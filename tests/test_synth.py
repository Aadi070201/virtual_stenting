from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from vstent.cli import app
from vstent.config import load_config
from vstent.data.card import write_dataset_card
from vstent.data.synth import (
    SHAPES,
    SamplingConfig,
    SynthConfig,
    content_hash,
    generate_dataset,
    generate_shard,
    lesion_bump,
    load_manifest,
    load_split,
    sample_vessel,
)
from vstent.physics.solver import PhysicsParams, solve_ffr_batch

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synth_v1.yaml"
MM = 1e-3


def _small_config(**split_sizes) -> SynthConfig:
    cfg = load_config(CONFIG, SynthConfig)
    sizes = {"train": 30, "val": 10, "test": 10, "ood": 10} | split_sizes
    splits = {
        name: split.model_copy(update={"n": sizes[name]}) for name, split in cfg.splits.items()
    }
    return cfg.model_copy(update={"splits": splits, "shard_size": 16})


def test_config_loads_with_plan_ranges():
    cfg = load_config(CONFIG, SynthConfig)
    assert {k: v.n for k, v in cfg.splits.items()} == {
        "train": 100_000,
        "val": 10_000,
        "test": 10_000,
        "ood": 5_000,
    }
    assert len({v.seed for v in cfg.splits.values()}) == 4
    ood = cfg.sampling_for("ood")
    assert ood.length_mm == (80.0, 120.0)
    assert ood.lesion_count_probs == [0.0, 0.0, 0.0, 0.5, 0.5]
    assert cfg.physics == PhysicsParams()


def test_sampling_config_validation():
    with pytest.raises(ValueError):
        SamplingConfig(length_mm=(80.0, 30.0))
    with pytest.raises(ValueError):
        SamplingConfig(lesion_count_probs=[0.5, 0.6])
    with pytest.raises(ValueError):
        SamplingConfig(shape_probs={"star": 1.0})


@pytest.mark.parametrize("shape", SHAPES)
def test_lesion_bump_peak_and_support(shape):
    s = np.linspace(0, 50 * MM, 256)
    bump = lesion_bump(s, 20 * MM, 10 * MM, shape, np.random.default_rng(0), SamplingConfig())
    assert bump.max() == pytest.approx(1.0)
    assert bump.min() >= 0.0
    outside = (s < 20 * MM) | (s > 30 * MM)
    assert np.all(bump[outside] == 0.0)


def test_sampled_vessels_respect_constraints():
    sampling = SamplingConfig()
    rng = np.random.default_rng(0)
    counts = np.zeros(len(sampling.lesion_count_probs), dtype=int)
    for _ in range(300):
        v = sample_vessel(rng, sampling)
        p = v.profile
        assert p.n == 256
        assert 30 * MM <= p.length <= 80 * MM
        assert 2.25 * MM <= 2 * p.r_ref[0] <= 4.5 * MM
        assert np.all(p.r > 0)
        counts[len(v.lesions)] += 1
        prev_end = -np.inf
        for les in v.lesions:
            assert les.start >= 5 * MM - 1e-12
            assert les.start + les.length <= p.length - 5 * MM + 1e-12
            assert les.start >= prev_end + 2 * MM - 1e-12  # non-overlapping, gap kept
            assert 20.0 <= les.ds_pct <= 90.0
            prev_end = les.start + les.length
        assert 70 * 133.322 <= v.pa <= 110 * 133.322
        assert 0.25 <= v.v_hyp <= 0.45
    assert counts[1] > counts[2] > counts[3]  # 55% / 25% / 10%


def test_noise_free_throat_matches_nominal_ds():
    sampling = SamplingConfig(
        noise_sigma_frac=(0.0, 0.0), diffuse_prob=0.0, lesion_count_probs=[0, 1.0]
    )
    rng = np.random.default_rng(3)
    for _ in range(20):
        v = sample_vessel(rng, sampling)
        measured = 100 * (1 - (v.profile.r / v.profile.r_ref).min())
        assert measured == pytest.approx(v.lesions[0].ds_pct, abs=1e-9)


def test_ood_overrides_applied():
    cfg = load_config(CONFIG, SynthConfig)
    rng = np.random.default_rng(1)
    for _ in range(50):
        v = sample_vessel(rng, cfg.sampling_for("ood"))
        assert 80 * MM <= v.profile.length <= 120 * MM
        assert len(v.lesions) in (3, 4)


def test_shard_is_reproducible_and_labels_match_stored_profile():
    sampling, physics = SamplingConfig(), PhysicsParams()
    a = generate_shard(sampling, physics, 25, 256, (5, 0))
    b = generate_shard(sampling, physics, 25, 256, (5, 0))
    c = generate_shard(sampling, physics, 25, 256, (5, 1))
    assert content_hash(a) == content_hash(b) != content_hash(c)
    assert a["r"].dtype == np.float32
    res = solve_ffr_batch(
        a["r"].astype(np.float64),
        a["length"] / 255,
        a["r_ref"].astype(np.float64),
        physics,
        pa=a["pa"],
        pv=a["pv"],
        v_hyp=a["v_hyp"],
    )
    np.testing.assert_allclose(a["ffr"], res.ffr, atol=1e-6)
    np.testing.assert_array_equal(a["q"], res.q)


def test_generate_dataset_independent_of_workers(tmp_path):
    cfg = _small_config()
    m1 = generate_dataset(cfg, out_dir=tmp_path / "w1", workers=1)
    m2 = generate_dataset(cfg, out_dir=tmp_path / "w2", workers=2)
    for split in cfg.splits:
        assert [s["sha256"] for s in m1["splits"][split]["shards"]] == [
            s["sha256"] for s in m2["splits"][split]["shards"]
        ]
    assert len(m1["splits"]["train"]["shards"]) == 2  # 30 samples, shard size 16
    train = load_split(tmp_path / "w1", "train")
    assert train["r"].shape == (30, 256)
    assert train["lesion_start"].shape == (30, 4)
    meta = yaml.safe_load((tmp_path / "w1" / "run_meta.yaml").read_text(encoding="utf-8"))
    assert "git_commit" in meta
    assert load_config(tmp_path / "w1" / "config.yaml", SynthConfig) == cfg
    assert load_manifest(tmp_path / "w1")["shapes"] == list(SHAPES)


def test_generate_dataset_refuses_to_overwrite(tmp_path):
    cfg = _small_config(train=5, val=5, test=5, ood=5)
    generate_dataset(cfg, out_dir=tmp_path, workers=1)
    with pytest.raises(FileExistsError):
        generate_dataset(cfg, out_dir=tmp_path, workers=1)
    generate_dataset(cfg, out_dir=tmp_path, workers=1, overwrite=True)


def test_dataset_card(tmp_path):
    cfg = _small_config(train=200, val=50, test=50, ood=20)
    generate_dataset(cfg, out_dir=tmp_path / "data", workers=1)
    card = write_dataset_card(tmp_path / "data", tmp_path / "reports")
    assert [t["split"] for t in card["targets"]] == ["train", "val", "test"]
    assert card["splits"]["train"]["n"] == 200
    assert (tmp_path / "reports" / "figures" / "dataset_hist.png").stat().st_size > 0
    text = (tmp_path / "reports" / "dataset_card.md").read_text(encoding="utf-8")
    assert "Research use only" in text


def test_synth_cli(tmp_path):
    cfg = _small_config(train=200, val=50, test=50, ood=20)
    cfg_path = tmp_path / "synth.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    args = ["synth", "--config", str(cfg_path), "--out", str(tmp_path / "data")]
    args += ["--workers", "1", "--report-dir", str(tmp_path / "reports")]
    result = CliRunner().invoke(app, args)
    assert "dataset_card.md" in result.output
    assert (tmp_path / "reports" / "dataset_card.json").exists()
    rerun = CliRunner().invoke(app, args)
    assert rerun.exit_code == 1
    assert "--overwrite" in rerun.output
