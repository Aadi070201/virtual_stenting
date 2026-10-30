import json

import numpy as np
import pytest
import torch

from vstent.data.synth import load_split
from vstent.physics.geometry import cosine_lesion_profile
from vstent.physics.solver import solve_ffr
from vstent.surrogate.baseline import BASELINE_FEATURES, hand_features
from vstent.surrogate.dataset import SurrogateDataset, load_raw
from vstent.surrogate.evaluate import _signed, distal_metrics, evaluate_run, mae_by_ds_bin
from vstent.surrogate.features import (
    GLOBAL_FEATURES,
    POINT_CHANNELS,
    NormStats,
    RawInputs,
    global_features,
    point_features,
    prepare_inputs,
)
from vstent.surrogate.model import FFRUNet, ModelConfig, monotone_ffr_head
from vstent.surrogate.train import (
    SurrogateConfig,
    TrainConfig,
    load_checkpoint,
    train_surrogate,
)

MM = 1e-3
SMALL_MODEL = ModelConfig(base_channels=8, levels=3, groups=4, cond_dim=16)


def _raw_from_profile():
    prof = cosine_lesion_profile(60 * MM, 3.0 * MM, 10 * MM, 0.6)
    res = solve_ffr(prof)
    return RawInputs.from_profile(prof, res.pa, res.pv, res.r_micro), res


def test_point_and_global_features():
    raw, res = _raw_from_profile()
    xp, xg = point_features(raw), global_features(raw)
    assert xp.shape == (1, len(POINT_CHANNELS), 256)
    assert xg.shape == (1, len(GLOBAL_FEATURES))
    np.testing.assert_allclose(xp[0, 0], raw.r[0] / raw.r_ref[0, 0])
    np.testing.assert_allclose(xp[0, 1], np.log(xp[0, 0]))
    assert xp[0, 2, 0] == 0.0 and xp[0, 2, -1] == 1.0
    assert abs(xp[0, 3, :20]).max() < 1e-9  # healthy start: area ratio constant
    np.testing.assert_allclose(xg[0], [60.0, 1.5, 90.0, 5.0, np.log(res.r_micro)], rtol=1e-9)


def test_from_profile_resamples():
    prof = cosine_lesion_profile(60 * MM, 3.0 * MM, 10 * MM, 0.6, n=401)
    raw = RawInputs.from_profile(prof, 12000.0, 700.0, 4e9)
    assert raw.r.shape == (1, 256)


def test_norm_stats_roundtrip_and_standardisation(tmp_path, tiny_data):
    raw, _ = load_raw(tiny_data, "train")
    stats = NormStats.fit(raw)
    stats.save(tmp_path / "stats.json")
    assert NormStats.load(tmp_path / "stats.json") == stats
    xp, xg = prepare_inputs(raw, stats)
    assert xp.dtype == np.float32 and xg.dtype == np.float32
    np.testing.assert_allclose(xp.mean(axis=(0, 2)), 0.0, atol=1e-4)
    np.testing.assert_allclose(xg.std(axis=0), 1.0, atol=1e-3)
    with pytest.raises(ValueError, match="points"):
        prepare_inputs(
            RawInputs(
                *(
                    np.atleast_2d(v)[:, :100] if v.ndim == 2 else v
                    for v in (raw.r, raw.r_ref, raw.length, raw.pa, raw.pv, raw.r_micro)
                )
            ),
            stats,
        )


def test_head_monotone_and_starts_at_one():
    z = torch.randn(16, 256) * 5
    ffr = monotone_ffr_head(z)
    assert ffr.shape == (16, 256)
    assert torch.all(ffr[:, 0] == 1.0)
    assert torch.all(ffr[:, 1:] <= ffr[:, :-1])


@pytest.mark.parametrize("config", [ModelConfig(), SMALL_MODEL])
def test_model_output_shape_and_properties(config):
    torch.manual_seed(0)
    model = FFRUNet(config)
    with torch.no_grad():
        for p in model.out.parameters():  # break the zero init so outputs vary along s
            p.normal_(0, 0.5)
    xp = torch.randn(3, config.in_channels, 256)
    xg = torch.randn(3, config.global_dim)
    ffr = model(xp, xg)
    assert ffr.shape == (3, 256)
    assert torch.all(ffr[:, 0] == 1.0)
    assert torch.all(torch.diff(ffr, dim=1) <= 0)
    assert model(xp[:1], xg[:1]).shape == (1, 256)


def test_untrained_model_starts_near_ffr_08():
    ffr = FFRUNet(SMALL_MODEL)(torch.randn(2, 4, 256), torch.randn(2, 5))
    np.testing.assert_allclose(ffr[:, -1].detach().numpy(), 0.80, atol=0.01)


def test_model_gradients_flow():
    model = FFRUNet(SMALL_MODEL)
    loss = model(torch.randn(4, 4, 256), torch.randn(4, 5)).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_dataset_batches(tiny_data):
    raw, ffr = load_raw(tiny_data, "train")
    ds = SurrogateDataset(raw, ffr, NormStats.fit(raw))
    assert len(ds) == 96
    xp, xg, y = ds[0]
    assert xp.shape == (4, 256) and xg.shape == (5,) and y.shape == (256,)
    sizes = [b[2].shape[0] for b in ds.batches(40, shuffle=True, generator=torch.Generator())]
    assert sizes == [40, 40, 16]


def test_baseline_hand_features(tiny_data):
    arrays = load_split(tiny_data, "train")
    feats = hand_features(RawInputs.from_arrays(arrays))
    assert feats.shape == (96, len(BASELINE_FEATURES))
    max_ds = 100 * (1 - (arrays["r"] / arrays["r_ref"]).min(axis=1))
    np.testing.assert_allclose(feats[:, 0], max_ds, rtol=1e-5)


def test_distal_metrics():
    true = np.array([0.70, 0.75, 0.85, 0.90])
    pred = np.array([0.72, 0.82, 0.84, 0.90])
    m = distal_metrics(pred, true)
    assert m["mae"] == pytest.approx(0.025)  # errors +0.02, +0.07, -0.01, 0
    assert m["bias"] == pytest.approx(0.02)
    assert m["agreement"] == pytest.approx(0.75)  # 0.75 -> 0.82 crosses the cut-off
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(1.0)
    bins = mae_by_ds_bin(pred, true, np.array([10.0, 40.0, 60.0, 95.0]))
    assert [b["n"] for b in bins] == [1, 1, 1, 1]


def test_signed_format_has_no_negative_zero():
    assert _signed(-0.00002) == "+0.0000"
    assert _signed(-0.0057) == "-0.0057"
    assert _signed(0.0056) == "+0.0056"


def test_train_and_evaluate_smoke(tmp_path, tiny_data):
    cfg = SurrogateConfig(
        data_dir=str(tiny_data),
        runs_dir=str(tmp_path / "runs"),
        model=SMALL_MODEL,
        train=TrainConfig(epochs=2, batch_size=32, device="cpu", patience=5),
    )
    run_dir = train_surrogate(
        cfg, run_dir=tmp_path / "runs" / "smoke", checkpoint_root=tmp_path / "checkpoints"
    )
    for name in (
        "config.yaml",
        "run_meta.yaml",
        "norm_stats.json",
        "metrics.csv",
        "best.pt",
        "summary.json",
    ):
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / "summary.json").read_text())["epochs_run"] == 2
    model = load_checkpoint(run_dir)
    assert not model.training

    report = evaluate_run(run_dir, tmp_path / "reports", device="cpu")
    assert set(report["splits"]) == {"val", "test", "ood"}
    assert len(report["targets"]) == 3
    for fig in ("scatter", "bland_altman", "error_by_ds", "examples", "training"):
        assert (tmp_path / "reports" / "figures" / f"surrogate_{fig}.png").exists()
    assert "Research use only" in (tmp_path / "reports" / "surrogate_eval.md").read_text(
        encoding="utf-8"
    )
