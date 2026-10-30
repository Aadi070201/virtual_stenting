from pathlib import Path

import numpy as np
import pytest

from vstent.config import load_config
from vstent.data.synth import content_hash, generate_dataset, load_split
from vstent.data.synth_tree import (
    ABSENT,
    LESION_KINDS,
    MAX_BRANCHES,
    TreeSamplingConfig,
    TreeSynthConfig,
    generate_tree_shard,
    sample_tree,
    tree_from_arrays,
)
from vstent.data.tree_card import write_tree_dataset_card
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.physics.tree import solve_tree

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synth_tree_v1.yaml"
PARAMS = PhysicsParams()
PHYSICS_FROM_YAML = load_config(CONFIG.parent / "physics.yaml", PhysicsParams)


def _small(**sizes) -> TreeSynthConfig:
    cfg = load_config(CONFIG, TreeSynthConfig)
    n = {"train": 60, "val": 20, "test": 20, "cal": 20, "ood": 10} | sizes
    splits = {k: v.model_copy(update={"n": n[k]}) for k, v in cfg.splits.items()}
    return cfg.model_copy(update={"splits": splits, "shard_size": 40})


def test_config_loads():
    cfg = load_config(CONFIG, TreeSynthConfig)
    assert set(cfg.splits) == {"train", "val", "test", "cal", "ood"}
    assert len({s.seed for s in cfg.splits.values()}) == 5
    assert cfg.sampling_for("ood").side_count_probs == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert cfg.physics == PHYSICS_FROM_YAML


def test_sampled_trees_are_valid_and_follow_the_geometry_rules():
    cfg = TreeSamplingConfig()
    rng = np.random.default_rng(0)
    kinds = set()
    for _ in range(200):
        t = sample_tree(rng, cfg, 128)
        tree = t.tree
        assert 2 <= tree.n_branches <= MAX_BRANCHES
        main = tree.branches[0]
        for k in range(1, tree.n_branches):
            att = tree.attach[k]
            d_before = 2 * main.r_ref[att]
            d_side = 2 * tree.branches[k].r_ref[0]
            d_after = 2 * main.r_ref[att + 1]
            assert 0.45 - 1e-9 <= d_side / d_before <= 0.85 + 1e-9
            assert d_after < d_before  # the main vessel steps down after every take-off
        for b in tree.branches:
            assert np.all(b.r > 0) and np.all(b.r <= 1.2 * b.r_ref)
        kinds |= {les.kind for les in t.lesions}
    assert kinds == set(LESION_KINDS)  # bifurcation and ostial lesions do occur


def test_shard_labels_match_the_solvers_on_the_stored_radii():
    a = generate_tree_shard(TreeSamplingConfig(), PARAMS, 6, 128, (7, 0))
    assert a["r"].dtype == np.float32 and a["r"].shape == (6, MAX_BRANCHES, 128)
    for i in range(6):
        nb = int(a["branch_count"][i])
        assert np.all(a["parent"][i, nb:] == ABSENT) and np.all(a["r"][i, nb:] == 0)
        tree = tree_from_arrays(a, i)
        res = solve_tree(tree, PARAMS, pa=a["pa"][i], pv=a["pv"][i], v_hyp=a["v_hyp"][i])
        for k in range(nb):
            np.testing.assert_allclose(a["ffr"][i, k], res.ffr[k], rtol=1e-6)
        np.testing.assert_allclose(a["q_outlet"][i, :nb], res.q_outlet, rtol=1e-12)
        single = solve_ffr_batch(
            a["r"][i, 0].astype(float), a["length"][i, 0] / 127, a["r_ref"][i, 0].astype(float),
            PARAMS, pa=a["pa"][i], pv=a["pv"][i], v_hyp=a["v_hyp"][i],
        )  # fmt: skip
        np.testing.assert_allclose(a["ffr_main_single"][i], single.ffr[0], rtol=1e-6)
    # side branches lower the main vessel's flow, so distal FFR is (almost always) higher
    assert np.median(a["ffr"][:, 0, -1] - a["ffr_main_single"][:, -1]) > 0


def test_generation_is_deterministic(tmp_path):
    cfg = _small()
    m1 = generate_dataset(cfg, out_dir=tmp_path / "a", workers=1)
    m2 = generate_dataset(cfg, out_dir=tmp_path / "b", workers=2)
    for split in cfg.splits:
        h1 = [s["sha256"] for s in m1["splits"][split]["shards"]]
        h2 = [s["sha256"] for s in m2["splits"][split]["shards"]]
        assert h1 == h2
    assert m1["kind"] == "tree" and m1["max_branches"] == MAX_BRANCHES
    arrays = load_split(tmp_path / "a", "train")
    assert arrays["r"].shape[0] == 60
    assert content_hash(arrays) == content_hash(load_split(tmp_path / "b", "train"))


def test_tree_dataset_card(tmp_path):
    cfg = _small(train=120)
    generate_dataset(cfg, out_dir=tmp_path / "data", workers=1)
    card = write_tree_dataset_card(tmp_path / "data", tmp_path / "reports")
    assert [t["split"] for t in card["targets"]] == ["train", "val", "test", "cal"]
    assert card["splits"]["train"]["n"] == 120
    assert card["splits"]["ood"]["side_branches"]["4"] == 10
    md = (tmp_path / "reports" / "tree_dataset_card.md").read_text(encoding="utf-8")
    assert "Research use only" in md and "Tree − single" in md
    assert (tmp_path / "reports" / "figures" / "tree_dataset_hist.png").stat().st_size > 0


@pytest.mark.parametrize("bad", [{"side_count_probs": [0.5, 0.6]}, {"split_exponent": [3, 2]}])
def test_sampling_validation(bad):
    with pytest.raises(ValueError):
        TreeSamplingConfig(**bad)
