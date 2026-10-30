import csv

import numpy as np
import pytest
import torch

from vstent import checkpoints as ck
from vstent.data.synth import load_split
from vstent.data.synth_tree import tree_from_arrays
from vstent.surrogate.train import TrainConfig
from vstent.surrogate.tree_features import (
    TreeNormStats,
    TreeRawInputs,
    prepare_tree_inputs,
)
from vstent.surrogate.tree_model import TreeFFRNet, TreeModelConfig
from vstent.surrogate.tree_train import (
    TreeSurrogateConfig,
    load_tree_checkpoint,
    load_tree_raw,
    predict_tree_batches,
    resume_tree_surrogate,
    train_tree_surrogate,
)

SMALL = TreeModelConfig(base_channels=8, levels=3, groups=4, cond_dim=16, message_layers=2)


def test_features_from_trees_match_the_dataset_arrays(tiny_tree_data):
    arrays = load_split(tiny_tree_data, "test")
    raw = TreeRawInputs.from_arrays(arrays)
    trees = [tree_from_arrays(arrays, i) for i in range(4)]
    again = TreeRawInputs.from_trees(trees, arrays["pa"][:4], arrays["pv"][:4],
                                     list(arrays["r_outlet"][:4]))  # fmt: skip
    for key in ("r", "r_ref", "length", "parent", "attach", "r_outlet"):
        np.testing.assert_allclose(getattr(again, key), getattr(raw, key)[:4], rtol=1e-6)
    stats = TreeNormStats.fit(raw)
    xp, xb, parent, attach, mask = prepare_tree_inputs(raw, stats)
    assert xp.shape[1:] == (5, 4, 256) and xb.shape[1:] == (5, 7)
    assert np.all(xp[mask == 0] == 0) and np.all(xb[mask == 0] == 0)
    assert np.all(parent.sum(axis=2) == (raw.parent >= 0))  # one parent per non-root branch
    assert np.all(attach.sum(axis=2) == (raw.parent >= 0))


def test_tree_head_is_continuous_and_monotone(tiny_tree_data):
    raw, _ = load_tree_raw(tiny_tree_data, "test")
    xs = [torch.from_numpy(x) for x in prepare_tree_inputs(raw, TreeNormStats.fit(raw))]
    torch.manual_seed(0)
    model = TreeFFRNet(SMALL)
    with torch.no_grad():
        for p in model.out.parameters():  # random head so increments vary
            p.normal_(0.0, 1.0)
        ffr = model(*xs)
    mask = xs[4].bool()
    assert torch.allclose(ffr[:, 0, 0], torch.ones(len(raw)))
    assert torch.all(ffr[mask][:, 1:] <= ffr[mask][:, :-1] + 1e-6)
    for i in range(len(raw)):
        for k in np.flatnonzero(raw.parent[i] >= 0):
            p, t = raw.parent[i, k], raw.attach[i, k]
            assert float(ffr[i, k, 0]) == pytest.approx(float(ffr[i, p, t]), abs=1e-6)


def test_tiny_run_trains_and_predicts(tiny_tree_run, tiny_tree_data):
    model = load_tree_checkpoint(tiny_tree_run)
    stats = TreeNormStats.load(tiny_tree_run / "tree_norm_stats.json")
    raw, ffr = load_tree_raw(tiny_tree_data, "test")
    xs = [torch.from_numpy(x) for x in prepare_tree_inputs(raw, stats)]
    pred = predict_tree_batches(model, xs).numpy()
    assert pred.shape == ffr.shape and np.isfinite(pred).all()
    assert (tiny_tree_run / "summary.json").exists()


def _metrics(run_dir):
    with (run_dir / "metrics.csv").open(encoding="utf-8") as f:
        return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(f)]


def test_stop_and_resume_matches_uninterrupted_run(tiny_tree_data, tmp_path):
    cfg = TreeSurrogateConfig(
        data_dir=str(tiny_tree_data), runs_dir=str(tmp_path / "runs"), model=SMALL,
        train=TrainConfig(epochs=3, batch_size=16, device="cpu", patience=50),
    )  # fmt: skip
    roots = tmp_path / "ck"
    straight = train_tree_surrogate(cfg, run_dir=tmp_path / "runs" / "a", checkpoint_root=roots)

    def stop_after_first(msg):
        if msg.startswith("epoch   1"):
            ck.request_stop(roots / "b")

    run = train_tree_surrogate(
        cfg, run_dir=tmp_path / "runs" / "b", checkpoint_root=roots, log=stop_after_first
    )
    status = ck.read_status(roots / "b")
    assert status["status"] == "stopped" and status["epoch"] == 1
    assert status["resume_command"] == "vstent train-tree-surrogate --resume b"
    resume_tree_surrogate("b", checkpoint_root=roots)
    assert ck.read_status(roots / "b")["status"] == "finished"
    assert _metrics(run) == _metrics(straight)
    a, b = load_tree_checkpoint(straight), load_tree_checkpoint(run)
    for (name, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(pa, pb), name
