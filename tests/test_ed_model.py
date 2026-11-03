"""P4l M2: the learned ED-frame model: decoding, scoring, shapes, a resumable run."""

import numpy as np
import pytest

from vstent.imaging import ed_data

torch = pytest.importorskip("torch")


def test_decode_and_score():
    from vstent.imaging.ed_model import decode, score

    prob = np.zeros(40)
    for k in (5, 15, 25, 35):
        prob[k] = 0.9
    prob[16] = 0.6  # a shoulder of the second peak, too close to count
    assert decode(prob) == (5, 15, 25, 35)
    runs = [{"ed": (5, 15, 26), "well": np.r_[np.zeros(4, bool), np.ones(36, bool)],
             "fps": 15.0}, {"ed": (2,), "well": np.ones(10, bool), "fps": 30.0}]  # fmt: skip
    s = score(runs, [(5, 15, 25, 35), ()])
    assert s["beats"] == 4 and s["rate"] == 0.75 and s["by_fps"]["30.0"]["rate"] == 0.0


def _fake(root, split, n, rng):
    out = root / split
    out.mkdir(parents=True)
    for i in range(n):
        t = int(rng.integers(20, 30))
        frames = rng.integers(0, 255, (t, 16, 16)).astype(np.uint8)
        np.savez(out / f"{i}_0.npz", frames=frames, ed=np.array([3, 13]), well=np.ones(t, bool),
                 m0=np.array([4]), fps=15.0)  # fmt: skip


def test_network_and_resumable_training(tmp_path, monkeypatch):
    from vstent.imaging import ed_model as em

    net = em.EdNet(8, pretrained=False)
    out = net(torch.zeros(2, 7, 1, 32, 32), torch.tensor([0.1, 0.033]))
    assert out.shape == (2, 7)
    rng = np.random.default_rng(0)
    _fake(tmp_path / "data", "train", 3, rng)
    _fake(tmp_path / "data", "val", 2, rng)
    monkeypatch.setattr(ed_data, "SIZE", 32)
    data, runs = str(tmp_path / "data"), str(tmp_path / "runs")
    cfg = em.EdTrainConfig(epochs=2, batch_runs=2, features=8, pretrained=False, device="cpu",
                           data_dir=data, runs_dir=runs)  # fmt: skip
    orig = em.stop_requested
    em.stop_requested = lambda d: True
    try:
        run = em.train_ed(cfg, checkpoint_root=tmp_path / "ckpt")
    finally:
        em.stop_requested = orig
    assert not (run / em.SUMMARY).exists()
    run = em.resume_ed(run.name, checkpoint_root=tmp_path / "ckpt")
    assert (run / em.SUMMARY).exists() and (run / em.BEST).exists()
    probs = em.predict(em.load_best(run), ed_data.load_split("val", tmp_path / "data"), "cpu")
    assert len(probs) == 2 and all(p.min() >= 0 and p.max() <= 1 for p in probs)


@pytest.mark.parametrize("head", ["tcn3", "tcn5", "gru", "transformer"])
def test_temporal_heads_respect_padding(head):
    """M2b: every head gives one logit per frame; the GRU and the transformer ignore the
    padding after a shorter run."""
    from vstent.imaging import ed_model as em

    torch.manual_seed(0)
    net = em.EdNet(8, pretrained=False, temporal=head).eval()
    x = torch.randn(1, 9, 1, 32, 32)
    ft = torch.tensor([0.1])
    alone = net(x, ft)
    padded = net(torch.cat([x, torch.zeros(1, 4, 1, 32, 32)], 1), ft, torch.tensor([9]))
    assert alone.shape == (1, 9) and padded.shape == (1, 13)
    if head in ("gru", "transformer"):
        assert torch.allclose(alone, padded[:, :9], atol=1e-5)


def test_soft_targets_and_augmentation():
    from vstent.imaging import ed_model as em

    y = em.targets((3, 8), 12, 0.6)
    assert y[3] == 1.0 and y[8] == 1.0 and 0.2 < y[4] < 0.3 and y[0] < 1e-5
    assert em.targets((3,), 6, 0.0).tolist() == [0, 0, 0, 1, 0, 0]
    gen = torch.Generator().manual_seed(0)
    frames = np.random.default_rng(0).integers(0, 255, (40, 16, 16)).astype(np.uint8)
    for _ in range(20):
        out, ed = em.augment(frames, (5, 20, 35), gen)
        assert out.dtype == np.uint8 and out.shape[1:] == (16, 16) and 28 <= len(out) <= 40
        assert all(0 <= e < len(out) for e in ed) and len(ed) >= 1


def test_m2b_training_resumes_with_its_schedule(tmp_path, monkeypatch):
    from vstent.imaging import ed_model as em

    rng = np.random.default_rng(1)
    _fake(tmp_path / "data", "train", 3, rng)
    _fake(tmp_path / "data", "val", 2, rng)
    monkeypatch.setattr(ed_data, "SIZE", 32)
    cfg = em.EdTrainConfig(epochs=3, batch_runs=2, features=8, pretrained=False, device="cpu",
                           data_dir=str(tmp_path / "data"), runs_dir=str(tmp_path / "runs"),
                           temporal="gru", soft_sigma=0.6, augment=True,
                           schedule="cosine")  # fmt: skip
    orig = em.stop_requested
    em.stop_requested = lambda d: True
    try:
        run = em.train_ed(cfg, checkpoint_root=tmp_path / "ckpt")
    finally:
        em.stop_requested = orig
    saved = em.load_state(tmp_path / "ckpt" / run.name)
    assert saved["scheduler"]["last_epoch"] == 1
    run = em.resume_ed(run.name, checkpoint_root=tmp_path / "ckpt")
    assert (run / em.SUMMARY).exists()
    assert em.load_best(run).kind == "gru"
