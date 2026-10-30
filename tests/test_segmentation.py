import csv

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from vstent import checkpoints as ck
from vstent.cli import app
from vstent.data.arcade import load_split, polygons_to_mask
from vstent.imaging.seg_train import (
    EpochSampler,
    SegConfig,
    SegDataset,
    SegModelConfig,
    SegTrainConfig,
    load_checkpoint,
    resume_seg,
    seg_loss,
    train_seg,
)
from vstent.imaging.segment import (
    AugmentConfig,
    augment,
    dice_score,
    preprocess_image,
    probability_to_image,
    to_unit,
)

TINY_MODEL = SegModelConfig(encoder="resnet18", encoder_weights=None, image_size=64)


def test_preprocess_image_any_size_and_dtype():
    img8 = np.full((300, 400), 255, np.uint8)
    x = preprocess_image(img8)
    assert x.shape == (1, 512, 512) and x.dtype == np.float32
    assert np.allclose(x, (1.0 - 0.449) / 0.226)
    assert np.allclose(to_unit(np.full((4, 4), 65535, np.uint16)), 1.0)
    assert np.allclose(to_unit(np.full((4, 4, 3), 255, np.uint8)), 1.0)  # colour → grey
    prob = probability_to_image(np.zeros((512, 512)), (300, 400))
    assert prob.shape == (300, 400) and prob.dtype == np.float32


def test_augment_ranges_and_determinism():
    rng_img = np.random.default_rng(0)
    img = rng_img.integers(0, 256, (96, 96), dtype=np.uint8)
    mask = np.zeros((96, 96), np.uint8)
    mask[40:56, 10:86] = 1
    a = augment(img, mask, np.random.default_rng(5), AugmentConfig())
    b = augment(img, mask, np.random.default_rng(5), AugmentConfig())
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    x, m = a
    assert x.dtype == np.float32 and x.min() >= 0.0 and x.max() <= 1.0
    assert set(np.unique(m)) <= {0, 1} and m.shape == mask.shape
    assert 0.7 < m.sum() / mask.sum() < 1.3  # scale 0.9–1.1 keeps the area roughly


def test_dice_score():
    a = np.zeros((2, 8, 8), bool)
    a[0, :4] = True
    b = a.copy()
    assert np.allclose(dice_score(a, b), [1.0, 1.0])  # second pair: both empty → 1
    c = np.zeros_like(a)
    c[0, 4:] = True
    assert dice_score(a, c)[0] == 0.0
    half = np.zeros_like(a)
    half[0, :2] = True
    assert dice_score(half, a)[0] == pytest.approx(2 * 16 / (16 + 32))


def test_polygons_to_mask_and_fake_arcade(tiny_arcade):
    m = polygons_to_mask([[1, 1, 6, 1, 6, 6, 1, 6]], 8, 8)
    assert m.dtype == np.uint8 and m[3, 3] == 1 and m[7, 7] == 0
    syntax = load_split(tiny_arcade, "syntax", "train")
    assert len(syntax) == 12 and syntax.images.shape == (12, 64, 64)
    assert syntax.boxes is None and list(syntax.image_ids) == sorted(syntax.image_ids)
    assert all(m.sum() > 0 for m in syntax.masks)
    sten = load_split(tiny_arcade, "stenosis", "val", max_images=2)
    assert sten.masks is None and len(sten.boxes) == 2 and len(sten.boxes[0][0]) == 4
    with pytest.raises(ValueError, match="unknown"):
        load_split(tiny_arcade, "other", "train")


def test_sampler_seeds_follow_generator():
    a = list(EpochSampler(5, torch.Generator().manual_seed(1)))
    b = list(EpochSampler(5, torch.Generator().manual_seed(1)))
    assert a == b and sorted(i for i, _ in a) == list(range(5))
    assert len({seed for _, seed in a}) == 1


def test_dataset_and_loss(tiny_arcade):
    s = load_split(tiny_arcade, "syntax", "train", max_images=2)
    ds = SegDataset(s.images, s.masks, 32, AugmentConfig())
    x, y = ds[(0, 7)]
    assert x.shape == y.shape == (1, 32, 32) and set(torch.unique(y).tolist()) <= {0.0, 1.0}
    logits = torch.full((1, 1, 32, 32), -20.0)
    assert seg_loss(logits, torch.zeros_like(logits), 1.0) < 1e-3


def _seg_config(tiny_arcade, tmp_path, epochs):
    return SegConfig(
        raw_dir=str(tiny_arcade),
        runs_dir=str(tmp_path / "runs"),
        model=TINY_MODEL,
        train=SegTrainConfig(epochs=epochs, batch_size=4, device="cpu", workers=0, patience=50),
    )


def _metrics(run_dir):
    with (run_dir / "metrics.csv").open(encoding="utf-8") as f:
        return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(f)]


def test_seg_stop_and_resume_matches_uninterrupted(tiny_arcade, tmp_path):
    cfg = _seg_config(tiny_arcade, tmp_path, epochs=2)
    root = tmp_path / "ckpt"
    straight = train_seg(cfg, run_dir=tmp_path / "runs" / "straight", checkpoint_root=root)

    def stop_after_first(msg):
        if msg.startswith("epoch   1"):
            ck.request_stop(root / "split")

    run = tmp_path / "runs" / "split"
    train_seg(cfg, run_dir=run, checkpoint_root=root, log=stop_after_first)
    status = ck.read_status(root / "split")
    assert status["status"] == "stopped" and status["epoch"] == 1
    assert status["best_metric"] == "val_dice"
    resume_seg("split", checkpoint_root=root)
    assert ck.read_status(root / "split")["status"] == "finished"
    assert _metrics(run) == _metrics(straight)
    a, b = load_checkpoint(straight), load_checkpoint(run)
    for (name, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(pa, pb), name


def test_train_seg_cli_needs_exactly_one_mode():
    assert CliRunner().invoke(app, ["train-seg"]).exit_code == 2
