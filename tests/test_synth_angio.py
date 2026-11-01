"""P8 sim-to-real: rendered angiograms, training with them, the comparison report."""

import json

import numpy as np
import pytest

from vstent.config import load_config
from vstent.data.synth_angio import (
    SynthAngioConfig,
    generate,
    load_synth_split,
    render_sample,
    split_seeds,
    write_card,
)
from vstent.data.synth_tree import TreeSynthConfig
from vstent.imaging.seg_train import SegConfig, SegModelConfig, SegSyntheticData, SegTrainConfig

# small images with a coarse pixel so whole trees fit (the real data: 512 px, ARCADE pixel size)
SMALL = SynthAngioConfig(
    n={"train": 3, "val": 1, "test": 2}, workers=1,
    scene={"image_size": 64, "iso_pixel_mm": [1.8, 2.0], "texture_scale_px": 6.0},
    small_branches=(2, 3), shadows=(1, 1), ribs=(1, 1),
)  # fmt: skip
TINY_MODEL = SegModelConfig(encoder="resnet18", encoder_weights=None, image_size=64)


def test_render_sample_is_deterministic_and_labels_major_vessels_only():
    tree_cfg = load_config(SMALL.scene.tree_config, TreeSynthConfig)
    img, mask, facts = render_sample(5, SMALL, tree_cfg)
    img2, mask2, _ = render_sample(5, SMALL, tree_cfg)
    assert np.array_equal(img, img2) and np.array_equal(mask, mask2)
    assert img.dtype == np.uint8 and img.shape == (64, 64) and set(np.unique(mask)) <= {0, 1}
    assert mask.any() and facts["labelled"] >= 1 and facts["small_branches"] >= 2
    # vessels are dark: labelled pixels are darker than the rest of the image on average
    assert img[mask > 0].mean() < img[mask == 0].mean()
    strict = SMALL.model_copy(update={"label_min_diameter_mm": 50.0})  # nothing is that wide
    with pytest.raises(ValueError, match="no view shows the vessels"):
        render_sample(5, strict.model_copy(update={"tries": 3}), tree_cfg)
    seeds = {s: split_seeds(SMALL, s) for s in ("train", "val", "test")}
    assert not set(seeds["train"]) & set(seeds["val"]) and len(seeds["test"]) == 2


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    cfg = SMALL.model_copy(update={"out_dir": str(tmp_path_factory.mktemp("synth_angio"))})
    return generate(cfg, log=lambda _m: None)


def test_generate_and_card(synth_dir, tmp_path):
    images, masks = load_synth_split(synth_dir, "train")
    assert images.shape == masks.shape == (3, 64, 64)
    manifest = json.loads((synth_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["test"]["n"] == 2 and len(manifest["splits"]["train"]["facts"]) == 3
    stats = write_card(synth_dir, tmp_path)
    text = (tmp_path / "synth_angio_card.md").read_text(encoding="utf-8")
    assert "Research use only" in text and stats["train"]["n"] == 3
    assert (tmp_path / "figures" / "synth_angio_examples.png").exists()


def test_training_with_rendered_images_and_comparison(synth_dir, tiny_arcade, tmp_path):
    from vstent.imaging.seg_train import train_seg
    from vstent.imaging.sim_to_real import compare_runs, paired_interval

    train = SegTrainConfig(epochs=1, batch_size=4, device="cpu", workers=0, patience=50)
    base_cfg = SegConfig(raw_dir=str(tiny_arcade), runs_dir=str(tmp_path / "runs"),
                         model=TINY_MODEL, train=train)  # fmt: skip
    new_cfg = base_cfg.model_copy(update={"synthetic": SegSyntheticData(dir=str(synth_dir),
                                                                        n_train=2)})  # fmt: skip
    logs: list[str] = []
    base = train_seg(base_cfg, run_dir=tmp_path / "runs" / "base", checkpoint_root=tmp_path / "c")
    new = train_seg(new_cfg, run_dir=tmp_path / "runs" / "new", checkpoint_root=tmp_path / "c",
                    log=logs.append)  # fmt: skip
    assert any("12 ARCADE + 2 rendered" in m for m in logs)
    report = compare_runs(base, new, synth_dir, tmp_path / "reports", device="cpu")
    assert [r["split"] for r in report["rows"]] == ["ARCADE val", "ARCADE test", "rendered test"]
    assert isinstance(report["adopt"], bool)
    text = (tmp_path / "reports" / "sim_to_real.md").read_text(encoding="utf-8")
    assert "Research use only" in text and "Adoption rule" in text
    mean, lo, hi = paired_interval(np.array([0.01, 0.02, 0.03, 0.02]))
    assert lo <= mean <= hi and mean == pytest.approx(0.02)
