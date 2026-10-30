from pathlib import Path

import pytest

from vstent.config import load_config
from vstent.data.synth import SynthConfig, generate_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """A tiny synthetic dataset (train 96, val 32, test 32, cal 32, ood 16) from synth_v1."""
    cfg = load_config(ROOT / "configs" / "synth_v1.yaml", SynthConfig)
    sizes = {"train": 96, "val": 32, "test": 32, "cal": 32, "ood": 16}
    splits = {k: v.model_copy(update={"n": sizes[k]}) for k, v in cfg.splits.items()}
    out = tmp_path_factory.mktemp("synth")
    generate_dataset(cfg.model_copy(update={"splits": splits}), out_dir=out, workers=1)
    return out


def _write_fake_arcade(root: Path, sizes: dict[str, int], size: int = 64, seed: int = 0) -> Path:
    """A tiny ARCADE-like folder: one random thick curve per image, COCO polygons / boxes."""
    import json

    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    for task in ("syntax", "stenosis"):
        for split, n in sizes.items():
            folder = root / "arcade" / task / split
            (folder / "images").mkdir(parents=True)
            (folder / "annotations").mkdir()
            images, anns = [], []
            for k in range(n):
                image_id = k + 1
                mask = np.zeros((size, size), np.uint8)
                pts = np.cumsum(rng.uniform(-6, 12, size=(6, 2)), axis=0) + size * 0.2
                pts = np.clip(pts, 2, size - 3).astype(np.int32)
                cv2.polylines(mask, [pts], False, 1, thickness=int(rng.integers(3, 7)))
                img = (rng.normal(150, 12, (size, size)) - 70 * mask).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(folder / "images" / f"{image_id}.png"), img)
                images.append(
                    {"id": image_id, "file_name": f"{image_id}.png", "width": size, "height": size}
                )
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in contours:
                    if task == "syntax" and len(c) >= 3:
                        poly = c[:, 0, :].astype(float).ravel().tolist()
                        anns.append(
                            {
                                "id": len(anns) + 1,
                                "image_id": image_id,
                                "category_id": 5,
                                "segmentation": [poly],
                                "iscrowd": 0,
                            }
                        )
                    elif task == "stenosis":
                        x, y, w, h = cv2.boundingRect(c)
                        anns.append(
                            {
                                "id": len(anns) + 1,
                                "image_id": image_id,
                                "category_id": 26,
                                "bbox": [x, y, w, h],
                                "iscrowd": 0,
                                "segmentation": [[x, y, x + w, y, x + w, y + h]],
                            }
                        )
            coco = {
                "images": images,
                "annotations": anns,
                "categories": [{"id": 5, "name": "5"}, {"id": 26, "name": "stenosis"}],
            }
            (folder / "annotations" / f"{split}.json").write_text(json.dumps(coco))
    return root


@pytest.fixture(scope="session")
def tiny_arcade(tmp_path_factory):
    """A fake ARCADE raw folder (train 12, val 4, test 4 images of 64 x 64 px)."""
    root = tmp_path_factory.mktemp("arcade_raw")
    return _write_fake_arcade(root, {"train": 12, "val": 4, "test": 4})


@pytest.fixture(scope="session")
def tiny_run(tiny_data, tmp_path_factory):
    """A small surrogate trained for 2 epochs on the tiny dataset (CPU)."""
    from vstent.surrogate.model import ModelConfig
    from vstent.surrogate.train import SurrogateConfig, TrainConfig, train_surrogate

    root = tmp_path_factory.mktemp("onnx")
    cfg = SurrogateConfig(
        data_dir=str(tiny_data),
        runs_dir=str(root / "runs"),
        model=ModelConfig(base_channels=8, levels=3, groups=4, cond_dim=16),
        train=TrainConfig(epochs=2, batch_size=32, device="cpu"),
    )
    return train_surrogate(cfg, run_dir=root / "runs" / "tiny", checkpoint_root=root / "ckpt")


@pytest.fixture(scope="session")
def tiny_export(tiny_run, tmp_path_factory):
    """The tiny surrogate exported to ONNX (model directory in ``["out_dir"]``)."""
    from vstent.surrogate.export import export_run

    return export_run(tiny_run, tmp_path_factory.mktemp("models") / "tiny")


@pytest.fixture(scope="session")
def tiny_ensemble(tiny_data, tmp_path_factory):
    """A 2-member ensemble of small surrogates trained for 2 epochs on the tiny dataset (CPU)."""
    from vstent.config import dump_config
    from vstent.surrogate.ensemble import EnsembleConfig, train_ensemble
    from vstent.surrogate.model import ModelConfig
    from vstent.surrogate.train import SurrogateConfig, TrainConfig

    root = tmp_path_factory.mktemp("ensemble")
    base = SurrogateConfig(
        data_dir=str(tiny_data),
        runs_dir=str(root / "runs"),
        model=ModelConfig(base_channels=8, levels=3, groups=4, cond_dim=16),
        train=TrainConfig(epochs=2, batch_size=32, device="cpu"),
    )
    dump_config(base, root / "base.yaml")
    cfg = EnsembleConfig(
        name="tens", runs_dir=str(root / "runs"), base_config=str(root / "base.yaml")
    )
    cfg = cfg.model_copy(update={"seeds": [0, 1]})
    return train_ensemble(cfg, run_dir=root / "runs" / "tens", checkpoint_root=root / "ckpt")


@pytest.fixture(scope="session")
def tiny_ensemble_export(tiny_ensemble, tmp_path_factory):
    """The tiny ensemble exported (mean + spread, screen model, conformal calibration on val)."""
    from vstent.surrogate.export import export_ensemble

    return export_ensemble(tiny_ensemble, tmp_path_factory.mktemp("models") / "tens")


@pytest.fixture(scope="session")
def dark_seg_dir(tmp_path_factory):
    """A stand-in ONNX 'segmenter' (dark pixels = vessel) exported like the real one."""
    import torch

    from vstent.imaging.seg_export import export_seg_onnx
    from vstent.imaging.segment import PIXEL_MEAN, PIXEL_STD
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE

    class DarkIsVessel(torch.nn.Module):
        def forward(self, x):
            return 20.0 * ((0.5 - PIXEL_MEAN) / PIXEL_STD - x)

    out = tmp_path_factory.mktemp("seg_model")
    meta = {"task": "vessel_segmentation", "image_size": "128"}
    export_seg_onnx(DarkIsVessel(), out / SEG_ONNX_FILE, 128, meta)
    return out
