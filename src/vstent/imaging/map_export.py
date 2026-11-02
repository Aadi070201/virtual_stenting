"""ONNX export of the probability-map models trained with the segmentation trainer's other label
modes (P4k): vessel identity (``vessel_id``, softmax over classes) and the whole-image lesion
detector (``lesion_boxes``, sigmoid). The vessel segmentation keeps ``seg_export``.

Writes ``models/<kind>/<run-name>/{map.onnx, config.yaml, export_meta.json}``, with parity
against PyTorch on the run's own validation images (the same pre-processing as the runner).
"""

import json
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from vstent import DISCLAIMER, __version__
from vstent.config import CONFIG_FILENAME, git_info, load_config
from vstent.imaging.seg_train import SegConfig, load_checkpoint
from vstent.inference.onnx_runtime import MAP_INPUT, MAP_ONNX_FILE, MAP_OUTPUT, MapRunner

OPSET = 17
PARITY_IMAGES = 4
PARITY_TOLERANCE = 1e-3
MODELS = {"vessel_id": Path("models/vessel_id"), "lesion_boxes": Path("models/lesion_det")}


class _WithHead(torch.nn.Module):
    """Logits → probabilities: softmax over classes, or a sigmoid for one class."""

    def __init__(self, model: torch.nn.Module, classes: int) -> None:
        super().__init__()
        self.model, self.classes = model, classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.softmax(logits, dim=1) if self.classes > 1 else torch.sigmoid(logits)


def _validation_images(cfg: SegConfig, n: int) -> np.ndarray:
    if cfg.labels == "vessel_id":
        from vstent.data.vessel_id import load_labels

        return load_labels(cfg.raw_dir, "val", n)[0]
    from vstent.data.lesion_boxes import dataset

    return dataset("val", max_images=n)[0]


def export_map_run(run_dir: str | Path, out_dir: str | Path | None = None, *,
                   parity_images: int = PARITY_IMAGES) -> dict[str, Any]:  # fmt: skip
    """Export a ``vessel_id`` or ``lesion_boxes`` run to ONNX and check parity."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / CONFIG_FILENAME, SegConfig)
    if cfg.labels not in MODELS:
        raise ValueError(f"{run_dir}: label mode {cfg.labels!r} is exported by export-seg")
    out_dir = Path(out_dir or MODELS[cfg.labels] / run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    size, classes = cfg.model.image_size, cfg.model.classes
    model = load_checkpoint(run_dir, "cpu")
    wrapped = _WithHead(model, classes).eval()
    path = out_dir / MAP_ONNX_FILE
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        axes = {MAP_INPUT: {0: "batch"}, MAP_OUTPUT: {0: "batch"}}
        torch.onnx.export(wrapped, (torch.zeros(1, 1, size, size),), str(path), dynamo=False,
                          opset_version=OPSET, input_names=[MAP_INPUT], output_names=[MAP_OUTPUT],
                          dynamic_axes=axes)  # fmt: skip
    metadata = {"task": cfg.labels, "vstent_version": __version__, "run": run_dir.name,
                "image_size": str(size), "classes": str(classes), "encoder": cfg.model.encoder,
                "output": "class probabilities (softmax)" if classes > 1 else
                "lesion probability (sigmoid)", "disclaimer": DISCLAIMER}  # fmt: skip
    proto = onnx.load(str(path))
    onnx.helper.set_model_props(proto, metadata)
    onnx.checker.check_model(proto, full_check=True)
    onnx.save(proto, str(path))
    shutil.copyfile(run_dir / CONFIG_FILENAME, out_dir / CONFIG_FILENAME)

    from vstent.imaging.segment import preprocess_image

    images = _validation_images(cfg, parity_images)
    runner = MapRunner(path)
    diffs = []
    for img in images:
        with torch.no_grad():
            ref = wrapped(torch.from_numpy(preprocess_image(img, size)[None])).numpy()[0]
        diffs.append(float(np.abs(runner.predict(img) - ref).max()))  # 512 px: no resize
    meta = {"run": run_dir.name, "task": cfg.labels, "classes": classes,
            "parity": {"n": len(diffs), "max_abs_diff": max(diffs),
                       "passed": max(diffs) <= PARITY_TOLERANCE},
            "provenance": {**git_info(), "created_utc": datetime.now(UTC).isoformat(
                timespec="seconds")}}  # fmt: skip
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not meta["parity"]["passed"]:
        raise ValueError(f"ONNX parity failed: max |torch - onnx| {max(diffs):.2e}")
    return meta
