"""ONNX export of the vessel-segmentation model (``vstent export-seg``).

Writes ``models/seg/<run-name>/``: ``seg.onnx`` (opset 17, sigmoid included, dynamic batch,
fixed S × S input), a copy of the run's ``config.yaml`` and ``export_meta.json``. Parity with
torch (float32, CPU) and CPU latency go to ``reports/seg_onnx.{md,json}``.
"""

import json
import shutil
import time
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from vstent import DISCLAIMER, __version__
from vstent.config import CONFIG_FILENAME, META_FILENAME, git_info, load_config, load_yaml
from vstent.data.arcade import load_split
from vstent.imaging.seg_train import SegConfig, load_checkpoint, validation_tensors
from vstent.imaging.segment import THRESHOLD, dice_score, probability_to_image
from vstent.inference.onnx_runtime import SEG_INPUT, SEG_ONNX_FILE, SEG_OUTPUT, SegmentationRunner
from vstent.surrogate.export import sha256
from vstent.sysinfo import cpu_name

OPSET = 17
MODELS_DIR = Path("models/seg")
EXPORT_META = "export_meta.json"
PARITY_IMAGES = 16
PARITY_TOL = 1e-4
LATENCY_RUNS = 20


class _WithSigmoid(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def export_seg_onnx(
    model: torch.nn.Module, path: str | Path, image_size: int, metadata: dict[str, str]
) -> Path:
    """Export ``model`` (logits) + sigmoid to ONNX with a dynamic batch axis (TorchScript
    exporter at opset 17, as for the surrogate; DECISIONS D16)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _WithSigmoid(model.cpu().eval()).eval()
    dummy = torch.zeros(1, 1, image_size, image_size)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(path),
            dynamo=False,
            opset_version=OPSET,
            input_names=[SEG_INPUT],
            output_names=[SEG_OUTPUT],
            dynamic_axes={SEG_INPUT: {0: "batch"}, SEG_OUTPUT: {0: "batch"}},
        )
    proto = onnx.load(str(path))
    onnx.helper.set_model_props(proto, metadata)
    onnx.checker.check_model(proto, full_check=True)
    onnx.save(proto, str(path))
    return path


def export_seg_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    report_dir: str | Path = "reports",
    *,
    parity_images: int = PARITY_IMAGES,
    latency_runs: int = LATENCY_RUNS,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Export a segmentation run, check parity on validation images, time CPU inference."""
    log = log or (lambda _msg: None)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    run_dir, report_dir = Path(run_dir), Path(report_dir)
    out_dir = Path(out_dir or MODELS_DIR / run_dir.name)
    cfg = load_config(run_dir / CONFIG_FILENAME, SegConfig)
    size = cfg.model.image_size
    model = load_checkpoint(run_dir, "cpu")
    metadata = {
        "task": "vessel_segmentation",
        "vstent_version": __version__,
        "run": run_dir.name,
        "image_size": str(size),
        "encoder": cfg.model.encoder,
        "output": "vessel probability in [0, 1] (sigmoid applied)",
        "disclaimer": DISCLAIMER,
    }
    onnx_path = export_seg_onnx(model, out_dir / SEG_ONNX_FILE, size, metadata)
    shutil.copyfile(run_dir / CONFIG_FILENAME, out_dir / CONFIG_FILENAME)
    log(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB, opset {OPSET})")

    val = load_split(cfg.raw_dir, "syntax", "val", parity_images)
    x, _ = validation_tensors(val.images, val.masks, size)
    with torch.no_grad():
        ref_model = torch.sigmoid(model(x)).numpy()[:, 0]
    # back to the image size exactly as the runner does (a no-op for 512 px ARCADE images)
    ref = np.stack(
        [probability_to_image(p, im.shape) for p, im in zip(ref_model, val.images, strict=True)]
    )
    runner = SegmentationRunner(onnx_path)
    out = np.stack([runner.predict(img) for img in val.images])
    diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    agree = dice_score(out >= THRESHOLD, ref >= THRESHOLD)
    parity = {
        "n": len(val),
        "max_abs_diff": float(diff.max()),
        "tolerance": PARITY_TOL,
        "min_mask_dice_torch_vs_onnx": float(agree.min()),
        "passed": bool(diff.max() < PARITY_TOL),
    }
    log(
        f"Parity on {parity['n']} val images: max |torch - onnx| {parity['max_abs_diff']:.2e} "
        f"(tolerance {PARITY_TOL:g})"
    )
    runner.predict(val.images[0])  # warm-up
    times = []
    for k in range(latency_runs):
        t0 = time.perf_counter()
        runner.predict(val.images[k % len(val)])
        times.append((time.perf_counter() - t0) * 1e3)
    latency = {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "runs": latency_runs,
        "cpu": cpu_name(),
        "onnxruntime": ort.__version__,
    }
    log(f"CPU latency per {size} x {size} image: median {latency['median_ms']:.0f} ms")

    run_meta = load_yaml(run_dir / META_FILENAME)
    meta = {
        "run": run_dir.as_posix(),
        "run_git_commit": run_meta.get("git_commit"),
        "run_git_dirty": run_meta.get("git_dirty"),
        "export": provenance,
        "onnx_file": SEG_ONNX_FILE,
        "onnx_sha256": sha256(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "opset": OPSET,
        "metadata": metadata,
        "parity": parity,
        "latency": latency,
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__},
    }
    (out_dir / EXPORT_META).write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "seg_onnx.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8", newline="\n"
    )
    md = _markdown(meta, out_dir)
    (report_dir / "seg_onnx.md").write_text(md, encoding="utf-8", newline="\n")
    return {**meta, "out_dir": out_dir.as_posix()}


def _markdown(meta: dict[str, Any], out_dir: Path) -> str:
    par, lat, md = meta["parity"], meta["latency"], meta["metadata"]
    return "\n".join(
        [
            "# Segmentation model: ONNX export",
            "",
            f"> **{DISCLAIMER}**",
            "",
            "Generated by `vstent export-seg`. Do not edit by hand.",
            "",
            "| Check | Value | Target | Pass |",
            "|---|---|---|---|",
            f"| Parity: max abs(prob_torch - prob_onnx), {par['n']} val images, all pixels "
            f"| {par['max_abs_diff']:.1e} | < {par['tolerance']:g} "
            f"| {'yes' if par['passed'] else 'NO'} |",
            f"| Lowest Dice between torch and ONNX masks (threshold {THRESHOLD}) "
            f"| {par['min_mask_dice_torch_vs_onnx']:.4f} | — | — |",
            "",
            f"- CPU latency per {md['image_size']} × {md['image_size']} image (pre-processing + "
            f"inference + resize back): median {lat['median_ms']:.0f} ms, p95 "
            f"{lat['p95_ms']:.0f} ms over {lat['runs']} runs; {lat['cpu']}, ONNX Runtime "
            f"{lat['onnxruntime']} (default threads). No latency target is set for segmentation.",
            f"- `{out_dir.as_posix()}/{meta['onnx_file']}`: {meta['onnx_bytes'] / 1e6:.1f} MB, "
            f"opset {meta['opset']}, SHA-256 `{meta['onnx_sha256'][:16]}…`; U-Net with "
            f"`{md['encoder']}` encoder, sigmoid included, dynamic batch axis.",
            "",
            "## Provenance",
            "",
            f"- Training run `{meta['run']}`, commit `{meta['run_git_commit']}` "
            f"(dirty: {meta['run_git_dirty']})",
            f"- Export commit `{meta['export']['git_commit']}` "
            f"(dirty: {meta['export']['git_dirty']}), {meta['export']['created_utc']}",
            "",
        ]
    )
