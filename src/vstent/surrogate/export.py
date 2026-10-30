"""ONNX export of a trained surrogate (``vstent export-surrogate``).

Writes ``models/surrogate/<run-name>/``: ``surrogate.onnx`` (opset 17, dynamic batch axis,
N fixed: resampling happens in pre-processing), copies of the run's ``norm_stats.json`` and
``config.yaml``, and ``export_meta.json`` (run, git commits, versions, SHA-256, parity vs torch).
"""

import hashlib
import json
import shutil
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
from vstent.inference.onnx_runtime import INPUT_NAMES, ONNX_FILE, OUTPUT_NAME, SurrogateRunner
from vstent.surrogate.dataset import load_raw
from vstent.surrogate.features import N_POINTS, NORM_STATS, NormStats, RawInputs, prepare_inputs
from vstent.surrogate.model import FFRUNet
from vstent.surrogate.train import SurrogateConfig, load_checkpoint, predict_batches

OPSET = 17
MODELS_DIR = Path("models/surrogate")
EXPORT_META = "export_meta.json"
PARITY_SPLIT = "test"
PARITY_SAMPLES = 1000
PARITY_TOL = 1e-4


def export_onnx(
    model: FFRUNet,
    path: str | Path,
    n_points: int = N_POINTS,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Export ``model`` (moved to CPU, eval mode) to ONNX with a dynamic batch axis.

    Uses the TorchScript exporter: the dynamo exporter cannot target opset 17 for this model
    (DECISIONS D16). ``metadata`` is stored in the model's ``metadata_props``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    cfg = model.config
    dummy = (torch.zeros(2, cfg.in_channels, n_points), torch.zeros(2, cfg.global_dim))
    batch = {0: "batch"}
    with warnings.catch_warnings():
        # GroupNorm's training-time batch-size check is traced as a constant (harmless; the
        # batch axis stays dynamic), and the TorchScript exporter itself is deprecated.
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model,
            dummy,
            str(path),
            dynamo=False,
            opset_version=OPSET,
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAMES[0]: batch, INPUT_NAMES[1]: batch, OUTPUT_NAME: batch},
        )
    proto = onnx.load(str(path))
    onnx.helper.set_model_props(proto, metadata or {})
    onnx.checker.check_model(proto, full_check=True)
    onnx.save(proto, str(path))
    return path


def model_metadata(stats: NormStats, run_name: str) -> dict[str, str]:
    """String metadata embedded in the ONNX file (checked by ``SurrogateRunner``)."""
    return {
        "vstent_version": __version__,
        "run": run_name,
        "feature_version": str(stats.feature_version),
        "n_points": str(stats.n_points),
        "point_channels": ",".join(stats.point_channels),
        "global_features": ",".join(stats.global_features),
        "output": "FFR(s), dimensionless, FFR(0) = 1, non-increasing",
        "disclaimer": DISCLAIMER,
    }


def sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parity(model: FFRUNet, runner: SurrogateRunner, raw: RawInputs) -> dict[str, Any]:
    """Max |FFR_torch − FFR_onnx| over all points of all vessels (both float32 on CPU)."""
    xp, xg = prepare_inputs(raw, runner.stats)
    ref = predict_batches(model.cpu(), torch.from_numpy(xp), torch.from_numpy(xg)).numpy()
    out = runner.predict(raw)
    diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    return {
        "n": len(raw),
        "max_abs_diff": float(diff.max()),
        "max_abs_diff_distal": float(diff[:, -1].max()),
        "tolerance": PARITY_TOL,
        "passed": bool(diff.max() < PARITY_TOL),
    }


def export_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    parity_samples: int = PARITY_SAMPLES,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Export a run's best model and check parity on ``parity_samples`` test vessels.

    Returns the export metadata (also written to ``<out_dir>/export_meta.json``).
    """
    log = log or (lambda _msg: None)
    provenance = {**git_info(), "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    run_dir = Path(run_dir)
    out_dir = Path(out_dir or MODELS_DIR / run_dir.name)
    cfg = load_config(run_dir / CONFIG_FILENAME, SurrogateConfig)
    stats = NormStats.load(run_dir / NORM_STATS)
    model = load_checkpoint(run_dir, "cpu")

    metadata = model_metadata(stats, run_dir.name)
    onnx_path = export_onnx(model, out_dir / ONNX_FILE, stats.n_points, metadata)
    for name in (NORM_STATS, CONFIG_FILENAME):
        shutil.copyfile(run_dir / name, out_dir / name)
    log(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB, opset {OPSET})")

    runner = SurrogateRunner(onnx_path)
    raw, _ = load_raw(cfg.data_dir, PARITY_SPLIT, parity_samples)
    check = {"split": PARITY_SPLIT, **parity(model, runner, raw)}
    log(
        f"Parity on {check['n']:,} {PARITY_SPLIT} vessels: max |torch - onnx| "
        f"{check['max_abs_diff']:.2e} (tolerance {PARITY_TOL:g})"
    )

    run_meta = load_yaml(run_dir / META_FILENAME)
    meta = {
        "run": run_dir.as_posix(),
        "run_git_commit": run_meta.get("git_commit"),
        "run_git_dirty": run_meta.get("git_dirty"),
        "export": provenance,
        "onnx_file": ONNX_FILE,
        "onnx_sha256": sha256(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "opset": OPSET,
        "versions": {
            "vstent": __version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "metadata": metadata,
        "parity": check,
    }
    (out_dir / EXPORT_META).write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return {**meta, "out_dir": out_dir.as_posix()}
