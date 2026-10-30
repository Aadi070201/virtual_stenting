"""ONNX export of a trained surrogate (``vstent export-surrogate``).

Writes ``models/surrogate/<run-name>/``: ``surrogate.onnx`` (opset 17, dynamic batch axis,
N fixed: resampling happens in pre-processing), copies of the run's ``norm_stats.json`` and
``config.yaml``, and ``export_meta.json`` (run, git commits, versions, SHA-256, parity vs torch).

``vstent export-ensemble`` (P1, :func:`export_ensemble`) does the same for an ensemble run and
adds the members' spread, a screening model and the conformal calibration.
"""

import hashlib
import json
import shutil
import warnings
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from vstent import DISCLAIMER, __version__
from vstent.config import CONFIG_FILENAME, META_FILENAME, git_info, load_config, load_yaml
from vstent.inference.onnx_runtime import (
    INPUT_NAMES,
    ONNX_FILE,
    OUTPUT_NAME,
    SCREEN_ONNX_FILE,
    SPREAD_NAME,
    SurrogateRunner,
)
from vstent.surrogate.dataset import load_raw
from vstent.surrogate.ensemble import MEMBER_CONFIG, member_dirs
from vstent.surrogate.features import N_POINTS, NORM_STATS, NormStats, RawInputs, prepare_inputs
from vstent.surrogate.model import FFRUNet
from vstent.surrogate.train import (
    SUMMARY,
    SurrogateConfig,
    load_checkpoint,
    predict_batches,
)
from vstent.uncertainty.conformal import (
    CONFORMAL_FILE,
    ConformalCalibration,
    coverage,
    fit_conformal,
)
from vstent.uncertainty.greyzone import UncertaintyConfig

OPSET = 17
MODELS_DIR = Path("models/surrogate")
EXPORT_META = "export_meta.json"
PARITY_SPLIT = "test"
PARITY_SAMPLES = 1000
PARITY_TOL = 1e-4


class EnsembleFFR(torch.nn.Module):
    """Ensemble members → (mean, sample SD across members) of FFR, each (B, N).

    The mean of monotone curves with FFR(0) = 1 is itself monotone with FFR(0) = 1.
    """

    def __init__(self, members: list[FFRUNet]) -> None:
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.config = members[0].config

    def forward(
        self, x_point: torch.Tensor, x_global: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ffr = torch.stack([m(x_point, x_global) for m in self.members], dim=1)
        return ffr.mean(dim=1), ffr.std(dim=1)


def export_onnx(
    model: FFRUNet | EnsembleFFR,
    path: str | Path,
    n_points: int = N_POINTS,
    metadata: dict[str, str] | None = None,
    output_names: tuple[str, ...] = (OUTPUT_NAME,),
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
            output_names=list(output_names),
            dynamic_axes={name: batch for name in (*INPUT_NAMES, *output_names)},
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


def ensemble_parity(
    ensemble: EnsembleFFR, runner: SurrogateRunner, raw: RawInputs, batch_size: int = 256
) -> dict[str, Any]:
    """Max |torch − ONNX| of the ensemble mean and SD over all points of all vessels (CPU)."""
    xp, xg = prepare_inputs(raw, runner.stats)
    ensemble = ensemble.cpu().eval()
    with torch.no_grad():
        outs = [
            ensemble(
                torch.from_numpy(xp[i : i + batch_size]), torch.from_numpy(xg[i : i + batch_size])
            )
            for i in range(0, len(raw), batch_size)
        ]
    ref_mean = torch.cat([o[0] for o in outs]).numpy().astype(np.float64)
    ref_std = torch.cat([o[1] for o in outs]).numpy().astype(np.float64)
    mean, std = runner.predict_with_spread(raw)
    d_mean = float(np.abs(mean - ref_mean).max())
    d_std = float(np.abs(std - ref_std).max())
    return {
        "n": len(raw),
        "max_abs_diff": d_mean,
        "max_abs_diff_std": d_std,
        "tolerance": PARITY_TOL,
        "passed": bool(max(d_mean, d_std) < PARITY_TOL),
    }


def calibrate_conformal(
    runner: SurrogateRunner, data_dir: str | Path, config: UncertaintyConfig
) -> tuple[ConformalCalibration, dict[str, Any]]:
    """Split-conformal calibration of distal FFR on ``config.calibration_split`` with the
    exported ensemble; returns the calibration and its coverage on that split."""
    raw, ffr = load_raw(data_dir, config.calibration_split)
    mean, std = runner.predict_with_spread(raw)
    pred, sd, true = mean[:, -1], std[:, -1], ffr[:, -1]
    cal = fit_conformal(
        pred,
        sd,
        true,
        level=config.level,
        score=config.conformal_score,
        sigma_floor=config.sigma_floor,
        split=config.calibration_split,
    )
    return cal, coverage(cal, pred, sd, true)


def export_ensemble(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    uncertainty: UncertaintyConfig | None = None,
    parity_samples: int = PARITY_SAMPLES,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Export a trained ensemble (``vstent export-ensemble``) and calibrate its interval.

    Writes ``models/surrogate/<run-name>/``: ``surrogate.onnx`` (outputs ``ffr``, the members'
    mean, and ``ffr_std``, their SD), ``screen.onnx`` (the first member alone, for fast
    screening), ``norm_stats.json``, the ensemble and member configs, ``conformal.json``
    (split-conformal calibration of distal FFR) and ``export_meta.json``.
    """
    log = log or (lambda _msg: None)
    ucfg = uncertainty or UncertaintyConfig()
    provenance = {**git_info(), "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    run_dir = Path(run_dir)
    out_dir = Path(out_dir or MODELS_DIR / run_dir.name)
    members = member_dirs(run_dir)
    unfinished = [m.name for m in members if not (m / SUMMARY).exists()]
    if unfinished:
        raise ValueError(f"ensemble members not finished: {', '.join(unfinished)}")
    stats = [NormStats.load(m / NORM_STATS) for m in members]
    if any(s != stats[0] for s in stats[1:]):
        raise ValueError("ensemble members use different normalisation statistics")
    models = [load_checkpoint(m, "cpu") for m in members]
    ensemble = EnsembleFFR(models)
    base = load_config(run_dir / MEMBER_CONFIG, SurrogateConfig)

    common = model_metadata(stats[0], run_dir.name)
    ens_meta = {
        **common,
        "ensemble_size": str(len(models)),
        "members": ",".join(m.name for m in members),
        "output": "ffr: mean FFR(s) of the members (FFR(0) = 1, non-increasing); "
        "ffr_std: sample SD across members",
    }
    onnx_path = export_onnx(
        ensemble, out_dir / ONNX_FILE, stats[0].n_points, ens_meta, (OUTPUT_NAME, SPREAD_NAME)
    )
    screen_meta = {**common, "run": members[0].name, "role": "screening: one ensemble member"}
    screen_path = export_onnx(models[0], out_dir / SCREEN_ONNX_FILE, stats[0].n_points, screen_meta)
    shutil.copyfile(members[0] / NORM_STATS, out_dir / NORM_STATS)
    for name in (CONFIG_FILENAME, MEMBER_CONFIG):
        shutil.copyfile(run_dir / name, out_dir / name)
    log(
        f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB, {len(models)} members) "
        f"and {screen_path.name} (opset {OPSET})"
    )

    runner = SurrogateRunner(onnx_path)
    raw, _ = load_raw(base.data_dir, PARITY_SPLIT, parity_samples)
    check = {"split": PARITY_SPLIT, **ensemble_parity(ensemble, runner, raw)}
    check_screen = {"split": PARITY_SPLIT, **parity(models[0], runner.screen, raw)}
    log(
        f"Parity on {check['n']:,} {PARITY_SPLIT} vessels: mean {check['max_abs_diff']:.2e}, "
        f"SD {check['max_abs_diff_std']:.2e}, screen {check_screen['max_abs_diff']:.2e} "
        f"(tolerance {PARITY_TOL:g})"
    )
    cal, cal_cov = calibrate_conformal(runner, base.data_dir, ucfg)
    cal.save(out_dir / CONFORMAL_FILE)
    log(
        f"Conformal {cal.level:.0%} ({cal.score}) on {cal.n_calibration:,} {cal.split} vessels: "
        f"q = {cal.q:.3f}, coverage {cal_cov['coverage']:.2%}, median width "
        f"{cal_cov['width_median']:.4f}"
    )
    runner.close()

    run_meta = load_yaml(run_dir / META_FILENAME)
    meta = {
        "run": run_dir.as_posix(),
        "run_git_commit": run_meta.get("git_commit"),
        "run_git_dirty": run_meta.get("git_dirty"),
        "members": [
            {"run": m.as_posix(), "git_commit": load_yaml(m / META_FILENAME).get("git_commit")}
            for m in members
        ],
        "export": provenance,
        "onnx_file": ONNX_FILE,
        "onnx_sha256": sha256(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "screen_file": SCREEN_ONNX_FILE,
        "screen_sha256": sha256(screen_path),
        "opset": OPSET,
        "versions": {
            "vstent": __version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "metadata": ens_meta,
        "parity": check,
        "parity_screen": check_screen,
        "conformal": {**asdict(cal), "calibration_coverage": cal_cov},
        "uncertainty_config": ucfg.model_dump(mode="json"),
    }
    (out_dir / EXPORT_META).write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return {**meta, "out_dir": out_dir.as_posix()}
