"""ONNX export of the tree surrogate (``vstent export-tree-surrogate``, P3).

The graph is padded: every input has a fixed number of branches (``max_branches``) and samples
(``n_points``); only the batch axis is dynamic. ``models/tree_surrogate/<run-name>/`` receives
``tree_surrogate.onnx`` (metadata checked by ``TreeSurrogateRunner``), the normalisation
statistics, the config and ``export_meta.json``; ``reports/tree_onnx.{md,json}`` gets the
parity check against PyTorch and the CPU latency.
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
from vstent.inference.tree_runtime import (
    TREE_INPUT_NAMES,
    TREE_MODELS,
    TREE_ONNX_FILE,
    TREE_OUTPUT,
    TreeSurrogateRunner,
)
from vstent.surrogate.export import OPSET, sha256
from vstent.surrogate.tree_features import (
    TREE_NORM_STATS,
    TreeNormStats,
    TreeRawInputs,
    prepare_tree_inputs,
)
from vstent.surrogate.tree_model import TreeFFRNet
from vstent.surrogate.tree_train import (
    TreeSurrogateConfig,
    load_tree_checkpoint,
    load_tree_raw,
    predict_tree_batches,
)
from vstent.sysinfo import cpu_name

PARITY_TOL = 1e-4
PARITY_SPLIT = "test"
PARITY_SAMPLES = 1000
LATENCY_BATCH = 256


def tree_metadata(stats: TreeNormStats, run_name: str) -> dict[str, str]:
    """String metadata embedded in the ONNX file (checked by ``TreeSurrogateRunner``)."""
    return {
        "vstent_version": __version__,
        "run": run_name,
        "tree_feature_version": str(stats.feature_version),
        "n_points": str(stats.n_points),
        "max_branches": str(stats.max_branches),
        "point_channels": ",".join(stats.point_channels),
        "branch_features": ",".join(stats.branch_features),
        "output": "FFR at every sample of every branch; FFR(0) = 1 at the inlet, continuous "
        "at junctions, non-increasing along every path; padded branches meaningless",
        "disclaimer": DISCLAIMER,
    }


def export_tree_onnx(
    model: TreeFFRNet, path: Path, stats: TreeNormStats, metadata: dict[str, str]
) -> Path:
    """Export with a dynamic batch axis (opset ``OPSET``), embed metadata, check the graph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m, n = stats.max_branches, stats.n_points
    c, g = len(stats.point_channels), len(stats.branch_features)
    parent = torch.zeros(2, m, m)
    parent[:, 1, 0] = 1.0
    attach = torch.zeros(2, m, n)
    attach[:, 1, n // 2] = 1.0
    mask = torch.zeros(2, m)
    mask[:, :2] = 1.0
    dummy = (torch.zeros(2, m, c, n), torch.zeros(2, m, g), parent, attach, mask)
    batch = {0: "batch"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model.cpu().eval(),
            dummy,
            str(path),
            dynamo=False,
            opset_version=OPSET,
            input_names=list(TREE_INPUT_NAMES),
            output_names=[TREE_OUTPUT],
            dynamic_axes={name: batch for name in (*TREE_INPUT_NAMES, TREE_OUTPUT)},
        )
    proto = onnx.load(str(path))
    onnx.helper.set_model_props(proto, metadata)
    onnx.checker.check_model(proto, full_check=True)
    onnx.save(proto, str(path))
    return path


def tree_parity(model: TreeFFRNet, runner: TreeSurrogateRunner, raw: TreeRawInputs) -> dict:
    """Max |FFR_torch − FFR_onnx| over every sample of every real branch (float32, CPU)."""
    xs = prepare_tree_inputs(raw, runner.stats)
    ref = predict_tree_batches(model.cpu(), [torch.from_numpy(x) for x in xs]).numpy()
    out = runner.predict_features(*xs)
    diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))[raw.mask]
    return {
        "n_trees": len(raw),
        "max_abs_diff": float(diff.max()),
        "max_abs_diff_outlet": float(diff[:, -1].max()),
        "tolerance": PARITY_TOL,
        "passed": bool(diff.max() < PARITY_TOL),
    }


def tree_latency(model_path: Path, raw: TreeRawInputs, repeats: int = 20) -> dict[str, Any]:
    """CPU latency: one tree (single-threaded session, default threads) and a batch."""
    one = raw.subset(slice(0, 1))
    out = {}
    for label, sample, kwargs in (
        ("one_tree_ms", one, {"workers": 1, "chunk_size": None}),
        (f"batch_{LATENCY_BATCH}_ms", raw, {}),
    ):
        runner = TreeSurrogateRunner(model_path, **kwargs)
        runner.predict(sample)  # warm-up
        times = []
        for _ in range(repeats if label == "one_tree_ms" else 5):
            t0 = time.perf_counter()
            runner.predict(sample)
            times.append(time.perf_counter() - t0)
        runner.close()
        out[label] = 1e3 * float(np.median(times))
    out["cpu"] = cpu_name()
    out["onnxruntime"] = ort.__version__
    return out


def _markdown(meta: dict[str, Any]) -> str:
    p, lat = meta["parity"], meta["latency"]
    return "\n".join(
        [
            "# Tree surrogate: ONNX export",
            "",
            f"> **{DISCLAIMER}**",
            "",
            "Generated by `vstent export-tree-surrogate`. Do not edit by hand.",
            "",
            "| Metric | Value | Target | Pass |",
            "|---|---|---|---|",
            f"| Max abs FFR difference, PyTorch vs ONNX Runtime ({p['n_trees']:,} test trees, "
            f"every sample of every real branch) | {p['max_abs_diff']:.2e} | < {p['tolerance']:g} "
            f"| {'yes' if p['passed'] else 'NO'} |",
            "",
            f"Padded graph: {meta['max_branches']} branches × {meta['n_points']} samples, dynamic "
            f"batch, opset {meta['opset']}, {meta['onnx_bytes'] / 1e6:.1f} MB.",
            "",
            "| CPU latency (median) | ms |",
            "|---|---|",
            f"| One tree | {lat['one_tree_ms']:.1f} |",
            f"| {LATENCY_BATCH} trees | {lat[f'batch_{LATENCY_BATCH}_ms']:.0f} |",
            "",
            f"CPU: {lat['cpu']}; ONNX Runtime {lat['onnxruntime']}. The tree solver runs about "
            "3,500 trees/s on one process (`reports/tree_physics.md`), so the surrogate is not "
            "faster than the reduced-order model; it exists for the pipeline that later takes "
            "3D-CFD labels (CLAUDE.md).",
            "",
            f"Run `{meta['run']}`; export commit `{meta['export']['git_commit']}` "
            f"(dirty: {meta['export']['git_dirty']}), {meta['export']['created_utc']}.",
            "",
        ]
    )


def export_tree_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    report_dir: str | Path = "reports",
    *,
    parity_samples: int = PARITY_SAMPLES,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Export a tree run's best model, check parity, time it; returns the export metadata."""
    log = log or (lambda _msg: None)
    provenance = {**git_info(), "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    run_dir = Path(run_dir)
    out_dir = Path(out_dir or TREE_MODELS / run_dir.name)
    cfg = load_config(run_dir / CONFIG_FILENAME, TreeSurrogateConfig)
    stats = TreeNormStats.load(run_dir / TREE_NORM_STATS)
    model = load_tree_checkpoint(run_dir, "cpu")
    path = export_tree_onnx(
        model, out_dir / TREE_ONNX_FILE, stats, tree_metadata(stats, run_dir.name)
    )
    for name in (TREE_NORM_STATS, CONFIG_FILENAME):
        shutil.copyfile(run_dir / name, out_dir / name)
    log(f"Exported {path} ({path.stat().st_size / 1e6:.1f} MB, opset {OPSET})")
    runner = TreeSurrogateRunner(path)
    raw, _ = load_tree_raw(cfg.data_dir, PARITY_SPLIT, parity_samples)
    check = {"split": PARITY_SPLIT, **tree_parity(model, runner, raw)}
    runner.close()
    log(
        f"Parity on {check['n_trees']:,} test trees: max |torch - onnx| {check['max_abs_diff']:.2e}"
    )
    batch = raw.subset(slice(0, LATENCY_BATCH))
    latency = tree_latency(path, batch)
    run_meta = load_yaml(run_dir / META_FILENAME)
    meta = {
        "run": run_dir.as_posix(),
        "run_git_commit": run_meta.get("git_commit"),
        "run_git_dirty": run_meta.get("git_dirty"),
        "export": provenance,
        "onnx_file": TREE_ONNX_FILE,
        "onnx_sha256": sha256(path),
        "onnx_bytes": path.stat().st_size,
        "opset": OPSET,
        "n_points": stats.n_points,
        "max_branches": stats.max_branches,
        "parity": check,
        "latency": latency,
        "versions": {
            "vstent": __version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },  # fmt: skip
    }
    (out_dir / "export_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8", newline="\n"
    )
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "tree_onnx.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8", newline="\n"
    )
    (report_dir / "tree_onnx.md").write_text(_markdown(meta), encoding="utf-8", newline="\n")
    return {**meta, "out_dir": out_dir.as_posix()}
