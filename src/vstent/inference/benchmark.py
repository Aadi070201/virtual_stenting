"""CPU latency of the ONNX surrogate → ``reports/onnx_benchmark.{md,json}``.

End-to-end timings start from SI arrays (``RawInputs``) and include the shared
pre-processing (``features.prepare_inputs``) and ONNX Runtime inference. Every number in the
report is measured here.
"""

import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from vstent import DISCLAIMER, __version__
from vstent.config import CONFIG_FILENAME, git_info, load_config
from vstent.data.synth import SynthConfig, load_split
from vstent.inference.onnx_runtime import ONNX_FILE, SurrogateRunner
from vstent.physics.solver import solve_ffr_batch
from vstent.surrogate.dataset import RAW_KEYS, load_raw
from vstent.surrogate.ensemble import surrogate_config
from vstent.surrogate.features import NORM_STATS, RawInputs, prepare_inputs
from vstent.sysinfo import cpu_name

TARGET_SINGLE_MS = 10.0
TARGET_BATCH_MS = 300.0
BATCH_SIZE = 1024
SINGLE_RUNS = 200
BATCH_RUNS = 20
WARMUP = 5
BENCH_SPLIT = "test"
INT8_MAX_MAE_RISE = 0.002
INT8_MIN_SPEEDUP = 1.1


def time_ms(fn: Callable[[int], object], runs: int, warmup: int = WARMUP) -> np.ndarray:
    """Wall-clock times (ms) of ``fn(i)`` for i = 0..runs−1, after ``warmup`` untimed calls."""
    for i in range(warmup):
        fn(i)
    out = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter_ns()
        fn(i)
        out[i] = (time.perf_counter_ns() - t0) / 1e6
    return out


def _summary(ms: np.ndarray) -> dict[str, float]:
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(ms.min()),
        "runs": int(ms.size),
    }


def benchmark_runner(
    runner: SurrogateRunner,
    raw: RawInputs,
    *,
    batch_size: int = BATCH_SIZE,
    single_runs: int = SINGLE_RUNS,
    batch_runs: int = BATCH_RUNS,
) -> dict[str, Any]:
    """Median / p95 latency for one vessel (a different vessel each run) and for a batch.

    If ``raw`` has fewer than ``batch_size`` vessels, they are repeated.
    """
    singles = [raw.subset(i % len(raw)) for i in range(single_runs)]
    single_feats = [prepare_inputs(v, runner.stats) for v in singles]
    batch = raw.subset(np.arange(batch_size) % len(raw))
    batch_feats = prepare_inputs(batch, runner.stats)
    n = len(singles)
    return {
        "single_end_to_end": _summary(time_ms(lambda i: runner.predict(singles[i % n]), n)),
        "single_inference": _summary(
            time_ms(lambda i: runner.predict_features(*single_feats[i % n]), n)
        ),
        "batch_end_to_end": _summary(time_ms(lambda _i: runner.predict(batch), batch_runs)),
        "batch_inference": _summary(
            time_ms(lambda _i: runner.predict_features(*batch_feats), batch_runs)
        ),
        "batch_size": batch_size,
    }


def benchmark_solver(raw: RawInputs, v_hyp: np.ndarray, synth: SynthConfig, runs: int) -> dict:
    """Latency of the reduced-order batch solver on the same vessels (context only)."""
    ds = raw.length / (raw.r.shape[1] - 1)

    def solve(_i: int) -> None:
        solve_ffr_batch(raw.r, ds, raw.r_ref, synth.physics, pa=raw.pa, pv=raw.pv, v_hyp=v_hyp)

    return {**_summary(time_ms(solve, runs)), "batch_size": len(raw)}


def _batch_median_ms(runner: SurrogateRunner, batch: RawInputs, runs: int) -> float:
    return float(np.median(time_ms(lambda _i: runner.predict(batch), runs)))


def int8_check(
    model_dir: str | Path, data_dir: str | Path, *, batch_size: int, batch_runs: int
) -> dict[str, Any]:
    """Optional dynamic INT8 quantisation (unsigned 8-bit weights) compared with FP32.

    Accuracy: distal-FFR MAE vs the solver on the whole test split. Speed: median end-to-end
    latency of a batch with the runner defaults. Kept only if the MAE rises by less than
    ``INT8_MAX_MAE_RISE`` and the batch is at least ``INT8_MIN_SPEEDUP`` times faster; the
    INT8 model itself is written to a temporary directory and not kept on disk.
    """
    model_dir = Path(model_dir)
    raw, ffr = load_raw(data_dir, BENCH_SPLIT)
    batch = raw.subset(np.arange(batch_size) % len(raw))
    preds, batch_ms = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logging.disable(logging.WARNING)  # quantize_dynamic logs pre-processing advice
        try:
            quantize_dynamic(model_dir / ONNX_FILE, tmp / ONNX_FILE, weight_type=QuantType.QUInt8)
        finally:
            logging.disable(logging.NOTSET)
        shutil.copyfile(model_dir / NORM_STATS, tmp / NORM_STATS)
        int8_bytes = (tmp / ONNX_FILE).stat().st_size
        for name, path in (("fp32", model_dir), ("int8", tmp)):
            runner = SurrogateRunner.from_dir(path)
            preds[name] = runner.predict(raw)
            batch_ms[name] = _batch_median_ms(runner, batch, batch_runs)
            runner.close()
            del runner  # release the model file before the temporary directory is removed
    mae = {k: float(np.abs(p[:, -1] - ffr[:, -1]).mean()) for k, p in preds.items()}
    rise = mae["int8"] - mae["fp32"]
    speedup = batch_ms["fp32"] / batch_ms["int8"]
    return {
        "weight_type": "QUInt8",
        "n": len(raw),
        "distal_mae_fp32": mae["fp32"],
        "distal_mae_int8": mae["int8"],
        "mae_rise": rise,
        "max_abs_diff": float(np.abs(preds["int8"] - preds["fp32"]).max()),
        "batch_size": batch_size,
        "batch_median_ms_fp32": batch_ms["fp32"],
        "batch_median_ms_int8": batch_ms["int8"],
        "speedup": speedup,
        "fp32_bytes": (model_dir / ONNX_FILE).stat().st_size,
        "int8_bytes": int8_bytes,
        "accuracy_ok": rise < INT8_MAX_MAE_RISE,
        "kept": bool(rise < INT8_MAX_MAE_RISE and speedup >= INT8_MIN_SPEEDUP),
    }


def runner_settings(runner: SurrogateRunner) -> dict[str, Any]:
    """How a runner splits and threads its work."""
    return {
        "workers": runner.workers,
        "chunk_size": runner.chunk_size,
        "intra_op_threads": runner.intra_op_threads or "ORT default",
    }


def _environment(runner: SurrogateRunner) -> dict[str, Any]:
    return {
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "providers": runner.session.get_providers(),
        "onnxruntime": ort.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def run_benchmark(
    model_dir: str | Path,
    report_dir: str | Path = "reports",
    *,
    export_meta: dict[str, Any],
    batch_size: int = BATCH_SIZE,
    single_runs: int = SINGLE_RUNS,
    batch_runs: int = BATCH_RUNS,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Benchmark the exported model on test vessels and write the ONNX report."""
    log = log or (lambda _msg: None)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    run_cfg = surrogate_config(model_dir)
    data_dir = Path(run_cfg.data_dir)
    arrays = load_split(data_dir, BENCH_SPLIT, (*RAW_KEYS, "v_hyp"))
    n = min(batch_size, arrays["r"].shape[0])
    arrays = {k: v[:n] for k, v in arrays.items()}
    raw = RawInputs.from_arrays(arrays)

    # The runner's defaults (what the app uses) and, for comparison, one ONNX call per batch
    # with ONNX Runtime's default threading.
    configs = {
        "default": SurrogateRunner.from_dir(model_dir),
        "one_call": SurrogateRunner.from_dir(model_dir, workers=1, chunk_size=None),
    }
    timings = {}
    for name, runner in configs.items():
        timings[name] = {
            **benchmark_runner(
                runner, raw, batch_size=batch_size, single_runs=single_runs, batch_runs=batch_runs
            ),
            "settings": runner_settings(runner),
        }
        runner.close()
    runner = configs["default"]
    synth = load_config(data_dir / CONFIG_FILENAME, SynthConfig)
    solver = benchmark_solver(raw, arrays["v_hyp"], synth, batch_runs)
    int8 = int8_check(model_dir, data_dir, batch_size=batch_size, batch_runs=batch_runs)
    log(
        f"INT8 (optional): distal MAE {int8['distal_mae_int8']:.4f} vs FP32 "
        f"{int8['distal_mae_fp32']:.4f}, speed-up {int8['speedup']:.2f}x -> "
        f"{'kept' if int8['kept'] else 'not kept'}"
    )
    single = timings["default"]["single_end_to_end"]
    batch = timings["default"]["batch_end_to_end"]
    par = export_meta["parity"]
    targets = [
        {
            "metric": f"Parity: max abs(FFR_torch - FFR_onnx) over {par['n']:,} {par['split']} "
            "vessels, all points",
            "value": f"{par['max_abs_diff']:.1e}",
            "target": f"< {par['tolerance']:g}",
            "passed": par["passed"],
        },
        {
            "metric": "Single-vessel latency, median (end-to-end)",
            "value": f"{single['median_ms']:.2f} ms",
            "target": f"< {TARGET_SINGLE_MS:g} ms",
            "passed": single["median_ms"] < TARGET_SINGLE_MS,
        },
        {
            "metric": f"Batch of {batch_size:,} vessels, median (end-to-end)",
            "value": f"{batch['median_ms']:.1f} ms",
            "target": f"< {TARGET_BATCH_MS:g} ms",
            "passed": batch["median_ms"] < TARGET_BATCH_MS,
        },
    ]
    for t in targets[1:]:
        log(f"{t['metric']}: {t['value']}")
    report = {
        "targets": targets,
        "targets_met": all(t["passed"] for t in targets),
        "timings": timings,
        "solver": solver,
        "int8": int8,
        "environment": _environment(runner),
        "model": {k: export_meta[k] for k in ("onnx_file", "onnx_sha256", "onnx_bytes", "opset")}
        | {
            "dir": model_dir.as_posix(),
            "n_points": runner.n_points,
            "base_channels": run_cfg.model.base_channels,
            "versions": export_meta["versions"],
        },
        "data": {"dir": data_dir.as_posix(), "split": BENCH_SPLIT, "n_vessels": n},
        "provenance": {
            "run": export_meta["run"],
            "run_git_commit": export_meta["run_git_commit"],
            "run_git_dirty": export_meta["run_git_dirty"],
            "export": export_meta["export"],
            "report": provenance,
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "onnx_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    (report_dir / "onnx_benchmark.md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    return report


def _markdown(report: dict[str, Any]) -> str:
    t, env, model, prov = (
        report["timings"],
        report["environment"],
        report["model"],
        report["provenance"],
    )
    bs = t["default"]["batch_size"]
    lines = [
        "# ONNX surrogate: export parity and CPU latency",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent export-surrogate`. Do not edit by hand.",
        "",
        "End-to-end = pre-processing with `surrogate.features.prepare_inputs` (the same function "
        "as in training) + ONNX Runtime inference, from SI arrays to the FFR pullback curve "
        f"(N = {model['n_points']} points).",
        "",
        "## Acceptance targets",
        "",
        "| Metric | Value | Target | Pass |",
        "|---|---|---|---|",
    ]
    for row in report["targets"]:
        lines.append(
            f"| {row['metric']} | {row['value']} | {row['target']} "
            f"| {'yes' if row['passed'] else 'NO'} |"
        )
    d = t["default"]["settings"]
    act_mb = bs * model["base_channels"] * model["n_points"] * 4 / 1e6  # float32, first level
    lines += [
        "",
        "## Latency",
        "",
        f"- **Runner default** (used for the targets and by the app): batches split into chunks "
        f"of {d['chunk_size']} vessels, run in parallel on {d['workers']} threads, "
        f"{d['intra_op_threads']} ONNX Runtime thread per call.",
        "- **One call**: the whole batch in a single ONNX Runtime call with its default "
        "intra-op threading (all physical cores).",
        "",
        "| Configuration | Case | Median | p95 | Min | Runs |",
        "|---|---|---|---|---|---|",
    ]
    for config, config_label in (("default", "Runner default"), ("one_call", "One call")):
        for key, label in (
            ("single_end_to_end", "1 vessel, end-to-end"),
            ("single_inference", "1 vessel, inference only"),
            ("batch_end_to_end", f"{bs:,} vessels, end-to-end"),
            ("batch_inference", f"{bs:,} vessels, inference only"),
        ):
            s = t[config][key]
            lines.append(
                f"| {config_label} | {label} | {s['median_ms']:.2f} ms | {s['p95_ms']:.2f} ms "
                f"| {s['min_ms']:.2f} ms | {s['runs']} |"
            )
    sv, q = report["solver"], report["int8"]
    lines += [
        "",
        "Single-vessel runs use a different test vessel each time; batch runs repeat the same "
        f"{bs:,} vessels. {WARMUP} untimed warm-up calls precede each case. With one call on "
        f"the whole batch, the activations (≈ {act_mb:.0f} MB per tensor at full resolution) no "
        "longer fit in the CPU cache and the normalisation / element-wise operations become "
        "memory-bound; small chunks keep them in cache (DECISIONS D17).",
        "",
        "## Context: reduced-order solver",
        "",
        f"The reduced-order batch solver computes the same {sv['batch_size']:,} vessels in "
        f"{sv['median_ms']:.1f} ms (median of {sv['runs']} runs, same CPU). In this MVP the solver "
        "is already fast and the surrogate is **not** claimed to be faster: it exists to build "
        "and validate the training → ONNX → app pipeline, so that in Phase 2 its labels can be "
        "swapped for 3D CFD, where a surrogate gives the real speed-up.",
        "",
        "## Optional: dynamic INT8 quantisation",
        "",
        "Weights quantised to unsigned 8-bit with ONNX Runtime `quantize_dynamic` (activations "
        "quantised at run time). Kept only if the distal-FFR MAE rises by less than "
        f"{INT8_MAX_MAE_RISE} and a batch runs at least {INT8_MIN_SPEEDUP}× faster. FP32 is "
        "timed again here, just before INT8, so its latency differs slightly from the table "
        "above (run-to-run variation).",
        "",
        "| | FP32 | INT8 |",
        "|---|---|---|",
        f"| Distal FFR MAE vs solver ({q['n']:,} test vessels) | {q['distal_mae_fp32']:.4f} "
        f"| {q['distal_mae_int8']:.4f} ({q['mae_rise']:+.4f}) |",
        f"| Max abs. difference from FP32, any point | — | {q['max_abs_diff']:.4f} |",
        f"| {q['batch_size']:,} vessels, median end-to-end | {q['batch_median_ms_fp32']:.1f} ms "
        f"| {q['batch_median_ms_int8']:.1f} ms ({q['speedup']:.2f}×) |",
        f"| File size | {q['fp32_bytes'] / 1e6:.2f} MB | {q['int8_bytes'] / 1e6:.2f} MB |",
        "",
        f"**INT8 kept: {'yes' if q['kept'] else 'no'}** "
        f"(accuracy criterion {'met' if q['accuracy_ok'] else 'not met'}; speed-up "
        f"{q['speedup']:.2f}× vs ≥ {INT8_MIN_SPEEDUP}× required). "
        + (
            "INT8 meets both criteria, but the app still uses FP32 until it is switched over."
            if q["kept"]
            else "The app uses the FP32 model."
        ),
        "",
        "## Environment",
        "",
        f"- CPU: {env['cpu']} ({env['logical_cpus']} logical CPUs)",
        f"- ONNX Runtime {env['onnxruntime']} ({', '.join(env['providers'])}); "
        f"Python {env['python']}; {env['platform']}",
        "",
        "## Model",
        "",
        f"- `{model['dir']}/{model['onnx_file']}`: {model['onnx_bytes'] / 1e6:.2f} MB, "
        f"opset {model['opset']}, SHA-256 `{model['onnx_sha256'][:16]}…`",
        f"- Inputs `x_point` (batch, 4, {model['n_points']}) and `x_global` (batch, 5), "
        f"float32, standardised; output `ffr` (batch, {model['n_points']}). Batch axis dynamic.",
        f"- Exported with torch {model['versions']['torch']}, onnx {model['versions']['onnx']}.",
        "",
        "## Provenance",
        "",
        f"- Training run `{prov['run']}`, commit `{prov['run_git_commit']}` "
        f"(dirty: {prov['run_git_dirty']})",
        f"- Export commit `{prov['export']['git_commit']}` "
        f"(dirty: {prov['export']['git_dirty']}), {prov['export']['created_utc']}",
        f"- Report commit `{prov['report']['git_commit']}` "
        f"(dirty: {prov['report']['git_dirty']}), vstent {prov['report']['vstent_version']}, "
        f"{prov['report']['created_utc']}",
        "",
        "## Limitations",
        "",
        "- Wall-clock timings on one laptop; they vary with CPU load, power settings and "
        "thread count.",
        "- Accuracy of the surrogate itself is reported in `reports/surrogate_eval.md`.",
        "",
    ]
    return "\n".join(lines)
