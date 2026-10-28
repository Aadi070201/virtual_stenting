"""Surrogate evaluation against the solver (``vstent eval-surrogate``).

Writes ``reports/surrogate_eval.{md,json}``, figures ``reports/figures/surrogate_*.png`` and a
copy of the metrics to ``<run>/eval.json``. Every number is computed here.
"""

import csv
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from matplotlib.figure import Figure

from vstent import DISCLAIMER, __version__
from vstent.config import META_FILENAME, git_info, load_config
from vstent.surrogate.baseline import fit_baseline, hand_features
from vstent.surrogate.dataset import load_raw
from vstent.surrogate.features import NormStats, prepare_inputs
from vstent.surrogate.train import (
    METRICS,
    NORM_STATS,
    SUMMARY,
    SurrogateConfig,
    load_checkpoint,
    predict_batches,
    resolve_device,
)

CUTOFF = 0.80
TARGET_DISTAL_MAE = 0.015
TARGET_LOA = 0.04
TARGET_AGREEMENT = 0.97
DS_BINS = (0.0, 30.0, 50.0, 70.0, 100.0)
SPLITS = ("val", "test", "ood")
EXAMPLE_TARGETS = (0.95, 0.85, 0.75, 0.55)

# Chart tokens (reference palette, light mode). Surrogate = slot 1, baseline = slot 2.
SURROGATE = "#2a78d6"
BASELINE = "#eb6834"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"


def distal_metrics(pred: np.ndarray, true: np.ndarray, cutoff: float = CUTOFF) -> dict[str, float]:
    """Error and Bland–Altman statistics of distal FFR, plus agreement at ``cutoff``.

    Positive class = FFR < cutoff (haemodynamically significant).
    """
    err = pred - true
    bias, sd = float(err.mean()), float(err.std(ddof=1))
    pos_true, pos_pred = true < cutoff, pred < cutoff
    return {
        "n": int(err.size),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "max_abs_error": float(np.abs(err).max()),
        "bias": bias,
        "sd": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "agreement": float(np.mean(pos_true == pos_pred)),
        "sensitivity": float(np.mean(pos_pred[pos_true])) if pos_true.any() else float("nan"),
        "specificity": float(np.mean(~pos_pred[~pos_true])) if (~pos_true).any() else float("nan"),
    }


def mae_by_ds_bin(pred: np.ndarray, true: np.ndarray, max_ds: np.ndarray) -> list[dict[str, Any]]:
    """Distal MAE in bins of the vessel's maximum measured DS (%)."""
    out = []
    for lo, hi in zip(DS_BINS[:-1], DS_BINS[1:], strict=True):
        m = (max_ds >= lo) & (max_ds < hi)
        out.append(
            {
                "bin": f"{lo:g}-{hi:g}%",
                "n": int(m.sum()),
                "mae": float(np.abs(pred[m] - true[m]).mean()) if m.any() else float("nan"),
            }
        )
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, lw=0.6, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=7.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def _figure(width: float, height: float, title: str, subtitle: str) -> Figure:
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE)
    fig.text(0.07, 0.94, title, fontsize=11, color=INK_PRIMARY)
    fig.text(0.07, 0.895, subtitle, fontsize=8, color=INK_SECONDARY)
    fig.text(0.07, 0.02, DISCLAIMER, fontsize=7, color=INK_MUTED)
    return fig


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def plot_scatter(true, pred_sur, pred_base, path: Path) -> None:
    fig = _figure(
        8.0,
        4.4,
        "Predicted vs solver distal FFR (test)",
        "Each dot is one vessel; diagonal = perfect agreement; dashed = 0.80 cut-off",
    )
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.18})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.14)
    for ax, pred, color, label in (
        (axes[0], pred_sur, SURROGATE, "Surrogate (1D U-Net)"),
        (axes[1], pred_base, BASELINE, "Baseline (gradient boosting)"),
    ):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1.0, zorder=1)
        ax.axvline(CUTOFF, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axhline(CUTOFF, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.scatter(true, pred, s=2, color=color, alpha=0.25, linewidths=0, zorder=2)
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.set_title(label, loc="left", fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel("Solver distal FFR", fontsize=8, color=INK_SECONDARY)
    axes[0].set_ylabel("Predicted distal FFR", fontsize=8, color=INK_SECONDARY)
    _save(fig, path)


def plot_bland_altman(true, pred, stats: dict[str, float], path: Path) -> None:
    fig = _figure(
        7.2,
        4.4,
        "Bland–Altman: surrogate vs solver distal FFR (test)",
        f"Shaded band = target limits of agreement ±{TARGET_LOA}",
    )
    ax = fig.add_axes((0.10, 0.14, 0.76, 0.68))
    _style(ax)
    mean, diff = (true + pred) / 2, pred - true
    ax.axhspan(-TARGET_LOA, TARGET_LOA, color=BAND, zorder=0.5)
    ax.scatter(mean, diff, s=2, color=SURROGATE, alpha=0.25, linewidths=0, zorder=2)
    for y, name, ls in (
        (stats["bias"], "bias", "-"),
        (stats["loa_low"], "−1.96 SD", (0, (4, 3))),
        (stats["loa_high"], "+1.96 SD", (0, (4, 3))),
    ):
        ax.axhline(y, color=INK_SECONDARY, lw=1.0, ls=ls, zorder=3)
        ax.annotate(
            f"{name} {y:+.4f}",
            xy=(1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK_SECONDARY,
            annotation_clip=False,
        )
    lim = max(TARGET_LOA * 1.5, float(np.abs(diff).max()) * 1.05)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Mean of surrogate and solver distal FFR", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Surrogate − solver", fontsize=8, color=INK_SECONDARY)
    _save(fig, path)


def plot_error_by_ds(by_sur: list[dict], by_base: list[dict], path: Path) -> None:
    fig = _figure(
        7.2,
        4.2,
        "Distal FFR error by maximum diameter stenosis (test)",
        "Mean absolute error vs the solver; bins of the vessel's maximum measured %DS",
    )
    ax = fig.add_axes((0.10, 0.14, 0.86, 0.66))
    _style(ax)
    ax.grid(False, axis="x")
    x = np.arange(len(by_sur))
    w = 0.38
    for offset, rows, color, label in (
        (-w / 2 - 0.01, by_sur, SURROGATE, "Surrogate"),
        (w / 2 + 0.01, by_base, BASELINE, "Baseline"),
    ):
        ax.bar(x + offset, [r["mae"] for r in rows], width=w, color=color, label=label, zorder=2)
    ax.set_xticks(x, [f"{r['bin']}\nn = {r['n']:,}" for r in by_sur])
    ax.set_ylabel("Distal FFR MAE", fontsize=8, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    _save(fig, path)


def plot_examples(s_norm, true_curves, pred_curves, idx, path: Path) -> None:
    fig = _figure(
        8.0,
        5.0,
        "FFR pullback curves: solver vs surrogate (test examples)",
        "Position along the vessel (0 = ostium, 1 = distal end)",
    )
    axes = fig.subplots(2, 2, gridspec_kw={"hspace": 0.45, "wspace": 0.15})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.10)
    for k, (ax, i) in enumerate(zip(axes.flat, idx, strict=True)):
        _style(ax)
        ax.plot(s_norm, true_curves[i], color=INK_SECONDARY, lw=2.0, label="Solver", zorder=2)
        ax.plot(
            s_norm,
            pred_curves[i],
            color=SURROGATE,
            lw=2.0,
            ls=(0, (5, 2)),
            label="Surrogate",
            zorder=3,
        )
        ax.axhline(CUTOFF, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_ylim(min(0.5, float(true_curves[i].min()) - 0.05), 1.02)
        ax.set_title(
            f"distal FFR: solver {true_curves[i, -1]:.3f}, surrogate {pred_curves[i, -1]:.3f}",
            loc="left",
            fontsize=8,
            color=INK_SECONDARY,
        )
        if k == 0:
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc="lower left")
    _save(fig, path)


def plot_training(run_dir: Path, path: Path) -> None:
    with (run_dir / METRICS).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    mae = [float(r["val_distal_mae"]) for r in rows]
    fig = _figure(
        7.2,
        4.0,
        "Validation distal FFR MAE per epoch",
        f"Dashed = target {TARGET_DISTAL_MAE} (evaluated on the test split)",
    )
    ax = fig.add_axes((0.10, 0.14, 0.86, 0.66))
    _style(ax)
    ax.plot(epochs, mae, color=SURROGATE, lw=2.0, marker="o", ms=3, zorder=2)
    ax.axhline(TARGET_DISTAL_MAE, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Distal FFR MAE (log scale)", fontsize=8, color=INK_SECONDARY)
    _save(fig, path)


def _targets(test: dict[str, Any]) -> list[dict[str, Any]]:
    m = test["surrogate"]
    return [
        {
            "metric": "Distal FFR MAE",
            "value": f"{m['mae']:.4f}",
            "target": f"≤ {TARGET_DISTAL_MAE}",
            "passed": m["mae"] <= TARGET_DISTAL_MAE,
        },
        {
            "metric": "95% limits of agreement",
            "value": f"[{m['loa_low']:+.4f}, {m['loa_high']:+.4f}]",
            "target": f"within ±{TARGET_LOA}",
            "passed": m["loa_low"] >= -TARGET_LOA and m["loa_high"] <= TARGET_LOA,
        },
        {
            "metric": f"Agreement at {CUTOFF}",
            "value": f"{m['agreement']:.2%}",
            "target": f"≥ {TARGET_AGREEMENT:.0%}",
            "passed": m["agreement"] >= TARGET_AGREEMENT,
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    tr, sp, prov = report["training"], report["splits"], report["provenance"]
    lines = [
        "# Surrogate evaluation (1D U-Net vs reduced-order solver)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent eval-surrogate`. Do not edit by hand.",
        "",
        "The reference is the reduced-order solver on the same synthetic vessels; these numbers "
        "measure how well the surrogate reproduces that model, not clinical accuracy. In this MVP "
        "the solver is already fast: the surrogate validates the training → ONNX → app pipeline "
        "for Phase 2 (3D-CFD labels). No speed-up over the solver is claimed.",
        "",
        "## Acceptance targets (test split)",
        "",
        "| Metric | Value | Target | Pass |",
        "|---|---|---|---|",
    ]
    for t in report["targets"]:
        lines.append(
            f"| {t['metric']} | {t['value']} | {t['target']} | {'yes' if t['passed'] else 'NO'} |"
        )
    lines += [
        "",
        "## Distal FFR metrics",
        "",
        "Positive class for sensitivity / specificity: FFR < 0.80 according to the solver.",
        "",
        "| Split | Model | n | MAE | Bias | 95% LoA | Agreement | Sens. | Spec. | Max error "
        "| Curve MAE |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split, res in sp.items():
        for model in ("surrogate", "baseline"):
            m = res[model]
            curve = f"{m['curve_mae']:.4f}" if "curve_mae" in m else "n/a"
            lines.append(
                f"| {split} | {model} | {m['n']:,} | {m['mae']:.4f} | {m['bias']:+.4f} "
                f"| [{m['loa_low']:+.4f}, {m['loa_high']:+.4f}] | {m['agreement']:.2%} "
                f"| {m['sensitivity']:.2%} | {m['specificity']:.2%} | {m['max_abs_error']:.3f} "
                f"| {curve} |"
            )
    lines += [
        "",
        "`ood` = 80–120 mm vessels with 3–4 lesions, outside the training distribution "
        "(reported, no target).",
        "",
        "## Error by stenosis severity (test)",
        "",
        "| Max DS bin | n | Surrogate MAE | Baseline MAE |",
        "|---|---|---|---|",
    ]
    test = sp["test"]
    for a, b in zip(test["surrogate"]["by_ds"], test["baseline"]["by_ds"], strict=True):
        lines.append(f"| {a['bin']} | {a['n']:,} | {a['mae']:.4f} | {b['mae']:.4f} |")
    lines += [
        "",
        "## Figures",
        "",
        "![Predicted vs solver](figures/surrogate_scatter.png)",
        "",
        "![Bland–Altman](figures/surrogate_bland_altman.png)",
        "",
        "![Error by stenosis](figures/surrogate_error_by_ds.png)",
        "",
        "![Pullback examples](figures/surrogate_examples.png)",
        "",
        "![Training curve](figures/surrogate_training.png)",
        "",
        "## Training",
        "",
        f"- Run `{report['run']}`: {tr['n_parameters']:,} parameters, {tr['n_train']:,} train / "
        f"{tr['n_val']:,} val samples, best epoch {tr['best_epoch']} of {tr['epochs_run']}, "
        f"{tr['train_seconds'] / 60:.1f} min on {tr['gpu'] or tr['device']} (AMP: {tr['amp']}).",
        f"- Baseline: histogram gradient boosting on {', '.join(report['baseline_features'])}, "
        f"fitted in {report['baseline_fit_seconds']:.1f} s.",
        "",
        "## Provenance",
        "",
        f"- Training run commit: `{report['run_provenance'].get('git_commit')}` "
        f"(dirty: {report['run_provenance'].get('git_dirty')})",
        f"- Evaluation commit: `{prov['git_commit']}` (dirty: {prov['git_dirty']}), "
        f"vstent {prov['vstent_version']}, {prov['created_utc']}",
        "",
        "## Limitations",
        "",
        "- Synthetic single-vessel data only; labels come from the reduced-order model.",
        "- Accuracy on real angiography-derived profiles (M5) is not measured here.",
        "",
    ]
    return "\n".join(lines)


def evaluate_run(
    run_dir: str | Path,
    report_dir: str | Path = "reports",
    *,
    device: str = "auto",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate a training run on val / test / ood against the solver and the baseline."""
    log = log or (lambda _msg: None)
    run_dir, report_dir = Path(run_dir), Path(report_dir)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    cfg = load_config(run_dir / "config.yaml", SurrogateConfig)
    stats = NormStats.load(run_dir / NORM_STATS)
    dev = resolve_device(device)
    model = load_checkpoint(run_dir, dev)

    t0 = time.perf_counter()
    raw_train, ffr_train = load_raw(cfg.data_dir, "train", cfg.train.max_train_samples)
    baseline = fit_baseline(raw_train, ffr_train[:, -1], seed=cfg.seed)
    baseline_seconds = time.perf_counter() - t0
    log(f"Baseline fitted in {baseline_seconds:.1f} s")

    splits: dict[str, Any] = {}
    test_arrays = None
    for split in SPLITS:
        raw, ffr = load_raw(cfg.data_dir, split)
        xp, xg = prepare_inputs(raw, stats)
        pred = (
            predict_batches(model, torch.from_numpy(xp).to(dev), torch.from_numpy(xg).to(dev))
            .cpu()
            .numpy()
        )
        feats = hand_features(raw)
        base = baseline.predict(feats)
        true = ffr[:, -1].astype(np.float64)
        splits[split] = {
            "surrogate": {
                **distal_metrics(pred[:, -1].astype(np.float64), true),
                "curve_mae": float(np.abs(pred - ffr).mean()),
                "by_ds": mae_by_ds_bin(pred[:, -1], true, feats[:, 0]),
            },
            "baseline": {
                **distal_metrics(base, true),
                "by_ds": mae_by_ds_bin(base, true, feats[:, 0]),
            },
        }
        log(
            f"{split}: surrogate distal MAE {splits[split]['surrogate']['mae']:.4f}, "
            f"baseline {splits[split]['baseline']['mae']:.4f}"
        )
        if split == "test":
            test_arrays = (ffr, pred, base)

    ffr, pred, base = test_arrays
    fig_dir = report_dir / "figures"
    plot_scatter(ffr[:, -1], pred[:, -1], base, fig_dir / "surrogate_scatter.png")
    plot_bland_altman(
        ffr[:, -1], pred[:, -1], splits["test"]["surrogate"], fig_dir / "surrogate_bland_altman.png"
    )
    plot_error_by_ds(
        splits["test"]["surrogate"]["by_ds"],
        splits["test"]["baseline"]["by_ds"],
        fig_dir / "surrogate_error_by_ds.png",
    )
    idx = [int(np.argmin(np.abs(ffr[:, -1] - t))) for t in EXAMPLE_TARGETS]
    plot_examples(
        np.linspace(0, 1, ffr.shape[1]), ffr, pred, idx, fig_dir / "surrogate_examples.png"
    )
    plot_training(run_dir, fig_dir / "surrogate_training.png")

    report = {
        "run": run_dir.as_posix(),
        "targets": _targets(splits["test"]),
        "splits": splits,
        "training": json.loads((run_dir / SUMMARY).read_text(encoding="utf-8")),
        "baseline_features": ["max DS", "lesion length (DS > 30%)", "D0", "L", "Pa", "R_micro"],
        "baseline_fit_seconds": baseline_seconds,
        "run_provenance": yaml.safe_load((run_dir / META_FILENAME).read_text(encoding="utf-8")),
        "provenance": provenance,
    }
    report["targets_met"] = all(t["passed"] for t in report["targets"])
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (run_dir / "eval.json").write_text(text, encoding="utf-8", newline="\n")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "surrogate_eval.json").write_text(text, encoding="utf-8", newline="\n")
    (report_dir / "surrogate_eval.md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    return report
