"""Uncertainty report (``vstent uncertainty-report``, P1) → ``reports/uncertainty.{md,json}``.

Every number is computed here: the exported ensemble and the M3 single model on the synthetic
splits (against the solver), the conformal intervals, the input-uncertainty Monte Carlo on
example cases, on test vessels and on planned stents, and the app latency from
``reports/app_latency.json``. Figures: ``reports/figures/uncertainty_*.png``.
"""

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vstent import DISCLAIMER, __version__
from vstent.data.synth import load_split
from vstent.inference.onnx_runtime import SurrogateRunner
from vstent.physics.geometry import VesselProfile, cosine_lesion_profile
from vstent.physics.solver import PhysicsParams
from vstent.planning.optimizer import PlannerConfig, StentCatalog, plan_stents
from vstent.planning.report import tandem_vessel
from vstent.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    new_figure,
    save_figure,
    style_axes,
)
from vstent.surrogate.dataset import load_raw
from vstent.surrogate.ensemble import surrogate_config
from vstent.surrogate.evaluate import DS_BINS
from vstent.surrogate.features import prepare_inputs
from vstent.surrogate.train import load_checkpoint, predict_batches, resolve_device
from vstent.uncertainty.conformal import ConformalCalibration, coverage, fit_conformal
from vstent.uncertainty.greyzone import UncertaintyConfig
from vstent.uncertainty.inputs import (
    SOURCE_LABELS,
    InputUncertainty,
    Measurement,
    QCAErrorModel,
    percentile_interval,
)

TARGET_COVERAGE = (0.88, 0.92)
FFR_BANDS = ((0.0, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))
SETTINGS = {
    "synthetic": ("synthetic geometry (known exactly)", Measurement()),
    "image": ("image-derived, 0.30 mm/px assumed", Measurement(0.30, True)),
}
N_FIGURE_VESSELS = 120


def max_ds(r: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    """Maximum diameter stenosis (%) of each vessel, (B,)."""
    return 100.0 * (1.0 - (r / r_ref).min(axis=1))


def _distal(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred[:, -1].astype(np.float64) - true[:, -1]
    return {
        "mae": float(np.abs(err).mean()),
        "max_abs_error": float(np.abs(err).max()),
        "curve_mae": float(np.abs(pred.astype(np.float64) - true).mean()),
    }


# --- ensemble vs single model ----------------------------------------------------------------


def ensemble_accuracy(
    ensemble: SurrogateRunner,
    single: SurrogateRunner,
    members: list[Path],
    data_dir: str,
    device: str,
    log: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Distal / curve error of the ensemble mean, the single M3 model and every member."""
    dev = resolve_device(device)
    nets = [load_checkpoint(m, dev) for m in members]
    out, arrays = {}, {}
    for split in ("val", "test", "ood"):
        raw, ffr = load_raw(data_dir, split)
        mean, std = ensemble.predict_with_spread(raw)
        arrays[split] = {
            "pred": mean[:, -1].astype(np.float64),
            "std": std[:, -1].astype(np.float64),
            "true": ffr[:, -1].astype(np.float64),
            "max_ds": max_ds(raw.r, raw.r_ref),
        }
        if split == "val":
            continue
        xp, xg = (torch.from_numpy(a).to(dev) for a in prepare_inputs(raw, ensemble.stats))
        member_rows = []
        for m, net in zip(members, nets, strict=True):
            pred = predict_batches(net, xp, xg).cpu().numpy()
            member_rows.append({"run": m.name, **_distal(pred, ffr)})
        out[split] = {
            "n": len(raw),
            "ensemble": {**_distal(mean, ffr), "median_std": float(np.median(std[:, -1]))},
            "single": _distal(single.predict(raw), ffr),
            "members": member_rows,
        }
        log(
            f"{split}: ensemble distal MAE {out[split]['ensemble']['mae']:.5f}, single "
            f"{out[split]['single']['mae']:.5f}, members "
            + ", ".join(f"{r['mae']:.5f}" for r in member_rows)
        )
    return out, arrays


# --- conformal ---------------------------------------------------------------------------------


def _by_ds(cal: ConformalCalibration, a: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in zip(DS_BINS[:-1], DS_BINS[1:], strict=True):
        m = (a["max_ds"] >= lo) & (a["max_ds"] < hi)
        cov = coverage(cal, a["pred"][m], a["std"][m], a["true"][m]) if m.any() else None
        rows.append({"bin": f"{lo:g}-{hi:g}%", "n": int(m.sum()), **(cov or {})})
    return rows


def conformal_section(
    cal: ConformalCalibration, arrays: dict[str, dict[str, np.ndarray]], threshold: float
) -> dict[str, Any]:
    """Coverage per split and DS bin, an absolute-score comparison and grey-zone statistics."""
    val = arrays["val"]
    absolute = fit_conformal(
        val["pred"], val["std"], val["true"], level=cal.level, score="absolute", split=cal.split
    )
    out: dict[str, Any] = {"calibration": asdict(cal), "absolute_q": absolute.q, "splits": {}}
    for split, a in arrays.items():
        lo, hi = cal.interval(a["pred"], a["std"])
        flagged = (lo <= threshold) & (threshold <= hi)
        disagree = (a["pred"] < threshold) != (a["true"] < threshold)
        out["splits"][split] = {
            **coverage(cal, a["pred"], a["std"], a["true"]),
            "absolute": coverage(absolute, a["pred"], a["std"], a["true"]),
            "by_ds": _by_ds(cal, a),
            "grey_zone": {
                "flagged_fraction": float(flagged.mean()),
                "n_disagree": int(disagree.sum()),
                "n_disagree_flagged": int((disagree & flagged).sum()),
            },
        }
    return out


# --- input uncertainty -------------------------------------------------------------------------


def _interval(mc_samples: np.ndarray, level: float) -> list[float]:
    return list(percentile_interval(mc_samples, level))


def preset_sensitivity(
    params: PhysicsParams, ucfg: UncertaintyConfig, qca_error: QCAErrorModel | None
) -> list[dict[str, Any]]:
    """One-at-a-time and combined input intervals for two example vessels in both settings."""
    cases = {
        "Focal: 60 mm, 3.0 mm, one 12 mm lesion of 70% DS": cosine_lesion_profile(
            0.06, 3.0e-3, 0.012, 0.7
        ),
        "Tandem: 70 mm, 3.0 mm, two 10 mm lesions of 60% DS": tandem_vessel(),
    }
    rows = []
    for name, profile in cases.items():
        for key, (label, meas) in SETTINGS.items():
            kw = {"measurement": meas, "config": ucfg.inputs, "qca_error": qca_error}
            full = InputUncertainty(profile, params, params.pa, params.pv, **kw)
            nominal = InputUncertainty(profile, params, params.pa, params.pv, sources=[], **kw)
            row = {
                "case": name,
                "setting": key,
                "setting_label": label,
                "ffr_nominal": float(nominal.pre_samples[0]),
                "sources": {},
                "combined": _interval(full.pre_samples, ucfg.level),
            }
            for src in full.sources:
                one = InputUncertainty(profile, params, params.pa, params.pv, sources=[src], **kw)
                row["sources"][src] = _interval(one.pre_samples, ucfg.level)
            rows.append(row)
    return rows


def _band_rows(ffr: np.ndarray, flagged: np.ndarray, width: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in FFR_BANDS:
        m = (ffr >= lo) & (ffr < hi)
        rows.append(
            {
                "band": f"{lo:.2f}-{min(hi, 1.0):.2f}",
                "n": int(m.sum()),
                "flagged_fraction": float(flagged[m].mean()) if m.any() else float("nan"),
                "median_width": float(np.median(width[m])) if m.any() else float("nan"),
            }
        )
    return rows


def _test_vessels(data_dir: str, n: int | None, only=None):
    keys = ("r", "r_ref", "length", "pa", "pv", "v_hyp", "ffr")
    arrays = load_split(data_dir, "test", keys)
    idx = np.arange(arrays["r"].shape[0]) if only is None else np.flatnonzero(only(arrays))
    for i in idx[:n]:
        s = np.linspace(0.0, float(arrays["length"][i]), arrays["r"].shape[1])
        yield (
            int(i),
            VesselProfile(s, arrays["r"][i], arrays["r_ref"][i]),
            float(arrays["pa"][i]),
            float(arrays["pv"][i]),
            float(arrays["v_hyp"][i]),
            float(arrays["ffr"][i, -1]),
        )


def population(
    data_dir: str,
    params: PhysicsParams,
    ucfg: UncertaintyConfig,
    qca_error: QCAErrorModel | None,
    n: int,
) -> dict[str, Any]:
    """Pre-PCI input intervals of the first ``n`` test vessels, in both settings."""
    t = ucfg.pre_pci_threshold
    rows = []
    for i, profile, pa, pv, v_hyp, ffr in _test_vessels(data_dir, n):
        vp = params.model_copy(update={"v_hyp": v_hyp})
        row = {"index": i, "ffr": ffr}
        for key, (_, meas) in SETTINGS.items():
            mc = InputUncertainty(
                profile, vp, pa, pv, measurement=meas, config=ucfg.inputs, qca_error=qca_error
            )
            row[key] = _interval(mc.pre_samples, ucfg.level)
        rows.append(row)
    ffr = np.array([r["ffr"] for r in rows])
    out: dict[str, Any] = {"n": len(rows), "rows": rows, "settings": {}}
    for key in SETTINGS:
        iv = np.array([r[key] for r in rows])
        width = iv[:, 1] - iv[:, 0]
        flagged = (iv[:, 0] <= t) & (t <= iv[:, 1])
        out["settings"][key] = {
            "median_width": float(np.median(width)),
            "p95_width": float(np.percentile(width, 95)),
            "flagged_fraction": float(flagged.mean()),
            "by_ffr": _band_rows(ffr, flagged, width),
        }
    return out


def plan_section(
    data_dir: str,
    runner: SurrogateRunner,
    params: PhysicsParams,
    planner: PlannerConfig,
    catalog: StentCatalog,
    ucfg: UncertaintyConfig,
    qca_error: QCAErrorModel | None,
    n: int,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Best surrogate plan for test vessels with pre-PCI FFR < 0.80, and its intervals."""
    target = planner.target_ffr
    rows = []
    vessels = _test_vessels(data_dir, n, only=lambda a: a["ffr"][:, -1] < ucfg.pre_pci_threshold)
    for i, profile, pa, pv, v_hyp, _ in vessels:
        vp = params.model_copy(update={"v_hyp": v_hyp})
        res = plan_stents(
            profile, catalog=catalog, config=planner.model_copy(update={"backend": "surrogate"}),
            params=vp, runner=runner, pa=pa, pv=pv,
        )  # fmt: skip
        if not res.plans:
            continue
        best = res.plans[0]
        row = {
            "index": i,
            "ffr": best.ffr,
            "ffr_solver": best.ffr_solver,
            "meets_target": best.meets_target,
            "model_interval": None if best.model_interval is None else list(best.model_interval),
        }
        for key, (_, meas) in SETTINGS.items():
            mc = InputUncertainty(
                profile, vp, pa, pv, measurement=meas, config=ucfg.inputs, qca_error=qca_error
            )
            row[key] = _interval(mc.post_samples(best.stents), ucfg.level)
        rows.append(row)
    log(f"Planned {len(rows)} test vessels with pre-PCI FFR < {ucfg.pre_pci_threshold}")

    def frac(key):
        iv = [r[key] for r in rows if r[key] is not None]
        return float(np.mean([lo <= target <= hi for lo, hi in iv])) if iv else float("nan")

    return {
        "n": len(rows),
        "target": target,
        "rows": rows,
        "meets_target_fraction": float(np.mean([r["meets_target"] for r in rows])),
        "flagged_model_fraction": frac("model_interval"),
        "flagged_synthetic_fraction": frac("synthetic"),
        "flagged_image_fraction": frac("image"),
        "median_width": {
            key: float(np.median([r[key][1] - r[key][0] for r in rows])) for key in SETTINGS
        }
        | {
            "model": float(
                np.median([r["model_interval"][1] - r["model_interval"][0] for r in rows])
            )
        },
    }


# --- figures -----------------------------------------------------------------------------------


def plot_coverage(conf: dict[str, Any], level: float, path: Path) -> None:
    fig = new_figure(
        7.2,
        4.2,
        f"Coverage of the {level:.0%} model interval by stenosis severity",
        "Share of vessels whose solver distal FFR lies inside the surrogate's conformal interval; "
        f"dashed = {level:.0%}",
    )
    ax = fig.add_axes((0.10, 0.17, 0.86, 0.63))
    style_axes(ax)
    ax.grid(False, axis="x")
    test, ood = conf["splits"]["test"]["by_ds"], conf["splits"]["ood"]["by_ds"]
    x = np.arange(len(test))
    w = 0.38
    for offset, rows, color, label in (
        (-w / 2 - 0.01, test, SERIES_1, "test (in distribution)"),
        (w / 2 + 0.01, ood, SERIES_2, "ood (80–120 mm, 3–4 lesions)"),
    ):
        vals = [r.get("coverage", np.nan) for r in rows]
        ax.bar(x + offset, vals, width=w, color=color, label=label, zorder=2)
        for xi, v in zip(x + offset, vals, strict=True):
            if np.isfinite(v):
                ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=7, color=INK_SECONDARY)
    ax.axhline(level, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.set_ylim(0, 1.08)
    labels = [f"{t['bin']}\nn = {t['n']:,} / {o['n']:,}" for t, o in zip(test, ood, strict=True)]
    ax.set_xticks(x, labels)
    ax.set_xlabel("Maximum diameter stenosis (n = test / ood)", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Coverage", fontsize=8, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower left")
    save_figure(fig, path)


def plot_input_intervals(pop: dict[str, Any], level: float, threshold: float, path: Path) -> None:
    rows = sorted(pop["rows"][:N_FIGURE_VESSELS], key=lambda r: r["ffr"])
    fig = new_figure(
        8.0,
        4.6,
        f"{level:.0%} input intervals of distal FFR ({len(rows)} test vessels)",
        f"Monte Carlo through the solver; sorted by solver FFR; orange = interval contains "
        f"{threshold:.2f} (grey zone)",
    )
    axes = fig.subplots(1, 2, sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.13)
    for ax, key in zip(axes, SETTINGS, strict=True):
        style_axes(ax)
        x = np.arange(len(rows))
        iv = np.array([r[key] for r in rows])
        flagged = (iv[:, 0] <= threshold) & (threshold <= iv[:, 1])
        for sel, color, label in ((~flagged, SERIES_1, "clear"), (flagged, SERIES_2, "grey zone")):
            ax.vlines(x[sel], iv[sel, 0], iv[sel, 1], color=color, lw=1.6, label=label, zorder=2)
        ax.scatter(x, [r["ffr"] for r in rows], s=6, color=INK_SECONDARY, zorder=3, lw=0)
        ax.axhline(threshold, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_title(SETTINGS[key][0], loc="left", fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlabel("Vessel (sorted by solver FFR)", fontsize=8, color=INK_SECONDARY)
        ax.set_xticks([])
    axes[0].set_ylabel("Distal FFR", fontsize=8, color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc="lower right")
    save_figure(fig, path)


# --- report ------------------------------------------------------------------------------------


def _latency(report_dir: Path) -> dict[str, Any] | None:
    path = report_dir / "app_latency.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    worst = max(m["server_max_ms"] for m in d["results"].values())
    return {"worst_server_ms": worst, "target_ms": d["target_ms"], "provenance": d["provenance"]}


def _targets(acc, conf, latency) -> list[dict[str, Any]]:
    test = conf["splits"]["test"]
    lo, hi = TARGET_COVERAGE
    rows = [
        {
            "metric": "Conformal coverage on test (90% target)",
            "value": f"{test['coverage']:.2%}",
            "target": f"{lo:.0%}–{hi:.0%}",
            "passed": lo <= test["coverage"] <= hi,
        },
        {
            "metric": "Ensemble-mean distal MAE ≤ single model (test)",
            "value": f"{acc['test']['ensemble']['mae']:.5f} vs {acc['test']['single']['mae']:.5f}",
            "target": "ensemble ≤ single",
            "passed": acc["test"]["ensemble"]["mae"] <= acc["test"]["single"]["mae"],
        },
    ]
    if latency is not None:
        rows.append(
            {
                "metric": "App stent-control update, worst server time",
                "value": f"{latency['worst_server_ms']:.0f} ms",
                "target": f"< {latency['target_ms']:.0f} ms",
                "passed": latency["worst_server_ms"] < latency["target_ms"],
            }
        )
    return rows


def write_uncertainty_report(
    model_dir: str | Path,
    single_dir: str | Path,
    report_dir: str | Path = "reports",
    *,
    config: UncertaintyConfig,
    params: PhysicsParams,
    planner: PlannerConfig,
    catalog: StentCatalog,
    n_vessels: int = 500,
    n_plans: int = 50,
    device: str = "auto",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate the exported ensemble and the input Monte Carlo; write the report."""
    from vstent.config import git_info

    log = log or (lambda _msg: None)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    model_dir, single_dir, report_dir = Path(model_dir), Path(single_dir), Path(report_dir)
    export_meta = json.loads((model_dir / "export_meta.json").read_text(encoding="utf-8"))
    single_meta = json.loads((single_dir / "export_meta.json").read_text(encoding="utf-8"))
    base = surrogate_config(model_dir)
    members = [Path(m["run"]) for m in export_meta["members"]]
    ensemble = SurrogateRunner.from_dir(model_dir)
    single = SurrogateRunner.from_dir(single_dir)
    if ensemble.conformal is None:
        raise ValueError(f"{model_dir} has no conformal calibration (run vstent export-ensemble)")
    qca_error = QCAErrorModel.from_report(config.inputs.qca_error_report)

    acc, arrays = ensemble_accuracy(ensemble, single, members, base.data_dir, device, log)
    conf = conformal_section(ensemble.conformal, arrays, config.pre_pci_threshold)
    log(
        f"Conformal coverage: val {conf['splits']['val']['coverage']:.2%}, test "
        f"{conf['splits']['test']['coverage']:.2%}, ood {conf['splits']['ood']['coverage']:.2%}"
    )
    presets = preset_sensitivity(params, config, qca_error)
    pop = population(base.data_dir, params, config, qca_error, n_vessels)
    log(f"Input intervals on {pop['n']} test vessels")
    plans = plan_section(
        base.data_dir, ensemble, params, planner, catalog, config, qca_error, n_plans, log
    )
    ensemble.close()
    single.close()
    latency = _latency(report_dir)

    fig_dir = report_dir / "figures"
    plot_coverage(conf, config.level, fig_dir / "uncertainty_coverage.png")
    plot_input_intervals(
        pop, config.level, config.pre_pci_threshold, fig_dir / "uncertainty_inputs.png"
    )
    report = {
        "model": model_dir.as_posix(),
        "single_model": single_dir.as_posix(),
        "targets": _targets(acc, conf, latency),
        "accuracy": acc,
        "conformal": conf,
        "inputs": {
            "config": config.inputs.model_dump(mode="json"),
            "qca_error_model": None if qca_error is None else asdict(qca_error),
            "presets": presets,
            "population": {k: v for k, v in pop.items() if k != "rows"},
            "population_rows": pop["rows"],
            "plans": plans,
        },
        "latency": latency,
        "model_provenance": {
            "ensemble_run": export_meta["run"],
            "ensemble_git_commit": export_meta["run_git_commit"],
            "export_git_commit": export_meta["export"]["git_commit"],
            "single_run": single_meta["run"],
            "single_git_commit": single_meta["run_git_commit"],
        },
        "provenance": provenance,
    }
    report["targets_met"] = all(t["passed"] for t in report["targets"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "uncertainty.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (report_dir / "uncertainty.md").write_text(
        _markdown(report, config), encoding="utf-8", newline="\n"
    )
    return report


def _iv(x) -> str:
    return "n/a" if x is None else f"{x[0]:.3f}–{x[1]:.3f}"


def _markdown(r: dict[str, Any], cfg: UncertaintyConfig) -> str:
    acc, conf, inp = r["accuracy"], r["conformal"], r["inputs"]
    cal = conf["calibration"]
    lvl = f"{cfg.level:.0%}"
    lines = [
        "# Uncertainty and grey-zone flagging (P1)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent uncertainty-report`. Do not edit by hand.",
        "",
        "**What the intervals cover.** The *model interval* covers only the surrogate's error "
        "against the reduced-order solver (split-conformal, calibrated on synthetic vessels). The "
        "*input interval* covers only the listed input assumptions (Monte Carlo through the "
        "solver). Neither covers the error of the reduced-order model itself against real "
        "pressure-wire FFR, which is unknown until P4 / P6.",
        "",
        "## Acceptance targets",
        "",
        "| Metric | Value | Target | Pass |",
        "|---|---|---|---|",
    ]
    for t in r["targets"]:
        lines.append(
            f"| {t['metric']} | {t['value']} | {t['target']} | {'yes' if t['passed'] else 'NO'} |"
        )
    if r["latency"] is None:
        lines += ["", "App latency: `reports/app_latency.json` missing (run `vstent app-latency`)."]
    lines += [
        "",
        f"OOD coverage (no target): {conf['splits']['ood']['coverage']:.2%}.",
        "",
        "## Ensemble vs single model",
        "",
        f"Five members, the M3 configuration with seeds 0–4. The single model is the M3 model "
        f"`{r['model_provenance']['single_run']}`. Errors against the solver; distal FFR unless "
        "stated.",
        "",
        "| Split | Model | n | Distal MAE | Max error | Curve MAE |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("test", "ood"):
        a = acc[split]
        rows = [("ensemble mean", a["ensemble"]), ("single (M3)", a["single"])]
        rows += [(f"member {m['run'].rsplit('-', 1)[-1]}", m) for m in a["members"]]
        for name, m in rows:
            lines.append(
                f"| {split} | {name} | {a['n']:,} | {m['mae']:.5f} | {m['max_abs_error']:.4f} "
                f"| {m['curve_mae']:.5f} |"
            )
    lines += [
        "",
        f"Median ensemble SD of distal FFR: test {acc['test']['ensemble']['median_std']:.5f}, "
        f"ood {acc['ood']['ensemble']['median_std']:.5f}.",
        "",
        f"## Model interval: split-conformal {lvl}",
        "",
        f"Calibrated on all {cal['n_calibration']:,} `{cal['split']}` vessels with the exported "
        f"ONNX ensemble. Score |y − ŷ| / (σ + {cal['sigma_floor']:g}), σ = ensemble SD of distal "
        f"FFR; q̂ = {cal['q']:.3f}. For comparison, an absolute score (one width for all vessels) "
        f"gives q̂ = {conf['absolute_q']:.5f}. The members were early-stopped on `val`, so the "
        "calibration split is not fully independent of training; the test coverage shows whether "
        "this matters.",
        "",
        "| Split | n | Coverage | Mean width | Median width | p95 width | Coverage (absolute) "
        "| Median width (absolute) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, s in conf["splits"].items():
        b = s["absolute"]
        lines.append(
            f"| {split} | {s['n']:,} | {s['coverage']:.2%} | {s['width_mean']:.4f} "
            f"| {s['width_median']:.4f} | {s['width_p95']:.4f} | {b['coverage']:.2%} "
            f"| {b['width_median']:.4f} |"
        )
    lines += [
        "",
        "Coverage by maximum diameter stenosis:",
        "",
        "| Max DS | test n | test coverage | test median width | ood n | ood coverage "
        "| ood median width |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, o in zip(conf["splits"]["test"]["by_ds"], conf["splits"]["ood"]["by_ds"], strict=True):

        def c(x, key, fmt):
            return format(x[key], fmt) if key in x else "n/a"

        lines.append(
            f"| {t['bin']} | {t['n']:,} | {c(t, 'coverage', '.2%')} | "
            f"{c(t, 'width_median', '.4f')} | {o['n']:,} | {c(o, 'coverage', '.2%')} | "
            f"{c(o, 'width_median', '.4f')} |"
        )
    lines += [
        "",
        "![Coverage by stenosis](figures/uncertainty_coverage.png)",
        "",
        f"**Grey zone at {cfg.pre_pci_threshold:.2f} (model interval only).** A vessel is "
        "flagged when its interval contains the threshold. *Disagreements* are vessels the "
        "surrogate and the solver put on different sides of the threshold.",
        "",
        "| Split | Flagged | Disagreements | Disagreements flagged |",
        "|---|---|---|---|",
    ]
    for split, s in conf["splits"].items():
        g = s["grey_zone"]
        share = f" ({g['n_disagree_flagged'] / g['n_disagree']:.0%})" if g["n_disagree"] else ""
        lines.append(
            f"| {split} | {g['flagged_fraction']:.2%} | {g['n_disagree']} "
            f"| {g['n_disagree_flagged']}{share} |"
        )
    ic = inp["config"]
    qe = inp["qca_error_model"]
    lines += [
        "",
        f"## Input interval: Monte Carlo through the solver ({ic['n_samples']} draws, "
        f"seed {ic['seed']})",
        "",
        "Varied, each uniformly unless noted:",
        "",
        f"- pixel size ±{ic['pixel_size_rel']:.0%} (only when assumed; lengths and diameters "
        "scale together; a stent keeps its physical length);",
        "- QCA diameter error: Gaussian, "
        + (
            f"SD {qe['sd_px']:.3f} px, correlation falling to 0.5 after {qe['corr_half_px']:.2f} "
            "px, as measured on rendered tubes (`reports/qca_synthetic.md`); image cases only. "
            "Rendered tubes are idealised, so real QCA error is likely larger;"
            if qe
            else "not included (`reports/qca_synthetic.json` missing);"
        ),
        f"- hyperemic velocity ±{ic['v_hyp_halfwidth']:g} m/s around the value used (half "
        "the synthetic sampling range), which recalibrates R_micro;",
        f"- Pa ±{ic['pa_halfwidth_mmhg']:g} mmHg and Pv ±{ic['pv_halfwidth_mmhg']:g} mmHg "
        "(half the synthetic sampling ranges).",
        "",
        "Two settings: *synthetic geometry* (known exactly: velocity and pressures only) and "
        "*image-derived* with an assumed 0.30 mm/px (all four).",
        "",
        f"### Example vessels: {lvl} interval per assumption (Pa 90, Pv 5 mmHg, 0.35 m/s)",
        "",
        "| Vessel | Setting | Solver FFR | Pixel size | QCA error | Velocity | Pa / Pv | All |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in inp["presets"]:
        src = p["sources"]
        lines.append(
            f"| {p['case']} | {p['setting_label']} | {p['ffr_nominal']:.3f} "
            + " ".join(f"| {_iv(src.get(k))}" for k in SOURCE_LABELS)
            + f" | {_iv(p['combined'])} |"
        )
    pop = inp["population"]
    lines += [
        "",
        f"### {pop['n']} test vessels (each with its own Pa, Pv and velocity as the centre)",
        "",
        "| Setting | Median width | p95 width | Flagged at "
        f"{cfg.pre_pci_threshold:.2f} | Flagged, FFR 0.70–0.80 | Flagged, FFR 0.80–0.90 |",
        "|---|---|---|---|---|---|",
    ]
    for key, s in pop["settings"].items():
        bands = {b["band"]: b for b in s["by_ffr"]}
        lines.append(
            f"| {SETTINGS[key][0]} | {s['median_width']:.3f} | {s['p95_width']:.3f} "
            f"| {s['flagged_fraction']:.1%} | {bands['0.70-0.80']['flagged_fraction']:.0%} "
            f"(n = {bands['0.70-0.80']['n']}) | {bands['0.80-0.90']['flagged_fraction']:.0%} "
            f"(n = {bands['0.80-0.90']['n']}) |"
        )
    pl = inp["plans"]
    mw = pl["median_width"]
    lines += [
        "",
        "![Input intervals](figures/uncertainty_inputs.png)",
        "",
        f"### Post-PCI: best plan for {pl['n']} test vessels with pre-PCI FFR < "
        f"{cfg.pre_pci_threshold:.2f}",
        "",
        f"Planned with the ensemble (screening by one member, final plans re-scored by the "
        f"ensemble). Grey zone = interval contains the planner target {pl['target']:.2f}. "
        f"{pl['meets_target_fraction']:.0%} of the best plans reach the target.",
        "",
        "| Interval | Median width | Flagged |",
        "|---|---|---|",
        f"| Model ({lvl}, surrogate) | {mw['model']:.4f} | {pl['flagged_model_fraction']:.0%} |",
        f"| Inputs, {SETTINGS['synthetic'][0]} | {mw['synthetic']:.3f} "
        f"| {pl['flagged_synthetic_fraction']:.0%} |",
        f"| Inputs, {SETTINGS['image'][0]} | {mw['image']:.3f} "
        f"| {pl['flagged_image_fraction']:.0%} |",
        "",
        "Post-PCI draws reuse the pre-PCI draws and their own R_micro, so the stent changes only "
        "the epicardial geometry.",
        "",
        "## Limitations",
        "",
        "- Both intervals are conditional on the reduced-order model being right; its own error "
        "against wire FFR or 3D CFD is not included (P4, P6).",
        "- Conformal coverage holds on average over vessels like the calibration split "
        "(synthetic, unstented, pre-PCI). It is not guaranteed for OOD vessels (see the table), "
        "stented vessels or image-derived profiles.",
        "- The input ranges are the synthetic sampling ranges, not measured patient-level "
        "uncertainties; the QCA error comes from idealised rendered tubes. Segmentation errors, "
        "overlap, foreshortening and side branches are not in the input interval.",
        "- The two intervals are reported side by side, not combined.",
        "",
        "## Provenance",
        "",
    ]
    mp, prov = r["model_provenance"], r["provenance"]
    lines += [
        f"- Ensemble `{mp['ensemble_run']}` (training commit `{mp['ensemble_git_commit']}`, "
        f"export commit `{mp['export_git_commit']}`); single model `{mp['single_run']}` "
        f"(commit `{mp['single_git_commit']}`).",
    ]
    if r["latency"] is not None:
        lp = r["latency"]["provenance"]
        lines.append(
            f"- App latency from `reports/app_latency.json` (commit `{lp['git_commit']}`, "
            f"dirty: {lp['git_dirty']}, {lp['created_utc']})."
        )
    lines += [
        f"- Report commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), vstent "
        f"{prov['vstent_version']}, {prov['created_utc']}.",
        "",
    ]
    return "\n".join(lines)
