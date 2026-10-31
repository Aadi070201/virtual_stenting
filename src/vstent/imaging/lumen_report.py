"""Lumen accuracy on rendered vessels with known lumens: intensity edges vs the imaging-chain
model, and what each does to FFR (``vstent lumen-synthetic``, P4c).

Vessels of 8, 12 and 16 px with one cosine lesion are rendered as X-ray projections with the
ARCADE-fitted contrast, blur, edge enhancement and noise ranges of the synthetic validation
study (``configs/validation_synthetic.yaml``, D39). The vessel mask given to QCA is the
*unnarrowed* vessel, as from a segmentation that smoothed over the lesion (P4b), so the lesion
is visible only in the image. Lesions: circular (50–95% DS) and eccentric (an elliptical lumen
narrowed by half either in the image plane or along the X-ray, i.e. invisible in width). For an
eccentric lumen the truth is the area-equivalent %DS, 1 − sqrt(width × depth) / reference.

FFR: the reduced-order solver on the true lumen (equivalent-circle area) and on each QCA
profile, default physics, calibrated pixel size. The renderer and the model share their
physics (Beer–Lambert, Gaussian blur, unsharp mask), so this is the model's best case; real
angiograms are measured in ``reports/stenosis_grading.md``. Units: px, %, FFR.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.imaging.centerline import smooth_and_resample
from vstent.imaging.qca import qca_from_centerline
from vstent.imaging.synthetic import (
    render_projection,
    render_tubes,
    sine_centerline,
    stenosis_diameter,
)
from vstent.physics.geometry import VesselProfile
from vstent.physics.solver import PhysicsParams, solve_ffr
from vstent.units import mm_to_m

SHAPE = (200, 260)
LENGTH_PX = 220.0
LESION_PX = 30.0
SEED = 4100
N_SEEDS = 4
REFERENCES_PX = (8.0, 12.0, 16.0)
LESIONS = ((0.5, "circular"), (0.7, "circular"), (0.85, "circular"), (0.95, "circular"),
           (0.5, "in-plane"), (0.5, "along the ray"), (0.0, "none"))  # fmt: skip
# imaging ranges of configs/validation_synthetic.yaml (ARCADE-fitted, D39)
RANGES = {"mm_per_px": (0.22, 0.30), "mu_per_mm": (0.04, 0.08), "background": (0.49, 0.64),
          "blur_px": (0.8, 1.2), "edge_amount": (2.0, 3.0), "edge_sigma_px": (1.5, 3.0),
          "noise_sd": (0.012, 0.022)}  # fmt: skip
TARGET_TIGHT_ERROR = 10.0  # model, circular lesions ≥ 85%: mean |%DS error| (points)
TARGET_ECCENTRIC_ERROR = 10.0  # model, eccentric lumens: mean |%DS error| (points)
METHODS = ("edges", "model")


def render_lesion(ref_px: float, severity: float, kind: str, seed: int) -> dict[str, Any]:
    """One rendered case: image (uint8), unnarrowed mask, centreline, truth and settings."""
    rng = np.random.default_rng(seed)
    imaging = {k: float(rng.uniform(*v)) for k, v in RANGES.items()}
    pts, s = sine_centerline((100.0, 20.0), LENGTH_PX, amplitude_px=10.0, wavelength_px=300.0)
    narrow = stenosis_diameter(s, ref_px, severity, s[-1] / 2, LESION_PX)
    full = np.full_like(narrow, ref_px)
    width, depth = {"circular": (narrow, narrow), "none": (full, full),
                    "in-plane": (narrow, full), "along the ray": (full, narrow)}[kind]  # fmt: skip
    img = render_projection(
        SHAPE, [(pts, width / 2, depth / 2)], mm_per_px=imaging["mm_per_px"],
        mu_per_mm=imaging["mu_per_mm"], background=imaging["background"],
        blur_sigma=imaging["blur_px"], edge_amount=imaging["edge_amount"],
        edge_sigma=imaging["edge_sigma_px"], noise_std=imaging["noise_sd"], seed=seed,
    )  # fmt: skip
    mask = render_tubes(SHAPE, [(pts, full)], blur_sigma=0, noise_std=0)
    return {"image": np.round(img * 255).astype(np.uint8), "mask": mask, "points": pts, "s": s,
            "d_eq": np.sqrt(width * depth), "imaging": imaging}  # fmt: skip


def true_ffr(case: dict[str, Any], ref_px: float, params: PhysicsParams) -> float:
    """Distal FFR of the true lumen (equivalent-circle area), trimmed by one reference
    diameter at each end like the QCA path."""
    s, d = case["s"], case["d_eq"]
    su = np.arange(ref_px, s[-1] - ref_px, 1.0)
    scale = mm_to_m(case["imaging"]["mm_per_px"])
    profile = VesselProfile(s=(su - su[0]) * scale, r=np.interp(su, s, d) / 2 * scale,
                            r_ref=np.full(su.size, ref_px / 2 * scale))  # fmt: skip
    return float(solve_ffr(profile, params).ffr[-1])


def measure(case: dict[str, Any], params: PhysicsParams) -> dict[str, dict[str, float]]:
    """%DS and distal FFR of each method's QCA (calibrated pixel size)."""
    cl = smooth_and_resample(np.round(case["points"][::8]))
    out = {}
    for method in METHODS:
        q = qca_from_centerline(case["mask"], cl, image=case["image"], lumen=method,
                                mm_per_px=case["imaging"]["mm_per_px"])  # fmt: skip
        out[method] = {"ds": q.max_percent_stenosis,
                       "ffr": float(solve_ffr(q.profile, params).ffr[-1])}  # fmt: skip
        if q.lumen_fit is not None:
            out[method]["fit"] = {"blur_px": q.lumen_fit.blur_px,
                                  "edge_amount": q.lumen_fit.edge_amount,
                                  "edge_sigma_px": q.lumen_fit.edge_sigma_px,
                                  "rms": q.lumen_fit.rms}  # fmt: skip
    return out


def run_cases(params: PhysicsParams | None = None) -> list[dict[str, Any]]:
    """Every (reference, lesion, seed) case with its truth and both methods' results."""
    params = params or PhysicsParams()
    rows = []
    k = 0
    for ref in REFERENCES_PX:
        for severity, kind in LESIONS:
            for _ in range(N_SEEDS):
                case = render_lesion(ref, severity, kind, SEED + k)
                k += 1
                true_ds = 100.0 * (1.0 - case["d_eq"].min() / ref)
                rows.append({"ref_px": ref, "severity": severity, "kind": kind,
                             "true_ds": float(true_ds), "true_ffr": true_ffr(case, ref, params),
                             "imaging": case["imaging"], **measure(case, params)})  # fmt: skip
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean absolute %DS and FFR errors per method and lesion group, and the targets."""

    def mae(group, method, key):
        sel = [r for r in rows if group(r)]
        truth = "true_ds" if key == "ds" else "true_ffr"
        return float(np.mean([abs(r[method][key] - r[truth]) for r in sel])) if sel else None

    groups = {
        "circular, true < 85% DS": lambda r: r["kind"] == "circular" and r["severity"] < 0.85,
        "circular, true ≥ 85% DS": lambda r: r["kind"] == "circular" and r["severity"] >= 0.85,
        "eccentric (area-equivalent 29% DS)": lambda r: r["kind"] in ("in-plane",
                                                                      "along the ray"),
        "no lesion": lambda r: r["kind"] == "none",
    }  # fmt: skip
    table = {g: {m: {"ds": mae(f, m, "ds"), "ffr": mae(f, m, "ffr")} for m in METHODS}
             for g, f in groups.items()}  # fmt: skip
    tight = table["circular, true ≥ 85% DS"]
    ecc = table["eccentric (area-equivalent 29% DS)"]
    targets = [
        {"metric": "Model, circular lesions ≥ 85%: mean |%DS error|",
         "value": f"{tight['model']['ds']:.1f} points (edges {tight['edges']['ds']:.1f})",
         "target": f"≤ {TARGET_TIGHT_ERROR:g} points",
         "passed": tight["model"]["ds"] <= TARGET_TIGHT_ERROR},
        {"metric": "Model, eccentric lumens: mean |%DS error|",
         "value": f"{ecc['model']['ds']:.1f} points (edges {ecc['edges']['ds']:.1f})",
         "target": f"≤ {TARGET_ECCENTRIC_ERROR:g} points",
         "passed": ecc["model"]["ds"] <= TARGET_ECCENTRIC_ERROR},
    ]  # fmt: skip
    return {"groups": table, "targets": targets, "targets_met": all(t["passed"] for t in targets)}


def _figure(rows: list[dict[str, Any]], path: Path) -> None:
    from vstent.plotting import INK_MUTED, SERIES_1, SERIES_3, new_figure, save_figure, style_axes

    fig = new_figure(10.0, 4.4, "Lumen measurement on rendered vessels with known lumens",
                     "Mask smoothed over the lesion; ARCADE-like blur and edge enhancement; "
                     "dots = cases, diagonal = truth")  # fmt: skip
    ax_ds, ax_ffr = fig.subplots(1, 2, gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.2)
    labels = {"edges": "intensity edges", "model": "imaging-chain model"}
    for ax, key, lim, name in ((ax_ds, "ds", (-2, 100), "%DS"), (ax_ffr, "ffr", (0.0, 1.02),
                                                                  "distal FFR")):  # fmt: skip
        style_axes(ax)
        truth = "true_ds" if key == "ds" else "true_ffr"
        ax.plot(lim, lim, color=INK_MUTED, lw=1, zorder=1)
        for method, color in (("edges", SERIES_1), ("model", SERIES_3)):
            ax.scatter([r[truth] for r in rows], [r[method][key] for r in rows], s=14, lw=0,
                       alpha=0.75, color=color, label=labels[method], zorder=3)  # fmt: skip
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel(f"True {name}", fontsize=8)
        ax.set_ylabel(f"Measured {name}", fontsize=8)
    ax_ds.legend(frameon=False, fontsize=8, loc="upper left")
    save_figure(fig, path)


def write_lumen_report(report_dir: str | Path = "reports") -> dict[str, Any]:
    """Run every case and write ``lumen_synthetic.{md,json}`` and the figure."""
    provenance = {**git_info(), "vstent_version": __version__,
                  "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    rows = run_cases()
    summary = summarise(rows)
    report = {**summary, "rows": rows, "provenance": provenance}
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "lumen_synthetic.json").write_text(json.dumps(report, indent=2) + "\n",
                                                     encoding="utf-8", newline="\n")  # fmt: skip
    _figure(rows, report_dir / "figures" / "lumen_synthetic.png")

    def fmt(v, digits):
        return "–" if v is None else f"{v:.{digits}f}"

    lines = [
        "# Lumen measurement on rendered vessels: intensity edges vs the imaging-chain model",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent lumen-synthetic`. Do not edit by hand.",
        "",
        "Vessels of " + ", ".join(f"{r:g}" for r in REFERENCES_PX) + " px with one "
        f"{LESION_PX:g} px cosine lesion, {N_SEEDS} renders each (seeds from {SEED}), rendered "
        "as X-ray projections with the ARCADE-fitted contrast, blur, edge enhancement and noise "
        "of the synthetic validation study. The mask given to QCA is the unnarrowed vessel (a "
        "segmentation that smoothed over the lesion), so the lesion is seen only in the image. "
        "Eccentric lumens are narrowed by half in the image plane or along the X-ray; their "
        "truth is the area-equivalent %DS. FFR: the reduced-order solver on the true lumen and on "
        "each QCA profile (default physics, calibrated pixel size).",
        "",
        "**Caveat:** the renderer and the model share their physics, so this is the model's best "
        "case. Real angiograms: `reports/stenosis_grading.md`.",
        "",
        "| Target | Value | Required | Status |",
        "|---|---|---|---|",
        *(f"| {t['metric']} | {t['value']} | {t['target']} | "
          f"{'PASS' if t['passed'] else 'FAIL'} |" for t in summary["targets"]),
        "",
        "Mean absolute error by lesion group:",
        "",
        "| Lesions | %DS, edges | %DS, model | FFR, edges | FFR, model |",
        "|---|---:|---:|---:|---:|",
        *(f"| {g} | {fmt(v['edges']['ds'], 1)} | {fmt(v['model']['ds'], 1)} | "
          f"{fmt(v['edges']['ffr'], 3)} | {fmt(v['model']['ffr'], 3)} |"
          for g, v in summary["groups"].items()),
        "",
        "![Measured vs true %DS and FFR](figures/lumen_synthetic.png)",
        "",
        "Means per case (over seeds):",
        "",
        "| Reference (px) | Lesion | True %DS | Edges %DS | Model %DS | True FFR | Edges FFR "
        "| Model FFR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]  # fmt: skip
    for ref in REFERENCES_PX:
        for severity, kind in LESIONS:
            sel = [r for r in rows if r["ref_px"] == ref and r["kind"] == kind
                   and r["severity"] == severity]  # fmt: skip

            def m(key, method=None, sel=sel):
                return float(np.mean([r[method][key] if method else r[key] for r in sel]))

            lines.append(
                f"| {ref:g} | {kind} {severity:.0%} | {m('true_ds'):.0f} | {m('ds', 'edges'):.0f} "
                f"| {m('ds', 'model'):.0f} | {m('true_ffr'):.2f} | {m('ffr', 'edges'):.2f} "
                f"| {m('ffr', 'model'):.2f} |"
            )
    lines += ["", f"Commit `{provenance['git_commit']}` (dirty: {provenance['git_dirty']}), "
              f"created {provenance['created_utc']}.", ""]  # fmt: skip
    (report_dir / "lumen_synthetic.md").write_text("\n".join(lines), encoding="utf-8",
                                                   newline="\n")  # fmt: skip
    return report
