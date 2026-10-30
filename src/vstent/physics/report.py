"""Physics checks report: ``figures/ffr_vs_ds.png`` and ``physics_checks.{json,md}``.

Every number in the report is computed here from the solver; nothing is hand-typed.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.physics.geometry import cosine_lesion_profile, cosine_narrowing
from vstent.physics.solver import PhysicsParams, solve_ffr, solve_ffr_batch
from vstent.sysinfo import cpu_name
from vstent.units import mm_to_m

N_POINTS = 256
VESSEL_DIAMETER_MM = 3.0
VESSEL_LENGTH_MM = 60.0
LESION_LENGTHS_MM = (5.0, 10.0, 20.0)
DS_SWEEP_PCT = np.arange(0.0, 90.0 + 0.5, 1.0)

# Chart tokens (reference palette, light mode): categorical slots 1-3 and chart chrome.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def ffr_vs_ds(
    params: PhysicsParams,
    lesion_lengths_mm=LESION_LENGTHS_MM,
    ds_pct: np.ndarray = DS_SWEEP_PCT,
    n: int = N_POINTS,
) -> dict[float, np.ndarray]:
    """Distal FFR (-) over a %DS sweep for centred cosine lesions in a 3.0 mm × 60 mm vessel."""
    length = mm_to_m(VESSEL_LENGTH_MM)
    s = np.linspace(0.0, length, n)
    r_ref = np.full((ds_pct.size, n), mm_to_m(VESSEL_DIAMETER_MM) / 2.0)
    curves = {}
    for lesion_mm in lesion_lengths_mm:
        factor = cosine_narrowing(s[None, :], length / 2, mm_to_m(lesion_mm), ds_pct[:, None] / 100)
        res = solve_ffr_batch(r_ref * factor, s[1] - s[0], r_ref, params)
        curves[float(lesion_mm)] = res.ffr_distal
    return curves


def sanity_checks(params: PhysicsParams) -> list[dict]:
    """The M1 sanity values, each with its target and whether it is met."""

    def lesion_ffr(severity: float, lesion_mm: float = 10.0) -> float:
        prof = cosine_lesion_profile(
            mm_to_m(VESSEL_LENGTH_MM),
            mm_to_m(VESSEL_DIAMETER_MM),
            mm_to_m(lesion_mm),
            severity,
            n=N_POINTS,
        )
        return solve_ffr(prof, params).ffr_distal

    healthy = solve_ffr(
        cosine_lesion_profile(mm_to_m(50.0), mm_to_m(3.0), mm_to_m(10.0), 0.0, n=N_POINTS), params
    ).ffr_distal
    sweep = [lesion_ffr(sev) for sev in np.arange(0.30, 0.901, 0.05)]
    ffr50, ffr70 = lesion_ffr(0.5), lesion_ffr(0.7)
    return [
        {
            "check": "Healthy 3.0 mm × 50 mm vessel, distal FFR",
            "value": healthy,
            "target": "≥ 0.95",
            "passed": healthy >= 0.95,
        },
        {
            "check": "3.0 mm × 60 mm, 10 mm cosine lesion, 50% DS",
            "value": ffr50,
            "target": "[0.84, 0.94]",
            "passed": 0.84 <= ffr50 <= 0.94,
        },
        {
            "check": "3.0 mm × 60 mm, 10 mm cosine lesion, 70% DS",
            "value": ffr70,
            "target": "[0.45, 0.70]",
            "passed": 0.45 <= ffr70 <= 0.70,
        },
        {
            "check": "Distal FFR strictly decreasing, DS 30% → 90% (5% steps)",
            "value": float(np.max(np.diff(sweep))),
            "target": "max step < 0",
            "passed": bool(np.all(np.diff(sweep) < 0)),
        },
    ]


def benchmark_batch(
    params: PhysicsParams, n_batch: int = 10_000, n: int = N_POINTS, repeats: int = 5, seed: int = 0
) -> dict:
    """Throughput of :func:`solve_ffr_batch` on random single-lesion vessels (best of repeats)."""
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, mm_to_m(VESSEL_LENGTH_MM), n)
    r_ref = np.full((n_batch, n), mm_to_m(VESSEL_DIAMETER_MM) / 2.0)
    centers = mm_to_m(rng.uniform(15.0, 45.0, n_batch))[:, None]
    lengths = mm_to_m(rng.uniform(3.0, 25.0, n_batch))[:, None]
    severity = rng.uniform(0.2, 0.9, n_batch)[:, None]
    r = r_ref * cosine_narrowing(s[None, :], centers, lengths, severity)
    solve_ffr_batch(r[:100], s[1], r_ref[:100], params)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        solve_ffr_batch(r, s[1], r_ref, params)
        times.append(time.perf_counter() - t0)
    best = min(times)
    return {
        "n_batch": n_batch,
        "n_points": n,
        "repeats": repeats,
        "best_seconds": best,
        "profiles_per_second": n_batch / best,
        "target_profiles_per_second": 5000,
        "passed": n_batch / best >= 5000,
        "cpu": cpu_name(),
    }


def _spread_labels(y: list[float], min_gap: float) -> list[float]:
    """Nudge label positions apart (keeping order) so neighbours are ≥ min_gap apart."""
    order = np.argsort(y)
    placed = np.array(y, dtype=float)[order]
    for i in range(1, placed.size):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


def plot_ffr_vs_ds(curves: dict[float, np.ndarray], ds_pct: np.ndarray, path: Path) -> None:
    """Line chart of distal FFR vs %DS, one line per lesion length, with the 0.80 cut-off."""
    fig = Figure(figsize=(7.2, 4.6), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes((0.10, 0.14, 0.76, 0.70), facecolor=SURFACE)

    ax.axhline(0.80, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(1.0, 0.81, "FFR 0.80 cut-off", color=INK_SECONDARY, fontsize=8, va="bottom")

    ends = []
    for (lesion_mm, ffr), color in zip(curves.items(), SERIES_COLORS, strict=True):
        ax.plot(ds_pct, ffr, color=color, lw=2.0, label=f"{lesion_mm:g} mm lesion", zorder=3)
        ends.append(float(ffr[-1]))
    for (lesion_mm, _), y_label in zip(curves.items(), _spread_labels(ends, 0.05), strict=True):
        ax.annotate(
            f"{lesion_mm:g} mm",
            xy=(ds_pct[-1], y_label),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
            annotation_clip=False,
        )

    ax.set_xlim(0, ds_pct[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Diameter stenosis (%)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Distal FFR", color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRIDLINE, lw=0.6, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    legend.set_zorder(4)

    fig.text(0.10, 0.94, "Distal FFR vs diameter stenosis", fontsize=11, color=INK_PRIMARY)
    fig.text(
        0.10,
        0.895,
        f"{VESSEL_DIAMETER_MM:.1f} mm × {VESSEL_LENGTH_MM:g} mm vessel, centred cosine lesion, "
        "reduced-order model (resistance mode)",
        fontsize=8,
        color=INK_SECONDARY,
    )
    fig.text(0.10, 0.025, DISCLAIMER, fontsize=7, color=INK_MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def _markdown(report: dict) -> str:
    lines = [
        "# Physics checks (reduced-order solver)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent physics-report`. Do not edit by hand.",
        "",
        "## Sanity checks",
        "",
        "| Check | Value | Target | Pass |",
        "|---|---|---|---|",
    ]
    for c in report["checks"]:
        passed = "yes" if c["passed"] else "NO"
        lines.append(f"| {c['check']} | {c['value']:.4f} | {c['target']} | {passed} |")
    b = report["benchmark"]
    lines += [
        "",
        "## Batch solver throughput",
        "",
        f"- {b['profiles_per_second']:,.0f} profiles/s"
        f" (target ≥ {b['target_profiles_per_second']:,});"
        f" batch {b['n_batch']:,} × {b['n_points']} points, best of {b['repeats']}"
        f" ({b['best_seconds'] * 1e3:.1f} ms)",
        f"- CPU: {b['cpu']}",
        "",
        "## Distal FFR vs %DS",
        "",
        "![Distal FFR vs diameter stenosis](figures/ffr_vs_ds.png)",
        "",
        "| DS (%) | " + " | ".join(f"{k} mm" for k in report["ffr_vs_ds"]["curves"]) + " |",
        "|---|" + "---|" * len(report["ffr_vs_ds"]["curves"]),
    ]
    ds_pct = report["ffr_vs_ds"]["ds_pct"]
    curves = list(report["ffr_vs_ds"]["curves"].values())
    for i, ds in enumerate(ds_pct):
        if ds % 10 == 0:
            lines.append(f"| {ds:g} | " + " | ".join(f"{c[i]:.3f}" for c in curves) + " |")
    prov = report["provenance"]
    lines += [
        "",
        "## Provenance",
        "",
        f"- git commit: `{prov['git_commit']}` (dirty: {prov['git_dirty']})",
        f"- vstent {prov['vstent_version']}, generated {prov['created_utc']}",
        f"- parameters: `{json.dumps(report['params'])}`",
        "",
    ]
    return "\n".join(lines)


def write_physics_report(out_dir: str | Path, params: PhysicsParams | None = None) -> dict:
    """Compute all checks and write the figure, JSON and Markdown into ``out_dir``."""
    params = params or PhysicsParams()
    out_dir = Path(out_dir)
    provenance = git_info()  # before writing anything, so our own outputs don't count as dirty
    curves = ffr_vs_ds(params)
    plot_ffr_vs_ds(curves, DS_SWEEP_PCT, out_dir / "figures" / "ffr_vs_ds.png")
    report = {
        "params": params.model_dump(mode="json"),
        "checks": sanity_checks(params),
        "benchmark": benchmark_batch(params),
        "ffr_vs_ds": {
            "vessel_diameter_mm": VESSEL_DIAMETER_MM,
            "vessel_length_mm": VESSEL_LENGTH_MM,
            "n_points": N_POINTS,
            "ds_pct": DS_SWEEP_PCT.tolist(),
            "curves": {f"{k:g}": v.tolist() for k, v in curves.items()},
        },
        "provenance": {
            **provenance,
            "vstent_version": __version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "physics_checks.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (out_dir / "physics_checks.md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    return report
