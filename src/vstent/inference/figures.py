"""Figure of one vessel analysis: image + centreline, diameter profile, FFR pullback."""

from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from vstent.inference.pipeline import VesselAnalysis
from vstent.plotting import (
    AXIS,
    BAND,
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    new_figure,
    save_figure,
    style_axes,
)
from vstent.units import m_to_mm

CUTOFF = 0.80


def plot_analysis(
    analysis: VesselAnalysis,
    image: np.ndarray,
    path: str | Path,
    *,
    title: str = "Vessel analysis",
    boxes: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Three panels; ``boxes`` (x, y, w, h px) are drawn on the image (e.g. ARCADE stenosis)."""
    qca, solver = analysis.qca, analysis.solver
    cl = qca.centerline
    calib = "assumed" if qca.calibration_assumed else "given"
    fig = new_figure(
        11.0,
        3.9,
        title,
        f"QCA at {qca.mm_per_px:g} mm/px ({calib}); max DS {qca.max_percent_stenosis:.0f}%, "
        f"MLD {qca.mld_mm:.2f} mm; distal FFR solver {analysis.ffr_distal_solver:.2f}"
        + (
            f", surrogate {analysis.ffr_distal_surrogate:.2f}"
            if analysis.surrogate_ffr is not None
            else ""
        ),
    )
    ax_img = fig.add_axes((0.02, 0.10, 0.28, 0.74))
    ax_d = fig.add_axes((0.37, 0.14, 0.27, 0.66))
    ax_f = fig.add_axes((0.71, 0.14, 0.27, 0.66))

    ax_img.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax_img.plot(cl.points[:, 1], cl.points[:, 0], color=SERIES_1, lw=1.2)
    ax_img.plot(*cl.points[0, ::-1], "o", color=SERIES_3, ms=4)
    ax_img.plot(*cl.points[-1, ::-1], "s", color=SERIES_3, ms=4)
    ax_img.plot(*qca.mld_point_px[::-1], "x", color=SERIES_2, ms=7, mew=2)
    for x, y, w, h in boxes or []:
        ax_img.add_patch(Rectangle((x, y), w, h, fill=False, ec=SERIES_2, lw=1.0, ls=(0, (3, 2))))
    ax_img.set_axis_off()

    s_mm = qca.s_mm
    style_axes(ax_d)
    if qca.capped.any():
        ax_d.plot(
            s_mm, qca.measured_diameter_mm, color=AXIS, lw=0.8, label="Measured (raw)",
            zorder=2,
        )  # fmt: skip
    ax_d.plot(s_mm, qca.diameter_mm, color=SERIES_1, lw=1.6, label="Lumen (used)", zorder=3)
    ax_d.plot(
        s_mm, qca.reference_diameter_mm, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)),
        label="Reference", zorder=2,
    )  # fmt: skip
    ax_d.axvline(s_mm[qca.mld_index], color=SERIES_2, lw=0.8, zorder=1)
    # raw spikes may run off the top; the axis is scaled to what the physics uses
    ax_d.set_ylim(0, max(qca.diameter_mm.max(), qca.reference_diameter_mm.max()) * 1.6)
    ax_d.set_xlabel("Distance from start (mm)", fontsize=8, color=INK_SECONDARY)
    ax_d.set_ylabel("Diameter (mm)", fontsize=8, color=INK_SECONDARY)
    ax_d.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc="lower left")

    style_axes(ax_f)
    lowest = float(solver.ffr.min())
    if analysis.surrogate_ffr is not None:
        lowest = min(lowest, float(analysis.surrogate_ffr.min()))
    lo = min(0.5, lowest - 0.05)
    ax_f.axhspan(lo, CUTOFF, color=BAND, zorder=0.5)
    ax_f.plot(m_to_mm(solver.s), solver.ffr, color=INK_SECONDARY, lw=2.0, label="Solver", zorder=2)
    if analysis.surrogate_ffr is not None:
        ax_f.plot(
            m_to_mm(analysis.surrogate_s), analysis.surrogate_ffr, color=SERIES_1, lw=2.0,
            ls=(0, (5, 2)), label="Surrogate (ONNX)", zorder=3,
        )  # fmt: skip
    ax_f.axhline(CUTOFF, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax_f.set_ylim(lo, 1.02)
    ax_f.set_xlabel("Distance from start (mm)", fontsize=8, color=INK_SECONDARY)
    ax_f.set_ylabel("FFR", fontsize=8, color=INK_SECONDARY)
    ax_f.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc="lower left")
    save_figure(fig, path)
