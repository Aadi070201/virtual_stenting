"""Figures of one analysis: image + centreline(s), diameters, FFR pullback (vessel or tree)."""

from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from vstent.inference.pipeline import TreeAnalysis, VesselAnalysis
from vstent.physics.tree import TreeFFRResult
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


def _label(ax, x, y, text: str) -> None:
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7,
                color=INK_SECONDARY, va="center")  # fmt: skip


def plot_tree_analysis(
    analysis: TreeAnalysis,
    image: np.ndarray,
    path: str | Path,
    *,
    title: str = "Vessel tree analysis",
    post: TreeFFRResult | None = None,
    boxes: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Image with every branch's centreline, and FFR along every root-to-outlet path: the tree
    model, the main path as a single vessel, and (``post``) after a virtual stent."""
    tq, res = analysis.qca, analysis.result
    tree = tq.tree
    calib = "assumed" if tq.calibration_assumed else "given"
    fig = new_figure(
        11.0, 4.6, title,
        f"QCA at {tq.mm_per_px:g} mm/px ({calib}); {tree.n_branches} branches; main-path distal "
        f"FFR {res.ffr[0][-1]:.2f} as a tree vs {analysis.single.ffr[-1]:.2f} as a single path",
    )  # fmt: skip
    ax_img = fig.add_axes((0.02, 0.08, 0.36, 0.78))
    ax_f = fig.add_axes((0.45, 0.13, 0.44, 0.71))
    ax_img.imshow(image, cmap="gray", vmin=0, vmax=255)
    for k, q in enumerate(tq.branches):
        pts = q.centerline.points
        color = SERIES_1 if k == 0 else SERIES_3
        ax_img.plot(pts[:, 1], pts[:, 0], color=color, lw=1.4 if k == 0 else 1.1)
        ax_img.annotate(tree.names[k], (pts[-1, 1], pts[-1, 0]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=color)  # fmt: skip
    ax_img.plot(*tq.branches[0].centerline.points[0, ::-1], "o", color=SERIES_1, ms=4)
    for x, y, w, h in boxes or []:
        ax_img.add_patch(Rectangle((x, y), w, h, fill=False, ec=SERIES_2, lw=1.0, ls=(0, (3, 2))))
    ax_img.set_axis_off()

    style_axes(ax_f)
    ax_f.axhline(CUTOFF, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for k in range(tree.n_branches):
        _, _, _, offset = tree.path_offsets(k)[-1]
        x = m_to_mm(offset + tree.branches[k].s)
        color = SERIES_1 if k == 0 else SERIES_3
        ax_f.plot(x, res.ffr[k], color=color, lw=2.0 if k == 0 else 1.3, zorder=3)
        _label(ax_f, x[-1], res.ffr[k][-1], tree.names[k] + (", tree" if k == 0 else ""))
        if post is not None:
            ax_f.plot(x, post.ffr[k], color=color, lw=1.2, ls=(0, (1, 1.5)), zorder=3)
    x0 = m_to_mm(tree.branches[0].s)
    ax_f.plot(x0, analysis.single.ffr, color=SERIES_2, lw=1.4, ls=(0, (4, 2)), zorder=2)
    _label(ax_f, x0[-1], analysis.single.ffr[-1], "main, single path")
    if post is not None:
        _label(ax_f, x0[-1], post.ffr[0][-1], "main, after stent (dotted)")
    ax_f.set_xlabel("Distance from the inlet along each path (mm)", fontsize=8,
                    color=INK_SECONDARY)  # fmt: skip
    ax_f.set_ylabel("FFR (tree model)", fontsize=8, color=INK_SECONDARY)
    save_figure(fig, path)
