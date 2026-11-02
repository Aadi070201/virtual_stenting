"""Figures for the presentation brief (docs/presentation/): the pipeline, the guided workflow on a
synthetic beating heart and a real CADICA run (CC BY 4.0), and charts from the generated
reports. Every figure is produced here from code and saved data.

    uv run python scripts/presentation_figures.py
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from vstent import DISCLAIMER
from vstent.app import core, guided
from vstent.imaging.run_analysis import analyse_frames
from vstent.inference.consensus import run_consensus
from vstent.inference.models import latest_model_dir
from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
from vstent.inference.pipeline import analyse_vessel
from vstent.planning.optimizer import plan_stents
from vstent.plotting import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    SURFACE,
    style_axes,
)
from vstent.units import m_to_mm

OUT = Path("docs/presentation/figures")
REPORT_FIGURES = {  # existing report figures used in the brief
    "physics_ffr_vs_ds.png": "reports/figures/ffr_vs_ds.png",
    "segmentation_examples.png": "reports/figures/seg_examples.png",
    "rendered_angiograms.png": "reports/figures/synth_angio_examples.png",
    "surrogate_vs_solver.png": "reports/figures/surrogate_scatter.png",
    "uncertainty_coverage.png": "reports/figures/uncertainty_coverage.png",
    "vessel_tree.png": "reports/figures/tree_example.png",
    "two_view_3d.png": "reports/figures/two_view_example.png",
    "synthetic_validation.png": "reports/figures/validation_synthetic.png",
    "stenosis_grading.png": "reports/figures/stenosis_grading.png",
    "fda_nozzle.png": "reports/figures/physics_nozzle.png",
}


def save(fig: Figure, name: str) -> None:
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=6.5, color=INK_MUTED)
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    print("wrote", OUT / name)


def save_rgb(rgb: np.ndarray, name: str) -> None:
    cv2.imwrite(str(OUT / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / name)


def pipeline() -> None:
    steps = [
        ("Cine angiogram", "multi-frame DICOM\n(+ ECG if recorded)"),
        ("Frames", "contrast curve,\nend-diastolic frames"),
        ("Vessels", "AI segmentation\n(U-Net, ONNX)"),
        ("Path & QCA", "centreline, diameters,\nlesions (edge\ntracking)"),
        ("Physics", "pressure drop along\nthe vessel → FFR\n(+ learned surrogate)"),
        ("Heartbeats", "median of up to 5\nend-diastolic frames"),
        ("Virtual stent", "planner: shortest\nstent reaching\nFFR ≥ 0.90"),
        ("Report", "FFR, range, plan,\naudit trail"),
    ]
    fig = Figure(figsize=(13, 2.3), facecolor=SURFACE)
    ax = fig.add_axes([0, 0.08, 1, 0.84])
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for k, (title, body) in enumerate(steps):
        x = k + 0.08
        color = SERIES_1 if k < 4 else (SERIES_3 if k < 6 else SERIES_2)
        ax.add_patch(FancyBboxPatch((x, 0.06), 0.84, 0.88, boxstyle="round,pad=0.02",
                                    fc="white", ec=color, lw=2))  # fmt: skip
        ax.text(x + 0.42, 0.8, title, ha="center", va="center", fontsize=10,
                color=INK_PRIMARY, weight="bold")  # fmt: skip
        ax.text(x + 0.42, 0.4, body, ha="center", va="center", fontsize=8.2, color=INK_SECONDARY)
        if k < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.86, 0.5), (x + 1.06, 0.5), arrowstyle="-|>",
                                         mutation_scale=12, color=INK_MUTED))  # fmt: skip
    save(fig, "pipeline.png")


def workflow(seg, label: str, case, prefix: str, boxes_frame: bool = False) -> dict:
    """The guided app's steps on one case, as images and charts."""
    run = guided.load_case(case)
    rf = analyse_frames(run, seg.predict)
    frame = rf.analysis_frame
    if boxes_frame and case.boxes:
        frame = sorted({b[0] for b in case.boxes})[len({b[0] for b in case.boxes}) // 2]
    img = run.frames[frame]
    prob = seg.predict(img)
    mm = run.pixel_spacing_mm
    boxes = [b[1:] for b in case.boxes if b[0] == frame]
    if boxes:
        ranked = sorted(boxes, key=lambda b: -guided.severity(b[4]))
        start, end = next(e for b in ranked if (e := guided.box_path(prob, b[:4])))
    else:
        start, end = guided.auto_path(prob, mm)
    params = core.load_settings()[0]

    def one(p, im, a, b):
        return analyse_vessel(p, a, b, params=params, mm_per_px=mm, image=im)

    ref = one(prob, img, start, end)
    cons = run_consensus(rf, frame, start, end, seg.predict, one, reference_analysis=ref)
    q = cons.chosen.analysis.qca
    lesions = guided.lesion_marks(q)

    # frames strip: the ED frames, the recommended one marked
    eds = [f for f in (rf.ed.frames if rf.ed.found else []) if 0 <= f < len(run)][:6]
    fig = Figure(figsize=(12, 2.6), facecolor=SURFACE)
    for k, f in enumerate(eds):
        ax = fig.add_axes([0.005 + k / 6, 0.12, 1 / 6 - 0.01, 0.76])
        ax.imshow(run.frames[f], cmap="gray")
        ax.set_axis_off()
        star = " ★" if f == rf.analysis_frame else ""
        ax.set_title(f"frame {f}{star}", fontsize=9, color=INK_PRIMARY if star else INK_SECONDARY)
    save(fig, f"{prefix}_ed_frames.png")
    save_rgb(guided.overlay(img, prob, ref.qca, guided.lesion_marks(ref.qca), points=[start, end],
                            boxes=boxes), f"{prefix}_vessel_and_lesions.png")  # fmt: skip
    if lesions:
        z_raw = guided.lesion_zoom(run.frames[cons.chosen.frame], q, lesions[0], edges=False)
        z_edges = guided.lesion_zoom(run.frames[cons.chosen.frame], q, lesions[0], edges=True)
        save_rgb(np.concatenate([z_raw, np.full((z_raw.shape[0], 8, 3), 255, np.uint8),
                                 z_edges], axis=1), f"{prefix}_lesion_closeup.png")  # fmt: skip
    pre = core.evaluate(q.profile, params, params.pa, params.pv)
    planner, catalog = core.load_settings()[1], core.load_settings()[2]
    solver_cfg = planner.model_copy(update={"backend": "solver"})
    plans = plan_stents(q.profile, catalog=catalog, config=solver_cfg, params=params)
    post, stent = None, None
    if plans.plans:
        stent = plans.plans[0].stents[0]
        _, post = core.stent_update(q.profile, [stent], pre, params, params.pa, params.pv, None)
        save_rgb(guided.overlay(run.frames[cons.chosen.frame], cons.chosen.analysis.prob, q,
                                lesions, [stent], [cons.chosen.start, cons.chosen.end]),
                 f"{prefix}_stent.png")  # fmt: skip
    fig = Figure(figsize=(7.5, 3.6), facecolor=SURFACE)
    ax = fig.add_axes([0.1, 0.18, 0.86, 0.72])
    style_axes(ax)
    for m in lesions:
        ax.axvspan(m.start_mm, m.end_mm, color="#dc2626", alpha=0.08, lw=0)
    ax.axhline(0.80, color=INK_MUTED, ls="--", lw=1)
    ax.text(m_to_mm(pre.s[-1]), 0.805, "0.80 cut-off", ha="right", fontsize=7.5, color=INK_MUTED)
    ax.plot(m_to_mm(pre.s), pre.solver, color=INK_SECONDARY, lw=2.2, label="now")
    if post is not None:
        ax.plot(m_to_mm(post.s), post.solver, color=SERIES_2, lw=2.2, label="after the stent")
    ax.set_xlabel("distance along the vessel (mm)", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("virtual FFR", fontsize=8.5, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{label}: virtual FFR {cons.ffr:.2f}" + (f" → {post.distal_solver:.2f} after "
                 "the stent" if post is not None else ""), fontsize=9.5, loc="left",
                 color=INK_PRIMARY)  # fmt: skip
    save(fig, f"{prefix}_ffr_curve.png")
    fig = Figure(figsize=(6, 3.2), facecolor=SURFACE)
    ax = fig.add_axes([0.12, 0.2, 0.84, 0.68])
    style_axes(ax)
    used = cons.used
    ax.bar([str(f.frame) for f in used], [f.values.ffr for f in used], color=SERIES_1, width=0.6)
    ax.axhline(cons.ffr, color=SERIES_2, lw=2, label=f"median {cons.ffr:.2f}")
    ax.axhline(0.80, color=INK_MUTED, ls="--", lw=1)
    ax.set_ylim(max(0.0, min(f.values.ffr for f in used) - 0.1), 1.0)
    ax.set_xlabel("end-diastolic frame (heartbeat)", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("virtual FFR", fontsize=8.5, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8)
    save(fig, f"{prefix}_heartbeats.png")
    return {"case": case.label, "frame": frame, "ffr": round(cons.ffr, 3),
            "heartbeats": [round(f.values.ffr, 3) for f in used],
            "lesions": [m.text() for m in lesions],
            "stent": None if stent is None else
            guided.stent_text([stent], post.distal_solver)}  # fmt: skip


def repeatability_chart() -> None:
    rep = json.loads(Path("reports/repeatability.json").read_text(encoding="utf-8"))["summary"]
    con = json.loads(Path("reports/consensus.json").read_text(encoding="utf-8"))["summary"]
    vals = [100 * rep["frames"]["ffr_cv"], 100 * con["test"]["ffr_cv"]]
    fig = Figure(figsize=(6, 3.4), facecolor=SURFACE)
    ax = fig.add_axes([0.12, 0.18, 0.84, 0.7])
    style_axes(ax)
    ax.bar(["one frame", "median of 5 frames"], vals, color=[INK_MUTED, SERIES_1], width=0.55)
    ax.axhline(9.4, color=SERIES_2, ls="--", lw=1.5)
    ax.text(1.3, 9.9, "QFR inter-observer CV 9.4% (QREP)", ha="right", fontsize=7.5,
            color=SERIES_2)  # fmt: skip
    for k, v in enumerate(vals):
        ax.text(k, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel("FFR coefficient of variation (%)", fontsize=8.5, color=INK_SECONDARY)
    ax.set_title("Repeatability on real angiograms (CADICA test patients)", fontsize=9.5,
                 loc="left", color=INK_PRIMARY)  # fmt: skip
    save(fig, "repeatability.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in REPORT_FIGURES.items():
        shutil.copyfile(src, OUT / name)
        print("copied", src)
    pipeline()
    repeatability_chart()
    seg = SegmentationRunner.from_dir(latest_model_dir(Path("models/seg"), SEG_ONNX_FILE))
    lib = guided.case_library()
    synthetic = next(c for c in lib if c.kind == "synthetic")
    facts = {"synthetic": workflow(seg, "Synthetic beating heart", synthetic, "synthetic")}
    real = next((c for c in lib if c.label.startswith("CADICA p11 run 16")), None)
    if real is not None:
        facts["cadica"] = workflow(seg, "CADICA p11 (expert 90–98%)", real, "cadica", True)
    (OUT / "figures.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
