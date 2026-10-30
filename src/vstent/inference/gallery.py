"""Qualitative QCA gallery on ARCADE ``stenosis`` images (``vstent qca-gallery``).

For each image: segmentation (ONNX) → longest vessel path (auto-selected) → QCA → FFR
(solver and surrogate), drawn with the ARCADE stenosis boxes. The table counts how often the
auto-selected path crosses an annotated stenosis box and whether the MLD falls inside one: a
coarse plausibility check, not a validation (boxes mark stenoses; FFR is not annotated).
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.data.arcade import load_split
from vstent.imaging.centerline import longest_path_endpoints
from vstent.imaging.segment import THRESHOLD
from vstent.inference.figures import plot_analysis, plot_tree_analysis
from vstent.inference.onnx_runtime import SegmentationRunner, SurrogateRunner
from vstent.inference.pipeline import analyse_tree, analyse_vessel
from vstent.physics.solver import PhysicsParams

GALLERY_DIR = Path("reports/figures/qca_examples")
TREE_GALLERY_DIR = Path("reports/figures/qca_tree_examples")
EXTREME_FFR = 0.30  # single-path distal FFR below this ...
SEVERE_DS_PCT = 70.0  # ... without a lesion of at least this %DS is implausible (P2)


def _inside(points_rc: np.ndarray, boxes) -> np.ndarray:
    """Which (row, col) points lie inside any (x, y, w, h) box."""
    pts = np.atleast_2d(points_rc)
    hit = np.zeros(len(pts), bool)
    for x, y, w, h in boxes:
        hit |= (pts[:, 1] >= x) & (pts[:, 1] <= x + w) & (pts[:, 0] >= y) & (pts[:, 0] <= y + h)
    return hit


def qca_gallery(
    raw_dir: str | Path,
    seg_dir: str | Path,
    surrogate_dir: str | Path,
    *,
    params: PhysicsParams,
    n_images: int = 20,
    split: str = "val",
    out_dir: str | Path = GALLERY_DIR,
    report_path: str | Path = "reports/qca_gallery.md",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Analyse the first ``n_images`` ``stenosis/<split>`` images; write figures and a table."""
    log = log or (lambda _msg: None)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out_dir, report_path = Path(out_dir), Path(report_path)
    data = load_split(raw_dir, "stenosis", split, n_images)
    seg = SegmentationRunner.from_dir(seg_dir)
    sur = SurrogateRunner.from_dir(surrogate_dir)
    rows = []
    for img, boxes, name in zip(data.images, data.boxes, data.file_names, strict=True):
        stem = f"{split}_{Path(name).stem}"
        prob = seg.predict(img)
        try:
            start, end = longest_path_endpoints(prob >= THRESHOLD)
            res = analyse_vessel(prob, start, end, params=params, surrogate=sur)
        except ValueError as exc:
            rows.append({"image": stem, "error": str(exc)})
            log(f"{stem}: {exc}")
            continue
        q = res.qca
        plot_analysis(
            res, img, out_dir / f"{stem}.png", title=f"ARCADE stenosis/{split} {name}", boxes=boxes
        )
        rows.append(
            {
                "image": stem,
                "path_mm": float(q.s_mm[-1]),
                "max_ds_pct": q.max_percent_stenosis,
                "mld_mm": q.mld_mm,
                "ffr_solver": res.ffr_distal_solver,
                "ffr_surrogate": res.ffr_distal_surrogate,
                "n_boxes": len(boxes),
                "path_crosses_box": bool(_inside(q.centerline.points, boxes).any()),
                "mld_in_box": bool(_inside(q.mld_point_px, boxes)[0]),
                "max_ds_in_box": bool(_inside(q.max_ds_point_px, boxes)[0]),
                "n_warnings": len(res.warnings),
                "out_of_range": any("training range" in w for w in res.warnings),
            }
        )
        log(
            f"{stem}: path {q.s_mm[-1]:.0f} mm, max DS {q.max_percent_stenosis:.0f}%, "
            f"FFR {res.ffr_distal_solver:.2f} / {res.ffr_distal_surrogate:.2f}"
        )
    sur.close()
    ok = [r for r in rows if "error" not in r]
    crossing = [r for r in ok if r["path_crosses_box"]]
    summary = {
        "n_images": len(rows),
        "n_analysed": len(ok),
        "n_failed": len(rows) - len(ok),
        "n_path_crosses_box": len(crossing),
        "n_mld_in_box_given_crossing": sum(r["mld_in_box"] for r in crossing),
        "n_max_ds_in_box_given_crossing": sum(r["max_ds_in_box"] for r in crossing),
        "n_out_of_surrogate_range": sum(r["out_of_range"] for r in ok),
        "median_abs_ffr_diff": float(
            np.median([abs(r["ffr_solver"] - r["ffr_surrogate"]) for r in ok])
        )
        if ok
        else None,
    }
    report = {
        "summary": summary,
        "rows": rows,
        "provenance": provenance,
        "seg_model": Path(seg_dir).as_posix(),
        "surrogate_model": Path(surrogate_dir).as_posix(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    report_path.write_text(_markdown(report, out_dir, split), encoding="utf-8", newline="\n")
    return report


def _markdown(report: dict[str, Any], out_dir: Path, split: str) -> str:
    s, prov = report["summary"], report["provenance"]
    lines = [
        "# QCA gallery on ARCADE stenosis images (qualitative)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent qca-gallery`. Do not edit by hand.",
        "",
        f"The first {s['n_images']} `stenosis/{split}` images (by COCO id). For each one the "
        "longest path of the largest segmented vessel component is selected automatically, so "
        "it often is **not** the vessel with the annotated stenosis. Pixel size: 0.30 mm/px "
        "**assumed** (ARCADE has no calibration), so lengths, diameters and FFR are "
        "illustrative only. The model has no side branches: the full inflow is carried along "
        "the whole path, so long paths that taper into thin distal branches show large "
        "viscous pressure losses that real flow, which leaves through the branches, would not "
        "produce. Dashed orange boxes: ARCADE stenosis annotations.",
        "",
        f"- Analysed: {s['n_analysed']} of {s['n_images']} (failed: {s['n_failed']}).",
        f"- Auto-selected path crosses an annotated stenosis box: {s['n_path_crosses_box']}; "
        f"of those, point of maximum %DS inside a box: {s['n_max_ds_in_box_given_crossing']}, "
        f"MLD (narrowest absolute diameter, often in a thin distal branch) inside a box: "
        f"{s['n_mld_in_box_given_crossing']}.",
        f"- Outside the surrogate's training range (mostly vessel length): "
        f"{s['n_out_of_surrogate_range']} of {s['n_analysed']}.",
        "- Median |distal FFR solver − surrogate|: "
        + (f"{s['median_abs_ffr_diff']:.3f}" if s["median_abs_ffr_diff"] is not None else "n/a"),
        "",
        "| Image | Path (mm) | Max DS (%) | MLD (mm) | FFR solver | FFR surrogate | Boxes "
        "| Path crosses box | Max DS in box | MLD in box |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        if "error" in r:
            lines.append(f"| {r['image']} | failed: {r['error']} | | | | | | | | |")
            continue
        lines.append(
            f"| [{r['image']}](figures/qca_examples/{r['image']}.png) | {r['path_mm']:.0f} "
            f"| {r['max_ds_pct']:.0f} | {r['mld_mm']:.2f} | {r['ffr_solver']:.2f} "
            f"| {r['ffr_surrogate']:.2f} | {r['n_boxes']} "
            f"| {'yes' if r['path_crosses_box'] else 'no'} "
            f"| {'yes' if r['max_ds_in_box'] else 'no'} | {'yes' if r['mld_in_box'] else 'no'} |"
        )
    lines += [
        "",
        f"Models: segmentation `{report['seg_model']}`, surrogate `{report['surrogate_model']}`. "
        f"Commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), {prov['created_utc']}.",
        "",
    ]
    return "\n".join(lines)


def tree_gallery(
    raw_dir: str | Path,
    seg_dir: str | Path,
    *,
    params: PhysicsParams,
    n_images: int = 20,
    split: str = "val",
    out_dir: str | Path = TREE_GALLERY_DIR,
    report_path: str | Path = "reports/qca_tree_gallery.md",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The QCA gallery's images and paths with tree physics (P2): main-path distal FFR as a
    tree vs as a single path, and the cases where the single path is implausibly low."""
    log = log or (lambda _msg: None)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out_dir, report_path = Path(out_dir), Path(report_path)
    data = load_split(raw_dir, "stenosis", split, n_images)
    seg = SegmentationRunner.from_dir(seg_dir)
    rows = []
    for img, boxes, name in zip(data.images, data.boxes, data.file_names, strict=True):
        stem = f"{split}_{Path(name).stem}"
        prob = seg.predict(img)
        try:
            start, end = longest_path_endpoints(prob >= THRESHOLD)
            res = analyse_tree(prob, start, end, params=params)
        except ValueError as exc:
            rows.append({"image": stem, "error": str(exc)})
            log(f"{stem}: {exc}")
            continue
        tq, sol = res.qca, res.result
        plot_tree_analysis(
            res, img, out_dir / f"{stem}.png", title=f"ARCADE stenosis/{split} {name}",
            boxes=boxes,
        )  # fmt: skip
        main = tq.branches[0]
        single, tree = float(res.single.ffr[-1]), float(sol.ffr[0][-1])
        rows.append(
            {
                "image": stem,
                "path_mm": float(main.s_mm[-1]),
                "max_ds_pct": main.max_percent_stenosis,
                "n_side": tq.tree.n_branches - 1,
                "n_dropped": len(tq.dropped),
                "side_mm": [float(q.s_mm[-1]) for q in tq.branches[1:]],
                "ffr_single": single,
                "ffr_tree": tree,
                "main_outlet_share": float(sol.q_outlet[0] / sol.q_inlet[0]),
                "side_outlet_ffr": [float(f[-1]) for f in sol.ffr[1:]],
                "extreme_single": single < EXTREME_FFR
                and main.max_percent_stenosis < SEVERE_DS_PCT,
                "extreme_tree": tree < EXTREME_FFR and main.max_percent_stenosis < SEVERE_DS_PCT,
            }
        )
        log(
            f"{stem}: {tq.tree.n_branches - 1} side branch(es); main FFR single {single:.2f} → "
            f"tree {tree:.2f}"
        )
    ok = [r for r in rows if "error" not in r]
    with_side = [r for r in ok if r["n_side"] > 0]
    summary = {
        "n_images": len(rows),
        "n_analysed": len(ok),
        "n_with_side_branches": len(with_side),
        "n_side_branches": sum(r["n_side"] for r in ok),
        "n_dropped": sum(r["n_dropped"] for r in ok),
        "median_change_with_side": float(
            np.median([r["ffr_tree"] - r["ffr_single"] for r in with_side])
        )
        if with_side
        else None,
        "n_extreme_single": sum(r["extreme_single"] for r in ok),
        "n_extreme_tree": sum(r["extreme_tree"] for r in ok),
    }
    report = {
        "summary": summary,
        "rows": rows,
        "criteria": {"extreme_ffr": EXTREME_FFR, "severe_ds_pct": SEVERE_DS_PCT},
        "provenance": provenance,
        "seg_model": Path(seg_dir).as_posix(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    report_path.write_text(_tree_markdown(report, split), encoding="utf-8", newline="\n")
    return report


def _tree_markdown(report: dict[str, Any], split: str) -> str:
    s, prov, c = report["summary"], report["provenance"], report["criteria"]
    change = s["median_change_with_side"]
    lines = [
        "# QCA gallery with vessel trees (P2, qualitative)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent qca-gallery --tree`. Do not edit by hand.",
        "",
        "The images and auto-selected paths of `reports/qca_gallery.md`. The side branches "
        "leaving the path are measured too (docs/PHYSICS.md, *Vessel trees*), and FFR comes "
        "from the tree solver. *Single path* is the main path alone with the single-vessel "
        "solver, the same value as the solver column in the single-path gallery. Pixel size "
        "0.30 mm/px assumed. There is no ground truth: this checks that the side-branch effect "
        "behaves plausibly. It is not a validation.",
        "",
        "**What limits it.** The segmentation model was trained on ARCADE's SYNTAX-segment "
        "masks, which leave smaller side branches unannotated. It therefore rarely segments "
        "them, so most trees have few side branches and their flow stays in the main path. "
        "Lowering the probability threshold does not help: on the first 10 images the skeleton "
        "had the same junctions at 0.5, 0.3 and 0.1.",
        "",
        f"- Analysed: {s['n_analysed']} of {s['n_images']}; with at least one side branch: "
        f"{s['n_with_side_branches']} ({s['n_side_branches']} side branches; {s['n_dropped']} "
        "further side paths too short, too thin or too deep to model).",
        "- Median change of main-path distal FFR where side branches were found: "
        + (f"{change:+.3f}" if change is not None else "n/a")
        + ".",
        f"- Implausibly low main-path FFR (< {c['extreme_ffr']:.2f} without a lesion of "
        f"≥ {c['severe_ds_pct']:.0f}% DS): {s['n_extreme_single']} as single paths, "
        f"{s['n_extreme_tree']} as trees.",
        "",
        "| Image | Main path (mm) | Max DS (%) | Side branches | FFR single path | FFR tree "
        "| Change | Inflow reaching the main outlet | Side outlet FFR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        if "error" in r:
            lines.append(f"| {r['image']} | failed: {r['error']} | | | | | | | |")
            continue
        flag = " ⚠" if r["extreme_single"] else ""
        sides = ", ".join(f"{f:.2f}" for f in r["side_outlet_ffr"]) or "—"
        lines.append(
            f"| [{r['image']}](figures/qca_tree_examples/{r['image']}.png) | {r['path_mm']:.0f} "
            f"| {r['max_ds_pct']:.0f} | {r['n_side']} | {r['ffr_single']:.2f}{flag} "
            f"| {r['ffr_tree']:.2f} | {r['ffr_tree'] - r['ffr_single']:+.2f} "
            f"| {r['main_outlet_share']:.0%} | {sides} |"
        )
    extreme = [r for r in report["rows"] if "error" not in r and r["extreme_single"]]
    lines += ["", "## Implausibly low single-path FFR, before and after", ""]
    if extreme:
        lines += [
            f"- {r['image']}: {r['ffr_single']:.2f} → {r['ffr_tree']:.2f} with "
            f"{r['n_side']} side branch(es); main path {r['path_mm']:.0f} mm, max DS "
            f"{r['max_ds_pct']:.0f}%."
            + (
                ""
                if r["n_side"]
                else " No side branch was segmented, so the tree cannot change it: the long "
                "path tapering into thin distal vessels still carries the full inflow."
            )
            for r in extreme
        ]
    else:
        lines.append("None in this set.")
    lines += [
        "",
        "⚠ = implausibly low as a single path. Figures: dotted lines would show a virtual "
        "stent (none here); the dashed orange line is the main path as a single vessel.",
        "",
        f"Segmentation model `{report['seg_model']}`. Commit `{prov['git_commit']}` "
        f"(dirty: {prov['git_dirty']}), {prov['created_utc']}.",
        "",
    ]
    return "\n".join(lines)
