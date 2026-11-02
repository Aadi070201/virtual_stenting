"""P4m: virtual stenting checked on real angiograms without FFR (docs/PLAN.md, P4m). Torch-free.

For CADICA lesions graded 70–98% (one box per lesion), the product measures the vessel on the
clinician-style path through the box and, when its FFR is ≤ 0.80, the guided app's planner
proposes stents. Checked: whether a stent covers the experts' lesion (its deepest point inside
the box), and the predicted post-PCI FFR against FFR-SEARCH (0.91 ± 0.07).

Arms: ``product``; ``wire`` (P4k-C2: the vessel up to the last point with a reference diameter
≥ 2.0 mm, its FFR read there).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from vstent.physics.solver import PhysicsParams, solve_ffr_batch

ARMS = ("product", "wire")
CUTOFF = 0.80
COVER_TARGET = 0.90
POST_MEAN_RANGE = (0.88, 0.94)  # FFR-SEARCH: 0.91 ± 0.07
POST_LOW_MAX = 0.15  # share ≤ 0.80 after stenting
GRADES = ("p70_90", "p90_98")


def middle_boxes(stage: str, raw_dir=None) -> list[tuple]:
    """(track, patient, category, box) of every 70–98% lesion of the ``stage`` patients, the
    box on the frame nearest the middle of the lesion's boxes."""
    from vstent.data import cadica
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    tracks = lesion_tracks(cadica.load_lesion_boxes(raw_dir or cadica.RAW_DIR))
    split = patient_split([t.patient for t in tracks])
    out = []
    for t in tracks:
        if t.category in GRADES and split[t.patient] == stage:
            boxes = sorted(t.boxes, key=lambda b: b.frame)
            out.append((t.key, t.patient, t.category, boxes[len(boxes) // 2]))
    return out


def _deepest_in_box(q, box) -> float | None:
    """Arc length (m) of the deepest point of the path inside ``box`` (x, y, w, h)."""
    x, y, w, h = box
    p = q.centerline.points
    inside = (p[:, 0] >= y) & (p[:, 0] <= y + h) & (p[:, 1] >= x) & (p[:, 1] <= x + w)
    if not inside.any():
        return None
    ds = 1.0 - q.profile.r / q.profile.reference_radius()
    return float(q.profile.s[int(np.argmax(np.where(inside, ds, -np.inf)))])


def _arm_profile(profile, arm: str):
    from vstent.imaging.qca import wire_index
    from vstent.physics.geometry import VesselProfile

    r_ref = profile.reference_radius()
    if arm == "product":
        return VesselProfile(profile.s, profile.r, r_ref)
    k = max(wire_index(r_ref), 2)
    return VesselProfile(profile.s[: k + 1], profile.r[: k + 1], r_ref[: k + 1])


def lesion_task(img, prob, item, params_dict) -> dict[str, Any]:
    """One lesion through the product and the planner, for every arm."""
    from vstent.app.core import load_settings
    from vstent.app.guided import box_path
    from vstent.inference.pipeline import analyse_vessel
    from vstent.planning.optimizer import plan_stents

    track, patient, category, b = item
    box = (b.x, b.y, b.w, b.h)
    params = PhysicsParams(**params_dict)
    _, planner, catalog = load_settings()
    planner = planner.model_copy(update={"backend": "solver"})
    row: dict[str, Any] = {"track": track, "patient": patient, "category": category}
    try:
        path = box_path(prob, box)
        if path is None:
            raise ValueError("no vessel in the box")
        q = analyse_vessel(prob, path[0], path[1], params=params, image=img).qca
        at = _deepest_in_box(q, box)
        if at is None:
            raise ValueError("the path does not enter the box")
        for arm in ARMS:
            prof = _arm_profile(q.profile, arm)
            pre = float(solve_ffr_batch(prof.r, prof.ds, prof.r_ref, params).ffr_distal[0])
            res: dict[str, Any] = {"pre": pre, "planned": False, "in_reach": bool(at <= prof.s[-1])}
            if pre <= CUTOFF:
                plans = plan_stents(prof, catalog=catalog, config=planner, params=params)
                if plans.plans:
                    best = plans.plans[0]
                    covered = any(s.s_start <= at <= s.s_end for s in best.stents)
                    res.update(planned=True, post=float(best.ffr),
                               reaches=bool(best.meets_target), n_stents=len(best.stents),
                               length_mm=1e3 * sum(s.length for s in best.stents),
                               covered=covered)  # fmt: skip
            row[arm] = res
    except Exception as exc:  # noqa: BLE001 — reported per lesion
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def measure(
    stage: str,
    segmenter,
    params: PhysicsParams,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """One row per lesion; segmentation here, the rest in ``workers`` processes."""
    from concurrent.futures import ProcessPoolExecutor

    from vstent.imaging.image_io import read_image_file

    items = middle_boxes(stage)
    pd = params.model_dump()
    if workers <= 1:
        rows = []
        for k, it in enumerate(items):
            img = read_image_file(it[3].image).image
            rows.append(lesion_task(img, segmenter.predict(img), it, pd))
            if progress:
                progress(k + 1, len(items))
        return rows
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = []
        for it in items:
            img = read_image_file(it[3].image).image
            futs.append(pool.submit(lesion_task, img, segmenter.predict(img), it, pd))
        rows = []
        for k, f in enumerate(futs):
            rows.append(f.result())
            if progress:
                progress(k + 1, len(items))
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per arm: lesions measured, significant, planned; the stent on the lesion; post-PCI FFR."""
    ok = [r for r in rows if "error" not in r]
    out: dict[str, Any] = {"n_lesions": len(rows), "n_measured": len(ok), "arms": {}}
    for arm in ARMS:
        sig = [r[arm] for r in ok if r[arm]["pre"] <= CUTOFF]
        planned = [x for x in sig if x["planned"]]
        post = np.array([x["post"] for x in planned])
        cover = float(np.mean([x["covered"] for x in planned])) if planned else None
        mean = float(post.mean()) if post.size else None
        low = float(np.mean(post <= CUTOFF)) if post.size else None
        out["arms"][arm] = {
            "significant": len(sig), "planned": len(planned), "covered": cover,
            "post_mean": mean, "post_sd": float(post.std()) if post.size else None,
            "post_low": low,
            "reaches_090": float(np.mean([x["reaches"] for x in planned])) if planned else None,
            "median_length_mm": float(np.median([x["length_mm"] for x in planned]))
            if planned else None,
            "targets": {
                "covered ≥ 90%": cover is not None and cover >= COVER_TARGET,
                "post-PCI mean 0.88–0.94": mean is not None
                and POST_MEAN_RANGE[0] <= mean <= POST_MEAN_RANGE[1],
                "post-PCI ≤ 0.80 at most 15%": low is not None and low <= POST_LOW_MAX,
            },
        }  # fmt: skip
    return out


def section(stage: str, s: dict[str, Any]) -> list[str]:
    """Report lines of one stage."""
    title = "Development (CADICA development patients)" if stage == "dev" else (
        "Test (CADICA test patients)")  # fmt: skip
    arms = list(s["arms"])
    lines = [f"## {title}", "", "| | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]

    def pct(x):
        return "–" if x is None else f"{100 * x:.0f}%"

    def num(x):
        return "–" if x is None else f"{x:.2f}"

    def row(label, f):
        lines.append(f"| {label} | " + " | ".join(f(s["arms"][a]) for a in arms) + " |")

    row("Lesions 70–98% with FFR ≤ 0.80", lambda a: str(a["significant"]))
    row("… with a stent plan", lambda a: str(a["planned"]))
    row("Stent on the experts' lesion (target ≥ 90%)", lambda a: pct(a["covered"]))
    row(
        "Post-PCI FFR, mean ± SD (FFR-SEARCH 0.91 ± 0.07)",
        lambda a: "–" if a["post_mean"] is None else f"{a['post_mean']:.2f} ± {a['post_sd']:.2f}",
    )
    row("Post-PCI FFR ≤ 0.80 (target ≤ 15%)", lambda a: pct(a["post_low"]))
    row("Plans reaching 0.90", lambda a: pct(a["reaches_090"]))
    row("Median total stent length (mm)", lambda a: num(a["median_length_mm"]))
    lines += ["", f"{s['n_measured']} of {s['n_lesions']} lesions measured.", ""]
    return lines
