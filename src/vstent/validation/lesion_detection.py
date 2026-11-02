"""P4k-E: the whole-image lesion detector against expert boxes (docs/PLAN.md, P4k-E). Torch-free.

A detection is a connected component (8-neighbour) of the detector's probability ≥ t. An expert
box is found when a detection overlaps it. False alarms: detections on lesion-free frames
(per frame), and detections overlapping no box on annotated frames (an upper bound: not every
lesion is boxed). The operating point is the highest t with validation sensitivity ≥ 95%.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import ndimage

from vstent.data.lesion_boxes import BoxImage, box_mask

THRESHOLDS = tuple(round(0.05 * k, 2) for k in range(1, 20))
TARGET_SENSITIVITY = 0.95
MAX_FREE_ALARMS = 2.0  # per lesion-free validation frame, for the app (P4k-E)
TEST_SENSITIVITY = 0.90
_EIGHT = np.ones((3, 3), bool)


def frame_counts(prob: np.ndarray, it: BoxImage, thresholds=THRESHOLDS) -> dict[str, list]:
    """Per threshold: boxes found (per box), detections, detections outside every box."""
    h, w = prob.shape
    side = max(h, w)  # boxes are in the frame's own pixels, as the probability map
    masks = [box_mask([b], side)[:h, :w].astype(bool) for b in it.boxes]
    found, n_det, n_out = [], [], []
    for t in thresholds:
        lab, n = ndimage.label(prob >= t, structure=_EIGHT)
        found.append([bool((lab[m] > 0).any()) for m in masks])
        inside = set()
        for m in masks:
            inside.update(int(v) for v in np.unique(lab[m]) if v)
        n_det.append(int(n))
        n_out.append(int(n - len(inside)))
    return {"found": found, "detections": n_det, "outside": n_out}


def measure(
    items: list[BoxImage],
    predict: Callable[[np.ndarray], np.ndarray],
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """One row per frame: its source, patient, lesion-free flag, grades and per-threshold
    counts. ``predict``: greyscale image → lesion probability (H, W)."""
    import cv2

    rows = []
    for k, it in enumerate(items):
        img = cv2.imread(str(it.path), cv2.IMREAD_GRAYSCALE)
        rows.append({"source": it.source, "patient": it.patient, "role": it.role,
                     "lesion_free": it.lesion_free, "grades": list(it.grades),
                     **frame_counts(predict(img), it)})  # fmt: skip
        if progress:
            progress(k + 1, len(items))
    return rows


def at_threshold(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Sensitivity, per-grade sensitivity and false alarms at ``THRESHOLDS[k]``."""
    found = [f for r in rows for f in r["found"][k]]
    free = [r["detections"][k] for r in rows if r["lesion_free"]]
    annotated = [r["outside"][k] for r in rows if not r["lesion_free"]]
    by_grade: dict[str, list[bool]] = {}
    for r in rows:
        for g, f in zip(r["grades"], r["found"][k], strict=False):
            by_grade.setdefault(g, []).append(f)
    return {
        "threshold": THRESHOLDS[k], "boxes": len(found),
        "sensitivity": float(np.mean(found)) if found else None,
        "free_frames": len(free), "alarms_per_free_frame": float(np.mean(free)) if free else None,
        "outside_per_annotated_frame": float(np.mean(annotated)) if annotated else None,
        "by_grade": {g: {"n": len(v), "sensitivity": float(np.mean(v))}
                     for g, v in sorted(by_grade.items())},
    }  # fmt: skip


def operating_point(rows: list[dict[str, Any]]) -> int | None:
    """Index of the highest threshold with sensitivity ≥ :data:`TARGET_SENSITIVITY`."""
    ok = [k for k in range(len(THRESHOLDS))
          if (at_threshold(rows, k)["sensitivity"] or 0.0) >= TARGET_SENSITIVITY]  # fmt: skip
    return max(ok) if ok else None


def dev_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The validation operating point, every source's numbers there, and the app rule."""
    k = operating_point(rows)
    if k is None:
        return {"threshold_index": None, "shown_in_app": False,
                "reason": "no threshold reaches 95% validation sensitivity"}  # fmt: skip
    pooled = at_threshold(rows, k)
    alarms = pooled["alarms_per_free_frame"]
    shown = alarms is not None and alarms <= MAX_FREE_ALARMS
    per_source = {s: at_threshold([r for r in rows if r["source"] == s], k)
                  for s in sorted({r["source"] for r in rows})}  # fmt: skip
    return {"threshold_index": k, "threshold": THRESHOLDS[k], "pooled": pooled,
            "per_source": per_source, "shown_in_app": shown,
            "reason": f"{alarms if alarms is None else round(alarms, 2)} detections per "
            f"lesion-free validation frame at t = {THRESHOLDS[k]} "
            f"(limit {MAX_FREE_ALARMS})"}  # fmt: skip


def test_results(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Each test source's numbers at the validation operating point."""
    return {s: at_threshold([r for r in rows if r["source"] == s], k)
            for s in sorted({r["source"] for r in rows})}  # fmt: skip
