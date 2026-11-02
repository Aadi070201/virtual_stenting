"""P4k-G: vessel identity (RCA, left main, LAD, LCx) on ARCADE and against CADICA's left / right
labels (docs/PLAN.md, P4k-G). Torch-free.

- ARCADE (val, then test): Dice per vessel (pooled over images) and the accuracy on vessel
  pixels (the predicted vessel of every annotated vessel pixel).
- CADICA (development, then test patients): a run is "right" when RCA pixels outnumber the
  left tree's on its analysis frame (the product's frame choice), compared with CADICA's
  ``CADICAprojections.json`` group.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vstent.data.vessel_id import CLASS_NAMES, N_CLASSES, side_of

MIN_ACCURACY = 0.90  # ARCADE val vessel-pixel accuracy, for the product (P4k-G)
MIN_AGREEMENT = 0.95  # CADICA development side agreement, for the product


def arcade_scores(images, labels, predict: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    """Dice per vessel and vessel-pixel accuracy; ``predict``: image → probabilities (C, H, W)."""
    inter, total = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    correct = vessel = 0
    for img, lab in zip(images, labels, strict=True):
        pred = predict(img).argmax(axis=0)
        for k in range(1, N_CLASSES):
            p, t = pred == k, lab == k
            inter[k] += (p & t).sum()
            total[k] += p.sum() + t.sum()
        on = lab > 0
        correct += int((pred[on] == lab[on]).sum())
        vessel += int(on.sum())
    dice = {CLASS_NAMES[k]: (float(2 * inter[k] / total[k]) if total[k] else None)
            for k in range(1, N_CLASSES)}  # fmt: skip
    return {"n": len(images), "dice": dice, "accuracy": correct / vessel if vessel else None}


def cadica_sides(
    stage: str,
    segmenter,
    predict,
    raw_dir=None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """One row per selected run of the ``stage`` patients: CADICA's side and the predicted one
    on the run's analysis frame."""
    from vstent.data import cadica
    from vstent.imaging.image_io import read_run
    from vstent.imaging.run_analysis import analyse_frames
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    raw_dir = raw_dir or cadica.RAW_DIR
    split = patient_split([t.patient for t in lesion_tracks(cadica.load_lesion_boxes(raw_dir))])
    groups = cadica.load_artery_groups(raw_dir)
    runs = [r for r in cadica.load_selected_runs(raw_dir)
            if split.get(r.patient) == stage and f"{r.patient}_{r.video}" in groups]  # fmt: skip
    rows = []
    for k, r in enumerate(runs):
        run = read_run(r.frames_dir)
        frame = analyse_frames(run, segmenter.predict).analysis_frame
        probs = predict(run.frames[frame])
        vessel = segmenter.predict(run.frames[frame]) >= 0.5
        labels = np.where(vessel, probs[1:].argmax(axis=0) + 1, 0)  # vessel pixels only
        key = f"{r.patient}_{r.video}"
        rows.append({"run": key, "patient": r.patient, "frame": int(frame),
                     "truth": groups[key], "predicted": side_of(labels)})  # fmt: skip
        if progress:
            progress(k + 1, len(runs))
    return rows


def agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The share of runs whose predicted side matches CADICA's (a run without vessel pixels
    counts as a miss), with a Wilson 95% interval."""
    from vstent.validation.stats import wilson

    hits = sum(r["predicted"] == r["truth"] for r in rows)
    n = len(rows)
    lo, hi = wilson(hits, n) if n else (None, None)
    return {"n": n, "agree": hits, "share": hits / n if n else None, "ci95": [lo, hi],
            "per_side": {s: sum(r["predicted"] == s for r in rows if r["truth"] == s)
                         for s in ("left", "right")},
            "n_side": {s: sum(r["truth"] == s for r in rows)
                       for s in ("left", "right")}}  # fmt: skip


def decision(val: dict[str, Any], dev: dict[str, Any]) -> dict[str, Any]:
    """The P4k-G rule: ARCADE val accuracy ≥ 90% and CADICA development agreement ≥ 95%."""
    acc, agr = val["accuracy"], dev["share"]
    ok = acc is not None and agr is not None and acc >= MIN_ACCURACY and agr >= MIN_AGREEMENT
    return {"used_in_product": bool(ok),
            "reason": f"ARCADE val vessel-pixel accuracy {acc:.1%} (≥ 90%), CADICA development "
            f"agreement {agr:.1%} (≥ 95%)" if acc is not None and agr is not None
            else "a statistic could not be computed"}  # fmt: skip
