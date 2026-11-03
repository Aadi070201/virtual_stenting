"""P4k-B: does the lesion-seeking automatic path reach the experts' lesions more often than the
main trunk? (docs/PLAN.md, P4k-B; pre-registered.)

Set: the lesion runs of CADICA's development (or test) patients. On each run, the
expert-annotated keyframe nearest the product's analysis frame (the end-diastolic rule, P7,
from the cached per-frame curves). Metric: the share of runs whose automatic path crosses at
least one expert box with ≥ 5 centreline points inside. Both paths are measured with the
product's QCA and lesion reading (``configs/product.yaml``). Adopt the lesion-seeking path if
its development share is ≥ 10 points above the main trunk's. Torch-free.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

DISCLAIMER = "Research use only — not for clinical decisions."
MIN_INSIDE = 5
MIN_GAIN = 0.10
ARMS = ("trunk", "seeking")
_READER: dict[str, Any] = {}


def crosses(points: np.ndarray, boxes, min_inside: int = MIN_INSIDE) -> bool:
    """Whether ≥ ``min_inside`` of the path's points (row, col) lie in one of ``boxes``
    ((x, y, w, h) px)."""
    r, c = points[:, 0], points[:, 1]
    return any(int(((c >= x) & (c <= x + w) & (r >= y) & (r <= y + h)).sum()) >= min_inside
               for x, y, w, h in boxes)  # fmt: skip


def run_items(stage: str, stats: dict[str, dict], raw_dir=None) -> list[dict[str, Any]]:
    """One item per lesion run of the ``stage`` patients: its analysis frame and the nearest
    expert keyframe (image and boxes)."""
    from vstent.data import cadica
    from vstent.validation.cine_frames import ed_frame, run_key
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    raw_dir = raw_dir or cadica.RAW_DIR
    boxes = cadica.load_lesion_boxes(raw_dir)
    split = patient_split([t.patient for t in lesion_tracks(boxes)])
    by_run: dict[tuple[str, str], dict[int, list]] = {}
    images: dict[tuple[str, str, int], Path] = {}
    for b in boxes:
        by_run.setdefault((b.patient, b.video), {}).setdefault(b.frame, []).append(
            (b.x, b.y, b.w, b.h)
        )
        images[(b.patient, b.video, b.frame)] = b.image
    items = []
    for run in cadica.load_selected_runs(raw_dir):
        key = (run.patient, run.video)
        if not run.lesion or split.get(run.patient) != stage or key not in by_run:
            continue
        st = stats[run_key(run)]
        frame = ed_frame(st["area"], st["spread"])  # 0-based
        kf = min(by_run[key], key=lambda f: (abs(f - (frame + 1)), f))  # file numbers: 1-based
        items.append({"run": run_key(run), "patient": run.patient, "analysis_frame": frame,
                      "keyframe": kf, "image": str(images[(*key, kf)]),
                      "boxes": by_run[key][kf]})  # fmt: skip
    return items


def _reader(reader_dir: str | None):
    if reader_dir not in _READER:
        from vstent.inference.product import product_reader

        _READER[reader_dir] = product_reader(reader_dir)
    return _READER[reader_dir]


def frame_task(image: np.ndarray, prob: np.ndarray, boxes, reader_dir: str | None,
               min_ref_mm: float | None) -> dict[str, Any]:  # fmt: skip
    """Both automatic paths on one keyframe: do they cross an expert box?"""
    from vstent.app.guided import auto_path, seeking_path
    from vstent.inference.pipeline import analyse_vessel
    from vstent.physics.solver import PhysicsParams

    reader, params = _reader(reader_dir), PhysicsParams()

    def measure(a, b):
        return analyse_vessel(prob, a, b, params=params, image=image, lesion_reader=reader,
                              wire=False, lesion_min_ref_mm=min_ref_mm)  # fmt: skip

    row: dict[str, Any] = {}
    try:
        trunk = auto_path(prob, None)
        row["trunk"] = crosses(measure(*trunk).qca.centerline.points, boxes)
        start, end, score = seeking_path(prob, None, measure)
        row["seeking"] = crosses(measure(start, end).qca.centerline.points, boxes)
        row["changed"] = (start, end) != trunk
        row["score"] = score
    except Exception as exc:  # noqa: BLE001 — a frame that cannot be measured is reported
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def measure_items(
    items: list[dict[str, Any]],
    segmenter,
    *,
    reader_dir: str | None,
    min_ref_mm: float | None,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """:func:`frame_task` on every item (segmentation in this process, paths in workers)."""
    from vstent.imaging.image_io import read_image_file

    def args(it):
        img = read_image_file(it["image"]).image
        return img, segmenter.predict(img), it["boxes"], reader_dir, min_ref_mm

    out: list[dict[str, Any]] = [{}] * len(items)
    if workers <= 1:
        for k, it in enumerate(items):
            out[k] = {**it, **frame_task(*args(it))}
            if progress:
                progress(k + 1, len(items))
        return out
    from concurrent.futures import ProcessPoolExecutor

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: dict = {}
        done = 0
        for k, it in enumerate(items):
            pending[pool.submit(frame_task, *args(it))] = k
            while len(pending) >= 2 * workers:
                fut = next(iter(pending))
                j = pending.pop(fut)
                out[j] = {**items[j], **fut.result()}
                done += 1
                if progress:
                    progress(done, len(items))
        for fut, j in pending.items():
            out[j] = {**items[j], **fut.result()}
            done += 1
            if progress:
                progress(done, len(items))
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each arm's share of runs crossing an expert box, the gain and the pre-registered rule."""
    from vstent.validation.stats import wilson

    ok = [r for r in rows if "error" not in r]
    out: dict[str, Any] = {"runs": len(rows), "measured": len(ok),
                           "patients": len({r["patient"] for r in ok}),
                           "changed": sum(1 for r in ok if r["changed"])}  # fmt: skip
    for arm in ARMS:
        k = sum(1 for r in ok if r[arm])
        lo, hi = wilson(k, len(ok)) if ok else (None, None)
        out[arm] = {"k": k, "n": len(ok), "share": k / len(ok) if ok else None, "ci": (lo, hi)}
    out["gain"] = (out["seeking"]["share"] - out["trunk"]["share"]) if ok else None
    out["only_seeking"] = sum(1 for r in ok if r["seeking"] and not r["trunk"])
    out["only_trunk"] = sum(1 for r in ok if r["trunk"] and not r["seeking"])
    out["adopt"] = bool(ok) and out["gain"] >= MIN_GAIN - 1e-12
    return out


def _arm_line(label: str, x: dict[str, Any]) -> str:
    lo, hi = x["ci"]
    ci = "" if lo is None else f" (95% CI {lo:.0%}–{hi:.0%})"
    return f"| {label} | {x['share']:.0%} ({x['k']}/{x['n']}){ci} |"


def section(title: str, s: dict[str, Any], *, rule: bool) -> list[str]:
    lines = [f"## {title}", "", "| Automatic path | Runs crossing an expert box |", "|---|---:|",
             _arm_line("Main trunk (product now)", s["trunk"]),
             _arm_line("Lesion-seeking (P4k-B)", s["seeking"]), "",
             f"{s['measured']} of {s['runs']} lesion runs measured ({s['patients']} patients); "
             f"the lesion-seeking path differs from the trunk on {s['changed']}. Only the "
             f"seeking path crosses a box on {s['only_seeking']} runs, only the trunk on "
             f"{s['only_trunk']}."]  # fmt: skip
    if rule:
        lines += ["", f"**Rule:** gain {100 * s['gain']:+.1f} points (adopt if ≥ +10) → "
                  f"{'adopt' if s['adopt'] else 'do not adopt'}."]  # fmt: skip
    return lines


def write_report(path: str | Path, sections: list[str], *, product: dict, seg_model: str,
                 git: dict[str, Any]) -> Path:  # fmt: skip
    import json

    lines = ["# P4k-B: the lesion-seeking automatic path", "", f"> **{DISCLAIMER}**", "",
             "Generated by `vstent target-path`. Do not edit by hand. Pre-registered in "
             "docs/PLAN.md, P4k-B. On the expert keyframe nearest each lesion run's analysis "
             "frame; a path crosses a box with ≥ 5 centreline points inside. Pixel size assumed "
             "0.30 mm.", "", *sections, "",
             f"Product: `{json.dumps(product)}`. Segmentation model `{seg_model}`. Commit "
             f"`{git.get('git_commit')}` (dirty: {git.get('git_dirty')})."]  # fmt: skip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
