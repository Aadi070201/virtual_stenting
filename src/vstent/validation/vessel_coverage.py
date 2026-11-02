"""Vessel coverage of the hybrid segmentation on real angiograms (P4j). Torch-free.

Does the segmentation reach the lesions the experts marked? For every expert box, a box is
covered when a measurable vessel path crosses it (``path_through_box`` on the mask's skeleton,
≥ 5 px), as the product needs. ARCADE's vessel masks check that the added vessels lie on
vessels (recall; the share of added pixels within :data:`NEAR_PX` of an annotated vessel).
Two stages, as pre-registered in docs/PLAN.md (P4j): ``dev`` (CADICA development patients,
ARCADE val; the threshold and the adoption decision), then ``test`` once (CADICA test patients,
ARCADE test, Kemerovo).
"""

import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from vstent import DISCLAIMER, __version__
from vstent.imaging.centerline import vessel_graph
from vstent.imaging.image_io import read_image_file
from vstent.imaging.vesselness import SEED_LEVEL, THRESHOLDS, added_mask, vesselness
from vstent.validation.stenosis_grading import path_through_box

NEAR_PX = 5.0
MIN_GAIN = 0.10  # adoption: development box coverage + 10 points
MIN_NEAR_SHARE = 0.50
DEV_REPORT, REPORT = "vessel_coverage_dev", "vessel_coverage"


def _masks(image, prob, thresholds) -> dict[str, np.ndarray]:
    v = vesselness(image)
    model = prob >= SEED_LEVEL
    return {"model": model, **{f"{t:g}": model | added_mask(prob, v, t) for t in thresholds}}


def _box_task(image, prob, boxes, thresholds) -> list[dict[str, bool]]:
    masks = _masks(image, prob, thresholds)
    graphs = {k: vessel_graph(m) for k, m in masks.items()}
    out = []
    for box in boxes:
        row = {}
        for k, g in graphs.items():
            path = path_through_box(g, box)
            row[k] = bool(path is not None and len(path) >= 5)
        out.append(row)
    return out


def _arcade_task(image, prob, gt, thresholds) -> dict[str, dict[str, float]]:
    masks = _masks(image, prob, thresholds)
    near = distance_transform_edt(~gt) <= NEAR_PX
    model = masks["model"]
    out = {}
    for k, m in masks.items():
        added = m & ~model
        out[k] = {"gt": float(gt.sum()), "hit": float((m & gt).sum()),
                  "added": float(added.sum()),
                  "added_near": float((added & near).sum())}  # fmt: skip
    return out


def _parallel(items, task, segmenter, workers, progress) -> list:
    """``task(image, model probability, arg, ...)`` per (path or image, arg) item."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    results: list = [None] * len(items)

    def image_of(src):
        return read_image_file(src).image if isinstance(src, (str, Path)) else src

    if workers <= 1:
        for k, (src, *args) in enumerate(items):
            img = image_of(src)
            results[k] = task(img, segmenter.predict_model(img), *args)
            if progress:
                progress(k + 1, len(items))
        return results
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for k, (src, *args) in enumerate(items):
            img = image_of(src)
            pending[pool.submit(task, img, segmenter.predict_model(img), *args)] = k
            while len(pending) >= 2 * workers:
                fut = next(iter(pending))
                results[pending.pop(fut)] = fut.result()
                done += 1
                if progress:
                    progress(done, len(items))
        for fut, k in pending.items():
            results[k] = fut.result()
            done += 1
            if progress:
                progress(done, len(items))
    return results


def cadica_boxes(split: str) -> list[tuple[Path, list]]:
    """Every CADICA expert box of the ``split`` patients, grouped by frame."""
    from vstent.data import cadica
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    boxes = cadica.load_lesion_boxes()
    which = patient_split([t.patient for t in lesion_tracks(boxes)])
    by: dict[Path, list] = {}
    for b in boxes:
        if which.get(b.patient) == split:
            by.setdefault(b.image, []).append((b.x, b.y, b.w, b.h))
    return sorted(by.items(), key=lambda kv: str(kv[0]))


def kemerovo_boxes(raw_dir="data/raw/kemerovo") -> list[tuple[Path, list]]:
    from vstent.data.kemerovo import load_boxes

    by: dict[Path, list] = {}
    for b in load_boxes(raw_dir):
        by.setdefault(b.image, []).append((b.x, b.y, b.w, b.h))
    return sorted(by.items(), key=lambda kv: str(kv[0]))


def box_coverage(frames, segmenter, thresholds, workers, progress=None) -> dict[str, Any]:
    items = [(path, boxes, tuple(thresholds)) for path, boxes in frames]
    rows = [r for out in _parallel(items, _box_task, segmenter, workers, progress) or []
            for r in out]  # fmt: skip
    keys = ["model", *(f"{t:g}" for t in thresholds)]
    return {"n_boxes": len(rows), **{k: float(np.mean([r[k] for r in rows])) if rows else None
                                     for k in keys}}  # fmt: skip


def arcade_coverage(split: str, segmenter, thresholds, workers, progress=None,
                    raw_dir="data/raw/arcade") -> dict[str, Any]:  # fmt: skip
    from vstent.data.arcade import load_split

    data = load_split(raw_dir, "syntax", split)
    items = [(img, m.astype(bool), tuple(thresholds))
             for img, m in zip(data.images, data.masks, strict=True)]  # fmt: skip
    outs = _parallel(items, _arcade_task, segmenter, workers, progress)
    out = {}
    for k in outs[0]:
        s = {f: sum(o[k][f] for o in outs) for f in ("gt", "hit", "added", "added_near")}
        near_share = s["added_near"] / s["added"] if s["added"] else None
        out[k] = {"recall": s["hit"] / s["gt"] if s["gt"] else None,
                  "added_px_per_image": s["added"] / len(outs),
                  "added_near_share": near_share}  # fmt: skip
    return {"n_images": len(outs), **out}


def choose(dev_boxes: dict[str, Any], arcade: dict[str, Any], thresholds) -> dict[str, Any]:
    """The pre-registered threshold choice and adoption rule."""
    ok = []
    for t in thresholds:
        k = f"{t:g}"
        a = arcade[k]
        if (a["recall"] >= arcade["model"]["recall"]
                and (a["added_near_share"] or 0.0) >= MIN_NEAR_SHARE):  # fmt: skip
            ok.append(t)
    if not ok:
        return {"threshold": None, "eligible": [], "adopted": False,
                "reason": "no threshold met the ARCADE conditions"}  # fmt: skip
    best = max(ok, key=lambda t: (dev_boxes[f"{t:g}"], -t))
    gain = dev_boxes[f"{best:g}"] - dev_boxes["model"]
    reason = f"coverage gain {100 * gain:.1f} points (≥ {100 * MIN_GAIN:.0f} needed)"
    return {"threshold": best, "eligible": ok, "gain": gain, "adopted": gain >= MIN_GAIN,
            "reason": reason}  # fmt: skip


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def write_report(report_dir, name: str, title: str, sections: list[str], data: dict,
                 git: dict, model: str) -> None:  # fmt: skip
    prov = {**git, "vstent_version": __version__, "seg_model": model,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    lines = [f"# {title}", "", f"> **{DISCLAIMER}**", "",
             "Generated by `vstent vessel-coverage`. Do not edit by hand.", "", *sections, "",
             f"Segmentation model `{model}`. Commit `{prov['git_commit']}` (dirty: "
             f"{prov['git_dirty']}), created {prov['created_utc']}.", ""]  # fmt: skip
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (report_dir / f"{name}.json").write_text(json.dumps({**data, "provenance": prov}, indent=1,
                                                        default=float),
                                             encoding="utf-8", newline="\n")  # fmt: skip


def dev_sections(d: dict[str, Any]) -> list[str]:
    b, a, c = d["cadica_dev"], d["arcade_val"], d["choice"]
    keys = ["model", *(f"{t:g}" for t in THRESHOLDS)]
    return [
        "## Development (CADICA development patients; ARCADE val)", "",
        "| Segmentation | CADICA boxes covered | ARCADE recall | Added px per image | "
        "Added px near annotated vessels |", "|---|---:|---:|---:|---:|",
        *(f"| {'model' if k == 'model' else 'hybrid T = ' + k} | {_pct(b[k])} | "
          f"{_pct(a[k]['recall'])} | {a[k]['added_px_per_image']:.0f} | "
          f"{_pct(a[k]['added_near_share'])} |" for k in keys), "",
        f"{b['n_boxes']} CADICA boxes; {a['n_images']} ARCADE val images.", "",
        f"**Rule** (docs/PLAN.md, P4j): eligible thresholds {c['eligible']}; chosen "
        f"{c['threshold']}; {c['reason']}.", "",
        f"**Decision: {'adopted' if c['adopted'] else 'not adopted'}.**",
    ]  # fmt: skip


def test_sections(t: dict[str, Any], threshold: float) -> list[str]:
    k = f"{threshold:g}"
    return [
        f"## Test (once; hybrid T = {k})", "",
        "| Data | Boxes / images | Model | Hybrid |", "|---|---:|---:|---:|",
        f"| CADICA test patients: boxes covered | {t['cadica_test']['n_boxes']} | "
        f"{_pct(t['cadica_test']['model'])} | {_pct(t['cadica_test'][k])} |",
        f"| Kemerovo: boxes covered | {t['kemerovo']['n_boxes']} | "
        f"{_pct(t['kemerovo']['model'])} | {_pct(t['kemerovo'][k])} |",
        f"| ARCADE test: recall | {t['arcade_test']['n_images']} | "
        f"{_pct(t['arcade_test']['model']['recall'])} | {_pct(t['arcade_test'][k]['recall'])} |",
        f"| ARCADE test: added px near annotated vessels | | – | "
        f"{_pct(t['arcade_test'][k]['added_near_share'])} |",
    ]  # fmt: skip


def run_dev(segmenter, workers: int, log: Callable[[str], None]) -> dict[str, Any]:
    def prog(name):
        return lambda k, n: log(f"  {name} {k} / {n}") if k % 250 == 0 or k == n else None

    boxes = box_coverage(cadica_boxes("dev"), segmenter, THRESHOLDS, workers, prog("cadica"))
    arcade = arcade_coverage("val", segmenter, THRESHOLDS, workers, prog("arcade val"))
    return {"cadica_dev": boxes, "arcade_val": arcade,
            "choice": choose(boxes, arcade, THRESHOLDS)}  # fmt: skip


def run_test(segmenter, threshold: float, workers: int,
             log: Callable[[str], None]) -> dict[str, Any]:  # fmt: skip
    def prog(name):
        return lambda k, n: log(f"  {name} {k} / {n}") if k % 500 == 0 or k == n else None

    ts = (threshold,)
    return {"cadica_test": box_coverage(cadica_boxes("test"), segmenter, ts, workers,
                                        prog("cadica test")),
            "kemerovo": box_coverage(kemerovo_boxes(), segmenter, ts, workers,
                                     prog("kemerovo")),
            "arcade_test": arcade_coverage("test", segmenter, ts, workers,
                                           prog("arcade test"))}  # fmt: skip
