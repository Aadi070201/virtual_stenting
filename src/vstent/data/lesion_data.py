"""Lesion samples from real angiograms for the learned lesion reading (P4i). Torch-free.

Every sample is the patch and scalars of :func:`vstent.imaging.lesion_patch.lesion_patch` at the
point of largest measured %DS inside an expert box (the vessel path through the box, P4b), or at
a candidate narrowing far from every box:

- **CADICA** (CC BY 4.0): every expert box graded 0–98%, with the grade's %DS target, the
  significance target (grade ≥ 50%) and the patient's development / test split (P4b);
- **ARCADE stenosis** (CC0), per split: every stenosis box (significance 1), and up to
  :data:`NEGATIVES_PER_IMAGE` candidates on the automatic path farther than one box diagonal
  from every stenosis box (significance 0);
- **Kemerovo** (CC BY 4.0): every stenosis box (external test only).

Each frame is segmented once in this process; the measurements run in worker processes. The
lumen is the intensity edges (the product's shape; the absolute-size factor does not change
%DS). Written to ``data/derived/lesion/<source>.npz`` (patches float16) + ``.json``.

**Version 2** (P4k-A, ``data/derived/lesion_v2/``):
- box samples are measured on the clinician-style path (from the tree's proximal end through
  the box to the vessel's end, ``guided.box_path``), falling back to the P4b path;
- normal-vessel candidates are added from CADICA's lesion-free runs (every 4th selected
  frame, both splits; ``cadica_free``);
- every sample has a %DS target interval (:func:`ds_interval`).
"""

import json
import os
import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vstent.imaging.centerline import smooth_and_resample, vessel_graph
from vstent.imaging.image_io import read_image_file
from vstent.imaging.lesion_patch import (
    GRADE_TARGETS,
    candidate_at,
    candidates,
    lesion_patch,
    path_ds,
)
from vstent.imaging.qca import LEVEL, qca_from_centerline
from vstent.validation.stenosis_grading import path_through_box

OUT_DIR = Path("data/derived/lesion")
OUT_DIR_V2 = Path("data/derived/lesion_v2")
NEGATIVES_PER_IMAGE = 2
SEED = 0
AUTO_PATH_PX = 100.0 / 0.30  # the guided app's automatic path length (100 mm at 0.30 mm/px)
FREE_FRAME_STEP = 4  # P4k-A: every 4th selected frame of a lesion-free run
GRADE_RANGES = {"p0_20": (0.0, 20.0), "p20_50": (20.0, 50.0), "p50_70": (50.0, 70.0),
                "p70_90": (70.0, 90.0), "p90_98": (90.0, 98.0)}  # fmt: skip
SIGNIFICANT_RANGE = (50.0, 100.0)  # an ungraded stenosis box (ARCADE, Kemerovo)
NORMAL_RANGE = (0.0, 40.0)  # a normal-vessel candidate


def ds_interval(meta: dict[str, Any]) -> tuple[float, float]:
    """The %DS target interval of a sample (P4k-A): a CADICA grade's range, ≥ 50 for an
    ungraded stenosis box, ≤ 40 for a normal-vessel candidate."""
    if meta.get("category") in GRADE_RANGES:
        return GRADE_RANGES[meta["category"]]
    return SIGNIFICANT_RANGE if meta["sig_target"] >= 0.5 else NORMAL_RANGE


def _inside(q, box) -> np.ndarray:
    x, y, w, h = box
    p = q.centerline.points
    return (p[:, 0] >= y) & (p[:, 0] <= y + h) & (p[:, 1] >= x) & (p[:, 1] <= x + w)


def _clinician_qca(image, prob, box):
    """QCA on the path from the tree's proximal end through ``box`` (``guided.box_path``), or
    ``None``."""
    from vstent.app.guided import box_path
    from vstent.imaging.centerline import trace_centerline

    ends = box_path(prob, box)
    if ends is None:
        return None
    try:
        q = qca_from_centerline(prob, trace_centerline(prob >= LEVEL, *ends), image=image,
                                lumen="edges")  # fmt: skip
    except ValueError:
        return None
    return q if _inside(q, box).any() else None


def box_sample(image, prob, graph, box, clinician: bool = False) -> dict[str, Any] | None:
    """Patch, scalars and measured %DS at the deepest point inside ``box`` (x, y, w, h); with
    ``clinician``, measured on the clinician-style path when there is one (P4k-A)."""
    q = _clinician_qca(image, prob, box) if clinician else None
    if q is None:
        path = path_through_box(graph, box)
        if path is None or len(path) < 5:
            return None
        try:
            q = qca_from_centerline(prob, smooth_and_resample(path), image=image, lumen="edges")
        except ValueError:
            return None
    inside = _inside(q, box)
    if not inside.any():
        return None
    k = int(np.argmax(np.where(inside, path_ds(q), -np.inf)))
    c = candidate_at(q, k)
    patch, scalars = lesion_patch(image, prob, q.centerline, c)
    return {"patch": patch, "scalars": scalars, "measured_ds": max(c.ds_pct, 0.0)}


def negative_samples(image, prob, boxes, key: str) -> list[dict[str, Any]]:
    """Candidates on the automatic path farther than one box diagonal from every box."""
    from vstent.imaging.centerline import main_trunk_endpoints, trace_centerline

    try:  # the guided app's automatic path: the main trunk, at most 100 mm at 0.30 mm/px
        a, b = main_trunk_endpoints(prob >= LEVEL, AUTO_PATH_PX)
        q = qca_from_centerline(prob, trace_centerline(prob >= LEVEL, a, b), image=image,
                                lumen="edges")  # fmt: skip
    except ValueError:
        return []
    far = []
    for c in candidates(q):
        r, col = q.centerline.points[c.k]
        if all(np.hypot(r - (y + h / 2), col - (x + w / 2)) > np.hypot(w, h)
               for x, y, w, h in boxes):  # fmt: skip
            far.append(c)
    rng = np.random.default_rng([SEED, zlib.crc32(key.encode())])
    pick = rng.permutation(len(far))[:NEGATIVES_PER_IMAGE]
    out = []
    for i in sorted(pick):
        patch, scalars = lesion_patch(image, prob, q.centerline, far[i])
        out.append({"patch": patch, "scalars": scalars, "measured_ds": max(far[i].ds_pct, 0.0)})
    return out


def _frame_task(image, prob, jobs) -> list[dict[str, Any]]:
    """Every job of one frame: ("box" or "box_clinician", meta, box) or ("negatives", meta,
    boxes)."""
    graph = vessel_graph(prob >= LEVEL)
    out = []
    for kind, meta, arg in jobs:
        if kind in ("box", "box_clinician"):
            s = box_sample(image, prob, graph, arg, clinician=kind == "box_clinician")
            out.append({**meta, **(s or {"error": "no vessel path through the box"})})
        else:
            out += [{**meta, **s} for s in negative_samples(image, prob, arg, meta["key"])]
    return out


def _run(frames: list[tuple[Path, list]], segmenter, workers: int,
         progress: Callable[[int, int], None] | None) -> list[dict[str, Any]]:  # fmt: skip
    results: list = [None] * len(frames)
    if workers <= 1:
        for k, (path, jobs) in enumerate(frames):
            img = read_image_file(path).image
            results[k] = _frame_task(img, segmenter.predict(img), jobs)
            if progress:
                progress(k + 1, len(frames))
        return [s for r in results for s in r or []]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    done = 0
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        pending = {}
        for k, (path, jobs) in enumerate(frames):
            img = read_image_file(path).image
            pending[pool.submit(_frame_task, img, segmenter.predict(img), jobs)] = k
            while len(pending) >= 2 * max(workers, 1):
                fut = next(iter(pending))
                results[pending.pop(fut)] = fut.result()
                done += 1
                if progress:
                    progress(done, len(frames))
        for fut, k in pending.items():
            results[k] = fut.result()
            done += 1
            if progress:
                progress(done, len(frames))
    return [s for r in results for s in r or []]


def cadica_frames(raw_dir: str | Path | None = None) -> list[tuple[Path, list]]:
    """CADICA boxes graded 0–98%, grouped by frame, with their split and targets."""
    from vstent.data import cadica
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    tracks = lesion_tracks(cadica.load_lesion_boxes(raw_dir or cadica.RAW_DIR))
    split = patient_split([t.patient for t in tracks])
    by_frame: dict[Path, list] = {}
    for t in tracks:
        if t.category not in GRADE_TARGETS:
            continue
        for b in t.boxes:
            meta = {"source": "cadica", "split": split[t.patient], "patient": t.patient,
                    "track": t.key, "frame": b.frame, "category": t.category,
                    "ds_target": GRADE_TARGETS[t.category],
                    "sig_target": float(GRADE_TARGETS[t.category] >= 50.0)}  # fmt: skip
            by_frame.setdefault(b.image, []).append(("box", meta, (b.x, b.y, b.w, b.h)))
    return sorted(by_frame.items(), key=lambda kv: str(kv[0]))


def arcade_frames(raw_dir: str | Path, split: str) -> list[tuple[Path, list]]:
    """ARCADE stenosis boxes of ``split`` (significance 1) and each image's distant
    candidates (significance 0)."""
    root = Path(raw_dir) / "arcade" / "stenosis" / split
    coco = json.loads((root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
    files = {im["id"]: root / "images" / im["file_name"] for im in coco["images"]}
    boxes: dict[int, list] = {}
    for a in coco["annotations"]:
        boxes.setdefault(a["image_id"], []).append(tuple(float(v) for v in a["bbox"]))
    frames = []
    for image_id, path in sorted(files.items()):
        bs = boxes.get(image_id, [])
        base = {"source": "arcade", "split": split, "patient": f"arcade_{split}_{image_id}",
                "track": f"arcade_{split}_{image_id}", "frame": 0, "category": "",
                "ds_target": float("nan")}  # fmt: skip
        jobs = [("box", {**base, "track": f"{base['track']}_{k}", "sig_target": 1.0}, b)
                for k, b in enumerate(bs)]  # fmt: skip
        jobs.append(("negatives", {**base, "track": f"{base['track']}_neg", "sig_target": 0.0,
                                   "key": base["track"]}, bs))  # fmt: skip
        frames.append((path, jobs))
    return frames


def cadica_free_frames(raw_dir: str | Path | None = None) -> list[tuple[Path, list]]:
    """Normal-vessel candidates on CADICA's lesion-free runs (P4k-A): every
    :data:`FREE_FRAME_STEP`-th selected frame, with the patient's split (P4b)."""
    from vstent.data import cadica
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    raw_dir = raw_dir or cadica.RAW_DIR
    split = patient_split([t.patient for t in lesion_tracks(cadica.load_lesion_boxes(raw_dir))])
    frames = []
    for r in cadica.load_selected_runs(raw_dir):
        if r.lesion or r.patient not in split:
            continue
        for f in r.selected[::FREE_FRAME_STEP]:
            key = f"{r.patient}_{r.video}_free_f{f}"
            meta = {"source": "cadica_free", "split": split[r.patient], "patient": r.patient,
                    "track": key, "frame": f, "category": "", "ds_target": float("nan"),
                    "sig_target": 0.0, "key": key}  # fmt: skip
            frames.append((r.frames_dir / f"{r.patient}_{r.video}_{f:05d}.png",
                           [("negatives", meta, [])]))  # fmt: skip
    return frames


def _clinician(frames: list[tuple[Path, list]]) -> list[tuple[Path, list]]:
    """The same frames with every box measured on the clinician-style path (version 2)."""
    return [(p, [("box_clinician" if k == "box" else k, m, a) for k, m, a in jobs])
            for p, jobs in frames]  # fmt: skip


def kemerovo_frames(raw_dir: str | Path) -> list[tuple[Path, list]]:
    """Kemerovo stenosis boxes (significant lesions; external test)."""
    from vstent.data.kemerovo import load_boxes

    by_frame: dict[Path, list] = {}
    for b in load_boxes(raw_dir):
        meta = {"source": "kemerovo", "split": "external", "patient": b.patient,
                "track": f"{b.patient}_{b.series}", "frame": 0, "category": "",
                "ds_target": float("nan"), "sig_target": 1.0}  # fmt: skip
        by_frame.setdefault(b.image, []).append(("box", meta, (b.x, b.y, b.w, b.h)))
    return sorted(by_frame.items(), key=lambda kv: str(kv[0]))


def save(samples: list[dict[str, Any]], name: str, out_dir: str | Path = OUT_DIR,
         extra: dict | None = None) -> Path:  # fmt: skip
    """``<name>.npz`` (patches, scalars) and ``<name>.json`` (one record per sample, failures
    included with their reason)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [s for s in samples if "patch" in s]
    patches = (np.stack([s["patch"] for s in ok]).astype(np.float16) if ok
               else np.zeros((0, 2, 1, 1), np.float16))  # fmt: skip
    scalars = np.stack([s["scalars"] for s in ok]) if ok else np.zeros((0, 3), np.float32)
    np.savez_compressed(out_dir / f"{name}.npz", patches=patches, scalars=scalars)
    meta = [{k: v for k, v in s.items() if k not in ("patch", "scalars")} for s in ok]
    failed = [{k: v for k, v in s.items() if k not in ("patch", "scalars")} for s in samples
              if "patch" not in s]  # fmt: skip
    (out_dir / f"{name}.json").write_text(json.dumps({"samples": meta, "failed": failed,
                                                      **(extra or {})}),
                                          encoding="utf-8")  # fmt: skip
    return out_dir / f"{name}.npz"


def load(name: str, out_dir: str | Path = OUT_DIR) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Patches (float32), scalars and per-sample records of ``<name>``."""
    out_dir = Path(out_dir)
    z = np.load(out_dir / f"{name}.npz")
    meta = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))["samples"]
    return z["patches"].astype(np.float32), z["scalars"].astype(np.float32), meta


SOURCES = ("cadica", "arcade", "kemerovo")


def build(segmenter, *, workers: int = 1, sources: tuple[str, ...] = SOURCES,
          kemerovo_dir: str | Path | None = None, arcade_dir: str | Path = "data/raw/arcade",
          git: dict | None = None, version: int = 1,
          log: Callable[[str], None] | None = None) -> dict[str, int]:  # fmt: skip
    """The samples of ``sources``; returns the number of samples per file. Kemerovo is skipped
    while it is not downloaded. ``version`` 2 (P4k-A): clinician-style paths for boxes, the
    lesion-free CADICA candidates, written to :data:`OUT_DIR_V2`."""
    log = log or (lambda _m: None)
    prep = _clinician if version == 2 else (lambda f: f)
    sets = [("cadica", prep(cadica_frames()))] if "cadica" in sources else []
    if "cadica" in sources and version == 2:
        sets.append(("cadica_free", cadica_free_frames()))
    if "arcade" in sources:
        sets += [(f"arcade_{s}", prep(arcade_frames(arcade_dir, s)))
                 for s in ("train", "val", "test")]  # fmt: skip
    if ("kemerovo" in sources and kemerovo_dir is not None
            and (Path(kemerovo_dir) / "train_labels.csv").exists()):  # fmt: skip
        sets.append(("kemerovo", prep(kemerovo_frames(kemerovo_dir))))
    out_dir = OUT_DIR_V2 if version == 2 else OUT_DIR
    counts = {}
    for name, frames in sets:
        log(f"{name}: {len(frames)} frames")

        def progress(k, n, name=name):
            if k % 500 == 0 or k == n:
                log(f"  {name}: {k} / {n}")

        samples = _run(frames, segmenter, workers, progress)
        save(samples, name, out_dir, extra={"git": git or {}, "version": version})
        counts[name] = sum("patch" in s for s in samples)
        log(f"  {name}: {counts[name]} samples ({len(samples) - counts[name]} failed)")
    return counts
