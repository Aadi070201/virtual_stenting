"""P4k-C: side-branch outflow (Murray's law, as in μQFR) against none, checked on CADICA without
invasive FFR, against two literature anchors (docs/PLAN.md, P4k-C). Torch-free.

1. **Lesion-free vessels** (CADICA lesion-free runs; the product's steps: the run's analysis
   frame, the automatic main-trunk path, the median of up to 5 heartbeats): the share with
   FFR ≤ 0.80. Anchor: De Bruyne 2001, FFR 0.89 ± 0.08 in non-stenotic arteries of patients
   with disease elsewhere, about 13% ≤ 0.80.
2. **Graded lesions** (CADICA expert boxes; the path from the tree's proximal end through the
   box, ``guided.box_path``; the median over a lesion's boxes): the share with FFR ≤ 0.80 per
   grade. Anchor: FAME, visual 50–70% / 70–90% / 90–99% → 35% / 80% / 96%.

Every arm is solved on the same measured profiles:
- ``none``: the product;
- ``murray``: side-branch outflow (P4k-C);
- ``wire``: the product's physics on the vessel cut at the last point whose reference diameter
  is ≥ 2.0 mm, where a pressure wire would sit (P4k-C2; ``qca.at_wire``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vstent.physics.solver import PhysicsParams, solve_ffr_batch

OUTFLOWS = ("none", "murray", "wire")  # the arms (``wire``: no outflow, FFR at the wire)
ARM_LABELS = {"none": "Product (now)", "murray": "Murray outflow", "wire": "FFR at the wire"}
CUTOFF = 0.80
LESION_FREE_ANCHOR = 0.13  # De Bruyne 2001: P(FFR ≤ 0.80) for 0.89 ± 0.08 (normal approx.)
FAME_SHARES = {"p50_70": 0.35, "p70_90": 0.80, "p90_98": 0.96}  # FFR ≤ 0.80 by visual grade
MAX_FAME_WORSENING = 0.10
FAME_LABELS = {"p50_70": "50–70%", "p70_90": "70–90%", "p90_98": "90–98%"}
OUT_DIR = Path("data/derived/p4k")


def distal_ffr(profile, params: PhysicsParams) -> dict[str, float]:
    """FFR (-) of a measured profile for every arm (reference from the QCA)."""
    from vstent.imaging.qca import wire_index

    out = {}
    r_ref = profile.reference_radius()
    for o in ("none", "murray"):
        p = params.model_copy(update={"outflow": o})
        out[o] = float(solve_ffr_batch(profile.r, profile.ds, r_ref, p).ffr_distal[0])
    # the wire arm solves the vessel cut at the wire position (``qca.at_wire``, D68)
    k = max(wire_index(r_ref), 2)
    cut = slice(0, k + 1)
    out["wire"] = float(solve_ffr_batch(profile.r[cut], profile.ds, r_ref[cut], params)
                        .ffr_distal[0]) if k < len(r_ref) - 1 else out["none"]  # fmt: skip
    return out


def _split_of(raw_dir) -> tuple[dict[str, str], list]:
    from vstent.data import cadica
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    tracks = lesion_tracks(cadica.load_lesion_boxes(raw_dir))
    return patient_split([t.patient for t in tracks]), tracks


_READERS: dict[str, Any] = {}


def _reader(reader_dir: str | None):
    """The learned lesion reader of ``reader_dir`` (P4k-C3), loaded once per process."""
    if reader_dir is None:
        return None
    if reader_dir not in _READERS:
        from vstent.inference.lesion_runtime import LesionRunner

        _READERS[reader_dir] = LesionRunner.from_dir(reader_dir)
    return _READERS[reader_dir]


def lesion_free_row(run_meta, split: str, segmenter, params: PhysicsParams,
                    reader=None) -> dict[str, Any]:  # fmt: skip
    """One lesion-free run through the product's steps; every arm on each heartbeat. With
    ``reader``, the learned lesion reading is applied to every measurement (P4k-C3)."""
    from vstent.app.guided import auto_path
    from vstent.imaging.image_io import read_run
    from vstent.imaging.run_analysis import analyse_frames
    from vstent.inference.consensus import run_consensus
    from vstent.inference.pipeline import analyse_vessel

    row: dict[str, Any] = {"patient": run_meta.patient, "video": run_meta.video, "split": split}
    try:
        run = read_run(run_meta.frames_dir)
        rf = analyse_frames(run, segmenter.predict)
        frame = rf.analysis_frame
        image = run.frames[frame]
        prob = segmenter.predict(image)
        start, end = auto_path(prob, run.pixel_spacing_mm)

        def one(p, im, a, b):
            return analyse_vessel(p, a, b, params=params, mm_per_px=run.pixel_spacing_mm,
                                  image=im, lesion_reader=reader)  # fmt: skip

        cons = run_consensus(rf, frame, start, end, segmenter.predict, one,
                             reference_analysis=one(prob, image, start, end))  # fmt: skip
        values = [distal_ffr(f.analysis.qca.profile, params) for f in cons.used]
        row.update(frame=int(frame), heartbeats=len(values),
                   length_mm=float(np.median([f.values.length_mm for f in cons.used])))  # fmt: skip
        row.update({f"ffr_{o}": float(np.median([v[o] for v in values])) for o in OUTFLOWS})
    except Exception as exc:  # noqa: BLE001 — a run that cannot be measured is reported
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _box_task(img, prob, items, params_dict, reader_dir=None) -> list[dict[str, Any]]:
    from vstent.app.guided import box_path
    from vstent.inference.pipeline import analyse_vessel

    params = PhysicsParams(**params_dict)
    reader = _reader(reader_dir)
    rows = []
    for track, patient, category, split, box in items:
        row: dict[str, Any] = {"track": track, "patient": patient, "category": category,
                               "split": split}  # fmt: skip
        try:
            path = box_path(prob, box)
            if path is None:
                raise ValueError("no vessel in the box")
            analysis = analyse_vessel(prob, path[0], path[1], params=params, image=img,
                                      lesion_reader=reader)  # fmt: skip
            row.update({f"ffr_{o}": v for o, v in distal_ffr(analysis.qca.profile, params).items()})
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def graded_rows(stage: str, segmenter, params: PhysicsParams, raw_dir=None, *,
                workers: int = 1, progress: Callable[[int, int], None] | None = None,
                reader_dir: str | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """One row per expert box (graded 50–98%) of the ``stage`` patients; segmentation here,
    measurements in ``workers`` processes."""
    from concurrent.futures import ProcessPoolExecutor

    from vstent.data import cadica
    from vstent.imaging.image_io import read_image_file

    split, tracks = _split_of(raw_dir or cadica.RAW_DIR)
    by_frame: dict[Path, list] = {}
    for t in tracks:
        if t.category not in FAME_SHARES or split[t.patient] != stage:
            continue
        for b in t.boxes:
            by_frame.setdefault(b.image, []).append((t.key, t.patient, t.category, stage,
                                                     (b.x, b.y, b.w, b.h)))  # fmt: skip
    frames = sorted(by_frame.items(), key=lambda kv: str(kv[0]))
    pd = params.model_dump()
    results: list = [None] * len(frames)
    if workers <= 1:
        for k, (image, items) in enumerate(frames):
            img = read_image_file(image).image
            results[k] = _box_task(img, segmenter.predict(img), items, pd, reader_dir)
            if progress:
                progress(k + 1, len(frames))
    else:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {}
            for k, (image, items) in enumerate(frames):
                img = read_image_file(image).image
                fut = pool.submit(_box_task, img, segmenter.predict(img), items, pd, reader_dir)
                pending[fut] = k
                while len(pending) >= 2 * workers:
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
    return [row for out in results for row in out or []]


def lesion_free_rows(stage: str, segmenter, params: PhysicsParams, raw_dir=None,
                     progress: Callable[[int, int], None] | None = None,
                     reader_dir: str | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """One row per lesion-free selected run of the ``stage`` patients (patients without a
    graded lesion have no split and are left out)."""
    from vstent.data import cadica

    raw_dir = raw_dir or cadica.RAW_DIR
    split, _ = _split_of(raw_dir)
    runs = [r for r in cadica.load_selected_runs(raw_dir)
            if not r.lesion and split.get(r.patient) == stage]  # fmt: skip
    rows = []
    for k, r in enumerate(runs):
        rows.append(lesion_free_row(r, stage, segmenter, params, _reader(reader_dir)))
        if progress:
            progress(k + 1, len(runs))
    return rows


def summarise(free: list[dict[str, Any]], graded: list[dict[str, Any]]) -> dict[str, Any]:
    """Shares with FFR ≤ 0.80 per outflow, the anchors' distances and the pre-registered rule."""
    ok_free = [r for r in free if "error" not in r]
    lesions: dict[str, list[dict[str, Any]]] = {}
    for r in graded:
        if "error" not in r:
            lesions.setdefault(r["track"], []).append(r)
    per_lesion = [{"category": rs[0]["category"],
                   **{o: float(np.median([x[f"ffr_{o}"] for x in rs])) for o in OUTFLOWS}}
                  for rs in lesions.values()]  # fmt: skip
    out: dict[str, Any] = {"n_free": len(ok_free), "n_free_failed": len(free) - len(ok_free),
                           "n_boxes": len(graded), "n_lesions": len(per_lesion),
                           "outflow": {}}  # fmt: skip
    for o in OUTFLOWS:
        ffr = np.array([r[f"ffr_{o}"] for r in ok_free])
        share = float(np.mean(ffr <= CUTOFF)) if ffr.size else None
        grades = {}
        for c in FAME_SHARES:
            vals = np.array([x[o] for x in per_lesion if x["category"] == c])
            share_c = float(np.mean(vals <= CUTOFF)) if vals.size else None
            grades[c] = {"n": int(vals.size), "share": share_c}
        diffs = [abs(g["share"] - FAME_SHARES[c]) for c, g in grades.items()
                 if g["share"] is not None]  # fmt: skip
        out["outflow"][o] = {
            "free_share": share,
            "free_median": float(np.median(ffr)) if ffr.size else None,
            "free_distance": None if share is None else abs(share - LESION_FREE_ANCHOR),
            "grades": grades,
            "fame_mad": float(np.mean(diffs)) if len(diffs) == len(FAME_SHARES) else None,
        }
    out["adopt"], out["reason"] = rule(out["outflow"]["none"], out["outflow"]["murray"])
    out["wire_adopt"], out["wire_reason"] = rule(out["outflow"]["none"], out["outflow"]["wire"])
    return out


def rule(n: dict[str, Any], m: dict[str, Any]) -> tuple[bool, str]:
    """The pre-registered rule for an arm ``m`` against the product ``n``: the lesion-free
    share closer to 13%, and the FAME distance not more than 10 points worse."""
    stats = (n["free_distance"], m["free_distance"], n["fame_mad"], m["fame_mad"])
    if any(v is None for v in stats):
        return False, "a statistic could not be computed"
    closer = m["free_distance"] < n["free_distance"]
    fame_ok = m["fame_mad"] <= n["fame_mad"] + MAX_FAME_WORSENING
    return bool(closer and fame_ok), (
        f"lesion-free share closer to 13%: {closer}; FAME distance grows by "
        f"{100 * (m['fame_mad'] - n['fame_mad']):+.1f} points (limit +10): "
        f"{'met' if fame_ok else 'not met'}")  # fmt: skip


def save(stage: str, free, graded, summary, out_dir: str | Path = OUT_DIR,
         name: str = "physics") -> Path:  # fmt: skip
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{stage}.json"
    path.write_text(json.dumps({"stage": stage, "summary": summary, "lesion_free": free,
                                "graded": graded}, indent=1), encoding="utf-8")  # fmt: skip
    return path


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def section(stage: str, s: dict[str, Any], part: str = "c") -> list[str]:
    """The report's lines for one stage: every arm in ``s``; the rule of ``part`` (``c``:
    Murray outflow; ``c2``: the FFR at the wire)."""
    title = "Development (CADICA development patients)" if stage == "dev" else (
        "Test (CADICA test patients)")  # fmt: skip
    arms = [a for a in OUTFLOWS if a in s["outflow"]]
    lines = [f"## {title}", "", "| | " + " | ".join(ARM_LABELS[a] for a in arms) + " |",
             "|---|" + "---:|" * len(arms)]  # fmt: skip

    def row(label, f):
        lines.append(f"| {label} | " + " | ".join(f(s["outflow"][a]) for a in arms) + " |")

    def med(v):
        return "–" if v is None else f"{v:.2f}"

    def mad(v):
        return "–" if v is None else f"{100 * v:.0f} points"

    n = s["outflow"]["none"]
    row("Lesion-free vessels: FFR ≤ 0.80 (anchor ≈ 13%)", lambda x: _pct(x["free_share"]))
    row("Lesion-free vessels: median FFR (anchor 0.89)", lambda x: med(x["free_median"]))
    for c, label in FAME_LABELS.items():
        title = (
            f"Lesions {label}: FFR ≤ 0.80 (FAME {_pct(FAME_SHARES[c])}; n = {n['grades'][c]['n']})"
        )
        row(title, lambda x, c=c: _pct(x["grades"][c]["share"]))
    row("Mean distance from FAME's shares", lambda x: mad(x["fame_mad"]))
    lines += ["", f"{s['n_free']} lesion-free runs measured ({s['n_free_failed']} could not be); "
              f"{s['n_boxes']} boxes of {s['n_lesions']} graded lesions.", ""]  # fmt: skip
    if stage == "dev":
        key, name = ("adopt", "`murray`") if part == "c" else ("wire_adopt", "the wire position")
        reason = s["reason"] if part == "c" else s["wire_reason"]
        decision = f"adopt {name}" if s[key] else "not adopted"
        lines += [f"**Rule** (docs/PLAN.md, P4k-{part.upper()}): {reason}.", "",
                  f"**Decision: {decision}.**", ""]  # fmt: skip
    return lines


C3_MAX_FREE = 0.25  # P4k-C3: the lesion-free share must also be ≤ 25%


def c3_rules(product: dict[str, Any], s: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """P4k-C3: every arm (with the reader) against the product's development numbers: C's rule,
    the lesion-free share ≤ 25%, and the combined distance used to choose among qualifiers."""
    out = {}
    for arm, x in s["outflow"].items():
        ok, reason = rule(product, x)
        share = x["free_share"]
        small = share is not None and share <= C3_MAX_FREE
        score = (None if share is None or x["fame_mad"] is None
                 else abs(share - LESION_FREE_ANCHOR) + x["fame_mad"])  # fmt: skip
        why = f"{reason}; lesion-free share {_pct(share)} (≤ 25%): {'met' if small else 'not met'}"
        out[arm] = {"qualifies": bool(ok and small), "reason": why, "score": score}
    return out
