"""Repeatability of the product's measurements on real angiograms (P4f). Torch-free.

A measurement that changes when nothing about the patient changes cannot be accurate, and this
can be checked without wire FFR. On CADICA's expert lesions (:mod:`vstent.data.cadica`), each
box is measured with the product's steps:

1. the vessel path through the box, extended on every side by a margin (the analysed
   segment; :func:`~vstent.validation.stenosis_grading.path_through_box`);
2. QCA with the product's lumen (:data:`~vstent.inference.pipeline.AUTO_LUMEN`);
3. %DS at the worst point inside the box, and the segment's FFR from the reduced-order solver
   (configs/physics.yaml; the assumed 0.3 mm/px, as CADICA has no pixel size).

Two kinds of repeatability (pre-registered in docs/PLAN.md, P4f):

- **frame to frame:** the same lesion on the keyframes of its run (margin = the box size);
- **operator to operator:** on each keyframe, two simulated operators choose the margin
  independently, uniformly from :data:`OPERATOR_MARGIN` × the box size.

The coefficient of variation is the root mean square of the per-lesion (or per-frame) ratio
of the standard deviation to the mean, as in the QREP study (Westra et al., EuroIntervention
2022). Units: %DS in percentage points, FFR dimensionless.
"""

import json
import math
import os
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.imaging.centerline import smooth_and_resample, vessel_graph
from vstent.imaging.image_io import read_image_file
from vstent.imaging.qca import LEVEL, qca_from_centerline
from vstent.inference.pipeline import AUTO_LUMEN
from vstent.physics.solver import PhysicsParams, solve_ffr
from vstent.validation.stenosis_grading import ORDINAL, Track, path_through_box

REPORT = "repeatability"
OPERATOR_MARGIN = (0.5, 1.5)  # × the box size
THRESHOLD = 0.80
TARGETS = {
    "frames_ds_cv": 0.119,  # QCA inter-observer CV with operator-selected frames (Ito et al.)
    "frames_ffr_cv": 0.094,  # QFR inter-observer CV (QREP, EuroIntervention 2022)
    "operators_ffr_cv": 0.094,
    "operators_ds_sd": 7.0,  # QCA inter-observer SD of %DS differences (points)
}
SEED = 0
N_BOOT = 2000


def measure_box(prob, graph, image, box, margin: float, params: PhysicsParams) -> dict[str, Any]:
    """%DS and segment FFR of the vessel crossing ``box`` with the analysed segment extended by
    ``margin`` px; ``{"error": ...}`` if it cannot be measured."""
    path = path_through_box(graph, box, margin)
    if path is None or len(path) < 5:
        return {"error": "no segmented vessel crosses the box"}
    try:
        q = qca_from_centerline(prob, smooth_and_resample(path), image=image, lumen=AUTO_LUMEN)
    except ValueError as exc:
        return {"error": f"QCA: {exc}"}
    x, y, w, h = box
    pts = q.centerline.points
    inside = (pts[:, 0] >= y) & (pts[:, 0] <= y + h) & (pts[:, 1] >= x) & (pts[:, 1] <= x + w)
    if not inside.any():
        return {"error": "the measured path misses the box"}
    ds = 100.0 * (1.0 - q.profile.r / q.profile.reference_radius())
    k = int(np.argmax(np.where(inside, ds, -np.inf)))
    return {"ds": float(max(ds[k], 0.0)), "ffr": float(solve_ffr(q.profile, params).ffr[-1])}


def _frame_task(img, prob, items, params_dict) -> list[dict[str, Any]]:
    """Every (lesion, box) of one frame: the default margin and the two operators'."""
    params = PhysicsParams.model_validate(params_dict)
    graph = vessel_graph(prob >= LEVEL)
    out = []
    for track_key, patient, category, frame, box in items:
        size = max(box[2], box[3])
        rng = np.random.default_rng([SEED, zlib.crc32(f"{track_key}:{frame}".encode())])
        margins = {"default": size, "op_a": size * rng.uniform(*OPERATOR_MARGIN),
                   "op_b": size * rng.uniform(*OPERATOR_MARGIN)}  # fmt: skip
        row = {"track": track_key, "patient": patient, "category": category, "frame": frame}
        for name, m in margins.items():
            res = measure_box(prob, graph, img, box, m, params)
            row[f"{name}_ds"] = res.get("ds")
            row[f"{name}_ffr"] = res.get("ffr")
            row[f"{name}_error"] = res.get("error", "")
        out.append(row)
    return out


def measure(
    tracks: list[Track],
    segmenter,
    params: PhysicsParams,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """One row per expert box (lesions graded 0–98%), each frame segmented once in this
    process; the measurements run in ``workers`` processes. Order does not depend on it."""
    from concurrent.futures import ProcessPoolExecutor

    by_frame: dict[Path, list] = {}
    for t in tracks:
        if t.category not in ORDINAL:
            continue
        for b in t.boxes:
            by_frame.setdefault(b.image, []).append((t.key, t.patient, t.category, b.frame,
                                                     (b.x, b.y, b.w, b.h)))  # fmt: skip
    frames = sorted(by_frame.items(), key=lambda kv: str(kv[0]))
    params_dict = params.model_dump()
    results: list = [None] * len(frames)
    if workers <= 1:
        for k, (image, items) in enumerate(frames):
            img = read_image_file(image).image
            results[k] = _frame_task(img, segmenter.predict(img), items, params_dict)
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
                pending[pool.submit(_frame_task, img, segmenter.predict(img), items,
                                    params_dict)] = k  # fmt: skip
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


def cv_rms(groups: list[list[float]]) -> float | None:
    """Root mean square over groups (≥ 2 values, positive mean) of SD / mean (sample SD)."""
    terms = [(np.std(g, ddof=1) / np.mean(g)) ** 2 for g in groups
             if len(g) >= 2 and np.mean(g) > 0]  # fmt: skip
    return float(math.sqrt(np.mean(terms))) if terms else None


def _boot(groups_by_patient: dict[str, list[list[float]]], stat, n_boot=N_BOOT) -> list | None:
    """95% patient-level bootstrap interval of ``stat`` (a function of a list of groups)."""
    patients = sorted(groups_by_patient)
    if len(patients) < 2:
        return None
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(patients), len(patients))
        groups = [g for i in pick for g in groups_by_patient[patients[i]]]
        v = stat(groups)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def _pairs_agree(groups: list[list[float]]) -> tuple[int, int]:
    agree = total = 0
    for g in groups:
        for a, b in combinations(g, 2):
            total += 1
            agree += (a <= THRESHOLD) == (b <= THRESHOLD)
    return agree, total


def _sd_diff(groups: list[list[float]]) -> float | None:
    d = [g[0] - g[1] for g in groups if len(g) == 2]
    return float(np.std(d, ddof=1)) if len(d) >= 2 else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frame-to-frame and operator-to-operator repeatability with bootstrap intervals."""
    frames: dict[str, list[dict]] = {}
    for r in rows:
        frames.setdefault(r["track"], []).append(r)

    def frame_groups(field, keep=lambda _r: True):
        by_pat: dict[str, list[list[float]]] = {}
        for ms in frames.values():
            g = [m[f"default_{field}"] for m in ms if m["default_ffr"] is not None and keep(m)]
            if len(g) >= 2:
                by_pat.setdefault(ms[0]["patient"], []).append(g)
        return by_pat

    def op_groups(field, keep=lambda _r: True):
        by_pat: dict[str, list[list[float]]] = {}
        for r in rows:
            a, b = r[f"op_a_{field}"], r[f"op_b_{field}"]
            if a is not None and b is not None and keep(r):
                by_pat.setdefault(r["patient"], []).append([a, b])
        return by_pat

    def flat(by_pat):
        return [g for gs in by_pat.values() for g in gs]

    out: dict[str, Any] = {
        "n_boxes": len(rows), "n_lesions": len(frames),
        "n_patients": len({r["patient"] for r in rows}),
        "measured_default": sum(r["default_ffr"] is not None for r in rows),
        "measured_both_operators": sum(r["op_a_ffr"] is not None and r["op_b_ffr"] is not None
                                       for r in rows),
    }  # fmt: skip
    for kind, groups in (("frames", frame_groups), ("operators", op_groups)):
        ds, ffr = groups("ds"), groups("ffr")
        agree, total = _pairs_agree(flat(ffr))
        out[kind] = {
            "n_groups": len(flat(ffr)), "n_patients": len(ffr),
            "ds_cv": cv_rms(flat(ds)), "ds_cv_ci": _boot(ds, cv_rms),
            "ffr_cv": cv_rms(flat(ffr)), "ffr_cv_ci": _boot(ffr, cv_rms),
            "ds_sd_within": _within_sd(flat(ds)), "ffr_sd_within": _within_sd(flat(ffr)),
            "ds_sd_diff": _sd_diff(flat(ds)) if kind == "operators" else None,
            "ds_sd_diff_ci": _boot(ds, _sd_diff) if kind == "operators" else None,
            "agree_080": agree / total if total else None, "pairs": total,
            "by_grade": {c: {"ffr_cv": cv_rms(flat(groups("ffr", lambda r, c=c: r["category"]
                                                           == c))),
                             "ds_cv": cv_rms(flat(groups("ds", lambda r, c=c: r["category"]
                                                         == c))),
                             "n": len(flat(groups("ffr", lambda r, c=c: r["category"] == c)))}
                         for c in ORDINAL},
        }  # fmt: skip
    f, o = out["frames"], out["operators"]
    out["targets"] = [
        _target("Frame to frame: %DS CV", f["ds_cv"], TARGETS["frames_ds_cv"]),
        _target("Frame to frame: FFR CV", f["ffr_cv"], TARGETS["frames_ffr_cv"]),
        _target("Operator to operator: FFR CV", o["ffr_cv"], TARGETS["operators_ffr_cv"]),
        _target("Operator to operator: SD of %DS differences (points)", o["ds_sd_diff"],
                TARGETS["operators_ds_sd"], fmt="{:.1f}"),
    ]  # fmt: skip
    return out


def _within_sd(groups: list[list[float]]) -> float | None:
    """Pooled within-group standard deviation."""
    num = sum((len(g) - 1) * np.var(g, ddof=1) for g in groups if len(g) >= 2)
    den = sum(len(g) - 1 for g in groups if len(g) >= 2)
    return float(math.sqrt(num / den)) if den else None


def _target(metric: str, value, target: float, fmt: str = "{:.1%}") -> dict[str, Any]:
    return {"metric": metric, "value": "–" if value is None else fmt.format(value),
            "target": "≤ " + fmt.format(target),
            "passed": bool(value is not None and value <= target)}  # fmt: skip


def _num(x, fmt: str) -> str:
    return "–" if x is None else fmt.format(x)


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _ci(ci, fmt="{:.1%}") -> str:
    return "–" if not ci else f"{fmt.format(ci[0])}–{fmt.format(ci[1])}"


def write_report(s: dict[str, Any], git: dict, seg_model: str,
                 report_dir: str | Path = "reports") -> None:  # fmt: skip
    """``reports/repeatability.{md,json}``."""
    report_dir = Path(report_dir)
    f, o = s["frames"], s["operators"]
    prov = {**git, "vstent_version": __version__, "seg_model": seg_model,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip

    def grade_rows(d):
        return [f"| {c} | {g['n']} | {_pct(g['ds_cv'])} | {_pct(g['ffr_cv'])} |"
                for c, g in d["by_grade"].items()]  # fmt: skip

    lines = [
        "# Repeatability on real angiograms (P4f)",
        "",
        f"> **{DISCLAIMER}** Repeatability, not accuracy: a measurement that varies when "
        "nothing about the patient changes cannot be accurate, but a repeatable one can still "
        "be wrong.",
        "",
        "Generated by `vstent repeatability`. Do not edit by hand.",
        "",
        f"**Data:** CADICA (CC BY 4.0): {s['n_lesions']} expert lesions graded 0–98% in "
        f"{s['n_patients']} patients, {s['n_boxes']} lesion boxes on keyframes; all patients "
        "(a reporting use; nothing was chosen on these results).",
        "",
        "**Measurement (the product's steps):** segmentation; the vessel path through the box, "
        "extended on every side by a margin; QCA with the product's lumen; %DS at the worst "
        "point inside the box; the segment's FFR from the reduced-order solver at the assumed "
        "0.3 mm/px (CADICA has no pixel size). This FFR is the lesion segment's, not a whole "
        "vessel's. Coefficient of variation (CV): root mean square of SD / mean per lesion (or "
        "per frame), as in QREP. Intervals: 95% patient-level bootstrap.",
        "",
        "## Pre-registered targets",
        "",
        "Each target is an accepted method's own variability: QFR's inter-observer CV of 9.4% "
        "(QREP, EuroIntervention 2022) and QCA's inter-observer variability (%DS CV 11.9% "
        "with operator-selected frames; SD of differences about 5–7 points).",
        "",
        "| Metric | Value | Target | Met |",
        "|---|---:|---:|---|",
        *(f"| {t['metric']} | {t['value']} | {t['target']} | "
          f"{'yes' if t['passed'] else '**no**'} |" for t in s["targets"]),
        "",
        "## Frame to frame (the same lesion on different keyframes of its run)",
        "",
        f"- {f['n_groups']} lesions with ≥ 2 measured keyframes ({f['n_patients']} patients).",
        f"- %DS: CV {_pct(f['ds_cv'])} (95% {_ci(f['ds_cv_ci'])}); within-lesion SD "
        f"{_num(f['ds_sd_within'], '{:.1f}')} points.",
        f"- FFR: CV {_pct(f['ffr_cv'])} (95% {_ci(f['ffr_cv_ci'])}); within-lesion SD "
        f"{_num(f['ffr_sd_within'], '{:.3f}')}.",
        f"- Pairs of keyframes on the same side of {THRESHOLD:.2f}: {_pct(f['agree_080'])} "
        f"of {f['pairs']}.",
        "",
        "| Expert grade | Lesions | %DS CV | FFR CV |",
        "|---|---:|---:|---:|",
        *grade_rows(f),
        "",
        "## Operator to operator (two choices of the analysed segment on the same frame)",
        "",
        f"Each operator extends the segment by {OPERATOR_MARGIN[0]:g}–{OPERATOR_MARGIN[1]:g} × "
        "the box size, drawn independently. Real operators also choose frames and click "
        "points, so this is a lower bound of operator variability.",
        "",
        f"- {o['n_groups']} keyframes measured by both ({o['n_patients']} patients).",
        f"- %DS: SD of the differences {_num(o['ds_sd_diff'], '{:.1f}')} points (95% "
        f"{_ci(o['ds_sd_diff_ci'], '{:.1f}')}); CV {_pct(o['ds_cv'])}.",
        f"- FFR: CV {_pct(o['ffr_cv'])} (95% {_ci(o['ffr_cv_ci'])}); within-frame SD "
        f"{_num(o['ffr_sd_within'], '{:.3f}')}.",
        f"- Same side of {THRESHOLD:.2f}: {_pct(o['agree_080'])} of {o['pairs']} frames.",
        "",
        "| Expert grade | Frames | %DS CV | FFR CV |",
        "|---|---:|---:|---:|",
        *grade_rows(o),
        "",
        f"Measured with the default margin: {s['measured_default']} of {s['n_boxes']} boxes; "
        f"by both operators: {s['measured_both_operators']}.",
        "",
        f"Segmentation model `{seg_model}`. Commit `{prov['git_commit']}` (dirty: "
        f"{prov['git_dirty']}), created {prov['created_utc']}.",
        "",
    ]  # fmt: skip
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{REPORT}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (report_dir / f"{REPORT}.json").write_text(
        json.dumps({"summary": s, "provenance": prov}, indent=1, default=float),
        encoding="utf-8", newline="\n")  # fmt: skip
