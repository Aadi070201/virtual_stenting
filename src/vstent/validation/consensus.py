"""Repeatability of a multi-frame consensus on real angiograms (P4g). Torch-free.

P4f (D57) found that one automatic frame does not measure a lesion stably. The consensus of K
frames is the median %DS and the median FFR over K frames of the same lesion. Its
repeatability is measured on P4f's per-keyframe measurements of CADICA's expert lesions
(:mod:`vstent.validation.repeatability`, default margin) by split halves, as pre-registered in
docs/PLAN.md (P4g):

- each lesion with ≥ 2K measured keyframes gives two disjoint sets of K frames: alternate
  keyframes in time order (the 1st, 3rd, … and the 2nd, 4th, …), the first K of each;
- the CV between the two sets' medians (root mean square over lesions, as in P4f);
- K is chosen on the development patients (P4b's split) and tested once on the test patients;
- a consensus is flagged as unstable when its K frames' FFR values lie on both sides of 0.80.

Units: %DS in percentage points, FFR dimensionless.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.inference.consensus import straddles as flagged
from vstent.validation import repeatability as rp

REPORT = "consensus"
KS = (1, 2, 3, 4, 5)
THRESHOLD = rp.THRESHOLD
TARGETS = {"ffr_cv": rp.TARGETS["frames_ffr_cv"], "ds_cv": rp.TARGETS["frames_ds_cv"]}
COMMON_MIN_FRAMES = 2 * max(KS)  # the supplementary comparison on one set of lesions


def lesion_frames(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Measured keyframes (default margin) of each lesion, in time order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["default_ffr"] is not None and r["default_ds"] is not None:
            out.setdefault(r["track"], []).append(r)
    return {k: sorted(v, key=lambda r: r["frame"]) for k, v in out.items()}


def halves(frames: list[Any], k: int) -> tuple[list[Any], list[Any]] | None:
    """Alternate items in time order, the first ``k`` of each; ``None`` if fewer than 2k."""
    if len(frames) < 2 * k:
        return None
    return frames[0::2][:k], frames[1::2][:k]


def split_half(
    lesions: dict[str, list[dict[str, Any]]], k: int, split: str | None = None,
    min_frames: int = 0,
) -> dict[str, Any]:  # fmt: skip
    """Split-half repeatability of the K-frame median over the lesions of ``split`` (``dev``,
    ``test`` or all) with ≥ max(2k, ``min_frames``) measured keyframes."""
    ds: dict[str, list[list[float]]] = {}
    ffr: dict[str, list[list[float]]] = {}
    agree = n = n_flag_sets = n_sets = 0
    disagree_flagged = n_disagree = agree_clear = n_agree = 0
    for ms in lesions.values():
        if (split and ms[0].get("split") != split) or len(ms) < min_frames:
            continue
        h = halves(ms, k)
        if h is None:
            continue
        a, b = h
        fa, fb = ([m["default_ffr"] for m in s] for s in (a, b))
        patient = ms[0]["patient"]
        ds.setdefault(patient, []).append([float(np.median([m["default_ds"] for m in s]))
                                           for s in (a, b)])  # fmt: skip
        ffr.setdefault(patient, []).append([float(np.median(fa)), float(np.median(fb))])
        ma, mb = ffr[patient][-1]
        same = (ma <= THRESHOLD) == (mb <= THRESHOLD)
        n += 1
        agree += same
        flags = flagged(fa), flagged(fb)
        n_sets += 2
        n_flag_sets += sum(flags)
        if same:
            n_agree += 1
            agree_clear += not any(flags)
        else:
            n_disagree += 1
            disagree_flagged += any(flags)

    def flat(by_pat):
        return [g for gs in by_pat.values() for g in gs]

    return {
        "k": k, "n_lesions": n, "n_patients": len(ffr),
        "ds_cv": rp.cv_rms(flat(ds)), "ds_cv_ci": rp._boot(ds, rp.cv_rms),
        "ffr_cv": rp.cv_rms(flat(ffr)), "ffr_cv_ci": rp._boot(ffr, rp.cv_rms),
        "ffr_sd_within": rp._within_sd(flat(ffr)),
        "agree_080": agree / n if n else None,
        "flagged_share": n_flag_sets / n_sets if n_sets else None,
        "n_disagree": n_disagree, "disagree_flagged": disagree_flagged / n_disagree
        if n_disagree else None,
        "n_agree": n_agree, "agree_unflagged": agree_clear / n_agree if n_agree else None,
    }  # fmt: skip


def meets(res: dict[str, Any]) -> bool:
    """Both pre-registered targets met."""
    ffr, ds = res["ffr_cv"], res["ds_cv"]
    if ffr is None or ds is None:
        return False
    return ffr <= TARGETS["ffr_cv"] and ds <= TARGETS["ds_cv"]


def choose_k(dev: dict[int, dict[str, Any]]) -> tuple[int, str]:
    """The pre-registered rule on the development results: the smallest K meeting both
    targets; if none does, the K with the lowest FFR CV."""
    ok = [k for k in sorted(dev) if meets(dev[k])]
    if ok:
        return ok[0], "the smallest K meeting both targets on the development patients"
    scored = [k for k in sorted(dev) if dev[k]["ffr_cv"] is not None]
    if not scored:
        raise ValueError("no development lesion has enough measured keyframes")
    best = min(scored, key=lambda k: dev[k]["ffr_cv"])
    return best, "no K met both targets on the development patients: the lowest FFR CV"


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Development results for every K, the chosen K and its one test result."""
    lesions = lesion_frames(rows)
    dev = {k: split_half(lesions, k, "dev") for k in KS}
    k, reason = choose_k(dev)
    test = split_half(lesions, k, "test")
    common = {k_: split_half(lesions, k_, "dev", COMMON_MIN_FRAMES) for k_ in KS}
    return {
        "n_boxes": len(rows), "n_lesions_measured": len(lesions),
        "n_patients": {s: len({r["patient"] for r in rows if r.get("split") == s})
                       for s in ("dev", "test")},
        "dev": dev, "chosen_k": k, "reason": reason, "test": test, "common_dev": common,
        "targets": [
            rp._target(f"Test, K = {k}: %DS CV", test["ds_cv"], TARGETS["ds_cv"]),
            rp._target(f"Test, K = {k}: FFR CV", test["ffr_cv"], TARGETS["ffr_cv"]),
        ],
    }  # fmt: skip


def _row(r: dict[str, Any]) -> str:
    return (f"| {r['k']} | {r['n_lesions']} | {r['n_patients']} | {rp._pct(r['ds_cv'])} "
            f"({rp._ci(r['ds_cv_ci'])}) | {rp._pct(r['ffr_cv'])} ({rp._ci(r['ffr_cv_ci'])}) | "
            f"{rp._pct(r['agree_080'])} |")  # fmt: skip


def write_report(s: dict[str, Any], git: dict, seg_model: str, rows_git: dict,
                 report_dir: str | Path = "reports") -> None:  # fmt: skip
    """``reports/consensus.{md,json}``. ``rows_git``: the commit the measurements came from."""
    report_dir = Path(report_dir)
    t, k = s["test"], s["chosen_k"]
    prov = {**git, "vstent_version": __version__, "seg_model": seg_model,
            "measurements_commit": rows_git.get("git_commit"),
            "measurements_dirty": rows_git.get("git_dirty"),
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    head = ["| K | Lesions | Patients | %DS CV (95%) | FFR CV (95%) | Medians agree at 0.80 |",
            "|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    lines = [
        "# Multi-frame consensus on real angiograms (P4g)",
        "",
        f"> **{DISCLAIMER}** Repeatability, not accuracy: a consensus that repeats can still "
        "be wrong.",
        "",
        "Generated by `vstent consensus-report`. Do not edit by hand.",
        "",
        "**Question:** P4f found that one automatic frame does not measure a lesion stably "
        "(D57). Does the median over K frames of the same lesion repeat well enough?",
        "",
        f"**Data:** P4f's measurements of CADICA's expert lesions (CC BY 4.0; default segment "
        f"margin): {s['n_boxes']} lesion boxes, {s['n_lesions_measured']} lesions with ≥ 1 "
        f"measured keyframe. Patients split as in P4b: {s['n_patients']['dev']} development, "
        f"{s['n_patients']['test']} test.",
        "",
        *([f"One frame, from the same measurements (P4f's frame-to-frame result, all "
           f"patients): %DS CV {rp._pct(s['single_frame']['ds_cv'])}, FFR CV "
           f"{rp._pct(s['single_frame']['ffr_cv'])} ({s['single_frame']['n_groups']} "
           "lesions).", ""] if s.get("single_frame") else []),
        "**Method (pre-registered in docs/PLAN.md, P4g, before any consensus was computed):** "
        "for each lesion with ≥ 2K measured keyframes, two disjoint sets of K frames "
        "(alternate keyframes in time order, the first K of each); the median %DS and the "
        "median segment FFR of each set; the CV between the two medians (root mean square "
        "over lesions, as in P4f). K is chosen on the development patients: the smallest K in "
        f"1–{max(KS)} meeting both targets (FFR CV ≤ {TARGETS['ffr_cv']:.1%}, QFR's "
        f"inter-observer CV in QREP; %DS CV ≤ {TARGETS['ds_cv']:.1%}, QCA's), otherwise the "
        "lowest FFR CV. The test patients are scored once, with that K. Intervals: 95% "
        "patient-level bootstrap.",
        "",
        "## Test result",
        "",
        "| Metric | Value | Target | Met |",
        "|---|---:|---:|---|",
        *(f"| {x['metric']} | {x['value']} | {x['target']} | "
          f"{'yes' if x['passed'] else '**no**'} |" for x in s["targets"]),
        "",
        f"- K = {k}: {s['reason']}.",
        f"- {t['n_lesions']} test lesions with ≥ {2 * k} measured keyframes "
        f"({t['n_patients']} patients). %DS CV {rp._pct(t['ds_cv'])} (95% "
        f"{rp._ci(t['ds_cv_ci'])}); FFR CV {rp._pct(t['ffr_cv'])} (95% "
        f"{rp._ci(t['ffr_cv_ci'])}); within-lesion FFR SD "
        f"{rp._num(t['ffr_sd_within'], '{:.3f}')}.",
        f"- The two medians on the same side of {THRESHOLD:.2f}: {rp._pct(t['agree_080'])}.",
        "",
        "### Instability flag (test patients)",
        "",
        f"A consensus is flagged when its {k} frames' FFR values lie on both sides of "
        f"{THRESHOLD:.2f}" + (" (one frame cannot be flagged)." if k == 1 else "."),
        "",
        f"- Consensus sets flagged: {rp._pct(t['flagged_share'])}.",
        f"- Lesions whose two medians fall on opposite sides of {THRESHOLD:.2f}: "
        f"{t['n_disagree']}; flagged in either set: {rp._pct(t['disagree_flagged'])}.",
        f"- Lesions whose medians agree: {t['n_agree']}; flagged in neither set: "
        f"{rp._pct(t['agree_unflagged'])}.",
        "",
        "## Development patients (where K was chosen)",
        "",
        *head,
        *(_row(r) for r in s["dev"].values()),
        "",
        "Each K uses the lesions with ≥ 2K measured keyframes, so the set of lesions shrinks as "
        "K grows. Supplementary, not pre-registered and not used for the choice: the same on "
        f"the development lesions with ≥ {COMMON_MIN_FRAMES} measured keyframes (one set of "
        "lesions for every K).",
        "",
        *head,
        *(_row(r) for r in s["common_dev"].values()),
        "",
        "## Limits",
        "",
        "- CADICA's keyframes span heart phases; the product takes its consensus over "
        "end-diastolic frames (P7), which differ less from one another.",
        "- Frames share the segmentation model's habits, so their errors are partly "
        "correlated: the median removes frame-to-frame noise, not a bias that every frame has "
        "(D44: severe lesions read too mild).",
        "- The FFR is the lesion segment's at the assumed 0.3 mm/px (CADICA has no pixel "
        "size), as in P4f. Nothing here is compared with wire FFR.",
        "",
        f"Segmentation model `{seg_model}`. Measurements from commit "
        f"`{prov['measurements_commit']}` (dirty: {prov['measurements_dirty']}); report from "
        f"commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), created "
        f"{prov['created_utc']}.",
        "",
    ]  # fmt: skip
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{REPORT}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (report_dir / f"{REPORT}.json").write_text(
        json.dumps({"summary": s, "provenance": prov}, indent=1, default=float),
        encoding="utf-8", newline="\n")  # fmt: skip
