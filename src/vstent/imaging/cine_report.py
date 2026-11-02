"""Beating-heart synthetic runs (P7) → ``reports/cine_synthetic.md``.

Every case (:mod:`vstent.imaging.synthetic_cine`) has two runs from two views. Each run is
written as a DICOM, read back with the product's loader and analysed with the product's steps
(:mod:`vstent.imaging.run_analysis`):

1. every frame is segmented with the exported segmenter;
2. ED frames are taken from the R-wave pointer, else the ECG, else the images;
3. the analysis frame is chosen;
4. the centreline is drawn between the landmarks clicked on that frame;
5. the contrast transit follows the heart;
6. the flow velocity is measured, on one view and on the two-view 3D path (both runs at their
   analysis frames, P5).

Compared with the truth:
- the ED frames: the ECG waveform's R peaks, and the image-based method on every run;
- the analysis frame;
- the velocity (the cycle-average front speed);
- the distal FFR of the true lumen with the measured velocity vs the true one (resting runs:
  the FAVOR Pilot conversion for both).

Targets were pre-registered in docs/PLAN.md (P7). ``--calibrate`` derives the velocity
intervals (:data:`~vstent.imaging.flow.VELOCITY_HALF_WIDTH`) on the development seeds.
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.config import load_config
from vstent.imaging.cardiac import ED_AREA_FRACTION, ed_from_ecg, ed_from_spread
from vstent.imaging.centerline import trace_centerline
from vstent.imaging.flow import (
    LEVEL,
    MAX_RUN_RATIO,
    VELOCITY_LOG_HALF_WIDTH,
    VELOCITY_LOG_HALF_WIDTH_TWO_RUNS,
    combine_runs,
    hyperemic_velocity,
)
from vstent.imaging.image_io import read_run_bytes
from vstent.imaging.run_analysis import analyse_frames, pixel_distance
from vstent.imaging.synthetic_cine import CineSynthConfig, render_run, run_dicom, sample_cine_case
from vstent.imaging.twoview import reconstruct
from vstent.physics.solver import PhysicsParams, solve_ffr

CONFIG = Path("configs/cine_synthetic.yaml")
PHYSICS = Path("configs/physics.yaml")
REPORT = "cine_synthetic"
CALIBRATION = "cine_calibration"
CALIBRATION_SEEDS = range(9164, 9200)  # development seeds not used to choose the method
TARGETS = {
    "ecg_beats": 0.98, "image_beats_15fps": 0.80, "analysis_frame": 0.80,
    "two_view_median": 0.10, "two_view_within": 0.90, "within": 0.25, "coverage": 0.85,
    "ffr_median": 0.02, "rejected": 0.20,
}  # fmt: skip


def _beats(true: tuple[int, ...], found: tuple[int, ...], keep=None) -> tuple[int, int]:
    """True ED frames (those passing ``keep``) with a found frame within ±1, and their count."""
    t = [f for f in true if keep is None or keep(f)]
    return sum(any(abs(f - g) <= 1 for g in found) for f in t), len(t)


def _velocity_row(fv, truth: float) -> dict[str, Any]:
    return {"cfv": fv.cfv, "lo": fv.interval[0], "hi": fv.interval[1], "accepted": fv.accepted,
            "reasons": list(fv.reasons), "err": fv.cfv / truth - 1.0 if np.isfinite(fv.cfv)
            else None, "covered": bool(fv.interval[0] <= truth <= fv.interval[1])}  # fmt: skip


def run_case(seed: int, cfg: CineSynthConfig, seg_dir: str) -> dict[str, Any]:
    """All metrics of one case (both runs, one view each and the two-view path)."""
    from vstent.inference.onnx_runtime import SegmentationRunner

    seg = SegmentationRunner.from_dir(seg_dir, intra_op_threads=1)
    params = load_config(PHYSICS, PhysicsParams)
    cc = sample_cine_case(seed, cfg)
    truth = cc.velocity_m_s
    row: dict[str, Any] = {"seed": seed, "velocity": truth, "heart_rate": 60.0 / cc.period_s,
                           "diastolic_ratio": cc.diastolic_ratio, "runs": []}  # fmt: skip
    rng = np.random.default_rng([seed, 99])
    analysed = []
    for k in range(2):
        run = cc.runs[k]
        frames = render_run(cc, k, probs=False).frames
        loaded = read_run_bytes(run_dicom(cc, k, frames, cfg), f"case{seed}_run{k}.dcm")
        rf = analyse_frames(loaded, seg.predict)
        true_ed = cc.ed_frames(k)
        op = rf.opacification
        image_ed = ed_from_spread(rf.spread, op.smoothed, loaded.frame_time_ms)
        well = op.smoothed >= ED_AREA_FRACTION * op.smoothed.max()
        r: dict[str, Any] = {
            "k": k, "fps": run.fps, "n_frames": run.n_frames, "has_ecg": run.has_ecg,
            "true_ed": list(true_ed), "source": rf.ed.source, "frame": rf.analysis_frame,
            "frame_err": int(min(abs(rf.analysis_frame - f) for f in true_ed)),
            "image_ed": list(image_ed.frames),
            "image_beats": _beats(true_ed, image_ed.frames, lambda f, w=well: bool(w[f])),
        }  # fmt: skip
        if loaded.ecg is not None:
            ecg = ed_from_ecg(loaded.ecg, len(loaded), loaded.frame_time_ms)
            r["ecg_beats"] = _beats(true_ed, ecg.frames)
        start, end = cc.landmarks(k, rf.analysis_frame)
        jitter = rng.normal(0.0, cfg.scene.click_sd_px, 4)
        start = (start[0] + jitter[0], start[1] + jitter[1])
        end = (end[0] + jitter[2], end[1] + jitter[3])
        try:
            mask = seg.predict(loaded.frames[rf.analysis_frame]) >= 0.5
            cl = trace_centerline(mask, start, end)
            fv, _ = rf.velocity(cl, rf.analysis_frame, pixel_distance(loaded.pixel_spacing_mm),
                                single_view=True)  # fmt: skip
            r["single"] = _velocity_row(fv, truth)
            analysed.append((rf, cl, loaded))
        except ValueError as exc:
            r["failed"] = str(exc)
            analysed.append(None)
        row["runs"].append(r)
    if all(a is not None for a in analysed):
        (rf1, cl1, in1), (rf2, cl2, in2) = analysed
        try:
            rec = reconstruct(in1.geometry, cl1, in2.geometry, cl2)
        except ValueError as exc:
            row["two_view_failed"] = str(exc)
            return row
        per = []
        for k, (rf, cl, _) in enumerate(analysed):
            fv, _ = rf.velocity(cl, rf.analysis_frame,
                                lambda s, k=k: np.interp(s, rec.view_s[:, k], rec.s),
                                single_view=False)  # fmt: skip
            per.append(fv)
        both = combine_runs(per)
        main = cc.case.tree.tree.branches[0]
        ffr_true = float(solve_ffr(main, params.model_copy(
            update={"v_hyp": hyperemic_velocity(truth)})).ffr[-1])  # fmt: skip
        two = {"length_err": rec.length / cc.case.main_length - 1.0,
               "runs": [_velocity_row(fv, truth) for fv in per],
               "combined": {**_velocity_row(both, truth), "n_runs": both.n_runs},
               "ffr_true_velocity": ffr_true}  # fmt: skip
        if both.accepted:
            two["ffr_measured"] = float(solve_ffr(main, params.model_copy(
                update={"v_hyp": both.v_hyp})).ffr[-1])  # fmt: skip
        row["two_view"] = two
    return row


def _task(args) -> dict[str, Any]:
    seed, cfg_dict, seg_dir = args
    return run_case(seed, CineSynthConfig.model_validate(cfg_dict), seg_dir)


def run_cases(cfg: CineSynthConfig, seeds, seg_dir: str, workers: int = 1) -> list[dict]:
    """``run_case`` for every seed; row order does not depend on ``workers``."""
    tasks = [(int(s), cfg.model_dump(mode="json"), str(seg_dir)) for s in seeds]
    if workers <= 1:
        return [_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_task, tasks, chunksize=1))


def conformal_quantile(scores, level: float = LEVEL) -> float | None:
    """Split-conformal quantile: the ⌈(n + 1)·level⌉-th smallest score."""
    s = np.sort(np.asarray(scores, float))
    if s.size == 0:
        return None
    return float(s[min(math.ceil((s.size + 1) * level), s.size) - 1])


def calibration(rows: list[dict]) -> dict[str, Any]:
    """Velocity interval half-widths: |ln(true / measured)| over accepted runs on the 3D path,
    and over the accepted means of both runs."""
    per, both = [], []
    for r in rows:
        tv = r.get("two_view")
        if not tv:
            continue
        per += [abs(math.log(r["velocity"] / x["cfv"])) for x in tv["runs"] if x["accepted"]]
        c = tv["combined"]
        if c["accepted"] and c["n_runs"] == 2:
            both.append(abs(math.log(r["velocity"] / c["cfv"])))
    return {"level": LEVEL, "per_run": conformal_quantile(per), "n_runs": len(per),
            "two_runs": conformal_quantile(both), "n_cases": len(both),
            "in_use": {"per_run": VELOCITY_LOG_HALF_WIDTH,
                       "two_runs": VELOCITY_LOG_HALF_WIDTH_TWO_RUNS}}  # fmt: skip


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def summarise(rows: list[dict]) -> dict[str, Any]:
    runs = [x for r in rows for x in r["runs"]]
    ecg = [x["ecg_beats"] for x in runs if "ecg_beats" in x]
    img = [x["image_beats"] for x in runs]
    img15 = [x["image_beats"] for x in runs if x["fps"] == 15.0]
    singles = [x["single"] for x in runs if "single" in x]
    acc_single = [s for s in singles if s["accepted"]]
    two = [r["two_view"] for r in rows if "two_view" in r]
    comb = [t["combined"] for t in two if t["combined"]["accepted"]]
    abs_err = np.abs([c["err"] for c in comb])
    ffr = [abs(t["ffr_measured"] - t["ffr_true_velocity"]) for t in two if "ffr_measured" in t]
    by_fps = {}
    for fps in sorted({x["fps"] for x in runs}):
        sel = [x for x in runs if x["fps"] == fps]
        k, n = map(sum, zip(*[x["image_beats"] for x in sel], strict=True)) if sel else (0, 0)
        e = [abs(x["single"]["err"]) for x in sel if "single" in x and x["single"]["accepted"]]
        by_fps[fps] = {"runs": len(sel), "image_beats": _rate(k, n),
                       "single_median": float(np.median(e)) if e else None}  # fmt: skip

    def beats(pairs):
        k, n = (sum(p[0] for p in pairs), sum(p[1] for p in pairs))
        return {"k": k, "n": n, "rate": _rate(k, n)}

    s = {
        "n_cases": len(rows), "n_runs": len(runs),
        "ecg_beats": beats(ecg), "image_beats": beats(img), "image_beats_15fps": beats(img15),
        "analysis_frame": {"k": sum(x["frame_err"] <= 1 for x in runs), "n": len(runs),
                           "rate": _rate(sum(x["frame_err"] <= 1 for x in runs), len(runs))},
        "analysis_frame_images": {
            "k": sum(x["frame_err"] <= 1 for x in runs if x["source"] == "images"),
            "n": sum(x["source"] == "images" for x in runs)},
        "by_fps": by_fps,
        "single": {"n": len(singles), "accepted": len(acc_single),
                   "median_err": float(np.median([s["err"] for s in acc_single]))
                   if acc_single else None,
                   "median_abs": float(np.median(np.abs([s["err"] for s in acc_single])))
                   if acc_single else None,
                   "coverage": _rate(sum(s["covered"] for s in acc_single), len(acc_single))},
        "rejected_runs": _rate(sum(not s["accepted"] for s in singles)
                               + sum("failed" in x for x in runs), len(runs)),
        "reasons": sorted({x.split(" (")[0] for s in singles for x in s["reasons"]}),
        "two_view": {"n": len(two), "accepted": len(comb), "failed": sum(
            "two_view_failed" in r for r in rows),
            "median_abs": float(np.median(abs_err)) if comb else None,
            "median_err": float(np.median([c["err"] for c in comb])) if comb else None,
            "within": _rate(int(np.sum(abs_err <= TARGETS["within"])), len(comb)),
            "coverage": _rate(sum(c["covered"] for c in comb), len(comb)),
            "ffr_median": float(np.median(ffr)) if ffr else None,
            "ffr_p90": float(np.percentile(ffr, 90)) if ffr else None,
            "length_abs": float(np.median(np.abs([t["length_err"] for t in two]))) if two
            else None},
    }  # fmt: skip
    t2, sg = s["two_view"], s["single"]

    def target(metric, value, goal, passed, fmt="{:.0%}"):
        return {
            "metric": metric,
            "value": "–" if value is None else fmt.format(value),
            "target": goal,
            "passed": bool(value is not None and passed),
        }

    s["targets"] = [
        target("ED frames from the ECG within ±1 frame (beats)", s["ecg_beats"]["rate"],
               "≥ 98%", (s["ecg_beats"]["rate"] or 0) >= TARGETS["ecg_beats"]),
        target("Image-based ED frames within ±1 frame at 15 fps (beats)",
               s["image_beats_15fps"]["rate"], "≥ 80%",
               (s["image_beats_15fps"]["rate"] or 0) >= TARGETS["image_beats_15fps"]),
        target("Analysis frame within ±1 frame of a true ED frame (runs)",
               s["analysis_frame"]["rate"], "≥ 80%",
               (s["analysis_frame"]["rate"] or 0) >= TARGETS["analysis_frame"]),
        target("Two views: median |CFV error| (cases)", t2["median_abs"], "≤ 10%",
               (t2["median_abs"] or 1) <= TARGETS["two_view_median"], "{:.1%}"),
        target("Two views: cases within ±25%", t2["within"], "≥ 90%",
               (t2["within"] or 0) >= TARGETS["two_view_within"]),
        target(f"Two views: {LEVEL:.0%} interval covers the true velocity (cases)",
               t2["coverage"], "≥ 85%", (t2["coverage"] or 0) >= TARGETS["coverage"]),
        target("Two views: median |FFR(measured) − FFR(true velocity)|", t2["ffr_median"],
               "≤ 0.02", (t2["ffr_median"] or 1) <= TARGETS["ffr_median"], "{:.3f}"),
        target(f"One view: {LEVEL:.0%} interval (widened) covers the true velocity (runs)",
               sg["coverage"], "≥ 85%", (sg["coverage"] or 0) >= TARGETS["coverage"]),
        target("Runs rejected by the guards", s["rejected_runs"], "≤ 20%",
               (s["rejected_runs"] if s["rejected_runs"] is not None else 1)
               <= TARGETS["rejected"]),
    ]  # fmt: skip
    return s


def _num(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def _pct(x, digits: int = 0) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def write_report(rows, cfg: CineSynthConfig, seeds, git: dict, seg_dir: str,
                 report_dir: str | Path = "reports") -> dict[str, Any]:  # fmt: skip
    """``reports/cine_synthetic.{md,json}``."""
    s = summarise(rows)
    report_dir = Path(report_dir)
    prov = {**git, "vstent_version": __version__, "seg_model": Path(seg_dir).as_posix(),
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    seeds = list(seeds)
    t2, sg = s["two_view"], s["single"]
    lines = [
        "# Cine runs: end-diastolic frames and flow velocity on synthetic beating hearts (P7)",
        "",
        f"> **{DISCLAIMER}** Synthetic data, not clinical data: these runs show that the "
        "method works where the truth is known, not that it agrees with real flow.",
        "",
        "Generated by `vstent cine-synthetic`. Do not edit by hand.",
        "",
        f"**Cases:** {s['n_cases']} synthetic cases (seeds {seeds[0]}–{seeds[-1]}, held out: "
        "no method or setting was chosen on them), two runs each from two C-arm views, "
        f"{s['n_runs']} runs in all (`configs/cine_synthetic.yaml`, "
        "`vstent.imaging.synthetic_cine`). P5's coronary trees beat on a heart sphere: the "
        "heart is largest at end-diastole (the ECG R wave) and contracts by "
        f"{cfg.contraction[0]:.0%}–{cfg.contraction[1]:.0%} with a twist. Breathing adds a "
        f"slow drift. Heart rate {cfg.heart_rate_bpm[0]:g}–{cfg.heart_rate_bpm[1]:g} bpm, "
        "7.5, 15 or 30 frames per second. Contrast fills the tree at a resting velocity of "
        f"{cfg.velocity_m_s[0]:g}–{cfg.velocity_m_s[1]:g} m/s (cycle average), "
        f"{cfg.diastolic_ratio[0]:g}–{cfg.diastolic_ratio[1]:g} times faster in diastole than "
        "in systole, then washes out. Each run is a DICOM with the frame time and C-arm "
        f"geometry; {cfg.ecg_prob:.0%} of runs also carry an ECG and the R-wave pointer.",
        "",
        "**Method (the product's steps):**",
        "- every frame is segmented with the exported segmenter;",
        "- ED frames: the R-wave pointer, else the ECG's R peaks, else the images (the vessel "
        "tree's spread, largest at end-diastole);",
        "- the analysis frame: the ED frame in the opacified window with the most vessel;",
        "- the centreline between the landmarks clicked on it;",
        "- the contrast front's arrival along it, following the heart;",
        "- the velocity: the slope of distance on flow time, which corrects for phasic flow "
        "with a diastolic / systolic ratio of 2 (D55). Distances come from one view or from "
        "the two-view 3D path.",
        "",
        f"Intervals: {LEVEL:.0%} split-conformal, the measured velocity × / ÷ "
        f"{math.exp(VELOCITY_LOG_HALF_WIDTH):.2f} (one run) and × / ÷ "
        f"{math.exp(VELOCITY_LOG_HALF_WIDTH_TWO_RUNS):.2f} (the mean of two runs, used only "
        f"when they agree within ×{MAX_RUN_RATIO:g}), calibrated on development seeds "
        f"{CALIBRATION_SEEDS[0]}–{CALIBRATION_SEEDS[-1]} (`reports/{CALIBRATION}.md`). One "
        "view: the upper end × 1.37 for foreshortening. FFR: the true lumen with the FAVOR "
        "Pilot hyperemic velocity of the measured vs the true resting velocity.",
        "",
        "## Pre-registered targets",
        "",
        "| Metric | Value | Target | Met |",
        "|---|---:|---:|---|",
        *(f"| {t['metric']} | {t['value']} | {t['target']} | {'yes' if t['passed'] else '**no**'} |"
          for t in s["targets"]),
        "",
        "## End-diastolic frames",
        "",
        f"- ECG waveform (R peaks; runs with an ECG): {s['ecg_beats']['k']} of "
        f"{s['ecg_beats']['n']} beats within ±1 frame.",
        f"- Images, all runs: {s['image_beats']['k']} of {s['image_beats']['n']} beats in the "
        f"well-opacified frames within ±1 frame ({_pct(s['image_beats']['rate'])}); at 15 fps "
        f"{_pct(s['image_beats_15fps']['rate'])}.",
        f"- Analysis frame within ±1 frame of a true ED frame: {s['analysis_frame']['k']} of "
        f"{s['analysis_frame']['n']} runs; with image-based ED frames "
        f"{s['analysis_frame_images']['k']} of {s['analysis_frame_images']['n']}.",
        "",
        "| Frames per second | Runs | Image-based ED beats within ±1 | One view: median "
        "|CFV error| |",
        "|---:|---:|---:|---:|",
        *(f"| {fps:g} | {v['runs']} | {_pct(v['image_beats'])} | {_pct(v['single_median'], 1)} |"
          for fps, v in s["by_fps"].items()),
        "",
        "## Flow velocity",
        "",
        f"- Two views (3D path; {t2['n']} cases, {t2['accepted']} with an accepted velocity, "
        f"{t2['failed']} reconstructions failed): median |error| {_pct(t2['median_abs'], 1)}, "
        f"median error {_pct(t2['median_err'], 1)}; within ±25%: {_pct(t2['within'])}; "
        f"interval coverage {_pct(t2['coverage'])}. 3D length error, median "
        f"|error| {_pct(t2['length_abs'], 1)}.",
        f"- FFR from the measured vs the true velocity (two views): median |difference| "
        f"{_num(t2['ffr_median'])}, 90th percentile {_num(t2['ffr_p90'])}.",
        f"- One view ({sg['n']} runs, {sg['accepted']} accepted): median error "
        f"{_pct(sg['median_err'], 1)} (foreshortening makes it negative), median |error| "
        f"{_pct(sg['median_abs'], 1)}; widened-interval coverage {_pct(sg['coverage'])}.",
        f"- Runs rejected by the guards (or not analysable): {_pct(s['rejected_runs'])}. "
        "Reasons seen: " + ("; ".join(s["reasons"]) or "none") + ".",
        "",
        "**Limits:**",
        "- The phasic correction assumes the same flow profile family that the generator "
        "draws from (systole ≈ 0.31 s, ratio 1.5–2.5), so its benefit here is partly by "
        "construction.",
        "- Real coronary flow profiles vary more (vessel, microvascular disease, arrhythmia).",
        "- The contrast injection itself adds flow.",
        "- Agreement with real flow needs a phantom or a Doppler wire (HUMAN_ACTIONS).",
        "",
        f"Segmentation model `{prov['seg_model']}`. Commit `{prov['git_commit']}` (dirty: "
        f"{prov['git_dirty']}), created {prov['created_utc']}.",
        "",
    ]  # fmt: skip
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{REPORT}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    payload = {"summary": s, "rows": rows, "provenance": prov, "seeds": seeds}
    (report_dir / f"{REPORT}.json").write_text(json.dumps(payload, indent=1, default=float),
                                               encoding="utf-8", newline="\n")  # fmt: skip
    return s


def write_calibration(rows, seeds, git: dict, seg_dir: str,
                      report_dir: str | Path = "reports") -> dict[str, Any]:  # fmt: skip
    """``reports/cine_calibration.{md,json}``: the velocity interval half-widths."""
    c = calibration(rows)
    report_dir = Path(report_dir)
    seeds = list(seeds)

    def q(x):
        return "–" if x is None else f"{x:.3f}"

    lines = [
        "# Velocity interval calibration (P7, development seeds)",
        "",
        f"> **{DISCLAIMER}** Synthetic data.",
        "",
        "Generated by `vstent cine-synthetic --calibrate`. Do not edit by hand.",
        "",
        f"Split-conformal {LEVEL:.0%} quantiles of |ln(true / measured velocity)| on "
        f"development seeds {seeds[0]}–{seeds[-1]}, which were not used to choose the method "
        "(D55):",
        "",
        "| Estimate | n | Quantile | In use (`vstent.imaging.flow`) |",
        "|---|---:|---:|---:|",
        f"| One run, 3D path | {c['n_runs']} runs | {q(c['per_run'])} | "
        f"{c['in_use']['per_run']:.2f} |",
        f"| Mean of both runs | {c['n_cases']} cases | {q(c['two_runs'])} | "
        f"{c['in_use']['two_runs']:.2f} |",
        "",
        f"Segmentation model `{Path(seg_dir).as_posix()}`. Commit `{git['git_commit']}` "
        f"(dirty: {git['git_dirty']}).",
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{CALIBRATION}.md").write_text("\n".join(lines), encoding="utf-8",
                                                  newline="\n")  # fmt: skip
    (report_dir / f"{CALIBRATION}.json").write_text(
        json.dumps({"calibration": c, "rows": rows, "seeds": seeds, **git}, indent=1,
                   default=float), encoding="utf-8", newline="\n")  # fmt: skip
    return c
