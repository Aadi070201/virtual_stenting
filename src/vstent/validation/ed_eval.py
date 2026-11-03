"""The learned ED frames (P4l, M2) against the product's image-based ED frames (M0), as
pre-registered in docs/PLAN.md, P4l.

- **Synthetic:** true ED beats in the well-opacified frames found within ±1 frame, per frame
  rate (as P7): validation runs for development, test runs once.
- **Real, without labels:** beat-to-beat reproducibility on CADICA runs (10 frames/s). For
  successive ED frames of a run, the mean distance between their vessel centrelines after an
  affine registration, divided by the same distance for both frames shifted by −1 and by +1.
  Lower is better: at the right phase the heart is back where it was one beat earlier.

Real runs are cached per run (resumable). Torch is needed only for the M2 predictions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from vstent.data.cadica import SelectedRun
from vstent.imaging.cardiac import ED_AREA_FRACTION, ed_from_spread
from vstent.imaging.motion import ECC_EPS, ECC_ITERATIONS, MAX_DRIFT

DISCLAIMER = "Research use only — not for clinical decisions."
CADICA_FRAME_MS = 100.0  # CADICA: 10 frames/s (docs/DATA.md)
REAL_DIR = Path("data/derived/ed_real")
SKELETON_LEVEL = 0.5  # centrelines of the smoothed half-size maps
MIN_SKELETON_PX = 50  # fewer centreline pixels: no distance
SHIFTS = (-1, 1)
METHODS = ("M0", "M2")
LABELS = {"M0": "M0: vessel-tree spread (product now)", "M2": "M2: learned (MobileNetV3 + time)"}

_WORKER: dict[str, Any] = {}


def beat_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """Mean distance (map px) between the vessel centrelines of two smoothed maps after an
    affine registration of ``b`` onto ``a`` (ECC), averaged both ways; ``None`` if the fit
    fails or drifts (as :mod:`vstent.imaging.motion`), or a map has too little vessel."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    w = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPS)
    try:
        _, w = cv2.findTransformECC(a, b, w, cv2.MOTION_AFFINE, criteria, None, 1)
    except cv2.error:
        return None
    if np.abs(w[:, :2] - np.eye(2)).max() > MAX_DRIFT:
        return None
    h, wd = a.shape
    bw = cv2.warpAffine(b, w, (wd, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    ska, skb = skeletonize(a >= SKELETON_LEVEL), skeletonize(bw >= SKELETON_LEVEL)
    if ska.sum() < MIN_SKELETON_PX or skb.sum() < MIN_SKELETON_PX:
        return None
    da, db = distance_transform_edt(~ska), distance_transform_edt(~skb)
    return 0.5 * (float(db[ska].mean()) + float(da[skb].mean()))


def reproducibility(
    maps: list[np.ndarray], frames: tuple[int, ...], well: np.ndarray
) -> dict[str, float] | None:
    """Beat-to-beat reproducibility of the ED ``frames``: over successive pairs whose frames
    and both shifted frames are well opacified, the mean centreline distance, the same for the
    pairs shifted by −1 and +1, and their ratio. ``None`` if no pair qualifies."""
    n = len(maps)

    def ok(i: int) -> bool:
        return 0 <= i < n and bool(well[i])

    d0, ds = [], []
    for a, b in zip(frames, frames[1:], strict=False):
        if not all(ok(a + s) and ok(b + s) for s in (0, *SHIFTS)):
            continue
        d = beat_distance(maps[a], maps[b])
        sh = [beat_distance(maps[a + s], maps[b + s]) for s in SHIFTS]
        if d is None or any(v is None for v in sh):
            continue
        d0.append(d)
        ds.append(float(np.mean(sh)))
    if not d0:
        return None
    return {"pairs": len(d0), "distance": float(np.mean(d0)), "shifted": float(np.mean(ds)),
            "ratio": float(np.mean(d0) / np.mean(ds))}  # fmt: skip


def _worker(seg_dir: str, model_dir: str):
    if not _WORKER:
        from vstent.imaging.ed_model import load_best
        from vstent.inference.onnx_runtime import SegmentationRunner

        _WORKER["seg"] = SegmentationRunner.from_dir(seg_dir)
        _WORKER["model"] = load_best(model_dir)
    return _WORKER["seg"], _WORKER["model"]


def real_run(run: SelectedRun, seg_dir: str, model_dir: str) -> dict[str, Any]:
    """M0 and M2 ED frames of one CADICA run and their reproducibility."""
    from vstent.imaging.ed_data import small_frames
    from vstent.imaging.ed_model import decode, predict
    from vstent.imaging.image_io import read_run
    from vstent.imaging.run_analysis import analyse_frames

    seg, model = _worker(seg_dir, model_dir)
    loaded = replace(read_run(run.frames_dir), frame_time_ms=CADICA_FRAME_MS)
    rf = analyse_frames(loaded, seg.predict)
    op = rf.opacification
    well = op.smoothed >= ED_AREA_FRACTION * op.smoothed.max()
    m0 = ed_from_spread(rf.spread, op.smoothed, CADICA_FRAME_MS).frames
    item = {"frames": small_frames(loaded.frames), "fps": 1000.0 / CADICA_FRAME_MS}
    m2 = decode(predict(model, [item], "cpu")[0])
    return {"key": f"{run.patient}_{run.video}", "patient": run.patient, "n_frames": len(loaded),
            "well": int(well.sum()), "M0": list(m0), "M2": list(m2),
            "rep_M0": reproducibility(rf.small, tuple(m0), well),
            "rep_M2": reproducibility(rf.small, tuple(m2), well)}  # fmt: skip


def real_rows(runs: list[SelectedRun], seg_dir: str, model_dir: str, cache_file: str | Path, *,
              workers: int = 1, progress: Callable[[int, int], None] | None = None,
              ) -> list[dict[str, Any]]:  # fmt: skip
    """:func:`real_run` for every run, cached in ``cache_file`` (saved after every run)."""
    cache_file = Path(cache_file)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for r in runs if f"{r.patient}_{r.video}" not in cache]

    def keep(row: dict[str, Any], k: int) -> None:
        cache[row["key"]] = row
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
        if progress:
            progress(len(runs) - len(todo) + k, len(runs))

    if workers <= 1:
        for k, r in enumerate(todo, 1):
            keep(real_run(r, seg_dir, model_dir), k)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(real_run, r, seg_dir, model_dir) for r in todo]
            for k, f in enumerate(as_completed(futs), 1):
                keep(f.result(), k)
    return [cache[f"{r.patient}_{r.video}"] for r in runs]


def synthetic_scores(run_dir: str | Path, split: str, data_dir: str | Path) -> dict[str, Any]:
    """M0 and M2 ±1-frame accuracy on the synthetic ``split`` runs, pooled and per frame rate."""
    from vstent.imaging import ed_data
    from vstent.imaging.ed_model import decode, load_best, predict, score

    runs = ed_data.load_split(split, data_dir)
    if not runs:
        raise ValueError(f"no ED data for {split} in {data_dir} (run `vstent ed-data`)")
    model = load_best(run_dir)
    m2 = [decode(p) for p in predict(model, runs, "cpu")]
    return {"split": split, "runs": len(runs), "M0": score(runs, [r["m0"] for r in runs]),
            "M2": score(runs, m2)}  # fmt: skip


def _bpm(frames: list[int]) -> float | None:
    if len(frames) < 2:
        return None
    return 60_000.0 / (float(np.median(np.diff(frames))) * CADICA_FRAME_MS)


def summarise_real(rows: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Mean reproducibility ratio per method on the runs where both have one, with patient
    bootstrap intervals; the beats found and their heart rates."""
    from vstent.validation.stats import cluster_bootstrap

    both = [r for r in rows if r["rep_M0"] and r["rep_M2"]]
    out: dict[str, Any] = {"runs": len(rows), "paired_runs": len(both),
                           "patients": len({r["patient"] for r in both})}  # fmt: skip
    ratio = {m: np.array([r[f"rep_{m}"]["ratio"] for r in both]) for m in METHODS}
    for m in METHODS:
        bpm = [b for b in (_bpm(r[m]) for r in rows) if b is not None]
        out[m] = {
            "with_value": sum(1 for r in rows if r[f"rep_{m}"]),
            "under_two": sum(1 for r in rows if len(r[m]) < 2),
            "beats_per_run": float(np.mean([len(r[m]) for r in rows])) if rows else None,
            "median_bpm": float(np.median(bpm)) if bpm else None,
            "bpm_outside_40_150": sum(1 for b in bpm if not 40 <= b <= 150),
            "mean_ratio": float(ratio[m].mean()) if both else None,
            "median_ratio": float(np.median(ratio[m])) if both else None,
            "mean_distance": (
                float(np.mean([r[f"rep_{m}"]["distance"] for r in both])) if both else None
            ),  # fmt: skip
        }
    if both:
        diff = ratio["M2"] - ratio["M0"]
        out["mean_difference"] = float(diff.mean())
        out["share_m2_lower"] = float((diff < 0).mean())
        patients = np.array([r["patient"] for r in both])
        ci = cluster_bootstrap(patients, lambda i: {"diff": float(diff[i].mean())}, n_boot=n_boot)
        out["difference_ci"] = ci["diff"][:2]
        out["m2_no_worse"] = bool(out["mean_difference"] <= 0.0)
    return out


def choose(synthetic: dict[str, Any], real: dict[str, Any]) -> tuple[str, str]:
    """P4l's choice: the best synthetic development ±1-frame accuracy among the methods whose
    real development reproducibility is no worse than M0's (M0 always qualifies)."""
    m0, m2 = synthetic["M0"]["rate"], synthetic["M2"]["rate"]
    if not real.get("paired_runs"):
        return "M0", "no real run has a reproducibility value for both methods"
    if not real["m2_no_worse"]:
        return "M0", (f"M2's mean reproducibility ratio {real['M2']['mean_ratio']:.3f} is worse "
                      f"than M0's {real['M0']['mean_ratio']:.3f}")  # fmt: skip
    if m2 > m0:
        return "M2", (f"M2 qualifies (ratio {real['M2']['mean_ratio']:.3f} ≤ "
                      f"{real['M0']['mean_ratio']:.3f}) and is more accurate ({m2:.1%} vs "
                      f"{m0:.1%})")  # fmt: skip
    return "M0", f"M2 qualifies but is not more accurate ({m2:.1%} vs {m0:.1%})"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.1%}"


def _synthetic_table(s: dict[str, Any]) -> list[str]:
    rates = sorted(s["M0"]["by_fps"], key=float)
    head = "| | " + " | ".join(f"{float(f):g} fps" for f in rates) + " | All |"
    lines = [head, "|---|" + "---:|" * (len(rates) + 1)]
    for m in METHODS:
        cells = [_pct(s[m]["by_fps"][f]["rate"]) for f in rates]
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + f" | {_pct(s[m]['rate'])} |")
    beats = [str(s["M0"]["by_fps"][f]["beats"]) for f in rates]
    lines.append("| ED beats scored | " + " | ".join(beats) + f" | {s['M0']['beats']} |")
    return lines


def _f(v: float | None, fmt: str = ".3f") -> str:
    return "—" if v is None else format(v, fmt)


def write_report(path: str | Path, *, stage: str, synthetic: dict[str, Any],
                 real: dict[str, Any] | None, model: str, seg_model: str,
                 git: dict[str, Any]) -> Path:  # fmt: skip
    """``reports/ed_model_dev.md`` (development) or ``reports/ed_model.md`` (test)."""
    path = Path(path)
    lines = [
        "# P4l: end-diastolic frames without an ECG", "", f"> **{DISCLAIMER}**", "",
        "Generated by `vstent ed-report`. Do not edit by hand. Pre-registered in docs/PLAN.md, "
        "P4l. Synthetic beating hearts with a known ECG at 7.5, 10, 15 and 30 fps; true ED "
        "beats in the well-opacified frames found within ±1 frame (as P7).", "",
        f"## Synthetic {'validation' if stage == 'dev' else 'test'} runs "
        f"({synthetic['runs']} runs)", "", *_synthetic_table(synthetic), "",
    ]  # fmt: skip
    if stage == "test":
        ok = {m: all((v["rate"] or 0) >= 0.95 for v in synthetic[m]["by_fps"].values())
              for m in METHODS}  # fmt: skip
        lines += [f"Target ≥ 95% at every frame rate: M0 {'met' if ok['M0'] else 'not met'}, "
                  f"M2 {'met' if ok['M2'] else 'not met'}.", ""]  # fmt: skip
    if real is not None:
        lines += [
            "## Real runs without labels: beat-to-beat reproducibility (CADICA development "
            "patients, 10 fps)", "",
            "Mean centreline distance between successive ED frames after an affine "
            "registration, divided by the same for both frames shifted by ±1. Lower is better; "
            "below 1 means the chosen frames repeat better than their neighbours.", "",
            "| | M0 | M2 |", "|---|---:|---:|",
        ]  # fmt: skip
        items = [("Runs with a value", "with_value", "d"), ("Runs with < 2 ED frames",
                 "under_two", "d"), ("ED frames per run", "beats_per_run", ".1f"),
                ("Median heart rate (bpm)", "median_bpm", ".0f"),
                ("Runs with a rate outside 40–150 bpm", "bpm_outside_40_150", "d"),
                ("Mean ratio (paired runs)", "mean_ratio", ".3f"),
                ("Median ratio (paired runs)", "median_ratio", ".3f"),
                ("Mean distance (half-size px)", "mean_distance", ".2f")]  # fmt: skip
        for label, key, fmt in items:
            lines.append(f"| {label} | {_f(real['M0'][key], fmt)} | {_f(real['M2'][key], fmt)} |")
        lines += ["", f"{real['runs']} runs; {real['paired_runs']} with a value for both "
                  f"methods ({real['patients']} patients)."]  # fmt: skip
        if real.get("paired_runs"):
            lo, hi = real["difference_ci"]
            lines += ["", f"M2 − M0 mean ratio: {real['mean_difference']:+.3f} (95% patient "
                      f"bootstrap {lo:+.3f} to {hi:+.3f}); M2 lower on "
                      f"{real['share_m2_lower']:.0%} of paired runs."]  # fmt: skip
        method, reason = choose(synthetic, real)
        lines += ["", f"**Choice (P4l rule):** {method}: {reason}.", ""]
    lines += [f"ED model `{model}`; segmentation model `{seg_model}`. Commit "
              f"`{git.get('git_commit')}` (dirty: {git.get('git_dirty')})."]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
