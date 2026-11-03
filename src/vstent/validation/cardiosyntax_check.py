"""P4o: the product on CardioSYNTAX against SYNTAX per side (docs/PLAN.md, P4o; pre-registered,
run once, no targets).

Studies without bypass whose runs are in zip parts 0 and 9; runs with ≥ 10 frames and a
confident LCA / RCA tag. Every run goes through the product's steps
(:func:`~vstent.validation.external.product_on_run`) at 15 fps. A side is flagged if any of its
runs is. Also each run's ED frames from M0 (images) and the learned model, for their implied
heart rate at a known frame rate. Rows are cached per run (resumable; a run that ran out of
memory is not cached, so it is tried again).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vstent.data.cardiosyntax import FRAME_TIME_MS, Study, Video, read_video

DISCLAIMER = "Research use only — not for clinical decisions."
PARTS = (0, 9)
MIN_FRAMES = 10
TAG_SURE = 0.1  # RCA probability ≤ 0.1: LCA; ≥ 0.9: RCA
ROWS_FILE = Path("data/derived/cardiosyntax/rows.json")
SYNTAX_BINS = ((1.0, 8.0, "1–7"), (8.0, 15.0, "8–14"), (15.0, float("inf"), "≥ 15"))
BPM_RANGE = (40.0, 150.0)
DE_BRUYNE = 0.13
_ED: dict[str, Any] = {}


def side_of(video: Video) -> str | None:
    """The run's side from the authors' tag, if confident."""
    if video.rca_prob <= TAG_SURE:
        return "left"
    if video.rca_prob >= 1.0 - TAG_SURE:
        return "right"
    return None


def tasks(studies: list[Study], parts: tuple[int, ...] = PARTS) -> list[tuple[str, str, Video]]:
    """(study, side, run) for every run of P4o's set."""
    return [(s.uid, side, v) for s in studies if not s.bypass for v in s.videos
            if v.part in parts and v.n_frames >= MIN_FRAMES and (side := side_of(v))]  # fmt: skip


def _ed_model(model_dir: str | None):
    if model_dir and model_dir not in _ED:
        from vstent.imaging.ed_model import load_best

        _ED[model_dir] = load_best(model_dir)
    return _ED.get(model_dir) if model_dir else None


def run_row(uid: str, side: str, video: Video, raw_dir: str, seg_dir: str,
            ed_model_dir: str | None, detail: bool = False) -> dict[str, Any]:  # fmt: skip
    """One run: the product's measurements and the ED frames of M0 and the learned model (with
    ``detail``, also the vessel profile, :func:`~vstent.validation.external.profile_of`)."""
    from vstent.imaging.image_io import RunInput
    from vstent.validation.external import product_on_run

    row: dict[str, Any] = {"key": video.path, "study": uid, "side": side}
    try:
        frames = read_video(raw_dir, video)
        run = RunInput(frames, Path(video.path).name, hashlib.sha256(frames.tobytes()).hexdigest(),
                       frame_time_ms=FRAME_TIME_MS, note="CardioSYNTAX run (15 fps)")  # fmt: skip
        measured, rf = product_on_run(run, seg_dir, detail=detail)
        row.update(measured)
        row["ed_m0"] = list(map(int, rf.ed.frames))
        model = _ed_model(ed_model_dir)
        if model is not None:
            from vstent.imaging.ed_data import small_frames
            from vstent.imaging.ed_model import decode, predict

            item = {"frames": small_frames(frames), "fps": 1000.0 / FRAME_TIME_MS}
            row["ed_m2"] = list(decode(predict(model, [item], "cpu")[0]))
    except MemoryError:
        row["error"] = "MemoryError"
    except Exception as exc:  # noqa: BLE001 — a run that cannot be measured is reported
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def rows(studies: list[Study], raw_dir: str | Path, seg_dir: str | Path,
         ed_model_dir: str | None, *, cache_file: str | Path = ROWS_FILE, workers: int = 1,
         progress: Callable[[int, int], None] | None = None,
         task_list: list[tuple[str, str, Video]] | None = None,
         detail: bool = False) -> list[dict[str, Any]]:  # fmt: skip
    """:func:`run_row` for every task (or ``task_list``), cached in ``cache_file`` (saved every
    10 runs)."""
    cache_file = Path(cache_file)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    all_tasks = tasks(studies) if task_list is None else task_list
    todo = [t for t in all_tasks if t[2].path not in cache]
    done = 0

    def keep(row: dict[str, Any]) -> None:
        nonlocal done
        if row.get("error") != "MemoryError":
            cache[row["key"]] = row
        done += 1
        if done % 10 == 0 or done == len(todo):
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        if progress:
            progress(len(all_tasks) - len(todo) + done, len(all_tasks))

    args = [(u, side, v, str(raw_dir), str(seg_dir), ed_model_dir, detail) for u, side, v in todo]
    if workers <= 1:
        for a in args:
            keep(run_row(*a))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(run_row, *a) for a in args]):
                keep(f.result())
    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[v.path] for _u, _s, v in all_tasks if v.path in cache]


def bpm(frames: list[int]) -> float | None:
    """Implied heart rate (beats/min) from successive ED frames at 15 fps."""
    if len(frames) < 2:
        return None
    return 60_000.0 / (float(np.median(np.diff(frames))) * FRAME_TIME_MS)


def _share(items: list[dict], key: str = "flagged") -> dict[str, Any]:
    n, k = len(items), sum(1 for s in items if s[key])
    return {"n": n, "k": k, "share": k / n if n else None}


def summarise(studies: list[Study], run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sides flagged by SYNTAX, FFR on SYNTAX-0 sides, ED plausibility."""
    by = {s.uid: s for s in studies}
    ok = [r for r in run_rows if "error" not in r]
    sides: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        sides.setdefault((r["study"], r["side"]), []).append(r)
    side_rows = []
    for (uid, side), rs in sides.items():
        sx = by[uid].side_syntax("RCA" if side == "right" else "LCA")
        if sx is None:
            continue
        side_rows.append({"syntax": float(sx), "side": side, "runs": len(rs),
                          "flagged": any(r["flagged"] for r in rs),
                          "by_marks": any(r["marked"] for r in rs),
                          "by_ffr": any(round(r["ffr"], 2) <= 0.80 for r in rs)})  # fmt: skip
    zero = [s for s in side_rows if s["syntax"] == 0]
    pos = [s for s in side_rows if s["syntax"] > 0]
    out: dict[str, Any] = {
        "runs": len(run_rows), "measured": len(ok), "studies": len({r["study"] for r in ok}),
        "sides": len(side_rows),
        "false_alarm": {k: _share(zero, k) for k in ("flagged", "by_marks", "by_ffr")},
        "sensitivity": {k: _share(pos, k) for k in ("flagged", "by_marks", "by_ffr")},
        "by_syntax": {label: _share([s for s in pos if lo <= s["syntax"] < hi])
                      for lo, hi, label in SYNTAX_BINS},
        "by_side": {side: {"zero": _share([s for s in zero if s["side"] == side]),
                           "positive": _share([s for s in pos if s["side"] == side])}
                    for side in ("left", "right")},
    }  # fmt: skip
    zero_keys = {(uid, side) for uid, side in sides
                 if by[uid].side_syntax("RCA" if side == "right" else "LCA") == 0}  # fmt: skip
    ffr0 = np.array([r["ffr"] for r in ok if (r["study"], r["side"]) in zero_keys])
    out["zero_runs_ffr"] = {"n": int(ffr0.size),
                            "le_080": float((np.round(ffr0, 2) <= 0.80).mean()) if ffr0.size
                            else None,
                            "median": float(np.median(ffr0)) if ffr0.size else None}  # fmt: skip
    for name in ("ed_m0", "ed_m2"):
        rates = [b for r in ok if name in r and (b := bpm(r[name])) is not None]
        have = sum(1 for r in ok if name in r)
        out[name] = {"runs": have, "with_rate": len(rates),
                     "median_bpm": float(np.median(rates)) if rates else None,
                     "in_range": float(np.mean([BPM_RANGE[0] <= b <= BPM_RANGE[1]
                                                for b in rates])) if rates else None}  # fmt: skip
    errors: dict[str, int] = {}
    for r in run_rows:
        if "error" in r:
            kind = r["error"].split(":")[0]
            errors[kind] = errors.get(kind, 0) + 1
    out["errors"] = errors
    return out


def _p(x: dict[str, Any]) -> str:
    return "—" if not x["n"] else f"{x['share']:.0%} ({x['k']}/{x['n']})"


def write_report(path: str | Path, s: dict[str, Any], *, seg_model: str, ed_model: str | None,
                 product: dict, git: dict[str, Any]) -> Path:  # fmt: skip
    """``reports/p4o_cardiosyntax.md``."""
    fa, se = s["false_alarm"], s["sensitivity"]
    lines = [
        "# P4o: the product on CardioSYNTAX (external; SYNTAX per side, no FFR)", "",
        f"> **{DISCLAIMER}**", "",
        "Generated by `vstent cardiosyntax-check`. Do not edit by hand. Pre-registered in "
        "docs/PLAN.md, P4o; run once, no targets. CardioSYNTAX (Zenodo 10.5281/zenodo.14005818, "
        "CC BY 4.0; Tyumen Cardiology Research Center): zip parts 0 and 9, studies without "
        "bypass, runs with ≥ 10 frames and a confident LCA / RCA tag, 15 fps. A side with "
        "SYNTAX 0 has no lesion ≥ 50% in a vessel ≥ 1.5 mm. Flagged: a run of the side with a "
        "marked lesion ≥ 50% or virtual FFR ≤ 0.80. Pixel size unknown (0.30 mm assumed).", "",
        f"{s['measured']} of {s['runs']} runs measured ({s['studies']} studies, {s['sides']} "
        "sides).", "",
        "## Sides flagged", "",
        "| Sides | Flagged | by a lesion ≥ 50% | by FFR ≤ 0.80 |", "|---|---:|---:|---:|",
        f"| SYNTAX 0 (false alarms) | {_p(fa['flagged'])} | {_p(fa['by_marks'])} | "
        f"{_p(fa['by_ffr'])} |",
        f"| SYNTAX > 0 (sensitivity) | {_p(se['flagged'])} | {_p(se['by_marks'])} | "
        f"{_p(se['by_ffr'])} |", "",
        "| Side's SYNTAX | Flagged |", "|---|---:|",
        *(f"| {label} | {_p(x)} |" for label, x in s["by_syntax"].items()), "",
        "| | SYNTAX 0 flagged | SYNTAX > 0 flagged |", "|---|---:|---:|",
        *(f"| {side.capitalize()} | {_p(x['zero'])} | {_p(x['positive'])} |"
          for side, x in s["by_side"].items()),
    ]  # fmt: skip
    z = s["zero_runs_ffr"]
    if z["n"]:
        lines += ["", f"Runs of SYNTAX-0 sides with virtual FFR ≤ 0.80: {z['le_080']:.0%} of "
                  f"{z['n']} (median FFR {z['median']:.2f}); De Bruyne about 13% in non-stenotic "
                  "arteries."]  # fmt: skip
    lines += ["", "## ED frames without an ECG at a known frame rate (plausibility, no truth)",
              "", "| Method | Runs | With a rate | Median implied heart rate | Within 40–150 bpm |",
              "|---|---:|---:|---:|---:|"]  # fmt: skip
    for name, label in (("ed_m0", "M0 (product)"), ("ed_m2", "M2b `tcn5`")):
        x = s[name]
        med = "—" if x["median_bpm"] is None else f"{x['median_bpm']:.0f} bpm"
        rng = "—" if x["in_range"] is None else f"{x['in_range']:.0%}"
        lines.append(f"| {label} | {x['runs']} | {x['with_rate']} | {med} | {rng} |")
    if s["errors"]:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(s["errors"].items()))
        lines += ["", f"Runs not measured: {kinds}."]
    lines += ["", "**Limits:** SYNTAX is scored per side, the automatic path follows one vessel "
              "per run, and a side's runs show different vessels; a flag is not tied to a "
              "graded lesion. The LCA / RCA tags come from the authors' classifier. No FFR.", "",
              f"Product: `{json.dumps(product)}`. Segmentation model `{seg_model}`; ED model "
              f"`{ed_model}`. Commit `{git.get('git_commit')}` "
              f"(dirty: {git.get('git_dirty')})."]  # fmt: skip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
