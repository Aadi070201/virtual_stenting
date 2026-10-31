"""Automatic analysis frames of cine runs vs the frames CADICA's clinicians selected (P4d).

For every selected CADICA run, the segmented vessel area of every frame gives the opacification
curve (:mod:`vstent.imaging.cine`); its analysis frame and opacified window are compared with
the frames the clinicians chose for that run. Patients are split as in the stenosis grading
(:func:`~vstent.validation.stenosis_grading.patient_split`); the window fraction was set on the
development runs (D47). Torch-free. Frames in reports are 1-based, as in CADICA's file names.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vstent.data.cadica import SelectedRun
from vstent.imaging.centerline import longest_path_endpoints, trace_centerline
from vstent.imaging.cine import (
    LEVEL,
    OPACIFIED_FRACTION,
    contrast_transit,
    opacification,
    vessel_areas,
)
from vstent.imaging.image_io import read_run
from vstent.validation.stats import cluster_bootstrap, wilson

DISCLAIMER = "Research use only — not for clinical decisions."
AREAS_DIR = Path("data/derived/cadica_cine")


def run_key(run: SelectedRun) -> str:
    return f"{run.patient}_{run.video}"


def run_areas(
    runs: list[SelectedRun],
    segment: Callable[[np.ndarray], np.ndarray],
    cache_file: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[int]]:
    """Vessel pixels per frame of every run (``segment``: frame → probability map), cached in
    ``cache_file`` (JSON, run key → areas) and saved after every run, so an interrupted pass
    resumes."""
    cache_file = Path(cache_file)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    for k, run in enumerate(runs, 1):
        key = run_key(run)
        if key not in cache:
            cache[key] = vessel_areas(read_run(run.frames_dir).frames, segment).tolist()
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        if progress:
            progress(k, len(runs))
    return {run_key(r): cache[run_key(r)] for r in runs}


def score_run(run: SelectedRun, areas: list[int], fraction: float = OPACIFIED_FRACTION) -> dict:
    """The run's automatic analysis frame and window against the clinicians' frames (1-based).

    ``hit``: the analysis frame is one of the selected frames; ``in_span``: it lies between the
    first and last selected frame; ``iou``: overlap of the window with the selected frames
    (intersection / union of frame sets); ``arrival_before``: contrast arrival is not later than
    the first selected frame."""
    op = opacification(np.asarray(areas), fraction)
    window = set(range(op.window[0] + 1, op.window[1] + 2))
    sel = set(run.selected)
    analysis, arrival = op.analysis_frame + 1, op.arrival + 1
    return {
        "patient": run.patient, "video": run.video, "lesion": run.lesion,
        "n_frames": len(areas), "selected_first": min(sel), "selected_last": max(sel),
        "n_selected": len(sel), "analysis": analysis, "window_first": op.window[0] + 1,
        "window_last": op.window[1] + 1, "arrival": arrival,
        "hit": analysis in sel, "in_span": min(sel) <= analysis <= max(sel),
        "iou": len(window & sel) / len(window | sel), "arrival_before": arrival <= min(sel),
    }  # fmt: skip


def summarise(rows: list[dict], n_boot: int = 2000) -> dict[str, Any]:
    """Rates with Wilson intervals (runs) and the median IoU with a patient-level bootstrap."""
    if not rows:
        return {"n_runs": 0}
    out: dict[str, Any] = {"n_runs": len(rows), "n_patients": len({r["patient"] for r in rows})}
    for name in ("hit", "in_span", "arrival_before"):
        k = sum(r[name] for r in rows)
        out[name] = {"k": k, "n": len(rows), "rate": k / len(rows), "ci": wilson(k, len(rows))}
    iou = np.array([r["iou"] for r in rows])
    patients = np.array([r["patient"] for r in rows])
    boot = cluster_bootstrap(patients, lambda idx: {"iou": float(np.median(iou[idx]))}, n_boot)
    out["iou"] = {"median": float(np.median(iou)), "q1": float(np.percentile(iou, 25)),
                  "q3": float(np.percentile(iou, 75)), "ci": boot["iou"][:2]}  # fmt: skip
    return out


def transit_for_run(
    run: SelectedRun, segment: Callable[[np.ndarray], np.ndarray], analysis_frame: int
) -> dict[str, Any]:
    """Contrast transit along the longest vessel path of the run's analysis frame (0-based)."""
    frames = read_run(run.frames_dir).frames
    mask = segment(frames[analysis_frame]) >= LEVEL
    out: dict[str, Any] = {"patient": run.patient, "video": run.video}
    try:
        a, b = longest_path_endpoints(mask)
        cl = trace_centerline(mask, a, b)
    except ValueError as exc:
        return {**out, "error": str(exc)}
    tr = contrast_transit(frames, cl)
    if tr is None:
        return {**out, "error": "no contrast arrival measured along the path"}
    return {**out, "path_px": float(cl.s[-1]), "px_per_frame": tr.px_per_frame,
            "low": tr.interval[0], "high": tr.interval[1], "n_points": tr.n_points}  # fmt: skip


def _pct(x: dict) -> str:
    lo, hi = x["ci"]
    return f"{x['k']} / {x['n']} ({100 * x['rate']:.0f}%; {100 * lo:.0f}–{100 * hi:.0f}%)"


def _example_figure(rows: list[dict], areas: dict[str, list[int]], path: Path) -> None:
    from vstent.plotting import (
        BAND,
        INK_SECONDARY,
        SERIES_1,
        SERIES_2,
        new_figure,
        save_figure,
        style_axes,
    )

    fig = new_figure(9.0, 5.2, "Opacification curves of CADICA runs (development patients)",
                     "Segmented vessel area per frame; grey: span of the frames the clinicians "
                     "selected; orange: automatic window and analysis frame")  # fmt: skip
    for k, r in enumerate(rows[:6]):
        ax = fig.add_axes((0.07 + (k % 3) * 0.31, 0.5 - (k // 3) * 0.4, 0.26, 0.3))
        style_axes(ax)
        a = np.asarray(areas[f"{r['patient']}_{r['video']}"], dtype=float)
        t = np.arange(1, a.size + 1)
        ax.axvspan(r["selected_first"] - 0.5, r["selected_last"] + 0.5, color=BAND, zorder=1)
        ax.axvspan(r["window_first"] - 0.5, r["window_last"] + 0.5, ymin=0.0, ymax=0.06,
                   color=SERIES_2, zorder=2)  # fmt: skip
        ax.plot(t, a / max(a.max(), 1.0), color=SERIES_1, lw=1.4, zorder=3)
        ax.axvline(r["analysis"], color=SERIES_2, lw=1.2, zorder=4)
        ax.set_title(f"{r['patient']} {r['video']}", fontsize=8, color=INK_SECONDARY)
        ax.set_ylim(0, 1.08)
    save_figure(fig, path)


def _quartiles(v: np.ndarray) -> str:
    if not v.size:
        return "–"
    return f"{np.median(v):.1f} ({np.percentile(v, 25):.1f}–{np.percentile(v, 75):.1f})"


def write_report(
    rows: list[dict],
    transit: list[dict],
    areas: dict[str, list[int]],
    out_md: str | Path,
    *,
    fraction: float,
    arteries: dict[str, str],
    model: str,
    git: dict,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """``reports/cine_frames.{md,json}`` and the example figure (``rows``: :func:`score_run`
    rows with a ``split``; ``transit``: :func:`transit_for_run` rows). Returns the summaries."""
    out_md = Path(out_md)
    stats = {sp: summarise([r for r in rows if r["split"] == sp], n_boot) for sp in ("dev", "test")}
    test = [r for r in rows if r["split"] == "test"]
    by_kind = {
        "lesion runs": summarise([r for r in test if r["lesion"]], n_boot),
        "runs without lesion": summarise([r for r in test if not r["lesion"]], n_boot),
    }
    ok = [t for t in transit if "px_per_frame" in t]
    speeds = np.array([t["px_per_frame"] for t in ok])
    rel = np.array([100 * (t["high"] - t["low"]) / t["px_per_frame"] for t in ok
                    if t["px_per_frame"] > 0])  # fmt: skip
    side = {
        name: np.array(
            [t["px_per_frame"] for t in ok if arteries.get(f"{t['patient']}_{t['video']}") == name]
        )  # fmt: skip
        for name in ("left", "right")
    }
    errors = sorted({t["error"].split(" (")[0] for t in transit if "error" in t})
    _example_figure([r for r in rows if r["split"] == "dev"], areas,
                    out_md.parent / "figures" / "cine_frames.png")  # fmt: skip

    def line(name: str, s: dict) -> str:
        if not s.get("n_runs"):
            return f"| {name} | 0 | – | – | – | – |"
        i = s["iou"]
        return (f"| {name} | {s['n_runs']} ({s['n_patients']}) | {_pct(s['hit'])} | "
                f"{_pct(s['in_span'])} | {i['median']:.2f} ({i['q1']:.2f}–{i['q3']:.2f}; "
                f"{i['ci'][0]:.2f}–{i['ci'][1]:.2f}) | {_pct(s['arrival_before'])} |")  # fmt: skip

    head = [
        "| Runs | Runs (patients) | Analysis frame among the selected frames | Within their "
        "span | Window overlap (IoU), median (IQR; 95% CI) | Contrast arrival ≤ first selected "
        "frame |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines = [
        "# Cine runs: automatic analysis frame vs the CADICA clinicians' selected frames",
        "",
        f"> {DISCLAIMER}",
        "",
        "Generated by `vstent cine-frames`. Do not edit by hand.",
        "",
        "**Data:** the CADICA dataset (CC BY 4.0, `docs/DATA.md`): full cine runs of 42 "
        f"patients and, for each of its {len(rows)} selected runs, the frames the clinicians "
        "selected (the keyframes; in lesion runs the lesion boxes are drawn on them). These "
        "frames are the reference here; no lesion grade, flow or FFR is involved.",
        "",
        "**Method (no user input):** every frame is segmented; the vessel area per frame, "
        "median-filtered over 3 frames, is the opacification curve. The analysis frame is its "
        "peak (the most completely filled vessels); the opacified window is the run of frames "
        f"around the peak above {fraction:.0%} of it, a fraction chosen on the development "
        "runs (D47); contrast arrival is the first frame above 50%. Patients are split as in "
        "`reports/stenosis_grading.md`; this is the first use of the test patients for this "
        "task. Intervals: Wilson (rates; runs as units) and a patient-level bootstrap (median "
        "IoU).",
        "",
        "## Results",
        "",
        *head,
        line("test patients", stats["test"]),
        line("development patients", stats["dev"]),
        "",
        "Test patients by run type:",
        "",
        *head,
        *(line(k, s) for k, s in by_kind.items()),
        "",
        "![Opacification curves](figures/cine_frames.png)",
        "",
        "## Contrast transit (feasibility; no reference)",
        "",
        "Along the longest vessel path of each run's analysis frame: the frame in which the "
        "contrast arrives at each point (the darkest value within 6 px, for cardiac motion), "
        "and the front speed, the Theil–Sen slope of the distance from the inflow on the "
        "arrival frame. CADICA gives neither the frame rate nor the pixel size, so speeds are "
        "in px per frame; without a flow reference this shows only how often the measurement "
        "is possible and how precise it is.",
        "",
        f"- Measured in {len(ok)} of {len(transit)} runs "
        f"({100 * len(ok) / max(len(transit), 1):.0f}%). Not measured: "
        + ("; ".join(errors) or "none")
        + ".",
        f"- Front speed, px per frame, median (IQR): all runs {_quartiles(speeds)}; left "
        f"coronary {_quartiles(side['left'])}; right coronary {_quartiles(side['right'])}.",
        f"- Width of the speed's 95% interval relative to the speed, %, median (IQR): "
        f"{_quartiles(rel)}.",
        "",
        f"Segmentation model `{model}`. Commit `{git['git_commit']}` (dirty: "
        f"{git['git_dirty']}). Per-run values: `cine_frames.json`.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    payload = {"fraction": fraction, "stats": stats, "by_kind": by_kind, "runs": rows,
               "transit": transit, "model": model, **git}  # fmt: skip
    out_md.with_suffix(".json").write_text(json.dumps(payload, indent=1, default=float),
                                           encoding="utf-8")  # fmt: skip
    return stats
