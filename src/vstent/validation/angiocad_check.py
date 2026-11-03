"""AngioCAD: the product on an external dataset with segment grades (docs/PLAN.md, P4k-J).

Every series of a patient's side goes through the product's steps: the run's analysis frame,
the automatic path, the product's QCA and physics (``configs/product.yaml``), the median over
up to five heartbeats. A side is **flagged** if any of its series has a marked lesion ≥ 50% or
a virtual FFR ≤ 0.80. Against the experts' segment grades:
1. sensitivity: significant sides (a segment ≥ 51%) flagged, by worst grade;
2. false alarms: normal sides (every segment NL) flagged;
3. normal sides' series with FFR ≤ 0.80 (De Bruyne 2001: about 13%).

Frame rate and pixel size are unknown (0.30 mm assumed, as in the product). No targets: run
once. Rows are cached per series (resumable). Frames and paths are kept, so the vessel
identity (G) and the detector (E) can later be read on the same frames.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vstent.data.angiocad import GRADES, SIDES, PatientLabels, series_dir

DISCLAIMER = "Research use only — not for clinical decisions."
ROWS_FILE = Path("data/derived/angiocad/rows.json")
DE_BRUYNE = 0.13


def series_row(patient: str, side: str, series: int, raw_dir: str, seg_dir: str) -> dict[str, Any]:
    """One series through the product's steps (:func:`~vstent.validation.external.
    product_on_run`)."""
    from vstent.imaging.image_io import read_run
    from vstent.validation.external import product_on_run

    row: dict[str, Any] = {"key": f"{patient}_{series}", "patient": patient, "side": side,
                           "series": series}  # fmt: skip
    folder = series_dir(raw_dir, patient, series)
    if not folder.is_dir():
        row["error"] = "missing: no frame folder"
        return row
    try:
        row.update(product_on_run(read_run(folder), seg_dir)[0])
    except Exception as exc:  # noqa: BLE001 — a series that cannot be measured is reported
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def tasks(labels: list[PatientLabels]) -> list[tuple[str, str, int]]:
    """(patient, side, series) for every listed series of every graded side."""
    return [(p.patient, side, s) for p in labels for side in SIDES if p.status(side)
            for s in p.series[side]]  # fmt: skip


def rows(labels: list[PatientLabels], raw_dir: str | Path, seg_dir: str | Path, *,
         cache_file: str | Path = ROWS_FILE, workers: int = 1,
         progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """:func:`series_row` for every task, cached in ``cache_file`` (saved every 10 series and
    at the end)."""
    cache_file = Path(cache_file)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks(labels) if f"{t[0]}_{t[2]}" not in cache]
    n_all, done = len(tasks(labels)), 0

    def keep(row: dict[str, Any]) -> None:
        nonlocal done
        cache[row["key"]] = row
        done += 1
        if done % 10 == 0 or done == len(todo):
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        if progress:
            progress(n_all - len(todo) + done, n_all)

    args = [(p, side, s, str(raw_dir), str(seg_dir)) for p, side, s in todo]
    if workers <= 1:
        for a in args:
            keep(series_row(*a))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(series_row, *a) for a in args]):
                keep(f.result())
    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[f"{p}_{s}"] for p, _side, s in tasks(labels)]


def summarise(labels: list[PatientLabels], series_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sides flagged by status and worst grade; normal sides' series with FFR ≤ 0.80."""
    by_side: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in series_rows:
        by_side.setdefault((r["patient"], r["side"]), []).append(r)
    sides = []
    for p in labels:
        for side in SIDES:
            status = p.status(side)
            if not status:
                continue
            rs = [r for r in by_side.get((p.patient, side), []) if "error" not in r]
            sides.append({"patient": p.patient, "side": side, "status": status,
                          "worst": p.worst(side), "measured": len(rs),
                          "flagged": any(r["flagged"] for r in rs) if rs else None,
                          "by_marks": any(r["marked"] for r in rs) if rs else None,
                          "by_ffr": any(round(r["ffr"], 2) <= 0.80 for r in rs) if rs else None,
                          })  # fmt: skip
    measured = [s for s in sides if s["measured"]]

    def share(items, key="flagged") -> dict[str, Any]:
        n = len(items)
        k = sum(1 for s in items if s[key])
        return {"n": n, "k": k, "share": k / n if n else None}

    out: dict[str, Any] = {
        "series": len(series_rows),
        "series_measured": sum(1 for r in series_rows if "error" not in r),
        "series_missing": sum(1 for r in series_rows if r.get("error", "").startswith("missing")),
        "sides": len(sides), "sides_measured": len(measured),
        "patients": len({s["patient"] for s in measured}),
    }  # fmt: skip
    for status in ("significant", "mild", "normal"):
        items = [s for s in measured if s["status"] == status]
        out[status] = {k: share(items, k) for k in ("flagged", "by_marks", "by_ffr")}
    out["by_worst"] = {g: share([s for s in measured if s["worst"] == g]) for g in GRADES}
    out["by_side"] = {side: {st: share([s for s in measured if s["side"] == side
                                        and s["status"] == st])
                             for st in ("significant", "normal")} for side in SIDES}  # fmt: skip
    normal_series = [r for r in series_rows if "error" not in r
                     and any(s["patient"] == r["patient"] and s["side"] == r["side"]
                             and s["status"] == "normal" for s in measured)]  # fmt: skip
    ffr = np.array([r["ffr"] for r in normal_series])
    out["normal_series"] = {"n": len(ffr), "ffr_le_080": float((np.round(ffr, 2) <= 0.80).mean())
                            if len(ffr) else None,
                            "median_ffr": float(np.median(ffr)) if len(ffr) else None}  # fmt: skip
    errors: dict[str, int] = {}
    for r in series_rows:
        if "error" in r:
            kind = r["error"].split(":")[0]
            errors[kind] = errors.get(kind, 0) + 1
    out["errors"] = errors
    return out


def _p(x: dict[str, Any]) -> str:
    return "—" if not x["n"] else f"{x['share']:.0%} ({x['k']}/{x['n']})"


def write_report(path: str | Path, s: dict[str, Any], *, seg_model: str, product: dict,
                 git: dict[str, Any]) -> Path:  # fmt: skip
    """``reports/p4k_angiocad.md``."""
    lines = [
        "# P4k-J: the product on AngioCAD (external, segment grades, no FFR)", "",
        f"> **{DISCLAIMER}**", "",
        "Generated by `vstent angiocad-check`. Do not edit by hand. Pre-registered in "
        "docs/PLAN.md, P4k-J (protocol written before any model was run on AngioCAD); run once, "
        "no targets. AngioCAD (Zenodo 10.5281/zenodo.15826856, CC BY 4.0): 413 patients, "
        "experts' grades of 15 segments; frame rate and pixel size unknown (0.30 mm assumed). "
        "A side is significant if a segment is ≥ 51%, normal if every segment is NL, mild "
        "otherwise. Flagged: a series of the side with a marked lesion ≥ 50% or virtual FFR "
        "≤ 0.80 (the product's analysis frame, automatic path, up to five heartbeats).", "",
        f"{s['series_measured']} of {s['series']} listed series measured "
        f"({s['series_missing']} without a frame folder); {s['sides_measured']} of "
        f"{s['sides']} graded sides ({s['patients']} patients).", "",
        "## Sides flagged", "",
        "| Side status | Flagged | by a lesion ≥ 50% | by FFR ≤ 0.80 |", "|---|---:|---:|---:|",
    ]  # fmt: skip
    for st in ("significant", "mild", "normal"):
        x = s[st]
        lines.append(f"| {st.capitalize()} | {_p(x['flagged'])} | {_p(x['by_marks'])} | "
                     f"{_p(x['by_ffr'])} |")  # fmt: skip
    lines += ["", "Sensitivity = significant sides flagged; false alarms = normal sides flagged.",
              "", "## By the side's worst grade", "", "| Worst grade | Flagged |",
              "|---|---:|"]  # fmt: skip
    lines += [f"| {g} | {_p(x)} |" for g, x in s["by_worst"].items()]
    lines += ["", "## By coronary side", "", "| | Significant flagged | Normal flagged |",
              "|---|---:|---:|"]  # fmt: skip
    for side, x in s["by_side"].items():
        lines.append(f"| {side.capitalize()} | {_p(x['significant'])} | {_p(x['normal'])} |")
    ns = s["normal_series"]
    if ns["n"]:
        lines += ["", "## Normal sides: series with virtual FFR ≤ 0.80", "",
                  f"{ns['ffr_le_080']:.0%} of {ns['n']} series (median FFR {ns['median_ffr']:.2f});"
                  f" anchor about {DE_BRUYNE:.0%} (De Bruyne 2001, non-stenotic arteries; the "
                  "automatic path may not follow the graded vessel)."]  # fmt: skip
    if s["errors"]:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(s["errors"].items()))
        lines += ["", f"Series not measured: {kinds}."]
    lines += ["", "**Limits:** grades are per segment, the automatic path follows one vessel per "
              "series, and a side's series show different vessels; a flag is not tied to the "
              "graded segment. No FFR: the FFR rows are descriptive.", "",
              f"Product: `{json.dumps(product)}`. Segmentation model `{seg_model}`. Commit "
              f"`{git.get('git_commit')}` (dirty: {git.get('git_dirty')})."]  # fmt: skip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
