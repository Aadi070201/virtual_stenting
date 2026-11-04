"""P4q: a narrowing counts only where the lesion detector agrees, on CardioSYNTAX against SYNTAX
per side (docs/PLAN.md, P4q; pre-registered). Torch-free.

Development: P4p's sample (parts 0 and 9, seen before), variants A and B against the product's
rows from P4p; the higher Youden J wins (within 0.02: A), and the test runs only if one of them
raises J by ≥ 0.10. Test (once): part 1, never used, the product and the product with the
chosen variant on the same run analysis, frames and path. A side is flagged if any of its runs
is; a side counts when at least one of its runs was measured in every arm.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from vstent.data.cardiosyntax import FRAME_TIME_MS, Study, Video, read_video

DISCLAIMER = "Research use only — not for clinical decisions."
DET = "models/lesion_det"
VARIANTS: dict[str, tuple[tuple[str, ...], float]] = {  # pre-registered (P4q)
    "A": ((f"{DET}/lesion_det_dn121-20261103-170432",), 0.5),
    "B": (tuple(f"{DET}/{n}" for n in (
        "lesion_det_dn121-20261103-170432", "lesion_det_effb4-20261103-204058",
        "lesion_det_mitb2-20261102-230824", "lesion_det_mnv3-20261103-184830",
        "lesion_det_r34-20261102-195951")), 0.05),
}  # fmt: skip
TEST_PARTS = (1,)
MIN_GAIN, TIE = 0.10, 0.02  # development: J gain over the product; A within this of B
MIN_DROP = 0.20  # test: false alarms fall by at least this
DEV_ROWS = Path("data/derived/cardiosyntax/gate_dev.json")
TEST_ROWS = Path("data/derived/cardiosyntax/gate_test.json")
DECISION = Path("data/derived/cardiosyntax/gate_dev_decision.json")
KEEP = ("flagged", "ffr", "marked", "max_ds", "length_mm", "heartbeats", "gate")


def run_row(uid: str, side: str, video: Video, raw_dir: str, seg_dir: str,
            arms: dict[str, tuple[tuple[str, ...], float] | None]) -> dict[str, Any]:  # fmt: skip
    """One run through the product once per arm (``None``: no gate; else detector dirs and
    threshold), sharing the run's frame analysis."""
    from vstent.imaging.image_io import RunInput
    from vstent.inference.product import lesion_gate
    from vstent.validation.external import product_on_run

    row: dict[str, Any] = {"key": video.path, "study": uid, "side": side, "arms": {}}
    try:
        frames = read_video(raw_dir, video)
        run = RunInput(frames, Path(video.path).name, hashlib.sha256(frames.tobytes()).hexdigest(),
                       frame_time_ms=FRAME_TIME_MS, note="CardioSYNTAX run (15 fps)")  # fmt: skip
        rf = None
        for name, spec in arms.items():
            gate = None if spec is None else lesion_gate(spec[0], spec[1])
            measured, rf = product_on_run(run, seg_dir, gate=gate, rf=rf)
            row["arms"][name] = {k: measured[k] for k in KEEP}
    except MemoryError:
        row["error"] = "MemoryError"
    except Exception as exc:  # noqa: BLE001 — a run that cannot be measured is reported
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def from_p4p(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """P4p's cached product rows in this module's form (one arm, ``"product"``)."""
    return [{"key": r["key"], "study": r["study"], "side": r["side"],
             "arms": {"product": {k: r.get(k) for k in KEEP}}} if "error" not in r
            else {"key": r["key"], "study": r["study"], "side": r["side"], "error": r["error"]}
            for r in rows]  # fmt: skip


def merge(*row_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows of the same runs with their arms combined (a run failing in any set keeps its
    error)."""
    out: dict[str, dict[str, Any]] = {}
    for rows in row_sets:
        for r in rows:
            m = out.setdefault(r["key"], {"key": r["key"], "study": r["study"],
                                          "side": r["side"], "arms": {}})  # fmt: skip
            if "error" in r:
                m["error"] = r["error"]
            else:
                m["arms"].update(r["arms"])
    return list(out.values())


def _syntax(study: Study, side: str) -> float | None:
    return study.side_syntax("RCA" if side == "right" else "LCA")


def sides(studies: list[Study], rows: list[dict[str, Any]], arms: list[str]) -> list[dict]:
    """Per side with a SYNTAX score and a measured run in every arm: the study, the score,
    and per arm whether any run is flagged and the lowest FFR."""
    by = {s.uid: s for s in studies}
    runs: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if "error" not in r and all(a in r["arms"] for a in arms):
            runs.setdefault((r["study"], r["side"]), []).append(r)
    out = []
    for (uid, side), rs in sorted(runs.items()):
        sx = _syntax(by[uid], side)
        if sx is None:
            continue
        flagged = {a: any(r["arms"][a]["flagged"] for r in rs) for a in arms}
        low = {a: min(r["arms"][a]["ffr"] for r in rs) for a in arms}
        out.append({"study": uid, "side": side, "syntax": sx, "runs": rs, "flagged": flagged,
                    "low_ffr": low})  # fmt: skip
    return out


def youden(flagged: np.ndarray, positive: np.ndarray) -> tuple[float, float, float]:
    """(false alarms, sensitivity, J)."""
    fa = float(flagged[~positive].mean()) if (~positive).any() else float("nan")
    se = float(flagged[positive].mean()) if positive.any() else float("nan")
    return fa, se, se - fa


def summarise(studies: list[Study], rows: list[dict[str, Any]], arms: list[str],
              base: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Per arm: false alarms, sensitivity, J, sensitivity by SYNTAX bin, AUC of the lowest FFR,
    SYNTAX-0 runs with FFR ≤ 0.80, marks per SYNTAX-0 run, candidate narrowings confirmed.
    With ``base``, each other arm's paired J difference against it (bootstrap by study)."""
    from vstent.validation.cardiosyntax_check import SYNTAX_BINS
    from vstent.validation.stats import auc_delong, cluster_bootstrap

    sd = sides(studies, rows, arms)
    pos = np.array([s["syntax"] > 0 for s in sd])
    syn = np.array([s["syntax"] for s in sd])
    out: dict[str, Any] = {"runs": len(rows), "measured": sum("error" not in r for r in rows),
                           "sides": len(sd), "zero_sides": int((~pos).sum()),
                           "positive_sides": int(pos.sum()), "arms": {}}  # fmt: skip
    zero_runs = [r for s in sd if s["syntax"] == 0 for r in s["runs"]]
    for a in arms:
        f = np.array([s["flagged"][a] for s in sd])
        fa, se, j = youden(f, pos)
        low = np.array([s["low_ffr"][a] for s in sd])
        auc = auc_delong(-low, pos)[0] if pos.any() and (~pos).any() else None
        gates = [r["arms"][a]["gate"] for s in sd for r in s["runs"] if r["arms"][a]["gate"]]
        cand = sum(g[1] for g in gates)
        out["arms"][a] = {
            "false_alarm": fa, "sensitivity": se, "j": j,
            "by_bin": [{"bin": lab, "n": int(m.sum()),
                        "sensitivity": float(f[m].mean()) if m.any() else None}
                       for lo, hi, lab in SYNTAX_BINS for m in [(syn >= lo) & (syn < hi)]],
            "auc_low_ffr": None if auc is None else float(auc),
            "zero_runs_le_080": float(np.mean([round(r["arms"][a]["ffr"], 2) <= 0.80
                                               for r in zero_runs])) if zero_runs else None,
            "marks_per_zero_run": float(np.mean([r["arms"][a]["marked"] for r in zero_runs]))
            if zero_runs else None,
            "confirmed": None if not cand else sum(g[0] for g in gates) / cand,
        }  # fmt: skip
    if base is not None:
        clusters = np.array([s["study"] for s in sd])
        for a in arms:
            if a == base:
                continue

            def stat(idx, a=a):
                p = pos[idx]
                fb = np.array([sd[i]["flagged"][base] for i in idx])
                fn = np.array([sd[i]["flagged"][a] for i in idx])
                return {"dj": youden(fn, p)[2] - youden(fb, p)[2]}

            lo, hi, used = cluster_bootstrap(clusters, stat)["dj"]
            x = out["arms"][a]
            x["dj"] = x["j"] - out["arms"][base]["j"]
            x["dj_interval"] = [lo, hi]
            x["dj_replicates"] = used
    return out


def dev_decision(s: dict[str, Any]) -> dict[str, Any]:
    """P4q's development rule: the higher J of A and B (within :data:`TIE`: A); the test runs
    only if A or B raises J by ≥ :data:`MIN_GAIN` over the product."""
    j = {a: s["arms"][a]["j"] for a in ("product", "A", "B")}
    chosen = "A" if abs(j["A"] - j["B"]) <= TIE else max(("A", "B"), key=j.get)
    go = max(j["A"], j["B"]) - j["product"] >= MIN_GAIN
    return {"chosen": chosen if go else None, "j": j, "variant": VARIANTS[chosen],
            "reason": f"J product {j['product']:.3f}, A {j['A']:.3f}, B {j['B']:.3f}: "
            + (f"{chosen} chosen; the test may run" if go else
               f"neither raises J by ≥ {MIN_GAIN}; no test")}  # fmt: skip


def test_decision(s: dict[str, Any], arm: str) -> dict[str, Any]:
    """P4q's adoption rule on the test: false alarms fall by ≥ :data:`MIN_DROP` and the 95%
    interval of the paired J difference lies above 0."""
    b, g = s["arms"]["product"], s["arms"][arm]
    drop = b["false_alarm"] - g["false_alarm"]
    ok1, ok2 = drop >= MIN_DROP, g["dj_interval"][0] > 0
    return {"drop": drop, "dj": g["dj"], "dj_interval": g["dj_interval"],
            "criteria": [ok1, ok2], "adopt": ok1 and ok2}  # fmt: skip


def _pct(x: float | None) -> str:
    return "—" if x is None or x != x else f"{x:.0%}"


def write_report(path: str | Path, s: dict[str, Any], *, title: str, intro: str,
                 names: dict[str, str], decision_lines: list[str],
                 git: dict[str, Any]) -> Path:  # fmt: skip
    """``reports/p4q_lesion_gate_dev.md`` or ``reports/p4q_lesion_gate.md``."""
    f2 = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [f"# {title}", "", f"> **{DISCLAIMER}**", "",
             "Generated by `vstent lesion-gate`. Do not edit by hand. Pre-registered in "
             "docs/PLAN.md, P4q. " + intro, "",
             f"{s['measured']} of {s['runs']} runs measured; {s['sides']} sides "
             f"({s['zero_sides']} SYNTAX 0, {s['positive_sides']} SYNTAX > 0).", "",
             "| Arm | SYNTAX-0 sides flagged (false alarms) | SYNTAX > 0 sides flagged "
             "(sensitivity) | Youden J | AUC, lowest FFR | SYNTAX-0 runs FFR ≤ 0.80 | Marks per "
             "SYNTAX-0 run | Narrowings confirmed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for a, x in s["arms"].items():
        lines.append(f"| {names.get(a, a)} | {_pct(x['false_alarm'])} | {_pct(x['sensitivity'])}"
                     f" | {x['j']:.2f} | {f2(x['auc_low_ffr'])} | {_pct(x['zero_runs_le_080'])}"
                     f" | {f2(x['marks_per_zero_run'])} | {_pct(x['confirmed'])} |")  # fmt: skip
    bins = [b["bin"] for b in next(iter(s["arms"].values()))["by_bin"]]
    lines += ["", "Sensitivity by SYNTAX score of the side:", "",
              "| Arm | " + " | ".join(bins) + " |", "|---|" + "---:|" * len(bins)]  # fmt: skip
    for a, x in s["arms"].items():
        lines.append(f"| {names.get(a, a)} | "
                     + " | ".join(f"{_pct(b['sensitivity'])} (n = {b['n']})" for b in x["by_bin"])
                     + " |")  # fmt: skip
    paired = [(a, x) for a, x in s["arms"].items() if "dj" in x]
    if paired:
        lines += ["", "Paired J difference against the product (95% interval, bootstrap by "
                  "study, 2,000 resamples, seed 0):", ""]  # fmt: skip
        for a, x in paired:
            lo, hi = x["dj_interval"]
            lines.append(f"- {names.get(a, a)}: {x['dj']:+.3f} ({lo:+.3f} to {hi:+.3f})")
    lines += ["", *decision_lines, "",
              "**Limits:** SYNTAX is per side, and a run's automatic path may not follow the "
              "graded vessel. The pixel size is assumed (0.30 mm). Without FFR or lesion "
              "positions, this measures flags per side, not lesions found.", "",
              f"Commit `{git.get('git_commit')}` (dirty: {git.get('git_dirty')})."]  # fmt: skip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_decision(path: str | Path = DECISION) -> dict[str, Any]:
    """The development decision (written by ``vstent lesion-gate --stage dev``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def why(studies: list[Study], rows: list[dict[str, Any]], arm: str = "A",
        base: str = "product") -> dict[str, Any]:  # fmt: skip
    """Exploratory (after D72): why ``arm`` still flags SYNTAX-0 sides, and what it misses
    that ``base`` caught. A flagged side is "mark" if one of its runs has a ≥ 50% mark, else
    "FFR only" (FFR ≤ 0.80 without a mark); FFR-only runs are split by whether the detector
    confirmed any narrowing, and compared in path length with the unflagged SYNTAX-0 runs."""
    ss = sides(studies, rows, [base, arm])
    zero = [s for s in ss if s["syntax"] == 0]
    flagged = [s for s in zero if s["flagged"][arm]]
    mark = [s for s in flagged if any(r["arms"][arm]["marked"] for r in s["runs"])]
    ffr_runs = [r["arms"][arm] for s in flagged if s not in mark for r in s["runs"]
                if r["arms"][arm]["flagged"]]  # fmt: skip
    quiet = [r["arms"][arm] for s in zero for r in s["runs"] if not r["arms"][arm]["flagged"]]
    none = [x for x in ffr_runs if x["gate"] and x["gate"][0] == 0]
    med = lambda xs, k: float(np.median([x[k] for x in xs])) if xs else None  # noqa: E731
    missed = [s for s in ss if s["syntax"] > 0 and s["flagged"][base] and not s["flagged"][arm]]
    return {
        "zero_sides": len(zero), "flagged": len(flagged), "mark_sides": len(mark),
        "ffr_only_sides": len(flagged) - len(mark), "ffr_only_runs": len(ffr_runs),
        "ffr_only_no_confirmed": len(none),
        "length_ffr_only": med(ffr_runs, "length_mm"), "length_quiet": med(quiet, "length_mm"),
        "ffr_ffr_only": med(ffr_runs, "ffr"),
        "max_ds_ffr_only": med(ffr_runs, "max_ds"), "max_ds_quiet": med(quiet, "max_ds"),
        "missed": len(missed), "missed_syntax": sorted(s["syntax"] for s in missed),
    }  # fmt: skip


def why_lines(w: dict[str, Any], git: dict[str, Any]) -> list[str]:
    """``reports/p4q_why_dev.md``."""
    f = lambda v, fmt: "—" if v is None else format(v, fmt)  # noqa: E731
    return [
        "# P4q, exploratory: why normal sides are still flagged (development)", "",
        f"> **{DISCLAIMER}**", "",
        "Generated by `vstent lesion-gate --stage dev --why`. Do not edit by hand. Exploratory, "
        "after D72, on the development rows (CardioSYNTAX parts 0 and 9, P4p's sample); the "
        "test data is not used. The product with the gate (A: DenseNet121, t = 0.5).", "",
        f"- SYNTAX-0 sides still flagged: {w['flagged']} of {w['zero_sides']}.",
        f"  - by a ≥ 50% mark the detector confirmed: {w['mark_sides']} sides;",
        f"  - by FFR ≤ 0.80 without any mark: {w['ffr_only_sides']} sides "
        f"({w['ffr_only_runs']} runs; in {w['ffr_only_no_confirmed']} of them the detector "
        "confirmed no narrowing at all).",
        f"- Flagged by FFR only vs unflagged SYNTAX-0 runs: median path length "
        f"{f(w['length_ffr_only'], '.0f')} vs {f(w['length_quiet'], '.0f')} mm (at the assumed "
        f"0.30 mm/px); deepest narrowing {f(w['max_ds_ffr_only'], '.0f')}% vs "
        f"{f(w['max_ds_quiet'], '.0f')}%; median FFR of the flagged runs "
        f"{f(w['ffr_ffr_only'], '.2f')}.",
        f"- SYNTAX > 0 sides the product flagged and the gate no longer does: {w['missed']} "
        f"(SYNTAX {', '.join(f'{x:g}' for x in w['missed_syntax']) or '—'}).", "",
        "**Reading:** a flag without any mark comes from the physics along the whole path (its "
        "length and diameters at the assumed pixel size), not from a lesion.", "",
        f"Commit `{git.get('git_commit')}` (dirty: {git.get('git_dirty')}).",
    ]  # fmt: skip
