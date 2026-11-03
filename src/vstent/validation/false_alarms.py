"""P4p (exploratory, development data): where along the vessel the product's false alarms arise
(docs/PLAN.md, P4p).

P4o found almost every SYNTAX-0 side flagged. Here a seeded sample of P4o's sides (CardioSYNTAX
parts 0 and 9, already reported once) goes through the product again, keeping each run's vessel
profile (:func:`~vstent.validation.external.profile_of`). Fixed candidate reading rules are then
applied to the saved profiles: which lesions count (by the reference diameter at the narrowest
point) and where the FFR is read (the path's end, or the last point whose reference diameter is
at least the limit, where a pressure wire would sit). Nothing here changes the product; a rule
can become the product only through a pre-registered test on data not used here. Torch-free.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from vstent.data.cardiosyntax import Study, Video

DISCLAIMER = "Research use only — not for clinical decisions."
PROFILES_FILE = Path("data/derived/cardiosyntax/profiles.json")
N_ZERO, N_POSITIVE, SEED = 60, 40, 0  # sides sampled: SYNTAX 0, SYNTAX > 0
MARK_DS = 50.0  # %DS of a marked lesion (the app's red mark)
CUTOFF = 0.80


class Rule:
    """A reading rule: lesions count only where the reference diameter at the narrowest point
    is ≥ ``marks_ref_mm`` (None: every lesion), and the FFR is read at the last point whose
    reference diameter is ≥ ``ffr_ref_mm`` (None: the path's end)."""

    def __init__(self, name: str, marks_ref_mm: float | None, ffr_ref_mm: float | None):
        self.name, self.marks_ref_mm, self.ffr_ref_mm = name, marks_ref_mm, ffr_ref_mm


RULES = (
    Rule("Product (every lesion, FFR at the path's end)", None, None),
    Rule("Lesions in vessels ≥ 2 mm", 2.0, None),
    Rule("FFR at the last point ≥ 2 mm", None, 2.0),
    Rule("Both at 2 mm (FAVOR II)", 2.0, 2.0),
    Rule("Both at 1.5 mm (SYNTAX)", 1.5, 1.5),
)


def _syntax(study: Study, side: str) -> float | None:
    return study.side_syntax("RCA" if side == "right" else "LCA")


def sample_tasks(studies: list[Study], n_zero: int = N_ZERO, n_positive: int = N_POSITIVE,
                 seed: int = SEED) -> list[tuple[str, str, Video]]:  # fmt: skip
    """Every P4o run of ``n_zero`` SYNTAX-0 and ``n_positive`` SYNTAX > 0 sides, drawn with
    ``seed`` from P4o's set (:func:`~vstent.validation.cardiosyntax_check.tasks`)."""
    from vstent.validation.cardiosyntax_check import tasks

    by = {s.uid: s for s in studies}
    runs: dict[tuple[str, str], list] = {}
    for t in tasks(studies):
        runs.setdefault((t[0], t[1]), []).append(t)
    keys = sorted(runs)
    zero = [k for k in keys if _syntax(by[k[0]], k[1]) == 0]
    pos = [k for k in keys if (_syntax(by[k[0]], k[1]) or 0) > 0]
    rng = random.Random(seed)
    chosen = rng.sample(zero, min(n_zero, len(zero))) + rng.sample(pos, min(n_positive, len(pos)))
    return [t for k in sorted(chosen) for t in runs[k]]


def wire_mm(profile: dict[str, Any], min_ref_mm: float) -> float:
    """Arc length (mm) of the last profile point whose reference diameter is ≥ ``min_ref_mm``
    (the path's end if none is, as :func:`~vstent.imaging.qca.wire_index`)."""
    ref = np.asarray(profile["ref_mm"])
    ok = np.flatnonzero(ref >= min_ref_mm)
    return float(profile["s_mm"][ok[-1] if ok.size else -1])


def read(row: dict[str, Any], rule: Rule) -> tuple[bool, float, float]:
    """(a marked lesion counts, the FFR read, the deepest counted %DS) of one run's profile
    under ``rule``."""
    p = row["profile"]
    deep = [les[3] for les in p["lesions"]
            if rule.marks_ref_mm is None or les[5] >= rule.marks_ref_mm]  # fmt: skip
    ffr = np.asarray(p["ffr"])
    at = p["s_mm"][-1] if rule.ffr_ref_mm is None else wire_mm(p, rule.ffr_ref_mm)
    value = float(np.interp(at, p["s_mm"], ffr))
    max_ds = max(deep, default=0.0)
    return max_ds >= MARK_DS, value, max_ds


def summarise(studies: list[Study], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per rule: sides flagged (SYNTAX 0: false alarms; > 0: sensitivity), runs of SYNTAX-0
    sides with FFR ≤ 0.80, and the AUC of each side's lowest FFR and deepest counted %DS for
    SYNTAX > 0."""
    from vstent.validation.stats import auc_delong

    by = {s.uid: s for s in studies}
    ok = [r for r in rows if "profile" in r]
    sides: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        sides.setdefault((r["study"], r["side"]), []).append(r)
    keys = [k for k in sides if _syntax(by[k[0]], k[1]) is not None]
    positive = np.array([_syntax(by[k[0]], k[1]) > 0 for k in keys])
    zero_runs = [r for k in keys if _syntax(by[k[0]], k[1]) == 0 for r in sides[k]]
    out: dict[str, Any] = {"runs": len(rows), "measured": len(ok), "sides": len(keys),
                           "zero_sides": int((~positive).sum()),
                           "positive_sides": int(positive.sum()), "rules": []}  # fmt: skip
    for rule in RULES:
        read_k = {k: [read(r, rule) for r in sides[k]] for k in keys}
        flagged = np.array([any(m or round(f, 2) <= CUTOFF for m, f, _ in read_k[k]) for k in keys])
        low_ffr = np.array([min(f for _, f, _ in read_k[k]) for k in keys])
        deep = np.array([max(d for _, _, d in read_k[k]) for k in keys])
        zr = np.array([read(r, rule)[1] for r in zero_runs])
        auc_ffr = auc_delong(-low_ffr, positive) if positive.any() and (~positive).any() else None
        auc_ds = auc_delong(deep, positive) if positive.any() and (~positive).any() else None
        out["rules"].append({
            "name": rule.name, "marks_ref_mm": rule.marks_ref_mm, "ffr_ref_mm": rule.ffr_ref_mm,
            "false_alarm": float(flagged[~positive].mean()) if (~positive).any() else None,
            "sensitivity": float(flagged[positive].mean()) if positive.any() else None,
            "zero_runs_le_080": float((np.round(zr, 2) <= CUTOFF).mean()) if zr.size else None,
            "zero_runs_median": float(np.median(zr)) if zr.size else None,
            "auc_ffr": None if auc_ffr is None else float(auc_ffr[0]),
            "auc_ds": None if auc_ds is None else float(auc_ds[0]),
        })  # fmt: skip
    # where the counted lesions of SYNTAX-0 runs sit: reference diameter at the narrowest point
    refs = np.array([les[5] for r in zero_runs for les in r["profile"]["lesions"]
                     if les[3] >= MARK_DS])  # fmt: skip
    bins = ((0.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 99.0))
    out["zero_marks_by_ref"] = [{"lo": lo, "hi": hi, "n": int(((refs >= lo) & (refs < hi)).sum())}
                                for lo, hi in bins]  # fmt: skip
    out["zero_marks"] = int(refs.size)
    out["zero_runs"] = len(zero_runs)
    out["product_flag_agrees"] = float(np.mean([r["flagged"] == _product_flag(r) for r in ok]))
    return out


def _product_flag(row: dict[str, Any]) -> bool:
    """The product's rule from the chosen heartbeat's profile (the cached flag uses the median
    FFR over heartbeats)."""
    marked, ffr, _ = read(row, RULES[0])
    return marked or round(ffr, 2) <= CUTOFF


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def write_report(path: str | Path, s: dict[str, Any], *, product: dict,
                 git: dict[str, Any]) -> Path:  # fmt: skip
    """``reports/p4p_false_alarms_dev.md``."""
    lines = [
        "# P4p: where the false alarms arise (exploratory, development data)", "",
        f"> **{DISCLAIMER}**", "",
        "Generated by `vstent false-alarms`. Do not edit by hand. Exploratory, after P4o's "
        "result (docs/PLAN.md, P4p): CardioSYNTAX parts 0 and 9, P4o's set, already reported "
        f"once. Sample: {N_ZERO} SYNTAX-0 and {N_POSITIVE} SYNTAX > 0 sides (seed {SEED}), every "
        "run of each through the product again with its vessel profile. The rules re-read the "
        "saved profiles of the chosen heartbeat (the product's FFR is the median over up to "
        "five). Pixel size unknown (0.30 mm assumed). No rule here changes the product.", "",
        f"{s['measured']} of {s['runs']} runs measured; {s['sides']} sides ({s['zero_sides']} "
        f"SYNTAX 0, {s['positive_sides']} SYNTAX > 0). The product's rule on the chosen "
        f"heartbeat agrees with its cached flag on {s['product_flag_agrees']:.0%} of runs.", "",
        "## Reading rules", "",
        "| Rule | SYNTAX-0 sides flagged | SYNTAX > 0 sides flagged | SYNTAX-0 runs FFR ≤ 0.80 "
        "(median) | AUC, lowest FFR | AUC, deepest %DS |",
        "|---|---:|---:|---:|---:|---:|",
    ]  # fmt: skip
    for x in s["rules"]:
        med = "—" if x["zero_runs_median"] is None else f"{x['zero_runs_median']:.2f}"
        auc = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {x['name']} | {_pct(x['false_alarm'])} | {_pct(x['sensitivity'])} | "
                     f"{_pct(x['zero_runs_le_080'])} ({med}) | {auc(x['auc_ffr'])} | "
                     f"{auc(x['auc_ds'])} |")  # fmt: skip
    lines += ["", "## Marked lesions (≥ 50%) on SYNTAX-0 runs, by the reference diameter at the "
              "narrowest point", "", "| Reference | Lesions |", "|---|---:|"]  # fmt: skip
    for b in s["zero_marks_by_ref"]:
        hi = "" if b["hi"] > 50 else f"–{b['hi']:g}"
        label = f"≥ {b['lo']:g} mm" if b["hi"] > 50 else f"{b['lo']:g}{hi} mm"
        share = b["n"] / s["zero_marks"] if s["zero_marks"] else 0.0
        lines.append(f"| {label} | {b['n']} ({share:.0%}) |")
    lines += [
        "", f"{s['zero_marks']} marked lesions on {s['zero_runs']} runs of SYNTAX-0 sides.", "",
        "**Limits:** exploratory, on data already reported once; SYNTAX is per side and a run's "
        "automatic path may not follow the graded vessel; the pixel size is assumed, so the "
        "reference diameter's mm limits are approximate. A rule chosen here needs a "
        "pre-registered test on unseen data.", "",
        f"Product: `{json.dumps(product)}`. Commit `{git.get('commit')}` (dirty: "
        f"{git.get('dirty')}).",
    ]  # fmt: skip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
