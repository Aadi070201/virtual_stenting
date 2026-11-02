"""Evaluation of the learned lesion reading (P4i): development decision, then the test. Torch-free.

Two stages, in this order (docs/PLAN.md, P4i):

1. ``dev``: CADICA development patients (out-of-fold predictions of the training run) and
   ARCADE val; the pre-registered adoption rule; ``reports/lesion_reading_dev.{md,json}``.
2. ``test``: only once the development decision is committed: CADICA test patients, ARCADE
   test and Kemerovo (external); ``reports/lesion_reading.{md,json}``.

A lesion's value is the median over its boxes (keyframes), as in P4b. "Edges" is the product's
measured %DS at the same point (the lumen the reader would correct).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.data import lesion_data
from vstent.validation.stenosis_grading import ORDINAL, grading_stats

DEV_REPORT = "lesion_reading_dev"
REPORT = "lesion_reading"
EDGES_DEV = {"kappa": 0.214, "auc_ge70": 0.818}  # P4h development values (pre-registered)
SIGNIFICANT = 50.0
FP_MARGIN = 10.0  # points
TARGETS = {"kappa": 0.40, "p70_90": 60.0, "p90_98": 70.0, "p0_20": 30.0, "arcade_sens": 0.70,
           "arcade_fp": 0.20, "kemerovo_sens": 0.70}  # fmt: skip


def lesion_grades(samples: list[dict[str, Any]], values: np.ndarray) -> list[dict[str, Any]]:
    """Per lesion (track): the median of ``values`` over its boxes."""
    by: dict[str, dict[str, Any]] = {}
    for s, v in zip(samples, values, strict=True):
        t = by.setdefault(s["track"], {"track": s["track"], "patient": s["patient"],
                                       "category": s.get("category", ""), "vals": []})  # fmt: skip
        t["vals"].append(float(v))
    return [{**{k: v for k, v in t.items() if k != "vals"}, "ds": float(np.median(t["vals"]))}
            for t in by.values()]  # fmt: skip


def _stats(samples, values) -> dict[str, Any]:
    s = grading_stats(lesion_grades(samples, values), n_boot=1000)
    return {k: s.get(k) for k in ("n_lesions", "n_measured", "kappa", "auc_ge70", "auc_ge50",
                                  "median_ds", "n_by_category", "bootstrap")}  # fmt: skip


def detection(samples, values) -> dict[str, Any]:
    """Shares read ≥ 50% among stenosis boxes (sensitivity) and distant candidates."""
    v = np.asarray(values, float)
    sig = np.array([s["sig_target"] for s in samples])
    pos, neg = v[sig == 1], v[sig == 0]
    return {"n_pos": int(pos.size), "n_neg": int(neg.size),
            "sensitivity": float(np.mean(pos >= SIGNIFICANT)) if pos.size else None,
            "false_positive": float(np.mean(neg >= SIGNIFICANT)) if neg.size else None}  # fmt: skip


def _predict(runner, name: str, data_dir) -> tuple[list[dict], np.ndarray, np.ndarray]:
    p, s, m = lesion_data.load(name, data_dir)
    return m, np.asarray([x["measured_ds"] for x in m], float), runner.predict(p, s)


def dev_stage(run_dir: str | Path, runner, data_dir=lesion_data.OUT_DIR) -> dict[str, Any]:
    """Out-of-fold CADICA development results, ARCADE val, and the pre-registered decision."""
    oof = json.loads((Path(run_dir) / "oof.json").read_text(encoding="utf-8"))["samples"]
    edges = np.array([s["measured_ds"] for s in oof], float)
    learned = np.array([s["oof_ds"] for s in oof], float)
    cad = {"edges": _stats(oof, edges), "learned": _stats(oof, learned)}
    m, e, lp = _predict(runner, "arcade_val", data_dir)
    arc = {"edges": detection(m, e), "learned": detection(m, lp)}
    le, ee = cad["learned"], cad["edges"]
    # the stricter of the pre-registered edge values and the edges on these very samples
    bar = {k: max(EDGES_DEV[k], ee[k] if ee[k] is not None else -np.inf)
           for k in ("kappa", "auc_ge70")}  # fmt: skip
    al, ae = arc["learned"], arc["edges"]

    def gt(x, y) -> bool:  # a statistic that could not be computed fails its condition
        return x is not None and y is not None and x > y

    rule = {
        "kappa_higher": gt(le.get("kappa"), bar["kappa"]),
        "auc_ge70_higher": gt(le.get("auc_ge70"), bar["auc_ge70"]),
        "median_0_20_le_30": (le.get("median_ds") or {}).get("p0_20") is not None
        and le["median_ds"]["p0_20"] <= TARGETS["p0_20"],
        "arcade_sensitivity_higher": gt(al["sensitivity"], ae["sensitivity"]),
        "arcade_fp_within_margin": al["false_positive"] is not None
        and ae["false_positive"] is not None
        and al["false_positive"] <= ae["false_positive"] + FP_MARGIN / 100.0,
    }  # fmt: skip
    return {"cadica_dev": cad, "edges_dev_recomputed": {"kappa": ee["kappa"],
                                                        "auc_ge70": ee["auc_ge70"]},
            "arcade_val": arc, "rule": rule, "adopted": all(rule.values())}  # fmt: skip


def test_stage(runner, data_dir=lesion_data.OUT_DIR) -> dict[str, Any]:
    """CADICA test patients, ARCADE test and Kemerovo (once)."""
    m, e, lp = _predict(runner, "cadica", data_dir)
    test = np.array([x["split"] == "test" for x in m])
    mt = [x for x, t in zip(m, test, strict=True) if t]
    out = {"cadica_test": {"edges": _stats(mt, e[test]), "learned": _stats(mt, lp[test])}}
    m, e, lp = _predict(runner, "arcade_test", data_dir)
    out["arcade_test"] = {"edges": detection(m, e), "learned": detection(m, lp)}
    kem = Path(data_dir) / "kemerovo.npz"
    if kem.exists():
        m, e, lp = _predict(runner, "kemerovo", data_dir)
        groups = {"edges": lesion_grades(m, e), "learned": lesion_grades(m, lp)}
        out["kemerovo"] = {k: _kemerovo(g) for k, g in groups.items()}
    free = Path(data_dir) / "cadica_free.npz"
    if free.exists():  # P4k-A: the CADICA test patients' lesion-free runs (never used before)
        m, e, lp = _predict(runner, "cadica_free", data_dir)
        test = np.array([x["split"] == "test" for x in m], bool)
        out["cadica_free_test"] = {"n": int(test.sum()), **{
            k: float(np.mean(v[test] >= SIGNIFICANT)) if test.any() else None
            for k, v in (("edges", e), ("learned", lp))}}  # fmt: skip
    return out


def _kemerovo(g: list[dict[str, Any]]) -> dict[str, Any]:
    ds = np.array([x["ds"] for x in g], float)
    return {"n_lesions": len(g), "read_ge50": float(np.mean(ds >= SIGNIFICANT)),
            "read_ge70": float(np.mean(ds >= 70.0)), "median_ds": float(np.median(ds))}  # fmt: skip


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _num(x, fmt="{:.2f}") -> str:
    return "–" if x is None else fmt.format(x)


def _grading_rows(d: dict[str, Any]) -> list[str]:
    rows = ["| Method | Lesions | Weighted kappa | AUC ≥ 70% | " +
            " | ".join(f"Median %DS {c}" for c in ORDINAL) + " |",
            "|---|---:|---:|---:|" + "---:|" * len(ORDINAL)]  # fmt: skip
    for name, s in d.items():
        med = s.get("median_ds") or {}
        rows.append(f"| {name} | {s['n_measured']} | {_num(s['kappa'])} | "
                    f"{_num(s['auc_ge70'])} | " +
                    " | ".join(_num(med.get(c), "{:.0f}") for c in ORDINAL) + " |")  # fmt: skip
    return rows


def write_report(report_dir: str | Path, name: str, title: str, sections: list[str],
                 data: dict[str, Any], git: dict, model: str) -> None:  # fmt: skip
    prov = {**git, "vstent_version": __version__, "model": model,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    lines = [f"# {title}", "", f"> **{DISCLAIMER}** Agreement with experts' visual grades and "
             "boxes on real angiograms, not accuracy against wire FFR.", "",
             "Generated by `vstent lesion-report`. Do not edit by hand.", "", *sections, "",
             f"Model `{model}`. Commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), "
             f"created {prov['created_utc']}.", ""]  # fmt: skip
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (report_dir / f"{name}.json").write_text(json.dumps({**data, "provenance": prov}, indent=1,
                                                        default=float),
                                             encoding="utf-8", newline="\n")  # fmt: skip


def dev_sections(d: dict[str, Any]) -> list[str]:
    a = d["arcade_val"]
    return [
        "## Development (CADICA development patients, out-of-fold; ARCADE val)", "",
        *_grading_rows(d["cadica_dev"]), "",
        f"ARCADE val: stenosis boxes read ≥ 50%: edges {_pct(a['edges']['sensitivity'])}, "
        f"learned {_pct(a['learned']['sensitivity'])} (of {a['learned']['n_pos']}); distant "
        f"candidates read ≥ 50%: edges {_pct(a['edges']['false_positive'])}, learned "
        f"{_pct(a['learned']['false_positive'])} (of {a['learned']['n_neg']}).", "",
        "**Pre-registered rule** (docs/PLAN.md, P4i):", "",
        *(f"- {k.replace('_', ' ')}: {'yes' if v else '**no**'}" for k, v in d["rule"].items()),
        "", f"**Decision: {'adopted' if d['adopted'] else 'not adopted'}.**",
    ]  # fmt: skip


def test_sections(t: dict[str, Any]) -> list[str]:
    a, out = t["arcade_test"], []
    out += ["## Test (once)", "", "### CADICA test patients", "", *_grading_rows(t["cadica_test"]),
            "", "### ARCADE test", "",
            f"Stenosis boxes read ≥ 50%: edges {_pct(a['edges']['sensitivity'])}, learned "
            f"{_pct(a['learned']['sensitivity'])} (of {a['learned']['n_pos']}; target ≥ 70%). "
            f"Distant candidates read ≥ 50%: edges {_pct(a['edges']['false_positive'])}, "
            f"learned {_pct(a['learned']['false_positive'])} (of {a['learned']['n_neg']}; "
            "target ≤ 20%).", ""]  # fmt: skip
    if "kemerovo" in t:
        k = t["kemerovo"]
        out += ["### Kemerovo (external; every lesion significant)", "",
                "| Method | Lesions | Read ≥ 50% | Read ≥ 70% | Median %DS |",
                "|---|---:|---:|---:|---:|",
                *(f"| {n} | {v['n_lesions']} | {_pct(v['read_ge50'])} | {_pct(v['read_ge70'])} | "
                  f"{v['median_ds']:.0f} |" for n, v in k.items()), ""]  # fmt: skip
    if "cadica_free_test" in t:
        f = t["cadica_free_test"]
        out += ["### CADICA test patients' lesion-free runs (fresh; P4k-A)", "",
                f"Candidates read ≥ 50% (target ≤ 20%): edges {_pct(f['edges'])}, learned "
                f"{_pct(f['learned'])} (of {f['n']}).", ""]  # fmt: skip
    return out
