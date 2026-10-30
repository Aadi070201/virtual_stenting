"""MVP report (``vstent mvp-report``) → ``reports/mvp_report.md``.

Assembled from the generated report files only: every number below is read from a ``*.json``
written by another ``vstent`` command (no hand-typed numbers). Missing reports are listed as
missing. The Phase 2 list is read from ``docs/PLAN.md``.
"""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.data.card import TARGET_BELOW_CUTOFF, TARGET_GREY_ZONE

LIMITATIONS = (
    "Research prototype: not a medical device, not validated, never for decisions about real "
    "patients.",
    "Physics: steady, Newtonian, single-vessel reduced-order model (Poiseuille + expansion "
    "loss) without side branches; FFR is very sensitive to the assumed microvascular "
    "resistance and hyperemic velocity.",
    "The surrogate is trained and evaluated on synthetic vessels with labels from the same "
    "reduced-order model: its accuracy is agreement with that model, not with invasive FFR. "
    "It is not faster than the reduced-order solver; it validates the training → ONNX → app "
    "pipeline for Phase 2 3D-CFD labels.",
    "Segmentation: one public dataset (ARCADE) whose masks cover only the annotated SYNTAX "
    "segments; no external validation.",
    "QCA: single 2D projection (foreshortening, overlap), uncalibrated pixel size unless "
    "given, crossing vessels can look like narrowings; automatic paths run through branches.",
    "Planning: the stent only widens the lumen to its radius (no recoil, malapposition or "
    "side-branch effects); two-stent search is heuristic; overlapping stents are not modelled.",
    "No patient outcome, invasive FFR or clinical data were used anywhere.",
)


def _load(reports: Path, name: str) -> dict[str, Any] | None:
    path = reports / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _row(milestone, area, metric, value, target, passed):
    return {
        "milestone": milestone,
        "area": area,
        "metric": metric,
        "value": value,
        "target": target,
        "passed": passed,
    }


def _prov(d: dict | None, *path: str) -> dict | None:
    for key in path:
        if d is None:
            return None
        d = d.get(key)
    return d


def collect(reports: Path) -> dict[str, Any]:
    """Targets, facts and provenance from every report file."""
    rows, facts, prov, missing = [], [], [], []

    def source(name: str, title: str, *prov_path: str) -> dict | None:
        d = _load(reports, name)
        if d is None:
            missing.append(name)
            return None
        p = _prov(d, *prov_path) or {}
        prov.append(
            {
                "report": name,
                "title": title,
                "commit": p.get("git_commit"),
                "dirty": p.get("git_dirty"),
                "created": p.get("created_utc") or d.get("created_utc"),
            }
        )
        return d

    d = source("physics_checks.json", "Physics checks", "provenance")
    if d:
        for c in d["checks"]:
            rows.append(
                _row("M1", "Physics", c["check"], f"{c['value']:.4f}", c["target"], c["passed"])
            )
        b = d["benchmark"]
        rows.append(
            _row(
                "M1",
                "Physics",
                f"Batch solver throughput ({b['cpu']})",
                f"{b['profiles_per_second']:,.0f} profiles/s",
                f"≥ {b['target_profiles_per_second']:,}",
                b["passed"],
            )
        )
    d = source("dataset_card.json", "Dataset card", "provenance")
    if d:
        for t in d["targets"]:
            rows.append(
                _row(
                    "M2",
                    "Synthetic data",
                    f"{t['split']}: FFR < 0.80 / grey zone",
                    f"{t['frac_ffr_below_cutoff']:.1%} / {t['frac_grey_zone']:.1%}",
                    f"≥ {TARGET_BELOW_CUTOFF:.0%} / ≥ {TARGET_GREY_ZONE:.0%}",
                    t["passed"],
                )
            )
        facts.append(
            f"Synthetic dataset: {d['n_total']:,} vessels generated in "
            f"{d['generation_seconds']:.1f} s."
        )
    d = source("surrogate_eval.json", "Surrogate evaluation", "provenance")
    if d:
        for t in d["targets"]:
            rows.append(
                _row(
                    "M3",
                    "Surrogate (test split)",
                    t["metric"],
                    t["value"],
                    t["target"],
                    t["passed"],
                )
            )
        sp = d["splits"]
        facts.append(
            f"Surrogate distal-FFR MAE: test {sp['test']['surrogate']['mae']:.4f} vs gradient-"
            f"boosting baseline {sp['test']['baseline']['mae']:.4f}; out-of-distribution "
            f"{sp['ood']['surrogate']['mae']:.4f} (no target)."
        )
    d = source("onnx_benchmark.json", "ONNX benchmark", "provenance", "report")
    if d:
        for t in d["targets"]:
            rows.append(
                _row("M4", "Surrogate ONNX", t["metric"], t["value"], t["target"], t["passed"])
            )
        facts.append(
            f"For context, the reduced-order solver computes {d['solver']['batch_size']:,} "
            f"vessels in {d['solver']['median_ms']:.1f} ms, the ONNX surrogate in "
            f"{d['timings']['default']['batch_end_to_end']['median_ms']:.0f} ms: no speed-up "
            "is claimed."
        )
        q = d["int8"]
        facts.append(
            f"INT8 quantisation: distal MAE {q['distal_mae_int8']:.4f} vs "
            f"{q['distal_mae_fp32']:.4f}, speed-up {q['speedup']:.2f}× → "
            f"{'kept' if q['kept'] else 'not kept'}."
        )
    d = source("seg_eval.json", "Segmentation evaluation", "provenance", "eval")
    if d:
        for t in d["targets"]:
            rows.append(
                _row("M5", "Segmentation", t["metric"], t["value"], t["target"], t["passed"])
            )
    d = source("seg_onnx.json", "Segmentation ONNX", "export")
    if d:
        p = d["parity"]
        rows.append(
            _row(
                "M5",
                "Segmentation ONNX",
                "Parity max abs(prob torch − onnx)",
                f"{p['max_abs_diff']:.1e}",
                f"< {p['tolerance']:g}",
                p["passed"],
            )
        )
    d = source("qca_synthetic.json", "QCA synthetic validation", "provenance")
    if d:
        for t in d["targets"]:
            rows.append(
                _row(
                    "M5", "QCA (synthetic tubes)", t["metric"], t["value"], t["target"], t["passed"]
                )
            )
    d = source("qca_gallery.json", "QCA gallery", "provenance")
    if d:
        s = d["summary"]
        facts.append(
            f"ARCADE gallery (qualitative): {s['n_analysed']}/{s['n_images']} images analysed "
            f"end-to-end; point of maximum DS inside an annotated stenosis box in "
            f"{s['n_max_ds_in_box_given_crossing']}/{s['n_path_crosses_box']}; "
            f"{s['n_out_of_surrogate_range']} outside the surrogate's training range."
        )
    d = source("planner_benchmark.json", "Planner benchmark", "provenance")
    if d:
        for t in d["targets"]:
            rows.append(_row("M6", "Planner", t["metric"], t["value"], t["target"], t["passed"]))
        s = d["study"]["summary"]
        facts.append(
            f"Planner on {s['n_vessels']} synthetic vessels: target reached "
            f"{s['sol_reaches_target']} (solver) / {s['sur_reaches_target']} (surrogate); "
            "solver confirms "
            f"{s['sur_plan_confirmed_by_solver']}/{s['sur_reaches_target']} surrogate plans; same "
            f"total length {s['same_length_both_meet']}/{s['n_both_meet']}."
        )
    d = source("app_latency.json", "App latency", "provenance")
    if d:
        modes = {k: v for k, v in d["results"].items() if v}
        worst = max(v["server_max_ms"] for v in modes.values())
        rows.append(
            _row(
                "M7",
                "App",
                "Stent-control update, server time (max over modes)",
                f"{worst:.0f} ms",
                f"< {d['target_ms']:g} ms",
                d["passed"],
            )
        )
        facts.append(
            "App update medians: "
            + ", ".join(f"{k} {v['server_median_ms']:.0f} ms" for k, v in modes.items())
            + "."
        )
    d = source("demo/demo.json", "Demo", "software")
    if d:
        facts.append(
            f"Demo (synthetic tandem lesions, {d['demo']['planning_backend']} backend): distal "
            f"FFR {d['pre_pci']['ffr_distal_solver']:.3f} → "
            f"{d['post_pci']['ffr_distal_solver']:.3f} (solver) with {d['demo']['best_plan']}; "
            f"planning {d['demo']['timings_s']['planning_s']:.2f} s."
        )
    return {"rows": rows, "facts": facts, "provenance": prov, "missing": missing}


def phase2_items(plan_path: Path = Path("docs/PLAN.md")) -> list[str]:
    """Phase 2 milestones of the plan (``## P<n>: title``) with their status from the task
    checkboxes: done (all ticked), in progress (some ticked) or not started."""
    if not plan_path.exists():
        return []
    text = plan_path.read_text(encoding="utf-8")
    items = []
    for m in re.finditer(r"^## (P\d+: .*?)$(.*?)(?=^## |\Z)", text, flags=re.M | re.S):
        done = len(re.findall(r"^- \[x\]", m.group(2), flags=re.M))
        todo = len(re.findall(r"^- \[ \]", m.group(2), flags=re.M))
        status = "done" if done and not todo else "in progress" if done else "not started"
        items.append(f"{m.group(1).strip()} ({status})")
    return items


def write_mvp_report(
    report_dir: str | Path = "reports", *, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Assemble ``mvp_report.md`` (and ``.json``) from the other reports."""
    log = log or (lambda _msg: None)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    report_dir = Path(report_dir)
    data = collect(report_dir)
    data["phase2"] = phase2_items()
    data["provenance_report"] = provenance
    data["all_targets_met"] = all(r["passed"] for r in data["rows"]) and not data["missing"]
    (report_dir / "mvp_report.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8", newline="\n"
    )
    (report_dir / "mvp_report.md").write_text(_markdown(data), encoding="utf-8", newline="\n")
    n_pass = sum(r["passed"] for r in data["rows"])
    log(f"{n_pass}/{len(data['rows'])} targets met; missing reports: {data['missing'] or 'none'}")
    return data


def _markdown(data: dict[str, Any]) -> str:
    rows, prov = data["rows"], data["provenance_report"]
    n_pass = sum(r["passed"] for r in rows)
    lines = [
        "# vstent MVP report",
        "",
        f"> **{DISCLAIMER}** This research prototype is not a medical device, is not validated "
        "and must never be used to make decisions about real patients.",
        "",
        "Assembled by `vstent mvp-report` from the generated report files in `reports/` (no "
        "hand-typed numbers). Each line links back to its source report; the provenance table "
        "lists the commit each one was generated at.",
        "",
        "## What the prototype does",
        "",
        "Angiogram → vessel segmentation (U-Net, ONNX) → centreline and 2D QCA → FFR pullback "
        "from a reduced-order physics model and a learned surrogate (ONNX) → virtual stent with "
        "instant post-PCI FFR → plan optimiser (shortest generic-catalogue plan reaching the "
        "target FFR) → Streamlit app and case report.",
        "",
        f"## Acceptance targets: {n_pass} of {len(rows)} met",
        "",
        "| Milestone | Area | Metric | Value | Target | Met |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['milestone']} | {r['area']} | {r['metric']} | {r['value']} "
            f"| {r['target']} | {'yes' if r['passed'] else '**NO**'} |"
        )
    if data["missing"]:
        lines += ["", f"**Missing reports:** {', '.join(data['missing'])}."]
    lines += ["", "## Further results (no target)", ""]
    lines += [f"- {f}" for f in data["facts"]]
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    lines += ["", "## Phase 2 (status from docs/PLAN.md)", ""]
    lines += [f"- {item}" for item in data["phase2"]] or ["- (plan not found)"]
    lines += [
        "",
        "## Sources and provenance",
        "",
        "| Report | Generated at commit | Dirty | Created (UTC) |",
        "|---|---|---|---|",
    ]
    for p in data["provenance"]:
        md = p["report"].replace(".json", ".md")
        commit = (p["commit"] or "?")[:10]
        lines.append(f"| [{p['title']}]({md}) | `{commit}` | {p['dirty']} | {p['created']} |")
    lines += [
        "",
        f"This summary: commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), vstent "
        f"{prov['vstent_version']}, {prov['created_utc']}.",
        "",
    ]
    return "\n".join(lines)
