"""End-to-end synthetic demo (``vstent demo``) → ``reports/demo/{demo.md, demo.json, demo.png}``.

The tandem-lesion preset goes through the whole chain: profile → pre-PCI FFR (solver and, if an
exported surrogate exists, the ONNX surrogate) → stent plan optimisation → the best plan applied
→ post-PCI FFR → report. Without an exported surrogate (e.g. on a fresh clone before training)
the solver is used for planning and the report says so.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from vstent import DISCLAIMER
from vstent.app import core
from vstent.inference.models import SURROGATE_MODELS, latest_model_dir
from vstent.inference.onnx_runtime import ONNX_FILE, SurrogateRunner
from vstent.planning.optimizer import plan_stents
from vstent.plotting import (
    BAND,
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    new_figure,
    save_figure,
    style_axes,
)
from vstent.units import m_to_mm

DEMO_PRESET = core.PRESETS[1]  # tandem lesions


def _figure(state: core.CaseState, stented, plans, path: Path) -> None:
    fig: Figure = new_figure(
        12.0,
        4.0,
        "vstent demo: synthetic tandem lesions",
        f"Best plan: {_plan_text(plans.plans[0])}; distal FFR {state.pre.distal_solver:.2f} → "
        f"{state.post.distal_solver:.2f} (solver)",
    )
    ax_d, ax_f, ax_p = fig.subplots(1, 3, gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.17)
    p, s = state.profile, m_to_mm(state.profile.s)
    style_axes(ax_d)
    for les in p.lesions():
        ax_d.axvspan(m_to_mm(les.s_start), m_to_mm(les.s_end), color=BAND, lw=0, zorder=0.5)
    for st in state.stents:
        ax_d.axvspan(m_to_mm(st.s_start), m_to_mm(st.s_end), color=SERIES_2, alpha=0.15, lw=0)
    ax_d.plot(s, m_to_mm(p.diameter), color=SERIES_1, lw=1.8, label="Lumen", zorder=3)
    ax_d.plot(s, m_to_mm(stented.diameter), color=SERIES_2, lw=1.4, label="After stents", zorder=2)
    ax_d.plot(
        s, m_to_mm(2 * p.reference_radius()), color=INK_MUTED, lw=1, ls="--", label="Reference"
    )
    ax_d.set_ylim(bottom=0)
    ax_d.set_xlabel("Distance (mm)", fontsize=8, color=INK_SECONDARY)
    ax_d.set_ylabel("Diameter (mm)", fontsize=8, color=INK_SECONDARY)
    ax_d.legend(frameon=False, fontsize=7, loc="lower left")
    style_axes(ax_f)
    for curves, color, label in ((state.pre, INK_SECONDARY, "pre"), (state.post, SERIES_2, "post")):
        ax_f.plot(m_to_mm(curves.s), curves.solver, color=color, lw=1.8, label=f"Solver {label}")
        if curves.surrogate is not None:
            ax_f.plot(
                m_to_mm(curves.s_surrogate),
                curves.surrogate,
                color=SERIES_1 if label == "pre" else color,
                lw=1.4,
                ls="--",
                label=f"Surrogate {label}",
            )
    ax_f.axhline(0.80, color=INK_MUTED, lw=0.8, ls=":")
    ax_f.set_xlabel("Distance (mm)", fontsize=8, color=INK_SECONDARY)
    ax_f.set_ylabel("FFR", fontsize=8, color=INK_SECONDARY)
    ax_f.legend(frameon=False, fontsize=7, loc="lower left")
    style_axes(ax_p)
    ax_p.plot(
        [m_to_mm(q.total_length) for q in plans.pareto],
        [q.ffr for q in plans.pareto],
        color=SERIES_1,
        marker="o",
        ms=3,
        lw=1.5,
    )
    ax_p.axhline(plans.target, color=INK_MUTED, lw=0.8, ls=":")
    ax_p.set_xlabel("Total stent length (mm)", fontsize=8, color=INK_SECONDARY)
    ax_p.set_ylabel(f"Post-PCI distal FFR ({plans.backend})", fontsize=8, color=INK_SECONDARY)
    ax_p.set_title("Pareto front", loc="left", fontsize=8, color=INK_SECONDARY)
    save_figure(fig, path)


def _plan_text(plan) -> str:
    return " + ".join(
        f"{m_to_mm(st.diameter):.2f} × {m_to_mm(st.length):.0f} mm at {m_to_mm(st.s_start):.1f} mm"
        for st in plan.stents
    )


def run_demo(
    out_dir: str | Path = "reports/demo", *, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Run the demo and write its report; returns the JSON content."""
    log = log or (lambda _msg: None)
    out_dir = Path(out_dir)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    params, planner, catalog = core.load_settings()
    sur_dir = latest_model_dir(SURROGATE_MODELS, ONNX_FILE)
    runner = SurrogateRunner.from_dir(sur_dir) if sur_dir else None
    timings["load_s"] = time.perf_counter() - t0
    backend = "surrogate" if runner is not None else "solver"
    log(f"Planning backend: {backend}" + ("" if runner else " (no exported surrogate found)"))

    case = core.synthetic_case(DEMO_PRESET, 0, params)
    t0 = time.perf_counter()
    pre = core.evaluate(case.profile, params, case.pa, case.pv, runner=runner)
    timings["pre_pci_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    plans = plan_stents(
        case.profile,
        catalog=catalog,
        params=params,
        runner=runner,
        config=planner.model_copy(update={"backend": backend}),
        pa=case.pa,
        pv=case.pv,
    )
    timings["planning_s"] = time.perf_counter() - t0
    best = plans.plans[0]
    t0 = time.perf_counter()
    stented, post = core.stent_update(
        case.profile, list(best.stents), pre, params, case.pa, case.pv, runner
    )
    timings["post_pci_s"] = time.perf_counter() - t0
    if runner is not None:
        runner.close()

    t0 = time.perf_counter()
    ucfg, qca_error = core.load_uncertainty()
    mc = core.input_uncertainty(
        case.profile, params, case.pa, case.pv, core.Measurement(), (ucfg, qca_error)
    )
    timings["uncertainty_s"] = time.perf_counter() - t0
    state = core.CaseState(
        "synthetic demo",
        params,
        case.pa,
        case.pv,
        case.profile,
        pre,
        list(best.stents),
        post,
        description=case.description,
        plans=plans,
        uncertainty=mc,
        pre_assessment=core.assess(pre, mc.pre_samples, ucfg.pre_pci_threshold, ucfg.level),
        post_assessment=core.assess(
            post, mc.post_samples(best.stents), planner.target_ffr, ucfg.level
        ),
        plan_assessments=core.plan_assessments(plans, mc, ucfg.level),
    )
    models = core.Models(runner, None, sur_dir, None)
    report = core.case_report(state, models)
    report["demo"] = {
        "preset": DEMO_PRESET,
        "planning_backend": backend,
        "best_plan": _plan_text(best),
        "n_candidates": plans.n_candidates,
        "n_feasible": plans.n_feasible,
        "n_pairs": plans.n_pairs,
        "timings_s": timings,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "demo.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    _figure(state, stented, plans, out_dir / "demo.png")
    (out_dir / "demo.md").write_text(_markdown(report, plans), encoding="utf-8", newline="\n")
    log(
        f"Pre-PCI FFR {pre.distal_solver:.3f} -> post-PCI {post.distal_solver:.3f} with "
        f"{_plan_text(best)}"
    )
    return report


def _markdown(report: dict[str, Any], plans) -> str:
    d, pre, post = report["demo"], report["pre_pci"], report["post_pci"]
    sw = report["software"]
    t = d["timings_s"]
    sur = report["models"]["surrogate"]

    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"

    lines = [
        "# vstent demo: synthetic tandem lesions",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent demo`. Do not edit by hand.",
        "",
        f"**Vessel:** {report['input']['description']} (synthetic; Pa "
        f"{report['assumptions']['pa_mmhg']:g} mmHg, Pv {report['assumptions']['pv_mmhg']:g} "
        "mmHg).",
        "",
        f"**Planning backend:** {d['planning_backend']}"
        + (
            ""
            if d["planning_backend"] == "surrogate"
            else " (no exported surrogate was found; the reduced-order solver planned the stents)"
        )
        + ".",
        "",
        "| Step | Result | Time |",
        "|---|---|---|",
        f"| Pre-PCI distal FFR (solver / surrogate) | {fmt(pre['ffr_distal_solver'])} / "
        f"{fmt(pre['ffr_distal_surrogate'])} | {t['pre_pci_s'] * 1e3:.0f} ms |",
        f"| Stent plan search | {d['n_feasible']} feasible of {d['n_candidates']} single-stent "
        f"candidates, {d['n_pairs']} pairs; {plans.message} | {t['planning_s'] * 1e3:.0f} ms |",
        f"| Best plan | {d['best_plan']} | |",
        f"| Post-PCI distal FFR (solver / surrogate) | {fmt(post['ffr_distal_solver'])} / "
        f"{fmt(post['ffr_distal_surrogate'])} | {t['post_pci_s'] * 1e3:.0f} ms |",
        "",
        "| Plan | Stents | Total length | Post FFR (planning backend) | Solver check |",
        "|---|---|---|---|---|",
    ]
    for k, pl in enumerate(report["plans"]["top"], 1):
        stents = " + ".join(
            f"{s['diameter_mm']:.2f} × {s['length_mm']:.0f} mm at {s['start_mm']:.1f} mm"
            for s in pl["stents"]
        )
        check = "—" if pl["ffr_solver_check"] is None else f"{pl['ffr_solver_check']:.3f}"
        lines.append(
            f"| {k} | {stents} | {pl['total_length_mm']:.0f} mm | {pl['ffr']:.3f} | {check} |"
        )
    u = report.get("uncertainty")
    if u is not None:

        def iv(x):
            return "n/a" if x is None else f"{x[0]:.3f}–{x[1]:.3f}"

        level = u["pre_pci"]["level"]
        lines += [
            "",
            f"**Uncertainty ({level:.0%} intervals).** Inputs varied: "
            f"{', '.join(u['input_assumptions_varied'])} ({u['input_samples']} Monte Carlo draws "
            f"through the solver; {t['uncertainty_s'] * 1e3:.0f} ms).",
            "",
            "| Distal FFR | Input interval (solver) | Model interval (surrogate) | Threshold "
            "| Grey zone |",
            "|---|---|---|---|---|",
        ]
        for label, a in (("Pre-PCI", u["pre_pci"]), ("Post-PCI, best plan", u["post_pci"])):
            lines.append(
                f"| {label} | {iv(a['input_interval'])} | {iv(a['model_interval'])} "
                f"| {a['threshold']:.2f} | {'yes' if a['grey_zone'] else 'no'} |"
            )
        lines += ["", u["note"]]
    lines += [
        "",
        "![Demo](demo.png)",
        "",
        "Post-PCI FFR keeps the pre-PCI microvascular resistance, Pa and Pv (only the lumen "
        "changes). Full case report with all assumptions: `demo.json`.",
        "",
        f"Surrogate model: `{sur['dir']}`; commit `{sw['git_commit']}` (dirty: "
        f"{sw['git_dirty']}), vstent {sw['vstent']}, {report['created_utc']}.",
        "",
    ]
    return "\n".join(lines)
