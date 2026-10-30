"""Tree physics checks and throughput (``vstent tree-report``, P2) → ``reports/tree_physics.*``.

Every check runs the tree solver (:mod:`vstent.physics.tree`) on generated data or on
controlled examples; nothing is hand-typed.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.data.synth import SamplingConfig, load_split, sample_vessel
from vstent.data.synth_tree import LESION_KINDS, tree_from_arrays
from vstent.physics.geometry import VesselProfile, cosine_lesion_profile, cosine_narrowing
from vstent.physics.solver import PhysicsParams, solve_ffr, solve_ffr_batch
from vstent.physics.tree import VesselTree, solve_tree, solve_tree_batch
from vstent.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    new_figure,
    save_figure,
    style_axes,
)
from vstent.sysinfo import cpu_name
from vstent.units import m_to_mm

TARGET_TREES_PER_S = 1000.0
N_TREES = 1000
FIGURE = "tree_example.png"


def _load_trees(data_dir: Path, n: int) -> tuple[list[VesselTree], dict[str, np.ndarray]]:
    arrays = load_split(data_dir, "test")
    n = min(n, arrays["r"].shape[0])
    arrays = {k: v[:n] for k, v in arrays.items()}
    return [tree_from_arrays(arrays, i) for i in range(n)], arrays


def _solve(trees, arrays, params):
    return solve_tree_batch(trees, params, pa=arrays["pa"], pv=arrays["pv"], v_hyp=arrays["v_hyp"])


def check_single_branch(params: PhysicsParams, n: int = 200) -> dict[str, Any]:
    """A one-branch tree against the M1 solver on sampled single vessels."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n):
        v = sample_vessel(rng, SamplingConfig())
        m1 = solve_ffr_batch(
            v.profile.r, v.profile.ds, v.profile.r_ref, params, pa=v.pa, pv=v.pv, v_hyp=v.v_hyp
        )
        res = solve_tree(VesselTree.single(v.profile), params, pa=v.pa, pv=v.pv, v_hyp=v.v_hyp)
        worst = max(worst, float(np.abs(res.ffr[0] - m1.ffr[0]).max()))
    return {"n": n, "max_abs_ffr_diff": worst, "tolerance": 1e-10, "passed": bool(worst <= 1e-10)}


def check_conservation(trees, results) -> dict[str, Any]:
    """Junction mass balance, pressure continuity and the outlet equations on every tree."""
    mass = pressure = outlet = 0.0
    for tree, res in zip(trees, results, strict=True):
        for k in range(tree.n_branches):
            kids = tree.children(k)
            balance = res.q_outlet[k] + sum(res.q_inlet[c] for c in kids)
            mass = max(mass, abs(res.q_inlet[k] - balance) / res.q_inlet[k])
            for c in kids:
                gap = abs(res.pressure[c][0] - res.pressure[k][tree.attach[c]]) / res.pa
                pressure = max(pressure, gap)
            lhs = res.pressure[k][-1] - res.pv
            outlet = max(outlet, abs(lhs - res.r_outlet[k] * res.q_outlet[k]) / lhs)
    return {
        "n": len(trees),
        "max_rel_mass_residual": mass,
        "max_rel_pressure_gap": pressure,
        "max_rel_outlet_residual": outlet,
        "tolerance": 1e-9,
        "passed": bool(max(mass, pressure, outlet) <= 1e-9),
    }


def check_murray(trees, arrays, params) -> dict[str, Any]:
    """Stenosis-free trees split flow at every junction as diameter^k.

    Trees with an outlet resistance at its floor (``r_outlet_clamped``) are excluded and
    counted: the floor overrides the calibration there, so their split differs by design.
    """
    healthy = [t.healthy() for t in trees]
    results = _solve(healthy, arrays, params)
    worst = 0.0
    n_junctions = n_clamped = 0
    for tree, res in zip(healthy, results, strict=True):
        if res.r_outlet_clamped.any():
            n_clamped += 1
            continue
        for c in range(1, tree.n_branches):
            p, t = tree.parent[c], tree.attach[c]
            later = [x for x in tree.children(p) if tree.attach[x] > t]
            q_cont = res.q_outlet[p] + sum(res.q_inlet[x] for x in later)
            d_side = 2 * tree.branches[c].reference_radius()[0]
            d_cont = 2 * tree.branches[p].reference_radius()[t + 1]
            expected = (d_side / d_cont) ** params.murray_exponent
            worst = max(worst, abs(res.q_inlet[c] / q_cont / expected - 1.0))
            n_junctions += 1
    return {
        "n_trees": len(trees) - n_clamped,
        "n_excluded_clamped": n_clamped,
        "n_junctions": n_junctions,
        "exponent": params.murray_exponent,
        "max_rel_error": worst,
        "tolerance": 1e-9,
        "passed": bool(worst <= 1e-9),
    }


def check_side_branch_effect(params: PhysicsParams) -> dict[str, Any]:
    """Controlled: one main vessel with a distal lesion, with and without a side branch taking
    off upstream of it (same upstream geometry), over a grid of lesion severities and side
    branch sizes."""
    rows = []
    for severity in (0.4, 0.6, 0.8):
        s = np.linspace(0.0, 0.07, 141)
        r_ref = np.full_like(s, 1.5e-3)
        main = VesselProfile(s, r_ref * cosine_narrowing(s, 0.05, 0.01, severity), r_ref)
        alone = solve_tree(VesselTree.single(main), params).ffr[0][-1]
        for ratio in (0.4, 0.6, 0.8):
            ss = np.linspace(0.0, 0.03, 61)
            side = VesselProfile(
                ss, np.full_like(ss, ratio * 1.5e-3), np.full_like(ss, ratio * 1.5e-3)
            )
            tree = VesselTree((main, side), (-1, 0), (0, 40))  # take-off at 20 mm
            with_side = solve_tree(tree, params).ffr[0][-1]
            rows.append(
                {"severity": severity, "side_ratio": ratio, "ffr_alone": float(alone),
                 "ffr_with_side": float(with_side), "increase": float(with_side - alone)}
            )  # fmt: skip
    return {
        "rows": rows,
        "min_increase": min(r["increase"] for r in rows),
        "passed": bool(all(r["increase"] > 0 for r in rows)),
    }


def check_vanishing_side_branch(params: PhysicsParams) -> dict[str, Any]:
    """A side branch carrying almost no flow, taking off inside a lesion's expansion run,
    leaves the main vessel unchanged: the split run keeps its full expansion loss."""
    main = cosine_lesion_profile(0.06, 3.0e-3, 0.012, 0.7, n=121)
    throat = int(np.argmin(main.r))
    ss = np.linspace(0.0, 0.02, 41)
    tiny = VesselProfile(ss, np.full_like(ss, 2.5e-5), np.full_like(ss, 2.5e-5))
    res = solve_tree(VesselTree((main, tiny), (-1, 0), (0, throat + 3)), params)
    diff = float(np.abs(res.ffr[0] - solve_ffr(main, params).ffr).max())
    return {"max_abs_ffr_diff": diff, "tolerance": 2e-4, "passed": bool(diff <= 2e-4),
            "side_flow_share": float(res.q_outlet[1] / res.q_inlet[0])}  # fmt: skip


def benchmark(trees, arrays, params, repeats: int = 3) -> dict[str, Any]:
    """Trees per second for one batched call (single process), median of ``repeats``."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _solve(trees, arrays, params)
        times.append(time.perf_counter() - t0)
    rate = len(trees) / float(np.median(times))
    return {
        "n_trees": len(trees),
        "branches_mean": float(np.mean([t.n_branches for t in trees])),
        "seconds_median": float(np.median(times)),
        "trees_per_s": rate,
        "target": TARGET_TREES_PER_S,
        "passed": bool(rate >= TARGET_TREES_PER_S),
        "cpu": cpu_name(),
    }


def _example_index(arrays: dict[str, np.ndarray]) -> int:
    """A test tree with a bifurcation lesion, 2+ side branches and a clear side-branch effect."""
    effect = arrays["ffr"][:, 0, -1] - arrays["ffr_main_single"][:, -1]
    bif = (arrays["lesion_kind"] == LESION_KINDS.index("bifurcation")).any(axis=1)
    ok = bif & (arrays["branch_count"] >= 3) & (effect > 0.05) & (arrays["ffr"][:, 0, -1] < 0.9)
    idx = np.flatnonzero(ok)
    return int(idx[0]) if idx.size else 0


def plot_example(tree: VesselTree, arrays, i: int, params: PhysicsParams, path: Path) -> dict:
    """Diameter and FFR along every branch of one tree; the main vessel also as a single path."""
    res = solve_tree(tree, params, pa=arrays["pa"][i], pv=arrays["pv"][i], v_hyp=arrays["v_hyp"][i])
    single = arrays["ffr_main_single"][i].astype(float)
    fig = new_figure(
        8.0, 5.6, "Example synthetic tree: side branches change the main vessel's FFR",
        "Test tree with a bifurcation lesion; side branches drawn from their take-off",
    )  # fmt: skip
    ax_d, ax_f = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    fig.subplots_adjust(left=0.09, right=0.86, top=0.84, bottom=0.11, hspace=0.12)
    for ax in (ax_d, ax_f):
        style_axes(ax)
    offsets = [0.0] * tree.n_branches
    for k in range(1, tree.n_branches):
        offsets[k] = offsets[tree.parent[k]] + float(
            tree.branches[tree.parent[k]].s[tree.attach[k]]
        )
    for k, b in enumerate(tree.branches):
        x = m_to_mm(b.s + offsets[k])
        color = SERIES_1 if k == 0 else SERIES_3
        ax_d.plot(x, m_to_mm(2 * b.r), color=color, lw=1.6 if k == 0 else 1.2)
        ax_f.plot(x, res.ffr[k], color=color, lw=2.0 if k == 0 else 1.4)
        ax_d.annotate(tree.names[k], (x[-1], m_to_mm(2 * b.r[-1])), xytext=(4, 0),
                      textcoords="offset points", fontsize=7, color=INK_SECONDARY,
                      va="center")  # fmt: skip
        if k > 0:
            ax_f.annotate(tree.names[k], (x[-1], res.ffr[k][-1]), xytext=(4, 0),
                          textcoords="offset points", fontsize=7, color=INK_SECONDARY,
                          va="center")  # fmt: skip
    x0 = m_to_mm(tree.branches[0].s)
    ax_f.plot(x0, single, color=SERIES_2, lw=1.6, ls=(0, (4, 2)))
    ax_f.annotate("main, tree", (x0[-1], res.ffr[0][-1]), xytext=(4, 0), textcoords="offset points",
                  fontsize=7, color=INK_SECONDARY, va="center")  # fmt: skip
    ax_f.annotate("main alone (single path)", (x0[-1], single[-1]), xytext=(4, 0),
                  textcoords="offset points", fontsize=7, color=INK_SECONDARY,
                  va="center")  # fmt: skip
    ax_f.axhline(0.80, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)))
    for k in range(1, tree.n_branches):
        for ax in (ax_d, ax_f):
            ax.axvline(m_to_mm(offsets[k]), color=INK_MUTED, lw=0.6, alpha=0.6)
    ax_d.set_ylabel("Diameter (mm)", fontsize=8, color=INK_SECONDARY)
    ax_f.set_ylabel("FFR", fontsize=8, color=INK_SECONDARY)
    ax_f.set_xlabel("Distance from the inlet along each branch (mm)", fontsize=8,
                    color=INK_SECONDARY)  # fmt: skip
    save_figure(fig, path)
    return {
        "test_index": i,
        "branches": tree.n_branches,
        "main_ffr_tree": float(res.ffr[0][-1]),
        "main_ffr_single": float(single[-1]),
        "outlet_ffr": [float(f[-1]) for f in res.ffr],
    }


def _markdown(report: dict[str, Any]) -> str:
    c = report["checks"]
    b = report["benchmark"]
    ex = report["example"]
    prov = report["provenance"]
    yes = {True: "yes", False: "NO"}
    se = c["side_branch_effect"]
    lines = [
        "# Tree physics checks (P2)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent tree-report`. Do not edit by hand. Model: docs/PHYSICS.md, "
        "*Vessel trees*; data: `reports/tree_dataset_card.md`.",
        "",
        "## Sanity checks",
        "",
        "| Check | Result | Tolerance | Pass |",
        "|---|---|---|---|",
        f"| One-branch tree = single-vessel (M1) solver, {c['single_branch']['n']} sampled "
        f"vessels | max abs FFR difference {c['single_branch']['max_abs_ffr_diff']:.1e} "
        f"| ≤ 1e-10 | {yes[c['single_branch']['passed']]} |",
        f"| Mass conserved at every junction, {c['conservation']['n']} test trees "
        f"| max relative residual {c['conservation']['max_rel_mass_residual']:.1e} | ≤ 1e-9 "
        f"| {yes[c['conservation']['passed']]} |",
        f"| Pressure continuous at every take-off | max gap "
        f"{c['conservation']['max_rel_pressure_gap']:.1e} × Pa | ≤ 1e-9 "
        f"| {yes[c['conservation']['passed']]} |",
        f"| Every outlet: P − Pv = R·q | max relative residual "
        f"{c['conservation']['max_rel_outlet_residual']:.1e} | ≤ 1e-9 "
        f"| {yes[c['conservation']['passed']]} |",
        f"| Healthy trees split flow by Murray's law (k = {c['murray']['exponent']:g}), "
        f"{c['murray']['n_junctions']:,} junctions in {c['murray']['n_trees']:,} trees "
        f"({c['murray']['n_excluded_clamped']} with an outlet resistance at its floor excluded) "
        f"| max relative error "
        f"{c['murray']['max_rel_error']:.1e} | ≤ 1e-9 | {yes[c['murray']['passed']]} |",
        f"| A side branch upstream of a lesion raises distal main FFR ({len(se['rows'])} "
        f"controlled cases) | smallest increase {se['min_increase']:.4f} | > 0 "
        f"| {yes[se['passed']]} |",
        f"| A side branch carrying almost no flow inside a lesion's expansion leaves the main "
        f"vessel unchanged | max abs FFR difference "
        f"{c['vanishing_side_branch']['max_abs_ffr_diff']:.1e} | ≤ 2e-4 "
        f"| {yes[c['vanishing_side_branch']['passed']]} |",
        "",
        "## Throughput",
        "",
        "| Metric | Value | Target | Pass |",
        "|---|---|---|---|",
        f"| Trees per second, one process ({b['n_trees']:,} test trees, mean "
        f"{b['branches_mean']:.1f} branches) | {b['trees_per_s']:,.0f} | ≥ {b['target']:,.0f} "
        f"| {yes[b['passed']]} |",
        "",
        f"CPU: {b['cpu']}. One batched call (padded Newton, ~5 iterations), median of 3 runs.",
        "",
        "## Side-branch effect (controlled cases)",
        "",
        "Main vessel 70 mm, 3.0 mm, lesion centred at 50 mm; side branch from 20 mm.",
        "",
        "| Lesion severity | Side / main diameter | FFR, main alone | FFR, with side branch "
        "| Increase |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['severity']:.0%} | {r['side_ratio']:.1f} | {r['ffr_alone']:.3f} "
        f"| {r['ffr_with_side']:.3f} | {r['increase']:+.3f} |"
        for r in se["rows"]
    ]
    lines += [
        "",
        "## Example",
        "",
        f"Test tree {ex['test_index']} ({ex['branches']} branches): main-vessel distal FFR "
        f"{ex['main_ffr_tree']:.3f} as a tree vs {ex['main_ffr_single']:.3f} as a single path.",
        "",
        f"![Example tree](figures/{FIGURE})",
        "",
        "## Limitations",
        "",
        "- Junction losses are neglected; the healthy flow split follows Murray's law, not "
        "measured territories.",
        "- These checks verify the implementation against its own equations; agreement with 3D "
        "CFD or invasive FFR is not tested here (P4, P6).",
        "",
        "## Provenance",
        "",
        f"- Commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), vstent "
        f"{prov['vstent_version']}, {prov['created_utc']}; data `{report['data_dir']}`.",
        "",
    ]
    return "\n".join(lines)


def write_tree_report(
    data_dir: str | Path = "data/synth/tree_v1",
    report_dir: str | Path = "reports",
    params: PhysicsParams | None = None,
    n_trees: int = N_TREES,
) -> dict[str, Any]:
    """Run every check and the benchmark; write ``tree_physics.{md,json}`` and the figure."""
    params = params or PhysicsParams()
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    provenance = git_info()
    trees, arrays = _load_trees(data_dir, n_trees)
    results = _solve(trees, arrays, params)
    checks = {
        "single_branch": check_single_branch(params),
        "conservation": check_conservation(trees, results),
        "murray": check_murray(trees, arrays, params),
        "side_branch_effect": check_side_branch_effect(params),
        "vanishing_side_branch": check_vanishing_side_branch(params),
    }
    i = _example_index(arrays)
    example = plot_example(trees[i], arrays, i, params, report_dir / "figures" / FIGURE)
    report = {
        "checks": checks,
        "checks_passed": all(c["passed"] for c in checks.values()),
        "benchmark": benchmark(trees, arrays, params),
        "example": example,
        "params": params.model_dump(mode="json"),
        "data_dir": data_dir.as_posix(),
        "provenance": {
            **provenance,
            "vstent_version": __version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "tree_physics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (report_dir / "tree_physics.md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    return report
