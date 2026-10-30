"""Tree planner benchmark (``vstent plan-benchmark --tree``, P3) → ``reports/tree_planner_*``.

1. Throughput: 5,000 candidates along the main path of one synthetic test tree, timed with
   the tree solver (target < 1 s, DECISIONS D37) and, if exported, the tree surrogate
   (reported; slower than the solver, no target).
2. Planning study: synthetic test trees whose main-vessel FFR is below 0.80, planned along the
   main path. For comparison, the same main vessel is planned as a single vessel (M6, solver),
   and that plan's FFR in the tree is reported: how much a single-path model would over-treat.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.data.synth import load_split
from vstent.data.synth_tree import tree_from_arrays
from vstent.inference.tree_runtime import TreeSurrogateRunner
from vstent.physics.solver import PhysicsParams
from vstent.physics.tree import solve_tree
from vstent.planning.optimizer import (
    PlannerConfig,
    StentCatalog,
    generate_candidates,
    plan_stents,
)
from vstent.planning.stent import apply_path_stents
from vstent.planning.tree_planner import (
    TreeSolverBackend,
    TreeSurrogateBackend,
    path_profile,
    path_radii,
    plan_tree_stents,
)
from vstent.sysinfo import cpu_name
from vstent.units import m_to_mm

N_CANDIDATES = 5000
TARGET_SECONDS = 1.0
RUNS = 5
N_TREES = 100
PRE_FFR_MAX = 0.80


def tree_throughput(tree, arrays, i, params, catalog, config, runner=None) -> dict[str, Any]:
    """Median time to build path radii and evaluate ``N_CANDIDATES`` candidates, per backend."""
    pre = solve_tree(tree, params, pa=arrays["pa"][i], pv=arrays["pv"][i],
                     v_hyp=arrays["v_hyp"][i])  # fmt: skip
    grid = path_profile(tree, 0)
    cand = generate_candidates(grid, grid.lesions(30.0), catalog, config)
    batch = cand.subset(np.arange(N_CANDIDATES) % len(cand))
    backends = [TreeSolverBackend(tree, params, pre.pa, pre.pv, pre.r_outlet)]
    if runner is not None:
        backends.append(TreeSurrogateBackend(runner, tree, pre.pa, pre.pv, pre.r_outlet))
    out: dict[str, Any] = {"n_candidates": N_CANDIDATES, "generated_for_tree": len(cand),
                           "branches": tree.n_branches}  # fmt: skip
    for be in backends:
        be.outlets(path_radii(tree, 0, [batch.subset(slice(0, 64))], config))  # warm-up
        build, evaluate = [], []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            radii = path_radii(tree, 0, [batch], config)
            t1 = time.perf_counter()
            be.outlets(radii)
            build.append(t1 - t0)
            evaluate.append(time.perf_counter() - t1)
        total = np.add(build, evaluate)
        out[be.name] = {
            "median_s": float(np.median(total)),
            "max_s": float(np.max(total)),
            "build_median_s": float(np.median(build)),
            "evaluate_median_s": float(np.median(evaluate)),
            "runs": RUNS,
        }
    return out


def tree_planning_study(arrays, params, catalog, config, n, runner=None) -> dict[str, Any]:
    """Plan the main path of the first ``n`` test trees with main FFR < 0.80."""
    idx = np.flatnonzero(arrays["ffr"][:, 0, -1] < PRE_FFR_MAX)[:n]
    solver_cfg = config.model_copy(update={"backend": "solver"})
    rows = []
    for i in idx:
        tree = tree_from_arrays(arrays, i)
        vp = params.model_copy(update={"v_hyp": float(arrays["v_hyp"][i])})
        kw = {"catalog": catalog, "config": config, "params": vp,
              "pa": float(arrays["pa"][i]), "pv": float(arrays["pv"][i])}  # fmt: skip
        res = plan_tree_stents(tree, 0, **kw)
        best = res.plans[0] if res.plans else None
        row = {
            "index": int(i),
            "branches": tree.n_branches,
            "pre_ffr": res.pre_ffr,
            "meets": bool(best and best.meets_target),
            "length_mm": m_to_mm(best.total_length) if best else None,
            "n_stents": len(best.stents) if best else 0,
            "seconds": res.seconds,
        }
        # the same main vessel planned as a single vessel (M6), then evaluated in the tree
        single = plan_stents(tree.branches[0], **{**kw, "config": solver_cfg})
        sb = single.plans[0] if single.plans else None
        if sb is not None:
            in_tree = solve_tree(apply_path_stents(tree, 0, sb.stents, edge=res.edge), vp,
                                 r_outlet=res.r_outlet)  # fmt: skip
            row["single_length_mm"] = m_to_mm(sb.total_length)
            row["single_plan_ffr_in_tree"] = float(in_tree.ffr[0][-1])
        if runner is not None and tree.n_branches <= runner.max_branches:
            sur = plan_tree_stents(tree, 0, **kw, backend="surrogate", runner=runner)
            sbest = sur.plans[0] if sur.plans else None
            row["surrogate_seconds"] = sur.seconds
            row["surrogate_same_length"] = bool(
                sbest and best and abs(sbest.total_length - best.total_length) < 1e-9
            )
            row["surrogate_plan_solver_ffr"] = None if sbest is None else sbest.ffr_solver
        rows.append(row)
    with_plan = [r for r in rows if r["length_mm"] is not None]
    both = [r for r in with_plan if r.get("single_length_mm") is not None and r["meets"]]
    summary = {
        "n_trees": len(rows),
        "reaches_target": sum(r["meets"] for r in rows),
        "two_stent_plans": sum(r["n_stents"] == 2 for r in rows),
        "median_length_mm": float(np.median([r["length_mm"] for r in with_plan]))
        if with_plan
        else None,
        "median_seconds": float(np.median([r["seconds"] for r in rows])),
        "p95_seconds": float(np.percentile([r["seconds"] for r in rows], 95)),
        "single_longer": sum(r["single_length_mm"] > r["length_mm"] + 1e-6 for r in both),
        "single_shorter": sum(r["single_length_mm"] < r["length_mm"] - 1e-6 for r in both),
        "n_compared": len(both),
        "median_extra_length_single_mm": float(
            np.median([r["single_length_mm"] - r["length_mm"] for r in both])
        )
        if both
        else None,
    }
    sur = [r for r in rows if "surrogate_seconds" in r]
    if sur:
        summary["surrogate"] = {
            "n": len(sur),
            "same_length": sum(r["surrogate_same_length"] for r in sur),
            "confirmed_by_solver": sum(
                (r["surrogate_plan_solver_ffr"] or 0) >= config.target_ffr for r in sur
            ),
            "median_seconds": float(np.median([r["surrogate_seconds"] for r in sur])),
        }
    return {"summary": summary, "rows": rows}


def _markdown(report: dict[str, Any]) -> str:
    t, s, prov = report["throughput"], report["study"]["summary"], report["provenance"]
    solver = t["solver"]
    passed = solver["median_s"] < TARGET_SECONDS
    lines = [
        "# Tree planner benchmark (P3)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent plan-benchmark --tree`. Do not edit by hand. Planner: "
        "docs/DECISIONS.md D37; trees: `reports/tree_dataset_card.md`.",
        "",
        "## Target",
        "",
        "| Metric | Value | Target | Pass |",
        "|---|---|---|---|",
        f"| {N_CANDIDATES:,} candidates along a tree path, tree solver (median) "
        f"| {solver['median_s']:.3f} s | < {TARGET_SECONDS:g} s | {'yes' if passed else 'NO'} |",
        "",
        "## Throughput",
        "",
        f"One synthetic test tree with {t['branches']} branches; {t['generated_for_tree']} "
        f"candidates generated, repeated to {N_CANDIDATES:,}. Median of {RUNS} runs.",
        "",
        "| Backend | Median | Max | Building radii | Evaluation |",
        "|---|---|---|---|---|",
    ]
    for name in ("solver", "surrogate"):
        if name in t:
            b = t[name]
            lines.append(
                f"| Tree {name} | {b['median_s']:.3f} s | {b['max_s']:.3f} s "
                f"| {1e3 * b['build_median_s']:.0f} ms | {1e3 * b['evaluate_median_s']:.0f} ms |"
            )
    lines += [
        "",
        f"CPU: {report['cpu']}. The tree surrogate is slower than the tree solver here and is "
        "not claimed to be faster; it validates the pipeline for later 3D-CFD labels.",
        "",
        "## Planning study",
        "",
        f"The first {s['n_trees']} synthetic `test` trees with main-vessel FFR < "
        f"{PRE_FFR_MAX:.2f}, planned along the main path (tree solver, target "
        f"{report['config']['target_ffr']:.2f}).",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Plans reaching the target | {s['reaches_target']} / {s['n_trees']} |",
        f"| Two-stent plans | {s['two_stent_plans']} / {s['n_trees']} |",
        f"| Median total stent length | {s['median_length_mm']:.0f} mm |",
        f"| Planning time per tree: median / p95 | {s['median_seconds']:.2f} s / "
        f"{s['p95_seconds']:.2f} s |",
        f"| Main vessel planned as a single vessel (M6): longer / shorter plan than the tree's "
        f"({s['n_compared']} compared) | {s['single_longer']} / {s['single_shorter']} |",
        f"| Median extra length of the single-vessel plan | "
        f"{s['median_extra_length_single_mm']:+.0f} mm |",
    ]
    if "surrogate" in s:
        g = s["surrogate"]
        lines += [
            f"| Tree surrogate backend: same total length as the solver's plan | "
            f"{g['same_length']} / {g['n']} |",
            f"| Tree surrogate plans confirmed by the solver (FFR ≥ target) | "
            f"{g['confirmed_by_solver']} / {g['n']} |",
            f"| Tree surrogate backend: planning time per tree (median) | "
            f"{g['median_seconds']:.2f} s |",
        ]
    lines += [
        "",
        "## Limitations",
        "",
        "- Stents only widen the lumen; side-branch jailing, carina shift and overlap are not "
        "modelled. Outlet resistances stay at their pre-PCI values.",
        "- Synthetic trees and the reduced-order tree model; no clinical outcome is involved.",
        "",
        f"Commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), vstent "
        f"{prov['vstent_version']}, {prov['created_utc']}.",
        "",
    ]
    return "\n".join(lines)


def run_tree_planner_benchmark(
    data_dir: str | Path,
    report_dir: str | Path = "reports",
    *,
    catalog: StentCatalog,
    config: PlannerConfig,
    params: PhysicsParams,
    tree_model_dir: str | Path | None = None,
    n_trees: int = N_TREES,
) -> dict[str, Any]:
    """Throughput and planning study on synthetic test trees; writes the report."""
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    arrays = load_split(data_dir, "test")
    runner = None if tree_model_dir is None else TreeSurrogateRunner.from_dir(tree_model_dir)
    low = np.flatnonzero((arrays["ffr"][:, 0, -1] < PRE_FFR_MAX) & (arrays["branch_count"] >= 3))
    i = int(low[0])
    throughput = tree_throughput(tree_from_arrays(arrays, i), arrays, i, params, catalog,
                                 config, runner)  # fmt: skip
    study = tree_planning_study(arrays, params, catalog, config, n_trees, runner)
    if runner is not None:
        runner.close()
    report = {
        "throughput": throughput,
        "study": study,
        "target_met": throughput["solver"]["median_s"] < TARGET_SECONDS,
        "config": config.model_dump(mode="json"),
        "catalog": catalog.model_dump(mode="json"),
        "tree_model": None if tree_model_dir is None else Path(tree_model_dir).as_posix(),
        "cpu": cpu_name(),
        "provenance": provenance,
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "tree_planner_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    (report_dir / "tree_planner_benchmark.md").write_text(
        _markdown(report), encoding="utf-8", newline="\n"
    )
    return report
