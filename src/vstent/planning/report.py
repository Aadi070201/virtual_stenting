"""Planner benchmark (``vstent plan-benchmark``) → ``reports/planner_benchmark.{md,json}``.

1. Throughput: 5,000 single-stent candidates on a fixed synthetic tandem-lesion vessel,
   stented radii + evaluation, for the surrogate (ONNX) and the solver backends.
2. Planning on synthetic test vessels with pre-PCI FFR < 0.80 (solver labels): both backends
   plan every vessel; the surrogate's chosen plan is re-checked with the solver.

Every number in the report is measured here.
"""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.config import CONFIG_FILENAME, git_info, load_config
from vstent.data.synth import load_split
from vstent.inference.onnx_runtime import SurrogateRunner
from vstent.physics.geometry import VesselProfile, cosine_narrowing
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.planning.optimizer import (
    PlannerConfig,
    SolverBackend,
    StentCatalog,
    SurrogateBackend,
    generate_candidates,
    plan_stents,
    planning_grid,
    stented_radii,
)
from vstent.surrogate.train import SurrogateConfig
from vstent.sysinfo import cpu_name
from vstent.units import m_to_mm

N_CANDIDATES = 5000
TARGET_SECONDS = 1.0
RUNS = 7
N_VESSELS = 100
PRE_FFR_MAX = 0.80


def tandem_vessel() -> VesselProfile:
    """70 mm, 3.0 mm reference, two 10 mm cosine lesions of 60% DS at 20 and 50 mm."""
    s = np.linspace(0.0, 0.07, 256)
    r_ref = np.full_like(s, 1.5e-3)
    r = r_ref * cosine_narrowing(s, 0.02, 0.01, 0.6) * cosine_narrowing(s, 0.05, 0.01, 0.6)
    return VesselProfile(s, r, r_ref)


def throughput(
    runner: SurrogateRunner, params: PhysicsParams, catalog: StentCatalog, config: PlannerConfig
) -> dict[str, Any]:
    """Median time to build radii and evaluate ``N_CANDIDATES`` candidates, per backend.

    The surrogate is timed with ``runner.screen``, the model the planner screens with.
    """
    runner = runner.screen
    grid = planning_grid(tandem_vessel())
    pre = solve_ffr_batch(grid.r, grid.ds, grid.reference_radius(), params)
    pa, pv, rm = pre.pa[0], pre.pv[0], pre.r_micro[0]
    cand = generate_candidates(grid, grid.lesions(30.0), catalog, config)
    batch = cand.subset(np.arange(N_CANDIDATES) % len(cand))
    out = {"n_candidates": N_CANDIDATES, "generated_for_vessel": len(cand)}
    backends = (SurrogateBackend(runner, grid, pa, pv, rm), SolverBackend(grid, params, pa, pv, rm))
    for backend in backends:
        backend.curves(stented_radii(grid, [batch.subset(slice(0, 256))], config))  # warm-up
        build, evaluate = [], []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            radii = stented_radii(grid, [batch], config)
            t1 = time.perf_counter()
            backend.curves(radii)
            t2 = time.perf_counter()
            build.append(t1 - t0)
            evaluate.append(t2 - t1)
        total = np.add(build, evaluate)
        out[backend.name] = {
            "median_s": float(np.median(total)),
            "max_s": float(np.max(total)),
            "build_median_s": float(np.median(build)),
            "evaluate_median_s": float(np.median(evaluate)),
            "runs": RUNS,
        }
    return out


def planning_study(
    data_dir: Path,
    runner: SurrogateRunner,
    params: PhysicsParams,
    catalog: StentCatalog,
    config: PlannerConfig,
    n_vessels: int,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Plan the first ``n_vessels`` test vessels with pre-PCI FFR < 0.80 with both backends."""
    keys = ("r", "r_ref", "length", "pa", "pv", "v_hyp", "ffr")
    arrays = load_split(data_dir, "test", keys)
    idx = np.flatnonzero(arrays["ffr"][:, -1] < PRE_FFR_MAX)[:n_vessels]
    rows = []
    for k, i in enumerate(idx):
        n = arrays["r"].shape[1]
        s = np.linspace(0.0, float(arrays["length"][i]), n)
        profile = VesselProfile(s, arrays["r"][i], arrays["r_ref"][i])
        vessel_params = params.model_copy(update={"v_hyp": float(arrays["v_hyp"][i])})
        kw = {
            "catalog": catalog,
            "params": vessel_params,
            "pa": float(arrays["pa"][i]),
            "pv": float(arrays["pv"][i]),
        }
        sur = plan_stents(
            profile, config=config.model_copy(update={"backend": "surrogate"}), runner=runner, **kw
        )
        sol = plan_stents(profile, config=config.model_copy(update={"backend": "solver"}), **kw)
        best_sur = sur.plans[0] if sur.plans else None
        best_sol = sol.plans[0] if sol.plans else None
        rows.append(
            {
                "index": int(i),
                "pre_ffr_solver": float(arrays["ffr"][i, -1]),
                "n_lesions": len(sol.lesions),
                "n_feasible": sol.n_feasible,
                "seconds_surrogate": sur.seconds,
                "seconds_solver": sol.seconds,
                "sur_meets": bool(best_sur and best_sur.meets_target),
                "sol_meets": bool(best_sol and best_sol.meets_target),
                "sur_length_mm": m_to_mm(best_sur.total_length) if best_sur else None,
                "sol_length_mm": m_to_mm(best_sol.total_length) if best_sol else None,
                "sur_n_stents": len(best_sur.stents) if best_sur else 0,
                "sur_ffr": best_sur.ffr if best_sur else None,
                "sur_ffr_checked": best_sur.ffr_solver if best_sur else None,
                "sol_ffr": best_sol.ffr if best_sol else None,
            }
        )
        if (k + 1) % 20 == 0:
            log(f"  planned {k + 1}/{len(idx)} vessels")
    return {"rows": rows, "summary": _summarise(rows, config.target_ffr)}


def _summarise(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    with_plan = [r for r in rows if r["sur_ffr"] is not None]
    sur_meets = [r for r in rows if r["sur_meets"]]
    both = [r for r in rows if r["sur_meets"] and r["sol_meets"]]
    return {
        "n_vessels": len(rows),
        "sol_reaches_target": sum(r["sol_meets"] for r in rows),
        "sur_reaches_target": len(sur_meets),
        "sur_plan_confirmed_by_solver": sum(r["sur_ffr_checked"] >= target for r in sur_meets),
        "median_abs_ffr_sur_vs_solver_check": float(
            np.median([abs(r["sur_ffr"] - r["sur_ffr_checked"]) for r in with_plan])
        )
        if with_plan
        else None,
        "same_length_both_meet": sum(r["sur_length_mm"] == r["sol_length_mm"] for r in both),
        "n_both_meet": len(both),
        "median_length_diff_mm": float(
            np.median([r["sur_length_mm"] - r["sol_length_mm"] for r in both])
        )
        if both
        else None,
        "two_stent_plans_surrogate": sum(r["sur_n_stents"] == 2 for r in rows),
        "median_seconds_surrogate": float(np.median([r["seconds_surrogate"] for r in rows])),
        "p95_seconds_surrogate": float(np.percentile([r["seconds_surrogate"] for r in rows], 95)),
        "median_seconds_solver": float(np.median([r["seconds_solver"] for r in rows])),
        "median_feasible": float(np.median([r["n_feasible"] for r in rows])),
    }


def run_planner_benchmark(
    model_dir: str | Path,
    report_dir: str | Path = "reports",
    *,
    catalog: StentCatalog,
    config: PlannerConfig,
    params: PhysicsParams,
    n_vessels: int = N_VESSELS,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run both parts and write the report."""
    log = log or (lambda _msg: None)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    data_dir = Path(load_config(model_dir / CONFIG_FILENAME, SurrogateConfig).data_dir)
    runner = SurrogateRunner.from_dir(model_dir)
    tp = throughput(runner, params, catalog, config)
    log(
        f"{N_CANDIDATES} candidates: surrogate {tp['surrogate']['median_s']:.3f} s, "
        f"solver {tp['solver']['median_s']:.3f} s"
    )
    study = planning_study(data_dir, runner, params, catalog, config, n_vessels, log)
    runner.close()
    report = {
        "targets": [
            {
                "metric": f"{N_CANDIDATES:,} candidates, surrogate backend (median)",
                "value": f"{tp['surrogate']['median_s']:.3f} s",
                "target": f"< {TARGET_SECONDS:g} s",
                "passed": tp["surrogate"]["median_s"] < TARGET_SECONDS,
            }
        ],
        "throughput": tp,
        "study": study,
        "config": config.model_dump(mode="json"),
        "catalog": catalog.model_dump(mode="json"),
        "environment": {"cpu": cpu_name(), "workers": runner.workers, "chunk": runner.chunk_size},
        "model_dir": model_dir.as_posix(),
        "data_dir": data_dir.as_posix(),
        "provenance": provenance,
    }
    report["targets_met"] = all(t["passed"] for t in report["targets"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "planner_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    (report_dir / "planner_benchmark.md").write_text(
        _markdown(report), encoding="utf-8", newline="\n"
    )
    return report


def _markdown(report: dict[str, Any]) -> str:
    tp, st, cfg, prov = (
        report["throughput"],
        report["study"]["summary"],
        report["config"],
        report["provenance"],
    )
    t = report["targets"][0]
    sur, sol = tp["surrogate"], tp["solver"]
    n = st["n_vessels"]
    lines = [
        "# Virtual stent planner: speed and behaviour on synthetic vessels",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent plan-benchmark`. Do not edit by hand.",
        "",
        "## Target",
        "",
        "| Metric | Value | Target | Pass |",
        "|---|---|---|---|",
        f"| {t['metric']} | {t['value']} | {t['target']} | {'yes' if t['passed'] else 'NO'} |",
        "",
        "## Throughput",
        "",
        f"{tp['n_candidates']:,} single-stent candidates on a synthetic tandem-lesion vessel "
        "(70 mm, 3.0 mm reference, two 60% lesions; the generator yields "
        f"{tp['generated_for_vessel']:,} candidates for it before constraints, repeated to "
        f"{tp['n_candidates']:,}). Time = building the stented radii + evaluating every "
        f"candidate; median of {sur['runs']} runs.",
        "",
        "| Backend | Median | Max | Building radii | Evaluation |",
        "|---|---|---|---|---|",
    ]
    for name, row in (("Surrogate (ONNX)", sur), ("Solver", sol)):
        lines.append(
            f"| {name} | {row['median_s']:.3f} s | {row['max_s']:.3f} s "
            f"| {row['build_median_s'] * 1e3:.0f} ms | {row['evaluate_median_s'] * 1e3:.0f} ms |"
        )
    env = report["environment"]
    lines += [
        "",
        f"CPU: {env['cpu']}; surrogate runner: chunks of {env['chunk']} on {env['workers']} "
        "threads. The reduced-order solver is faster than the surrogate here; the surrogate is "
        "not claimed to be faster (it validates the pipeline for Phase 2 CFD labels).",
        "",
        "## Planning synthetic test vessels",
        "",
        f"The first {n} vessels of the synthetic `test` split with pre-PCI FFR < "
        f"{PRE_FFR_MAX:.2f} (solver), planned with each backend (target FFR ≥ "
        f"{cfg['target_ffr']:.2f}, catalogue and constraints below). Synthetic vessels use their "
        "known reference radius.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Plans reaching the target, solver backend | {st['sol_reaches_target']} / {n} |",
        f"| Plans reaching the target, surrogate backend | {st['sur_reaches_target']} / {n} |",
        f"| … of which the solver confirms FFR ≥ {cfg['target_ffr']:.2f} "
        f"| {st['sur_plan_confirmed_by_solver']} / {st['sur_reaches_target']} |",
        "| Median abs(surrogate − solver) post-PCI FFR of the surrogate's plan | "
        + (
            f"{st['median_abs_ffr_sur_vs_solver_check']:.4f}"
            if st["median_abs_ffr_sur_vs_solver_check"] is not None
            else "n/a"
        )
        + " |",
        f"| Same total stent length chosen by both backends (both reach target) "
        f"| {st['same_length_both_meet']} / {st['n_both_meet']} |",
        "| Median length difference surrogate − solver (both reach target) | "
        + (
            f"{st['median_length_diff_mm']:+.1f} mm"
            if st["median_length_diff_mm"] is not None
            else "n/a"
        )
        + " |",
        f"| Two-stent plans (surrogate) | {st['two_stent_plans_surrogate']} / {n} |",
        f"| Feasible single-stent candidates per vessel (median) | {st['median_feasible']:.0f} |",
        f"| Planning time per vessel, surrogate: median / p95 "
        f"| {st['median_seconds_surrogate']:.2f} s / {st['p95_seconds_surrogate']:.2f} s |",
        f"| Planning time per vessel, solver: median | {st['median_seconds_solver']:.2f} s |",
        "",
        "## Settings",
        "",
        f"- Catalogue (generic): diameters {report['catalog']['diameters_mm']} mm, lengths "
        f"{report['catalog']['lengths_mm']} mm.",
        f"- Start positions every {cfg['start_step_mm']} mm; stent must cover a lesion MLD "
        f"(DS > 30%); nominal diameter ≤ {cfg['max_diameter_ratio']} × reference at the distal "
        f"end; DS < {cfg['landing_max_ds_pct']:g}% over the first and last "
        f"{cfg['landing_zone_mm']:g} mm; inside the vessel; up to {cfg['max_stents']} stents "
        f"(pairs from the best {cfg['top_k_per_lesion']} candidates per lesion, spread over all "
        "lengths, no overlap).",
        f"- Stent edges blended over {cfg['edge_mm']:g} mm inside each end; expansion "
        f"{cfg['expansion']:g}. Pa, Pv and R_micro fixed at their pre-PCI values.",
        "",
        "## Provenance",
        "",
        f"- Surrogate model `{report['model_dir']}`, data `{report['data_dir']}`.",
        f"- Commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), vstent "
        f"{prov['vstent_version']}, {prov['created_utc']}.",
        "",
        "## Limitations",
        "",
        "- Synthetic single vessels; post-PCI FFR comes from the same reduced-order model (or its "
        "surrogate), not from patients.",
        "- The stent model only widens the lumen to the stent radius; no recoil, malapposition, "
        "edge injury or side-branch effects.",
        "- Two-stent plans pair the best candidates per lesion, which is not an exhaustive "
        "search; overlapping stents are not considered.",
        "",
    ]
    return "\n".join(lines)
