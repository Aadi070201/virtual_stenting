import time

import numpy as np
import pytest

from vstent.config import load_config
from vstent.inference.onnx_runtime import SurrogateRunner
from vstent.physics.geometry import VesselProfile, cosine_lesion_profile, cosine_narrowing
from vstent.physics.solver import PhysicsParams, solve_ffr, solve_ffr_batch
from vstent.planning.optimizer import (
    PlannerConfig,
    SolverBackend,
    StentCatalog,
    _window_max,
    feasible,
    generate_candidates,
    pareto_front,
    plan_stents,
    planning_grid,
    stented_radii,
)
from vstent.planning.stent import Stent, apply_stent, apply_stents, stent_weight
from vstent.units import mm_to_m

CATALOG = load_config("configs/stent_catalog.yaml", StentCatalog)
SOLVER_CFG = load_config("configs/planner.yaml", PlannerConfig).model_copy(
    update={"backend": "solver"}
)
PARAMS = load_config("configs/physics.yaml", PhysicsParams)


def _focal():
    return cosine_lesion_profile(length=0.06, ref_diameter=3e-3, lesion_length=0.012, severity=0.7)


def _tandem(gap_center_mm=30.0):
    s = np.linspace(0.0, 0.08, 256)
    r_ref = np.full_like(s, 1.5e-3)
    c1, c2 = 0.02, 0.02 + gap_center_mm * 1e-3
    r = r_ref * cosine_narrowing(s, c1, 0.01, 0.6) * cosine_narrowing(s, c2, 0.01, 0.6)
    return VesselProfile(s, r, r_ref)


# --- stent model -------------------------------------------------------------------------


def test_stent_widens_inside_and_never_narrows():
    p = _focal()
    q = apply_stent(p, 0.02, 0.02, 3.0e-3, edge=1e-3)
    assert np.all(q.r >= p.r - 1e-15)
    full = (p.s >= 0.021) & (p.s <= 0.039)
    assert np.allclose(q.r[full], np.maximum(p.r[full], 1.5e-3))
    outside = (p.s < 0.02) | (p.s > 0.04)
    assert np.array_equal(q.r[outside], p.r[outside])
    np.testing.assert_allclose(q.reference_radius(), p.reference_radius())


def test_stent_never_reduces_a_wider_lumen():
    p = cosine_lesion_profile(length=0.05, ref_diameter=4e-3, lesion_length=0.01, severity=0.3)
    q = apply_stent(p, 0.01, 0.03, 2.25e-3)  # smaller than the vessel almost everywhere
    assert np.all(q.r >= p.r - 1e-15)


def test_stent_weight_edges_and_batch():
    s = np.linspace(0.0, 0.02, 201)
    w = stent_weight(s, 0.005, 0.015, 0.002)
    assert w[s < 0.005].max() == 0.0 and w[s > 0.015].max() == 0.0
    assert np.allclose(w[(s >= 0.007) & (s <= 0.013)], 1.0)
    assert np.all(np.diff(w[(s >= 0.005) & (s <= 0.007)]) >= 0)  # ramps up
    assert stent_weight(s, np.array([0.0, 0.005]), np.array([0.01, 0.02]), 0.001).shape == (2, 201)


def test_multiple_stents_expansion_and_bounds():
    p = _tandem()
    q = apply_stents(p, [Stent(0.012, 0.015, 3e-3), Stent(0.042, 0.015, 3e-3, expansion=1.1)])
    assert q.mld()[0] > p.mld()[0]
    mid2 = np.argmin(np.abs(p.s - 0.0495))
    assert q.r[mid2] == pytest.approx(1.1 * 1.5e-3)
    with pytest.raises(ValueError, match="outside the vessel"):
        apply_stent(p, 0.07, 0.02, 3e-3)
    with pytest.raises(ValueError, match="positive"):
        Stent(0.0, -1.0, 3e-3)


# --- physics with fixed R_micro --------------------------------------------------------------


def test_solver_fixed_r_micro_matches_calibration_and_flow_rises_after_stent():
    p = _focal()
    pre = solve_ffr(p, PARAMS)
    same = solve_ffr_batch(p.r, p.ds, p.reference_radius(), PARAMS, r_micro=pre.r_micro)
    np.testing.assert_allclose(same.ffr[0], pre.ffr)
    q = apply_stent(p, 0.022, 0.016, 3e-3)
    post = solve_ffr_batch(q.r, q.ds, q.reference_radius(), PARAMS, r_micro=pre.r_micro)
    assert post.q[0] > pre.q and post.ffr[0, -1] > pre.ffr[-1]
    with pytest.raises(ValueError, match="r_micro"):
        solve_ffr_batch(p.r, p.ds, p.reference_radius(), PARAMS, r_micro=-1.0)


# --- candidates and constraints ------------------------------------------------------------


def test_window_max_matches_brute_force():
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, 1.0, 101)
    y = rng.normal(size=101)
    a = rng.uniform(0.0, 0.9, 300)
    b = a + rng.uniform(0.0, 0.1, 300)
    brute = [
        max(y[(s >= ai) & (s <= bi)].max(initial=-np.inf), np.interp(ai, s, y), np.interp(bi, s, y))
        for ai, bi in zip(a, b, strict=True)
    ]
    np.testing.assert_allclose(_window_max(s, y, a, b), brute)


def test_candidates_cover_an_mld_on_the_step_grid():
    grid = planning_grid(_tandem())
    lesions = grid.lesions(30.0)
    cand = generate_candidates(grid, lesions, CATALOG, SOLVER_CFG)
    assert len(cand) > 0
    covers = np.zeros(len(cand), bool)
    for les in lesions:
        covers |= (cand.s_start <= les.s_mld + 1e-9) & (cand.s_end >= les.s_mld - 1e-9)
    assert covers.all()
    k = cand.s_start / mm_to_m(SOLVER_CFG.start_step_mm)
    assert np.allclose(k, np.round(k))
    assert set(np.round(cand.diameter * 1e3, 2)) == set(CATALOG.diameters_mm)


def _assert_constraints(result, cfg):
    grid = result.grid
    ref_d = 2.0 * grid.reference_radius()
    ds = 100.0 * (1.0 - grid.r / grid.reference_radius())
    zone = mm_to_m(cfg.landing_zone_mm)
    for plan in result.plans + result.pareto:
        for st in plan.stents:
            assert st.s_start >= -1e-9 and st.s_end <= grid.s[-1] + 1e-9
            assert st.diameter <= cfg.max_diameter_ratio * np.interp(st.s_end, grid.s, ref_d) + 1e-9
            for lo, hi in ((st.s_start, st.s_start + zone), (st.s_end - zone, st.s_end)):
                inside = (grid.s >= lo) & (grid.s <= hi)
                worst = max(
                    ds[inside].max(initial=-np.inf),
                    np.interp(lo, grid.s, ds),
                    np.interp(hi, grid.s, ds),
                )
                assert worst < cfg.landing_max_ds_pct
        if len(plan.stents) == 2:
            a, b = plan.stents
            assert a.s_end <= b.s_start + 1e-9


# --- optimiser ---------------------------------------------------------------------------------


def test_focal_lesion_shortest_plan_reaching_target():
    res = plan_stents(_focal(), catalog=CATALOG, config=SOLVER_CFG, params=PARAMS)
    best = res.plans[0]
    assert res.pre_ffr < 0.8 and best.meets_target and best.ffr >= 0.90
    les = res.lesions[0]
    st = best.stents[0]
    assert len(best.stents) == 1 and st.s_start <= les.s_mld <= st.s_end
    # brute force over every feasible single-stent candidate with the same solver
    grid = res.grid
    cand = generate_candidates(grid, res.lesions, CATALOG, SOLVER_CFG)
    cand = cand.subset(feasible(grid, cand, SOLVER_CFG))
    backend = SolverBackend(grid, PARAMS, res.pa, res.pv, res.r_micro)
    ffr = backend.curves(stented_radii(grid, [cand], SOLVER_CFG))[:, -1]
    assert best.total_length == pytest.approx(cand.length[ffr >= 0.90].min())
    np.testing.assert_allclose(res.best_ffr_curve[-1], best.ffr)
    _assert_constraints(res, SOLVER_CFG)


def test_tandem_lesions_one_long_or_two_stents():
    for gap in (18.0, 32.0):  # close: one long stent may cover both; far: two stents needed
        res = plan_stents(_tandem(gap), catalog=CATALOG, config=SOLVER_CFG, params=PARAMS)
        best = res.plans[0]
        assert len(res.lesions) == 2 and best.meets_target
        mlds = [les.s_mld for les in res.lesions]
        covered = [any(st.s_start <= m <= st.s_end for st in best.stents) for m in mlds]
        assert all(covered)
        _assert_constraints(res, SOLVER_CFG)
    assert len(best.stents) == 2  # 32 mm apart: no catalogue stent covers both


def test_two_stent_plan_is_shortest_with_exhaustive_pools():
    cat = StentCatalog(diameters_mm=[3.0, 3.25], lengths_mm=[8, 12, 18])
    cfg = SOLVER_CFG.model_copy(update={"start_step_mm": 1.0, "top_k_per_lesion": 10_000})
    res = plan_stents(_tandem(32.0), catalog=cat, config=cfg, params=PARAMS)
    grid = res.grid
    cand = generate_candidates(grid, res.lesions, cat, cfg)
    cand = cand.subset(feasible(grid, cand, cfg))
    backend = SolverBackend(grid, PARAMS, res.pa, res.pv, res.r_micro)
    i, j = np.triu_indices(len(cand), k=1)
    apart = (cand.s_end[i] <= cand.s_start[j] + 1e-9) | (cand.s_end[j] <= cand.s_start[i] + 1e-9)
    i, j = i[apart], j[apart]
    ffr = backend.curves(stented_radii(grid, [cand.subset(i), cand.subset(j)], cfg))[:, -1]
    lengths = cand.length[i] + cand.length[j]
    assert res.plans[0].total_length == pytest.approx(lengths[ffr >= cfg.target_ffr].min())


def test_no_lesion_and_unreachable_target():
    healthy = cosine_lesion_profile(
        length=0.05, ref_diameter=3e-3, lesion_length=0.01, severity=0.1
    )
    res = plan_stents(healthy, catalog=CATALOG, config=SOLVER_CFG, params=PARAMS)
    assert res.plans == [] and "No lesion" in res.message
    strict = SOLVER_CFG.model_copy(update={"target_ffr": 0.999})
    res = plan_stents(_focal(), catalog=CATALOG, config=strict, params=PARAMS)
    assert res.plans and not res.plans[0].meets_target and "No plan reaches" in res.message
    assert res.plans[0].ffr == max(p.ffr for p in res.pareto)


def test_pareto_front_is_monotone():
    res = plan_stents(_tandem(32.0), catalog=CATALOG, config=SOLVER_CFG, params=PARAMS)
    lengths = [p.total_length for p in res.pareto]
    ffrs = [p.ffr for p in res.pareto]
    assert lengths == sorted(lengths) and all(np.diff(ffrs) > 0)


def test_surrogate_backend_runs_and_is_rechecked(tiny_export):
    runner = SurrogateRunner.from_dir(tiny_export["out_dir"])
    cfg = SOLVER_CFG.model_copy(update={"backend": "surrogate"})
    res = plan_stents(_focal(), catalog=CATALOG, config=cfg, params=PARAMS, runner=runner)
    runner.close()
    assert res.backend == "surrogate" and res.plans
    assert all(p.ffr_solver is not None for p in res.plans)
    with pytest.raises(ValueError, match="SurrogateRunner"):
        plan_stents(_focal(), catalog=CATALOG, config=cfg, params=PARAMS)


def test_planning_grid_keeps_reference_and_length():
    p = _focal()
    g = planning_grid(p, n=256)
    assert g.n == 256 and g.length == pytest.approx(p.length) and g.r_ref is not None


def test_solver_backend_throughput_5000_candidates():
    grid = planning_grid(_tandem(32.0))
    pre = solve_ffr_batch(grid.r, grid.ds, grid.reference_radius(), PARAMS)
    cand = generate_candidates(grid, grid.lesions(30.0), CATALOG, SOLVER_CFG)
    batch = cand.subset(np.arange(5000) % len(cand))
    backend = SolverBackend(grid, PARAMS, pre.pa[0], pre.pv[0], pre.r_micro[0])
    t0 = time.perf_counter()
    backend.curves(stented_radii(grid, [batch], SOLVER_CFG))
    assert time.perf_counter() - t0 < 1.0


def test_pareto_front_helper():
    from vstent.planning.optimizer import Plan

    a = Plan((Stent(0, 0.012, 3e-3),), 0.85, False)
    b = Plan((Stent(0, 0.015, 3e-3),), 0.84, False)  # dominated by a
    c = Plan((Stent(0, 0.018, 3e-3),), 0.93, True)
    assert pareto_front([c, b, a]) == [a, c]


def test_equal_total_lengths_tie_on_ffr():
    """12 + 26 mm and one 38 mm stent have the same length, so the higher FFR ranks first."""
    from vstent.planning.optimizer import Plan, _rank

    pair = Plan((Stent(0, mm_to_m(12), 3e-3), Stent(0.02, mm_to_m(26), 3e-3)), 0.95, True)
    single = Plan((Stent(0, mm_to_m(38), 3e-3),), 0.92, True)
    assert pair.total_length == single.total_length
    assert _rank([single, pair]) == [pair, single]
