import numpy as np
import pytest

from tests.test_tree import PARAMS, branch, example_tree
from vstent.config import load_config
from vstent.physics.tree import VesselTree, solve_tree
from vstent.planning.optimizer import PlannerConfig, StentCatalog
from vstent.planning.stent import apply_path_stents
from vstent.planning.tree_planner import path_profile, plan_tree_stents
from vstent.units import mm_to_m

CATALOG = load_config("configs/stent_catalog.yaml", StentCatalog)
CFG = load_config("configs/planner.yaml", PlannerConfig)


def test_path_profile():
    tree = example_tree()
    main = path_profile(tree, 0)
    assert main.length == pytest.approx(tree.branches[0].length)
    np.testing.assert_allclose(main.r[[0, -1]], tree.branches[0].r[[0, -1]])
    side = path_profile(tree, 3)  # main → D1 → D1a
    assert side.length == pytest.approx(tree.path_length(3))
    assert side.r[-1] == pytest.approx(tree.branches[3].r[-1])


def test_plan_on_the_main_path_matches_an_explicit_solve():
    tree = example_tree()
    res = plan_tree_stents(tree, 0, catalog=CATALOG, config=CFG, params=PARAMS)
    best = res.plans[0]
    assert res.pre_ffr < 0.90 and best.meets_target and res.backend == "solver"
    les = res.lesions[0]
    assert any(st.s_start <= les.s_mld <= st.s_end for st in best.stents)
    post = solve_tree(apply_path_stents(tree, 0, best.stents, edge=res.edge), PARAMS,
                      r_outlet=res.r_outlet)  # fmt: skip
    np.testing.assert_allclose(best.ffr_outlets, post.ffr_outlets, atol=1e-12)
    assert best.ffr == pytest.approx(post.ffr_outlets[0], abs=1e-12)


def test_plan_on_a_side_branch_path():
    main = branch(60, 3.2, 121, d_end_mm=2.8)
    side = branch(40, 2.4, 81, lesion=(15, 8, 0.7), d_end_mm=2.0)
    tree = VesselTree((main, side), (-1, 0), (0, 40))
    res = plan_tree_stents(tree, 1, catalog=CATALOG, config=CFG, params=PARAMS)
    best = res.plans[0]
    assert best.meets_target and res.pre_ffr < 0.80
    offset = tree.path_offsets(1)[-1][3]  # the side branch starts 20 mm along the path
    assert all(st.s_start >= offset - mm_to_m(5) for st in best.stents)
    # stenting the side branch also changes the main outlet a little (more flow upstream)
    assert best.ffr_outlets[0] != pytest.approx(res.pre_ffr_outlets[0], abs=1e-6)


def test_no_lesion_no_plan():
    tree = VesselTree.single(branch(50, 3.0, 101))
    res = plan_tree_stents(tree, 0, catalog=CATALOG, config=CFG, params=PARAMS)
    assert res.plans == [] and "No lesion" in res.message


def test_surrogate_backend_with_solver_check(tiny_tree_run, tmp_path):
    from vstent.inference.tree_runtime import TreeSurrogateRunner
    from vstent.surrogate.tree_export import export_tree_run

    export_tree_run(tiny_tree_run, tmp_path / "m", tmp_path / "r", parity_samples=4)
    runner = TreeSurrogateRunner.from_dir(tmp_path / "m")
    tree = example_tree()
    res = plan_tree_stents(tree, 0, catalog=CATALOG, config=CFG, params=PARAMS,
                           backend="surrogate", runner=runner)  # fmt: skip
    runner.close()
    assert res.backend == "surrogate" and res.plans
    assert all(p.ffr_solver is not None for p in res.plans)


def test_tree_planner_benchmark_on_tiny_data(tiny_tree_data, tmp_path):
    from vstent.planning.tree_report import run_tree_planner_benchmark

    report = run_tree_planner_benchmark(
        tiny_tree_data, tmp_path, catalog=CATALOG, config=CFG, params=PARAMS, n_trees=3
    )
    assert report["throughput"]["n_candidates"] == 5000 and report["study"]["rows"]
    md = (tmp_path / "tree_planner_benchmark.md").read_text(encoding="utf-8")
    assert "Research use only" in md and "single vessel" in md
