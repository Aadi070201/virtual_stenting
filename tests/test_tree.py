import numpy as np
import pytest

from vstent.data.synth import SamplingConfig, sample_vessel
from vstent.physics.geometry import VesselProfile, cosine_lesion_profile, cosine_narrowing
from vstent.physics.solver import PhysicsParams, solve_ffr, solve_ffr_batch
from vstent.physics.tree import (
    VesselTree,
    murray_flows,
    solve_tree,
    solve_tree_batch,
    solve_tree_variants,
    tree_segments,
)

MM = 1e-3
PARAMS = PhysicsParams()


def branch(length_mm, d_mm, n=128, lesion=None, d_end_mm=None):
    """A straight branch with a linearly tapering reference and an optional cosine lesion
    ``(centre_mm, length_mm, severity)``."""
    s = np.linspace(0.0, length_mm * MM, n)
    r_ref = np.linspace(d_mm, d_end_mm or d_mm, n) * MM / 2
    r = r_ref.copy()
    if lesion is not None:
        c, ln, sev = lesion
        r = r_ref * cosine_narrowing(s, c * MM, ln * MM, sev)
    return VesselProfile(s, r, r_ref)


def example_tree():
    """Main vessel with a distal lesion, two side branches (one with an ostial lesion) and a
    second-level branch on the first side branch."""
    main = branch(70, 3.2, 141, lesion=(45, 10, 0.6), d_end_mm=2.4)
    d1 = branch(40, 2.2, 81, lesion=(4, 6, 0.5), d_end_mm=1.8)
    d2 = branch(30, 1.8, 61, d_end_mm=1.5)
    d1a = branch(20, 1.4, 41)
    return VesselTree(
        (main, d1, d2, d1a), (-1, 0, 0, 1), (0, 40, 100, 30), ("main", "D1", "D2", "D1a")
    )


def test_tree_validation():
    main, side = branch(50, 3.0, 51), branch(20, 2.0, 21)
    VesselTree((main, side), (-1, 0), (0, 25))
    with pytest.raises(ValueError, match="earlier branch"):
        VesselTree((main, side), (-1, 1), (0, 25))
    with pytest.raises(ValueError, match="inside its parent"):
        VesselTree((main, side), (-1, 0), (0, 50))
    with pytest.raises(ValueError, match="root"):
        VesselTree((main, side), (0, 0), (0, 25))


def test_segments_and_paths():
    tree = example_tree()
    segs = [(s.branch, s.start, s.end, s.outlets) for s in tree_segments(tree)]
    assert segs == [
        (0, 0, 40, (0, 1, 2, 3)),
        (0, 40, 100, (0, 2)),
        (0, 100, 140, (0,)),
        (1, 0, 30, (1, 3)),
        (1, 30, 80, (1,)),
        (2, 0, 60, (2,)),
        (3, 0, 40, (3,)),
    ]
    assert tree.path(3) == [(0, 0, 40), (1, 0, 30), (3, 0, 40)]
    assert tree.subtree(1) == [1, 3]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_single_branch_reproduces_m1_exactly(seed):
    v = sample_vessel(np.random.default_rng(seed), SamplingConfig())
    m1 = solve_ffr_batch(
        v.profile.r, v.profile.ds, v.profile.reference_radius(), PARAMS,
        pa=v.pa, pv=v.pv, v_hyp=v.v_hyp,
    )  # fmt: skip
    res = solve_tree(VesselTree.single(v.profile), PARAMS, pa=v.pa, pv=v.pv, v_hyp=v.v_hyp)
    np.testing.assert_allclose(res.ffr[0], m1.ffr[0], rtol=1e-12, atol=0)
    assert res.q_outlet[0] == pytest.approx(m1.q[0], rel=1e-12)
    assert res.r_outlet[0] == pytest.approx(m1.r_micro[0], rel=1e-12)
    assert res.q_ref == pytest.approx(m1.q_ref[0], rel=1e-12)


def test_mass_conservation_pressure_continuity_and_outlets():
    tree = example_tree()
    res = solve_tree(tree, PARAMS)
    for k in range(tree.n_branches):
        kids = tree.children(k)
        assert res.q_inlet[k] == pytest.approx(
            res.q_outlet[k] + sum(res.q_inlet[c] for c in kids), rel=1e-12
        )
        for c in kids:  # the side branch starts at its parent's pressure at the take-off
            assert res.pressure[c][0] == pytest.approx(res.pressure[k][tree.attach[c]], rel=1e-14)
        # outlet condition: P_out − Pv = R_out · q_out
        assert res.pressure[k][-1] - res.pv == pytest.approx(
            res.r_outlet[k] * res.q_outlet[k], rel=1e-10
        )
    for p in res.ffr:  # pressure falls along every branch and stays above Pv/Pa
        assert np.all(np.diff(p) <= 1e-15) and p[-1] > res.pv / res.pa


def test_healthy_tree_splits_flow_by_murrays_law():
    tree = example_tree().healthy()
    res = solve_tree(tree, PARAMS)
    k_exp = PARAMS.murray_exponent
    assert res.q_inlet[0] == pytest.approx(res.q_ref, rel=1e-12)
    np.testing.assert_allclose(res.q_outlet, res.q_outlet_healthy, rtol=1e-10)
    for c in range(1, tree.n_branches):
        p, t = tree.parent[c], tree.attach[c]
        later = [x for x in tree.children(p) if tree.attach[x] > t]
        q_cont = res.q_outlet[p] + sum(res.q_inlet[x] for x in later)
        d_side = 2 * tree.branches[c].reference_radius()[0]
        d_cont = 2 * tree.branches[p].reference_radius()[t + 1]
        assert res.q_inlet[c] / q_cont == pytest.approx((d_side / d_cont) ** k_exp, rel=1e-9)


def test_side_branch_raises_distal_main_ffr():
    main = branch(70, 3.0, 141, lesion=(50, 10, 0.6))
    alone = solve_tree(VesselTree.single(main), PARAMS)
    with_side = solve_tree(VesselTree((main, branch(30, 2.0, 61)), (-1, 0), (0, 30)), PARAMS)
    assert with_side.ffr[0][-1] > alone.ffr[0][-1] + 0.02
    # the lesion lowers the inflow below the healthy tree's, but less than for a single path
    assert alone.q_inlet[0] < with_side.q_inlet[0] < with_side.q_ref


def test_vanishing_side_branch_inside_a_lesion_recovers_m1():
    """A side branch that carries almost no flow, taking off inside the lesion's expansion
    run, must not change the main vessel: the split run keeps its full expansion loss."""
    main = cosine_lesion_profile(0.06, 3.0e-3, 0.012, 0.7, n=121)
    throat = int(np.argmin(main.r))
    tiny = branch(20, 0.05, 41)
    tree = VesselTree((main, tiny), (-1, 0), (0, throat + 3))
    m1 = solve_ffr(main, PARAMS)
    res = solve_tree(tree, PARAMS)
    assert res.q_outlet[1] / res.q_outlet[0] < 1e-3
    np.testing.assert_allclose(res.ffr[0], m1.ffr, atol=2e-4)


def test_batch_matches_single_solves_and_outlet_resistances_can_be_reused():
    trees = [example_tree(), VesselTree.single(branch(50, 3.0, 51, lesion=(25, 8, 0.7)))]
    batch = solve_tree_batch(trees, PARAMS, pa=[12000.0, 11000.0], pv=[600.0, 700.0])
    for tree, res, pa, pv in zip(trees, batch, (12000.0, 11000.0), (600.0, 700.0), strict=True):
        one = solve_tree(tree, PARAMS, pa=pa, pv=pv)
        for a, b in zip(res.ffr, one.ffr, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12)
        again = solve_tree(tree, PARAMS, pa=pa, pv=pv, r_outlet=res.r_outlet)
        np.testing.assert_allclose(again.q_outlet, res.q_outlet, rtol=1e-12)


def test_widening_a_lesion_with_fixed_outlets_raises_ffr():
    tree = example_tree()
    pre = solve_tree(tree, PARAMS)
    radii = [b.r for b in tree.branches]
    radii[0] = np.maximum(radii[0], tree.branches[0].reference_radius())  # "stent" the main
    post = solve_tree(tree.with_radii(radii), PARAMS, r_outlet=pre.r_outlet)
    assert post.ffr[0][-1] > pre.ffr[0][-1]
    assert post.q_inlet[0] > pre.q_inlet[0]


def test_murray_flows_sum_to_the_inlet():
    tree = example_tree()
    q = murray_flows(tree, 2e-6, 2.33)
    assert q.sum() == pytest.approx(2e-6, rel=1e-14) and np.all(q > 0)


def test_variants_solver_matches_explicit_trees():
    tree = example_tree()
    pre = solve_tree(tree, PARAMS)
    rng = np.random.default_rng(0)
    main = tree.branches[0]
    widen = [np.maximum(main.r, main.r_ref * f) for f in rng.uniform(0.5, 1.0, 6)]
    variants = solve_tree_variants(
        tree, {0: np.array(widen)}, PARAMS, pa=pre.pa, pv=pre.pv, r_outlet=pre.r_outlet
    )
    for i, r in enumerate(widen):
        radii = [r] + [b.r for b in tree.branches[1:]]
        ref = solve_tree(tree.with_radii(radii), PARAMS, r_outlet=pre.r_outlet)
        np.testing.assert_allclose(variants[i], ref.ffr_outlets, atol=1e-12)
