"""Virtual stent planning on vessel trees (P3): stents along one root-to-outlet path.

The path to the chosen outlet is treated like a single vessel for lesions, candidates and the
landing-zone and sizing constraints (M6), on a 256-point grid of path coordinates (distance
from the inlet along the path). A candidate widens only the branch pieces of that path, as
``planning.stent.apply_path_stents`` does; FFR at every outlet then follows with the pre-PCI
outlet resistances. The tree solver evaluates the candidates by default
(:func:`solve_tree_variants`: the topology is shared, so 5,000 candidates take tens of ms); the
tree surrogate can be used instead and is slower than that solver (CLAUDE.md). Plans are ranked
by the FFR at the chosen outlet exactly as in M6; the other outlets' FFR comes with each plan.
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vstent.inference.tree_runtime import TreeSurrogateRunner
from vstent.physics.geometry import Lesion, VesselProfile
from vstent.physics.solver import PhysicsParams
from vstent.physics.tree import VesselTree, solve_tree, solve_tree_variants
from vstent.planning.optimizer import (
    LESION_THRESHOLD_PCT,
    Candidates,
    Plan,
    PlannerConfig,
    StentCatalog,
    _pool,
    _rank,
    feasible,
    generate_candidates,
    pareto_front,
)
from vstent.planning.stent import Stent, apply_path_stents, stent_weight
from vstent.surrogate.tree_features import N_POINTS, TreeRawInputs
from vstent.units import mm_to_m

TreeBackendName = Literal["solver", "surrogate"]


@dataclass(frozen=True)
class TreePlan(Plan):
    """A plan on a tree path: ``ffr`` is the FFR at the chosen outlet; ``ffr_outlets`` at every
    outlet (same backend). Stent positions are path coordinates (m)."""

    ffr_outlets: tuple[float, ...] = ()


@dataclass(frozen=True, eq=False)
class TreePlanResult:
    """Output of :func:`plan_tree_stents`. ``path`` is the planning grid along the path."""

    tree: VesselTree
    outlet: int
    path: VesselProfile
    lesions: list[Lesion]
    backend: str
    target: float
    pre_ffr_outlets: np.ndarray
    plans: list[TreePlan]
    pareto: list[TreePlan]
    n_candidates: int
    n_feasible: int
    n_pairs: int
    seconds: float
    message: str
    pa: float
    pv: float
    r_outlet: np.ndarray
    edge: float

    @property
    def pre_ffr(self) -> float:
        """Pre-PCI FFR at the chosen outlet (planning backend)."""
        return float(self.pre_ffr_outlets[self.outlet])


def path_profile(tree: VesselTree, outlet: int, n: int = N_POINTS) -> VesselProfile:
    """The root-to-outlet path of branch ``outlet`` as one profile on ``n`` uniform points:
    radius and reference radius along path coordinates (m). At a take-off the side branch's
    first sample is used."""
    xs, rs, refs = [], [], []
    pieces = tree.path_offsets(outlet)
    for j, (k, i0, i1, offset) in enumerate(pieces):
        b = tree.branches[k]
        end = i1 + 1 if j == len(pieces) - 1 else i1  # the next piece starts at the take-off
        xs.append(offset + b.s[i0:end])
        rs.append(b.r[i0:end])
        refs.append(b.reference_radius()[i0:end])
    x, r, ref = (np.concatenate(v) for v in (xs, rs, refs))
    s = np.linspace(0.0, x[-1], n)
    return VesselProfile(s, np.interp(s, x, r), np.interp(s, x, ref))


def path_radii(
    tree: VesselTree, outlet: int, sets: list[Candidates], config: PlannerConfig
) -> dict[int, np.ndarray]:
    """Radii (C, n_k) of every branch on the path with one stent from each candidate set
    implanted (aligned sets; positions in path coordinates, m)."""
    edge = mm_to_m(config.edge_mm)
    out = {}
    for k, i0, i1, offset in tree.path_offsets(outlet):
        b = tree.branches[k]
        inside = np.zeros(b.n)
        inside[i0 : i1 + 1] = 1.0
        res = np.broadcast_to(b.r, (len(sets[0]), b.n)).copy()
        for c in sets:
            w = stent_weight(b.s, c.s_start - offset, c.s_end - offset, edge) * inside
            target = config.expansion * c.diameter[:, None] / 2.0
            res = np.maximum(res, b.r + w * np.maximum(0.0, target - b.r))
        out[k] = res
    return out


class TreeSolverBackend:
    """Tree solver with fixed pressures and outlet resistances."""

    name = "solver"

    def __init__(self, tree, params, pa, pv, r_outlet) -> None:
        self.tree, self.params = tree, params
        self.pa, self.pv, self.r_outlet = pa, pv, r_outlet

    def outlets(self, radii: dict[int, np.ndarray]) -> np.ndarray:
        """FFR at every outlet (C, n_branches)."""
        return solve_tree_variants(
            self.tree, radii, self.params, pa=self.pa, pv=self.pv, r_outlet=self.r_outlet
        )


class TreeSurrogateBackend:
    """Tree surrogate (ONNX) with fixed pressures and outlet resistances. Radii on the tree's
    own points are resampled to the model's grid, branch by branch."""

    name = "surrogate"

    def __init__(self, runner: TreeSurrogateRunner, tree, pa, pv, r_outlet) -> None:
        self.runner, self.tree = runner, tree
        self.base = TreeRawInputs.from_trees(
            [tree], pa, pv, [r_outlet], n=runner.n_points, max_branches=runner.max_branches
        )

    def outlets(self, radii: dict[int, np.ndarray]) -> np.ndarray:
        c = next(iter(radii.values())).shape[0] if radii else 1
        rep = {
            k: np.repeat(getattr(self.base, k), c, axis=0) for k in self.base.__dataclass_fields__
        }
        for k, rk in radii.items():
            b = self.tree.branches[k]
            grid = np.linspace(0.0, b.length, self.runner.n_points)
            rep["r"][:, k] = np.array([np.interp(grid, b.s, row) for row in rk])
        ffr = self.runner.predict(TreeRawInputs(**rep))
        return ffr[:, : self.tree.n_branches, -1].astype(np.float64)


def _plans(sets, ffr_out: np.ndarray, outlet: int, idx, config: PlannerConfig) -> list[TreePlan]:
    plans = []
    for i in np.atleast_1d(idx):
        stents = tuple(
            Stent(float(c.s_start[i]), float(c.length[i]), float(c.diameter[i]), config.expansion)
            for c in sets
        )
        f = float(ffr_out[i, outlet])
        plans.append(
            TreePlan(
                tuple(sorted(stents, key=lambda st: st.s_start)), f, bool(f >= config.target_ffr),
                ffr_outlets=tuple(float(x) for x in ffr_out[i]),
            )
        )  # fmt: skip
    return plans


def _stack_radii(tree, outlet, plans, edge) -> dict[int, np.ndarray]:
    """Path-branch radii of explicit plans (few), via ``apply_path_stents``."""
    trees = [apply_path_stents(tree, outlet, p.stents, edge=edge) for p in plans]
    return {k: np.array([t.branches[k].r for t in trees]) for k, *_ in tree.path_offsets(outlet)}


def plan_tree_stents(
    tree: VesselTree,
    outlet: int = 0,
    *,
    catalog: StentCatalog,
    config: PlannerConfig,
    params: PhysicsParams | None = None,
    backend: TreeBackendName = "solver",
    runner: TreeSurrogateRunner | None = None,
    pa: float | None = None,
    pv: float | None = None,
) -> TreePlanResult:
    """Best stent plans along the path to branch ``outlet`` (SI; pressures in Pa)."""
    t0 = time.perf_counter()
    params = params or PhysicsParams()
    pa = params.pa if pa is None else pa
    pv = params.pv if pv is None else pv
    if not 0 <= outlet < tree.n_branches:
        raise ValueError(f"outlet {outlet}: the tree has {tree.n_branches} branches")
    pre = solve_tree(tree, params, pa=pa, pv=pv)
    solver = TreeSolverBackend(tree, params, pa, pv, pre.r_outlet)
    if backend == "surrogate":
        if runner is None:
            raise ValueError("the surrogate backend needs a TreeSurrogateRunner")
        be = TreeSurrogateBackend(runner, tree, pa, pv, pre.r_outlet)
        pre_out = be.outlets({})[0]
    else:
        be, pre_out = solver, pre.ffr_outlets
    edge = mm_to_m(config.edge_mm)
    grid = path_profile(tree, outlet)
    lesions = grid.lesions(LESION_THRESHOLD_PCT)

    def result(plans, pareto, n_cand, n_feas, n_pairs, message):
        return TreePlanResult(
            tree, outlet, grid, lesions, be.name, config.target_ffr, np.asarray(pre_out),
            plans, pareto, n_cand, n_feas, n_pairs, time.perf_counter() - t0, message,
            float(pa), float(pv), pre.r_outlet, edge,
        )  # fmt: skip

    if not lesions:
        return result([], [], 0, 0, 0, "No lesion with DS > 30% on this path: no stent proposed.")
    cand = generate_candidates(grid, lesions, catalog, config)
    single = cand.subset(feasible(grid, cand, config))
    if len(single) == 0:
        message = "No candidate satisfies the landing-zone and sizing constraints."
        return result([], [], len(cand), 0, 0, message)
    out_single = be.outlets(path_radii(tree, outlet, [single], config))
    all_plans = _plans([single], out_single, outlet, np.arange(len(single)), config)
    n_pairs = 0
    if config.max_stents >= 2 and len(lesions) >= 2:
        ffr_single = out_single[:, outlet]
        pools = [_pool(single, ffr_single, les, config.top_k_per_lesion) for les in lesions]
        pairs = sorted(
            {
                (min(i, j), max(i, j))
                for a in range(len(pools))
                for b in range(a + 1, len(pools))
                for i in pools[a]
                for j in pools[b]
                if single.s_end[i] <= single.s_start[j] + 1e-9
                or single.s_end[j] <= single.s_start[i] + 1e-9
            }
        )
        if pairs:
            ii, jj = np.array(pairs).T
            sets = [single.subset(ii), single.subset(jj)]
            out_pair = be.outlets(path_radii(tree, outlet, sets, config))
            all_plans += _plans(sets, out_pair, outlet, np.arange(len(pairs)), config)
            n_pairs = len(pairs)
    ranked = _rank(all_plans)[: config.n_plans]
    if be is not solver:  # re-check the proposed plans with the tree solver
        checked = solver.outlets(_stack_radii(tree, outlet, ranked, edge))[:, outlet]
        ranked = [
            TreePlan(p.stents, p.ffr, p.meets_target, float(f), ffr_outlets=p.ffr_outlets)
            for p, f in zip(ranked, checked, strict=True)
        ]
    if ranked[0].meets_target:
        message = f"Shortest plan reaching FFR >= {config.target_ffr:.2f} at {tree.names[outlet]}."
    else:
        message = (
            f"No plan reaches FFR >= {config.target_ffr:.2f} at {tree.names[outlet]}; the plan "
            "with the highest predicted FFR is shown."
        )
    return result(ranked, pareto_front(all_plans), len(cand), len(single), n_pairs, message)
