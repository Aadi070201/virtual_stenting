"""Virtual stent plan optimiser.

Candidates: catalogue diameter × length × start position (``start_step`` grid) such that the
stent covers the MLD of a lesion (DS > 30%, M1 definition). Constraints:

- the stent lies inside the vessel;
- nominal diameter ≤ ``max_diameter_ratio`` × reference diameter at the distal stent end;
- DS < ``landing_max_ds_pct`` over the first and last ``landing_zone`` of the stent.

Objective: among plans whose post-PCI distal FFR reaches ``target_ffr``, the shortest total
stent length (ties: higher FFR); if none reaches it, the highest FFR. Up to two stents: pairs of
the best single-stent candidates of two different lesions, without overlap.

Plans are evaluated in batches by the ONNX surrogate or by the reduced-order solver on a
256-point planning grid. Pa, Pv and R_micro keep their pre-PCI values: a stent changes the
epicardial geometry, not the microvasculature. With the surrogate backend the returned plans are
re-checked with the solver (``Plan.ffr_solver``).

Lengths in metres internally; the config files use mm.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from pydantic import Field, field_validator

from vstent.config import StrictModel
from vstent.inference.onnx_runtime import SurrogateRunner
from vstent.physics.geometry import Lesion, VesselProfile, percent_diameter_stenosis
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.planning.stent import Stent, stent_weight, stented_radius
from vstent.surrogate.features import N_POINTS, RawInputs
from vstent.units import mm_to_m

LESION_THRESHOLD_PCT = 30.0


class StentCatalog(StrictModel):
    """Generic catalogue (``configs/stent_catalog.yaml``): nominal sizes in mm."""

    diameters_mm: list[float]
    lengths_mm: list[float]

    @field_validator("diameters_mm", "lengths_mm")
    @classmethod
    def _positive_sorted(cls, v: list[float]) -> list[float]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("catalogue sizes must be positive and non-empty")
        return sorted(set(v))


class PlannerConfig(StrictModel):
    """Planner settings (``configs/planner.yaml``); lengths in mm."""

    target_ffr: float = Field(0.90, gt=0, lt=1)
    start_step_mm: float = Field(0.5, gt=0)
    max_diameter_ratio: float = Field(1.1, gt=0)
    landing_zone_mm: float = Field(2.0, ge=0)
    landing_max_ds_pct: float = Field(30.0, gt=0, lt=100)
    edge_mm: float = Field(1.0, ge=0)
    expansion: float = Field(1.0, gt=0)
    max_stents: int = Field(2, ge=1, le=2)
    top_k_per_lesion: int = Field(20, ge=1)
    n_plans: int = Field(3, ge=1)
    backend: Literal["surrogate", "solver"] = "surrogate"


@dataclass(frozen=True)
class Plan:
    """One stent plan and its predicted post-PCI distal FFR."""

    stents: tuple[Stent, ...]
    ffr: float
    meets_target: bool
    ffr_solver: float | None = None

    @property
    def total_length(self) -> float:
        """Sum of stent lengths (m)."""
        return float(sum(st.length for st in self.stents))


@dataclass(frozen=True, eq=False)
class PlanResult:
    """Output of :func:`plan_stents`. Curves are on ``grid.s``."""

    grid: VesselProfile
    lesions: list[Lesion]
    backend: str
    target: float
    pre_ffr_curve: np.ndarray
    plans: list[Plan]
    pareto: list[Plan]
    best_ffr_curve: np.ndarray | None
    n_candidates: int
    n_feasible: int
    n_pairs: int
    seconds: float
    message: str
    pa: float
    pv: float
    r_micro: float
    extra: dict = field(default_factory=dict)

    @property
    def pre_ffr(self) -> float:
        """Pre-PCI distal FFR (planning backend)."""
        return float(self.pre_ffr_curve[-1])


class Backend(Protocol):
    name: str

    def curves(self, r: np.ndarray) -> np.ndarray: ...


class SolverBackend:
    """Reduced-order solver with fixed Pa, Pv, R_micro on the planning grid."""

    name = "solver"

    def __init__(self, grid: VesselProfile, params: PhysicsParams, pa, pv, r_micro) -> None:
        self.grid, self.params = grid, params
        self.pa, self.pv, self.r_micro = pa, pv, r_micro
        self.r_ref = grid.reference_radius()

    def curves(self, r: np.ndarray) -> np.ndarray:
        """FFR curves (B, N) for radii ``r`` (B, N) in m."""
        r = np.atleast_2d(r)
        res = solve_ffr_batch(
            r,
            self.grid.ds,
            np.broadcast_to(self.r_ref, r.shape),
            self.params,
            pa=self.pa,
            pv=self.pv,
            r_micro=self.r_micro,
        )
        return res.ffr


class SurrogateBackend:
    """ONNX surrogate with fixed Pa, Pv, R_micro on the planning grid (N = 256)."""

    name = "surrogate"

    def __init__(self, runner: SurrogateRunner, grid: VesselProfile, pa, pv, r_micro) -> None:
        if grid.n != runner.n_points:
            raise ValueError(f"planning grid has {grid.n} points, model needs {runner.n_points}")
        self.runner, self.grid = runner, grid
        self.pa, self.pv, self.r_micro = pa, pv, r_micro
        self.r_ref = grid.reference_radius()

    def curves(self, r: np.ndarray) -> np.ndarray:
        """FFR curves (B, N) for radii ``r`` (B, N) in m."""
        r = np.atleast_2d(r)
        b = r.shape[0]
        raw = RawInputs(
            r=r,
            r_ref=np.broadcast_to(self.r_ref, r.shape),
            length=np.full(b, self.grid.length),
            pa=np.full(b, self.pa),
            pv=np.full(b, self.pv),
            r_micro=np.full(b, self.r_micro),
        )
        return self.runner.predict(raw).astype(np.float64)


def planning_grid(profile: VesselProfile, n: int = N_POINTS) -> VesselProfile:
    """``profile`` on ``n`` uniform points with its reference radius made explicit."""
    s = np.linspace(profile.s[0], profile.s[-1], n)
    return VesselProfile(
        s - s[0],
        np.interp(s, profile.s, profile.r),
        np.interp(s, profile.s, profile.reference_radius()),
    )


@dataclass(frozen=True)
class Candidates:
    """Single-stent candidates: start, length and nominal diameter arrays (m)."""

    s_start: np.ndarray
    length: np.ndarray
    diameter: np.ndarray

    def __len__(self) -> int:
        return self.s_start.size

    def subset(self, idx) -> "Candidates":
        return Candidates(self.s_start[idx], self.length[idx], self.diameter[idx])

    @property
    def s_end(self) -> np.ndarray:
        return self.s_start + self.length


def generate_candidates(
    grid: VesselProfile, lesions: Sequence[Lesion], catalog: StentCatalog, config: PlannerConfig
) -> Candidates:
    """All catalogue sizes × start positions (on the step grid) that cover a lesion MLD.

    Duplicates (a stent covering several MLDs) are listed once. No constraint applied yet.
    """
    step = mm_to_m(config.start_step_mm)
    lengths = mm_to_m(np.asarray(catalog.lengths_mm))
    diameters = mm_to_m(np.asarray(catalog.diameters_mm))
    rows = set()
    for les in lesions:
        for length in lengths:
            k0 = int(np.ceil((les.s_mld - length) / step - 1e-9))
            k1 = int(np.floor(les.s_mld / step + 1e-9))
            for k in range(k0, k1 + 1):
                rows.add((round(k * step, 9), float(length)))
    if not rows:
        empty = np.empty(0)
        return Candidates(empty, empty, empty)
    starts, lens = np.array(sorted(rows)).T
    n_d = diameters.size
    return Candidates(np.repeat(starts, n_d), np.repeat(lens, n_d), np.tile(diameters, starts.size))


def feasible(grid: VesselProfile, cand: Candidates, config: PlannerConfig) -> np.ndarray:
    """Boolean mask of candidates that satisfy all constraints."""
    s, s0, s1 = grid.s, cand.s_start, cand.s_end
    tol = 1e-9
    inside = (s0 >= s[0] - tol) & (s1 <= s[-1] + tol)
    ref_d = 2.0 * grid.reference_radius()
    distal_ref = np.interp(s1, s, ref_d)
    sized = cand.diameter <= config.max_diameter_ratio * distal_ref + tol
    ds = percent_diameter_stenosis(grid.r, grid.reference_radius())
    zone = mm_to_m(config.landing_zone_mm)
    landing_ok = _window_max(s, ds, s0, s0 + zone) < config.landing_max_ds_pct
    landing_ok &= _window_max(s, ds, s1 - zone, s1) < config.landing_max_ds_pct
    return inside & sized & landing_ok


def _window_max(s: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """max of y over [a, b] for each window: grid points inside plus the interpolated ends."""
    i0 = np.searchsorted(s, a, side="left")
    i1 = np.searchsorted(s, b, side="right")
    # sparse table for O(1) range maxima over the grid points
    levels = [y]
    while (1 << len(levels)) <= y.size:
        prev, w = levels[-1], 1 << (len(levels) - 1)
        levels.append(np.maximum(prev[:-w], prev[w:]))
    ends = np.maximum(np.interp(a, s, y), np.interp(b, s, y))
    has = i1 > i0
    n = np.where(has, i1 - i0, 1)
    k = np.floor(np.log2(n)).astype(int)
    out = ends.copy()
    for lv in np.unique(k[has]):
        sel = has & (k == lv)
        table = levels[lv]
        lo, hi = i0[sel], i1[sel] - (1 << lv)
        out[sel] = np.maximum(out[sel], np.maximum(table[lo], table[hi]))
    return out


def stented_radii(
    grid: VesselProfile, stents: Sequence[Candidates], config: PlannerConfig
) -> np.ndarray:
    """Radii (C, N) with one stent from each of ``stents`` (aligned candidate sets) implanted."""
    r = grid.r
    edge = mm_to_m(config.edge_mm)
    out = np.broadcast_to(r, (len(stents[0]), r.size)).copy()
    for c in stents:
        w = stent_weight(grid.s, c.s_start, c.s_end, edge)
        target = config.expansion * c.diameter[:, None] / 2.0
        out = np.maximum(out, r + w * np.maximum(0.0, target - r))
    return out


def _covers(cand: Candidates, lesion: Lesion) -> np.ndarray:
    return (cand.s_start <= lesion.s_mld + 1e-9) & (cand.s_end >= lesion.s_mld - 1e-9)


def _pool(cand: Candidates, ffr: np.ndarray, lesion: Lesion, top_k: int) -> np.ndarray:
    """Candidates covering ``lesion`` kept for pairing: the best by FFR *for every length*.

    Ranking by FFR alone would keep only the longest stents and miss short two-stent plans.
    """
    idx = np.flatnonzero(_covers(cand, lesion))
    lengths = np.unique(cand.length[idx])
    per_length = max(1, top_k // max(lengths.size, 1))
    keep = []
    for length in lengths:
        sel = idx[cand.length[idx] == length]
        keep.extend(sel[np.argsort(-ffr[sel])][:per_length])
    return np.array(keep, dtype=int)


def _to_plans(
    sets: Sequence[Candidates], ffr: np.ndarray, idx, config: PlannerConfig
) -> list[Plan]:
    plans = []
    for i in np.atleast_1d(idx):
        stents = tuple(
            Stent(float(c.s_start[i]), float(c.length[i]), float(c.diameter[i]), config.expansion)
            for c in sets
        )
        plans.append(
            Plan(
                tuple(sorted(stents, key=lambda st: st.s_start)),
                float(ffr[i]),
                bool(ffr[i] >= config.target_ffr),
            )
        )
    return plans


def _rank(plans: list[Plan]) -> list[Plan]:
    """Target-reaching plans first, by total length then FFR; then the rest by FFR."""
    return sorted(
        plans,
        key=lambda p: (not p.meets_target, p.total_length if p.meets_target else 0.0, -p.ffr),
    )


def pareto_front(plans: Sequence[Plan]) -> list[Plan]:
    """Plans not dominated in (shorter total length, higher FFR), sorted by length."""
    front, best = [], -np.inf
    for p in sorted(plans, key=lambda p: (p.total_length, -p.ffr)):
        if p.ffr > best + 1e-12:
            front.append(p)
            best = p.ffr
    return front


def plan_stents(
    profile: VesselProfile,
    *,
    catalog: StentCatalog,
    config: PlannerConfig,
    params: PhysicsParams | None = None,
    runner: SurrogateRunner | None = None,
    pa: float | None = None,
    pv: float | None = None,
) -> PlanResult:
    """Best stent plans for ``profile`` (SI). Pressures in Pa (default: ``params``)."""
    t0 = time.perf_counter()
    params = params or PhysicsParams()
    grid = planning_grid(profile)
    pre = solve_ffr_batch(grid.r, grid.ds, grid.reference_radius(), params, pa=pa, pv=pv)
    pa_, pv_, rm = float(pre.pa[0]), float(pre.pv[0]), float(pre.r_micro[0])
    solver = SolverBackend(grid, params, pa_, pv_, rm)
    if config.backend == "surrogate":
        if runner is None:
            raise ValueError("the surrogate backend needs a SurrogateRunner")
        backend: Backend = SurrogateBackend(runner, grid, pa_, pv_, rm)
    else:
        backend = solver
    pre_curve = backend.curves(grid.r[None])[0]
    lesions = grid.lesions(LESION_THRESHOLD_PCT)

    def result(plans, pareto, best_curve, n_cand, n_feas, n_pairs, message):
        return PlanResult(
            grid,
            lesions,
            backend.name,
            config.target_ffr,
            pre_curve,
            plans,
            pareto,
            best_curve,
            n_cand,
            n_feas,
            n_pairs,
            time.perf_counter() - t0,
            message,
            pa_,
            pv_,
            rm,
        )

    if not lesions:
        return result([], [], None, 0, 0, 0, "No lesion with DS > 30%: no stent proposed.")
    cand = generate_candidates(grid, lesions, catalog, config)
    ok = feasible(grid, cand, config)
    single = cand.subset(ok)
    if len(single) == 0:
        return result(
            [],
            [],
            None,
            len(cand),
            0,
            0,
            "No candidate satisfies the landing-zone and sizing constraints.",
        )
    ffr_single = backend.curves(stented_radii(grid, [single], config))[:, -1]
    all_plans = _to_plans([single], ffr_single, np.arange(len(single)), config)

    n_pairs = 0
    if config.max_stents >= 2 and len(lesions) >= 2:
        pools = [_pool(single, ffr_single, les, config.top_k_per_lesion) for les in lesions]
        # A stent covering several MLDs sits in several pools: keep each pair once.
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
            first, second = single.subset(ii), single.subset(jj)
            ffr_pair = backend.curves(stented_radii(grid, [first, second], config))[:, -1]
            all_plans += _to_plans([first, second], ffr_pair, np.arange(len(pairs)), config)
            n_pairs = len(pairs)

    ranked = _rank(all_plans)[: config.n_plans]
    if backend.name != "solver":  # re-check the proposed plans with the solver
        checked = solver.curves(_radii_for(grid, ranked, config))[:, -1]
        ranked = [
            Plan(p.stents, p.ffr, p.meets_target, float(f))
            for p, f in zip(ranked, checked, strict=True)
        ]
    best_curve = backend.curves(_radii_for(grid, ranked[:1], config))[0]
    if ranked[0].meets_target:
        message = f"Shortest plan reaching FFR >= {config.target_ffr:.2f}."
    else:
        message = (
            f"No plan reaches FFR >= {config.target_ffr:.2f}; the plan with the highest predicted "
            "FFR is shown."
        )
    return result(
        ranked, pareto_front(all_plans), best_curve, len(cand), len(single), n_pairs, message
    )


def _radii_for(grid: VesselProfile, plans: Sequence[Plan], config: PlannerConfig) -> np.ndarray:
    """Radii (len(plans), N) with each plan's stents implanted."""
    edge = mm_to_m(config.edge_mm)
    return np.array([stented_radius(grid.s, grid.r, p.stents, edge) for p in plans])
