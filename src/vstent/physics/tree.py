"""Reduced-order FFR on vessel trees with side branches (P2).

Equations and assumptions are in docs/PHYSICS.md ("Vessel trees"). Units are SI throughout:
m, m³/s, Pa; coefficients as in :mod:`vstent.physics.solver`.

A tree is a set of branches. Branch 0 is the root: its inlet is at Pa. Every other branch ``k``
starts at sample ``attach[k]`` of its parent branch ``parent[k] < k``. Every branch ends in one
outlet, which drains to Pv through an outlet resistance. Each branch keeps one continuous radius
profile, so the M1 coefficients a(s) and b(s) are computed per branch exactly as for a single
vessel. Along a branch the flow is piecewise constant: it drops at every side-branch take-off.
A segment (a branch piece between take-offs) with flow Q loses Δa·Q + Δb·Q², where Δa and Δb
are the increase of a(s) and b(s) over the segment. An expansion that spans a take-off therefore
keeps its full loss, each part with its own flow. Junction losses are neglected.

**Calibration.** In the stenosis-free tree (radii = reference radii) the root carries
Q_ref = v_hyp·π·r_ref,root(0)². At every junction the flow splits between the side branch(es)
and the continuing parent in proportion to diameter^k (Murray's law; reference diameters at the
junction). The healthy pressures (viscous loss only, as in M1) then give every outlet resistance
R_j = (P_j − Pv)/Q_j, floored at ``R_MICRO_MIN_FRACTION``·(Pa − Pv)/Q_j. With one branch this is
exactly the M1 calibration.

**Solve.** The unknowns are the outlet flows q. Segment flows are Q = W·q, with W_ej = 1 when
segment e lies on the path to outlet j. Every outlet satisfies
Σ_e W_ej (A_e Q_e + B_e Q_e²) + R_j q_j = Pa − Pv. These equations are the gradient of a strictly
convex function of q, so the solution is unique; the Jacobian Wᵀ diag(A + 2 B Q) W + diag(R) is
symmetric positive definite. Newton's method starts from the linear solution (B = 0); trees are
solved as one padded batch.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vstent.physics.geometry import VesselProfile
from vstent.physics.solver import (
    R_MICRO_MIN_FRACTION,
    PhysicsParams,
    expansion_coefficient,
    viscous_coefficient,
)

NEWTON_TOL = 1e-13  # relative change of the outlet flows
NEWTON_MAX_ITER = 50


@dataclass(frozen=True)
class VesselTree:
    """Branches with their radius profiles and how they connect.

    Attributes:
        branches: one :class:`VesselProfile` per branch (positions from the branch's own start);
            branch 0 is the root.
        parent: parent branch of every branch (``-1`` for the root; ``parent[k] < k``).
        attach: sample index on the parent where the branch starts (``0`` for the root;
            ``1 ≤ attach[k] ≤ n_parent − 2``, so every take-off is inside its parent).
        names: optional labels, e.g. ``("main", "side 1")``.
    """

    branches: tuple[VesselProfile, ...]
    parent: tuple[int, ...]
    attach: tuple[int, ...]
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.branches)
        object.__setattr__(self, "branches", tuple(self.branches))
        object.__setattr__(self, "parent", tuple(int(p) for p in self.parent))
        object.__setattr__(self, "attach", tuple(int(a) for a in self.attach))
        object.__setattr__(
            self, "names", tuple(self.names) or tuple(f"branch {k}" for k in range(n))
        )
        if n == 0:
            raise ValueError("a tree needs at least one branch")
        if len(self.parent) != n or len(self.attach) != n or len(self.names) != n:
            raise ValueError("parent, attach and names need one entry per branch")
        if self.parent[0] != -1 or self.attach[0] != 0:
            raise ValueError("branch 0 is the root: parent -1, attach 0")
        for k in range(1, n):
            p = self.parent[k]
            if not 0 <= p < k:
                raise ValueError(f"branch {k}: parent must be an earlier branch, got {p}")
            if not 1 <= self.attach[k] <= self.branches[p].n - 2:
                raise ValueError(f"branch {k}: attach index must be inside its parent")

    @classmethod
    def single(cls, profile: VesselProfile, name: str = "main") -> "VesselTree":
        """A tree with one branch (the M1 single-vessel case)."""
        return cls((profile,), (-1,), (0,), (name,))

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    def children(self, k: int) -> list[int]:
        """Branches that start on branch ``k``, in index order."""
        return [c for c in range(k + 1, self.n_branches) if self.parent[c] == k]

    def subtree(self, k: int) -> list[int]:
        """Branch ``k`` and all branches downstream of it."""
        out = [k]
        for c in range(k + 1, self.n_branches):
            if self.parent[c] in out:
                out.append(c)
        return out

    def path(self, j: int) -> list[tuple[int, int, int]]:
        """Root-to-outlet path of branch ``j`` as (branch, first index, last index) pieces."""
        pieces = []
        k, end = j, self.branches[j].n - 1
        while k >= 0:
            start = 0
            pieces.append((k, start, end))
            k, end = self.parent[k], self.attach[k]
        return pieces[::-1]

    def path_offsets(self, j: int) -> list[tuple[int, int, int, float]]:
        """Root-to-outlet path of branch ``j`` as (branch, first index, last index, offset):
        ``offset`` (m) is where the branch starts in path coordinates (distance from the inlet
        along the path), so a branch sample at ``s`` sits at ``offset + s`` on the path."""
        out, offset = [], 0.0
        for k, i0, i1 in self.path(j):
            out.append((k, i0, i1, offset))
            offset += float(self.branches[k].s[i1])
        return out

    def path_length(self, j: int) -> float:
        """Length (m) of the root-to-outlet path of branch ``j``."""
        k, _, i1, offset = self.path_offsets(j)[-1]
        return offset + float(self.branches[k].s[i1])

    def with_radii(self, radii: Sequence[np.ndarray]) -> "VesselTree":
        """Same tree and reference radii with new lumen radii (m), one array per branch."""
        if len(radii) != self.n_branches:
            raise ValueError("one radius array per branch")
        branches = tuple(
            VesselProfile(b.s, r, b.reference_radius())
            for b, r in zip(self.branches, radii, strict=True)
        )
        return VesselTree(branches, self.parent, self.attach, self.names)

    def healthy(self) -> "VesselTree":
        """The stenosis-free tree: every lumen radius set to its reference radius."""
        return self.with_radii([b.reference_radius() for b in self.branches])


@dataclass(frozen=True)
class _Segment:
    branch: int
    start: int  # first sample index on the branch
    end: int  # last sample index
    outlets: tuple[int, ...]  # outlets (= branches) whose flow passes through


def tree_segments(tree: VesselTree) -> list[_Segment]:
    """Branch pieces between take-offs, root first, with the outlets each one feeds."""
    segments = []
    for k in range(tree.n_branches):
        kids = tree.children(k)
        cuts = sorted({tree.attach[c] for c in kids})
        bounds = [0, *cuts, tree.branches[k].n - 1]
        for p0, p1 in zip(bounds[:-1], bounds[1:], strict=True):
            fed = [k]
            for c in kids:
                if tree.attach[c] >= p1:
                    fed += tree.subtree(c)
            segments.append(_Segment(k, p0, p1, tuple(sorted(fed))))
    return segments


def murray_flows(tree: VesselTree, q_root: float, exponent: float) -> np.ndarray:
    """Outlet flows (m³/s) of the stenosis-free tree carrying ``q_root`` at the inlet.

    At each junction the arriving flow splits between the side branch(es) starting there and
    the continuing parent in proportion to diameter^``exponent``: the side branch's reference
    diameter at its start, the parent's one sample after the take-off.
    """
    q_in = np.zeros(tree.n_branches)
    q_out = np.zeros(tree.n_branches)
    q_in[0] = q_root
    for k in range(tree.n_branches):  # parents come before their children
        r_ref = tree.branches[k].reference_radius()
        flow = q_in[k]
        groups: dict[int, list[int]] = {}
        for c in tree.children(k):
            groups.setdefault(tree.attach[c], []).append(c)
        for t in sorted(groups):
            weights = {c: (2.0 * tree.branches[c].reference_radius()[0]) ** exponent
                       for c in groups[t]}  # fmt: skip
            total = sum(weights.values()) + (2.0 * r_ref[t + 1]) ** exponent
            for c, w in weights.items():
                q_in[c] = flow * w / total
            flow -= sum(q_in[c] for c in groups[t])
        q_out[k] = flow
    return q_out


@dataclass(frozen=True, eq=False)
class TreeFFRResult:
    """Steady hyperemic solution on one tree. Arrays per branch follow ``tree.branches``.

    Attributes:
        ffr: FFR = P/Pa at every sample of every branch (-).
        pressure: P at every sample of every branch (Pa).
        q_outlet: outlet flow of every branch (m³/s).
        q_inlet: inflow of every branch (m³/s).
        q_outlet_healthy: outlet flows of the stenosis-free tree (Murray split, m³/s).
        r_outlet: outlet resistance of every branch (Pa·s/m³).
        r_outlet_clamped: whether that resistance hit its floor.
        q_ref: inlet flow of the stenosis-free tree (m³/s).
        pa, pv: aortic and venous pressure (Pa).
        iterations: Newton iterations used for the batch this tree was solved in.
    """

    ffr: tuple[np.ndarray, ...]
    pressure: tuple[np.ndarray, ...]
    q_outlet: np.ndarray
    q_inlet: np.ndarray
    q_outlet_healthy: np.ndarray
    r_outlet: np.ndarray
    r_outlet_clamped: np.ndarray
    q_ref: float
    pa: float
    pv: float
    iterations: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def ffr_outlets(self) -> np.ndarray:
        """FFR at every branch's distal end (-)."""
        return np.array([f[-1] for f in self.ffr])


def _coefficients(
    profiles: Sequence[np.ndarray], ds: Sequence[float], params: PhysicsParams, expansion: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """a(s) and b(s) for many radius arrays, batched by length (units as in the M1 solver)."""
    out: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(profiles)
    by_len: dict[int, list[int]] = {}
    for i, r in enumerate(profiles):
        by_len.setdefault(r.size, []).append(i)
    for idx in by_len.values():
        r = np.stack([profiles[i] for i in idx])
        a = viscous_coefficient(r, np.array([ds[i] for i in idx]), params.mu)
        if expansion:
            b = expansion_coefficient(r, params.rho, params.kt, params.expansion_threshold)
        else:
            b = np.zeros_like(a)
        for row, i in enumerate(idx):
            out[i] = (a[row], b[row])
    return out  # type: ignore[return-value]


def _pressures(
    tree: VesselTree,
    segments: list[_Segment],
    coeffs: list[tuple[np.ndarray, np.ndarray]],
    q_seg: np.ndarray,
    pa: float,
) -> list[np.ndarray]:
    """Pressure (Pa) at every sample of every branch for segment flows ``q_seg`` (m³/s)."""
    pressure: list[np.ndarray | None] = [None] * tree.n_branches
    by_branch: dict[int, list[int]] = {}
    for e, seg in enumerate(segments):
        by_branch.setdefault(seg.branch, []).append(e)
    for k in range(tree.n_branches):  # parents first, so the junction pressure is known
        a, b = coeffs[k]
        p = np.empty(tree.branches[k].n)
        p_start = pa if k == 0 else pressure[tree.parent[k]][tree.attach[k]]
        for e in by_branch[k]:
            seg, q = segments[e], q_seg[e]
            sl = slice(seg.start, seg.end + 1)
            drop = (a[sl] - a[seg.start]) * q + (b[sl] - b[seg.start]) * q * q
            p[sl] = p_start - drop
            p_start = p[seg.end]
        pressure[k] = p
    return pressure  # type: ignore[return-value]


def _newton(
    w: np.ndarray, a: np.ndarray, b: np.ndarray, r: np.ndarray, rhs: np.ndarray
) -> tuple[np.ndarray, int]:
    """Outlet flows of a padded batch: W (T, S, m), A and B (T, S), R and rhs (T, m).

    Padded outlets have an all-zero W column, R = 1 and rhs = 0, so their flow is 0.
    """
    wt = np.swapaxes(w, 1, 2)
    eye = np.eye(w.shape[2])
    real = rhs > 0

    def residual(q):
        qs = np.einsum("tsm,tm->ts", w, q)
        return np.einsum("tms,ts->tm", wt, a * qs + b * qs * qs) + r * q - rhs, qs

    jac0 = wt @ (a[:, :, None] * w) + r[:, :, None] * eye
    q = np.linalg.solve(jac0, rhs[..., None])[..., 0]  # linear losses only: an upper bound
    for it in range(1, NEWTON_MAX_ITER + 1):
        f, qs = residual(q)
        jac = wt @ ((a + 2.0 * b * qs)[:, :, None] * w) + r[:, :, None] * eye
        step = np.linalg.solve(jac, f[..., None])[..., 0]
        lam = np.ones(len(q))
        for _ in range(60):  # damp only if a step would make a real outlet flow non-positive
            trial = q - lam[:, None] * step
            bad = np.any(real & (trial <= 0.0), axis=1)
            if not bad.any():
                break
            lam[bad] *= 0.5
        q = q - lam[:, None] * step
        scale = np.maximum(np.max(np.abs(q), axis=1), 1e-300)
        if np.all(np.max(np.abs(lam[:, None] * step), axis=1) <= NEWTON_TOL * scale):
            return q, it
    raise RuntimeError("tree Newton iteration did not converge")


def solve_tree_batch(
    trees: Sequence[VesselTree],
    params: PhysicsParams | None = None,
    *,
    pa=None,
    pv=None,
    v_hyp=None,
    r_outlet: Sequence[np.ndarray] | None = None,
) -> list[TreeFFRResult]:
    """Solve FFR on many trees at once.

    Args:
        trees: the trees.
        params: model constants (``murray_exponent`` sets the healthy flow split).
        pa, pv: aortic / venous pressure (Pa), scalar or one per tree; default from ``params``.
        v_hyp: hyperemic reference velocity (m/s), scalar or one per tree.
        r_outlet: outlet resistances (Pa·s/m³) to use instead of the calibration, one array per
            tree, e.g. after a virtual stent when the microvasculature is unchanged.
    """
    params = params or PhysicsParams()
    n = len(trees)
    if n == 0:
        return []
    pa = np.broadcast_to(np.asarray(params.pa if pa is None else pa, float), (n,))
    pv = np.broadcast_to(np.asarray(params.pv if pv is None else pv, float), (n,))
    v_hyp = np.broadcast_to(np.asarray(params.v_hyp if v_hyp is None else v_hyp, float), (n,))
    if np.any(pv >= pa) or np.any(v_hyp <= 0):
        raise ValueError("need pv < pa and v_hyp > 0")

    flat = [b for t in trees for b in t.branches]
    ds = [b.ds for b in flat]
    coeffs = _coefficients([b.r for b in flat], ds, params, expansion=True)
    coeffs_ref = _coefficients([b.reference_radius() for b in flat], ds, params, expansion=False)

    segs, offsets, calib = [], [], []
    start = 0
    for i, tree in enumerate(trees):
        offsets.append(start)
        segments = tree_segments(tree)
        segs.append(segments)
        q_ref = v_hyp[i] * np.pi * tree.branches[0].reference_radius()[0] ** 2
        q_h = murray_flows(tree, q_ref, params.murray_exponent)
        if r_outlet is None:
            ref = coeffs_ref[start : start + tree.n_branches]
            q_seg = np.array([q_h[list(s.outlets)].sum() for s in segments])
            p_out = np.array([p[-1] for p in _pressures(tree, segments, ref, q_seg, pa[i])])
            r = (p_out - pv[i]) / q_h
            floor = R_MICRO_MIN_FRACTION * (pa[i] - pv[i]) / q_h
            clamped = r < floor
            r = np.maximum(r, floor)
        else:
            r = np.asarray(r_outlet[i], float)
            if r.shape != (tree.n_branches,) or np.any(r <= 0):
                raise ValueError("r_outlet: one positive resistance per branch")
            clamped = np.zeros(tree.n_branches, bool)
        calib.append((q_ref, q_h, r, clamped))
        start += tree.n_branches

    s_max = max(len(s) for s in segs)
    m_max = max(t.n_branches for t in trees)
    w = np.zeros((n, s_max, m_max))
    a_seg, b_seg = np.zeros((n, s_max)), np.zeros((n, s_max))
    r_pad, rhs = np.ones((n, m_max)), np.zeros((n, m_max))
    for i, (tree, segments) in enumerate(zip(trees, segs, strict=True)):
        for e, s in enumerate(segments):
            a, b = coeffs[offsets[i] + s.branch]
            w[i, e, list(s.outlets)] = 1.0
            a_seg[i, e] = a[s.end] - a[s.start]
            b_seg[i, e] = b[s.end] - b[s.start]
        r_pad[i, : tree.n_branches] = calib[i][2]
        rhs[i, : tree.n_branches] = pa[i] - pv[i]
    q, iterations = _newton(w, a_seg, b_seg, r_pad, rhs)

    results = []
    for i, (tree, segments) in enumerate(zip(trees, segs, strict=True)):
        q_out = q[i, : tree.n_branches]
        q_seg = np.array([q_out[list(s.outlets)].sum() for s in segments])
        own = coeffs[offsets[i] : offsets[i] + tree.n_branches]
        pressure = _pressures(tree, segments, own, q_seg, pa[i])
        q_ref, q_h, r, clamped = calib[i]
        results.append(
            TreeFFRResult(
                ffr=tuple(p / pa[i] for p in pressure),
                pressure=tuple(pressure),
                q_outlet=q_out.copy(),
                q_inlet=np.array([q_out[tree.subtree(k)].sum() for k in range(tree.n_branches)]),
                q_outlet_healthy=q_h,
                r_outlet=r,
                r_outlet_clamped=clamped,
                q_ref=float(q_ref),
                pa=float(pa[i]),
                pv=float(pv[i]),
                iterations=iterations,
            )
        )
    return results


def solve_tree(
    tree: VesselTree,
    params: PhysicsParams | None = None,
    *,
    pa: float | None = None,
    pv: float | None = None,
    v_hyp: float | None = None,
    r_outlet: np.ndarray | None = None,
) -> TreeFFRResult:
    """Solve FFR on one tree (see :func:`solve_tree_batch`)."""
    return solve_tree_batch(
        [tree], params, pa=pa, pv=pv, v_hyp=v_hyp,
        r_outlet=None if r_outlet is None else [r_outlet],
    )[0]  # fmt: skip
