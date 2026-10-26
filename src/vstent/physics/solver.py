"""Reduced-order FFR solver: ΔP(Q) = a·Q + b·Q² along a single vessel.

Equations and assumptions are in docs/PHYSICS.md. Units are SI throughout: m, m², m³/s, Pa,
Pa·s, kg/m³. Coefficients: a in Pa·s/m³, b in Pa·s²/m⁶, R_micro in Pa·s/m³.

The vectorised :func:`solve_ffr_batch` is the single implementation; :func:`solve_ffr` is a
batch of one.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from vstent.config import StrictModel
from vstent.physics.geometry import VesselProfile
from vstent.units import mmhg_to_pa

FlowMode = Literal["resistance", "fixed_velocity"]

R_MICRO_MIN_FRACTION = 0.01
"""Floor on R_micro as a fraction of (Pa − Pv)/Q_ref. A numerical safeguard for the spec's
"clamp R_micro > 0"; it is only reached when the reference vessel's own viscous loss at Q_ref
would exceed Pa − Pv."""


class PhysicsParams(StrictModel):
    """Constants of the reduced-order model.

    Pressures are entered in mmHg (config boundary); use :attr:`pa` / :attr:`pv` for Pa.
    """

    mu: float = Field(3.5e-3, gt=0, description="blood dynamic viscosity (Pa·s)")
    rho: float = Field(1060.0, gt=0, description="blood density (kg/m³)")
    pa_mmhg: float = Field(90.0, gt=0, description="mean aortic pressure (mmHg)")
    pv_mmhg: float = Field(5.0, ge=0, description="venous pressure (mmHg)")
    kt: float = Field(1.52, ge=0, description="expansion loss coefficient (-)")
    v_hyp: float = Field(0.35, gt=0, description="hyperemic reference velocity (m/s)")
    expansion_threshold: float = Field(
        1.05, ge=1.0, description="ignore diverging runs with A_max/A_min below this (-)"
    )

    @model_validator(mode="after")
    def _pa_above_pv(self) -> "PhysicsParams":
        if self.pv_mmhg >= self.pa_mmhg:
            raise ValueError("pv_mmhg must be below pa_mmhg")
        return self

    @property
    def pa(self) -> float:
        """Mean aortic pressure (Pa)."""
        return float(mmhg_to_pa(self.pa_mmhg))

    @property
    def pv(self) -> float:
        """Venous pressure (Pa)."""
        return float(mmhg_to_pa(self.pv_mmhg))


def viscous_coefficient(r: np.ndarray, ds, mu: float) -> np.ndarray:
    """Cumulative Poiseuille coefficient a(s) = ∫ 8μ/(π r⁴) ds (Pa·s/m³), trapezoid rule.

    Args:
        r: radii (m), shape (..., N).
        ds: uniform point spacing (m), scalar or shape (...).
        mu: dynamic viscosity (Pa·s).

    Returns:
        a with shape (..., N) and a[..., 0] = 0.
    """
    r = np.asarray(r, dtype=float)
    inv_r4 = 1.0 / (r * r) ** 2
    seg = (4.0 * mu / np.pi) * (inv_r4[..., :-1] + inv_r4[..., 1:])
    seg *= np.asarray(ds, dtype=float)[..., None]
    a = np.zeros_like(r)
    a[..., 1:] = np.cumsum(seg, axis=-1)
    return a


def expansion_coefficient(r: np.ndarray, rho: float, kt: float, threshold: float) -> np.ndarray:
    """Cumulative expansion (separation) loss coefficient b(s) (Pa·s²/m⁶).

    Every maximal diverging run (area non-decreasing between consecutive points) with
    A_max/A_min ≥ ``threshold`` contributes b_run = kt·ρ/2·(1/A_min − 1/A_max)², spread over
    the run's segments in proportion to their decrease of 1/A². Converging sections are
    lossless.

    Args:
        r: radii (m), shape (..., N).
        rho: blood density (kg/m³).
        kt: expansion loss coefficient (-).
        threshold: minimum A_max/A_min for a run to count (-).

    Returns:
        b with shape (..., N) and b[..., 0] = 0.
    """
    r = np.asarray(r, dtype=float)
    rows = r.reshape(-1, r.shape[-1])
    area = np.pi * rows * rows
    inv_a = 1.0 / area

    diverging = np.diff(area, axis=1) >= 0.0  # (B, N-1), per segment
    starts = diverging.copy()
    starts[:, 1:] &= ~diverging[:, :-1]
    # Global run id per segment; flattening row-major keeps runs from different rows apart
    # because the first segment of every row that diverges is itself a start.
    div = diverging.ravel()
    run_of_segment = np.cumsum(starts.ravel()) - 1
    ids = run_of_segment[div]
    n_runs = int(starts.sum())

    d_inv = (inv_a[:, :-1] - inv_a[:, 1:]).ravel()[div]  # ≥ 0 on diverging segments
    d_inv2 = d_inv * (inv_a[:, :-1] + inv_a[:, 1:]).ravel()[div]  # decrease of 1/A²
    run_d_inv = np.bincount(ids, weights=d_inv, minlength=n_runs)  # 1/A_min − 1/A_max
    run_d_inv2 = np.bincount(ids, weights=d_inv2, minlength=n_runs)  # 1/A_min² − 1/A_max²
    inv_a_min = inv_a[:, :-1].ravel()[starts.ravel()]  # 1/A at each run start, in id order
    inv_a_max = inv_a_min - run_d_inv

    active = (inv_a_min >= threshold * inv_a_max) & (run_d_inv2 > 0.0)
    b_run = np.where(active, 0.5 * kt * rho * run_d_inv**2, 0.0)
    scale = np.divide(b_run, run_d_inv2, out=np.zeros(n_runs), where=active)

    contrib = np.zeros(div.size)
    contrib[div] = scale[ids] * d_inv2
    b = np.zeros_like(rows)
    b[:, 1:] = np.cumsum(contrib.reshape(rows.shape[0], -1), axis=1)
    return b.reshape(r.shape)


def calibrate_microvascular_resistance(
    r_ref: np.ndarray, ds, pa, pv, v_hyp, mu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R_micro such that the reference vessel carries Q_ref = v_hyp · π r_ref(0)².

    R_micro = (Pa − Pv)/Q_ref − a_ref(L), floored at R_MICRO_MIN_FRACTION · (Pa − Pv)/Q_ref.

    Args:
        r_ref: reference radii (m), shape (..., N).
        ds: point spacing (m), scalar or shape (...).
        pa, pv: aortic and venous pressure (Pa), scalar or shape (...).
        v_hyp: hyperemic reference velocity (m/s), scalar or shape (...).
        mu: dynamic viscosity (Pa·s).

    Returns:
        (Q_ref in m³/s, R_micro in Pa·s/m³, boolean mask of clamped samples).
    """
    r_ref = np.asarray(r_ref, dtype=float)
    q_ref = v_hyp * np.pi * r_ref[..., 0] ** 2
    r_total = (pa - pv) / q_ref
    r_micro = r_total - viscous_coefficient(r_ref, ds, mu)[..., -1]
    floor = R_MICRO_MIN_FRACTION * r_total
    clamped = r_micro < floor
    return q_ref, np.maximum(r_micro, floor), clamped


def flow_from_resistance(a_total, b_total, r_micro, dp_total):
    """Positive root Q (m³/s) of b·Q² + (a + R_micro)·Q − ΔP_total = 0.

    Uses the cancellation-free form Q = 2ΔP / (c + √(c² + 4bΔP)), c = a + R_micro,
    which is also exact for b = 0.

    Args:
        a_total: viscous coefficient over the whole vessel (Pa·s/m³).
        b_total: expansion coefficient over the whole vessel (Pa·s²/m⁶).
        r_micro: microvascular resistance (Pa·s/m³).
        dp_total: Pa − Pv (Pa).
    """
    c = a_total + r_micro
    return 2.0 * dp_total / (c + np.sqrt(c * c + 4.0 * b_total * dp_total))


@dataclass(frozen=True, eq=False)
class BatchFFRResult:
    """Solutions for B vessels on N points. Arrays are (B,) or (B, N); units in module doc."""

    q: np.ndarray
    q_ref: np.ndarray
    r_micro: np.ndarray
    r_micro_clamped: np.ndarray
    pa: np.ndarray
    pv: np.ndarray
    a: np.ndarray
    b: np.ndarray
    dp: np.ndarray
    ffr: np.ndarray
    mode: FlowMode

    @property
    def ffr_distal(self) -> np.ndarray:
        """FFR at the distal end, shape (B,)."""
        return self.ffr[:, -1]

    @property
    def dp_viscous(self) -> np.ndarray:
        """Viscous pressure loss over the vessel a(L)·Q (Pa)."""
        return self.a[:, -1] * self.q

    @property
    def dp_expansion(self) -> np.ndarray:
        """Expansion pressure loss over the vessel b(L)·Q² (Pa)."""
        return self.b[:, -1] * self.q**2

    @property
    def dp_micro(self) -> np.ndarray:
        """Microvascular pressure drop R_micro·Q (Pa)."""
        return self.r_micro * self.q


def _per_sample(value, n: int, name: str) -> np.ndarray:
    arr = np.broadcast_to(np.asarray(value, dtype=float), (n,))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def solve_ffr_batch(
    r: np.ndarray,
    ds,
    r_ref: np.ndarray,
    params: PhysicsParams | None = None,
    *,
    pa=None,
    pv=None,
    v_hyp=None,
    mode: FlowMode = "resistance",
) -> BatchFFRResult:
    """Solve the FFR pullback for a batch of vessels.

    Args:
        r: radii (m), shape (B, N) or (N,).
        ds: uniform point spacing (m), scalar or shape (B,).
        r_ref: reference radii used for R_micro calibration (m), same shape as ``r``.
        params: model constants; defaults to :class:`PhysicsParams`.
        pa, pv: per-sample aortic / venous pressure (Pa); default ``params.pa`` / ``params.pv``.
        v_hyp: per-sample hyperemic reference velocity (m/s); default ``params.v_hyp``.
        mode: ``"resistance"`` (Q from the R_micro model) or ``"fixed_velocity"`` (Q = Q_ref).
    """
    params = params or PhysicsParams()
    r = np.atleast_2d(np.asarray(r, dtype=float))
    r_ref = np.atleast_2d(np.asarray(r_ref, dtype=float))
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("r must have shape (B, N) with N >= 2")
    if r_ref.shape != r.shape:
        raise ValueError(f"r_ref shape {r_ref.shape} does not match r shape {r.shape}")
    for name, arr in (("r", r), ("r_ref", r_ref)):
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} must be finite and positive")
    n_batch = r.shape[0]
    ds = _per_sample(ds, n_batch, "ds")
    pa = _per_sample(params.pa if pa is None else pa, n_batch, "pa")
    pv = _per_sample(params.pv if pv is None else pv, n_batch, "pv")
    v_hyp = _per_sample(params.v_hyp if v_hyp is None else v_hyp, n_batch, "v_hyp")
    if np.any(ds <= 0) or np.any(v_hyp <= 0):
        raise ValueError("ds and v_hyp must be positive")
    if np.any(pv >= pa):
        raise ValueError("pv must be below pa")

    a = viscous_coefficient(r, ds, params.mu)
    b = expansion_coefficient(r, params.rho, params.kt, params.expansion_threshold)
    q_ref, r_micro, clamped = calibrate_microvascular_resistance(
        r_ref, ds, pa, pv, v_hyp, params.mu
    )
    if mode == "resistance":
        q = flow_from_resistance(a[:, -1], b[:, -1], r_micro, pa - pv)
    elif mode == "fixed_velocity":
        q = q_ref
    else:
        raise ValueError(f"unknown flow mode {mode!r}")

    dp = a * q[:, None] + b * (q * q)[:, None]
    ffr = 1.0 - dp / pa[:, None]
    return BatchFFRResult(
        q=q,
        q_ref=q_ref,
        r_micro=r_micro,
        r_micro_clamped=clamped,
        pa=pa,
        pv=pv,
        a=a,
        b=b,
        dp=dp,
        ffr=ffr,
        mode=mode,
    )


@dataclass(frozen=True, eq=False)
class FFRResult:
    """FFR pullback for one vessel.

    Attributes:
        s: positions (m).
        ffr: FFR(s) = (Pa − ΔP(s)) / Pa (-).
        dp: ΔP(s) = a(s)·Q + b(s)·Q² (Pa).
        a: cumulative viscous coefficient (Pa·s/m³).
        b: cumulative expansion coefficient (Pa·s²/m⁶).
        q: hyperemic flow (m³/s).
        q_ref: reference flow v_hyp · π r_ref(0)² (m³/s).
        r_micro: microvascular resistance (Pa·s/m³).
        r_micro_clamped: whether R_micro hit its floor.
        pa, pv: aortic and venous pressure (Pa).
        mode: flow mode used.
    """

    s: np.ndarray
    ffr: np.ndarray
    dp: np.ndarray
    a: np.ndarray
    b: np.ndarray
    q: float
    q_ref: float
    r_micro: float
    r_micro_clamped: bool
    pa: float
    pv: float
    mode: FlowMode

    @property
    def ffr_distal(self) -> float:
        """FFR at the distal end (-)."""
        return float(self.ffr[-1])

    @property
    def pd(self) -> np.ndarray:
        """Pressure along the vessel Pa − ΔP(s) (Pa)."""
        return self.pa - self.dp

    @property
    def loss_breakdown(self) -> dict[str, float]:
        """Pressure drops over the whole circuit (Pa): viscous, expansion, microvascular."""
        return {
            "viscous": float(self.a[-1] * self.q),
            "expansion": float(self.b[-1] * self.q**2),
            "microvascular": float(self.r_micro * self.q),
        }


def solve_ffr(
    profile: VesselProfile, params: PhysicsParams | None = None, mode: FlowMode = "resistance"
) -> FFRResult:
    """Solve the FFR pullback for one vessel.

    R_micro is calibrated on ``profile.reference_radius()`` (the known reference if set,
    otherwise the robust estimate).
    """
    params = params or PhysicsParams()
    res = solve_ffr_batch(profile.r, profile.ds, profile.reference_radius(), params, mode=mode)
    return FFRResult(
        s=profile.s,
        ffr=res.ffr[0],
        dp=res.dp[0],
        a=res.a[0],
        b=res.b[0],
        q=float(res.q[0]),
        q_ref=float(res.q_ref[0]),
        r_micro=float(res.r_micro[0]),
        r_micro_clamped=bool(res.r_micro_clamped[0]),
        pa=float(res.pa[0]),
        pv=float(res.pv[0]),
        mode=mode,
    )
