"""Coronary flow velocity from the contrast transit of a cine run (P7). Torch-free.

**CFV** (contrast flow velocity, m/s): the speed of the contrast front along the analysed
path, from the arrival of contrast at each point (:func:`~vstent.imaging.cine.contrast_transit`,
with the points following the heart, :mod:`vstent.imaging.motion`). Distances are metres along
the path: the 2D arc length × the pixel size at the vessel in one view, or the 3D arc length
of a two-view reconstruction (P5).

**Phasic flow** (D55): coronary flow is faster in diastole than in systole, so a front timed
over part of a beat would give the systolic or the diastolic speed. With the ED frames known,
arrival times are converted to flow time (:func:`flow_time`): ``SYSTOLE_S`` after each R wave
runs at the systolic speed, the rest of the beat ``DS_RATIO_PRIOR`` times faster, one beat of
flow time per beat. CFV is the Theil–Sen slope of distance on flow time. The ratio is a
population prior: fitting it per run was worse on the development cases.

**Interval** (D55): CFV × exp(± ``VELOCITY_LOG_HALF_WIDTH``), a split-conformal 90%
quantile of |ln(true / measured)| on development cases; the mean of a case's two runs (two
views) × exp(± ``VELOCITY_LOG_HALF_WIDTH_TWO_RUNS``), used only when the two runs agree within
``MAX_RUN_RATIO``. One view foreshortens the path, so its CFV is biased low: the point estimate
is kept and the upper end is multiplied by ``SINGLE_VIEW_UPPER``.

**Hyperemic velocity:** a run recorded under hyperemia (adenosine; DICOM does not record it, the
user says so) uses CFV directly. A resting run uses the FAVOR Pilot relation
HFV = 0.10 + 1.55·CFV − 0.93·CFV² (m/s; Tu et al., JACC Cardiovasc Interv 2016;9:2024–35, as
quoted by Tar et al., Cardiol J 2021;28:163–165, doi:10.5603/CJ.a2020.0144).

**Guards** (docs/PLAN.md, P7): a forward-moving front, a frame time, a calibrated pixel size, a
known heart cycle, CFV within ``CFV_RANGE``, and a Theil–Sen 95% half-width ≤
``MAX_REL_HALF_WIDTH`` of CFV. If one fails, the fixed velocity of configs/physics.yaml stays.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import theilslopes

from vstent.imaging.cine import ContrastTransit

FAVOR_PILOT = (0.10, 1.55, -0.93)  # HFV = a0 + a1·CFV + a2·CFV² (m/s)
FAVOR_VERTEX = -FAVOR_PILOT[1] / (2.0 * FAVOR_PILOT[2])  # 0.833 m/s: the relation's maximum
CFV_RANGE = (0.05, 0.80)  # m/s accepted
MAX_REL_HALF_WIDTH = 0.5  # Theil–Sen 95% half-width ≤ this share of CFV
# 95th percentile of true / single-view length (pixel size at the isocentre) on the P5 report's
# synthetic cases, seeds 5001–5100, 200 views (`foreshortening_quantile`; median 1.138). Those
# cases are not used in any P7 evaluation.
SINGLE_VIEW_UPPER = 1.37
# Split-conformal 90% quantiles of |ln(true / measured)| on the P7 calibration seeds
# 9164–9199 (not used to choose the method): accepted runs on the 3D path, and the mean of both
# runs (`vstent cine-synthetic --calibrate`, reports/cine_calibration.md).
VELOCITY_LOG_HALF_WIDTH = 0.38
VELOCITY_LOG_HALF_WIDTH_TWO_RUNS = 0.34
MAX_RUN_RATIO = 1.5  # two runs of one vessel farther apart than this: one transit failed
LEVEL = 0.90


def hyperemic_velocity(cfv):
    """FAVOR Pilot hyperemic velocity (m/s) from a resting CFV (m/s); CFV above the relation's
    maximum (0.833 m/s) is held there."""
    c = np.minimum(np.asarray(cfv, float), FAVOR_VERTEX)
    a0, a1, a2 = FAVOR_PILOT
    out = a0 + a1 * c + a2 * c**2
    return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class FlowVelocity:
    """A measured flow velocity and what it gives the solver.

    Attributes:
        cfv: contrast flow velocity (m/s), NaN if not measured.
        interval: its 90% interval (m/s); one view: upper end × ``SINGLE_VIEW_UPPER``.
        hyperemic_run: the run was recorded under hyperemia.
        single_view: distances from one view (foreshortened).
        reasons: failed guards (empty: accepted).
        n_runs: runs combined (two views: 2).
    """

    cfv: float
    interval: tuple[float, float]
    hyperemic_run: bool
    single_view: bool
    reasons: tuple[str, ...]
    n_runs: int = 1

    @property
    def accepted(self) -> bool:
        return not self.reasons

    @property
    def v_hyp(self) -> float:
        """Hyperemic velocity for the solver (m/s)."""
        return self.cfv if self.hyperemic_run else hyperemic_velocity(self.cfv)

    @property
    def v_hyp_interval(self) -> tuple[float, float]:
        """The interval passed through the conversion (m/s)."""
        lo, hi = self.interval
        if self.hyperemic_run:
            return lo, hi
        return hyperemic_velocity(max(lo, 0.0)), hyperemic_velocity(hi)

    def note(self) -> str:
        if not np.isfinite(self.cfv):
            return "Flow velocity not measured: " + "; ".join(self.reasons) + "."
        lo, hi = self.interval
        runs = f", mean of {self.n_runs} runs" if self.n_runs > 1 else ""
        text = (f"Contrast flow velocity {self.cfv:.2f} m/s ({LEVEL:.0%}: {lo:.2f}–{hi:.2f}"
                f"{runs})")  # fmt: skip
        if self.single_view:
            text += "; one view: foreshortening can only make it higher (two views remove this)"
        if not self.accepted:
            return text + ". Not used: " + "; ".join(self.reasons) + "."
        conv = ("hyperemic run, used directly" if self.hyperemic_run
                else "resting run, FAVOR Pilot conversion")  # fmt: skip
        vlo, vhi = self.v_hyp_interval
        return (f"{text}; hyperemic velocity {self.v_hyp:.2f} m/s ({vlo:.2f}–{vhi:.2f}; "
                f"{conv}).")  # fmt: skip


def _distances(tr: ContrastTransit, distance_m) -> np.ndarray:
    s = tr.s_px[tr.used]
    return np.abs(np.asarray(distance_m(s), float)
                  - float(distance_m(np.array([tr.inflow_px]))[0]))  # fmt: skip


def transit_slope(
    tr: ContrastTransit, distance_m: Callable[[np.ndarray], np.ndarray]
) -> tuple[float, float, float]:
    """Front speed (m per frame) and its 95% interval on plain time, with the transit's points
    at distances ``distance_m(s_px)`` (m) along the path."""
    slope, _, lo, hi = theilslopes(_distances(tr, distance_m), tr.arrival[tr.used], alpha=0.95)
    return float(slope), float(lo), float(hi)


def flow_velocity(
    tr: ContrastTransit | None,
    frame_time_ms: float | None,
    distance_m: Callable[[np.ndarray], np.ndarray] | None,
    *,
    ed_frames: Sequence[int] | None,
    period_frames: float | None,
    single_view: bool,
    hyperemic_run: bool = False,
) -> FlowVelocity:
    """CFV of one run from its contrast transit, with the guards. ``distance_m`` maps the
    transit's 2D arc length (px) to metres along the path (``None``: no calibrated pixel
    size); ``ed_frames`` and ``period_frames`` give the heart cycle (flow time)."""
    nan = float("nan")
    if tr is None:
        return FlowVelocity(nan, (nan, nan), hyperemic_run, single_view,
                            ("no contrast arrival measured along the path",))  # fmt: skip
    reasons = []
    if not tr.plausible:
        reasons.append("the contrast front does not advance steadily along the path")
    if not frame_time_ms:
        reasons.append("no frame time in the run")
    if distance_m is None:
        reasons.append("no calibrated pixel size (DICOM geometry)")
    if tr.used.size < 2:
        reasons.append("too few points with a contrast arrival")
    if not frame_time_ms or distance_m is None or tr.used.size < 2:
        return FlowVelocity(nan, (nan, nan), hyperemic_run, single_view, tuple(reasons))
    if ed_frames and period_frames:
        slope, lo, hi, _ = phasic_slope(tr, distance_m, frame_time_ms, ed_frames,
                                        period_frames, DS_RATIO_PRIOR)  # fmt: skip
    else:
        reasons.append("heart cycle unknown (no ECG and no beat found in the images)")
        slope, lo, hi = (v * 1000.0 / frame_time_ms for v in transit_slope(tr, distance_m))
    if not CFV_RANGE[0] <= slope <= CFV_RANGE[1]:
        reasons.append(f"{slope:.2f} m/s is outside {CFV_RANGE[0]:g}–{CFV_RANGE[1]:g} m/s")
    half = (hi - lo) / 2.0
    if slope > 0 and half > MAX_REL_HALF_WIDTH * slope:
        reasons.append(f"imprecise (95% interval of the slope ±{half / slope:.0%})")
    q = float(np.exp(VELOCITY_LOG_HALF_WIDTH))
    interval = (slope / q, slope * q * (SINGLE_VIEW_UPPER if single_view else 1.0))
    return FlowVelocity(slope, interval, hyperemic_run, single_view, tuple(dict.fromkeys(reasons)))


def combine_runs(runs: Sequence[FlowVelocity]) -> FlowVelocity:
    """One velocity from a case's runs (two views): the mean of the accepted runs' CFVs,
    × exp(± ``VELOCITY_LOG_HALF_WIDTH_TWO_RUNS``) if both were accepted and agree within
    ``MAX_RUN_RATIO`` (otherwise not accepted); a single accepted run as it is; none accepted:
    the first run, with every reason."""
    ok = [r for r in runs if r.accepted]
    if len(ok) == 1:
        return ok[0]
    if not ok:
        reasons = tuple(dict.fromkeys(x for r in runs for x in r.reasons))
        first = runs[0]
        return FlowVelocity(first.cfv, first.interval, first.hyperemic_run, first.single_view,
                            reasons or ("no run measured",), len(runs))  # fmt: skip
    cfvs = [r.cfv for r in ok]
    v = float(np.mean(cfvs))
    upper = SINGLE_VIEW_UPPER if any(r.single_view for r in ok) else 1.0
    q = float(np.exp(VELOCITY_LOG_HALF_WIDTH_TWO_RUNS))
    ratio = max(cfvs) / min(cfvs) if min(cfvs) > 0 else float("inf")
    reasons = ()
    if ratio > MAX_RUN_RATIO:
        reasons = (f"the two runs disagree (×{ratio:.1f}): one transit is unreliable",)
    return FlowVelocity(v, (v / q, v * q * upper), all(r.hyperemic_run for r in ok),
                        any(r.single_view for r in ok), reasons, len(ok))  # fmt: skip


SYSTOLE_S = 0.31  # assumed R wave to end-systole (s)
DS_RATIO_PRIOR = 2.0  # assumed diastolic / systolic front speed when it cannot be fitted
DS_RATIO_GRID = np.round(np.arange(1.0, 3.01, 0.1), 2)


def flow_time(
    t_s, ed_s: np.ndarray, period_s: float, ratio: float, systole_s: float = SYSTOLE_S
) -> np.ndarray:
    """Time weighted by a phasic speed profile (s): it runs at g_s in systole (``systole_s``
    after each R wave) and ``ratio`` · g_s in diastole, 1 on average over a beat. R waves at
    ``ed_s`` (s), continued every ``period_s`` before and after."""
    t = np.asarray(t_s, float)
    sys = min(systole_s / period_s, 0.6)
    g_s = 1.0 / (sys + (1.0 - sys) * ratio)
    x = (t - float(np.min(ed_s))) / period_s
    cycles = np.floor(x)
    phase = x - cycles
    within = g_s * np.minimum(phase, sys) + ratio * g_s * np.maximum(phase - sys, 0.0)
    return (cycles + within) * period_s  # fmt: skip


def phasic_slope(tr: ContrastTransit, distance_m, frame_time_ms: float, ed_frames,
                 period_frames: float, ratio: float | None = None):  # fmt: skip
    """Front speed (m/s) in flow time (:func:`flow_time`) and its 95% interval, with the
    diastolic / systolic ratio given, or fitted (``None``: the ratio of the grid whose line
    fits best, by median absolute residual). Returns (speed, lo, hi, ratio)."""
    dt = frame_time_ms / 1000.0
    d = _distances(tr, distance_m)
    t = tr.arrival[tr.used] * dt
    ed_s = np.asarray(ed_frames, float) * dt
    period = period_frames * dt

    def fit(rho):
        g = flow_time(t, ed_s, period, rho)
        slope, icpt, lo, hi = theilslopes(d, g, alpha=0.95)
        return slope, lo, hi, float(np.median(np.abs(d - icpt - slope * g)))

    if ratio is None:
        ratio = min(DS_RATIO_GRID, key=lambda r: fit(r)[3])
    slope, lo, hi, _ = fit(ratio)
    return float(slope), float(lo), float(hi), float(ratio)


def foreshortening_quantile(cfg, seeds, q: float = 95.0) -> tuple[float, float]:
    """The ``q``-th percentile and the median of true / single-view main-vessel length over
    both views of the synthetic two-view cases ``seeds`` (``cfg``:
    :class:`~vstent.imaging.synthetic3d.TwoViewSynthConfig`), the single-view length being
    the projected length × the pixel size at the isocentre. Gives :data:`SINGLE_VIEW_UPPER`."""
    from vstent.imaging.synthetic3d import sample_case

    ratios = []
    for seed in seeds:
        case = sample_case(int(seed), cfg)
        for g in case.views:
            p = g.project(case.curves[0])
            length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) * g.iso_pixel_m
            ratios.append(case.main_length / length)
    r = np.asarray(ratios)
    return float(np.percentile(r, q)), float(np.median(r))
