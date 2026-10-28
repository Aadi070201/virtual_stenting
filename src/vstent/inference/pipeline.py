"""End-to-end analysis of one vessel (no torch needed).

angiogram → vessel probability (ONNX segmentation) → centreline between two points → QCA
diameter profile (``VesselProfile``, SI units) → FFR pullback from the reduced-order solver and
from the ONNX surrogate. R_micro is calibrated by the solver on the QCA reference diameter and
the same Pa, Pv and R_micro are given to the surrogate.

Warnings flag what the user must know: an assumed pixel size, many fallback diameter
measurements, and inputs outside the surrogate's synthetic training ranges.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from vstent.data.synth import SamplingConfig
from vstent.imaging.qca import DEFAULT_MM_PER_PX, QCAResult, run_qca
from vstent.inference.onnx_runtime import SurrogateRunner
from vstent.physics.geometry import VesselProfile
from vstent.physics.solver import FFRResult, PhysicsParams, solve_ffr
from vstent.units import m_to_mm, pa_to_mmhg

MAX_FALLBACK_FRACTION = 0.10


class Segmenter(Protocol):
    def predict(self, image: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, eq=False)
class VesselAnalysis:
    """Result of :func:`analyse_vessel`.

    Attributes:
        prob: vessel probability map (H, W).
        qca: centreline, diameters and the SI ``VesselProfile``.
        solver: reduced-order FFR pullback on the QCA profile.
        surrogate_s: positions of the surrogate curve (m), or ``None`` without a surrogate.
        surrogate_ffr: surrogate FFR pullback, or ``None``.
        params: physics constants used.
        warnings: plain-language caveats for this analysis.
    """

    prob: np.ndarray
    qca: QCAResult
    solver: FFRResult
    surrogate_s: np.ndarray | None
    surrogate_ffr: np.ndarray | None
    params: PhysicsParams
    warnings: list[str] = field(default_factory=list)

    @property
    def ffr_distal_solver(self) -> float:
        """Distal FFR from the reduced-order solver."""
        return float(self.solver.ffr[-1])

    @property
    def ffr_distal_surrogate(self) -> float | None:
        """Distal FFR from the surrogate (``None`` without a surrogate)."""
        return None if self.surrogate_ffr is None else float(self.surrogate_ffr[-1])


def range_warnings(
    profile: VesselProfile, pa: float, pv: float, sampling: SamplingConfig | None = None
) -> list[str]:
    """Inputs outside the surrogate's synthetic training ranges. Pressures in Pa."""
    sampling = sampling or SamplingConfig()
    checks = (
        ("vessel length", m_to_mm(profile.length), sampling.length_mm, "mm"),
        ("proximal reference diameter", m_to_mm(2 * profile.reference_radius()[0]),
         sampling.d0_mm, "mm"),
        ("aortic pressure", pa_to_mmhg(pa), sampling.pa_mmhg, "mmHg"),
        ("venous pressure", pa_to_mmhg(pv), sampling.pv_mmhg, "mmHg"),
    )  # fmt: skip
    return [
        f"{name} {value:.1f} {unit} is outside the surrogate's training range "
        f"{low:g}–{high:g} {unit}; its prediction is an extrapolation"
        for name, value, (low, high), unit in checks
        if not low <= value <= high
    ]


def analyse_vessel(
    prob: np.ndarray,
    start,
    end,
    *,
    params: PhysicsParams | None = None,
    surrogate: SurrogateRunner | None = None,
    mm_per_px: float | None = None,
    sampling: SamplingConfig | None = None,
) -> VesselAnalysis:
    """QCA + FFR for the vessel between ``start`` and ``end`` ((row, col) px) on ``prob``."""
    params = params or PhysicsParams()
    qca = run_qca(prob, start, end, mm_per_px=mm_per_px)
    solver = solve_ffr(qca.profile, params)
    warnings = []
    if qca.calibration_assumed:
        warnings.append(
            f"Pixel size not calibrated: {DEFAULT_MM_PER_PX} mm/px assumed; absolute diameters, "
            "length and FFR depend on it"
        )
    if qca.fallback_fraction > MAX_FALLBACK_FRACTION:
        warnings.append(
            f"{qca.fallback_fraction:.0%} of diameters used the distance-transform fallback "
            "(edges not found along the normal)"
        )
    if solver.r_micro_clamped:
        warnings.append("R_micro hit its lower bound during calibration")
    s = ffr = None
    if surrogate is not None:
        s, ffr = surrogate.predict_profile(qca.profile, solver.pa, solver.pv, solver.r_micro)
        warnings += range_warnings(qca.profile, solver.pa, solver.pv, sampling)
    return VesselAnalysis(prob, qca, solver, s, ffr, params, warnings)


def analyse_image(
    image: np.ndarray, start, end, *, segmenter: Segmenter, **kwargs
) -> VesselAnalysis:
    """Segment ``image`` (greyscale (H, W)) and analyse the vessel between two points."""
    return analyse_vessel(segmenter.predict(image), start, end, **kwargs)
