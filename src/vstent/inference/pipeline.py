"""End-to-end analysis of one vessel (no torch needed).

angiogram → vessel probability (ONNX segmentation) → centreline between two points → QCA
diameter profile (``VesselProfile``, SI units) → FFR pullback from the reduced-order solver and
from the ONNX surrogate. R_micro is calibrated by the solver on the QCA reference diameter and
the same Pa, Pv and R_micro are given to the surrogate.

Lumen edges: from the angiogram's intensity when the image is given (``image``; the default
for an automatic segmentation, D43), else from the probability map (an analyst's mask).

Warnings flag what the user must know: an assumed pixel size, many fallback diameter
measurements, and inputs outside the surrogate's synthetic training ranges.

Two views (P5, :func:`analyse_two_views`): the same vessel between the same landmarks in two
projections with known geometry → 3D centreline (:mod:`vstent.imaging.twoview`) → a lumen
profile along the true 3D length, with diameters from both views at their exact magnification
→ the same solver and surrogate.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from vstent.data.synth import SamplingConfig
from vstent.imaging.centerline import trace_centerline
from vstent.imaging.projection import ProjectionGeometry
from vstent.imaging.qca import (
    DEFAULT_MM_PER_PX,
    LEVEL,
    NO_LUMEN_PX,
    QCAResult,
    qca_from_centerline,
    run_qca,
)
from vstent.imaging.qca_tree import MAX_DEPTH, TreeQCA, tree_qca
from vstent.imaging.twoview import Reconstruction, TwoViewProfile, reconstruct, two_view_profile
from vstent.inference.onnx_runtime import SurrogateRunner
from vstent.physics.geometry import VesselProfile
from vstent.physics.solver import FFRResult, PhysicsParams, solve_ffr
from vstent.physics.tree import TreeFFRResult, solve_tree
from vstent.uncertainty.greyzone import Interval, model_interval
from vstent.units import m_to_mm, pa_to_mmhg

# Lumen measurement when the angiogram is given (an automatic segmentation): "edges" (D43; the
# shape, chosen by the pre-registered rule D44) with the absolute size from the imaging-chain fit
# ("scaled", P5b; chosen by the pre-registered rule in docs/PLAN.md, D50). "model" is an option.
AUTO_LUMEN = "scaled"
MAX_FALLBACK_FRACTION = 0.10
MAX_CAPPED_FRACTION = 0.10


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
        surrogate_ffr: surrogate FFR pullback (ensemble mean), or ``None``.
        params: physics constants used.
        warnings: plain-language caveats for this analysis.
        surrogate_std: ensemble SD along ``surrogate_s`` (``None`` for a single model).
        model_interval: calibrated interval of the surrogate's distal FFR, if available.
    """

    prob: np.ndarray
    qca: QCAResult
    solver: FFRResult
    surrogate_s: np.ndarray | None
    surrogate_ffr: np.ndarray | None
    params: PhysicsParams
    warnings: list[str] = field(default_factory=list)
    surrogate_std: np.ndarray | None = None
    model_interval: Interval | None = None

    @property
    def ffr_distal_solver(self) -> float:
        """Distal FFR from the reduced-order solver."""
        return float(self.solver.ffr[-1])

    @property
    def ffr_distal_surrogate(self) -> float | None:
        """Distal FFR from the surrogate (``None`` without a surrogate)."""
        return None if self.surrogate_ffr is None else float(self.surrogate_ffr[-1])


def _lumen(image: np.ndarray | None, lumen: str | None) -> str:
    """The lumen option for :func:`~vstent.imaging.qca.qca_from_centerline`."""
    if image is None:
        if lumen == "model":
            raise ValueError("the model-based lumen needs the image")
        return "edges"
    return AUTO_LUMEN if lumen is None else lumen


def lumen_warnings(qca: QCAResult, name: str = "") -> list[str]:
    """Bridged gaps of the segmentation and stretches without a measurable lumen (P4c).
    ``name`` prefixes the messages (a tree branch)."""
    out = []
    prefix = f"{name}: " if name else ""
    if qca.size_note:
        out.append(
            f"{prefix}absolute size from the intensity edges only ({qca.size_note}); on rendered "
            "vessels the edges read healthy lumens about 14% narrow"
        )
    if qca.bridged_mm > 0:
        out.append(
            f"{prefix}the segmentation missed {qca.bridged_mm:.1f} mm of the vessel (a gap "
            "between aligned vessel ends, bridged); that part was measured from the image only"
        )
    for a, b in qca.no_lumen_segments:
        out.append(
            f"{prefix}no measurable lumen (below {NO_LUMEN_PX:g} px = "
            f"{NO_LUMEN_PX * qca.mm_per_px:.2f} mm) over {b - a:.1f} mm at {a:.0f}–{b:.0f} mm "
            "from the start: critical stenosis or total occlusion. A total occlusion is supplied "
            "by collaterals, which the flow model does not represent: if the vessel is occluded, "
            "the FFR shown is not valid"
        )
    return out


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
    image: np.ndarray | None = None,
    lumen: str | None = None,
    lesion_reader=None,
) -> VesselAnalysis:
    """QCA + FFR for the vessel between ``start`` and ``end`` ((row, col) px) on ``prob``.
    ``image``: the angiogram ``prob`` was segmented from (lumen edges from its intensity);
    ``None`` when ``prob`` is an analyst's mask. ``lumen``: ``"edges"`` or ``"model"`` (needs
    ``image``); default :data:`AUTO_LUMEN` with an image, edges without. ``lesion_reader``:
    a :class:`~vstent.inference.lesion_runtime.LesionRunner` whose predicted %DS sets each
    candidate lesion's depth (P4i; a research option, not adopted, D63)."""
    params = params or PhysicsParams()
    qca = run_qca(prob, start, end, mm_per_px=mm_per_px, image=image,
                  lumen=_lumen(image, lumen))  # fmt: skip
    if lesion_reader is not None and image is not None:
        from vstent.imaging.lesion_patch import apply_learned

        qca, _ = apply_learned(qca, image, prob, lesion_reader.predict)
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
    if qca.capped_fraction > MAX_CAPPED_FRACTION:
        warnings.append(
            f"{qca.capped_fraction:.0%} of diameters exceeded 1.15 × the reference (side "
            "branches or overlapping vessels) and were capped"
        )
    if solver.r_micro_clamped:
        warnings.append("R_micro hit its lower bound during calibration")
    warnings += lumen_warnings(qca)
    s = ffr = std = interval = None
    if surrogate is not None:
        s, ffr, std = surrogate.predict_profile_with_spread(
            qca.profile, solver.pa, solver.pv, solver.r_micro
        )
        interval = model_interval(surrogate, ffr[-1], None if std is None else std[-1])
        warnings += range_warnings(qca.profile, solver.pa, solver.pv, sampling)
        floor = solver.pv / solver.pa  # FFR cannot fall below Pv / Pa (zero flow)
        if ffr.min() < floor:
            warnings.append(
                f"Surrogate FFR reaches {ffr.min():.2f}, below the physical minimum "
                f"Pv/Pa = {floor:.2f}: the input is far outside what it was trained on"
            )
    return VesselAnalysis(prob, qca, solver, s, ffr, params, warnings, std, interval)


@dataclass(frozen=True, eq=False)
class TwoViewAnalysis:
    """Result of :func:`analyse_two_views` (P5).

    Attributes:
        probs: vessel probability map of each view.
        qca: each view's QCA along its traced centreline (ends not trimmed; isocentre pixel
            size).
        reconstruction: the 3D centreline.
        lumen: the lumen profile along it (``lumen.profile`` goes to the physics).
        solver: reduced-order FFR pullback along the 3D centreline.
        surrogate_s, surrogate_ffr, surrogate_std, model_interval: as in
            :class:`VesselAnalysis`.
        params: physics constants used.
        warnings: plain-language caveats for this analysis.
    """

    probs: tuple[np.ndarray, np.ndarray]
    qca: tuple[QCAResult, QCAResult]
    reconstruction: Reconstruction
    lumen: TwoViewProfile
    solver: FFRResult
    surrogate_s: np.ndarray | None
    surrogate_ffr: np.ndarray | None
    params: PhysicsParams
    warnings: list[str] = field(default_factory=list)
    surrogate_std: np.ndarray | None = None
    model_interval: Interval | None = None

    @property
    def profile(self) -> VesselProfile:
        return self.lumen.profile

    @property
    def ffr_distal_solver(self) -> float:
        return float(self.solver.ffr[-1])

    @property
    def ffr_distal_surrogate(self) -> float | None:
        return None if self.surrogate_ffr is None else float(self.surrogate_ffr[-1])

    @property
    def single_view_lengths_mm(self) -> tuple[float, float]:
        """Each view's 2D path length at its isocentre pixel size (mm): what a single view
        would report."""
        return tuple(  # type: ignore[return-value]
            m_to_mm(q.centerline.length * g.iso_pixel_m)
            for q, g in zip(self.qca, self.reconstruction.geometries, strict=True)
        )


def analyse_two_views(
    probs: tuple[np.ndarray, np.ndarray],
    starts,
    ends,
    geometries: tuple[ProjectionGeometry, ProjectionGeometry],
    *,
    params: PhysicsParams | None = None,
    surrogate: SurrogateRunner | None = None,
    sampling: SamplingConfig | None = None,
    images: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    lumen: str | None = None,
    correct_shift: bool = True,
) -> TwoViewAnalysis:
    """QCA + FFR of one vessel seen in two views. ``starts`` / ``ends``: the same two
    landmarks ((row, col) px) in each view; ``geometries``: each view's projection (DICOM).
    ``images``, ``lumen``: as in :func:`analyse_vessel`. ``correct_shift``: estimate motion
    between the runs (:func:`~vstent.imaging.twoview.estimate_shift`)."""
    params = params or PhysicsParams()
    qcas = []
    for k in range(2):
        cl = trace_centerline(np.asarray(probs[k]) >= LEVEL, starts[k], ends[k])
        qcas.append(qca_from_centerline(
            probs[k], cl, mm_per_px=m_to_mm(geometries[k].iso_pixel_m), image=images[k],
            lumen=_lumen(images[k], lumen), trim_diameters=0,
        ))  # fmt: skip
    rec = reconstruct(geometries[0], qcas[0].centerline, geometries[1], qcas[1].centerline,
                      correct_shift=correct_shift)  # fmt: skip
    prof = two_view_profile(
        rec,
        tuple(q.diameters.diameter_px * q.size_factor for q in qcas),
        (qcas[0].centerline.s, qcas[1].centerline.s),
    )
    solver = solve_ffr(prof.profile, params)
    warnings = rec.warnings()
    for k, q in enumerate(qcas):
        name = f"View {k + 1}"
        if q.fallback_fraction > MAX_FALLBACK_FRACTION:
            warnings.append(
                f"{name}: {q.fallback_fraction:.0%} of diameters used the distance-transform "
                "fallback (edges not found along the normal)"
            )
        warnings += lumen_warnings(q, name)
    if prof.capped.mean() > MAX_CAPPED_FRACTION:
        warnings.append(
            f"{prof.capped.mean():.0%} of diameters exceeded 1.15 × the reference (side "
            "branches or overlapping vessels) and were capped"
        )
    if solver.r_micro_clamped:
        warnings.append("R_micro hit its lower bound during calibration")
    s = ffr = std = interval = None
    if surrogate is not None:
        s, ffr, std = surrogate.predict_profile_with_spread(
            prof.profile, solver.pa, solver.pv, solver.r_micro
        )
        interval = model_interval(surrogate, ffr[-1], None if std is None else std[-1])
        warnings += range_warnings(prof.profile, solver.pa, solver.pv, sampling)
    return TwoViewAnalysis(tuple(probs), tuple(qcas), rec, prof, solver, s, ffr, params, warnings,
                           std, interval)  # fmt: skip


def analyse_image(
    image: np.ndarray, start, end, *, segmenter: Segmenter, **kwargs
) -> VesselAnalysis:
    """Segment ``image`` (greyscale (H, W)) and analyse the vessel between two points; lumen
    edges from the image (the automatic-segmentation default)."""
    return analyse_vessel(segmenter.predict(image), start, end, image=image, **kwargs)


@dataclass(frozen=True, eq=False)
class TreeAnalysis:
    """Result of :func:`analyse_tree` (P2): the vessel tree and its FFR.

    Attributes:
        prob: vessel probability map (H, W).
        qca: the tree's QCA (``qca.tree`` is what the tree solver received).
        result: tree-solver FFR at every sample of every branch.
        single: the main path alone with the single-vessel solver, for comparison.
        params: physics constants used.
        warnings: plain-language caveats for this analysis.
    """

    prob: np.ndarray
    qca: TreeQCA
    result: TreeFFRResult
    single: FFRResult
    params: PhysicsParams
    warnings: list[str] = field(default_factory=list)


def analyse_tree(
    prob: np.ndarray,
    start,
    end=None,
    *,
    params: PhysicsParams | None = None,
    mm_per_px: float | None = None,
    max_depth: int = MAX_DEPTH,
    image: np.ndarray | None = None,
    lumen: str | None = None,
) -> TreeAnalysis:
    """Tree QCA from ``start`` (to ``end`` or the farthest point) and tree-solver FFR.
    ``image``, ``lumen``: as :func:`analyse_vessel`.

    No surrogate yet: the tree surrogate is P3. Pressures and v_hyp come from ``params``.
    """
    params = params or PhysicsParams()
    tq = tree_qca(prob, start, end, mm_per_px=mm_per_px, max_depth=max_depth, image=image,
                  lumen=_lumen(image, lumen))  # fmt: skip
    result = solve_tree(tq.tree, params)
    single = solve_ffr(tq.tree.branches[0], params)
    warnings = []
    if tq.calibration_assumed:
        warnings.append(
            f"Pixel size not calibrated: {DEFAULT_MM_PER_PX} mm/px assumed; absolute diameters, "
            "lengths and FFR depend on it"
        )
    for name, q in zip(tq.tree.names, tq.branches, strict=True):
        warnings += lumen_warnings(q, name)
        if q.fallback_fraction > MAX_FALLBACK_FRACTION:
            warnings.append(f"{name}: {q.fallback_fraction:.0%} of diameters used the fallback")
        if q.capped_fraction > MAX_CAPPED_FRACTION:
            warnings.append(
                f"{name}: {q.capped_fraction:.0%} of diameters exceeded 1.15 × the reference "
                "(overlap or further branches) and were capped"
            )
    if result.r_outlet_clamped.any():
        names = ", ".join(
            n for n, c in zip(tq.tree.names, result.r_outlet_clamped, strict=True) if c
        )
        warnings.append(f"outlet resistance at its lower bound for: {names}")
    if tq.dropped:
        warnings.append(
            f"{len(tq.dropped)} further side path(s) on the skeleton were not modelled "
            "(too short, too thin or deeper than one level); their flow is lumped into the "
            "branches they leave"
        )
    return TreeAnalysis(prob, tq, result, single, params, warnings)
