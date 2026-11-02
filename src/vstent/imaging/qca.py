"""2D quantitative coronary analysis (QCA) on a vessel probability map.

**Edge method.** With the angiogram itself (``image``), the lumen edges come from its intensity,
guided by the probability map (:func:`measure_diameters_intensity`): the product default
wherever the vessel map is an automatic segmentation, because the segmenter smooths over
narrowings (``reports/stenosis_grading.md``, D43). Without it (an analyst's mask, or rendered
test tubes), the edges are the probability map's, as follows.

The lumen diameter at each centreline point is the distance between the two crossings of the
probability map with the 0.5 level along the local normal (bilinear sampling, linear sub-pixel
interpolation of the crossing). If one edge is much farther than the other (a side branch or
an overlapping vessel on that side), 2 × the nearer edge distance is used. Fallbacks: no
crossing within the search range → 2 × the distance transform of the binary mask; centreline
point outside the vessel (smoothing cut a sharp bend) → interpolation from its neighbours.

One median vessel diameter is trimmed from both ends of the centreline (end effects of the
mask, not lumen narrowing). Pixels are converted to millimetres with a calibration factor.
**Without a calibration from the image (e.g. DICOM pixel spacing or a catheter of known size)
the default 0.30 mm/px is an assumption**, and every result records whether it was assumed.

**Absolute size** (P5b, ``lumen="scaled"``): intensity edges sit inside the blurred,
edge-enhanced vessel wall, so on rendered vessels they read healthy lumens about 14% narrow.
The scaled lumen keeps the edges' shape (%DS unchanged) and takes the absolute size from the
imaging-chain fit (:mod:`vstent.imaging.lumen_model`) on the healthy part: one factor per
vessel, the median of fit / edge diameters there. An implausible factor or a failed fit keeps
the edges' size, with a note.

The reference diameter is a two-sided robust line fit (:func:`robust_reference`); diameters
above 1.15 × reference (branches, overlaps) are capped before the profile, in SI units with
that explicit reference, goes to the M1 functions (%DS, lesions, MLD) and the physics.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, median_filter
from scipy.stats import theilslopes

from vstent.imaging.centerline import Centerline, trace_centerline
from vstent.imaging.densitometry import DOMAIN, densitometric_diameter
from vstent.imaging.lumen_model import ChainCalibration, LumenFit, fit_lumen, fit_windows
from vstent.physics.geometry import Lesion, VesselProfile
from vstent.units import m_to_mm, mm_to_m

DEFAULT_MM_PER_PX = 0.30
LEVEL = 0.5
EDGE_STEP_PX = 0.1
MIN_DIAMETER_PX = 0.5
END_TRIM_DIAMETERS = 1.0
MIN_POINTS_AFTER_TRIM = 10
# One edge farther than BRANCH_RATIO × the other + BRANCH_SLACK_PX → side branch on that side.
# The slack keeps a sub-pixel off-centre centreline in a thin vessel from triggering it.
BRANCH_RATIO = 1.5
BRANCH_SLACK_PX = 1.0
# Reference fit: points within [REF_LOWER, REF_UPPER] × the line count as healthy. Diameters
# above REF_UPPER × reference are capped before the physics (a spike would otherwise act as a
# narrowing followed by an expansion loss).
REF_LOWER = 0.85
REF_UPPER = 1.15
REF_MAX_ITER = 5
# A lumen narrower than NO_LUMEN_PX cannot be told apart from none in one frame; a run of at
# least NO_LUMEN_MIN_POINTS such points is reported (critical stenosis or total occlusion).
NO_LUMEN_PX = 1.0
NO_LUMEN_MIN_POINTS = 3
LUMEN_METHOD_LABELS = {  # for reports
    "intensity": "image intensity edges, guided by the automatic segmentation (D43)",
    "probability": "outline of the vessel mask",
    "model": "imaging-chain model fitted to the image (blur, edge enhancement; P4c)",
    "scaled": "image intensity edges, absolute size from the imaging-chain fit (P5b)",
    "dense": "intensity edges narrowed by densitometry (integrated contrast, P4h), absolute "
    "size from the imaging-chain fit",
    "learned": "lesion depths learned from real angiograms (P4i) on the intensity edges",
}
# densitometric lumen (P4h): the product's domain, or one named for the comparison
DENSE_DOMAINS = {"dense": None, "dense_log": "log", "dense_linear": "linear"}
LUMEN_OPTIONS = ("edges", "model", "scaled", *DENSE_DOMAINS)
# Scaled lumen (P5b): the size factor is accepted within SIZE_FACTOR_RANGE, from at least
# SIZE_MIN_POINTS healthy points (set on development renders, D50)
SIZE_FACTOR_RANGE = (1.0, 1.4)
SIZE_MIN_POINTS = 10
SIZE_WINDOW_POINTS = 30  # the fit runs on up to SIZE_MAX_WINDOWS healthy windows of this length
SIZE_MAX_WINDOWS = 4


@dataclass(frozen=True, eq=False)
class DiameterProfile:
    """Lumen diameter along a centreline (all in pixels).

    Attributes:
        diameter_px: (n,) diameter.
        left_px, right_px: (n,) signed edge offsets along the normal (left ≤ 0 ≤ right);
            NaN where the fallback was used.
        fallback: (n,) True where an edge was not found (2 × distance transform, or
            interpolation from neighbours where the centreline left the vessel).
        branch: (n,) True where one edge was much farther than the other (a side branch or
            overlap on that side) and 2 × the nearer edge distance was used.
    """

    diameter_px: np.ndarray
    left_px: np.ndarray
    right_px: np.ndarray
    fallback: np.ndarray
    branch: np.ndarray


def _first_crossing(vals: np.ndarray, t: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """First crossing below ``level`` along axis 1 (vals[:, 0] is the centre).

    Returns the interpolated position (same units as ``t``) and a found-flag per row.
    """
    below = vals < level
    found = below.any(axis=1) & ~below[:, 0]
    k = np.where(found, np.argmax(below, axis=1), 1)
    rows = np.arange(vals.shape[0])
    v0, v1 = vals[rows, k - 1], vals[rows, k]
    frac = np.clip((v0 - level) / np.where(v0 > v1, v0 - v1, 1.0), 0.0, 1.0)
    return t[k - 1] + frac * (t[k] - t[k - 1]), found


def measure_diameters(
    prob: np.ndarray,
    centerline: Centerline,
    *,
    level: float = LEVEL,
    step_px: float = EDGE_STEP_PX,
    max_half_width_px: float | None = None,
) -> DiameterProfile:
    """Diameter at every centreline point from a probability map ``prob`` (H, W) in [0, 1]."""
    prob = np.asarray(prob, dtype=np.float32)
    edt = distance_transform_edt(prob >= level)
    half = max_half_width_px or float(2.0 * edt.max() + 3.0)
    t = np.arange(0.0, half + step_px / 2, step_px)
    pts, nrm = centerline.points, centerline.normals

    def sample(sign: float) -> np.ndarray:
        q = pts[:, None, :] + sign * t[None, :, None] * nrm[:, None, :]
        coords = [q[..., 0].ravel(), q[..., 1].ravel()]
        vals = map_coordinates(prob, coords, order=1, mode="constant", cval=0.0)
        return vals.reshape(q.shape[:2])

    right, ok_r = _first_crossing(sample(+1.0), t, level)
    left, ok_l = _first_crossing(sample(-1.0), t, level)
    both, either = ok_r & ok_l, ok_r | ok_l
    near = np.where(both, np.minimum(left, right), np.where(ok_l, left, right))
    far = np.maximum(left, right)
    # A side branch / overlap on one side: that edge is much farther, or beyond the search range.
    branch = (both & (far > BRANCH_RATIO * near + BRANCH_SLACK_PX)) | (either & ~both)
    diameter = np.where(branch, 2.0 * near, right + left)
    # Fallbacks. Centre inside the vessel but no edge found: 2 × distance transform.
    # Centre outside (smoothing cut a sharp bend): interpolate from the measured neighbours.
    coords = [pts[:, 0], pts[:, 1]]
    inside = map_coordinates(prob, coords, order=1, mode="nearest") >= level
    edt_d = 2.0 * map_coordinates(edt, coords, order=1, mode="nearest")
    diameter = np.where(either, diameter, edt_d)
    if (~inside).any() and inside.any():
        s = centerline.s
        diameter[~inside] = np.interp(s[~inside], s[inside], diameter[inside])
    return DiameterProfile(
        diameter_px=np.maximum(diameter, MIN_DIAMETER_PX),
        left_px=np.where(ok_l, -left, np.nan),
        right_px=np.where(ok_r, right, np.nan),
        fallback=~either | ~inside,
        branch=branch & inside,
    )


INTENSITY_SMOOTH_PX = 1.0  # Gaussian smoothing of the image before edge detection
INTENSITY_SLACK_PX = 4.0  # edge search: up to 2 × the guide's half-width + this
MIN_CONTRAST = 0.04  # background − lumen (image in [0, 1]); less → keep the guide's diameter
EDGE_MEDIAN_POINTS = 5  # median filter of each edge along the centreline (points ≈ px)
EDGE_MAX_SLOPE = 1.0  # an edge moves at most this far (px) per px along the centreline (D45)
EDGE_MOVE_COST = 0.1  # edge-tracking cost per px of edge movement (edge cost per point: 0–1)


def track_edge(grad: np.ndarray, max_jump: int, move_cost: float) -> np.ndarray:
    """Minimum-cost edge contour, the classical QCA edge criterion: one search position (column)
    per centreline point (row) minimising Σ (1 − gradient / the row's largest gradient) +
    ``move_cost`` × |column change|, with at most ``max_jump`` columns between neighbouring
    rows. A wall is a continuous contour, so the edge cannot flicker between two candidate
    gradients from one point to the next. ``grad`` (n, m): −inf outside the search range.
    Returns the column of every row."""
    n, m = grad.shape
    finite = np.isfinite(grad)
    top = np.max(np.where(finite, grad, 0.0), axis=1, keepdims=True)
    rel = np.clip(np.where(finite, grad, 0.0), 0.0, None) / np.where(top > 0, top, 1.0)
    cost = np.where(finite, 1.0 - rel, np.inf)
    offsets = np.arange(-max_jump, max_jump + 1)
    source = np.arange(m)[None, :] + offsets[:, None]  # (2 J + 1, m) column in the row above
    valid = (source >= 0) & (source < m)
    source = np.clip(source, 0, m - 1)
    penalty = move_cost * np.abs(offsets)[:, None]
    cols = np.arange(m)
    acc, back = cost[0], np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        cand = np.where(valid, acc[source], np.inf) + penalty
        best = np.argmin(cand, axis=0)
        back[i] = source[best, cols]
        acc = cand[best, cols] + cost[i]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmin(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def measure_diameters_intensity(
    image: np.ndarray,
    centerline: Centerline,
    guide: DiameterProfile,
    *,
    smooth_px: float = INTENSITY_SMOOTH_PX,
    step_px: float = EDGE_STEP_PX,
    median_points: int = EDGE_MEDIAN_POINTS,
) -> DiameterProfile:
    """Lumen edges from the image intensity (vessels dark): on each side of the centreline the
    edge follows the steep intensity rises along the normals as one continuous contour
    (:func:`track_edge`; parabolic sub-pixel peak), searched up to 2 × the ``guide``
    half-width + ``INTENSITY_SLACK_PX``. The classical edge criterion of QCA; it does not
    depend on how the segmentation draws the vessel's outline.

    Where the contrast (brightest − centre intensity in the search range) is below
    ``MIN_CONTRAST``, an edge is missing, or the guide shows a side branch / overlap, the
    ``guide`` diameter (probability-map edges) is kept. Pixels throughout.
    """
    img = np.asarray(image, dtype=np.float32)
    img = img / 255.0 if img.max() > 1.0 else img
    img = gaussian_filter(img, smooth_px) if smooth_px > 0 else img
    pts, nrm = centerline.points, centerline.normals
    limit = guide.diameter_px + INTENSITY_SLACK_PX  # 2 × half-width + slack
    t = np.arange(0.0, float(limit.max()) + step_px / 2, step_px)
    in_range = t[None, :] <= limit[:, None]
    centre = map_coordinates(img, [pts[:, 0], pts[:, 1]], order=1, mode="nearest")
    spacing = float(np.median(np.diff(centerline.s))) if len(pts) > 1 else 1.0
    max_jump = max(1, int(round(EDGE_MAX_SLOPE * spacing / step_px)))

    def edge(sign: float) -> tuple[np.ndarray, np.ndarray]:
        q = pts[:, None, :] + sign * t[None, :, None] * nrm[:, None, :]
        vals = map_coordinates(img, [q[..., 0].ravel(), q[..., 1].ravel()], order=1,
                               mode="nearest").reshape(q.shape[:2])  # fmt: skip
        grad = np.gradient(vals, step_px, axis=1)
        grad = np.where(in_range, grad, -np.inf)
        k = track_edge(grad, max_jump, EDGE_MOVE_COST * step_px)
        rows = np.arange(len(k))
        km, kp = np.clip(k - 1, 0, len(t) - 1), np.clip(k + 1, 0, len(t) - 1)
        g0, g1, g2 = grad[rows, km], grad[rows, k], grad[rows, kp]
        ok_peak = np.isfinite(g0) & np.isfinite(g2)
        g0, g2 = np.where(ok_peak, g0, g1), np.where(ok_peak, g2, g1)  # no inf − inf
        den = g0 - 2.0 * g1 + g2
        ok_peak &= den < 0
        shift = np.where(ok_peak, 0.5 * (g0 - g2) / np.where(den < 0, den, -1.0), 0.0)
        pos = t[k] + np.clip(shift, -0.5, 0.5) * step_px
        bright = np.max(np.where(in_range, vals, -np.inf), axis=1)
        found = (grad[rows, k] > 0) & (bright - centre >= MIN_CONTRAST) & (k > 0)
        return pos, found

    right, ok_r = edge(+1.0)
    left, ok_l = edge(-1.0)
    if median_points > 1:  # edge contours are smooth; single-point noise is not a narrowing
        right, left = (median_filter(v, size=median_points, mode="nearest") for v in (right, left))
    # Side branches / overlaps are where the segmentation's outline shows one; the guide's
    # diameter is kept there, as superimposed vessels put intensity steps inside the lumen (D45).
    # One edge much nearer than the other is not a branch: the centreline (the mask's skeleton)
    # need not lie on the image's lumen axis, and an eccentric lesion narrows one side only.
    use = ok_r & ok_l & ~guide.branch
    return DiameterProfile(
        diameter_px=np.maximum(np.where(use, right + left, guide.diameter_px), MIN_DIAMETER_PX),
        left_px=np.where(use, -left, guide.left_px),
        right_px=np.where(use, right, guide.right_px),
        fallback=np.where(use, False, guide.fallback),
        branch=guide.branch.copy(),
    )


@dataclass(frozen=True, eq=False)
class QCAResult:
    """QCA of one vessel path.

    ``profile`` (SI units, m) is what the physics solver and the surrogate receive: diameters
    above REF_UPPER × the reference (``capped``) are capped there, and ``r_ref`` is the
    two-sided robust reference. ``diameters`` keeps the raw measurements in pixels.
    """

    centerline: Centerline
    diameters: DiameterProfile
    mm_per_px: float
    calibration_assumed: bool
    profile: VesselProfile
    capped: np.ndarray
    lumen_method: str = "probability"  # "intensity" (D43), "model" (P4c): LUMEN_METHOD_LABELS
    lumen_fit: LumenFit | None = None  # the imaging-chain fit, for lumen_method "model"
    size_factor: float = 1.0  # absolute size × this (scaled lumen, P5b); px diameters unscaled
    size_note: str = ""  # why a requested size factor was not applied

    @property
    def s_mm(self) -> np.ndarray:
        """Arc length from the start point (mm)."""
        return m_to_mm(self.profile.s)

    @property
    def diameter_mm(self) -> np.ndarray:
        """Lumen diameter used for the physics (mm): measured, capped at REF_UPPER × reference."""
        return m_to_mm(self.profile.diameter)

    @property
    def measured_diameter_mm(self) -> np.ndarray:
        """Raw measured diameter (mm), before capping."""
        return self.diameters.diameter_px * self.mm_per_px * self.size_factor

    @property
    def reference_diameter_mm(self) -> np.ndarray:
        """Healthy reference diameter (mm), see :func:`robust_reference`."""
        return m_to_mm(2.0 * self.profile.reference_radius())

    @property
    def capped_fraction(self) -> float:
        """Share of points capped at REF_UPPER × reference (branches, overlaps)."""
        return float(self.capped.mean())

    @property
    def percent_stenosis(self) -> np.ndarray:
        """Diameter stenosis along the vessel (%)."""
        return self.profile.percent_stenosis()

    @property
    def max_percent_stenosis(self) -> float:
        """Largest diameter stenosis (%)."""
        return float(self.percent_stenosis.max())

    @property
    def max_ds_point_px(self) -> np.ndarray:
        """Image position (row, col) of the largest diameter stenosis (px)."""
        return self.centerline.points[int(np.argmax(self.percent_stenosis))]

    @property
    def mld_mm(self) -> float:
        """Minimal lumen diameter (mm)."""
        return float(m_to_mm(self.profile.mld()[0]))

    @property
    def mld_index(self) -> int:
        """Centreline index of the MLD."""
        return int(self.profile.mld()[1])

    @property
    def mld_point_px(self) -> np.ndarray:
        """Image position (row, col) of the MLD (px)."""
        return self.centerline.points[self.mld_index]

    @property
    def lesions(self) -> list[Lesion]:
        """Segments with DS above the M1 threshold (positions in m)."""
        return self.profile.lesions()

    @property
    def no_lumen_segments(self) -> list[tuple[float, float]]:
        """(start, end) arc length (mm) of every run of at least ``NO_LUMEN_MIN_POINTS`` points
        whose measured lumen is at most ``NO_LUMEN_PX``: critical stenosis or total occlusion."""
        low = self.diameters.diameter_px <= NO_LUMEN_PX
        edges = np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0]))
        s = self.s_mm
        return [(float(s[a]), float(s[b - 1])) for a, b in zip(edges[::2], edges[1::2],
                strict=True) if b - a >= NO_LUMEN_MIN_POINTS]  # fmt: skip

    @property
    def bridged_mm(self) -> float:
        """Length (mm) of the path on bridged gaps of the mask (measured from the image)."""
        return float(self.centerline.bridged_mask.sum() * np.mean(np.diff(self.s_mm)))

    @property
    def fallback_fraction(self) -> float:
        """Share of points measured with the distance-transform fallback."""
        return float(self.diameters.fallback.mean())


GAP_SMOOTH_PX = 1.5  # image smoothing for re-centring a bridged gap


def recentre_gaps(image: np.ndarray, centerline: Centerline, guide: DiameterProfile) -> Centerline:
    """Bridged gaps (``centerline.bridged``) are straight lines between the vessel ends; move each
    of their points along its normal to the darkest point of the smoothed image within half the
    median vessel diameter (median-filtered along the gap), so that the lumen is measured from
    its centre. Other points and the arc length are kept. Units: px."""
    gap = centerline.bridged_mask
    if not gap.any() or gap.all():
        return centerline
    img = np.asarray(image, dtype=np.float32)
    img = gaussian_filter(img / 255.0 if img.max() > 1.0 else img, GAP_SMOOTH_PX)
    half = 0.5 * float(np.median(guide.diameter_px[~gap]))
    t = np.arange(-half, half + EDGE_STEP_PX / 2, EDGE_STEP_PX)
    pts, nrm = centerline.points[gap], centerline.normals[gap]
    q = pts[:, None, :] + t[None, :, None] * nrm[:, None, :]
    vals = map_coordinates(img, [q[..., 0].ravel(), q[..., 1].ravel()], order=1, mode="nearest")
    offset = t[np.argmin(vals.reshape(q.shape[:2]), axis=1)]
    offset = median_filter(offset, size=EDGE_MEDIAN_POINTS, mode="nearest")
    points = centerline.points.copy()
    points[gap] = pts + offset[:, None] * nrm
    return Centerline(points=points, s=centerline.s, bridged=centerline.bridged)


def _gap_guide(centerline: Centerline, guide: DiameterProfile) -> DiameterProfile:
    """The guide for the intensity edges, with the median measured diameter on bridged gaps:
    there the mask has no vessel (so no side branch either), and its tapering ends would limit
    the edge search."""
    gap = centerline.bridged_mask
    if not gap.any() or gap.all():
        return guide
    d = np.where(gap, float(np.median(guide.diameter_px[~gap])), guide.diameter_px)
    return DiameterProfile(
        d, guide.left_px, guide.right_px, guide.fallback | gap, guide.branch & ~gap
    )


def _no_lumen_in_gaps(centerline: Centerline, diameters: DiameterProfile) -> DiameterProfile:
    """Points on a bridged gap of the mask (``centerline.bridged``) whose lumen was not found
    (``fallback``: no edge in the image, or the mask has none) get ``MIN_DIAMETER_PX``, never a
    diameter interpolated from the healthy vessel around the gap (P4c)."""
    gap = centerline.bridged_mask & diameters.fallback
    if not gap.any():
        return diameters
    return DiameterProfile(
        diameter_px=np.where(gap, MIN_DIAMETER_PX, diameters.diameter_px),
        left_px=diameters.left_px,
        right_px=diameters.right_px,
        fallback=diameters.fallback,
        branch=diameters.branch & ~gap,
    )


def robust_reference(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Healthy reference diameter along the vessel (same units as ``d``).

    A Theil–Sen line (insensitive to up to ~29% outliers) is refined by least-squares refits on
    the points within [REF_LOWER, REF_UPPER] × the current line, at most REF_MAX_ITER times.
    Points below are narrowing; points above are side branches or overlapping vessels, which
    the M1 estimator (made for synthetic profiles without them) does not exclude.
    """
    slope, intercept, _, _ = theilslopes(d, s)
    fit = intercept + slope * s
    for _ in range(REF_MAX_ITER):
        keep = (d >= REF_LOWER * fit) & (d <= REF_UPPER * fit)
        if keep.sum() < 3:
            break
        new = np.polyval(np.polyfit(s[keep], d[keep], 1), s)
        if np.allclose(new, fit):
            break
        fit = new
    return np.maximum(fit, MIN_DIAMETER_PX)


def trim_ends(
    centerline: Centerline,
    diameters: DiameterProfile,
    trim_px: float,
    trim_end_px: float | None = None,
) -> tuple[Centerline, DiameterProfile]:
    """Drop the first ``trim_px`` and the last ``trim_end_px`` (default ``trim_px``) px of arc
    length; s restarts at 0.

    Nothing is trimmed if fewer than ``MIN_POINTS_AFTER_TRIM`` points would remain.
    """
    s = centerline.s
    end_px = trim_px if trim_end_px is None else trim_end_px
    keep = (s >= trim_px) & (s <= s[-1] - end_px)
    if keep.sum() < MIN_POINTS_AFTER_TRIM:
        return centerline, diameters
    bridged = None if centerline.bridged is None else centerline.bridged[keep]
    cl = Centerline(points=centerline.points[keep], s=s[keep] - s[keep][0], bridged=bridged)
    d = DiameterProfile(
        diameter_px=diameters.diameter_px[keep],
        left_px=diameters.left_px[keep],
        right_px=diameters.right_px[keep],
        fallback=diameters.fallback[keep],
        branch=diameters.branch[keep],
    )
    return cl, d


def qca_from_centerline(
    prob: np.ndarray,
    centerline: Centerline,
    *,
    mm_per_px: float | None = None,
    level: float = LEVEL,
    trim_diameters: float = END_TRIM_DIAMETERS,
    trim_start_diameters: float | None = None,
    image: np.ndarray | None = None,
    lumen: str = "edges",
    chain: ChainCalibration | None = None,
    size_factor: float | None = None,
) -> QCAResult:
    """QCA along a given centreline. ``mm_per_px=None`` uses the assumed default 0.30 mm/px.
    With ``image`` (the greyscale angiogram) the lumen edges come from its intensity
    (:func:`measure_diameters_intensity`, guided by the probability-map edges).
    ``lumen="model"`` (needs ``image``) replaces the edge diameters by the imaging-chain fit
    (:func:`~vstent.imaging.lumen_model.fit_lumen`), started from the intensity edges; where
    the fit cannot run (too few points), the edges are kept. ``chain``: the frame's calibrated
    blur and edge enhancement (:mod:`vstent.imaging.calibration`), fixed in that fit.

    ``trim_diameters`` × the median diameter is removed at both ends: where the centreline
    runs into the end of the mask (a vessel tip, an ostium, a cut-off branch) the measured
    width falls off for geometric reasons that are not lumen narrowing. ``trim_start_diameters``
    overrides the start (0 for a side branch that already starts at its parent's wall).

    ``lumen="scaled"`` (needs ``image``): edges, with the absolute size from the imaging-chain
    fit (:func:`size_factor_from_fit`). ``size_factor``: a known factor instead of the fit (a
    side branch takes its main vessel's).

    ``lumen="dense"`` (needs ``image``; P4h): the scaled lumen, narrowed where the integrated
    contrast shows less lumen than the edges (:mod:`vstent.imaging.densitometry`).
    ``"dense_log"`` / ``"dense_linear"`` name the domain explicitly.
    """
    if lumen not in LUMEN_OPTIONS:
        raise ValueError(f"lumen must be one of {LUMEN_OPTIONS}, not {lumen!r}")
    dense = lumen in DENSE_DOMAINS
    if dense and image is None:
        raise ValueError(f"the {lumen} lumen needs the image")
    if lumen in ("model", "scaled") and image is None and size_factor is None:
        raise ValueError(f"the {lumen} lumen needs the image")
    assumed = mm_per_px is None
    scale = DEFAULT_MM_PER_PX if assumed else float(mm_per_px)
    diameters = measure_diameters(prob, centerline, level=level)
    if image is not None:
        if centerline.bridged_mask.any():
            centerline = recentre_gaps(image, centerline, diameters)
            diameters = measure_diameters(prob, centerline, level=level)
        diameters = measure_diameters_intensity(
            image, centerline, _gap_guide(centerline, diameters)
        )
    diameters = _no_lumen_in_gaps(centerline, diameters)
    start = trim_diameters if trim_start_diameters is None else trim_start_diameters
    if trim_diameters > 0 or start > 0:
        median = float(np.median(diameters.diameter_px))
        centerline, diameters = trim_ends(
            centerline, diameters, start * median, trim_diameters * median
        )
    method = "probability" if image is None else "intensity"
    fit = None
    factor, note = 1.0, ""
    if lumen == "scaled" and size_factor is None:
        factor, note = size_factor_from_image(image, centerline, diameters)
        method = "scaled" if not note else method
    if dense:
        if size_factor is None:
            factor, note = size_factor_from_image(image, centerline, diameters)
        domain = DENSE_DOMAINS[lumen] or DOMAIN
        d_dens = densitometric_diameter(image, centerline, diameters.diameter_px, domain)
        d_new = np.maximum(np.minimum(diameters.diameter_px, d_dens), MIN_DIAMETER_PX)
        found = np.isfinite(diameters.left_px) & np.isfinite(diameters.right_px)
        mid = np.where(found, (diameters.left_px + diameters.right_px) / 2, 0.0)
        diameters = DiameterProfile(
            diameter_px=d_new,
            left_px=np.where(np.isfinite(diameters.left_px), mid - d_new / 2, np.nan),
            right_px=np.where(np.isfinite(diameters.right_px), mid + d_new / 2, np.nan),
            fallback=diameters.fallback,
            branch=diameters.branch,
        )
        method = "dense"
    if lumen == "model":
        edge_ref = robust_reference(centerline.s, diameters.diameter_px)
        offset = (diameters.right_px + diameters.left_px) / 2.0
        fit = fit_lumen(image, centerline, diameters.diameter_px, offset, edge_ref, chain=chain)
        if fit is not None:
            half = fit.diameter_px / 2.0
            diameters = DiameterProfile(
                diameter_px=np.maximum(fit.diameter_px, MIN_DIAMETER_PX),
                left_px=fit.offset_px - half,
                right_px=fit.offset_px + half,
                fallback=diameters.fallback,
                branch=diameters.branch,
            )
            method = "model"
    d = diameters.diameter_px
    d_ref = robust_reference(centerline.s, d)
    capped = d > REF_UPPER * d_ref
    d_used = np.where(capped, REF_UPPER * d_ref, d)
    if size_factor is not None:
        factor = float(size_factor)
        method = "scaled" if factor != 1.0 and not dense else method
    profile = VesselProfile(
        s=mm_to_m(centerline.s * scale),
        r=mm_to_m(d_used * scale * factor) / 2.0,
        r_ref=mm_to_m(d_ref * scale * factor) / 2.0,
    )
    return QCAResult(centerline, diameters, scale, assumed, profile, capped, method, fit, factor,
                     note)  # fmt: skip


def healthy_windows(healthy: np.ndarray, length: int, max_windows: int) -> list[slice]:
    """Up to ``max_windows`` non-overlapping windows of ``length`` consecutive healthy points,
    spread along the vessel."""
    ok = np.convolve(healthy.astype(int), np.ones(length, int), mode="valid") == length
    starts = np.flatnonzero(ok)
    chosen: list[int] = []
    for q in np.linspace(0.0, 1.0, max_windows + 2)[1:-1] if starts.size else []:
        free = [i for i in starts if all(abs(i - c) >= length for c in chosen)]
        if free:
            target = starts[0] + q * (starts[-1] - starts[0])
            chosen.append(min(free, key=lambda i: abs(i - target)))
    return [slice(int(i), int(i) + length) for i in sorted(chosen)]


def size_factor_from_image(
    image: np.ndarray, centerline: Centerline, diameters: DiameterProfile
) -> tuple[float, str]:
    """Absolute size factor of the scaled lumen (P5b): the imaging chain fitted jointly on up
    to ``SIZE_MAX_WINDOWS`` healthy windows (edges within [REF_LOWER, REF_UPPER] × their
    reference, no fallback, no side branch), and the median of fit / edge diameters there.
    Returns (factor, note): (1, reason) when no window fits, too few points were fitted, or
    the factor lies outside ``SIZE_FACTOR_RANGE``."""
    d = diameters.diameter_px
    edge_ref = robust_reference(centerline.s, d)
    healthy = ((d >= REF_LOWER * edge_ref) & (d <= REF_UPPER * edge_ref) & ~diameters.fallback
               & ~diameters.branch)  # fmt: skip
    windows = healthy_windows(healthy, SIZE_WINDOW_POINTS, SIZE_MAX_WINDOWS)
    offset = (diameters.right_px + diameters.left_px) / 2.0
    fitted = fit_windows(image, centerline, d, offset, edge_ref, windows) if windows else None
    if fitted is None:
        return 1.0, "no healthy stretch to fit the imaging chain on"
    ratio = np.concatenate([f / d[w] for f, w in zip(fitted, windows, strict=True)])
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size < SIZE_MIN_POINTS:
        return 1.0, "too few healthy points for the imaging-chain fit"
    factor = float(np.median(ratio))
    lo, hi = SIZE_FACTOR_RANGE
    if not lo <= factor <= hi:
        return 1.0, f"implausible size factor {factor:.2f} (accepted {lo:g}–{hi:g})"
    return factor, ""


def run_qca(
    prob: np.ndarray,
    start,
    end,
    *,
    mm_per_px: float | None = None,
    level: float = LEVEL,
    image: np.ndarray | None = None,
    lumen: str = "edges",
) -> QCAResult:
    """QCA of the vessel path between ``start`` and ``end`` ((row, col) px) on ``prob``; lumen
    from ``image`` when it is given (``lumen``: :func:`qca_from_centerline`)."""
    centerline = trace_centerline(np.asarray(prob) >= level, start, end)
    return qca_from_centerline(prob, centerline, mm_per_px=mm_per_px, level=level, image=image,
                               lumen=lumen)  # fmt: skip
