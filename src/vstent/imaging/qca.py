"""2D quantitative coronary analysis (QCA) on a vessel probability map.

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

The reference diameter is a two-sided robust line fit (:func:`robust_reference`); diameters
above 1.15 × reference (branches, overlaps) are capped before the profile, in SI units with
that explicit reference, goes to the M1 functions (%DS, lesions, MLD) and the physics.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, median_filter
from scipy.stats import theilslopes

from vstent.imaging.centerline import Centerline, trace_centerline
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
    edge is the steepest intensity rise along the normal (parabolic sub-pixel peak), searched
    up to 2 × the ``guide`` half-width + ``INTENSITY_SLACK_PX``. The classical edge criterion
    of QCA; it does not depend on how the segmentation draws the vessel's outline.

    Where the contrast (brightest − centre intensity in the search range) is below
    ``MIN_CONTRAST`` or an edge is missing, the ``guide`` diameter (probability-map edges) is
    kept. Side branches / overlaps: as :func:`measure_diameters`. Pixels throughout.
    """
    img = np.asarray(image, dtype=np.float32)
    img = img / 255.0 if img.max() > 1.0 else img
    img = gaussian_filter(img, smooth_px) if smooth_px > 0 else img
    pts, nrm = centerline.points, centerline.normals
    limit = guide.diameter_px + INTENSITY_SLACK_PX  # 2 × half-width + slack
    t = np.arange(0.0, float(limit.max()) + step_px / 2, step_px)
    in_range = t[None, :] <= limit[:, None]
    centre = map_coordinates(img, [pts[:, 0], pts[:, 1]], order=1, mode="nearest")

    def edge(sign: float) -> tuple[np.ndarray, np.ndarray]:
        q = pts[:, None, :] + sign * t[None, :, None] * nrm[:, None, :]
        vals = map_coordinates(img, [q[..., 0].ravel(), q[..., 1].ravel()], order=1,
                               mode="nearest").reshape(q.shape[:2])  # fmt: skip
        grad = np.gradient(vals, step_px, axis=1)
        grad = np.where(in_range, grad, -np.inf)
        k = np.argmax(grad, axis=1)
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
    both = ok_r & ok_l
    near, far = np.minimum(left, right), np.maximum(left, right)
    branch = both & (far > BRANCH_RATIO * near + BRANCH_SLACK_PX)
    diameter = np.where(branch, 2.0 * near, right + left)
    use = both
    return DiameterProfile(
        diameter_px=np.maximum(np.where(use, diameter, guide.diameter_px), MIN_DIAMETER_PX),
        left_px=np.where(use, -left, guide.left_px),
        right_px=np.where(use, right, guide.right_px),
        fallback=np.where(use, False, guide.fallback),
        branch=np.where(use, branch, guide.branch),
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
        return self.diameters.diameter_px * self.mm_per_px

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
    def fallback_fraction(self) -> float:
        """Share of points measured with the distance-transform fallback."""
        return float(self.diameters.fallback.mean())


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
    cl = Centerline(points=centerline.points[keep], s=s[keep] - s[keep][0])
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
) -> QCAResult:
    """QCA along a given centreline. ``mm_per_px=None`` uses the assumed default 0.30 mm/px.
    With ``image`` (the greyscale angiogram) the lumen edges come from its intensity
    (:func:`measure_diameters_intensity`, guided by the probability-map edges).

    ``trim_diameters`` × the median diameter is removed at both ends: where the centreline
    runs into the end of the mask (a vessel tip, an ostium, a cut-off branch) the measured
    width falls off for geometric reasons that are not lumen narrowing. ``trim_start_diameters``
    overrides the start (0 for a side branch that already starts at its parent's wall).
    """
    assumed = mm_per_px is None
    scale = DEFAULT_MM_PER_PX if assumed else float(mm_per_px)
    diameters = measure_diameters(prob, centerline, level=level)
    if image is not None:
        diameters = measure_diameters_intensity(image, centerline, diameters)
    start = trim_diameters if trim_start_diameters is None else trim_start_diameters
    if trim_diameters > 0 or start > 0:
        median = float(np.median(diameters.diameter_px))
        centerline, diameters = trim_ends(
            centerline, diameters, start * median, trim_diameters * median
        )
    d = diameters.diameter_px
    d_ref = robust_reference(centerline.s, d)
    capped = d > REF_UPPER * d_ref
    d_used = np.where(capped, REF_UPPER * d_ref, d)
    profile = VesselProfile(
        s=mm_to_m(centerline.s * scale),
        r=mm_to_m(d_used * scale) / 2.0,
        r_ref=mm_to_m(d_ref * scale) / 2.0,
    )
    return QCAResult(centerline, diameters, scale, assumed, profile, capped)


def run_qca(
    prob: np.ndarray,
    start,
    end,
    *,
    mm_per_px: float | None = None,
    level: float = LEVEL,
) -> QCAResult:
    """QCA of the vessel path between ``start`` and ``end`` ((row, col) px) on ``prob``."""
    centerline = trace_centerline(np.asarray(prob) >= level, start, end)
    return qca_from_centerline(prob, centerline, mm_per_px=mm_per_px, level=level)
