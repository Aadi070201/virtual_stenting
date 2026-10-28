"""2D quantitative coronary analysis (QCA) on a vessel probability map.

The lumen diameter at each centreline point is the distance between the two crossings of the
probability map with the 0.5 level along the local normal (bilinear sampling, linear sub-pixel
interpolation of the crossing). Where that fails (centre below the level, or no crossing within
the search range) the fallback is 2 × the distance transform of the binary mask.

One median vessel diameter is trimmed from both ends of the centreline (end effects of the
mask, not lumen narrowing). Pixels are converted to millimetres with a calibration factor.
**Without a calibration from the image (e.g. DICOM pixel spacing or a catheter of known size)
the default 0.30 mm/px is an assumption**, and every result records whether it was assumed.
The profile is then handed to the M1 functions (reference diameter, %DS, lesions, MLD) in SI
units.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from vstent.imaging.centerline import Centerline, trace_centerline
from vstent.physics.geometry import Lesion, VesselProfile
from vstent.units import m_to_mm, mm_to_m

DEFAULT_MM_PER_PX = 0.30
LEVEL = 0.5
EDGE_STEP_PX = 0.1
MIN_DIAMETER_PX = 0.5
END_TRIM_DIAMETERS = 1.0
MIN_POINTS_AFTER_TRIM = 10


@dataclass(frozen=True, eq=False)
class DiameterProfile:
    """Lumen diameter along a centreline (all in pixels).

    Attributes:
        diameter_px: (n,) diameter.
        left_px, right_px: (n,) signed edge offsets along the normal (left ≤ 0 ≤ right);
            NaN where the fallback was used.
        fallback: (n,) True where 2 × distance transform was used.
    """

    diameter_px: np.ndarray
    left_px: np.ndarray
    right_px: np.ndarray
    fallback: np.ndarray


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
    ok = ok_r & ok_l
    fallback_d = 2.0 * map_coordinates(edt, [pts[:, 0], pts[:, 1]], order=1, mode="nearest")
    diameter = np.where(ok, right + left, fallback_d)
    return DiameterProfile(
        diameter_px=np.maximum(diameter, MIN_DIAMETER_PX),
        left_px=np.where(ok, -left, np.nan),
        right_px=np.where(ok, right, np.nan),
        fallback=~ok,
    )


@dataclass(frozen=True, eq=False)
class QCAResult:
    """QCA of one vessel path. ``profile`` is in SI units (m) for the physics solver."""

    centerline: Centerline
    diameters: DiameterProfile
    mm_per_px: float
    calibration_assumed: bool
    profile: VesselProfile

    @property
    def s_mm(self) -> np.ndarray:
        """Arc length from the start point (mm)."""
        return m_to_mm(self.profile.s)

    @property
    def diameter_mm(self) -> np.ndarray:
        """Measured lumen diameter (mm)."""
        return m_to_mm(self.profile.diameter)

    @property
    def reference_diameter_mm(self) -> np.ndarray:
        """Estimated healthy reference diameter (mm), robust linear fit (M1)."""
        return m_to_mm(2.0 * self.profile.reference_radius())

    @property
    def percent_stenosis(self) -> np.ndarray:
        """Diameter stenosis along the vessel (%)."""
        return self.profile.percent_stenosis()

    @property
    def max_percent_stenosis(self) -> float:
        """Largest diameter stenosis (%)."""
        return float(self.percent_stenosis.max())

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


def trim_ends(
    centerline: Centerline, diameters: DiameterProfile, trim_px: float
) -> tuple[Centerline, DiameterProfile]:
    """Drop the first and last ``trim_px`` (px) of arc length; s restarts at 0.

    Nothing is trimmed if fewer than ``MIN_POINTS_AFTER_TRIM`` points would remain.
    """
    s = centerline.s
    keep = (s >= trim_px) & (s <= s[-1] - trim_px)
    if keep.sum() < MIN_POINTS_AFTER_TRIM:
        return centerline, diameters
    cl = Centerline(points=centerline.points[keep], s=s[keep] - s[keep][0])
    d = DiameterProfile(
        diameter_px=diameters.diameter_px[keep],
        left_px=diameters.left_px[keep],
        right_px=diameters.right_px[keep],
        fallback=diameters.fallback[keep],
    )
    return cl, d


def qca_from_centerline(
    prob: np.ndarray,
    centerline: Centerline,
    *,
    mm_per_px: float | None = None,
    level: float = LEVEL,
    trim_diameters: float = END_TRIM_DIAMETERS,
) -> QCAResult:
    """QCA along a given centreline. ``mm_per_px=None`` uses the assumed default 0.30 mm/px.

    ``trim_diameters`` × the median diameter is removed at both ends: where the centreline
    runs into the end of the mask (a vessel tip, an ostium, a cut-off branch) the measured
    width falls off for geometric reasons that are not lumen narrowing.
    """
    assumed = mm_per_px is None
    scale = DEFAULT_MM_PER_PX if assumed else float(mm_per_px)
    diameters = measure_diameters(prob, centerline, level=level)
    if trim_diameters > 0:
        trim = trim_diameters * float(np.median(diameters.diameter_px))
        centerline, diameters = trim_ends(centerline, diameters, trim)
    profile = VesselProfile(
        s=mm_to_m(centerline.s * scale),
        r=mm_to_m(diameters.diameter_px * scale) / 2.0,
    )
    return QCAResult(centerline, diameters, scale, assumed, profile)


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
