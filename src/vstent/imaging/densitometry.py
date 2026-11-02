"""Densitometric lumen (P4h): the contrast a vessel carries, not its blurred edges. Torch-free.

Edge-based QCA has a resolution floor: a lumen narrower than the blur still shows two edges a
few pixels apart, so tight lesions in small vessels read mild (D44, D60). What such a lesion
does change is how much contrast passes: by Beer–Lambert the attenuation along a ray through
the lumen is μ × chord, and summed across the vessel it is μ × the lumen's cross-sectional
area. Blur and the X-ray system's edge enhancement are linear filters that keep that sum.

At every centreline point (pre-registered in docs/PLAN.md, P4h):

1. the image is sampled along the normal every :data:`STEP_PX` over ± (half the edges' robust
   reference diameter + :data:`MARGIN_PX`), as the mean of three parallel profiles
   (:data:`TANGENT_OFFSETS_PX` along the tangent);
2. the background is a straight line through the medians of the outer :data:`BAND_PX` on each
   side, its parameters median-filtered over :data:`BACKGROUND_MEDIAN_POINTS` along the vessel;
3. the integrated contrast (IOD) is the signed sum of background − image over the window,
   median-filtered over :data:`IOD_MEDIAN_POINTS`, then smoothed along the vessel by a
   Gaussian of :data:`IOD_SIGMA_PX` (amendment 1: a tight lesion's IOD is of the order of the
   noise); in the ``log`` domain (an image proportional to the transmitted X-rays) or the
   ``linear`` one (grey levels already log-transformed by the X-ray system);
4. the densitometric diameter is the edges' reference diameter × √(IOD / reference IOD), the
   reference IOD being :func:`~vstent.imaging.qca.robust_reference` of √IOD.

The lumen is then the smaller of the edge and the densitometric diameter
(:func:`~vstent.imaging.qca.qca_from_centerline`, ``lumen="dense"``): an overlapping vessel or a
branch raises the IOD and is left to the edges. Units: px.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter

from vstent.imaging.centerline import Centerline

STEP_PX = 0.5
MARGIN_PX = 6.0
BAND_PX = 3.0
TANGENT_OFFSETS_PX = (-1.0, 0.0, 1.0)
BACKGROUND_MEDIAN_POINTS = 9
IOD_MEDIAN_POINTS = 3
IOD_SIGMA_PX = 3.0  # then a Gaussian along the vessel (P4h amendment 1)
DOMAINS = ("log", "linear")
DOMAIN = "log"  # chosen by the P4h rule on the development patients (D61)
_SCALE = 10.0  # √IOD is brought to diameter-like magnitudes for robust_reference's floor


def integrated_contrast(
    image: np.ndarray, centerline: Centerline, ref_diameter_px: np.ndarray, domain: str = DOMAIN
) -> np.ndarray:
    """(n,) integrated contrast (IOD) at each centreline point: px × (log or grey) units.
    ``ref_diameter_px``: (n,) the edges' healthy reference diameter, which sets the window."""
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, not {domain!r}")
    pts, nrm, tan = centerline.points, centerline.normals, centerline.tangents
    half = 0.5 * np.asarray(ref_diameter_px, float) + MARGIN_PX
    x = np.arange(-float(half.max()), float(half.max()) + 1e-9, STEP_PX)
    img = np.asarray(image, np.float64)
    if domain == "log":
        img = np.log(np.maximum(img, 1.0))
    prof = np.zeros((len(pts), x.size))
    for off in TANGENT_OFFSETS_PX:
        rows = pts[:, :1] + off * tan[:, :1] + x[None, :] * nrm[:, :1]
        cols = pts[:, 1:] + off * tan[:, 1:] + x[None, :] * nrm[:, 1:]
        prof += map_coordinates(img, [rows, cols], order=1, mode="nearest")
    prof /= len(TANGENT_OFFSETS_PX)
    lo, hi = -half[:, None], half[:, None]
    inside = (x[None, :] >= lo) & (x[None, :] <= hi)
    left = inside & (x[None, :] <= lo + BAND_PX)
    right = inside & (x[None, :] >= hi - BAND_PX)
    b_left = np.nanmedian(np.where(left, prof, np.nan), axis=1)
    b_right = np.nanmedian(np.where(right, prof, np.nan), axis=1)
    centre = half - BAND_PX / 2  # the bands' centres at ± this
    slope = (b_right - b_left) / (2 * centre)
    level = (b_left + b_right) / 2
    slope = median_filter(slope, BACKGROUND_MEDIAN_POINTS, mode="nearest")
    level = median_filter(level, BACKGROUND_MEDIAN_POINTS, mode="nearest")
    background = level[:, None] + slope[:, None] * x[None, :]
    iod = np.sum(np.where(inside, background - prof, 0.0), axis=1) * STEP_PX
    iod = median_filter(iod, IOD_MEDIAN_POINTS, mode="nearest")
    return gaussian_filter1d(iod, IOD_SIGMA_PX, mode="nearest")


def densitometric_diameter(
    image: np.ndarray, centerline: Centerline, edge_diameter_px: np.ndarray, domain: str = DOMAIN
) -> np.ndarray:
    """(n,) lumen diameter from the integrated contrast, scaled to the edges' healthy
    reference: reference × √(IOD / reference IOD)."""
    from vstent.imaging.qca import robust_reference

    ref = robust_reference(centerline.s, np.asarray(edge_diameter_px, float))
    e = np.sqrt(np.maximum(integrated_contrast(image, centerline, ref, domain), 0.0))
    typical = float(np.median(e[e > 0])) if (e > 0).any() else 0.0
    if typical <= 0:
        return np.full(e.shape, np.inf)  # no contrast measured: leave the edges alone
    k = _SCALE / typical
    e_ref = robust_reference(centerline.s, e * k) / k
    return ref * e / e_ref
