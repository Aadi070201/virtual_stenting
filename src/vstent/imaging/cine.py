"""Cine runs: opacification curve and the analysis frame (P4d). Torch-free.

Contrast arrives, fills the vessels for a while and washes out. The segmented vessel area per
frame traces this; the opacified window is the run of frames around its peak that stay above
``OPACIFIED_FRACTION`` of the peak, and the analysis frame is the frame of the largest
(smoothed) vessel area: the vessels most completely filled.

Contrast transit (:func:`contrast_transit`): along the analysed vessel's centreline, the frame
in which the contrast arrives at each point; the front's speed (px per frame) is the robust
slope of arc length on arrival frame, the quantity frame-count methods turn into flow. Frames
are 0-based; lengths in px.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter, minimum_filter
from scipy.stats import theilslopes

from vstent.imaging.centerline import Centerline

LEVEL = 0.5  # vessel probability threshold, as in QCA
SMOOTH_FRAMES = 3  # median filter of the area curve (single-frame segmentation dropouts)
OPACIFIED_FRACTION = 0.8  # window: frames above this share of the peak area (D47)
ARRIVAL_FRACTION = 0.5  # contrast arrival: first frame above this share of the peak area


@dataclass(frozen=True)
class Opacification:
    """Opacification of one run.

    Attributes:
        area_px: (T,) segmented vessel pixels per frame.
        smoothed: (T,) the same after a ``SMOOTH_FRAMES`` median filter.
        arrival: first frame whose smoothed area reaches ``ARRIVAL_FRACTION`` of the peak.
        window: (first, last) frame of the opacified window, inclusive.
        analysis_frame: frame of the largest smoothed area.
    """

    area_px: np.ndarray
    smoothed: np.ndarray
    arrival: int
    window: tuple[int, int]
    analysis_frame: int


def opacification(area_px: np.ndarray, fraction: float = OPACIFIED_FRACTION) -> Opacification:
    """Opacified window and analysis frame from the vessel area per frame."""
    area = np.asarray(area_px, dtype=float)
    if area.ndim != 1 or area.size == 0:
        raise ValueError("area_px must be a non-empty 1-D array")
    smooth = median_filter(area, SMOOTH_FRAMES, mode="nearest") if area.size >= 3 else area
    peak = int(np.argmax(smooth))
    top = smooth[peak]
    above = smooth >= fraction * top
    first = last = peak
    while first > 0 and above[first - 1]:
        first -= 1
    while last < area.size - 1 and above[last + 1]:
        last += 1
    arrival = int(np.argmax(smooth >= ARRIVAL_FRACTION * top))
    return Opacification(area, smooth, arrival, (first, last), peak)


def vessel_areas(
    frames: np.ndarray, segment: Callable[[np.ndarray], np.ndarray], level: float = LEVEL
) -> np.ndarray:
    """(T,) vessel pixels per frame: ``segment`` (frame → probability map) at ``level``."""
    return np.array([int((segment(f) >= level).sum()) for f in frames])


TRANSIT_SEARCH_PX = 6  # each point is searched this far in every frame (cardiac motion)
TRANSIT_SMOOTH_PX = 1.0  # frames smoothed before the search
TRANSIT_LEVEL = 0.5  # arrival: the frame the point reaches half its plateau contrast
TRANSIT_STEP_PX = 2.0  # spacing of the sampled centreline points
MIN_CONTRAST = 0.03  # plateau contrast (image in [0, 1]) below this: no arrival measured
MIN_TRANSIT_POINTS = 10


@dataclass(frozen=True)
class ContrastTransit:
    """Contrast arrival along a centreline.

    Attributes:
        s_px: (n,) arc length of the sampled points.
        arrival: (n,) arrival frame (fractional; NaN where no contrast was measured).
        inflow_px: arc length of the earliest arrival (where contrast enters the path).
        px_per_frame: front speed, Theil–Sen slope of the distance from the inflow on
            ``arrival``, over the longer side of the path from the inflow.
        interval: 95% interval of that slope.
        n_points: points used for the slope.
    """

    s_px: np.ndarray
    arrival: np.ndarray
    inflow_px: float
    px_per_frame: float
    interval: tuple[float, float]
    n_points: int

    def velocity_mm_s(self, mm_per_px: float, frame_time_ms: float) -> float:
        """Front speed in mm/s (needs the pixel size and the frame time)."""
        return self.px_per_frame * mm_per_px / (frame_time_ms / 1000.0)


def contrast_transit(
    frames: np.ndarray, centerline: Centerline, *, search_px: int = TRANSIT_SEARCH_PX
) -> ContrastTransit | None:
    """Contrast arrival along ``centerline`` (drawn on one frame of the run ``frames`` (T, H, W),
    vessels dark). In every frame each point takes the darkest value within ``search_px``; its
    contrast is its brightest such value over the run (no contrast) minus the current one, and
    its arrival the first frame at ``TRANSIT_LEVEL`` of its plateau, interpolated between
    frames. ``None`` if fewer than ``MIN_TRANSIT_POINTS`` points show contrast."""
    s_all = centerline.s
    s = np.arange(0.0, s_all[-1] + 1e-9, TRANSIT_STEP_PX)
    pts = np.stack([np.interp(s, s_all, centerline.points[:, k]) for k in range(2)], axis=1)
    yy, xx = np.mgrid[-search_px : search_px + 1, -search_px : search_px + 1]
    disk = yy**2 + xx**2 <= search_px**2
    imgs = np.asarray(frames, dtype=np.float32)
    imgs = imgs / 255.0 if imgs.max() > 1.0 else imgs
    dark = np.stack([
        map_coordinates(minimum_filter(gaussian_filter(f, TRANSIT_SMOOTH_PX), footprint=disk),
                        [pts[:, 0], pts[:, 1]], order=1, mode="nearest")
        for f in imgs
    ])  # fmt: skip  # (T, n)
    contrast = np.percentile(dark, 90, axis=0)[None, :] - dark
    plateau = np.percentile(contrast, 90, axis=0)
    arrival = np.full(s.size, np.nan)
    for i in np.where(plateau >= MIN_CONTRAST)[0]:
        c = contrast[:, i] / plateau[i]
        k = int(np.argmax(c >= TRANSIT_LEVEL))
        if k == 0:
            continue  # opacified from the first frame: arrival not seen
        frac = (TRANSIT_LEVEL - c[k - 1]) / max(c[k] - c[k - 1], 1e-9)
        arrival[i] = k - 1 + float(np.clip(frac, 0.0, 1.0))
    ok = np.isfinite(arrival)
    if ok.sum() < MIN_TRANSIT_POINTS:
        return None
    # a path may run through the inflow (e.g. LAD tip → left main → LCx tip): contrast arrives
    # first there, so the longer side from the earliest (median-smoothed) arrival is used
    first = int(np.argmin(median_filter(np.where(ok, arrival, np.inf), 5, mode="nearest")))
    side = np.arange(first, s.size) if s.size - first >= first + 1 else np.arange(first, -1, -1)
    side = side[ok[side]]
    if side.size < MIN_TRANSIT_POINTS or np.ptp(arrival[side]) <= 0:
        return None
    dist = np.abs(s[side] - s[first])
    slope, _, low, high = theilslopes(dist, arrival[side], alpha=0.95)
    return ContrastTransit(s, arrival, float(s[first]), float(slope), (float(low), float(high)),
                           int(side.size))  # fmt: skip
