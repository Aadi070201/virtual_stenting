"""Cardiac motion of the vessel tree across a cine run (P7). Torch-free.

Each frame's vessel probability map is registered to the analysis frame's with an affine
transform: OpenCV's ECC on half-size, smoothed maps, starting from the neighbouring frame's
transform and working outwards from the analysis frame. Frames with little vessel (before the
contrast arrives, after washout) take their neighbour's transform. The transforms map
analysis-frame (row, col) to each frame, so points on the analysed centreline can be followed
through the run (contrast transit, P7). Coordinates in px.
"""

from collections.abc import Sequence

import cv2
import numpy as np

SCALE = 0.5  # registration on maps resized by this factor
SMOOTH_PX = 2.0  # Gaussian smoothing of the resized maps
MIN_AREA_FRACTION = 0.2  # frames with less vessel than this share of the reference: no fit
ECC_ITERATIONS = 60
ECC_EPS = 1e-4
MAX_DRIFT = 0.25  # a fitted transform may not change scale / shear by more than this


def small_map(prob: np.ndarray) -> np.ndarray:
    """A probability map resized by ``SCALE`` and smoothed, as registered."""
    img = cv2.resize(np.asarray(prob, np.float32), None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_AREA)  # fmt: skip
    return cv2.GaussianBlur(img, (0, 0), SMOOTH_PX)


def _to_rowcol(w: np.ndarray) -> np.ndarray:
    """OpenCV (x = col, y = row) 2×3 warp at the small scale → (row, col) at full size."""
    a, b, tx = w[0]
    c, d, ty = w[1]
    return np.array([[d, c, ty / SCALE], [b, a, tx / SCALE]])


def track_affine(probs: Sequence[np.ndarray] | np.ndarray, ref: int) -> np.ndarray:
    """(T, 2, 3) affine transforms mapping frame ``ref``'s (row, col) to each frame's."""
    area = np.array([float((np.asarray(p) >= 0.5).sum()) for p in probs])
    return track_affine_maps([small_map(p) for p in probs], area, ref)


def track_affine_maps(small: Sequence[np.ndarray], area: np.ndarray, ref: int) -> np.ndarray:
    """:func:`track_affine` from :func:`small_map` maps and the vessel pixels per frame."""
    area = np.asarray(area, float)
    template = small[ref]
    eye = np.eye(2, 3, dtype=np.float32)
    warps: list[np.ndarray | None] = [None] * len(small)
    warps[ref] = eye.copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPS)
    for order in (range(ref + 1, len(small)), range(ref - 1, -1, -1)):
        prev = eye.copy()
        for j in order:
            w = prev.copy()
            if area[j] >= MIN_AREA_FRACTION * max(area[ref], 1.0):
                try:
                    _, fit = cv2.findTransformECC(template, small[j], w.copy(),
                                                  cv2.MOTION_AFFINE, criteria, None, 1)  # fmt: skip
                    if np.all(np.abs(fit[:, :2] - eye[:, :2]) <= MAX_DRIFT):
                        w = fit.astype(np.float32)
                except cv2.error:
                    pass
            warps[j] = w
            prev = w
    return np.stack([_to_rowcol(w) for w in warps])


def apply(warp: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(n, 2) (row, col) points moved by one (2, 3) transform."""
    return np.asarray(points, float) @ warp[:, :2].T + warp[:, 2]
