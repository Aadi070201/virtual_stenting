"""Synthetic vessel images with a known centreline and diameter (tests and demo).

A tube is rendered by giving every pixel the coverage ``clip(r(s*) − dist + 0.5, 0, 1)``,
where ``dist`` is the distance to the nearest point of a densely sampled centreline and
``r(s*)`` the radius there, then blurring and adding noise. All units are pixels.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


@dataclass(frozen=True, eq=False)
class SyntheticTube:
    """Rendered tube and its ground truth.

    Attributes:
        image: (H, W) float32 in [0, 1]; 1 inside the lumen (a probability-like map).
        points: (n, 2) dense centreline (row, col) px.
        s: (n,) arc length (px).
        diameter: (n,) true diameter (px).
    """

    image: np.ndarray
    points: np.ndarray
    s: np.ndarray
    diameter: np.ndarray


def sine_centerline(
    origin: tuple[float, float],
    length_px: float,
    *,
    angle_deg: float = 0.0,
    amplitude_px: float = 20.0,
    wavelength_px: float = 200.0,
    n: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Gently curved centreline: a sine wave rotated by ``angle_deg``. Returns (points, s)."""
    x = np.linspace(0.0, length_px, n)
    y = amplitude_px * np.sin(2.0 * np.pi * x / wavelength_px)
    a = np.deg2rad(angle_deg)
    rows = origin[0] + x * np.sin(a) + y * np.cos(a)
    cols = origin[1] + x * np.cos(a) - y * np.sin(a)
    pts = np.stack([rows, cols], axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, s


def stenosis_diameter(
    s: np.ndarray, ref_px: float, severity: float, center_px: float, length_px: float
) -> np.ndarray:
    """Constant reference diameter with one cosine narrowing of DS = 100 · severity %."""
    d = np.full_like(s, ref_px, dtype=float)
    inside = np.abs(s - center_px) < length_px / 2
    phase = 2.0 * np.pi * (s[inside] - center_px) / length_px
    d[inside] *= 1.0 - severity * 0.5 * (1.0 + np.cos(phase))
    return d


def render_tube(
    shape: tuple[int, int],
    points: np.ndarray,
    s: np.ndarray,
    diameter: np.ndarray,
    *,
    blur_sigma: float = 1.0,
    noise_std: float = 0.03,
    seed: int = 0,
) -> SyntheticTube:
    """Render a tube into an image of ``shape`` (H, W) with blur and Gaussian noise."""
    img = render_tubes(
        shape, [(points, diameter)], blur_sigma=blur_sigma, noise_std=noise_std, seed=seed
    )
    return SyntheticTube(img, points, s, diameter)


def render_tubes(
    shape: tuple[int, int],
    tubes: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    blur_sigma: float = 1.0,
    noise_std: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    """Union of several tubes, each ``(points, diameter)``, rendered as in :func:`render_tube`
    (e.g. a vessel and its side branch). Returns the (H, W) float32 image in [0, 1]."""
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]]
    pix = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(float)
    cover = np.zeros(pix.shape[0])
    for points, diameter in tubes:
        dist, idx = cKDTree(points).query(pix)
        cover = np.maximum(cover, np.clip(diameter[idx] / 2.0 - dist + 0.5, 0.0, 1.0))
    cover = cover.reshape(shape)
    img = gaussian_filter(cover, blur_sigma) if blur_sigma > 0 else cover
    img = img + np.random.default_rng(seed).normal(0.0, noise_std, size=shape)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
