"""The virtual X-ray: tubes (arteries, the catheter) projected through a C-arm onto a
detector, with a heart shadow, background, blur, edge enhancement and noise.

**Tubes.** At each centreline point the lumen is an ellipse (half-axes ``ra``, ``rb``). Seen
along the ray direction d, its half-width in the image is the ellipse's extent across both d
and the vessel (√((ra·(ea·n))² + (rb·(eb·n))²), n ∝ d × tangent), and the ray through its
centre crosses a chord c₀ / sin ∠(d, tangent) (c₀: the chord through the cross-section along
d's component in it), capped at 1 / ``MIN_SINE`` where a vessel runs along the ray
(foreshortening: a short, dark vessel). Off the centreline the chord falls as c·√(1 − u²),
u = distance / half-width (the chord profile of any ellipse).

**Sums.** X-ray attenuation adds along a ray: where vessels cross, their chords add (a crossing
looks darker), also where a vessel crosses itself. A tube is drawn in pieces of
``CHUNK_POINTS`` points; a pixel beyond a piece's end belongs to the next piece, so pieces do
not double-count where they meet. Transmission exp(−Σ μ·chord) is averaged over
``supersample``² sub-pixels (thin lumens fade as on a real detector).

**Imaging chain** (ranges fitted to ARCADE in vstent, D39; copied, docs/DECISIONS.md V2):
contrast attenuation μ per mm of lumen, background level and texture, Gaussian blur, edge
enhancement (unsharp mask) and Gaussian noise. Units: SI (m) unless named; chords in mm.
"""

import math
from dataclasses import dataclass

import numpy as np
from pydantic import Field
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree

from vcathlab.config import Range, StrictModel
from vcathlab.geometry import CArm

MIN_SINE = 0.3  # a vessel along the ray: chord ≤ 2r / 0.3
CHUNK_POINTS = 150  # 15 mm at the 0.1 mm centreline step
JOIN_MARGIN_M = 0.2e-3  # lumens closer than r₁ + r₂ + this are joined (they touch)


class ImagingConfig(StrictModel):
    """The imaging chain (uniform over [low, high]; ARCADE-fitted in vstent, D39)."""

    mu_per_mm: Range = (0.04, 0.08)  # contrast attenuation per mm of lumen
    background: Range = (0.49, 0.64)
    texture_sd: float = Field(0.04, ge=0)
    texture_scale_px: float = Field(40.0, gt=0)
    blur_px: Range = (0.8, 1.2)
    edge_amount: Range = (2.0, 3.0)
    edge_sigma_px: Range = (1.5, 3.0)
    noise_sd: Range = (0.012, 0.022)
    heart_mu_per_mm: Range = (0.001, 0.002)  # the heart's soft-tissue shadow
    wall_mu_share: float = Field(0.15, ge=0)  # catheter wall attenuation / contrast's
    diaphragm_prob: float = Field(0.6, ge=0, le=1)
    ribs: tuple[int, int] = (0, 4)
    vignetting: Range = (0.0, 0.15)
    supersample: int = Field(2, ge=1)


@dataclass(frozen=True)
class Imaging:
    """One image's drawn imaging-chain values."""

    mu_per_mm: float
    background: float
    blur_px: float
    edge_amount: float
    edge_sigma_px: float
    noise_sd: float
    heart_mu_per_mm: float
    vignetting: float

    @classmethod
    def sample(cls, rng: np.random.Generator, cfg: ImagingConfig) -> "Imaging":
        return cls(
            *(
                rng.uniform(*getattr(cfg, k))
                for k in (
                    "mu_per_mm",
                    "background",
                    "blur_px",
                    "edge_amount",
                    "edge_sigma_px",
                    "noise_sd",
                    "heart_mu_per_mm",
                    "vignetting",
                )
            )
        )


@dataclass(frozen=True)
class Projected:
    """A tube in one view: image points (n, 2) px, half-width (n,) px, central chord (n,) mm,
    unit image tangent (n, 2), a label id per point (n,) (e.g. the segment), the tube's id, its
    material ("contrast" or "wall"), and its 3D points (n, 3) m and largest radius (n,) m."""

    px: np.ndarray
    half_width: np.ndarray
    chord_mm: np.ndarray
    tangent: np.ndarray
    label: np.ndarray
    tube: int
    material: str = "contrast"
    points3d: np.ndarray | None = None
    radius3d: np.ndarray | None = None


def project_tube(
    carm: CArm,
    points,
    ra,
    rb,
    ea,
    eb,
    tangent,
    label,
    tube: int,
    material: str = "contrast",
) -> Projected:
    """Project an elliptical-lumen tube (points (n, 3) m; half-axes (n,) m along unit axes
    (n, 3) ⟂ the unit tangent (n, 3)) into ``carm``'s image."""
    d = carm.ray_directions(points)
    cos = np.sum(d * tangent, axis=1)
    sine = np.maximum(np.sqrt(np.clip(1.0 - cos**2, 0.0, 1.0)), MIN_SINE)
    nrm = np.cross(d, tangent)
    nn = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(nn > 1e-9, nrm / np.maximum(nn, 1e-12), ea)
    w3 = np.sqrt((ra * np.sum(ea * nrm, axis=1)) ** 2 + (rb * np.sum(eb * nrm, axis=1)) ** 2)
    m = d - cos[:, None] * tangent
    mn = np.linalg.norm(m, axis=1, keepdims=True)
    m = np.where(mn > 1e-9, m / np.maximum(mn, 1e-12), eb)
    with np.errstate(divide="ignore"):
        c0 = 2.0 / np.sqrt((np.sum(m * ea, axis=1) / ra) ** 2 + (np.sum(m * eb, axis=1) / rb) ** 2)
    px = carm.project(points)
    tan = np.gradient(px, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    return Projected(
        px,
        w3 / carm.pixel_size_at(points),
        1e3 * c0 / sine,
        tan,
        np.asarray(label),
        tube,
        material,
        np.asarray(points, float),
        np.maximum(ra, rb),
    )


@dataclass(frozen=True)
class Layers:
    """Per-pixel maps of one view at ``supersample`` resolution: summed chord (mm) per
    material, the label with the largest chord, and where two tubes (or distant parts of one)
    overlap."""

    contrast_mm: np.ndarray
    wall_mm: np.ndarray
    label: np.ndarray
    overlap: np.ndarray
    supersample: int


def rasterise(shape: tuple[int, int], tubes: list[Projected], supersample: int = 2) -> Layers:
    """Every tube's chords on a ``supersample``-times finer grid (see the module docstring):
    where a pixel's contributions come from points joined in 3D (closer than the sum of their
    radii plus ``JOIN_MARGIN_M``: a branch point, two pieces of one vessel) the larger counts;
    otherwise (a crossing) they add, and the pixel is marked as an overlap."""
    ss = supersample
    h, w = shape[0] * ss, shape[1] * ss
    acc = {"contrast": np.zeros((h, w)), "wall": np.zeros((h, w))}
    top = np.zeros((h, w))  # the largest contrast contribution so far ...
    top_p = np.zeros((h, w, 3))  # ... its 3D point (m) and radius (m)
    top_r = np.zeros((h, w))
    label = np.full((h, w), -1, dtype=np.int32)
    overlap = np.zeros((h, w), dtype=bool)
    for t in tubes:
        n = len(t.px)
        if n < 2:
            continue
        pts = (t.px + 0.5) * ss - 0.5  # sub-pixel coordinates
        hw = t.half_width * ss
        bounds = list(range(0, n - 1, CHUNK_POINTS)) + [n - 1]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            idx = np.arange(a, b + 1)
            reach = float(hw[idx].max()) + 1.0
            lo = np.floor(pts[idx].min(axis=0) - reach).astype(int)
            hi = np.ceil(pts[idx].max(axis=0) + reach).astype(int) + 1
            lo, hi = np.maximum(lo, 0), np.minimum(hi, (h, w))
            if np.any(hi <= lo):
                continue
            rr, cc = np.mgrid[lo[0] : hi[0], lo[1] : hi[1]]
            q = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(float)
            dist, j = cKDTree(pts[idx]).query(q, distance_upper_bound=reach)
            ok = np.isfinite(dist)
            i = idx[np.where(ok, j, 0)]
            u = np.where(ok, dist / np.maximum(hw[i], 1e-9), 2.0)
            inside = ok & (u < 1.0)
            if not inside.any():
                continue
            i = i[inside]
            c = t.chord_mm[i] * np.sqrt(1.0 - u[inside] ** 2)
            r, col = rr.ravel()[inside], cc.ravel()[inside]
            if t.material == "wall":
                np.maximum.at(acc["wall"], (r, col), c)
                continue
            prev = top[r, col]
            gap = np.linalg.norm(t.points3d[i] - top_p[r, col], axis=1)
            joined = (prev > 0) & (gap < t.radius3d[i] + top_r[r, col] + JOIN_MARGIN_M)
            acc["contrast"][r, col] += np.where(joined, np.maximum(c - prev, 0.0), c)
            cross = (prev > 0) & ~joined
            overlap[r[cross], col[cross]] = True
            better = c > prev
            rb, cb = r[better], col[better]
            top[rb, cb] = c[better]
            top_p[rb, cb] = t.points3d[i[better]]
            top_r[rb, cb] = t.radius3d[i[better]]
            label[rb, cb] = t.label[i[better]]
    return Layers(acc["contrast"], acc["wall"], label, overlap, ss)


def heart_shadow(carm: CArm, centre, axis, a: float, b: float, grid: int = 64) -> np.ndarray:
    """Chord (mm) of each pixel's ray through the heart spheroid (centre (3,) m, unit long
    ``axis``, half-axes a, b m), computed on a ``grid``² grid and smoothly enlarged."""
    rows = np.linspace(0, carm.shape[0] - 1, grid)
    cols = np.linspace(0, carm.shape[1] - 1, grid)
    u, down, right = carm.axes()
    r0, c0 = carm.centre_px()
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    v = (
        carm.sid_m * u
        + ((rr - r0) * carm.pixel_m)[..., None] * down
        + ((cc - c0) * carm.pixel_m)[..., None] * right
    ).reshape(-1, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    o = carm.source - np.asarray(centre)

    # spheroid: (x·axis)²/a² + |x − (x·axis) axis|²/b² = 1, with x = o + t v
    def quad(x, y):
        return (x @ axis) * (y @ axis) / a**2 + (np.sum(x * y, -1) - (x @ axis) * (y @ axis)) / b**2

    qa, qb, qc = quad(v, v), 2.0 * quad(v, o[None, :].repeat(len(v), 0)), quad(o, o) - 1.0
    disc = np.maximum(qb**2 - 4.0 * qa * qc, 0.0)
    chord = 1e3 * np.sqrt(disc) / qa  # (t₂ − t₁) · |v| with |v| = 1, in mm
    return zoom(chord.reshape(grid, grid), (carm.shape[0] / grid, carm.shape[1] / grid), order=1)


def background_map(
    rng: np.random.Generator, shape: tuple[int, int], im: Imaging, cfg: ImagingConfig
) -> np.ndarray:
    """The background level with low-frequency texture, a diaphragm, rib bands and
    vignetting (H, W) in [0, 1]."""
    h, w = shape
    bg = np.full(shape, im.background)
    tex = gaussian_filter(rng.normal(size=shape), cfg.texture_scale_px)
    bg *= 1.0 + cfg.texture_sd * tex / max(float(np.std(tex)), 1e-9)
    rr, cc = np.mgrid[0:h, 0:w]
    if rng.random() < cfg.diaphragm_prob:
        level = rng.uniform(0.7, 1.0) * h + 0.15 * h * np.sin(
            cc / w * math.pi * rng.uniform(0.5, 1.5)
        )
        bg *= 1.0 - rng.uniform(0.1, 0.3) / (1.0 + np.exp(-(rr - level) / rng.uniform(8, 25)))
    for _ in range(int(rng.integers(cfg.ribs[0], cfg.ribs[1] + 1))):
        y0, tilt, width = rng.uniform(0, h), rng.uniform(-0.4, 0.4), rng.uniform(12, 30)
        band = np.exp(-0.5 * ((rr - (y0 + tilt * (cc - w / 2))) / width) ** 2)
        bg *= 1.0 - rng.uniform(0.02, 0.06) * band
    radius2 = ((rr - h / 2) / (h / 2)) ** 2 + ((cc - w / 2) / (w / 2)) ** 2
    bg *= 1.0 - im.vignetting * np.clip(radius2 / 2.0, 0.0, 1.0)
    return np.clip(bg, 0.0, 1.0)


def expose(
    rng: np.random.Generator,
    layers: Layers,
    heart_mm: np.ndarray,
    bg: np.ndarray,
    im: Imaging,
    cfg: ImagingConfig,
) -> np.ndarray:
    """The detector image (H, W) uint8: transmission through contrast, catheter wall and heart,
    times the background, then blur, edge enhancement and noise."""
    ss = layers.supersample
    att = im.mu_per_mm * (layers.contrast_mm + cfg.wall_mu_share * layers.wall_mm)
    trans = np.exp(-att)
    h, w = trans.shape[0] // ss, trans.shape[1] // ss
    trans = trans.reshape(h, ss, w, ss).mean(axis=(1, 3))
    img = trans * np.exp(-im.heart_mu_per_mm * heart_mm) * bg
    img = gaussian_filter(img, im.blur_px)
    img = img + im.edge_amount * (img - gaussian_filter(img, im.edge_sigma_px))
    img = img + rng.normal(0.0, im.noise_sd, size=img.shape)
    return np.round(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def downsample_layers(layers: Layers) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-level maps: lumen coverage (share of sub-pixels inside a contrast lumen), the
    label of the sub-pixel with the largest chord, and overlap (any sub-pixel)."""
    ss = layers.supersample
    h, w = layers.contrast_mm.shape[0] // ss, layers.contrast_mm.shape[1] // ss
    cover = (layers.contrast_mm > 0).reshape(h, ss, w, ss).mean(axis=(1, 3))
    blocks = layers.contrast_mm.reshape(h, ss, w, ss).transpose(0, 2, 1, 3).reshape(h, w, -1)
    labels = layers.label.reshape(h, ss, w, ss).transpose(0, 2, 1, 3).reshape(h, w, -1)
    pick = np.take_along_axis(labels, blocks.argmax(axis=2)[..., None], axis=2)[..., 0]
    over = layers.overlap.reshape(h, ss, w, ss).any(axis=(1, 3))
    return cover, pick, over
