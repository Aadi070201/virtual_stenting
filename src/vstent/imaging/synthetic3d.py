"""Synthetic coronary trees in 3D, seen from two C-arm views (P5 test data). Not clinical data.

**Trees.** Lumen profiles, lesions and side branches come from the P2 tree sampler
(:func:`vstent.data.synth_tree.sample_tree`). Each branch is laid on the surface of a sphere
the size of the heart (the coronary arteries run on the epicardium): a curve of the branch's
length that starts at a random point (side branches at their take-off, at an angle to the
parent), turns with a smooth random curvature in the surface, and rises and falls slightly off
it. The heart's centre is offset at random from the isocentre.

**Views.** Two projections (:class:`~vstent.imaging.projection.ProjectionGeometry`) with
angles from the clinical range, at least ``separation_deg[0]`` apart. One C-arm: the source to
isocentre distance is shared, the detector distance varies per run, and the detector pixel is
chosen so that the pixel size at the isocentre matches ARCADE's (D39). In the motion scenario
the heart is shifted between the two runs. With ``avoid_overlap`` (the default), views are
chosen as QCA practice asks: the analysed (main) vessel overlaps neither itself nor another
branch in either view; foreshortening is not avoided (two views correct it). Without it, views
are random (a stress test).

**Images** (:func:`render_view`): the X-ray-like projection of
:func:`~vstent.imaging.synthetic.render_projection`; a segment at angle θ to the ray is crossed
along a chord of 2·r / sin θ. The segmentation stand-in is the unnarrowed reference tube, as in
the lumen report (an automatic segmentation smooths over narrowings). Units: SI unless named.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree

from vstent.config import StrictModel, load_config
from vstent.data.synth import Range
from vstent.data.synth_tree import SyntheticTree, TreeSynthConfig, sample_tree
from vstent.imaging.projection import ProjectionGeometry
from vstent.imaging.synthetic import render_projection, render_tubes
from vstent.units import mm_to_m

FINE_STEP_M = 0.05e-3  # 3D curve step (≤ 0.25 px in the image)
MIN_SINE = 0.15  # chord factor capped at 1 / MIN_SINE where a vessel runs along the ray
CURVE_MARGIN = 1.1  # integrate this much further, then cut at the branch length


class TwoViewSynthConfig(StrictModel):
    """Sampling of synthetic two-view cases (uniform over [low, high] unless noted). May be
    tuned; the physics comes from configs/physics.yaml."""

    name: str = "two_view_synthetic"
    seed: int = 5001
    n_cases: int = Field(100, ge=1)
    n_stress: int = Field(50, ge=0)  # report: random views, overlap allowed
    tree_config: str = "configs/synth_tree_v1.yaml"  # P2 sampler ...
    sampling: dict[str, Any] = Field(default_factory=dict)  # ... with these overrides
    n_points: int = Field(256, ge=16)
    # heart
    heart_radius_mm: Range = (40.0, 55.0)
    heart_offset_sd_mm: float = Field(15.0, ge=0)  # centre vs isocentre, SD per axis
    curvature_sd_per_mm: float = Field(0.03, ge=0)  # curvature within the surface
    curvature_corr_mm: float = Field(10.0, gt=0)
    wobble_sd_mm: float = Field(1.5, ge=0)  # rise and fall off the surface
    wobble_corr_mm: float = Field(15.0, gt=0)
    side_angle_deg: Range = (35.0, 75.0)
    # views
    primary_deg: Range = (-45.0, 60.0)
    secondary_deg: Range = (-35.0, 40.0)
    separation_deg: Range = (30.0, 110.0)
    sid_mm: Range = (900.0, 1200.0)
    sod_mm: Range = (720.0, 800.0)
    iso_pixel_mm: Range = (0.22, 0.30)
    image_size: int = Field(512, ge=64)
    margin_px: float = Field(16.0, ge=0)
    layout_tries: int = Field(100, ge=1)  # layouts and views per tree ...
    max_trees: int = Field(50, ge=1)  # ... and trees per case before giving up
    motion_sd_mm: float = Field(2.0, ge=0)  # heart shift between the runs, SD per axis
    avoid_overlap: bool = True  # views in which the main vessel overlaps nothing
    overlap_gap_px: float = Field(2.0, ge=0)  # lumens closer than this count as overlapping
    click_sd_px: float = Field(1.0, ge=0)  # start / end point placement, SD per axis
    # imaging (ARCADE-fitted, configs/validation_synthetic.yaml, D39)
    mu_per_mm: Range = (0.04, 0.08)
    background: Range = (0.49, 0.64)
    texture_sd: float = Field(0.04, ge=0)
    texture_scale_px: float = Field(40.0, gt=0)
    blur_px: Range = (0.8, 1.2)
    edge_amount: Range = (2.0, 3.0)
    edge_sigma_px: Range = (1.5, 3.0)
    noise_sd: Range = (0.012, 0.022)


@dataclass(frozen=True, eq=False)
class TwoViewCase:
    """One synthetic case and its ground truth.

    Attributes:
        seed: the case's seed.
        tree: the P2 tree (lumen profiles, pressures).
        curves: fine 3D centreline of each branch (m), ``FINE_STEP_M`` apart.
        curve_s: arc length of each curve (m).
        views: the two projections.
        motion_m: (3,) shift of the heart in the second run (motion scenario) (m).
        imaging: contrast, blur, edge enhancement and noise settings.
        centre_m: (3,) the heart sphere's centre (m); the branches lie on that sphere.
    """

    seed: int
    tree: SyntheticTree
    curves: tuple[np.ndarray, ...]
    curve_s: tuple[np.ndarray, ...]
    views: tuple[ProjectionGeometry, ProjectionGeometry]
    motion_m: np.ndarray
    imaging: dict[str, float]
    centre_m: np.ndarray | None = None

    @property
    def main_length(self) -> float:
        """True 3D length of the main vessel (m)."""
        return float(self.curve_s[0][-1])

    def radii(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Lumen and reference radius (m) of branch ``k`` along its fine curve."""
        b = self.tree.tree.branches[k]
        s = self.curve_s[k]
        return np.interp(s, b.s, b.r), np.interp(s, b.s, b.reference_radius())


def _smooth(rng, n: int, corr_steps: float, sd: float) -> np.ndarray:
    x = gaussian_filter1d(rng.standard_normal(n), max(corr_steps, 1e-9))
    return sd * x / max(float(x.std()), 1e-12)


def surface_curve(rng, centre, radius: float, p0, t0, length: float, cfg) -> np.ndarray:
    """A curve of ``length`` (m) on a sphere (``centre``, ``radius`` in m) from ``p0`` with
    initial direction ``t0``: smooth random curvature within the surface, smooth rise and fall
    off it. Returns (k, 3) points equally spaced in arc length, at most ``FINE_STEP_M`` apart,
    whose arc length is ``length`` (m)."""
    n = int(math.ceil(CURVE_MARGIN * length / FINE_STEP_M)) + 1
    kappa = _smooth(
        rng, n, mm_to_m(cfg.curvature_corr_mm) / FINE_STEP_M, cfg.curvature_sd_per_mm * 1e3
    )
    wobble = _smooth(rng, n, mm_to_m(cfg.wobble_corr_mm) / FINE_STEP_M, mm_to_m(cfg.wobble_sd_mm))
    wobble -= wobble[0]
    c = np.asarray(centre, float)
    p, t = np.asarray(p0, float), np.asarray(t0, float)
    out = np.empty((n, 3))
    out[0] = p
    for k in range(1, n):
        nrm = (p - c) / np.linalg.norm(p - c)
        t = t - (t @ nrm) * nrm
        t /= np.linalg.norm(t)
        phi = kappa[k] * FINE_STEP_M
        t = t * math.cos(phi) + np.cross(nrm, t) * math.sin(phi)
        q = p + FINE_STEP_M * t - c
        p = c + (radius + wobble[k]) * q / np.linalg.norm(q)
        out[k] = p
    arc = _arc(out)
    if arc[-1] < length:
        raise ValueError("surface curve shorter than requested")
    s = np.linspace(0.0, length, int(math.ceil(length / FINE_STEP_M)) + 1)
    return np.stack([np.interp(s, arc, out[:, k]) for k in range(3)], axis=1)


def _arc(points: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def _random_unit(rng) -> np.ndarray:
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def _tangent_at(points: np.ndarray, k: int) -> np.ndarray:
    a, b = points[max(k - 5, 0)], points[min(k + 5, len(points) - 1)]
    return (b - a) / np.linalg.norm(b - a)


def layout_tree(rng, tree: SyntheticTree, centre: np.ndarray, radius: float, cfg, start=None):
    """3D curves of every branch of ``tree`` on the sphere, the main vessel from ``start`` (a
    point on the sphere; random if ``None``). Returns (curves, arc lengths)."""
    curves, arcs = [], []
    for k, branch in enumerate(tree.tree.branches):
        length = float(branch.s[-1])
        if k == 0:
            p0 = centre + radius * _random_unit(rng) if start is None else np.asarray(start)
            nrm = (p0 - centre) / np.linalg.norm(p0 - centre)
            t0 = np.cross(nrm, _random_unit(rng))
        else:
            parent = tree.tree.parent[k]
            at = float(tree.tree.branches[parent].s[tree.tree.attach[k]])
            j = int(np.clip(np.searchsorted(arcs[parent], at), 0, len(arcs[parent]) - 1))
            p0 = curves[parent][j]
            nrm = (p0 - centre) / np.linalg.norm(p0 - centre)
            tp = _tangent_at(curves[parent], j)
            angle = math.radians(rng.uniform(*cfg.side_angle_deg)) * rng.choice((-1.0, 1.0))
            t0 = tp * math.cos(angle) + np.cross(nrm, tp) * math.sin(angle)
        pts = surface_curve(rng, centre, float(np.linalg.norm(p0 - centre)), p0, t0, length, cfg)
        arc = _arc(pts)
        curves.append(pts)
        arcs.append(arc)
    return tuple(curves), tuple(arcs)


def sample_views(rng, cfg: TwoViewSynthConfig) -> tuple[ProjectionGeometry, ProjectionGeometry]:
    """Two projections of one C-arm, ``separation_deg`` apart."""
    sod = mm_to_m(rng.uniform(*cfg.sod_mm))
    sids = [mm_to_m(rng.uniform(*cfg.sid_mm)) for _ in range(2)]
    pixel = mm_to_m(rng.uniform(*cfg.iso_pixel_mm)) * sids[0] / sod
    shape = (cfg.image_size, cfg.image_size)
    while True:
        angles = [
            (rng.uniform(*cfg.primary_deg), rng.uniform(*cfg.secondary_deg)) for _ in range(2)
        ]
        views = tuple(
            ProjectionGeometry(a, b, sid, sod, (pixel, pixel), shape)
            for (a, b), sid in zip(angles, sids, strict=True)
        )
        if cfg.separation_deg[0] <= views[0].separation_deg(views[1]) <= cfg.separation_deg[1]:
            return views


def main_overlaps(case_curves, radii, g: ProjectionGeometry, gap_px: float, shift=None) -> bool:
    """Whether the main vessel (curve 0) overlaps itself or another branch in view ``g``, with
    the heart shifted by ``shift`` (m). ``radii``: reference radius (m) along each curve (the
    drawn vessel's extent). Parts of a side branch within three main radii (3D) of the main
    vessel (its take-off) do not count; nor do main points less than three diameters apart
    along the vessel."""
    shift = np.zeros(3) if shift is None else np.asarray(shift, float)
    main = case_curves[0][::4] + shift
    p0 = g.project(main)
    r0 = radii[0][::4] / g.pixel_size_at(main)
    s0 = _arc(main)
    tree0 = cKDTree(p0)
    pairs = tree0.query_pairs(float(2.0 * r0.max() + gap_px), output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        apart = np.abs(s0[i] - s0[j]) > 6.0 * np.maximum(radii[0][::4][i], radii[0][::4][j])
        close = np.linalg.norm(p0[i] - p0[j], axis=1) < r0[i] + r0[j] + gap_px
        if np.any(apart & close):
            return True
    main3d = cKDTree(main)
    for curve, r in zip(case_curves[1:], radii[1:], strict=True):
        x = curve[::4] + shift
        far = main3d.query(x)[0] > 3.0 * float(radii[0].max())
        if not far.any():
            continue
        pk = g.project(x[far])
        rk = r[::4][far] / g.pixel_size_at(x[far])
        dist, idx = tree0.query(pk)
        if np.any(dist < r0[idx] + rk + gap_px):
            return True
    return False


def _inside(g: ProjectionGeometry, points: np.ndarray, margin: float) -> bool:
    p = g.project(points)
    return bool(np.all(p >= margin) and np.all(p <= np.asarray(g.shape) - 1 - margin))


def sample_case(seed: int, cfg: TwoViewSynthConfig) -> TwoViewCase:
    """One case: a tree laid on a heart-sized sphere and two views in which the whole main
    vessel is inside the image (and stays inside after the motion)."""
    rng = np.random.default_rng(seed)
    tree_cfg = load_config(cfg.tree_config, TreeSynthConfig)
    sampling = type(tree_cfg.sampling).model_validate(
        {**tree_cfg.sampling.model_dump(), **cfg.sampling}
    )
    for _ in range(cfg.max_trees):
        tree = sample_tree(rng, sampling, cfg.n_points)
        radius = mm_to_m(rng.uniform(*cfg.heart_radius_mm))
        for _ in range(cfg.layout_tries):
            centre = rng.normal(0.0, mm_to_m(cfg.heart_offset_sd_mm), 3)
            curves, arcs = layout_tree(rng, tree, centre, radius, cfg)
            views = sample_views(rng, cfg)
            motion = rng.normal(0.0, mm_to_m(cfg.motion_sd_mm), 3)
            main = curves[0]
            if not (
                _inside(views[0], main, cfg.margin_px)
                and _inside(views[1], main, cfg.margin_px)
                and _inside(views[1], main + motion, cfg.margin_px)
            ):
                continue
            radii = [
                np.interp(a, b.s, b.reference_radius())
                for a, b in zip(arcs, tree.tree.branches, strict=True)
            ]
            if not cfg.avoid_overlap or not any(
                main_overlaps(curves, radii, g, cfg.overlap_gap_px, shift)
                for g, shift in (
                    (views[0], np.zeros(3)),
                    (views[1], np.zeros(3)),
                    (views[1], motion),
                )
            ):
                imaging = {
                    name: float(rng.uniform(*getattr(cfg, name)))
                    for name in (
                        "mu_per_mm",
                        "background",
                        "blur_px",
                        "edge_amount",
                        "edge_sigma_px",
                        "noise_sd",
                    )
                }
                return TwoViewCase(seed, tree, curves, arcs, views, motion, imaging, centre)
    raise ValueError(
        f"seed {seed}: no layout fits the images in {cfg.max_trees} trees × {cfg.layout_tries} "
        "tries "
        "(image too small for the vessels, or overlap impossible to avoid)"
    )


@dataclass(frozen=True, eq=False)
class RenderedView:
    """One rendered view.

    Attributes:
        image: (H, W) uint8 angiogram-like image.
        prob: (H, W) float32 segmentation stand-in (unnarrowed reference tubes).
        geometry: the projection.
        start, end: the main vessel's start and end landmarks as clicked (row, col) (px).
    """

    image: np.ndarray
    prob: np.ndarray
    geometry: ProjectionGeometry
    start: tuple[float, float]
    end: tuple[float, float]


def render_view(
    case: TwoViewCase,
    k: int,
    cfg: TwoViewSynthConfig,
    *,
    moved: bool = False,
) -> RenderedView:
    """View ``k`` (0 or 1) of ``case``; ``moved``: the heart shifted by ``case.motion_m``."""
    g = case.views[k]
    rng = np.random.default_rng([case.seed, k, int(moved)])
    shift = case.motion_m if moved else np.zeros(3)
    tubes, refs = [], []
    for b, curve in enumerate(case.curves):
        x = curve + shift
        p = g.project(x)
        size = g.pixel_size_at(x)
        r, r_ref = case.radii(b)
        tangent = np.gradient(x, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        ray = x - g.source
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        sine = np.maximum(np.linalg.norm(np.cross(tangent, ray), axis=1), MIN_SINE)
        tubes.append((p, r / size, r / size / sine))
        refs.append((p, 2.0 * r_ref / size))
    n = cfg.image_size
    texture = gaussian_filter(rng.standard_normal((n, n)), cfg.texture_scale_px)
    texture *= cfg.texture_sd / max(float(texture.std()), 1e-12)
    im = case.imaging
    img = render_projection(
        (n, n),
        tubes,
        mm_per_px=g.iso_pixel_m * 1e3,
        mu_per_mm=im["mu_per_mm"],
        background=im["background"] + texture,
        blur_sigma=im["blur_px"],
        edge_amount=im["edge_amount"],
        edge_sigma=im["edge_sigma_px"],
        noise_std=im["noise_sd"],
        seed=int(rng.integers(2**31)),
    )
    prob = render_tubes((n, n), refs, blur_sigma=0.0, noise_std=0.0)
    ends = g.project(case.curves[0][[0, -1]] + shift)
    ends += rng.normal(0.0, cfg.click_sd_px, ends.shape)
    ends = np.clip(ends, 0, n - 1)
    return RenderedView(
        np.round(img * 255).astype(np.uint8),
        prob,
        g,
        (float(ends[0, 0]), float(ends[0, 1])),
        (float(ends[1, 0]), float(ends[1, 1])),
    )
