"""Two-view 3D reconstruction of a vessel centreline (P5). Torch-free.

Input: two angiograms of the same vessel from different angles, each with its projection
geometry (:mod:`vstent.imaging.projection`) and the vessel's 2D centreline between the same two
landmarks (the user's start and end points in each view).

1. **Correspondence.** Each centreline point defines a ray from its view's X-ray source; points
   that image the same 3D point have intersecting rays (the epipolar constraint). Pairing point
   i of view 1 with point j of view 2 costs the distance between their rays (m) plus a weak
   preference for equal fractional positions along the two centrelines; dynamic programming
   finds the monotone pairing from start to end with the least total cost (as in dynamic time
   warping).
2. **Triangulation.** Each pair gives the midpoint of its rays' closest approach. Where the
   vessel runs along the epipolar lines in either view, the pairing cannot fix the depth: such
   pairs are weighted by the sine of the angle between vessel and epipolar line.
3. **Smoothing.** A weighted least-squares cubic spline (one knot per ``KNOT_SPACING_M`` of
   vessel) through the points, resampled by 3D arc length: the length free of foreshortening.
4. **Check.** The 3D centreline is projected back into both views; its distance from the 2D
   centrelines (the reprojection residual) and the mean gap between paired rays show a
   geometry that does not fit (motion between the runs, wrong angles or distances).
   **Smoothing sensitivity** (P5b, D50, D51): where depth is poorly determined the result
   depends on the smoothing, so the curve is refitted with twice the knot spacing and the
   length change is reported (``reliable``: change ≤ ``MAX_SMOOTHING_CHANGE`` and residual
   ≤ ``MAX_FIT_RESIDUAL_PX``). It is information, not a warning: as a warning it flagged 18%
   of synthetic cases and missed the one large error.

**Motion between the runs.** The two runs are acquired one after the other, so the heart may
have moved (breathing, table). A shift of view 2 across its epipolar lines breaks the pairing;
:func:`estimate_shift` finds the shift that minimises the ray gaps, using the vessel itself as
the calibration object. A shift along the epipolar lines cannot be seen in two views; it moves
the reconstruction in depth with little change of shape.

**Diameters.** Each view's QCA diameter (px) is converted to metres with the pixel size at the
reconstructed 3D point (its exact magnification), and the two are combined as the
equivalent-circle diameter of an ellipse with those axes, sqrt(d1 · d2).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import make_lsq_spline
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from vstent.imaging.centerline import Centerline
from vstent.imaging.projection import ProjectionGeometry
from vstent.imaging.qca import (
    END_TRIM_DIAMETERS,
    MIN_POINTS_AFTER_TRIM,
    REF_UPPER,
    robust_reference,
)
from vstent.physics.geometry import VesselProfile

MIN_SEPARATION_DEG = 30.0  # views closer than this give poorly conditioned depth (warning)
SPEED_WEIGHT_M = 0.5e-3  # DP cost (m) of pairing points whose fractional positions differ by 1
KNOT_SPACING_M = 8e-3  # spline: one interior knot per this much vessel (D50)
MIN_WEIGHT = 0.05  # floor of the conditioning weight of a pair
WEAK_SINE = 0.26  # a pair is poorly conditioned below sin 15° between vessel and epipolar line
EPIPOLAR_STEP_M = 1e-3  # step along the baseline used to find the local epipolar direction
STEP_M = 0.1e-3  # 3D centreline resampling
SHIFT_RANGE_M = 10e-3  # motion search: ± this at the isocentre, across the epipolar lines
SHIFT_STEPS = 41
MAX_GAP_M = 0.5e-3  # check: mean ray gap above this → the two views do not fit (warning)
MAX_RESIDUAL_PX = 2.0  # check: median reprojection residual above this (warning)
# smoothing sensitivity (D50, set on development seeds; reported, not a warning: D51)
MAX_SMOOTHING_CHANGE = 0.025  # |length change| when the knot spacing is doubled
MAX_FIT_RESIDUAL_PX = 0.30  # median reprojection residual in either view


def ray_gaps(g1: ProjectionGeometry, p1, g2: ProjectionGeometry, p2) -> np.ndarray:
    """(n, m) distance (m) between the ray of each point (row, col) px of view 1 and of view 2
    at their closest approach."""
    d1, d2 = g1.rays(p1), g2.rays(p2)
    cross = np.cross(d1[:, None, :], d2[None, :, :])
    norm = np.linalg.norm(cross, axis=2)
    return np.abs(cross @ (g2.source - g1.source)) / np.maximum(norm, 1e-12)


def monotone_path(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-cost monotone path through ``cost`` (n, m) from (0, 0) to (n − 1, m − 1) with
    steps (1, 0), (0, 1), (1, 1) (dynamic time warping). Returns the (k, 2) index pairs and the
    total cost."""
    n, m = cost.shape
    acc = np.empty_like(cost, dtype=float)
    acc[0] = np.cumsum(cost[0])
    inf = np.array([np.inf])
    for i in range(1, n):
        prev = acc[i - 1]
        best = cost[i] + np.minimum(prev, np.concatenate([inf, prev[:-1]]))
        cum = np.cumsum(cost[i])
        # a run of (0, 1) steps along row i: acc[i, j] = min_k≤j best[k] + cost[i, k+1..j]
        acc[i] = cum + np.minimum.accumulate(best - cum)
    i, j, path = n - 1, m - 1, [(n - 1, m - 1)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin((acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])))
            i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
        path.append((i, j))
    return np.asarray(path[::-1]), float(acc[-1, -1])


def _pairing(g1, c1: Centerline, g2, c2: Centerline) -> tuple[np.ndarray, np.ndarray, float]:
    """DP pairing of the two centrelines: index pairs, their ray gaps (m), mean path cost (m)."""
    gaps = ray_gaps(g1, c1.points, g2, c2.points)
    f1, f2 = c1.s / max(c1.s[-1], 1e-9), c2.s / max(c2.s[-1], 1e-9)
    cost = gaps + SPEED_WEIGHT_M * np.abs(f1[:, None] - f2[None, :])
    path, total = monotone_path(cost)
    return path, gaps[path[:, 0], path[:, 1]], total / len(path)


def closest_points(o1, d1, o2, d2) -> tuple[np.ndarray, np.ndarray]:
    """Midpoints (k, 3) and gaps (k,) (m) of the closest approach of rays ``o + t·d`` (origins
    (3,), unit directions (k, 3))."""
    w0 = np.asarray(o1, float) - np.asarray(o2, float)
    b = np.sum(d1 * d2, axis=1)
    d, e = d1 @ w0, d2 @ w0
    den = np.maximum(1.0 - b**2, 1e-12)
    x1 = o1 + ((b * e - d) / den)[:, None] * d1
    x2 = o2 + ((e - b * d) / den)[:, None] * d2
    return 0.5 * (x1 + x2), np.linalg.norm(x1 - x2, axis=1)


def epipolar_sines(g1, t1: np.ndarray, g2, t2: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(k, 2) sine of the angle between the vessel ((k, 2) unit tangents (row, col) in each
    view) and the epipolar line through the image of each 3D point (k, 3) (m) in that view."""
    base = g2.source - g1.source
    step = EPIPOLAR_STEP_M * base / np.linalg.norm(base)
    out = []
    for g, t in ((g1, t1), (g2, t2)):
        e = g.project(points + step) - g.project(points)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        out.append(np.abs(e[:, 0] * t[:, 1] - e[:, 1] * t[:, 0]))
    return np.stack(out, axis=1)


def _across_epipolar(g1, g2, c2: Centerline) -> np.ndarray:
    """Unit image direction (row, col) across view 2's epipolar lines (mean over the vessel)."""
    near = g2.source + g2.rays(c2.points) * g2.sod_m
    base = g1.source - g2.source
    e = g2.project(near + EPIPOLAR_STEP_M * base / np.linalg.norm(base)) - g2.project(near)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    e *= np.sign(e @ e[0])[:, None]
    mean = e.mean(axis=0)
    mean /= np.linalg.norm(mean)
    return np.array([-mean[1], mean[0]])


def estimate_shift(g1, c1: Centerline, g2, c2: Centerline) -> tuple[np.ndarray, float]:
    """Shift (row, col) px of view 2's principal point, across its epipolar lines, that best
    pairs the two centrelines (least mean DP cost): motion of the heart between the runs.
    Searched over ± ``SHIFT_RANGE_M`` at the isocentre. Returns the shift and its mean cost (m)."""
    normal = _across_epipolar(g1, g2, c2)
    reach = SHIFT_RANGE_M / g2.iso_pixel_m

    def cost(t: float) -> float:
        return _pairing(g1, c1, g2.shifted(g2.shift_px + t * normal), c2)[2]

    grid = np.linspace(-reach, reach, SHIFT_STEPS)
    values = [cost(t) for t in grid]
    k = int(np.argmin(values))
    step = grid[1] - grid[0]
    res = minimize_scalar(
        cost, bounds=(grid[k] - step, grid[k] + step), method="bounded", options={"xatol": 0.05}
    )
    t, value = (res.x, res.fun) if res.fun < values[k] else (grid[k], values[k])
    return np.asarray(g2.shift_px) + t * normal, float(value)


@dataclass(frozen=True, eq=False)
class Reconstruction:
    """A vessel centreline in 3D from two views.

    Attributes:
        points: (n, 3) positions (m, patient coordinates), uniformly spaced in 3D arc length.
        s: (n,) 3D arc length from the start landmark (m).
        view_s: (n, 2) arc length (px) of the corresponding point on each view's 2D centreline.
        geometries: the two views (view 2 with the estimated shift, if any).
        gap_m: mean distance between paired rays (m).
        residual_px: (2,) median reprojection residual in each view (px).
        weak_fraction: share of pairs whose depth is poorly conditioned (vessel within 15° of
            the epipolar line in either view).
        shift_px: estimated shift of view 2 (row, col) (px).
        smoothing_change: relative length change when the curve is refitted with twice the
            knot spacing (sensitivity to the smoothing).
    """

    points: np.ndarray
    s: np.ndarray
    view_s: np.ndarray
    geometries: tuple[ProjectionGeometry, ProjectionGeometry]
    gap_m: float
    residual_px: np.ndarray
    weak_fraction: float
    shift_px: np.ndarray = field(default_factory=lambda: np.zeros(2))
    smoothing_change: float = 0.0

    @property
    def reliable(self) -> bool:
        """The reliability check (D50): the length barely depends on the smoothing and the 3D
        path follows both traced paths closely."""
        return (abs(self.smoothing_change) <= MAX_SMOOTHING_CHANGE
                and float(np.max(self.residual_px)) <= MAX_FIT_RESIDUAL_PX)  # fmt: skip

    @property
    def length(self) -> float:
        """3D length of the centreline (m)."""
        return float(self.s[-1])

    @property
    def separation_deg(self) -> float:
        return self.geometries[0].separation_deg(self.geometries[1])

    def warnings(self) -> list[str]:
        """Plain-language checks of the reconstruction."""
        out = []
        if self.separation_deg < MIN_SEPARATION_DEG:
            out.append(
                f"The two views are only {self.separation_deg:.0f}° apart (at least "
                f"{MIN_SEPARATION_DEG:.0f}° is recommended): depth is poorly determined"
            )
        if self.gap_m > MAX_GAP_M or np.max(self.residual_px) > MAX_RESIDUAL_PX:
            out.append(
                f"The two views do not fit one 3D vessel well (mean ray gap "
                f"{self.gap_m * 1e3:.2f} mm, residual {np.max(self.residual_px):.1f} px): check "
                "the start and end points, the projection angles, and motion between the runs"
            )
        if self.weak_fraction > 0.5:
            out.append(
                f"{self.weak_fraction:.0%} of the vessel runs along the epipolar lines: "
                "its 3D course there is interpolated"
            )
        return out


def reconstruct(
    g1: ProjectionGeometry,
    c1: Centerline,
    g2: ProjectionGeometry,
    c2: Centerline,
    *,
    correct_shift: bool = True,
) -> Reconstruction:
    """3D centreline from the 2D centrelines ``c1`` and ``c2`` (each from the start landmark to
    the end landmark, (row, col) px) of two views. ``correct_shift``: estimate and remove a
    shift of view 2 between the runs (:func:`estimate_shift`)."""
    if len(c1.s) < 4 or len(c2.s) < 4:
        raise ValueError("each view needs a centreline of at least 4 points")
    shift = np.asarray(g2.shift_px, float)
    if correct_shift:
        shift, _ = estimate_shift(g1, c1, g2, c2)
        g2 = g2.shifted(shift)
    path, gaps, _ = _pairing(g1, c1, g2, c2)
    i, j = path[:, 0], path[:, 1]
    mid, _ = closest_points(g1.source, g1.rays(c1.points[i]), g2.source, g2.rays(c2.points[j]))
    sines = epipolar_sines(g1, c1.tangents[i], g2, c2.tangents[j], mid).min(axis=1)
    weight = np.maximum(sines, MIN_WEIGHT)

    tau = c1.s[i] + c2.s[j]  # strictly increasing along the path
    est_len = max(c1.s[-1] * g1.iso_pixel_m, c2.s[-1] * g2.iso_pixel_m)
    spline, dense, arc = _fit_curve(tau, mid, weight, est_len, KNOT_SPACING_M)
    smoother = _fit_curve(tau, mid, weight, est_len, 2.0 * KNOT_SPACING_M)[2]
    s = np.linspace(0.0, arc[-1], max(2, int(round(arc[-1] / STEP_M)) + 1))
    t_s = np.interp(s, arc, dense)
    points = spline(t_s)
    view_s = np.stack([np.interp(t_s, tau, c1.s[i]), np.interp(t_s, tau, c2.s[j])], axis=1)

    residual = []
    for g, c in ((g1, c1), (g2, c2)):
        fine = np.linspace(0.0, c.s[-1], 4 * len(c.s))  # ≤ 0.25 px apart
        line = np.stack([np.interp(fine, c.s, c.points[:, k]) for k in range(2)], axis=1)
        dist, _ = cKDTree(line).query(g.project(points))
        residual.append(float(np.median(dist)))
    return Reconstruction(
        points,
        s,
        view_s,
        (g1, g2),
        float(np.mean(gaps)),
        np.asarray(residual),
        float(np.mean(sines < WEAK_SINE)),
        shift,
        float(smoother[-1] / arc[-1] - 1.0),
    )


def _fit_curve(tau, mid, weight, est_len: float, spacing: float):
    """Weighted least-squares cubic spline through the triangulated points, one interior knot
    per ``spacing`` (m) of the estimated length. Returns the spline, a dense parameter grid and
    the arc length (m) along it."""
    n_inner = max(0, int(round(est_len / spacing)) - 1)
    n_inner = min(n_inner, max(0, len(tau) // 4 - 4))
    inner = np.linspace(tau[0], tau[-1], n_inner + 2)[1:-1]
    knots = np.concatenate([[tau[0]] * 4, inner, [tau[-1]] * 4])
    spline = make_lsq_spline(tau, mid, knots, k=3, w=weight)
    dense = np.linspace(tau[0], tau[-1], max(20 * len(tau), 200))
    curve = spline(dense)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    return spline, dense, arc


@dataclass(frozen=True, eq=False)
class TwoViewProfile:
    """Lumen profile along the 3D centreline.

    Attributes:
        profile: SI profile for the physics (trimmed ends, capped like single-view QCA).
        offset_m: where ``profile.s = 0`` lies on the reconstruction's arc length (m).
        d_view_m: (n, 2) diameter seen in each view (m) at the profile's points.
        capped: (n,) points capped at REF_UPPER × the reference.
    """

    profile: VesselProfile
    offset_m: float
    d_view_m: np.ndarray
    capped: np.ndarray


def two_view_profile(
    recon: Reconstruction,
    diameters_px: tuple[np.ndarray, np.ndarray],
    s_px: tuple[np.ndarray, np.ndarray],
    *,
    trim_diameters: float = END_TRIM_DIAMETERS,
) -> TwoViewProfile:
    """Lumen profile from both views' QCA diameters (px; ``diameters_px[k]`` at arc lengths
    ``s_px[k]`` of view k's centreline, the one used for the reconstruction). Each diameter is
    scaled by the pixel size at its 3D point; the lumen is the ellipse-equivalent diameter
    sqrt(d1 · d2). ``trim_diameters`` × the median diameter is removed at both ends, as in
    single-view QCA."""
    d_view = np.stack(
        [
            np.interp(recon.view_s[:, k], s_px[k], diameters_px[k])
            * recon.geometries[k].pixel_size_at(recon.points)
            for k in range(2)
        ],
        axis=1,
    )
    d = np.sqrt(np.prod(np.maximum(d_view, 0.0), axis=1))
    trim = trim_diameters * float(np.median(d))
    keep = (recon.s >= trim) & (recon.s <= recon.s[-1] - trim)
    if keep.sum() < MIN_POINTS_AFTER_TRIM:
        keep = np.ones_like(keep)
    s, d, d_view = recon.s[keep], d[keep], d_view[keep]
    unit = recon.geometries[0].iso_pixel_m  # robust_reference works in pixel units
    d_ref = robust_reference(s / unit, d / unit) * unit
    capped = d > REF_UPPER * d_ref
    d_used = np.where(capped, REF_UPPER * d_ref, d)
    profile = VesselProfile(s=s - s[0], r=d_used / 2.0, r_ref=d_ref / 2.0)
    return TwoViewProfile(profile, float(s[0]), d_view, capped)
