"""Model-based lumen measurement: the X-ray imaging chain fitted to the straightened vessel (P4c).

Edge-based QCA has a resolution floor. An edge is where the blurred intensity profile is
steepest, so a lumen much narrower than the blur still shows two edges a few pixels apart: on
rendered vessels of 8–16 px, intensity edges never read more than ~40–70% diameter stenosis,
however tight the lesion. This module instead fits a forward model of how a contrast-filled
lumen appears in the angiogram to the image around the centreline, and reads the lumen radius
from the fitted model.

Forward model, in the straightened frame (z: centreline samples, 1 px apart; x: along the
normal, ``X_STEP_PX`` apart):

1. chord through a circular lumen of radius r(z) centred at x0(z): 2·sqrt(r² − (x − x0)²),
   averaged over each x cell (closed form, so a sub-pixel lumen keeps its exact area);
2. transmission exp(−μ · chord) (Beer–Lambert; μ per px of chord);
3. system blur: Gaussian, σ_b px;
4. the X-ray system's edge enhancement: unsharp mask (1 + a)·Y − a·Gauss(Y, σ_e);
5. times the background, estimated from a band on each side beyond the fitted window.

The chain parameters (μ, σ_b, a, σ_e) are shared along the path, r and x0 are per centreline
point. Robust nonlinear least squares (soft-L1 loss, so an overlapping vessel or a catheter in
the window is down-weighted) with a curvature penalty on r and x0, initialised from the edge
diameters. The Jacobian is analytic and sparse: blur and edge enhancement are linear filters,
so the derivative with respect to r(z) is the filtered derivative of row z's transmission.
Because the model accounts for blur and edge enhancement, a tight lumen's contrast deficit, not
its blurred edges, sets its radius. Units: px.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, median_filter
from scipy.optimize import least_squares

from vstent.imaging.centerline import Centerline

X_STEP_PX = 1.0  # straightened-frame step along the normal
WINDOW_MARGIN_PX = 10.0  # fitted half-width = half the widest reference + this (blur, halo)
BAND_PX = 5.0  # background band on each side of the window
BACKGROUND_MEDIAN_POINTS = 9  # background smoothed along the centreline
SMOOTHNESS = 0.15  # curvature penalty on r and x0 (× 1 / noise)
LOSS_SCALE = 3.0  # soft-L1 scale in noise units: larger residuals are down-weighted
MAX_NFEV = 60
MIN_POINTS = 10
# Bounds of the chain parameters: μ (1/px), σ_b (px), a, σ_e (px).
CHAIN_LOWER = (1e-4, 0.3, 0.0, 0.5)
CHAIN_UPPER = (1.0, 3.0, 5.0, 5.0)
CHAIN_START = (None, 1.0, 1.0, 2.0)  # μ from the healthy contrast


@dataclass(frozen=True, eq=False)
class LumenFit:
    """Fitted lumen and imaging chain.

    Attributes:
        diameter_px: (n,) lumen diameter 2·r.
        offset_px: (n,) lumen centre along the normal, relative to the centreline.
        mu_per_px: attenuation per px of chord.
        blur_px: system blur σ_b.
        edge_amount: unsharp-mask amount a.
        edge_sigma_px: unsharp-mask σ_e.
        rms: root-mean-square residual in noise units (≈ 1 for a model that explains the image).
        converged: the optimiser reported convergence.
    """

    diameter_px: np.ndarray
    offset_px: np.ndarray
    mu_per_px: float
    blur_px: float
    edge_amount: float
    edge_sigma_px: float
    rms: float
    converged: bool


def straighten(
    image: np.ndarray, centerline: Centerline, half_width_px: float, step_px: float = X_STEP_PX
) -> tuple[np.ndarray, np.ndarray]:
    """Image resampled along the centreline's normals: (n, m) values and the m offsets (px)."""
    x = np.arange(-half_width_px, half_width_px + step_px / 2, step_px)
    q = centerline.points[:, None, :] + x[None, :, None] * centerline.normals[:, None, :]
    v = map_coordinates(image, [q[..., 0].ravel(), q[..., 1].ravel()], order=1, mode="nearest")
    return v.reshape(q.shape[:2]), x


def cell_chord(x: np.ndarray, x0: np.ndarray, r: np.ndarray, step_px: float = X_STEP_PX):
    """Mean chord (px) through circles (centre ``x0``, radius ``r``, both (n,)) over cells of
    width ``step_px`` centred at ``x`` (m,): exact, from the antiderivative of 2·sqrt(r² − u²)."""
    r = np.maximum(np.asarray(r, dtype=float), 1e-9)[:, None]

    def antiderivative(u):
        u = np.clip(u, -r, r)
        return u * np.sqrt(np.maximum(r * r - u * u, 0.0)) + r * r * np.arcsin(u / r)

    u = x[None, :] - np.asarray(x0, dtype=float)[:, None]
    return (antiderivative(u + step_px / 2) - antiderivative(u - step_px / 2)) / step_px


def chord_derivatives(x: np.ndarray, x0: np.ndarray, r: np.ndarray,
                      step_px: float = X_STEP_PX) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Derivatives of :func:`cell_chord` with respect to r and x0, each (n, m)."""
    r = np.maximum(np.asarray(r, dtype=float), 1e-9)[:, None]
    u = x[None, :] - np.asarray(x0, dtype=float)[:, None]
    hi, lo = u + step_px / 2, u - step_px / 2

    def chord_at(v):
        return 2.0 * np.sqrt(np.maximum(r * r - v * v, 0.0))

    d_r = 2.0 * r * (np.arcsin(np.clip(hi / r, -1, 1)) - np.arcsin(np.clip(lo / r, -1, 1)))
    return d_r / step_px, -(chord_at(hi) - chord_at(lo)) / step_px


def _filters(y: np.ndarray, blur: float, sigma: float, step_px: float):
    """Gauss(y, σ_b) and Gauss(Gauss(y, σ_b), σ_e) in the straightened frame."""
    gb = gaussian_filter(y, (blur, blur / step_px), mode="nearest")
    return gb, gaussian_filter(gb, (sigma, sigma / step_px), mode="nearest")


def forward(chain, r, x0, x, step_px: float = X_STEP_PX) -> np.ndarray:
    """Model image / background in the straightened frame, (n, m). ``chain`` = (μ, σ_b, a,
    σ_e); ``r``, ``x0`` (n,) px; ``x`` (m,) px."""
    mu, blur, amount, sigma = chain
    gb, geb = _filters(np.exp(-mu * cell_chord(x, x0, r, step_px)), blur, sigma, step_px)
    return (1.0 + amount) * gb - amount * geb


def jacobian(chain, r, x0, x, step_px: float = X_STEP_PX) -> sparse.csr_matrix:
    """Sparse Jacobian of :func:`forward` (rows: the (n, m) model raveled; columns: μ, σ_b, a,
    σ_e, r (n), x0 (n)). Exact for r, x0, μ and a; σ_b and σ_e by forward differences.

    The model is L(T) with L = (1 + a)·G_b − a·G_e·G_b linear and separable, so the column of
    r(z) is L applied to an image whose only non-zero row is z: the row responses of the
    Gaussians along z (filtering the identity, which keeps the boundary handling exact) times
    the filtered derivative row along x.
    """
    mu, blur, amount, sigma = chain
    n, m = len(r), len(x)
    chord = cell_chord(x, x0, r, step_px)
    t = np.exp(-mu * chord)
    gb, geb = _filters(t, blur, sigma, step_px)
    base = (1.0 + amount) * gb - amount * geb
    cols = []
    gb_mu, geb_mu = _filters(-chord * t, blur, sigma, step_px)
    cols.append((1.0 + amount) * gb_mu - amount * geb_mu)
    eps = 1e-4
    cols.append((forward((mu, blur + eps, amount, sigma), r, x0, x, step_px) - base) / eps)
    cols.append(gb - geb)
    cols.append((forward((mu, blur, amount, sigma + eps), r, x0, x, step_px) - base) / eps)
    chain_block = sparse.csr_matrix(np.stack([c.ravel() for c in cols], axis=1))

    row_b = gaussian_filter1d(np.eye(n), blur, axis=0, mode="nearest")
    row_eb = gaussian_filter1d(row_b, sigma, axis=0, mode="nearest")
    zp, zq = np.nonzero(np.abs(row_b) + np.abs(row_eb) > 1e-12)  # (z', z) pairs
    lanes = np.arange(m)
    blocks = []
    for deriv in chord_derivatives(x, x0, r, step_px):
        d = -mu * t * deriv
        a_b = gaussian_filter1d(d, blur / step_px, axis=1, mode="nearest")
        a_eb = gaussian_filter1d(a_b, sigma / step_px, axis=1, mode="nearest")
        vals = ((1.0 + amount) * row_b[zp, zq][:, None] * a_b[zq]
                - amount * row_eb[zp, zq][:, None] * a_eb[zq])  # fmt: skip
        rows = (zp[:, None] * m + lanes[None, :]).ravel()
        blocks.append(sparse.csr_matrix((vals.ravel(), (rows, np.repeat(zq, m))),
                                        shape=(n * m, n)))  # fmt: skip
    return sparse.hstack([chain_block, *blocks], format="csr")


def _second_difference(n: int) -> sparse.csr_matrix:
    """(n − 2, n) matrix of second differences."""
    k = max(n - 2, 0)
    return sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(k, n), format="csr")


@dataclass(frozen=True, eq=False)
class ChainCalibration:
    """Imaging-chain settings shared by every vessel of one frame (P4d).

    Attributes:
        blur_px: system blur σ_b.
        edge_amount: unsharp-mask amount a.
        edge_sigma_px: unsharp-mask σ_e.
        n_segments: vessel segments fitted jointly.
        rms: root-mean-square residual in noise units over all segments.
        converged: the optimiser reported convergence.
    """

    blur_px: float
    edge_amount: float
    edge_sigma_px: float
    n_segments: int
    rms: float
    converged: bool


@dataclass(frozen=True, eq=False)
class _Segment:
    """One vessel prepared for fitting: straightened, background-normalised image (n, m) at
    offsets ``x``, its noise and the start values."""

    obs: np.ndarray
    x: np.ndarray
    noise: float
    r0: np.ndarray
    x0_start: np.ndarray
    mu0: float
    r_upper: float
    x0_bound: float


def background_bands(
    image: np.ndarray, centerline: Centerline, reference_px: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Straightened image and its background: (values (n, m), offsets x (m,), background
    (n, m), fitted half-width). The background is the median of a ``BAND_PX`` band on each side
    beyond the fitted window (smoothed along the centreline), interpolated across it."""
    half_in = 0.5 * float(np.max(reference_px)) + WINDOW_MARGIN_PX
    values, x = straighten(image, centerline, half_in + BAND_PX)
    left, right = x < -half_in, x > half_in
    b_left = median_filter(np.median(values[:, left], axis=1), BACKGROUND_MEDIAN_POINTS,
                           mode="nearest")  # fmt: skip
    b_right = median_filter(np.median(values[:, right], axis=1), BACKGROUND_MEDIAN_POINTS,
                            mode="nearest")  # fmt: skip
    frac = np.clip((x + half_in) / (2.0 * half_in), 0.0, 1.0)
    background = b_left[:, None] + frac[None, :] * (b_right - b_left)[:, None]
    return values, x, background, half_in


def _as_unit(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    return img / 255.0 if img.max() > 1.0 else img


def _prepare(img, centerline, edge_diameter_px, edge_offset_px, reference_px) -> _Segment | None:
    d_edge = np.asarray(edge_diameter_px, dtype=float)
    n = d_edge.size
    if n < MIN_POINTS:
        return None
    ref = np.asarray(reference_px, dtype=float)
    values, x, background, half_in = background_bands(img, centerline, ref)
    observed = values / np.maximum(background, 1e-3)
    band = np.abs(x) > half_in
    obs = observed[:, ~band]
    noise = max(float(np.std(observed[:, band] - 1.0)), 1e-3)
    r0 = np.maximum(d_edge / 2.0, 0.5)
    healthy = (d_edge > 0.85 * ref) & (d_edge < 1.15 * ref)
    healthy = healthy if healthy.sum() >= 3 else np.ones(n, bool)
    depth = -np.log(np.clip(obs[healthy].min(axis=1), 0.05, 0.999))
    mu0 = float(np.clip(np.median(depth / (2.0 * r0[healthy])), CHAIN_LOWER[0], CHAIN_UPPER[0]))
    x0_bound = 0.5 * float(ref.max())
    x0_start = np.clip(np.nan_to_num(np.asarray(edge_offset_px, dtype=float)),
                       -x0_bound + 1e-6, x0_bound - 1e-6)  # fmt: skip
    return _Segment(obs, x[~band], noise, r0, x0_start, mu0, float(ref.max()) + 2.0, x0_bound)


def _solve(segments: list[_Segment], chain: ChainCalibration | None, max_nfev: int):
    """Robust joint fit (soft-L1, curvature penalty on r and x0). Parameters: (σ_b, a, σ_e)
    shared by all ``segments`` unless ``chain`` fixes them, then μ, r (n), x0 (n) per segment.
    Returns the solution and a function mapping it to per-segment (chain tuple, r, x0)."""
    free = chain is None
    n_shared = 3 if free else 0
    sizes = [s.r0.size for s in segments]
    starts = n_shared + np.concatenate([[0], np.cumsum([1 + 2 * n for n in sizes])[:-1]])
    lower = list(CHAIN_LOWER[1:]) if free else []
    upper = list(CHAIN_UPPER[1:]) if free else []
    p0 = list(CHAIN_START[1:]) if free else []
    for s, n in zip(segments, sizes, strict=True):
        lower += [CHAIN_LOWER[0], *np.zeros(n), *np.full(n, -s.x0_bound)]
        upper += [CHAIN_UPPER[0], *np.full(n, s.r_upper), *np.full(n, s.x0_bound)]
        p0 += [s.mu0, *s.r0, *s.x0_start]
    lower, upper = np.array(lower), np.array(upper)
    p0 = np.clip(np.array(p0, dtype=float), lower + 1e-9, upper - 1e-9)
    weights = [SMOOTHNESS / s.noise for s in segments]
    d2s = [_second_difference(n) for n in sizes]

    def unpack(p):
        shared = p[:3] if free else (chain.blur_px, chain.edge_amount, chain.edge_sigma_px)
        out = []
        for k, n in zip(starts, sizes, strict=True):
            out.append(((p[k], *shared), p[k + 1 : k + 1 + n], p[k + 1 + n : k + 1 + 2 * n]))
        return out

    def residuals(p):
        parts = []
        for s, (ch, r, x0), w, d2 in zip(segments, unpack(p), weights, d2s, strict=True):
            parts += [((forward(ch, r, x0, s.x) - s.obs) / s.noise).ravel(), w * (d2 @ r),
                      w * (d2 @ x0)]  # fmt: skip
        return np.concatenate(parts)

    def jac(p):
        rows, cols, vals, row0 = [], [], [], 0
        for s, (ch, r, x0), w, d2, k, n in zip(
            segments, unpack(p), weights, d2s, starts, sizes, strict=True
        ):
            block = (jacobian(ch, r, x0, s.x) / s.noise).tocoo()
            shared = [0, 1, 2] if free else [-1, -1, -1]
            col_map = np.array([k, *shared, *range(k + 1, k + 1 + 2 * n)])
            keep = col_map[block.col] >= 0
            rows.append(block.row[keep] + row0)
            cols.append(col_map[block.col][keep])
            vals.append(block.data[keep])
            row0 += block.shape[0]
            pen = (w * d2).tocoo()
            for first in (k + 1, k + 1 + n):  # r, then x0
                rows.append(pen.row + row0)
                cols.append(pen.col + first)
                vals.append(pen.data)
                row0 += d2.shape[0]
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(row0, p.size),
        )

    sol = least_squares(
        residuals, p0, jac=jac, bounds=(lower, upper), x_scale="jac", loss="soft_l1",
        f_scale=LOSS_SCALE, max_nfev=max_nfev,
    )  # fmt: skip
    return sol, unpack


def fit_lumen(
    image: np.ndarray,
    centerline: Centerline,
    edge_diameter_px: np.ndarray,
    edge_offset_px: np.ndarray,
    reference_px: np.ndarray,
    *,
    chain: ChainCalibration | None = None,
    max_nfev: int = MAX_NFEV,
) -> LumenFit | None:
    """Fit the imaging-chain model along ``centerline`` (see the module docstring).

    ``image``: greyscale angiogram, vessels dark (uint8 or float in [0, 1]).
    ``edge_diameter_px``, ``edge_offset_px``: (n,) edge-measured diameter and lumen-centre
    offset (start values); ``reference_px``: (n,) healthy reference diameter (healthy points set
    the start value of μ). ``chain``: the frame's calibrated blur and edge enhancement
    (:func:`calibrate_chain`), fixed during the fit; without it they are fitted to this vessel
    alone. Returns ``None`` for fewer than ``MIN_POINTS`` points.
    """
    seg = _prepare(_as_unit(image), centerline, edge_diameter_px, edge_offset_px, reference_px)
    if seg is None:
        return None
    sol, unpack = _solve([seg], chain, max_nfev)
    (mu, blur, amount, sigma), r, x0 = unpack(sol.x)[0]
    n, m = seg.obs.shape
    data = sol.fun[: n * m]
    return LumenFit(
        diameter_px=2.0 * np.asarray(r),
        offset_px=np.asarray(x0).copy(),
        mu_per_px=float(mu),
        blur_px=float(blur),
        edge_amount=float(amount),
        edge_sigma_px=float(sigma),
        rms=float(np.sqrt(np.mean(data**2))),
        converged=bool(sol.success),
    )


def calibrate_chain(
    image: np.ndarray,
    vessels: list[tuple[Centerline, np.ndarray, np.ndarray, np.ndarray]],
    *,
    max_nfev: int = MAX_NFEV,
) -> ChainCalibration | None:
    """Blur and edge enhancement of one frame, fitted jointly over its ``vessels`` (each:
    centreline, edge diameter, edge offset, reference diameter, as for :func:`fit_lumen`).

    One X-ray system and its image processing made every vessel of the frame, so σ_b, a and
    σ_e are shared, while μ (contrast concentration), r and x0 are per vessel. Fitted to one
    vessel, the chain trades off against the lumen; shared across vessels of different sizes,
    it is pinned down. Returns ``None`` if no vessel has ``MIN_POINTS`` points.
    """
    img = _as_unit(image)
    segments = [s for s in (_prepare(img, *v) for v in vessels) if s is not None]
    if not segments:
        return None
    sol, unpack = _solve(segments, None, max_nfev)
    data = np.concatenate([
        ((forward(ch, r, x0, s.x) - s.obs) / s.noise).ravel()
        for s, (ch, r, x0) in zip(segments, unpack(sol.x), strict=True)
    ])  # fmt: skip
    return ChainCalibration(
        blur_px=float(sol.x[0]),
        edge_amount=float(sol.x[1]),
        edge_sigma_px=float(sol.x[2]),
        n_segments=len(segments),
        rms=float(np.sqrt(np.mean(data**2))),
        converged=bool(sol.success),
    )
