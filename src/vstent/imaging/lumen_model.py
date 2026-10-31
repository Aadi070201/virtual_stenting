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


def fit_lumen(
    image: np.ndarray,
    centerline: Centerline,
    edge_diameter_px: np.ndarray,
    edge_offset_px: np.ndarray,
    reference_px: np.ndarray,
    *,
    max_nfev: int = MAX_NFEV,
) -> LumenFit | None:
    """Fit the imaging-chain model along ``centerline`` (see the module docstring).

    ``image``: greyscale angiogram, vessels dark (uint8 or float in [0, 1]).
    ``edge_diameter_px``, ``edge_offset_px``: (n,) edge-measured diameter and lumen-centre
    offset (start values); ``reference_px``: (n,) healthy reference diameter (healthy points set
    the start value of μ). Returns ``None`` for fewer than ``MIN_POINTS`` points.
    """
    d_edge = np.asarray(edge_diameter_px, dtype=float)
    n = d_edge.size
    if n < MIN_POINTS:
        return None
    ref = np.asarray(reference_px, dtype=float)
    img = np.asarray(image, dtype=np.float32)
    img = img / 255.0 if img.max() > 1.0 else img
    half_in = 0.5 * float(ref.max()) + WINDOW_MARGIN_PX
    values, x = straighten(img, centerline, half_in + BAND_PX)
    left, right = x < -half_in, x > half_in
    b_left = median_filter(np.median(values[:, left], axis=1), BACKGROUND_MEDIAN_POINTS,
                           mode="nearest")  # fmt: skip
    b_right = median_filter(np.median(values[:, right], axis=1), BACKGROUND_MEDIAN_POINTS,
                            mode="nearest")  # fmt: skip
    frac = np.clip((x + half_in) / (2.0 * half_in), 0.0, 1.0)
    background = b_left[:, None] + frac[None, :] * (b_right - b_left)[:, None]
    observed = values / np.maximum(background, 1e-3)
    inner = ~(left | right)
    xi, obs = x[inner], observed[:, inner]
    noise = max(float(np.std(observed[:, left | right] - 1.0)), 1e-3)
    m = xi.size

    r0 = np.maximum(d_edge / 2.0, 0.5)
    healthy = (d_edge > 0.85 * ref) & (d_edge < 1.15 * ref)
    healthy = healthy if healthy.sum() >= 3 else np.ones(n, bool)
    depth = -np.log(np.clip(obs[healthy].min(axis=1), 0.05, 0.999))
    mu0 = float(np.clip(np.median(depth / (2.0 * r0[healthy])), CHAIN_LOWER[0], CHAIN_UPPER[0]))
    chain0 = np.array([mu0, *CHAIN_START[1:]], dtype=float)
    lower = np.concatenate([CHAIN_LOWER, np.zeros(n), np.full(n, -0.5 * ref.max())])
    upper = np.concatenate([CHAIN_UPPER, np.full(n, ref.max() + 2.0), np.full(n, 0.5 * ref.max())])
    x0_start = np.clip(np.nan_to_num(np.asarray(edge_offset_px, dtype=float)),
                       lower[4 + n :] + 1e-6, upper[4 + n :] - 1e-6)  # fmt: skip
    p0 = np.clip(np.concatenate([chain0, r0, x0_start]), lower + 1e-9, upper - 1e-9)
    weight = SMOOTHNESS / noise
    d2 = _second_difference(n)
    zero = sparse.csr_matrix((d2.shape[0], n))
    penalty = sparse.vstack([
        sparse.hstack([sparse.csr_matrix((d2.shape[0], 4)), weight * d2, zero]),
        sparse.hstack([sparse.csr_matrix((d2.shape[0], 4)), zero, weight * d2]),
    ], format="csr")  # fmt: skip

    def residuals(p):
        r, x0 = p[4 : 4 + n], p[4 + n :]
        model = forward(p[:4], r, x0, xi)
        return np.concatenate([((model - obs) / noise).ravel(), weight * (d2 @ r),
                               weight * (d2 @ x0)])  # fmt: skip

    def jac(p):
        data = jacobian(p[:4], p[4 : 4 + n], p[4 + n :], xi) / noise
        return sparse.vstack([data, penalty], format="csr")

    sol = least_squares(
        residuals, p0, jac=jac, bounds=(lower, upper), x_scale="jac", loss="soft_l1",
        f_scale=LOSS_SCALE, max_nfev=max_nfev,
    )  # fmt: skip
    data = sol.fun[: n * m]
    return LumenFit(
        diameter_px=2.0 * sol.x[4 : 4 + n],
        offset_px=sol.x[4 + n :].copy(),
        mu_per_px=float(sol.x[0]),
        blur_px=float(sol.x[1]),
        edge_amount=float(sol.x[2]),
        edge_sigma_px=float(sol.x[3]),
        rms=float(np.sqrt(np.mean(data**2))),
        converged=bool(sol.success),
    )
