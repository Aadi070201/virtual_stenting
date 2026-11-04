"""Lesion patches and learned lesion depth (P4i). Torch-free.

The one pre-processing function for training, ONNX inference and the product
(:func:`lesion_patch`), the candidate narrowings of a QCA path (:func:`candidates`) and the
rescaling of a candidate's depth to a predicted %DS (:func:`apply_learned`), as pre-registered in
docs/PLAN.md (P4i). A patch is straightened along the centreline and scaled to the vessel's
reference diameter, so image size and magnification do not matter. Units: image px; %DS in
percent.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import map_coordinates

from vstent.imaging.centerline import Centerline
from vstent.imaging.qca import MIN_DIAMETER_PX, DiameterProfile, QCAResult
from vstent.physics.geometry import VesselProfile

PATCH_ALONG = 64
PATCH_ACROSS = 32
ALONG_REF = 6.0  # ± this × the reference diameter along the path
ACROSS_REF = 2.0  # ± this × the reference diameter across it
MIN_REF_PX = 4.0
BG_COLUMNS = 4
N_SCALARS = 3
CANDIDATE_MIN_DS = 15.0
CANDIDATE_SPACING_REF = 2.0
MAX_DEPTH_SCALE = 4.0
GRADE_TARGETS = {"p0_20": 10.0, "p20_50": 35.0, "p50_70": 60.0, "p70_90": 80.0, "p90_98": 94.0}


@dataclass(frozen=True)
class Candidate:
    """A narrowing of a QCA path: its deepest point ``k``, reference diameter (image px),
    measured %DS, extent [start, end] (indices; where %DS ≥ half the deepest) and length (px)."""

    k: int
    ref_px: float
    ds_pct: float
    start: int
    end: int
    length_px: float


def path_ds(qca: QCAResult) -> np.ndarray:
    """(n,) %DS along the path (the physics profile's radius against its reference)."""
    p = qca.profile
    return 100.0 * (1.0 - p.r / p.reference_radius())


def reference_px(qca: QCAResult) -> np.ndarray:
    """(n,) reference diameter in image px (without the absolute-size factor)."""
    return qca.reference_diameter_mm / (qca.mm_per_px * qca.size_factor)


def extent(ds: np.ndarray, k: int) -> tuple[int, int]:
    """Indices around ``k`` where %DS ≥ half of %DS at ``k``."""
    half = ds[k] / 2.0
    a = k
    while a > 0 and ds[a - 1] >= half:
        a -= 1
    b = k
    while b < len(ds) - 1 and ds[b + 1] >= half:
        b += 1
    return a, b


def candidate_at(qca: QCAResult, k: int) -> Candidate:
    """The candidate whose deepest point is ``k``."""
    ds, ref, s = path_ds(qca), reference_px(qca), qca.centerline.s
    a, b = extent(ds, k)
    return Candidate(k, float(max(ref[k], MIN_REF_PX)), float(ds[k]), a, b, float(s[b] - s[a]))


def candidates(qca: QCAResult) -> list[Candidate]:
    """Local %DS maxima ≥ :data:`CANDIDATE_MIN_DS`, deepest first, at least
    :data:`CANDIDATE_SPACING_REF` reference diameters apart."""
    ds, s = path_ds(qca), qca.centerline.s
    if ds.size < 3:
        return []
    peak = np.zeros(ds.size, bool)
    peak[1:-1] = (ds[1:-1] >= ds[:-2]) & (ds[1:-1] >= ds[2:])
    peak &= ds >= CANDIDATE_MIN_DS
    chosen: list[Candidate] = []
    for k in sorted(np.flatnonzero(peak), key=lambda i: -ds[i]):
        c = candidate_at(qca, int(k))
        if all(abs(s[k] - s[o.k]) >= CANDIDATE_SPACING_REF * max(c.ref_px, o.ref_px)
               for o in chosen):  # fmt: skip
            chosen.append(c)
    return chosen


def _along(centerline: Centerline, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points and unit normals at arc lengths ``q`` (px), extrapolated along the end tangents."""
    s, pts, tan = centerline.s, centerline.points, centerline.tangents
    out = np.stack([np.interp(q, s, pts[:, i]) for i in range(2)], axis=1)
    lo, hi = q < s[0], q > s[-1]
    out[lo] = pts[0] + (q[lo] - s[0])[:, None] * tan[0]
    out[hi] = pts[-1] + (q[hi] - s[-1])[:, None] * tan[-1]
    t = np.stack([np.interp(q, s, tan[:, i]) for i in range(2)], axis=1)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return out, np.stack([t[:, 1], -t[:, 0]], axis=1)


def lesion_patch(
    image: np.ndarray, prob: np.ndarray, centerline: Centerline, c: Candidate
) -> tuple[np.ndarray, np.ndarray]:
    """(2, :data:`PATCH_ALONG`, :data:`PATCH_ACROSS`) float32 patch at candidate ``c``: log
    contrast against the background (median of the outer :data:`BG_COLUMNS` columns on each
    side) and the segmentation probability; and the (:data:`N_SCALARS`,) scalars: measured
    %DS ÷ 100, log reference diameter (px), lesion length ÷ reference diameter."""
    d = max(c.ref_px, MIN_REF_PX)
    a = np.linspace(-ALONG_REF * d, ALONG_REF * d, PATCH_ALONG)
    x = np.linspace(-ACROSS_REF * d, ACROSS_REF * d, PATCH_ACROSS)
    pts, nrm = _along(centerline, centerline.s[c.k] + a)
    rows = pts[:, :1] + x[None, :] * nrm[:, :1]
    cols = pts[:, 1:] + x[None, :] * nrm[:, 1:]
    img = map_coordinates(np.asarray(image, np.float64), [rows, cols], order=1, mode="nearest")
    pr = map_coordinates(np.asarray(prob, np.float64), [rows, cols], order=1, mode="nearest")
    bg = float(np.median(np.concatenate([img[:, :BG_COLUMNS], img[:, -BG_COLUMNS:]], axis=1)))
    contrast = np.log(bg + 1.0) - np.log(img + 1.0)
    patch = np.stack([contrast, pr]).astype(np.float32)
    scalars = np.array([c.ds_pct / 100.0, np.log(d), c.length_px / d], np.float32)
    return patch, scalars


def rescale(profile: VesselProfile, c: Candidate, ds_pred: float, floor_m: float) -> np.ndarray:
    """Radii (m) with candidate ``c``'s depth rescaled to ``ds_pred`` over its extent:
    r' = ref − k (ref − r), k = predicted ÷ measured %DS (≤ :data:`MAX_DEPTH_SCALE`), r' ≥
    ``floor_m``."""
    r, ref = profile.r.copy(), profile.reference_radius()
    k = float(np.clip(ds_pred / max(c.ds_pct, 1e-6), 0.0, MAX_DEPTH_SCALE))
    sel = slice(c.start, c.end + 1)
    r[sel] = np.maximum(ref[sel] - k * (ref[sel] - profile.r[sel]), floor_m)
    return r


def apply_learned(
    qca: QCAResult,
    image: np.ndarray,
    prob: np.ndarray,
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    min_ref_mm: float | None = None,
) -> tuple[QCAResult, list[tuple[Candidate, float]]]:
    """The QCA with every candidate's depth set to its predicted %DS (``predict``: patches
    (n, 2, 64, 32), scalars (n, 3) → %DS (n,)); and the (candidate, prediction) pairs. With
    ``min_ref_mm`` (P4n), candidates whose reference diameter is smaller keep the edges'
    depth."""
    cands = candidates(qca)
    if not cands:
        return qca, []
    batch = [lesion_patch(image, prob, qca.centerline, c) for c in cands]
    preds = np.asarray(predict(np.stack([b[0] for b in batch]), np.stack([b[1] for b in batch])),
                       float).reshape(-1)  # fmt: skip
    profile = qca.profile
    m_per_px = qca.mm_per_px * qca.size_factor * 1e-3
    r = profile.r
    mm_per_px = qca.mm_per_px * qca.size_factor
    for c, p in zip(cands, preds, strict=True):
        if min_ref_mm is not None and c.ref_px * mm_per_px < min_ref_mm:
            continue  # P4n: a thin vessel keeps the edges' depth
        r = rescale(VesselProfile(profile.s, r, profile.reference_radius()), c, float(p),
                    MIN_DIAMETER_PX * m_per_px / 2.0)  # fmt: skip
    return with_radii(qca, r, "learned"), list(zip(cands, preds.tolist(), strict=True))


def with_radii(qca: QCAResult, r: np.ndarray, lumen_method: str) -> QCAResult:
    """``qca`` with the physics profile's radii ``r`` (m) and the image diameters to match (the
    reference unchanged; each edge pair moved symmetrically about its midpoint)."""
    profile = qca.profile
    m_per_px = qca.mm_per_px * qca.size_factor * 1e-3
    new_profile = VesselProfile(profile.s, r, profile.reference_radius())
    d_px = 2.0 * r / m_per_px
    old = qca.diameters
    found = np.isfinite(old.left_px) & np.isfinite(old.right_px)
    mid = np.where(found, (old.left_px + old.right_px) / 2.0, 0.0)
    diam = DiameterProfile(
        diameter_px=d_px,
        left_px=np.where(np.isfinite(old.left_px), mid - d_px / 2.0, np.nan),
        right_px=np.where(np.isfinite(old.right_px), mid + d_px / 2.0, np.nan),
        fallback=old.fallback,
        branch=old.branch,
    )
    return replace(qca, diameters=diam, profile=new_profile, lumen_method=lumen_method)
