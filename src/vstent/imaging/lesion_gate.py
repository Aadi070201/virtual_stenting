"""P4q: a narrowing counts only where the whole-image lesion detector agrees (docs/PLAN.md, P4q).
Torch-free.

The detector's probability map ≥ ``threshold`` gives the detections. A candidate narrowing
(:func:`~vstent.imaging.lesion_patch.candidates`: local deepest points with %DS ≥ 15%) is
**confirmed** when a detection lies within one reference diameter of its narrowed stretch (where
its %DS is ≥ half its deepest). An unconfirmed narrowing is set to its reference diameter (as
if healthy), for the marks and the physics. Applied after the learned reader, so the reader
cannot deepen what is left of a removed narrowing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from vstent.imaging.lesion_patch import Candidate, candidates, rescale, with_radii
from vstent.imaging.qca import QCAResult
from vstent.physics.geometry import VesselProfile


@dataclass(frozen=True)
class LesionGate:
    """``predict``: greyscale image (H, W) → lesion probability (H, W); detections are the
    pixels ≥ ``threshold``."""

    predict: Callable[[np.ndarray], np.ndarray]
    threshold: float


def confirmed(qca: QCAResult, detections: np.ndarray, cands: list[Candidate]) -> list[bool]:
    """Per candidate: a detection (``detections``, bool (H, W) in the image's pixels) lies within
    one reference diameter (image px) of a centreline point of its narrowed stretch."""
    if not detections.any():
        return [False] * len(cands)
    dist = ndimage.distance_transform_edt(~detections)  # px to the nearest detection
    hi = np.array(dist.shape) - 1
    out = []
    for c in cands:
        ij = np.clip(np.rint(qca.centerline.points[c.start : c.end + 1]).astype(int), 0, hi)
        out.append(bool(dist[ij[:, 0], ij[:, 1]].min() <= c.ref_px))
    return out


def apply_gate(
    qca: QCAResult, image: np.ndarray, gate: LesionGate
) -> tuple[QCAResult, list[tuple[Candidate, bool]]]:
    """The QCA with every unconfirmed candidate narrowing set to its reference diameter; and
    the (candidate, confirmed) pairs."""
    cands = candidates(qca)
    if not cands:
        return qca, []
    ok = confirmed(qca, np.asarray(gate.predict(image)) >= gate.threshold, cands)
    if all(ok):
        return qca, list(zip(cands, ok, strict=True))
    profile, r = qca.profile, qca.profile.r
    for c, keep in zip(cands, ok, strict=True):
        if not keep:
            r = rescale(VesselProfile(profile.s, r, profile.reference_radius()), c, 0.0, 0.0)
    return with_radii(qca, r, qca.lumen_method), list(zip(cands, ok, strict=True))
