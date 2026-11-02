"""Hybrid vessel segmentation (P4j): the model's vessels plus the vesselness-filter vessels that
connect to them. Torch-free.

The segmenter was trained on ARCADE, whose masks cover only the SYNTAX segments, so on other
hospitals' angiograms it misses much of the tree. A Frangi vesselness filter needs no training
and sees every tubular dark structure, but also rib and lung edges. Pre-registered in
docs/PLAN.md (P4j): components of {vesselness ≥ T} that touch the model's mask are added to it
with probability :data:`ADDED_PROB`; components that touch nothing the model saw are not. The
run's curves (opacification, end-diastole) count only confident pixels (≥ 0.9) and are
unchanged. Units: px.
"""

import numpy as np
from scipy import ndimage
from skimage.filters import frangi

SIGMAS_PX = (1.0, 2.0, 3.0, 4.0, 5.0)  # at 512 px, scaled with the image size
NORM_PERCENTILE = 99.5
SEED_LEVEL = 0.5
SEED_DILATE_PX = 2
MIN_COMPONENT_PX = 30
ADDED_PROB = 0.6
THRESHOLDS = (0.05, 0.10, 0.15, 0.20)  # the pre-registered candidates
HYBRID_T: float | None = None  # the product's threshold, set by the P4j rule (None: off)
_EIGHT = np.ones((3, 3), bool)


def vesselness(image: np.ndarray) -> np.ndarray:
    """Frangi vesselness of dark ridges, divided by its 99.5th percentile (float32, H × W)."""
    img = np.asarray(image, np.float64)
    img = img / 255.0 if img.max() > 1.0 else img
    scale = max(img.shape) / 512.0
    v = frangi(img, sigmas=[s * scale for s in SIGMAS_PX], black_ridges=True)
    p = float(np.percentile(v, NORM_PERCENTILE))
    return (v / p if p > 0 else v).astype(np.float32)


def added_mask(prob: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """Pixels added to the model's mask: components of {v ≥ t} ∪ model touching the model's
    mask (dilated by :data:`SEED_DILATE_PX`), minus the model's mask, components ≥
    :data:`MIN_COMPONENT_PX`."""
    seeds = prob >= SEED_LEVEL
    if not seeds.any():
        return np.zeros(seeds.shape, bool)
    labels, _ = ndimage.label((v >= t) | seeds, structure=_EIGHT)
    near = ndimage.binary_dilation(seeds, iterations=SEED_DILATE_PX)
    touching = np.unique(labels[near & (labels > 0)])
    add = np.isin(labels, touching) & ~seeds
    lab2, n = ndimage.label(add, structure=_EIGHT)
    if n:
        sizes = ndimage.sum(np.ones_like(lab2), lab2, index=np.arange(1, n + 1))
        add &= np.isin(lab2, np.flatnonzero(sizes >= MIN_COMPONENT_PX) + 1)
    return add


def hybrid_probability(image: np.ndarray, prob: np.ndarray, t: float,
                       v: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """The model's probability with the added vessels set to at least :data:`ADDED_PROB`."""
    add = added_mask(prob, vesselness(image) if v is None else v, t)
    out = np.asarray(prob, np.float32).copy()
    out[add] = np.maximum(out[add], ADDED_PROB)
    return out
