"""Vessel geometry: centreline radius profiles and stenosis metrics.

Lengths are in m, areas in m². Percent diameter stenosis (DS) is in percent.
"""

from dataclasses import dataclass, replace

import numpy as np

LESION_THRESHOLD_PCT = 30.0
"""A lesion is a contiguous run of points with DS above this value (%)."""


def _readonly_copy(x) -> np.ndarray:
    arr = np.array(x, dtype=float)
    arr.setflags(write=False)
    return arr


def _check_radius(r: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if r.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {r.shape}")
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, eq=False)
class VesselProfile:
    """Lumen radius sampled along a vessel centreline.

    Attributes:
        s: arc-length positions (m), 1-D, strictly increasing, uniformly spaced.
        r: lumen radius at each position (m), positive.
        r_ref: known healthy reference radius (m), e.g. for synthetic vessels. ``None`` means
            it is estimated from ``r`` with :func:`estimate_reference_radius`.

    Arrays are copied and made read-only.
    """

    s: np.ndarray
    r: np.ndarray
    r_ref: np.ndarray | None = None

    def __post_init__(self) -> None:
        s = _readonly_copy(self.s)
        if s.ndim != 1 or s.size < 2:
            raise ValueError("s must be 1-D with at least 2 points")
        if not np.all(np.isfinite(s)):
            raise ValueError("s must be finite")
        steps = np.diff(s)
        if np.any(steps <= 0):
            raise ValueError("s must be strictly increasing")
        if not np.allclose(steps, steps.mean(), rtol=1e-6, atol=0.0):
            raise ValueError("s must be uniformly spaced (resample first)")
        r = _readonly_copy(self.r)
        _check_radius(r, "r", s.shape)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "r", r)
        if self.r_ref is not None:
            r_ref = _readonly_copy(self.r_ref)
            _check_radius(r_ref, "r_ref", s.shape)
            object.__setattr__(self, "r_ref", r_ref)

    @property
    def n(self) -> int:
        """Number of points."""
        return self.s.size

    @property
    def length(self) -> float:
        """Centreline length (m)."""
        return float(self.s[-1] - self.s[0])

    @property
    def ds(self) -> float:
        """Uniform point spacing (m)."""
        return self.length / (self.n - 1)

    @property
    def area(self) -> np.ndarray:
        """Lumen area πr² (m²)."""
        return np.pi * self.r**2

    @property
    def diameter(self) -> np.ndarray:
        """Lumen diameter 2r (m)."""
        return 2.0 * self.r

    def resample(self, n: int) -> "VesselProfile":
        """Linearly resample onto ``n`` uniformly spaced points over the same span."""
        if n < 2:
            raise ValueError("n must be at least 2")
        s_new = np.linspace(self.s[0], self.s[-1], n)
        r_ref = None if self.r_ref is None else np.interp(s_new, self.s, self.r_ref)
        return VesselProfile(s_new, np.interp(s_new, self.s, self.r), r_ref)

    def with_radius(self, r: np.ndarray) -> "VesselProfile":
        """Same positions and reference, new radius (m)."""
        return replace(self, r=r)

    def reference_radius(self) -> np.ndarray:
        """Known reference radius if set, otherwise the robust estimate (m)."""
        if self.r_ref is not None:
            return self.r_ref
        return estimate_reference_radius(self.s, self.r)

    def percent_stenosis(self) -> np.ndarray:
        """Percent diameter stenosis at each point (%)."""
        return percent_diameter_stenosis(self.r, self.reference_radius())

    def lesions(self, threshold_pct: float = LESION_THRESHOLD_PCT) -> list["Lesion"]:
        """Lesions (contiguous runs with DS > ``threshold_pct``)."""
        return find_lesions(self.s, self.r, self.reference_radius(), threshold_pct)

    def mld(self) -> tuple[float, int]:
        """Minimal lumen diameter (m) and its index."""
        return minimal_lumen_diameter(self.r)


@dataclass(frozen=True)
class Lesion:
    """A contiguous segment with diameter stenosis above the threshold.

    Indices are inclusive. Positions and diameters are in m, DS in %.
    """

    start_idx: int
    end_idx: int
    s_start: float
    s_end: float
    mld: float
    mld_idx: int
    s_mld: float
    max_ds_pct: float

    @property
    def length(self) -> float:
        """Distance from the first to the last point above the threshold (m)."""
        return self.s_end - self.s_start


def estimate_reference_radius(
    s: np.ndarray, r: np.ndarray, *, threshold: float = 0.85, max_iter: int = 5
) -> np.ndarray:
    """Robust linear fit of the healthy segments (m).

    Fit r(s) linearly, exclude points below ``threshold`` × fit, refit; repeat at most
    ``max_iter`` times or until the set of included points stops changing.

    Args:
        s: positions (m).
        r: radii (m).
        threshold: points with r < threshold · fit are treated as diseased.
        max_iter: maximum number of exclude-and-refit iterations.
    """
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    included = np.ones(r.shape, dtype=bool)
    coef = np.polyfit(s, r, 1)
    for _ in range(max_iter):
        new_included = r >= threshold * np.polyval(coef, s)
        if new_included.sum() < 2 or np.array_equal(new_included, included):
            break
        included = new_included
        coef = np.polyfit(s[included], r[included], 1)
    return np.polyval(coef, s)


def percent_diameter_stenosis(r: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    """DS = 100 · (1 − r / r_ref) (%). Negative where the lumen exceeds the reference."""
    return 100.0 * (1.0 - np.asarray(r, dtype=float) / np.asarray(r_ref, dtype=float))


def find_lesions(
    s: np.ndarray, r: np.ndarray, r_ref: np.ndarray, threshold_pct: float = LESION_THRESHOLD_PCT
) -> list[Lesion]:
    """Contiguous runs of points with DS > ``threshold_pct``, ordered along s.

    Args:
        s: positions (m).
        r: radii (m).
        r_ref: reference radii (m).
        threshold_pct: DS threshold (%).
    """
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    ds_pct = percent_diameter_stenosis(r, r_ref)
    above = np.concatenate(([0], (ds_pct > threshold_pct).astype(np.int8), [0]))
    edges = np.diff(above)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    lesions = []
    for i0, i1 in zip(starts, ends, strict=True):
        j = int(i0 + np.argmin(r[i0 : i1 + 1]))
        lesions.append(
            Lesion(
                start_idx=int(i0),
                end_idx=int(i1),
                s_start=float(s[i0]),
                s_end=float(s[i1]),
                mld=float(2.0 * r[j]),
                mld_idx=j,
                s_mld=float(s[j]),
                max_ds_pct=float(ds_pct[i0 : i1 + 1].max()),
            )
        )
    return lesions


def minimal_lumen_diameter(r: np.ndarray) -> tuple[float, int]:
    """Minimal lumen diameter 2·min(r) (m) and its index."""
    r = np.asarray(r, dtype=float)
    j = int(np.argmin(r))
    return float(2.0 * r[j]), j


def cosine_narrowing(s: np.ndarray, center: float, length: float, severity: float) -> np.ndarray:
    """Multiplicative radius factor of a cosine lesion (dimensionless).

    ``1 − severity · ½(1 + cos(2π (s − center) / length))`` inside the lesion, 1 outside,
    so the throat has DS = 100 · severity %.

    Args:
        s: positions (m).
        center: lesion centre (m).
        length: lesion length (m).
        severity: fractional diameter reduction at the throat, in [0, 1).
    """
    s = np.asarray(s, dtype=float)
    x = (s - center) / length
    bump = np.where(np.abs(x) <= 0.5, 0.5 * (1.0 + np.cos(2.0 * np.pi * x)), 0.0)
    return 1.0 - severity * bump


def cosine_lesion_profile(
    length: float,
    ref_diameter: float,
    lesion_length: float,
    severity: float,
    *,
    center: float | None = None,
    n: int = 256,
) -> VesselProfile:
    """Straight vessel of constant reference diameter with one cosine lesion.

    Args:
        length: vessel length (m).
        ref_diameter: reference diameter (m).
        lesion_length: lesion length (m).
        severity: fractional diameter stenosis at the throat, in [0, 1).
        center: lesion centre (m); defaults to the middle of the vessel.
        n: number of points.

    The true reference radius is attached as ``r_ref``.
    """
    s = np.linspace(0.0, length, n)
    r_ref = np.full(n, ref_diameter / 2.0)
    c = length / 2.0 if center is None else center
    return VesselProfile(s, r_ref * cosine_narrowing(s, c, lesion_length, severity), r_ref)
