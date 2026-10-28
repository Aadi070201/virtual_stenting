"""Virtual stent: widen the lumen of a ``VesselProfile`` over a segment.

Inside a stent from ``s_start`` to ``s_start + length`` the target radius is
``expansion · diameter / 2``. Over the first and last ``edge`` metres the stent effect is
blended in with a half-cosine (0 at the stent end, 1 after ``edge``), so the lumen changes
smoothly and a stent never acts outside its own length. The lumen is never reduced:
``r_new = r + w · max(0, r_stent − r)``. Several stents combine through the maximum.

All lengths in metres.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vstent.physics.geometry import VesselProfile

DEFAULT_EDGE = 1e-3


@dataclass(frozen=True)
class Stent:
    """One stent: start position, length and nominal diameter (m), expansion factor (−)."""

    s_start: float
    length: float
    diameter: float
    expansion: float = 1.0

    def __post_init__(self) -> None:
        if self.length <= 0 or self.diameter <= 0 or self.expansion <= 0:
            raise ValueError("stent length, diameter and expansion must be positive")

    @property
    def s_end(self) -> float:
        """Distal end (m)."""
        return self.s_start + self.length

    @property
    def radius(self) -> float:
        """Expanded radius (m)."""
        return self.expansion * self.diameter / 2.0


def stent_weight(s: np.ndarray, s_start, s_end, edge: float = DEFAULT_EDGE) -> np.ndarray:
    """Blending weight in [0, 1]: 0 outside, half-cosine ramps over ``edge`` inside each end.

    ``s`` (N,) positions; ``s_start`` / ``s_end`` scalars or (B,) arrays → result (N,) or (B, N).
    """
    s = np.asarray(s, dtype=float)
    s0 = np.asarray(s_start, dtype=float)[..., None]
    s1 = np.asarray(s_end, dtype=float)[..., None]
    d = np.minimum(s - s0, s1 - s)  # distance to the nearer stent end (negative outside)
    if edge > 0:
        e = np.minimum(edge, np.maximum((s1 - s0) / 2.0, 1e-12))  # at most half the stent
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(d / e, 0.0, 1.0)))
    else:
        ramp = np.ones_like(d)
    return np.where(d >= 0, ramp, 0.0)


def stented_radius(
    s: np.ndarray, r: np.ndarray, stents: Sequence[Stent], edge: float = DEFAULT_EDGE
) -> np.ndarray:
    """Radius (m) after implanting ``stents`` in a vessel with radius ``r`` at positions ``s``."""
    r = np.asarray(r, dtype=float)
    out = r.copy()
    for st in stents:
        w = stent_weight(s, st.s_start, st.s_end, edge)
        out = np.maximum(out, r + w * np.maximum(0.0, st.radius - r))
    return out


def check_inside(profile: VesselProfile, stents: Sequence[Stent]) -> None:
    """Raise ``ValueError`` if a stent extends beyond the vessel."""
    tol = 1e-9
    for st in stents:
        if st.s_start < profile.s[0] - tol or st.s_end > profile.s[-1] + tol:
            raise ValueError(
                f"stent {st.s_start * 1e3:.1f}–{st.s_end * 1e3:.1f} mm is outside the vessel "
                f"({profile.s[0] * 1e3:.1f}–{profile.s[-1] * 1e3:.1f} mm)"
            )


def apply_stents(
    profile: VesselProfile, stents: Sequence[Stent], *, edge: float = DEFAULT_EDGE
) -> VesselProfile:
    """Profile with ``stents`` implanted; the reference radius is kept (known or estimated)."""
    check_inside(profile, stents)
    r_new = stented_radius(profile.s, profile.r, stents, edge)
    return VesselProfile(profile.s, r_new, profile.reference_radius())


def apply_stent(
    profile: VesselProfile,
    s_start: float,
    length: float,
    diameter: float,
    *,
    expansion: float = 1.0,
    edge: float = DEFAULT_EDGE,
) -> VesselProfile:
    """One stent (positions, length and diameter in m). See :func:`apply_stents`."""
    return apply_stents(profile, [Stent(s_start, length, diameter, expansion)], edge=edge)
