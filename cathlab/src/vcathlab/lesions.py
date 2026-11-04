"""Lesions in the virtual patient's arteries.

A lesion narrows a stretch of ``length_mm`` with a smooth (cosine) profile, deepest at its
centre. Its severity ``ds_pct`` is the diameter stenosis of the lumen's area-equivalent
diameter: the lumen's area at the throat is π (r_ref (1 − DS))². The lumen can be an ellipse
(``eccentricity`` = minor / major axis at the throat, 1 = round) whose major axis keeps a fixed
direction around the vessel (``orientation_deg`` in a twist-free frame), so its apparent
severity depends on the view, as for eccentric plaque. Several lesions on one vessel multiply.
Units: SI (m) unless named.
"""

import math
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from vcathlab.anatomy import Patient, Vessel
from vcathlab.config import StrictModel


class LesionSpec(StrictModel):
    """One lesion: where (a SYNTAX ``segment`` and the ``position`` within it, 0 = its start)
    and its shape."""

    segment: str
    position: float = Field(0.5, ge=0, le=1)
    ds_pct: float = Field(ge=0, lt=100)  # area-equivalent diameter stenosis at the throat
    length_mm: float = Field(10.0, gt=0)
    eccentricity: float = Field(1.0, gt=0, le=1)  # minor / major axis at the throat
    orientation_deg: float = 0.0  # the major axis, around the vessel


@dataclass(frozen=True)
class Lumen:
    """A vessel's lumen: half-axes ``ra`` ≥ ``rb`` (n,) m along the unit axes ``ea``, ``eb``
    (n, 3), both perpendicular to the unit ``tangent`` (n, 3); ``ref`` (n,) the reference
    radius (m)."""

    ra: np.ndarray
    rb: np.ndarray
    ea: np.ndarray
    eb: np.ndarray
    tangent: np.ndarray
    ref: np.ndarray


@dataclass(frozen=True)
class LesionTruth:
    """A placed lesion: its spec, vessel, point indices (``start``, ``centre``, ``end``), the
    reference and minimal (area-equivalent) lumen diameters at the throat (mm), and its length
    along the vessel (mm)."""

    spec: LesionSpec
    vessel: str
    start: int
    centre: int
    end: int
    ref_mm: float
    mld_mm: float
    length_mm: float

    @property
    def ds_pct(self) -> float:
        return self.spec.ds_pct

    @property
    def area_stenosis_pct(self) -> float:
        """Area stenosis (%): 1 − (1 − DS)²."""
        return 100.0 * (1.0 - (1.0 - self.spec.ds_pct / 100.0) ** 2)


def frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Twist-free frame along a curve: unit tangent, normal and binormal (n, 3) each (the
    normal is carried along by projection, so it does not spin about the curve)."""
    t = np.gradient(points, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.zeros_like(t)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n[0] = ref - (ref @ t[0]) * t[0]
    n[0] /= np.linalg.norm(n[0])
    for i in range(1, len(t)):
        v = n[i - 1] - (n[i - 1] @ t[i]) * t[i]
        norm = np.linalg.norm(v)
        n[i] = v / norm if norm > 1e-12 else n[i - 1]
    return t, n, np.cross(t, n)


def _bump(s: np.ndarray, centre: float, length: float) -> np.ndarray:
    """1 at ``centre``, falling as a cosine to 0 at ``length`` / 2 either side."""
    x = np.abs(s - centre) / (length / 2.0)
    return np.where(x < 1.0, 0.5 * (1.0 + np.cos(math.pi * np.minimum(x, 1.0))), 0.0)


def apply_lesions(
    patient: Patient, specs: list[LesionSpec]
) -> tuple[dict[str, Lumen], list[LesionTruth]]:
    """Every vessel's lumen with the lesions applied, and each lesion's truth."""
    lumens, truths = {}, []
    by_vessel: dict[str, list[tuple[LesionSpec, int]]] = {}
    for spec in specs:
        name, i0, i1 = patient.segment_span(spec.segment)
        by_vessel.setdefault(name, []).append((spec, int(round(i0 + spec.position * (i1 - i0)))))
    for name, v in patient.vessels.items():
        lumens[name], placed = _lumen(v, by_vessel.get(name, []))
        truths += placed
    return lumens, truths


def _lumen(v: Vessel, lesions: list[tuple[LesionSpec, int]]) -> tuple[Lumen, list[LesionTruth]]:
    t, n, b = frames(v.points)
    s, ref = v.s, v.radius
    req, ecc = ref.copy(), np.ones_like(ref)
    psi = np.zeros_like(ref)
    truths = []
    for spec, c in lesions:
        length = spec.length_mm * 1e-3
        f = _bump(s, s[c], length)
        req *= 1.0 - spec.ds_pct / 100.0 * f
        ecc *= 1.0 - (1.0 - spec.eccentricity) * f
        psi = np.where(f > 0, math.radians(spec.orientation_deg), psi)
        inside = np.flatnonzero(f > 0)
        start, end = (int(inside[0]), int(inside[-1])) if inside.size else (c, c)
        truths.append(
            LesionTruth(
                spec,
                v.name,
                start,
                c,
                end,
                2e3 * float(ref[c]),
                2e3 * float(ref[c]) * (1.0 - spec.ds_pct / 100.0),
                1e3 * float(s[end] - s[start]),
            )
        )
    ra = np.minimum(req / np.sqrt(ecc), ref)
    rb = req**2 / ra
    ea = np.cos(psi)[:, None] * n + np.sin(psi)[:, None] * b
    eb = np.cross(t, ea)
    return Lumen(ra, rb, ea, eb, t, ref), truths
