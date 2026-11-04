"""The virtual patient's heart and coronary arteries (no lesions; see :mod:`vcathlab.lesions`).

**Heart.** A prolate spheroid (half-axes ``a`` along the long axis, ``b`` across) whose long
axis points from the base towards the apex: to the patient's left, anterior and inferior. A
surface point is given by (θ, φ): θ from the base pole (0) to the apex (π), φ around the long
axis (0: the anterior face; +90°: the acute margin, right and inferior; −90°: the obtuse
margin, left and superior; 180°: the posterior face).

**Arteries** run on the epicardium (``fat_m`` above the surface) along the grooves:
- the atrioventricular (AV) groove, a ring at θ = ``theta_av``: the RCA (from its ostium on the
  right, round the acute margin to the crux at φ = 180°) and the circumflex (round the obtuse
  margin);
- the anterior interventricular groove (φ ≈ 0): the LAD, from the left main's bifurcation to
  the apex; its diagonals run over the anterolateral wall;
- the posterior interventricular groove (φ ≈ 180°): the PDA, from the crux towards the apex;
  the obtuse marginals run down the lateral wall.

The left main runs straight from the aortic root to the bifurcation (it is not on the
epicardium). Segments are numbered as in the SYNTAX score (``SEGMENTS``). Right dominance: the
RCA reaches the crux and gives the PDA (4) and a posterolateral branch (16); left dominance:
the circumflex reaches the crux and gives the PDA (15); the RCA ends at the acute margin.

**Sizes.** Proximal reference diameters are normal values (Dodge et al., Circulation 1992:
left main 4.5 ± 0.5 mm, proximal LAD 3.7 ± 0.4 mm, proximal RCA 3.9 ± 0.6 mm in right and
2.8 ± 0.5 mm in left dominance) times a patient scale. The circumflex follows Murray's law at
the bifurcation (d_LM³ = d_LAD³ + d_LCx³), side branches start at a sampled share of their
parent's diameter, and the parent's diameter after a take-off follows Murray's law again.
Between take-offs a vessel tapers gently. Units: SI (m) unless named.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from vcathlab.config import Range, StrictModel

STEP_M = 0.1e-3  # centreline step (m)
MIN_DIAMETER_M = 0.8e-3  # vessels end before narrowing below this
END_TAPER_M, END_SHARE = 8e-3, 0.4  # the last 8 mm narrow to 40% (vessels fade out)

SEGMENTS = {
    "1": "proximal RCA",
    "2": "mid RCA",
    "3": "distal RCA",
    "4": "posterior descending (RCA)",
    "16": "posterolateral (RCA)",
    "5": "left main",
    "6": "proximal LAD",
    "7": "mid LAD",
    "8": "apical LAD",
    "9": "first diagonal",
    "10": "second diagonal",
    "11": "proximal circumflex",
    "12a": "first obtuse marginal",
    "12b": "second obtuse marginal",
    "13": "distal circumflex",
    "15": "posterior descending (circumflex)",
}


class AnatomyConfig(StrictModel):
    """Sampling of a virtual patient's anatomy (uniform over [low, high] unless noted). May be
    tuned; normal diameters are from Dodge et al. 1992 (see the module docstring)."""

    heart_a_mm: Range = (55.0, 65.0)  # half long axis
    heart_b_mm: Range = (40.0, 48.0)  # half short axis
    heart_offset_sd_mm: float = Field(12.0, ge=0)  # centre vs isocentre, SD per axis
    axis_tilt_sd_deg: float = Field(8.0, ge=0)  # long axis vs its nominal direction
    theta_av_deg: Range = (52.0, 62.0)
    fat_mm: Range = (1.0, 2.5)
    size_scale_sd: float = Field(0.10, ge=0)  # patient scale of all diameters, SD
    left_dominance_prob: float = Field(0.1, ge=0, le=1)
    lm_length_mm: Range = (6.0, 15.0)
    lm_diameter_mm: float = 4.5
    lad_diameter_mm: float = 3.7
    rca_diameter_mm: float = 3.9  # right dominance
    rca_left_dominant_mm: float = 2.8
    side_ratio: Range = (0.55, 0.8)  # side branch start diameter / parent diameter
    taper_per_100mm: Range = (0.10, 0.25)  # share lost per 100 mm between take-offs
    wobble_deg: float = Field(2.5, ge=0)  # smooth random deviation from the grooves
    lad_end_theta_deg: Range = (150.0, 172.0)
    diagonal_length_mm: Range = (35.0, 75.0)
    marginal_length_mm: Range = (40.0, 80.0)
    pda_length_mm: Range = (40.0, 75.0)
    plv_length_mm: Range = (20.0, 45.0)


@dataclass(frozen=True)
class Heart:
    """The heart's frame: ``centre`` (3,) m, unit ``axis`` base → apex, ``e1`` (φ = 0, the
    anterior face) and ``e2`` (φ = +90°, the acute margin), half-axes ``a``, ``b`` (m),
    ``theta_av`` (rad) and ``fat`` (m)."""

    centre: np.ndarray
    axis: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    a: float
    b: float
    theta_av: float
    fat: float

    def surface(self, theta, phi, lift: float | None = None) -> np.ndarray:
        """(n, 3) points (m) ``lift`` (default: the fat layer) above the surface at (θ, φ)."""
        theta, phi = np.atleast_1d(theta).astype(float), np.atleast_1d(phi).astype(float)
        radial = np.cos(phi)[:, None] * self.e1 + np.sin(phi)[:, None] * self.e2
        p = (
            self.centre
            - (self.a * np.cos(theta))[:, None] * self.axis
            + (self.b * np.sin(theta))[:, None] * radial
        )
        n = (-np.cos(theta) / self.a)[:, None] * self.axis + (np.sin(theta) / self.b)[
            :, None
        ] * radial
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return p + (self.fat if lift is None else lift) * n

    @property
    def aortic_root(self) -> np.ndarray:
        """(3,) the aortic root (m): just above the base pole, slightly anterior."""
        return self.centre - 1.05 * self.a * self.axis + 0.25 * self.b * self.e1


@dataclass(eq=False)
class Vessel:
    """One artery as a tube: ``points`` (n, 3) m, ``STEP_M`` apart, the reference radius
    ``radius`` (n,) m (lesions narrow it later), the SYNTAX ``segment`` of each point, the
    ``parent`` vessel's name and the parent's point index at the take-off (``None`` for an
    ostium). ``theta``, ``phi``: the surface coordinates (``None`` off the surface)."""

    name: str
    points: np.ndarray
    radius: np.ndarray
    segment: np.ndarray
    parent: str | None = None
    takeoff: int | None = None
    theta: np.ndarray | None = None
    phi: np.ndarray | None = None

    @property
    def s(self) -> np.ndarray:
        """(n,) arc length (m)."""
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    @property
    def length(self) -> float:
        return float(self.s[-1])


@dataclass(eq=False)
class Patient:
    """A virtual patient: the heart, the left or right coronary tree (``artery``), dominance,
    the patient scale of the diameters, and the vessels by name (parents first)."""

    heart: Heart
    artery: str
    dominance: str
    scale: float
    vessels: dict[str, Vessel] = field(default_factory=dict)

    def segment_span(self, segment: str) -> tuple[str, int, int]:
        """(vessel name, first index, last index) of a SYNTAX segment."""
        for v in self.vessels.values():
            idx = np.flatnonzero(v.segment == segment)
            if idx.size:
                return v.name, int(idx[0]), int(idx[-1])
        raise KeyError(f"segment {segment} is not in this patient's {self.artery} tree")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def sample_heart(rng: np.random.Generator, cfg: AnatomyConfig) -> Heart:
    """A heart with sampled size, position and long-axis direction."""
    nominal = _unit(np.array([0.6, -0.55, -0.55]))  # left, anterior, inferior
    tilt = math.radians(cfg.axis_tilt_sd_deg)
    axis = _unit(nominal + tilt * rng.normal(size=3))
    anterior = np.array([0.0, -1.0, 0.0])
    e1 = _unit(anterior - (anterior @ axis) * axis)
    e2 = np.cross(axis, e1)
    return Heart(
        centre=rng.normal(0.0, cfg.heart_offset_sd_mm, 3) * 1e-3,
        axis=axis,
        e1=e1,
        e2=e2,
        a=rng.uniform(*cfg.heart_a_mm) * 1e-3,
        b=rng.uniform(*cfg.heart_b_mm) * 1e-3,
        theta_av=math.radians(rng.uniform(*cfg.theta_av_deg)),
        fat=rng.uniform(*cfg.fat_mm) * 1e-3,
    )


def _groove_path(rng, heart: Heart, waypoints_deg, wobble_deg: float, length: float | None = None):
    """A smooth surface path through (θ, φ) ``waypoints_deg`` with a little random wobble
    (none at the start, so a branch leaves its parent's centreline),
    resampled every ``STEP_M``; cut at ``length`` (m) if given. Returns (points, θ, φ)."""
    wp = np.radians(np.asarray(waypoints_deg, float))
    k = np.arange(len(wp))
    fine = np.linspace(0.0, k[-1], 4000)
    theta, phi = PchipInterpolator(k, wp[:, 0])(fine), PchipInterpolator(k, wp[:, 1])(fine)
    if wobble_deg > 0:
        for arr in (theta, phi):
            noise = gaussian_filter1d(rng.normal(size=fine.size), 300.0, mode="nearest")
            noise *= np.clip(np.arange(fine.size) / 400.0, 0.0, 1.0)  # none at the start
            arr += math.radians(wobble_deg) * noise / max(float(np.std(noise)), 1e-9)
    pts = heart.surface(theta, phi)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    end = s[-1] if length is None else min(length, s[-1])
    out = np.arange(0.0, end, STEP_M)
    th, ph = np.interp(out, s, theta), np.interp(out, s, phi)
    return heart.surface(th, ph), th, ph


def _taper(n: int, r0: float, per_100mm: float) -> np.ndarray:
    """(n,) radius (m) from ``r0`` losing ``per_100mm`` of itself per 100 mm."""
    return r0 * (1.0 - per_100mm) ** (np.arange(n) * STEP_M / 0.1)


def _branch(
    rng,
    heart,
    parent: Vessel,
    at: float,
    name: str,
    segment: str,
    heading_deg,
    length: float,
    cfg: AnatomyConfig,
) -> Vessel:
    """A side branch leaving ``parent`` at ``at`` (share of its length) along the surface,
    through (Δθ, Δφ) ``heading_deg`` waypoints relative to the take-off; it starts on the
    parent's centreline (the renderer joins the two). The parent's radius after the take-off
    follows Murray's law."""
    j = int(round(at * (len(parent.points) - 1)))
    t0, p0 = math.degrees(parent.theta[j]), math.degrees(parent.phi[j])
    wps = [(t0, p0)] + [(t0 + dt, p0 + dp) for dt, dp in heading_deg]
    pts, th, ph = _groove_path(rng, heart, wps, cfg.wobble_deg, length)
    r_parent = float(parent.radius[j])
    r0 = r_parent * rng.uniform(*cfg.side_ratio)
    radius = _taper(len(pts), r0, rng.uniform(*cfg.taper_per_100mm))
    after = max(r_parent**3 - r0**3, (0.5 * r_parent) ** 3) ** (1 / 3)
    ramp = np.clip((np.arange(len(parent.radius)) - j) * STEP_M / 3e-3, 0.0, 1.0)
    parent.radius = parent.radius * (1.0 - ramp * (1.0 - after / r_parent))
    return Vessel(
        name, pts, radius, np.full(len(pts), segment, dtype=object), parent.name, j, th, ph
    )


def _cut_thin(v: Vessel) -> Vessel:
    """``v`` ending where its radius falls below ``MIN_DIAMETER_M`` / 2, its last
    ``END_TAPER_M`` narrowing to ``END_SHARE`` of the radius (vessels fade out)."""
    n = (
        int(np.argmax(v.radius < MIN_DIAMETER_M / 2))
        if np.any(v.radius < MIN_DIAMETER_M / 2)
        else len(v.radius)
    )
    n = max(n, 2)
    th = None if v.theta is None else v.theta[:n]
    ph = None if v.phi is None else v.phi[:n]
    k = min(int(END_TAPER_M / STEP_M), n)
    radius = v.radius[:n].copy()
    radius[n - k :] *= np.linspace(1.0, END_SHARE, k)
    return Vessel(v.name, v.points[:n], radius, v.segment[:n], v.parent, v.takeoff, th, ph)


def _segments(n: int, cuts: list[float], names: list[str]) -> np.ndarray:
    """(n,) segment name per point: ``names[k]`` up to the share ``cuts[k]`` of the length."""
    share = np.arange(n) / max(n - 1, 1)
    out = np.full(n, names[-1], dtype=object)
    for cut, name in zip(reversed(cuts), reversed(names[:-1]), strict=True):
        out[share <= cut] = name
    return out


def sample_patient(
    rng: np.random.Generator, cfg: AnatomyConfig, artery: str, dominance: str | None = None
) -> Patient:
    """A virtual patient's ``artery`` tree ("left" or "right"); ``dominance`` sampled unless
    given."""
    if artery not in ("left", "right"):
        raise ValueError("artery is left or right")
    heart = sample_heart(rng, cfg)
    if dominance is None:
        dominance = "left" if rng.random() < cfg.left_dominance_prob else "right"
    scale = float(np.clip(rng.normal(1.0, cfg.size_scale_sd), 0.75, 1.3))
    patient = Patient(heart, artery, dominance, scale)
    tav = math.degrees(heart.theta_av)
    w = cfg.wobble_deg
    if artery == "right":
        d0 = (cfg.rca_diameter_mm if dominance == "right" else cfg.rca_left_dominant_mm) * scale
        end_phi = 180.0 if dominance == "right" else rng.uniform(95.0, 120.0)
        wps = [(tav - 14, 22), (tav - 4, 40), (tav, 70), (tav, 110), (tav + 2, end_phi)]
        pts, th, ph = _groove_path(rng, heart, wps, w)
        r = _taper(len(pts), d0 * 1e-3 / 2, rng.uniform(*cfg.taper_per_100mm))
        cuts = [0.35, 0.7] if dominance == "right" else [0.5, 0.8]
        rca = Vessel("RCA", pts, r, _segments(len(pts), cuts, ["1", "2", "3"]), None, None, th, ph)
        patient.vessels["RCA"] = rca
        if dominance == "right":
            patient.vessels["PDA"] = _branch(
                rng,
                heart,
                rca,
                0.985,
                "PDA",
                "4",
                [(30, 2), (60, 0), (90, -3)],
                rng.uniform(*cfg.pda_length_mm) * 1e-3,
                cfg,
            )
            patient.vessels["PLV"] = _branch(
                rng,
                heart,
                rca,
                1.0,
                "PLV",
                "16",
                [(3, 15), (15, 30), (35, 40)],
                rng.uniform(*cfg.plv_length_mm) * 1e-3,
                cfg,
            )
    else:
        lm_d = cfg.lm_diameter_mm * scale * 1e-3
        lad_d = min(cfg.lad_diameter_mm * scale * rng.normal(1.0, 0.04) * 1e-3, 0.92 * lm_d)
        lcx_d = (lm_d**3 - lad_d**3) ** (1 / 3)
        bif_phi = rng.uniform(-28.0, -12.0)
        bif = heart.surface(math.radians(tav), math.radians(bif_phi))[0]
        lm_len = rng.uniform(*cfg.lm_length_mm) * 1e-3
        ost = bif + lm_len * _unit(heart.aortic_root - bif)
        lm_pts = ost + np.linspace(0.0, 1.0, max(int(lm_len / STEP_M), 2))[:, None] * (bif - ost)
        lm = Vessel(
            "LM", lm_pts, np.full(len(lm_pts), lm_d / 2), np.full(len(lm_pts), "5", dtype=object)
        )
        end_theta = rng.uniform(*cfg.lad_end_theta_deg)
        lad_wps = [(tav, bif_phi), (tav + 20, bif_phi + 10), (tav + 55, 2), (end_theta, 8)]
        pts, th, ph = _groove_path(rng, heart, lad_wps, w)
        lad = Vessel(
            "LAD",
            pts,
            _taper(len(pts), lad_d / 2, rng.uniform(*cfg.taper_per_100mm)),
            _segments(len(pts), [0.3, 0.65], ["6", "7", "8"]),
            "LM",
            len(lm_pts) - 1,
            th,
            ph,
        )
        lcx_end = -175.0 if dominance == "left" else rng.uniform(-125.0, -95.0)
        lcx_wps = [(tav, bif_phi), (tav + 1, bif_phi - 30), (tav + 2, -80), (tav + 2, lcx_end)]
        pts, th, ph = _groove_path(rng, heart, lcx_wps, w)
        lcx = Vessel(
            "LCx",
            pts,
            _taper(len(pts), lcx_d / 2, rng.uniform(*cfg.taper_per_100mm)),
            _segments(len(pts), [0.4], ["11", "13"]),
            "LM",
            len(lm_pts) - 1,
            th,
            ph,
        )
        patient.vessels.update(LM=lm, LAD=lad, LCx=lcx)
        patient.vessels["D1"] = _branch(
            rng,
            heart,
            lad,
            0.28,
            "D1",
            "9",
            [(15, -25), (35, -45), (55, -60)],
            rng.uniform(*cfg.diagonal_length_mm) * 1e-3,
            cfg,
        )
        patient.vessels["D2"] = _branch(
            rng,
            heart,
            lad,
            0.55,
            "D2",
            "10",
            [(12, -22), (28, -38), (45, -50)],
            0.8 * rng.uniform(*cfg.diagonal_length_mm) * 1e-3,
            cfg,
        )
        patient.vessels["OM1"] = _branch(
            rng,
            heart,
            lcx,
            0.38,
            "OM1",
            "12a",
            [(25, -3), (55, 0), (85, 5)],
            rng.uniform(*cfg.marginal_length_mm) * 1e-3,
            cfg,
        )
        patient.vessels["OM2"] = _branch(
            rng,
            heart,
            lcx,
            0.7,
            "OM2",
            "12b",
            [(25, -5), (50, -5), (75, 0)],
            0.85 * rng.uniform(*cfg.marginal_length_mm) * 1e-3,
            cfg,
        )
        if dominance == "left":
            patient.vessels["LPDA"] = _branch(
                rng,
                heart,
                lcx,
                0.985,
                "LPDA",
                "15",
                [(30, -2), (60, 0), (90, 3)],
                rng.uniform(*cfg.pda_length_mm) * 1e-3,
                cfg,
            )
    for name, v in list(patient.vessels.items()):
        patient.vessels[name] = _cut_thin(v)
    return patient
