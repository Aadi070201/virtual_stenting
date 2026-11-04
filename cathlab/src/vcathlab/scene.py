"""A scene (one virtual patient, one injection, one C-arm view) and its X-ray frame with every
label.

A :class:`Scene` is fully specified and editable: the patient's ``seed`` (anatomy and imaging
details), the injected artery, dominance, the lesions, the view and the catheter. Editing a
lesion or the view keeps the same patient. :func:`sample_scene` draws a random scene;
:func:`simulate` renders it.

**Labels** of a :class:`Frame`: the vessel (lumen coverage ≥ 0.5), the SYNTAX segment with the
largest chord at each pixel, where vessels overlap, the catheter, and per lesion its true
severity, its box in the image, its apparent severity in this view (projected width vs the
reference), the share of it lying on an overlap, how foreshortened it is (the smallest sine of
the angle between the vessel and the ray), and the pixel size at its depth. Units: SI inside;
the frame's truth in mm and px as named.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from pydantic import Field

from vcathlab.anatomy import SEGMENTS, AnatomyConfig, Patient, sample_patient
from vcathlab.config import Range, StrictModel
from vcathlab.geometry import CArm
from vcathlab.lesions import LesionSpec, LesionTruth, apply_lesions, frames
from vcathlab.render import (
    MIN_SINE,
    Imaging,
    ImagingConfig,
    background_map,
    downsample_layers,
    expose,
    heart_shadow,
    project_tube,
    rasterise,
)

SEGMENT_CODES = list(SEGMENTS)  # segment map values: index into this list; −1: no vessel
BOX_MARGIN_PX = 2.0

STANDARD_VIEWS = {  # (primary, secondary) degrees
    "left": {
        "RAO caudal": (-30, -25),
        "AP cranial": (-10, 40),
        "LAO cranial": (45, 25),
        "LAO caudal (spider)": (45, -30),
        "AP caudal": (0, -30),
        "RAO cranial": (-30, 30),
    },
    "right": {"LAO": (30, 0), "RAO": (-30, 0), "LAO cranial": (30, 20), "AP cranial": (0, 25)},
}


class ViewSpec(StrictModel):
    """The C-arm: angles (degrees), source-to-detector and source-to-isocentre distances (mm),
    the pixel size at the isocentre (mm) and the image size (px)."""

    primary_deg: float
    secondary_deg: float
    sid_mm: float = 1100.0
    sod_mm: float = 760.0
    iso_pixel_mm: float = Field(0.28, gt=0)
    image_size: int = Field(512, ge=64)
    pan_px: tuple[float, float] = (0.0, 0.0)  # framing offset from the tree's centre (row, col)

    def carm(self) -> CArm:
        detector = self.iso_pixel_mm * self.sid_mm / self.sod_mm
        return CArm(
            self.primary_deg,
            self.secondary_deg,
            self.sid_mm * 1e-3,
            self.sod_mm * 1e-3,
            detector * 1e-3,
            (self.image_size, self.image_size),
        )


class Scene(StrictModel):
    """One fully specified image."""

    seed: int
    artery: Literal["left", "right"]
    dominance: Literal["right", "left"] = "right"
    lesions: list[LesionSpec] = Field(default_factory=list)
    view: ViewSpec
    catheter_french: float = Field(6.0, gt=0)


class SceneConfig(StrictModel):
    """Sampling of random scenes (uniform over [low, high] unless noted). May be tuned."""

    anatomy: AnatomyConfig = Field(default_factory=AnatomyConfig)
    imaging: ImagingConfig = Field(default_factory=ImagingConfig)
    left_prob: float = Field(0.65, ge=0, le=1)
    lesions: tuple[int, int] = (0, 3)
    significant_prob: float = Field(0.55, ge=0, le=1)  # ≥ 50%; otherwise a mild 20–49%
    mild_ds_pct: Range = (20.0, 49.0)
    significant_ds_pct: Range = (50.0, 95.0)
    lesion_length_mm: Range = (4.0, 25.0)
    eccentricity: Range = (0.5, 1.0)
    concentric_prob: float = Field(0.4, ge=0, le=1)
    min_ref_mm: float = Field(1.5, gt=0)  # lesions only where the reference is at least this
    view_jitter_deg: float = Field(8.0, ge=0)
    pan_sd_px: float = Field(25.0, ge=0)  # framing: the tree's centre off the image centre
    sid_mm: Range = (900.0, 1200.0)
    sod_mm: Range = (720.0, 800.0)
    iso_pixel_mm: Range = (0.18, 0.40)
    image_size: int = Field(512, ge=64)
    catheter_french: tuple[float, ...] = (5.0, 6.0, 6.0, 7.0)


@dataclass
class Frame:
    """A rendered scene: the image (H, W) uint8 and its labels (see the module docstring)."""

    scene: Scene
    image: np.ndarray
    vessel: np.ndarray  # uint8 0/1
    segments: np.ndarray  # int16, index into SEGMENT_CODES, −1 none
    overlap: np.ndarray  # uint8 0/1
    catheter: np.ndarray  # uint8 0/1
    lesions: list[dict[str, Any]] = field(default_factory=list)
    truth: dict[str, Any] = field(default_factory=dict)


def _rngs(seed: int) -> tuple[np.random.Generator, ...]:
    """Independent streams for the anatomy, the imaging and the noise of one patient."""
    return tuple(np.random.default_rng([seed, k]) for k in range(3))


def build_patient(scene: Scene, cfg: SceneConfig) -> Patient:
    """The scene's patient (anatomy from the seed only)."""
    return sample_patient(_rngs(scene.seed)[0], cfg.anatomy, scene.artery, scene.dominance)


def sample_scene(seed: int, cfg: SceneConfig) -> Scene:
    """A random scene: artery, dominance, lesions on vessels ≥ ``min_ref_mm``, a standard view
    with jitter, the pixel size and the catheter."""
    rng = np.random.default_rng([seed, 99])
    artery = "left" if rng.random() < cfg.left_prob else "right"
    dominance = "left" if rng.random() < cfg.anatomy.left_dominance_prob else "right"
    probe = Scene(
        seed=seed, artery=artery, dominance=dominance, view=ViewSpec(primary_deg=0, secondary_deg=0)
    )
    patient = build_patient(probe, cfg)
    spans = []
    for v in patient.vessels.values():
        for seg in dict.fromkeys(v.segment):
            idx = np.flatnonzero(v.segment == seg)
            if 2e3 * float(np.median(v.radius[idx])) >= cfg.min_ref_mm and idx.size > 50:
                spans.append(str(seg))
    lesions = []
    for _ in range(int(rng.integers(cfg.lesions[0], cfg.lesions[1] + 1)) if spans else 0):
        sig = rng.random() < cfg.significant_prob
        ecc = 1.0 if rng.random() < cfg.concentric_prob else rng.uniform(*cfg.eccentricity)
        lesions.append(
            LesionSpec(
                segment=str(rng.choice(spans)),
                position=float(rng.uniform(0.15, 0.85)),
                ds_pct=float(rng.uniform(*(cfg.significant_ds_pct if sig else cfg.mild_ds_pct))),
                length_mm=float(rng.uniform(*cfg.lesion_length_mm)),
                eccentricity=float(ecc),
                orientation_deg=float(rng.uniform(0, 180)),
            )
        )
    name = str(rng.choice(list(STANDARD_VIEWS[artery])))
    a, b = STANDARD_VIEWS[artery][name]
    j = cfg.view_jitter_deg
    view = ViewSpec(
        primary_deg=float(a + rng.uniform(-j, j)),
        secondary_deg=float(b + rng.uniform(-j, j)),
        sid_mm=float(rng.uniform(*cfg.sid_mm)),
        sod_mm=float(rng.uniform(*cfg.sod_mm)),
        iso_pixel_mm=float(rng.uniform(*cfg.iso_pixel_mm)),
        image_size=cfg.image_size,
        pan_px=(float(rng.normal(0, cfg.pan_sd_px)), float(rng.normal(0, cfg.pan_sd_px))),
    )
    return Scene(
        seed=seed,
        artery=artery,
        dominance=dominance,
        lesions=lesions,
        view=view,
        catheter_french=float(rng.choice(cfg.catheter_french)),
    )


def framed(carm: CArm, patient: Patient, pan_px=(0.0, 0.0)) -> CArm:
    """``carm`` with the table panned so the injected tree's bounding box is centred, then
    moved by ``pan_px`` (row, col)."""
    pts = np.vstack([carm.project(v.points) for v in patient.vessels.values()])
    mid = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    centre = np.array([(carm.shape[0] - 1) / 2.0, (carm.shape[1] - 1) / 2.0])
    return replace(carm, shift_px=tuple(float(x) for x in centre - mid + np.asarray(pan_px)))


def _catheter(patient: Patient, french: float) -> tuple[np.ndarray, float]:
    """The guiding catheter's centreline (k, 3) m, from the injected artery's ostium back to
    the aortic root and up the ascending aorta, and its outer radius (m)."""
    first = patient.vessels["LM" if patient.artery == "left" else "RCA"].points
    tip, root = first[0], patient.heart.aortic_root
    up = np.array([-0.15, -0.1, 1.0])
    up /= np.linalg.norm(up)
    bend = root + 0.02 * up
    t = np.linspace(0.0, 1.0, 300)[:, None]
    curve = (1 - t) ** 2 * tip + 2 * (1 - t) * t * root + t**2 * bend  # quadratic Bézier
    shaft = bend + np.linspace(0.0, 0.15, 1500)[1:, None] * up
    pts = np.vstack([curve, shaft])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    even = np.arange(0.0, s[-1], 0.1e-3)
    pts = np.stack([np.interp(even, s, pts[:, k]) for k in range(3)], axis=1)
    return pts, french / 3.0 * 1e-3 / 2.0


def simulate(scene: Scene, cfg: SceneConfig | None = None) -> Frame:
    """Render ``scene`` with all its labels."""
    cfg = cfg or SceneConfig()
    _, rng_img, rng_noise = _rngs(scene.seed)
    patient = build_patient(scene, cfg)
    lumens, truths = apply_lesions(patient, scene.lesions)
    carm = framed(scene.view.carm(), patient, scene.view.pan_px)
    im = Imaging.sample(rng_img, cfg.imaging)
    names = list(patient.vessels)
    tubes = []
    for k, name in enumerate(names):
        v, lu = patient.vessels[name], lumens[name]
        codes = np.array([SEGMENT_CODES.index(str(sg)) for sg in v.segment], dtype=np.int32)
        tubes.append(project_tube(carm, v.points, lu.ra, lu.rb, lu.ea, lu.eb, lu.tangent, codes, k))
    cat_pts, cat_r = _catheter(patient, scene.catheter_french)
    t, n, b = frames(cat_pts)
    inner = np.full(len(cat_pts), max(cat_r - 0.1e-3, 0.2e-3))
    cat_id = len(names)
    outer = np.full(len(cat_pts), cat_r)
    no_label = np.full(len(cat_pts), -1, dtype=np.int32)
    cat_lumen = project_tube(carm, cat_pts, inner, inner, n, b, t, no_label, cat_id)
    cat_wall = project_tube(carm, cat_pts, outer, outer, n, b, t, no_label, cat_id, "wall")
    layers = rasterise(carm.shape, [*tubes, cat_lumen, cat_wall], cfg.imaging.supersample)
    heart = patient.heart
    heart_mm = heart_shadow(carm, heart.centre, heart.axis, heart.a, heart.b)
    bg = background_map(rng_img, carm.shape, im, cfg.imaging)
    image = expose(rng_noise, layers, heart_mm, bg, im, cfg.imaging)
    cover, seg, over = downsample_layers(layers)
    cat_only = rasterise(carm.shape, [cat_lumen], cfg.imaging.supersample)
    cat_cover = downsample_layers(cat_only)[0]
    vessel = (cover >= 0.5) & (seg >= 0)
    frame = Frame(
        scene,
        image,
        vessel.astype(np.uint8),
        np.where(vessel, seg, -1).astype(np.int16),
        (over & vessel).astype(np.uint8),
        (cat_cover >= 0.5).astype(np.uint8),
    )
    frame.lesions = [_lesion_label(tr, patient, lumens, carm, over) for tr in truths]
    frame.truth = {
        "artery": scene.artery,
        "dominance": scene.dominance,
        "scale": patient.scale,
        "view": carm.label(),
        "iso_pixel_mm": 1e3 * carm.iso_pixel_m,
        "detector_pixel_mm": 1e3 * carm.pixel_m,
        "sid_mm": scene.view.sid_mm,
        "sod_mm": scene.view.sod_mm,
        "heart_centre_pixel_mm": float(1e3 * carm.pixel_size_at(heart.centre)[0]),
        "catheter_french": scene.catheter_french,
        "catheter_outer_mm": 2e3 * cat_r,
        "imaging": im.__dict__,
    }
    return frame


def _lesion_label(
    tr: LesionTruth, patient: Patient, lumens, carm: CArm, overlap: np.ndarray
) -> dict[str, Any]:
    """One lesion's labels in this view."""
    v, lu = patient.vessels[tr.vessel], lumens[tr.vessel]
    sl = slice(tr.start, tr.end + 1)
    p = project_tube(
        carm,
        v.points[sl],
        lu.ra[sl],
        lu.rb[sl],
        lu.ea[sl],
        lu.eb[sl],
        lu.tangent[sl],
        np.zeros(tr.end - tr.start + 1),
        0,
    )
    pix = carm.pixel_size_at(v.points[sl])
    ref_px = lu.ref[sl] / pix
    apparent = 100.0 * (1.0 - float(np.min(p.half_width / ref_px)))
    d = carm.ray_directions(v.points[sl])
    sine = np.sqrt(np.clip(1.0 - np.sum(d * lu.tangent[sl], axis=1) ** 2, 0.0, 1.0))
    h, w = carm.shape
    lo = (p.px - ref_px[:, None]).min(axis=0) - BOX_MARGIN_PX
    hi = (p.px + ref_px[:, None]).max(axis=0) + BOX_MARGIN_PX
    inside = (p.px[:, 0] >= 0) & (p.px[:, 0] < h) & (p.px[:, 1] >= 0) & (p.px[:, 1] < w)
    lo, hi = np.maximum(lo, 0.0), np.minimum(hi, (h - 1.0, w - 1.0))
    box = None
    if inside.any() and np.all(hi > lo):
        box = [float(lo[1]), float(lo[0]), float(hi[1] - lo[1]), float(hi[0] - lo[0])]
    rc = np.clip(np.rint(p.px).astype(int), 0, (h - 1, w - 1))
    return {
        "segment": tr.spec.segment,
        "segment_name": SEGMENTS[tr.spec.segment],
        "vessel": tr.vessel,
        "ds_pct": tr.ds_pct,
        "area_stenosis_pct": tr.area_stenosis_pct,
        "length_mm": tr.length_mm,
        "ref_mm": tr.ref_mm,
        "mld_mm": tr.mld_mm,
        "eccentricity": tr.spec.eccentricity,
        "orientation_deg": tr.spec.orientation_deg,
        "apparent_ds_pct": apparent,
        "box_xywh": box,
        "inside_share": float(inside.mean()),
        "overlap_share": float(overlap[rc[:, 0], rc[:, 1]].mean()),
        "min_sine": float(max(sine.min(), 0.0)),
        "foreshortened": bool(sine.min() < 2 * MIN_SINE),
        "pixel_mm": float(1e3 * pix[tr.centre - tr.start]),
    }


def overlay(frame: Frame) -> np.ndarray:
    """An RGB (H, W, 3) uint8 picture of the frame with its lesion boxes (≥ 50%: red, mild:
    yellow)."""
    import cv2

    rgb = np.repeat(frame.image[..., None], 3, axis=2).copy()
    for les in frame.lesions:
        if les["box_xywh"] is None:
            continue
        x, y, w, h = (int(round(v)) for v in les["box_xywh"])
        color = (220, 40, 40) if les["ds_pct"] >= 50 else (230, 200, 40)
        cv2.rectangle(rgb, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            rgb,
            f"{les['segment']} {les['ds_pct']:.0f}%",
            (x, max(y - 3, 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return rgb


def seg_colours(frame: Frame) -> np.ndarray:
    """An RGB (H, W, 3) uint8 picture of the segment map (one colour per segment)."""
    rng = np.random.default_rng(7)
    palette = rng.integers(60, 255, size=(len(SEGMENT_CODES), 3)).astype(np.uint8)
    out = np.zeros((*frame.segments.shape, 3), dtype=np.uint8)
    ok = frame.segments >= 0
    out[ok] = palette[frame.segments[ok]]
    return out
