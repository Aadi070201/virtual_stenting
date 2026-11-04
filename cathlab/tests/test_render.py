"""The virtual X-ray: geometry, tube widths, lesions, crossings and eccentric lumens."""

import numpy as np
import pytest

from vcathlab.geometry import CArm
from vcathlab.lesions import LesionSpec
from vcathlab.render import downsample_layers, project_tube, rasterise
from vcathlab.scene import Scene, SceneConfig, ViewSpec, build_patient, sample_scene, simulate


def _carm(**kw) -> CArm:
    return CArm(kw.get("a", 0.0), kw.get("b", 0.0), 1.1, 0.75, 0.3e-3, (256, 256))


def _straight(n=600, along=(1.0, 0.0, 0.0), offset=(0.0, 0.0, 0.0)):
    t = np.linspace(-30e-3, 30e-3, n)[:, None] * np.asarray(along) + np.asarray(offset)
    tan = np.repeat(np.asarray(along, float)[None, :], n, axis=0)
    e1 = np.repeat(np.array([[0.0, 1.0, 0.0]]), n, axis=0) if along[1] == 0 else None
    if e1 is None:
        e1 = np.repeat(np.array([[1.0, 0.0, 0.0]]), n, axis=0)
    e2 = np.cross(tan, e1)
    return t, tan, e1, e2


def test_projection_and_pixel_size():
    c = _carm()
    assert c.project([[0.0, 0.0, 0.0]])[0] == pytest.approx([127.5, 127.5])
    assert c.iso_pixel_m == pytest.approx(0.3e-3 * 0.75 / 1.1)
    # a point towards the patient's left moves right in the AP view; towards the head, up
    assert c.project([[0.01, 0, 0]])[0, 1] > 127.5 and c.project([[0, 0, 0.01]])[0, 0] < 127.5


def test_tube_width_in_the_image():
    c = _carm()
    pts, tan, e1, e2 = _straight()
    r = np.full(len(pts), 1.5e-3)
    tube = project_tube(c, pts, r, r, e1, e2, tan, np.zeros(len(pts)), 0)
    cover, _, _ = downsample_layers(rasterise(c.shape, [tube], 2))
    width_px = cover[:, 128].sum()  # the tube runs along the columns
    assert width_px == pytest.approx(3.0e-3 / c.iso_pixel_m, abs=1.0)
    assert tube.chord_mm[300] == pytest.approx(3.0, rel=1e-6)  # perpendicular to the ray


def test_crossing_adds_and_is_marked():
    c = _carm()
    a = _straight(along=(1.0, 0.0, 0.0))
    b = _straight(along=(0.0, 0.0, 1.0), offset=(0.0, 10e-3, 0.0))  # behind, 10 mm deeper
    r = np.full(600, 1.5e-3)
    ta = project_tube(c, a[0], r, r, a[2], a[3], a[1], np.zeros(600), 0)
    tb = project_tube(c, b[0], r, r, b[2], b[3], b[1], np.ones(600), 1)
    lay = rasterise(c.shape, [ta, tb], 1)
    centre = tuple(np.round(c.project([[0, 0, 0]])[0]).astype(int))
    assert lay.overlap[centre] and lay.contrast_mm[centre] > 1.8 * 3.0 * 0.9
    lay_joined = rasterise(c.shape, [ta, ta], 1)  # the same lumen twice: joined, not summed
    assert not lay_joined.overlap.any()
    assert lay_joined.contrast_mm.max() == pytest.approx(3.0, rel=0.02)  # one 3 mm lumen


def test_lesion_narrows_the_vessel_mask():
    cfg = SceneConfig()
    base = Scene(
        seed=3, artery="left", view=ViewSpec(primary_deg=-30, secondary_deg=-25, iso_pixel_mm=0.15)
    )
    lesion = LesionSpec(segment="6", position=0.5, ds_pct=70.0, length_mm=12.0)
    with_lesion = base.model_copy(update={"lesions": [lesion]})
    f0, f1 = simulate(base, cfg), simulate(with_lesion, cfg)
    les = f1.lesions[0]
    assert les["ds_pct"] == 70.0 and les["box_xywh"] is not None
    assert les["mld_mm"] == pytest.approx(0.3 * les["ref_mm"])
    x, y, w, h = (int(round(v)) for v in les["box_xywh"])
    removed = f0.vessel[y : y + h, x : x + w].sum() - f1.vessel[y : y + h, x : x + w].sum()
    assert removed > 0  # the lumen is narrower inside the box
    assert les["apparent_ds_pct"] == pytest.approx(70.0, abs=3.0)  # round lumen: any view


def test_eccentric_lesion_looks_different_from_two_views():
    cfg = SceneConfig()
    lesion = LesionSpec(
        segment="6",
        position=0.5,
        ds_pct=50.0,
        length_mm=10.0,
        eccentricity=0.5,
        orientation_deg=30.0,
    )
    views = [ViewSpec(primary_deg=a, secondary_deg=b) for a, b in ((-30, -25), (45, 25))]
    apparent = [
        simulate(Scene(seed=3, artery="left", lesions=[lesion], view=v), cfg).lesions[0][
            "apparent_ds_pct"
        ]
        for v in views
    ]
    assert abs(apparent[0] - apparent[1]) > 3.0


def test_random_scenes_are_connected_and_labelled():
    cfg = SceneConfig()
    for seed in range(6):
        scene = sample_scene(seed, cfg)
        p = build_patient(scene, cfg)
        for v in p.vessels.values():
            if v.parent:
                parent = p.vessels[v.parent]
                gap = np.linalg.norm(
                    v.points[0] - parent.points[min(v.takeoff, len(parent.points) - 1)]
                )
                assert gap < 1e-6
        f = simulate(scene, cfg)
        assert f.image.dtype == np.uint8 and f.image.shape == (512, 512)
        assert f.vessel.sum() > 1000 and set(np.unique(f.segments)) <= set(range(-1, 16))
        assert np.all(f.vessel[f.overlap > 0] == 1)
        assert len(f.lesions) == len(scene.lesions)
        assert 0.15 < f.truth["iso_pixel_mm"] < 0.45
