import networkx as nx
import numpy as np
import pytest
from scipy.spatial import cKDTree

from vstent.imaging.centerline import (
    clean_mask,
    longest_path_endpoints,
    prune_spurs,
    skeleton_graph,
    smooth_and_resample,
    snap_to_graph,
    trace_centerline,
)
from vstent.imaging.qca import DEFAULT_MM_PER_PX, qca_from_centerline, run_qca
from vstent.imaging.synthetic import render_tube, sine_centerline, stenosis_diameter
from vstent.units import m_to_mm

SHAPE = (400, 400)
# (angle °, amplitude px, reference diameter px, blur sigma px, noise std)
CASES = [
    (0.0, 0.0, 12.0, 1.0, 0.03),
    (20.0, 25.0, 12.0, 1.0, 0.03),
    (-35.0, 30.0, 10.0, 1.2, 0.05),
    (60.0, 20.0, 14.0, 1.0, 0.03),
    (100.0, 35.0, 8.0, 1.0, 0.03),
]


def _tube(angle, amplitude, ref, blur, noise, seed=0, severity=0.6, length=280.0):
    pts, s = sine_centerline((0.0, 0.0), length, angle_deg=angle, amplitude_px=amplitude)
    pts = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2 + np.array(SHAPE) / 2  # centred
    d = stenosis_diameter(s, ref, severity, s[-1] / 2, 40.0)
    return render_tube(SHAPE, pts, s, d, blur_sigma=blur, noise_std=noise, seed=seed)


@pytest.mark.parametrize("case", CASES)
def test_qca_on_synthetic_stenosis_meets_targets(case):
    tube = _tube(*case, seed=CASES.index(case))
    p, d = tube.points, tube.diameter
    res = run_qca(tube.image, p[0] + [1.5, -2.0], p[-1] + [-2.0, 1.0])  # imprecise clicks
    _, idx = cKDTree(p).query(res.centerline.points)
    err = res.diameters.diameter_px - d[idx]
    assert np.median(np.abs(err)) <= 0.7
    throat = p[np.argmin(np.abs(tube.s - tube.s[-1] / 2))]
    assert np.linalg.norm(res.mld_point_px - throat) <= 3.0
    assert abs(res.max_percent_stenosis - 60.0) <= 7.0
    assert res.fallback_fraction < 0.05
    assert res.centerline.length > 0.9 * tube.s[-1]


def test_qca_units_and_calibration_flag():
    tube = _tube(0.0, 0.0, 12.0, 1.0, 0.0)
    assumed = run_qca(tube.image, tube.points[0], tube.points[-1])
    assert assumed.calibration_assumed and assumed.mm_per_px == DEFAULT_MM_PER_PX
    given = qca_from_centerline(tube.image, assumed.centerline, mm_per_px=0.2)
    assert not given.calibration_assumed
    np.testing.assert_allclose(given.diameter_mm, given.diameters.diameter_px * 0.2)
    assert given.s_mm[-1] == pytest.approx(given.centerline.length * 0.2)
    assert m_to_mm(given.profile.ds) == pytest.approx(0.2 * assumed.centerline.s[1])
    assert given.mld_mm == pytest.approx(given.diameter_mm.min())
    assert len(given.lesions) == 1 and given.reference_diameter_mm.shape == given.s_mm.shape


def test_skeleton_graph_and_spur_pruning():
    mask = np.zeros((40, 80), bool)
    mask[18:23, 5:75] = True  # horizontal bar
    mask[14:18, 40:43] = True  # short spur on top: ~6 px of skeleton up to the bar axis
    mask[23:40, 20:23] = True  # long branch (17 px) below
    from skimage.morphology import skeletonize

    g = skeleton_graph(skeletonize(clean_mask(mask, 1, 1)))
    pruned = prune_spurs(g, min_length=10.0)
    ends = [n for n, deg in pruned.degree() if deg == 1]
    assert len(ends) == 3  # bar ends + long branch; the 8 px spur is gone
    assert not any(r < 15 for r, _ in pruned.nodes)
    assert max(dict(pruned.degree()).values()) == 3


def test_snapping_and_disconnected_points():
    mask = np.zeros((40, 80), bool)
    mask[5:10, 5:75] = True
    mask[30:35, 5:75] = True
    from skimage.morphology import skeletonize

    g = skeleton_graph(skeletonize(mask))
    node = snap_to_graph(g, (0.0, 40.0))
    assert abs(node[0] - 7) <= 1 and abs(node[1] - 40) <= 1
    with pytest.raises(ValueError, match="no vessel path"):
        trace_centerline(mask, (7, 10), (32, 70))
    with pytest.raises(ValueError, match="no skeleton"):
        snap_to_graph(nx.Graph(), (0, 0))


def test_smooth_and_resample_is_uniform():
    path = np.stack([np.arange(50), np.round(np.linspace(0, 10, 50))], axis=1)
    cl = smooth_and_resample(path)
    steps = np.diff(cl.s)
    assert np.allclose(steps, steps[0]) and steps[0] == pytest.approx(1.0, rel=0.05)
    assert np.allclose(np.linalg.norm(cl.normals, axis=1), 1.0)
    assert np.allclose((cl.normals * cl.tangents).sum(axis=1), 0.0, atol=1e-9)


def test_end_trimming_removes_one_diameter_per_end():
    tube = _tube(0.0, 0.0, 12.0, 1.0, 0.0)
    full = run_qca(tube.image, tube.points[0], tube.points[-1])
    untrimmed = qca_from_centerline(tube.image, full.centerline, trim_diameters=0.0)
    assert untrimmed.centerline.length == pytest.approx(full.centerline.length)
    raw = trace_centerline(tube.image >= 0.5, tube.points[0], tube.points[-1])
    trimmed = qca_from_centerline(tube.image, raw)
    removed = raw.length - trimmed.centerline.length
    assert removed == pytest.approx(2 * 12.0, abs=2.0)  # median diameter ≈ reference 12 px
    assert trimmed.centerline.s[0] == 0.0


def test_longest_path_starts_at_wider_end():
    pts, s = sine_centerline((0.0, 0.0), 250.0, angle_deg=30.0, amplitude_px=10.0)
    pts = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2 + np.array(SHAPE) / 2
    wide_first = np.linspace(16.0, 6.0, s.size)  # tapering vessel
    for diameter, wide_end in ((wide_first, pts[0]), (wide_first[::-1], pts[-1])):
        tube = render_tube(SHAPE, pts, s, diameter, noise_std=0.0)
        a, _ = longest_path_endpoints(tube.image >= 0.5)
        assert np.hypot(a[0] - wide_end[0], a[1] - wide_end[1]) < 15


def test_side_branch_does_not_inflate_diameter():
    ref = 10.0
    pts, s = sine_centerline((0.0, 0.0), 240.0, amplitude_px=0.0)
    pts = pts + np.array([200.0, 80.0])  # horizontal vessel, row 200
    main = render_tube(SHAPE, pts, s, np.full(s.size, ref), noise_std=0.0, blur_sigma=0.8)
    bpts = np.stack([np.linspace(200.0, 60.0, 2000), np.full(2000, 200.0)], axis=1)
    bs = np.linspace(0.0, 140.0, 2000)
    branch = render_tube(SHAPE, bpts, bs, np.full(2000, 8.0), noise_std=0.0, blur_sigma=0.8)
    image = np.maximum(main.image, branch.image)
    res = run_qca(image, pts[0], pts[-1])
    assert res.diameters.branch.any()
    near_branch = np.abs(res.centerline.points[:, 1] - 200.0) < 6
    assert near_branch.any()
    assert np.all(res.diameters.diameter_px[near_branch] < 1.3 * ref)
    assert np.median(np.abs(res.diameters.diameter_px - ref)) < 0.5


def test_centreline_outside_vessel_is_interpolated():
    from vstent.imaging.centerline import Centerline
    from vstent.imaging.qca import measure_diameters

    tube = _tube(0.0, 0.0, 12.0, 1.0, 0.0, severity=0.0)
    cl = trace_centerline(tube.image >= 0.5, tube.points[0], tube.points[-1])
    pts = cl.points.copy()
    mid = len(pts) // 2
    pts[mid] = pts[mid] + 15.0 * cl.normals[mid]  # push one point well outside the tube
    d = measure_diameters(tube.image, Centerline(points=pts, s=cl.s))
    assert d.fallback[mid]
    assert d.diameter_px[mid] == pytest.approx(12.0, abs=1.0)


def test_robust_reference_ignores_spikes_and_dips():
    from vstent.imaging.qca import robust_reference

    s = np.arange(300.0)
    d = 12.0 - 0.01 * s
    d_meas = d.copy()
    d_meas[40:55] = 40.0  # overlap / branch spikes
    d_meas[150:180] = 5.0  # stenosis
    ref = robust_reference(s, d_meas)
    np.testing.assert_allclose(ref, d, atol=0.3)


def test_crossing_vessel_is_capped_not_a_stenosis():
    ref = 10.0
    pts, s = sine_centerline((0.0, 0.0), 260.0, amplitude_px=0.0)
    pts = pts + np.array([200.0, 70.0])
    main = render_tube(SHAPE, pts, s, np.full(s.size, ref), noise_std=0.0, blur_sigma=0.8)
    cpts = np.stack([np.linspace(60.0, 340.0, 2000), np.full(2000, 200.0)], axis=1)
    cs = np.linspace(0.0, 280.0, 2000)
    cross = render_tube(SHAPE, cpts, cs, np.full(2000, 10.0), noise_std=0.0, blur_sigma=0.8)
    res = run_qca(np.maximum(main.image, cross.image), pts[0], pts[-1])
    assert res.capped.any() and res.capped_fraction < 0.2
    np.testing.assert_allclose(res.reference_diameter_mm / res.mm_per_px, ref, atol=0.5)
    assert np.all(res.diameter_mm <= 1.15 * res.reference_diameter_mm + 1e-9)
    assert res.max_percent_stenosis < 10.0
