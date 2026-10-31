"""P4c: model-based lumen (imaging-chain fit) and gap bridging, on rendered vessels with
known lumens."""

import json

import numpy as np
import pytest

from vstent.imaging.centerline import (
    bridge_gaps,
    smooth_and_resample,
    trace_centerline,
    vessel_graph,
)
from vstent.imaging.lumen_model import cell_chord, forward, jacobian
from vstent.imaging.qca import qca_from_centerline
from vstent.imaging.synthetic import (
    render_projection,
    render_tubes,
    sine_centerline,
    stenosis_diameter,
)


def _vessel(ref_px, severity, kind="circular", seed=0):
    """Rendered vessel (ARCADE-like contrast and edge enhancement) with one cosine lesion,
    its filled mask (a segmentation that smoothed over the lesion) and the centreline."""
    pts, s = sine_centerline((100.0, 20.0), 220.0, amplitude_px=10.0, wavelength_px=300.0)
    narrow = stenosis_diameter(s, ref_px, severity, s[-1] / 2, 30.0)
    full = np.full_like(narrow, ref_px)
    width, depth = {"circular": (narrow, narrow), "depth": (full, narrow)}[kind]
    img = render_projection((200, 260), [(pts, width / 2, depth / 2)], mm_per_px=0.25,
                            mu_per_mm=0.06, background=0.6, blur_sigma=1.0, edge_amount=2.5,
                            edge_sigma=2.0, noise_std=0.015, seed=seed)  # fmt: skip
    filled = render_tubes((200, 260), [(pts, full)], blur_sigma=0, noise_std=0)
    true_ds = 100.0 * (1.0 - np.sqrt((width * depth).min()) / ref_px)
    centerline = smooth_and_resample(np.round(pts[::8]))
    return np.round(img * 255).astype(np.uint8), filled, centerline, true_ds


def test_cell_chord_keeps_the_exact_area():
    x = np.arange(-10.0, 10.0 + 0.25, 0.5)
    for r, x0 in ((4.0, 0.3), (0.2, -0.1), (0.05, 0.24)):  # incl. lumens below the cell size
        area = cell_chord(x, np.array([x0]), np.array([r]), step_px=0.5).sum() * 0.5
        assert area == pytest.approx(np.pi * r * r, rel=1e-9)
    # without blur or edge enhancement the model is the Beer–Lambert transmission
    y = forward(
        (0.1, 1e-3, 0.0, 1.0), np.array([3.0]), np.array([0.0]), np.array([0.0]), step_px=0.01
    )
    assert y[0, 0] == pytest.approx(np.exp(-0.1 * 6.0), rel=1e-3)


@pytest.mark.parametrize("severity", [0.7, 0.9])
def test_model_measures_tight_lesions_edges_cannot(severity):
    img, filled, cl, true_ds = _vessel(12.0, severity)
    edges = qca_from_centerline(filled, cl, image=img)
    model = qca_from_centerline(filled, cl, image=img, lumen="model")
    assert model.lumen_method == "model" and model.lumen_fit is not None
    assert abs(model.max_percent_stenosis - true_ds) <= 5.0
    if severity >= 0.9:  # a lumen of ~1 px: below the edge resolution floor
        assert edges.max_percent_stenosis < true_ds - 10.0
    fit = model.lumen_fit
    assert 0.6 <= fit.blur_px <= 1.6 and 1.0 <= fit.edge_amount <= 4.0  # chain recovered


def test_model_sees_a_lesion_narrowed_along_the_ray():
    """An eccentric lumen narrowed in depth only: edges see its full width, the model its
    reduced area (equivalent-circle %DS)."""
    img, filled, cl, true_ds = _vessel(12.0, 0.5, kind="depth")
    edges = qca_from_centerline(filled, cl, image=img)
    model = qca_from_centerline(filled, cl, image=img, lumen="model")
    assert edges.max_percent_stenosis < 15.0
    # P4c criterion: ≤ 10 points (this case: 7.9 before D45's edge changes, 8.3 after)
    assert abs(model.max_percent_stenosis - true_ds) <= 10.0


def test_lumen_option_checks():
    img, filled, cl, _ = _vessel(12.0, 0.5)
    with pytest.raises(ValueError, match="needs the image"):
        qca_from_centerline(filled, cl, lumen="model")
    with pytest.raises(ValueError, match="lumen must be one of"):
        qca_from_centerline(filled, cl, image=img, lumen="densitometric")


def test_analytic_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    n, x = 20, np.arange(-12.0, 12.5, 1.0)
    r, x0 = rng.uniform(0.3, 5.0, n), rng.uniform(-1.0, 1.0, n)
    chain = (0.02, 1.1, 2.0, 1.8)
    jac = jacobian(chain, r, x0, x).toarray()
    p = np.concatenate([chain, r, x0])

    def f(q):
        return forward(q[:4], q[4 : 4 + n], q[4 + n :], x).ravel()

    for k in range(p.size):
        step = np.zeros_like(p)
        step[k] = 1e-6 * max(1.0, abs(p[k]))
        num = (f(p + step) - f(p - step)) / (2 * step[k])
        assert np.abs(jac[:, k] - num).max() <= 1e-3 * np.abs(num).max() + 1e-9


def _gap_case(lumen_in_gap: float):
    """A vessel whose mask has a 16 px gap over its lesion (the segmentation lost it); the image
    shows a lesion of ``lumen_in_gap`` × 12 px (0: a total occlusion)."""
    pts, s = sine_centerline((100.0, 20.0), 220.0, amplitude_px=10.0, wavelength_px=300.0)
    mid = np.abs(s - s[-1] / 2) < 8.0
    width = np.where(mid, 12.0 * lumen_in_gap, 12.0)
    img = render_projection((200, 260), [(pts, width / 2)], mm_per_px=0.25, mu_per_mm=0.06,
                            background=0.6, blur_sigma=1.0, edge_amount=2.5, edge_sigma=2.0,
                            noise_std=0.015, seed=2)  # fmt: skip
    mask = render_tubes((200, 260), [(pts, np.full_like(s, 12.0))], blur_sigma=0,
                        noise_std=0) >= 0.5  # fmt: skip
    rr, cc = np.mgrid[0:200, 0:260]
    for p in pts[mid][::10]:
        mask[(rr - p[0]) ** 2 + (cc - p[1]) ** 2 <= 81] = False
    return np.round(img * 255).astype(np.uint8), mask.astype(np.float32), pts


@pytest.mark.parametrize("lumen_in_gap", [0.5, 0.0])
def test_a_gap_in_the_mask_is_bridged_and_measured_from_the_image(lumen_in_gap):
    img, mask, pts = _gap_case(lumen_in_gap)
    cl = trace_centerline(mask >= 0.5, pts[0], pts[-1])  # no path without the bridge
    assert cl.bridged_mask.sum() >= 10
    q = qca_from_centerline(mask, cl, image=img)
    true_ds = 100.0 * (1.0 - lumen_in_gap)
    # the gap is never filled with the healthy diameter around it
    assert q.max_percent_stenosis == pytest.approx(true_ds, abs=12.0 if lumen_in_gap else 8.0)
    # an analyst's mask with a gap (no image): no lumen there
    assert qca_from_centerline(mask, cl).max_percent_stenosis > 90.0


def test_gap_bridging_needs_aligned_nearby_ends():
    bars = np.zeros((60, 300), bool)
    bars[25:35, 10:120] = True
    bars[25:35, 180:290] = True  # 60 px gap: too far
    with pytest.raises(ValueError, match="no vessel path"):
        trace_centerline(bars, (30, 20), (30, 280))
    bars[25:35, 120:150] = True  # now 30 px, aligned: bridged
    cl = trace_centerline(bars, (30, 20), (30, 280))
    assert cl.bridged_mask.any() and 25 <= cl.points[cl.bridged_mask][:, 0].mean() <= 35
    graph = bridge_gaps(vessel_graph(bars))
    # the gap between the skeleton ends: 30 px plus the half-widths the skeleton stops short
    assert 30 <= sum(graph.nodes[n].get("bridge", False) for n in graph) <= 50
    offset = np.zeros((80, 300), bool)
    offset[10:20, 10:140] = True
    offset[50:60, 160:290] = True  # ends 40 px apart vertically, 20 px horizontally: not aligned
    assert not any(d.get("bridge") for _, d in bridge_gaps(vessel_graph(offset)).nodes(data=True))


def test_occlusion_and_gap_are_reported_to_the_user():
    from vstent.inference.pipeline import analyse_tree, analyse_vessel

    img, mask, pts = _gap_case(0.0)
    res = analyse_vessel(mask, pts[0], pts[-1], image=img, mm_per_px=0.25)
    assert res.qca.bridged_mm > 5.0 and res.qca.no_lumen_segments
    a, b = res.qca.no_lumen_segments[0]
    assert 15.0 < a < b < 40.0  # in the middle of the 55 mm path
    assert any("segmentation missed" in w for w in res.warnings)
    assert any("no measurable lumen" in w and "total occlusion" in w for w in res.warnings)
    tree = analyse_tree(mask, pts[0], pts[-1], image=img, mm_per_px=0.25)
    assert any(w.startswith("main: no measurable lumen") for w in tree.warnings)
    img1, mask1, _ = _gap_case(1.0)
    open_ = analyse_vessel(mask1, pts[0], pts[-1], image=img1, mm_per_px=0.25)
    assert not open_.qca.no_lumen_segments  # an open lumen in the gap is not flagged


def test_lumen_report(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from vstent.cli import app
    from vstent.imaging import lumen_report

    monkeypatch.setattr(lumen_report, "REFERENCES_PX", (12.0,))
    monkeypatch.setattr(lumen_report, "N_SEEDS", 1)
    monkeypatch.setattr(lumen_report, "LESIONS", ((0.9, "circular"), (0.5, "along the ray"),
                                                  (0.0, "none")))  # fmt: skip
    result = CliRunner().invoke(app, ["lumen-synthetic", "--report-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    rep = json.loads((tmp_path / "lumen_synthetic.json").read_text(encoding="utf-8"))
    tight = next(r for r in rep["rows"] if r["kind"] == "circular")
    assert abs(tight["model"]["ds"] - tight["true_ds"]) < abs(
        tight["edges"]["ds"] - tight["true_ds"]
    )
    assert tight["true_ffr"] < 0.8 and abs(tight["model"]["ffr"] - tight["true_ffr"]) < 0.1
    md = (tmp_path / "lumen_synthetic.md").read_text(encoding="utf-8")
    assert "Research use only" in md and "best case" in md
    assert (tmp_path / "figures" / "lumen_synthetic.png").exists()
