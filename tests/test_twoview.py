"""P5: projection geometry, DICOM geometry, two-view reconstruction, synthetic two-view cases."""

import dataclasses
import io

import numpy as np
import pytest
from scipy.spatial import cKDTree
from typer.testing import CliRunner

from vstent.cli import app
from vstent.imaging.centerline import smooth_and_resample
from vstent.imaging.image_io import read_image
from vstent.imaging.projection import ProjectionGeometry, read_projection
from vstent.imaging.synthetic3d import (
    TwoViewSynthConfig,
    main_overlaps,
    render_view,
    sample_case,
)
from vstent.imaging.twoview import (
    _across_epipolar,
    closest_points,
    estimate_shift,
    monotone_path,
    reconstruct,
    two_view_profile,
)
from vstent.inference.pipeline import analyse_two_views
from vstent.physics.solver import PhysicsParams

PIXEL = 0.35e-3


def _geometry(primary, secondary, sid=1.1, sod=0.75, shape=(512, 512)):
    return ProjectionGeometry(primary, secondary, sid, sod, (PIXEL, PIXEL), shape)


def _curve(n: int = 4000) -> np.ndarray:
    """A smooth 3D curve around the isocentre (m), about 100 mm long."""
    t = np.linspace(0.0, 1.0, n)
    th = 0.2 + 1.6 * t
    return np.stack([0.045 * np.cos(th), 0.036 * np.sin(th) + 0.01 * np.sin(6 * t),
                     0.03 * np.cos(3 * t) - 0.015], axis=1)  # fmt: skip


def _arc(x: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1)))


def test_projection_axes_and_conventions():
    ap = _geometry(0.0, 0.0)
    centre = (np.array(ap.shape) - 1) / 2.0
    assert np.allclose(ap.project([[0.0, 0.0, 0.0]])[0], centre)
    # patient's left → image right; head → image up; 10 mm at the isocentre = 10 mm × M on the
    # detector
    left = ap.project([[0.01, 0.0, 0.0]])[0] - centre
    head = ap.project([[0.0, 0.0, 0.01]])[0] - centre
    assert left[1] == pytest.approx(0.01 * ap.magnification / PIXEL) and abs(left[0]) < 1e-9
    assert head[0] == pytest.approx(-0.01 * ap.magnification / PIXEL) and abs(head[1]) < 1e-9
    assert np.allclose(_geometry(90.0, 0.0).axes()[0], [1.0, 0.0, 0.0], atol=1e-12)  # LAO 90
    assert np.allclose(_geometry(0.0, 90.0).axes()[0], [0.0, 0.0, 1.0], atol=1e-12)  # CRA 90
    assert np.allclose(_geometry(-90.0, 0.0).source, [0.75, 0.0, 0.0])  # RAO 90: source left
    for g in (_geometry(-30.0, 20.0), _geometry(45.0, -25.0)):
        u, down, right = g.axes()
        assert np.allclose(np.array([u, down, right]) @ np.array([u, down, right]).T, np.eye(3))
    g = _geometry(-30.0, 20.0)
    assert g.label() == "RAO 30° CRA 20°" and _geometry(0.0, 0.0).label() == "AP"
    assert _geometry(40.0, -25.0).label() == "LAO 40° CAU 25°"
    assert g.separation_deg(_geometry(60.0, 20.0)) == pytest.approx(
        np.degrees(np.arccos(g.axes()[0] @ _geometry(60.0, 20.0).axes()[0]))
    )
    with pytest.raises(ValueError, match="SOD < SID"):
        ProjectionGeometry(0.0, 0.0, 0.7, 0.75, (PIXEL, PIXEL), (64, 64))


def test_rays_invert_the_projection_and_pixel_size_follows_depth():
    rng = np.random.default_rng(1)
    g = _geometry(25.0, -15.0).shifted((3.0, -2.0))
    pts = rng.normal(0.0, 0.04, (50, 3))
    pix = g.project(pts)
    rays = g.rays(pix)
    along = (pts - g.source) / np.linalg.norm(pts - g.source, axis=1, keepdims=True)
    assert np.allclose(rays, along, atol=1e-12)
    assert g.pixel_size_at([[0.0, 0.0, 0.0]])[0] == pytest.approx(g.iso_pixel_m)
    near = g.source + 0.5 * g.sod_m * g.axes()[0]  # half way to the isocentre
    assert g.pixel_size_at([near])[0] == pytest.approx(0.5 * g.iso_pixel_m)
    with pytest.raises(ValueError, match="behind"):
        g.project([g.source - g.axes()[0]])


def _dataset(**tags):
    from pydicom.dataset import Dataset

    ds = Dataset()
    for k, v in tags.items():
        setattr(ds, k, v)
    return ds


def test_read_projection_from_dicom_tags():
    full = read_projection(_dataset(
        ImagerPixelSpacing=[0.30, 0.30], DistanceSourceToDetector=1100.0,
        DistanceSourceToPatient=750.0, PositionerPrimaryAngle=-30.0,
        PositionerSecondaryAngle=20.0, Rows=512, Columns=512,
    ))  # fmt: skip
    assert full.iso_spacing_mm == pytest.approx(0.30 * 750 / 1100)
    assert full.geometry is not None and full.geometry.label() == "RAO 30° CRA 20°"
    assert full.geometry.sid_m == pytest.approx(1.1) and "isocentre" in full.note
    emf = read_projection(_dataset(ImagerPixelSpacing=[0.2, 0.2],
                                   EstimatedRadiographicMagnificationFactor=1.25))  # fmt: skip
    assert emf.iso_spacing_mm == pytest.approx(0.16) and emf.geometry is None
    detector_only = read_projection(_dataset(ImagerPixelSpacing=[0.25, 0.25]))
    assert detector_only.iso_spacing_mm is None and detector_only.detector_spacing_mm == 0.25
    assert "cannot be corrected" in detector_only.note
    assert read_projection(_dataset()).iso_spacing_mm is None


def _dicom_bytes(image: np.ndarray, primary: float, secondary: float, sid=1100.0, sod=750.0,
                 pixel=PIXEL * 1e3) -> bytes:  # fmt: skip
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, generate_uid()
    ds.Rows, ds.Columns = image.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.ImagerPixelSpacing = [pixel, pixel]
    ds.DistanceSourceToDetector, ds.DistanceSourceToPatient = sid, sod
    ds.PositionerPrimaryAngle, ds.PositionerSecondaryAngle = primary, secondary
    ds.PixelData = np.ascontiguousarray(image, np.uint8).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_loader_corrects_the_magnification():
    img = np.full((64, 64), 200, np.uint8)
    out = read_image(_dicom_bytes(img, 30.0, 0.0), "a.dcm")
    assert out.pixel_spacing_mm == pytest.approx(PIXEL * 1e3 * 750 / 1100)
    assert out.detector_spacing_mm == pytest.approx(PIXEL * 1e3)
    assert out.geometry is not None and out.geometry.shape == (64, 64)


def test_monotone_path_and_closest_points():
    cost = np.ones((4, 6))
    cost[[0, 1, 1, 2, 3], [0, 1, 2, 3, 5]] = 0.0
    cost[3, 4] = 0.0
    path, total = monotone_path(cost)
    assert path[0].tolist() == [0, 0] and path[-1].tolist() == [3, 5]
    assert np.all(np.diff(path, axis=0) >= 0) and total == 0.0
    o1, o2 = np.zeros(3), np.array([1.0, 0.0, 0.0])
    target = np.array([[0.5, 1.0, 0.2]])
    d1 = (target - o1) / np.linalg.norm(target - o1)
    d2 = (target - o2) / np.linalg.norm(target - o2)
    mid, gap = closest_points(o1, d1, o2, d2)
    assert np.allclose(mid, target) and gap[0] < 1e-12


def _views_of(curve, g1, g2):
    return [smooth_and_resample(g.project(curve), window_px=3) for g in (g1, g2)]


def test_reconstruction_recovers_a_projected_curve():
    curve = _curve()
    truth = cKDTree(curve)
    for g1, g2 in ((_geometry(-30, 20), _geometry(45, 25, sid=1.0)),
                   (_geometry(0, 30), _geometry(90, 0))):  # fmt: skip
        c1, c2 = _views_of(curve, g1, g2)
        rec = reconstruct(g1, c1, g2, c2, correct_shift=False)
        assert rec.length == pytest.approx(_arc(curve), rel=0.005)
        assert truth.query(rec.points)[0].max() < 0.2e-3
        assert rec.gap_m < 0.1e-3 and np.all(rec.residual_px < 0.5) and rec.warnings() == []
        assert rec.reliable and abs(rec.smoothing_change) < 0.01
        # a single view is shorter (foreshortening)
        assert c1.length * g1.iso_pixel_m < 0.97 * rec.length
    close = _geometry(10, 35), _geometry(30, 30)
    rec = reconstruct(close[0], *_views_of(curve, *close)[:1], close[1],
                      _views_of(curve, *close)[1], correct_shift=False)  # fmt: skip
    assert any("apart" in w for w in rec.warnings())


def test_reliability_check_warns_when_the_length_depends_on_the_smoothing():
    curve = _curve()
    g1, g2 = _geometry(-30, 20), _geometry(45, 25)
    rec = reconstruct(g1, *_views_of(curve, g1, g2)[:1], g2, _views_of(curve, g1, g2)[1])
    shaky = dataclasses.replace(rec, smoothing_change=0.04)
    assert not shaky.reliable and any("poorly determined" in w for w in shaky.warnings())
    off = dataclasses.replace(rec, residual_px=np.array([0.1, 0.5]))
    assert not off.reliable


def test_motion_between_runs_is_corrected():
    curve = _curve()
    g1, g2 = _geometry(-30, 20), _geometry(45, 25)
    c1 = _views_of(curve, g1, g2)[0]
    # the heart moved 4 mm across view 2's epipolar lines before the second run
    across = _across_epipolar(g1, g2, _views_of(curve, g1, g2)[1])
    _, down, right = g2.axes()
    moved = curve + 0.004 * (across[0] * down + across[1] * right)
    c2 = smooth_and_resample(g2.project(moved), window_px=3)
    off = reconstruct(g1, c1, g2, c2, correct_shift=False)
    assert abs(off.length / _arc(curve) - 1.0) > 0.02 and off.warnings()
    shift, _ = estimate_shift(g1, c1, g2, c2)
    assert np.linalg.norm(shift) > 3.0
    rec = reconstruct(g1, c1, g2, c2)
    assert rec.length == pytest.approx(_arc(curve), rel=0.01) and rec.warnings() == []


def test_two_view_profile_uses_the_magnification_at_the_3d_point():
    curve = _curve()
    g1, g2 = _geometry(-30, 20), _geometry(45, 25)
    c1, c2 = _views_of(curve, g1, g2)
    rec = reconstruct(g1, c1, g2, c2, correct_shift=False)
    r_true = 1.5e-3
    # each view sees a 2 r wide vessel (px at the local magnification); view 2 sees an ellipse
    d_px = []
    for g, c in ((g1, c1), (g2, c2)):
        pts = curve[cKDTree(g.project(curve)).query(c.points)[1]]
        d_px.append(2 * r_true / g.pixel_size_at(pts))
    d_px[1] = d_px[1] * 0.64
    prof = two_view_profile(rec, tuple(d_px), (c1.s, c2.s))
    assert np.median(prof.profile.r) == pytest.approx(r_true * 0.8, rel=0.01)  # sqrt(1 · 0.64)
    assert prof.profile.length == pytest.approx(rec.length - 4 * r_true * 0.8, rel=0.02)
    assert np.allclose(np.median(prof.d_view_m, axis=0), [2 * r_true, 2 * r_true * 0.64],
                       rtol=0.01)  # fmt: skip


SMALL = TwoViewSynthConfig(n_cases=1, n_stress=1, sampling={"length_mm": [40.0, 50.0]})


def test_synthetic_case_and_views():
    case = sample_case(7, SMALL)
    again = sample_case(7, SMALL)
    assert np.allclose(case.curves[0], again.curves[0])
    for curve, branch in zip(case.curves, case.tree.tree.branches, strict=True):
        assert _arc(curve) == pytest.approx(float(branch.s[-1]), rel=0.01)
    assert SMALL.separation_deg[0] <= case.views[0].separation_deg(case.views[1])
    radii = [case.radii(k)[1] for k in range(len(case.curves))]
    assert not any(main_overlaps(case.curves, radii, g, SMALL.overlap_gap_px) for g in case.views)
    view = render_view(case, 0, SMALL)
    assert view.image.dtype == np.uint8 and view.image.shape == (512, 512)
    assert view.prob.min() >= 0.0 and view.prob.max() <= 1.0
    end = case.views[0].project(case.curves[0][[-1]])[0]
    assert np.hypot(*(np.array(view.end) - end)) < 6 * SMALL.click_sd_px
    assert view.prob[int(round(view.start[0])), int(round(view.start[1]))] > 0.5
    with pytest.raises(ValueError, match="no layout fits"):
        sample_case(
            1, SMALL.model_copy(update={"image_size": 64, "layout_tries": 2, "max_trees": 1})
        )


def test_analyse_two_views_on_a_synthetic_case():
    case = sample_case(11, SMALL)
    views = [render_view(case, k, SMALL) for k in range(2)]
    result = analyse_two_views(
        tuple(v.prob for v in views), tuple(v.start for v in views),
        tuple(v.end for v in views), case.views, params=PhysicsParams(),
        images=tuple(v.image for v in views),
    )  # fmt: skip
    assert result.reconstruction.length == pytest.approx(case.main_length, rel=0.05)
    assert 0.0 < result.ffr_distal_solver <= 1.0 and result.surrogate_ffr is None
    longest = result.reconstruction.length * 1e3 * 1.02
    assert max(result.single_view_lengths_mm) < longest


def test_cli_two_views(dark_seg_dir, tiny_export, tmp_path):
    case = sample_case(11, SMALL)
    paths, points = [], []
    for k, g in enumerate(case.views):
        prob = render_view(case, k, SMALL).prob
        image = ((1.0 - 0.6 * prob) * 255).astype(np.uint8)  # dark vessel, bright background
        path = tmp_path / f"view{k + 1}.dcm"
        path.write_bytes(_dicom_bytes(image, g.primary_deg, g.secondary_deg, g.sid_m * 1e3,
                                      g.sod_m * 1e3, g.pixel_m[0] * 1e3))  # fmt: skip
        paths.append(path)
        main = case.curves[0]  # landmarks 5% inside the ends (the stand-in segmenter is coarse)
        ends = np.round(g.project(main[[len(main) // 20, -len(main) // 20]])).astype(int)
        points.append([str(v) for v in ends.ravel()])
    args = ["analyse", str(paths[0]), "--view2", str(paths[1]),
            "--start", *points[0][:2], "--end", *points[0][2:],
            "--start2", *points[1][:2], "--end2", *points[1][2:],
            "--seg-model", str(dark_seg_dir), "--surrogate-model", tiny_export["out_dir"],
            "--figure", str(tmp_path / "two.png")]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "3D path length" in result.output and "Distal FFR: solver" in result.output
    assert "Research use only" in result.output and (tmp_path / "two.png").exists()
    missing = CliRunner().invoke(app, args[:4] + args[-6:-2])
    assert missing.exit_code == 2 and "needs --start" in missing.output


def test_report_on_one_case(tmp_path):
    from vstent.imaging.twoview_report import run_case, summarise

    rows = [run_case(5, SMALL, "static"), run_case(6, SMALL, "stress")]
    assert "failed" not in rows[0]
    for key in ("two_view", "two_view_no_correction", "motion", "motion_no_correction"):
        assert np.isfinite(rows[0][key]["length_err"]) and rows[0][key]["err3d_mm"] >= 0
    assert len(rows[0]["single"]) == 2 and isinstance(rows[1]["overlap"], bool)
    summary = summarise(rows)
    assert summary["n_static"] == 1 and summary["stress"]["n"] == 1
    assert [t["passed"] for t in summary["targets"][:2]] == [True, True]
    assert summary["check"]["flag_rate"] is not None


@pytest.mark.slow
def test_write_report(tmp_path):
    from vstent.imaging.twoview_report import write_two_view_report

    config = tmp_path / "c.yaml"
    config.write_text("n_cases: 2\nn_stress: 1\nsampling: {length_mm: [40.0, 50.0]}\n")
    report = write_two_view_report(tmp_path, config, workers=1)
    assert (tmp_path / "two_view_synthetic.md").exists() and report["n_static"] == 2
    assert (tmp_path / "figures" / "two_view_errors.png").exists()
