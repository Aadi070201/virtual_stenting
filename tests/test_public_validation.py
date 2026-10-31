"""P4b: public-data validation — FDA nozzle physics, CADICA loading, stenosis grading,
intensity-edge QCA."""

import io
import json
import zipfile
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from vstent.data import cadica, fda_nozzle
from vstent.imaging.centerline import smooth_and_resample, vessel_graph
from vstent.imaging.qca import measure_diameters, measure_diameters_intensity, qca_from_centerline
from vstent.imaging.synthetic import (
    render_projection,
    render_tubes,
    sine_centerline,
    stenosis_diameter,
)
from vstent.physics import nozzle
from vstent.physics.solver import PhysicsParams
from vstent.validation import stenosis_grading as sg
from vstent.validation.stats import weighted_kappa

PARAMS = PhysicsParams()


# --- FDA nozzle --------------------------------------------------------------------------------


def _run_text(lab: str, z, p, q=5.2e-6, deleted=False) -> str:
    key = (
        "deleted-plot-wall-distribution-pressure" if deleted else "plot-wall-distribution-pressure"
    )
    rows = "\n".join(f"{a:.6E}  {b:.5E}" for a, b in zip(z, p, strict=True))
    return (f'dataset-description "x"\ndataset-code {lab}\nfluid-density 1056\n'
            f"fluid-viscosity 0.0035\nfluid-volumetric-flow-rate {q:.5E}\n{key}\n{len(z)}\n"
            f"{rows}\nplot-z-distribution-pressure\n1\n0.0 0.0\n")  # fmt: skip


def _fake_nozzle_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for orientation in ("SE", "CD"):
        for re_t in fda_nozzle.REYNOLDS:
            z = np.linspace(-0.09, 0.03, 9) if orientation == "SE" else np.linspace(-0.03, 0.09, 9)
            p = np.linspace(300.0, 0.0, 9)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("experiment/a_243.txt", _run_text("243", z, p))
                zf.writestr("experiment/a_468.txt", _run_text("468", z, p + 20))
                zf.writestr("experiment/a_297.txt", _run_text("297", z, p, deleted=True))
                zf.writestr("experiment/piv_999.txt", "dataset-code 999\nfluid-density 1056\n")
            (root / fda_nozzle.archive_name(orientation, re_t)).write_bytes(buf.getvalue())
    return root


def test_nozzle_files_parse_and_report_skips(tmp_path):
    raw = _fake_nozzle_dir(tmp_path)
    runs, skipped = fda_nozzle.load_runs("SE", 500, raw)
    assert [r.lab for r in runs] == ["243", "468"] and runs[0].q == pytest.approx(5.2e-6)
    assert np.all(np.diff(runs[0].z) > 0) and runs[0].rho == 1056
    reasons = {s["file"]: s["reason"] for s in skipped}
    assert "deleted" in reasons["a_297.txt"] and "velocity data only" in reasons["piv_999.txt"]


def test_nozzle_geometry_and_model_loss():
    assert pytest.approx(0.022685, abs=1e-6) == nozzle.L_CONE  # (6 − 2) mm / tan 10°
    se = nozzle.nozzle_radius(np.array([-0.08, -0.05, -0.02, 0.0, 1e-4]), "SE")
    assert se[[0, 2, 3, 4]] == pytest.approx([0.006, 0.002, 0.002, 0.006])
    assert 0.002 < se[1] < 0.006
    cd = nozzle.nozzle_radius(np.array([-0.01, 0.0, 0.02, 0.05, 0.07]), "CD")
    assert cd[[0, 1, 2, 4]] == pytest.approx([0.006, 0.002, 0.002, 0.006])
    q, rho, mu = 5.2e-6, 1056.0, 0.0035
    se_loss = nozzle.model_loss("SE", -0.09, 0.03, q, rho, mu, PARAMS)
    cd_loss = nozzle.model_loss("CD", -0.03, 0.09, q, rho, mu, PARAMS)
    # the model depends only on areas: the mirrored nozzle loses the same
    assert se_loss["total"] == pytest.approx(cd_loss["total"], rel=1e-3)
    a_t, a_d = np.pi * 0.002**2, np.pi * 0.006**2
    expected_exp = PARAMS.kt * rho / 2 * (1 / a_t - 1 / a_d) ** 2 * q * q
    assert se_loss["expansion"] == pytest.approx(expected_exp, rel=1e-6)
    poiseuille_throat = 8 * mu * 0.04 * q / (np.pi * 0.002**4)
    assert poiseuille_throat < se_loss["viscous"] < 1.2 * poiseuille_throat  # + cone and pipes


def test_nozzle_report(tmp_path):
    raw = _fake_nozzle_dir(tmp_path / "raw")
    res = nozzle.write_nozzle_report(PARAMS, tmp_path / "r" / "physics_nozzle.md", raw)
    case = res["cases"][0]
    # median(300, 262.5, 225) − median(75, 37.5, 0) for both labs
    assert case["orientation"] == "SE" and case["measured_median"] == pytest.approx(225.0)
    md = (tmp_path / "r" / "physics_nozzle.md").read_text(encoding="utf-8")
    assert "Nothing is fitted" in md and "a_297.txt" in md
    assert (tmp_path / "r" / "figures" / "physics_nozzle.png").exists()


# --- CADICA ------------------------------------------------------------------------------------


def _fake_cadica(root: Path) -> Path:
    """Two patients, one lesion video each: a dark tube with a 60% stenosis, one expert box."""
    for k, patient in enumerate(("p1", "p2")):
        v = root / "CADICA" / "selectedVideos" / patient / "v1"
        (v / "input").mkdir(parents=True)
        (v / "groundtruth").mkdir()
        pts, s = sine_centerline((128.0, 20.0), 216.0, amplitude_px=0.0)
        d = stenosis_diameter(s, 12.0, 0.6, s[-1] / 2, 30.0)
        img = render_projection((256, 256), [(pts, d / 2)], mm_per_px=0.25, mu_per_mm=0.4,
                                background=0.6, blur_sigma=1.0, noise_std=0.01, seed=k)  # fmt: skip
        cv2.imwrite(str(v / "input" / f"{patient}_v1_00003.png"),
                    np.round(img * 255).astype(np.uint8))  # fmt: skip
        (v / "groundtruth" / f"{patient}_v1_00003.txt").write_text(
            "113 113 30 30 p50_70\n", encoding="utf-8"
        )
    return root


def test_cadica_boxes_and_fetch(tmp_path, monkeypatch):
    raw = _fake_cadica(tmp_path / "raw")
    boxes = cadica.load_lesion_boxes(raw)
    assert [(b.patient, b.frame, b.category) for b in boxes] == [("p1", 3, "p50_70"),
                                                                 ("p2", 3, "p50_70")]  # fmt: skip
    assert boxes[0].image.exists() and boxes[0].ds_range == (50.0, 70.0)
    bad = raw / "CADICA" / "selectedVideos" / "p1" / "v1" / "groundtruth" / "p1_v1_00004.txt"
    bad.write_text("1 1 2 2 severe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown lesion category"):
        cadica.load_lesion_boxes(raw)
    bad.unlink()
    # fetch: an archive already present is verified (SHA-256), then extracted
    archive = tmp_path / "dl" / cadica.ARCHIVE
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("CADICA/readme.txt", "hello")
    monkeypatch.setattr(cadica, "SIZE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(cadica, "SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        cadica.fetch_cadica(archive.parent)
    from vstent.data.arcade import file_digest

    monkeypatch.setattr(cadica, "SHA256", file_digest(archive, "sha256"))
    src = cadica.fetch_cadica(archive.parent)
    assert (archive.parent / "CADICA" / "readme.txt").exists() and src["licence"].startswith("CC")
    assert json.loads((archive.parent / "SOURCE.json").read_text())["doi"] == cadica.DOI


# --- stenosis grading --------------------------------------------------------------------------


def test_lesion_tracks_and_split():
    def box(frame, x, cat="p70_90"):
        return cadica.LesionBox("p1", "v1", frame, x, 50, 20, 20, cat, Path("x.png"))

    tracks = sg.lesion_tracks([box(1, 10), box(2, 14), box(1, 200), box(2, 205),
                               box(1, 100, "p20_50")])  # fmt: skip
    assert sorted(len(t.boxes) for t in tracks) == [1, 2, 2]
    # two lesions of one grade starting on the same keyframe keep separate keys
    assert len({t.key for t in tracks}) == len(tracks) == 3
    split = sg.patient_split([f"p{k}" for k in range(1, 11)])
    assert sorted(split.values()).count("dev") == 5 and split == sg.patient_split(list(split))


def test_path_and_stenosis_in_box():
    pts, s = sine_centerline((128.0, 20.0), 216.0, amplitude_px=0.0)
    d = stenosis_diameter(s, 12.0, 0.6, s[-1] / 2, 30.0)
    prob = render_tubes((256, 256), [(pts, d)], blur_sigma=0.0, noise_std=0.0)
    graph = vessel_graph(prob >= 0.5)
    box = (113.0, 113.0, 30.0, 30.0)
    path = sg.path_through_box(graph, box)
    assert path is not None and path[:, 1].min() < 113 - 20 and path[:, 1].max() > 143 + 20
    res = sg.stenosis_in_box(prob, graph, box)
    assert res["ds"] == pytest.approx(60.0, abs=5.0)
    assert "error" in sg.stenosis_in_box(prob, graph, (10.0, 10.0, 20.0, 20.0))
    assert sg.path_through_box(nx.Graph(), box) is None


def test_intensity_edges_recover_a_stenosis_the_segmentation_filled():
    pts, s = sine_centerline((150.0, 30.0), 340.0, amplitude_px=15.0)
    d = stenosis_diameter(s, 12.0, 0.5, s[-1] / 2, 30.0)
    img = render_projection((300, 400), [(pts, d / 2)], mm_per_px=0.25, mu_per_mm=0.06,
                            background=0.6, blur_sigma=1.0, edge_amount=2.5, edge_sigma=2.0,
                            noise_std=0.015, seed=1)  # fmt: skip
    img8 = np.round(img * 255).astype(np.uint8)
    filled = render_tubes((300, 400), [(pts, np.full_like(d, 12.0))], blur_sigma=0, noise_std=0)
    cl = smooth_and_resample(np.round(pts[::8]))

    def max_ds(q):
        return float((100 * (1 - q.profile.r / q.profile.reference_radius())).max())

    assert max_ds(qca_from_centerline(filled, cl)) < 5  # probability edges miss it
    assert max_ds(qca_from_centerline(filled, cl, image=img8)) == pytest.approx(50.0, abs=8.0)
    # no contrast at all: the guide (probability) diameters are kept
    guide = measure_diameters(filled, cl)
    flat = measure_diameters_intensity(np.full((300, 400), 128, np.uint8), cl, guide)
    assert np.allclose(flat.diameter_px, guide.diameter_px)


def test_grading_statistics():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 200), rng.integers(0, 5, 200)
    assert weighted_kappa(a, b, 5) == pytest.approx(cohen_kappa_score(a, b, weights="linear"))
    assert [sg.ordinal(v) for v in (0, 19.9, 20, 55, 75, 95)] == [0, 0, 1, 2, 3, 4]
    pairs = [(c, sg.MIDPOINT[c]) for c in sg.ORDINAL] * 8
    grades = [{"patient": f"p{k % 8}", "category": c, "ds": mid, "split": "test"}
              for k, (c, mid) in enumerate(pairs)]  # fmt: skip
    grades.append({"patient": "p0", "category": "p50_70", "ds": None, "split": "test"})
    st = sg.grading_stats(grades, n_boot=100)
    assert st["exact"] == 1.0 and st["kappa"] == pytest.approx(1.0) and st["auc_ge70"] == 1.0
    assert st["n_measured"] == 40 and st["feasibility"] == pytest.approx(40 / 41)


def test_grading_end_to_end(tmp_path, dark_seg_dir):
    from vstent.inference.onnx_runtime import SegmentationRunner

    raw = _fake_cadica(tmp_path / "raw")
    tracks = sg.lesion_tracks(cadica.load_lesion_boxes(raw))
    split = sg.patient_split([t.patient for t in tracks])
    seg = SegmentationRunner.from_dir(dark_seg_dir)
    rows = sg.measure_tracks(tracks, seg, split=split, methods=tuple(sg.EDGE_METHODS))
    assert list(rows) == list(sg.EDGE_METHODS)
    assert all(r["status"] == "ok" for rs in rows.values() for r in rs)
    for rs in rows.values():
        assert all(abs(float(r["ds"]) - 60.0) < 15.0 for r in rs)
    source = {"doi": cadica.DOI, "licence": cadica.LICENCE}
    sg.write_grading_report(rows, tmp_path / "rep" / "grading.md", source=source, n_boot=100)
    md = (tmp_path / "rep" / "grading.md").read_text(encoding="utf-8")
    assert "visual estimates" in md and "Development patients" in md
    assert (tmp_path / "rep" / "grading" / "lesions_intensity.csv").exists()
