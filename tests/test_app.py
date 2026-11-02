import gc
import io
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from vstent.app import core
from vstent.cli import APP_SCRIPT, app
from vstent.imaging.image_io import read_image
from vstent.physics.solver import PhysicsParams
from vstent.planning.stent import Stent

PARAMS = PhysicsParams()
SLIDER_BUDGET_MS = 300.0


# --- core ------------------------------------------------------------------------------------


def _dicom_bytes(frames: int = 2, spacing=(0.25, 0.25)) -> bytes:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, generate_uid()
    ds.Rows, ds.Columns, ds.NumberOfFrames = 32, 40, frames
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    if spacing:
        ds.ImagerPixelSpacing = list(spacing)
    pixels = np.zeros((frames, 32, 40), np.uint16)
    pixels[0, :, 20:] = 4000
    ds.PixelData = pixels.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_read_image_png_jpg_dicom():
    img = np.zeros((20, 30), np.uint8)
    img[:, 15:] = 200
    for ext in (".png", ".jpg"):
        ok, enc = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        out = read_image(enc.tobytes(), f"x{ext}")
        assert out.image.shape == (20, 30) and out.image.dtype == np.uint8
        assert out.pixel_spacing_mm is None and len(out.sha256) == 64
    dcm = read_image(_dicom_bytes(), "angio.dcm")
    assert dcm.image.shape == (32, 40) and dcm.image[:, 30].mean() > dcm.image[:, 5].mean()
    # detector spacing without SID / SOD: kept for the record, not offered as a calibration
    assert dcm.pixel_spacing_mm is None and dcm.detector_spacing_mm == pytest.approx(0.25)
    assert "magnification" in dcm.note and "Frame 0 of 2" in dcm.note
    second = read_image(_dicom_bytes(), "angio.dcm", frame=1)  # frame 1 is uniform
    assert second.image.max() == second.image.min() and "Frame 1 of 2" in second.note
    with pytest.raises(ValueError, match="frame 2 requested, the image has 2"):
        read_image(_dicom_bytes(), "angio.dcm", frame=2)
    with pytest.raises(ValueError, match="frame 1 requested, the image has 1"):
        read_image(enc.tobytes(), "x.jpg", frame=1)
    assert read_image(_dicom_bytes(1, None), "a.dcm").pixel_spacing_mm is None
    with pytest.raises(ValueError, match="not a readable"):
        read_image(b"not an image", "x.png")


def test_presets_and_fixed_r_micro_after_stent(tiny_export):
    runner = core.SurrogateRunner.from_dir(tiny_export["out_dir"])
    for k, preset in enumerate(core.PRESETS):
        case = core.synthetic_case(preset, 5, PARAMS)
        assert case.profile.n == 256 and case.description
        pre = core.evaluate(case.profile, PARAMS, case.pa, case.pv, runner=runner)
        assert pre.surrogate.shape == (256,) and 0 < pre.distal_solver <= 1
        if k == 0:
            st_ = [Stent(0.02, 0.018, 3e-3)]
            stented, post = core.stent_update(
                case.profile, st_, pre, PARAMS, case.pa, case.pv, runner
            )
            assert post.r_micro == pre.r_micro and post.distal_solver > pre.distal_solver
            assert post.q > pre.q and np.all(stented.r >= case.profile.r - 1e-15)
    runner.close()


def test_slider_update_is_fast(tiny_export):
    runner = core.SurrogateRunner.from_dir(tiny_export["out_dir"])
    case = core.synthetic_case(core.PRESETS[1], 0, PARAMS)
    pre = core.evaluate(case.profile, PARAMS, case.pa, case.pv, runner=runner)
    times = []
    for k in range(10):
        t0 = time.perf_counter()
        stents = [Stent(0.005 + 0.002 * k, 0.018, 3e-3)]
        stented, post = core.stent_update(
            case.profile, stents, pre, PARAMS, case.pa, case.pv, runner
        )
        core.profile_figure(case.profile, stented, stents)
        core.ffr_figure(pre, post)
        times.append((time.perf_counter() - t0) * 1e3)
    runner.close()
    assert np.median(times) < SLIDER_BUDGET_MS


def test_overlay_and_report_without_pixels(tiny_export):
    runner = core.SurrogateRunner.from_dir(tiny_export["out_dir"])
    case = core.synthetic_case(core.PRESETS[0], 0, PARAMS)
    pre = core.evaluate(case.profile, PARAMS, case.pa, case.pv, runner=runner)
    stents = [Stent(0.02, 0.018, 3e-3)]
    _, post = core.stent_update(case.profile, stents, pre, PARAMS, case.pa, case.pv, runner)
    runner.close()
    image = core.ImageInput(np.zeros((64, 64), np.uint8), "img.png", "ab" * 32)
    state = core.CaseState(
        "synthetic",
        PARAMS,
        case.pa,
        case.pv,
        case.profile,
        pre,
        stents,
        post,
        image=image,
        points=[(1, 2), (60, 61)],
    )
    models = core.Models(None, None, Path(tiny_export["out_dir"]), None)
    rep = json.loads(core.report_json(state, models))
    assert rep["disclaimer"].startswith("Research use only")
    assert rep["stents"][0]["length_mm"] == 18.0 and rep["post_pci"]["ffr_distal_solver"] > 0.9
    assert rep["models"]["surrogate"]["onnx_sha256"] == tiny_export["onnx_sha256"]
    assert "flow_model" in rep["assumptions"] and rep["input"]["image"]["sha256"] == "ab" * 32
    assert len(json.dumps(rep)) < 20_000  # numbers only, no image data
    assert core.report_png(state)[:8] == b"\x89PNG\r\n\x1a\n"
    ov = core.overlay_image(np.zeros((64, 64), np.uint8), np.ones((64, 64)), None, [], [(5, 5)])
    assert ov.shape == (64, 64, 3) and ov.dtype == np.uint8


# --- AppTest smoke tests ---------------------------------------------------------------------


@pytest.fixture
def app_env(monkeypatch, tiny_export, dark_seg_dir, tiny_arcade):
    monkeypatch.setenv(core.ENV_SURROGATE, tiny_export["out_dir"])
    monkeypatch.setenv(core.ENV_SEG, str(dark_seg_dir))
    monkeypatch.setenv(core.ENV_ARCADE, str(tiny_arcade))
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _server_ms(at: AppTest) -> float:
    text = [c.value for c in at.caption if c.value.startswith("Server time")][-1]
    return float(re.search(r"(\d+) ms", text).group(1))


def _slide_ms(at: AppTest, key: str, value: float) -> float:
    """Move a slider; return the app's server time for that update (ms). Garbage is collected
    first: with the whole test suite loaded (torch etc., which the app never imports) a full
    collection takes ~250 ms and can land in any timed update. `vstent app-latency` measures
    the real app process, collections included."""
    gc.collect()
    at.slider(key=key).set_value(value).run()
    assert not at.exception
    return _server_ms(at)


def test_app_synthetic_mode_end_to_end(app_env):
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    assert not at.exception
    assert any("Research use only" in e.value for e in at.error)
    assert [m.label for m in at.metric][:3] == ["Path length", "MLD", "Max DS"]
    at.toggle(key="place").set_value(True).run()
    assert not at.exception and "after stent" in at.metric[3].delta
    for start in (10.0, 20.0, 25.0):
        assert _slide_ms(at, "stent_start", start) < SLIDER_BUDGET_MS
    at.button(key="suggest").click().run()
    assert not at.exception and len(at.dataframe) == 1
    at.radio(key="show").set_value("Plan 1").run()
    assert not at.exception
    at.button(key="prepare").click().run()
    assert not at.exception
    at.selectbox(key="preset").set_value(core.PRESETS[2]).run()
    at.number_input(key="seed").set_value(7).run()
    assert not at.exception
    assert any("from the synthetic sampler" in c.value for c in at.sidebar.caption)


def test_app_image_mode_with_example(app_env):
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    at.radio(key="mode").set_value("Angiogram image").run()
    assert not at.exception and any("Upload" in i.value for i in at.info)
    at.radio(key="source").set_value("ARCADE example").run()
    assert not at.exception and any("start and an end point" in i.value for i in at.info)
    at.button(key="auto").click().run()
    assert not at.exception
    assert any(m.label == "Distal FFR (solver)" for m in at.metric)
    assert any("Pixel size not calibrated" in w.value for w in at.warning)
    at.selectbox(key="lumen").set_value("Imaging-chain model").run()
    assert not at.exception
    assert any(m.label == "Distal FFR (solver)" for m in at.metric)
    at.toggle(key="place").set_value(True).run()
    assert not at.exception
    at.button(key="reset").click().run()
    assert not at.exception and any("start and an end point" in i.value for i in at.info)


def test_plans_and_stents_belong_to_one_geometry_and_pressure(app_env):
    """Plans are dropped when Pv changes; the stent controls reset when the pixel size changes
    (a new geometry: old stents could lie outside the shorter vessel)."""
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    at.button(key="suggest").click().run()
    assert not at.exception and at.dataframe  # plans for the focal preset
    pv = next(n for n in at.number_input if n.label.startswith("Venous pressure"))
    pv.set_value(8.0).run()
    assert not at.exception and not at.dataframe

    at.radio(key="mode").set_value("Angiogram image").run()
    at.radio(key="source").set_value("ARCADE example").run()
    at.button(key="auto").click().run()
    at.toggle(key="place").set_value(True).run()
    at.radio(key="calib").set_value("Enter pixel size").run()
    at.number_input(key="mm_px").set_value(0.10).run()
    assert not at.exception and at.toggle(key="place").value is False


def test_pressures_outside_training_range_are_flagged_in_both_modes(app_env):
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    pa = next(n for n in at.number_input if n.label.startswith("Aortic pressure"))
    pa.set_value(130.0).run()
    assert not at.exception
    assert any("aortic pressure 130.0 mmHg" in w.value for w in at.warning)
    at.radio(key="mode").set_value("Angiogram image").run()
    at.radio(key="source").set_value("ARCADE example").run()
    at.button(key="auto").click().run()
    assert not at.exception
    assert any("aortic pressure 130.0 mmHg" in w.value for w in at.warning)


def test_cli_app_launches_streamlit_on_localhost(monkeypatch):
    import subprocess

    calls = []
    monkeypatch.setattr(subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    result = CliRunner().invoke(app, ["app", "--headless", "--port", "8600"])
    assert result.exit_code == 0 and "Research use only" in result.output
    cmd = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"] and Path(cmd[4]) == APP_SCRIPT
    assert cmd[cmd.index("--server.address") + 1] == "localhost"
    assert cmd[cmd.index("--browser.gatherUsageStats") + 1] == "false"
    assert "--server.headless" in cmd and "8600" in cmd


def test_new_vessel_resets_stent_controls(app_env):
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    at.toggle(key="place").set_value(True).run()
    at.slider(key="stent_start").set_value(30.0).run()
    at.radio(key="mode").set_value("Angiogram image").run()
    at.radio(key="source").set_value("ARCADE example").run()
    at.button(key="auto").click().run()
    assert not at.exception
    assert at.toggle(key="place").value is False
    assert not any(m.delta for m in at.metric)  # no stent carried over from the synthetic case


def test_oversized_manual_stent_is_flagged():
    case = core.synthetic_case(core.PRESETS[0], 0, PARAMS)
    ok = core.stent_warnings(case.profile, [Stent(0.02, 0.018, 3.25e-3)], 1.1)
    assert ok == []
    big = core.stent_warnings(case.profile, [Stent(0.02, 0.018, 4.0e-3)], 1.1)
    assert len(big) == 1 and "1.3×" in big[0] and "1.1×" in big[0]


# --- uncertainty (P1) ------------------------------------------------------------------------


def test_core_intervals_flags_and_report(tiny_ensemble_export):
    from vstent.planning.optimizer import PlannerConfig, StentCatalog, plan_stents

    runner = core.SurrogateRunner.from_dir(tiny_ensemble_export["out_dir"])
    case = core.synthetic_case(core.PRESETS[0], 0, PARAMS)
    pre = core.evaluate(case.profile, PARAMS, case.pa, case.pv, runner=runner)
    assert pre.model_interval is not None and pre.surrogate_std.shape == (256,)
    assert pre.model_interval.low <= pre.distal_surrogate <= pre.model_interval.high
    from vstent.uncertainty.greyzone import load_uncertainty, plan_assessments
    from vstent.uncertainty.inputs import Measurement

    settings = load_uncertainty()
    mc = core.input_uncertainty(case.profile, PARAMS, case.pa, case.pv, Measurement(), settings)
    assert mc.sources == ["v_hyp", "pa_pv"]
    a_pre = core.assess(pre, mc.pre_samples, 0.80, 0.90)
    stents = [Stent(0.02, 0.018, 3e-3)]
    _, post = core.stent_update(case.profile, stents, pre, PARAMS, case.pa, case.pv, runner)
    a_post = core.assess(post, mc.post_samples(stents), 0.90, 0.90)
    assert a_post.inputs.low > a_pre.inputs.high  # the stent clearly helps in every draw
    plans = plan_stents(
        case.profile,
        catalog=StentCatalog(diameters_mm=[2.75, 3.0], lengths_mm=[12, 18, 22]),
        config=PlannerConfig(n_plans=2),
        params=PARAMS,
        runner=runner,
        pa=case.pa,
        pv=case.pv,
    )
    assert all(p.model_interval is not None and p.ffr_std >= 0 for p in plans.plans)
    assert all(p.model_interval.level == runner.conformal.level for p in plans.plans)
    # the Pareto front carries the ensemble's FFRs too, so plan table and front agree
    assert plans.pareto and all(p.ffr_std is not None for p in plans.pareto)
    state = core.CaseState(
        "synthetic", PARAMS, case.pa, case.pv, case.profile, pre, stents, post, plans=plans,
        uncertainty=mc, pre_assessment=a_pre, post_assessment=a_post,
        plan_assessments=plan_assessments(plans, mc, 0.90),
    )  # fmt: skip
    models = core.Models(runner, None, Path(tiny_ensemble_export["out_dir"]), None)
    rep = json.loads(core.report_json(state, models))
    u = rep["uncertainty"]
    assert u["input_assumptions_varied"] == ["hyperemic velocity", "Pa / Pv"]
    assert u["pre_pci"]["threshold"] == 0.80 and len(u["pre_pci"]["model_interval"]) == 2
    assert u["post_pci"]["threshold"] == 0.90 and "Neither covers" in u["note"]
    assert all("grey_zone" in p["uncertainty"] for p in rep["plans"]["top"])
    fig = core.ffr_figure(pre, post, (a_pre, a_post))
    assert sum("inputs" in (t.name or "") for t in fig.data) == 2
    runner.close()


@pytest.fixture
def app_env_ensemble(monkeypatch, tiny_ensemble_export, dark_seg_dir, tiny_arcade):
    monkeypatch.setenv(core.ENV_SURROGATE, tiny_ensemble_export["out_dir"])
    monkeypatch.setenv(core.ENV_SEG, str(dark_seg_dir))
    monkeypatch.setenv(core.ENV_ARCADE, str(tiny_arcade))
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_app_shows_intervals_and_grey_zone(app_env_ensemble):
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Pre-PCI distal FFR, 90% intervals: inputs" in text and "model" in text
    assert any("Input assumptions varied" in c.value for c in at.caption)
    at.toggle(key="place").set_value(True).run()
    assert "Post-PCI distal FFR, 90% intervals" in " ".join(m.value for m in at.markdown)
    for start in (10.0, 20.0):
        assert _slide_ms(at, "stent_start", start) < SLIDER_BUDGET_MS
    at.button(key="suggest").click().run()
    assert not at.exception
    columns = list(at.dataframe[0].value.columns)
    assert {"Model 90%", "Inputs 90%", "Grey zone"} <= set(columns)
    at.radio(key="mode").set_value("Angiogram image").run()
    at.radio(key="source").set_value("ARCADE example").run()
    at.button(key="auto").click().run()
    assert not at.exception
    caption = " ".join(c.value for c in at.caption)
    assert "pixel size" in caption  # QCA error too once reports/qca_synthetic.json has it


def test_random_preset_uses_its_own_velocity(app_env):
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    at.selectbox(key="preset").set_value(core.PRESETS[2]).run()
    at.number_input(key="seed").set_value(7).run()
    assert not at.exception
    case = core.synthetic_case(core.PRESETS[2], 7, PARAMS)
    params = PARAMS.model_copy(update={"v_hyp": case.v_hyp})
    expected = core.evaluate(case.profile, params, case.pa, case.pv).distal_solver
    shown = next(m.value for m in at.metric if m.label == "Distal FFR (solver)")
    assert shown == f"{expected:.3f}"


def test_app_tree_mode(app_env, tiny_tree_export, monkeypatch):
    from vstent.app import tree_view

    monkeypatch.setenv(tree_view.ENV_TREE_SURROGATE, tiny_tree_export["out_dir"])
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    at.radio(key="mode").set_value("Angiogram image").run()
    at.radio(key="source").set_value("ARCADE example").run()
    at.toggle(key="tree_mode").set_value(True).run()
    assert not at.exception and any("vessel tree" in i.value for i in at.info)
    at.button(key="auto").click().run()
    assert not at.exception
    assert any(h.value == "Vessel tree" for h in at.subheader)
    assert any(m.label == "Main path as a single vessel" for m in at.metric)
    assert "Outlet FFR (tree surrogate)" in at.dataframe[0].value.columns
    at.toggle(key="tree_place").set_value(True).run()
    # the fake image's path (~5 mm) is shorter than any catalogue stent: no stent, slider off
    assert not at.exception and at.slider(key="tree_start").disabled
    at.button(key="tree_suggest").click().run()
    assert not at.exception
    at.toggle(key="tree_mode").set_value(False).run()  # back to the single path
    assert not at.exception and any(m.label == "Distal FFR (solver)" for m in at.metric)


def test_tree_view_pieces():
    import json as _json

    from tests.test_qca_tree import MM_PER_PX, y_vessel
    from vstent.app import tree_view
    from vstent.inference.pipeline import analyse_tree
    from vstent.planning.stent import Stent

    img = y_vessel()
    res = analyse_tree(img, (100, 12), (100, 388), params=PARAMS, mm_per_px=MM_PER_PX)
    stent = Stent(0.06, 0.012, 3e-3)
    overlay = tree_view.tree_overlay((img * 255).astype(np.uint8), img, res.qca, 0, [stent])
    assert overlay.shape == (*img.shape, 3)
    fig = tree_view.tree_ffr_figure(res, 1)
    assert sum("pre-PCI" in (t.name or "") for t in fig.data) == res.qca.tree.n_branches
    rows = tree_view.branch_table(res)
    assert [r["Branch"] for r in rows] == ["main", "side 1"]
    rep = _json.loads(tree_view.tree_report_json(res, 0, [stent], None, None, "y.png"))
    assert rep["disclaimer"].startswith("Research use only") and len(rep["branches"]) == 2
    assert rep["input"]["lumen_edges"] == "outline of the vessel mask"


def test_cine_and_flow_captions():
    from vstent.imaging.cardiac import EndDiastole
    from vstent.imaging.cine import ContrastTransit, opacification
    from vstent.imaging.flow import FlowVelocity
    from vstent.imaging.image_io import RunInput
    from vstent.imaging.run_analysis import RunFrames

    area = np.array([0, 0, 10, 60, 90, 100, 95, 85, 40, 5], float)
    op = opacification(area)
    ed = EndDiastole((5,), "images", None, "Image-based: frames [5].")
    run = RunInput(np.zeros((10, 4, 4), np.uint8), "run", "x")
    rf = RunFrames(run, area, area, [], op, ed, 5, "End-diastolic frame (images) in the window.")
    text = core.cine_caption(10, rf)
    assert "Cine run of 10 frames" in text and "Frame 5 (0-based) is analysed" in text
    fig = core.opacification_figure(rf, 5)
    assert [d.name for d in fig.data] == ["Vessel area (smoothed)", "End-diastolic (images)"]
    assert list(fig.data[1].x) == [5]
    assert "Image-based" in text
    nan = float("nan")
    unmeasured = FlowVelocity(nan, (nan, nan), False, True, ("no frame time in the run",))
    tr = ContrastTransit(np.arange(3.0), np.arange(3.0), 0.0, 20.0, (18.0, 22.0), 30)
    cap = core.flow_caption(unmeasured, tr)
    assert "20.0 px/frame (95%: 18.0–22.0)" in cap and "no frame time" in cap
    fv = FlowVelocity(0.2, (0.124, 0.38), False, True, ())
    cap = core.flow_caption(fv)
    assert "0.20 m/s" in cap and "FAVOR" in cap and "fixed velocity stays the default" in cap
    d = core.flow_dict(fv, used=False, ed=ed)
    assert d["accepted"] and not d["used_for_ffr"] and d["end_diastole"]["source"] == "images"
    assert d["hyperemic_velocity_m_s"] == pytest.approx(0.10 + 1.55 * 0.2 - 0.93 * 0.04, abs=1e-4)
    assert core.flow_dict(unmeasured, used=False)["contrast_flow_velocity_m_s"] is None


def test_app_two_view_example(app_env):
    """P5c: the synthetic two-view example end to end: both views, the 3D centreline, FFR on
    the 3D length, a stent drawn in both views, plans and the case report."""
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=300).run()
    at.radio(key="mode").set_value("Angiogram image").run()
    at.radio(key="source").set_value("Two-view example (synthetic)").run()
    assert not at.exception and any("both views" in i.value for i in at.info)
    at.button(key="marks").click().run()
    assert not at.exception
    assert any(m.label == "3D path length" for m in at.metric)
    assert any(c.value.startswith("Two views:") for c in at.caption)
    assert not any("one view" in c.value for c in at.caption)  # lengths are 3D
    at.toggle(key="place").set_value(True).run()
    assert not at.exception
    at.button(key="suggest").click().run()
    assert not at.exception and at.dataframe


def test_two_view_overlay_and_report_fields():
    example = core.two_view_example()
    (a1, b1), (a2, b2) = example.landmarks
    g1, g2 = (v.geometry for v in example.views)
    for point, truth in ((a1, a2), (b1, b2)):  # the true view-2 landmark lies on the guide
        (r0, c0), (r1, c1) = core.epipolar_guide(g1, point, g2)
        dist = abs((r1 - r0) * (truth[1] - c0) - (c1 - c0) * (truth[0] - r0)) / np.hypot(
            r1 - r0, c1 - c0
        )
        assert dist < 1.5
    from vstent.inference.pipeline import analyse_two_views

    two = analyse_two_views(example.probs, (a1, a2), (b1, b2), (g1, g2), params=PARAMS,
                            images=tuple(v.image for v in example.views))  # fmt: skip
    img = core.two_view_overlay(example.views[1].image, example.probs[1], 1, two,
                                [Stent(0.01, 0.012, 3e-3)], [a2, b2],
                                [core.epipolar_guide(g1, a1, g2)])  # fmt: skip
    assert img.shape == (*example.views[1].image.shape, 3)
    pa, pv = two.solver.pa, two.solver.pv
    pre = core.evaluate(two.profile, PARAMS, pa, pv)
    state = core.CaseState("image", PARAMS, pa, pv, two.profile, pre,
                           image=example.views[0], points=[a1, b1], two_view=two,
                           image2=example.views[1], points2=[a2, b2])  # fmt: skip
    report = core.case_report(state, core.Models(None, None, None, None))
    assert report["two_view"]["length_3d_mm"] > max(report["two_view"]["single_view_lengths_mm"])
    assert report["input"]["image2"]["sha256"] == example.views[1].sha256
    assert "pixels" not in json.dumps(report)
