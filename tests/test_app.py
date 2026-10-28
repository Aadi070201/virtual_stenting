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
        out = core.read_image(enc.tobytes(), f"x{ext}")
        assert out.image.shape == (20, 30) and out.image.dtype == np.uint8
        assert out.pixel_spacing_mm is None and len(out.sha256) == 64
    dcm = core.read_image(_dicom_bytes(), "angio.dcm")
    assert dcm.image.shape == (32, 40) and dcm.image[:, 30].mean() > dcm.image[:, 5].mean()
    assert dcm.pixel_spacing_mm == pytest.approx(0.25)
    assert "magnification" in dcm.note and "First of 2 frames" in dcm.note
    assert core.read_image(_dicom_bytes(1, None), "a.dcm").pixel_spacing_mm is None
    with pytest.raises(ValueError, match="not a readable"):
        core.read_image(b"not an image", "x.png")


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


def test_app_synthetic_mode_end_to_end(app_env):
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    assert not at.exception
    assert any("Research use only" in e.value for e in at.error)
    assert [m.label for m in at.metric][:3] == ["Path length", "MLD", "Max DS"]
    at.toggle(key="place").set_value(True).run()
    assert not at.exception and "after stent" in at.metric[3].delta
    for start in (10.0, 20.0, 25.0):
        at.slider(key="stent_start").set_value(start).run()
        assert not at.exception
        assert _server_ms(at) < SLIDER_BUDGET_MS
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
    at.toggle(key="place").set_value(True).run()
    assert not at.exception
    at.button(key="reset").click().run()
    assert not at.exception and any("start and an end point" in i.value for i in at.info)


def test_cli_app_launches_streamlit_on_localhost(monkeypatch):
    import subprocess

    calls = []
    monkeypatch.setattr(subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    result = CliRunner().invoke(app, ["app", "--headless", "--port", "8600"])
    assert result.exit_code == 0 and "Research use only" in result.output
    cmd = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"] and Path(cmd[4]) == APP_SCRIPT
    assert cmd[cmd.index("--server.address") + 1] == "localhost"
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
