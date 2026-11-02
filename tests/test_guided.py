"""The guided app (``vstent app``): main-trunk path, plain-language helpers, the whole flow."""

from types import SimpleNamespace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from vstent.app import core, guided
from vstent.cli import GUIDED_SCRIPT
from vstent.imaging.centerline import main_trunk_endpoints
from vstent.inference import consensus as pc
from vstent.planning.stent import Stent


def test_main_trunk_takes_the_wider_branch_and_stops():
    mask = np.zeros((120, 260), bool)
    mask[54:67, 10:250] = True  # the main vessel, 13 px wide, wider at the left end
    mask[50:71, 10:40] = True
    for k in range(60):  # a thin side branch leaving at column 120, going down-right
        mask[66 + k : 66 + k + 1, 120 + k : 125 + k] = True
    a, b = main_trunk_endpoints(mask)
    assert a[1] < 30 and b[1] > 230 and 55 <= b[0] <= 66  # along the main vessel to its end
    a2, b2 = main_trunk_endpoints(mask, max_length_px=100)
    assert a2 == a and 100 <= b2[1] - a2[1] <= 115  # stopped after ~100 px


def _cons(ffrs, errors=()):
    frames = [pc.FrameResult(10 * k, (0, 0), (1, 1), None, pc.Values(v, 50.0, 30.0))
              for k, v in enumerate(ffrs)]  # fmt: skip
    frames += [pc.FrameResult(99, (0, 0), (1, 1), error=e) for e in errors]
    return pc.Consensus(tuple(frames), 5, "end-diastolic frames from the ECG")


def test_plain_language_helpers():
    assert guided.verdict(0.72, False).label == "Significant"
    assert guided.verdict(0.86, False).label == "Not significant"
    assert guided.verdict(0.79, True).label == "Borderline"
    ok, text = guided.stability(_cons([0.71, 0.74, 0.73]))
    assert ok and "3 heartbeats" in text and "Consistent" in text
    bad, text = guided.stability(_cons([0.76, 0.84], errors=["the path was lost"]))
    assert not bad and "disagree" in text and "frame 99: the path was lost" in text
    one, text = guided.stability(_cons([0.8]))
    assert one and "one heartbeat only" in text
    st = Stent(0.010, 0.018, 0.003)
    assert guided.stent_text([st], 0.91) == ("3.00 × 18 mm stent from 10 to 28 mm along the "
                                             "path: virtual FFR after stenting 0.91")  # fmt: skip
    smoothed = np.zeros(60)
    smoothed[10:50] = 100.0
    rf = SimpleNamespace(opacification=SimpleNamespace(smoothed=smoothed, window=(8, 52)),
                         ed=SimpleNamespace(found=True, frames=(3, 14, 25, 36, 47, 58)),
                         analysis_frame=25)  # fmt: skip
    assert guided.frame_choices(rf) == [14, 25, 36, 47]
    assert guided.frame_choices(rf, limit=2) == [14, 25]
    img = np.full((64, 80), 128, np.uint8)
    out = guided.overlay(img, np.zeros((64, 80)), points=[(10, 10), (50, 70)])
    assert out.shape == (64, 80, 3) and out.dtype == np.uint8


@pytest.fixture
def guided_env(monkeypatch, tiny_export, dark_seg_dir):
    monkeypatch.setenv(core.ENV_SURROGATE, tiny_export["out_dir"])
    monkeypatch.setenv(core.ENV_SEG, str(dark_seg_dir))
    monkeypatch.setattr(guided, "CADICA_EXAMPLE", guided.Path("no/such/folder"))


def test_guided_app_whole_flow(guided_env):
    """Demo run → frame → automatic path → virtual FFR over the heartbeats → suggested stent →
    report, with the stand-in models."""
    at = AppTest.from_file(str(GUIDED_SCRIPT), default_timeout=300).run()
    assert not at.exception and "Research use only" in at.caption[0].value
    at.radio(key="source").set_value(guided.DEMOS["synthetic"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "ECG markers" in at.success[0].value
    headers = [m.value for m in at.markdown if m.value.startswith("### ")]
    assert headers[:3] == ["### 1 · Load a cine angiogram", "### 2 · Choose the frame",
                           "### 3 · Vessel and lesions"]  # fmt: skip
    frame_radio = next(r for r in at.radio if r.label == "Frame")
    assert any("★" in str(frame_radio.format_func(o)) for o in frame_radio.options)
    labels = [m.label for m in at.metric]
    assert "Virtual FFR at the end of the path" in labels
    assert any("heartbeat" in i.value for i in [*at.info, *at.warning])
    at.button(key="suggest").click().run()  # stand-in models: maybe no lesion to plan for
    assert not at.exception, [e.value for e in at.exception]
    # the path through the stand-in's mask: the demo's main vessel on the shown frame
    cc, _ = guided.demo_case()
    main = cc.case.curves[0]
    inner = main[[len(main) // 10, (7 * len(main)) // 10]]
    t_s = frame_radio.value / cc.runs[0].fps
    a, b = np.round(cc.case.views[0].project(cc.position(0, t_s, inner))).astype(int)
    at.session_state["start"], at.session_state["end"] = tuple(map(int, a)), tuple(map(int, b))
    at.run()
    own = next(t.key for t in at.toggle if t.key and t.key.startswith("own_"))
    at.toggle(key=own).set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Virtual FFR after the stent" in [m.label for m in at.metric]
    assert len(at.get("download_button")) == 2
