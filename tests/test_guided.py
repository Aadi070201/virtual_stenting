"""The guided app (``vstent app``): case library, main-trunk path, plain-language helpers, the
whole flow."""

from types import SimpleNamespace

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from vstent.app import core, guided
from vstent.cli import GUIDED_SCRIPT
from vstent.data import cadica
from vstent.imaging.cardiac import EndDiastole
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
    assert guided.verdict(0.72).label == "Significant"
    assert guided.verdict(0.78).label == "Grey zone"
    assert guided.verdict(0.80).label == "Grey zone"  # 0.80 is significant (≤ 0.80)
    assert guided.verdict(0.86).label == "Not significant"
    assert [g[1] for g in guided.FFR_GUIDE][:3] == ["above 0.80", "0.76–0.80", "0.75 or below"]
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
                         ed=EndDiastole((3, 14, 25, 36, 47, 58), "images", 11.0, ""),
                         analysis_frame=25)  # fmt: skip
    assert guided.frame_label(25, rf) == "25 ★" and guided.frame_label(3, rf) == "3 faint"
    assert guided.frame_label(14, rf, all_frames=True) == "14 beat"  # no ECG: not called ED
    rf.ed = EndDiastole(rf.ed.frames, "ECG", 11.0, "")
    assert guided.frame_label(14, rf, all_frames=True) == "14 ED"
    assert guided.frame_label(25, rf, expert=frozenset({25})) == "25 ★ ◆"
    assert guided.severity("70–90%") == 70.0 and guided.severity("99%") == 99.0
    prob = np.zeros((100, 200))
    prob[45:56, 5:195] = 1.0  # a vessel, wider at the left end
    prob[42:59, 5:30] = 1.0
    start, end = guided.box_path(prob, (90, 40, 20, 20))
    assert start[1] < 30 and end[1] > 180  # proximal start, through the box, to the vessel end
    assert guided.box_path(prob, (90, 75, 10, 10)) is None  # no vessel in that box
    img = np.full((64, 80), 128, np.uint8)
    out = guided.overlay(img, np.zeros((64, 80)), points=[(10, 10), (50, 70)],
                         boxes=[(5, 5, 20, 20, "70–90%")])  # fmt: skip
    assert out.shape == (64, 80, 3) and out.dtype == np.uint8
    mark = guided.LesionMark(1, 3, 3, 60.0, 5.0, 1.2, 10.0, 8.0, 12.0)  # near the corner
    zoom = guided.lesion_zoom(img, None, mark, zoom=3, half_px=10)
    assert zoom.shape == (60, 60, 3)


def test_case_library_on_cadica(tmp_path):
    from tests.test_public_validation import _fake_cadica

    raw = _fake_cadica(tmp_path / "raw")
    for p in ("p1", "p2"):  # the selected frames file makes a run part of the library
        (raw / "CADICA" / "selectedVideos" / p / "v1" / f"{p}_v1_selectedFrames.txt").write_text(
            f"{p}_v1_00003\n", encoding="utf-8")  # fmt: skip
    lib = guided.case_library(raw, n_synthetic=2)
    assert [c.kind for c in lib] == ["synthetic", "synthetic", "cadica", "cadica"]
    real = lib[2]
    assert real.label == "CADICA p1 run 1 · worst lesion 50–70%"
    assert real.boxes == ((0, 113.0, 113.0, 30.0, 30.0, "50–70%"),)  # frame 00003 is index 0
    run = guided.load_case(real)
    assert len(run) == 1 and run.name == real.label
    assert [c.kind for c in guided.case_library(tmp_path / "none", n_synthetic=1)] == ["synthetic"]


@pytest.fixture
def guided_env(monkeypatch, tiny_export, dark_seg_dir, tmp_path):
    monkeypatch.setenv(core.ENV_SURROGATE, tiny_export["out_dir"])
    monkeypatch.setenv(core.ENV_SEG, str(dark_seg_dir))
    monkeypatch.setattr(cadica, "RAW_DIR", tmp_path / "no_cadica")  # synthetic cases only
    st.cache_resource.clear()  # the app's caches outlive one AppTest (library, models)
    st.cache_data.clear()


def test_guided_app_whole_flow(guided_env):
    """Library case → frame → automatic path → virtual FFR over the heartbeats → stent → report,
    with the stand-in models."""
    at = AppTest.from_file(str(GUIDED_SCRIPT), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Research use only" in at.caption[0].value
    assert at.selectbox(key="case").value.startswith("Synthetic 1")
    text = " ".join(m.value for m in at.markdown)
    for step in ("1 · Frame", "2 · Vessel and lesions", "3 · Virtual FFR", "0.76–0.80"):
        assert step in text
    assert "Virtual FFR (end of the path)" in text
    assert any("heartbeat" in c.value for c in at.caption)
    # the path through the stand-in's mask: the case's main vessel on the analysed frame
    cc, _ = guided.synthetic_case(guided.DEMO_SEED)
    main = cc.case.curves[0]
    inner = main[[len(main) // 10, (7 * len(main)) // 10]]
    frame = at.slider[0].value  # the frame slider (every frame of the run)
    a, b = np.round(cc.case.views[0].project(cc.position(0, frame / cc.runs[0].fps, inner)))
    at.session_state["start"], at.session_state["end"] = tuple(map(int, a)), tuple(map(int, b))
    at.run()
    at.button(key="suggest").click().run()  # stand-in models: maybe no lesion to plan for
    assert not at.exception, [e.value for e in at.exception]
    own = next(t.key for t in at.toggle if t.key and t.key.startswith("own_"))
    at.toggle(key=own).set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "After the stent" in " ".join(m.value for m in at.markdown)
    n_key = next(r.key for r in at.radio if r.key and r.key.startswith("n_"))
    at.radio(key=n_key).set_value(2).run()  # two stents, each with its own controls
    assert not at.exception, [e.value for e in at.exception]
    dias = [s.key for s in at.selectbox if s.key and s.key.startswith("dia_")]
    assert len(dias) == 2 and dias[0] != dias[1]
    assert any("S2:" in c.value for c in at.caption)
    text = " ".join(m.value for m in at.markdown)
    assert "heartbeat frame" in text or "end-diastolic frame" in text  # the frame's state
    assert "contrast" in text
    assert len(at.get("download_button")) == 2


def test_frame_states_and_no_plan_reasons():
    from types import SimpleNamespace as NS

    from vstent.physics.geometry import VesselProfile

    smoothed = np.array([0, 0, 5, 30, 80, 95, 100, 98, 92, 60, 20, 5], float)
    rf = NS(opacification=NS(smoothed=smoothed, arrival=3, window=(3, 10)),
            ed=EndDiastole((2, 6, 10), "images", 4.0, ""), analysis_frame=6)  # fmt: skip
    assert guided.contrast_state(rf, 1)[0] == "no contrast yet"
    assert guided.contrast_state(rf, 3)[0] == "filling"
    assert guided.contrast_state(rf, 6) == ("full", 100.0)
    assert guided.contrast_state(rf, 10)[0] == "washing out"
    assert len(guided.frame_chart(rf, 6).data) == 3
    s = np.linspace(0, 0.06, 61)
    healthy = VesselProfile(s, np.full_like(s, 0.0008), np.full_like(s, 0.0008))
    catalog = NS(diameters_mm=[2.25, 3.0])
    cfg = NS(max_diameter_ratio=1.1)
    assert "spread along the whole path" in guided.no_plan_reason(
        healthy, NS(lesions=[]), catalog, cfg)  # fmt: skip
    assert "smaller than the smallest stent" in guided.no_plan_reason(
        healthy, NS(lesions=[1]), catalog, cfg)  # fmt: skip
    big = VesselProfile(s, np.full_like(s, 0.0015), np.full_like(s, 0.0015))
    assert "end of the path" in guided.no_plan_reason(big, NS(lesions=[1]), catalog, cfg)


def test_hand_placed_stents():
    plan = [Stent(0.006, 0.038, 0.00275)]
    m1 = guided.LesionMark(1, 0, 0, 60.0, 5.0, 1.2, 20.0, 18.0, 22.0)  # under the plan's stent
    m2 = guided.LesionMark(2, 0, 0, 75.0, 4.0, 1.0, 80.0, 78.0, 82.0)
    assert guided.own_stent_defaults(plan, [m1, m2], 1, 120.0) == [(2.75, 38.0, 6.0)]
    two = guided.own_stent_defaults(plan, [m1, m2], 2, 120.0)
    assert two[1] == (3.0, 18.0, 71.0)  # the uncovered lesion gets the second stent
    three = guided.own_stent_defaults(plan, [m1, m2], 3, 120.0)
    assert len(three) == 3 and three[2][2] == pytest.approx(91.0)  # 2 mm past the last
    a, b = Stent(0.006, 0.038, 0.003), Stent(0.040, 0.018, 0.003)
    assert guided.overlap_notes([a, b]) == ["S1 and S2 overlap by 4 mm."]
    assert guided.overlap_notes([a, Stent(0.050, 0.018, 0.003)]) == []
    assert guided.stent_text([a, b]).startswith("S1: 3.00 × 38 mm")
    s = np.linspace(0.0, 0.1, 101)
    diffuse = SimpleNamespace(s=s, solver=1.0 - 4.0 * s, distal_solver=0.6)  # a steady decline
    assert "spread along the path" in guided.diffuse_note(diffuse, [m2])
    focal = SimpleNamespace(s=s, solver=np.where(s < 0.081, 1.0, 0.6), distal_solver=0.6)
    assert guided.diffuse_note(focal, [m2]) is None
    good = SimpleNamespace(s=s, solver=1.0 - s, distal_solver=0.9)
    assert guided.diffuse_note(good, [m2]) is None


def test_target_path_and_detection_check():
    from vstent.imaging.centerline import target_endpoints

    mask = np.zeros((120, 260), bool)
    mask[54:67, 10:250] = True  # main vessel, wider at the left
    mask[50:71, 10:40] = True
    for k in range(60):  # a thin branch leaving at column 120, going down-right
        mask[66 + k : 66 + k + 1, 120 + k : 125 + k] = True
    a, b = target_endpoints(mask, (100, 157))  # click on the branch
    assert a[1] < 30 and b[0] > 110  # from the proximal end down to the branch's end
    with pytest.raises(ValueError, match="not on the vessel"):
        target_endpoints(mask, (10, 200))
    m1 = guided.LesionMark(1, 60, 100, 70.0, 5.0, 0.8, 30.0, 28.0, 32.0)
    m2 = guided.LesionMark(2, 60, 200, 55.0, 4.0, 1.2, 60.0, 58.0, 62.0)
    checks, extra = guided.detection_check(None, [m1, m2], [(90, 50, 20, 20, "70–90%"),
                                                            (10, 10, 5, 5, "50–70%")])  # fmt: skip
    assert [c.status for c in checks] == ["found", "off the path"]
    assert checks[0].letter == "A" and checks[0].lesion == 1 and extra == [m2]


def test_go_to_an_expert_frame(guided_env, monkeypatch):
    """The jump to the experts' frames must not touch the slider after it is drawn."""
    case = guided.Case("synthetic:9500", "Synthetic with boxes", "synthetic", seed=9500,
                       boxes=((5, 100.0, 100.0, 30.0, 30.0, "70–90%"),))  # fmt: skip
    monkeypatch.setattr(guided, "case_library", lambda *a, **k: [case])
    at = AppTest.from_file(str(GUIDED_SCRIPT), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="to_expert").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.slider[0].value == 5


def test_contours_lesion_table_and_loss_chart():
    """The QFR-style display: lumen contours, lesions filled between them, a lesion table and
    the FFR lost per mm under the FFR curve (P4k-D)."""
    from vstent.imaging.qca import run_qca
    from vstent.physics.solver import PhysicsParams

    prob = np.zeros((120, 400))
    cols = np.arange(400)
    half = np.where(np.abs(cols - 200) < 12, 2.5, 6.0)  # a vessel 12 px wide, 5 px at a lesion
    for c in range(20, 380):
        prob[int(60 - half[c]) : int(60 + half[c]) + 1, c] = 1.0
    q = run_qca(prob, (60, 25), (60, 375), mm_per_px=0.25)
    marks = guided.lesion_marks(q)
    assert len(marks) == 1 and 180 <= marks[0].col <= 220
    img = np.full(prob.shape, 120, np.uint8)
    out = guided.overlay(img, prob, q, marks, points=[(60, 25), (60, 375)])
    plain = guided.overlay(img, prob, q, marks, contours=False)
    assert out.shape == (120, 400, 3) and not np.array_equal(out, plain)
    assert out[60, marks[0].col, 0] > out[60, marks[0].col, 2]  # filled red at the lesion
    p = PhysicsParams()
    pre = core.evaluate(q.profile, p, p.pa, p.pv)
    rows = guided.lesion_table(q, pre, marks)
    assert rows[0]["Lesion"] == "1" and float(rows[0]["FFR drop"]) > 0.0
    assert rows[0]["Reference"].endswith(" mm") and rows[0]["Stenosis"].endswith("%")
    drop = guided.ffr_drop_per_mm(pre)
    s_mm = pre.s * 1e3
    assert drop.shape == pre.solver.shape and drop.min() >= 0.0
    assert abs(s_mm[np.argmax(drop)] - marks[0].at_mm) < 8.0  # the loss peaks at the lesion
    assert len(guided.ffr_chart(pre, None, marks).data) == 2  # FFR and loss per mm


def test_path_starts_at_the_left_main_not_a_branch_tip():
    """A short, wide left main splitting into two long branches: the longest path runs between
    the branch tips, but the path must start at the left main (the widest end)."""
    mask = np.zeros((200, 300), bool)
    mask[90:111, 5:40] = True  # left main: 21 px wide, 35 px long
    for k in range(200):  # LAD: 9 px wide, down-right
        mask[100 + k // 2 - 4 : 100 + k // 2 + 5, 40 + k] = True
    for k in range(150):  # a long branch: 7 px wide, up-right
        mask[100 - k // 2 - 3 : 100 - k // 2 + 4, 40 + k] = True
    a, b = main_trunk_endpoints(mask)
    assert a[1] < 20 and 90 <= a[0] <= 111  # the left main
    assert b[1] > 200 and b[0] > 150  # down the LAD
    from vstent.imaging.centerline import target_endpoints

    a2, b2 = target_endpoints(mask, (40, 160))  # click on the branch
    assert a2 == a and b2[0] < 40


def test_faint_vessels_are_outlined_not_measured():
    img = np.full((80, 120), 150, np.uint8)
    img[38:43, 10:110] = 60  # a dark vessel
    img[20:60, 70:73] = 70  # a fainter crossing branch
    prob = np.zeros(img.shape)
    prob[38:43, 10:110] = 1.0  # the model found only the first
    faint = guided.faint_vessels(img, prob)
    assert faint.dtype == bool and faint.shape == img.shape and not faint[39, 50]
    a = guided.overlay(img, prob)
    b = guided.overlay(img, prob, faint=np.pad(np.ones((10, 10), bool), ((5, 65), (5, 105))))
    assert not np.array_equal(a, b)


def test_ffr_read_where_a_wire_would_sit():
    """D68: the product cuts the vessel where its reference falls below 2 mm, keeps the whole
    path for display; a user's own path (wire=False) is not cut."""
    from vstent.inference.pipeline import analyse_vessel

    prob = np.zeros((100, 420))
    for c in range(10, 410):
        half = 7.0 - 4.5 * (c - 10) / 400  # 3.5 mm → 1.25 mm at 0.25 mm/px
        prob[int(50 - half) : int(50 + half) + 1, c] = 1.0
    a = analyse_vessel(prob, (50, 15), (50, 405), mm_per_px=0.25)
    assert a.full_qca is not None and a.qca.s_mm[-1] < a.full_qca.s_mm[-1]
    assert any("pressure wire" in w for w in a.warnings)
    b = analyse_vessel(prob, (50, 15), (50, 405), mm_per_px=0.25, wire=False)
    assert b.full_qca is None and b.solver.ffr[-1] < a.solver.ffr[-1]  # the thin end adds loss
    img = np.full(prob.shape, 120, np.uint8)
    plain = guided.overlay(img, prob, a.qca)
    shown = guided.overlay(img, prob, a.qca, beyond=a.full_qca)
    assert not np.array_equal(plain, shown)  # the dashed rest and the FFR marker


def test_vessel_3d_tube_coloured_by_ffr_with_lesions_and_stent():
    """The 3D view: a tube of the measured radius along the path, coloured by FFR, lesions
    marked, a stent drawn as a sleeve over its span."""
    from vstent.imaging.qca import run_qca
    from vstent.physics.solver import PhysicsParams
    from vstent.planning.stent import apply_stents

    prob = np.zeros((120, 400))
    cols = np.arange(400)
    half = np.where(np.abs(cols - 200) < 12, 2.5, 6.0)
    for c in range(20, 380):
        prob[int(60 - half[c]) : int(60 + half[c]) + 1, c] = 1.0
    q = run_qca(prob, (60, 25), (60, 375), mm_per_px=0.25)
    marks = guided.lesion_marks(q)
    p = PhysicsParams()
    pre = core.evaluate(q.profile, p, p.pa, p.pv)
    fig = guided.vessel_3d(q, pre, marks)
    tube = fig.data[0]
    n = len(tube.x) // guided.TUBE_SIDES
    assert len(tube.x) == n * guided.TUBE_SIDES and len(tube.i) == 2 * (n - 1) * guided.TUBE_SIDES
    pts = guided.path_mm(q, pre.s)
    assert np.allclose(pts[:, 2], 0.0) and pts[-1, 0] - pts[0, 0] > 80.0  # ~87 mm, image plane
    xyz = np.c_[tube.x, tube.y, tube.z].reshape(n, guided.TUBE_SIDES, 3)
    radius = np.linalg.norm(xyz - xyz.mean(axis=1, keepdims=True), axis=2).mean(axis=1)
    k = int(np.argmin(radius))
    assert abs(pts[0, 0] + (pts[-1, 0] - pts[0, 0]) * k / (n - 1) - 200 * 0.25) < 4.0  # waist
    assert tube.intensity[0] > tube.intensity[-1]  # FFR falls along the path
    assert any(t.type == "scatter3d" and t.text and "1:" in t.text[0] for t in fig.data)
    st_ = Stent(marks[0].start_mm * 1e-3 - 2e-3, 16e-3, 3e-3)
    stented = apply_stents(q.profile, [st_])
    post = core.evaluate(stented, p, p.pa, p.pv, r_micro=pre.r_micro)
    fig2 = guided.vessel_3d(q, post, marks, [st_], radius_m=stented.r)
    sleeves = [t for t in fig2.data if t.type == "mesh3d" and t.opacity is not None]
    assert len(sleeves) == 1 and sleeves[0].name.startswith("Stent 3.00 × 16")
    assert fig2.data[0].intensity[-1] > tube.intensity[-1]  # better FFR after the stent


def _tree_with_branch_lesion() -> np.ndarray:
    import cv2

    img = np.zeros((260, 420), np.uint8)
    for k, x in enumerate(range(20, 380, 30)):  # a healthy trunk tapering 17 → 13 px
        cv2.line(img, (x, 60), (x + 30, 60), 1, 17 - (k * 4) // 12)
    a, b = np.array([150.0, 60.0]), np.array([330.0, 240.0])  # a branch, 11 px wide
    for t0 in np.linspace(0, 0.95, 20):
        p, q = a + (b - a) * t0, a + (b - a) * (t0 + 0.05)
        w = 3 if 0.5 <= t0 < 0.6 else 11  # a tight lesion halfway along the branch
        cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), 1, w)
    return img.astype(float)


def test_lesion_seeking_path_takes_the_branch_with_the_lesion():
    """P4k-B: the trunk is healthy, the branch has a ≥ 50% lesion where the vessel is ≥ 2 mm:
    the seeking path goes down the branch; without a lesion it keeps the trunk."""
    from vstent.imaging.centerline import branch_ends
    from vstent.inference.pipeline import analyse_vessel
    from vstent.physics.solver import PhysicsParams
    from vstent.validation.target_path import crosses

    prob = _tree_with_branch_lesion()
    a, ends = branch_ends(prob >= 0.5, 30.0 / 0.30)
    assert a[1] < 40 and len(ends) == 2  # the trunk's wide end; the trunk's and the branch's

    def measure(s, e):
        return analyse_vessel(prob, s, e, params=PhysicsParams(), mm_per_px=0.30)

    trunk = guided.auto_path(prob, 0.30)
    assert abs(trunk[1][0] - 60) < 10  # the main trunk stays on the trunk
    start, end, score = guided.seeking_path(prob, 0.30, measure)
    assert end[0] > 200 and score >= guided.SEEK_MIN_DS  # down the branch to its end
    q = measure(start, end).qca
    assert crosses(q.centerline.points, [(230, 140, 40, 40)])  # the lesion near (row 160, col 240)
    assert not crosses(measure(*trunk).qca.centerline.points, [(230, 140, 40, 40)])
    healthy = prob.copy()
    healthy[130:200, 200:290] = 0
    import cv2

    cv2.line(healthy, (150, 60), (330, 240), 1.0, 11)  # the branch without the lesion

    def measure_healthy(s, e):
        return analyse_vessel(healthy, s, e, params=PhysicsParams(), mm_per_px=0.30)

    trunk_healthy = guided.auto_path(healthy, 0.30)
    assert guided.seeking_path(healthy, 0.30, measure_healthy)[:2] == trunk_healthy


def test_contrast_text():
    from vstent.imaging.cine import ContrastTransit

    s = np.arange(0.0, 40.0, 2.0)
    arrival = 5.0 + s / 4.0  # 4 px per frame
    level = np.r_[np.linspace(0, 1, 16), np.ones(4), [0.5, 0.2]]
    tr = ContrastTransit(s, arrival, 0.0, 4.0, (3.5, 4.5), s.size, np.arange(s.size),
                         arrival + 2, level)  # fmt: skip
    text = guided.contrast_text(tr, 17, 1000.0 / 15.0)
    # ends: the median of 3 points (5.5 and 14.0 frames) → 8.5 frames at 15 fps → 17 at 30
    assert "full on frame 17" in text and "TIMI frame count 17 (at 30 fps)" in text
    assert "needs the frame rate" in guided.contrast_text(tr, 5, None)
    assert "filling" in guided.contrast_text(tr, 5, None)
    assert "not measured" in guided.contrast_text(None, 5, 33.3)


def test_ppg_focal_and_diffuse():
    """Collet's PPG on virtual pullbacks: a 10 mm focal drop → ~0.95; the same drop spread
    evenly over 100 mm → ~0.1; no drop → None."""
    s = np.linspace(0.0, 0.1, 1001)  # m
    focal = np.interp(s, [0.0, 0.04, 0.05, 0.1], [1.0, 1.0, 0.75, 0.75])
    diffuse = 1.0 - 2.5 * s
    p_f = guided.ppg(SimpleNamespace(s=s, solver=focal))
    p_d = guided.ppg(SimpleNamespace(s=s, solver=diffuse))
    assert p_f == pytest.approx(0.95, abs=0.01) and p_d == pytest.approx(0.1, abs=0.01)
    assert "focal" in guided.ppg_text(SimpleNamespace(s=s, solver=focal))
    assert "diffuse" in guided.ppg_text(SimpleNamespace(s=s, solver=diffuse))
    assert guided.ppg(SimpleNamespace(s=s, solver=np.ones_like(s))) is None
