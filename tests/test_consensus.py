"""P4g: multi-frame consensus (split-half repeatability on CADICA rows; the product's consensus
on a beating-heart run; ``vstent validate`` on a cine run)."""

from types import SimpleNamespace

import numpy as np
import pytest

from vstent.imaging.cardiac import EndDiastole
from vstent.imaging.synthetic_cine import (
    CineSynthConfig,
    render_run,
    run_dicom,
    sample_cine_case,
)
from vstent.inference import consensus as pc
from vstent.physics.solver import PhysicsParams
from vstent.validation import consensus as vc


def _frame(frame, ffr, ds=50.0, error=""):
    if error:
        return pc.FrameResult(frame, (0, 0), (1, 1), error=error)
    return pc.FrameResult(frame, (0, 0), (1, 1), None, pc.Values(ffr, ds, 30.0, ffr + 0.01))


def test_consensus_median_choice_and_flag():
    c = pc.Consensus((_frame(10, 0.84), _frame(22, 0.78, 60.0), _frame(34, 0.81),
                      _frame(46, 0.0, error="x")), 3, "ED")  # fmt: skip
    assert c.ffr == pytest.approx(0.81) and c.ffr_surrogate == pytest.approx(0.82)
    assert c.ds == pytest.approx(50.0) and c.chosen.frame == 34 and c.flagged
    assert "Unstable" in c.note() and "Frame 46 left out: x." in c.note()
    d = c.as_dict()
    assert d["chosen_frame"] == 34 and d["unstable"] and d["frames"][3]["left_out"] == "x"
    tie = pc.Consensus((_frame(10, 0.5), _frame(22, 0.75)), 3, "ED")
    assert tie.ffr == pytest.approx(0.625) and tie.chosen.frame == 10 and not tie.flagged
    assert "2 of the 3 frames" in tie.note()
    one = pc.Consensus((_frame(5, 0.7),), 3, "heartbeat frames unknown")
    assert "on one frame only (heartbeat frames unknown)" in one.note("Main-path FFR")
    assert not pc.straddles([0.80, 0.79]) and pc.straddles([0.80, 0.81])


def _fake_run():
    smoothed = np.zeros(70)
    smoothed[14:56] = 92.0
    smoothed[[16, 27, 38, 49]] = (95.0, 99.0, 100.0, 97.0)
    op = SimpleNamespace(window=(12, 60), smoothed=smoothed)
    ed = EndDiastole((5, 16, 27, 38, 49, 60), "R-wave pointer", 11.0, "")
    return SimpleNamespace(opacification=op, ed=ed)


def test_consensus_frames_on_the_fully_opacified_path():
    rf = _fake_run()
    assert pc.opacified_frames(rf, None) == (14, 55)  # the plateau: 5 and 60 are outside
    assert pc.consensus_frames(rf, 38, 3) == ([38, 27, 49], "end-diastolic frames from the "
                                              "R-wave pointer")  # fmt: skip
    assert pc.consensus_frames(rf, 38, 3, filled=30)[0] == [38, 49]  # 16, 27: still filling
    assert pc.consensus_frames(rf, 38, 1) == ([38], "one frame asked for")
    rf.ed = EndDiastole((), "none", None, "")
    assert pc.consensus_frames(rf, 38, 3)[0] == [38]
    tr = SimpleNamespace(filled=np.array([10.0, 11.0, np.nan, 12.0, 20.0]))
    assert pc.filled_frame(tr) == 19 and pc.filled_frame(None) is None


def _rows(track, patient, split, ffrs, ds=50.0):
    return [{"track": track, "patient": patient, "split": split, "frame": 3 * f,
             "category": "p50_70", "default_ffr": v, "default_ds": ds}
            for f, v in enumerate(ffrs)]  # fmt: skip


def test_split_half_choice_of_k_and_report(tmp_path):
    assert vc.halves(list(range(7)), 2) == ([0, 2], [1, 3]) and vc.halves([0, 1, 2], 2) is None
    noisy = [0.60, 0.95, 0.80, 0.80, 0.95, 0.60]  # alternate halves: medians equal from K = 3
    rows = [r for p in ("p1", "p2", "p3") for r in _rows(f"L{p}", p, "dev", noisy)]
    rows += [r for p in ("p4", "p5") for r in _rows(f"L{p}", p, "test", [0.75] * 6)]
    rows[0]["default_ffr"] = None  # not measured: dropped, the lesion keeps 5 frames
    lesions = vc.lesion_frames(rows)
    assert [m["frame"] for m in lesions["Lp1"]] == [3, 6, 9, 12, 15]
    dev = {k: vc.split_half(lesions, k, "dev") for k in vc.KS}
    assert dev[1]["ffr_cv"] > 0.094 and dev[3]["n_lesions"] == 2  # Lp1 has < 6 frames
    assert dev[3]["ffr_cv"] == pytest.approx(0.0) and dev[3]["flagged_share"] == 1.0
    assert dev[4]["ffr_cv"] is None and vc.choose_k(dev)[0] == 3
    s = vc.summarise(rows)
    assert s["chosen_k"] == 3 and all(t["passed"] for t in s["targets"])
    assert s["test"]["agree_080"] == 1.0 and s["test"]["agree_unflagged"] == 1.0
    vc.write_report(s, {"git_commit": "x", "git_dirty": False}, "m",
                    {"git_commit": "y", "git_dirty": False}, tmp_path)  # fmt: skip
    text = (tmp_path / "consensus.md").read_text(encoding="utf-8")
    assert "Research use only" in text and "pre-registered" in text and "K = 3" in text


def test_choose_k_without_a_qualifying_k():
    dev = {1: {"ffr_cv": 0.2, "ds_cv": 0.3}, 2: {"ffr_cv": 0.15, "ds_cv": 0.3},
           3: {"ffr_cv": None, "ds_cv": None}}  # fmt: skip
    assert vc.choose_k(dev) == (2, "no K met both targets on the development patients: the "
                                "lowest FFR CV")  # fmt: skip


@pytest.fixture(scope="module")
def beating_run(tmp_path_factory, dark_seg_dir):
    """A beating-heart run (R-wave pointer, breathing 3–4 mm) as DICOM, its run analysis and
    the true main-vessel position at every frame."""
    from vstent.imaging.image_io import read_run_bytes
    from vstent.imaging.run_analysis import analyse_frames
    from vstent.inference.onnx_runtime import SegmentationRunner

    cfg = CineSynthConfig(
        scene={"image_size": 256, "iso_pixel_mm": (0.45, 0.5), "background": (0.6, 0.64)},
        fps_choices=(15.0,), plateau_beats=(3.0, 3.0), ecg_prob=1.0, breathing_mm=(3.0, 4.0),
    )  # fmt: skip
    cc = sample_cine_case(9100, cfg)
    path = tmp_path_factory.mktemp("cine") / "run.dcm"
    path.write_bytes(run_dicom(cc, 0, render_run(cc, 0, probs=False).frames, cfg))
    seg = SegmentationRunner.from_dir(dark_seg_dir)
    rf = analyse_frames(read_run_bytes(path.read_bytes(), "run.dcm"), seg.predict)
    main = cc.case.curves[0]
    inner = main[[len(main) // 10, (7 * len(main)) // 10]]  # inside the stand-in's mask

    def at(frame):
        return cc.case.views[0].project(cc.position(0, frame / cc.runs[0].fps, inner))

    return path, rf, seg, at


def test_run_consensus_follows_the_heart(beating_run):
    from vstent.inference.pipeline import analyse_vessel

    _, rf, seg, at = beating_run

    def one(p, im, a, b):
        return analyse_vessel(p, a, b, params=PhysicsParams(), image=im)

    systolic = rf.analysis_frame + round(0.3 * rf.ed.period_frames)
    for ref in (rf.analysis_frame, systolic):
        start, end = np.round(at(ref)).astype(int)
        c = pc.run_consensus(rf, ref, start, end, seg.predict, one, k=3)
        assert [f.frame for f in c.frames][0] == ref and len(c.frames) == 3
        assert all(f.frame in rf.ed.frames for f in c.frames[1:])
        for f in c.frames:  # the moved points are where the vessel is on that frame
            assert np.max(np.abs(np.array([f.start, f.end]) - at(f.frame))) <= 2.0
        assert len(c.used) == 3 and c.ffr == pytest.approx(np.median([f.ffr for f in c.used]))
        assert c.flagged == pc.straddles([f.ffr for f in c.used])
        assert c.chosen.analysis.ffr_distal_solver == pytest.approx(c.ffr)  # K odd: a frame


def test_validate_reports_the_consensus_on_a_run(beating_run, tmp_path):
    from tests.test_validation import FAST_UNCERTAINTY
    from vstent.validation.manifest import CaseRow
    from vstent.validation.run import predict_case

    path, rf, seg, at = beating_run
    ref = rf.analysis_frame
    (sr, sc), (er, ec) = np.round(at(ref)).astype(int)
    case = CaseRow(case_id="C1", patient_id="P1", image=str(path), frame=ref, start_row=sr,
                   start_col=sc, end_row=er, end_col=ec)  # fmt: skip
    row = predict_case(case, tmp_path, segmenter=seg, surrogate=None, params=PhysicsParams(),
                       uncertainty=FAST_UNCERTAINTY, qca_error=None)  # fmt: skip
    assert row["single_status"] == "ok" and row["tree_status"] == "ok", row
    frames = [int(f) for f in row["consensus_frames"].split(";")]
    assert frames[0] == ref and 3 <= len(frames) <= pc.CONSENSUS_K
    assert row["tree_consensus_frames"]
    assert row["frame_single_solver"] != "" and row["frame_tree_solver"] != ""
    assert isinstance(row["consensus_unstable"], bool)


def test_merged_tracks_are_found_and_checked(tmp_path):
    rows = [r for p in ("p1", "p2") for r in _rows(f"L{p}", p, "dev", [0.7, 0.9] * 5)]
    rows += [r for p in ("p3", "p4") for r in _rows(f"L{p}", p, "test", [0.75] * 10)]
    rows += [dict(rows[0], default_ffr=0.5)]  # a second box on Lp1's first keyframe
    assert vc.merged_tracks(rows) == {"Lp1"}
    s = vc.summarise(rows)
    assert s["sensitivity"]["n_tracks"] == 1 and s["sensitivity"]["by_split"]["dev"] == 1
    vc.write_report(s, {"git_commit": "x", "git_dirty": False}, "m",
                    {"git_commit": "y", "git_dirty": False}, tmp_path)  # fmt: skip
    assert "two boxes on one keyframe" in (tmp_path / "consensus.md").read_text(encoding="utf-8")
