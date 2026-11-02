"""P7: end-diastolic frames (ECG, R-wave pointer, images), motion tracking, flow velocity."""

import io
import re
from dataclasses import replace

import numpy as np
import pydicom
import pytest

from vstent.imaging.cardiac import (
    EcgSignal,
    ed_analysis_frame,
    ed_from_ecg,
    ed_from_spread,
    end_diastole,
    frames_at,
    r_peaks,
    r_wave_pointer,
    read_ecg,
)
from vstent.imaging.cine import ContrastTransit, opacification
from vstent.imaging.flow import (
    CFV_RANGE,
    SINGLE_VIEW_UPPER,
    VELOCITY_LOG_HALF_WIDTH,
    VELOCITY_LOG_HALF_WIDTH_TWO_RUNS,
    combine_runs,
    flow_time,
    flow_velocity,
    hyperemic_velocity,
    phasic_slope,
)
from vstent.imaging.image_io import read_run_bytes
from vstent.imaging.motion import _same_phase, apply, plateau, track_affine
from vstent.imaging.synthetic_cine import (
    CineSynthConfig,
    ecg_waveform,
    run_dicom,
    sample_cine_case,
)

FS = 500.0


def _ecg(r_times_s, duration_s, noise=0.03, seed=0):
    t = np.arange(0.0, duration_s, 1 / FS)
    sig = 0.1 * np.sin(2 * np.pi * 0.25 * t)
    for r in r_times_s:
        for dt, amp, sd in ((-0.16, 0.12, 0.025), (0.0, 1.0, 0.012), (0.03, -0.25, 0.012),
                            (0.25, 0.3, 0.05)):  # fmt: skip
            sig += amp * np.exp(-0.5 * ((t - r - dt) / sd) ** 2)
    return sig + np.random.default_rng(seed).normal(0.0, noise, t.size)


def test_r_peaks_and_frames():
    r_times = np.arange(0.3, 6.0, 0.8)
    peaks = r_peaks(_ecg(r_times, 6.2), FS)
    assert peaks.size == r_times.size
    assert np.max(np.abs(peaks / FS - r_times)) < 0.01
    # an inverted lead (QRS mostly negative) is found too
    assert r_peaks(-_ecg(r_times, 6.2), FS).size == r_times.size
    assert frames_at(np.array([-20.0, 0.0, 120.0, 930.0, 1000.0]), 15, 66.7) == (0, 2, 14)
    ecg = EcgSignal(_ecg(r_times, 6.2), FS, start_ms=-100.0)  # starts 100 ms before frame 0
    ed = ed_from_ecg(ecg, 90, 1000 / 15)
    expected = tuple(int(round((r - 0.1) * 15)) for r in r_times if 0 <= round((r - 0.1) * 15) < 90)
    assert ed.source == "ECG" and ed.frames == expected
    assert ed.period_frames == pytest.approx(0.8 * 15, rel=0.02)


@pytest.fixture(scope="module")
def cine_case():
    return sample_cine_case(9100, CineSynthConfig())  # P7 development seed


def test_dicom_r_wave_pointer_and_ecg_round_trip(cine_case):
    run = replace(cine_case.runs[0], has_ecg=True)
    cc = replace(cine_case, runs=(run, cine_case.runs[1]))
    frames = np.zeros((run.n_frames, 16, 16), np.uint8)
    data = run_dicom(cc, 0, frames, CineSynthConfig())
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    assert r_wave_pointer(ds, run.n_frames) == cc.ed_frames(0)
    ecg = read_ecg(ds)
    sig, start_ms = ecg_waveform(cc, 0, 500.0, 0.03)
    assert ecg is not None and ecg.start_ms == pytest.approx(start_ms, abs=1e-3)
    assert ed_from_ecg(ecg, run.n_frames, run.frame_time_ms).frames == cc.ed_frames(0)
    loaded = read_run_bytes(data, "run.dcm")
    assert loaded.r_wave_frames == cc.ed_frames(0) and loaded.ecg is not None
    assert loaded.frame_time_ms == pytest.approx(run.frame_time_ms, rel=1e-4)
    assert "R-wave pointer" in loaded.note
    ed = end_diastole(run.n_frames, run.frame_time_ms, pointer=loaded.r_wave_frames,
                      ecg=loaded.ecg)  # fmt: skip
    assert ed.source == "R-wave pointer" and ed.frames == cc.ed_frames(0)
    assert ed.period_frames == pytest.approx(cc.period_s * run.fps, abs=1.0)
    assert end_diastole(run.n_frames, run.frame_time_ms, ecg=loaded.ecg).source == "ECG"


def test_ed_from_spread_finds_the_largest_tree():
    t = np.arange(80)
    period, first = 11.0, 4.0
    spread = 100.0 * (1.0 + 0.05 * np.cos(2 * np.pi * (t - first) / period)) * (1 + 0.002 * t)
    spread += np.random.default_rng(1).normal(0.0, 0.3, t.size)
    area = np.clip((t - 5) * 40.0, 0, 1000) * (t < 70)  # filling, plateau, washout
    op = opacification(area)
    ed = ed_from_spread(spread, op.smoothed, 1000 / 15)
    truth = [f for f in np.arange(first, 80, period).round().astype(int) if area[f] >= 500]
    assert ed.source == "images" and ed.period_frames == pytest.approx(period, abs=1)
    assert all(min(abs(f - g) for g in ed.frames) <= 1 for f in truth[1:-1])
    frame, note = ed_analysis_frame(op, ed)
    assert frame in ed.frames and op.window[0] <= frame <= op.window[1] and "End-diastolic" in note
    none = ed_from_spread(spread, np.zeros(80))
    assert not none.found and ed_analysis_frame(op, none)[0] == op.analysis_frame


def test_flow_time_and_phasic_slope():
    ed_s, period = np.array([0.1]), 0.8
    g = flow_time(np.array([0.3, 1.1, 1.9]), ed_s, period, 2.0)
    assert np.diff(g) == pytest.approx([period, period])  # one beat of flow time per beat
    assert flow_time(np.array([0.2, 0.6]), ed_s, period, 1.0) == pytest.approx(
        [0.2 - 0.1, 0.6 - 0.1]
    )
    # a front with phasic speed (ratio 2) over part of a beat: plain time is biased, flow time
    # with the right ratio recovers the cycle average
    v = 0.2
    t = np.linspace(0.12, 0.55, 40)  # mostly systole
    d = v * (flow_time(t, ed_s, period, 2.0) - flow_time(np.array([0.12]), ed_s, period, 2.0))
    dt_ms = 1000 / 30
    tr = ContrastTransit(d / 1e-3, t / (dt_ms / 1000), 0.0, 1.0, (0.5, 1.5), 40, np.arange(40))
    speed, lo, hi, ratio = phasic_slope(tr, lambda s: s * 1e-3, dt_ms, [0.1 * 30], period * 30, 2.0)
    assert speed == pytest.approx(v, rel=0.01) and ratio == 2.0
    plain = np.polyfit(t, d, 1)[0]
    assert plain < 0.8 * v
    fitted = phasic_slope(tr, lambda s: s * 1e-3, dt_ms, [0.1 * 30], period * 30, None)
    assert fitted[3] == pytest.approx(2.0, abs=0.15)


def _phasic_transit(v, px_m, dt_ms, ed_frames, period_frames, n=60):
    """A transit whose front moves at cycle average ``v`` (m/s) with a diastolic / systolic
    ratio of 2 (the prior)."""
    dt = dt_ms / 1000.0
    a = np.linspace(3.0, 3.0 + 0.6 * period_frames, n)
    ed_s, period = np.asarray(ed_frames, float) * dt, period_frames * dt
    d = v * (flow_time(a * dt, ed_s, period, 2.0) - flow_time(a[:1] * dt, ed_s, period, 2.0))
    s = d / px_m
    slope = float(np.polyfit(a, s, 1)[0])
    return ContrastTransit(s, a, 0.0, slope, (0.9 * slope, 1.1 * slope), n, np.arange(n))


def test_flow_velocity_conversion_and_guards():
    assert hyperemic_velocity(0.2) == pytest.approx(0.10 + 1.55 * 0.2 - 0.93 * 0.04)
    assert hyperemic_velocity(2.0) == pytest.approx(hyperemic_velocity(1.55 / 1.86))  # capped
    px_m, dt_ms, ed, period = 0.25e-3, 1000 / 15, [2, 14, 26], 12.0
    tr = _phasic_transit(0.15, px_m, dt_ms, ed, period)
    cycle = {"ed_frames": ed, "period_frames": period}
    fv = flow_velocity(tr, dt_ms, lambda x: x * px_m, single_view=True, **cycle)
    assert fv.accepted and fv.cfv == pytest.approx(0.15, rel=1e-3)
    q = np.exp(VELOCITY_LOG_HALF_WIDTH)
    assert fv.interval == pytest.approx((0.15 / q, 0.15 * q * SINGLE_VIEW_UPPER), rel=1e-3)
    assert fv.v_hyp == pytest.approx(hyperemic_velocity(fv.cfv))
    assert "one view" in fv.note() and "FAVOR" in fv.note()
    hyp = flow_velocity(tr, dt_ms, lambda x: x * px_m, single_view=False, hyperemic_run=True,
                        **cycle)  # fmt: skip
    assert hyp.v_hyp == pytest.approx(hyp.cfv) and hyp.interval[1] < fv.interval[1]

    def run(v):
        return flow_velocity(_phasic_transit(v, px_m, dt_ms, ed, period), dt_ms,
                             lambda x: x * px_m, single_view=False, hyperemic_run=True,
                             **cycle)  # fmt: skip

    two = combine_runs([hyp, run(0.20)])
    assert two.accepted and two.n_runs == 2 and two.cfv == pytest.approx(0.175, rel=1e-3)
    q2 = np.exp(VELOCITY_LOG_HALF_WIDTH_TWO_RUNS)
    assert two.interval == pytest.approx((0.175 / q2, 0.175 * q2), rel=1e-3)
    apart = combine_runs([hyp, run(0.25)])  # ×1.67: one of them failed
    assert not apart.accepted and "disagree" in apart.reasons[0]
    slow = flow_velocity(tr, dt_ms, lambda x: x * px_m * 0.1, single_view=False, **cycle)
    assert not slow.accepted and f"{CFV_RANGE[0]:g}" in slow.reasons[0]
    assert combine_runs([slow, fv]) is fv
    unknown = flow_velocity(tr, dt_ms, lambda x: x * px_m, single_view=False,
                            ed_frames=None, period_frames=None)  # fmt: skip
    assert not unknown.accepted and "heart cycle unknown" in unknown.reasons[0]
    assert not flow_velocity(tr, None, lambda x: x, single_view=False, **cycle).accepted
    assert not flow_velocity(tr, 66.7, None, single_view=False, **cycle).accepted
    assert not flow_velocity(None, 66.7, lambda x: x, single_view=False, **cycle).accepted
    s = np.arange(0.0, 200.0, 2.0)
    back = ContrastTransit(s, -s / 20.0, 0.0, -20.0, (-21.0, -19.0), s.size, np.arange(s.size))
    assert "does not advance" in flow_velocity(back, 66.7, lambda x: x * px_m,
                                               single_view=False, **cycle).reasons[0]  # fmt: skip


def test_track_affine_follows_scale_and_shift():
    yy, xx = np.mgrid[0:200, 0:200]
    ref = ((np.abs(yy - 100) < 2) & (np.abs(xx - 100) < 60)) | (
        (np.abs(xx - 80) < 2) & (np.abs(yy - 100) < 50)
    )
    # the same cross, 8% smaller about (100, 100) and shifted by (5, -7)
    src_y, src_x = (yy - 100 - 5) / 0.92 + 100, (xx - 100 + 7) / 0.92 + 100
    moved = ((np.abs(src_y - 100) < 2) & (np.abs(src_x - 100) < 60)) | (
        (np.abs(src_x - 80) < 2) & (np.abs(src_y - 100) < 50)
    )
    w = track_affine([ref.astype(np.float32), moved.astype(np.float32)], 0)
    p = np.array([[100.0, 40.0], [60.0, 80.0]])
    expected = (p - 100) * 0.92 + 100 + np.array([5.0, -7.0])
    assert np.max(np.abs(apply(w[1], p) - expected)) < 1.5
    assert np.allclose(w[0], np.eye(2, 3))


def test_cine_report_end_to_end(dark_seg_dir, tmp_path):
    """The P7 report pipeline on one small development case (stand-in segmenter)."""
    from vstent.imaging import cine_report as cr

    small = CineSynthConfig(
        scene={"image_size": 256, "iso_pixel_mm": (0.45, 0.5), "background": (0.6, 0.64)},
        fps_choices=(7.5,), plateau_beats=(1.5, 1.5), ecg_prob=1.0,
    )  # fmt: skip
    row = cr.run_case(9100, small, str(dark_seg_dir))
    assert len(row["runs"]) == 2 and all(r["source"] == "R-wave pointer" for r in row["runs"])
    assert all(r["frame_err"] <= 1 for r in row["runs"])
    assert all(r["ecg_beats"][0] == r["ecg_beats"][1] for r in row["runs"])
    s = cr.write_report([row], small, [9100], {"git_commit": "x", "git_dirty": False},
                        str(dark_seg_dir), tmp_path)  # fmt: skip
    text = (tmp_path / "cine_synthetic.md").read_text(encoding="utf-8")
    assert "Research use only" in text and "Pre-registered targets" in text
    assert len(s["targets"]) == 9
    c = cr.write_calibration([row], [9100], {"git_commit": "x", "git_dirty": False},
                             str(dark_seg_dir), tmp_path)  # fmt: skip
    assert (tmp_path / "cine_calibration.md").exists() and c["in_use"][
        "per_run"
    ] == VELOCITY_LOG_HALF_WIDTH


def test_plateau_and_same_phase_transforms():
    assert plateau(np.array([0, 10, 80, 95, 88, 100, 92, 40, 0.0])) == (3, 6)
    assert _same_phase(0, 10, 20, 12.0) == 12 and _same_phase(30, 10, 20, 12.0) == 18
    assert _same_phase(2, 10, 14, 30.0) == 10  # no frame a whole beat away: the nearest
    assert _same_phase(25, 10, 14, None) == 14


def test_transit_with_tracking_ignores_partial_filling():
    """Registering a partly filled vessel slides it along itself; only the plateau is
    registered, so a still vessel keeps its true front speed."""
    from vstent.imaging.centerline import trace_centerline
    from vstent.imaging.cine import contrast_transit
    from vstent.imaging.synthetic import render_cine, sine_centerline

    pts, s = sine_centerline((64.0, 10.0), 110.0, amplitude_px=6.0, wavelength_px=120.0)
    frames = render_cine((128, 128), [(pts, np.full(s.size, 4.0))], n_frames=24,
                         arrival_frame=3.0, front_px_per_frame=10.0, washout_frame=18.0,
                         background=0.6, blur_sigma=0.8, noise_std=0.01, mu_per_mm=0.2,
                         motion_px=3.0)  # fmt: skip
    probs = [(f < 0.6 * 255 - 25).astype(np.float32) for f in frames]
    op = opacification(np.array([p.sum() for p in probs]))
    ref = op.analysis_frame
    cl = trace_centerline(probs[ref] >= 0.5, tuple(np.round(pts[2]).astype(int)),
                          tuple(np.round(pts[-3]).astype(int)))  # fmt: skip
    w = track_affine(probs, ref, window=plateau(op.smoothed), period=12.0)
    assert contrast_transit(frames, cl, warps=w).px_per_frame == pytest.approx(10.0, rel=0.08)


def test_cli_analyse_beating_heart_run_with_measured_flow(dark_seg_dir, tiny_export, tmp_path):
    """``vstent analyse <run.dcm>``: ED frame from the R-wave pointer, measured velocity, and
    ``--flow measured`` uses it."""
    from typer.testing import CliRunner

    from vstent.cli import app
    from vstent.imaging import cine_report as cr
    from vstent.imaging.run_analysis import analyse_frames
    from vstent.imaging.synthetic_cine import render_run
    from vstent.inference.onnx_runtime import SegmentationRunner

    small = CineSynthConfig(
        scene={"image_size": 256, "iso_pixel_mm": (0.45, 0.5), "background": (0.6, 0.64)},
        fps_choices=(15.0,), plateau_beats=(1.5, 1.5), ecg_prob=1.0,
    )  # fmt: skip
    cc = sample_cine_case(9100, small)
    frames = render_run(cc, 0, probs=False).frames
    dcm = tmp_path / "run.dcm"
    dcm.write_bytes(run_dicom(cc, 0, frames, small))
    rf = analyse_frames(read_run_bytes(dcm.read_bytes(), "run.dcm"),
                        SegmentationRunner.from_dir(dark_seg_dir).predict)  # fmt: skip
    main = cc.case.curves[0]
    inner = main[[len(main) // 10, (7 * len(main)) // 10]]  # inside the stand-in's mask
    t_s = rf.analysis_frame / cc.runs[0].fps
    start, end = np.round(cc.case.views[0].project(cc.position(0, t_s, inner))).astype(int)
    args = ["analyse", str(dcm), "--start", *map(str, start), "--end", *map(str, end),
            "--seg-model", str(dark_seg_dir),
            "--surrogate-model", tiny_export["out_dir"]]  # fmt: skip
    out = CliRunner().invoke(app, args)
    assert out.exit_code == 0, out.output
    assert (
        "DICOM R-wave pointer" in out.output
        and f"Analysis frame {rf.analysis_frame}:" in out.output
    )
    assert "Contrast flow velocity" in out.output and "Flow: fixed" in out.output
    assert re.search(r"Distal FFR(, median of \d end-diastolic frames| on one frame only)",
                     out.output)  # P4g  # fmt: skip
    one = CliRunner().invoke(app, [*args, "--consensus", "1", "--tree"])
    assert one.exit_code == 0, one.output
    assert "Main-path distal FFR (tree) on one frame only (one frame asked for)" in one.output
    measured = CliRunner().invoke(app, [*args, "--flow", "measured"])
    assert measured.exit_code == 0 and "Flow: measured velocity used" in measured.output
    bad = CliRunner().invoke(app, [*args, "--flow", "guess"])
    assert bad.exit_code == 2
    assert cr.TARGETS["coverage"] == 0.85
