"""Cine runs (P4d): run loading, opacification window, contrast transit, CADICA frame scoring."""

import io

import cv2
import numpy as np
import pytest

from vstent.data.cadica import load_selected_runs
from vstent.imaging.centerline import smooth_and_resample
from vstent.imaging.cine import contrast_transit, opacification, vessel_areas
from vstent.imaging.image_io import read_run
from vstent.imaging.synthetic import render_cine, sine_centerline
from vstent.validation.cine_frames import score_run, summarise

SHAPE = (160, 360)


def _run(motion_px=0.0, seed=0):
    pts, s = sine_centerline((80.0, 20.0), 320.0, amplitude_px=15.0, wavelength_px=300.0)
    frames = render_cine(SHAPE, [(pts, np.full(s.size, 5.0))], n_frames=40, arrival_frame=6.0,
                         front_px_per_frame=12.0, washout_frame=30.0, motion_px=motion_px,
                         background=0.6, blur_sigma=1.0, edge_amount=1.0, noise_std=0.01,
                         seed=seed)  # fmt: skip
    return frames, pts


def _dark(frame):
    """Stand-in segmenter: clearly darker than the background is vessel."""
    return (frame < 0.6 * 255 - 25).astype(float)


def test_opacification_window_and_frames():
    area = np.array([0, 0, 0, 10, 50, 80, 95, 100, 0, 97, 90, 60, 20, 5], float)  # one dropout
    op = opacification(area, fraction=0.8)
    # the dropout (frame 8) is smoothed over: median(100, 0, 97) = 97, the smoothed peak
    assert op.analysis_frame == 8 and op.window == (5, 10)
    assert op.arrival == 4
    with pytest.raises(ValueError):
        opacification(np.array([]))


def test_synthetic_run_analysis_frame_is_fully_opacified():
    frames, _ = _run()
    op = opacification(vessel_areas(frames, _dark))
    # the front reaches the end (~330 px) at frame 6 + 330 / 12 ≈ 33.5; washout from frame 30
    assert 30 - 8 <= op.analysis_frame <= 31
    assert op.arrival <= 20 and op.window[0] <= op.analysis_frame <= op.window[1]


@pytest.mark.parametrize("motion_px", [0.0, 3.0])
def test_contrast_transit_recovers_the_front_speed(motion_px):
    frames, pts = _run(motion_px)
    tr = contrast_transit(frames, smooth_and_resample(np.round(pts[::8])))
    assert tr is not None and tr.n_points > 50
    assert tr.px_per_frame == pytest.approx(12.0, rel=0.15)
    assert tr.interval[0] <= tr.px_per_frame <= tr.interval[1]
    assert tr.velocity_mm_s(0.25, 1000 / 15) == pytest.approx(tr.px_per_frame * 0.25 * 15)
    # a run without contrast has no transit
    assert contrast_transit(np.full((10, *SHAPE), 150, np.uint8),
                            smooth_and_resample(np.round(pts[::8]))) is None  # fmt: skip


def _dicom_run(frames: np.ndarray, frame_time_ms: float) -> bytes:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, generate_uid()
    t, h, w = frames.shape
    ds.Rows, ds.Columns, ds.NumberOfFrames = h, w, t
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.FrameTime = frame_time_ms
    ds.PixelData = frames.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_read_run_from_a_folder_and_a_dicom(tmp_path):
    frames = np.zeros((3, 20, 30), np.uint8)
    frames[1, :, 10:] = 100
    frames[2, :, 10:] = 200
    folder = tmp_path / "run"
    folder.mkdir()
    for k, f in enumerate(frames):  # names sort in frame order
        cv2.imwrite(str(folder / f"r_{k + 1:05d}.png"), f)
    run = read_run(folder)
    assert run.frames.shape == (3, 20, 30) and np.array_equal(run.frames, frames)
    assert run.frame_time_ms is None and "unknown" in run.note
    one = run.frame(2)
    assert one.image.max() == 200 and "Frame 2 of 3" in one.note
    with pytest.raises(ValueError, match="frame 3 requested"):
        run.frame(3)
    # 12-bit DICOM: one intensity scale for the whole run, not per frame
    dcm = tmp_path / "run.dcm"
    dcm.write_bytes(_dicom_run(frames.astype(np.uint16) * 16, 66.7))
    d = read_run(dcm)
    assert d.frame_time_ms == pytest.approx(66.7) and d.frames.shape == (3, 20, 30)
    assert d.frames[1, 0, 20] < d.frames[2, 0, 20]  # frame 1 stays half as dark as frame 2
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no frame images"):
        read_run(empty)


def test_cadica_runs_and_frame_scores(tmp_path):
    root = tmp_path / "CADICA" / "selectedVideos"
    for p, v, ids, lesion in (("p1", "v2", (5, 6, 7, 9), True), ("p2", "v1", (3, 4), False)):
        vdir = root / p / v
        (vdir / "input").mkdir(parents=True)
        (vdir / f"{p}_{v}_selectedFrames.txt").write_text(
            "\n".join(f"{p}_{v}_{i:05d}" for i in ids), encoding="utf-8"
        )
        if lesion:
            (vdir / "groundtruth").mkdir()
    runs = load_selected_runs(tmp_path)
    assert [(r.patient, r.video, r.selected, r.lesion) for r in runs] == [
        ("p1", "v2", (5, 6, 7, 9), True), ("p2", "v1", (3, 4), False)]  # fmt: skip
    # peak at 1-based frame 7, window 6–9 (fraction 0.8)
    row = score_run(runs[0], [0, 0, 0, 10, 50, 85, 100, 90, 85, 40], fraction=0.8)
    assert row["analysis"] == 7 and row["hit"] and row["in_span"]
    assert (row["window_first"], row["window_last"]) == (6, 9)
    assert row["iou"] == pytest.approx(3 / 5)  # {6,7,8,9} vs {5,6,7,9}
    miss = score_run(runs[1], [0, 0, 0, 0, 0, 10, 100, 90], fraction=0.8)
    assert not miss["hit"] and not miss["in_span"] and miss["iou"] == 0.0
    s = summarise([row, miss], n_boot=50)
    assert s["n_runs"] == 2 and s["hit"]["k"] == 1 and s["iou"]["median"] == pytest.approx(0.3)


def test_contrast_transit_on_a_path_through_the_inflow():
    """Tip → inflow → tip (as LAD → left main → LCx): the inflow is found and the longer side
    gives the front speed."""
    x = np.linspace(0.0, 1.0, 2000)
    right = np.stack([np.full_like(x, 80.0), 120.0 + 220.0 * x], axis=1)  # 220 px
    left = np.stack([np.full_like(x, 80.0), 120.0 - 100.0 * x], axis=1)  # 100 px
    r = np.full(x.size, 5.0)
    frames = render_cine(SHAPE, [(right, r), (left, r)], n_frames=40, arrival_frame=6.0,
                         front_px_per_frame=12.0, washout_frame=32.0, background=0.6,
                         blur_sigma=1.0, noise_std=0.01)  # fmt: skip
    path = np.concatenate([left[::-1][::8], right[::8]])
    tr = contrast_transit(frames, smooth_and_resample(np.round(path)))
    assert tr is not None and tr.inflow_px == pytest.approx(100.0, abs=8.0)
    assert tr.px_per_frame == pytest.approx(12.0, rel=0.15)


def test_cine_report_end_to_end(tmp_path, dark_seg_dir):
    """Fake CADICA runs (synthetic cine frames as PNGs) → areas (cached) → scores, transit and
    the report."""
    from vstent.data.cadica import load_artery_groups
    from vstent.inference.onnx_runtime import SegmentationRunner
    from vstent.validation import cine_frames as cf

    root = tmp_path / "raw" / "CADICA" / "selectedVideos"
    pts, s = sine_centerline((64.0, 10.0), 110.0, amplitude_px=6.0, wavelength_px=120.0)
    frames = render_cine((128, 128), [(pts, np.full(s.size, 4.0))], n_frames=24,
                         arrival_frame=3.0, front_px_per_frame=10.0, washout_frame=18.0,
                         background=0.6, blur_sigma=0.8, noise_std=0.01, mu_per_mm=0.2)  # fmt: skip
    for p in ("p1", "p2"):
        vdir = root / p / "v1"
        (vdir / "input").mkdir(parents=True)
        for k, f in enumerate(frames):
            cv2.imwrite(str(vdir / "input" / f"{p}_v1_{k + 1:05d}.png"), f)
        (vdir / f"{p}_v1_selectedFrames.txt").write_text(
            "\n".join(f"{p}_v1_{i:05d}" for i in range(16, 20)), encoding="utf-8"
        )
    (root / "CADICAprojections.json").write_text(
        '{"videosLCA": ["p1_v1"], "videosRCA": ["p2_v1"]}', encoding="utf-8"
    )
    runs = load_selected_runs(tmp_path / "raw")
    seg = SegmentationRunner.from_dir(dark_seg_dir)
    cache = tmp_path / "areas.json"
    areas = cf.run_areas(runs, seg.predict, cache)
    assert cache.exists() and len(areas["p1_v1"]) == 24
    assert cf.run_areas(runs, lambda f: 1 / 0, cache) == areas  # from the cache, no segmenting
    rows = [{**cf.score_run(r, areas[cf.run_key(r)]), "split": sp}
            for r, sp in zip(runs, ("dev", "test"), strict=True)]  # fmt: skip
    assert all(r["in_span"] for r in rows)  # filled ~16 (1-based) to washout at 19
    transit = [cf.transit_for_run(r, seg.predict, r_["analysis"] - 1)
               for r, r_ in zip(runs, rows, strict=True)]  # fmt: skip
    assert all(t.get("px_per_frame", 0) == pytest.approx(10.0, rel=0.3) for t in transit)
    git = {"git_commit": "abc", "git_dirty": False}
    cf.write_report(rows, transit, areas, tmp_path / "rep" / "cine.md", fraction=0.8,
                    arteries=load_artery_groups(tmp_path / "raw"), model="m", git=git,
                    n_boot=50)  # fmt: skip
    md = (tmp_path / "rep" / "cine.md").read_text(encoding="utf-8")
    assert "clinicians" in md and "Measured in 2 of 2 runs" in md and "left coronary" in md
    assert (tmp_path / "rep" / "figures" / "cine_frames.png").exists()
    assert (tmp_path / "rep" / "cine.json").exists()
