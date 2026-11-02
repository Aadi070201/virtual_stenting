"""P4f: repeatability of %DS and FFR (frame to frame, operator to operator)."""

import numpy as np
import pytest

from tests.test_public_validation import _fake_cadica
from vstent.data import cadica
from vstent.physics.solver import PhysicsParams
from vstent.validation import repeatability as rp
from vstent.validation import stenosis_grading as sg


def test_cv_and_agreement():
    assert rp.cv_rms([[0.8, 0.8], [0.5, 0.5]]) == pytest.approx(0.0)
    # one group 0.7 / 0.9: SD 0.1414, mean 0.8 → CV 0.177; RMS over two equal groups the same
    assert rp.cv_rms([[0.7, 0.9], [0.7, 0.9]]) == pytest.approx(np.std([0.7, 0.9], ddof=1) / 0.8)
    assert rp.cv_rms([[1.0]]) is None
    assert rp._pairs_agree([[0.75, 0.85], [0.7, 0.72, 0.9]]) == (1, 4)
    assert rp._sd_diff([[1.0, 2.0], [3.0, 3.0], [5.0, 4.0]]) == pytest.approx(
        np.std([-1.0, 0.0, 1.0], ddof=1)
    )


def _row(track, patient, frame, ds, ffr, a, b, category="p50_70"):
    return {"track": track, "patient": patient, "category": category, "frame": frame,
            "default_ds": ds, "default_ffr": ffr, "default_error": "",
            "op_a_ds": a[0], "op_a_ffr": a[1], "op_a_error": "",
            "op_b_ds": b[0], "op_b_ffr": b[1], "op_b_error": ""}  # fmt: skip


def test_summarise_targets():
    rows = []
    for k in range(6):
        p = f"p{k}"
        for f in range(3):  # the same lesion, nearly the same on every frame and operator
            rows.append(_row(f"L{k}", p, f, 50.0 + f, 0.80 + 0.005 * f, (50.0, 0.80),
                             (51.0, 0.805)))  # fmt: skip
    s = rp.summarise(rows)
    assert s["n_lesions"] == 6 and s["frames"]["n_groups"] == 6
    assert all(t["passed"] for t in s["targets"])
    assert s["frames"]["ffr_cv_ci"] is not None and s["operators"]["agree_080"] < 1.0


def test_measure_end_to_end(tmp_path, dark_seg_dir):
    from vstent.inference.onnx_runtime import SegmentationRunner

    raw = _fake_cadica(tmp_path / "raw")
    tracks = sg.lesion_tracks(cadica.load_lesion_boxes(raw))
    seg = SegmentationRunner.from_dir(dark_seg_dir)
    rows = rp.measure(tracks, seg, PhysicsParams())
    assert rows and all(r["default_ffr"] is not None for r in rows)
    assert all(0.0 < r["default_ffr"] <= 1.0 and r["default_ds"] > 10.0 for r in rows)
    again = rp.measure(tracks, seg, PhysicsParams())  # operators' margins are seeded
    assert [r["op_a_ffr"] for r in again] == [r["op_a_ffr"] for r in rows]
    s = rp.summarise(rows)
    rp.write_report(s, {"git_commit": "x", "git_dirty": False}, "m", tmp_path)
    text = (tmp_path / "repeatability.md").read_text(encoding="utf-8")
    assert "Research use only" in text and "Pre-registered targets" in text
