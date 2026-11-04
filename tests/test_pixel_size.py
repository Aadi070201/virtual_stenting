"""P4r: the true pixel size from CARDIAG's geometry, the catheter width and calibration."""

import json
import zipfile

import cv2
import numpy as np
import pytest

from vstent.validation import pixel_size as ps


def _bar(width: int, shape=(120, 200)) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    m[50 : 50 + width, 20:180] = 255
    return m


def test_catheter_width_of_a_bar():
    assert ps.catheter_width_px(_bar(9))[0] == pytest.approx(9.0)
    assert ps.catheter_width_px(_bar(10))[0] == pytest.approx(9.0)  # even widths: one narrow
    assert ps.catheter_width_px(np.zeros((50, 50))) == (None, 0)
    width, n = ps.catheter_width_px(_bar(9, (60, 30)))  # a short stub: too few skeleton pixels
    assert width is None and 0 < n < ps.MIN_SKELETON


def test_isocentre_pixel_size_from_the_metadata():
    meta = {"sod": "784.441978", "sid": "1119", "spacing": "['0.278875', '0.278875']"}
    assert ps.isocentre_mm(meta) == pytest.approx(0.278875 * 784.441978 / 1119)
    assert ps.isocentre_mm({"sod": "800", "sid": "1100", "spacing": [0.3, 0.3]}) == pytest.approx(
        0.3 * 800 / 1100
    )
    assert ps.isocentre_mm({"sod": "800", "spacing": "0.3"}) is None
    assert ps.isocentre_mm({"sod": "0", "sid": "1100", "spacing": "0.3"}) is None


def _png(a: np.ndarray) -> bytes:
    return cv2.imencode(".png", a)[1].tobytes()


def test_read_and_summarise_a_small_archive(tmp_path):
    zp = tmp_path / "CARDIAG.zip"
    # true 0.2 mm/px; catheters 11 and 9 px wide (2.2 and 1.8 mm), and one frame without
    with zipfile.ZipFile(zp, "w") as zf:
        for k, (centre, width) in enumerate([("x", 11), ("x", 9), ("y", 0)]):
            meta = {"sod": "800", "sid": "1200", "spacing": "['0.3', '0.3']",
                    "center_id": centre, "patient_id": f"p{k}"}  # fmt: skip
            zf.writestr(f"{k:03d}/metadata.json", json.dumps(meta))
            zf.writestr(f"{k:03d}/img.png", _png(np.zeros((512, 512), np.uint8)))
            zf.writestr(f"{k:03d}/catheter.png", _png(_bar(width) if width else
                                                      np.zeros((120, 200), np.uint8)))  # fmt: skip
        zf.writestr("other_keyframes", "")
    samples = ps.read_samples(zp)
    assert [s.name for s in samples] == ["000", "001", "002"]
    assert samples[0].mm_per_px == pytest.approx(0.2) and samples[0].shape == (512, 512)
    s = ps.summarise(samples)
    a, b, c = s["a"], s["b"], s["c"]
    assert a["mm_per_px"]["median"] == pytest.approx(0.2) and a["centres"] == 2
    assert a["overstatement"]["median"] == pytest.approx(1.5) and a["inside_input_range"] == 0.0
    assert a["shapes"] == {"512×512": 3}
    assert (b["measured"], b["no_catheter"]) == (2, 1)
    assert b["width_mm"]["median"] == pytest.approx(2.0) and b["scale_holds"]
    # 2.0 mm / 11 px and / 9 px against 0.2 mm/px: −9.1% and +11.1%
    assert c["median_abs_error"] == pytest.approx(
        (abs(2 / 11 / 0.2 - 1) + abs(2 / 9 / 0.2 - 1)) / 2
    )
    assert c["within_10"] == 0.5 and c["within_20"] == 1.0 and not c["worth_building"]
    md = "\n".join(ps.report_lines(s, {"git_commit": "abc", "git_dirty": False}))
    assert "Research use only" in md and "Scale holds" in md and "not met" in md
