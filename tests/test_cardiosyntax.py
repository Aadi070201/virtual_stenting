"""P4o: CardioSYNTAX metadata, runs read from the zips, the side summary."""

import io
import json
import zipfile

import numpy as np
import pytest

from vstent.data import cardiosyntax as cs
from vstent.validation import cardiosyntax_check as cc


def _write(tmp_path):
    studies = [
        {"study_uid": "a", "syntax": 0.0, "syntax_left": 0.0, "syntax_right": 0.0,
         "dominance": None, "bypass": False, "videos": [
             {"path": "0/a/v1.npy", "artery": "LCA", "artery_prob": 0.01, "shape": [12, 8, 8],
              "PositionerPrimaryAngle": 1.0, "PositionerSecondaryAngle": 2.0},
             {"path": "0/a/v2.npy", "artery": "RCA", "artery_prob": 0.5, "shape": [12, 8, 8]},
             {"path": "3/a/v3.npy", "artery": "RCA", "artery_prob": 0.99, "shape": [12, 8, 8]}]},
        {"study_uid": "b", "syntax": 12.0, "syntax_left": 12.0, "syntax_right": 0.0,
         "dominance": "right", "bypass": True, "videos": [
             {"path": "9/b/v1.npy", "artery": "LCA", "artery_prob": 0.0, "shape": [30, 8, 8]}]},
    ]  # fmt: skip
    (tmp_path / "all.json").write_text(json.dumps(studies), encoding="utf-8")
    buf = io.BytesIO()
    np.save(buf, np.arange(12 * 8 * 8, dtype=np.uint8).reshape(12, 8, 8))
    with zipfile.ZipFile(tmp_path / "0.zip", "w") as z:
        z.writestr("0/a/v1.npy", buf.getvalue())


def test_studies_runs_and_tasks(tmp_path):
    _write(tmp_path)
    studies = cs.load_studies(tmp_path)
    assert [s.uid for s in studies] == ["a", "b"] and studies[0].videos[0].part == 0
    assert studies[1].side_syntax("LCA") == 12.0 and studies[1].side_syntax("RCA") == 0.0
    v = cs.read_video(tmp_path, studies[0].videos[0])
    assert v.shape == (12, 8, 8) and v.dtype == np.uint8
    with pytest.raises(KeyError):
        cs.read_video(tmp_path, "0/a/missing.npy")
    t = cc.tasks(studies)  # part 3 excluded; unsure tag excluded; bypass study excluded
    assert [(u, side, v.path) for u, side, v in t] == [("a", "left", "0/a/v1.npy")]


def test_bpm_and_summary():
    assert cc.bpm([0, 15, 30]) == pytest.approx(60.0) and cc.bpm([4]) is None
    studies = [cs.Study("a", 0, 0.0, 0.0, None, False, ()),
               cs.Study("b", 12, 12.0, 0.0, None, False, ())]  # fmt: skip
    rows = [
        {"key": "1", "study": "a", "side": "left", "ffr": 0.91, "marked": 0, "flagged": False,
         "ed_m0": [0, 10, 20], "ed_m2": [0, 15, 30]},
        {"key": "2", "study": "a", "side": "right", "ffr": 0.78, "marked": 0, "flagged": True},
        {"key": "3", "study": "b", "side": "left", "ffr": 0.70, "marked": 1, "flagged": True},
        {"key": "4", "study": "b", "side": "left", "error": "ValueError: no vessel"},
    ]  # fmt: skip
    s = cc.summarise(studies, rows)
    assert s["false_alarm"]["flagged"] == {"n": 2, "k": 1, "share": 0.5}
    assert s["sensitivity"]["flagged"]["k"] == 1 and s["by_syntax"]["8–14"]["k"] == 1
    assert s["zero_runs_ffr"]["n"] == 2 and s["zero_runs_ffr"]["le_080"] == pytest.approx(0.5)
    assert s["ed_m0"]["median_bpm"] == pytest.approx(90.0) and s["ed_m2"]["in_range"] == 1.0
    assert s["errors"] == {"ValueError": 1}
