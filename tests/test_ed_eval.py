"""P4l: ED evaluation helpers (reproducibility without labels, summary, choice)."""

import cv2
import numpy as np
import pytest

from vstent.validation import ed_eval


def _tree(shift=(0.0, 0.0), bend=0.0) -> np.ndarray:
    img = np.zeros((128, 128), np.float32)
    dx, dy = shift
    pts = [((20, 20), (100, 40)), ((60, 30), (50, 110)), ((30, 90), (110, 100 + bend))]
    for (x0, y0), (x1, y1) in pts:
        cv2.line(img, (int(x0 + dx), int(y0 + dy)), (int(x1 + dx), int(y1 + dy)), 1.0, 5)
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def test_beat_distance_registers_a_shift():
    a = _tree()
    assert ed_eval.beat_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert ed_eval.beat_distance(a, _tree(shift=(3, 2))) < 0.6  # registered away
    assert ed_eval.beat_distance(a, _tree(bend=25)) > ed_eval.beat_distance(a, _tree(shift=(3, 2)))


def test_beat_distance_needs_vessel():
    assert ed_eval.beat_distance(np.zeros((64, 64), np.float32), _tree()[:64, :64]) is None


def test_reproducibility_ratio(monkeypatch):
    # pairs of the ED frames repeat exactly; every other pair is 1 apart
    ed = (5, 15, 25)
    monkeypatch.setattr(ed_eval, "beat_distance",
                        lambda a, b: 0.0 if (a in ed and b in ed) else 1.0)  # fmt: skip
    maps = list(range(30))
    well = np.ones(30, bool)
    rep = ed_eval.reproducibility(maps, ed, well)
    assert rep["pairs"] == 2 and rep["ratio"] == 0.0 and rep["shifted"] == 1.0
    late = ed_eval.reproducibility(maps, (6, 16, 26), well)  # one frame late: its −1 is ED
    assert late["shifted"] == pytest.approx(0.5) and late["ratio"] == pytest.approx(2.0)
    assert ed_eval.reproducibility(maps, (8, 18, 28), well)["ratio"] == pytest.approx(1.0)
    well[26] = False  # the second pair loses a shifted frame
    assert ed_eval.reproducibility(maps, ed, well)["pairs"] == 1
    assert ed_eval.reproducibility(maps, (5,), well) is None


def _row(p, k, r0, r2, m0=(0, 8, 16), m2=(1, 9, 17)):
    rep = lambda r: None if r is None else {"pairs": 2, "distance": r, "shifted": 1.0, "ratio": r}  # noqa: E731
    return {"key": f"{p}_v{k}", "patient": p, "M0": list(m0), "M2": list(m2),
            "rep_M0": rep(r0), "rep_M2": rep(r2)}  # fmt: skip


def test_summary_and_choice():
    rows = [_row("p1", 1, 0.8, 0.6), _row("p2", 1, 0.9, 0.7), _row("p3", 1, 0.7, None, m2=(3,))]
    s = ed_eval.summarise_real(rows, n_boot=200)
    assert s["paired_runs"] == 2 and s["patients"] == 2
    assert s["M2"]["mean_ratio"] == pytest.approx(0.65)
    assert s["mean_difference"] == pytest.approx(-0.2) and s["m2_no_worse"]
    assert s["M2"]["under_two"] == 1 and s["M0"]["median_bpm"] == pytest.approx(75.0)
    syn = {"M0": {"rate": 0.78}, "M2": {"rate": 0.93}}
    assert ed_eval.choose(syn, s)[0] == "M2"
    assert ed_eval.choose({"M0": {"rate": 0.95}, "M2": {"rate": 0.93}}, s)[0] == "M0"
    worse = ed_eval.summarise_real([_row("p1", 1, 0.5, 0.9)], n_boot=50)
    assert ed_eval.choose(syn, worse)[0] == "M0"
