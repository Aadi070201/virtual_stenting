"""P4k-B: box crossing and the adoption rule."""

import numpy as np

from vstent.validation import target_path as tp


def test_crosses_needs_five_points_in_one_box():
    pts = np.c_[np.full(20, 50.0), np.arange(20.0)]  # row 50, cols 0..19
    assert tp.crosses(pts, [(10, 40, 10, 20)])  # cols 10..20: 10 points
    assert not tp.crosses(pts, [(16, 40, 10, 20)])  # cols 16..19: 4 points
    assert not tp.crosses(pts, [(0, 60, 30, 10)])  # other rows


def test_summary_and_rule():
    rows = [{"patient": f"p{k}", "trunk": k < 4, "seeking": k < 6, "changed": k >= 4,
             "score": 60.0} for k in range(10)] + [{"patient": "p9", "error": "x"}]  # fmt: skip
    s = tp.summarise(rows)
    assert s["measured"] == 10 and s["trunk"]["share"] == 0.4 and s["seeking"]["share"] == 0.6
    assert abs(s["gain"] - 0.2) < 1e-12 and s["adopt"] and s["only_seeking"] == 2
    rows[5]["seeking"] = False
    rows[4]["seeking"] = False
    assert not tp.summarise(rows)["adopt"]  # +0 points
    assert "adopt" in " ".join(tp.section("Dev", s, rule=True))
