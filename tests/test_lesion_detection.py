"""P4k-E: detections against expert boxes, the operating point and the app rule."""

from pathlib import Path

import numpy as np

from vstent.data.lesion_boxes import BoxImage
from vstent.validation import lesion_detection as ld


def test_counts_operating_point_and_rule():
    it = BoxImage(Path("x.png"), ((10, 10, 8, 8), (40, 40, 6, 6)), "cadica", "p1", "val",
                  grades=("p70_90", "p90_98"))  # fmt: skip
    prob = np.zeros((64, 64))
    prob[12:16, 12:16] = 0.9  # a confident detection in the first box
    prob[42:44, 42:44] = 0.3  # a weak one in the second
    prob[55:58, 2:5] = 0.6  # a false alarm
    c = ld.frame_counts(prob, it)
    k01, k05 = ld.THRESHOLDS.index(0.1), ld.THRESHOLDS.index(0.5)
    assert c["found"][k01] == [True, True] and c["found"][k05] == [True, False]
    assert c["detections"][k01] == 3 and c["outside"][k01] == 1
    free = BoxImage(Path("y.png"), (), "cadica", "p1", "val", lesion_free=True)
    rows = [{"source": "cadica", "patient": "p1", "role": "val", "lesion_free": False,
             "grades": list(it.grades), **c},
            {"source": "cadica", "patient": "p1", "role": "val", "lesion_free": True,
             "grades": [], **ld.frame_counts(prob * 0.5, free)}]  # fmt: skip
    k = ld.operating_point(rows)
    assert ld.THRESHOLDS[k] == 0.3  # the highest t still finding both boxes
    d = ld.dev_decision(rows)
    assert d["pooled"]["sensitivity"] == 1.0 and d["pooled"]["by_grade"]["p90_98"]["n"] == 1
    assert d["pooled"]["alarms_per_free_frame"] == 2.0 and d["shown_in_app"]  # 0.45 and 0.30
    assert ld.test_results(rows, k)["cadica"]["boxes"] == 2


def test_candidate_table_reports_every_candidate():
    it = BoxImage(Path("x.png"), ((10, 10, 8, 8),), "cadica", "p1", "val")
    free = BoxImage(Path("y.png"), (), "cadica", "p1", "val", lesion_free=True)
    prob = np.zeros((64, 64))
    prob[12:16, 12:16] = 0.3  # found at t = 0.05, missed at t = 0.5
    prob[55:58, 2:5] = 0.6
    base = {"source": "cadica", "patient": "p1", "role": "val", "grades": []}
    rows = [{**base, "lesion_free": False, **ld.frame_counts(prob, it)},
            {**base, "lesion_free": True, **ld.frame_counts(prob, free)}]  # fmt: skip
    lines = ld.candidate_table({"a": rows})
    assert lines[0].count("|") == 6 and lines[2] == "| a | 100.0% | 2.00 | 0.0% | 1.00 |"
