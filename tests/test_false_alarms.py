"""P4p: reading rules on saved vessel profiles, the side sample."""

import pytest

from vstent.data import cardiosyntax as cs
from vstent.validation import false_alarms as fa


def _profile(ref, ffr, lesions):
    return {"s_mm": [float(i) for i in range(len(ref))], "d_mm": list(ref), "ref_mm": list(ref),
            "ffr": list(ffr), "lesions": lesions}  # fmt: skip


def test_wire_point_and_rules():
    # reference 3 mm for 3 points, then 1.6 mm; a 60% lesion in the thin part at 4 mm
    p = _profile([3.0, 3.0, 3.0, 1.6, 1.6, 1.6], [1.0, 0.98, 0.96, 0.90, 0.75, 0.70],
                 [[3.5, 4.0, 4.5, 60.0, 0.6, 1.6]])  # fmt: skip
    assert fa.wire_mm(p, 2.0) == 2.0 and fa.wire_mm(p, 1.5) == 5.0 and fa.wire_mm(p, 4.0) == 5.0
    row = {"profile": p}
    assert fa.read(row, fa.RULES[0]) == (True, pytest.approx(0.70), 60.0)  # product
    assert fa.read(row, fa.RULES[1])[0] is False  # the thin-vessel lesion does not count
    assert fa.read(row, fa.RULES[2])[1] == pytest.approx(0.96)  # FFR at the 2 mm point
    marked, ffr, ds = fa.read(row, fa.RULES[3])
    assert (marked, ds) == (False, 0.0) and ffr == pytest.approx(0.96)
    assert fa.read(row, fa.RULES[4])[0] is True  # 1.6 mm ≥ 1.5 mm counts


def test_summary_per_rule():
    studies = [cs.Study("a", 0, 0.0, 0.0, None, False, ()),
               cs.Study("b", 12, 12.0, 0.0, None, False, ())]  # fmt: skip
    thin = _profile([3.0, 3.0, 1.6, 1.6], [1.0, 0.97, 0.80, 0.70],
                    [[2.5, 3.0, 3.5, 60.0, 0.6, 1.6]])  # fmt: skip
    proximal = _profile([3.0, 3.0, 3.0], [1.0, 0.85, 0.70], [[0.5, 1.0, 1.5, 70.0, 0.9, 3.0]])
    rows = [{"key": "1", "study": "a", "side": "left", "flagged": True, "profile": thin},
            {"key": "2", "study": "b", "side": "left", "flagged": True, "profile": proximal},
            {"key": "3", "study": "b", "side": "left",
             "error": "ValueError: no vessel"}]  # fmt: skip
    s = fa.summarise(studies, rows)
    assert (s["measured"], s["sides"], s["zero_sides"], s["positive_sides"]) == (2, 2, 1, 1)
    product, _, _, both, _ = s["rules"]
    assert product["false_alarm"] == 1.0 and product["sensitivity"] == 1.0
    assert both["false_alarm"] == 0.0 and both["sensitivity"] == 1.0
    assert both["zero_runs_le_080"] == 0.0 and product["zero_runs_le_080"] == 1.0
    assert s["zero_marks"] == 1 and s["zero_marks_by_ref"][1]["n"] == 1  # 1.5–2 mm
    assert s["product_flag_agrees"] == 1.0


def test_report_names_the_commit(tmp_path):
    studies = [cs.Study("a", 0, 0.0, 0.0, None, False, ())]
    p = _profile([3.0, 3.0], [1.0, 0.9], [])
    s = fa.summarise(studies, [{"key": "1", "study": "a", "side": "left", "flagged": False,
                                "profile": p}])  # fmt: skip
    path = fa.write_report(tmp_path / "r.md", s, product={},
                           git={"git_commit": "abc123", "git_dirty": False})  # fmt: skip
    assert "Commit `abc123` (dirty: False)" in path.read_text(encoding="utf-8")


def test_sample_tasks_takes_whole_sides(monkeypatch):
    v = lambda path: cs.Video(path, "LCA", 0.0, 20, 0.0, 0.0)  # noqa: E731
    studies = [cs.Study(u, sx, sx, 0.0, None, False, (v(f"0/{u}/1.npy"), v(f"0/{u}/2.npy")))
               for u, sx in (("a", 0.0), ("b", 0.0), ("c", 5.0))]  # fmt: skip
    t = fa.sample_tasks(studies, n_zero=1, n_positive=1, seed=0)
    sides = {(u, side) for u, side, _ in t}
    assert len(t) == 4 and len(sides) == 2 and ("c", "left") in sides
    assert t == fa.sample_tasks(studies, n_zero=1, n_positive=1, seed=0)


def test_profile_of_a_measured_tube():
    import numpy as np

    from vstent.imaging.synthetic import render_tube, sine_centerline, stenosis_diameter
    from vstent.inference.pipeline import analyse_vessel
    from vstent.physics.solver import PhysicsParams
    from vstent.validation.external import profile_of

    pts, s = sine_centerline((0.0, 0.0), 180.0, angle_deg=15.0, amplitude_px=15.0)
    pts = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2 + 120.0
    tube = render_tube((240, 240), pts, s, stenosis_diameter(s, 12.0, 0.6, s[-1] / 2, 30.0),
                       blur_sigma=1.0, noise_std=0.02)  # fmt: skip
    a = analyse_vessel(tube.image, pts[0], pts[-1], params=PhysicsParams(), wire=False)
    p = profile_of(a)
    n = len(p["s_mm"])
    assert p["s_mm"][0] == 0.0 and p["s_mm"][1] == 1.0 and n == int(a.qca.s_mm[-1]) + 1
    assert all(len(p[k]) == n for k in ("d_mm", "ref_mm", "ffr"))
    assert p["ffr"][-1] == pytest.approx(float(a.solver.ffr[-1]), abs=0.01)
    deepest = max(p["lesions"], key=lambda les: les[3])
    assert deepest[0] <= deepest[1] <= deepest[2] and deepest[3] > 40
    assert deepest[5] == pytest.approx(float(np.median(a.qca.reference_diameter_mm)), rel=0.2)
