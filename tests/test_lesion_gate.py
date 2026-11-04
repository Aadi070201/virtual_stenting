"""P4q: a narrowing counts only where the lesion detector agrees; the development and test
rules."""

import numpy as np
import pytest

from vstent.imaging.lesion_gate import LesionGate, apply_gate
from vstent.validation import lesion_gate_check as lg


def _tube():
    from vstent.imaging.synthetic import render_tube, sine_centerline, stenosis_diameter

    pts, s = sine_centerline((0.0, 0.0), 180.0, angle_deg=15.0, amplitude_px=15.0)
    pts = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2 + 120.0
    tube = render_tube((240, 240), pts, s, stenosis_diameter(s, 12.0, 0.6, s[-1] / 2, 30.0),
                       blur_sigma=1.0, noise_std=0.02)  # fmt: skip
    return tube, pts


def _blob(at, shape=(240, 240), radius=5.0):
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    return ((rr - at[0]) ** 2 + (cc - at[1]) ** 2 <= radius**2).astype(np.float32)


def test_gate_keeps_a_confirmed_narrowing_and_removes_the_rest():
    from vstent.inference.pipeline import analyse_vessel
    from vstent.physics.solver import PhysicsParams

    tube, pts = _tube()
    params = PhysicsParams()
    a = analyse_vessel(tube.image, pts[0], pts[-1], params=params, wire=False)
    q = a.qca
    deepest = max(q.profile.lesions(), key=lambda les: les.max_ds_pct)
    assert deepest.max_ds_pct > 40
    mid = q.centerline.points[int(np.argmin(np.abs(q.s_mm - deepest.s_mld * 1e3)))]
    on = LesionGate(lambda img: _blob(mid), 0.5)
    off = LesionGate(lambda img: _blob((5, 5)), 0.5)
    kept, pairs_on = apply_gate(q, tube.image, on)
    gone, pairs_off = apply_gate(q, tube.image, off)
    assert pairs_on and any(ok for _c, ok in pairs_on)
    assert kept.profile.r.min() == q.profile.r.min()  # the confirmed narrowing stays
    assert not any(ok for _c, ok in pairs_off)
    assert max((les.max_ds_pct for les in gone.profile.lesions()), default=0.0) < 30
    assert np.all(gone.profile.r >= q.profile.r - 1e-12)
    assert gone.diameters.diameter_px.min() > q.diameters.diameter_px.min()
    # through the pipeline: the physics sees the healthy vessel, and the counts are kept
    b = analyse_vessel(tube.image, pts[0], pts[-1], params=params, wire=False,
                       image=tube.image, lesion_gate=off)  # fmt: skip
    c = analyse_vessel(tube.image, pts[0], pts[-1], params=params, wire=False,
                       image=tube.image)  # fmt: skip
    assert b.gate is not None and b.gate[0] == 0 and b.gate[1] >= 1 and c.gate is None
    assert b.solver.ffr[-1] > c.solver.ffr[-1]


def test_no_detection_confirms_nothing():
    from vstent.imaging.lesion_gate import confirmed
    from vstent.imaging.lesion_patch import candidates
    from vstent.inference.pipeline import analyse_vessel

    tube, pts = _tube()
    q = analyse_vessel(tube.image, pts[0], pts[-1], wire=False).qca
    cands = candidates(q)
    assert cands and confirmed(q, np.zeros((240, 240), bool), cands) == [False] * len(cands)


def _row(key, study, side, **arms):
    return {"key": key, "study": study, "side": side,
            "arms": {a: {"flagged": f, "ffr": ffr, "marked": int(f), "max_ds": 0.0,
                         "length_mm": 50.0, "heartbeats": 1, "gate": g}
                     for a, (f, ffr, g) in arms.items()}}  # fmt: skip


def test_summary_and_rules():
    from vstent.data import cardiosyntax as cs

    studies = [cs.Study(u, sx, sx, 0.0, None, False, ())
               for u, sx in (("a", 0.0), ("b", 0.0), ("c", 10.0), ("d", 20.0))]  # fmt: skip
    rows = [_row("1", "a", "left", product=(True, 0.75, None), gated=(False, 0.90, [0, 2])),
            _row("2", "b", "left", product=(True, 0.85, None), gated=(False, 0.92, [1, 2])),
            _row("3", "c", "left", product=(True, 0.70, None), gated=(True, 0.72, [1, 1])),
            _row("4", "d", "left", product=(True, 0.60, None), gated=(True, 0.60, [2, 2])),
            {"key": "5", "study": "d", "side": "left", "error": "ValueError: x"}]  # fmt: skip
    s = lg.summarise(studies, rows, ["product", "gated"], base="product")
    p, g = s["arms"]["product"], s["arms"]["gated"]
    assert (s["sides"], s["zero_sides"], s["positive_sides"]) == (4, 2, 2)
    assert (p["false_alarm"], p["sensitivity"], p["j"]) == (1.0, 1.0, 0.0)
    assert (g["false_alarm"], g["sensitivity"], g["j"]) == (0.0, 1.0, 1.0)
    assert g["dj"] == 1.0 and g["confirmed"] == pytest.approx(4 / 7) and p["confirmed"] is None
    assert p["zero_runs_le_080"] == 0.5 and g["zero_runs_le_080"] == 0.0
    assert [b["n"] for b in g["by_bin"]] == [0, 1, 1]
    td = lg.test_decision(s, "gated")
    assert td["drop"] == 1.0 and td["criteria"][0]
    dev = {"arms": {"product": {"j": 0.02}, "A": {"j": 0.20}, "B": {"j": 0.21}}}
    assert lg.dev_decision(dev)["chosen"] == "A"  # within 0.02: one model
    dev["arms"]["B"]["j"] = 0.30
    assert lg.dev_decision(dev)["chosen"] == "B"
    dev["arms"] = {"product": {"j": 0.02}, "A": {"j": 0.08}, "B": {"j": 0.11}}
    assert lg.dev_decision(dev)["chosen"] is None  # no variant gains 0.10


def test_rows_merge_with_p4p():
    base = lg.from_p4p([{"key": "1", "study": "a", "side": "left", "flagged": True, "ffr": 0.7,
                         "marked": 1, "max_ds": 60.0, "length_mm": 80.0, "heartbeats": 3,
                         "profile": {}},
                        {"key": "2", "study": "a", "side": "left",
                         "error": "ValueError: x"}])  # fmt: skip
    new = [_row("1", "a", "left", A=(False, 0.9, [0, 1])), _row("2", "a", "left")]
    m = {r["key"]: r for r in lg.merge(base, new)}
    assert set(m["1"]["arms"]) == {"product", "A"} and m["1"]["arms"]["product"]["gate"] is None
    assert "error" in m["2"]


def test_product_without_detector_has_no_gate():
    from vstent.inference.product import ProductConfig, product_gate

    assert product_gate(ProductConfig()) is None
    cfg = ProductConfig(lesion_detector=["a", "b"], detector_threshold=0.05)
    assert cfg.lesion_detector == ("a", "b")
