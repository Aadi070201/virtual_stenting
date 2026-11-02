"""P4k-C: the outflow check's summary and pre-registered rule."""

import pytest

from vstent.validation import physics_check as pc


def _free(none, murray):
    return [{"ffr_none": a, "ffr_murray": b, "ffr_wire": b}
            for a, b in zip(none, murray, strict=True)]  # fmt: skip


def _graded(category, none, murray):
    return [{"track": f"{category}{k}", "category": category, "ffr_none": a, "ffr_murray": b,
             "ffr_wire": a}
            for k, (a, b) in enumerate(zip(none, murray, strict=True))]  # fmt: skip


def test_rule_adopts_only_when_both_conditions_hold():
    free = _free([0.7, 0.75, 0.9, 0.95], [0.85, 0.88, 0.95, 0.97])  # 50% → 0% ≤ 0.80
    graded = (_graded("p50_70", [0.7, 0.9], [0.75, 0.95])
              + _graded("p70_90", [0.7, 0.75], [0.72, 0.78])
              + _graded("p90_98", [0.5, 0.6], [0.55, 0.65]))  # fmt: skip
    s = pc.summarise(free + [{"error": "x"}], graded)
    n, m = s["outflow"]["none"], s["outflow"]["murray"]
    assert n["free_share"] == 0.5 and m["free_share"] == 0.0 and s["n_free_failed"] == 1
    assert m["free_distance"] == pytest.approx(0.13) and n["free_distance"] == pytest.approx(0.37)
    assert n["grades"]["p50_70"]["share"] == 0.5 and m["fame_mad"] == pytest.approx(
        (0.15 + 0.2 + 0.04) / 3)  # fmt: skip
    assert s["adopt"]
    worse = _graded("p50_70", [0.7], [0.9]) + _graded("p70_90", [0.7], [0.9]) + _graded(
        "p90_98", [0.7], [0.9])  # fmt: skip
    assert not pc.summarise(free, worse)["adopt"]  # FAME distance grows by > 10 points
    assert pc.summarise(free, [])["reason"] == "a statistic could not be computed"
    lines = pc.section("dev", s)
    assert any("Decision: adopt `murray`" in x for x in lines)


def test_wire_arm():
    """P4k-C2: the FFR read at the last point with a reference diameter ≥ 2.0 mm."""
    import numpy as np

    from vstent.imaging.qca import wire_index
    from vstent.physics.geometry import VesselProfile
    from vstent.physics.solver import PhysicsParams

    assert wire_index(np.array([1.5e-3, 1.2e-3, 0.99e-3, 0.8e-3])) == 1
    assert wire_index(np.array([0.9e-3, 0.8e-3])) == 1  # nothing to cut
    s = np.linspace(0.0, 0.12, 241)
    r = np.linspace(1.6e-3, 0.5e-3, s.size)  # tapers below 2 mm diameter at about 2/3
    f = pc.distal_ffr(VesselProfile(s, r, r), PhysicsParams())
    assert f["wire"] > f["none"] and f["murray"] > f["none"]
    free = _free([0.7, 0.75, 0.9, 0.95], [0.85, 0.88, 0.95, 0.97])
    s2 = pc.summarise(free, _graded("p50_70", [0.7], [0.9]) + _graded("p70_90", [0.7], [0.9])
                      + _graded("p90_98", [0.7], [0.9]))  # fmt: skip
    assert s2["wire_adopt"] and not s2["adopt"]  # wire keeps the lesions' readings here
    assert any("adopt the wire position" in x for x in pc.section("dev", s2, "c2"))


def test_c3_rule_needs_a_small_lesion_free_share():
    """P4k-C3: C's rule against the product, plus a lesion-free share ≤ 25%."""
    product = {"free_distance": 0.39, "fame_mad": 0.23, "free_share": 0.52}
    s = {"outflow": {
        "none": {"free_share": 0.30, "free_distance": 0.17, "fame_mad": 0.20},
        "murray": {"free_share": 0.12, "free_distance": 0.01, "fame_mad": 0.30},
        "wire": {"free_share": 0.10, "free_distance": 0.03, "fame_mad": 0.40},
    }}  # fmt: skip
    r = pc.c3_rules(product, s)
    assert not r["none"]["qualifies"]  # 30% > 25%
    assert r["murray"]["qualifies"] and abs(r["murray"]["score"] - 0.31) < 1e-9
    assert not r["wire"]["qualifies"]  # FAME 17 points worse


def test_at_wire_cuts_the_measured_vessel():
    """D68: the QCA cut where the reference diameter falls below 2 mm; every per-point array
    cut alike, the reference kept from the whole path."""
    import numpy as np

    from vstent.imaging.qca import at_wire, run_qca

    prob = np.zeros((100, 420))
    for c in range(10, 410):
        half = 7.0 - 4.5 * (c - 10) / 400  # 14 px wide (3.5 mm at 0.25) → 5 px (1.25 mm)
        prob[int(50 - half) : int(50 + half) + 1, c] = 1.0
    q = run_qca(prob, (50, 15), (50, 405), mm_per_px=0.25)
    cut, k = at_wire(q)
    n = len(q.profile.s)
    assert 2 < k < n - 1 and len(cut.profile.s) == k + 1
    assert len(cut.centerline.points) == k + 1 and len(cut.diameters.diameter_px) == k + 1
    ref = 2e3 * cut.profile.reference_radius()
    assert ref[-1] >= 2.0 and np.allclose(ref, 2e3 * q.profile.reference_radius()[: k + 1])
    wide = np.zeros((60, 200))
    wide[22:38, 10:190] = 1.0  # 4 mm everywhere: nothing to cut
    w = run_qca(wide, (30, 15), (30, 185), mm_per_px=0.25)
    assert at_wire(w)[0] is w
