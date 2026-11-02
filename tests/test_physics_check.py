"""P4k-C: the outflow check's summary and pre-registered rule."""

import pytest

from vstent.validation import physics_check as pc


def _free(none, murray):
    return [{"ffr_none": a, "ffr_murray": b} for a, b in zip(none, murray, strict=True)]


def _graded(category, none, murray):
    return [{"track": f"{category}{k}", "category": category, "ffr_none": a, "ffr_murray": b}
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
