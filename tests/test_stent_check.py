"""P4m: the virtual-stenting check's summary."""

from vstent.validation import stent_check as sc


def _row(pre, post=None, covered=False, reaches=False):
    arm = {"pre": pre, "planned": post is not None, "in_reach": True}
    if post is not None:
        arm.update(post=post, covered=covered, reaches=reaches, n_stents=1, length_mm=18.0)
    return {"product": arm, "wire": arm, "pipeline": arm}


def test_summary_and_targets():
    rows = [_row(0.7, 0.92, True, True), _row(0.75, 0.89, True), _row(0.6, 0.78, False),
            _row(0.9), {"error": "no vessel in the box"}]  # fmt: skip
    s = sc.summarise(rows)
    a = s["arms"]["product"]
    assert s["n_measured"] == 4 and a["significant"] == 3 and a["planned"] == 3
    assert abs(a["covered"] - 2 / 3) < 1e-9 and abs(a["post_low"] - 1 / 3) < 1e-9
    assert a["targets"]["post-PCI mean 0.88–0.94"] is False  # mean 0.863
    assert not a["targets"]["covered ≥ 90%"]
    assert any("Stent on the experts' lesion" in x for x in sc.section("dev", s))
