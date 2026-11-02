"""P4k-G: vessel identity scores, side agreement and the product rule."""

import numpy as np

from vstent.validation import vessel_id_eval as ve


def test_scores_agreement_and_rule():
    lab = np.zeros((1, 10, 10), np.uint8)
    lab[0, 2:4, :] = 1  # RCA
    lab[0, 6:8, :] = 3  # LAD
    probs = np.zeros((5, 10, 10))
    probs[0] = 1.0
    probs[1, 2:4, :] = 2.0
    probs[3, 6:8, :5] = 2.0
    probs[4, 6:8, 5:] = 2.0  # half the LAD called LCx
    s = ve.arcade_scores(np.zeros((1, 10, 10), np.uint8), lab, lambda img: probs)
    assert s["accuracy"] == 0.75 and s["dice"]["RCA"] == 1.0 and s["dice"]["LAD"] == 2 * 10 / 30
    rows = [{"truth": "left", "predicted": "left"}, {"truth": "right", "predicted": "right"},
            {"truth": "right", "predicted": None}]  # fmt: skip
    a = ve.agreement(rows)
    assert a["agree"] == 2 and a["n"] == 3 and a["per_side"] == {"left": 1, "right": 1}
    assert not ve.decision({"accuracy": 0.95}, a)["used_in_product"]
    assert ve.decision({"accuracy": 0.95}, {"share": 0.96})["used_in_product"]
