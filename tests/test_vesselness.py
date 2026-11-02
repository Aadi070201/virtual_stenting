"""P4j: hybrid segmentation (vesselness vessels connected to the model's mask)."""

import numpy as np

from vstent.imaging import vesselness as vs
from vstent.validation import vessel_coverage as vc


def _scene():
    img = np.full((160, 200), 200, np.uint8)
    img[78:84, 10:190] = 90  # a vessel
    img[40:44, 60:140] = 90  # a separate structure, not connected to the vessel
    prob = np.zeros(img.shape, np.float32)
    prob[78:84, 10:60] = 0.9  # the model saw only the vessel's first part
    return img, prob


def test_added_vessels_connect_to_the_model():
    img, prob = _scene()
    v = vs.vesselness(img)
    add = vs.added_mask(prob, v, 0.1)
    assert add[80, 120] and add[80, 180]  # the rest of the vessel is added
    assert not add[42, 100]  # the unconnected structure is not
    out = vs.hybrid_probability(img, prob, 0.1, v)
    assert out[80, 120] == vs.ADDED_PROB and out[80, 30] == 0.9 and out[42, 100] == 0.0
    assert not vs.added_mask(np.zeros_like(prob), v, 0.1).any()


def test_coverage_rule():
    boxes = {"n_boxes": 10, "model": 0.4, "0.05": 0.7, "0.1": 0.65, "0.15": 0.55, "0.2": 0.5}
    arc = {"n_images": 3,
           "model": {"recall": 0.8, "added_px_per_image": 0, "added_near_share": None},
           "0.05": {"recall": 0.85, "added_px_per_image": 900, "added_near_share": 0.3},
           "0.1": {"recall": 0.84, "added_px_per_image": 600, "added_near_share": 0.6},
           "0.15": {"recall": 0.83, "added_px_per_image": 400, "added_near_share": 0.7},
           "0.2": {"recall": 0.82, "added_px_per_image": 300, "added_near_share": 0.8}}  # fmt: skip
    c = vc.choose(boxes, arc, vs.THRESHOLDS)
    assert c["threshold"] == 0.1 and c["eligible"] == [0.1, 0.15, 0.2] and c["adopted"]
    arc["0.1"]["recall"] = arc["0.15"]["recall"] = arc["0.2"]["recall"] = 0.7
    assert vc.choose(boxes, arc, vs.THRESHOLDS)["threshold"] is None


def test_box_coverage_in_process():
    img, prob = _scene()

    class Seg:
        def predict_model(self, image):
            return prob

    out = vc.box_coverage([(img, [(150, 70, 20, 20)])], Seg(), (0.1,), 1)
    assert out == {"n_boxes": 1, "model": 0.0, "0.1": 1.0}
