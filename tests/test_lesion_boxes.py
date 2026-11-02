"""P4k-E: the detector's box masks."""

import numpy as np

from vstent.data import lesion_boxes as lb


def test_box_masks_are_drawn_on_demand():
    m = lb.box_mask([(10.2, 20.0, 5.0, 4.0), (-3.0, -3.0, 6.0, 6.0)], size=32)
    assert m.dtype == np.uint8 and m[20:24, 10:16].all() and m[0:3, 0:3].all()
    assert m.sum() == 4 * 6 + 3 * 3
    masks = lb.BoxMasks([[(1, 1, 2, 2)], []], size=8)
    assert len(masks) == 2 and [int(x.sum()) for x in masks] == [4, 0]
