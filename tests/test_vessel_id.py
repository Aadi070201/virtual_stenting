"""P4k-G: vessel identity labels, the multi-class loss and Dice, the run's side."""

import numpy as np
import pytest

from vstent.data import vessel_id as vid


def test_label_map_groups_and_overlaps():
    names = {1: "6", 2: "9", 3: "2", 4: "stenosis"}
    big = {"category_id": 1, "area": 400.0, "segmentation": [[0, 0, 20, 0, 20, 20, 0, 20]]}
    small = {"category_id": 3, "area": 25.0, "segmentation": [[5, 5, 10, 5, 10, 10, 5, 10]]}
    other = {"category_id": 4, "area": 4.0, "segmentation": [[30, 30, 32, 30, 32, 32, 30, 32]]}
    lab = vid.label_map([small, big, other], names, 40, 40)
    assert lab[2, 2] == vid.LAD and lab[7, 7] == vid.RCA  # the smaller segment is on top
    assert lab[31, 31] == 0  # not a SYNTAX segment
    assert vid.side_of(lab) == "left" and vid.side_of(np.zeros((4, 4), np.uint8)) is None
    assert vid.SEGMENT_CLASS["16b"] == vid.RCA and vid.SEGMENT_CLASS["12a"] == vid.LCX


def test_class_loss_and_dice():
    torch = pytest.importorskip("torch")
    from vstent.imaging.seg_train import class_dice, class_loss

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:5, 2:5] = 3.0
    good = torch.full((1, 5, 8, 8), -5.0)
    good[:, 0] = 5.0
    good[0, 0, 2:5, 2:5], good[0, 3, 2:5, 2:5] = -5.0, 5.0
    bad = torch.zeros(1, 5, 8, 8)
    assert float(class_loss(good, target, 1.0)) < float(class_loss(bad, target, 1.0))
    pred = target[0, 0].numpy().astype(np.uint8)
    d = class_dice(pred[None], pred[None], 5)
    assert d[2] == 1.0 and np.isnan(d[0])
