"""Vessel identity labels from ARCADE's SYNTAX segments (P4k-G). Torch-free.

ARCADE (CC0) labels every visible coronary segment with its SYNTAX number. The segments are
grouped into four vessels (docs/PLAN.md, P4k-G); each pixel's class is its segment's vessel
(0 = background). Where segments overlap, the smaller one is drawn last.
"""

from pathlib import Path

import numpy as np

from vstent.data.arcade import load_coco, load_split, polygons_to_mask

CLASS_NAMES = ("background", "RCA", "left main", "LAD", "LCx")
N_CLASSES = len(CLASS_NAMES)
RCA, LM, LAD, LCX = 1, 2, 3, 4
GROUPS = {
    RCA: ("1", "2", "3", "4", "16", "16a", "16b", "16c"),
    LM: ("5",),
    LAD: ("6", "7", "8", "9", "10", "9a", "10a"),
    LCX: ("11", "12", "13", "14", "15", "12a", "12b", "14a", "14b"),
}
SEGMENT_CLASS = {name: k for k, names in GROUPS.items() for name in names}
LEFT = (LM, LAD, LCX)


def label_map(annotations: list[dict], names: dict[int, str], height: int,
              width: int) -> np.ndarray:  # fmt: skip
    """uint8 (H, W) vessel classes from one image's SYNTAX annotations (``names``: category id
    → SYNTAX name). Segments that are not in :data:`SEGMENT_CLASS` are left out."""
    out = np.zeros((height, width), np.uint8)
    drawn = [a for a in annotations if names.get(a["category_id"]) in SEGMENT_CLASS]
    for a in sorted(drawn, key=lambda a: -float(a.get("area", 0.0))):  # small segments on top
        m = polygons_to_mask(a["segmentation"], height, width).astype(bool)
        out[m] = SEGMENT_CLASS[names[a["category_id"]]]
    return out


def load_labels(raw_dir: str | Path, split: str,
                max_images: int | None = None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Images (N, H, W) uint8 and vessel classes (N, H, W) uint8 of one ARCADE ``syntax``
    split, in :func:`~vstent.data.arcade.load_split`'s order (COCO image id)."""
    base = load_split(raw_dir, "syntax", split, max_images)
    coco = load_coco(raw_dir, "syntax", split)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    h, w = base.images.shape[1:]
    labels = np.stack([label_map(by_image.get(int(i), []), names, h, w)
                       for i in base.image_ids])  # fmt: skip
    return base.images, labels


def side_of(labels: np.ndarray) -> str | None:
    """The run's side: ``"right"`` when RCA pixels outnumber the left tree's (left main, LAD,
    LCx), ``"left"`` when fewer, ``None`` without vessel pixels."""
    right = int((labels == RCA).sum())
    left = int(np.isin(labels, LEFT).sum())
    if right == left == 0:
        return None
    return "right" if right > left else "left"
