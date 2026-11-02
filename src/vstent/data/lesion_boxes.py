"""Whole-image lesion detector data (P4k-E): every expert lesion box on real angiograms, and
lesion-free frames. Torch-free.

Roles (docs/PLAN.md, P4k-E):
- **CADICA** (P4b patient split): every expert box (all grades) on the annotated frames, and
  every 4th selected frame of the lesion-free runs (no boxes). Development patients train,
  except 5 (seed 0) held out for validation; test patients are the test.
- **ARCADE stenosis:** train / val / test as published.
- **Kemerovo:** patients split 50 / 50 (seed 0). The first half trains, except 10 of its
  patients (seed 0) held out for validation; the second half is the test. Every 4th frame of a
  series (its frames are consecutive and nearly identical).

Images are held at :data:`SIZE` px (greyscale uint8) and boxes scaled with them; the filled
box masks are drawn when a sample is used (:class:`BoxMasks`).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SIZE = 512
SEED = 0
CADICA_VAL_PATIENTS = 5
KEMEROVO_VAL_PATIENTS = 10
FRAME_STEP = 4

Box = tuple[float, float, float, float]  # x, y, w, h (px)


@dataclass(frozen=True)
class BoxImage:
    """One frame: its file, expert boxes, source, patient, role (train / val / test) and
    whether it comes from a lesion-free run."""

    path: Path
    boxes: tuple[Box, ...]
    source: str
    patient: str
    role: str
    lesion_free: bool = False


def _held_out(patients: list[str], n: int, seed: int = SEED) -> set[str]:
    order = sorted(set(patients))
    return {order[i] for i in np.random.default_rng(seed).permutation(len(order))[:n]}


def cadica_items(raw_dir=None) -> list[BoxImage]:
    """CADICA's annotated frames (every box) and lesion-free frames, with their roles."""
    from vstent.data import cadica
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    raw_dir = raw_dir or cadica.RAW_DIR
    boxes = cadica.load_lesion_boxes(raw_dir)
    split = patient_split([t.patient for t in lesion_tracks(boxes)])
    val = _held_out([p for p, s in split.items() if s == "dev"], CADICA_VAL_PATIENTS)

    def role(p: str) -> str:
        return "test" if split[p] == "test" else ("val" if p in val else "train")

    by_image: dict[Path, list] = {}
    for b in boxes:
        if b.patient in split:
            by_image.setdefault(b.image, [b.patient, []])[1].append((b.x, b.y, b.w, b.h))
    out = [BoxImage(path, tuple(bs), "cadica", p, role(p))
           for path, (p, bs) in sorted(by_image.items(), key=lambda kv: str(kv[0]))]  # fmt: skip
    for r in cadica.load_selected_runs(raw_dir):
        if r.lesion or r.patient not in split:
            continue
        for f in r.selected[::FRAME_STEP]:
            path = r.frames_dir / f"{r.patient}_{r.video}_{f:05d}.png"
            out.append(BoxImage(path, (), "cadica", r.patient, role(r.patient), lesion_free=True))
    return out


def arcade_items(raw_dir: str | Path = "data/raw/arcade") -> list[BoxImage]:
    """ARCADE stenosis images and boxes; role = the published split."""
    import json

    out = []
    for split, role in (("train", "train"), ("val", "val"), ("test", "test")):
        root = Path(raw_dir) / "arcade" / "stenosis" / split
        coco = json.loads((root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
        boxes: dict[int, list] = {}
        for a in coco["annotations"]:
            boxes.setdefault(a["image_id"], []).append(tuple(float(v) for v in a["bbox"]))
        for im in sorted(coco["images"], key=lambda im: im["id"]):
            out.append(BoxImage(root / "images" / im["file_name"],
                                tuple(boxes.get(im["id"], [])), "arcade",
                                f"arcade_{split}_{im['id']}", role))  # fmt: skip
    return out


def kemerovo_items(raw_dir: str | Path = "data/raw/kemerovo") -> list[BoxImage]:
    """Kemerovo frames (every :data:`FRAME_STEP`-th of a series) with their roles."""
    from vstent.data.kemerovo import load_boxes

    boxes = load_boxes(raw_dir)
    patients = sorted({b.patient for b in boxes})
    perm = np.random.default_rng(SEED).permutation(len(patients))
    first = {patients[i] for i in perm[: len(patients) // 2]}
    val = _held_out(sorted(first), KEMEROVO_VAL_PATIENTS)
    series: dict[tuple[str, str], list] = {}
    for b in boxes:
        series.setdefault((b.patient, b.series), []).append(b)
    out = []
    for (patient, _), bs in sorted(series.items()):
        role = "test" if patient not in first else ("val" if patient in val else "train")
        for b in sorted(bs, key=lambda b: b.image.name)[::FRAME_STEP]:
            out.append(BoxImage(b.image, ((b.x, b.y, b.w, b.h),), "kemerovo", patient, role))
    return out


def all_items(kemerovo_dir: str | Path = "data/raw/kemerovo") -> list[BoxImage]:
    """Every source's frames (Kemerovo if downloaded)."""
    items = cadica_items() + arcade_items()
    if (Path(kemerovo_dir) / "train_labels.csv").exists():
        items += kemerovo_items(kemerovo_dir)
    return items


def load_images(items: list[BoxImage], size: int = SIZE) -> tuple[np.ndarray, list]:
    """Greyscale images (N, size, size) uint8 and each image's boxes scaled to ``size``."""
    images, boxes = [], []
    for it in items:
        img = cv2.imread(str(it.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"{it.path}: unreadable")
        sy, sx = size / img.shape[0], size / img.shape[1]
        images.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
        boxes.append([(x * sx, y * sy, w * sx, h * sy) for x, y, w, h in it.boxes])
    return np.stack(images), boxes


def box_mask(boxes, size: int = SIZE) -> np.ndarray:
    """uint8 {0, 1} (size, size) with every box filled."""
    m = np.zeros((size, size), np.uint8)
    for x, y, w, h in boxes:
        x0, y0 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
        m[y0 : int(np.ceil(y + h)), x0 : int(np.ceil(x + w))] = 1
    return m


class BoxMasks:
    """The filled box masks of a list of images, drawn when indexed (saves memory)."""

    def __init__(self, boxes: list, size: int = SIZE) -> None:
        self.boxes, self.size = boxes, size

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, i: int) -> np.ndarray:
        return box_mask(self.boxes[i], self.size)

    def __iter__(self) -> Iterator[np.ndarray]:
        return (self[i] for i in range(len(self)))


def dataset(role: str, kemerovo_dir: str | Path = "data/raw/kemerovo",
            max_images: int | None = None) -> tuple[np.ndarray, BoxMasks]:  # fmt: skip
    """Images and box masks of one role (train / val / test)."""
    items = [it for it in all_items(kemerovo_dir) if it.role == role][:max_images]
    images, boxes = load_images(items)
    return images, BoxMasks(boxes)
