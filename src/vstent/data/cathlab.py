"""Datasets exported by the virtual cath lab (``cathlab/``, a separate app): format
``vcathlab-frames`` v1 (cathlab/docs/DATA_FORMAT.md). Torch-free; nothing from the cath lab is
imported, only its files are read.

Every simulated lesion is listed with its true severity; which ones count as lesions here is
this reader's rule (:func:`positives`), stated wherever it is used. Simulated data only.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FORMAT, VERSION = "vcathlab-frames", 1
Box = tuple[float, float, float, float]  # x, y, w, h (px)


def read_manifest(root: str | Path) -> dict[str, Any]:
    """The dataset's manifest, after checking its format and version."""
    m = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    if m.get("format") != FORMAT or m.get("version") != VERSION:
        raise ValueError(f"{root}: not a {FORMAT} v{VERSION} dataset "
                         f"({m.get('format')} v{m.get('version')})")  # fmt: skip
    return m


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def positives(lesions: list[dict[str, Any]], min_ds: float = 50.0,
              min_inside: float = 0.5) -> list[Box]:  # fmt: skip
    """The boxes of the lesions that count: true diameter stenosis ≥ ``min_ds`` %, at least
    ``min_inside`` of it inside the image. Milder lesions are left unmarked (as in SYNTAX)."""
    return [tuple(le["box_xywh"]) for le in lesions
            if le["box_xywh"] is not None and le["ds_pct"] >= min_ds
            and le["inside_share"] >= min_inside]  # fmt: skip


def load_split(root: str | Path, split: str, *, min_ds: float = 50.0, min_inside: float = 0.5,
               max_images: int | None = None, verify: bool = True,
               ) -> tuple[np.ndarray, list[list[Box]], list[dict[str, Any]]]:  # fmt: skip
    """Images (N, H, W) uint8, each image's lesion boxes by :func:`positives`, and the records
    (scene, truth, every lesion) of ``split``. ``verify``: check every file's SHA-256."""
    root = Path(root)
    info = read_manifest(root)["splits"][split]
    rec_path = root / info["records"]
    if verify and _sha256(rec_path) != info["records_sha256"]:
        raise ValueError(f"{rec_path}: SHA-256 differs from the manifest")
    records = [json.loads(x) for x in rec_path.read_text(encoding="utf-8").splitlines()]
    n = len(records) if max_images is None else min(max_images, len(records))
    images = []
    for sh in info["shards"]:
        if sh["first"] >= n:
            break
        path = root / sh["file"]
        if verify and _sha256(path) != sh["sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from the manifest")
        with np.load(path, allow_pickle=False) as z:
            images.append(z["images"][: n - sh["first"]])
    records = records[:n]
    return np.concatenate(images), [positives(r["lesions"], min_ds, min_inside)
                                    for r in records], records  # fmt: skip
