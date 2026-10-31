"""ARCADE coronary angiography dataset (Zenodo record 10390295): download, verification, loading.

The archive is downloaded to ``data/raw/arcade/arcade.zip`` (streamed to a ``.part`` file and
renamed only after the MD5 checksum matches), then extracted next to it with every member path
checked to stay inside the target folder. ``SOURCE.json`` records URL, checksum and licence.
The images are de-identified research data; they are never committed (``data/`` is git-ignored).

Layout and categories are documented in ``docs/DATA.md``. Vessel masks are the union of all
``syntax``-task segment polygons of an image; the ``stenosis`` task supplies boxes for overlays.
"""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ZENODO_RECORD = "10390295"
ARCHIVE = "arcade.zip"
URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ARCHIVE}/content"
MD5 = "c5b1973ade06f7dff210f878161e1a76"
SIZE_BYTES = 451_573_813
LICENCE = "CC0 1.0 (Zenodo licence id: cc-zero)"
DOI = "10.5281/zenodo.10390295"
RAW_DIR = Path("data/raw/arcade")
SOURCE_FILE = "SOURCE.json"
EXTRACTED_MARKER = ".extracted"
CHUNK = 1 << 20


def file_digest(path: str | Path, algorithm: str = "md5") -> str:
    """Hex digest of a file with ``algorithm`` (``md5``, ``sha256``; read in 1 MiB chunks)."""
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def md5sum(path: str | Path) -> str:
    """MD5 hex digest of a file."""
    return file_digest(path, "md5")


def download(
    url: str,
    dest: str | Path,
    *,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    log: Callable[[str], None] | None = None,
    timeout: float = 60.0,
) -> Path:
    """Stream ``url`` to ``dest`` via ``dest.part``; rename only if the checksum matches (SHA-256
    if given, else MD5; one of them is required).

    Raises ``ValueError`` on a checksum or size mismatch (the partial file is removed).
    """
    if expected_md5 is None and expected_sha256 is None:
        raise ValueError("download needs an expected MD5 or SHA-256")
    algorithm, expected = ("sha256", expected_sha256) if expected_sha256 else ("md5", expected_md5)
    log = log or (lambda _msg: None)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.new(algorithm)
    done, next_report = 0, 0
    request = urllib.request.Request(url, headers={"User-Agent": "vstent-research-prototype"})
    with urllib.request.urlopen(request, timeout=timeout) as resp, part.open("wb") as f:
        total = int(resp.headers.get("Content-Length") or 0) or expected_size
        for block in iter(lambda: resp.read(CHUNK), b""):
            f.write(block)
            h.update(block)
            done += len(block)
            if total and done >= next_report:
                log(f"  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
                next_report += total // 10
    if expected_size is not None and done != expected_size:
        part.unlink()
        raise ValueError(f"{url}: got {done} bytes, expected {expected_size}")
    if h.hexdigest() != expected:
        part.unlink()
        raise ValueError(f"{url}: {algorithm.upper()} {h.hexdigest()} != expected {expected}")
    part.replace(dest)
    return dest


def safe_extract(archive: str | Path, out_dir: str | Path) -> list[str]:
    """Extract a zip, refusing absolute paths and members that would land outside ``out_dir``.

    Returns the member names.
    """
    out_dir = Path(out_dir).resolve()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for name in names:
            target = (out_dir / name).resolve()
            if Path(name).is_absolute() or not target.is_relative_to(out_dir):
                raise ValueError(f"{archive}: unsafe member path {name!r}")
        zf.extractall(out_dir)
    return names


def fetch_arcade(
    raw_dir: str | Path = RAW_DIR,
    *,
    url: str = URL,
    expected_md5: str = MD5,
    expected_size: int | None = SIZE_BYTES,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Download (unless a verified archive exists), verify and extract ARCADE into ``raw_dir``.

    Returns the source record that is also written to ``raw_dir/SOURCE.json``.
    """
    log = log or (lambda _msg: None)
    raw_dir = Path(raw_dir)
    archive = raw_dir / ARCHIVE
    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    if archive.exists() and md5sum(archive) == expected_md5:
        log(f"{archive} already present, MD5 verified")
    else:
        log(f"Downloading {url}")
        download(url, archive, expected_md5=expected_md5, expected_size=expected_size, log=log)
        log(f"MD5 verified: {expected_md5}")
    marker = raw_dir / EXTRACTED_MARKER
    if not marker.exists():
        names = safe_extract(archive, raw_dir)
        marker.write_text(f"{len(names)}\n", encoding="utf-8", newline="\n")
        log(f"Extracted {len(names):,} archive members into {raw_dir}")
    source = {
        "dataset": "ARCADE",
        "zenodo_record": ZENODO_RECORD,
        "doi": DOI,
        "url": url,
        "archive": ARCHIVE,
        "md5": expected_md5,
        "bytes": archive.stat().st_size,
        "licence": LICENCE,
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (raw_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8", newline="\n")
    return source


SPLITS = ("train", "val", "test")
DATASET_DIR = "arcade"  # top-level folder inside the archive


@dataclass(frozen=True, eq=False)
class ArcadeSplit:
    """Images of one ARCADE task / split, ordered by COCO image id.

    Attributes:
        images: (N, H, W) uint8 greyscale.
        masks: (N, H, W) uint8 in {0, 1} (``syntax`` task), else ``None``.
        boxes: per image, a list of (x, y, w, h) boxes in px (``stenosis`` task), else ``None``.
        file_names: PNG names, e.g. ``"12.png"``.
        image_ids: (N,) COCO image ids.
    """

    images: np.ndarray
    masks: np.ndarray | None
    boxes: list[list[tuple[float, float, float, float]]] | None
    file_names: list[str]
    image_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.file_names)


def task_dir(raw_dir: str | Path, task: str, split: str) -> Path:
    """``<raw_dir>/arcade/<task>/<split>`` (task ``syntax`` or ``stenosis``)."""
    return Path(raw_dir) / DATASET_DIR / task / split


def load_coco(raw_dir: str | Path, task: str, split: str) -> dict[str, Any]:
    """The COCO annotation file of one task / split."""
    path = task_dir(raw_dir, task, split) / "annotations" / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def polygons_to_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """Union of COCO polygons ([x1, y1, x2, y2, ...] in px) as a uint8 {0, 1} mask (H, W)."""
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    rles = mask_utils.frPyObjects(polygons, height, width)
    return np.asarray(mask_utils.decode(mask_utils.merge(rles)), dtype=np.uint8)


def load_split(
    raw_dir: str | Path, task: str, split: str, max_images: int | None = None
) -> ArcadeSplit:
    """Load images plus vessel masks (``syntax``) or stenosis boxes (``stenosis``)."""
    if task not in ("syntax", "stenosis"):
        raise ValueError(f"unknown ARCADE task {task!r}")
    coco = load_coco(raw_dir, task, split)
    images_meta = sorted(coco["images"], key=lambda im: im["id"])[:max_images]
    by_image: dict[int, list[dict]] = {im["id"]: [] for im in images_meta}
    for ann in coco["annotations"]:
        if ann["image_id"] in by_image:
            by_image[ann["image_id"]].append(ann)
    folder = task_dir(raw_dir, task, split) / "images"
    images, masks, boxes = [], [], []
    for im in images_meta:
        img = cv2.imread(str(folder / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (im["height"], im["width"]):
            raise ValueError(f"{folder / im['file_name']}: unreadable or unexpected size")
        images.append(img)
        anns = by_image[im["id"]]
        if task == "syntax":
            polys = [p for a in anns for p in a["segmentation"]]
            masks.append(polygons_to_mask(polys, im["height"], im["width"]))
        else:
            boxes.append([tuple(float(v) for v in a["bbox"]) for a in anns])
    return ArcadeSplit(
        images=np.stack(images),
        masks=np.stack(masks) if task == "syntax" else None,
        boxes=boxes if task == "stenosis" else None,
        file_names=[im["file_name"] for im in images_meta],
        image_ids=np.array([im["id"] for im in images_meta]),
    )
