"""CADICA invasive coronary angiography dataset: download, verification, loading.

Jiménez-Partinen A. et al., "CADICA: A new dataset for coronary artery disease detection by
using invasive coronary angiography", Expert Systems 2024;41(12):e13708. Mendeley Data
doi:10.17632/p9bpx9ctcv.5, licence CC BY 4.0 (attribution required; commercial use allowed).

Cine angiograms of 42 patients (University of Málaga), converted by the authors from DICOM to
512 × 512 PNG frames (10 frames/s), with expert-drawn lesion boxes on keyframes. Each box has a
lesion-degree category (diameter stenosis estimated visually by the clinicians, e.g. ``p70_90``
= 70–90%). Pixel size and projection geometry are not included. Files go to
``data/raw/cadica/`` (git-ignored); the archive's SHA-256 is the one Mendeley publishes.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vstent.data.arcade import download, file_digest, safe_extract

URL = (
    "https://data.mendeley.com/public-files/datasets/p9bpx9ctcv/files/"
    "9088fc7c-629b-4a23-8553-5a4d3a5282e1/file_downloaded"
)
ARCHIVE = "CADICA.zip"
SHA256 = "ac945f946d076ac426975b9096d030df0b5a00ff5ac2c284eb5747b43dbad97b"
SIZE_BYTES = 3_076_112_179
DOI = "10.17632/p9bpx9ctcv.5"
LICENCE = "CC BY 4.0 (Mendeley Data licence field)"
CITATION = (
    "Jiménez-Partinen A, Molina-Cabello MA, Thurnhofer-Hemsi K, Palomo EJ, Rodríguez-Capitán "
    "J, Molina-Ramos AI, Jiménez-Navarro M. CADICA: A new dataset for coronary artery disease "
    "detection by using invasive coronary angiography. Expert Systems 2024;41(12):e13708."
)
RAW_DIR = Path("data/raw/cadica")
SOURCE_FILE = "SOURCE.json"
EXTRACTED_MARKER = ".extracted"
ROOT = "CADICA"
# lesion-degree categories: label → (low, high) visual diameter stenosis (%)
CATEGORIES: dict[str, tuple[float, float]] = {
    "p0_20": (0.0, 20.0),
    "p20_50": (20.0, 50.0),
    "p50_70": (50.0, 70.0),
    "p70_90": (70.0, 90.0),
    "p90_98": (90.0, 98.0),
    "p99": (99.0, 99.0),
    "p100": (100.0, 100.0),
}


def fetch_cadica(
    raw_dir: str | Path = RAW_DIR, *, force: bool = False, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Download (unless a verified archive exists), verify SHA-256 and extract CADICA into
    ``raw_dir``; writes ``SOURCE.json``."""
    log = log or (lambda _msg: None)
    raw_dir = Path(raw_dir)
    archive = raw_dir / ARCHIVE
    if not force and archive.exists() and archive.stat().st_size == SIZE_BYTES:
        log(f"{archive} present; verifying SHA-256")
        if file_digest(archive, "sha256") != SHA256:
            raise ValueError(f"{archive}: SHA-256 differs from the published value")
    else:
        log(f"Downloading {URL} ({SIZE_BYTES / 1e9:.1f} GB)")
        download(URL, archive, expected_sha256=SHA256, expected_size=SIZE_BYTES, log=log)
    marker = raw_dir / EXTRACTED_MARKER
    if force or not marker.exists():
        names = safe_extract(archive, raw_dir)
        marker.write_text(f"{len(names)}\n", encoding="utf-8", newline="\n")
        log(f"Extracted {len(names):,} archive members into {raw_dir}")
    source = {
        "dataset": "CADICA",
        "doi": DOI,
        "url": URL,
        "archive": ARCHIVE,
        "sha256": SHA256,
        "bytes": archive.stat().st_size,
        "licence": LICENCE,
        "citation": CITATION,
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (raw_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8", newline="\n")
    return source


@dataclass(frozen=True)
class LesionBox:
    """One expert lesion box on one keyframe. Box (x, y, w, h) in px, (x, y) = top-left."""

    patient: str
    video: str
    frame: int
    x: float
    y: float
    w: float
    h: float
    category: str
    image: Path

    @property
    def ds_range(self) -> tuple[float, float]:
        """Visual diameter-stenosis range of the category (%)."""
        return CATEGORIES[self.category]


_FRAME = re.compile(r"^(p\d+)_(v\d+)_(\d+)\.txt$")


def load_lesion_boxes(raw_dir: str | Path = RAW_DIR) -> list[LesionBox]:
    """Every expert lesion box of the selected videos, ordered by patient, video and frame.
    Rows are ``x y w h category`` (whitespace-separated)."""
    root = Path(raw_dir) / ROOT / "selectedVideos"
    boxes = []
    for gt in sorted(root.glob("p*/v*/groundtruth/*.txt")):
        m = _FRAME.match(gt.name)
        if not m:
            continue
        patient, video, frame = m.group(1), m.group(2), int(m.group(3))
        image = gt.parent.parent / "input" / f"{patient}_{video}_{frame:05d}.png"
        for line in gt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            x, y, w, h = (float(v) for v in parts[:4])
            if parts[4] not in CATEGORIES:
                raise ValueError(f"{gt}: unknown lesion category {parts[4]!r}")
            boxes.append(LesionBox(patient, video, frame, x, y, w, h, parts[4], image))
    key = lambda b: (int(b.patient[1:]), int(b.video[1:]), b.frame)  # noqa: E731
    return sorted(boxes, key=key)
