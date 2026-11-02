"""Kemerovo angiographic dataset for stenosis detection: download, verification, loading.

Danilov VV, Klyshnikov KY, Gerget OM, Kutikhin AG, Ganyukov VI, Frangi AF, Ovcharenko EA.
"Real-time coronary artery stenosis detection based on modern neural networks". Sci Rep
2021;11:7582. Data: Mendeley Data "Angiographic dataset for stenosis detection", version 2,
doi:10.17632/ydrm75xywg.2, licence CC BY 4.0.

100 patients with one-vessel coronary artery disease (Research Institute for Complex Issues of
Cardiovascular Diseases, Kemerovo; Siemens Coroscop and GE Innova), each with ≥ 70% diameter
stenosis by QCA, or 50–69% with FFR ≤ 0.80 or ischaemia on stress echocardiography. 8,325
greyscale frames (512–1000 px) selected by a cardiologist, each with a box around the stenosis
(the only class, "Stenosis"). No severity grades, no FFR values, no pixel size (checked in the
archive's files, 2026-11-02). Used as an external test of lesion reading (P4i): every box is a
significant lesion. Only the frames and labels are extracted; the archive's detector videos
are not. Files go to ``data/raw/kemerovo/`` (git-ignored).
"""

import csv
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vstent.data.arcade import download, file_digest

URL = (
    "https://data.mendeley.com/public-files/datasets/ydrm75xywg/files/"
    "61f788d6-65ce-4265-a23a-5ba16931d18b/file_downloaded"
)
ARCHIVE = "Stenosis detection.zip"
SHA256 = "36af6306a3994fe370354e117be8b8fe33457f6ea8c04f202340949b73732e14"
SIZE_BYTES = 8_300_927_983
DOI = "10.17632/ydrm75xywg.2"
LICENCE = "CC BY 4.0 (Mendeley Data licence field)"
CITATION = (
    "Danilov VV, Klyshnikov KY, Gerget OM, Kutikhin AG, Ganyukov VI, Frangi AF, Ovcharenko EA. "
    "Real-time coronary artery stenosis detection based on modern neural networks. Sci Rep "
    "2021;11:7582. Data: Mendeley Data doi:10.17632/ydrm75xywg.2."
)
RAW_DIR = Path("data/raw/kemerovo")
SOURCE_FILE = "SOURCE.json"
EXTRACTED_MARKER = ".extracted"
KEEP = ("dataset/", "train_labels.csv", "test_labels.csv")  # members extracted


@dataclass(frozen=True)
class StenosisBox:
    """One expert stenosis box: frame image, patient, (x, y, w, h) px with (x, y) top-left."""

    image: Path
    patient: str
    series: str
    x: float
    y: float
    w: float
    h: float


def extract_labelled(archive: str | Path, out_dir: str | Path) -> list[str]:
    """Extract the frames and label files only, refusing members outside ``out_dir``."""
    out_dir = Path(out_dir).resolve()
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.startswith(KEEP) and not n.endswith("/")]
        for name in names:
            target = (out_dir / name).resolve()
            if Path(name).is_absolute() or not target.is_relative_to(out_dir):
                raise ValueError(f"{archive}: unsafe member path {name!r}")
        for name in names:
            zf.extract(name, out_dir)
    return names


def fetch_kemerovo(
    raw_dir: str | Path = RAW_DIR, *, force: bool = False, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Download (unless a verified archive exists), verify SHA-256 and extract the frames and
    labels into ``raw_dir``; writes ``SOURCE.json``."""
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
        names = extract_labelled(archive, raw_dir)
        marker.write_text(f"{len(names)}\n", encoding="utf-8", newline="\n")
        log(f"Extracted {len(names):,} frames and label files into {raw_dir}")
    source = {
        "dataset": "Kemerovo angiographic dataset for stenosis detection",
        "doi": DOI, "url": URL, "archive": ARCHIVE, "sha256": SHA256,
        "bytes": archive.stat().st_size, "licence": LICENCE, "citation": CITATION,
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    (raw_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8", newline="\n")
    return source


def load_boxes(raw_dir: str | Path = RAW_DIR) -> list[StenosisBox]:
    """Every stenosis box of ``train_labels.csv`` and ``test_labels.csv`` (the authors' split is
    not used: all are external test data here). File names are 14_<patient>_<series>_<frame>."""
    raw_dir = Path(raw_dir)
    out = []
    for name in ("train_labels.csv", "test_labels.csv"):
        with (raw_dir / name).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["class"] != "Stenosis":
                    raise ValueError(f"{name}: unknown class {row['class']!r}")
                parts = Path(row["filename"]).stem.split("_")
                x0, y0 = float(row["xmin"]), float(row["ymin"])
                out.append(StenosisBox(raw_dir / "dataset" / row["filename"], f"k{parts[1]}",
                                       parts[2], x0, y0, float(row["xmax"]) - x0,
                                       float(row["ymax"]) - y0))  # fmt: skip
    return out
