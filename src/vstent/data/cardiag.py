"""CARDIAG coronary angiography dataset: download and verification.

"CARDIAG: A Multi-Center Dataset and Benchmark for SYNTAX Segmentation of Coronary Angiograms"
(Gdańsk University of Technology and partners; five cardiology centres in northern Poland;
Siemens Axiom Artis). Zenodo doi:10.5281/zenodo.19958730, licence CC BY 4.0 (Zenodo licence
field, checked 2026-11-03); paper arXiv:2607.22139. 644 annotated frames of 114 patients: the
X-ray frame, SYNTAX segment masks (26 classes), expert uncertainty masks, the contrast catheter
mask, stenoses, up to 30 earlier and 10 later frames, and the acquisition geometry (C-arm
angles, SID, SOD and imager pixel spacing, mm). No FFR, no ECG. Only ``CARDIAG.zip`` is used
(``models.zip``, the authors' weights, is not downloaded). Files go to ``data/raw/cardiag/``
(git-ignored); the archive is listed and checked before anything is extracted, and nothing in
it is run.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vstent.data.arcade import download, md5sum

RECORD = "https://zenodo.org/api/records/19958730/files/{name}/content"
DOI = "10.5281/zenodo.19958730"
LICENCE = "CC BY 4.0 (Zenodo licence field)"
CITATION = (
    "Lau DB, Malinowski H, Szyjut J, Brzeski A, Dziubich T, Targoński R, Figatowski T, "
    "Zielińska N. CARDIAG: A Multi-Center Dataset and Benchmark for SYNTAX Segmentation of "
    "Coronary Angiograms. Zenodo 2026, doi:10.5281/zenodo.19958730; arXiv:2607.22139."
)
FILES = {"CARDIAG.zip": (4_569_236_433, "1aa8f3b77c1d8dd6b101410a496728f3")}  # Zenodo record
RAW_DIR = Path("data/raw/cardiag")
SOURCE_FILE = "SOURCE.json"


def fetch_cardiag(
    raw_dir: str | Path = RAW_DIR, *, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Download ``CARDIAG.zip`` if not present and verified (MD5) into ``raw_dir``; writes
    ``SOURCE.json``. Extraction is a separate, checked step."""
    log = log or (lambda _msg: None)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (size, md5) in FILES.items():
        path = raw_dir / name
        if path.exists() and path.stat().st_size == size:
            log(f"{name} present; verifying MD5")
            if md5sum(path) != md5:
                raise ValueError(f"{path}: MD5 differs from the published value")
            continue
        log(f"Downloading {name} ({size / 1e9:.2f} GB)")
        download(RECORD.format(name=name), path, expected_md5=md5, expected_size=size, log=log)
    source = {
        "dataset": "CARDIAG", "doi": DOI, "files": {n: {"bytes": s, "md5": m}
                                                     for n, (s, m) in FILES.items()},
        "licence": LICENCE, "citation": CITATION,
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    (raw_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8", newline="\n")
    return source
