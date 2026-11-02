"""AngioCAD X-ray angiography dataset: download and verification (P4k-J).

"AngioCAD: A Publicly Available X-ray Angiography Dataset for Coronary Artery Stenosis and CAD
Detection" (University of Isfahan, Isfahan University of Technology, Isfahan University of
Medical Sciences). Zenodo doi:10.5281/zenodo.15826856, licence CC BY 4.0 (Zenodo API licence
field, checked 2026-11-02). 413 patients: anonymised cine frames (PNG, extracted from DICOM),
each patient's 15 coronary segments graded by experts in seven severity categories, clinical
and laboratory data. No FFR, no ECG. Used as an external, segment-level test (docs/PLAN.md,
P4k-J). Files go to ``data/raw/angiocad/`` (git-ignored). The image archives are multi-part RAR
files: they are listed and checked before anything is extracted, and nothing in them is run.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vstent.data.arcade import download, md5sum

RECORD = "https://zenodo.org/api/records/15826856/files/{name}/content"
DOI = "10.5281/zenodo.15826856"
LICENCE = "CC BY 4.0 (Zenodo licence field)"
CITATION = (
    "Hosseini MS, Naghsh-Nilchi AR, Miramirkhani A, Safayani M, Sadeghi M, Shirvani E, Danesh M. "
    "AngioCAD: A Publicly Available X-ray Angiography Dataset for Coronary Artery Stenosis and "
    "CAD Detection. Zenodo 2025, doi:10.5281/zenodo.15826856."
)
FILES = {  # name: (bytes, MD5), from the Zenodo record
    "AngioCAD_Labels.xlsx": (43_267, "fc97ffc2978f6cee93a1059e8b1c282f"),
    "AngioCAD_Features.xlsx": (129_493, "1617d242df42367787f445392729f35e"),
    "AngioCAD_Dataset.part1.rar": (4_194_304_000, "3f406741a35ce45a5b6caa60e3ac8ab9"),
    "AngioCAD_Dataset.part2.rar": (4_194_304_000, "0596e0a58a7a252a19b31e8bec37dc96"),
    "AngioCAD_Dataset.part3.rar": (4_194_304_000, "f425ca88df9e6024dbeeed3cea582ce9"),
    "AngioCAD_Dataset.part4.rar": (3_818_488_789, "a8f24f0523a129ede37112533eb9629b"),
}
RAW_DIR = Path("data/raw/angiocad")
SOURCE_FILE = "SOURCE.json"


def fetch_angiocad(
    raw_dir: str | Path = RAW_DIR, *, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Download every file not yet present and verified (MD5) into ``raw_dir``; writes
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
        "dataset": "AngioCAD", "doi": DOI, "files": {n: {"bytes": s, "md5": m}
                                                      for n, (s, m) in FILES.items()},
        "licence": LICENCE, "citation": CITATION,
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    (raw_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8", newline="\n")
    return source
