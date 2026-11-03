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
from dataclasses import dataclass
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


# --- labels (AngioCAD_Labels.xlsx) ----------------------------------------------------------------

LABELS_FILE = "AngioCAD_Labels.xlsx"
FRAMES_DIR = "frames/AngioCAD_Dataset"
GRADES = ("NL", "1-25", "26-50", "51-75", "76-90", "91-99", "100")
SIDES = {
    "right": ("Prox RCA", "Mid RCA", "Dist RCA", "PDA", "PLB"),
    "left": ("LM", "Prox LAD", "Mid LAD", "Dist LAD", "1st dig", "2nd dig", "Prox LCX",
             "Mid LCX", "Dist LCX", "OM"),
}  # fmt: skip
SERIES_COLUMNS = {"right": "Right Coronary Series", "left": "Left Coronary Series"}
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass(frozen=True)
class PatientLabels:
    """One AngioCAD patient: the series of each side and each segment's grade (one of
    :data:`GRADES`, or ``None`` if blank)."""

    patient: str
    series: dict[str, tuple[int, ...]]
    grades: dict[str, str | None]

    def worst(self, side: str) -> str | None:
        """The side's worst grade (``None`` if no segment is graded)."""
        found = [g for s in SIDES[side] if (g := self.grades.get(s)) in GRADES]
        return max(found, key=GRADES.index) if found else None

    def status(self, side: str) -> str | None:
        """``significant`` (a segment ≥ 51%), ``normal`` (all NL) or ``mild``; ``None`` if
        ungraded."""
        worst = self.worst(side)
        if worst is None:
            return None
        if GRADES.index(worst) >= GRADES.index("51-75"):
            return "significant"
        return "normal" if worst == "NL" else "mild"


def _column(ref: str) -> str:
    return "".join(ch for ch in ref if ch.isalpha())


def xlsx_rows(path: str | Path) -> list[dict[str, str | None]]:
    """The first worksheet of an .xlsx file as rows keyed by the header row (cell text; numbers
    as written). Read with the standard library only; nothing in the file is run."""
    import xml.etree.ElementTree as ET
    import zipfile

    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{_NS}t"))
                      for si in root.findall(f"{_NS}si")]  # fmt: skip
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    table = []
    for row in sheet.iter(f"{_NS}row"):
        cells = {}
        for c in row.findall(f"{_NS}c"):
            v = c.find(f"{_NS}v")
            if v is None or v.text is None:
                inline = c.find(f"{_NS}is")
                text = (
                    None
                    if inline is None
                    else "".join(t.text or "" for t in inline.iter(f"{_NS}t"))
                )
            else:
                text = shared[int(v.text)] if c.get("t") == "s" else v.text
            cells[_column(c.get("r", ""))] = text
        table.append(cells)
    if not table:
        return []
    header = {col: name for col, name in table[0].items() if name}
    return [{name: cells.get(col) for col, name in header.items()} for cells in table[1:]]


def parse_series(text: str | None) -> tuple[int, ...]:
    """Series numbers from "1-2", "3", "1-6", "1,3" or "1-2, 5" (empty if blank)."""
    out: list[int] = []
    for part in str(text or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(float(a)), int(float(b)) + 1))
        else:
            out.append(int(float(part)))
    return tuple(out)


def _grade(text: str | None) -> str | None:
    if text is None or not str(text).strip():
        return None
    t = str(text).strip()
    return {"100.0": "100"}.get(t, t)


def load_labels(raw_dir: str | Path = RAW_DIR) -> list[PatientLabels]:
    """Every labelled patient of ``AngioCAD_Labels.xlsx`` (rows without an ID skipped)."""
    out = []
    for row in xlsx_rows(Path(raw_dir) / LABELS_FILE):
        pid = row.get("ID")
        if pid is None or not str(pid).strip():
            continue
        series = {side: parse_series(row.get(col)) for side, col in SERIES_COLUMNS.items()}
        grades = {seg: _grade(row.get(seg)) for segs in SIDES.values() for seg in segs}
        out.append(PatientLabels(str(int(float(pid))), series, grades))
    return out


def series_dir(raw_dir: str | Path, patient: str, series: int) -> Path:
    """The folder of one series' frames."""
    return Path(raw_dir) / FRAMES_DIR / patient / str(series)
