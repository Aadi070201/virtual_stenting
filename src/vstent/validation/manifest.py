"""Validation case lists (P4): the index-test manifest and the reference file.

Two files, so that whoever runs the software stays blinded to the wire values
(``docs/validation/PROTOCOL.md``); the columns are specified in ``docs/validation/DATA_SPEC.md``:

- ``cases.csv`` (index test), one row per vessel: ``case_id, patient_id, vessel, image, frame,
  start_row, start_col, end_row, end_col, mm_per_px, calibration, pa_mmhg, pv_mmhg, mask``;
- ``reference.csv`` (held by the data manager until the predictions are locked):
  ``case_id, ref_type, ref_value, excluded``.

File paths in a manifest are relative to the manifest's folder. An empty cell means "not given".
Unknown columns are refused (a misspelt column must not be silently ignored).

:func:`check_local_only` refuses patient data, or results derived from it, inside this
repository's tracked files: they must live outside the repository or in a git-ignored folder.
"""

import csv
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from vstent.config import StrictModel

CASE_COLUMNS = (
    "case_id", "patient_id", "vessel", "image", "frame", "start_row", "start_col", "end_row",
    "end_col", "mm_per_px", "calibration", "pa_mmhg", "pv_mmhg", "mask",
)  # fmt: skip
REFERENCE_COLUMNS = ("case_id", "ref_type", "ref_value", "excluded")


class ManifestError(ValueError):
    """A manifest that cannot be used; the message lists every problem found."""


class CaseRow(StrictModel):
    """One vessel of ``cases.csv``. Points (row, col) px; ``end`` is the pressure-sensor
    position. Pixel size in mm/px, pressures in mmHg."""

    case_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    vessel: str = ""
    image: str = Field(min_length=1)
    frame: int | None = Field(None, ge=0)  # multi-frame DICOM: the analysed frame (0-based)
    start_row: int = Field(ge=0)
    start_col: int = Field(ge=0)
    end_row: int = Field(ge=0)
    end_col: int = Field(ge=0)
    mm_per_px: float | None = Field(None, gt=0, lt=5)
    calibration: str = ""
    pa_mmhg: float | None = Field(None, gt=0, lt=300)
    pv_mmhg: float | None = Field(None, ge=0, lt=50)
    mask: str = ""  # optional vessel mask (8-bit PNG; analyst-corrected contour)

    @property
    def start(self) -> tuple[int, int]:
        return self.start_row, self.start_col

    @property
    def end(self) -> tuple[int, int]:
        return self.end_row, self.end_col


class ReferenceRow(StrictModel):
    """One vessel of ``reference.csv``: wire FFR or iFR at the sensor position. A non-empty
    ``excluded`` (the reason) removes the vessel from the accuracy analysis."""

    case_id: str = Field(min_length=1)
    ref_type: Literal["FFR", "iFR"]
    ref_value: float = Field(gt=0, le=1.2)
    excluded: str = ""

    @field_validator("ref_type", mode="before")
    @classmethod
    def _type(cls, v):
        return {"ffr": "FFR", "ifr": "iFR"}.get(str(v).strip().lower(), v)


def _read_rows(path: Path, model, columns: tuple[str, ...]) -> list:
    if not path.is_file():
        raise ManifestError(f"{path}: file not found")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = [c.strip() for c in reader.fieldnames or []]
        problems = []
        unknown = [c for c in header if c not in columns]
        if unknown:
            problems.append(f"unknown column(s) {unknown}; allowed: {list(columns)}")
        rows, ids = [], {}
        for line, raw in enumerate(reader, start=2):
            data = {k.strip(): v.strip() for k, v in raw.items() if k and v and v.strip()}
            if not data:
                continue  # blank line
            try:
                row = model.model_validate(data)
            except ValidationError as exc:
                for err in exc.errors():
                    where = ".".join(str(x) for x in err["loc"])
                    problems.append(f"line {line}, {where}: {err['msg']}")
                continue
            if row.case_id in ids:
                problems.append(
                    f"line {line}: case_id {row.case_id!r} repeats line {ids[row.case_id]}"
                )
            ids[row.case_id] = line
            rows.append(row)
    if problems:
        raise ManifestError(f"{path}:\n  " + "\n  ".join(problems))
    if not rows:
        raise ManifestError(f"{path}: no rows")
    return rows


def read_cases(path: str | Path) -> list[CaseRow]:
    """Validated rows of an index-test manifest (raises :class:`ManifestError`)."""
    return _read_rows(Path(path), CaseRow, CASE_COLUMNS)


def read_reference(path: str | Path) -> list[ReferenceRow]:
    """Validated rows of a reference file (raises :class:`ManifestError`)."""
    return _read_rows(Path(path), ReferenceRow, REFERENCE_COLUMNS)


def resolve(base: str | Path, name: str) -> Path:
    """A manifest path: absolute, or relative to the manifest's folder ``base``."""
    p = Path(name)
    return p if p.is_absolute() else Path(base) / p


def case_files(cases: list[CaseRow], base: str | Path) -> list[Path]:
    """Every image and mask a manifest names."""
    return [resolve(base, n) for c in cases for n in (c.image, c.mask) if n]


def _repo_root() -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out.stdout.strip()).resolve()


def tracked_paths(paths: list[Path]) -> list[Path]:
    """The ``paths`` that lie inside this repository and are not git-ignored (``[]`` when the
    package does not run from a git checkout)."""
    root = _repo_root()
    if root is None:
        return []
    inside = []
    for p in paths:
        q = Path(p).resolve()
        if q == root or root in q.parents:
            inside.append(q)
    if not inside:
        return []
    # NUL-separated POSIX paths in and out: git quotes and escapes other paths (e.g. "a\b")
    rel = [q.relative_to(root).as_posix() for q in inside]
    out = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"], cwd=root, capture_output=True, text=True,
        input="\0".join(rel) + "\0",
    )  # fmt: skip
    if out.returncode not in (0, 1):  # 1 = none ignored; anything else is an error
        raise RuntimeError(f"git check-ignore failed: {out.stderr.strip()}")
    ignored = {(root / p).resolve() for p in out.stdout.split("\0") if p}
    return [q for q in inside if q not in ignored]


def check_local_only(paths: list[Path], what: str) -> None:
    """Raise ``ValueError`` if any of ``paths`` lies in this repository's tracked files:
    patient data and anything derived from it must never be committed."""
    bad = tracked_paths(paths)
    if bad:
        shown = ", ".join(str(p) for p in bad[:5]) + (" …" if len(bad) > 5 else "")
        raise ValueError(
            f"{what} inside this repository must be in a git-ignored folder (data/, runs/) or "
            f"outside the repository; no patient data may ever be committed: {shown}"
        )
