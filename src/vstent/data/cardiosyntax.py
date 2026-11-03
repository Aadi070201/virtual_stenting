"""CardioSYNTAX: real cine runs with SYNTAX scores per side (external data).

"CardioSyntax: end-to-end SYNTAX score prediction — dataset, benchmark and method" (Ponomarchuk
et al., WACV 2025; Sber AI Lab and Tyumen Cardiology Research Center). Zenodo
doi:10.5281/zenodo.14005818, licence CC BY 4.0 (Zenodo licence field, checked 2026-11-03).
1,844 studies, 14,219 cine runs (512 × 512, 15 frames/s, Philips Allura Clarity / Azurion),
each run tagged LCA or RCA by the authors' classifier, with the C-arm angles; per study the
SYNTAX score in all and for the left and right coronary arteries (an interventional
cardiologist's), dominance on 400 studies, a bypass tag. No ECG, no lesion boxes, no FFR.

A side with SYNTAX 0 has no lesion of ≥ 50% diameter stenosis in a vessel ≥ 1.5 mm; a side
with SYNTAX > 0 has at least one. Files go to ``data/raw/cardiosyntax/`` (git-ignored). The
runs are NumPy ``.npy`` files inside ten zip parts; they are read straight from the zips
(``allow_pickle=False``), never extracted or run.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent.data.arcade import download, md5sum

RECORD = "https://zenodo.org/api/records/14005818/files/{name}/content"
DOI = "10.5281/zenodo.14005818"
LICENCE = "CC BY 4.0 (Zenodo licence field)"
CITATION = (
    "Ponomarchuk A, Kruzhilov I, Zubkova G, Shadrin A, Utegenov R, Bessonov I, Blinov P. "
    "CardioSyntax: end-to-end SYNTAX score prediction - dataset, benchmark and method. "
    "WACV 2025; Zenodo doi:10.5281/zenodo.14005818."
)
FRAME_TIME_MS = 1000.0 / 15.0  # "a frame rate of 15 fps" (the paper)
FILES = {  # name: (bytes, MD5), from the Zenodo record
    "all.json": (7_226_950, "66a4cb32b2f3b7c6888bc31053a5309a"),
    "three_experts.json": (9_562, "5d6e0240b3a3ad006d683a5948967420"),
    "0.zip": (15_806_175_343, "debf55d991f3bf078a7ca1a91480d376"),
    "1.zip": (16_051_666_092, "5d174917c61acbb968926c0bea459091"),
    "2.zip": (15_312_455_164, "7dd536f2c968426a2f383c69def9ae5b"),
    "3.zip": (15_373_699_955, "a54ec3cd82b658383103426acf71c0b8"),
    "4.zip": (16_363_949_016, "c181f668463a6ad79bd91423297b56ad"),
    "5.zip": (15_554_826_439, "ae1fe10a956c58e5cc5cecda896f7253"),
    "6.zip": (15_249_754_046, "d3286d4a13e5c90adf366895d29026bc"),
    "7.zip": (15_865_806_927, "7ab9dd95a80c86919200c1a7a4a8e5a4"),
    "8.zip": (15_799_671_802, "bd67b3dc345ad132c9076b7710833dca"),
    "9.zip": (3_178_756_015, "be1cd1f084458f1151f448c5194052a0"),
}
RAW_DIR = Path("data/raw/cardiosyntax")
SOURCE_FILE = "SOURCE.json"


def fetch_cardiosyntax(raw_dir: str | Path = RAW_DIR, parts: tuple[int, ...] = (9,), *,
                       log: Callable[[str], None] | None = None) -> dict[str, Any]:  # fmt: skip
    """Download the metadata and the zip ``parts`` not yet present and verified (MD5) into
    ``raw_dir``; writes ``SOURCE.json`` listing what is there."""
    log = log or (lambda _msg: None)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = ["all.json", "three_experts.json", *(f"{p}.zip" for p in parts)]
    for name in names:
        size, md5 = FILES[name]
        path = raw_dir / name
        if path.exists() and path.stat().st_size == size:
            log(f"{name} present; verifying MD5")
            if md5sum(path) != md5:
                raise ValueError(f"{path}: MD5 differs from the published value")
            continue
        log(f"Downloading {name} ({size / 1e9:.2f} GB)")
        download(RECORD.format(name=name), path, expected_md5=md5, expected_size=size, log=log)
    present = {n: {"bytes": s, "md5": m} for n, (s, m) in FILES.items()
               if (raw_dir / n).exists() and (raw_dir / n).stat().st_size == s}  # fmt: skip
    source = {"dataset": "CardioSYNTAX", "doi": DOI, "files": present, "licence": LICENCE,
              "citation": CITATION,
              "verified_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    (raw_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8", newline="\n")
    return source


@dataclass(frozen=True)
class Video:
    """One cine run: its path inside the part zips, the authors' LCA / RCA tag (classifier
    probability of RCA), frames and the C-arm angles (degrees)."""

    path: str
    artery: str
    rca_prob: float
    n_frames: int
    primary_deg: float | None
    secondary_deg: float | None

    @property
    def part(self) -> int:
        return int(self.path.split("/", 1)[0])


@dataclass(frozen=True)
class Study:
    """One patient's study: SYNTAX in all, left and right; dominance; bypass; its runs."""

    uid: str
    syntax: float | None
    syntax_left: float | None
    syntax_right: float | None
    dominance: str | None
    bypass: bool
    videos: tuple[Video, ...]

    def side_syntax(self, artery: str) -> float | None:
        return self.syntax_right if artery == "RCA" else self.syntax_left


def load_studies(raw_dir: str | Path = RAW_DIR) -> list[Study]:
    """Every study of ``all.json``."""
    data = json.loads((Path(raw_dir) / "all.json").read_text(encoding="utf-8"))
    out = []
    for s in data:
        videos = tuple(Video(v["path"], v["artery"], float(v["artery_prob"]), int(v["shape"][0]),
                             v.get("PositionerPrimaryAngle"), v.get("PositionerSecondaryAngle"))
                       for v in s["videos"])  # fmt: skip
        out.append(Study(s["study_uid"], s.get("syntax"), s.get("syntax_left"),
                         s.get("syntax_right"), s.get("dominance"), bool(s.get("bypass")),
                         videos))  # fmt: skip
    return out


def read_video(raw_dir: str | Path, video: Video | str) -> np.ndarray:
    """(T, H, W) uint8 frames of one run, read from its part zip (no pickles, no extraction)."""
    path = video.path if isinstance(video, Video) else video
    part = int(path.split("/", 1)[0])
    with zipfile.ZipFile(Path(raw_dir) / f"{part}.zip") as z:
        names = set(z.namelist())
        member = path if path in names else path.split("/", 1)[1]
        if member not in names:
            raise KeyError(f"{path} is not in {part}.zip")
        with z.open(member) as f:
            arr = np.load(f, allow_pickle=False)
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected (T, H, W) frames, got {arr.shape}")
    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        arr = np.clip((arr - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    return arr
