"""Expert end-diastolic frame labels without an ECG (P4l): the label set, storage and the
agreement of two readers. No Streamlit (the page is ``label_app.py``).

Readers mark one ED frame per heart beat in each run: the frame where the heart and its
coronary tree are largest, just before they contract. Labels are saved per reader as JSON in
``data/labels/ed/<reader>/<run>.json`` (git-ignored, like all data).
"""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from vstent.app.guided import Case, case_library

LABEL_DIR = Path("data/labels/ed")
N_CADICA = 30  # P4l: 30 CADICA test patients' runs (and 30 AngioCAD runs once available)
SEED = 0
MATCH_FRAMES = 1  # two readers agree on a beat when their frames are within ±1


def label_set(raw_dir=None, n: int = N_CADICA, seed: int = SEED) -> list[Case]:
    """The runs to label: ``n`` CADICA runs of test patients (seed fixed), in a fixed order; the
    synthetic library cases when CADICA is absent (for trying the page out)."""
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    cases = case_library(raw_dir)
    cadica = [c for c in cases if c.kind == "cadica"]
    if not cadica:
        return [c for c in cases if c.kind == "synthetic"]
    from vstent.data import cadica as cad

    split = patient_split([t.patient for t in lesion_tracks(cad.load_lesion_boxes(
        raw_dir or cad.RAW_DIR))])  # fmt: skip
    test = sorted((c for c in cadica if split.get(c.key.split(":")[1].split("_")[0]) == "test"),
                  key=lambda c: c.key)  # fmt: skip
    pick = np.random.default_rng(seed).permutation(len(test))[:n]
    return [test[i] for i in sorted(pick)]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()) or "reader"


def label_path(reader: str, case_key: str, root: Path | None = None) -> Path:
    """Where a reader's label of a run is saved (``root`` read at call time: default
    :data:`LABEL_DIR`)."""
    return Path(root or LABEL_DIR) / _safe(reader) / f"{_safe(case_key)}.json"


@dataclass(frozen=True)
class EdLabel:
    """One reader's ED frames (0-based) for one run."""

    case_key: str
    reader: str
    frames: tuple[int, ...]
    n_frames: int
    saved_utc: str = ""


def save_label(label: EdLabel, root: Path | None = None) -> Path:
    path = label_path(label.reader, label.case_key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"case": label.case_key, "reader": label.reader, "ed_frames": sorted(label.frames),
            "n_frames": label.n_frames,
            "saved_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def load_label(reader: str, case_key: str, root: Path | None = None) -> EdLabel | None:
    path = label_path(reader, case_key, root)
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return EdLabel(d["case"], d["reader"], tuple(d["ed_frames"]), d["n_frames"], d["saved_utc"])


def match(a: tuple[int, ...], b: tuple[int, ...], tol: int = MATCH_FRAMES) -> int:
    """Beats on which two readers agree: frames paired greedily, closest first, within ±tol."""
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b))
    used_a, used_b, n = set(), set(), 0
    for d, i, j in pairs:
        if d <= tol and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            n += 1
    return n


def agreement(reader_a: str, reader_b: str, root: Path | None = None) -> dict:
    """Runs both readers labelled; beats matched within ±1 frame as a share of each reader's
    beats (Meng et al. 2022 reported 93% between their readers)."""
    runs = n_a = n_b = agree = 0
    for path in sorted((Path(root or LABEL_DIR) / _safe(reader_a)).glob("*.json")):
        a = json.loads(path.read_text(encoding="utf-8"))
        b = load_label(reader_b, a["case"], root)
        if b is None:
            continue
        runs += 1
        n_a, n_b = n_a + len(a["ed_frames"]), n_b + len(b.frames)
        agree += match(tuple(a["ed_frames"]), b.frames)
    return {"runs": runs, "beats_a": n_a, "beats_b": n_b, "agree": agree,
            "share": agree / max(n_a, n_b) if max(n_a, n_b) else None}  # fmt: skip
