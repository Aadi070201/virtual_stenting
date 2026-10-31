"""FDA benchmark nozzle: inter-laboratory wall-pressure measurements (public domain).

The U.S. FDA's first computational round-robin used an axisymmetric nozzle: 0.012 m pipe, a 20°
cone, a 0.04 m long and 0.004 m wide throat, and a sudden step back to 0.012 m (Stewart et al.,
Cardiovasc Eng Technol 2012;3:139–160). Flow one way gives a conical collector and a sudden
expansion (``SE``), the other way a sudden contraction and a conical diffuser (``CD``). Wall
pressures were measured with a pressure-tap model in three laboratories at throat Reynolds
numbers 500 (laminar), 2000 (transitional), 3500, 5000 and 6500 (turbulent), with a Newtonian
blood analogue (1056 kg/m³, 3.5 mPa·s).

Data: github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks, ``Nozzle/Data`` (Round Robin 1 data
sets v2.0, Hariharan & Malinauskas, doi:10.17917/C78G69), pinned to commit :data:`COMMIT`.
A work of the U.S. Government, not subject to copyright (17 U.S.C. §105); the repository asks
that derived works say so. Files go to ``data/raw/fda_nozzle/`` (git-ignored). Units: SI as in
the files (m, Pa, m³/s, kg/m³, Pa·s).
"""

import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent.data.arcade import download, file_digest

COMMIT = "76cdd3423845a398fc75f8121be4130caca4de90"
BASE_URL = (
    f"https://raw.githubusercontent.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks/{COMMIT}/"
    "Nozzle/Data/"
)
RAW_DIR = Path("data/raw/fda_nozzle")
LICENCE = "U.S. Government work, public domain (17 U.S.C. §105); FDA asks derived works to say so"
DOI = "10.17917/C78G69"
REYNOLDS = (500, 2000, 3500, 5000, 6500)
# archive name → SHA-256 (computed on download at the pinned commit)
FILES: dict[str, str] = {
    "SE_exp_0500.zip": "c33d8d604c072edd0298890274e9b8d625bb39281e2b691c51dac639aaf95d58",
    "SE_exp_2000.zip": "b03a1f115eb952c030b94379ac41f85542b39639751c0971162bfd94ca02f7ea",
    "SE_exp_3500.zip": "abb47dacee13beeb811cdf3514d02eb31fb31da0f7d9bf272cad4f7e2c02d3a1",
    "SE_exp_5000.zip": "2491e33610bdfff03a259b11e86c9073d5a901c8d6f2bd5d1a46ed186cd82b75",
    "SE_exp_6500.zip": "5e326233651435b45631fcffd1fd5d972a69725036566600f086667819d2551f",
    "CD_Re_500.zip": "32ac63b390bd04b3c3908bc82c046a2cd2a63513f6f40b05b87f7a94342d3f10",
    "CD_Re_2000.zip": "9d931249aab7f3688031b60856dedd9de19e238f1ab02f688caca3fb487fa49d",
    "CD_Re_3500.zip": "e4fc21825322b7e6885c631738207f77e111f870b2cd2d9956584e733c209d3f",
    "CD_Re_5000.zip": "81392fc4c14c1c67e70258b6cc322a620ff4494298477b237e6c9d4222875e26",
    "CD_Re_6500.zip": "4e86d9d3ab26c72f41524e0c9727223b1a276b7f42cc423f3c415e544ac6d2ca",
}
SOURCE_FILE = "SOURCE.json"


def archive_name(orientation: str, reynolds: int) -> str:
    """The data archive for ``orientation`` (``SE`` or ``CD``) at throat Reynolds number."""
    return f"SE_exp_{reynolds:04d}.zip" if orientation == "SE" else f"CD_Re_{reynolds}.zip"


def fetch_fda_nozzle(
    raw_dir: str | Path = RAW_DIR, *, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Download (unless already verified) every nozzle data archive; writes ``SOURCE.json``."""
    log = log or (lambda _msg: None)
    raw_dir = Path(raw_dir)
    for name, sha in FILES.items():
        path = raw_dir / name
        if path.exists() and file_digest(path, "sha256") == sha:
            continue
        log(f"Downloading {name}")
        download(BASE_URL + name, path, expected_sha256=sha)
    source = {
        "dataset": "FDA benchmark nozzle, Round Robin 1 experimental data v2.0",
        "repository": "https://github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks",
        "commit": COMMIT,
        "doi": DOI,
        "files": FILES,
        "licence": LICENCE,
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8", newline="\n")
    return source


@dataclass(frozen=True, eq=False)
class NozzleRun:
    """One laboratory's wall-pressure measurement.

    Attributes:
        orientation: ``SE`` (collector → throat → sudden expansion) or ``CD`` (sudden
            contraction → throat → conical diffuser); flow towards +z.
        reynolds: throat Reynolds number.
        lab: the data set's blinded code.
        rho, mu, q: density (kg/m³), viscosity (Pa·s), flow rate (m³/s) from the file.
        z, p: wall-tap positions (m; 0 at the step) and pressures (Pa, gauge).
    """

    orientation: str
    reynolds: int
    lab: str
    rho: float
    mu: float
    q: float
    z: np.ndarray
    p: np.ndarray


def parse_run(text: str, orientation: str, reynolds: int) -> NozzleRun:
    """Parse one experiment file (``key value`` header lines, then ``plot-*`` sections each
    followed by a point count and that many ``z value`` rows)."""
    lines = [ln.strip() for ln in text.splitlines()]
    if "deleted-plot-wall-distribution-pressure" in lines:
        raise ValueError("wall pressures marked deleted by the data curators")
    header: dict[str, str] = {}
    z = p = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln == "plot-wall-distribution-pressure":
            n = int(lines[i + 1])
            rows = np.array([[float(v) for v in lines[i + 2 + k].split()[:2]] for k in range(n)])
            z, p = rows[:, 0], rows[:, 1]
            i += 2 + n
            continue
        if ln and not ln.startswith("plot-") and " " in ln:
            key, value = ln.split(None, 1)
            header.setdefault(key, value.strip().strip('"'))
        i += 1
    if z is None:
        raise ValueError("no wall pressures (velocity data only)")
    order = np.argsort(z)
    return NozzleRun(
        orientation=orientation,
        reynolds=reynolds,
        lab=header.get("dataset-code", "?"),
        rho=float(header["fluid-density"]),
        mu=float(header["fluid-viscosity"]),
        q=float(header["fluid-volumetric-flow-rate"]),
        z=z[order],
        p=p[order],
    )


def load_runs(
    orientation: str, reynolds: int, raw_dir: str | Path = RAW_DIR
) -> tuple[list[NozzleRun], list[dict[str, str]]]:
    """Every laboratory's wall-pressure run in one archive, and the files without usable wall
    pressures with the reason (``{"file", "reason"}``)."""
    runs, skipped = [], []
    with zipfile.ZipFile(Path(raw_dir) / archive_name(orientation, reynolds)) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".txt"):
                continue
            text = io.TextIOWrapper(zf.open(name), encoding="latin-1").read()
            try:
                runs.append(parse_run(text, orientation, reynolds))
            except (ValueError, KeyError, IndexError) as exc:
                skipped.append({"file": Path(name).name, "reason": str(exc)})
    return runs, skipped
