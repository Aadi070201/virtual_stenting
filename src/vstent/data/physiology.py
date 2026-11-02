"""Open invasive coronary physiology data (P4e): DEFINE-FLOW and CONTRAST. Not angiograms.

- **DEFINE-FLOW** (Johnson, Kirkeeide and Gould, JACC Cardiovasc Interv 2021,
  doi:10.1016/j.jcin.2021.07.041; data doi:10.5061/dryad.h18931zm6, Zenodo record 5508241):
  455 subjects, 668 lesions, site and core-lab FFR and CFR, Doppler average peak velocity (APV)
  at baseline and at hyperemia, aortic and distal pressures, and raw pressure / Doppler / ECG
  tracings (one CSV per core-lab-accepted measurement).
- **CONTRAST** (Johnson et al., JACC Cardiovasc Interv 2016, doi:10.1016/j.jcin.2015.12.273;
  data doi:10.5061/dryad.f76nv, Zenodo record 4992130): raw Pa / Pd / ECG tracings of 763
  subjects at rest, during intracoronary contrast and during adenosine, 100 Hz.

Both are CC0 1.0 and open (no sign-in). Files go to ``data/raw/<name>/`` (git-ignored) and are
verified against the MD5 that Zenodo publishes; archives are extracted with tar's ``data``
filter (no absolute paths, links or files outside the folder). The downloaded files are data
only: nothing in them is executed.
"""

import json
import tarfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vstent.data.arcade import download, file_digest

LICENCE = "CC0 1.0 (Zenodo / Dryad); the authors ask that the paper be cited"
SOURCE_FILE = "SOURCE.json"
DATASETS: dict[str, dict[str, Any]] = {
    "define_flow": {
        "title": "DEFINE-FLOW: Combined pressure and flow measurements to guide treatment of "
        "coronary stenoses",
        "record": "https://zenodo.org/records/5508241",
        "doi": "10.5061/dryad.h18931zm6",
        "paper": "Johnson NP, Kirkeeide RL, Gould KL et al. JACC Cardiovasc Interv 2021; "
        "doi:10.1016/j.jcin.2021.07.041",
        "files": {
            "README.txt": "4d6e6a924e3ac17df8b397ae5b8e20e7",
            "DFLOW_data.tgz": "44dd4875cca5fb901812d5ce4295d3e6",
            "DFLOW_tracings.tgz": "073c57c66c7f7cbec9899f7bc49cbec5",
        },
    },
    "contrast": {
        "title": "CONTRAST: Continuum of vasodilator stress from rest to contrast medium to "
        "adenosine hyperemia for fractional flow reserve assessment",
        "record": "https://zenodo.org/records/4992130",
        "doi": "10.5061/dryad.f76nv",
        "paper": "Johnson NP, Jeremias A, Zimmermann FM et al. JACC Cardiovasc Interv 2016; "
        "doi:10.1016/j.jcin.2015.12.273",
        "files": {
            "README_for_CONTRAST.txt": "aea39aca788cb329c4bc3f0088ab3370",
            "CONTRAST.tgz": "72908ccf181fb10206e1142cdf7dccb7",
        },
    },
}
RAW_ROOT = Path("data/raw")


def _url(record: str, name: str) -> str:
    return f"{record}/files/{name}?download=1"


def fetch_physiology(
    name: str, raw_dir: str | Path | None = None, *, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Download (unless already verified), MD5-check and extract dataset ``name``
    (``define_flow`` or ``contrast``); writes ``SOURCE.json``. Raises ``ValueError`` on a
    checksum mismatch."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; use {sorted(DATASETS)}")
    log = log or (lambda _msg: None)
    meta = DATASETS[name]
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_ROOT / name
    for fname, md5 in meta["files"].items():
        path = raw_dir / fname
        if not (path.exists() and file_digest(path, "md5") == md5):
            log(f"Downloading {fname}")
            download(_url(meta["record"], fname), path, expected_md5=md5, log=log)
        if fname.endswith(".tgz"):
            target = raw_dir / fname.removesuffix(".tgz")
            if not target.exists():
                log(f"Extracting {fname}")
                with tarfile.open(path, "r:gz") as tar:
                    tar.extractall(target, filter="data")
    source = {**{k: v for k, v in meta.items()}, "licence": LICENCE,
              "verified_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    (raw_dir / SOURCE_FILE).write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return source
