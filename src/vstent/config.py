"""Pydantic config models, YAML loading and run metadata.

Every run directory receives the fully resolved config (``config.yaml``) and the
git commit it was produced from (``run_meta.yaml``), so that any result can be
reproduced from what is saved next to it.
"""

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

from vstent import __version__

CONFIG_FILENAME = "config.yaml"
META_FILENAME = "run_meta.yaml"


class StrictModel(BaseModel):
    """Base class for all configs: unknown keys are rejected and fields are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping. An empty file gives ``{}``."""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    return data


def load_config(path: str | Path, model: type[ConfigT]) -> ConfigT:
    """Load a YAML file and validate it against ``model`` (defaults filled in)."""
    return model.model_validate(load_yaml(path))


def dump_config(config: BaseModel, path: str | Path) -> None:
    """Write the fully resolved config (including defaults) to YAML."""
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)


def git_info() -> dict[str, Any]:
    """Return the current commit hash and whether the working tree is dirty.

    Values are ``None`` when git or the repository is unavailable.
    """
    cwd = Path(__file__).resolve().parent

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


def save_run_metadata(run_dir: str | Path, config: BaseModel) -> Path:
    """Create ``run_dir`` and save the resolved config plus provenance metadata.

    Returns the run directory as a ``Path``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, run_dir / CONFIG_FILENAME)
    meta = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_class": type(config).__name__,
    }
    with (run_dir / META_FILENAME).open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return run_dir
