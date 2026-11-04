"""Config base: strict pydantic models (unknown keys are errors), YAML loading, and the git
commit for every saved output."""

import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

Range = tuple[float, float]  # sampled uniformly over [low, high]


class StrictModel(BaseModel):
    """A config section: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


def load_yaml(path: str | Path, model: type[StrictModel]) -> StrictModel:
    """``model`` from a YAML file."""
    return model.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def git_info() -> dict[str, Any]:
    """The repository's commit and whether the working tree has changes (``None`` outside
    git)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}
