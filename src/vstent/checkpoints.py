"""Resumable training checkpoints, shared by every trainer.

Layout (git-ignored)::

    checkpoints/<run-name>/
        last.pt      full training state after the latest completed epoch (model, optimiser,
                     scheduler, AMP scaler, RNG states, history); replaced atomically
        state.json   readable status: kind, run_dir, epoch, epochs, best metric, status,
                     resume command, resume history
        STOP         if present, the trainer saves and exits after the current epoch

``<run-name>`` is the name of the matching ``runs/<run-name>/`` directory, which keeps the
deliverables (config, metrics, best model). Stop with ``vstent stop <run-name>``, list with
``vstent checkpoints``, continue with the ``resume_command`` shown there.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

CHECKPOINTS_DIR = Path("checkpoints")
STATE_FILE = "last.pt"
STATUS_FILE = "state.json"
STOP_FILE = "STOP"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def checkpoint_dir(run_name: str, root: str | Path = CHECKPOINTS_DIR) -> Path:
    """Checkpoint folder of a run."""
    return Path(root) / run_name


def resolve_checkpoint(name_or_path: str | Path, root: str | Path = CHECKPOINTS_DIR) -> Path:
    """Accept a run name, ``checkpoints/<name>`` or ``runs/<name>``; return the checkpoint dir."""
    path = Path(name_or_path)
    candidates = [path, Path(root) / path.name]
    for candidate in candidates:
        if (candidate / STATUS_FILE).exists():
            return candidate
    raise FileNotFoundError(f"no checkpoint found for {name_or_path!s} (looked in {root})")


def save_state(ckpt_dir: Path, state: dict[str, Any], status: dict[str, Any]) -> None:
    """Write ``last.pt`` atomically (temp file + replace) and update ``state.json``."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / (STATE_FILE + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, ckpt_dir / STATE_FILE)
    write_status(ckpt_dir, status)


def load_state(ckpt_dir: Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Load ``last.pt`` (tensors and plain Python containers only)."""
    return torch.load(ckpt_dir / STATE_FILE, map_location=map_location, weights_only=True)


def write_status(ckpt_dir: Path, status: dict[str, Any]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps({**status, "updated_utc": utc_now()}, indent=2)
    (ckpt_dir / STATUS_FILE).write_text(text, encoding="utf-8", newline="\n")


def read_status(ckpt_dir: Path) -> dict[str, Any]:
    return json.loads((ckpt_dir / STATUS_FILE).read_text(encoding="utf-8"))


def list_checkpoints(root: str | Path = CHECKPOINTS_DIR) -> list[dict[str, Any]]:
    """Status of every checkpoint under ``root``, most recently updated first."""
    root = Path(root)
    if not root.exists():
        return []
    rows = [
        {"name": d.name, **read_status(d)} for d in root.iterdir() if (d / STATUS_FILE).exists()
    ]
    return sorted(rows, key=lambda r: r.get("updated_utc", ""), reverse=True)


def request_stop(ckpt_dir: Path) -> None:
    """Ask a running trainer to save and exit after its current epoch."""
    (ckpt_dir / STOP_FILE).write_text(utc_now(), encoding="utf-8")


def stop_requested(ckpt_dir: Path) -> bool:
    return (ckpt_dir / STOP_FILE).exists()


def clear_stop(ckpt_dir: Path) -> None:
    (ckpt_dir / STOP_FILE).unlink(missing_ok=True)


def rng_states(generator: torch.Generator | None = None) -> dict[str, Any]:
    """Torch CPU / CUDA RNG states, plus a trainer's own generator if given."""
    states: dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        states["generator"] = generator.get_state()
    return states


def restore_rng_states(states: dict[str, Any], generator: torch.Generator | None = None) -> None:
    torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    if generator is not None and "generator" in states:
        generator.set_state(states["generator"])
