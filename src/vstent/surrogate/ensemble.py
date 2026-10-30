"""Ensemble of surrogates trained with different seeds (``vstent train-ensemble``, P1).

Layout::

    runs/<name>-<UTC timestamp>/
        config.yaml, run_meta.yaml      resolved EnsembleConfig + git commit
        surrogate_config.yaml           the resolved member config (seed set per member)
        ensemble.json                   members (seed, run directory, status)
        members/<name>-<stamp>-s<seed>/ ordinary surrogate runs (see ``surrogate.train``)

Every member is an ordinary resumable run with its own ``checkpoints/<member-name>/``. The
ensemble has a status of its own in ``checkpoints/<ensemble-name>/`` (kind
``surrogate_ensemble``): ``vstent stop <ensemble-name>`` stops the member that is training after
its current epoch and starts no further member; ``vstent train-ensemble --resume
<ensemble-name>`` skips finished members, continues the stopped one and trains the rest.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from vstent.checkpoints import (
    CHECKPOINTS_DIR,
    STATUS_FILE,
    checkpoint_dir,
    clear_stop,
    read_status,
    request_stop,
    resolve_checkpoint,
    stop_requested,
    write_status,
)
from vstent.config import (
    CONFIG_FILENAME,
    StrictModel,
    dump_config,
    load_config,
    save_run_metadata,
)
from vstent.surrogate.train import SUMMARY, SurrogateConfig, resume_surrogate, train_surrogate

KIND = "surrogate_ensemble"
MEMBER_CONFIG = "surrogate_config.yaml"
ENSEMBLE_FILE = "ensemble.json"
MEMBERS_DIR = "members"


class EnsembleConfig(StrictModel):
    """Ensemble config (``configs/ensemble.yaml``): one member per seed, same settings."""

    name: str = "unet1d_ens5"
    runs_dir: str = "runs"
    base_config: str = "configs/surrogate.yaml"
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])

    @field_validator("seeds")
    @classmethod
    def _distinct(cls, v: list[int]) -> list[int]:
        if len(v) < 2 or len(set(v)) != len(v):
            raise ValueError("an ensemble needs at least two distinct seeds")
        return v


def member_name(ensemble_dir: Path, seed: int) -> str:
    """Run name of the member with ``seed`` (unique across ensembles, used for checkpoints)."""
    return f"{ensemble_dir.name}-s{seed}"


def member_dirs(ensemble_dir: str | Path) -> list[Path]:
    """Run directories of the members, in seed order of the config."""
    ensemble_dir = Path(ensemble_dir)
    cfg = load_config(ensemble_dir / CONFIG_FILENAME, EnsembleConfig)
    return [ensemble_dir / MEMBERS_DIR / member_name(ensemble_dir, s) for s in cfg.seeds]


def surrogate_config(path: str | Path) -> SurrogateConfig:
    """Surrogate training config of a run or exported model directory: the member config
    (``surrogate_config.yaml``) for an ensemble, else ``config.yaml``."""
    path = Path(path)
    member = path / MEMBER_CONFIG
    return load_config(member if member.exists() else path / CONFIG_FILENAME, SurrogateConfig)


def train_ensemble(
    config: EnsembleConfig,
    *,
    run_dir: str | Path | None = None,
    checkpoint_root: str | Path = CHECKPOINTS_DIR,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Start a new ensemble; returns its run directory (members trained one after another)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir or Path(config.runs_dir) / f"{config.name}-{stamp}")
    base = load_config(config.base_config, SurrogateConfig)
    save_run_metadata(run_dir, config)
    dump_config(base, run_dir / MEMBER_CONFIG)
    return _train_members(run_dir, checkpoint_root, log or (lambda _msg: None))


def resume_ensemble(
    name_or_path: str | Path,
    *,
    checkpoint_root: str | Path = CHECKPOINTS_DIR,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Continue a stopped or interrupted ensemble (name, ``runs/<name>`` or checkpoint dir)."""
    ckpt_dir = resolve_checkpoint(name_or_path, checkpoint_root)
    status = read_status(ckpt_dir)
    if status.get("kind") != KIND:
        raise ValueError(f"{ckpt_dir} is a {status.get('kind')!r} checkpoint, not {KIND!r}")
    return _train_members(Path(status["run_dir"]), checkpoint_root, log or (lambda _msg: None))


def stop_ensemble(ckpt_dir: Path) -> str | None:
    """Ask a running ensemble to stop: its current member after this epoch, then the loop.

    Members watch the ensemble's STOP file themselves, so a stop sent while a member is still
    loading data is not lost. Returns the name of the current member (``None`` if none).
    """
    request_stop(ckpt_dir)
    return read_status(ckpt_dir).get("current_member")


def _member_status(name: str, checkpoint_root: Path) -> str | None:
    path = checkpoint_dir(name, checkpoint_root) / STATUS_FILE
    return json.loads(path.read_text(encoding="utf-8"))["status"] if path.exists() else None


def _train_members(run_dir: Path, checkpoint_root: str | Path, log: Callable[[str], None]) -> Path:
    checkpoint_root = Path(checkpoint_root)
    cfg = load_config(run_dir / CONFIG_FILENAME, EnsembleConfig)
    base = load_config(run_dir / MEMBER_CONFIG, SurrogateConfig)
    ckpt_dir = checkpoint_dir(run_dir.name, checkpoint_root)
    resume_command = f"vstent train-ensemble --resume {run_dir.name}"
    names = [member_name(run_dir, s) for s in cfg.seeds]

    def write(status: str, current: str | None) -> None:
        members = [
            {
                "seed": seed,
                "run_dir": (run_dir / MEMBERS_DIR / name).as_posix(),
                "status": _member_status(name, checkpoint_root) or "not started",
            }
            for seed, name in zip(cfg.seeds, names, strict=True)
        ]
        n_done = sum(m["status"] == "finished" for m in members)
        write_status(
            ckpt_dir,
            {
                "kind": KIND,
                "run_dir": run_dir.as_posix(),
                "status": status,
                "epoch": n_done,  # finished members (``vstent checkpoints`` shows "members")
                "epochs": len(members),
                "current_member": current,
                "resume_command": resume_command,
            },
        )
        (run_dir / ENSEMBLE_FILE).write_text(
            json.dumps({"seeds": cfg.seeds, "members": members}, indent=2),
            encoding="utf-8",
            newline="\n",
        )

    clear_stop(ckpt_dir)
    write("running", None)
    for seed, name in zip(cfg.seeds, names, strict=True):
        member_dir = run_dir / MEMBERS_DIR / name
        state = _member_status(name, checkpoint_root)
        if state == "finished":
            log(f"Member seed {seed}: finished earlier")
            continue
        if stop_requested(ckpt_dir):
            break
        write("running", name)
        log(f"Member seed {seed} ({len(cfg.seeds)} members): {member_dir}")
        if state is None:
            member = base.model_copy(update={"seed": seed, "name": name})
            train_surrogate(
                member, run_dir=member_dir, checkpoint_root=checkpoint_root, log=log,
                parent_stop=ckpt_dir,
            )  # fmt: skip
        else:
            resume_surrogate(name, checkpoint_root=checkpoint_root, log=log, parent_stop=ckpt_dir)
        if _member_status(name, checkpoint_root) != "finished":
            break  # the member was stopped
    done = all(_member_status(n, checkpoint_root) == "finished" for n in names)
    clear_stop(ckpt_dir)
    write("finished" if done else "stopped", None)
    if not done:
        log(f"Ensemble stopped. Resume with: {resume_command}")
        return run_dir
    summary: dict[str, Any] = {"members": []}
    for seed, name in zip(cfg.seeds, names, strict=True):
        s = json.loads((run_dir / MEMBERS_DIR / name / SUMMARY).read_text(encoding="utf-8"))
        summary["members"].append(
            {"seed": seed, "run": name, **{k: s[k] for k in ("best_epoch", "best_val_distal_mae")}}
        )
        summary["train_seconds"] = summary.get("train_seconds", 0.0) + s["train_seconds"]
    (run_dir / SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    log(f"Ensemble finished: {len(names)} members in {run_dir}")
    return run_dir
