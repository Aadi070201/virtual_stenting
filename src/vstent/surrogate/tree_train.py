"""Training of the tree FFR surrogate (``vstent train-tree-surrogate``, P3).

The same run layout and resumable state as the single-vessel trainer (``surrogate.train``):
``runs/<name>-<UTC timestamp>/`` holds ``config.yaml`` + ``run_meta.yaml``,
``tree_norm_stats.json``, ``metrics.csv``, ``best.pt`` (lowest validation outlet-FFR MAE) and
``summary.json``; the full state goes to ``checkpoints/<run-name>/`` after every epoch, the
STOP file is honoured and ``--resume`` continues exactly.
"""

import csv
import json
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vstent.checkpoints import (
    CHECKPOINTS_DIR,
    STATE_FILE,
    checkpoint_dir,
    clear_stop,
    load_state,
    read_status,
    resolve_checkpoint,
    restore_rng_states,
    rng_states,
    save_state,
    saved_epoch,
    stop_requested,
    utc_now,
    write_status,
)
from vstent.config import CONFIG_FILENAME, StrictModel, git_info, load_config, save_run_metadata
from vstent.data.synth import load_split
from vstent.surrogate.model import count_parameters
from vstent.surrogate.train import TrainConfig, resolve_device
from vstent.surrogate.tree_features import (
    RAW_TREE_KEYS,
    TREE_NORM_STATS,
    TreeNormStats,
    TreeRawInputs,
    prepare_tree_inputs,
)
from vstent.surrogate.tree_model import TreeFFRNet, TreeModelConfig

KIND = "tree_surrogate"
CHECKPOINT = "best.pt"
METRICS = "metrics.csv"
SUMMARY = "summary.json"


class TreeSurrogateConfig(StrictModel):
    """Full tree-surrogate training config (``configs/tree_surrogate.yaml``)."""

    name: str = "tree_unet"
    data_dir: str = "data/synth/tree_v1"
    runs_dir: str = "runs"
    seed: int = 0
    model: TreeModelConfig = TreeModelConfig()
    train: TrainConfig = TrainConfig(batch_size=128, eval_batch_size=512)


def load_tree_raw(
    data_dir: str | Path, split: str, max_samples: int | None = None
) -> tuple[TreeRawInputs, np.ndarray]:
    """Raw SI inputs and the tree-solver FFR (B, M, N) of one split."""
    arrays = load_split(data_dir, split, (*RAW_TREE_KEYS, "ffr"))
    if max_samples is not None:
        arrays = {k: v[:max_samples] for k, v in arrays.items()}
    return TreeRawInputs.from_arrays(arrays), arrays["ffr"].astype(np.float32)


class TreeDataset:
    """Standardised tree inputs and FFR targets as tensors (moved whole to the device)."""

    def __init__(self, raw: TreeRawInputs, ffr: np.ndarray, stats: TreeNormStats) -> None:
        self.tensors = [torch.from_numpy(x) for x in prepare_tree_inputs(raw, stats)]
        self.ffr = torch.from_numpy(np.ascontiguousarray(ffr, dtype=np.float32))

    @classmethod
    def from_split(cls, data_dir, split, stats, max_samples=None) -> "TreeDataset":
        raw, ffr = load_tree_raw(data_dir, split, max_samples)
        return cls(raw, ffr, stats)

    def __len__(self) -> int:
        return self.ffr.shape[0]

    @property
    def mask(self) -> torch.Tensor:
        return self.tensors[4]

    def to(self, device: torch.device) -> "TreeDataset":
        self.tensors = [t.to(device) for t in self.tensors]
        self.ffr = self.ffr.to(device)
        return self

    def batches(
        self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None
    ) -> Iterator[tuple[list[torch.Tensor], torch.Tensor]]:
        """Yield ([x_point, x_branch, parent, attach, mask], ffr) batches."""
        n = len(self)
        device = self.ffr.device
        order = (
            torch.randperm(n, generator=generator, device=device)
            if shuffle
            else torch.arange(n, device=device)
        )
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield [t[idx] for t in self.tensors], self.ffr[idx]


def tree_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, distal_weight: float
) -> torch.Tensor:
    """MSE over every sample of every real branch + ``distal_weight`` × MSE at the outlets."""
    w = mask.float()
    n = pred.shape[-1]
    curve = (((pred - target) ** 2).sum(dim=-1) * w).sum() / (w.sum() * n)
    outlet = (((pred[..., -1] - target[..., -1]) ** 2) * w).sum() / w.sum()
    return curve + distal_weight * outlet


def tree_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """MAE over every sample of real branches, at every outlet, and at the main outlet."""
    m = mask.bool()
    err = (pred - target).abs()
    return {
        "node_mae": float(err[m].mean()),
        "outlet_mae": float(err[..., -1][m].mean()),
        "main_outlet_mae": float(err[:, 0, -1].mean()),
    }


@torch.no_grad()
def predict_tree_batches(
    model: TreeFFRNet, tensors: list[torch.Tensor], batch_size: int = 512
) -> torch.Tensor:
    """FFR (B, M, N) in eval mode, float32, on the inputs' device."""
    model.eval()
    n = tensors[0].shape[0]
    return torch.cat(
        [model(*(t[i : i + batch_size] for t in tensors)) for i in range(0, n, batch_size)]
    )


def load_tree_checkpoint(run_dir: str | Path, device: torch.device | str = "cpu") -> TreeFFRNet:
    """Rebuild the best model of a run, in eval mode."""
    ckpt = torch.load(Path(run_dir) / CHECKPOINT, map_location=device, weights_only=True)
    model = TreeFFRNet(TreeModelConfig.model_validate(ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def train_tree_surrogate(
    config: TreeSurrogateConfig,
    *,
    run_dir: str | Path | None = None,
    checkpoint_root: str | Path = CHECKPOINTS_DIR,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Start a new run; returns its directory (stoppable and resumable, see module doc)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir or Path(config.runs_dir) / f"{config.name}-{stamp}")
    save_run_metadata(run_dir, config)
    ckpt_dir = checkpoint_dir(run_dir.name, checkpoint_root)
    return _fit(config, run_dir, ckpt_dir, log or (lambda _msg: None), resume=False)


def resume_tree_surrogate(
    name_or_path: str | Path,
    *,
    checkpoint_root: str | Path = CHECKPOINTS_DIR,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Continue a stopped or interrupted run from its last completed epoch."""
    ckpt_dir = resolve_checkpoint(name_or_path, checkpoint_root)
    status = read_status(ckpt_dir)
    if status.get("kind") != KIND:
        raise ValueError(f"{ckpt_dir} is a {status.get('kind')!r} checkpoint, not {KIND!r}")
    run_dir = Path(status["run_dir"])
    config = load_config(run_dir / CONFIG_FILENAME, TreeSurrogateConfig)
    return _fit(config, run_dir, ckpt_dir, log or (lambda _msg: None), resume=True)


def _fit(
    config: TreeSurrogateConfig,
    run_dir: Path,
    ckpt_dir: Path,
    log: Callable[[str], None],
    *,
    resume: bool,
) -> Path:
    tc = config.train
    torch.manual_seed(config.seed)
    device = resolve_device(tc.device)
    use_amp = tc.amp and device.type == "cuda"

    raw_train, ffr_train = load_tree_raw(config.data_dir, "train", tc.max_train_samples)
    if resume and (run_dir / TREE_NORM_STATS).exists():
        stats = TreeNormStats.load(run_dir / TREE_NORM_STATS)
    else:
        stats = TreeNormStats.fit(raw_train)
        stats.save(run_dir / TREE_NORM_STATS)
    train = TreeDataset(raw_train, ffr_train, stats).to(device)
    del raw_train, ffr_train
    val = TreeDataset.from_split(config.data_dir, "val", stats, tc.max_val_samples).to(device)

    model = TreeFFRNet(config.model).to(device)
    n_params = count_parameters(model)
    steps_per_epoch = -(-len(train) // tc.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.epochs * steps_per_epoch
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    best_mae, best_epoch, stale, train_seconds = float("inf"), -1, 0, 0.0
    rows: list[dict[str, Any]] = []
    resumes: list[dict[str, Any]] = []

    def save_best() -> None:  # the current weights are the best so far
        torch.save(
            {
                "model_config": config.model.model_dump(mode="json"),
                "state_dict": model.state_dict(),
                "epoch": best_epoch,
                "val_outlet_mae": best_mae,
            },
            run_dir / CHECKPOINT,
        )

    if resume and (ckpt_dir / STATE_FILE).exists():
        state = load_state(ckpt_dir)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng_states(state["rng"], generator)
        rows, resumes = state["history"], state["resumes"]
        best_mae, best_epoch = state["best_mae"], state["best_epoch"]
        stale, train_seconds = state["stale"], state["train_seconds"]
        if rows and best_epoch == len(rows) and saved_epoch(run_dir / CHECKPOINT) != best_epoch:
            save_best()  # interrupted after the state was saved, before best.pt
        if state["finished"]:
            log(f"Run {run_dir.name} already finished ({len(rows)} epochs); nothing to resume")
            return run_dir
    if resume:
        resumes.append(
            {"after_epoch": len(rows), "git_commit": git_info()["git_commit"], "utc": utc_now()}
        )
    resume_command = f"vstent train-tree-surrogate --resume {run_dir.name}"

    def status_fields(status: str) -> dict[str, Any]:
        return {
            "kind": KIND,
            "run_dir": run_dir.as_posix(),
            "status": status,
            "epoch": len(rows),
            "epochs": tc.epochs,
            "best_epoch": best_epoch,
            "best_metric": "val_outlet_mae",
            "best_value": best_mae if rows else None,
            "resume_command": resume_command,
            "resumes": resumes,
        }

    # Status exists from the start, so `vstent stop` / `vstent checkpoints` work during epoch 1.
    clear_stop(ckpt_dir)
    write_status(ckpt_dir, status_fields("running"))
    log(
        f"{'Resuming' if resume else 'Run'} {run_dir} "
        f"{f'after epoch {len(rows)} ' if resume else ''}| device {gpu or device} | "
        f"amp {use_amp} | {n_params:,} parameters | train {len(train):,} / val {len(val):,} "
        f"trees | checkpoints in {ckpt_dir}"
    )

    def save_checkpoint(status: str) -> None:
        save_state(
            ckpt_dir,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_states(generator),
                "history": rows,
                "resumes": resumes,
                "best_mae": best_mae,
                "best_epoch": best_epoch,
                "stale": stale,
                "train_seconds": train_seconds,
                "finished": status == "finished",
            },
            status_fields(status),
        )

    for epoch in range(len(rows) + 1, tc.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        for xs, y in train.batches(tc.batch_size, shuffle=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(*xs)
            loss = tree_loss(pred, y, xs[4], tc.distal_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach() * y.shape[0]

        pred = predict_tree_batches(model, val.tensors, tc.eval_batch_size)
        metrics = tree_metrics(pred, val.ffr, val.mask)
        row = {
            "epoch": epoch,
            "train_loss": float(loss_sum) / len(train),
            "val_loss": float(tree_loss(pred, val.ffr, val.mask, tc.distal_weight)),
            "val_outlet_mae": metrics["outlet_mae"],
            "val_node_mae": metrics["node_mae"],
            "val_main_outlet_mae": metrics["main_outlet_mae"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        train_seconds += row["seconds"]
        improved = row["val_outlet_mae"] < best_mae
        if improved:  # best.pt is written after the state (see ``saved_epoch``)
            best_mae, best_epoch, stale = row["val_outlet_mae"], epoch, 0
        else:
            stale += 1
        log(
            f"epoch {epoch:3d}  train {row['train_loss']:.2e}  val {row['val_loss']:.2e}  "
            f"outlet MAE {row['val_outlet_mae']:.4f}  node MAE {row['val_node_mae']:.4f}  "
            f"{row['seconds']:.1f}s{'  *' if improved else ''}"
        )
        with (run_dir / METRICS).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if stale >= tc.patience and epoch >= tc.min_epochs:
            log(f"Early stop: no improvement for {stale} epochs")
            break
        if stop_requested(ckpt_dir):
            save_checkpoint("stopped")
            if improved:
                save_best()
            clear_stop(ckpt_dir)
            log(f"Stopped after epoch {epoch}. Resume with: {resume_command}")
            return run_dir
        save_checkpoint("running")
        if improved:
            save_best()

    save_checkpoint("finished")
    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_outlet_mae": best_mae,
        "epochs_run": len(rows),
        "train_seconds": train_seconds,
        "resumes": resumes,
        "n_parameters": n_params,
        "n_train": len(train),
        "n_val": len(val),
        "device": str(device),
        "gpu": gpu,
        "amp": use_amp,
        "torch": torch.__version__,
    }
    (run_dir / SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    log(f"Best val outlet MAE {best_mae:.4f} at epoch {best_epoch}")
    return run_dir
