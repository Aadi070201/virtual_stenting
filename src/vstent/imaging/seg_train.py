"""Training of the vessel-segmentation U-Net on ARCADE ``syntax`` masks (``vstent train-seg``).

A run directory ``runs/<name>-<UTC timestamp>/`` receives ``config.yaml`` + ``run_meta.yaml``
(git commit), ``metrics.csv``, ``best.pt`` (highest mean validation Dice) and ``summary.json``.
The resumable state lives in ``checkpoints/<run-name>/`` (see ``vstent.checkpoints``).
"""

import csv
import json
import os
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from torch.utils.data import DataLoader, Dataset, Sampler

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
    stop_requested,
    utc_now,
    write_status,
)
from vstent.config import CONFIG_FILENAME, StrictModel, git_info, load_config, save_run_metadata
from vstent.data.arcade import load_split
from vstent.imaging.segment import (
    IMAGE_SIZE,
    THRESHOLD,
    AugmentConfig,
    augment,
    dice_score,
    resize,
    standardise,
    to_unit,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows cache without symlinks
import segmentation_models_pytorch as smp  # noqa: E402

KIND = "segmentation"
CHECKPOINT = "best.pt"
METRICS = "metrics.csv"
SUMMARY = "summary.json"


class SegModelConfig(StrictModel):
    """U-Net from segmentation-models-pytorch with one input channel and one output class."""

    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    image_size: int = Field(IMAGE_SIZE, ge=32)


class SegTrainConfig(StrictModel):
    """Optimisation settings."""

    epochs: int = Field(60, gt=0)
    batch_size: int = Field(8, gt=0)
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    dice_weight: float = Field(1.0, ge=0)
    patience: int = Field(15, gt=0)
    grad_clip: float = Field(1.0, gt=0)
    amp: bool = True
    device: str = "auto"
    workers: int = Field(2, ge=0)  # each worker holds a copy of the images (~0.5 GB)
    eval_batch_size: int = Field(16, gt=0)
    max_train_images: int | None = Field(None, gt=0)
    max_val_images: int | None = Field(None, gt=0)


class SegConfig(StrictModel):
    """Full segmentation-training config (``configs/seg.yaml``)."""

    name: str = "seg_unet_r34"
    raw_dir: str = "data/raw/arcade"
    runs_dir: str = "runs"
    seed: int = 0
    model: SegModelConfig = SegModelConfig()
    train: SegTrainConfig = SegTrainConfig()
    augment: AugmentConfig = AugmentConfig()


def resolve_device(spec: str = "auto") -> torch.device:
    """``"auto"`` -> CUDA if available, else CPU."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def build_model(cfg: SegModelConfig, *, pretrained: bool = True) -> torch.nn.Module:
    """smp U-Net with ``cfg.encoder``; ImageNet weights only when ``pretrained``."""
    return smp.Unet(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights if pretrained else None,
        in_channels=1,
        classes=1,
    )


def prepare_pair(
    image: np.ndarray, mask: np.ndarray, size: int, cfg: AugmentConfig | None, seed=None
) -> tuple[np.ndarray, np.ndarray]:
    """One training / validation sample: input (1, size, size) float32, target (1, size, size).

    With ``cfg`` the pair is augmented with ``np.random.default_rng(seed)``; the input then goes
    through the same ``resize`` + ``standardise`` as ``segment.preprocess_image``.
    """
    if cfg is None:
        unit, msk = to_unit(image), mask
    else:
        unit, msk = augment(image, mask, np.random.default_rng(seed), cfg)
    x = standardise(resize(unit, size))[None]
    y = resize(msk, size, cv2.INTER_NEAREST)[None].astype(np.float32)
    return x, y


class SegDataset(Dataset):
    """ARCADE images + vessel masks; keys are (index, epoch seed) pairs from the sampler."""

    def __init__(
        self, images: np.ndarray, masks: np.ndarray, size: int, cfg: AugmentConfig | None
    ) -> None:
        self.images, self.masks, self.size, self.cfg = images, masks, size, cfg

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, key: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        i, epoch_seed = key
        x, y = prepare_pair(self.images[i], self.masks[i], self.size, self.cfg, [epoch_seed, i])
        return torch.from_numpy(x), torch.from_numpy(y)


class EpochSampler(Sampler):
    """Shuffled indices, each paired with a per-epoch seed drawn from ``generator``.

    Augmentations then depend only on the generator state (saved in checkpoints), not on the
    number of DataLoader workers.
    """

    def __init__(self, n: int, generator: torch.Generator) -> None:
        self.n, self.generator = n, generator

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[tuple[int, int]]:
        seed = int(torch.randint(2**31 - 1, (1,), generator=self.generator))
        order = torch.randperm(self.n, generator=self.generator).tolist()
        return iter([(i, seed) for i in order])


def _worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(1)


def seg_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float) -> torch.Tensor:
    """BCE + ``dice_weight`` × soft Dice loss (batch-level), in float32."""
    logits = logits.float()
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (prob.sum() + target.sum() + 1.0)
    return F.binary_cross_entropy_with_logits(logits, target) + dice_weight * dice


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module, x: torch.Tensor, batch_size: int, device: torch.device, amp: bool
) -> np.ndarray:
    """Vessel probabilities (N, H, W) float32 for standardised inputs ``x`` (N, 1, H, W)."""
    model.eval()
    out = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(xb)
        out.append(torch.sigmoid(logits.float())[:, 0].cpu().numpy())
    return np.concatenate(out)


def load_checkpoint(run_dir: str | Path, device: torch.device | str = "cpu") -> torch.nn.Module:
    """Rebuild the best model of a run, in eval mode (no weight download)."""
    ckpt = torch.load(Path(run_dir) / CHECKPOINT, map_location=device, weights_only=True)
    model = build_model(SegModelConfig.model_validate(ckpt["model_config"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def validation_tensors(images: np.ndarray, masks: np.ndarray, size: int):
    """Un-augmented standardised inputs (N, 1, size, size) and binary targets (N, size, size)."""
    pairs = [prepare_pair(im, m, size, None) for im, m in zip(images, masks, strict=True)]
    x = torch.from_numpy(np.stack([p[0] for p in pairs]))
    y = np.stack([p[1][0] for p in pairs]).astype(bool)
    return x, y


def train_seg(
    config: SegConfig,
    *,
    run_dir: str | Path | None = None,
    checkpoint_root: str | Path = CHECKPOINTS_DIR,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Start a new segmentation run; returns the run directory (stoppable, resumable)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir or Path(config.runs_dir) / f"{config.name}-{stamp}")
    save_run_metadata(run_dir, config)
    ckpt_dir = checkpoint_dir(run_dir.name, checkpoint_root)
    return _fit(config, run_dir, ckpt_dir, log or (lambda _msg: None), resume=False)


def resume_seg(
    name_or_path: str | Path,
    *,
    checkpoint_root: str | Path = CHECKPOINTS_DIR,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Continue a stopped or interrupted segmentation run from its last completed epoch."""
    ckpt_dir = resolve_checkpoint(name_or_path, checkpoint_root)
    status = read_status(ckpt_dir)
    if status.get("kind") != KIND:
        raise ValueError(f"{ckpt_dir} is a {status.get('kind')!r} checkpoint, not {KIND!r}")
    run_dir = Path(status["run_dir"])
    config = load_config(run_dir / CONFIG_FILENAME, SegConfig)
    return _fit(config, run_dir, ckpt_dir, log or (lambda _msg: None), resume=True)


def _fit(
    config: SegConfig,
    run_dir: Path,
    ckpt_dir: Path,
    log: Callable[[str], None],
    *,
    resume: bool,
) -> Path:
    tc, size = config.train, config.model.image_size
    torch.manual_seed(config.seed)
    device = resolve_device(tc.device)
    use_amp = tc.amp and device.type == "cuda"

    train_split = load_split(config.raw_dir, "syntax", "train", tc.max_train_images)
    val_split = load_split(config.raw_dir, "syntax", "val", tc.max_val_images)
    train = SegDataset(train_split.images, train_split.masks, size, config.augment)
    x_val, y_val = validation_tensors(val_split.images, val_split.masks, size)

    # ImageNet weights are only downloaded for a new run; a resumed run loads its own state.
    model = build_model(config.model, pretrained=not resume).to(device)
    torch.backends.cudnn.benchmark = device.type == "cuda"  # fixed input size
    n_params = sum(p.numel() for p in model.parameters())
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train,
        batch_size=tc.batch_size,
        sampler=EpochSampler(len(train), generator),
        num_workers=tc.workers,
        persistent_workers=tc.workers > 0,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init,
        drop_last=len(train) > tc.batch_size,
    )
    steps_per_epoch = len(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.epochs * steps_per_epoch
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    best_dice, best_epoch, stale, train_seconds = -1.0, -1, 0, 0.0
    rows: list[dict[str, Any]] = []
    resumes: list[dict[str, Any]] = []
    if resume and (ckpt_dir / STATE_FILE).exists():
        state = load_state(ckpt_dir)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng_states(state["rng"], generator)
        rows, resumes = state["history"], state["resumes"]
        best_dice, best_epoch = state["best_dice"], state["best_epoch"]
        stale, train_seconds = state["stale"], state["train_seconds"]
        if state["finished"]:
            log(f"Run {run_dir.name} already finished ({len(rows)} epochs); nothing to resume")
            return run_dir
    elif resume:  # killed before the first checkpoint: start over, with pretrained weights
        model.load_state_dict(build_model(config.model).state_dict())
    if resume:
        resumes.append(
            {"after_epoch": len(rows), "git_commit": git_info()["git_commit"], "utc": utc_now()}
        )
    resume_command = f"vstent train-seg --resume {run_dir.name}"

    def status_fields(status: str) -> dict[str, Any]:
        return {
            "kind": KIND,
            "run_dir": run_dir.as_posix(),
            "status": status,
            "epoch": len(rows),
            "epochs": tc.epochs,
            "best_epoch": best_epoch,
            "best_metric": "val_dice",
            "best_value": best_dice if rows else None,
            "resume_command": resume_command,
            "resumes": resumes,
        }

    clear_stop(ckpt_dir)
    write_status(ckpt_dir, status_fields("running"))
    log(
        f"{'Resuming' if resume else 'Run'} {run_dir} "
        f"{f'after epoch {len(rows)} ' if resume else ''}| device {gpu or device} | "
        f"amp {use_amp} | {n_params:,} parameters | train {len(train):,} / val {len(y_val):,} "
        f"images at {size} px | checkpoints in {ckpt_dir}"
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
                "best_dice": best_dice,
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
        loss_sum, n_seen = torch.zeros((), device=device), 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
            loss = seg_loss(logits, yb, tc.dice_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach() * yb.shape[0]
            n_seen += yb.shape[0]

        prob = predict_probs(model, x_val, tc.eval_batch_size, device, use_amp)
        dice = dice_score(prob >= THRESHOLD, y_val)
        row = {
            "epoch": epoch,
            "train_loss": float(loss_sum) / max(n_seen, 1),
            "val_dice": float(dice.mean()),
            "val_dice_median": float(np.median(dice)),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        train_seconds += row["seconds"]
        improved = row["val_dice"] > best_dice
        if improved:
            best_dice, best_epoch, stale = row["val_dice"], epoch, 0
            torch.save(
                {
                    "model_config": config.model.model_dump(mode="json"),
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": best_dice,
                },
                run_dir / CHECKPOINT,
            )
        else:
            stale += 1
        log(
            f"epoch {epoch:3d}  train loss {row['train_loss']:.4f}  val Dice {row['val_dice']:.4f} "
            f"(median {row['val_dice_median']:.4f})  {row['seconds']:.1f}s"
            f"{'  *' if improved else ''}"
        )
        with (run_dir / METRICS).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if stale >= tc.patience:
            log(f"Early stop: no improvement for {tc.patience} epochs")
            break
        if stop_requested(ckpt_dir):
            save_checkpoint("stopped")
            clear_stop(ckpt_dir)
            log(f"Stopped after epoch {epoch}. Resume with: {resume_command}")
            return run_dir
        save_checkpoint("running")

    save_checkpoint("finished")
    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "epochs_run": len(rows),
        "train_seconds": train_seconds,
        "resumes": resumes,
        "n_parameters": n_params,
        "n_train": len(train),
        "n_val": len(y_val),
        "image_size": size,
        "device": str(device),
        "gpu": gpu,
        "amp": use_amp,
        "torch": torch.__version__,
        "smp": smp.__version__,
    }
    (run_dir / SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    log(f"Best val Dice {best_dice:.4f} at epoch {best_epoch}")
    return run_dir
