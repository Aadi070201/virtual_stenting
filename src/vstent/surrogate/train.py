"""Training of the FFR surrogate (``vstent train-surrogate``).

A run directory ``runs/<name>-<UTC timestamp>/`` receives: ``config.yaml`` + ``run_meta.yaml``
(git commit), ``norm_stats.json``, ``metrics.csv`` (one row per epoch), ``best.pt`` (lowest
validation distal-FFR MAE) and ``summary.json``.
"""

import csv
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import Field

from vstent.config import StrictModel, save_run_metadata
from vstent.surrogate.dataset import SurrogateDataset, load_raw
from vstent.surrogate.features import NormStats
from vstent.surrogate.model import FFRUNet, ModelConfig, count_parameters

CHECKPOINT = "best.pt"
NORM_STATS = "norm_stats.json"
METRICS = "metrics.csv"
SUMMARY = "summary.json"


class TrainConfig(StrictModel):
    """Optimisation settings."""

    epochs: int = Field(50, gt=0)
    batch_size: int = Field(256, gt=0)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    distal_weight: float = Field(2.0, ge=0)
    patience: int = Field(10, gt=0)
    grad_clip: float = Field(1.0, gt=0)
    amp: bool = True
    device: str = "auto"
    eval_batch_size: int = Field(4096, gt=0)
    max_train_samples: int | None = Field(None, gt=0)
    max_val_samples: int | None = Field(None, gt=0)


class SurrogateConfig(StrictModel):
    """Full surrogate-training config (``configs/surrogate.yaml``)."""

    name: str = "unet1d"
    data_dir: str = "data/synth/v1"
    runs_dir: str = "runs"
    seed: int = 0
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def resolve_device(spec: str = "auto") -> torch.device:
    """``"auto"`` -> CUDA if available, else CPU."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def surrogate_loss(pred: torch.Tensor, target: torch.Tensor, distal_weight: float) -> torch.Tensor:
    """MSE over the whole FFR curve + ``distal_weight`` × MSE on distal FFR."""
    return F.mse_loss(pred, target) + distal_weight * F.mse_loss(pred[:, -1], target[:, -1])


@torch.no_grad()
def predict_batches(
    model: FFRUNet, x_point: torch.Tensor, x_global: torch.Tensor, batch_size: int = 4096
) -> torch.Tensor:
    """FFR predictions (n, N) in eval mode, float32, on the inputs' device."""
    model.eval()
    return torch.cat(
        [
            model(x_point[i : i + batch_size], x_global[i : i + batch_size])
            for i in range(0, x_point.shape[0], batch_size)
        ]
    )


def load_checkpoint(run_dir: str | Path, device: torch.device | str = "cpu") -> FFRUNet:
    """Rebuild the best model of a run, in eval mode."""
    ckpt = torch.load(Path(run_dir) / CHECKPOINT, map_location=device, weights_only=True)
    model = FFRUNet(ModelConfig.model_validate(ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def train_surrogate(
    config: SurrogateConfig,
    *,
    run_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Train the surrogate and return the run directory."""
    log = log or (lambda _msg: None)
    tc = config.train
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir or Path(config.runs_dir) / f"{config.name}-{stamp}")
    save_run_metadata(run_dir, config)

    torch.manual_seed(config.seed)
    device = resolve_device(tc.device)
    use_amp = tc.amp and device.type == "cuda"

    raw_train, ffr_train = load_raw(config.data_dir, "train", tc.max_train_samples)
    stats = NormStats.fit(raw_train)
    stats.save(run_dir / NORM_STATS)
    train = SurrogateDataset(raw_train, ffr_train, stats).to(device)
    val = SurrogateDataset.from_split(config.data_dir, "val", stats, tc.max_val_samples).to(device)

    model = FFRUNet(config.model).to(device)
    n_params = count_parameters(model)
    steps_per_epoch = -(-len(train) // tc.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.epochs * steps_per_epoch
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    log(
        f"Run {run_dir} | device {gpu or device} | amp {use_amp} | {n_params:,} parameters | "
        f"train {len(train):,} / val {len(val):,}"
    )

    best_mae, best_epoch, stale = float("inf"), -1, 0
    rows = []
    t_start = time.perf_counter()
    for epoch in range(1, tc.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        for xp, xg, y in train.batches(tc.batch_size, shuffle=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(xp, xg)
            loss = surrogate_loss(pred, y, tc.distal_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach() * y.shape[0]

        pred = predict_batches(model, val.x_point, val.x_global, tc.eval_batch_size)
        row = {
            "epoch": epoch,
            "train_loss": float(loss_sum) / len(train),
            "val_loss": float(surrogate_loss(pred, val.ffr, tc.distal_weight)),
            "val_distal_mae": float((pred[:, -1] - val.ffr[:, -1]).abs().mean()),
            "val_curve_mae": float((pred - val.ffr).abs().mean()),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        improved = row["val_distal_mae"] < best_mae
        if improved:
            best_mae, best_epoch, stale = row["val_distal_mae"], epoch, 0
            torch.save(
                {
                    "model_config": config.model.model_dump(mode="json"),
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_distal_mae": best_mae,
                },
                run_dir / CHECKPOINT,
            )
        else:
            stale += 1
        log(
            f"epoch {epoch:3d}  train {row['train_loss']:.2e}  val {row['val_loss']:.2e}  "
            f"distal MAE {row['val_distal_mae']:.4f}  curve MAE {row['val_curve_mae']:.4f}  "
            f"{row['seconds']:.1f}s{'  *' if improved else ''}"
        )
        with (run_dir / METRICS).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if stale >= tc.patience:
            log(f"Early stop: no improvement for {tc.patience} epochs")
            break

    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_distal_mae": best_mae,
        "epochs_run": len(rows),
        "train_seconds": time.perf_counter() - t_start,
        "n_parameters": n_params,
        "n_train": len(train),
        "n_val": len(val),
        "device": str(device),
        "gpu": gpu,
        "amp": use_amp,
        "torch": torch.__version__,
    }
    (run_dir / SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    log(f"Best val distal MAE {best_mae:.4f} at epoch {best_epoch}")
    return run_dir
