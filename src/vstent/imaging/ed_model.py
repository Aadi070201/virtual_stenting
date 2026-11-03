"""Learned end-diastolic frames (P4l, M2): a per-frame MobileNetV3-Small with dilated temporal
convolutions, trained on synthetic beating hearts whose ECG is known (docs/PLAN.md, P4l).

Input: a run's frames through the shared pre-processing at 128 px, and its frame time (s), so
one model serves 7.5 to 30 fps. Output: a per-frame logit of "this is an ED frame". Training is
stoppable and resumable (``vstent.checkpoints``); the epoch with the best validation ±1-frame
accuracy is kept.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field
from scipy.signal import find_peaks
from torch import nn

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
from vstent.config import StrictModel, git_info, load_config, save_run_metadata
from vstent.imaging import ed_data
from vstent.imaging.segment import preprocess_image

KIND = "ed"
CONFIG_FILENAME = "config.yaml"
BEST = "best.pt"
SUMMARY = "summary.json"
PEAK_LEVEL = 0.5
PEAK_GAP = 0.6  # peaks at least this share of the median gap apart


class EdTrainConfig(StrictModel):
    """Pre-registered training of the ED model (docs/PLAN.md, P4l M2)."""

    name: str = "ed_mnv3"
    seed: int = 0
    epochs: int = Field(30, ge=1)
    batch_runs: int = Field(4, ge=1)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    pos_weight: float = Field(5.0, gt=0)
    features: int = Field(64, ge=1)
    pretrained: bool = True
    # M2b (docs/PLAN.md, P4l); the defaults reproduce M2
    temporal: Literal["tcn3", "tcn5", "gru", "transformer"] = "tcn3"
    soft_sigma: float = Field(0.0, ge=0)  # frames; 0: one positive frame per beat
    augment: bool = False
    schedule: Literal["constant", "cosine"] = "constant"
    data_dir: str = str(ed_data.OUT_DIR)
    runs_dir: str = "runs"
    device: str = "auto"


DILATIONS = {"tcn3": (1, 2, 4), "tcn5": (1, 2, 4, 8, 16)}
PE_PERIODS_S = (0.05, 10.0)  # transformer: time encodings with periods from 50 ms to 10 s


def time_encoding(t_s: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encodings (…, dim) of times in seconds, periods log-spaced over
    :data:`PE_PERIODS_S`."""
    k = dim // 2
    periods = torch.logspace(np.log10(PE_PERIODS_S[0]), np.log10(PE_PERIODS_S[1]), k,
                             device=t_s.device)  # fmt: skip
    ang = 2.0 * np.pi * t_s[..., None] / periods
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class EdNet(nn.Module):
    """MobileNetV3-Small per frame (one input channel), 64 features + the frame time, then a
    temporal head and a per-frame logit. Heads: ``tcn3`` (M2: dilated convolutions, kernel 5,
    dilation 1, 2, 4), ``tcn5`` (dilation up to 16), ``gru`` (two-layer bidirectional GRU) or
    ``transformer`` (two encoder layers on time encodings in seconds)."""

    def __init__(self, features: int = 64, pretrained: bool = True, temporal: str = "tcn3") -> None:
        super().__init__()
        import torchvision

        m = torchvision.models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        conv = m.features[0][0]
        one = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                        bias=False)  # fmt: skip
        one.weight = nn.Parameter(conv.weight.detach().sum(dim=1, keepdim=True))
        m.features[0][0] = one
        self.encoder = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Linear(576, features)
        self.kind = temporal
        if temporal in DILATIONS:
            layers: list[nn.Module] = []
            c = features + 1
            for d in DILATIONS[temporal]:
                layers += [nn.Conv1d(c, features, 5, padding=2 * d, dilation=d), nn.ReLU()]
                c = features
            self.temporal = nn.Sequential(*layers, nn.Conv1d(features, 1, 1))
        elif temporal == "gru":
            self.temporal = nn.GRU(features + 1, features, num_layers=2, batch_first=True,
                                   bidirectional=True)  # fmt: skip
            self.head = nn.Linear(2 * features, 1)
        elif temporal == "transformer":
            self.inp = nn.Linear(features + 1, features)
            layer = nn.TransformerEncoderLayer(features, 4, 2 * features, dropout=0.1,
                                               batch_first=True)  # fmt: skip
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(features, 1)
        else:
            raise ValueError(f"unknown temporal head {temporal!r}")

    def forward(self, frames: torch.Tensor, frame_time: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
        """``frames`` (B, T, 1, H, W) standardised, ``frame_time`` (B,) s, ``lengths`` (B,)
        frames per run (padding after) → logits (B, T)."""
        b, t = frames.shape[:2]
        f = self.proj(self.encoder(frames.flatten(0, 1))).view(b, t, -1)
        x = torch.cat([f, frame_time.view(b, 1, 1).expand(b, t, 1)], dim=-1)
        if self.kind in DILATIONS:
            return self.temporal(x.transpose(1, 2))[:, 0]
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long)
        if self.kind == "gru":
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                       enforce_sorted=False)  # fmt: skip
            out, _ = self.temporal(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=t)
            return self.head(out)[..., 0]
        times = torch.arange(t, device=x.device)[None] * frame_time.view(b, 1)
        h = self.inp(x) + time_encoding(times, x.shape[-1] - 1)
        pad = torch.arange(t, device=x.device)[None] >= lengths.to(x.device)[:, None]
        return self.head(self.temporal(h, src_key_padding_mask=pad))[..., 0]


def run_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, 1, 128, 128) float32 through the shared pre-processing."""
    return torch.from_numpy(np.stack([preprocess_image(f, ed_data.SIZE) for f in frames]))


def targets(ed: tuple[int, ...], n: int, sigma: float) -> np.ndarray:
    """(n,) per-frame targets: 1 at each ED frame, spread as exp(−Δ²/(2σ²)) to its neighbours
    if ``sigma`` > 0 (the maximum over beats)."""
    y = np.zeros(n, np.float32)
    idx = np.arange(n)
    for e in ed:
        if 0 <= e < n:
            y = np.maximum(
                y, (idx == e) if sigma <= 0 else np.exp(-((idx - e) ** 2) / (2 * sigma**2))
            )
    return y.astype(np.float32)


def augment(frames: np.ndarray, ed: tuple[int, ...], gen: torch.Generator):
    """M2b's augmentation of one training run (uint8 frames): the same flips on every frame,
    brightness, contrast, gamma, noise, and a contiguous crop of 70–100% of the frames."""

    def u(lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand(1, generator=gen))

    n = len(frames)
    keep = max(1, int(round(n * u(0.7, 1.0))))
    first = int(torch.randint(0, n - keep + 1, (1,), generator=gen))
    x = frames[first : first + keep].astype(np.float32) / 255.0
    if u(0, 1) < 0.5:
        x = x[:, :, ::-1]
    if u(0, 1) < 0.5:
        x = x[:, ::-1, :]
    x = np.clip(x, 0, 1) ** u(0.8, 1.25)
    mean = x.mean()
    x = (x - mean) * u(0.8, 1.2) + mean * u(0.9, 1.1)
    noise = torch.randn(x.shape, generator=gen).numpy() * (u(0.0, 3.0) / 255.0)
    out = np.clip((x + noise) * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out), tuple(e - first for e in ed if first <= e < first + keep)


def decode(prob: np.ndarray) -> tuple[int, ...]:
    """ED frames: probability peaks ≥ 0.5, at least 0.6 × the median gap between peaks apart."""
    peaks, _ = find_peaks(np.r_[0.0, prob, 0.0], height=PEAK_LEVEL)
    peaks = peaks - 1
    if len(peaks) >= 3:
        gap = float(np.median(np.diff(peaks)))
        peaks, _ = find_peaks(np.r_[0.0, prob, 0.0], height=PEAK_LEVEL,
                              distance=max(1, int(PEAK_GAP * gap)))  # fmt: skip
        peaks = peaks - 1
    return tuple(int(p) for p in peaks)


def score(runs: list[dict], found: list[tuple[int, ...]]) -> dict[str, Any]:
    """True ED beats in the well-opacified frames found within ±1 frame (as P7), pooled and per
    frame rate."""
    by: dict[float, list[int]] = {}
    for r, f in zip(runs, found, strict=True):
        true = [e for e in r["ed"] if r["well"][e]]
        hit = sum(any(abs(e - g) <= 1 for g in f) for e in true)
        a = by.setdefault(r["fps"], [0, 0])
        a[0], a[1] = a[0] + hit, a[1] + len(true)
    hits, total = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    return {"rate": hits / total if total else None, "beats": total,
            "by_fps": {str(k): {"rate": v[0] / v[1] if v[1] else None, "beats": v[1]}
                       for k, v in sorted(by.items())}}  # fmt: skip


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@torch.no_grad()
def predict(model: nn.Module, runs: list[dict], device) -> list[np.ndarray]:
    """Per-frame ED probabilities of every run."""
    model.eval()
    out = []
    for r in runs:
        x = run_tensor(r["frames"])[None].to(device)
        ft = torch.tensor([1.0 / r["fps"]], device=device)
        out.append(torch.sigmoid(model(x, ft))[0].float().cpu().numpy())
    return out


def train_ed(config: EdTrainConfig, *, checkpoint_root: str | Path = CHECKPOINTS_DIR,
             log: Callable[[str], None] | None = None) -> Path:  # fmt: skip
    """Start a new run (stoppable, resumable); returns its directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.runs_dir) / f"{config.name}-{stamp}"
    save_run_metadata(run_dir, config)
    return _fit(config, run_dir, checkpoint_dir(run_dir.name, checkpoint_root),
                log or (lambda _m: None), resume=False)  # fmt: skip


def resume_ed(name_or_path: str | Path, *, checkpoint_root: str | Path = CHECKPOINTS_DIR,
              log: Callable[[str], None] | None = None) -> Path:  # fmt: skip
    """Continue a stopped or interrupted run from its last completed epoch."""
    ckpt = resolve_checkpoint(name_or_path, checkpoint_root)
    status = read_status(ckpt)
    if status.get("kind") != KIND:
        raise ValueError(f"{ckpt} is a {status.get('kind')!r} checkpoint, not {KIND!r}")
    run_dir = Path(status["run_dir"])
    cfg = load_config(run_dir / CONFIG_FILENAME, EdTrainConfig)
    return _fit(cfg, run_dir, ckpt, log or (lambda _m: None), resume=True)


def _fit(cfg: EdTrainConfig, run_dir: Path, ckpt: Path, log, *, resume: bool) -> Path:
    torch.manual_seed(cfg.seed)
    device = _device(cfg.device)
    train = ed_data.load_split("train", cfg.data_dir)
    val = ed_data.load_split("val", cfg.data_dir)
    if not train or not val:
        raise ValueError(f"no ED data in {cfg.data_dir} (run `vstent ed-data`)")
    model = EdNet(cfg.features, pretrained=cfg.pretrained and not resume,
                  temporal=cfg.temporal).to(device)  # fmt: skip
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
             if cfg.schedule == "cosine" else None)  # fmt: skip
    gen = torch.Generator().manual_seed(cfg.seed)
    state: dict[str, Any] = {"epoch": 0, "history": [], "best": -1.0, "best_epoch": None,
                             "resumes": [], "seconds": 0.0}  # fmt: skip
    if resume and (ckpt / STATE_FILE).exists():
        saved = load_state(ckpt)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        if sched is not None:
            sched.load_state_dict(saved["scheduler"])
        restore_rng_states(saved["rng"], gen)
        state = saved["state"]
        commit = git_info()["git_commit"]
        state["resumes"].append({"after_epoch": state["epoch"], "git_commit": commit,
                                 "utc": utc_now()})  # fmt: skip
    resume_command = f"vstent train-ed --resume {run_dir.name}"

    def status(s: str) -> dict[str, Any]:
        return {"kind": KIND, "run_dir": run_dir.as_posix(), "status": s,
                "epoch": state["epoch"], "epochs": cfg.epochs, "best_epoch": state["best_epoch"],
                "best_metric": "val_within1", "best_value": state["best"],
                "resume_command": resume_command, "resumes": state["resumes"]}  # fmt: skip

    def checkpoint(s: str) -> None:
        save_state(ckpt, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                          "scheduler": None if sched is None else sched.state_dict(),
                          "rng": rng_states(gen), "state": state}, status(s))  # fmt: skip

    clear_stop(ckpt)
    write_status(ckpt, status("running"))
    log(f"{'Resuming' if resume else 'Run'} {run_dir} | {device} | train {len(train)} runs, "
        f"val {len(val)} runs | {cfg.epochs} epochs")  # fmt: skip
    pos = torch.tensor(cfg.pos_weight, device=device)
    for epoch in range(state["epoch"] + 1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = torch.randperm(len(train), generator=gen).tolist()
        total = 0.0
        for i in range(0, len(order), cfg.batch_runs):
            batch = [train[j] for j in order[i : i + cfg.batch_runs]]
            items = [augment(r["frames"], r["ed"], gen) if cfg.augment else (r["frames"], r["ed"])
                     for r in batch]  # fmt: skip
            t_max = max(len(f) for f, _ in items)
            x = torch.zeros(len(batch), t_max, 1, ed_data.SIZE, ed_data.SIZE)
            y = torch.zeros(len(batch), t_max)
            mask = torch.zeros(len(batch), t_max)
            for b, (frames, ed) in enumerate(items):
                n = len(frames)
                x[b, :n] = run_tensor(frames)
                y[b, :n] = torch.from_numpy(targets(ed, n, cfg.soft_sigma))
                mask[b, :n] = 1.0
            ft = torch.tensor([1.0 / r["fps"] for r in batch])
            lengths = torch.tensor([len(f) for f, _ in items])
            logits = model(x.to(device), ft.to(device), lengths)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y.to(device), weight=mask.to(device), pos_weight=pos
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        if sched is not None:
            sched.step()
        probs = predict(model, val, device)
        sc = score(val, [decode(p) for p in probs])
        seconds = time.perf_counter() - t0
        state["epoch"], state["seconds"] = epoch, state["seconds"] + seconds
        state["history"].append({"epoch": epoch, "loss": total / len(train),
                                 "val_within1": sc["rate"], "seconds": seconds})  # fmt: skip
        better = (sc["rate"] or 0.0) > state["best"]
        if better:
            state["best"], state["best_epoch"] = sc["rate"] or 0.0, epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump(),
                        "epoch": epoch}, run_dir / BEST)  # fmt: skip
        log(f"epoch {epoch:3d}  loss {total / len(train):.4f}  val ±1 {sc['rate']:.3f}  "
            f"{seconds:.0f}s{'  *' if better else ''}")  # fmt: skip
        if stop_requested(ckpt):
            checkpoint("stopped")
            clear_stop(ckpt)
            log(f"Stopped after epoch {epoch}. Resume with: {resume_command}")
            return run_dir
        checkpoint("running")
    checkpoint("finished")
    (run_dir / SUMMARY).write_text(json.dumps({"best_epoch": state["best_epoch"],
                                               "best_val_within1": state["best"],
                                               "seconds": state["seconds"]}, indent=2),
                                   encoding="utf-8")  # fmt: skip
    log(f"Best validation ±1-frame accuracy {state['best']:.3f} at epoch {state['best_epoch']}")
    return run_dir


def load_best(run_dir: str | Path, device="cpu") -> EdNet:
    """The kept epoch of a run, in eval mode (no weight download)."""
    ck = torch.load(Path(run_dir) / BEST, map_location=device, weights_only=True)
    model = EdNet(ck["config"]["features"], pretrained=False,
                  temporal=ck["config"].get("temporal", "tcn3"))  # fmt: skip
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()
