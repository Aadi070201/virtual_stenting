"""Learned lesion reading (P4i): network, 5-fold training, ONNX export. Needs torch.

As pre-registered in docs/PLAN.md (P4i): a small CNN reads a straightened lesion patch
(:func:`vstent.imaging.lesion_patch.lesion_patch`) and its scalars and predicts the lesion's %DS
and whether it is significant (≥ 50%). Trained on CADICA's development patients (graded boxes)
and ARCADE's stenosis train split (significance only), in 5 folds of the development patients;
every development prediction is out-of-fold, and the product model is the mean of the 5 fold
models. Stoppable and resumable through :mod:`vstent.checkpoints` (``vstent stop <run>``,
``vstent train-lesion --resume <run>``).
"""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field
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
from vstent.data import lesion_data
from vstent.imaging.lesion_patch import N_SCALARS, PATCH_ACROSS, PATCH_ALONG

KIND = "lesion"
CONFIG_FILENAME = "config.yaml"
OOF_FILE = "oof.json"
SUMMARY = "summary.json"


class LesionTrainConfig(StrictModel):
    """Pre-registered training of the lesion reader (docs/PLAN.md, P4i)."""

    name: str = "lesion_cnn"
    seed: int = 0
    folds: int = Field(5, ge=2)
    epochs: int = Field(40, ge=1)
    batch_size: int = Field(64, ge=1)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    huber_delta: float = Field(0.1, gt=0)  # on %DS ÷ 100
    bce_weight: float = Field(0.5, ge=0)
    dropout: float = Field(0.2, ge=0, lt=1)
    ds_loss: Literal["midpoint", "interval"] = "midpoint"  # P4k-A: "interval"
    free_negatives: bool = False  # P4k-A: CADICA lesion-free candidates in training
    data_dir: str = str(lesion_data.OUT_DIR)
    runs_dir: str = "runs"
    device: str = "cpu"


class LesionNet(nn.Module):
    """Three conv blocks (16, 32, 64), global average pooling, the normalised scalars, FC 64;
    heads: %DS (sigmoid × 100) and a significance logit."""

    def __init__(self, scalar_mean, scalar_std, dropout: float = 0.2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.MaxPool2d(2))  # fmt: skip

        self.features = nn.Sequential(block(2, 16), block(16, 32), block(32, 64))
        self.register_buffer("s_mean", torch.as_tensor(scalar_mean, dtype=torch.float32))
        self.register_buffer("s_std", torch.as_tensor(scalar_std, dtype=torch.float32))
        self.head = nn.Sequential(nn.Linear(64 + N_SCALARS, 64), nn.ReLU(), nn.Dropout(dropout))
        self.ds = nn.Linear(64, 1)
        self.sig = nn.Linear(64, 1)

    def forward(self, patch: torch.Tensor, scalars: torch.Tensor):
        h = self.features(patch).mean(dim=(2, 3))
        z = self.head(torch.cat([h, (scalars - self.s_mean) / self.s_std], dim=1))
        return torch.sigmoid(self.ds(z)).squeeze(1) * 100.0, self.sig(z).squeeze(1)


class LesionEnsemble(nn.Module):
    """The mean of the fold models' %DS and significance logits (the product model)."""

    def __init__(self, members: list[LesionNet]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, patch: torch.Tensor, scalars: torch.Tensor):
        outs = [m(patch, scalars) for m in self.members]
        ds = torch.stack([o[0] for o in outs]).mean(dim=0)
        sig = torch.stack([o[1] for o in outs]).mean(dim=0)
        return ds, sig


def training_data(data_dir: str | Path, free: bool = False) -> dict[str, Any]:
    """CADICA development samples (graded), ARCADE stenosis train samples (significance) and,
    with ``free``, the development patients' lesion-free candidates (P4k-A)."""
    names = ["cadica", "arcade_train"] + (["cadica_free"] if free else [])
    loaded = {n: lesion_data.load(n, data_dir) for n in names}
    for n, (p, s, _) in loaded.items():
        if not (np.isfinite(p).all() and np.isfinite(s).all()):
            raise ValueError(f"{n}: non-finite values in the lesion samples")

    def dev_only(n):
        p, s, m = loaded[n]
        dev = np.array([x["split"] == "dev" for x in m], bool)
        return p[dev], s[dev], [x for x, d in zip(m, dev, strict=True) if d]

    out = {"cadica": dev_only("cadica"), "arcade": loaded["arcade_train"]}
    if free:
        out["free"] = dev_only("cadica_free")
    return out


def fold_of(patients: list[str], folds: int, seed: int) -> dict[str, int]:
    """Development patient → fold (fixed with ``seed``)."""
    order = sorted(set(patients), key=lambda p: int(p[1:]))
    perm = np.random.default_rng(seed).permutation(len(order))
    return {order[i]: k % folds for k, i in enumerate(perm)}


def _augment(patch: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Mirror across, reverse along, contrast gain × U(0.8, 1.2), noise σ 0.02, shift ± 2."""
    n = patch.shape[0]
    flip_x = torch.rand(n, generator=gen) < 0.5
    patch = torch.where(flip_x[:, None, None, None], patch.flip(3), patch)
    flip_a = torch.rand(n, generator=gen) < 0.5
    patch = torch.where(flip_a[:, None, None, None], patch.flip(2), patch)
    gain = 0.8 + 0.4 * torch.rand(n, generator=gen)
    noise = 0.02 * torch.randn(patch[:, :1].shape, generator=gen)
    contrast = patch[:, :1] * gain[:, None, None, None] + noise
    patch = torch.cat([contrast, patch[:, 1:]], dim=1)
    shift = int(torch.randint(-2, 3, (1,), generator=gen))
    return torch.roll(patch, shifts=shift, dims=2)


def _loss(ds_pred, logit, ds_target, sig_target, cfg: LesionTrainConfig,
          lo=None, hi=None) -> torch.Tensor:  # fmt: skip
    """BCE on the significance logit, plus a Huber loss on %DS ÷ 100: against the midpoint
    target, or (``ds_loss="interval"``, P4k-A) on the distance outside [``lo``, ``hi``]."""
    loss = cfg.bce_weight * nn.functional.binary_cross_entropy_with_logits(logit, sig_target)
    if cfg.ds_loss == "interval":
        outside = (torch.relu(lo - ds_pred) + torch.relu(ds_pred - hi)) / 100.0
        return loss + nn.functional.huber_loss(outside, torch.zeros_like(outside),
                                               delta=cfg.huber_delta)  # fmt: skip
    graded = torch.isfinite(ds_target)
    if graded.any():
        loss = loss + nn.functional.huber_loss(ds_pred[graded] / 100.0,
                                               ds_target[graded] / 100.0,
                                               delta=cfg.huber_delta)  # fmt: skip
    return loss


def _predict(model: nn.Module, patches: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(patches), 512):
            ds, _ = model(torch.as_tensor(patches[i : i + 512]),
                          torch.as_tensor(scalars[i : i + 512]))  # fmt: skip
            out.append(ds.numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_lesion(config: LesionTrainConfig, *, checkpoint_root: str | Path = CHECKPOINTS_DIR,
                 log: Callable[[str], None] | None = None) -> Path:  # fmt: skip
    """Start a new run; returns its directory (stoppable and resumable)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.runs_dir) / f"{config.name}-{stamp}"
    save_run_metadata(run_dir, config)
    return _fit(config, run_dir, checkpoint_dir(run_dir.name, checkpoint_root),
                log or (lambda _m: None), resume=False)  # fmt: skip


def resume_lesion(name_or_path: str | Path, *, checkpoint_root: str | Path = CHECKPOINTS_DIR,
                  log: Callable[[str], None] | None = None) -> Path:  # fmt: skip
    """Continue a stopped or interrupted run from its last completed epoch."""
    ckpt_dir = resolve_checkpoint(name_or_path, checkpoint_root)
    status = read_status(ckpt_dir)
    if status.get("kind") != KIND:
        raise ValueError(f"{ckpt_dir} is a {status.get('kind')!r} checkpoint, not {KIND!r}")
    run_dir = Path(status["run_dir"])
    config = load_config(run_dir / CONFIG_FILENAME, LesionTrainConfig)
    return _fit(config, run_dir, ckpt_dir, log or (lambda _m: None), resume=True)


def _fit(cfg: LesionTrainConfig, run_dir: Path, ckpt_dir: Path, log, *, resume: bool) -> Path:
    torch.manual_seed(cfg.seed)
    data = training_data(cfg.data_dir, cfg.free_negatives)
    pc, sc, mc = data["cadica"]
    pa, sa, ma = data["arcade"]
    pf, sf, mf = data.get("free", (pc[:0], sc[:0], []))
    fold = fold_of([m["patient"] for m in mc], cfg.folds, cfg.seed)
    fold_c = np.array([fold[m["patient"]] for m in mc])
    fold_f = np.array([fold.get(m["patient"], -1) for m in mf], int)  # -1: never held out
    ds_c = np.array([m["ds_target"] for m in mc], np.float32)
    sig_c = np.array([m["sig_target"] for m in mc], np.float32)
    sig_a = np.array([m["sig_target"] for m in ma], np.float32)
    sig_f = np.zeros(len(mf), np.float32)

    def bounds(ms):
        iv = np.array([lesion_data.ds_interval(m) for m in ms], np.float32).reshape(-1, 2)
        return iv[:, 0], iv[:, 1]

    (lo_c, hi_c), (lo_a, hi_a), (lo_f, hi_f) = bounds(mc), bounds(ma), bounds(mf)
    all_s = np.concatenate([sc, sa, sf])
    s_mean, s_std = all_s.mean(axis=0), all_s.std(axis=0) + 1e-6
    gen = torch.Generator().manual_seed(cfg.seed)
    state = {"fold": 0, "epoch": 0, "history": [], "oof": [None] * len(mc), "resumes": [],
             "train_seconds": 0.0}  # fmt: skip
    model = opt = sched = None

    def new_fold():
        m = LesionNet(s_mean, s_std, cfg.dropout)
        o = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n_train = (int((fold_c != state["fold"]).sum()) + len(pa)
                   + int((fold_f != state["fold"]).sum()))  # fmt: skip
        steps = cfg.epochs * -(-n_train // cfg.batch_size)
        return m, o, torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps)

    if resume and (ckpt_dir / STATE_FILE).exists():
        saved = load_state(ckpt_dir)
        state = saved["state"]
        restore_rng_states(saved["rng"], gen)
        if state["fold"] < cfg.folds:
            model, opt, sched = new_fold()
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            sched.load_state_dict(saved["scheduler"])
        commit = git_info()["git_commit"]
        state["resumes"].append({"fold": state["fold"], "epoch": state["epoch"],
                                 "git_commit": commit, "utc": utc_now()})  # fmt: skip
    resume_command = f"vstent train-lesion --resume {run_dir.name}"

    def status(s: str) -> dict[str, Any]:
        return {"kind": KIND, "run_dir": run_dir.as_posix(), "status": s,
                "epoch": state["fold"] * cfg.epochs + state["epoch"],
                "epochs": cfg.folds * cfg.epochs, "best_epoch": None, "best_metric": None,
                "best_value": None, "resume_command": resume_command,
                "resumes": state["resumes"]}  # fmt: skip

    def checkpoint(s: str) -> None:
        save_state(ckpt_dir, {"state": state, "rng": rng_states(gen),
                              "model": None if model is None else model.state_dict(),
                              "optimizer": None if opt is None else opt.state_dict(),
                              "scheduler": None if sched is None else sched.state_dict()},
                   status(s))  # fmt: skip

    clear_stop(ckpt_dir)
    write_status(ckpt_dir, status("running"))
    log(f"{'Resuming' if resume else 'Run'} {run_dir} | CADICA dev {len(mc)} + ARCADE "
        f"{len(ma)} samples | {cfg.folds} folds × {cfg.epochs} epochs")  # fmt: skip
    while state["fold"] < cfg.folds:
        k = state["fold"]
        if model is None or state["epoch"] == 0:
            model, opt, sched = new_fold()
        train_c = np.flatnonzero(fold_c != k)
        train_f = np.flatnonzero(fold_f != k)
        patches = torch.as_tensor(np.concatenate([pc[train_c], pa, pf[train_f]]))
        scalars = torch.as_tensor(np.concatenate([sc[train_c], sa, sf[train_f]]))
        no_ds = np.full(len(pa) + len(train_f), np.nan, np.float32)
        ds_t = torch.as_tensor(np.concatenate([ds_c[train_c], no_ds]))
        sig_t = torch.as_tensor(np.concatenate([sig_c[train_c], sig_a, sig_f[train_f]]))
        lo_t = torch.as_tensor(np.concatenate([lo_c[train_c], lo_a, lo_f[train_f]]))
        hi_t = torch.as_tensor(np.concatenate([hi_c[train_c], hi_a, hi_f[train_f]]))
        for epoch in range(state["epoch"] + 1, cfg.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            order = torch.randperm(len(patches), generator=gen)
            total = 0.0
            for i in range(0, len(order), cfg.batch_size):
                idx = order[i : i + cfg.batch_size]
                ds_pred, logit = model(_augment(patches[idx], gen), scalars[idx])
                loss = _loss(ds_pred, logit, ds_t[idx], sig_t[idx], cfg, lo_t[idx], hi_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(idx)
            seconds = time.perf_counter() - t0
            state["epoch"] = epoch
            state["train_seconds"] += seconds
            state["history"].append({"fold": k, "epoch": epoch, "loss": total / len(order),
                                     "seconds": seconds})  # fmt: skip
            if epoch % 10 == 0 or epoch == cfg.epochs:
                log(f"fold {k} epoch {epoch:3d}  loss {total / len(order):.4f}  {seconds:.1f}s")
            if stop_requested(ckpt_dir) and epoch < cfg.epochs:
                checkpoint("stopped")
                clear_stop(ckpt_dir)
                log(f"Stopped at fold {k}, epoch {epoch}. Resume with: {resume_command}")
                return run_dir
            checkpoint("running")
        held = np.flatnonzero(fold_c == k)
        for i, v in zip(held, _predict(model, pc[held], sc[held]), strict=True):
            state["oof"][int(i)] = float(v)
        torch.save({"state_dict": model.state_dict(), "fold": k, "s_mean": s_mean,
                    "s_std": s_std, "dropout": cfg.dropout}, run_dir / f"fold_{k}.pt")  # fmt: skip
        state["fold"], state["epoch"] = k + 1, 0
        model = opt = sched = None
        checkpoint("running")
    (run_dir / OOF_FILE).write_text(json.dumps({"samples": [
        {**m, "oof_ds": v, "fold": int(f)} for m, v, f in zip(mc, state["oof"], fold_c,
                                                               strict=True)]}),
        encoding="utf-8")  # fmt: skip
    (run_dir / SUMMARY).write_text(json.dumps({"folds": cfg.folds, "epochs": cfg.epochs,
                                               "n_cadica_dev": len(mc), "n_arcade": len(ma),
                                               "train_seconds": state["train_seconds"],
                                               "resumes": state["resumes"],
                                               "torch": torch.__version__}, indent=2),
                                   encoding="utf-8")  # fmt: skip
    checkpoint("finished")
    log(f"Finished: {cfg.folds} fold models and out-of-fold predictions in {run_dir}")
    return run_dir


def load_ensemble(run_dir: str | Path) -> LesionEnsemble:
    """The fold models of a finished run, as one averaging model."""
    members = []
    for path in sorted(Path(run_dir).glob("fold_*.pt")):
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = LesionNet(d["s_mean"], d["s_std"], d["dropout"])
        m.load_state_dict(d["state_dict"])
        members.append(m.eval())
    if not members:
        raise ValueError(f"{run_dir}: no fold models")
    return LesionEnsemble(members).eval()


def export_lesion(run_dir: str | Path, out_root: str | Path = "models/lesion",
                  log: Callable[[str], None] | None = None) -> dict[str, Any]:  # fmt: skip
    """``lesion.onnx`` (the fold models' mean) with a parity check against torch."""
    import hashlib

    import onnxruntime as ort

    from vstent.inference.lesion_runtime import ONNX_FILE

    log = log or (lambda _m: None)
    run_dir = Path(run_dir)
    model = load_ensemble(run_dir)
    out = Path(out_root) / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    patch = torch.zeros(2, 2, PATCH_ALONG, PATCH_ACROSS)
    scal = torch.zeros(2, N_SCALARS)
    torch.onnx.export(model, (patch, scal), str(out / ONNX_FILE), input_names=["patch", "scalars"],
                      output_names=["ds", "sig_logit"], opset_version=17, dynamo=False,
                      dynamic_axes={"patch": {0: "n"}, "scalars": {0: "n"}, "ds": {0: "n"},
                                    "sig_logit": {0: "n"}})  # fmt: skip
    cfg = load_config(run_dir / CONFIG_FILENAME, LesionTrainConfig)
    pc, sc, _ = lesion_data.load("cadica", Path(cfg.data_dir))
    x, s = pc[:256], sc[:256]
    with torch.no_grad():
        ref = model(torch.as_tensor(x), torch.as_tensor(s))[0].numpy()
    sess = ort.InferenceSession(str(out / ONNX_FILE), providers=["CPUExecutionProvider"])
    got = sess.run(["ds"], {"patch": x, "scalars": s})[0]
    diff = float(np.max(np.abs(got - ref)))
    meta = {"run": run_dir.name, "members": len(model.members), "parity_max_abs_ds": diff,
            "parity_passed": diff < 1e-3, "onnx_sha256":
            hashlib.sha256((out / ONNX_FILE).read_bytes()).hexdigest(), **git_info(),
            "created_utc": utc_now()}  # fmt: skip
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Exported {out / ONNX_FILE}: parity max |Δ%DS| {diff:.2e}")
    return meta
