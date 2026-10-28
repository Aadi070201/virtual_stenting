import csv

import pytest
import torch
from typer.testing import CliRunner

from vstent import checkpoints as ck
from vstent.cli import app
from vstent.config import save_run_metadata
from vstent.surrogate.model import ModelConfig
from vstent.surrogate.train import (
    SurrogateConfig,
    TrainConfig,
    load_checkpoint,
    resume_surrogate,
    train_surrogate,
)

SMALL_MODEL = ModelConfig(base_channels=8, levels=3, groups=4, cond_dim=16)


def _config(tiny_data, tmp_path, epochs=3):
    return SurrogateConfig(
        data_dir=str(tiny_data),
        runs_dir=str(tmp_path / "runs"),
        model=SMALL_MODEL,
        train=TrainConfig(epochs=epochs, batch_size=32, device="cpu", patience=50),
    )


def _metrics(run_dir):
    with (run_dir / "metrics.csv").open(encoding="utf-8") as f:
        return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(f)]


def _stop_after(ckpt_dir, epoch):
    def log(msg):
        if msg.startswith(f"epoch {epoch:3d}"):
            ck.request_stop(ckpt_dir)

    return log


def test_stop_and_resume_matches_uninterrupted_run(tiny_data, tmp_path):
    cfg = _config(tiny_data, tmp_path)
    roots = tmp_path / "ckpt"
    straight = train_surrogate(cfg, run_dir=tmp_path / "runs" / "straight", checkpoint_root=roots)

    run = tmp_path / "runs" / "split"
    stopped = train_surrogate(
        cfg, run_dir=run, checkpoint_root=roots, log=_stop_after(roots / "split", 1)
    )
    status = ck.read_status(roots / "split")
    assert status["status"] == "stopped" and status["epoch"] == 1
    assert status["resume_command"] == "vstent train-surrogate --resume split"
    assert not (roots / "split" / ck.STOP_FILE).exists()
    assert not (stopped / "summary.json").exists()

    resume_surrogate("split", checkpoint_root=roots)
    status = ck.read_status(roots / "split")
    assert status["status"] == "finished" and status["epoch"] == 3
    assert len(status["resumes"]) == 1 and status["resumes"][0]["after_epoch"] == 1

    assert _metrics(run) == _metrics(straight)
    a, b = load_checkpoint(straight), load_checkpoint(run)
    for (name, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(pa, pb), name


def test_resume_finished_run_is_a_no_op(tiny_data, tmp_path):
    cfg = _config(tiny_data, tmp_path, epochs=1)
    roots = tmp_path / "ckpt"
    run = train_surrogate(cfg, run_dir=tmp_path / "runs" / "done", checkpoint_root=roots)
    before = (run / "metrics.csv").read_text(encoding="utf-8")
    messages = []
    resume_surrogate(run, checkpoint_root=roots, log=messages.append)  # runs/<name> path works
    assert any("already finished" in m for m in messages)
    assert (run / "metrics.csv").read_text(encoding="utf-8") == before


def test_resume_run_killed_before_first_checkpoint(tiny_data, tmp_path):
    cfg = _config(tiny_data, tmp_path, epochs=1)
    roots = tmp_path / "ckpt"
    run = tmp_path / "runs" / "early"
    save_run_metadata(run, cfg)  # what exists if the process dies during epoch 1
    ck.write_status(roots / "early", {"kind": "surrogate", "run_dir": run.as_posix()})
    resume_surrogate("early", checkpoint_root=roots)
    assert ck.read_status(roots / "early")["status"] == "finished"
    assert (run / "best.pt").exists()


def test_resolve_and_list(tmp_path):
    root = tmp_path / "ckpt"
    with pytest.raises(FileNotFoundError):
        ck.resolve_checkpoint("missing", root)
    ck.save_state(root / "a", {"x": torch.ones(2)}, {"kind": "surrogate", "status": "running"})
    assert ck.resolve_checkpoint("a", root) == root / "a"
    assert ck.resolve_checkpoint(tmp_path / "runs" / "a", root) == root / "a"
    assert not (root / "a" / (ck.STATE_FILE + ".tmp")).exists()
    assert torch.equal(ck.load_state(root / "a")["x"], torch.ones(2))
    assert [r["name"] for r in ck.list_checkpoints(root)] == ["a"]
    assert ck.list_checkpoints(tmp_path / "nothing") == []


def test_resume_rejects_other_kind(tmp_path):
    root = tmp_path / "ckpt"
    ck.write_status(root / "seg", {"kind": "segmentation", "run_dir": "runs/seg"})
    with pytest.raises(ValueError, match="segmentation"):
        resume_surrogate("seg", checkpoint_root=root)


def test_cli_stop_and_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert "No checkpoints" in runner.invoke(app, ["checkpoints"]).output
    ck.write_status(
        ck.CHECKPOINTS_DIR / "r1",
        {
            "kind": "surrogate",
            "status": "stopped",
            "epoch": 4,
            "epochs": 50,
            "best_epoch": 3,
            "best_val_distal_mae": 0.01,
            "resume_command": "vstent train-surrogate --resume r1",
        },
    )
    out = runner.invoke(app, ["checkpoints"]).output
    assert "r1" in out and "epoch 4/50" in out and "--resume r1" in out
    assert runner.invoke(app, ["stop", "r1"]).exit_code == 0
    assert (ck.CHECKPOINTS_DIR / "r1" / ck.STOP_FILE).exists()
    assert runner.invoke(app, ["stop", "nope"]).exit_code == 1


def test_train_surrogate_cli_needs_exactly_one_mode():
    result = CliRunner().invoke(app, ["train-surrogate"])
    assert result.exit_code == 2
