import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from vstent import checkpoints as ck
from vstent.cli import app
from vstent.config import dump_config
from vstent.surrogate.ensemble import (
    ENSEMBLE_FILE,
    EnsembleConfig,
    is_ensemble_run,
    member_dirs,
    resume_ensemble,
    stop_ensemble,
    surrogate_config,
    train_ensemble,
)
from vstent.surrogate.model import ModelConfig
from vstent.surrogate.train import SurrogateConfig, TrainConfig

SMALL_MODEL = ModelConfig(base_channels=8, levels=3, groups=4, cond_dim=16)


def _ensemble_config(tiny_data, tmp_path, seeds=(0, 1), epochs=1):
    base = SurrogateConfig(
        data_dir=str(tiny_data),
        runs_dir=str(tmp_path / "runs"),
        model=SMALL_MODEL,
        train=TrainConfig(epochs=epochs, batch_size=32, device="cpu", patience=50),
    )
    dump_config(base, tmp_path / "base.yaml")
    return EnsembleConfig(
        name="ens", runs_dir=str(tmp_path / "runs"), base_config=str(tmp_path / "base.yaml"),
        seeds=list(seeds),
    )  # fmt: skip


def test_config_needs_distinct_seeds():
    with pytest.raises(ValueError, match="distinct"):
        EnsembleConfig(seeds=[1, 1])
    with pytest.raises(ValueError, match="distinct"):
        EnsembleConfig(seeds=[3])


def test_train_ensemble_layout(tiny_data, tmp_path):
    roots = tmp_path / "ckpt"
    run = train_ensemble(
        _ensemble_config(tiny_data, tmp_path), run_dir=tmp_path / "runs" / "e1",
        checkpoint_root=roots,
    )  # fmt: skip
    assert is_ensemble_run(run) and not is_ensemble_run(member_dirs(run)[0])
    members = member_dirs(run)
    assert [m.name for m in members] == ["e1-s0", "e1-s1"]
    for seed, m in zip((0, 1), members, strict=True):
        assert (m / "best.pt").exists()
        assert yaml.safe_load((m / "config.yaml").read_text(encoding="utf-8"))["seed"] == seed
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert [s["seed"] for s in summary["members"]] == [0, 1]
    status = ck.read_status(roots / "e1")
    assert status["kind"] == "surrogate_ensemble" and status["status"] == "finished"
    assert status["epoch"] == status["epochs"] == 2
    listed = json.loads((run / ENSEMBLE_FILE).read_text(encoding="utf-8"))["members"]
    assert [m["status"] for m in listed] == ["finished", "finished"]


def test_stop_mid_member_and_resume(tiny_data, tmp_path):
    roots = tmp_path / "ckpt"
    seen = []

    def log(msg):  # stop while the second member is in its first epoch
        seen.append(msg)
        if msg.startswith("epoch   1") and sum(m.startswith("Member seed") for m in seen) == 2:
            assert stop_ensemble(roots / "e2") == "e2-s1"

    cfg = _ensemble_config(tiny_data, tmp_path, seeds=(0, 1, 2), epochs=2)
    run = train_ensemble(cfg, run_dir=tmp_path / "runs" / "e2", checkpoint_root=roots, log=log)
    status = ck.read_status(roots / "e2")
    assert status["status"] == "stopped" and status["epoch"] == 1
    assert status["resume_command"] == "vstent train-ensemble --resume e2"
    assert ck.read_status(roots / "e2-s1")["epoch"] == 1
    assert not (roots / "e2-s2").exists()
    assert not (run / "summary.json").exists()

    messages = []
    resume_ensemble("e2", checkpoint_root=roots, log=messages.append)
    assert any("seed 0: finished earlier" in m for m in messages)
    assert ck.read_status(roots / "e2")["status"] == "finished"
    assert ck.read_status(roots / "e2-s1")["resumes"][0]["after_epoch"] == 1
    assert all((m / "summary.json").exists() for m in member_dirs(run))


def test_resume_rejects_other_kind(tmp_path):
    ck.write_status(tmp_path / "ckpt" / "x", {"kind": "surrogate", "run_dir": "runs/x"})
    with pytest.raises(ValueError, match="surrogate"):
        resume_ensemble("x", checkpoint_root=tmp_path / "ckpt")


def test_cli_lists_and_stops_ensemble(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    ck.write_status(
        ck.CHECKPOINTS_DIR / "ens-a",
        {
            "kind": "surrogate_ensemble",
            "status": "running",
            "epoch": 2,
            "epochs": 5,
            "current_member": "ens-a-s2",
            "resume_command": "vstent train-ensemble --resume ens-a",
        },
    )
    ck.write_status(
        ck.CHECKPOINTS_DIR / "ens-a-s2",
        {
            "kind": "surrogate",
            "status": "running",
            "epoch": 3,
            "epochs": 50,
            "resume_command": "vstent train-surrogate --resume ens-a-s2",
        },
    )
    assert "members 2/5" in runner.invoke(app, ["checkpoints"]).output
    result = runner.invoke(app, ["stop", "ens-a"])
    assert result.exit_code == 0 and "ens-a-s2" in result.output
    assert (ck.CHECKPOINTS_DIR / "ens-a" / ck.STOP_FILE).exists()
    assert (ck.CHECKPOINTS_DIR / "ens-a-s2" / ck.STOP_FILE).exists()
    assert runner.invoke(app, ["train-ensemble"]).exit_code == 2


def test_surrogate_config_of_model_dirs(tiny_data, tiny_export, tiny_ensemble_export):
    # an ensemble model's config.yaml is the EnsembleConfig; the member config has data_dir
    for export in (tiny_export, tiny_ensemble_export):
        assert Path(surrogate_config(export["out_dir"]).data_dir) == Path(tiny_data)


def test_planner_benchmark_runs_on_an_ensemble_model(tiny_ensemble_export, tmp_path):
    from vstent.config import load_config
    from vstent.physics.solver import PhysicsParams
    from vstent.planning.optimizer import PlannerConfig, StentCatalog
    from vstent.planning.report import run_planner_benchmark

    root = Path(__file__).resolve().parents[1] / "configs"
    report = run_planner_benchmark(
        tiny_ensemble_export["out_dir"],
        tmp_path,
        catalog=load_config(root / "stent_catalog.yaml", StentCatalog),
        config=load_config(root / "planner.yaml", PlannerConfig),
        params=load_config(root / "physics.yaml", PhysicsParams),
        n_vessels=2,
    )
    assert (tmp_path / "planner_benchmark.md").exists() and report["throughput"]
