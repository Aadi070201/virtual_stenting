import json
from pathlib import Path

from typer.testing import CliRunner

from vstent.cli import app
from vstent.imaging.qca_validation import write_qca_validation
from vstent.reporting import demo as demo_mod
from vstent.reporting.mvp import collect, phase2_items, write_mvp_report


def test_demo_without_surrogate_uses_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_mod, "latest_model_dir", lambda *_: None)
    report = demo_mod.run_demo(tmp_path)
    assert report["demo"]["planning_backend"] == "solver"
    assert report["post_pci"]["ffr_distal_solver"] > 0.90 > report["pre_pci"]["ffr_distal_solver"]
    assert report["pre_pci"]["ffr_distal_surrogate"] is None
    md = (tmp_path / "demo.md").read_text(encoding="utf-8")
    assert "Research use only" in md and "no exported surrogate" in md
    assert (tmp_path / "demo.png").read_bytes()[:4] == b"\x89PNG"
    assert json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))["stents"]


def test_demo_with_surrogate(tmp_path, monkeypatch, tiny_export):
    monkeypatch.setattr(demo_mod, "latest_model_dir", lambda *_: Path(tiny_export["out_dir"]))
    report = demo_mod.run_demo(tmp_path)
    assert report["demo"]["planning_backend"] == "surrogate"
    assert report["pre_pci"]["ffr_distal_surrogate"] is not None
    assert all(p["ffr_solver_check"] is not None for p in report["plans"]["top"])


def test_mvp_report_from_generated_files(tmp_path, monkeypatch):
    write_qca_validation(tmp_path)
    monkeypatch.setattr(demo_mod, "latest_model_dir", lambda *_: None)
    demo_mod.run_demo(tmp_path / "demo")
    data = write_mvp_report(tmp_path)
    assert {r["milestone"] for r in data["rows"]} == {"M5"}  # only the QCA report is present
    assert "physics_checks.json" in data["missing"] and not data["all_targets_met"]
    assert any(f.startswith("Demo (synthetic tandem lesions") for f in data["facts"])
    md = (tmp_path / "mvp_report.md").read_text(encoding="utf-8")
    assert "Research use only" in md and "Missing reports" in md and "Phase 2" in md


def test_mvp_collect_on_committed_reports():
    data = collect(Path("reports"))
    milestones = {r["milestone"] for r in data["rows"]}
    assert {"M1", "M2", "M3", "M4", "M5", "M6", "M7"} <= milestones
    assert all(p["commit"] for p in data["provenance"])


def test_phase2_items(tmp_path):
    plan = tmp_path / "PLAN.md"
    plan.write_text(
        "## M8: x\n- [ ] a\n\n## P1: Done one\n- [x] a\n- [x] b\n\n## P2: Half\n- [x] a\n"
        "- [ ] b\n\n## P3: Later [needs human]\n- [ ] a\n",
        "utf-8",
    )
    assert phase2_items(plan) == [
        "P1: Done one (done)",
        "P2: Half (in progress)",
        "P3: Later [needs human] (not started)",
    ]
    assert phase2_items(tmp_path / "missing.md") == []
    assert phase2_items()[0] == "P1: Uncertainty and grey-zone flagging (done)"  # the real plan


def test_cli_demo_and_qca_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_mod, "latest_model_dir", lambda *_: None)
    result = CliRunner().invoke(app, ["demo", "--out", str(tmp_path / "demo")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "demo" / "demo.md").exists()
    result = CliRunner().invoke(app, ["qca-synthetic", "--report-dir", str(tmp_path)])
    assert result.exit_code == 0 and "[PASS]" in result.output


def test_mvp_provenance_has_creation_times():
    data = collect(Path("reports"))
    assert all(p["created"] for p in data["provenance"])
