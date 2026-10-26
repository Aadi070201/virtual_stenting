import json
from pathlib import Path

import pytest

from vstent.config import load_config
from vstent.physics.report import write_physics_report
from vstent.physics.solver import PhysicsParams

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "physics.yaml"


def test_physics_yaml_matches_spec_defaults():
    assert load_config(CONFIG, PhysicsParams) == PhysicsParams()


@pytest.mark.slow
def test_write_physics_report(tmp_path):
    report = write_physics_report(tmp_path)
    assert all(c["passed"] for c in report["checks"])
    assert (tmp_path / "figures" / "ffr_vs_ds.png").stat().st_size > 0
    saved = json.loads((tmp_path / "physics_checks.json").read_text(encoding="utf-8"))
    assert saved["checks"] == report["checks"]
    assert "Research use only" in (tmp_path / "physics_checks.md").read_text(encoding="utf-8")
