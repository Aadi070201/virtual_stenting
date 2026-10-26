import re

import pytest
import yaml
from pydantic import ValidationError

from vstent.config import (
    CONFIG_FILENAME,
    META_FILENAME,
    StrictModel,
    load_config,
    save_run_metadata,
)


class _DemoConfig(StrictModel):
    seed: int = 0
    length_mm: float = 50.0
    name: str = "demo"


def test_load_config_fills_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 3\n", encoding="utf-8")
    cfg = load_config(path, _DemoConfig)
    assert cfg.seed == 3
    assert cfg.length_mm == 50.0


def test_empty_yaml_gives_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path, _DemoConfig) == _DemoConfig()


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("sed: 3\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path, _DemoConfig)


def test_non_mapping_yaml_rejected(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path, _DemoConfig)


def test_save_run_metadata_roundtrip(tmp_path):
    cfg = _DemoConfig(seed=7, name="run")
    run_dir = save_run_metadata(tmp_path / "runs" / "r1", cfg)

    assert load_config(run_dir / CONFIG_FILENAME, _DemoConfig) == cfg

    meta = yaml.safe_load((run_dir / META_FILENAME).read_text(encoding="utf-8"))
    assert meta["config_class"] == "_DemoConfig"
    assert "git_commit" in meta
    if meta["git_commit"] is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", meta["git_commit"])
