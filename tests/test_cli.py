from typer.testing import CliRunner

from vstent import DISCLAIMER
from vstent.cli import app

runner = CliRunner()

COMMANDS = [
    "physics-report",
    "synth",
    "train-surrogate",
    "eval-surrogate",
    "export-surrogate",
    "data",
    "train-seg",
    "demo",
    "app",
    "app-latency",
    "qca-synthetic",
    "mvp-report",
    "plan-benchmark",
    "qca-gallery",
    "analyse",
]


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


def test_help_shows_disclaimer():
    result = runner.invoke(app, ["--help"])
    assert "Research use only" in result.output
    assert DISCLAIMER.startswith("Research use only")


def test_data_group_lists_arcade():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "arcade" in result.output
