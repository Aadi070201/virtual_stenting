from typer.testing import CliRunner

from vstent import DISCLAIMER
from vstent.cli import app

runner = CliRunner()

COMMANDS = [
    "physics-report",
    "synth",
    "synth-tree",
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
    "train-ensemble",
    "export-ensemble",
    "uncertainty-report",
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


def test_console_output_never_crashes_on_legacy_code_page(monkeypatch):
    import io
    import sys

    from vstent.cli import _safe_console

    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)
    _safe_console()
    print("DS 30% → 90%, FFR ≥ 0.90")  # → and ≥ are not in cp1252
    stream.flush()
    assert buf.getvalue() == b"DS 30% ? 90%, FFR ? 0.90\n"
