"""Command-line interface: ``vstent <command>``.

Commands are stubs until their milestone in docs/PLAN.md is implemented.
"""

from pathlib import Path
from typing import Annotated

import typer

from vstent import DISCLAIMER, __version__

app = typer.Typer(
    name="vstent",
    help=f"Virtual stenting planner v{__version__}. {DISCLAIMER}",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Download and prepare datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config file.")
]
RunOpt = Annotated[
    Path, typer.Option("--run", exists=True, file_okay=False, help="Run directory (runs/<name>).")
]


def _not_implemented(milestone: str) -> None:
    typer.echo(f"Not implemented yet (planned in {milestone}, see docs/PLAN.md).", err=True)
    raise typer.Exit(code=1)


@app.command()
def synth(config: ConfigOpt) -> None:
    """Generate the synthetic vessel dataset."""
    _not_implemented("M2")


@app.command()
def train_surrogate(config: ConfigOpt) -> None:
    """Train the FFR surrogate model."""
    _not_implemented("M3")


@app.command()
def eval_surrogate(run: RunOpt) -> None:
    """Evaluate a trained surrogate against the physics solver."""
    _not_implemented("M3")


@app.command()
def export_surrogate(run: RunOpt) -> None:
    """Export a trained surrogate to ONNX."""
    _not_implemented("M4")


@data_app.command()
def arcade() -> None:
    """Download and verify the ARCADE angiography dataset."""
    _not_implemented("M5")


@app.command()
def train_seg(config: ConfigOpt) -> None:
    """Train the vessel segmentation model."""
    _not_implemented("M5")


@app.command()
def demo() -> None:
    """Run a synthetic case end-to-end and write reports/demo/."""
    _not_implemented("M8")


@app.command("app")
def launch_app() -> None:
    """Launch the Streamlit app."""
    _not_implemented("M7")


if __name__ == "__main__":
    app()
