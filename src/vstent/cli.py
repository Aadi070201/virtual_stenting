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
PHYSICS_CONFIG = Path("configs/physics.yaml")


def _not_implemented(milestone: str) -> None:
    typer.echo(f"Not implemented yet (planned in {milestone}, see docs/PLAN.md).", err=True)
    raise typer.Exit(code=1)


@app.command()
def physics_report(
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("reports"),
) -> None:
    """Run the physics sanity checks and write the ffr_vs_ds figure and physics_checks report."""
    from vstent.config import load_config
    from vstent.physics.report import write_physics_report
    from vstent.physics.solver import PhysicsParams

    report = write_physics_report(out, load_config(config, PhysicsParams))
    for c in report["checks"]:
        typer.echo(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['value']:.4f}")
    bench = report["benchmark"]
    typer.echo(f"Batch solver: {bench['profiles_per_second']:,.0f} profiles/s ({bench['cpu']})")
    typer.echo(f"Wrote {out / 'physics_checks.md'} and {out / 'figures' / 'ffr_vs_ds.png'}")
    if not all(c["passed"] for c in report["checks"]) or not bench["passed"]:
        raise typer.Exit(code=1)


@app.command()
def synth(
    config: ConfigOpt,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output directory (default: out_dir in config).")
    ] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Worker processes (default: all cores).")
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace existing data.")] = False,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write the dataset card.")
    ] = Path("reports"),
) -> None:
    """Generate the synthetic vessel dataset and its dataset card."""
    from vstent.config import load_config
    from vstent.data.card import write_dataset_card
    from vstent.data.synth import SynthConfig, generate_dataset

    cfg = load_config(config, SynthConfig)
    try:
        manifest = generate_dataset(
            cfg, out_dir=out, workers=workers, overwrite=overwrite, log=typer.echo
        )
    except FileExistsError as exc:
        typer.echo(f"{exc}. Pass --overwrite to replace it.", err=True)
        raise typer.Exit(code=1) from exc
    n_total = sum(split["n"] for split in manifest["splits"].values())
    typer.echo(
        f"Generated {n_total:,} samples in {manifest['generation_seconds']:.1f} s "
        f"({manifest['workers']} workers)"
    )
    card = write_dataset_card(out or Path(cfg.out_dir), report_dir)
    for t in card["targets"]:
        typer.echo(
            f"[{'PASS' if t['passed'] else 'FAIL'}] {t['split']}: FFR < 0.80 "
            f"{t['frac_ffr_below_cutoff']:.1%}, grey zone {t['frac_grey_zone']:.1%}"
        )
    typer.echo(f"Wrote {report_dir / 'dataset_card.md'}")
    if not card["targets_met"]:
        raise typer.Exit(code=1)


@app.command()
def train_surrogate(
    config: ConfigOpt,
    name: Annotated[str | None, typer.Option("--name", help="Override the run name.")] = None,
) -> None:
    """Train the FFR surrogate model; prints the run directory."""
    from vstent.config import load_config
    from vstent.surrogate.train import SurrogateConfig, train_surrogate

    cfg = load_config(config, SurrogateConfig)
    if name is not None:
        cfg = cfg.model_copy(update={"name": name})
    run_dir = train_surrogate(cfg, log=typer.echo)
    typer.echo(f"Run directory: {run_dir}")


@app.command()
def eval_surrogate(
    run: RunOpt,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write surrogate_eval.md.")
    ] = Path("reports"),
    device: Annotated[str, typer.Option("--device", help="auto, cpu or cuda.")] = "auto",
) -> None:
    """Evaluate a trained surrogate against the physics solver and the baseline."""
    from vstent.surrogate.evaluate import evaluate_run

    report = evaluate_run(run, report_dir, device=device, log=typer.echo)
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Wrote {report_dir / 'surrogate_eval.md'}")
    if not report["targets_met"]:
        raise typer.Exit(code=1)


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
