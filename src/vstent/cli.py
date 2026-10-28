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
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config (new run)."),
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue a stopped run (name or path).")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Override the run name.")] = None,
) -> None:
    """Train the FFR surrogate (new run with --config, or continue one with --resume)."""
    from vstent.config import load_config
    from vstent.surrogate.train import SurrogateConfig, resume_surrogate, train_surrogate

    if (config is None) == (resume is None):
        typer.echo("Give exactly one of --config (new run) or --resume (continue).", err=True)
        raise typer.Exit(code=2)
    if resume is not None:
        run_dir = resume_surrogate(resume, log=typer.echo)
    else:
        cfg = load_config(config, SurrogateConfig)
        if name is not None:
            cfg = cfg.model_copy(update={"name": name})
        run_dir = train_surrogate(cfg, log=typer.echo)
    typer.echo(f"Run directory: {run_dir}")


@app.command()
def checkpoints() -> None:
    """List resumable training runs in checkpoints/ with their status."""
    from vstent.checkpoints import list_checkpoints

    rows = list_checkpoints()
    if not rows:
        typer.echo("No checkpoints.")
        return
    for r in rows:
        metric = r.get("best_metric", "val_distal_mae")
        best = r.get("best_value", r.get("best_val_distal_mae"))
        best_txt = ""
        if best is not None:
            best_txt = f", best {metric} {best:.4f} (epoch {r['best_epoch']})"
        typer.echo(
            f"{r['name']}  [{r['kind']}]  {r['status']}  epoch {r['epoch']}/{r['epochs']}"
            f"{best_txt}  updated {r['updated_utc']}"
        )
        if r["status"] != "finished":
            typer.echo(f"    resume: {r['resume_command']}")


@app.command()
def stop(
    run: Annotated[str, typer.Argument(help="Run name (as shown by `vstent checkpoints`).")],
) -> None:
    """Ask a running training to save its state and exit after the current epoch."""
    from vstent.checkpoints import request_stop, resolve_checkpoint

    try:
        ckpt_dir = resolve_checkpoint(run)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    request_stop(ckpt_dir)
    typer.echo(f"Stop requested for {ckpt_dir.name}; it will stop after the current epoch.")


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
def export_surrogate(
    run: RunOpt,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Model directory (default: models/surrogate/<run-name>)."),
    ] = None,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write onnx_benchmark.md.")
    ] = Path("reports"),
) -> None:
    """Export a trained surrogate to ONNX, check parity with torch and benchmark it on CPU."""
    from vstent.inference.benchmark import run_benchmark
    from vstent.surrogate.export import export_run

    meta = export_run(run, out, log=typer.echo)
    report = run_benchmark(meta["out_dir"], report_dir, export_meta=meta, log=typer.echo)
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Model in {meta['out_dir']}; wrote {report_dir / 'onnx_benchmark.md'}")
    if not report["targets_met"]:
        raise typer.Exit(code=1)


@data_app.command()
def arcade(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path("data/raw/arcade"),
    force: Annotated[bool, typer.Option("--force", help="Delete and download again.")] = False,
) -> None:
    """Download, verify (MD5) and extract the ARCADE angiography dataset (Zenodo 10390295)."""
    from urllib.error import URLError

    from vstent.data.arcade import URL, fetch_arcade

    try:
        source = fetch_arcade(out, force=force, log=typer.echo)
    except (URLError, OSError, ValueError) as exc:
        typer.echo(
            f"ARCADE download failed: {exc}\nDownload {URL} manually into {out}/ "
            "and run this command again (the archive is verified, then extracted).",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"ARCADE ready in {out} (licence {source['licence']}, MD5 {source['md5']})")


@app.command()
def train_seg(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config (new run)."),
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue a stopped run (name or path).")
    ] = None,
) -> None:
    """Train the vessel segmentation model (new run with --config, or continue with --resume)."""
    from vstent.config import load_config
    from vstent.imaging.seg_train import SegConfig, resume_seg, train_seg

    if (config is None) == (resume is None):
        typer.echo("Give exactly one of --config (new run) or --resume (continue).", err=True)
        raise typer.Exit(code=2)
    if resume is not None:
        run_dir = resume_seg(resume, log=typer.echo)
    else:
        run_dir = train_seg(load_config(config, SegConfig), log=typer.echo)
    typer.echo(f"Run directory: {run_dir}")


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
