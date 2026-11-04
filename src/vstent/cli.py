"""Command-line interface: ``vstent <command>`` (research use only)."""

import math
import sys
from pathlib import Path
from typing import Annotated, Any

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


def _safe_console() -> None:
    # Messages contain characters such as "→", "≥" or "×". A Windows console or pipe in a
    # legacy code page (cp1252) cannot encode some of them; print a replacement instead of
    # crashing after the work is done (found in the fresh-clone test).
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if hasattr(stream, "reconfigure") and encoding != "utf8":
            stream.reconfigure(errors="replace")


@app.callback()
def _main() -> None:
    _safe_console()


ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config file.")
]
RunOpt = Annotated[
    Path, typer.Option("--run", exists=True, file_okay=False, help="Run directory (runs/<name>).")
]
PHYSICS_CONFIG = Path("configs/physics.yaml")
PLANNER_CONFIG = Path("configs/planner.yaml")
CATALOG_CONFIG = Path("configs/stent_catalog.yaml")
UNCERTAINTY_CONFIG = Path("configs/uncertainty.yaml")
VALIDATION_CONFIG = Path("configs/validation.yaml")


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
def synth_tree(
    config: ConfigOpt,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output directory (default: out_dir in config).")
    ] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Worker processes (default: all cores).")
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace existing data.")] = False,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write the tree dataset card.")
    ] = Path("reports"),
) -> None:
    """Generate the synthetic vessel-tree dataset (P2) and its dataset card."""
    from vstent.config import load_config
    from vstent.data.synth import generate_dataset
    from vstent.data.synth_tree import TreeSynthConfig
    from vstent.data.tree_card import CARD, write_tree_dataset_card

    cfg = load_config(config, TreeSynthConfig)
    try:
        manifest = generate_dataset(
            cfg, out_dir=out, workers=workers, overwrite=overwrite, log=typer.echo
        )
    except FileExistsError as exc:
        typer.echo(f"{exc}. Pass --overwrite to replace it.", err=True)
        raise typer.Exit(code=1) from exc
    n_total = sum(split["n"] for split in manifest["splits"].values())
    typer.echo(
        f"Generated {n_total:,} trees in {manifest['generation_seconds']:.1f} s "
        f"({manifest['workers']} workers)"
    )
    card = write_tree_dataset_card(out or Path(cfg.out_dir), report_dir)
    for t in card["targets"]:
        typer.echo(
            f"[{'PASS' if t['passed'] else 'FAIL'}] {t['split']}: main FFR < 0.80 "
            f"{t['frac_ffr_below_cutoff']:.1%}, grey zone {t['frac_grey_zone']:.1%}"
        )
    typer.echo(f"Wrote {report_dir / (CARD + '.md')}")
    if not card["targets_met"]:
        raise typer.Exit(code=1)


@app.command()
def tree_report(
    data: Annotated[
        Path, typer.Option("--data", help="Synthetic tree dataset (vstent synth-tree).")
    ] = Path("data/synth/tree_v1"),
    physics: Annotated[
        Path, typer.Option("--physics", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    n_trees: Annotated[int, typer.Option("--n", min=10, help="Test trees to check.")] = 1000,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write tree_physics.md.")
    ] = Path("reports"),
) -> None:
    """Tree physics sanity checks and solver throughput (P2) → reports/tree_physics.md."""
    from vstent.config import load_config
    from vstent.physics.solver import PhysicsParams
    from vstent.physics.tree_report import write_tree_report

    if not (data / "manifest.json").exists():
        typer.echo(f"No tree dataset in {data}: run vstent synth-tree first.", err=True)
        raise typer.Exit(code=1)
    report = write_tree_report(data, report_dir, load_config(physics, PhysicsParams), n_trees)
    for name, check in report["checks"].items():
        typer.echo(f"[{'PASS' if check['passed'] else 'FAIL'}] {name}")
    b = report["benchmark"]
    typer.echo(
        f"[{'PASS' if b['passed'] else 'FAIL'}] throughput {b['trees_per_s']:,.0f} trees/s "
        f"(target {b['target']:,.0f})"
    )
    typer.echo(f"Wrote {report_dir / 'tree_physics.md'}")
    if not (report["checks_passed"] and b["passed"]):
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
def train_tree_surrogate(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config (new run)."),
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue a stopped run (name or path).")
    ] = None,
) -> None:
    """Train the tree FFR surrogate (P3; new run with --config, or continue with --resume)."""
    from vstent.config import load_config
    from vstent.surrogate.tree_train import (
        TreeSurrogateConfig,
        resume_tree_surrogate,
        train_tree_surrogate,
    )

    if (config is None) == (resume is None):
        typer.echo("Give exactly one of --config (new run) or --resume (continue).", err=True)
        raise typer.Exit(code=2)
    if resume is not None:
        run_dir = resume_tree_surrogate(resume, log=typer.echo)
    else:
        run_dir = train_tree_surrogate(load_config(config, TreeSurrogateConfig), log=typer.echo)
    typer.echo(f"Run directory: {run_dir}")


@app.command()
def eval_tree_surrogate(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False, help="Run dir.")],
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
    device: Annotated[str, typer.Option("--device", help="auto, cpu or cuda.")] = "auto",
) -> None:
    """Evaluate a tree surrogate against the tree solver → reports/tree_surrogate_eval.md."""
    from vstent.surrogate.tree_evaluate import evaluate_tree_run

    report = evaluate_tree_run(run, report_dir, device)
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']:.4f}")
    typer.echo(f"Wrote {report_dir / 'tree_surrogate_eval.md'}")


@app.command()
def export_tree_surrogate(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False, help="Run dir.")],
    out: Annotated[
        Path | None, typer.Option("--out", help="Default: models/tree_surrogate/<run-name>/.")
    ] = None,
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """Export a tree surrogate to ONNX (padded graphs), parity + CPU latency → tree_onnx.md."""
    from vstent.surrogate.tree_export import export_tree_run

    meta = export_tree_run(run, out, report_dir, log=typer.echo)
    typer.echo(f"[{'PASS' if meta['parity']['passed'] else 'FAIL'}] ONNX parity")
    typer.echo(f"Model in {meta['out_dir']}")
    if not meta["parity"]["passed"]:
        raise typer.Exit(code=1)


@app.command()
def train_ensemble(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config (new run)."),
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue a stopped ensemble (name or path).")
    ] = None,
) -> None:
    """Train a surrogate ensemble, one member per seed (new with --config, or --resume)."""
    from vstent.config import load_config
    from vstent.surrogate.ensemble import EnsembleConfig, resume_ensemble, train_ensemble

    if (config is None) == (resume is None):
        typer.echo("Give exactly one of --config (new run) or --resume (continue).", err=True)
        raise typer.Exit(code=2)
    if resume is not None:
        run_dir = resume_ensemble(resume, log=typer.echo)
    else:
        run_dir = train_ensemble(load_config(config, EnsembleConfig), log=typer.echo)
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
        unit = "members" if r["kind"] == "surrogate_ensemble" else "epoch"
        typer.echo(
            f"{r['name']}  [{r['kind']}]  {r['status']}  {unit} {r['epoch']}/{r['epochs']}"
            f"{best_txt}  updated {r['updated_utc']}"
        )
        if r["status"] != "finished":
            typer.echo(f"    resume: {r['resume_command']}")


@app.command()
def stop(
    run: Annotated[str, typer.Argument(help="Run name (as shown by `vstent checkpoints`).")],
) -> None:
    """Ask a running training to save its state and exit after the current epoch."""
    from vstent.checkpoints import read_status, request_stop, resolve_checkpoint
    from vstent.surrogate.ensemble import KIND as ENSEMBLE_KIND
    from vstent.surrogate.ensemble import stop_ensemble

    try:
        ckpt_dir = resolve_checkpoint(run)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if read_status(ckpt_dir).get("kind") == ENSEMBLE_KIND:
        member = stop_ensemble(ckpt_dir)
        typer.echo(
            f"Stop requested for ensemble {ckpt_dir.name}"
            + (f"; member {member} stops after its current epoch." if member else ".")
        )
        return
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


@app.command()
def export_ensemble(
    run: RunOpt,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Model directory (default: models/surrogate/<run-name>)."),
    ] = None,
    uncertainty: Annotated[
        Path, typer.Option("--uncertainty", exists=True, dir_okay=False, help="Uncertainty YAML.")
    ] = UNCERTAINTY_CONFIG,
) -> None:
    """Export a surrogate ensemble to ONNX (mean + spread), check parity, calibrate intervals."""
    from vstent.config import load_config
    from vstent.surrogate.export import export_ensemble as run_export
    from vstent.uncertainty.greyzone import UncertaintyConfig

    meta = run_export(
        run, out, uncertainty=load_config(uncertainty, UncertaintyConfig), log=typer.echo
    )
    passed = meta["parity"]["passed"] and meta["parity_screen"]["passed"]
    typer.echo(f"[{'PASS' if passed else 'FAIL'}] ONNX parity (ensemble mean, SD and screen)")
    typer.echo(f"Model in {meta['out_dir']}")
    if not passed:
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


@data_app.command()
def cadica(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path("data/raw/cadica"),
    force: Annotated[bool, typer.Option("--force", help="Download and extract again.")] = False,
) -> None:
    """Download (3.1 GB), verify (SHA-256) and extract the CADICA angiography dataset."""
    from urllib.error import URLError

    from vstent.data.cadica import URL, fetch_cadica

    try:
        source = fetch_cadica(out, force=force, log=typer.echo)
    except (URLError, OSError, ValueError) as exc:
        typer.echo(
            f"CADICA download failed: {exc}\nDownload {URL} manually as {out}/CADICA.zip "
            "and run this command again (the archive is verified, then extracted).",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"CADICA ready in {out} (licence {source['licence']})")


@data_app.command()
def angiocad(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path("data/raw/angiocad"),
) -> None:
    """Download (16.4 GB) and verify (MD5) the AngioCAD dataset (CC BY 4.0; 413 patients,
    segment-level stenosis grades; external test, P4k-J). Extraction is a separate step."""
    from urllib.error import URLError

    from vstent.data.angiocad import fetch_angiocad

    try:
        source = fetch_angiocad(out, log=typer.echo)
    except (URLError, OSError, ValueError) as exc:
        typer.echo(f"AngioCAD download failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified {len(source['files'])} files in {out} ({source['licence']})")


@data_app.command()
def cardiag(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path("data/raw/cardiag"),
) -> None:
    """Download (4.6 GB) and verify (MD5) the CARDIAG dataset (CC BY 4.0; 114 patients, SYNTAX
    segment masks, catheter masks, C-arm geometry with pixel spacing). Extraction is a separate
    step."""
    from urllib.error import URLError

    from vstent.data.cardiag import fetch_cardiag

    try:
        source = fetch_cardiag(out, log=typer.echo)
    except (URLError, OSError, ValueError) as exc:
        typer.echo(f"CARDIAG download failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified {len(source['files'])} file in {out} ({source['licence']})")


@data_app.command()
def cardiosyntax(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path(
        "data/raw/cardiosyntax"
    ),
    parts: Annotated[str, typer.Option("--parts", help="Zip parts 0-9, comma-separated.")] = "9",
) -> None:
    """Download and verify (MD5) CardioSYNTAX (CC BY 4.0; 1,844 studies, 14,219 cine runs at
    15 fps, SYNTAX per side): the metadata and the chosen zip parts (~16 GB each, part 9
    3.2 GB). The runs are read from the zips; nothing is extracted or run."""
    from urllib.error import URLError

    from vstent.data.cardiosyntax import fetch_cardiosyntax

    chosen = tuple(int(p) for p in parts.split(",") if p.strip())
    if any(not 0 <= p <= 9 for p in chosen):
        raise typer.BadParameter("--parts are 0-9")
    try:
        source = fetch_cardiosyntax(out, chosen, log=typer.echo)
    except (URLError, OSError, ValueError) as exc:
        typer.echo(f"CardioSYNTAX download failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified {len(source['files'])} files in {out} ({source['licence']})")


@data_app.command()
def kemerovo(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path("data/raw/kemerovo"),
    force: Annotated[bool, typer.Option("--force", help="Download and extract again.")] = False,
) -> None:
    """Download (8.3 GB), verify (SHA-256) and extract the frames and labels of the Kemerovo
    stenosis dataset (CC BY 4.0; external test of lesion reading, P4i)."""
    from urllib.error import URLError

    from vstent.data.kemerovo import ARCHIVE, URL, fetch_kemerovo

    try:
        source = fetch_kemerovo(out, force=force, log=typer.echo)
    except (URLError, OSError, ValueError) as exc:
        typer.echo(
            f"Kemerovo download failed: {exc}\nDownload {URL} manually as {out}/{ARCHIVE} "
            "and run this command again (the archive is verified, then extracted).",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"Kemerovo ready in {out} (licence {source['licence']})")


@data_app.command("fda-nozzle")
def fda_nozzle(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path(
        "data/raw/fda_nozzle"
    ),
) -> None:
    """Download and verify the FDA benchmark nozzle's experimental data (public domain)."""
    from urllib.error import URLError

    from vstent.data.fda_nozzle import fetch_fda_nozzle

    try:
        source = fetch_fda_nozzle(out, log=typer.echo)
    except (URLError, OSError, ValueError) as exc:
        typer.echo(f"FDA nozzle download failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"FDA nozzle data ready in {out} (commit {source['commit'][:12]})")


@data_app.command("define-flow")
def define_flow(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path(
        "data/raw/define_flow"
    ),
) -> None:
    """Download and verify DEFINE-FLOW: invasive FFR, CFR and Doppler velocity (CC0; P4e)."""
    _fetch_physiology("define_flow", out)


@data_app.command("contrast")
def contrast_data(
    out: Annotated[Path, typer.Option("--out", help="Target folder.")] = Path("data/raw/contrast"),
) -> None:
    """Download and verify CONTRAST: raw coronary pressure tracings (CC0; P4e)."""
    _fetch_physiology("contrast", out)


def _fetch_physiology(name: str, out: Path) -> None:
    from urllib.error import URLError

    from vstent.data.physiology import fetch_physiology

    try:
        source = fetch_physiology(name, out, log=typer.echo)
    except (URLError, OSError, ValueError) as exc:
        typer.echo(f"{name} download failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{source['title'].split(':')[0]} ready in {out} ({source['licence']})")


@app.command()
def train_seg(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config (new run)."),
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue a stopped run (name or path).")
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", min=0, help="DataLoader processes (memory; results unchanged)."),
    ] = None,
) -> None:
    """Train the vessel segmentation model (new run with --config, or continue with --resume).
    ``--workers`` overrides the config's DataLoader processes: each holds a copy of the images,
    and augmentations depend only on the saved generator, so the results do not change."""
    from vstent.config import load_config
    from vstent.imaging.seg_train import SegConfig, resume_seg, train_seg

    if (config is None) == (resume is None):
        typer.echo("Give exactly one of --config (new run) or --resume (continue).", err=True)
        raise typer.Exit(code=2)
    if resume is not None:
        run_dir = resume_seg(resume, log=typer.echo, workers=workers)
    else:
        cfg = load_config(config, SegConfig)
        if workers is not None:
            cfg = cfg.model_copy(update={"train": cfg.train.model_copy(
                update={"workers": workers})})  # fmt: skip
        run_dir = train_seg(cfg, log=typer.echo)
    typer.echo(f"Run directory: {run_dir}")


@app.command()
def synth_angio(
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config.")
    ] = Path("configs/synth_angio_v1.yaml"),
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write synth_angio_card.md.")
    ] = Path("reports"),
    workers: Annotated[int | None, typer.Option("--workers", help="Processes.")] = None,
) -> None:
    """Render synthetic angiograms with vessel labels for segmentation training (P8)."""
    from vstent.config import load_config
    from vstent.data.synth_angio import SynthAngioConfig, generate, write_card

    cfg = load_config(config, SynthAngioConfig)
    if workers is not None:
        cfg = cfg.model_copy(update={"workers": workers})
    out = generate(cfg, log=typer.echo)
    write_card(out, report_dir)
    typer.echo(f"Wrote {out} and {report_dir / 'synth_angio_card.md'}")


@app.command()
def sim_to_real_report(
    base: Annotated[Path, typer.Option("--base", exists=True, help="Baseline run (M5).")],
    new: Annotated[Path, typer.Option("--new", exists=True, help="Run with rendered data.")],
    synth_dir: Annotated[Path, typer.Option("--synth-dir", help="Rendered dataset.")] = Path(
        "data/synth_angio/v1"
    ),
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write sim_to_real.md.")
    ] = Path("reports"),
    device: Annotated[str, typer.Option("--device", help="auto, cpu or cuda.")] = "auto",
) -> None:
    """Both segmenters on ARCADE val / test and the rendered test split, with the
    pre-registered adoption rule (P8)."""
    from vstent.imaging.sim_to_real import compare_runs

    typer.echo(DISCLAIMER)
    report = compare_runs(base, new, synth_dir, report_dir, device=device)
    for r in report["rows"]:
        typer.echo(f"{r['split']}: baseline {r['baseline']:.4f}, new {r['new']:.4f}, "
                   f"difference {r['diff']:+.4f} ({r['lo']:+.4f} to {r['hi']:+.4f})")  # fmt: skip
    typer.echo(f"Adoption rule: {'adopt' if report['adopt'] else 'keep the baseline'}")
    typer.echo(f"Wrote {report_dir / 'sim_to_real.md'}")


@app.command()
def eval_seg(
    run: RunOpt,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write seg_eval.md.")
    ] = Path("reports"),
    device: Annotated[str, typer.Option("--device", help="auto, cpu or cuda.")] = "auto",
) -> None:
    """Evaluate a segmentation run on ARCADE syntax val / test (Dice)."""
    from vstent.imaging.seg_eval import evaluate_seg_run

    report = evaluate_seg_run(run, report_dir, device=device, log=typer.echo)
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Wrote {report_dir / 'seg_eval.md'}")
    if not report["targets_met"]:
        raise typer.Exit(code=1)


@app.command()
def export_seg(
    run: RunOpt,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Model directory (default: models/seg/<run-name>)."),
    ] = None,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write seg_onnx.md.")
    ] = Path("reports"),
) -> None:
    """Export a segmentation run to ONNX and check parity with torch."""
    from vstent.imaging.seg_export import export_seg_run

    meta = export_seg_run(run, out, report_dir, log=typer.echo)
    passed = meta["parity"]["passed"]
    typer.echo(f"[{'PASS' if passed else 'FAIL'}] parity {meta['parity']['max_abs_diff']:.1e}")
    typer.echo(f"Model in {meta['out_dir']}; wrote {report_dir / 'seg_onnx.md'}")
    if not passed:
        raise typer.Exit(code=1)


def _run_frame(run, segmenter, frame: int | None):
    """The frame of a cine run to analyse (``frame``, or the end-diastolic frame chosen by
    :func:`~vstent.imaging.run_analysis.analyse_frames`), the run's analysis and the frame."""
    from vstent.imaging.run_analysis import analyse_frames

    rf = analyse_frames(run, segmenter.predict)
    op = rf.opacification
    first, last = op.window
    typer.echo(
        f"Cine run of {len(run)} frames: contrast arrival at frame {op.arrival}, opacified "
        f"frames {first}–{last} (0-based). {rf.ed.note}"
    )
    t = rf.analysis_frame if frame is None else frame
    if frame is None:
        typer.echo(f"Analysis frame {t}: {rf.frame_note}")
    elif not first <= frame <= last:
        typer.echo(f"Warning: frame {frame} is outside the opacified frames.")
    typer.echo(f"Analysing frame {t}.")
    return run.frame(t), rf, t


def _measure_flow(rf, frame: int, centerline, distance_m, *, single_view: bool,
                  hyperemic_run: bool):  # fmt: skip
    """Flow velocity along the analysed centreline of a cine run (P7), echoed."""
    fv, tr = rf.velocity(centerline, frame, distance_m, single_view=single_view,
                         hyperemic_run=hyperemic_run)  # fmt: skip
    if tr is not None and tr.plausible and math.isnan(fv.cfv):  # not converted to m/s
        lo, hi = tr.interval
        typer.echo(f"Contrast front {tr.px_per_frame:.1f} px/frame (95%: {lo:.1f}–{hi:.1f}).")
    typer.echo(fv.note())
    return fv


def _echo_flow_ffr(fv, profile, params, chosen: bool, fixed_v: float) -> None:
    """The distal FFR with the other velocity than the one used above (``fixed_v``: the
    configured hyperemic velocity, m/s)."""
    from vstent.physics.solver import solve_ffr

    if not fv.accepted:
        return
    if chosen:
        other = solve_ffr(profile, params.model_copy(update={"v_hyp": fixed_v})).ffr[-1]
        typer.echo(f"Flow: measured velocity used. With the fixed {fixed_v:g} m/s the "
                   f"distal FFR (solver) would be {other:.3f}.")  # fmt: skip
    else:
        other = solve_ffr(profile, params.model_copy(update={"v_hyp": fv.v_hyp})).ffr[-1]
        typer.echo(f"Flow: fixed {params.v_hyp:g} m/s used (the default). With the measured "
                   f"velocity the distal FFR (solver) would be {other:.3f}; --flow measured "
                   "uses it (not yet validated against real flow).")  # fmt: skip


def _analyse_tree_cli(prob, img, start, end, name, params, mm_per_px, stent, outlet, figure,
                      plan=None, lumen=None):  # fmt: skip
    """``vstent analyse --tree``: tree QCA, tree FFR (solver; surrogate if exported),
    optional virtual stent or plans on the path to ``outlet``. ``plan`` = (catalog, config)."""
    from vstent.inference.pipeline import analyse_tree
    from vstent.physics.tree import solve_tree
    from vstent.planning.stent import Stent, apply_path_stents
    from vstent.units import m_to_mm, mm_to_m

    res = analyse_tree(prob, start, end, params=params, mm_per_px=mm_per_px, image=img,
                       lumen=lumen)  # fmt: skip
    tq, sol = res.qca, res.result
    typer.echo(DISCLAIMER)
    typer.echo(
        f"Vessel tree: {tq.tree.n_branches} branches at {tq.mm_per_px:g} mm/px "
        f"({'assumed' if tq.calibration_assumed else 'given'}); "
        f"{len(tq.dropped)} further side path(s) not modelled"
    )
    for k, (q, nm) in enumerate(zip(tq.branches, tq.tree.names, strict=True)):
        where = "" if k == 0 else f", from {m_to_mm(tq.tree.path_offsets(k)[-1][3]):.1f} mm"
        typer.echo(
            f"  {nm}: {q.s_mm[-1]:.1f} mm{where}; MLD {q.mld_mm:.2f} mm, max DS "
            f"{q.max_percent_stenosis:.0f}%; outlet FFR {sol.ffr[k][-1]:.3f}, "
            f"{sol.q_outlet[k] / sol.q_inlet[0]:.0%} of the inflow"
        )
    typer.echo(
        f"Main-path distal FFR: tree {sol.ffr[0][-1]:.3f}; as a single path (M1) "
        f"{res.single.ffr[-1]:.3f}"
    )
    from vstent.inference.models import latest_model_dir
    from vstent.inference.tree_runtime import (
        TREE_MODELS,
        TREE_ONNX_FILE,
        TreeSurrogateRunner,
        floor_warnings,
    )

    tree_dir = latest_model_dir(TREE_MODELS, TREE_ONNX_FILE)
    if tree_dir is not None:
        runner = TreeSurrogateRunner.from_dir(tree_dir)
        try:
            _, sur = runner.predict_tree(tq.tree, sol.pa, sol.pv, sol.r_outlet)
            typer.echo(
                "Tree surrogate outlet FFR: "
                + ", ".join(f"{nm} {f[-1]:.3f}" for nm, f in zip(tq.tree.names, sur, strict=True))
            )
            for w in floor_warnings(sur, tq.tree.names, sol.pa, sol.pv):
                typer.echo(f"Warning: {w}")
        except ValueError as exc:  # more branches than the model takes
            typer.echo(f"Tree surrogate not used: {exc}")
        runner.close()
    for w in res.warnings:
        typer.echo(f"Warning: {w}")
    post = None
    if stent is not None:
        if outlet >= tq.tree.n_branches:
            typer.echo(f"--outlet {outlet}: the tree has {tq.tree.n_branches} branches", err=True)
            raise typer.Exit(code=2)
        st = Stent(mm_to_m(stent[0]), mm_to_m(stent[1]), mm_to_m(stent[2]))
        try:
            stented = apply_path_stents(tq.tree, outlet, [st])
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
        post = solve_tree(stented, params, r_outlet=sol.r_outlet)
        typer.echo(
            f"Virtual stent {stent[2]:.2f} × {stent[1]:.0f} mm at {stent[0]:.1f} mm on the path to "
            f"{tq.tree.names[outlet]}: outlet FFR "
            + ", ".join(
                f"{nm} {a[-1]:.3f} → {b[-1]:.3f}"
                for nm, a, b in zip(tq.tree.names, sol.ffr, post.ffr, strict=True)
            )
        )
    if plan is not None:
        from vstent.planning.tree_planner import plan_tree_stents

        if outlet >= tq.tree.n_branches:
            typer.echo(f"--outlet {outlet}: the tree has {tq.tree.n_branches} branches", err=True)
            raise typer.Exit(code=2)
        pr = plan_tree_stents(tq.tree, outlet, catalog=plan[0], config=plan[1], params=params)
        typer.echo(
            f"Stent plans along the path to {tq.tree.names[outlet]} ({pr.backend}; pre-PCI FFR "
            f"{pr.pre_ffr:.3f}; {pr.n_feasible} feasible of {pr.n_candidates} candidates, "
            f"{pr.n_pairs} pairs, {pr.seconds:.2f} s): {pr.message}"
        )
        for k, p in enumerate(pr.plans, 1):
            stents = " + ".join(
                f"{m_to_mm(s.diameter):.2f} × {m_to_mm(s.length):.0f} mm at "
                f"{m_to_mm(s.s_start):.1f} mm"
                for s in p.stents
            )
            outs = ", ".join(
                f"{nm} {f:.3f}" for nm, f in zip(tq.tree.names, p.ffr_outlets, strict=True)
            )
            typer.echo(f"  {k}. {stents}: outlet FFR {outs}")
    if figure is not None:
        from vstent.inference.figures import plot_tree_analysis

        plot_tree_analysis(res, img, figure, title=f"Vessel tree analysis: {name}", post=post)
        typer.echo(f"Wrote {figure}")


def _analyse_two_views_cli(
    inputs, starts, ends, segmenter, physics, surrogate_model, figure, plan_paths, lumen,
    uncertainty, *, runs=None, flow="fixed", hyperemic_run=False,
) -> None:  # fmt: skip
    """``vstent analyse --view2``: two-view QCA + FFR (P5); with two cine runs, their flow
    velocity along the 3D path (P7)."""
    from vstent.config import load_config
    from vstent.inference.onnx_runtime import ONNX_FILE, SurrogateRunner
    from vstent.inference.pipeline import analyse_two_views
    from vstent.physics.solver import PhysicsParams
    from vstent.uncertainty.greyzone import assessment, load_uncertainty, plan_assessments
    from vstent.uncertainty.inputs import SOURCE_LABELS, InputUncertainty, Measurement

    missing = [i.name for i in inputs if i.geometry is None]
    if missing:
        typer.echo(
            f"No complete projection geometry in {', '.join(missing)}: two-view analysis needs "
            "DICOM with PositionerPrimaryAngle, PositionerSecondaryAngle, "
            "DistanceSourceToDetector, DistanceSourceToPatient and ImagerPixelSpacing.",
            err=True,
        )
        raise typer.Exit(code=2)
    params = load_config(physics, PhysicsParams)
    surrogate = SurrogateRunner.from_dir(
        surrogate_model or _latest_model_dir(Path("models/surrogate"), ONNX_FILE)
    )
    images = tuple(i.image for i in inputs)
    try:
        result = analyse_two_views(
            tuple(segmenter.predict(img) for img in images), starts, ends,
            tuple(i.geometry for i in inputs), params=params, surrogate=surrogate,
            images=images, lumen=lumen,
        )  # fmt: skip
    except ValueError as exc:
        typer.echo(f"Two-view analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    rec = result.reconstruction
    fv, fixed_v = None, params.v_hyp
    if runs is not None:
        import numpy as np

        from vstent.imaging.flow import combine_runs

        per = [
            _measure_flow(rf, t, result.qca[k].centerline,
                          lambda s, k=k: np.interp(s, rec.view_s[:, k], rec.s),
                          single_view=False, hyperemic_run=hyperemic_run)
            for k, (rf, t) in enumerate(runs)
        ]  # fmt: skip
        fv = combine_runs(per)
        typer.echo("Both runs: " + fv.note())
        if flow == "measured" and fv.accepted:
            params = params.model_copy(update={"v_hyp": fv.v_hyp})
            result = analyse_two_views(
                tuple(segmenter.predict(img) for img in images), starts, ends,
                tuple(i.geometry for i in inputs), params=params, surrogate=surrogate,
                images=images, lumen=lumen,
            )  # fmt: skip
            rec = result.reconstruction
    single = result.single_view_lengths_mm
    typer.echo(DISCLAIMER)
    typer.echo(
        f"Two views: {rec.geometries[0].label()} and {rec.geometries[1].label()}, "
        f"{rec.separation_deg:.0f}° apart; mean ray gap {rec.gap_m * 1e3:.2f} mm, residual "
        f"{rec.residual_px[0]:.1f} / {rec.residual_px[1]:.1f} px; view 2 shifted by "
        f"({rec.shift_px[0]:.1f}, {rec.shift_px[1]:.1f}) px (motion between the runs); length "
        f"change with smoother fitting {100 * rec.smoothing_change:+.1f}%."
    )
    prof = result.profile
    d_ref = 2e3 * prof.reference_radius()
    typer.echo(
        f"3D path length {rec.length * 1e3:.1f} mm (single views {single[0]:.1f} and "
        f"{single[1]:.1f} mm); analysed {prof.length * 1e3:.1f} mm; reference diameter "
        f"{d_ref[0]:.2f} → {d_ref[-1]:.2f} mm; MLD {2e3 * prof.r.min():.2f} mm"
    )
    typer.echo(
        f"Distal FFR: solver {result.ffr_distal_solver:.3f}, surrogate "
        f"{result.ffr_distal_surrogate:.3f}"
    )
    ucfg, qca_error = load_uncertainty(uncertainty)
    mc = InputUncertainty(
        prof, params, result.solver.pa, result.solver.pv,
        measurement=Measurement(rec.geometries[0].iso_pixel_m * 1e3, False),
        config=ucfg.inputs, qca_error=qca_error,
        v_hyp_range=fv.v_hyp_interval if fv and fv.accepted and flow == "measured" else None,
    )  # fmt: skip
    pre = assessment(ucfg.pre_pci_threshold, result.ffr_distal_solver,
                     result.ffr_distal_surrogate, result.model_interval, mc.pre_samples,
                     ucfg.level)  # fmt: skip
    typer.echo(f"{pre.level_text} intervals: {_interval_text(pre)}")
    typer.echo(f"  inputs varied: {', '.join(SOURCE_LABELS[s] for s in mc.sources)}")
    for reason in pre.reasons("Pre-PCI distal FFR"):
        typer.echo(f"Grey zone: {reason}")
    for w in result.warnings:
        typer.echo(f"Warning: {w}")
    if fv is not None:
        _echo_flow_ffr(fv, prof, params, flow == "measured" and fv.accepted, fixed_v)
    if plan_paths is not None:
        from vstent.planning.optimizer import PlannerConfig, StentCatalog, plan_stents

        planned = plan_stents(prof, catalog=load_config(plan_paths[0], StentCatalog),
                              config=load_config(plan_paths[1], PlannerConfig), params=params,
                              runner=surrogate)  # fmt: skip
        _echo_plans(planned, plan_assessments(planned, mc, ucfg.level))
    surrogate.close()
    if figure is not None:
        from vstent.inference.figures import plot_two_view_analysis

        title = f"Two-view analysis: {inputs[0].name} + {inputs[1].name}"
        plot_two_view_analysis(result, images, figure, title=title)
        typer.echo(f"Wrote {figure}")


def _interval_text(a) -> str:
    parts = [f"inputs {a.inputs.text()}"] if a.inputs else []
    parts += [f"model {a.model.text()}"] if a.model else []
    flag = "GREY ZONE" if a.flagged else "clear"
    return f"{', '.join(parts)}; {a.threshold:.2f}: {flag}"


SINGLE_VIEW_LENGTHS = (
    "Lengths are measured in one view: foreshortening makes them shorter than the vessel"
)


def _echo_plans(planned, assessments=None) -> None:
    typer.echo(
        f"Stent plans ({planned.backend}; pre-PCI FFR {planned.pre_ffr:.3f}; "
        f"{planned.n_feasible} feasible of {planned.n_candidates} candidates, "
        f"{planned.n_pairs} pairs, {planned.seconds:.2f} s): {planned.message}"
    )
    for k, p in enumerate(planned.plans, 1):
        stents = " + ".join(
            f"{st.diameter * 1e3:.2f} x {st.length * 1e3:.0f} mm at {st.s_start * 1e3:.1f} mm"
            for st in p.stents
        )
        check = f" (solver {p.ffr_solver:.3f})" if p.ffr_solver is not None else ""
        typer.echo(f"  {k}. {stents}: post-PCI FFR {p.ffr:.3f}{check}")
        if assessments:
            typer.echo(f"     intervals: {_interval_text(assessments[k - 1])}")


def _latest_model_dir(root: Path, filename: str) -> Path:
    from vstent.inference.models import latest_model_dir

    found = latest_model_dir(root, filename)
    if found is None:
        typer.echo(f"No exported model in {root}/ (run the export command first).", err=True)
        raise typer.Exit(code=1)
    return found


@app.command()
def analyse(
    image: Annotated[
        Path,
        typer.Argument(
            exists=True, help="Angiogram image, or a cine run: multi-frame DICOM / frame folder."
        ),
    ],
    start: Annotated[
        tuple[int, int] | None, typer.Option("--start", help="Start point: ROW COL (px).")
    ] = None,
    end: Annotated[
        tuple[int, int] | None, typer.Option("--end", help="End point: ROW COL (px).")
    ] = None,
    mm_per_px: Annotated[
        float | None, typer.Option("--mm-per-px", help="Pixel size; default 0.30 (assumed).")
    ] = None,
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    surrogate_model: Annotated[
        Path | None,
        typer.Option("--surrogate-model", help="Default: newest in models/surrogate/."),
    ] = None,
    physics: Annotated[
        Path, typer.Option("--physics", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    figure: Annotated[Path | None, typer.Option("--figure", help="Save a PNG figure.")] = None,
    plan: Annotated[bool, typer.Option("--plan", help="Also propose stent plans.")] = False,
    planner: Annotated[
        Path, typer.Option("--planner", dir_okay=False, help="Planner YAML (with --plan).")
    ] = PLANNER_CONFIG,
    catalog: Annotated[
        Path, typer.Option("--catalog", dir_okay=False, help="Stent catalogue YAML (with --plan).")
    ] = CATALOG_CONFIG,
    uncertainty: Annotated[
        Path, typer.Option("--uncertainty", dir_okay=False, help="Uncertainty YAML.")
    ] = UNCERTAINTY_CONFIG,
    tree: Annotated[
        bool, typer.Option("--tree", help="Vessel tree: the path and its side branches (P2).")
    ] = False,
    stent: Annotated[
        tuple[float, float, float] | None,
        typer.Option("--stent", help="With --tree: START LENGTH DIAMETER (mm) along a path."),
    ] = None,
    outlet: Annotated[
        int, typer.Option("--outlet", min=0, help="With --tree: path to this branch (0 = main).")
    ] = 0,
    lumen: Annotated[
        str | None,
        typer.Option(
            "--lumen", help="Lumen measurement: scaled, edges or model (default: the product's)."
        ),
    ] = None,
    frame: Annotated[
        int | None,
        typer.Option(
            "--frame", min=0, help="Cine run: analyse this frame (0-based), not the auto one."
        ),
    ] = None,
    view2: Annotated[
        Path | None,
        typer.Option(
            "--view2", exists=True, help="Second view (DICOM with geometry): 3D analysis (P5)."
        ),
    ] = None,
    start2: Annotated[
        tuple[int, int] | None,
        typer.Option("--start2", help="With --view2: the start point in view 2: ROW COL (px)."),
    ] = None,
    end2: Annotated[
        tuple[int, int] | None,
        typer.Option("--end2", help="With --view2: the end point in view 2: ROW COL (px)."),
    ] = None,
    flow: Annotated[
        str,
        typer.Option("--flow", help="Cine runs: fixed (default) or measured flow velocity (P7)."),
    ] = "fixed",
    hyperemic_run: Annotated[
        bool,
        typer.Option("--hyperemic-run", help="The run was recorded under hyperemia (adenosine)."),
    ] = False,
    consensus: Annotated[
        int | None,
        typer.Option(
            "--consensus",
            min=1,
            help="Cine runs: heartbeats in the median (default 5; 1 = one frame).",
        ),
    ] = None,
) -> None:
    """Segment an angiogram and compute QCA + FFR for one vessel (research use only).

    A cine run (multi-frame DICOM or a folder of frames) is analysed on its end-diastolic frame
    (R-wave pointer, ECG or the images; or --frame), and the contrast's flow velocity along the
    vessel is measured (P7). The fixed hyperemic velocity stays the default; --flow measured
    uses the measured one (FAVOR Pilot conversion for resting runs; --hyperemic-run: as is).
    The vessel is then measured on up to --consensus end-diastolic frames, the points following
    the heart, and the median FFR is reported with every frame's value (P4g). The frame closest
    to the median gives the profile for the intervals and the plans.

    Without --start / --end the longest vessel path in the mask is used. With --plan the
    virtual stent planner proposes the shortest plan reaching the target FFR. Distal FFR is
    reported with 90% intervals (surrogate ensemble; input assumptions) and grey-zone flags.

    With --tree the side branches leaving the path are measured too and FFR comes from the tree
    solver (and the tree surrogate, if exported; no intervals for trees yet). --plan proposes
    plans and --stent places a virtual stent along the path to branch --outlet; outlet
    resistances stay as before the stent.

    With --view2 (P5) the same vessel in a second projection gives its 3D centreline: length
    free of foreshortening, diameters at their true magnification. Both images need the DICOM
    projection geometry, and the same two landmarks in each view (--start/--end,
    --start2/--end2).
    """
    from vstent.config import load_config
    from vstent.imaging.centerline import longest_path_endpoints
    from vstent.imaging.image_io import is_run, read_image_file, read_run
    from vstent.imaging.segment import THRESHOLD
    from vstent.inference.onnx_runtime import (
        ONNX_FILE,
        SEG_ONNX_FILE,
        SegmentationRunner,
        SurrogateRunner,
    )
    from vstent.inference.pipeline import analyse_vessel
    from vstent.physics.solver import PhysicsParams

    seg_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    segmenter = SegmentationRunner.from_dir(seg_dir)
    run = None
    try:  # the app's loader: the same pixels reach the segmentation model
        if is_run(image):
            run = read_run(image)
            image_input, run_frames, run_t = _run_frame(run, segmenter, frame)
        elif frame not in (None, 0):
            raise ValueError(f"--frame {frame}: a single image only has frame 0")
        else:
            image_input = read_image_file(image)
    except ValueError as exc:
        typer.echo(f"Cannot read {image}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    img = image_input.image
    if mm_per_px is None and image_input.pixel_spacing_mm:
        mm_per_px = image_input.pixel_spacing_mm  # magnification-corrected (P5)
        typer.echo(f"Calibration from the DICOM geometry. {image_input.note}")
    elif mm_per_px is None and image_input.detector_spacing_mm:
        typer.echo(f"{image_input.note} Pass --mm-per-px to set the pixel size.")
    if view2 is not None:
        if tree or stent is not None:
            typer.echo("--view2 does not combine with --tree or --stent yet.", err=True)
            raise typer.Exit(code=2)
        if not (start and end and start2 and end2) or None in (*start, *end, *start2, *end2):
            typer.echo("--view2 needs --start, --end, --start2 and --end2 (the same two "
                       "landmarks in each view).", err=True)  # fmt: skip
            raise typer.Exit(code=2)
        try:
            run2 = read_run(view2) if is_run(view2) else None
            runs = None
            if run2:
                input2, run_frames2, run_t2 = _run_frame(run2, segmenter, None)
                if run is not None:
                    runs = ((run_frames, run_t), (run_frames2, run_t2))
            else:
                input2 = read_image_file(view2)
        except ValueError as exc:
            typer.echo(f"Cannot read {view2}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        _analyse_two_views_cli(
            (image_input, input2), (start, start2), (end, end2), segmenter, physics,
            surrogate_model, figure, (catalog, planner) if plan else None, lumen, uncertainty,
            runs=runs, flow=flow, hyperemic_run=hyperemic_run,
        )  # fmt: skip
        return
    if tree and plan and stent is not None:
        typer.echo("With --tree give either --plan or --stent, not both.", err=True)
        raise typer.Exit(code=2)
    if stent is not None and not tree:
        typer.echo("--stent needs --tree (single paths: use --plan or the app).", err=True)
        raise typer.Exit(code=2)
    if lumen not in (None, "scaled", "edges", "model"):
        typer.echo("--lumen: use scaled, edges or model.", err=True)
        raise typer.Exit(code=2)
    if flow not in ("fixed", "measured"):
        typer.echo("--flow: use fixed or measured.", err=True)
        raise typer.Exit(code=2)
    sur_dir = (
        None if tree else surrogate_model or _latest_model_dir(Path("models/surrogate"), ONNX_FILE)
    )
    prob = segmenter.predict(img)
    if not start or not end or start[0] is None or end[0] is None:
        start, end = longest_path_endpoints(prob >= THRESHOLD)
        typer.echo(f"Auto-selected path: start {start}, end {end} (row, col)")
    params = load_config(physics, PhysicsParams)
    if tree:
        from vstent.planning.optimizer import PlannerConfig, StentCatalog

        plan_cfg = (
            (load_config(catalog, StentCatalog), load_config(planner, PlannerConfig))
            if plan
            else None
        )
        if run is not None:
            from vstent.inference.consensus import CONSENSUS_K, run_consensus, tree_values
            from vstent.inference.pipeline import analyse_tree

            def one_tree(p, im, a, b):
                return analyse_tree(p, a, b, params=params, mm_per_px=mm_per_px, image=im,
                                    lumen=lumen)  # fmt: skip

            try:
                cons = run_consensus(run_frames, run_t, start, end, segmenter.predict, one_tree,
                                     k=consensus or CONSENSUS_K, values=tree_values)  # fmt: skip
            except ValueError as exc:
                typer.echo(f"Cannot trace the vessel tree: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            typer.echo(cons.note("Main-path distal FFR (tree)"))
            c = cons.chosen
            prob, img, start, end = c.analysis.prob, run.frames[c.frame], c.start, c.end
        _analyse_tree_cli(prob, img, start, end, image.name, params, mm_per_px, stent, outlet,
                          figure, plan_cfg, lumen)  # fmt: skip
        return
    surrogate = SurrogateRunner.from_dir(sur_dir)
    from vstent.inference.product import (
        product_config,
        product_gate,
        product_params,
        product_reader,
    )

    prod = product_config()  # the product's FFR pipeline (configs/product.yaml)
    params = product_params(params, prod)
    reader, gate = product_reader(prod.lesion_reader), product_gate(prod)
    result = analyse_vessel(
        prob, start, end, params=params, surrogate=surrogate, mm_per_px=mm_per_px, image=img,
        lumen=lumen, lesion_reader=reader, wire=prod.wire,
        lesion_min_ref_mm=prod.lesion_min_ref_mm, lesion_gate=gate,
    )  # fmt: skip
    q = result.qca
    fv, fixed_v = None, params.v_hyp
    if run is not None:
        from vstent.imaging.run_analysis import pixel_distance

        fv = _measure_flow(run_frames, run_t, q.centerline,
                           None if q.calibration_assumed else pixel_distance(q.mm_per_px),
                           single_view=True, hyperemic_run=hyperemic_run)  # fmt: skip
        if flow == "measured" and fv.accepted:
            params = params.model_copy(update={"v_hyp": fv.v_hyp})
            result = analyse_vessel(
                prob, start, end, params=params, surrogate=surrogate, mm_per_px=mm_per_px,
                image=img, lumen=lumen, lesion_reader=reader, wire=prod.wire,
                lesion_min_ref_mm=prod.lesion_min_ref_mm, lesion_gate=gate,
            )  # fmt: skip
            q = result.qca
        elif flow == "measured":
            typer.echo("--flow measured: the velocity was not accepted; the fixed one is used.")
    cons, where = None, ""
    if run is not None:
        from vstent.inference.consensus import CONSENSUS_K, run_consensus

        def one_frame(p, im, a, b):
            return analyse_vessel(p, a, b, params=params, surrogate=surrogate,
                                  mm_per_px=mm_per_px, image=im, lumen=lumen,
                                  lesion_reader=reader, wire=prod.wire,
                                  lesion_min_ref_mm=prod.lesion_min_ref_mm,
                                  lesion_gate=gate)  # fmt: skip

        cons = run_consensus(run_frames, run_t, start, end, segmenter.predict, one_frame,
                             k=consensus or CONSENSUS_K, reference_analysis=result)  # fmt: skip
        result, img = cons.chosen.analysis, run.frames[cons.chosen.frame]
        q = result.qca
        where = f"Frame {cons.chosen.frame}: " if len(cons.used) > 1 else ""
    typer.echo(f"{DISCLAIMER}")
    if cons is not None:
        typer.echo(cons.note())
    typer.echo(
        f"{where}Path length {q.s_mm[-1]:.1f} mm at {q.mm_per_px:g} mm/px "
        f"({'assumed' if q.calibration_assumed else 'given'}); MLD {q.mld_mm:.2f} mm; "
        f"max DS {q.max_percent_stenosis:.0f}%; lesions {len(q.lesions)}; lumen: "
        f"{q.lumen_method}"
    )
    if result.gate is not None:
        typer.echo(
            f"{where}Lesion detector: {result.gate[0]} of {result.gate[1]} narrowings confirmed "
            "(the rest read as healthy; D72)"
        )
    typer.echo(
        f"{where}Distal FFR: solver {result.ffr_distal_solver:.3f}, surrogate "
        f"{result.ffr_distal_surrogate:.3f}"
    )
    from vstent.uncertainty.greyzone import assessment, load_uncertainty, plan_assessments
    from vstent.uncertainty.inputs import SOURCE_LABELS, InputUncertainty, Measurement

    ucfg, qca_error = load_uncertainty(uncertainty)
    mc = InputUncertainty(
        q.profile,
        params,
        result.solver.pa,
        result.solver.pv,
        measurement=Measurement(q.mm_per_px, q.calibration_assumed),
        config=ucfg.inputs,
        qca_error=qca_error,
        v_hyp_range=fv.v_hyp_interval if fv and fv.accepted and flow == "measured" else None,
    )
    pre = assessment(
        ucfg.pre_pci_threshold,
        result.ffr_distal_solver,
        result.ffr_distal_surrogate,
        result.model_interval,
        mc.pre_samples,
        ucfg.level,
    )
    typer.echo(f"{pre.level_text} intervals: {_interval_text(pre)}")
    typer.echo(
        f"  inputs varied: {', '.join(SOURCE_LABELS[s] for s in mc.sources)} "
        f"({mc.n_samples} draws); model = surrogate-vs-solver error only"
        + ("" if result.model_interval else " (no calibrated ensemble: no model interval)")
    )
    for reason in pre.reasons("Pre-PCI distal FFR"):
        typer.echo(f"Grey zone: {reason}")
    for w in result.warnings:
        typer.echo(f"Warning: {w}")
    if fv is not None:
        _echo_flow_ffr(fv, q.profile, params, flow == "measured" and fv.accepted, fixed_v)
    if plan:
        from vstent.planning.optimizer import PlannerConfig, StentCatalog, plan_stents

        planned = plan_stents(
            q.profile,
            catalog=load_config(catalog, StentCatalog),
            config=load_config(planner, PlannerConfig),
            params=params,
            runner=surrogate,
        )
        _echo_plans(planned, plan_assessments(planned, mc, ucfg.level))
        typer.echo(f"  {SINGLE_VIEW_LENGTHS} (two views: --view2).")
    surrogate.close()
    if figure is not None:
        from vstent.inference.figures import plot_analysis

        plot_analysis(result, img, figure, title=f"Vessel analysis: {image.name}")
        typer.echo(f"Wrote {figure}")


@app.command()
def qca_gallery(
    n_images: Annotated[int, typer.Option("--n", min=1, help="Number of images.")] = 20,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="ARCADE folder.")] = Path(
        "data/raw/arcade"
    ),
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    surrogate_model: Annotated[
        Path | None,
        typer.Option("--surrogate-model", help="Default: newest in models/surrogate/."),
    ] = None,
    physics: Annotated[
        Path, typer.Option("--physics", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    tree: Annotated[
        bool, typer.Option("--tree", help="Tree physics → reports/qca_tree_gallery.md (P2).")
    ] = False,
) -> None:
    """Qualitative QCA gallery on ARCADE stenosis/val images → reports/qca_gallery.md."""
    from vstent.config import load_config
    from vstent.inference.gallery import qca_gallery as run_gallery
    from vstent.inference.gallery import tree_gallery
    from vstent.inference.onnx_runtime import ONNX_FILE, SEG_ONNX_FILE
    from vstent.physics.solver import PhysicsParams

    if tree:
        report = tree_gallery(
            raw_dir,
            seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE),
            params=load_config(physics, PhysicsParams),
            n_images=n_images,
            log=typer.echo,
        )
        s = report["summary"]
        typer.echo(
            f"Analysed {s['n_analysed']}/{s['n_images']}; side branches in "
            f"{s['n_with_side_branches']}; implausibly low main FFR: {s['n_extreme_single']} as "
            f"single paths, {s['n_extreme_tree']} as trees"
        )
        typer.echo("Wrote reports/qca_tree_gallery.md and reports/figures/qca_tree_examples/")
        return
    report = run_gallery(
        raw_dir,
        seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE),
        surrogate_model or _latest_model_dir(Path("models/surrogate"), ONNX_FILE),
        params=load_config(physics, PhysicsParams),
        n_images=n_images,
        log=typer.echo,
    )
    s = report["summary"]
    typer.echo(
        f"Analysed {s['n_analysed']}/{s['n_images']}; path crosses a stenosis box in "
        f"{s['n_path_crosses_box']}; max-DS point inside a box in "
        f"{s['n_max_ds_in_box_given_crossing']}, MLD in {s['n_mld_in_box_given_crossing']}"
    )
    typer.echo("Wrote reports/qca_gallery.md and reports/figures/qca_examples/")


@app.command()
def plan_benchmark(
    surrogate_model: Annotated[
        Path | None,
        typer.Option("--surrogate-model", help="Default: newest in models/surrogate/."),
    ] = None,
    n_vessels: Annotated[int, typer.Option("--n", min=1, help="Synthetic test vessels.")] = 100,
    planner: Annotated[
        Path, typer.Option("--planner", exists=True, dir_okay=False, help="Planner YAML.")
    ] = PLANNER_CONFIG,
    catalog: Annotated[
        Path, typer.Option("--catalog", exists=True, dir_okay=False, help="Stent catalogue YAML.")
    ] = CATALOG_CONFIG,
    physics: Annotated[
        Path, typer.Option("--physics", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write planner_benchmark.md.")
    ] = Path("reports"),
    tree: Annotated[
        bool, typer.Option("--tree", help="Tree planner → reports/tree_planner_benchmark.md.")
    ] = False,
    tree_data: Annotated[Path, typer.Option("--tree-data")] = Path("data/synth/tree_v1"),
) -> None:
    """Planner speed (5,000 candidates) and behaviour on synthetic test vessels (or trees)."""
    from vstent.config import load_config
    from vstent.inference.onnx_runtime import ONNX_FILE
    from vstent.physics.solver import PhysicsParams
    from vstent.planning.optimizer import PlannerConfig, StentCatalog
    from vstent.planning.report import run_planner_benchmark

    if tree:
        from vstent.inference.models import latest_model_dir
        from vstent.inference.tree_runtime import TREE_MODELS, TREE_ONNX_FILE
        from vstent.planning.tree_report import run_tree_planner_benchmark

        report = run_tree_planner_benchmark(
            tree_data,
            report_dir,
            catalog=load_config(catalog, StentCatalog),
            config=load_config(planner, PlannerConfig),
            params=load_config(physics, PhysicsParams),
            tree_model_dir=latest_model_dir(TREE_MODELS, TREE_ONNX_FILE),
            n_trees=n_vessels,
        )
        t = report["throughput"]["solver"]
        typer.echo(
            f"[{'PASS' if report['target_met'] else 'FAIL'}] 5,000 candidates, tree solver: "
            f"{t['median_s']:.3f} s"
        )
        typer.echo(f"Wrote {report_dir / 'tree_planner_benchmark.md'}")
        if not report["target_met"]:
            raise typer.Exit(code=1)
        return
    report = run_planner_benchmark(
        surrogate_model or _latest_model_dir(Path("models/surrogate"), ONNX_FILE),
        report_dir,
        catalog=load_config(catalog, StentCatalog),
        config=load_config(planner, PlannerConfig),
        params=load_config(physics, PhysicsParams),
        n_vessels=n_vessels,
        log=typer.echo,
    )
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Wrote {report_dir / 'planner_benchmark.md'}")
    if not report["targets_met"]:
        raise typer.Exit(code=1)


@app.command()
def uncertainty_report(
    model: Annotated[
        Path | None,
        typer.Option("--model", help="Ensemble model dir (default: newest calibrated one)."),
    ] = None,
    single_model: Annotated[
        Path | None,
        typer.Option("--single-model", help="Single model to compare (default: the M3 model)."),
    ] = None,
    n_vessels: Annotated[int, typer.Option("--n", min=1, help="Test vessels, input MC.")] = 500,
    n_plans: Annotated[int, typer.Option("--n-plans", min=1, help="Vessels to plan.")] = 50,
    uncertainty: Annotated[
        Path, typer.Option("--uncertainty", exists=True, dir_okay=False, help="Uncertainty YAML.")
    ] = UNCERTAINTY_CONFIG,
    physics: Annotated[
        Path, typer.Option("--physics", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    planner: Annotated[
        Path, typer.Option("--planner", exists=True, dir_okay=False, help="Planner YAML.")
    ] = PLANNER_CONFIG,
    catalog: Annotated[
        Path, typer.Option("--catalog", exists=True, dir_okay=False, help="Stent catalogue YAML.")
    ] = CATALOG_CONFIG,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write uncertainty.md.")
    ] = Path("reports"),
    device: Annotated[str, typer.Option("--device", help="auto, cpu or cuda.")] = "auto",
) -> None:
    """Ensemble, conformal coverage and input-uncertainty report → reports/uncertainty.md."""
    import json

    from vstent.config import load_config
    from vstent.physics.solver import PhysicsParams
    from vstent.planning.optimizer import PlannerConfig, StentCatalog
    from vstent.uncertainty.conformal import CONFORMAL_FILE
    from vstent.uncertainty.greyzone import UncertaintyConfig
    from vstent.uncertainty.report import write_uncertainty_report

    model = model or _latest_model_dir(Path("models/surrogate"), CONFORMAL_FILE)
    if single_model is None:
        run = json.loads((report_dir / "surrogate_eval.json").read_text(encoding="utf-8"))["run"]
        single_model = Path("models/surrogate") / Path(run).name
    report = write_uncertainty_report(
        model,
        single_model,
        report_dir,
        config=load_config(uncertainty, UncertaintyConfig),
        params=load_config(physics, PhysicsParams),
        planner=load_config(planner, PlannerConfig),
        catalog=load_config(catalog, StentCatalog),
        n_vessels=n_vessels,
        n_plans=n_plans,
        device=device,
        log=typer.echo,
    )
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Wrote {report_dir / 'uncertainty.md'}")
    if not report["targets_met"]:
        raise typer.Exit(code=1)


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", help="Output folder.")] = Path("reports/demo"),
) -> None:
    """Run a synthetic case end-to-end (profile → FFR → optimise → report) into reports/demo/."""
    from vstent.reporting.demo import run_demo

    typer.echo(DISCLAIMER)
    run_demo(out, log=typer.echo)
    typer.echo(f"Wrote {out / 'demo.md'}, {out / 'demo.json'} and {out / 'demo.png'}")


@app.command()
def qca_synthetic(
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write qca_synthetic.md.")
    ] = Path("reports"),
) -> None:
    """QCA accuracy on rendered synthetic tubes with a known 60% stenosis."""
    from vstent.imaging.qca_validation import write_qca_validation

    report = write_qca_validation(report_dir)
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Wrote {report_dir / 'qca_synthetic.md'}")
    if not report["targets_met"]:
        raise typer.Exit(code=1)


@app.command()
def lumen_synthetic(
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write lumen_synthetic.md.")
    ] = Path("reports"),
) -> None:
    """Lumen accuracy on rendered vessels with known lumens: intensity edges vs the
    imaging-chain model, and their FFR (P4c)."""
    from vstent.imaging.lumen_report import write_lumen_report

    report = write_lumen_report(report_dir)
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Wrote {report_dir / 'lumen_synthetic.md'}")
    if not report["targets_met"]:
        raise typer.Exit(code=1)


@app.command()
def two_view_synthetic(
    config: Annotated[Path, typer.Option("--config", help="Synthetic two-view cases.")] = Path(
        "configs/two_view_synthetic.yaml"
    ),
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write two_view_synthetic.md.")
    ] = Path("reports"),
    workers: Annotated[int, typer.Option("--workers", help="Parallel processes.")] = 1,
) -> None:
    """Two-view 3D reconstruction on synthetic angiograms with known geometry: 3D centreline,
    length, diameter and FFR against the truth and against single-view QCA (P5)."""
    from vstent.imaging.twoview_report import write_two_view_report

    typer.echo(DISCLAIMER)
    report = write_two_view_report(report_dir, config, workers)
    for t in report["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Wrote {report_dir / 'two_view_synthetic.md'}")
    if not report["targets_met"]:
        raise typer.Exit(code=1)


@app.command()
def cine_synthetic(
    config: Annotated[Path, typer.Option("--config", help="Synthetic cine cases.")] = Path(
        "configs/cine_synthetic.yaml"
    ),
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write cine_synthetic.md.")
    ] = Path("reports"),
    workers: Annotated[int, typer.Option("--workers", help="Parallel processes.")] = 1,
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    calibrate: Annotated[
        bool, typer.Option("--calibrate", help="Velocity intervals on the development seeds.")
    ] = False,
) -> None:
    """End-diastolic frames and flow velocity on synthetic beating-heart runs against the
    truth (P7) → reports/cine_synthetic.md; --calibrate → reports/cine_calibration.md."""
    from vstent.config import git_info, load_config
    from vstent.imaging import cine_report as cr
    from vstent.imaging.synthetic_cine import CineSynthConfig
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE

    typer.echo(DISCLAIMER)
    git = git_info()  # before anything is written
    cfg = load_config(config, CineSynthConfig)
    seg_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    if calibrate:
        rows = cr.run_cases(cfg, cr.CALIBRATION_SEEDS, seg_dir, workers)
        c = cr.write_calibration(rows, cr.CALIBRATION_SEEDS, git, seg_dir, report_dir)
        typer.echo(f"Velocity half-widths ({c['level']:.0%}): one run {c['per_run']:.3f} "
                   f"(n = {c['n_runs']}), mean of two runs {c['two_runs']:.3f} "
                   f"(n = {c['n_cases']}); in use {c['in_use']}")  # fmt: skip
        typer.echo(f"Wrote {report_dir / 'cine_calibration.md'}")
        return
    seeds = range(cfg.seed, cfg.seed + cfg.n_cases)
    rows = cr.run_cases(cfg, seeds, seg_dir, workers)
    s = cr.write_report(rows, cfg, seeds, git, seg_dir, report_dir)
    for t in s["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']}")
    typer.echo(f"Wrote {report_dir / 'cine_synthetic.md'}")
    if not all(t["passed"] for t in s["targets"]):
        raise typer.Exit(code=1)


@app.command()
def artery_velocity(
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """Hyperemic velocity by artery from DEFINE-FLOW's Doppler data (P4k-C4)
    → reports/artery_velocity.md."""
    import json

    from vstent.config import git_info
    from vstent.validation.physiology_real import analyse
    from vstent.validation.physiology_real import artery_velocity as av

    typer.echo(DISCLAIMER)
    git = git_info()
    res = av(analyse()["lesions"])
    lines = ["# P4k-C4: hyperemic velocity by artery (DEFINE-FLOW)", "",
             "> **Research use only — not for clinical decisions.**", "",
             "Generated by `vstent artery-velocity`. Do not edit by hand. Pre-registered in "
             "docs/PLAN.md, P4k-C4: the stenosis-free hyperemic mean velocity per lesion is "
             f"k × APV at peak flow ÷ FFR (k = {res['k']}, as P4e). An artery's own velocity is "
             "used if its median differs from the pooled median by ≥ 10%.", "",
             "| Artery | Lesions | Median (m/s) | Change from pooled |", "|---|---:|---:|---:|",
             f"| Pooled (LAD, LCx, RCA) | {res['n_pooled']} | {res['pooled']:.3f} "
             "| – |"]  # fmt: skip
    for g, x in res["arteries"].items():
        name = "Left (LAD + LCx)" if g == "left" else g
        lines.append(f"| {name} | {x['n']} | {x['median']:.3f} | {100 * x['change']:+.1f}% |")
    rule = "per-artery velocities" if res["use_per_artery"] else "one velocity (0.35 m/s) stays"
    lines += ["", f"**Rule:** {rule}.", "",
              f"Commit `{git['git_commit']}` (dirty: {git['git_dirty']})."]  # fmt: skip
    out = Path("data/derived/p4k")
    out.mkdir(parents=True, exist_ok=True)
    (out / "artery_velocity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "artery_velocity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {report_dir / 'artery_velocity.md'}")


@app.command()
def physiology_real(
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="DEFINE-FLOW folder.")] = Path(
        "data/raw/define_flow"
    ),
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write physiology_real.md.")
    ] = Path("reports"),
) -> None:
    """The model's physiology assumptions (phasic flow, hyperemic velocity, FAVOR conversion,
    pressures) against DEFINE-FLOW's invasive measurements → reports/physiology_real.md (P4e)."""
    import json

    from vstent.config import git_info
    from vstent.data.physiology import SOURCE_FILE, fetch_physiology
    from vstent.validation import physiology_real as pr

    typer.echo(DISCLAIMER)
    git = git_info()  # before anything is written
    if not (raw_dir / SOURCE_FILE).exists():
        fetch_physiology("define_flow", raw_dir, log=typer.echo)
    source = json.loads((raw_dir / SOURCE_FILE).read_text(encoding="utf-8"))
    result = pr.analyse(raw_dir)
    pr.write_report(result, git, report_dir, source)
    rules = result["rules"]
    typer.echo(f"{result['n_lesions']} lesions; median resting diastolic / systolic ratio "
               f"{rules['phasic_median_ratio']:.2f}; prior replaced: "
               f"{rules['phasic_prior_replaced']}; fixed velocity to revisit: "
               f"{rules['fixed_velocity_revisit']}")  # fmt: skip
    typer.echo(f"Wrote {report_dir / 'physiology_real.md'}")


@app.command()
def repeatability(
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CADICA folder.")] = Path(
        "data/raw/cadica"
    ),
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Processes for the measurements.")
    ] = 1,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write repeatability.md.")
    ] = Path("reports"),
    rows_out: Annotated[
        Path | None, typer.Option("--rows-out", help="Also save every box's row (JSON).")
    ] = None,
    report: Annotated[bool, typer.Option("--report/--no-report", help="Write the report.")] = True,
) -> None:
    """Frame-to-frame and operator-to-operator repeatability of %DS and FFR on CADICA's real
    angiograms → reports/repeatability.md (P4f)."""
    from vstent.config import git_info, load_config
    from vstent.data.cadica import SOURCE_FILE, fetch_cadica, load_lesion_boxes
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
    from vstent.physics.solver import PhysicsParams
    from vstent.validation import repeatability as rp
    from vstent.validation.stenosis_grading import lesion_tracks

    typer.echo(DISCLAIMER)
    git = git_info()  # before anything is written
    if not (raw_dir / SOURCE_FILE).exists():
        fetch_cadica(raw_dir, log=typer.echo)
    seg_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    seg = SegmentationRunner.from_dir(seg_dir)
    tracks = lesion_tracks(load_lesion_boxes(raw_dir))

    def progress(k, n):
        if k % 500 == 0 or k == n:
            typer.echo(f"  {k} / {n} frames")

    params = load_config(Path("configs/physics.yaml"), PhysicsParams)
    rows = rp.measure(tracks, seg, params, workers=workers, progress=progress)
    if rows_out is not None:
        import json

        from vstent.validation.stenosis_grading import patient_split

        split = patient_split([t.patient for t in tracks])
        rows_out.parent.mkdir(parents=True, exist_ok=True)
        rows_out.write_text(json.dumps({"rows": [{**r, "split": split[r["patient"]]}
                                                 for r in rows], "git": git,
                                        "seg_model": Path(seg_dir).as_posix()}),
                            encoding="utf-8")  # fmt: skip
        typer.echo(f"Wrote {rows_out}")
    if not report:
        return
    s = rp.summarise(rows)
    rp.write_report(s, git, Path(seg_dir).as_posix(), report_dir)
    for t in s["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']} "
                   f"(target {t['target']})")  # fmt: skip
    typer.echo(f"Wrote {report_dir / 'repeatability.md'}")


@app.command()
def physics_check(
    stage: Annotated[str, typer.Option("--stage", help="dev, then test (after the decision).")],
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CADICA folder.")] = Path(
        "data/raw/cadica"
    ),
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Processes for the graded lesions.")
    ] = 1,
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
    part: Annotated[
        str, typer.Option("--part", help="c (Murray outflow), c2 (wire) or c3 (with a reader).")
    ] = "c",
    reader: Annotated[
        Path | None, typer.Option("--reader", help="c3: an exported lesion reader directory.")
    ] = None,
    from_saved: Annotated[
        bool, typer.Option("--from-saved", help="Re-write the report from the saved rows.")
    ] = False,
    min_ref_mm: Annotated[
        float | None,
        typer.Option("--min-ref-mm", help="c3: the learned depth only in vessels ≥ this (P4n)."),
    ] = None,
    gate: Annotated[
        bool, typer.Option("--gate", help="c3: also the product's lesion gate (D72; post hoc).")
    ] = False,
) -> None:
    """Side-branch outflow (Murray, P4k-C), the FFR at the wire position (P4k-C2), and both
    with a learned lesion reader (P4k-C3) vs the product on CADICA, against De Bruyne 2001
    and FAME → reports/p4k_physics[_dev].md, p4k_wire[_dev].md or
    p4k_reader_<reader>[_ref<mm>][_gated][_dev].md."""
    import json

    from vstent.config import git_info, load_config
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
    from vstent.physics.solver import PhysicsParams
    from vstent.validation import physics_check as pc

    if stage not in ("dev", "test") or part not in ("c", "c2", "c3"):
        raise typer.BadParameter("--stage is dev or test; --part is c, c2 or c3")
    if (part == "c3") != (reader is not None):
        raise typer.BadParameter("--reader goes with --part c3")
    stem = {"c": "physics", "c2": "wire"}.get(part) or f"reader_{Path(reader).name}"
    if min_ref_mm is not None:
        if part != "c3":
            raise typer.BadParameter("--min-ref-mm goes with --part c3")
        stem += f"_ref{min_ref_mm:g}"
    gate_spec, ungated = None, stem
    if gate:  # D72, described after the decision: separate files, no rule applied
        from vstent.inference.product import product_config

        cfg = product_config()
        if part != "c3" or not cfg.lesion_detector:
            raise typer.BadParameter("--gate goes with --part c3 and a lesion_detector in "
                                     "configs/product.yaml")  # fmt: skip
        gate_spec, stem = (tuple(cfg.lesion_detector), cfg.detector_threshold), stem + "_gated"
    dev_file = pc.OUT_DIR / f"{stem}_dev.json"
    if stage == "test" and not dev_file.exists():
        raise typer.BadParameter("run --stage dev first and commit its decision")
    typer.echo(DISCLAIMER)
    git = git_info()  # before anything is written
    seg_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    seg = SegmentationRunner.from_dir(seg_dir)
    params = load_config(Path("configs/physics.yaml"), PhysicsParams)

    def progress(what):
        def show(k, n):
            if k % 50 == 0 or k == n:
                typer.echo(f"  {what}: {k} / {n}")

        return show

    rd = None if reader is None else str(reader)
    saved = pc.OUT_DIR / f"{stem}_{stage}.json"
    if from_saved:  # the same rows, the report written again (no new measurement)
        rec = json.loads(saved.read_text(encoding="utf-8"))
        free, graded = rec["lesion_free"], rec["graded"]
    else:
        free = pc.lesion_free_rows(stage, seg, params, raw_dir, progress("lesion-free runs"),
                                   reader_dir=rd, min_ref_mm=min_ref_mm,
                                   gate=gate_spec)  # fmt: skip
        graded = pc.graded_rows(stage, seg, params, raw_dir, workers=workers,
                                progress=progress("frames with graded lesions"),
                                reader_dir=rd, min_ref_mm=min_ref_mm, gate=gate_spec)  # fmt: skip
    summary = pc.summarise(free, graded)
    summary["git"] = git
    if not from_saved:
        typer.echo(f"Wrote {pc.save(stage, free, graded, summary, name=stem)}")
    title = {
        "c": "P4k-C: side-branch outflow in the physics",
        "c2": "P4k-C2: the FFR where a pressure wire would sit",
        "c3": f"P4k-C3: learned lesion reading `{stem[7:]}` with each physics arm",
    }[part] + (" and the lesion gate (D72, post hoc)" if gate else "")
    lines = [f"# {title}", "",
             "> **Research use only — not for clinical decisions.**", "",
             "Generated by `vstent physics-check`. Do not edit by hand. Pre-registered in "
             f"docs/PLAN.md, P4k-{part.upper()}. No invasive FFR: the anchors are De Bruyne 2001 "
             "(non-stenotic arteries, FFR 0.89 ± 0.08) and FAME (FFR ≤ 0.80 by visual grade). "
             "CADICA has no pixel size: 0.30 mm assumed, as in the product.", ""]  # fmt: skip
    if gate:
        before = pc.OUT_DIR / f"{ungated}_{stage}.json"
        lines += ["**Post hoc** (after D72): the same check with the product's lesion gate, to "
                  "describe what it changes on CADICA's expert-graded lesions. No rule is applied "
                  "and no decision is taken from it.", ""]  # fmt: skip
        if before.exists():
            prior = json.loads(before.read_text(encoding="utf-8"))["summary"]
            lines += [f"Without the gate (`{ungated}`, saved rows):", ""]
            lines += pc.section(stage, prior, part)
            lines += ["With the gate:", ""]
    elif stage == "test":
        dev_summary = json.loads(dev_file.read_text(encoding="utf-8"))["summary"]
        lines += pc.section("dev", dev_summary, part)
    lines += pc.section(stage, summary, part)
    if part == "c3" and not gate:  # every arm with the reader against the product's (C)
        product_file = pc.OUT_DIR / f"physics_{stage}.json"
        product = json.loads(product_file.read_text(encoding="utf-8"))["summary"]
        rules = pc.c3_rules(product["outflow"]["none"], summary)
        summary["c3"] = rules
        pc.save(stage, free, graded, summary, name=stem)
        lines += [f"**Against the product** (`reports/p4k_physics{'_dev' if stage == 'dev' else ''}"
                  ".md`, P4k-C3 rule):", ""]  # fmt: skip
        for arm, r in rules.items():
            score = "–" if r["score"] is None else f"{r['score']:.2f}"
            lines.append(f"- {pc.ARM_LABELS[arm]}: {r['reason']}; combined distance {score} → "
                         f"{'qualifies' if r['qualifies'] else 'does not qualify'}.")  # fmt: skip
        lines.append("")
    if from_saved:
        measured = rec.get("summary", {}).get("git", {}).get("git_commit", "an earlier commit")
        lines.append(f"Re-written from the saved rows (`--from-saved`); measured at {measured}.")
    lines.append(f"Segmentation model `{Path(seg_dir).as_posix()}`. Commit "
                 f"`{git['git_commit']}` (dirty: {git['git_dirty']}).")  # fmt: skip
    name = f"p4k_{stem}_dev.md" if stage == "dev" else f"p4k_{stem}.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if part != "c3":
        key = "adopt" if part == "c" else "wire_adopt"
        typer.echo(f"Decision rule: {summary[key.replace('adopt', 'reason')]} → "
                   f"adopt: {summary[key]}")  # fmt: skip
    typer.echo(f"Wrote {report_dir / name}")


@app.command()
def ed_data(
    split: Annotated[str, typer.Option("--split", help="train, val or test.")],
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Synthetic runs for the learned ED-frame model (P4l, M2) → data/derived/ed/<split>/
    (resumable)."""
    from vstent.imaging.ed_data import SPLITS, generate
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE

    if split not in SPLITS:
        raise typer.BadParameter(f"--split is one of {', '.join(SPLITS)}")
    typer.echo(DISCLAIMER)
    seg_dir = str(_latest_model_dir(Path("models/seg"), SEG_ONNX_FILE))

    def progress(k, n):
        if k % 10 == 0 or k == n:
            typer.echo(f"  {split}: {k} / {n} cases")

    n = generate(split, seg_dir, workers=workers, progress=progress)
    typer.echo(f"Wrote {n} runs to data/derived/ed/{split}")


@app.command()
def train_ed(
    config: Annotated[Path | None, typer.Option("--config", help="configs/ed.yaml")] = None,
    resume: Annotated[str | None, typer.Option("--resume", help="Run name to continue.")] = None,
) -> None:
    """Train the learned ED-frame model (P4l, M2) on synthetic runs; stoppable, resumable."""
    from vstent.config import load_config
    from vstent.imaging.ed_model import EdTrainConfig, resume_ed, train_ed

    if (config is None) == (resume is None):
        raise typer.BadParameter("give --config or --resume")
    typer.echo(DISCLAIMER)
    run_dir = (resume_ed(resume, log=typer.echo) if resume
               else train_ed(load_config(config, EdTrainConfig), log=typer.echo))  # fmt: skip
    typer.echo(f"Run directory: {run_dir}")


@app.command()
def target_path(
    stage: Annotated[str, typer.Option("--stage", help="dev, then test (once).")],
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CADICA folder.")] = Path(
        "data/raw/cadica"
    ),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """The lesion-seeking automatic path vs the main trunk on CADICA's lesion runs (P4k-B):
    --stage dev applies the pre-registered rule → reports/p4k_target_path_dev.md; --stage test
    (after the dev decision is committed) → reports/p4k_target_path.md."""
    import json
    import subprocess

    from vstent.config import git_info
    from vstent.data.cadica import load_selected_runs
    from vstent.imaging.run_analysis import RUN_LEVEL
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
    from vstent.inference.product import product_config
    from vstent.validation import cine_frames as cf
    from vstent.validation import target_path as tp

    if stage not in ("dev", "test"):
        raise typer.BadParameter("--stage is dev or test")
    git = git_info()  # before anything is written
    typer.echo(DISCLAIMER)
    dev_json = report_dir / "p4k_target_path_dev.json"
    if stage == "test":
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(dev_json)],
                                 capture_output=True).returncode == 0  # fmt: skip
        if not tracked or git["git_dirty"]:
            typer.echo("The development decision must be committed (and the tree clean) "
                       "before the test stage.", err=True)  # fmt: skip
            raise typer.Exit(code=1)
    seg_dir = _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    seg = SegmentationRunner.from_dir(seg_dir)
    stats = cf.run_stats(load_selected_runs(raw_dir), seg.predict,
                         cf.AREAS_DIR / f"stats_{seg_dir.name}_{RUN_LEVEL:g}.json",
                         level=RUN_LEVEL)  # fmt: skip
    items = tp.run_items(stage, stats, raw_dir)
    cfg = product_config()
    typer.echo(f"{len(items)} lesion runs ({stage}) …")

    def progress(k, n):
        if k % 10 == 0 or k == n:
            typer.echo(f"  {k} / {n} runs")

    rows = tp.measure_items(items, seg, reader_dir=cfg.lesion_reader,
                            min_ref_mm=cfg.lesion_min_ref_mm, workers=workers,
                            progress=progress)  # fmt: skip
    s = tp.summarise(rows)
    product = cfg.model_dump()
    if stage == "dev":
        sections = tp.section("Development (CADICA development patients)", s, rule=True)
        name, data = "p4k_target_path_dev.md", {"dev": s, "rows": rows}
    else:
        d = json.loads(dev_json.read_text(encoding="utf-8"))["dev"]
        sections = [*tp.section("Development (CADICA development patients)", d, rule=True), "",
                    *tp.section("Test (CADICA test patients, once)", s, rule=False)]  # fmt: skip
        name, data = "p4k_target_path.md", {"dev": d, "test": s, "rows": rows}
    path = tp.write_report(report_dir / name, sections, product=product,
                           seg_model=seg_dir.as_posix(), git=git)  # fmt: skip
    path.with_suffix(".json").write_text(json.dumps({**data, "git": git}, indent=1, default=str),
                                         encoding="utf-8")  # fmt: skip
    if stage == "dev":
        typer.echo(f"Gain {100 * s['gain']:+.1f} points → adopt: {s['adopt']}")
    typer.echo(f"Wrote {path}")


@app.command()
def cardiosyntax_check(
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CardioSYNTAX folder.")] = Path(
        "data/raw/cardiosyntax"
    ),
    ed_model: Annotated[
        Path | None, typer.Option("--ed-model", help="Learned ED model run (M2b).")
    ] = Path("runs/ed_m2b_tcn5-20261103-023054"),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """The product on CardioSYNTAX against SYNTAX per side (P4o; external, run once;
    resumable) → reports/p4o_cardiosyntax.md."""
    import json

    from vstent.config import git_info
    from vstent.data.cardiosyntax import load_studies
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE
    from vstent.inference.product import product_config
    from vstent.validation import cardiosyntax_check as cc

    git = git_info()  # before anything is written
    typer.echo(DISCLAIMER)
    studies = load_studies(raw_dir)
    missing = [p for p in cc.PARTS if not (raw_dir / f"{p}.zip").exists()]
    if missing:
        typer.echo(f"Zip parts {missing} missing: `vstent data cardiosyntax --parts 0,9`.",
                   err=True)  # fmt: skip
        raise typer.Exit(code=1)
    seg_dir = _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    ed_dir = str(ed_model) if ed_model and ed_model.exists() else None
    typer.echo(f"{len(cc.tasks(studies))} runs …")

    def progress(k, n):
        if k % 50 == 0 or k == n:
            typer.echo(f"  {k} / {n} runs")

    rows = cc.rows(studies, raw_dir, seg_dir, ed_dir, workers=workers, progress=progress)
    summary = cc.summarise(studies, rows)
    product = product_config().model_dump()
    path = cc.write_report(report_dir / "p4o_cardiosyntax.md", summary,
                           seg_model=seg_dir.as_posix(), ed_model=ed_dir, product=product,
                           git=git)  # fmt: skip
    payload = {"summary": summary, "product": product, "git": git}
    path.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    typer.echo(f"Wrote {path}")


@app.command()
def false_alarms(
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CardioSYNTAX folder.")] = Path(
        "data/raw/cardiosyntax"
    ),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """Where the product's false alarms arise: reading rules on vessel profiles of a sample of
    P4o's sides (P4p; exploratory, development data; resumable) →
    reports/p4p_false_alarms_dev.md."""
    import json

    from vstent.config import git_info
    from vstent.data.cardiosyntax import load_studies
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE
    from vstent.inference.product import product_config
    from vstent.validation import cardiosyntax_check as cc
    from vstent.validation import false_alarms as fa

    git = git_info()  # before anything is written
    typer.echo(DISCLAIMER)
    studies = load_studies(raw_dir)
    seg_dir = _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    task_list = fa.sample_tasks(studies)
    typer.echo(f"{len(task_list)} runs …")

    def progress(k, n):
        if k % 25 == 0 or k == n:
            typer.echo(f"  {k} / {n} runs")

    rows = cc.rows(studies, raw_dir, seg_dir, None, cache_file=fa.PROFILES_FILE, workers=workers,
                   progress=progress, task_list=task_list, detail=True)  # fmt: skip
    summary = fa.summarise(studies, rows)
    product = product_config().model_dump()
    path = fa.write_report(report_dir / "p4p_false_alarms_dev.md", summary, product=product,
                           git=git)  # fmt: skip
    payload = {"summary": summary, "product": product, "git": git}
    path.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    typer.echo(f"Wrote {path}")


@app.command()
def lesion_gate(
    stage: Annotated[str, typer.Option("--stage", help="dev, then test (once, if dev allows).")],
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CardioSYNTAX folder.")] = Path(
        "data/raw/cardiosyntax"
    ),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
    why: Annotated[
        bool, typer.Option("--why", help="dev: why normal sides are still flagged (exploratory).")
    ] = False,
) -> None:
    """A narrowing counts only where the lesion detector agrees (P4q; resumable): variants A and
    B on P4p's sample → reports/p4q_lesion_gate_dev.md; the chosen one against the product on
    CardioSYNTAX part 1 → reports/p4q_lesion_gate.md; ``--why`` → reports/p4q_why_dev.md."""
    import json

    from vstent.config import git_info
    from vstent.data.cardiosyntax import load_studies
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE
    from vstent.validation import cardiosyntax_check as cc
    from vstent.validation import false_alarms as fa
    from vstent.validation import lesion_gate_check as lg

    if stage not in ("dev", "test"):
        raise typer.BadParameter("--stage is dev or test")
    git = git_info()  # before anything is written
    typer.echo(DISCLAIMER)
    studies = load_studies(raw_dir)
    seg_dir = _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)

    def progress(k, n):
        if k % 25 == 0 or k == n:
            typer.echo(f"  {k} / {n} runs")

    names = {"product": "Product", "A": "A: DenseNet121, t = 0.5",
             "B": "B: mean of five, t = 0.05"}  # fmt: skip
    if why:  # exploratory, from the saved development rows only
        if stage != "dev":
            raise typer.BadParameter("--why reads the development rows only")
        saved = json.loads(lg.DEV_ROWS.read_text(encoding="utf-8"))
        p4p = json.loads(fa.PROFILES_FILE.read_text(encoding="utf-8"))
        rows = lg.merge(lg.from_p4p(list(p4p.values())),
                        saved if isinstance(saved, list) else list(saved.values()))  # fmt: skip
        path = report_dir / "p4q_why_dev.md"
        path.write_text("\n".join(lg.why_lines(lg.why(studies, rows), git)) + "\n",
                        encoding="utf-8")  # fmt: skip
        typer.echo(f"Wrote {path}")
        return
    if stage == "dev":
        task_list = fa.sample_tasks(studies)
        typer.echo(f"{len(task_list)} runs …")
        arms = {"A": lg.VARIANTS["A"], "B": lg.VARIANTS["B"]}
        new = cc.rows(studies, raw_dir, seg_dir, None, cache_file=lg.DEV_ROWS, workers=workers,
                      progress=progress, task_list=task_list, row_fn=lg.run_row,
                      row_extra=(arms,))  # fmt: skip
        p4p = json.loads(fa.PROFILES_FILE.read_text(encoding="utf-8"))  # P4p's product rows
        base = lg.from_p4p([p4p[v.path] for _u, _s, v in task_list if v.path in p4p])
        s = lg.summarise(studies, lg.merge(base, new), ["product", "A", "B"], base="product")
        d = lg.dev_decision(s)
        lg.DECISION.write_text(json.dumps({"decision": d, "summary": s, "git": git}, indent=1),
                               encoding="utf-8")  # fmt: skip
        lines = [f"**Decision (P4q, development):** {d['reason']}."]
        path = lg.write_report(
            report_dir / "p4q_lesion_gate_dev.md", s, title="P4q: narrowings confirmed by the "
            "lesion detector (development)", intro="CardioSYNTAX parts 0 and 9, P4p's sample "
            "(seen before). The product's rows are P4p's.", names=names, decision_lines=lines,
            git=git)  # fmt: skip
    else:
        d = lg.load_decision()["decision"]
        if d["chosen"] is None:
            raise typer.BadParameter("development did not allow the test (P4q)")
        dirs, t = d["variant"]
        names["gated"] = names[d["chosen"]]
        arms = {"product": None, "gated": (tuple(dirs), float(t))}
        task_list = cc.tasks(studies, parts=lg.TEST_PARTS)
        typer.echo(f"{len(task_list)} runs …")
        rows = cc.rows(studies, raw_dir, seg_dir, None, cache_file=lg.TEST_ROWS, workers=workers,
                       progress=progress, task_list=task_list, row_fn=lg.run_row,
                       row_extra=(arms,))  # fmt: skip
        s = lg.summarise(studies, rows, ["product", "gated"], base="product")
        td = lg.test_decision(s, "gated")
        ok = ["met" if c else "not met" for c in td["criteria"]]
        lines = [
            f"**Decision (P4q, test):** {'adopt' if td['adopt'] else 'not adopted'}. "
            f"1. False alarms fall by ≥ {lg.MIN_DROP:.0%}: {td['drop']:.1%} ({ok[0]}). "
            f"2. Paired J difference above 0: {td['dj']:+.3f} "
            f"({td['dj_interval'][0]:+.3f} to {td['dj_interval'][1]:+.3f}; {ok[1]})."
        ]
        path = lg.write_report(
            report_dir / "p4q_lesion_gate.md", s, title="P4q: narrowings confirmed by the "
            "lesion detector (test, run once)", intro=f"CardioSYNTAX part 1 (never used), "
            f"variant {d['chosen']} chosen on development.", names=names, decision_lines=lines,
            git=git)  # fmt: skip
        (report_dir / "p4q_lesion_gate.json").write_text(
            json.dumps({"summary": s, "decision": td, "git": git}, indent=1), encoding="utf-8"
        )
    typer.echo(f"Wrote {path}")


@app.command()
def pixel_size(
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CARDIAG folder.")] = Path(
        "data/raw/cardiag"
    ),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """The true pixel size on CARDIAG and a catheter calibration check (P4r; no model run)
    → reports/pixel_size.md and .json."""
    import json

    from vstent.config import git_info
    from vstent.uncertainty.greyzone import load_uncertainty
    from vstent.validation import pixel_size as ps

    git = git_info()
    typer.echo(DISCLAIMER)
    archive = raw_dir / "CARDIAG.zip"
    if not archive.exists():
        raise typer.BadParameter(f"{archive} not found (run `vstent data cardiag`)")
    samples = ps.read_samples(archive)
    s = ps.summarise(samples, input_rel=load_uncertainty()[0].inputs.pixel_size_rel)
    report_dir.mkdir(parents=True, exist_ok=True)
    md = "\n".join(ps.report_lines(s, git)) + "\n"
    (report_dir / "pixel_size.md").write_text(md, encoding="utf-8")
    (report_dir / "pixel_size.json").write_text(json.dumps({"summary": s, "git": git}, indent=1),
                                                encoding="utf-8")  # fmt: skip
    typer.echo(f"Wrote {report_dir / 'pixel_size.md'}")


@app.command()
def angiocad_check(
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="AngioCAD folder.")] = Path(
        "data/raw/angiocad"
    ),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """The product on AngioCAD's 413 patients against the experts' segment grades (P4k-J;
    external, run once; resumable) → reports/p4k_angiocad.md."""
    import json

    from vstent.config import git_info
    from vstent.data.angiocad import load_labels
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE
    from vstent.inference.product import product_config
    from vstent.validation import angiocad_check as ac

    git = git_info()  # before anything is written
    typer.echo(DISCLAIMER)
    labels = load_labels(raw_dir)
    seg_dir = _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    typer.echo(f"{len(labels)} patients, {len(ac.tasks(labels))} series …")

    def progress(k, n):
        if k % 50 == 0 or k == n:
            typer.echo(f"  {k} / {n} series")

    rows = ac.rows(labels, raw_dir, seg_dir, workers=workers, progress=progress)
    summary = ac.summarise(labels, rows)
    product = product_config().model_dump()
    path = ac.write_report(report_dir / "p4k_angiocad.md", summary, seg_model=seg_dir.as_posix(),
                           product=product, git=git)  # fmt: skip
    path.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "product": product, "git": git}, indent=1),
        encoding="utf-8",
    )
    typer.echo(f"Wrote {path}")


@app.command()
def ed_measure_check(
    run: Annotated[Path, typer.Option("--run", help="runs/<ED model run>")],
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """Is beat-to-beat reproducibility a fair stand-in for ED accuracy? On synthetic validation
    runs (P4l), where the true ED frames are known → reports/ed_measure_check.md."""
    import json

    from vstent.config import git_info
    from vstent.imaging.ed_data import SPLITS
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE
    from vstent.validation import ed_eval

    git = git_info()  # before anything is written
    typer.echo(DISCLAIMER)
    seg_dir = _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)

    def progress(k, n):
        if k % 10 == 0 or k == n:
            typer.echo(f"  {k} / {n} runs")

    rows = ed_eval.synthetic_measure(SPLITS["val"], str(seg_dir), str(run), workers=workers,
                                     progress=progress)  # fmt: skip
    s = ed_eval.summarise_measure(rows)
    lines = ["# P4l: the reproducibility measure against the truth (synthetic validation runs)",
             "", f"> **{ed_eval.DISCLAIMER}**", "",
             "Generated by `vstent ed-measure-check`. Do not edit by hand. Beat-to-beat "
             "reproducibility (`vstent ed-report`) of the true ED frames, M0 and the learned "
             "model, with each one's ±1-frame accuracy. A fair measure gives the true frames a "
             "ratio below 1 and ranks the methods as their accuracy does.", "",
             "| | Runs with a value | Mean ratio | Median ratio | Ratio < 1 | ±1 frame |",
             "|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    for name in ("truth", "M0", "M2"):
        x = s[name]
        lines.append(f"| {name} | {x['with_value']} | {ed_eval._f(x['mean_ratio'])} | "
                     f"{ed_eval._f(x['median_ratio'])} | {ed_eval._pct(x['share_below_1'])} | "
                     f"{ed_eval._pct(x['within1'])} |")  # fmt: skip
    lines += ["", f"Runs where M0 and the model differ in accuracy and both have a ratio: "
              f"{s['order_runs']}; the more accurate one has the lower ratio on "
              f"{ed_eval._pct(s['order_agrees'])}.", "",
              f"ED model `{run.as_posix()}`; segmentation model `{seg_dir.as_posix()}`. Commit "
              f"`{git.get('git_commit')}` (dirty: {git.get('git_dirty')})."]  # fmt: skip
    path = report_dir / "ed_measure_check.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"summary": s, "rows": rows, "git": git}
    path.with_suffix(".json").write_text(json.dumps(payload, indent=1, default=str),
                                         encoding="utf-8")  # fmt: skip
    typer.echo(f"Wrote {path}")


@app.command()
def ed_report(
    run: Annotated[Path, typer.Option("--run", help="runs/<ED model run>")],
    stage: Annotated[str, typer.Option("--stage", help="dev, then test.")] = "dev",
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CADICA folder.")] = Path(
        "data/raw/cadica"
    ),
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """Learned ED frames (M2) vs the product's (M0), P4l: synthetic ±1 frame per frame rate;
    for development also beat-to-beat reproducibility on CADICA runs →
    reports/ed_model_dev.md, then reports/ed_model.md (synthetic test runs, once)."""
    import json

    from vstent.config import git_info, load_config
    from vstent.imaging.ed_model import CONFIG_FILENAME, EdTrainConfig
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE
    from vstent.validation import ed_eval

    if stage not in ("dev", "test"):
        raise typer.BadParameter("--stage is dev or test")
    git = git_info()  # before anything is written
    typer.echo(DISCLAIMER)
    cfg = load_config(run / CONFIG_FILENAME, EdTrainConfig)
    synthetic = ed_eval.synthetic_scores(run, "val" if stage == "dev" else "test", cfg.data_dir)
    seg_dir = _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    real = None
    if stage == "dev":
        from vstent.data.cadica import load_lesion_boxes, load_selected_runs
        from vstent.validation.stenosis_grading import lesion_tracks, patient_split

        split = patient_split([t.patient for t in lesion_tracks(load_lesion_boxes(raw_dir))])
        runs = [r for r in load_selected_runs(raw_dir) if split[r.patient] == "dev"]
        typer.echo(f"{len(runs)} CADICA development runs …")

        def progress(k, n):
            if k % 10 == 0 or k == n:
                typer.echo(f"  {k} / {n} runs")

        rows = ed_eval.real_rows(runs, str(seg_dir), str(run),
                                 ed_eval.REAL_DIR / f"{run.name}_dev.json", workers=workers,
                                 progress=progress)  # fmt: skip
        real = ed_eval.summarise_real(rows)
    name = "ed_model_dev.md" if stage == "dev" else "ed_model.md"
    path = ed_eval.write_report(report_dir / name, stage=stage, synthetic=synthetic, real=real,
                                model=run.as_posix(), seg_model=seg_dir.as_posix(),
                                git=git)  # fmt: skip
    path.with_suffix(".json").write_text(
        json.dumps({"synthetic": synthetic, "real": real, "git": git}, indent=1), encoding="utf-8"
    )
    if real is not None:
        typer.echo("Choice: {} ({})".format(*ed_eval.choose(synthetic, real)))
    typer.echo(f"Wrote {path}")


@app.command()
def stent_check(
    stage: Annotated[str, typer.Option("--stage", help="dev, then test.")],
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
    min_ref_mm: Annotated[
        float | None,
        typer.Option(
            "--min-ref-mm", help="The pipeline arm's learned depth only in vessels ≥ this (P4n)."
        ),
    ] = None,
    gate: Annotated[
        bool, typer.Option("--gate", help="The pipeline arm with the lesion gate (D72; post hoc).")
    ] = False,
) -> None:
    """Virtual stenting on CADICA without FFR (P4m): the stent on the experts' lesion, and the
    post-PCI FFR against FFR-SEARCH → reports/stent_check_dev.md, then stent_check.md."""
    import json

    from vstent.config import git_info, load_config
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
    from vstent.physics.solver import PhysicsParams
    from vstent.validation import stent_check as sc

    if stage not in ("dev", "test"):
        raise typer.BadParameter("--stage is dev or test")
    typer.echo(DISCLAIMER)
    git = git_info()
    seg_dir = _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    params = load_config(Path("configs/physics.yaml"), PhysicsParams)

    def progress(k, n):
        if k % 25 == 0 or k == n:
            typer.echo(f"  {k} / {n} lesions")

    rows = sc.measure(stage, SegmentationRunner.from_dir(seg_dir), params, workers=workers,
                      progress=progress, min_ref_mm=min_ref_mm, gate=gate)  # fmt: skip
    tag = ("" if min_ref_mm is None else f"_ref{min_ref_mm:g}") + ("_gated" if gate else "")
    summary = sc.summarise(rows)
    out = Path("data/derived/p4k")
    out.mkdir(parents=True, exist_ok=True)
    record = {"summary": summary, "rows": rows, "git": git}
    (out / f"stent{tag}_{stage}.json").write_text(json.dumps(record), encoding="utf-8")
    lines = ["# P4m: virtual stenting on real angiograms (no FFR)", "",
             "> **Research use only — not for clinical decisions.**", "",
             "Generated by `vstent stent-check`. Do not edit by hand. Pre-registered in "
             "docs/PLAN.md, P4m. CADICA lesions graded 70–98%, measured on the "
             "clinician-style path; the guided app's planner (solver). Pixel size assumed "
             "0.30 mm.", ""]  # fmt: skip
    if gate:
        lines += ["**Post hoc** (after D72): the pipeline arm with the product's lesion gate, "
                  "to describe what it changes on CADICA's 70–98% lesions. No rule is applied "
                  "and no decision is taken from it.", ""]  # fmt: skip
        before = out / f"stent{tag.removesuffix('_gated')}_{stage}.json"
        if before.exists():
            prior = json.loads(before.read_text(encoding="utf-8"))["summary"]
            lines += ["Without the gate (saved rows):", "", *sc.section(stage, prior),
                      "With the gate:", ""]  # fmt: skip
    elif stage == "test" and (out / f"stent{tag}_dev.json").exists():
        dev = json.loads((out / f"stent{tag}_dev.json").read_text(encoding="utf-8"))["summary"]
        lines += sc.section("dev", dev)
    lines += sc.section(stage, summary)
    lines.append(f"Segmentation model `{Path(seg_dir).as_posix()}`. Commit "
                 f"`{git['git_commit']}` (dirty: {git['git_dirty']}).")  # fmt: skip
    name = f"stent_check{tag}_dev.md" if stage == "dev" else f"stent_check{tag}.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {report_dir / name}")


@app.command()
def export_map(
    run: Annotated[Path, typer.Option("--run", help="A vessel_id or lesion_boxes run.")],
) -> None:
    """ONNX export (with parity) of a vessel-identity or lesion-detector run (P4k-E, P4k-G)
    → models/vessel_id/ or models/lesion_det/."""
    from vstent.imaging.map_export import export_map_run

    meta = export_map_run(run)
    typer.echo(f"Exported {meta['task']} ({meta['classes']} classes); parity max |diff| "
               f"{meta['parity']['max_abs_diff']:.2e}")  # fmt: skip


def _map_runners(models: str | None, root: Path) -> dict[str, Any]:
    from vstent.inference.onnx_runtime import MAP_ONNX_FILE, MapRunner

    dirs = ([Path(m) for m in models.split(",") if m.strip()] if models
            else sorted(p.parent for p in root.glob(f"*/{MAP_ONNX_FILE}")))  # fmt: skip
    if not dirs:
        raise typer.BadParameter(f"no exported model in {root} (run `vstent export-map`)")
    return {d.name: MapRunner.from_dir(d) for d in dirs}


@app.command()
def lesion_detection(
    stage: Annotated[str, typer.Option("--stage", help="dev, then test (after the decision).")],
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated model dirs (default: all).")
    ] = None,
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """Whole-image lesion detector vs expert boxes (P4k-E): candidates compared on validation
    → reports/lesion_detection_dev.md; the chosen one on test → reports/lesion_detection.md."""
    import json

    from vstent.config import git_info
    from vstent.data.lesion_boxes import all_items
    from vstent.validation import lesion_detection as ld

    if stage not in ("dev", "test"):
        raise typer.BadParameter("--stage is dev or test")
    typer.echo(DISCLAIMER)
    git = git_info()
    out = Path("data/derived/p4k")
    out.mkdir(parents=True, exist_ok=True)
    runners = _map_runners(models, Path("models/lesion_det"))
    head = ["# P4k-E: whole-image lesion detector", "",
            "> **Research use only — not for clinical decisions.**", "",
            "Generated by `vstent lesion-detection`. Do not edit by hand. Pre-registered in "
            "docs/PLAN.md, P4k-E and amendment 2. A box is found when a detection (a connected "
            "region of probability ≥ t) overlaps it.", ""]  # fmt: skip

    def progress(k, n):
        if k % 200 == 0 or k == n:
            typer.echo(f"  {k} / {n} frames")

    if stage == "dev":
        rows_file = out / "detection_dev_rows.json"  # per-frame counts: re-summarised later
        cached = json.loads(rows_file.read_text(encoding="utf-8")) if rows_file.exists() else {}
        if cached.get("models") == sorted(runners):
            rows = cached["rows"]
        else:
            items = [it for it in all_items() if it.role == "val"]
            preds = {n: (lambda img, r=r: r.predict(img)[0]) for n, r in runners.items()}
            rows = ld.measure_candidates(items, preds, progress)
            rows_file.write_text(json.dumps({"models": sorted(runners), "rows": rows}),
                                 encoding="utf-8")  # fmt: skip
        decisions = {n: ld.dev_decision(r) for n, r in rows.items()}
        chosen = ld.choose(decisions, rows)
        d = decisions[chosen]
        record = {"chosen": chosen, "models": [str(Path("models/lesion_det") / n)
                                               for n in runners], "decision": d,
                  "candidates": {n: {k: v for k, v in x.items() if k != "per_source"}
                                 for n, x in decisions.items()}, "git": git}  # fmt: skip
        (out / "detection_dev.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        lines = head + ["## Validation: every candidate at its 95%-sensitivity point", "",
                        "| Candidate | t | Sensitivity | Detections per lesion-free frame |",
                        "|---|---:|---:|---:|"]  # fmt: skip
        for n, x in decisions.items():
            p = x.get("pooled") or {}
            sens, al = p.get("sensitivity"), p.get("alarms_per_free_frame")
            lines.append(f"| {n} | {x.get('threshold', '–')} | "
                         f"{'–' if sens is None else f'{100 * sens:.1f}%'} | "
                         f"{'–' if al is None else f'{al:.2f}'} |")  # fmt: skip
        lines += ["", "## Validation: every candidate at fixed thresholds", ""]
        lines += ld.candidate_table(rows)
        lines += [f"**Chosen (amendment 2):** {chosen}.", ""]
        if d["threshold_index"] is not None:
            lines += ld.section("Validation", chosen, d["per_source"])
        else:  # amendment 2's fallback point
            k5 = ld.THRESHOLDS.index(0.5)
            lines += ld.section("Validation at t = 0.5", chosen,
                                ld.test_results(rows[chosen], k5))  # fmt: skip
        shown = "shown in the app" if d["shown_in_app"] else "research option"
        lines += [f"**App rule** (≤ {ld.MAX_FREE_ALARMS} detections per lesion-free frame): "
                  f"{d['reason']} → {shown}.",
                  "", f"Commit `{git['git_commit']}` (dirty: {git['git_dirty']})."]  # fmt: skip
        name = "lesion_detection_dev.md"
    else:
        rec = json.loads((out / "detection_dev.json").read_text(encoding="utf-8"))
        k = rec["decision"]["threshold_index"]
        if k is None:
            k = ld.THRESHOLDS.index(0.5)
        chosen = rec["chosen"]
        names = list(runners) if chosen.startswith("mean of") else [chosen]
        sel = {n: runners[n] for n in names}

        def predict(img):
            import numpy as np

            return np.mean([r.predict(img)[0] for r in sel.values()], axis=0)

        items = [it for it in all_items() if it.role == "test"]
        rows = ld.measure_candidates(items, {chosen: predict}, progress)[chosen]
        res = ld.test_results(rows, k)
        (out / "detection_test.json").write_text(json.dumps({"chosen": chosen, "k": k,
                                                             "results": res, "git": git}),
                                                 encoding="utf-8")  # fmt: skip
        lines = head + [f"Chosen on validation: {chosen}, t = {ld.THRESHOLDS[k]}.", ""]
        lines += ld.section("Test", chosen, res)
        lines += ["Targets: sensitivity ≥ 90% per source. CADICA test lesion-free frames are "
                  "fresh; the other test sets were seen once in P4i.", "",
                  f"Commit `{git['git_commit']}` (dirty: {git['git_dirty']})."]  # fmt: skip
        name = "lesion_detection.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {report_dir / name}")


@app.command()
def vessel_id_report(
    stage: Annotated[str, typer.Option("--stage", help="dev, then test (after the decision).")],
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated model dirs (default: all).")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    report_dir: Annotated[Path, typer.Option("--report-dir")] = Path("reports"),
) -> None:
    """Vessel identity (P4k-G): the candidate with the best validation Dice, its ARCADE scores
    and CADICA side agreement → reports/vessel_id_dev.md, then reports/vessel_id.md."""
    import json

    from vstent.config import git_info
    from vstent.data.vessel_id import load_labels
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
    from vstent.validation import vessel_id_eval as ve

    if stage not in ("dev", "test"):
        raise typer.BadParameter("--stage is dev or test")
    typer.echo(DISCLAIMER)
    git = git_info()
    out = Path("data/derived/p4k")
    out.mkdir(parents=True, exist_ok=True)
    runners = _map_runners(models, Path("models/vessel_id"))
    seg = SegmentationRunner.from_dir(_latest_model_dir(Path("models/seg"), SEG_ONNX_FILE))
    if stage == "dev":
        val_dice = {n: json.loads((Path("runs") / n / "summary.json").read_text())["best_val_dice"]
                    for n in runners}  # fmt: skip
        chosen = max(val_dice, key=val_dice.get)
    else:
        chosen = json.loads((out / "vessel_id_dev.json").read_text(encoding="utf-8"))["chosen"]
    r = runners[chosen]
    images, labels = load_labels("data/raw/arcade", "val" if stage == "dev" else "test")
    arcade = ve.arcade_scores(images, labels, r.predict)

    def progress(k, n):
        if k % 25 == 0 or k == n:
            typer.echo(f"  CADICA runs {k} / {n}")

    rows = ve.cadica_sides(stage, seg, r.predict, progress=progress)
    agr = ve.agreement(rows)
    rec = {"chosen": chosen, "arcade": arcade, "cadica": agr, "rows": rows, "git": git}
    if stage == "dev":
        rec["val_dice"] = val_dice
        rec["decision"] = ve.decision(arcade, agr)
    (out / f"vessel_id_{stage}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    lines = ["# P4k-G: vessel identity (RCA, left main, LAD, LCx)", "",
             "> **Research use only — not for clinical decisions.**", "",
             "Generated by `vstent vessel-id-report`. Do not edit by hand. Pre-registered in "
             "docs/PLAN.md, P4k-G and amendment 2.", ""]  # fmt: skip
    if stage == "dev":
        each = "; ".join(f"{n} {v:.3f}" for n, v in val_dice.items())
        lines += [f"Validation Dice (mean of the four vessels) per candidate: {each}. "
                  f"Chosen: **{chosen}**.", ""]  # fmt: skip
    split = "val" if stage == "dev" else "test"
    lines += [f"ARCADE {split} ({arcade['n']} images): vessel-pixel accuracy "
              f"{100 * arcade['accuracy']:.1f}%; Dice " + "; ".join(
                  f"{k} {v:.3f}" for k, v in arcade["dice"].items() if v is not None) + ".", "",
              f"CADICA {'development' if stage == 'dev' else 'test'} patients: side agrees with "
              f"CADICA's label in {agr['agree']} of {agr['n']} runs ({100 * agr['share']:.1f}%, "
              f"95% CI {100 * agr['ci95'][0]:.1f}–{100 * agr['ci95'][1]:.1f}%); left "
              f"{agr['per_side']['left']} / {agr['n_side']['left']}, right "
              f"{agr['per_side']['right']} / {agr['n_side']['right']}.", ""]  # fmt: skip
    if stage == "dev":
        d = rec["decision"]
        lines += [
            f"**Rule:** {d['reason']} → "
            f"{'used in the product' if d['used_in_product'] else 'research option'}.",
            "",
        ]
    lines.append(f"Commit `{git['git_commit']}` (dirty: {git['git_dirty']}).")
    name = "vessel_id_dev.md" if stage == "dev" else "vessel_id.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {report_dir / name}")


@app.command()
def lesion_data(
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Processes for the measurements.")
    ] = 1,
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    kemerovo_dir: Annotated[
        Path, typer.Option("--kemerovo", help="Kemerovo folder (used if present).")
    ] = Path("data/raw/kemerovo"),
    sources: Annotated[
        str, typer.Option("--sources", help="Comma-separated: cadica, arcade, kemerovo.")
    ] = "cadica,arcade,kemerovo",
    version: Annotated[int, typer.Option("--version", min=1, max=2, help="2: P4k-A.")] = 1,
) -> None:
    """Lesion patches from real angiograms (CADICA, ARCADE stenosis, Kemerovo) for the learned
    lesion reading (P4i) → data/derived/lesion/; version 2 (P4k-A: clinician-style paths,
    lesion-free CADICA candidates) → data/derived/lesion_v2/."""
    from vstent.config import git_info
    from vstent.data.lesion_data import build
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner

    typer.echo(DISCLAIMER)
    git = git_info()
    seg_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    counts = build(SegmentationRunner.from_dir(seg_dir), workers=workers,
                   sources=tuple(s.strip() for s in sources.split(",") if s.strip()),
                   kemerovo_dir=kemerovo_dir, git={**git, "seg_model": Path(seg_dir).as_posix()},
                   version=version, log=typer.echo)  # fmt: skip
    typer.echo(f"Samples: {counts}")


@app.command()
def train_lesion(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False, help="Training YAML.")
    ] = Path("configs/lesion.yaml"),
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue this run (name or folder).")
    ] = None,
) -> None:
    """Train the learned lesion reader in 5 patient folds (P4i; stoppable: vstent stop <run>)."""
    from vstent.config import load_config
    from vstent.imaging.lesion_model import LesionTrainConfig, resume_lesion, train_lesion

    typer.echo(DISCLAIMER)
    if resume:
        run_dir = resume_lesion(resume, log=typer.echo)
    else:
        run_dir = train_lesion(load_config(config, LesionTrainConfig), log=typer.echo)
    typer.echo(f"Run folder: {run_dir}")


@app.command()
def export_lesion(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False, help="Run folder.")],
) -> None:
    """Export the lesion reader (mean of the fold models) to ONNX with a parity check."""
    from vstent.imaging.lesion_model import export_lesion as export

    meta = export(run, log=typer.echo)
    typer.echo(f"[{'PASS' if meta['parity_passed'] else 'FAIL'}] parity "
               f"{meta['parity_max_abs_ds']:.1e}")  # fmt: skip
    if not meta["parity_passed"]:
        raise typer.Exit(code=1)


@app.command()
def lesion_report(
    stage: Annotated[str, typer.Option("--stage", help="dev, then test (once).")],
    model: Annotated[
        Path, typer.Option("--model", exists=True, file_okay=False, help="models/lesion/<run>.")
    ],
    report_dir: Annotated[Path, typer.Option("--report-dir", help="Output folder.")] = Path(
        "reports"
    ),
    version: Annotated[int, typer.Option("--version", min=1, max=2, help="2: P4k-A.")] = 1,
) -> None:
    """The learned lesion reading (P4i; version 2: P4k-A): --stage dev applies the
    pre-registered rule on the development data; --stage test (only after the dev decision is
    committed) runs the test."""
    import json
    import subprocess

    from vstent.config import git_info
    from vstent.inference.lesion_runtime import LesionRunner
    from vstent.validation import lesion_reading as lr

    typer.echo(DISCLAIMER)
    git = git_info()
    runner = LesionRunner.from_dir(model)
    run_dir = Path("runs") / model.name
    from vstent.data.lesion_data import OUT_DIR, OUT_DIR_V2

    data_dir = OUT_DIR_V2 if version == 2 else OUT_DIR
    dev_name, name = ((lr.DEV_REPORT, lr.REPORT) if version == 1
                      else ("lesion_reading_v2_dev", "lesion_reading_v2"))  # fmt: skip
    title = "" if version == 1 else ", version 2 (P4k-A)"
    if stage == "dev":
        d = lr.dev_stage(run_dir, runner, data_dir)
        lr.write_report(report_dir, dev_name, f"Learned lesion reading{title}: development "
                        "decision", lr.dev_sections(d), d, git, model.as_posix())  # fmt: skip
        for k, v in d["rule"].items():
            typer.echo(f"[{'PASS' if v else 'FAIL'}] {k}")
        typer.echo(f"Decision: {'adopted' if d['adopted'] else 'not adopted'}. Commit "
                   f"{report_dir / (dev_name + '.md')} before --stage test.")  # fmt: skip
        return
    if stage != "test":
        typer.echo("--stage: dev or test", err=True)
        raise typer.Exit(code=2)
    dev_file = report_dir / f"{dev_name}.json"
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(dev_file)],
                             capture_output=True).returncode == 0  # fmt: skip
    if not tracked or git["git_dirty"]:
        typer.echo("The development decision must be committed (and the tree clean) before the "
                   "test stage.", err=True)  # fmt: skip
        raise typer.Exit(code=1)
    d = json.loads(dev_file.read_text(encoding="utf-8"))
    t = lr.test_stage(runner, data_dir)
    lr.write_report(report_dir, name, f"Learned lesion reading on real angiograms{title}",
                    [*lr.dev_sections(d), "", *lr.test_sections(t)], {"dev": d, "test": t}, git,
                    model.as_posix())  # fmt: skip
    typer.echo(f"Wrote {report_dir / (name + '.md')}")


@app.command()
def vessel_coverage(
    stage: Annotated[str, typer.Option("--stage", help="dev, then test (once).")],
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Processes for the measurements.")
    ] = 1,
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    report_dir: Annotated[Path, typer.Option("--report-dir", help="Output folder.")] = Path(
        "reports"
    ),
) -> None:
    """Does the segmentation reach the experts' lesions? Model vs hybrid (P4j): --stage dev
    chooses the threshold and decides; --stage test (after the dev decision is committed)."""
    import json
    import subprocess

    from vstent.config import git_info
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
    from vstent.validation import vessel_coverage as vc

    typer.echo(DISCLAIMER)
    git = git_info()
    seg_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    seg = SegmentationRunner.from_dir(seg_dir, hybrid_threshold=None)
    model = Path(seg_dir).as_posix()
    if stage == "dev":
        d = vc.run_dev(seg, workers, typer.echo)
        vc.write_report(report_dir, vc.DEV_REPORT, "Vessel coverage: development decision",
                        vc.dev_sections(d), d, git, model)  # fmt: skip
        c = d["choice"]
        typer.echo(f"Threshold {c['threshold']}; {c['reason']}; "
                   f"{'adopted' if c['adopted'] else 'not adopted'}.")  # fmt: skip
        return
    if stage != "test":
        typer.echo("--stage: dev or test", err=True)
        raise typer.Exit(code=2)
    dev_file = report_dir / f"{vc.DEV_REPORT}.json"
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(dev_file)],
                             capture_output=True).returncode == 0  # fmt: skip
    if not tracked or git["git_dirty"]:
        typer.echo("Commit the development decision (clean tree) before the test.", err=True)
        raise typer.Exit(code=1)
    d = json.loads(dev_file.read_text(encoding="utf-8"))
    threshold = d["choice"]["threshold"]
    if threshold is None:
        typer.echo("No threshold was eligible on the development data: nothing to test.")
        return
    t = vc.run_test(seg, threshold, workers, typer.echo)
    vc.write_report(report_dir, vc.REPORT, "Vessel coverage on real angiograms (P4j)",
                    [*vc.dev_sections(d), "", *vc.test_sections(t, threshold)],
                    {"dev": d, "test": t}, git, model)  # fmt: skip
    typer.echo(f"Wrote {report_dir / (vc.REPORT + '.md')}")


@app.command()
def consensus_report(
    rows: Annotated[
        Path, typer.Option("--rows", exists=True, dir_okay=False, help="From --rows-out.")
    ] = Path("data/derived/repeatability/rows.json"),
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write consensus.md.")
    ] = Path("reports"),
) -> None:
    """Split-half repeatability of the K-frame median on CADICA, K chosen on the development
    patients and tested once (P4g) → reports/consensus.md. Needs the rows saved by
    `vstent repeatability --rows-out`."""
    import json

    from vstent.config import git_info
    from vstent.validation import consensus as cs
    from vstent.validation import repeatability as rp

    typer.echo(DISCLAIMER)
    git = git_info()  # before anything is written
    saved = json.loads(rows.read_text(encoding="utf-8"))
    s = cs.summarise(saved["rows"])
    f = rp.summarise(saved["rows"])["frames"]
    s["single_frame"] = {"ds_cv": f["ds_cv"], "ffr_cv": f["ffr_cv"], "n_groups": f["n_groups"]}
    cs.write_report(s, git, saved["seg_model"], saved["git"], report_dir)
    typer.echo(f"Single frame (P4f, from these rows): %DS CV {f['ds_cv']:.1%}, FFR CV "
               f"{f['ffr_cv']:.1%} ({f['n_groups']} lesions).")  # fmt: skip
    for k, r in s["dev"].items():
        typer.echo(f"  dev K = {k}: {r['n_lesions']} lesions, %DS CV {rp._pct(r['ds_cv'])}, "
                   f"FFR CV {rp._pct(r['ffr_cv'])}")  # fmt: skip
    typer.echo(f"Chosen K = {s['chosen_k']}: {s['reason']}.")
    for t in s["targets"]:
        typer.echo(f"[{'PASS' if t['passed'] else 'FAIL'}] {t['metric']}: {t['value']} "
                   f"(target {t['target']})")  # fmt: skip
    typer.echo(f"Wrote {report_dir / 'consensus.md'}")


@app.command()
def mvp_report(
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Folder with the generated reports.")
    ] = Path("reports"),
) -> None:
    """Assemble reports/mvp_report.md from the generated report files."""
    from vstent.reporting.mvp import write_mvp_report

    data = write_mvp_report(report_dir, log=typer.echo)
    typer.echo(f"Wrote {report_dir / 'mvp_report.md'}")
    if not data["all_targets_met"]:
        raise typer.Exit(code=1)


def _optional_model_dir(given: Path | None, root: Path, filename: str) -> Path | None:
    from vstent.inference.models import latest_model_dir

    return given if given is not None else latest_model_dir(root, filename)


@app.command()
def validate(
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", exists=True, dir_okay=False, help="Case list (cases.csv)."),
    ] = None,
    reference: Annotated[
        Path | None,
        typer.Option(
            "--reference",
            exists=True,
            dir_okay=False,
            help="Wire FFR / iFR (reference.csv); omit to stay blinded.",
        ),
    ] = None,
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            exists=True,
            dir_okay=False,
            help="Score locked predictions (predictions.csv) instead of --manifest.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output folder (default runs/validation/<manifest>-<time>)."),
    ] = None,
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    surrogate_model: Annotated[
        Path | None,
        typer.Option("--surrogate-model", help="Default: newest in models/surrogate/."),
    ] = None,
    physics: Annotated[
        Path, typer.Option("--physics", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    uncertainty: Annotated[
        Path, typer.Option("--uncertainty", dir_okay=False, help="Uncertainty YAML.")
    ] = UNCERTAINTY_CONFIG,
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False, help="Validation YAML.")
    ] = VALIDATION_CONFIG,
    ignore_masks: Annotated[
        bool, typer.Option("--ignore-masks", help="Segment every image, even with a mask.")
    ] = False,
) -> None:
    """Validation against wire FFR on a local, de-identified case list (research use only).

    Step 1, blinded: --manifest cases.csv runs the pipeline on every vessel and locks
    predictions.csv (SHA-256 in predictions_meta.json). Step 2: --predictions
    <folder>/predictions.csv --reference reference.csv writes report.md, report.json and
    figures/. Give --manifest and --reference together to do both. Case data and outputs must
    be outside this repository or in a git-ignored folder (docs/validation/DATA_SPEC.md).
    """
    from datetime import datetime

    from vstent.config import load_config
    from vstent.inference.onnx_runtime import (
        ONNX_FILE,
        SEG_ONNX_FILE,
        SegmentationRunner,
        SurrogateRunner,
    )
    from vstent.physics.solver import PhysicsParams
    from vstent.uncertainty.greyzone import load_uncertainty
    from vstent.validation.manifest import ManifestError, check_local_only, read_reference
    from vstent.validation.report import write_validation_report
    from vstent.validation.run import (
        INDEX_TESTS,
        ValidationConfig,
        model_info,
        read_predictions,
        run_predictions,
    )

    if (manifest is None) == (predictions is None):
        typer.echo("Give either --manifest (run the pipeline) or --predictions (score).", err=True)
        raise typer.Exit(code=2)
    if predictions is not None and reference is None:
        typer.echo("--predictions needs --reference.", err=True)
        raise typer.Exit(code=2)
    cfg = load_config(config, ValidationConfig)
    typer.echo(DISCLAIMER)
    try:
        if manifest is not None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out = out or Path("runs/validation") / f"{manifest.stem}-{stamp}"
            seg_dir = _optional_model_dir(seg_model, Path("models/seg"), SEG_ONNX_FILE)
            sur_dir = _optional_model_dir(surrogate_model, Path("models/surrogate"), ONNX_FILE)
            if seg_dir is None:
                typer.echo("No segmentation model: only cases with a mask can be analysed.")
            ucfg, qca_error = load_uncertainty(uncertainty)

            def progress(k, n, row):
                if k % 10 == 0 or k == n:
                    typer.echo(f"  {k} / {n} vessels")

            predictions = run_predictions(
                manifest, out,
                segmenter=None if seg_dir is None else SegmentationRunner.from_dir(seg_dir),
                surrogate=None if sur_dir is None else SurrogateRunner.from_dir(sur_dir),
                params=load_config(physics, PhysicsParams), uncertainty=ucfg,
                qca_error=qca_error, use_masks=not ignore_masks,
                models={"segmentation": model_info(seg_dir, SEG_ONNX_FILE),
                        "surrogate": model_info(sur_dir, ONNX_FILE)},
                progress=progress,
            )  # fmt: skip
            typer.echo(f"Predictions locked: {predictions}")
        out = out or predictions.parent
        if reference is None:
            typer.echo("No --reference: scoring left for the unblinded step.")
            return
        check_local_only([reference], "Reference file")
        check_local_only([out], "Validation output")
        rows, meta = read_predictions(predictions)
        summary = write_validation_report(
            rows, meta, read_reference(reference), cfg, out / "report.md",
            out / "figures" / "validation.png", title="Validation against wire FFR",
            intro=[f"Case list `{meta['manifest']['path']}`; reference `{reference}`."
                   if meta else f"Predictions `{predictions}`; reference `{reference}`."],
        )  # fmt: skip
    except (ManifestError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    r = summary["index_tests"].get("FFR", {}).get(cfg.primary_index_test)
    if r and r["n"]:
        ba = r.get("bland_altman")
        typer.echo(
            f"Primary ({INDEX_TESTS[cfg.primary_index_test]}): {r['n']} vessels, AUC "
            f"{r['auc']:.3f}, sensitivity {r['sensitivity']:.3f}, specificity "
            f"{r['specificity']:.3f}" + (f", bias {ba['bias']:+.3f}" if ba else "")
        )
    typer.echo(f"Wrote {out / 'report.md'}")


@app.command()
def validate_synthetic(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False, help="Stand-in study YAML.")
    ] = Path("configs/validation_synthetic.yaml"),
    validation: Annotated[
        Path, typer.Option("--validation", exists=True, dir_okay=False, help="Validation YAML.")
    ] = VALIDATION_CONFIG,
    n_patients: Annotated[
        int | None, typer.Option("--n-patients", min=1, help="Override the study size.")
    ] = None,
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    surrogate_model: Annotated[
        Path | None,
        typer.Option("--surrogate-model", help="Default: newest in models/surrogate/."),
    ] = None,
    physics: Annotated[
        Path, typer.Option("--physics", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    uncertainty: Annotated[
        Path, typer.Option("--uncertainty", dir_okay=False, help="Uncertainty YAML.")
    ] = UNCERTAINTY_CONFIG,
) -> None:
    """The validation tool end to end on synthetic stand-in cases → reports/validation_synthetic.md
    (not clinical data; P4)."""
    from vstent.config import load_config
    from vstent.inference.onnx_runtime import ONNX_FILE, SEG_ONNX_FILE
    from vstent.physics.solver import PhysicsParams
    from vstent.validation.run import INDEX_TESTS, ValidationConfig
    from vstent.validation.synthetic import SyntheticStudyConfig
    from vstent.validation.synthetic_report import REPORT, run_synthetic_validation

    study = load_config(config, SyntheticStudyConfig)
    if n_patients is not None:
        study = study.model_copy(update={"n_patients": n_patients})
    cfg = load_config(validation, ValidationConfig)

    def progress(arm, k, n):
        if k % 20 == 0 or k == n:
            typer.echo(f"  {arm}: {k} / {n} vessels")

    out = run_synthetic_validation(
        study, cfg, load_config(physics, PhysicsParams),
        seg_dir=_optional_model_dir(seg_model, Path("models/seg"), SEG_ONNX_FILE),
        surrogate_dir=_optional_model_dir(surrogate_model, Path("models/surrogate"), ONNX_FILE),
        uncertainty_path=uncertainty, progress=progress,
    )  # fmt: skip
    r = out["summary"]["index_tests"]["FFR"][cfg.primary_index_test]
    typer.echo(
        f"{out['study']['n_cases']} synthetic vessels; primary "
        f"({INDEX_TESTS[cfg.primary_index_test]}): AUC {r['auc']:.3f}, MAE {r['mae']:.3f}"
    )
    typer.echo(f"Wrote {REPORT}")


@app.command()
def sample_size(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False, help="Validation YAML.")
    ] = VALIDATION_CONFIG,
    out: Annotated[Path, typer.Option("--out", help="Report path.")] = Path(
        "reports/validation_sample_size.md"
    ),
) -> None:
    """Sample size for the validation study from the planning assumptions (placeholders)."""
    from vstent.config import load_config
    from vstent.validation.run import ValidationConfig
    from vstent.validation.sample_size import write_sample_size_report

    res = write_sample_size_report(load_config(config, ValidationConfig).sample_size, out)
    typer.echo(
        f"{res['vessels']} vessels in about {res['patients']} patients (governing endpoint: "
        f"{res['governing']}); the assumptions are placeholders to set with a statistician."
    )
    typer.echo(f"Wrote {out}")


@app.command()
def physics_nozzle(
    physics: Annotated[
        Path, typer.Option("--physics", exists=True, dir_okay=False, help="Physics YAML.")
    ] = PHYSICS_CONFIG,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="FDA nozzle data folder.")] = Path(
        "data/raw/fda_nozzle"
    ),
) -> None:
    """The loss model against the FDA benchmark nozzle experiments → reports/physics_nozzle.md."""
    from vstent.config import load_config
    from vstent.data.fda_nozzle import fetch_fda_nozzle
    from vstent.physics.nozzle import LAMINAR_RE, write_nozzle_report
    from vstent.physics.solver import PhysicsParams

    fetch_fda_nozzle(raw_dir, log=typer.echo)
    res = write_nozzle_report(load_config(physics, PhysicsParams), raw_dir=raw_dir)
    for c in res["cases"]:
        if c["reynolds"] == LAMINAR_RE and c["labs"]:
            typer.echo(
                f"{c['orientation']} Re {LAMINAR_RE}: model {c['model_total']:.0f} Pa vs measured "
                f"median {c['measured_median']:.0f} Pa (ratio {c['ratio']:.2f})"
            )
    typer.echo("Wrote reports/physics_nozzle.md")


@app.command()
def stenosis_grading(
    edges: Annotated[
        str,
        typer.Option(
            "--edges",
            help="Comma-separated: probability, intensity, model, dense, dense_log, dense_linear.",
        ),
    ] = "probability,intensity,model",
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CADICA folder.")] = Path(
        "data/raw/cadica"
    ),
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    n_boot: Annotated[int, typer.Option("--n-boot", min=100, help="Bootstrap resamples.")] = 2000,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Processes for the measurements.")
    ] = 1,
) -> None:
    """Automatic %DS at every CADICA expert lesion vs the experts' grades
    → reports/stenosis_grading.md (public data; P4b, P4c)."""
    import json

    from vstent.config import git_info
    from vstent.data.cadica import SOURCE_FILE, fetch_cadica, load_lesion_boxes
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
    from vstent.validation import stenosis_grading as sg

    methods = [m.strip() for m in edges.split(",") if m.strip()]
    unknown = [m for m in methods if m not in sg.EDGE_METHODS]
    if unknown or not methods:
        typer.echo(f"--edges: unknown {unknown}; use {list(sg.EDGE_METHODS)}", err=True)
        raise typer.Exit(code=2)
    git = git_info()  # before anything is written
    if not (raw_dir / SOURCE_FILE).exists():
        fetch_cadica(raw_dir, log=typer.echo)
    source = json.loads((raw_dir / SOURCE_FILE).read_text(encoding="utf-8"))
    tracks = sg.lesion_tracks(load_lesion_boxes(raw_dir))
    split = sg.patient_split([t.patient for t in tracks])
    seg_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    seg = SegmentationRunner.from_dir(seg_dir)
    typer.echo(f"{len(tracks)} expert lesions in {len({t.patient for t in tracks})} patients")

    def progress(k, n):
        if k % 500 == 0 or k == n:
            typer.echo(f"  {k} / {n} frames")

    rows = sg.measure_tracks(tracks, seg, split=split, methods=tuple(methods), progress=progress,
                             workers=workers)  # fmt: skip
    stats = sg.write_grading_report(rows, Path("reports/stenosis_grading.md"), source=source,
                                    n_boot=n_boot, git=git,
                                    seg_model=Path(seg_dir).as_posix())  # fmt: skip
    for m, s in stats.items():
        t = s["test"]
        if "kappa" in t:
            typer.echo(f"{m}: test kappa {t['kappa']:.2f}, AUC>=70% {t['auc_ge70']:.2f}, "
                       f"measured {t['n_measured']}/{t['n_lesions']}")  # fmt: skip
    typer.echo("Wrote reports/stenosis_grading.md")


@app.command()
def cine_frames(
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="CADICA folder.")] = Path(
        "data/raw/cadica"
    ),
    seg_model: Annotated[
        Path | None, typer.Option("--seg-model", help="Default: newest in models/seg/.")
    ] = None,
    n_boot: Annotated[int, typer.Option("--n-boot", min=100, help="Bootstrap resamples.")] = 2000,
) -> None:
    """Automatic analysis frame and opacified window of every CADICA run vs the frames its
    clinicians selected, plus contrast transit → reports/cine_frames.md (public data; P4d)."""
    from vstent.config import git_info
    from vstent.data.cadica import (
        SOURCE_FILE,
        fetch_cadica,
        load_artery_groups,
        load_lesion_boxes,
        load_selected_runs,
    )
    from vstent.imaging.cine import OPACIFIED_FRACTION, opacification
    from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
    from vstent.validation import cine_frames as cf
    from vstent.validation.stenosis_grading import lesion_tracks, patient_split

    git = git_info()  # before anything is written
    if not (raw_dir / SOURCE_FILE).exists():
        fetch_cadica(raw_dir, log=typer.echo)
    model_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    seg = SegmentationRunner.from_dir(model_dir)
    runs = load_selected_runs(raw_dir)
    split = patient_split([t.patient for t in lesion_tracks(load_lesion_boxes(raw_dir))])
    typer.echo(f"{len(runs)} selected runs of {len({r.patient for r in runs})} patients")

    def progress(k, n):
        if k % 50 == 0 or k == n:
            typer.echo(f"  {k} / {n} runs")

    areas = cf.run_areas(runs, seg.predict, cf.AREAS_DIR / f"areas_{model_dir.name}.json",
                         progress)  # fmt: skip
    rows = [{**cf.score_run(r, areas[cf.run_key(r)]), "split": split[r.patient]} for r in runs]
    from vstent.imaging.run_analysis import RUN_LEVEL

    typer.echo(f"End-diastolic rule (curves at probability ≥ {RUN_LEVEL:g}) …")
    stats_file = cf.AREAS_DIR / f"stats_{model_dir.name}_{RUN_LEVEL:g}.json"
    run_stats = cf.run_stats(runs, seg.predict, stats_file, progress, level=RUN_LEVEL)
    ed_rows = []
    for r in runs:
        st = run_stats[cf.run_key(r)]
        frame_ed = cf.ed_frame(st["area"], st["spread"])
        ed_rows.append({**cf.score_run(r, st["area"], analysis_frame=frame_ed),
                        "split": split[r.patient]})  # fmt: skip
    typer.echo("Contrast transit …")
    transit = []
    for k, r in enumerate(runs, 1):
        frame = opacification(areas[cf.run_key(r)]).analysis_frame
        transit.append(cf.transit_for_run(r, seg.predict, frame))
        progress(k, len(runs))
    stats = cf.write_report(
        rows, transit, areas, Path("reports/cine_frames.md"), fraction=OPACIFIED_FRACTION,
        arteries=load_artery_groups(raw_dir), model=model_dir.as_posix(), git=git, n_boot=n_boot,
        ed_rows=ed_rows, ed_level=RUN_LEVEL,
    )  # fmt: skip
    for sp, st in stats.items():
        typer.echo(f"{sp}: analysis frame among the selected frames {st['hit']['k']}/"
                   f"{st['hit']['n']}, median window IoU {st['iou']['median']:.2f}")  # fmt: skip
    typer.echo("Wrote reports/cine_frames.md")


APP_SCRIPT = Path(__file__).resolve().parent / "app" / "streamlit_app.py"  # the research view
GUIDED_SCRIPT = Path(__file__).resolve().parent / "app" / "guided_app.py"
LABEL_SCRIPT = Path(__file__).resolve().parent / "app" / "label_app.py"  # P4l: ED labels


@app.command()
def regulatory(
    check: Annotated[
        bool, typer.Option("--check", help="Write nothing; fail if a document is out of date.")
    ] = False,
) -> None:
    """Regulatory drafts: requirements, hazards, traceability, SOUP list and SBOM →
    docs/regulatory/ (drafts for review by a qualified person; not regulatory advice)."""
    from vstent.regulatory.build import REG_DIR, write_all

    typer.echo(DISCLAIMER)
    try:
        changed = write_all(check=check)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from None
    if check:
        for path in changed:
            typer.echo(f"Out of date: {path.as_posix()} (run `vstent regulatory`)")
        if changed:
            raise typer.Exit(code=1)
        typer.echo(f"All documents in {REG_DIR.as_posix()} match their sources.")
    else:
        for path in changed:
            typer.echo(f"Wrote {path.as_posix()}")
        typer.echo(f"{len(changed)} documents changed in {REG_DIR.as_posix()}.")


@app.command()
def app_latency(
    n: Annotated[int, typer.Option("--n", min=1, help="Start-slider updates per mode.")] = 20,
    report_dir: Annotated[
        Path, typer.Option("--report-dir", help="Where to write app_latency.md.")
    ] = Path("reports"),
) -> None:
    """Time the app's response to stent-control changes (synthetic and image mode)."""
    from vstent.app.latency import write_latency_report

    report = write_latency_report(report_dir, n, log=typer.echo)
    typer.echo(f"[{'PASS' if report['passed'] else 'FAIL'}] every update < 300 ms (server time)")
    typer.echo(f"Wrote {report_dir / 'app_latency.md'}")
    if not report["passed"]:
        raise typer.Exit(code=1)


@app.command("app")
def launch_app(
    port: Annotated[int, typer.Option("--port", help="Server port.")] = 8501,
    address: Annotated[
        str, typer.Option("--address", help="Bind address (default: this computer only).")
    ] = "localhost",
    headless: Annotated[
        bool, typer.Option("--headless", help="Do not open a browser window.")
    ] = False,
    advanced: Annotated[
        bool,
        typer.Option("--advanced", help="The full research view instead of the guided app."),
    ] = False,
) -> None:
    """Launch the app (research use only): the guided cine workflow, or with --advanced the
    full research view."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_SCRIPT if advanced else GUIDED_SCRIPT),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.address",
        address,
    ]
    if headless:
        cmd += ["--server.headless", "true"]
    typer.echo(f"{DISCLAIMER} Starting the app on http://{address}:{port} (Ctrl+C to stop).")
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("label-ed")
def label_ed(
    port: Annotated[int, typer.Option("--port", help="Server port.")] = 8502,
    headless: Annotated[bool, typer.Option("--headless", help="No browser window.")] = False,
) -> None:
    """Expert ED-frame labelling page (P4l; research use only): mark one end-diastolic frame
    per heart beat → data/labels/ed/<reader>/."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "streamlit", "run", str(LABEL_SCRIPT), "--server.port",
           str(port), "--browser.gatherUsageStats", "false",
           "--server.address", "localhost"]  # fmt: skip
    if headless:
        cmd += ["--server.headless", "true"]
    typer.echo(f"{DISCLAIMER} ED labelling on http://localhost:{port} (Ctrl+C to stop).")
    raise typer.Exit(code=subprocess.call(cmd))


if __name__ == "__main__":
    app()
