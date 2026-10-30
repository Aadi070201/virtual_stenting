"""Command-line interface: ``vstent <command>`` (research use only)."""

import sys
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


def _echo_plans(planned) -> None:
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


def _latest_model_dir(root: Path, filename: str) -> Path:
    from vstent.inference.models import latest_model_dir

    found = latest_model_dir(root, filename)
    if found is None:
        typer.echo(f"No exported model in {root}/ (run the export command first).", err=True)
        raise typer.Exit(code=1)
    return found


@app.command()
def analyse(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Angiogram image.")],
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
) -> None:
    """Segment an angiogram and compute QCA + FFR for one vessel (research use only).

    Without --start / --end the longest vessel path in the mask is used. With --plan the
    virtual stent planner proposes the shortest plan reaching the target FFR.
    """
    import cv2

    from vstent.config import load_config
    from vstent.imaging.centerline import longest_path_endpoints
    from vstent.imaging.segment import THRESHOLD
    from vstent.inference.onnx_runtime import (
        ONNX_FILE,
        SEG_ONNX_FILE,
        SegmentationRunner,
        SurrogateRunner,
    )
    from vstent.inference.pipeline import analyse_vessel
    from vstent.physics.solver import PhysicsParams

    img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if img is None:
        typer.echo(f"Cannot read {image}", err=True)
        raise typer.Exit(code=1)
    seg_dir = seg_model or _latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    sur_dir = surrogate_model or _latest_model_dir(Path("models/surrogate"), ONNX_FILE)
    prob = SegmentationRunner.from_dir(seg_dir).predict(img)
    if not start or not end or start[0] is None or end[0] is None:
        start, end = longest_path_endpoints(prob >= THRESHOLD)
        typer.echo(f"Auto-selected path: start {start}, end {end} (row, col)")
    surrogate = SurrogateRunner.from_dir(sur_dir)
    params = load_config(physics, PhysicsParams)
    result = analyse_vessel(
        prob, start, end, params=params, surrogate=surrogate, mm_per_px=mm_per_px
    )
    q = result.qca
    typer.echo(f"{DISCLAIMER}")
    typer.echo(
        f"Path length {q.s_mm[-1]:.1f} mm at {q.mm_per_px:g} mm/px "
        f"({'assumed' if q.calibration_assumed else 'given'}); MLD {q.mld_mm:.2f} mm; "
        f"max DS {q.max_percent_stenosis:.0f}%; lesions {len(q.lesions)}"
    )
    typer.echo(
        f"Distal FFR: solver {result.ffr_distal_solver:.3f}, surrogate "
        f"{result.ffr_distal_surrogate:.3f}"
    )
    for w in result.warnings:
        typer.echo(f"Warning: {w}")
    if plan:
        from vstent.planning.optimizer import PlannerConfig, StentCatalog, plan_stents

        planned = plan_stents(
            q.profile,
            catalog=load_config(catalog, StentCatalog),
            config=load_config(planner, PlannerConfig),
            params=params,
            runner=surrogate,
        )
        _echo_plans(planned)
    surrogate.close()
    if figure is not None:
        from vstent.inference.figures import plot_analysis

        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plot_analysis(result, gray, figure, title=f"Vessel analysis: {image.name}")
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
) -> None:
    """Qualitative QCA gallery on ARCADE stenosis/val images → reports/qca_gallery.md."""
    from vstent.config import load_config
    from vstent.inference.gallery import qca_gallery as run_gallery
    from vstent.inference.onnx_runtime import ONNX_FILE, SEG_ONNX_FILE
    from vstent.physics.solver import PhysicsParams

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
) -> None:
    """Planner speed (5,000 candidates) and behaviour on synthetic test vessels."""
    from vstent.config import load_config
    from vstent.inference.onnx_runtime import ONNX_FILE
    from vstent.physics.solver import PhysicsParams
    from vstent.planning.optimizer import PlannerConfig, StentCatalog
    from vstent.planning.report import run_planner_benchmark

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


APP_SCRIPT = Path(__file__).resolve().parent / "app" / "streamlit_app.py"


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
) -> None:
    """Launch the Streamlit app (research use only)."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_SCRIPT),
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


if __name__ == "__main__":
    app()
