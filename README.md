# vstent: Virtual Stenting Planner

> **⚠ Research use only — not for clinical decisions.**
> This software is a research prototype. It is not a medical device, is not validated for
> clinical use, and must never be used to make decisions about real patients.

vstent is a research prototype that:

1. segments coronary vessels in an angiogram image,
2. measures the lumen diameter profile along a user-selected vessel path (2D QCA),
3. computes an FFR pullback curve with a reduced-order physics model and a learned
   surrogate model (ONNX Runtime),
4. lets you place a virtual stent and see the predicted post-PCI FFR,
5. suggests the shortest stent that brings distal FFR back above target.

The build plan is in [docs/PLAN.md](docs/PLAN.md); progress is logged in
[docs/PROGRESS.md](docs/PROGRESS.md).

## Status

Done: M0 (setup), M1 (reduced-order physics solver), M2 (synthetic data factory), M3 (FFR
surrogate, see [reports/surrogate_eval.md](reports/surrogate_eval.md)), M4 (ONNX export and
runtime, see [reports/onnx_benchmark.md](reports/onnx_benchmark.md)), M5 (ARCADE vessel
segmentation, QCA and the end-to-end pipeline, see [reports/seg_eval.md](reports/seg_eval.md)
and [reports/qca_gallery.md](reports/qca_gallery.md)), M6 (virtual stent and plan optimiser,
see [reports/planner_benchmark.md](reports/planner_benchmark.md)) and M7 (Streamlit app, see
[Using the app](#using-the-app) and [reports/app_latency.md](reports/app_latency.md)). `vstent
demo` is a stub until M8.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (on Windows: `winget install astral-sh.uv`).

```bash
# 1. create .venv and install everything (Python version pinned in .python-version)
uv sync --all-extras

# 2. run the checks
uv run ruff check . && uv run ruff format --check . && uv run pytest -m "not slow" -q

# 3. list the commands
uv run vstent --help
```

Without uv: `python -m venv .venv`, activate it, then `pip install -e ".[dev,imaging,app]"`.

**GPU:** on Windows and Linux, PyTorch is installed from the CUDA 12.6 wheel index
(needs an NVIDIA driver ≥ 560). It also runs on CPU, just slower.

## Commands

| Command | What it does | Milestone |
|---|---|---|
| `vstent physics-report` | physics sanity checks → `reports/physics_checks.md` + FFR-vs-%DS figure | M1 |
| `vstent synth --config configs/synth_v1.yaml` | generate the synthetic dataset (~0.5 GB in `data/synth/v1/`) + `reports/dataset_card.md` | M2 |
| `vstent train-surrogate --config configs/surrogate.yaml` | train the FFR surrogate | M3 |
| `vstent stop <run-name>` / `vstent checkpoints` | pause a training after the current epoch / list resumable runs | — |
| `vstent train-surrogate --resume <run-name>` | continue a stopped or interrupted training exactly | M3 |
| `vstent eval-surrogate --run runs/<name>` | evaluate the surrogate vs the solver | M3 |
| `vstent export-surrogate --run runs/<name>` | export to ONNX (`models/surrogate/<name>/`), check parity with torch, CPU benchmark → `reports/onnx_benchmark.md` | M4 |
| `vstent data arcade` | download, MD5-verify and extract the ARCADE dataset (CC0, ~450 MB) into `data/raw/arcade/` | M5 |
| `vstent train-seg --config configs/seg.yaml` | train vessel segmentation (`--resume <run-name>` to continue) | M5 |
| `vstent eval-seg --run runs/<name>` | Dice on ARCADE val / test → `reports/seg_eval.md` | M5 |
| `vstent export-seg --run runs/<name>` | export segmentation to ONNX, parity → `reports/seg_onnx.md` | M5 |
| `vstent analyse <image> [--start R C --end R C]` | segment an angiogram, QCA + FFR (solver and surrogate) for one vessel | M5 |
| `vstent qca-gallery` | qualitative QCA gallery on 20 ARCADE stenosis images → `reports/qca_gallery.md` | M5 |
| `vstent analyse <image> --plan` | … plus the virtual stent planner: shortest plan reaching FFR ≥ 0.90 (generic catalogue) | M6 |
| `vstent plan-benchmark` | planner speed (5,000 candidates) and behaviour on synthetic vessels → `reports/planner_benchmark.md` | M6 |
| `vstent qca-synthetic` | QCA accuracy on rendered synthetic tubes → `reports/qca_synthetic.md` | M5 |
| `vstent demo` | synthetic case end-to-end (FFR → plan → post-PCI) → `reports/demo/` | M8 |
| `vstent mvp-report` | assemble `reports/mvp_report.md` from all generated reports | M8 |
| `vstent app` | launch the Streamlit app on http://localhost:8501 | M7 |
| `vstent app-latency` | time the app's response to stent-control changes → `reports/app_latency.md` | M7 |

## Using the app

`uv run vstent app` opens the planner in your browser (it listens on this computer only). The
red banner is permanent: **research use only**.

1. **Synthetic vessel** (sidebar): pick a preset (focal, tandem) or a random synthetic vessel.
   **Angiogram image**: upload a PNG / JPG / DICOM or pick an ARCADE example; the vessel is
   segmented automatically. Click the start (proximal) and end points on the image, or use
   *Auto path* / the manual row-col form. Set the pixel size (otherwise 0.30 mm/px is assumed
   and flagged).
2. Read the diameter profile (reference, lesions, MLD) and the FFR pullback from the solver and
   the surrogate, with their difference.
3. **Virtual stent** (sidebar): switch on *Place a stent*, choose catalogue length and diameter,
   move the start slider. Post-PCI FFR updates immediately.
4. **Suggest plan**: the shortest plan reaching FFR ≥ 0.90 (top 3, with the solver check); pick
   one under *Show on plots* to draw it.
5. **Prepare case report** → download the JSON (all assumptions, model versions, no image
   pixels) and a PNG of the plots.

## Repository layout

```
configs/      YAML configs
docs/         plan, progress log, decisions, physics and data notes
src/vstent/   the package: physics, data, surrogate, imaging, planning, inference, app
tests/        pytest suite (slow tests marked @pytest.mark.slow)
reports/      generated metrics and figures
notebooks/    exploration only, never the source of truth
data/ models/ runs/   generated artefacts (git-ignored)
checkpoints/          resumable training state, one folder per run (git-ignored)
```

## Limitations

Single vessel, no side branches, steady flow, Newtonian blood, geometry derived from a
single 2D projection. See [docs/PHYSICS.md](docs/PHYSICS.md) for the full list.
