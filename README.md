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

M0 (setup), M1 (reduced-order physics solver), M2 (synthetic data factory) and M3 (FFR
surrogate, see [reports/surrogate_eval.md](reports/surrogate_eval.md)) are done. The remaining
commands are stubs until their milestone is implemented.

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
| `vstent export-surrogate --run runs/<name>` | export the surrogate to ONNX | M4 |
| `vstent data arcade` | download the ARCADE dataset | M5 |
| `vstent train-seg --config configs/seg.yaml` | train vessel segmentation | M5 |
| `vstent demo` | synthetic case end-to-end → `reports/demo/` | M8 |
| `vstent app` | launch the Streamlit app | M7 |

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
