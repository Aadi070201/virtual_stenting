# vstent — Virtual Stenting Planner (research MVP)

> **Research use only.** This software is a research prototype. It is not a medical device, is not validated for clinical use, and must never be used to make decisions about real patients. Every UI screen, report and README must say this.

## Mission

In about 3 weeks, build a working research prototype that:

1. Takes a coronary angiogram image and segments the coronary vessels.
2. Measures the lumen diameter profile along a vessel path the user selects (2D QCA).
3. Computes an **FFR pullback curve** (FFR at every point along the vessel) using a reduced-order physics model **and** a learned surrogate model running in **ONNX Runtime**.
4. Lets the user place a **virtual stent** (start position, length, diameter) and immediately see the predicted **post-PCI FFR**.
5. Suggests the best stent plan: the shortest stent that brings distal FFR back above target.

The detailed, step-by-step plan with acceptance criteria is in **`docs/PLAN.md`**. That file is the source of truth for what to build next.

## How to work (autonomous loop)

1. At the start of every session, read `docs/PLAN.md` and `docs/PROGRESS.md`. Pick the first unchecked task in the lowest-numbered unfinished milestone.
2. Implement it with tests. Run the checks (see **Commands**).
3. When the task's acceptance criteria are met:
   - tick its checkbox in `docs/PLAN.md`
   - append a short entry to `docs/PROGRESS.md` (date, what was done, key metrics, open issues)
   - `git commit` with message `M<n>: <short summary>`
4. Continue to the next task **without waiting for confirmation**, until milestone M8 is complete or you are blocked.
5. Record every non-trivial design decision in `docs/DECISIONS.md` (one short paragraph: decision, reason, alternatives).
6. Keep this file up to date if commands or structure change.

**Stop and ask the human only when:**
- a dataset download fails or needs a manual licence acceptance
- credentials, paid services or cloud accounts are needed
- an acceptance criterion is still not met after 2 serious attempts (write the analysis and options in `PROGRESS.md` first)
- anything would involve real patient data
- you think Phase 2 work should begin

## Ground rules

- **Never fabricate results.** Every number in a report must be produced by a script in this repo and be reproducible from saved configs. If a target is missed, report the real number.
- **Honest claims about the surrogate.** In this MVP the reduced-order solver is already fast. The surrogate exists to build and validate the training → ONNX → app pipeline, so that in Phase 2 its labels can be swapped for 3D CFD, where it gives the real speed-up. Do not claim a speed-up over the reduced-order solver.
- **Do not tune physics constants to make tests pass.** If a physics sanity test fails, find the implementation bug. Sampling distributions for synthetic data *may* be tuned.
- **SI units internally** (m, m², m³/s, Pa, Pa·s, kg/m³). Convert only at input/output boundaries. 1 mmHg = 133.322 Pa. Every function docstring states units.
- **Reproducibility:** fixed seeds; every generated dataset, trained model and report is saved next to its config and the git commit hash.
- **Data hygiene:** `data/`, `models/` and `runs/` are git-ignored. No patient data in the repo, ever. Record the source and licence of every dataset in `docs/DATA.md`.
- **One source of truth for pre-processing:** the same function prepares inputs for training, ONNX inference and the app.
- **Generic stent catalogue only** (diameters and lengths). No commercial brand names.
- **Keep it simple first.** Get a working, tested version, then improve. No notebooks as source of truth (`notebooks/` is for exploration only).
- **No GPU? Don't block.** Train with smaller configs on CPU, note it in `PROGRESS.md`, and continue.
- **Do not start Phase 2** (listed at the end of `PLAN.md`) without explicit approval.

## Tech stack

- Python ≥ 3.11, packaged with `pyproject.toml` (src layout). Use `uv` if available, otherwise `venv` + `pip`.
- Core: `numpy`, `scipy`, `pydantic` (configs), `pyyaml`, `typer` (CLI), `networkx`
- ML: `torch`, `segmentation-models-pytorch`, `onnx`, `onnxruntime`
- Imaging: `scikit-image`, `opencv-python-headless`, `pycocotools`, `pydicom` (optional)
- App: `streamlit`, `plotly`, `streamlit-image-coordinates`
- Dev: `pytest`, `ruff`. Mark slow tests with `@pytest.mark.slow`.

## Repo layout

```
vstent/
  CLAUDE.md
  README.md
  pyproject.toml        # deps, extras, ruff, pytest; torch from the CUDA 12.6 index via uv
  .python-version       # Python pinned for uv (3.14)
  uv.lock               # committed lock file
  configs/              # YAML configs: physics, synth_v1, surrogate, seg, stent_catalog
  docs/                 # PLAN.md, PROGRESS.md, DECISIONS.md, PHYSICS.md, DATA.md, INVENTION_NOTES.md
  src/vstent/
    physics/            # geometry.py, solver.py
    data/               # synth.py, arcade.py
    surrogate/          # features.py, model.py, train.py, evaluate.py, export.py
    imaging/            # segment.py, centerline.py, qca.py
    planning/           # stent.py, optimizer.py
    inference/          # onnx_runtime.py, pipeline.py
    app/                # streamlit_app.py
    config.py           # StrictModel base, YAML loader, save_run_metadata (config + git hash)
    cli.py
  tests/
  reports/              # generated metrics (*.md, *.json) and figures/
  data/  models/  runs/ # git-ignored
```

## Commands (create these in M0, keep them working)

```bash
# setup
uv sync --all-extras            # or: python -m venv .venv && pip install -e ".[dev,imaging,app]"

# checks: must pass before every commit
uv run ruff check . && uv run ruff format --check . && uv run pytest -m "not slow" -q

# pipeline (Typer CLI: `vstent`)
uv run vstent synth --config configs/synth_v1.yaml
uv run vstent train-surrogate --config configs/surrogate.yaml
uv run vstent eval-surrogate --run runs/<name>
uv run vstent export-surrogate --run runs/<name>
uv run vstent data arcade
uv run vstent train-seg --config configs/seg.yaml
uv run vstent demo              # end-to-end synthetic case → report
uv run vstent app               # launches the Streamlit app
```

## Physics in one paragraph (full spec in PLAN.md, M1)

A single vessel is a radius profile r(s) along its centreline. Hyperemic flow Q is set by aortic pressure Pa, venous pressure Pv, a microvascular resistance R_micro and the vessel's own pressure losses: **ΔP(Q) = a·Q + b·Q²**. Here *a* is Poiseuille viscous loss, ∫ 8μ/(π r⁴) ds, and *b* is the expansion (separation) loss after each narrowing, Kt·ρ/2·(1/A_min − 1/A_max)² with Kt = 1.52. Solving **Pa − Pv = a·Q + b·Q² + R_micro·Q** is a quadratic with a closed-form positive root. Then **FFR(s) = (Pa − ΔP(s)) / Pa**. R_micro is calibrated so the stenosis-free reference vessel carries a hyperemic velocity of 0.35 m/s. A virtual stent sets r(s) = max(r(s), r_stent) over the stented segment.

## MVP definition of done

- All milestones M0–M8 in `docs/PLAN.md` are ticked, or any shortfall is documented honestly.
- `vstent demo` and `vstent app` work end-to-end on a fresh clone after setup.
- `reports/mvp_report.md` is generated from code and lists metrics and limitations.
- The checks pass and `v0.1.0` is tagged.
