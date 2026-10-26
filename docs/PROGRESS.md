# Progress log

Newest entries at the bottom. Each entry: date, what was done, key metrics, open issues.

---

## 2026-10-26 — M0: project setup

**Done**
- Environment: `uv` 0.13.5, Python 3.14.5 (pinned in `.python-version`), `.venv/` created by `uv sync`.
- PyTorch 2.14.1 + torchvision 0.29.1 from the CUDA 12.6 index (machine: AMD Ryzen 7 6800H, NVIDIA RTX 3060 Laptop 6 GB, driver 566.07). Verified `torch.cuda.is_available()` and a torchvision CUDA op on the GPU.
- Git repo on `main`; `.gitignore` covers `/data/`, `/models/`, `/runs/`, `.venv/`, `*.onnx` (except `tests/fixtures/`), caches and secrets.
- `pyproject.toml` (src layout, hatchling) with extras `dev`, `imaging`, `app`; ruff and pytest configured; `slow` marker registered.
- Typer CLI `vstent` with stub subcommands for every command in CLAUDE.md.
- `vstent.config`: strict Pydantic base model, YAML loader, `save_run_metadata` (resolved config + git commit + dirty flag).
- `README.md` (disclaimer + quickstart), `docs/PROGRESS.md`, `docs/DECISIONS.md`, `docs/DATA.md`.
- `PLAN.md` moved to `docs/PLAN.md` (the location CLAUDE.md refers to).

**Metrics**
- `ruff check`, `ruff format --check`: clean. `pytest -m "not slow"`: 9 passed in 0.9 s.

**Open issues**
- torchvision (pulled in by segmentation-models-pytorch) first resolved to the CPU build from PyPI and failed on CUDA tensors; fixed by listing it directly in the `imaging` extra with the cu126 source.
- uv was installed with winget; restart VS Code / the shell so `uv` is on PATH.

---

## 2026-10-26 — M1: reduced-order physics solver

**Done**
- `physics/geometry.py`: `VesselProfile` (validated, read-only arrays, resampling), robust reference-radius estimate, %DS, lesion finder (DS > 30%), MLD, cosine-lesion helpers.
- `physics/solver.py`: `PhysicsParams` (config `configs/physics.yaml`), vectorised `solve_ffr_batch` (Poiseuille + Young–Tsai expansion losses spread along diverging runs, R_micro calibration, stable quadratic root, `resistance` / `fixed_velocity` modes, loss breakdown); `solve_ffr` = batch of one.
- `physics/report.py` + `vstent physics-report` → `reports/physics_checks.{md,json}`, `reports/figures/ffr_vs_ds.png` (generated from a clean tree; the commit hash is recorded in the report).
- `docs/PHYSICS.md`: equations, guarantees, safeguards, assumptions, limitations, references.
- 32 physics/geometry tests: every M1 spec test plus tapered-tube integral, Borda–Carnot step, batch-row isolation, grid convergence, batch-vs-single equivalence, input validation.

**Metrics** (from `reports/physics_checks.md`)
- Healthy 3.0 × 50 mm: distal FFR 0.982 (≥ 0.95).
- 3.0 × 60 mm, 10 mm cosine lesion: 50% DS → 0.903 (band 0.84–0.94); 70% DS → 0.620 (band 0.45–0.70). No constants were changed.
- Batch solver: ~50,000 profiles/s for 10,000 × 256 points on AMD Ryzen 7 6800H (target ≥ 5,000).
- Checks: ruff clean; `pytest`: 47 passed (incl. 2 slow).

**Open issues / observations**
- In this model FFR is only weakly sensitive to lesion length (the expansion term depends only on A_min/A_max); e.g. at 70% DS: 5 mm → 0.639, 20 mm → 0.594. Documented as a limitation in PHYSICS.md.
- `fixed_velocity` mode gives FFR < 0 for very severe lesions (≈ −80 at 90% DS); kept unclipped as a comparison mode, documented.
- Measurement noise can split an expansion into several diverging runs and lower the expansion loss; relevant to M2 noisy labels (labels are computed from the same noisy profile, so they stay self-consistent).
