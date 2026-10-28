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

---

## 2026-10-28 — M2: synthetic data factory

**Done**
- `data/synth.py`: `SynthConfig` / `SamplingConfig` (`configs/synth_v1.yaml`), `sample_vessel` (tapered reference, 0–4 lesions of 4 shapes with exact throat DS, diffuse disease, smoothed noise, Pa/Pv/v_hyp), `generate_shard` (float32 profile → labels from that exact profile), `generate_dataset` (multiprocessing, shard seeds = (split seed, shard index), per-shard content SHA-256, resolved config + git commit), `load_split`.
- `data/card.py` + `vstent synth` → `reports/dataset_card.{md,json}`, `reports/figures/dataset_hist.png`.
- 14 tests: config/ranges, shape bumps, placement constraints (margins, gaps, counts), exact nominal DS, OOD overrides, shard reproducibility, labels = solver on stored profile, worker-count independence, overwrite protection, card, CLI.

**Sampling tuning (allowed; physics untouched).** With lesion DS ~ Beta(2, 2) the train grey zone was 14.3% (< 15%). Sweep on 10k samples each (seed (7, 0)):

| Beta(a, b) | mass in 40–80% DS | FFR < 0.80 | grey zone | median FFR |
|---|---|---|---|---|
| (2.0, 2.0) | 0.746 | 59.1% | 14.2% | 0.728 |
| (2.0, 2.5) | 0.715 | 51.4% | 16.6% | 0.791 |
| (2.5, 2.5) | 0.796 | 59.3% | 15.2% | 0.733 |
| (2.5, 3.0) | 0.765 | 52.9% | 17.5% | 0.783 |
| (3.0, 3.0) | 0.832 | 59.8% | 17.0% | 0.738 |
| **(3.0, 3.5)** | **0.803** | **52.7%** | **18.4%** | **0.783** |
| (3.0, 4.0) | 0.765 | 47.4% | 19.2% | 0.811 |

Chose (3.0, 3.5) (DECISIONS D10).

**Metrics** (from `reports/dataset_card.md`, generated at commit 86ec53c, clean tree)
- FFR < 0.80 / grey zone [0.75, 0.85]: train 53.3% / 18.4%, val 54.2% / 18.1%, test 53.8% / 17.8% (targets ≥ 30% / ≥ 15%: met). OOD: 96.1% / 7.4% (no target).
- 125,000 samples in 8.4 s (13 processes, AMD Ryzen 7 6800H); target < 15 min. 504 MB on disk.
- Reproducibility: full regeneration with 4 workers gave 13/13 identical shard content hashes.
- R_micro clamped: 0 samples. Checks: ruff clean; `pytest -m "not slow"`: 59 passed.

**Open issues / observations**
- Train median max DS 55%; ~10% of vessels have no focal lesion (max DS mostly < 10% from noise / diffuse disease).
- OOD is much more severe (median distal FFR 0.52), as expected for 3–4 lesions in 80–120 mm vessels.

---

## 2026-10-28 — M3: surrogate model

**Done**
- `surrogate/features.py`: the single pre-processing path (`RawInputs` → `prepare_inputs`) with 4 per-point channels, 5 global features and `NormStats` (training-set statistics saved as JSON, version-checked).
- `surrogate/model.py`: 1D U-Net (4 levels, base 32, GroupNorm, GELU, FiLM conditioning, global-context bottleneck) with a monotone softplus–cumsum head (FFR(0) = 1, non-increasing by construction); 1,058,881 parameters.
- `surrogate/dataset.py`, `surrogate/train.py` (`vstent train-surrogate`): AdamW + per-step cosine LR, float16 AMP, early stopping on val distal MAE, resumable via `vstent.checkpoints` (`vstent stop`, `vstent checkpoints`, `--resume`; DECISIONS D15).
- `surrogate/baseline.py`: histogram gradient boosting on hand features (max DS, lesion length, D0, L, Pa, R_micro).
- `surrogate/evaluate.py` (`vstent eval-surrogate`) → `reports/surrogate_eval.{md,json}` + 5 figures.
- Tests: features, head monotonicity / FFR(0) = 1, shapes, training smoke, metrics, evaluation, exact stop/resume reproducibility.

**Metrics** (from `reports/surrogate_eval.md`; run `runs/unet1d-20261028-125810` trained at commit deb7b35, evaluated at commit 4e93e88, both clean)
- Test split (10,000) vs solver: distal MAE **0.0013** (target ≤ 0.015), bias +0.0000, 95% LoA **[−0.0057, +0.0056]** (target within ±0.04), agreement at 0.80 **99.76%** (target ≥ 97%), sensitivity 99.80%, specificity 99.72%, curve MAE 0.0009, max |error| 0.087. All three targets met.
- Baseline on test: distal MAE 0.0342, LoA [−0.0906, +0.0891], agreement 93.39%.
- OOD (5,000; 80–120 mm, 3–4 lesions; no target): surrogate MAE 0.0039, LoA [−0.0115, +0.0118], agreement 99.90%; baseline MAE 0.0505.
- Test MAE by max DS bin: 0–30% 0.0002, 30–50% 0.0008, 50–70% 0.0016, 70–100% 0.0021.
- Training: 50 epochs, 12.8 min on RTX 3060 Laptop GPU (≈ 15 s/epoch, AMP), best val distal MAE 0.0013 at epoch 50.
- Checks: ruff clean; `pytest -m "not slow"`: 79 passed.

**Open issues / observations**
- Best epoch = last epoch: most of the final gain comes from the tail of the cosine schedule (LR → 0); training longer was not needed for the targets.
- A few test vessels fall outside ±0.04 (max |error| 0.087; see the Bland–Altman figure), mostly in the 0.6–0.8 FFR range.
- OOD specificity (98.48%) rests on few negatives: 96% of OOD vessels have FFR < 0.80.
- The first M3 run (`runs/unet1d-20261028-091227`) was stopped at epoch 30 on request before checkpointing existed; it is superseded and not used in any report.
- Stretch goal (5-seed ensemble with uncertainty) not done.
- These numbers measure agreement with the reduced-order solver on synthetic data only; accuracy on real angiography-derived profiles is not measured until M5–M6.

---

## 2026-10-28 — M4: ONNX export and runtime

**Done**
- `surrogate/export.py`: TorchScript exporter at opset 17 (the dynamo exporter cannot target 17 for this model; DECISIONS D16), dynamic batch axis, N = 256, metadata embedded (feature version, N, channels, run, disclaimer) and `onnx.checker` full check. Writes `models/surrogate/<run>/` with `surrogate.onnx`, `norm_stats.json`, `config.yaml` and `export_meta.json` (run and export commits, versions, SHA-256, parity).
- `inference/onnx_runtime.py`: torch-free `SurrogateRunner` (`predict(RawInputs)`, `predict_features`, `predict_profile(VesselProfile, pa, pv, r_micro)`), using the shared `features.prepare_inputs`, checking model metadata against the normalisation stats, and splitting large batches into chunks of 16 run in parallel (D17).
- `inference/benchmark.py`: CPU latency (runner default and single-call), reduced-order solver on the same vessels for context, optional INT8 check → `reports/onnx_benchmark.{md,json}`.
- `vstent export-surrogate --run runs/<name>`: export + parity + benchmark; exit code 1 if a target is missed.
- `RawInputs.subset`; `NORM_STATS` moved to `features.py` so the runtime does not import torch.
- 10 new tests (1 slow): files and provenance, opset / metadata / dynamic axis, parity, batch sizes 1 / 7 / 33 with chunked = serial, profile prediction vs torch, metadata mismatch rejected, subset, report, CLI.

**Metrics** (from `reports/onnx_benchmark.md`; model from run `runs/unet1d-20261028-125810`, exported and reported at commit d9c8d1e, clean; AMD Ryzen 7 6800H)
- Parity: max |FFR_torch − FFR_onnx| = **5.4e-07** over 1,000 test vessels, all 256 points (target < 1e-4).
- Single vessel, end-to-end median: **1.40 ms** (target < 10 ms).
- Batch of 1,024, end-to-end median: **204.7 ms** (target < 300 ms). The same batch in one ONNX call: 582.4 ms (missed the target; see open issues).
- Context: the reduced-order solver computes the same 1,024 vessels in 17.3 ms. No speed-up over the solver is claimed.
- Optional INT8 (unsigned weights): distal MAE 0.0030 vs 0.0013 (+0.0017, within the 0.002 allowance) but only 1.03× faster → not kept; the app uses FP32.
- Checks: ruff clean; `pytest -m "not slow"`: 88 passed; full suite 91 passed.

**Open issues / observations**
- The first benchmark (one multi-threaded ONNX call on the whole batch) missed the batch target at ≈ 0.57 s. Profiling showed the model is memory-bound at batch 1,024 (≈ 43% of the time in normalisation / element-wise ops; no gain beyond 4 threads). Chunked parallel inference fixed it without touching the model (D17). The cost is single-vessel latency, up from ≈ 0.95 ms to ≈ 1.4 ms.
- The TorchScript exporter is deprecated in torch; it works with the pinned torch 2.14.1. Moving to the dynamo exporter means opset 18 and the `onnxscript` dependency (D16).
- Signed INT8 weights were about 10× slower in an exploration run; only unsigned weights are checked (D18).
- Timings are wall-clock on one laptop and vary by a few percent between runs (e.g. FP32 batch 205 ms in the main table vs 222 ms in the INT8 comparison pass).
