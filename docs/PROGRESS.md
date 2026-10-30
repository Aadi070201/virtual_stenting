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

---

## 2026-10-28 — M5: ARCADE segmentation, 2D QCA and the end-to-end pipeline

**Done**
- **Data:** `vstent data arcade` streams Zenodo 10390295 `arcade.zip` (451.6 MB), verifies MD5 `c5b1973a…`, extracts with path checks and writes `SOURCE.json`. Licence verified from the Zenodo record: **CC0 1.0**. Structure, categories, duplicates and cross-task overlap are documented in `docs/DATA.md` (D19). Masks are the union of the `syntax` segment polygons (pycocotools).
- **Segmentation:** `imaging/segment.py` (shared pre-processing, augmentation, Dice; torch-free); `seg_train.py` (smp U-Net resnet34, ImageNet weights, BCE + Dice, AMP, resumable via `vstent.checkpoints`, augmentation seeds follow the checkpointed generator; D20); `seg_eval.py` → `reports/seg_eval.md`; `seg_export.py` → `models/seg/<run>/seg.onnx` + `reports/seg_onnx.md`; `SegmentationRunner` in `inference/onnx_runtime.py`.
- **QCA:** `imaging/centerline.py` (clean mask → skeleton → 8-connected graph → spur pruning < 10 px → snapping → shortest path → Savitzky–Golay → 1 px resampling; longest-path auto-selection, wider end first). `imaging/qca.py` measures the 0.5-level width along normals, with sub-pixel interpolation, a one-sided branch rule, fallbacks, end trimming, a two-sided robust reference and capping, and a 0.30 mm/px default flagged as assumed (D21). `imaging/synthetic.py` renders test tubes.
- **Pipeline:** `inference/pipeline.py` (torch-free) runs image → probability → QCA → solver and surrogate with the same Pa, Pv and R_micro. It warns about an assumed calibration, many fallbacks or capped points, R_micro at its floor, out-of-range surrogate inputs, and a surrogate FFR below Pv/Pa (D22). `inference/figures.py` draws an analysis figure; `inference/gallery.py` runs the gallery.
- **CLI:** `data arcade`, `train-seg` (`--resume`), `eval-seg`, `export-seg`, `analyse <image> [--start R C --end R C]`, `qca-gallery`.
- `vstent/plotting.py` holds the shared chart style (surrogate eval moved onto it).
- Tests: 127 in total (124 fast + 3 slow). Covered: downloader (local `file://`), fake-ARCADE loader, pre-processing and augmentation, exact segmentation stop/resume, segmentation export parity and eval report, QCA targets on 5 synthetic tubes, branch, crossing and off-vessel cases, robust reference, end trimming, orientation, pipeline, the `analyse` CLI end-to-end with a stand-in ONNX segmenter, and the gallery.

**Metrics** (from committed reports)
- `reports/seg_eval.md` (run `seg_unet_r34-20261028-141743`, trained at 0d4afee, evaluated at 226fc57, both clean):

  | Split | Mean Dice | Median | Min | Pixel precision | Pixel recall |
  |---|---|---|---|---|---|
  | val | **0.8325** (target ≥ 0.75, stretch 0.80: met) | 0.8473 | 0.62 | 0.841 | 0.825 |
  | test | **0.7925** | 0.8183 | 0.21 | 0.836 | 0.768 |

  Training: early stop at epoch 53 (best 38), 27.0 min on the RTX 3060 Laptop GPU (≈ 30 s/epoch).
- `reports/seg_onnx.md`: parity max |prob_torch − prob_onnx| **1.8e-05** on 16 val images (< 1e-4); 97.7 MB; CPU ≈ 104 ms per 512 × 512 image.
- QCA synthetic tests: all 5 configurations meet the plan's targets (median diameter error ≤ 0.7 px, MLD within 3 px, %DS within ±7 points).
- `reports/qca_gallery.md` (20 `stenosis/val` images, auto-selected longest paths, 0.30 mm/px assumed):
  - 20/20 analysed end-to-end;
  - the point of maximum %DS falls inside an annotated stenosis box in 8/20, the MLD in 2/20;
  - 19/20 paths lie outside the surrogate's training range (length);
  - median |distal FFR solver − surrogate| 0.033.

**Open issues / observations**
- **Test Dice is lower than val** (0.79 vs 0.83), with a long low tail (minimum 0.21). ARCADE annotates only the SYNTAX segments, so correctly segmented side branches count as false positives (visible in `seg_examples.png`).
- **Pixel size is unknown** for ARCADE: 0.30 mm/px is assumed, so absolute diameters, lengths and FFR are illustrative.
- **Real images needed four QCA fixes:**
  1. Auto-selected paths had to be oriented proximal-first.
  2. End trimming removes the mask-tip artefact.
  3. One-sided branches and centreline points that left the vessel needed their own rules.
  4. A two-sided robust reference and capping stop overlap spikes from inflating the reference (16.6 → 2.3 mm on val 107) or acting as fake stenoses.

  All are documented in D21. Two problems remain:
  - crossing vessels can leave thin bridges in the mask, which read as severe narrowings (val 107, MLD 0.33 mm);
  - the wider-end-first rule can pick the wrong end when both ends are thin.
- **Auto-selected paths are long** (68–265 mm) and run into thin distal branches. With constant flow and no side branches the model then overstates pressure losses (val 10: 31% max DS, FFR 0.06). User-selected single segments (the app in M7) are the intended use.
- **Surrogate out of range:** it extrapolates on most real paths (length > 80 mm) and once went below Pv/Pa before the warning was added. Retraining on longer and larger synthetic vessels would be a Phase 2 or M6+ consideration.
- The superseded first M3 run folder (`runs/unet1d-20261028-091227`) is still on disk.

---

## 2026-10-28 — M6: virtual stent and plan optimiser

**Done**
- `planning/stent.py`: `Stent`, `stent_weight`, `stented_radius`, `apply_stent(s)`. The lumen is never reduced, edges are half-cosine blended inside the stent ends, multiple stents are supported, and stents outside the vessel are rejected (D23).
- `solve_ffr_batch(..., r_micro=...)`: post-PCI evaluation with the pre-PCI R_micro, Pa and Pv (API addition only).
- `configs/stent_catalog.yaml` (generic: 2.25–4.00 mm in 0.25 mm steps; 8–38 mm) and `configs/planner.yaml`.
- `planning/optimizer.py` (D24):
  - candidates covering a lesion MLD on a 0.5 mm grid;
  - vectorised constraints: inside the vessel, diameter ≤ 1.1 × distal reference, DS < 30% over 2 mm landing zones;
  - ranking: shortest plan reaching FFR ≥ 0.90, else the highest FFR;
  - up to two stents (per-length pools per lesion, deduplicated, non-overlapping) and a Pareto front;
  - surrogate or solver backend; surrogate plans are re-checked by the solver.
- `planning/report.py` + `vstent plan-benchmark` → `reports/planner_benchmark.md`; `vstent analyse <image> --plan`.
- **Runner tuning for the 5,000-candidate target:** `SurrogateRunner` now defaults to chunks of 8 on all logical CPUs and pre-processes batches of ≥ 512 vessels in parallel (bit-identical results). The M4 report was regenerated with these defaults: batch of 1,024 now 188 ms (was 205 ms), single vessel 1.53 ms.
- Tests: 19 new planning/CLI tests, including:
  - focal lesion: the plan is the shortest feasible one reaching the target (brute force over all candidates);
  - tandem lesions: one long or two stents;
  - the two-stent optimum against exhaustive pairs;
  - every constraint checked on every returned plan;
  - no lesion; unreachable target; Pareto monotonicity; surrogate backend; solver throughput.

  Full suite: 145 passed.

**Metrics** (`reports/planner_benchmark.md`, commit 8c89eb0, clean; AMD Ryzen 7 6800H)
- 5,000 candidates (stented radii + evaluation): **surrogate 0.959 s median** (max 1.024 s over 7 runs; target < 1 s); solver 0.154 s.
- 100 synthetic test vessels with pre-PCI FFR < 0.80:
  - both backends reach FFR ≥ 0.90 in 84/100;
  - the solver confirms 82 of the surrogate's 84 plans;
  - median |surrogate − solver| post-PCI FFR of the chosen plan: 0.0003;
  - same total stent length in 81/84;
  - 25 two-stent plans;
  - planning time per vessel: 0.09 s median (0.35 s p95) with the surrogate, 0.01 s with the solver.
- On ARCADE `stenosis/val` 110 (assumed 0.30 mm/px): 588 feasible candidates and 188 pairs in 0.17 s; best plan 3.00 × 12 mm + 2.75 × 15 mm, surrogate 0.907 (solver 0.948).

**Open issues / observations**
- **Narrow throughput margin:** the 5,000-candidate target is met by about 4%, and the slowest of 7 runs took 1.02 s. Real plans evaluate far fewer candidates (median 382 feasible), so planning itself takes ~0.1 s.
- **Two surrogate plans fail the solver check:** for 2 of 84 vessels the surrogate predicts ≥ 0.90 but the solver gives < 0.90. The app should show the solver check next to the surrogate value (the planner already returns it).
- **Unreachable targets:** 16/100 vessels cannot reach 0.90 under the constraints, e.g. disease in the landing zones, more than two lesions, or narrow distal references limiting the diameter. The planner then returns the highest-FFR plan and says so.
- **Pairs are heuristic:** pools hold the best candidates per stent length, not an exhaustive search; overlapping stents (lesions longer than 38 mm) are not supported.
- **Tie-break:** at equal length the highest FFR wins, so the largest allowed diameter (≤ 1.1 × distal reference) is chosen. A sizing-to-reference rule could be considered with clinical input.

---

## 2026-10-28 — M7: Streamlit app

**Done**
- `app/core.py` (no Streamlit), covering:
  - model loading (newest exports, or environment overrides);
  - PNG / JPG / DICOM reading (DICOM: first frame, `ImagerPixelSpacing` with a magnification note);
  - synthetic presets (focal, tandem, random from the synthetic sampler);
  - pre-PCI and post-PCI evaluation (post-PCI keeps R_micro, Pa, Pv);
  - Plotly figures (diameter with reference / lesions / MLD / stents; FFR from solver and surrogate with their difference and the 0.80 line);
  - the image overlay (mask, centreline, points, stents);
  - the case report (JSON with disclaimer, assumptions and model provenance, no pixels; PNG of the plots).
- `app/streamlit_app.py`:
  - a permanent research-use banner;
  - synthetic and image modes (upload or ARCADE example; click points / auto path / manual form; calibration choice);
  - Pa / Pv inputs; virtual-stent controls (catalogue length and diameter, start slider) with instant post-PCI FFR;
  - "Suggest plan" (top 3 with the solver check, plans drawable, Pareto front);
  - case report download;
  - cached segmentation and QCA, and its own server rerun time shown.
- `vstent app` (localhost by default, `--port`, `--address`, `--headless`); `vstent app-latency` → `reports/app_latency.md`. `inference/models.py` locates exported models (shared with the CLI).
- Tests: 9 new in `tests/test_app.py`, including `AppTest` smoke tests of synthetic mode end to end (stent, slider < 300 ms, plan, plan display, report, random preset) and image mode (example, auto path, stent, reset), plus the regression test below. Full suite: 153 passed.
- Checked in a real browser: Edge, driven by a temporary Playwright install that was not added to the project. Synthetic mode, a stent + suggested plan, and image mode all render with no console errors.

**Metrics** (`reports/app_latency.md`, commit 9eebddb, clean; real models; AMD Ryzen 7 6800H)
- Stent-control updates (26 per mode: start slider, length, diameter), server rerun time:
  - synthetic: median **50 ms**, max 130 ms;
  - image (ARCADE `stenosis/val` 1, after segmentation): median **76 ms**, max 80 ms.
  - Target < 300 ms: met. Including the test harness: 72 / 98 ms median.
- One-off image costs, cached afterwards: segmentation ≈ 0.12 s; automatic path + QCA + both FFR models ≈ 0.2 s.

**Open issues / observations**
- **Bug found in the browser, now fixed:** after switching from a synthetic case to an image, the previous stent was applied while the toggle looked off. A new vessel now resets the stent controls and plan selection; regression test added.
- The click-to-select component cannot be exercised by `AppTest`; the same path is reachable through the auto-path button and the manual form, which are tested.
- Browser rendering time is not part of the latency numbers.
- The app runs without a DICOM viewer: multi-frame DICOM uses frame 1, and no frame selection or ECG gating is offered.

---

## 2026-10-28 — M8: reports, docs, demo; MVP complete (v0.1.0)

**Done**
- **`vstent demo`** (`reporting/demo.py`): the tandem-lesion preset end to end (profile → FFR → plan optimisation → best plan → post-PCI FFR) → `reports/demo/` (Markdown, JSON case report, figure). It uses the exported surrogate if present, else the solver, and says so.
- **`vstent qca-synthetic`** (`imaging/qca_validation.py`): the five synthetic QCA cases as a report; `tests/test_qca.py` imports the same cases.
- **`vstent mvp-report`** (`reporting/mvp.py`) → `reports/mvp_report.md`, assembled only from the generated JSON reports:
  - the target table, facts and provenance per source;
  - fixed limitations;
  - the Phase 2 list parsed from the plan.

  Missing reports are listed rather than skipped.
- **`docs/INVENTION_NOTES.md`:** dated notes, with commit references, on the monotone-increment head, the constrained optimiser with solver re-check, and the two-sided QCA reference with capping, plus the planned sim-to-real idea. Each lists the prior art to check, with a clear "not legal advice" and disclosure caution.
- **README** rewritten: screenshots (app in synthetic and image mode), setup, what works without models, full reproduction steps with timings, the command table, app guide and limitations.
- **Fresh-clone test** (`git clone` into a new folder, `uv sync --all-extras`, README steps):
  - checks: 157 passed;
  - `physics-report`, `qca-synthetic` and `demo` all run; the demo falls back to the solver without trained models;
  - `vstent app` serves and works in synthetic mode with solver planning (driven in Edge: no exceptions, no console errors).

**Bugs found and fixed in M8**
- `vstent physics-report` crashed when its output went to a Windows pipe in code page cp1252 ("→" in a check name). The CLI now replaces unencodable characters instead of crashing, for every command; regression test added.
- A real-browser run hit "partially initialized module 'pandas'". Plotly and `st.dataframe` import pandas / pyarrow lazily on first use, and a click during that first import interrupted it. The app now imports both up front; re-checked on a fresh server.
- Manual stents could be much larger than the vessel without warning (e.g. 3.0 mm in a ~1 mm distal segment, which the model then sees as a sudden widening). The app and the report now warn when a stent exceeds the planner's 1.1 × distal-reference rule.
- The MVP report showed no creation time for the demo (field location); fixed and tested.

**Final metrics** (all from `reports/mvp_report.md`, assembled at a clean commit; every source report was generated at a clean commit)
- **21 of 21 acceptance targets met, M1–M7.** Highlights:
  - surrogate test distal-FFR MAE 0.0013 (≤ 0.015);
  - ONNX parity 5.4e-07;
  - segmentation val Dice 0.8325 (≥ 0.75; stretch 0.80 met); test 0.7925;
  - QCA synthetic worst-case median diameter error 0.10 px (≤ 0.7);
  - planner, 5,000 candidates: 0.959 s (< 1 s);
  - app stent-control updates ≤ 179 ms (< 300 ms).
- Demo: distal FFR 0.705 → 0.976 (solver) with 3.25 × 12 mm + 3.25 × 12 mm; planning 0.37 s.
- Tests: 162 in total (159 fast + 3 slow).

**Open issues / not done**
- **M3 stretch:** the 5-seed ensemble with uncertainty was not done.
- **Thin margins:**
  - the planner's 5,000-candidate time is ≈ 4% under target;
  - the app's worst update varies between runs (80–180 ms in image mode).
- **Physics limits:** all of them are recorded in the MVP report and `docs/PHYSICS.md`, in particular single vessel without side branches, synthetic-only surrogate validation, an assumed pixel size on ARCADE, and 2D overlap artefacts.
- **Superseded run:** the first M3 run folder (`runs/unet1d-20261028-091227`) is still on disk (git-ignored).
- **Phase 2** has not been started; it needs explicit approval (CLAUDE.md).

---

## 2026-10-28 — Phase 2 approved and planned

The human approved Phase 2 ("go ahead with phase 2"). The backlog was turned into milestones P1–P8 with tasks and acceptance criteria in `docs/PLAN.md`; the ordering rationale is D27. CLAUDE.md now records the approval and that [needs human] items wait for the human. No Phase 2 code yet.

**Needs the human, in time:**
- **P6:** register for ImageCAS (Kaggle) and / or ASOCA (grand-challenge.org) and accept their licences; decide on a compute budget for 3D CFD.
- **P4:** a hospital partner and ethics approval for comparison with wire FFR.
- **P7:** de-identified, licensed cine DICOM examples, if available.
- **P8:** a regulatory adviser to review the drafts.

---

## 2026-10-30 — P1: Uncertainty and grey-zone flagging (done)

**What was done**
- **Ensemble:** 5 surrogates with the M3 configuration and seeds 0–4, trained with the new resumable `vstent train-ensemble` (D28).
  - Training was paused at the human's request during seed 2 (after epoch 31) and resumed exactly from there.
  - Total training time: 4,857 s (≈ 81 min) on the RTX 3060 Laptop GPU.
  - Best val distal MAE per member: 0.0013–0.0014.
- **Discarded first ensemble:** the first ensemble (`runs/unet1d_ens5-20261030-100555`) was discarded. Its seed 3 early-stopped at epoch 21 on high-learning-rate validation noise (val distal MAE 0.0030). Early stopping now starts at epoch 40 (D31). The decision used `val` only.
- **ONNX export** (`vstent export-ensemble`):
  - one model with mean and SD outputs;
  - `screen.onnx` (member 0) for the planner's candidate screening, after which the final plans are re-scored by the ensemble (D28);
  - parity: mean 2.7e-07, SD 3.6e-07, screen 5.4e-07.
- **Split-conformal 90% interval** for distal FFR, calibrated on `val` with an ensemble-normalised score (D29).
  - The σ floor was lowered from 0.001 to 0.0002 (the 10th percentile of val σ). The median val σ is 0.00073, so 0.001 had made the score half absolute. Coverage by σ quintile on val went from 77–98% to 85–94%.
  - The floor was chosen on `val` only, before any `test` evaluation.
- **Input-uncertainty Monte Carlo** through the solver: 500 fixed-seed draws per case (D30).
  - Sources: pixel size ±10% (only when assumed), QCA diameter error, hyperemic velocity ±0.10 m/s, Pa ±20 / Pv ±3 mmHg.
  - The QCA error model (SD 0.107 px, half-correlation lag 0.62 px) is now measured by `vstent qca-synthetic`.
- **Grey-zone flags:** a flag is raised when either interval contains 0.80 (pre-PCI) or the planner target 0.90 (post-PCI). Flags and intervals appear in the app (FFR plot bars, results, plan table), the case report JSON, `vstent analyse` and the demo.
- New commands: `train-ensemble`, `export-ensemble`, `uncertainty-report`. `vstent stop` and `vstent checkpoints` handle ensembles.

**Bugs found and fixed**
- **App random synthetic preset (from M7):** pre-PCI FFR used the default velocity of 0.35 m/s, although the caption said the velocity came from the sampler. Fixed and tested.
- **Benchmarks on an ensemble model folder:** `vstent plan-benchmark` (and the ONNX benchmark) crashed. They read the ensemble's `config.yaml` as a single-model config. A shared `surrogate_config()` now reads the member config. Regression tests added.

**Code review of P1 (after the first reports), and fixes**

A review of the P1 code found ten issues; all were checked against the code before changing anything.
- **Lost stop request (would break the stoppable-training rule):** `vstent stop <ensemble>` sent while a member was still loading data was lost, and that member then trained every epoch. Members now watch the ensemble's own STOP file. Regression test added; it fails on the old code.
- **Interval labels:** plan model intervals were labelled with the config's level, not the level they were calibrated at. They were the same (90%), so no output was wrong. Each interval now carries its own level.
- **Pareto front vs plan table:** with the ensemble, the Pareto front kept the screening model's FFRs while the plan table showed the ensemble's. The front is now re-scored by the ensemble, which adds about 50 ms per plan search.
- **Stent edge:** the input Monte Carlo and the app's manual stents used a hard-coded 1 mm edge, the planner `edge_mm` from its config (also 1 mm). Everything now uses the planner's value. A test proves the Monte Carlo reproduces the planner's solver value exactly, also at 3 mm; the old code was off by 4.6e-4.
- **Positions:** stent and lesion positions assumed profiles start at s = 0. Every producer already did; `VesselProfile` now enforces it.
- **Calibration split (research rigour):** conformal calibration used `val`, which had chosen each member's best epoch and the σ floor, so the 90% guarantee did not formally hold. A new `cal` split (10,000 vessels, seed 1005, used for nothing else) was added. Train, val, test and ood are byte-identical (same shard hashes). Decision D29 updated.
- **Smaller fixes:**
  - one shared `load_uncertainty` / `model_interval` instead of three copies;
  - removed dead code;
  - the report no longer runs 500 identical solves for one nominal value;
  - the report's plan summary handles an empty set.

**Metrics** (final; all reports generated at clean commits after the fixes)
- **Acceptance targets, from `reports/uncertainty.md`:**
  - conformal coverage on test: **89.72%** (target 88–92%); `cal` 90.01%, `val` 89.56%;
  - ensemble-mean distal MAE on test: **0.00099** vs 0.00129 for the single M3 model (target ≤);
  - app worst stent-control update: **211 ms** server time (< 300 ms).
- **OOD (no target):**
  - coverage 77.96%; an absolute score would give 44.80%;
  - distal MAE 0.00376 (ensemble) vs 0.00391 (single).
- **Grey zone (model interval):** 0.65% of test vessels are flagged. 14 of the 16 test vessels that the surrogate and solver put on opposite sides of 0.80 are flagged.
- **Input intervals, 500 test vessels:**
  - median width 0.128 (synthetic geometry) and 0.129 (image-derived, assumed pixel size);
  - flagged at 0.80: 26.8% and 28.6%.
- **Planner benchmark** (ensemble, 5,000 candidates): 0.984 s (< 1 s). The same code measured 0.874 s an hour earlier, so the margin depends on background load (browser and chat apps were running). 83 of 100 plans reach FFR ≥ 0.90; the solver confirms 81.
- **App latency:** median 69 ms (synthetic) and 116 ms (image), up from 50 / 77 ms (ensemble plus Monte Carlo).
- **Demo:**
  - pre-PCI 0.705: input interval 0.622–0.791, model interval 0.704–0.708;
  - post-PCI 0.976: input interval 0.967–0.984, model interval 0.975–0.976;
  - no grey-zone flag.
- **Tests:** 190 fast tests and 3 slow tests pass.
- **Seed 0 check:** member seed 0 reproduces the M3 single model (same seed and configuration: test distal MAE 0.00129 for both).

**Open issues**
- **Coverage falls with stenosis severity.** On test it goes from 99% (0–30% DS) to 85.0% (70–100% DS). OOD coverage is 76–79% in every bin. The guarantee is marginal; per-bin coverage is not guaranteed.
- **Input intervals are wide** (median ≈ 0.13), mostly because of the ±0.10 m/s velocity range. With these ranges, 85–91% of vessels with FFR 0.70–0.80 are flagged. The ranges are the synthetic sampling ranges, not measured patient-level uncertainties.
- **QCA error model:** it comes from idealised rendered tubes, so real QCA error is likely larger.
- **Thin planner margin:** 0.87–0.98 s against the 1 s target, depending on background load.
- **`reports/mvp_report.md` is still the v0.1.0 snapshot.** Its planner (0.959 s) and app-latency numbers predate the ensemble; the current values are in the source reports.
- **Old run folders:** the discarded first ensemble is still on disk (git-ignored).

---

## 2026-10-30 — Code review of the MVP (M0–M8) and fixes

An independent read-only review covered physics, data, surrogate, imaging, planning, inference, app and reporting. Each finding was checked against the code before any change, and each fix has a regression test that fails on the old code.

**Fixed**
- **Planner vs app on image paths (medium).** On image paths the planner's solver used a 256-point resample while the app solved the profile itself. A plan's "Solver check" could therefore differ from the app's post-PCI value for the same stents (0.765 vs 0.752 on one ARCADE image). The planner's solver now runs on the profile's own points (D34). Synthetic results are unchanged.
- **Stale plans (medium).** The app reused plans and stent controls after the pixel size, Pv or v_hyp changed. With a shorter path this could crash (a stent outside the vessel). Plans and stents are now keyed on the geometry and all pressures.
- **Extrapolation warnings (low–medium).** In image mode the warnings used the config pressures instead of the user's, and synthetic mode never checked ranges. Both now use the user's Pa / Pv.
- **Plan tie-break (low).** Floating-point stent lengths broke ties: 12 + 26 mm ≠ 38 mm. Lengths are now rounded to 1 nm.
- **Image loading (low).** The CLI and the app loaded 16-bit images differently. There is now one loader (`imaging/image_io.py`).
- **Interrupted saves (low).** An interruption between `best.pt` and the resumable state could leave them inconsistent. `best.pt` is now written after the state, and resume repairs it (both trainers).
- **Segmentation report (low).** It showed only the val Dice, and val also chose the model. A held-out test row was added: 0.7925 meets the 0.75 target but not the 0.80 stretch.
- **MVP report (low).** Its "Phase 2 not started" heading now shows each milestone's status from the plan.

**Checked and found correct:** solver vs PHYSICS.md; split independence and normalisation (train only); resumable training; ARCADE has no image overlap between splits; one pre-processing path for segmentation; QCA geometry; planner constraints.

**Not changed (documented):** the surrogate still resamples image paths to 256 points (D34). This only downsamples paths longer than about 77 mm at 0.30 mm/px, already outside the surrogate's training range.

**Reports regenerated:** `seg_eval.md` (same numbers, honest labels). App latency, planner benchmark and demo are unaffected for synthetic cases; they will be regenerated together at the end of P2.

---

## 2026-10-30 — P2: Vessel trees with side branches (done)

**What was done**
- **Tree physics** (`physics/tree.py`, D32, D33):
  - Each branch keeps one continuous radius profile with the M1 coefficients, and the flow drops at every take-off.
  - Outlet resistances are calibrated so the stenosis-free tree splits flow by Murray's law (k = 2.33, `murray_exponent` in `configs/physics.yaml`).
  - A batched Newton solve on the outlet flows gives FFR at every sample of every branch.
- **Synthetic trees** (`vstent synth-tree`, `configs/synth_tree_v1.yaml`, `reports/tree_dataset_card.md`):
  - a main vessel with 1–4 side branches, stepping down after each take-off;
  - bifurcation and ostial lesions;
  - 54,000 trees in 13.4 s;
  - the main vessel's single-path FFR is stored for comparison.
- **Tree checks** (`vstent tree-report`, `reports/tree_physics.md`), all passing:
  - a one-branch tree reproduces M1 to 4e-16;
  - mass is conserved and pressure continuous at every junction (≤ 3e-16 on 1,000 test trees);
  - outlet equations hold to 2e-14;
  - the Murray split holds to 9e-16 (one tree whose outlet resistance hit its floor is excluded and counted);
  - a side branch raises distal main FFR in all 9 controlled cases (+0.007 to +0.126);
  - a side branch with almost no flow inside a lesion's expansion run changes main FFR by only 2e-5.
- **Throughput:** **3,474 trees/s** on one process (target ≥ 1,000).
- **Trees from the image** (`imaging/qca_tree.py`, D35):
  - shortest-path tree on the skeleton, with side branches of the chosen path;
  - each side branch is measured from the parent's wall, so ostial lesions are kept.
- **Tree analysis:** `vstent analyse --tree`, with an optional `--stent START LEN DIA --outlet K` on any root-to-outlet path (outlet resistances kept from before the stent), and a tree figure.
- **Tree gallery** (`vstent qca-gallery --tree`, `reports/qca_tree_gallery.md`):
  - the same 20 ARCADE images and paths as the single-path gallery;
  - side branches were found in 14 (24 modelled, 10 too short, thin or deep);
  - median change of main-path distal FFR where side branches were found: **+0.149**.

**"Done when" (PLAN.md)**
- All sanity checks pass: yes.
- Throughput ≥ 1,000 trees/s: yes (3,474).
- Extreme single-path cases reported before and after: 3 cases with FFR < 0.30 and no lesion ≥ 70% DS. They remain extreme as trees:
  - in 2 of them no side branch was segmented;
  - in val_10 the low value comes from the path's thin distal tail, downstream of its only side branch.

**Also in this session:** the code review of P1 and of the MVP, with all fixes (see the two entries above), and the `cal` split.

**Reports regenerated after the fixes (clean commits):**
- app latency: worst 224 ms (< 300 ms);
- demo: unchanged;
- uncertainty: test coverage 89.72%;
- **planner benchmark: 1.345 s (fail; target < 1 s)**. It was measured while a game and other apps kept the CPU about 70% busy. The solver in the same run was also 48% slower (0.226 vs 0.153 s). The code path is unchanged, and earlier runs measured 0.874–0.984 s. It needs re-measuring on an idle machine; the thin margin was already an open issue.

**Open issues**
- **Few side branches in real images.** The segmentation model, trained on ARCADE's SYNTAX-segment masks, rarely segments small side branches, so real-image trees have few of them. Better segmentation (annotated side branches, or P8's synthetic angiograms) would help more than physics.
- **Automatic start point.** The automatic start (the wider end of the longest path) is not always the ostium. In val_107 the vessel to the true inlet became a "side branch". Users set the start point in the app and the CLI.
- **Model simplifications:** junction losses are neglected, the healthy split follows Murray's law, and skeleton junctions sit up to about one vessel width downstream of where the centrelines cross.
- **Not yet available for trees:** surrogate, intervals, planner, and the app (P3).
- **Tests:** 223 fast and 3 slow pass.

---

## 2026-10-30 — P3: Tree surrogate (graph network) + ONNX (done)

**What was done**
- **Tree surrogate** (`surrogate/tree_model.py`, D36):
  - the M3 FiLM 1D U-Net applied to every branch (shared weights), with 3 rounds of message passing between branch summaries (children, parent, whole tree) in plain PyTorch;
  - a monotone tree head: FFR = 1 at the inlet, continuous at every junction and non-increasing along every path, by construction;
  - 1.31 M parameters;
  - one pre-processing path (`surrogate/tree_features.py`) for training, ONNX and the app.
- **Training** (`vstent train-tree-surrogate`): resumable, STOP / `--resume` (tested bit-exact). 40,000 synthetic trees, 50 epochs, 23 min on the RTX 3060 Laptop GPU. Best validation outlet MAE 0.0043 at epoch 50.
- **Evaluation** (`vstent eval-tree-surrogate`, `reports/tree_surrogate_eval.md`): against the tree solver on test and ood trees.
- **ONNX** (`vstent export-tree-surrogate`, `reports/tree_onnx.md`):
  - padded graph (5 branches × 256 samples, dynamic batch), 5.3 MB;
  - `TreeSurrogateRunner` (torch-free).
- **Tree planner** (`planning/tree_planner.py`, D37):
  - stents along one root-to-outlet path, possibly into a side branch;
  - every outlet's FFR with the pre-PCI outlet resistances;
  - the tree solver evaluates candidates through `solve_tree_variants` (shared topology); the tree surrogate is optional, with a solver re-check.
- **Benchmark:** `vstent plan-benchmark --tree` (`reports/tree_planner_benchmark.md`).
- **Integration:**
  - `vstent analyse --tree --plan` / `--stent`, showing the tree surrogate's outlet FFR next to the solver's;
  - a tree mode in the app: side branches on the image, FFR along every path, a stent or plans on a chosen path, tree surrogate column, JSON tree report.

**"Done when" (PLAN.md)**
| Target | Result |
|---|---|
| Node-wise FFR MAE ≤ 0.015 (test trees) | **0.0038** |
| Distal (every outlet) MAE ≤ 0.015 | **0.0043** |
| ONNX parity < 1e-4 | **7.2e-07** |
| Tree planner, 5,000 candidates < 1 s (target set from P2, D37) | **0.136 s** |

**Further results**
- **OOD trees** (100–140 mm main vessel, 4 side branches), errors about 2× test: node MAE 0.0090, outlet MAE 0.0094.
- **Agreement with the tree solver at 0.80 (test outlets):** sensitivity 0.992, specificity 0.994.
- **For context:** ignoring side branches (main vessel alone, single-vessel solver) would err by 0.152 at the main outlet (0.326 on ood).
- **Tree surrogate speed:** 3.3 ms for one tree on the CPU, and 5.4 s for 5,000 planning candidates vs 0.136 s for the tree solver. It is not faster than the reduced-order model and is not claimed to be.
- **Planning study** (first 100 test trees with main FFR < 0.80):
  - **56 reach FFR ≥ 0.90.**
  - **30 get no feasible plan**: in 24 of them every lesion lies in a vessel thinner than 2.05 mm, where no catalogue stent fits. The synthetic main vessels narrow after each take-off.
  - 14 have a best plan below the target.
  - Planning the main vessel as a single vessel (M6) gave a longer plan in 17 of 56 compared trees and never a shorter one.
  - The tree surrogate backend chose the same total length in 67 of 100 trees; the solver confirmed 56 of its plans.

**Open issues**
- **Rare large surrogate errors.** 56 of 12,817 test outlets (0.4%) are off by more than 0.05; the largest is 0.23. All are in severely diseased trees, often a healthy side branch behind a severe main-vessel lesion. One prediction was negative (−0.05): like M3's, the tree head is not bounded below by Pv/Pa. The CLI and the app now warn when the surrogate falls below Pv/Pa. A bounded head or a tree ensemble with intervals (P1 for trees) would address it.
- **Stenting small vessels.** Many synthetic trees have lesions in main-vessel segments thinner than the smallest catalogue stent allows. That is realistic, but it limits the planning-study success rate. The catalogue and sizing rules are unchanged.
- **Trees with more than 5 branches** (possible from images) fall back to the solver for the surrogate.
- **No uncertainty intervals for trees yet.**
- **App latency** was not re-measured after adding tree mode. Its single-path code path is unchanged, and tree mode is a separate branch of the script.

## 2026-10-31 — P4: Clinical-validation readiness (done; real data waits for the human)

**What was done**
- **Protocol template** (`docs/validation/PROTOCOL.md`):
  - retrospective diagnostic-accuracy design (STARD 2015) against wire FFR (iFR secondary);
  - primary endpoint: per-vessel AUC for FFR ≤ 0.80, with sensitivity, specificity,
    Bland–Altman, feasibility and intention to diagnose as secondary endpoints;
  - blinding and locking, sample size, analysis plan, data protection, ethics and limitations;
  - every clinical choice in [brackets] for the investigators to confirm.
- **Data specification** (`docs/validation/DATA_SPEC.md`):
  - the data needed per vessel;
  - the DICOM attributes to keep, with tag numbers checked against pydicom's data dictionary;
  - a de-identification checklist (PS3.15 Annex E as the reference);
  - the two CSV files, with header-only templates (`docs/validation/templates/`);
  - the local-only rule.
- **`vstent validate`** (`src/vstent/validation/`, D38):
  - **Blinded step:** `--manifest cases.csv` runs the `vstent analyse` pipeline on every
    vessel, in single-path (solver, surrogate, P1 intervals) and tree modes.
  - **Locking:** it writes `predictions.csv` and locks it with its SHA-256 (plus the commit,
    model, config and manifest hashes).
  - **Scoring:** `--predictions --reference reference.csv` refuses changed predictions and
    reports:
    - the flow of vessels;
    - the primary analysis with analytic and patient-bootstrap intervals;
    - all index tests;
    - the grey zone and a hybrid strategy;
    - per-territory tables;
    - every vessel that could not be analysed, with the reason.
  - **Local-only guard:** case lists, images, masks, references and outputs inside the
    repository's tracked files are refused (`git check-ignore`).
  - **Other inputs:** an optional analyst-corrected mask per vessel; multi-frame DICOM
    `frame` selection (`image_io`).
- **Statistics** (`validation/stats.py`):
  - Wilson intervals;
  - DeLong AUC standard error (logit-scale interval);
  - Bland–Altman limits with their sqrt(3 s² / n) intervals and the proportional-bias slope;
  - Pearson r with a Fisher-z interval;
  - a patient-level cluster bootstrap.

  Tests check them against hand-calculated values, scipy's Mann–Whitney statistic and a
  simulation of the AUC's spread.
- **Sample size** (`vstent sample-size`, `reports/validation_sample_size.md`):
  - methods: Buderer for sensitivity / specificity, Hanley–McNeil for the AUC, Bland–Altman
    for the limits of agreement;
  - adjustments: a clustering design effect and non-analysable vessels;
  - a table of how the total changes with the assumptions.
- **Synthetic stand-in study** (`vstent validate-synthetic`, `reports/validation_synthetic.md`,
  D39):
  - 60 patients, 102 vessels (P2 trees, 0–3 side branches) rendered as X-ray-like
    projections;
  - contrast, edge enhancement and noise fitted to ARCADE centreline statistics;
  - stand-in "wire FFR" = the tree solver on the true geometry with sampled Pa, Pv and
    velocity;
  - a main run on the true lumen masks and a second with the segmenter.

**"Done when" (PLAN.md):** the protocol and data specification exist, and the validation tool
runs end to end on synthetic stand-in data. **Met.** Real data waits for the human (ethics
approval and a data agreement).

**Results on the stand-in study (synthetic; not a clinical validation)**
| Index test (masks run, 102 vessels, 60 patients) | AUC (patient bootstrap) | Sens. / spec. | Bias; limits of agreement | MAE |
|---|---|---|---|---|
| Tree, solver (primary) | 0.973 (0.946–0.993) | 0.894 / 0.909 | +0.008; −0.135 to +0.152 | 0.045 |
| Single path, solver | 0.960 | 0.936 / 0.873 | −0.022; −0.185 to +0.141 | 0.056 |
| Single path, surrogate | 0.966 | 0.936 / 0.855 | −0.023; −0.183 to +0.138 | 0.054 |

- **Grey zone (P1):**
  - 31% of vessels are flagged;
  - accuracy 0.957 when not flagged vs 0.781 when flagged;
  - a hybrid strategy (wire only when flagged) gives 0.971.
- **Error budget of the tree index test** (MAE):
  - full pipeline 0.045;
  - measurement error only (true pixel size, Pv and velocity) 0.034;
  - assumption error only (true geometry, fixed Pv and velocity) 0.027.
- **Where the measurement error comes from.** The QCA's inlet reference diameter sets the flow
  (Q = v_hyp · π r_ref(0)²); its error correlates with the FFR error (Spearman −0.66). The
  largest outliers (+0.36, +0.25) had the inlet reference underestimated by 44–47%. Tiny lumens
  (MLD 0.5 → 0.14 mm) and a missed side branch cause the others. MLD itself is measured to
  +0.01 mm (median).
- **The automatic segmenter on the renders:** only 38% of vessels can be analysed, so
  intention-to-diagnose accuracy is 0.275–0.284. This is a domain gap, not a clinical result: it
  persists with ARCADE-matched contrast because the renders lack the anatomy of clinical
  angiograms (catheter, other vessels, ribs). Closing it is P8's task.
- **Sample size with placeholder assumptions:** 286 vessels in about 220 patients, governed by
  the precision of sensitivity.

**Fixes found while building P4**
- **Points off the segmented vessel were silently snapped** (D40). A click 60–90 px beyond a
  partly segmented vessel measured a path half the length, with no warning. Points farther
  than the vessel's half-width + 5 px are now refused in single-path and tree QCA, and tree
  QCA no longer moves an end point onto the start's vessel. Regression tests fail on the old
  code.
- **The new local-only guard on Windows:** `git check-ignore` quotes backslash paths, so
  git-ignored data was reported as tracked. A test caught it; paths now go in as POSIX on
  stdin with NUL separators, and a git failure raises instead of passing silently.

**Open issues**
- **Real validation needs the human:** a clinical partner, ethics approval, a data agreement
  and a statistician for the assumptions (`docs/HUMAN_ACTIONS.md`).
- **The QCA reference-diameter estimate** is the largest measurement-error source in the
  stand-in study. A more robust proximal reference, or a user-adjustable one, is a candidate
  improvement; it is not done here.
- **No uncertainty intervals for trees.** The grey-zone analysis therefore uses the
  single-path intervals.
- **The segmenter's domain gap on rendered images** is left to P8.
- **App latency** is still not re-measured with tree mode (carried over from P3).
