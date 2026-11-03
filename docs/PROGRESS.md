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

## 2026-10-31 — P4b: Validation on public data (done; FFR ground truth waits for the human)

The human cannot obtain a hospital partner, so validation moved to public data with
independent ground truth (D41). No openly downloadable dataset pairs angiograms with invasive
FFR (searched 2026-10-31). ICADD (70 Indian patients, angiography + invasive FFR) must be
requested from its authors.

**What was done**
- **Datasets:**
  - `vstent data cadica`: CADICA, 42 patients, cine angiograms with expert lesion boxes and
    visual grades; CC BY 4.0; SHA-256 as published.
  - `vstent data fda-nozzle`: FDA benchmark nozzle, inter-laboratory wall pressures; public
    domain; pinned commit, per-file SHA-256.
- **`vstent physics-nozzle`** (`reports/physics_nozzle.md`, D42): the unchanged loss model on
  the exact nozzle geometry against every laboratory's net pressure loss.
- **`vstent stenosis-grading`** (`reports/stenosis_grading.md`):
  - automatic %DS at every expert lesion (6,161 boxes, 497 lesion tracks; path through the box
    from the skeleton, the product's QCA, median over keyframes) against the experts' grades;
  - development / test split by patient;
  - edge-method rule committed before any result (D43).
- **Intensity-edge QCA** (`imaging/qca.py`, `measure_diameters_intensity`): each lumen edge is
  the steepest intensity rise along the normal, guided by the segmentation, with a 5-point
  median along the vessel. It is an option for now; the product default is unchanged.

**Results**
- **Physics (FDA nozzle, laminar Re = 500, the coronary regime):**
  - sudden expansion: model 237 Pa vs measured median 390 Pa (labs 219–441), ratio 0.61;
  - conical diffuser: 237 vs 224 Pa, ratio 1.06.

  The model's loss depends only on the areas, while the measured losses differ by a factor of
  1.7 with the exit shape. Abrupt-exit lesions are likely underestimated, so their FFR would be
  too high. Transitional and turbulent Reynolds numbers: ratios 0.70–0.94. Nothing was fitted.
- **Stenosis grading, current product QCA (probability-map edges), test half** (208 / 214
  lesions measured, 21 patients):
  - weighted kappa 0.12 (0.06–0.19), AUC for ≥ 70% 0.69;
  - median measured %DS of 90–98% lesions: 32;
  - no lesion measured ≥ 70%;
  - total occlusions measured as open (10 / 11, median 6% DS).

  **The segmentation smooths over narrowings, so the current QCA compresses severity.** This
  is the most important product finding so far.
- **With intensity edges, test half:**
  - weighted kappa 0.22 (0.15–0.31), within one grade 71% (vs 54%), AUC for ≥ 70% 0.73
    (0.61–0.84), AUC for ≥ 50% 0.76;
  - median %DS of 70–90% lesions 53, of 90–98% lesions 48.

  On the development half: kappa 0.36, AUC for ≥ 70% 0.81. Better on both halves, but
  agreement is still modest and severe lesions are still measured low.
- **Why severe lesions read low** (inspected on development patients only): of the 4 lesions
  graded ≥ 70% but measured below 20%,
  - in 2 the narrowing lies outside the mask (an ostial lesion; a gap at a branch origin), so
    the measured path starts beyond it;
  - in 1 the path ran through two adjacent vessels;
  - in 1 the vessel is segmented but neither edge method finds the narrowing on that frame.

  Together with the compression seen in the probability edges, this points to the segmenter
  (trained on ARCADE's segment-level masks) as the main limit.

**Open issues and next steps**
- **Product default.** Make intensity edges the QCA default for automatically segmented
  vessels (analyst masks keep mask edges), then regenerate the image-based reports.
- **Segmentation at stenoses.** Train with lumen-accurate labels at narrowings: rendered
  angiograms with exact lumens (P8, sim-to-real). Total occlusions are bridged by the
  segmentation and measured as open; there is no occlusion detection yet.
- **Visual grades are not ground truth.** Visual and quantitative grading are known to differ,
  so part of the gap may lie in the reference.
- **Physics.** Quantify the abrupt-exit effect with CFD (P6). A literature developing-flow
  correction was not applied because its published form could not be verified.
- **Real FFR ground truth.** ICADD access requested by the human.

## 2026-10-31 — P4c: Lesion-accurate QCA (done; severity on real angiograms still compressed)

P4b showed that the automatic QCA compresses stenosis severity on real angiograms and reads
total occlusions as open. P4c tried to fix both. Methods were chosen on CADICA's development
patients only, with each rule committed before its results (D44, D45).

**What was done**
- **Intensity edges are the default** for automatic segmentations (D43); analyst masks keep
  their own edges. Reports state the edge method.
- **Imaging-chain lumen model** (`imaging/lumen_model.py`, `--lumen model`, the app's "Lumen
  measurement" selector): Beer–Lambert transmission, blur and the X-ray system's edge
  enhancement fitted to the straightened vessel; analytic sparse Jacobian, ~1 s per vessel.
  `vstent lumen-synthetic` tests it on rendered vessels with known lumens.
- **Mask gaps and occlusions:** skeleton ends facing each other across a gap are bridged; the
  bridge is measured from the image only; a stretch without a measurable lumen is reported as
  "critical stenosis or total occlusion" with an FFR warning.
- **Intensity-edge corrections (D45)**, found when the ARCADE gallery was regenerated: side
  branches only where the mask's outline shows one (the 2 × near-edge rule turned off-centre
  centrelines, one-sided lesions and gradients inside the lumen into deep narrowings);
  minimum-cost edge tracking along each wall; bridged points never count as branches.
- `vstent stenosis-grading --edges … --workers N` (parallel, same results as serial).
- Slider-latency tests collect garbage before each timed update: with the whole test suite
  loaded, one full collection took ~250 ms (the app itself ~60 ms).

**Results**
- **Rendered vessels** (`reports/lumen_synthetic.md`, both criteria PASS):
  - lesions ≥ 85% DS: model error 2.3 points, edges 30.7; FFR error 0.035 vs 0.53;
  - eccentric lumens: 5.5 vs 19.7 points;
  - healthy vessels: edges read 6.5 points of false narrowing (11.6 before D45).
- **ARCADE gallery, 20 whole paths** (`reports/qca_gallery.md`): with the uncorrected
  intensity edges, 8 paths read ≥ 85% DS and 13 had a distal FFR ≤ 0.2; after D45, 0 and 5
  (probability edges: 0 and 2). Two of the 5 are as low with probability edges (long paths into
  thin branches). Tree gallery: implausibly low main-path FFR in 5 single paths / 4 trees
  (3 / 3 with probability edges).
- **CADICA expert grades** (`reports/stenosis_grading.md`, commit `2c93bbd`; 21 test patients,
  214 lesions; development half in brackets):

  | Method | Weighted kappa | AUC ≥ 70% | ±1 grade | Median %DS, 90–98% lesions |
  |---|---:|---:|---:|---:|
  | probability edges | 0.12 (0.12) | 0.69 (0.73) | 54% | 32 |
  | intensity edges, corrected (default) | 0.14 (0.19) | 0.72 (0.82) | 62% | 42 |
  | imaging-chain model | 0.23 (0.25) | 0.74 (0.73) | 70% | 53 |

  - D44: the model lost to the uncorrected intensity edges on development patients (kappa 0.25
    vs 0.36, AUC 0.73 vs 0.81), so it stayed an option. That test run is kept in history
    (commit `8d78d72`).
  - D45: the corrections kept intensity edges ahead of probability edges on development
    patients, so they stay the default. **But kappa fell from 0.36 to 0.19 (development) and
    from 0.22 to 0.14 (test):** much of the earlier agreement came from the bug, which
    exaggerated one-sided narrowings.
  - The model is the most consistent across halves and the best on the test half, but it
    missed D44's AUC criterion on development patients. Nothing was chosen on the test half,
    which has now been used three times; a further method choice needs a fresh test set.
- **Occlusions:** none of the 11 total occlusions was read as critical by any method (10
  measured as open). The segmentation draws a vessel through them, so there is no gap to
  bridge. False alarms on lesions graded 0–50%: 0 / 223 (intensity), 1 / 223 (model).
- **Synthetic validation, automatic-segmenter arm:** 42 of 102 vessels analysable (39 before),
  AUC 0.92 (0.91).

**Open issues and next steps**
- **Severity on real angiograms is still compressed:** median 42% DS for lesions graded
  90–98%. Candidate causes: the segmentation at narrowings (P8 sim-to-real labels), and
  physics missing from the imaging-chain model (scatter / veiling glare, background
  structure, overlapping vessels). Part of the gap may lie in the visual grades themselves.
- **Occlusion detection needs another signal**, e.g. contrast arrival and washout over the
  cine frames.
- **A fresh, independent test set** is needed before choosing between intensity edges and
  the model (ICADD, or a partner's data).
- **Patents:** for the model fit (invention note 5), the evidence is synthetic only.
  Gap bridging (note 6) did not work on real occlusions.
- `reports/app_latency.md` was not re-run: the stent controls' code path is unchanged, and it
  needs an idle machine.

## 2026-11-01 — P4d: Cine runs on real angiograms (done)

P4c left severity on real angiograms compressed. P4d first tried to calibrate the imaging
chain once per frame, to insert physics-consistent lesions into real angiograms as a
benchmark with exact lumens; that did not hold on real frames (D46), so P4d turned to the
cine run, the form in which clinics record angiograms (parts of P7 moved here).

**What was done**
- **Frame-level imaging-chain calibration** (`imaging/calibration.py`, `calibrate_chain`,
  `fit_lumen(chain=...)`): kept as an option, not the default (D46).
- **One loader for runs** (`imaging/image_io.py`: `read_run`, `read_run_bytes`, `is_run`):
  multi-frame DICOM (frame time from FrameTime / CineRate) or a folder of frames, on one
  intensity scale for the whole run (frame-wise stretching would distort the opacification).
- **Opacification and analysis frame** (`imaging/cine.py`): segmented vessel area per frame →
  contrast arrival, the opacified window (frames above 70% of the peak, D47) and the analysis
  frame (the peak).
- **Contrast transit**: arrival frame at each centreline point (darkest value within 6 px, for
  cardiac motion), inflow found on the path, Theil–Sen front speed with its 95% interval;
  shown only when the front advances (D47), never used for FFR (runs are at rest).
- **Product:** `vstent analyse` accepts a run (`--frame` to override) and reports the
  transit; the app treats an uploaded multi-frame DICOM as a run and analyses its most
  opacified frame (before, it used frame 0, usually recorded before the contrast arrives).
- **`vstent cine-frames`** (`validation/cine_frames.py`, `reports/cine_frames.md`).
- Synthetic cine runs (`synthetic.render_cine`: contrast front, washout, cardiac motion) for
  the tests.

**Results**
- **Rendered frames (D46):** the joint calibration recovers blur 1.01 (1.0) and edge amount
  2.20 (2.5). **Real ARCADE frames:** blur 0.6–3.0 px and edge amount 0.2–5.0 across frames,
  often at a bound; fixing them barely raises the residual. Lesion insertion was dropped.
- **Synthetic runs:** front speed recovered within 15% (10.0 px/frame for 10 in the CLI test;
  12 px/frame with 3 px of cardiac motion; a path through the inflow).
- **CADICA, 382 runs of 42 patients, the clinicians' selected frames as reference**
  (`reports/cine_frames.md`, commit `296b430`):

  | | Test (177 runs, 21 patients) | Development (205 runs) |
  |---|---:|---:|
  | Analysis frame among the selected frames (primary) | 77% (70–82%) | 84% |
  | Within their span | 88% (82–92%) | 90% |
  | Window overlap, median IoU | 0.61 (0.55–0.67) | 0.62 |
  | Contrast arrival ≤ first selected frame | 94% | 95% |

  Lesion runs: 72% among the selected frames (runs without a lesion 85%).
- **Contrast transit on CADICA:** a plausible, advancing front in 213 of 382 runs (56%); the
  95% interval's width is a median 44% of the speed. Too imprecise to set the flow; shown
  only.

**Open issues and next steps**
- **Transit:** the longest path often crosses overlapping vessels or the catheter. Measuring
  along the user's selected vessel (the app does), and a dedicated catheter / ostium detector,
  should raise feasibility; turning a resting transit into a hyperaemic flow needs a validated
  relation or an adenosine run.
- **Frame choice** misses most often in lesion runs (the clinicians pick frames where the
  lesion is best seen, not only the most filled); a lesion-aware choice needs the user's
  vessel.
- **Novelty:** this is established practice (invention notes, "not considered inventive"); it
  makes the product's input match the clinic's.
- Real FFR ground truth (ICADD) still waits for the human.

## 2026-11-01 — P8 (part): Regulatory groundwork drafts

The founder asked for whatever makes the product robust and pitchable. A hospital or device
company asks first for the intended use, the classification and the risk file, so the
regulatory half of P8 came before two-view 3D (P5). Everything is a draft for review by a
qualified person, not regulatory advice.

**What was done** (`docs/regulatory/`, D48)
- **Hand-written:**
  - INTENDED_USE: intended use, users, exclusions, foreseeable misuse, preliminary class;
  - SAFETY_CLASS: IEC 62304 class C and the class C gap table;
  - DEVELOPMENT_PLAN, ARCHITECTURE (11 software items, interfaces, data flow);
  - RISK_MANAGEMENT_PLAN (ISO 14971 policy and method).
- **Generated and checked** (`vstent regulatory`, `src/vstent/regulatory/`):
  - REQUIREMENTS (42), HAZARD_ANALYSIS (19 hazards), TRACEABILITY, SOUP, a CycloneDX SBOM;
  - sources: `requirements.yaml`, `hazards.yaml`, `soup.yaml` and `uv.lock`;
  - every code reference (`path::name`), test, report and risk control must resolve, and a
    test checks that the committed documents match their sources;
  - the SOUP list walks the locked dependency graph for CPython 3.14 on Windows and Linux.
- **Sources checked** rather than recalled:
  - the CDSCO software guidance (CDSCO/MD/GD/MDSW/01/2026, read from the PDF): Table 2
    classes, licensing authorities, test licence and clinical-investigation forms, the
    algorithm change protocol;
  - FDA 510(k) entries for the angiography-FFR precedents (FFRangio K182149 / QEK; QAngio XA
    3D K182611, CAAS vFFR K232147, QFR K243769 / QHA), found via database mirrors and marked
    "verify".
- **Fixes the checks found:**
  - pandas and pyarrow were imported by the app but not declared; they are now in the `app`
    extra, with no version change in the lock;
  - new tests for the research-use notice on every report and regulatory page, the unit
    conversions, and the app's usage statistics being off.

**Results**
- **Preliminary class:**
  - India: C (critical × drive clinical management; D with a diagnosis claim);
  - EU: IIb likely;
  - US: class II, 510(k) under 21 CFR 870.1415;
  - IEC 62304: C.
- **Requirements:** 39 of 42 verified by tests or generated reports; 3 are gaps:
  - pixel size from the projection geometry (P5);
  - foreshortening-free lengths (P5);
  - usability engineering.
- **Hazards:** after today's controls, 13 of 19 residual risks are unacceptable for clinical
  use, 2 need reduction and 4 are acceptable. The unacceptable ones rest on measured
  weaknesses:
  - severity compression on CADICA (kappa 0.14);
  - no total occlusion flagged (0 of 10 measured);
  - the physics' loss at 0.61 × measured on the FDA nozzle;
  - 12% of analysis frames outside the clinicians' span;
  - single-view geometry.
  This confirms that research use only is the right status, and each hazard names the
  evidence that would lower it.
- **SOUP:** 108 packages shipped (22 declared directly), 4 development-only; anomaly lists
  not yet reviewed (recorded as a gap).
- Checks: ruff clean; 302 tests passed (3 slow deselected).

**Open issues and next steps**
- Human (HUMAN_ACTIONS.md): a regulatory adviser, an ISO 13485 quality system, an
  independent reviewer, an interventional cardiologist for usability; ICADD or a study for
  clinical evidence.
- Code, before any release:
  - package the analysis path without torch and the tools;
  - move the training ranges into the model metadata;
  - signed model files;
  - an issue tracker with safety assessment;
  - CI to run the checks.
- Novelty: none claimed; generated, checked traceability is established practice.
- P8's other half (sim-to-real angiograms) and P5 remain.

## 2026-11-01 — P5: Projection geometry and two-view 3D reconstruction (done on synthetic data)

**What was done** (D49)
- `imaging/projection.py`: C-arm geometry from the DICOM positioner angles, SID, SOD and
  ImagerPixelSpacing; projection, back-projection rays, pixel size at any 3D point.
- **Magnification fix.** The loader used to offer the DICOM detector spacing as the pixel
  size, which overstates every size by SID / SOD. It now corrects it to the isocentre;
  `vstent analyse` uses it automatically and the app offers it first.
- `imaging/twoview.py`:
  - ray-distance DP pairing of the two centrelines;
  - triangulation and a weighted spline;
  - motion correction between the runs;
  - reprojection and ray-gap checks;
  - the lumen profile from both views at each point's true magnification.
- `vstent analyse VIEW1 --view2 VIEW2 --start --end --start2 --end2`: 3D length, two-view FFR
  (solver and surrogate), intervals, plans, and a figure with the 3D centreline projected back
  onto both views. Single-view stent plans now state that their lengths are foreshortened.
- `imaging/synthetic3d.py`: P2 trees on a heart-sized sphere, rendered in two C-arm views with
  the ARCADE-fitted imaging chain.
- `vstent two-view-synthetic` → `reports/two_view_synthetic.md`. The scenarios:
  - 100 static cases, with views chosen without overlap;
  - the same cases with motion between the runs;
  - 50 random-view stress cases.
- Regulatory: SR-40 and SR-41 implemented and traced; H-03, H-05 and H-09 updated with
  measured evidence; new hazard H-20 (wrong 3D reconstruction not noticed).

**Results** (report seeds; method fixed beforehand on development seeds)
- **Targets met:** 3D centreline error 0.23 mm (median over cases; target ≤ 1 mm); length
  error 0.6% (median abs.; target ≤ 5%), 90th percentile 2.0%, 95% of cases within 5%.
- **Silent failures:** 5 of the 100 cases were off by more than 5% (worst −21% with 5.9 mm 3D
  error, views 33° apart; the others +7% to +12%), and the check flagged none of them. These
  are poorly conditioned cases (strong foreshortening, part of the vessel along the epipolar
  lines), where the spline is not held well. The method was not changed afterwards (it was
  fixed on development seeds); a better check is the next step (H-20).
- **Single view vs two views, length:**
  - single view at the isocentre pixel size: 13.0% short (median), 16% of views within 5%;
  - uncorrected detector spacing: 20.3% off (median abs.);
  - the isocentre pixel size itself is 4.0% off at the vessel (median; 90th percentile
    6.7%).
- **FFR error from geometry alone** (true lumen, each method's length and pixel size):
  - single view: 0.008 median, 0.030 at the 90th percentile, max 0.088;
  - uncorrected detector spacing: 0.023 median, max 0.163;
  - two views: 0.000 median, max 0.027.
- **End to end** (product lumen measurement), FFR error median 0.07–0.08 for all three
  methods: the lumen measurement dominates, so two views do not yet improve the end-to-end
  FFR.
- **Motion** (median 3.1 mm between the runs):
  - uncorrected: length error 8.5% median, 27.8% at the 90th percentile;
  - corrected: 0.7% and 2.6%.
- **Stress test** (random views): 37 of 50 cases overlap. Their two-view length error is 1.1%
  median, 5.7% at the 90th percentile, and their 3D error 1.3 mm at the 90th percentile; a
  wrong path through a crossing that still fits the other view is not flagged.
- Checks: ruff clean; fast suite passed (see the commit).

**Open issues and next steps**
- **New finding: absolute diameter bias of intensity edges.** On these renders, intensity
  edges read healthy vessels 10–12% narrow (checked in pixels against the rendered width; the
  mask edges are exact), and the reference diameter about 15% narrow. The %DS-based lumen
  reports could not show this, since %DS cancels a uniform scale. It biases stent diameters
  small (H-09). The imaging-chain model lumen fits the blur and is the candidate fix, to be
  checked on development data first.
- The angle convention (D49) and real C-arm distances are unverified: real two-view DICOM (a
  phantom or de-identified runs) is needed (HUMAN_ACTIONS).
- Two views are in the CLI only; the app and `vstent validate` take one view.
- The single-view isocentre pixel size is not yet varied in the input intervals (4% median
  error at the vessel).
- Not modelled: different cardiac phases in the two runs (P7: end-diastolic frames), image
  distortion, overlap detection on the traced path.
- Novelty: none claimed. Two-view reconstruction with epipolar DP matching is established
  (QAngio XA 3D, CAAS, QFR).

## 2026-11-01 — P5b: Absolute lumen size and two-view reliability (done; the check missed its target)

**What was done** (D50; plan and default rule committed first, 7437238)
- **Scaled lumen** (`lumen="scaled"`, now the product default). Intensity edges give the
  shape (%DS unchanged, so the CADICA grading is unaffected), and the imaging chain fitted on
  up to four healthy windows gives the absolute size. Side branches inherit the factor, both
  views of a two-view analysis use it, and the case report stores it. A failed fit or a factor
  outside 1.0–1.4 keeps the edges' size with a warning. It takes about 1 s per vessel; the
  whole-vessel fit took 6–16 s.
- **Two views:** knot spacing 4 → 8 mm (chosen on 199 development cases), and a reliability
  check (length sensitivity to doubled smoothing > 2.5%, or residual > 0.30 px).
- Regulatory: new requirement SR-43 (absolute lumen size); SR-41 and H-09 updated.

**Results**
- **Default rule met** (pre-registered), so `AUTO_LUMEN` is now `scaled`:
  - held-out renders: median size error 0.0%, 97% of views within ±15%; intensity edges alone
    −13%;
  - real ARCADE frames (no ground truth, consistency only): factor accepted on 18 of 20,
    median 1.148, inside the development range 1.120–1.209.
  - The whole-vessel variant also met the rule (held-out −0.7%; ARCADE 85%, median 1.133)
    but was too slow.
- **Two-view report, fresh seeds** (200 static + 50 stress cases):
  - 3D error 0.20 mm and length error 0.7% (median);
  - only 1 of 200 cases is off by more than 5%, against 5 of 100 with 4 mm knots;
  - single-view reference diameter −3.4% (median) with the scaled lumen, −15.8% with edges
    alone;
  - FFR error from geometry alone is unchanged in size: single view 0.008, two views 0.000.
- **Reliability check: targets not met.** Warnings fired on 35 of 200 cases (18%; target
  ≤ 10%): 21 from the new check, the others from P5's "along the epipolar lines" warning. The
  one large error (+17%) was not flagged (target ≥ 80% flagged; with one such case, its
  sensitivity cannot be estimated). Flagged cases do have larger errors (median 1.0% vs
  0.6%). Not retuned on the held-out data.
- **Other reports regenerated:**
  - QCA galleries: MLDs about ×1.1–1.3, FFR about +0.02–0.04;
  - validation stand-in, automatic-segmentation arm: accuracy 0.76 → 0.83, FFR MAE
    0.113 → 0.106. The mask arm is unchanged.

**Open issues and next steps**
- The absolute size on real angiograms is unverified. If real edges were unbiased, the
  scaled lumen would oversize by about 15% (H-09). A phantom or IVUS / OCT reference
  diameters are needed.
- The reliability check: decide whether to keep it (more alarms than targeted, no measured
  catch) or drop it now that 8 mm knots make large errors rare (0.5%). Evaluating it would need
  far more large-error cases.
- Two views in the app, which is still single view (the next product step).
- Novelty: none claimed. Model-based edge and size estimation in QCA is established.

## 2026-11-01 — P5c: Two views in the app (done) and the smoothing check made information

**What was done**
- D51 (the founder's choice): the P5b reliability check no longer warns. Its number is
  printed by `vstent analyse --view2` and stored in the case report.
- App (D52):
  - a second DICOM view (a toggle when the first image has projection geometry) and a
    synthetic two-view example;
  - view-2 points clicked on the second image, with epipolar guide lines;
  - both views show the 3D centreline projected back and the stents along it;
  - results ("3D path length"), stents, planner, intervals and the case report use the 3D
    profile, and the case report stores the two-view details. Single-view foreshortening
    notes are not shown for two-view cases.
- Regulatory: SR-41 now covers the app; D51 and D52 recorded.

**Results**
- The synthetic example runs end to end in the app (AppTest): two views, 3D length 78.6 mm
  (single views 65.0 and 74.9 mm), a stent drawn in both views, plans, the case report.
- The view-2 landmarks lie on their epipolar guides (< 1.5 px).
- Checks: ruff clean, fast suite passed. App latency not re-measured (the stent controls are
  unchanged; it needs an idle PC).

**Open issues**
- Real two-view DICOM is still needed to confirm the geometry (HUMAN_ACTIONS).
- Two-view cases are single path only: tree mode stays single-view.

## 2026-11-01 — P8: Sim-to-real angiograms for segmentation (done; angiogram → FFR deferred)

**What was done**
- Plan and adoption rule pre-registered before any rendered data or training (commit
  9537e4e).
- `vstent synth-angio` (`data/synth_angio.py`) renders angiograms from the P5 3D trees:
  - one or two trees from one ostium, random clinical views, the ARCADE-fitted imaging chain
    (D39);
  - unlabelled distractors: a catheter, shadows, rib-like bands, 12–30 small branches;
  - labels: every vessel with a reference diameter ≥ 1.5 mm;
  - 2,400 images (train 2,000, val 200, test 200). Appearance was tuned by eye against
    ARCADE val frames only. Card: `reports/synth_angio_card.md`.
- Training (`configs/seg_synth.yaml`): the M5 configuration with 1,000 rendered images added
  to ARCADE's 1,000. RTX 3060 Laptop GPU, 57 epochs in 79 min (early stop; best ARCADE val at
  epoch 42).
- `vstent sim-to-real-report` (`imaging/sim_to_real.py`) evaluates both models on ARCADE val /
  test and the rendered test split, with paired per-image Dice differences, 95% bootstrap
  intervals and the adoption rule.
- The adopted model is exported to `models/seg/seg_unet_r34_synth-20261101-182811` (parity
  1.7e-05). It is the newest model, so the app and the CLI use it (D53).

**Results** (`reports/sim_to_real.md`, `reports/seg_synth/`)
- ARCADE test (n 300): mean Dice 0.7925 → 0.7947, difference +0.0022 (95% interval −0.0052
  to +0.0095). **Adoption rule met.**
- ARCADE val (n 200; it chose both models' epochs): 0.8325 → 0.8278, −0.0047 (−0.0105 to
  +0.0012).
- Rendered test (n 200): 0.600 → 0.907; 10th percentile 0.303 → 0.828.
- Reading: no measurable change on real angiograms, and the gap on rendered angiograms is
  closed. This is not evidence of a real-world gain.
- ONNX CPU latency: median 134 ms per image (M5's export: 101 ms). The architecture is the
  same, so the difference is the machine's load at the time.

**Open issues**
- Reports produced with M5 are not yet regenerated with the adopted model: QCA gallery,
  stenosis grading, cine frames, validation stand-in, app latency. Stenosis grading would use
  the CADICA test half a fourth time, for reporting only.
- Angiogram → FFR models on rendered images: deferred (D53).
- ARCADE's partial labels cap what its Dice can show. Real angiograms with every major vessel
  labelled would measure the difference.
- Novelty: none claimed. Training segmentation on rendered images is established practice.

## 2026-11-02 — P8: reports regenerated with the adopted segmenter

**What was done**
- Every report that uses the segmenter was regenerated from clean commits: QCA gallery
  (single path and trees), validation stand-in, CADICA stenosis grading, CADICA cine frames,
  app latency.
- Report text that described the old model is now model-aware or corrected. The tree
  gallery's "what limits it" paragraph describes the current labels. The stand-in's
  segmenter paragraph depends on whether the model was trained with rendered images. The
  stenosis report names its segmentation model.
- Hazard numbers updated (H-01, H-02, H-04, H-11, H-12); D54 recorded.

**Results** (with M5 in brackets)
- CADICA stenosis grading, intensity edges, test patients:
  - weighted kappa 0.13 (0.14);
  - AUC ≥ 70% 0.71 (0.72);
  - median read for 70–90% lesions 44% (44%);
  - total occlusions measured 9 of 11 (10), none read as occluded.
  Unchanged within the intervals.
- QCA gallery: max-DS point inside a stenosis box 10 of 20 (10); MLD inside 4 (2). The
  auto-selected paths changed.
- Tree gallery: side branches in 16 of 20 images (14); implausibly low tree FFR 1 (4).
- Validation stand-in, automatic arm: analysable 96 of 102 (42); accuracy counting the
  failures 0.81 (0.34); tree MAE 0.083 (0.105). The renders are close to the new training
  data, so this is not a clinical gain.
- **CADICA cine frames, worse:** the analysis frame lies within the clinicians' span in 82%
  of test runs (88%) and 81% of development runs (90%). McNemar development p = 0.002, test
  p = 0.06 (D54).
- App: stent-control updates 74 / 131 ms median (synthetic / image), max 233 ms (target
  < 300 ms, met). The one-off image set-up is now 925 ms (236 ms in the previous
  measurement, which was taken before the scaled-lumen fit of P5b).

**Open issues**
- Recovering the frame choice is now a P7 target (D54).

## 2026-11-02 — P7: Cine angiography (done; one target missed)

**What was done**
- Plan and targets pre-registered (commit dc16be1), amended once before any data existed
  (two-view velocity targets, single-view foreshortening; fcb0506).
- End-diastolic frames (`imaging/cardiac.py`), in order:
  - the DICOM R-wave pointer;
  - the R peaks of a DICOM ECG waveform, mapped to frames with FrameDelay, FrameTime and
    the waveform's time offset;
  - the vessel tree's spread in the images, its beat by autocorrelation.

  The analysis frame is the ED frame in the opacified window with the most vessel. The
  loader carries the pointer and the ECG.
- Heart motion (`imaging/motion.py`): affine registration of the vessel maps, on the
  opacification plateau only. Filling and washout frames take the transform of the same
  phase a whole number of beats away.
- Flow velocity (`imaging/flow.py`):
  - the contrast front's slope of distance on flow time, a phasic-flow correction from the
    ED frames (D55);
  - guards; a resting run converted with the FAVOR Pilot relation;
  - 90% split-conformal intervals (× / ÷ 1.68 for one run, 1.23 for two runs that agree
    within ×1.5), widened for one view's foreshortening;
  - the input Monte Carlo draws from the interval when the velocity is used.
- One pass over a run (`imaging/run_analysis.py`), shared by `vstent analyse` (`--flow
  measured`, `--hyperemic-run`), the app and the report.
  - The app shows the opacification curve with the ED frames and the measured velocity, with
    a toggle to use it; the case report has a `flow` section.
  - The fixed velocity stays the default.
- Synthetic beating hearts (`imaging/synthetic_cine.py`):
  - P5's trees contract about the heart's centre with a twist, plus breathing;
  - phasic contrast fronts;
  - an ECG; written as XA DICOM.
- Reports: `vstent cine-synthetic` (held-out seeds 9200–9259), `--calibrate` (development
  seeds 9164–9199), and the ED rule in `vstent cine-frames`.
- Regulatory: SR-06 revised, SR-44–SR-46 added (46 requirements, 45 verified); H-11 and H-12
  revised (H-12: a wrong measured velocity used); D55; INVENTION_NOTES §7; HUMAN_ACTIONS
  (ECG runs, flow ground truth, freedom to operate).

**Results** (held-out synthetic cases unless named)
- ED frames:
  - from the ECG: 347 of 348 beats within ±1 frame;
  - from the images: 79% of beats at 15 fps (target ≥ 80%, **not met**);
  - analysis frame within ±1 frame of a true ED frame: 114 of 120 runs (95%).
- Velocity, two views (3D path): median |error| 5.8% (bias −1.2%), 98% of cases within
  ±25%, interval coverage 95%. The FFR changes by 0.005 (median; 90th percentile 0.022) from
  the velocity error alone. 40 of 51 analysable cases had an accepted velocity.
- Velocity, one view: median error −16% (foreshortening), widened-interval coverage 95%.
  Guards rejected 13% of runs.
- On development cases the phasic correction cut the median error from 19.9% to 9.7% (per
  run, 3D path). Plateau tracking removed a +3.8% bias from registering partly filled
  frames.
- CADICA test patients: the analysis frame within the clinicians' span in 149 of 177 runs
  (84%; target ≥ 83%, met). The peak rule with the same segmenter gave 82%, with M5 88%.
- The intervals are wide for one run (× / ÷ 1.68): a single run's velocity is barely more
  informative than the fixed one. Two agreeing runs narrow it to × / ÷ 1.23.

**Open issues**
- Image-based ED at 15 fps missed its target by one point; 7.5 fps runs are the weakest.
  Not retuned on the held-out data.
- Untested on real flow and on real ECG-carrying DICOM (HUMAN_ACTIONS). The phasic
  correction assumes the generator's flow-profile family.
- M5's segmentation for the run curves would give 88–89% on CADICA, at the cost of a second
  segmentation pass and model (D55).
- Freedom to operate for the FAVOR Pilot conversion and frame-count velocity (QFR patents).
- Fast test suite now about 5 minutes (the P7 CLI test renders a run).

## 2026-11-02 — P4e: Real coronary physiology (DEFINE-FLOW; done)

**What was done**
- A thorough search for open FFR data: Zenodo, Dryad, Figshare, Mendeley, PhysioNet, Kaggle and
  GitHub, grand-challenge, BioLINCC / Vivli, and a curated catalogue. Still no open angiograms
  with invasive FFR. Found two open (CC0) physiology datasets:
  - DEFINE-FLOW: 455 subjects; FFR, CFR, Doppler velocity, raw pressure, Doppler and ECG
    tracings;
  - CONTRAST: 763 subjects; raw pressure tracings.
  The other leads are in `docs/DATA.md` and `docs/HUMAN_ACTIONS.md`.
- Plan and rules pre-registered before the download (commit d45a676).
- `vstent data define-flow` and `vstent data contrast` (`data/physiology.py`): MD5-verified,
  safely extracted.
- `vstent physiology-real` (`validation/physiology_real.py`) compares, on real patients:
  - the phasic profile per beat (P7's flow-time step), from ECG R peaks or the recorder's flags;
  - the stenosis-free hyperemic velocity;
  - the FAVOR conversion;
  - Pa.

**Results** (455 lesions of 379 patients)
- Resting diastolic / systolic ratio: median 1.73 (prior 2.0); LAD 1.77, LCx 1.81, **RCA
  1.35**. Systole 0.39 s (assumed 0.31 s).
- Hyperemic stenosis-free velocity: 0.21–0.33 m/s (median, k 0.5–0.8) against the fixed
  0.35 m/s. P1's range holds 27–56% of lesions.
- FAVOR conversion: median +4% to +28%, wide spread. Pa inside P1's range: 78%.
- No pre-registered rule fired; nothing changed (D56). CONTRAST is not analysed: no condition
  markers.

**Open issues**
- Candidate changes to pre-register: an artery-specific phasic prior; wider P1 velocity and Pa
  ranges.
- DEFINE-FLOW's angiograms are not deposited. With them (a request to the corresponding
  author), the product could be compared with wire FFR end to end (HUMAN_ACTIONS, first item).

## 2026-11-02 — P4f: Repeatability on real angiograms (done; 3 of 4 targets not met)

**What was done**
- Plan and targets pre-registered (commit 7873b9a), set from accepted methods' own
  variability:
  - QREP: QFR inter-observer CV 9.4%;
  - QCA inter-observer: %DS CV 11.9%, SD of differences 5–7 points.
- `vstent repeatability` (`validation/repeatability.py`) runs on CADICA's expert lesions,
  graded 0–98%, with the product's steps. Each box is measured with the default segment
  margin and with two simulated operators' margins (0.5–1.5 × the box size, seeded). The box
  path takes an adjustable margin (`stenosis_grading.path_through_box`).

**Results** (5,894 boxes, 414 lesions with ≥ 2 keyframes, 42 patients)
- Frame to frame:
  - %DS within-lesion SD 10 points, CV 44.6% (not met);
  - segment-FFR SD 0.124, CV 18.7% (not met), worst in severe lesions;
  - 85% of keyframe pairs agree on the side of 0.80.
- Operator to operator:
  - FFR CV 17.1% (not met), 90% agreement at 0.80;
  - %DS SD of differences 6.0 points (met).
- D57: a real weakness of single-frame automatic measurement, partly made worse by the
  design (mixed heart phases, short segments, no manual editing).

**Open issues**
- Multi-frame consensus (median over frames), pre-registered before it is tried: frame count
  chosen on CADICA development patients, reported on test patients.
- Contour editing in the app; a lesion-measurement model trained on AngioCAD / CARDIAG.

## 2026-11-02 — P4g: Multi-frame consensus (done; FFR target met, %DS not)

**What was done**
- Plan and rules pre-registered (commit 4ecbd0b). P4f's CADICA measurements re-made with
  every row saved (`vstent repeatability --rows-out`); the single-frame result reproduced
  exactly.
- `vstent consensus-report` (`validation/consensus.py`): split halves of alternate keyframes,
  the median of K frames each, CV between the medians; K chosen on development patients and
  tested once.
- Product (`inference/consensus.py`):
  - for a cine run, the vessel is measured on up to K end-diastolic frames on which the path
    is fully opacified, the points following the heart;
  - the output is the median FFR, every frame's value and the instability flag; the profile
    for intervals and plans is the frame closest to the median;
  - single path and tree, in `vstent analyse` (`--consensus K`), the app (the chosen frame
    shown with its stents) and `vstent validate` (the medians as index tests, `frame_*`
    columns, PROTOCOL updated).
- The P7 transit now also records when each point is filled (90% of its plateau contrast).

**Results** (`reports/consensus.md`)
- Development: no K met both targets; K = 5 by the rule (FFR CV 12.0%, %DS CV 21.7%).
- Test, K = 5 (111 lesions, 20 patients):
  - FFR CV 5.1% (95% 3.4–6.8%), met (≤ 9.4%); within-lesion SD 0.040 (one frame: 0.124);
  - %DS CV 27.0%, not met;
  - 91% of the medians agree at 0.80.
- Flag: all 10 discordant lesions flagged, but 31.5% of sets flagged.
- Synthetic beating heart: moved points within 1 px of the truth; an outlier frame
  (FFR 0.35 vs 0.92) was absorbed by the median and flagged.

**Open issues**
- %DS repeatability and the severity compression common to all frames (D44).
- The flag's specificity.
- ED-only frames on real runs are untested (CADICA has no ECG).
- Accuracy against wire FFR.

## 2026-11-02 — Guided app (D59) and a merged-track check of P4g

**What was done**
- `vstent app` is now a guided workflow for one cine run: load (upload or a demo), frame
  (end-diastolic thumbnails, ★ recommended), vessel and lesions (automatic main-trunk path,
  *Change path* with two clicks, lesions of 50% or more numbered), virtual FFR (the
  multi-frame median in plain words, its range, the heartbeats' agreement, the curve), stent
  (suggested plans drawn on the vessel, before and after, adjustable), report. The research
  view is `vstent app --advanced`.
- `centerline.main_trunk_endpoints`: an automatic path that follows the widest branch and
  stops at 100 mm.
- P4g check: 16 CADICA lesion tracks hold two boxes on the same keyframes. Without them the
  test FFR CV is 4.9% (reported 5.1%) and K is unchanged. `reports/consensus.md` now carries
  this sensitivity section, generated by code (D58 unchanged).

**Open issues**
- The automatic path may pick the wrong vessel on real angiograms.
- Without an ECG, tracking between heartbeats often fails on real runs.
- The image-based end-diastolic frames on CADICA are unevenly spaced; to be investigated.

## 2026-11-02 — Guided app, second version (D60)

**What was done**
- Case library in the sidebar: 10 synthetic beating hearts and CADICA's 382 real runs (most
  severe lesions first); *Random*, *New study*, upload.
- A frame scroller (★ recommended, ◆ expert box) that measures nothing until *Analyse this
  frame*.
- Lesion close-ups with the measured edges; experts' boxes; *Through the expert box*; plain
  angiogram on request.
- A permanent FFR guide: > 0.80 no stent; 0.76–0.80 grey zone; ≤ 0.75 significant; ≥ 0.90
  after a stent.
- Much less text.

**Seen:** expert 90–98% lesions read 32–35% DS. The close-ups show density loss without a
resolvable width change: densitometry is the candidate fix.

## 2026-11-02 — P4h: Densitometric lumen (done; not adopted by the rule; D61)

**What was done**
- `imaging/densitometry.py`: integrated contrast across the vessel → lumen area. The lumen
  is the narrower of the edges and densitometry. Lumen options `dense`, `dense_log` and
  `dense_linear` are wired into QCA, the grading and the rendered-lesion report.
- Pre-registered first (031074a). Amendment 1 (IOD smoothing σ = 3 px) was chosen on separate
  development renders before any CADICA measurement (8aa3def).

**Results**
- Development: kappa 0.214 → 0.294, but AUC for ≥ 70% lesions 0.818 → 0.803. Rule not met:
  **not adopted**.
- Rendered lesions: tight lesions within 5.9 points (edges 30.7).
- Test (once): kappa 0.13 → 0.25; AUC 0.71 → 0.77; lesions graded 90–98% read at a median
  49% (edges 40%). Targets not met.

**Analysis after two attempts** (D44: the imaging-chain fit; D61: densitometry). Both read
rendered lesions almost perfectly and both raise agreement on real angiograms, but neither
brings severe lesions near the experts' grades (≈ 45–50% against a visual 70–98%). Causes,
most likely first:
1. **The real images:** the X-ray systems' processing (local contrast equalisation, edge
   enhancement, 8-bit export) removes most of the density signal that physics-based methods
   need. Our renderer does not imitate that processing, so methods tuned on it look better
   than they are.
2. **The reference:** visual grades exceed QCA by about 10–20 points in tight lesions; a
   careful QCA would read a 90–98% visual lesion at perhaps 70–80%. A gap of 20–30 points
   remains.
3. **The segmentation:** it smooths over narrowings, and the edge search starts from it.

**Options for the next step (for the human to choose):**
- **A. Learned lesion severity on real angiograms.**
  - What: a model trained on real lesion patches with expert labels, used to narrow the
    lumen at lesions. Labels from CADICA development patients (grades) and ARCADE stenosis
    boxes; the untouched Kemerovo dataset (8.3 GB, CC BY 4.0, all lesions significant) as a
    fresh external test.
  - For: it learns what real images look like.
  - Against: the labels are coarse, visual (biased high) and few; a fresh pre-registration
    is needed.
- **B. Segmenter retrained with rendered tight stenoses** (as in P8), so the mask narrows at
  lesions; judged by the same pre-registered rule.
  - For: cheap and controlled.
  - Against: the sim-to-real gap of point 1.
- **C. Outline correction in the app**, as QFR analysts do. A product feature; accuracy then
  depends on the user, as in QFR's own validations.
- **D. Re-test `dense` on a fresh external dataset** (Kemerovo) under a new pre-registered rule.
- **E. Wait for wire-FFR data** and calibrate lesion measurement against FFR directly on a
  development part, testing on the rest.

Recommendation: C now (the product needs it in any case), then A with Kemerovo as the
external test. E remains the final word.

## 2026-11-02 — P4i / P4j: lesion reading learned from real angiograms; hybrid segmentation (D62, D63)

**What was done**
- Kemerovo dataset (100 patients, CC BY 4.0) downloaded and verified (`vstent data
  kemerovo`): an external test.
- Lesion patches from CADICA, ARCADE and Kemerovo (`vstent lesion-data`): one patch function
  for training and the product.
- A CNN lesion reader trained in 5 patient folds (`vstent train-lesion`, resumable), exported
  to ONNX (`vstent export-lesion`) and evaluated in two stages (`vstent lesion-report`).
- Hybrid segmentation (vesselness grown from the model's mask) evaluated with
  `vstent vessel-coverage`.
- Guided app:
  - every frame on the slider, with its contrast state;
  - an automatic stent plan when significant, and reasons when no plan is possible;
  - Target paths, and a detection check against the experts' boxes;
  - jumps to the experts' frames;
  - the expert-box path starts proximally;
  - a learned-reading switch.

**Results**
- P4j: not adopted. The model already crosses 95% of the development boxes, and the added
  vessels failed the near-annotation guard.
- P4i: not adopted (false alarms). On test, severe lesions (90–98%) read at a median 65%
  instead of 40%, kappa 0.32 instead of 0.13, and Kemerovo's significant lesions read ≥ 50% in
  54% instead of 18%. But 29% of ARCADE's unmarked narrowings were read ≥ 50% (edges 4%).

**Open issues**
- A second learned reader (%DS targets at non-lesion points, or a significance gate),
  pre-registered.
- The clinician-style proximal path for measurement.
- Frame choice without an ECG.

## 2026-11-02/03 — P4k: lesion reading with physics, wire position, virtual stenting on real angiograms (D64–D69)

**What was done**
- **Guided app:**
  - every stent has its own controls; plans come from the solver (D64);
  - the path starts at the tree's widest end, not a branch tip (D65);
  - QFR-style display: lumen contours, filled lesions (red ≥ 50%, amber 30–50%), a lesion
    table, the FFR loss per mm, every vessel outlined, faint vessels from the vesselness filter
    (D67);
  - an FFR marker where the FFR is read.
- **Pre-registered P4k A–J, P4l and P4m** before any training or evaluation.
- **Data:**
  - AngioCAD downloaded and verified (413 patients; 121,566 frames; three frames lost at
    archive volume boundaries);
  - its test protocol written before any model was run on it.
- **Models:**
  - lesion reader v2 (interval loss, normal-vessel negatives, clinician-style path);
  - vessel identity (ARCADE SYNTAX, ResNet34; validation Dice 0.816);
  - whole-image lesion detector (training, resumable; four more encoders queued, amendment 2);
  - learned ED model (MobileNetV3-Small + temporal convolutions; its synthetic data being
    generated);
  - an ED labelling page (`vstent label-ed`).

**Results (CADICA, no FFR; anchors De Bruyne 2001 and FAME)**
- **P4k-C (Murray outflow alone):** not adopted. Lesion-free 52% → 6% ≤ 0.80, but lesions
  under-read (D66).
- **P4k-C2 (wire position):** adopted (D68). Lesion-free 52% → 42% (corrected run).
- **P4k-C4 (velocity by artery):** RCA −9.1% in DEFINE-FLOW; the 10% rule did not fire.
- **Reader v2:** not adopted at the patch level (false alarms 30% vs limit 19%).
- **P4k-C3 (reader + physics, end to end):**
  - reader v2 + Murray outflow adopted as the product's FFR pipeline (D69,
    `configs/product.yaml`);
  - development: lesion-free 19%; test, once: lesion-free 54% → 20%;
  - 90–98% lesions read ≤ 0.80 in 52% on test (FAME 96%).
- **P4m (virtual stenting, development):**
  - with the pipeline, plans reaching 0.90 13% → 66%, median post-PCI FFR 0.91 (FFR-SEARCH
    0.91), median stent 39 → 27 mm;
  - a failure mode: about 8 cases where the reader reads a thin distal narrowing as ~92%
    (lumen 0.17 mm), giving FFR ~0.06 that no stent fixes.

**Open issues**
- Severe lesions under-called (test).
- The near-occlusion failure mode in thin distal vessels (P4n, pre-registered next).
- Detector and vessel-identity encoder comparisons (GPU queue).
- ED model (data generating).
- AngioCAD external run (after the decisions).
- Memory: 15 GB RAM. Jobs now run one per queue, with PID files (two crashes from leftover
  queue loops, fixed).

## 2026-11-03 — P4n test, reader v2 test, ED model on real runs, lesion-seeking path, 3D view, contrast (TIMI), AngioCAD tooling

**What was done**
- **Guided app:**
  - 3D vessel view: the measured lumen as a tube along the path, coloured by virtual FFR,
    lesions marked, stents as sleeves (one view: drawn in the image plane, depth not
    measured);
  - contrast along the analysed vessel: its state on the frame (filling / full / washing out)
    and the TIMI frame count at 30 fps when the frame rate is known (P4k-I, display only).
- **New tools:** `vstent ed-report` (ED methods per frame rate; beat-to-beat reproducibility on
  real runs without labels), `vstent target-path` (P4k-B), `vstent angiocad-check` (P4k-J;
  the labels read from the .xlsx with the standard library).
- **Docs:** README and CLAUDE.md list every command since P4i, and the product settings layer.

**Results**
- **P4n test (once):** the thin-vessel guard holds. Lesion-free vessels ≤ 0.80: 20% → 16%;
  combined distance 0.35 → 0.34. P4m on test patients (first run): post-PCI ≤ 0.80 12%
  (target ≤ 15%, met), median post-PCI FFR 0.92, plans reaching 0.90 73%, stent on the
  experts' lesion 82% (target 90%), median stent 26 mm.
- **Reader v2 test (once):** few false alarms (fresh lesion-free candidates read ≥ 50%: 7%;
  ARCADE distant candidates 20%, met), but severe lesions still read shallow (70–90%: median
  46%; 90–98%: 57%; Kemerovo read ≥ 50%: 51%).
- **ED without an ECG (P4l):**
  - M2 (synthetic-trained network): 85.7% within ±1 frame on synthetic validation (M0 80.9%),
    but on real CADICA runs it finds ~2 ED frames per run: it fires at frame 12 and at the last
    frames whatever the images show (position cues of the synthetic runs). The rule keeps M0.
  - M2b (pre-registered; amendment 1, disclosed: augmentation dropped because it stopped the
    model learning): `tcn5` 92.5% on synthetic validation (99 / 98 / 92 / 80% at 7.5 / 10 /
    15 / 30 fps); `gru` and `transformer` training; then the real-run check.
- **P4k-B (lesion-seeking path):** not adopted. The main trunk already crosses an expert box
  on 84% of development lesion runs; the seeking path 79%, because it follows branches the
  reader over-reads.

**Open issues**
- Severe lesions read shallow by every method (edges, imaging-chain model, densitometry,
  readers v1 and v2). Likely causes: visual grades read above QCA, and processed 8-bit exports.
  Real QCA- or FFR-annotated data would settle it (the user is seeking ECG + FFR data).
- ED: synthetic-only training does not transfer; M3 (adaptation to real runs, with a patient
  split between adaptation and evaluation) to be pre-registered after M2b's real result.
  Expert ED labels (`vstent label-ed`) would give the real test set.
- P4k-B test patients (reported once), the frame-rate test (H, with the TIMI frame count per
  frame rate), the detector and vessel-identity encoder comparisons (GPU), then AngioCAD (J).
- Memory: training with CUDA commits ~10 GB on Windows; jobs stay one per queue.
