# vstent: build plan (MVP, about 3 weeks)

Work top to bottom. Tick a box only when its acceptance criteria are met and the checks pass. Rough timing: Week 1 = M0–M2, Week 2 = M3–M5, Week 3 = M6–M8.

---

## M0: Project setup

- [x] `git init`; `.gitignore` covering `data/`, `models/`, `runs/`, `.venv/`, `*.onnx` (except small test fixtures), `__pycache__/`, `.streamlit/secrets.toml`
- [x] `pyproject.toml` with package `vstent` (src layout, Python ≥ 3.11) and optional extras `dev`, `imaging`, `app`
- [x] `ruff` (lint + format) and `pytest` configured; register the `slow` marker
- [x] Typer CLI `vstent` with stub subcommands matching CLAUDE.md
- [x] Create `docs/PROGRESS.md`, `docs/DECISIONS.md`, `docs/DATA.md`, and a `README.md` with the research-use-only disclaimer and a quickstart
- [x] Pydantic config models + YAML loader; every run saves the resolved config and the git hash

**Done when:** `vstent --help` lists all commands, the checks pass, first commit made.

---

## M1: Reduced-order physics solver (`src/vstent/physics/`)

### Specification

Inputs: positions `s_i` (m, uniform spacing Δs, i = 0..N−1), radii `r_i` (m), parameters below.

| Parameter | Default | Notes |
|---|---|---|
| μ (viscosity) | 3.5e-3 Pa·s | Newtonian blood |
| ρ (density) | 1060 kg/m³ | |
| Pa | 90 mmHg | mean aortic pressure |
| Pv | 5 mmHg | venous pressure |
| Kt | 1.52 | expansion loss coefficient (Young & Tsai) |
| v_hyp | 0.35 m/s | hyperemic reference velocity (as used for fixed-flow QFR) |
| expansion threshold | 1.05 | ignore diverging runs with A_max/A_min below this |

1. Area A_i = π r_i².
2. **Viscous coefficient** (cumulative, trapezoid rule): a(s_k) = Σ_{i<k} 8μ/π · ½(r_i⁻⁴ + r_{i+1}⁻⁴) · Δs.
3. **Expansion losses:** find every maximal *diverging run* (area non-decreasing from a local minimum A_min to the next local maximum A_max). If A_max/A_min ≥ threshold, its total coefficient is b_run = Kt·ρ/2·(1/A_min − 1/A_max)². Spread b_run along the run in proportion to the local decrease of 1/A², so the pullback curve stays continuous. Converging sections are treated as lossless. Cumulative b(s) = sum of the spread contributions up to s.
4. **Reference vessel:** r_ref(s) comes from `estimate_reference_radius` (robust linear fit of the healthy segments: iteratively exclude points below 85% of the fit, then refit; at most 5 iterations). For synthetic data the true reference is known and is used instead.
5. **Microvascular resistance (calibration):** Q_ref = v_hyp · π r_ref(0)². R_micro = (Pa − Pv)/Q_ref − a_ref(L), where a_ref is the viscous coefficient of the reference vessel. Clamp R_micro > 0.
6. **Flow:** solve b(L)·Q² + (a(L) + R_micro)·Q − (Pa − Pv) = 0 and take the positive root (use the numerically stable form of the quadratic formula).
7. **Pressure and FFR:** ΔP(s) = a(s)·Q + b(s)·Q²; Pd(s) = Pa − ΔP(s); FFR(s) = Pd(s)/Pa. Report `ffr_distal = FFR(L)`.
8. Two flow modes: `resistance` (default, steps 5–6) and `fixed_velocity` (Q = Q_ref, for comparison only; it overestimates pressure loss in severe lesions).

### Tasks
- [x] `geometry.py`: `VesselProfile` dataclass (s, r; validation), resampling to N points, area, `estimate_reference_radius`, percent diameter stenosis, lesion finder (contiguous runs with DS > 30%), minimal lumen diameter (MLD)
- [x] `solver.py`: `solve_ffr(profile, params, mode) -> FFRResult` (Q, a, b, ΔP(s), FFR(s), ffr_distal, loss breakdown)
- [x] Vectorised batch solver (NumPy) for dataset generation; target ≥ 5,000 profiles/s on a laptop CPU
- [x] `docs/PHYSICS.md`: equations, assumptions, limitations (no side branches, steady flow, Newtonian, 2D-derived geometry, no entrance-length effects)
- [x] Figure `reports/figures/ffr_vs_ds.png`: distal FFR vs %DS for 5, 10 and 20 mm lesions in a 3.0 mm vessel

### Tests (`tests/test_physics.py`)
- [x] Uniform tube with fixed Q: ΔP equals 8μLQ/(πr⁴) within 0.5%
- [x] Healthy 3.0 mm × 50 mm vessel, default params: `ffr_distal ≥ 0.95`
- [x] Distal FFR strictly decreases as DS goes 30% → 90% (10 mm cosine lesion)
- [x] FFR(s) is non-increasing along s, and FFR(0) = 1
- [x] Sanity bands for a 3.0 mm, 60 mm vessel with a 10 mm cosine lesion: 50% DS → FFR in [0.84, 0.94]; 70% DS → FFR in [0.45, 0.70]. If these fail, look for a bug; do **not** change constants.
- [x] Stenosed vessel flow < Q_ref; quadratic root is positive and finite
- [x] Restoring the lesion to the reference radius brings FFR to within 0.01 of the healthy vessel

**Done when:** all tests pass, PHYSICS.md is written, the figure is generated.

---

## M2: Synthetic data factory (`src/vstent/data/synth.py`)

### Sampling (all ranges in `configs/synth_v1.yaml`)
- Length L ~ U(30, 80) mm; N = 256 points
- Proximal reference diameter D0 ~ U(2.25, 4.5) mm; linear taper to D0 × U(0.6, 0.95) at the distal end
- Number of lesions: 0 (10%), 1 (55%), 2 (25%), 3 (10%)
- Each lesion: DS from a Beta distribution scaled to [20%, 90%] that oversamples 40–80%; length U(3, 25) mm; non-overlapping; ≥ 5 mm from both ends; shape ∈ {cosine, skewed cosine, plateau/diffuse, irregular (smooth random perturbation)}
- Diffuse disease with p = 0.2: smooth random narrowing of 0–20% over a long segment
- Measurement noise: smoothed Gaussian, σ ~ U(0, 0.03) × r_ref. **The label is computed from the same noisy profile the model sees.**
- Pa ~ U(70, 110) mmHg; Pv ~ U(2, 8) mmHg; v_hyp ~ U(0.25, 0.45) m/s (microvascular variability)

### Tasks
- [x] Generator with per-split seeds: train 100k, val 10k, test 10k, plus `ood` 5k (L 80–120 mm, 3–4 lesions)
- [x] Store per sample: r (N), L, r_ref (N), Pa, Pv, R_micro, Q, FFR(s) (N), ΔP(s) (N), lesion metadata. Save as `.npz` shards in `data/synth/v1/` with the config
- [x] Multiprocessing; full generation < 15 min on a laptop CPU
- [x] `reports/dataset_card.md` + histograms (D0, L, DS, distal FFR, Q)

**Done when:** the dataset card shows ≥ 30% of samples with distal FFR < 0.80 and ≥ 15% in the grey zone [0.75, 0.85] (adjust sampling, never physics), and generation is reproducible from the seed.

---

## M3: Surrogate model (`src/vstent/surrogate/`)

### Design
- **Features** (`features.py`, shared with ONNX runtime and app): per-point channels r/r_ref(0), log(r/r_ref(0)), s/L, d(A/A_ref)/ds. Global features: L, r_ref(0) in mm, Pa, Pv, log R_micro (standardised with training-set statistics saved to JSON).
- **Model:** 1D U-Net (4 levels, base 32 channels, GroupNorm, GELU) with FiLM conditioning on the global features.
- **Physics-consistent head:** predict N−1 non-negative increments via softplus, take the cumulative sum as ΔP̂(s)/Pa, then FFR̂(s) = 1 − cumsum. This guarantees FFR(0) = 1 and a monotone curve by construction.
- **Loss:** MSE over the whole FFR curve + 2 × MSE on distal FFR.
- **Training:** AdamW, lr 1e-3 with cosine decay, batch 256, up to 50 epochs, early stopping on val distal MAE, mixed precision on GPU. Logs go to `runs/<name>/` (`metrics.csv`, `config.yaml`, `best.pt`, normalisation stats).

### Tasks
- [x] Dataset/dataloader over `.npz` shards
- [x] Model + head + unit tests (output shape, monotonicity, FFR(0) = 1)
- [x] Training script and CLI command
- [x] Baseline for comparison: gradient boosting on hand features (max DS, lesion length, D0, L, Pa, R_micro) → distal FFR
- [x] `evaluate.py` → `reports/surrogate_eval.md` + figures: distal MAE, curve MAE, Bland–Altman (bias, 95% limits of agreement), agreement / sensitivity / specificity at the 0.80 cut-off vs the solver, error by DS bin, OOD results, baseline comparison
- [ ] *(Stretch)* 5-seed ensemble; report the standard deviation as an uncertainty estimate

**Done when (test set vs solver):** distal-FFR MAE ≤ 0.015; 95% limits of agreement within ±0.04; ≥ 97% agreement at the 0.80 cut-off; OOD numbers reported (no target).

---

## M4: ONNX export and runtime (`surrogate/export.py`, `inference/onnx_runtime.py`)

- [x] Export with opset 17 and a dynamic batch axis (N fixed at 256; resampling happens in pre-processing)
- [x] `SurrogateRunner` class: loads the `.onnx` file + normalisation stats and uses the shared `features.py`
- [x] Parity test: max |FFR_torch − FFR_onnx| < 1e-4 on 1,000 test samples
- [x] Benchmark → `reports/onnx_benchmark.md` (record CPU model): median single-sample latency < 10 ms; batch of 1,024 < 300 ms
- [x] *(Optional)* dynamic INT8 quantisation; keep it only if distal MAE rises by < 0.002 — evaluated in every export, not kept: no speed-up (DECISIONS D18)

**Done when:** parity and benchmark targets are met and documented.

---

## M5: Imaging: ARCADE segmentation + 2D QCA (`src/vstent/imaging/`)

### Data
- [ ] `vstent data arcade`: download the ARCADE dataset from Zenodo (record 10390295, `arcade.zip`, ~450 MB) into `data/raw/arcade/`, then verify the MD5 checksum shown on the Zenodo page. If the download is blocked, **stop and ask the human** to download it manually.
- [ ] Inspect the real folder structure and COCO categories (do not assume them); document both and the licence in `docs/DATA.md`
- [ ] Build binary vessel masks = union of all segment polygons from the `syntax` task. Use the `stenosis` task only for evaluation.

### Segmentation
- [ ] U-Net (`segmentation-models-pytorch`, `resnet34` or `efficientnet-b0` encoder, ImageNet weights, `in_channels=1`), 512 × 512 input, Dice + BCE loss
- [ ] Augmentations: rotation ±15°, scale 0.9–1.1, brightness / contrast / gamma, Gaussian noise, CLAHE. No vertical flips.
- [ ] Evaluation → `reports/seg_eval.md`. Target: val Dice ≥ 0.75 (stretch 0.80). If it is not reached after reasonable tuning, report the real value and continue.
- [ ] Export the segmentation model to ONNX + parity test

### Centreline and QCA
- [ ] `centerline.py`: skeletonise the mask, prune spurs < 10 px, build a pixel graph, snap the user's start and end points to the nearest skeleton nodes, find the shortest path (networkx), smooth it (Savitzky–Golay), resample at uniform arc length
- [ ] `qca.py`: diameter at each centreline point = width of the probability map along the local normal at the 0.5 level, with sub-pixel interpolation (fallback: 2 × distance transform). Convert px → mm with a calibration factor (default 0.30 mm/px, configurable and clearly flagged as an assumption). Then compute the reference diameter, %DS, lesions and MLD using the M1 functions
- [ ] Synthetic image tests: render curved tubes with a known diameter function including a 60% stenosis, plus blur and noise. Targets: median absolute diameter error ≤ 0.7 px; MLD location within 3 px; DS within ±7 percentage points
- [ ] `inference/pipeline.py`: image → mask → path → `VesselProfile` → solver and surrogate
- [ ] *(Stretch)* qualitative gallery `reports/figures/qca_examples/`: 20 val images, longest path auto-selected, with ARCADE stenosis boxes overlaid

**Done when:** the segmentation report exists, the QCA synthetic tests pass, and the pipeline runs on an ARCADE image end-to-end.

---

## M6: Virtual stent + plan optimiser (`src/vstent/planning/`)

- [ ] `stent.py`: `apply_stent(profile, s_start, length, diameter, expansion=1.0, edge_mm=1.0)`. Over the stented segment, r_new(s) = max(r(s), expansion · diameter/2), with cosine blending at the edges over `edge_mm`. The lumen is never reduced. Support multiple stents.
- [ ] `configs/stent_catalog.yaml` (generic): diameters 2.25–4.00 mm in 0.25 mm steps; lengths 8, 12, 15, 18, 22, 26, 30, 34, 38 mm
- [ ] `optimizer.py`:
  - **Candidates:** catalogue diameters × lengths × start positions (0.5 mm steps) that cover the lesion MLD
  - **Constraints:** diameter ≤ 1.1 × reference diameter at the distal landing zone; both landing zones in segments with DS < 30%; stent inside the vessel
  - **Objective:** among plans reaching the target post-FFR (default 0.90), choose the minimum total stent length; if none reaches it, maximise post-FFR
  - **Output:** top 3 plans + Pareto front (length vs post-FFR)
  - **Multi-lesion:** allow up to 2 stents (combine the top-k single-stent candidates for each lesion)
  - Evaluate in batches through the ONNX surrogate, with the solver as a switchable backend
- [ ] Performance: ~5,000 candidates in < 1 s on CPU
- [ ] Tests: focal lesion → the best plan covers the MLD, post-FFR ≥ 0.90 and is the shortest such plan; tandem lesions → handled with one long or two stents; every constraint enforced

**Done when:** all tests pass and the performance target is met.

---

## M7: Streamlit app (`src/vstent/app/streamlit_app.py`)

- [ ] Permanent banner: "Research use only — not for clinical decisions"
- [ ] **Image mode:**
  1. Upload PNG/JPG (DICOM optional via `pydicom`: use the first frame; read pixel spacing from `ImagerPixelSpacing` if present and note that magnification is not corrected)
  2. Segmentation overlay
  3. Click start and end points (`streamlit-image-coordinates`)
  4. Calibration input
- [ ] **Synthetic mode:** pick or generate a synthetic vessel (for demos without images)
- [ ] Plots (Plotly): diameter profile with reference and lesions marked; FFR pullback from the solver and the surrogate (with their difference); 0.80 line
- [ ] Virtual stent controls (start, catalogue length, catalogue diameter) → instant post-PCI FFR curve and distal value
- [ ] "Suggest plan" button → top-3 table; the selected plan is drawn on the profile and the image
- [ ] Export the case report (JSON + PNG of the plots); the report includes the disclaimer and all assumptions (calibration, flow mode, model versions)
- [ ] Smoke test with `streamlit.testing.v1.AppTest` running synthetic mode end-to-end

**Done when:** `vstent app` launches, the smoke test passes, and any slider update after segmentation takes < 300 ms.

---

## M8: Reports, docs, demo

- [ ] `vstent demo`: runs a synthetic case end-to-end (profile → FFR → optimise → report) and writes `reports/demo/`
- [ ] `reports/mvp_report.md`, **assembled by a script** from the generated report files (no hand-typed numbers): physics checks, dataset card summary, surrogate metrics, ONNX benchmark, segmentation Dice, QCA test results, optimiser tests, limitations, Phase 2 next steps
- [ ] `docs/INVENTION_NOTES.md`: dated technical notes on potentially novel elements, for discussion with a patent attorney. Candidates: the monotone-increment surrogate head, the optimiser formulation, planned sim-to-real training. Note that this is a record of ideas, not a patent draft or legal advice.
- [ ] README: setup, commands, screenshots, limitations, disclaimer
- [ ] Fresh-clone test: follow the README from scratch in a new directory; fix anything that breaks
- [ ] All checks pass; `git tag v0.1.0`

---

## Phase 2 backlog (do NOT start without human approval)

- 3D CFD labels: coronary trees from ImageCAS / ASOCA, steady hyperemic simulations (SimVascular or OpenFOAM); multi-fidelity surrogate training
- Graph neural network / graph transformer surrogate for vessel trees with side branches; flow split by Murray's law
- Two-view 3D reconstruction from DICOM angles; foreshortening correction
- Cine DICOM: end-diastolic frame selection; TIMI frame-count flow estimate
- Sim-to-real: render synthetic angiograms from 3D trees to train angiogram → FFR end-to-end
- Calibrated uncertainty and conformal intervals; grey-zone flagging
- Retrospective clinical validation against wire FFR with a partner hospital (needs ethics approval; human-led)
- Regulatory groundwork: IEC 62304 software lifecycle docs, ISO 14971 risk file
