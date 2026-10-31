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
- [x] `vstent data arcade`: download the ARCADE dataset from Zenodo (record 10390295, `arcade.zip`, ~450 MB) into `data/raw/arcade/`, then verify the MD5 checksum shown on the Zenodo page. If the download is blocked, **stop and ask the human** to download it manually.
- [x] Inspect the real folder structure and COCO categories (do not assume them); document both and the licence in `docs/DATA.md`
- [x] Build binary vessel masks = union of all segment polygons from the `syntax` task. Use the `stenosis` task only for evaluation.

### Segmentation
- [x] U-Net (`segmentation-models-pytorch`, `resnet34` or `efficientnet-b0` encoder, ImageNet weights, `in_channels=1`), 512 × 512 input, Dice + BCE loss
- [x] Augmentations: rotation ±15°, scale 0.9–1.1, brightness / contrast / gamma, Gaussian noise, CLAHE. No vertical flips.
- [x] Evaluation → `reports/seg_eval.md`. Target: val Dice ≥ 0.75 (stretch 0.80). If it is not reached after reasonable tuning, report the real value and continue.
- [x] Export the segmentation model to ONNX + parity test

### Centreline and QCA
- [x] `centerline.py`: skeletonise the mask, prune spurs < 10 px, build a pixel graph, snap the user's start and end points to the nearest skeleton nodes, find the shortest path (networkx), smooth it (Savitzky–Golay), resample at uniform arc length
- [x] `qca.py`: diameter at each centreline point = width of the probability map along the local normal at the 0.5 level, with sub-pixel interpolation (fallback: 2 × distance transform). Convert px → mm with a calibration factor (default 0.30 mm/px, configurable and clearly flagged as an assumption). Then compute the reference diameter, %DS, lesions and MLD using the M1 functions
- [x] Synthetic image tests: render curved tubes with a known diameter function including a 60% stenosis, plus blur and noise. Targets: median absolute diameter error ≤ 0.7 px; MLD location within 3 px; DS within ±7 percentage points
- [x] `inference/pipeline.py`: image → mask → path → `VesselProfile` → solver and surrogate
- [x] *(Stretch)* qualitative gallery `reports/figures/qca_examples/`: 20 val images, longest path auto-selected, with ARCADE stenosis boxes overlaid

**Done when:** the segmentation report exists, the QCA synthetic tests pass, and the pipeline runs on an ARCADE image end-to-end.

---

## M6: Virtual stent + plan optimiser (`src/vstent/planning/`)

- [x] `stent.py`: `apply_stent(profile, s_start, length, diameter, expansion=1.0, edge_mm=1.0)`. Over the stented segment, r_new(s) = max(r(s), expansion · diameter/2), with cosine blending at the edges over `edge_mm`. The lumen is never reduced. Support multiple stents.
- [x] `configs/stent_catalog.yaml` (generic): diameters 2.25–4.00 mm in 0.25 mm steps; lengths 8, 12, 15, 18, 22, 26, 30, 34, 38 mm
- [x] `optimizer.py`:
  - **Candidates:** catalogue diameters × lengths × start positions (0.5 mm steps) that cover the lesion MLD
  - **Constraints:** diameter ≤ 1.1 × reference diameter at the distal landing zone; both landing zones in segments with DS < 30%; stent inside the vessel
  - **Objective:** among plans reaching the target post-FFR (default 0.90), choose the minimum total stent length; if none reaches it, maximise post-FFR
  - **Output:** top 3 plans + Pareto front (length vs post-FFR)
  - **Multi-lesion:** allow up to 2 stents (combine the top-k single-stent candidates for each lesion)
  - Evaluate in batches through the ONNX surrogate, with the solver as a switchable backend
- [x] Performance: ~5,000 candidates in < 1 s on CPU
- [x] Tests: focal lesion → the best plan covers the MLD, post-FFR ≥ 0.90 and is the shortest such plan; tandem lesions → handled with one long or two stents; every constraint enforced

**Done when:** all tests pass and the performance target is met.

---

## M7: Streamlit app (`src/vstent/app/streamlit_app.py`)

- [x] Permanent banner: "Research use only — not for clinical decisions"
- [x] **Image mode:**
  1. Upload PNG/JPG (DICOM optional via `pydicom`: use the first frame; read pixel spacing from `ImagerPixelSpacing` if present and note that magnification is not corrected)
  2. Segmentation overlay
  3. Click start and end points (`streamlit-image-coordinates`)
  4. Calibration input
- [x] **Synthetic mode:** pick or generate a synthetic vessel (for demos without images)
- [x] Plots (Plotly): diameter profile with reference and lesions marked; FFR pullback from the solver and the surrogate (with their difference); 0.80 line
- [x] Virtual stent controls (start, catalogue length, catalogue diameter) → instant post-PCI FFR curve and distal value
- [x] "Suggest plan" button → top-3 table; the selected plan is drawn on the profile and the image
- [x] Export the case report (JSON + PNG of the plots); the report includes the disclaimer and all assumptions (calibration, flow mode, model versions)
- [x] Smoke test with `streamlit.testing.v1.AppTest` running synthetic mode end-to-end

**Done when:** `vstent app` launches, the smoke test passes, and any slider update after segmentation takes < 300 ms.

---

## M8: Reports, docs, demo

- [x] `vstent demo`: runs a synthetic case end-to-end (profile → FFR → optimise → report) and writes `reports/demo/`
- [x] `reports/mvp_report.md`, **assembled by a script** from the generated report files (no hand-typed numbers): physics checks, dataset card summary, surrogate metrics, ONNX benchmark, segmentation Dice, QCA test results, optimiser tests, limitations, Phase 2 next steps
- [x] `docs/INVENTION_NOTES.md`: dated technical notes on potentially novel elements, for discussion with a patent attorney. Candidates: the monotone-increment surrogate head, the optimiser formulation, planned sim-to-real training. Note that this is a record of ideas, not a patent draft or legal advice.
- [x] README: setup, commands, screenshots, limitations, disclaimer
- [x] Fresh-clone test: follow the README from scratch in a new directory; fix anything that breaks
- [x] All checks pass; `git tag v0.1.0`

---

## Phase 2 backlog (approved by the human on 2026-10-28; worked through as P1–P8 below)

- 3D CFD labels: coronary trees from ImageCAS / ASOCA, steady hyperemic simulations (SimVascular or OpenFOAM); multi-fidelity surrogate training
- Graph neural network / graph transformer surrogate for vessel trees with side branches; flow split by Murray's law
- Two-view 3D reconstruction from DICOM angles; foreshortening correction
- Cine DICOM: end-diastolic frame selection; TIMI frame-count flow estimate
- Sim-to-real: render synthetic angiograms from 3D trees to train angiogram → FFR end-to-end
- Calibrated uncertainty and conformal intervals; grey-zone flagging
- Retrospective clinical validation against wire FFR with a partner hospital (needs ethics approval; human-led)
- Regulatory groundwork: IEC 62304 software lifecycle docs, ISO 14971 risk file

---

## Phase 2 plan

Order: what removes the biggest known weakness, and can run on this laptop (RTX 3060 6 GB, Ryzen 7 6800H) without outside help, comes first. Items that need data registrations, paid compute or a hospital partner come later and are marked **[needs human]**. The MVP stays tagged `v0.1.0`. Phase 2 milestones keep the same rules: tests, reports from clean commits, decisions, progress log, no patient data in the repo, and a check-in with the human after each milestone.

| # | Milestone | Main weakness addressed | Needs from the human |
|---|---|---|---|
| P1 | Uncertainty and grey-zone flagging | single numbers without confidence | — |
| P2 | Vessel trees with side branches (physics + image) | single-tube physics overstates loss in long paths | — |
| P3 | Tree surrogate (graph network) + ONNX | surrogate only handles single tubes | — |
| P4 | Clinical-validation readiness (protocol + analysis tools) | no comparison with wire FFR | a hospital partner, ethics approval (later) |
| P5 | Projection geometry and two-view 3D reconstruction | assumed pixel size, foreshortening, overlap | multi-view DICOM data (later; built on synthetic first) |
| P6 | 3D CFD labels and multi-fidelity training | reduced-order model error unknown | dataset registration (ImageCAS / ASOCA), compute budget |
| P7 | Cine angiography: frame selection and frame-count flow | fixed hyperemic velocity for everyone | cine DICOM examples |
| P8 | Sim-to-real angiograms and regulatory groundwork documents | domain gap; regulatory path | regulatory adviser to review |

## P1: Uncertainty and grey-zone flagging

- [x] **Model uncertainty:** an ensemble of 5 surrogates (M3 config, seeds 0–4; resumable training). The mean and spread are exported to ONNX and used by `SurrogateRunner`.
- [x] **Split-conformal 90% intervals** for distal FFR, calibrated on a dedicated `cal` split (D29), reported on `test` and `ood`.
- [x] **Input-uncertainty propagation:** Monte Carlo through the solver over the assumptions that matter most, giving a 90% FFR interval per case:
  - pixel size (±10% when assumed);
  - hyperemic velocity (its sampling range);
  - Pa / Pv;
  - QCA diameter error (from `reports/qca_synthetic.md`).
- [x] **Grey-zone flag:** a case or plan is flagged when its interval crosses 0.80 (pre-PCI) or the planner target (post-PCI). Shown in the app, the case report and `vstent analyse`.
- [x] Report `reports/uncertainty.md`.

**Done when:**
- conformal coverage on `test` is 88–92% for a 90% target;
- OOD coverage is reported (no target);
- the ensemble-mean distal MAE is ≤ the single model's;
- the app shows intervals and flags, while slider updates stay < 300 ms.

The report must state that model intervals cover only surrogate-vs-solver error, and input intervals cover only the listed assumptions.

## P2: Vessel trees with side branches

- [x] **Tree physics** (`physics/tree.py`):
  - segments use the M1 viscous and expansion coefficients;
  - each terminal branch gets one outlet resistance, split from the total microvascular resistance by Murray's law (flow ∝ diameter^k, k configurable, default 2.33);
  - calibrated so the stenosis-free tree carries the reference flow;
  - the steady nonlinear network is solved by Newton iteration, giving FFR at every node.
- [x] **Sanity checks:**
  - a tree with one segment reproduces the M1 solver exactly;
  - mass is conserved at every junction;
  - a healthy tree splits flow as Murray's law;
  - adding a side branch raises distal main-vessel FFR for the same upstream geometry.
- [x] **Synthetic tree generator:** a main vessel plus 1–4 side branches, with lesions anywhere, including at bifurcations; dataset card.
- [x] **Image path:** QCA on the skeleton *tree* between a start point and every reachable end point (or user-chosen branches) → tree profile → tree physics. Adds `vstent analyse --tree` and virtual stents on a branch path.
- [x] Re-run the ARCADE gallery with tree physics and compare with the single-path version.

**Done when:**
- all sanity checks pass;
- tree solver throughput is ≥ 1,000 trees/s;
- on the ARCADE gallery, cases with extreme single-path FFR (< 0.30 without a severe focal lesion) are reported before and after. There is no target; this is a qualitative check that the side-branch effect behaves plausibly.

## P3: Tree surrogate (graph network) + ONNX

- [x] A graph network over segment graphs: message passing in plain PyTorch (no extra graph library), with node-wise FFR from a head that is monotone along the flow (generalising M3).
- [x] Training on synthetic trees (resumable); evaluation against the tree solver; OOD trees.
- [x] ONNX export with padded graphs (fixed maximum nodes / edges) plus a parity test. Integration in the runner, the planner (stents on branch paths) and the app.

**Done when:**
- node-wise FFR MAE ≤ 0.015 and distal MAE ≤ 0.015 on test trees;
- ONNX parity < 1e-4;
- the planner on trees meets its speed target (to be set from P2 measurements).

## P4: Clinical-validation readiness [needs human: partner, ethics approval]

- [x] `docs/validation/PROTOCOL.md` template:
  - retrospective design;
  - wire FFR (or iFR) as the reference;
  - primary endpoint (agreement at 0.80, AUC);
  - sample-size calculation;
  - inclusion / exclusion criteria;
  - blinding;
  - analysis plan.
- [x] Data specification:
  - required DICOM fields (cine runs, angles, distances, pixel spacing);
  - pressure-wire position and values;
  - a de-identification checklist;
  - local-only processing: no patient data ever enters this repository.
- [x] `vstent validate --manifest <csv>`: runs the pipeline on a local, de-identified case list and reports Bland–Altman, AUC, sensitivity / specificity and per-vessel tables. Tested only on synthetic stand-in cases.

**Done when:** the protocol and data specification exist, and the validation tool runs end to end on synthetic stand-in data. Real data only after the human confirms ethics approval and a data agreement.

## P4b: Validation on public data (added 2026-10-31: no hospital partner, D41)

The human cannot obtain a hospital partner. Validation therefore uses public data with
independent ground truth, part by part. No public dataset pairs angiograms with invasive FFR
(searched 2026-10-31), so this measures each component; it does not replace a clinical study.

- [ ] `vstent data cadica`: download, SHA-256-verify and extract CADICA (42 patients, cine
  angiograms with expert lesion boxes and visual lesion grades; CC BY 4.0).
- [ ] **Stenosis grading on real angiograms.** The product's QCA, measured automatically at
  every expert lesion box, against the expert grades:
  - weighted kappa, AUC for ≥ 70% and ≥ 50%, rank correlation, feasibility;
  - patient-level bootstrap;
  - a fixed development / test split by patient; any improvement to the measurement is chosen
    on the development half and reported on the test half.
- [ ] **Physics against a bench experiment.** The reduced-order loss model against the FDA
  benchmark nozzle's inter-laboratory wall-pressure data (public domain): net pressure loss at
  throat Re 500–6500, nothing fitted.
- [ ] Record sources, licences and hashes in `docs/DATA.md`.
- [ ] [needs human] **Invasive FFR ground truth:** request ICADD (70 Indian patients, angiography
  + invasive FFR; available from its authors, not openly downloadable).

**Done when:**
- the CADICA grading report and the nozzle report exist, generated from code with every
  number traceable;
- missed expectations are reported as measured.

## P5: Projection geometry and two-view 3D reconstruction

- [ ] Read the projection geometry from DICOM (`PositionerPrimaryAngle`, `PositionerSecondaryAngle`, `DistanceSourceToDetector`, `DistanceSourceToPatient`, `ImagerPixelSpacing`) → magnification-corrected pixel size at the vessel.
- [ ] Synthetic 3D trees projected into two views with known geometry, as test data.
- [ ] Epipolar matching of two-view centrelines → 3D centreline; diameters from both views; lengths free of foreshortening.

**Done when:**
- on synthetic two-view pairs, the 3D centreline error is ≤ 1 mm (median) and the length error is ≤ 5%;
- the length error of single-view QCA vs two-view is reported.

## P6: 3D CFD labels and multi-fidelity training [needs human: data registration, compute]

- [ ] Coronary trees from ImageCAS and / or ASOCA (registration and licence acceptance by the human).
- [ ] Surface and volume meshing, then steady hyperemic CFD (OpenFOAM or SimVascular; the choice is recorded). Verified on an analytic Poiseuille pipe and a published stenosis benchmark.
- [ ] Reduced-order vs CFD comparison: the first measurement of the reduced-order model's error.
- [ ] Multi-fidelity surrogate: pre-trained on reduced-order labels, fine-tuned on CFD labels.

**Done when:**
- verification cases are within 5% of the reference;
- the reduced-order vs CFD error is reported for every simulated case;
- the multi-fidelity surrogate beats the reduced-order model against CFD on held-out cases, or the real result is reported.

## P7: Cine angiography

- [ ] Multi-frame DICOM: end-diastolic frame selection (ECG if present, image-based otherwise).
- [ ] TIMI frame count → a patient-specific flow velocity instead of a fixed hyperemic velocity, with its uncertainty feeding P1.

**Done when:** frame selection and frame counting are tested on synthetic cine sequences, and on real cine examples if the human supplies de-identified, licensed ones.

## P8: Sim-to-real angiograms and regulatory groundwork [needs human: regulatory review]

- [ ] Render synthetic angiograms from 3D trees (projection, contrast, noise, overlap) to train and test segmentation and angiogram → FFR models.
- [ ] Regulatory groundwork drafts (not regulatory advice):
  - intended-use statement;
  - IEC 62304 software safety class rationale;
  - software development plan;
  - requirements and architecture;
  - SOUP list;
  - ISO 14971 hazard analysis draft.

**Done when:**
- rendered angiograms improve segmentation on the ARCADE test split, or the real result is reported;
- the regulatory drafts exist and are marked for review by a qualified person.
