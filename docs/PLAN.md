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

- [x] `vstent data cadica`: download, SHA-256-verify and extract CADICA (42 patients, cine
  angiograms with expert lesion boxes and visual lesion grades; CC BY 4.0).
- [x] **Stenosis grading on real angiograms.** The product's QCA, measured automatically at
  every expert lesion box, against the expert grades:
  - weighted kappa, AUC for ≥ 70% and ≥ 50%, rank correlation, feasibility;
  - patient-level bootstrap;
  - a fixed development / test split by patient; any improvement to the measurement is chosen
    on the development half and reported on the test half.
- [x] **Physics against a bench experiment.** The reduced-order loss model against the FDA
  benchmark nozzle's inter-laboratory wall-pressure data (public domain): net pressure loss at
  throat Re 500–6500, nothing fitted.
- [x] Record sources, licences and hashes in `docs/DATA.md`.
- [ ] [needs human] **Invasive FFR ground truth:** request ICADD (70 Indian patients, angiography
  + invasive FFR; available from its authors, not openly downloadable).

**Done when:**
- the CADICA grading report and the nozzle report exist, generated from code with every
  number traceable;
- missed expectations are reported as measured.

## P4c: Lesion-accurate QCA (added 2026-10-31: the weakest product link found in P4b)

P4b showed that on real angiograms the automatic QCA compresses stenosis severity and reads
total occlusions as open. FFR is computed from this geometry, so this is fixed before new
features.

- [x] **Intensity edges are the QCA default** wherever the vessel map is an automatic
  segmentation (`vstent analyse`, the app, `vstent validate` without masks, the galleries);
  analyst masks keep their own edges (D43). Case reports state the edge method.
- [x] **Model-based lumen** (changed 2026-10-31 from plain densitometry, which overestimated
  severity under edge enhancement; D44): the X-ray imaging chain (Beer–Lambert, blur, the
  system's edge enhancement) is fitted to the straightened vessel, and the lumen radius is read
  from the fit. It measures lumens narrower than the edge detector resolves and lumens narrowed
  along the X-ray. Tested on rendered vessels with known lumens (`vstent lumen-synthetic`).
- [x] **Mask gaps and occlusions:** gaps in the segmentation between aligned vessel ends are
  bridged and measured from the image (never interpolated). A stretch without a measurable
  lumen is reported as "critical stenosis or total occlusion", with the warning that the FFR is
  not valid if the vessel is occluded (a single frame cannot tell 95% from 100%).
- [x] **CADICA re-run:** candidate measurements are explored on the development patients only,
  the choice is committed (D44, D45) before the test patients are run; occlusion-flag
  sensitivity on total occlusions and false-flag rate on lesions graded 0–50%.
- [x] Image-based reports regenerated; invention notes with a first prior-art search.

**Done when:**
- on rendered vessels, the model's mean %DS error is ≤ 10 points for lesions ≥ 85% and for
  eccentric lumens (edges are reported alongside);
- on rendered vessels, an occluded segment in a mask gap is flagged and an open one is not;
- the CADICA report is regenerated from code, and every result is reported as measured.

## P4d: Cine runs on real angiograms (added 2026-10-31, revised 2026-11-01)

P4c left severity on real angiograms compressed, and the imaging-chain model does not
describe real frames well enough to insert lesions into them (D46). Clinical angiograms are
recorded as cine runs, and the clinically validated angiography-FFR methods use them: frames
are chosen in the opacified phase, and the contrast's transit gives the flow. CADICA contains
the full runs (19–101 frames) and, for each, the window of frames its clinicians selected:
real expert ground truth. Parts of P7 move here.

- [x] **Frame-level imaging-chain calibration** (D46): recovers the chain on rendered frames;
  on real frames the settings are not physically consistent, so it stays an option and lesion
  insertion is dropped.
- [x] **Cine input:** a run as a folder of frames or a multi-frame DICOM (frame time when
  present), one loader for the CLI, the app and the evaluations.
- [x] **Opacification curve and analysis window:** vessel area per frame from the
  segmentation → contrast arrival, the opacified window and the analysis frame. Agreement with
  the CADICA clinicians' selected frames (development patients; test patients once).
- [x] **Contrast transit:** arrival time of the contrast front along the analysed vessel →
  front velocity (frame count), with its uncertainty, as an optional patient-specific flow for
  the FFR model. Tested on synthetic cine runs with a known front velocity and cardiac
  motion; on CADICA reported descriptively (no flow ground truth, no frame rate).
- [x] `vstent analyse` and the app accept a run and pick the analysis frame; reports and
  invention notes updated.

**Done when:**
- on synthetic runs, the analysis frame lies in the opacified window and the front velocity
  is recovered within 15%;
- the CADICA window-agreement report is generated from code, with every result reported as
  measured.

## P4e: Real coronary physiology (added 2026-11-02; pre-registered before the data were downloaded)

**Data** (open, CC0, no sign-in; no angiograms, so these test the model's physiology
assumptions, not the image pipeline):
- **DEFINE-FLOW:** Johnson, Kirkeeide and Gould, JACC Cardiovasc Interv 2021,
  doi:10.1016/j.jcin.2021.07.041. Data: Dryad doi:10.5061/dryad.h18931zm6 (Zenodo 5508241).
  455 subjects and 668 lesions; site and core-lab FFR and CFR; Doppler average peak velocity
  (APV) at baseline and at hyperemia; aortic and distal pressures; raw pressure, Doppler and
  ECG tracings.
- **CONTRAST:** Johnson et al., JACC Cardiovasc Interv 2016, doi:10.1016/j.jcin.2015.12.273.
  Data: Dryad doi:10.5061/dryad.f76nv (Zenodo 4992130). Raw Pa, Pd and ECG tracings of 763
  subjects at rest, during intracoronary contrast and during adenosine.

Analyses use core-lab-accepted measurements only. An absolute mean velocity is k × APV, with
k bracketed at 0.5 (Doucette et al., Circulation 1992, parabolic profile) and 0.8 (in vitro,
Ultrasound Med Biol 1999), because the true factor varies; ratios do not depend on k.

- [x] `vstent data define-flow` and `vstent data contrast`: download, MD5 check, extract into
  `data/raw/`; source and licence in `docs/DATA.md`.
- [x] **Phasic flow profile** (independent of k): per resting beat, fit the P7 flow-time step
  profile (systole duration after the R wave, diastolic / systolic ratio) to the Doppler
  velocity between R waves. Compare with the P7 prior (0.31 s; ratio 2.0; generator range
  1.5–2.5). **Rule:** if the median resting ratio lies outside 1.5–2.5, or differs from 2.0
  by more than 25%, the prior is replaced by the measured median (systole likewise); P7's
  calibration and held-out report are then rerun on their seeds, and a decision is recorded.
  Otherwise the prior stays and the real distribution is reported. **Result:** median 1.73
  (IQR 1.50–2.01, 455 lesions; LAD 1.77, LCx 1.81, RCA 1.35); systole 0.39 s. The rule did
  not fire; the prior stays (D56).
- [x] **Hyperemic velocity:** the stenosis-free hyperemic mean velocity per lesion, estimated
  as k × vAPV / FFR_c (FFR = the ratio of maximal flows, measured at the same distal site),
  against the fixed 0.35 m/s and P1's range of 0.25–0.45 m/s. **Rule:** if 0.35 m/s lies
  outside the interquartile range at both k = 0.5 and k = 0.8, the fixed velocity is revisited
  in a decision, never changed silently. Otherwise it is reported. **Result:** median 0.21
  m/s (k 0.5) and 0.33 m/s (k 0.8); 0.35 m/s is inside the IQR at k 0.8, so the rule did
  not fire.
- [x] **FAVOR Pilot conversion:** the hyperemic mean velocity predicted from the resting one,
  FAVOR(k × bAPV), against the measured k × vAPV: bias and 90% range at k = 0.5 and 0.8.
  FAVOR was fitted on contrast frame counts, not Doppler, so this is reported, not tuned.
  **Result:** median +28% (k 0.5) and +4% (k 0.8); 5th–95th percentile −29% to +72% at k 0.8.
- [x] **Pressures:** Pa at hyperemia against P1's 70–110 mmHg (reported). Pv is not measured.
- [x] **CONTRAST:** if the condition segments can be identified in the tracings, Pd / Pa at
  rest, during contrast and during adenosine: how far a contrast injection is from rest
  (reported). **Result:** the tracings carry no condition markers; not analysed.
- [x] Report: `vstent physiology-real` → `reports/physiology_real.md`.

**Done when:** the report exists and every rule that fires has been acted on.

## P4f: Repeatability on real angiograms (added 2026-11-02; pre-registered before any measurement)

No wire FFR is needed: a measurement that changes when nothing about the patient changes cannot
be accurate. Data: CADICA's expert lesions (all 42 patients; a reporting use only, no method is
chosen on it). A lesion is linked across the keyframes of one run; CADICA does not link lesions
across runs, so between-run repeatability is not measured.

Measurement (the product's steps): segmentation; the vessel path through the expert box,
extended on every side by a margin (the analysed segment); QCA with the product's lumen (D50);
%DS at its worst point inside the box; and the segment's FFR from the reduced-order solver with
configs/physics.yaml. CADICA has no pixel size, so the default 0.3 mm/px is assumed (the same
within a run). This FFR is the lesion segment's, not a whole vessel's. Lesions graded 0–98%;
total occlusions are left out.

- [x] **Frame to frame:** the same lesion on the keyframes of its run (margin = the box size).
  Per lesion with ≥ 2 measured frames: the within-lesion standard deviation and the
  coefficient of variation (CV; root-mean-square over lesions, as in QREP). **Targets:**
  - %DS CV ≤ 11.9%: QCA's inter-observer CV with operator-selected frames (Ito et al.);
  - FFR CV ≤ 9.4%: QFR's inter-observer CV in QREP (EuroIntervention 2022), which includes
    each observer's own frame choice.

  **Result:** %DS CV 44.6% and FFR CV 18.7% (414 lesions): **both not met** (D57).
- [x] **Operator to operator:** on each keyframe, two simulated operators choose the analysed
  segment's margin independently, uniformly from 0.5–1.5 × the box size (fixed seeds). The
  CV between the two. **Targets:** FFR CV ≤ 9.4% (QREP); %DS standard deviation of the
  differences ≤ 7 points (QCA inter-observer, operator-selected frames). Real operators also
  choose frames and points, so this is a lower bound of operator variability. **Result:**
  FFR CV 17.1% (**not met**); %DS SD of differences 6.0 points (met).
- [x] Also reported, no targets: agreement on the side of 0.80 for pairs of frames and of
  operators; results by expert grade; patient-level bootstrap intervals.
- [x] Report: `vstent repeatability` → `reports/repeatability.md`.

**Done when:** the report exists with every target's result.

## P4g: Multi-frame consensus (added 2026-11-02; pre-registered before any consensus was computed)

P4f found that one automatic frame does not measure a lesion stably (D57). The consensus is
the median %DS and the median FFR over K frames of the same lesion. Data: the same CADICA
per-keyframe measurements as P4f (re-measured with the rows saved; the method is unchanged).
Patients split as in P4b: 21 development, 21 test.

- [ ] **Repeatability of a K-frame consensus:** for each lesion with ≥ 2K measured keyframes,
  two disjoint sets of K frames (alternate keyframes in time order: the 1st, 3rd, … and the
  2nd, 4th, …, the first K of each); the median of each set; the CV between the two medians
  (root mean square over lesions, as in P4f).
- [ ] **K, chosen on the development patients only:** the smallest K in 1–5 whose FFR CV is
  ≤ 9.4% and whose %DS CV is ≤ 11.9% (P4f's targets). If none qualifies, the K with the
  lowest FFR CV.
- [ ] **Test patients, once, with the chosen K:** FFR CV ≤ 9.4% and %DS CV ≤ 11.9% (the same
  targets), plus the two medians' agreement on the side of 0.80 (reported).
- [ ] **Instability flag:** a consensus is flagged when its K frames' FFR values lie on both
  sides of 0.80. On the test patients, reported: the share flagged; among lesions whose two
  medians disagree at 0.80, the share flagged in either set (sensitivity); among those that
  agree, the share flagged in neither.
- [ ] **Product:** for a cine run, the analysis is repeated on up to K end-diastolic frames in
  the opacified window (the user's points move to each frame with the heart, P7). The median
  FFR is reported with every frame's value; the planner uses the profile of the frame
  closest to the median; the flag is shown. `vstent analyse`, the app and `vstent validate`
  use the same function. A single image is still measured once.
- [ ] Tests on a synthetic beating-heart run: the points follow the heart, and the median and
  the flag are computed as specified.
- [ ] Report: `vstent consensus-report` → `reports/consensus.md`.

**Done when:** the report exists with the test result, and the product uses the consensus.

## P5: Projection geometry and two-view 3D reconstruction

- [x] Read the projection geometry from DICOM (`PositionerPrimaryAngle`, `PositionerSecondaryAngle`, `DistanceSourceToDetector`, `DistanceSourceToPatient`, `ImagerPixelSpacing`) → magnification-corrected pixel size at the vessel.
- [x] Synthetic 3D trees projected into two views with known geometry, as test data.
- [x] Epipolar matching of two-view centrelines → 3D centreline; diameters from both views; lengths free of foreshortening.

**Done when:**
- on synthetic two-view pairs, the 3D centreline error is ≤ 1 mm (median) and the length error is ≤ 5%;
- the length error of single-view QCA vs two-view is reported.

## P5b: Absolute lumen size and two-view reliability (added 2026-11-01)

P5 found two weaknesses on synthetic data:
- intensity edges read healthy vessels about 14% narrow (P4c's %DS reports cancel a uniform
  scale, so they could not show this). This biases stent sizing and FFR;
- 5 of 100 two-view reconstructions were off by more than 5% in length, and the check
  flagged none of them.

Method choices are made on fresh development seeds (910000+). They are reported on held-out
seeds (920000+ for the lumen, 7001+ for the two-view report) and, for the lumen, on real
ARCADE frames without ground truth.

- [x] **Scaled lumen:** intensity edges keep the lumen's shape (%DS unchanged by
  construction, so the CADICA grading is unaffected); the imaging-chain fit on the healthy
  part sets the absolute size (one factor per vessel; fallback to edges with a warning when
  the fit fails or the factor is implausible). Used in single view, trees, two views and the
  planner's sizing.
- [x] **Default rule (pre-registered).** `AUTO_LUMEN` becomes the scaled lumen only if both
  hold, otherwise edges stay the default and the scaled lumen is an option:
  1. on held-out renders, its median healthy-diameter error is within ±5% and 90% of views
     are within ±15%;
  2. on the 20 ARCADE gallery frames, the size factor is accepted on ≥ 80% of paths, and its
     median lies within the central 80% of the factors on development renders (real frames
     behave like the renders).
- [x] **Two-view reliability check** that catches poorly conditioned reconstructions,
  designed on development seeds.
- [x] Reports regenerated: the two-view report on fresh seeds with the check's sensitivity
  and false-alarm rate; the lumen report with absolute size; the image-based reports whose
  default changes.

**Done when:**
- the default rule has been applied and its outcome recorded;
- on fresh two-view seeds, the check flags ≥ 80% of reconstructions with |length error|
  > 5% while flagging ≤ 10% of all cases; or the real result is reported.

## P5c: Two views in the app (added 2026-11-01)

- [x] The app accepts a second DICOM view (when the first has projection geometry) and offers
  a synthetic two-view example (rendered masks, true landmarks), so the feature can be shown
  without patient data.
- [x] View-2 points are clicked on the second image, guided by the epipolar lines of the view-1
  points. Both views show the 3D centreline projected back and the stents along it.
- [x] Results, stent controls, planner, intervals and the case report use the 3D profile; the
  report stores both views, the 3D and single-view lengths, the ray gap, the residual, the
  shift and the smoothing sensitivity.

**Done when:** the synthetic example runs end to end in the app (AppTest), the same landmark
lies on its epipolar guide, and the checks pass.

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

- [x] Multi-frame DICOM: end-diastolic frame selection (ECG if present, image-based otherwise). (Run loading and the opacified analysis window: P4d.)
- [x] TIMI frame count → a patient-specific flow velocity instead of a fixed hyperemic velocity, with its uncertainty feeding P1. (Contrast-front transit on single runs: P4d.) Available on request; the fixed velocity stays the default until real flow confirms it (D55).

**Done when:** frame selection and frame counting are tested on synthetic cine sequences, and on real cine examples if the human supplies de-identified, licensed ones.

**P7 plan (pre-registered 2026-11-01, before any beating-heart run is rendered or measured):**

Development seeds 9100+ (methods and settings chosen here); held-out seeds 9200+ (the report,
run once). CADICA development patients for real-data choices; CADICA test patients once.

- [x] Synthetic beating-heart cine runs (`vstent cine-synthetic`): P5's 3D trees on a heart
  sphere whose radius follows a cardiac volume curve: largest at end-diastole (the ECG R
  wave), end-systolic radius 8–15% smaller, heart rate 50–110 bpm, frame rate 7.5, 15 or
  30 fps, 2–5 beats per run. Breathing drift (a slow 3D shift). A contrast front advances
  along every branch at a known mean velocity (resting runs 0.10–0.35 m/s), then washes out.
  The D39 imaging chain renders each frame. Each case has two runs from two views (P5),
  with their own heart-cycle phase and breathing. Coronary flow is phasic: the front moves
  faster in diastole (diastolic / systolic speed ratio 1.5–2.5); the true velocity is the
  cycle average. Each run is written as a DICOM with FrameTime and C-arm geometry; half the
  runs also carry an ECG waveform (WaveformSequence) and the R-wave pointer.
- [x] End-diastolic (ED) frames:
  - from the DICOM when present: the R-wave pointer (0028,6040); else R peaks of the ECG
    waveform (band-pass, derivative energy, 250 ms refractory period), mapped to frames
    with FrameTime (or the frame time vector);
  - otherwise from the images: the spread of the segmented vessel tree (RMS distance of
    vessel pixels from their centroid), detrended for opacification; its period from the
    autocorrelation; ED frames are its maxima, one per beat (the heart, and with it the
    epicardial tree, is largest at end-diastole);
  - the analysis frame becomes the ED frame inside the opacified window (P4d) with the
    largest smoothed vessel area. With no ED frame in the window, P4d's peak frame is kept,
    with a note.

  **Targets (held-out synthetic runs):** ED frames from the ECG within ±1 frame of the true
  ED frame for ≥ 98% of beats; image-based ED frames within ±1 frame for ≥ 80% of beats at
  15 fps; analysis frame within ±1 frame of a true ED frame in ≥ 80% of runs.
  **Real data (CADICA: no ECG, no ED labels):** the image-based rule's settings are chosen on
  the development runs. On the test runs its analysis frame lies inside the clinicians'
  selected span in ≥ 83% of runs: M5's peak rule reached 88%, the current segmenter's 82%
  (D54).
  **Results:** ECG 347 of 348 beats (met); image-based at 15 fps 79% (**not met**, target
  80%); analysis frame 95% of runs (met); CADICA test 149 of 177 (84%, met; run curves at
  probability ≥ 0.9, chosen on the development runs, D55).
- [x] Flow velocity from the contrast transit:
  - CFV (contrast flow velocity) = the P4d front speed (Theil–Sen) × the pixel size at the
    vessel ÷ the frame time; on a two-view path (both runs at their ED frames), the 3D arc
    length is used instead;
  - one view foreshortens the path, so its CFV is biased low (P5b: lengths −13% median). Its
    point estimate is not corrected; its 95% interval's upper end is multiplied by the 95th
    percentile of true / single-view length on the P5 development cases (seeds 5001), and
    the user is told that two views remove the bias;
  - a run marked hyperemic (adenosine; the user says so, DICOM does not record it) uses CFV
    as the hyperemic velocity. A resting run uses the FAVOR Pilot relation HFV = 0.10 +
    1.55·CFV − 0.93·CFV² (m/s), as quoted by Tar et al., Cardiol J 2021;28:163–165,
    doi:10.5603/CJ.a2020.0144;
  - guards: a forward-moving front (P4d), a frame time and a calibrated pixel size, CFV
    within 0.05–0.80 m/s (the quadratic rises up to 0.83 m/s), and the 95% interval's
    half-width ≤ 50% of CFV. If any guard fails, the fixed 0.35 m/s is used, with a note;
  - the CFV's 95% interval, passed through the conversion, replaces P1's ±0.10 m/s velocity
    half-width in the input Monte Carlo.

  **Targets (held-out synthetic cases, true velocity known, pixel size from the DICOM):**
  - two views (3D path): median |CFV error| ≤ 10%; ≥ 90% of accepted cases within ±25%;
    the 95% interval covers the true velocity in ≥ 85% of accepted cases; median
    |FFR(measured velocity) − FFR(true velocity)| ≤ 0.02;
  - one view: the widened interval covers the true velocity in ≥ 85% of accepted runs;
    the median error is reported;
  - the guards reject ≤ 20% of runs.

  (Amended 2026-11-02, before any beating-heart run existed: the first version set the
  accuracy targets for any run, but one view cannot meet them because of foreshortening.)

  **Results** (`reports/cine_synthetic.md`; methods chosen on development seeds, D55: motion
  tracking on the opacification plateau, flow time for phasic flow, 90% split-conformal
  intervals instead of the Theil–Sen 95% interval, a two-run agreement guard). Two views:
  median error 5.8%, 98% within ±25%, coverage 95%, FFR 0.005 (all met). One view: coverage
  95% (met), median error −16%. Guards rejected 13% of runs (met).
- [x] Product: the fixed velocity stays the default until a measured velocity has been
  checked against real flow (a phantom or a Doppler wire, HUMAN_ACTIONS). `vstent analyse
  <run>` and the app also show the FFR from the measured velocity, with its interval and
  source (rest / hyperemic), and the user can choose it as the primary value. The app's
  run view shows the ED frames on the opacification curve. The case report stores the ED
  source, the CFV, the conversion and the guards.
- [x] Reports: `reports/cine_synthetic.md` (ED frames and velocity on the held-out
  runs); `reports/cine_frames.md` adds the ED rule on CADICA.
- [x] Regulatory: requirements for ED selection and the measured velocity; H-11 and H-12
  updated (the transit becomes an optional FFR input), with a hazard for a wrong measured
  velocity.

## P8: Sim-to-real angiograms and regulatory groundwork [needs human: regulatory review]

- [x] Render synthetic angiograms from 3D trees (projection, contrast, noise, overlap) to train and test segmentation (sim-to-real plan below; D53). (Lesion insertion into real angiograms was tried in P4d and dropped, D46.)
- [ ] Angiogram → FFR models trained on rendered angiograms: deferred until there are labels
  better than the reduced-order model's own output (CFD, P6, or wire FFR, P4); D53.
- [x] Regulatory groundwork drafts (not regulatory advice; `docs/regulatory/`, D48):
  - intended-use statement;
  - IEC 62304 software safety class rationale;
  - software development plan;
  - requirements and architecture;
  - SOUP list;
  - ISO 14971 hazard analysis draft.
  Requirements, hazards, traceability and the SOUP list (with an SBOM) are generated by
  `vstent regulatory` and checked against the code by the test suite.

**Done when:**
- rendered angiograms improve segmentation on the ARCADE test split, or the real result is reported;
- the regulatory drafts exist and are marked for review by a qualified person.

**Sim-to-real plan (pre-registered 2026-11-01, before any rendered data or training):**
- [x] `vstent synth-angio`: rendered angiograms from the P5 3D trees (one or two trees from one
  ostium, random clinical views, the ARCADE-fitted imaging chain of D39). Distractors are
  drawn but not labelled: a contrast-filled catheter, smooth anatomy-like shadows, rib-like
  bands, and small branches. Labels follow ARCADE's annotation of the major (SYNTAX)
  segments: the lumen of every vessel whose reference diameter is ≥ 1.5 mm. Splits: train
  2,000, val 200, test 200; dataset card. Done: `reports/synth_angio_card.md`.
- [x] Training `configs/seg_synth.yaml`: the M5 configuration (same seed, schedule, early
  stopping) with 1,000 rendered images added to ARCADE's 1,000 training images. The epoch is
  chosen on ARCADE val, as in M5. Done: 57 epochs, best ARCADE val Dice 0.8278 at epoch 42.
- [x] Evaluation: the new run on ARCADE val and test (`reports/seg_synth/`); both models on
  the rendered test split; `reports/sim_to_real.md` with the paired per-image Dice
  difference on ARCADE test and its 95% bootstrap interval. Done: ARCADE test +0.0022
  (−0.0052 to +0.0095).
- [x] **Adoption rule:** the new model replaces the product's segmenter only if its mean
  ARCADE test Dice is higher than the M5 model's and the lower bound of the paired
  difference's 95% interval is above −0.01. Otherwise M5 stays and the result is reported.
  Outcome: met (0.7947 vs 0.7925; lower bound −0.0052), so the new model is exported and is
  the product's segmenter (D53).
