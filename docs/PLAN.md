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

- [x] **Repeatability of a K-frame consensus:** for each lesion with ≥ 2K measured keyframes,
  two disjoint sets of K frames (alternate keyframes in time order: the 1st, 3rd, … and the
  2nd, 4th, …, the first K of each); the median of each set; the CV between the two medians
  (root mean square over lesions, as in P4f).
- [x] **K, chosen on the development patients only:** the smallest K in 1–5 whose FFR CV is
  ≤ 9.4% and whose %DS CV is ≤ 11.9% (P4f's targets). If none qualifies, the K with the
  lowest FFR CV. **Result:** none qualified; **K = 5** (FFR CV 12.0%, %DS CV 21.7%).
- [x] **Test patients, once, with the chosen K:** FFR CV ≤ 9.4% and %DS CV ≤ 11.9% (the same
  targets), plus the two medians' agreement on the side of 0.80 (reported). **Result:** FFR CV
  **5.1%, met**; %DS CV **27.0%, not met**; 91% agree at 0.80 (D58).
- [x] **Instability flag:** a consensus is flagged when its K frames' FFR values lie on both
  sides of 0.80. On the test patients, reported: the share flagged; among lesions whose two
  medians disagree at 0.80, the share flagged in either set (sensitivity); among those that
  agree, the share flagged in neither. **Result:** 31.5% of sets flagged; all 10 discordant
  lesions flagged; 62% of concordant lesions flagged in neither set.
- [x] **Product:** for a cine run, the analysis is repeated on up to K end-diastolic frames in
  the opacified window (the user's points move to each frame with the heart, P7). The median
  FFR is reported with every frame's value; the planner uses the profile of the frame
  closest to the median; the flag is shown. `vstent analyse`, the app and `vstent validate`
  use the same function. A single image is still measured once. Done (D58), with two rules
  set on principle: only frames on which the contrast has filled the analysed path, up to the
  end of the opacification plateau; a frame whose path length differs by > 30% is left out.
- [x] Tests on a synthetic beating-heart run: the points follow the heart, and the median and
  the flag are computed as specified.
- [x] Report: `vstent consensus-report` → `reports/consensus.md`.

**Done when:** the report exists with the test result, and the product uses the consensus.

## P4h: Densitometric lumen at lesions (added 2026-11-02; pre-registered before any measurement with it)

Why: the product's lumen reads lesions the CADICA experts graded 70–90% at a median 42% DS and
90–98% at 43%. On those lesions it calls segment FFR ≤ 0.80 in 16–23%, where FAME found wire
FFR ≤ 0.80 in about 80% and 96% of such visual grades. The close-ups (D60) show tight lesions as
a fading of the contrast column, narrower than the blur, so their edges stay apart. The
imaging-chain fit (D44) did not fix this; it smooths the lumen along the vessel and starts
from the edges.

**Method, fixed now** (`imaging/densitometry.py`, lumen option `dense`):
- **Straighten the vessel:** at every centreline point, sample the image along the normal
  every 0.5 px over ± (half the edges' robust reference diameter + 6 px). Average three
  parallel profiles (−1, 0, +1 px along the tangent).
- **Background:** a straight line through the medians of the outer 3 px on each side, its
  two parameters median-filtered over 9 points along the vessel.
- **Integrated contrast (IOD):** the signed sum over the window of (background − image), in
  one of two domains:
  - `log`: log of intensity, for an image proportional to transmitted X-rays;
  - `linear`: grey levels, for an image already log-transformed by the X-ray system.

  Blur and the X-ray system's edge enhancement are linear filters that keep this integral.
  Median over 3 points along the vessel.
- **Lumen diameter:**
  - densitometric diameter = (edges' robust reference diameter) × √(IOD / IOD reference),
    where the IOD reference is `robust_reference` of √IOD;
  - lumen = min(edge diameter, densitometric diameter). Only densitometry can narrow a
    lumen: an overlapping vessel or a branch raises the IOD and is ignored;
  - the absolute size is the scaled lumen's (P5b), so %DS comes from the shape alone.
- No constant above is changed after this commit. The domain is the only choice, made by the
  rule below.
- **Amendment 1** (2026-11-02, before any CADICA measurement with `dense`): after the IOD's
  median over 3 points, a Gaussian along the vessel, σ = 3 px:
  - **Why:** on development renders (seeds 5100+, not the report's 4100+), the contrast left in
    a tight lesion was about the size of the image noise. The narrowest point then picked
    noise dips, and circular 70% lesions read +18 points (many at the 0.5 px floor).
  - **Rule, written before the variants were run:** the smallest σ in {0, 1, 2, 3} px with mean
    |%DS error| ≤ 10 points for 70% lesions and ≤ 8 for vessels without a lesion.
  - **Result:**
    - σ = 0: +18.4 for 70% lesions, +6.2 without a lesion;
    - σ = 1: +17.4, +6.1;
    - σ = 2: +12.8, +6.1;
    - σ = 3: +4.4, +6.1 (85% lesions −2.8, 95% −0.4).
  - σ = 3 px is also the scale of the X-ray system's edge enhancement (1.5–3 px).

**Development** (CADICA development patients, as split in P4b; rendered lesions,
`vstent lumen-synthetic`):
- [x] **Domain:** the variant with the higher weighted kappa against the expert grades on the
  development patients (a tie: `log`). **Result:** `log` (kappa 0.294 vs 0.281).
- [x] **Adopt** `dense` as the product lumen only if, with that domain:
  - on the development patients, both its weighted kappa and its AUC for ≥ 70% lesions are
    higher than the intensity edges' (the D43 / D44 rule), and its median %DS of lesions
    graded 0–20% is ≤ 30%;
  - on the rendered lesions, its mean |%DS error| is ≤ 10 points for circular lesions
    ≥ 85% and ≤ 8 points for vessels without a lesion.

  Otherwise `dense` stays an option and nothing changes.

  **Result: not adopted** (D61).
  - Kappa higher (0.294 vs 0.214); median %DS of 0–20% lesions 19.6.
  - Both rendered targets met (5.9 and 5.5 points).
  - But the AUC for ≥ 70% lesions was lower (0.803 vs 0.818).
- During development, lesions of development patients may be viewed (p11, p12 so far). One
  test patient (p24, run 15) was shown in the app demo before this plan; nothing is tuned on
  it.

**Test** (CADICA test patients, once, every method reported whatever it shows; the test half
was used before in P4b, D44, P4f, P4g and the exploratory FAME comparison of 2026-11-02):
- [x] Weighted kappa and AUC for ≥ 70%, against the intensity edges (reported; target kappa
  ≥ 0.40). **Result:** `dense` kappa 0.25 (edges 0.13), AUC 0.77 (0.71); target not met.
- [x] Median %DS by grade (targets: 70–90% ≥ 50%; 90–98% ≥ 60%; 0–20% ≤ 30%).
  **Result:** `dense` 46% and 49% (edges 44% and 40%): not met; 0–20%: 19%, met.
- [x] If adopted: the P4f measurements re-made with the new lumen. **Not run:** not
  adopted. Then the share of lesions
  whose segment FFR (median of up to 5 keyframes) is ≤ 0.80, by grade (targets: 70–98% ≥ 50%;
  0–50% ≤ 10%), and the P4g repeatability with K = 5 (reported).
- [x] Report: `vstent stenosis-grading --edges intensity,dense`, `vstent lumen-synthetic`,
  `vstent repeatability`, `vstent consensus-report` (the last two not re-run: not adopted).

**Done when:** the decision is recorded, the reports exist with the test results, and the
product uses the rule's outcome.

## P4i: Lesion severity learned from real angiograms (added 2026-11-02; pre-registered before any training)

Why: physics-based lumen methods (D44, D61) read rendered lesions accurately but real ones too
mild, because X-ray systems process images before export. The product only sees real images,
so the lesion reading is learned from real angiograms with expert labels. The physics, the
multi-frame median and the planner stay as they are.

**Inputs (one function, `imaging/lesion_patch.py`, for training, ONNX inference and the
product):**
- A candidate narrowing is a point k of a QCA path (the product's lumen) with reference
  diameter d (px, at least 4).
- **Patch:** 64 × 32 samples, straightened along the centreline: ± 6 d along the path
  (extrapolated along the end tangents) and ± 2 d across. Two channels:
  - log contrast, log(background) − log(image), the background being the median of the
    outer 4 columns on each side;
  - the segmentation probability.
- **Scalars:** the edges' %DS at k ÷ 100, log d, and the measured lesion length ÷ d.

**Model and training (`learn/lesion_model.py`, torch; ONNX for inference):**
- **Network:** three conv blocks (16, 32, 64 channels; 3 × 3, batch norm, ReLU, 2 × 2 max
  pooling), global average pooling, the scalars concatenated, then FC 64, ReLU, dropout 0.2.
  Two heads: %DS (sigmoid × 100) and a significance logit.
- **Training data:**
  - CADICA **development** patients' expert boxes graded 0–98%, at the point of largest
    measured %DS inside each box. %DS target: the grade's midpoint (10, 35, 60, 80, 94);
    significance target: grade ≥ 50%.
  - ARCADE **stenosis train** split: each stenosis box at its point of largest measured %DS
    (significance 1, no %DS target). Plus up to 2 candidates per image farther than one box
    diagonal from every stenosis box (significance 0), chosen with seed 0.
- **Loss:** Huber (δ = 0.1 on %DS ÷ 100) + 0.5 × binary cross-entropy.
- **Augmentation:** mirror across, reverse along, contrast gain × U(0.8, 1.2), Gaussian
  noise σ 0.02, shift ± 2 samples along.
- **Optimiser:** AdamW, lr 1e-3, weight decay 1e-4, batch 64, cosine schedule, 40 epochs
  without early stopping, seed 0.
- **Folds:** 5 folds of the development patients (fixed with seed 0); every fold also trains
  on ARCADE train.
- **The product model** is the mean of the 5 fold models. Every development prediction is
  out-of-fold.

**In the product (lumen `learned`, the product's lumen if adopted):**
- **Candidates:** local minima of the path's diameter with %DS ≥ 15%, greedily from the
  deepest, at least 2 d apart.
- Each candidate's depth is rescaled to the predicted %DS over its extent (where its %DS ≥
  half its minimum): d'(s) = ref(s) − k · (ref(s) − d(s)), with k = predicted ÷ measured
  %DS, k ≤ 4, d' ≥ the 0.5 px floor.
- Everything after (physics, multi-frame median, planner) is unchanged.

**Development** (CADICA development patients, out-of-fold; ARCADE val):
- [x] **Adopt** `learned` only if all of these hold:
  - weighted kappa and AUC for ≥ 70% lesions both higher than the intensity edges' (the
    D43 / D44 / D61 rule; development edges: kappa 0.214, AUC 0.818);
  - median %DS of 0–20% lesions ≤ 30%;
  - on ARCADE val, the share of stenosis boxes read ≥ 50% is higher than the edges', and
    the share of distant candidates read ≥ 50% is at most the edges' + 10 points.

  Otherwise `learned` stays an option and nothing changes.

  **Result: not adopted** (D63).
  - Met: kappa 0.39 vs 0.21; AUC 0.84 vs 0.82; median of 0–20% lesions 13%; ARCADE val
    stenoses read ≥ 50% in 79% vs 36%.
  - Not met: distant candidates read ≥ 50% in 46% vs 7%.

**Test** (once; every method reported whatever it shows):
- [x] CADICA test patients: weighted kappa (target ≥ 0.40); median %DS by grade (targets
  70–90% ≥ 60%, 90–98% ≥ 70%, 0–20% ≤ 30%). **Result:** kappa 0.32 (edges 0.13); 53%, 65%
  and 11% (edges 44%, 40%, 18%). Only the 0–20% target was met. The test half was used before (P4b, D44, P4f,
  P4g, P4h) but never for training.
- [x] ARCADE test: the share of stenosis boxes read ≥ 50% (target ≥ 70%); distant
  candidates read ≥ 50% (target ≤ 20%). **Result:** 73% (edges 29%), met; 29% (edges 4%),
  not met.
- [x] Kemerovo (external: another hospital and other X-ray systems; every box a significant
  lesion, `docs/DATA.md`): the share of lesions read ≥ 50% (target ≥ 70%), a lesion being
  a patient's run (median over its boxes). **Result:** 54% (edges 18%), not met.
- [x] If adopted: the P4f measurements re-made with the new lumen. Then the share of lesions
  whose segment FFR (median of up to 5 keyframes) is ≤ 0.80, by grade (targets: 70–98% ≥ 50%;
  0–50% ≤ 10%), and the P4g repeatability with K = 5 (reported). **Not run:** not adopted.
- [x] Report: `vstent lesion-report` → `reports/lesion_reading.md`.
- **Amendment 2** (2026-11-02, before any development result of P4i was computed): if P4j
  adopts the hybrid segmentation, the lesion samples are re-extracted with it and the reader
  is retrained with the same configuration, and the run trained on the model-only
  segmentation (`runs/lesion_cnn-20261102-171855`) is not evaluated. If P4j does not adopt
  it, that run is evaluated as planned.

**Done when:** the decision is recorded, the report exists with the test results, and the
product uses the rule's outcome.

## P4j: Vessel coverage on real angiograms: hybrid segmentation (added 2026-11-02; pre-registered before any measurement)

Why: on CADICA frames the segmenter (trained on ARCADE, whose masks cover only the SYNTAX
segments) misses most of the tree. On one development frame (p12 run 1) it marks the proximal
and distal vessel but not the mid-vessel holding all three expert lesions (1 of 3 boxes
reached). A training-free vesselness filter (Frangi) reaches all 3. Paths then cannot be
placed on the missed vessels, and lesions sit in gaps.

**Method (`imaging/vesselness.py`):**
- **Vesselness:** Frangi (scikit-image, dark ridges) at σ = 1–5 px × (image size ÷ 512),
  divided by its 99.5th percentile.
- **Added vessels:** the connected components of {vesselness ≥ T} that touch the model's mask
  (probability ≥ 0.5, dilated by 2 px); components < 30 px dropped.
- **Hybrid probability:** max(model, 0.6 on the added pixels).
- The run's curves (opacification, end-diastole) keep counting the model's confident pixels
  (≥ 0.9) and are unchanged.

**Development** (CADICA development patients' expert boxes; ARCADE val vessel masks):
- [x] **T** in {0.05, 0.10, 0.15, 0.20}: the one with the highest share of development boxes
  crossed by a measurable path (`path_through_box`, ≥ 5 px), among those meeting both:
  - ARCADE val recall (annotated vessel pixels covered) ≥ the model's;
  - ≥ 50% of the added pixels within 5 px of an annotated vessel (ARCADE annotates only the
    main segments, so a correct small branch counts against this).
- [x] **Adopt** the hybrid segmentation for the product only if it raises the development box
  coverage by ≥ 10 points over the model alone. Otherwise it stays an option.

  **Result: not adopted** (D62).
  - No threshold was eligible: only 15–24% of the added pixels lay near annotated vessels.
  - The model alone already crosses 95.4% of the development boxes; the hybrid crosses
    98–99%.

**Test** (once):
- [x] Box coverage on CADICA test patients and on Kemerovo. **Not run:** no threshold was
  eligible.
- [x] ARCADE test recall and the near-annotation share of the added pixels. **Not run.**
- [x] If adopted: P4i continues on the adopted segmentation (amendment 2, before its
  development decision); the grading is re-run. **Not adopted:** P4i evaluates its run on the
  model's segmentation, as amendment 2 says.
- [ ] Report: `vstent vessel-coverage` → `reports/vessel_coverage.md`.

## P4k: Do not miss lesions; target vessel; side-branch physics (added 2026-11-02; pre-registered before any code, training or evaluation)

Why: the user compared the guided app with commercial QFR software. Three problems:
- severe lesions read mild (D63);
- the automatic path is the main trunk, not the vessel with the lesion;
- long paths lose pressure "diffusely" because the model carries the whole inflow to the
  path's thin end (D64).

The user's priority is not missing lesions. The test sets of P4i (CADICA test patients,
ARCADE test, Kemerovo) have each been seen once; where they are used again, the report says
so. The CADICA test patients' lesion-free runs have never been used for lesion reading or FFR.

### A. Lesion reader, version 2 (CPU)
Changes from P4i. Everything else is as in P4i: the patch function, the network, the
optimiser, 40 epochs, 5 development folds, seed 0, and how the product uses the reader.
1. **Grades as intervals:**
   - the %DS loss is a Huber loss (δ = 0.1 on %DS ÷ 100) on the distance outside the target
     interval;
   - CADICA grades: their ranges (0–20, 20–50, 50–70, 70–90, 90–98);
   - ARCADE stenosis boxes: ≥ 50;
   - normal-vessel candidates: ≤ 40.
2. **Normal-vessel examples:**
   - ARCADE's distant candidates, now with the ≤ 40 interval;
   - CADICA **development** patients' lesion-free runs: every 4th selected frame, up to 2
     candidates per frame on the automatic path (seed 0), ≤ 40.
3. **Clinician-style paths for box samples:**
   - from the vessel tree's proximal end through the box to the vessel's end
     (`guided.box_path`), so the healthy proximal vessel sets the reference;
   - otherwise, the P4b path.
- [ ] **Development:** adopt only if all of P4i's conditions hold (unchanged).
- [ ] **Test:**
  - P4i's sets with P4i's targets (a second look);
  - **fresh:** candidates on the CADICA test patients' lesion-free runs read ≥ 50%
    (target ≤ 20%; the edges reported too).

### B. Automatic target vessel (CPU)
"Lesion-seeking" path:
- **Candidates:** from the tree's proximal end (as `main_trunk_endpoints`) to every skeleton
  end of the same vessel tree, at least 30 mm long.
- **Measurement:** each candidate measured with the product's QCA. Its score is the highest
  lesion %DS where the reference diameter is ≥ 2.0 mm (FAVOR II's vessel size).
- **Choice:** the candidate with the highest score. With no lesion ≥ 50%, the main trunk
  (as now).

Evaluation:
- **Set:** CADICA development patients' lesion runs. On each run, the expert-annotated frame
  nearest to the product's analysis frame.
- **Metric:** the share of runs whose automatic path crosses ≥ 1 expert box (≥ 5 px inside, as
  P4j).
- [ ] **Adopt** if the share is ≥ 10 points above the main trunk's. Test patients reported.

### C. Side-branch outflow in the physics (CPU)
μQFR (Murray's law) lets flow leave through side branches. Here, the flow at s is
Q(s) = φ(s)·Q_in with φ(s) = (min over s' ≤ s of r_ref(s') ÷ r_ref(0))³. The viscous and
expansion coefficients carry φ and φ², so ΔP = a·Q_in + b·Q_in² keeps its closed form. The
distal microvascular term becomes R_micro·φ(L)·Q_in, calibrated exactly as now. Kt, μ, ρ
and the velocity are unchanged. φ ≡ 1 for a vessel without taper, so every M1 check is
unchanged. Option: `PhysicsParams.outflow` in {`none`, `murray`}.

Evaluation (no invasive FFR; two literature anchors; product pipeline: automatic main-trunk
path, K = 5 consensus, pixel size assumed):
1. **Lesion-free vessels:** CADICA development patients' lesion-free runs. The share with
   FFR ≤ 0.80. Anchor: De Bruyne 2001 (Circulation 104:2401) measured FFR 0.89 ± 0.08 in
   non-stenotic arteries of patients with disease elsewhere, about 13% ≤ 0.80 (normal
   approximation).
2. **Graded lesions:** CADICA development patients' lesions, measured on the path from the
   proximal end through the box (`guided.box_path`; median over the lesion's boxes). The share
   with FFR ≤ 0.80 per grade. Anchor: FAME, visual 50–70% / 70–90% / 90–99% → 35% / 80% /
   96% with FFR ≤ 0.80.
- [x] **Adopt** `murray` only if both hold:
  - the lesion-free share moves closer to 13%;
  - the mean absolute difference from FAME's three shares does not grow by more than
    10 points.

  Test patients reported. If adopted, the planner and the app use it.

  **Result: not adopted** (D66; `reports/p4k_physics_dev.md`).
  - Lesion-free vessels read FFR ≤ 0.80: 52% now, 6% with `murray` (anchor 13%).
  - FFR ≤ 0.80 by grade (50–70 / 70–90 / 90–98%): 88 / 78 / 81% now; 0 / 14 / 19% with
    `murray` (FAME 35 / 80 / 96%). The FAME distance grows by 36 points (limit 10).

  **Test** (CADICA test patients, `reports/p4k_physics.md`): the same pattern.
  - Lesion-free vessels read ≤ 0.80: 54% now, 10% with `murray`.
  - By grade: 64 / 66 / 73% now; 5 / 9 / 10% with `murray`.
  - FAME distance: 22 vs 62 points.

  **Amendment 1** (2026-11-02, after this result, before any result of A or E): if A or E is
  adopted, this development check is run again with the adopted lesion reading, same rule.
  Reason: the lesions' own pressure loss depends on their measured depth, which the edges
  under-read (D63).

### C2. The path ends where a pressure wire would sit (CPU; added 2026-11-02, before running)
Why: the user's RCA case. After two stents the FFR stays above 0.90 until the last ~15 mm,
then falls to 0.72 inside a thin terminal branch. FAVOR II enrolled vessels with a reference
diameter ≥ 2 mm, and a wire measures in the distal vessel, not in a terminal branch.
- **Rule:** the FFR is read at the last point of the path where the QCA reference diameter is
  still ≥ 2.0 mm. The path beyond it is shown but not used for the FFR.
- **Physics unchanged:** `outflow: none`.
- **Evaluation:** C's check, same sets and anchors, as a third arm.
- [x] **Adopt** for the product (automatic path, Target, expert-box path) only if both hold:
  - the lesion-free share moves closer to 13% than now;
  - the FAME distance does not grow by more than 10 points.

  Test patients reported.

  **Result: adopted** (D68; `reports/p4k_wire_dev.md`).
  - Lesion-free vessels ≤ 0.80: 52% → 37% (median FFR 0.73 → 0.86).
  - By grade (50–70 / 70–90 / 90–98%): 88 / 78 / 81% → 53 / 60 / 69% (FAME 35 / 80 / 96%).
  - FAME distance 23 → 22 points.

  **Correction, before the product change** (2026-11-02):
  - The check solved the whole path, including the part beyond the wire point, then read the
    FFR at that point.
  - The text above says the path beyond is "not used for the FFR", and P4m cut the vessel.
  - The shared rule is now `qca.at_wire`: the vessel is cut, the reference kept from the whole
    path.
  - The development check is run again with it, under the same rule, and both runs are
    reported. The product follows only if the corrected run also meets the rule.

### C3. Learned lesion reading together with the physics, judged end to end (CPU; added 2026-11-02, before running)
Why: C showed the FFR depends on how deep the lesions are measured. In the model a 3 mm
vessel's lesion read at 40% loses about 2 mmHg; read at 65% (reader v1's median for 90–98%
lesions on test) it loses about 38 mmHg. Reader v1 failed P4i's patch-level false-alarm rule.
The end-to-end check judges false alarms where they matter: lesion-free vessels read
FFR ≤ 0.80.
- **Arms:** C's three (no outflow, Murray outflow, the wire position), each with the
  learned reader applied to the measured lumen (`analyse_vessel(..., lesion_reader=...)`).
- **Readers:** v1 (exported, P4i) now. v2 once trained, whatever its own P4k-A decision.
- **Data, anchors and rule:** C's (development patients).
- [ ] **Adopt** the reader and the arm together as the product's FFR pipeline if:
  - C's rule holds against the product (lesion-free share closer to 13%; FAME distance not
    more than 10 points worse);
  - **and** the lesion-free share is ≤ 25%.

  Of several qualifying combinations, the smallest mean distance from the two anchors wins:
  |lesion-free share − 13%| plus the FAME distance.
- An adopted combination supersedes D63's patch-level decision for the FFR pipeline: the
  end-to-end rule is the stricter clinical test. Test patients once, for the chosen one.
- P4m is then re-run with it as an arm.

### C4. Hyperemic velocity by artery from real Doppler data (CPU; added 2026-11-02, before the per-artery numbers were looked at)
Why: the user asked whether the vessel (LAD, LCx, RCA) makes a difference. The model uses one
velocity, 0.35 m/s, for every artery. DEFINE-FLOW has Doppler velocities measured in real
patients, with each lesion's artery (P4e). An input measured in patients is not a constant
tuned to the anchors.
- **Estimate:** per artery (left coronary = LAD + LCx, and RCA), the median stenosis-free
  hyperemic mean velocity (k = 0.8, P4e's definition).
  - For LAD vs LCx separately, only if G is used in the product.
- **Use:** only if an artery's median differs from the pooled median by ≥ 10%. Otherwise
  0.35 m/s stays.
- **Arm:** per-artery velocity, with the artery from CADICA's left / right label (in the
  product: the user's choice, or G). Judged on C's data with C3's rule: alone, and with the
  best C3 combination.
- Test patients once, for an adopted arm.

**Result** (`reports/artery_velocity.md`; DEFINE-FLOW, 447 lesions):
- pooled median 0.333 m/s; left coronary (LAD + LCx) 0.335 m/s (+0.6%); RCA 0.303 m/s (−9.1%).
- The 10% rule did not fire: one velocity (0.35 m/s) stays, and the arm is not run.
- RCA is close to the limit. Recorded, not acted on.

### D. Display like a clinical QFR tool (no measurement change)
- The lumen contours along the whole path.
- Lesions as shaded segments labelled A, B…
- A lesion table: MLD, reference diameter, length, %DS, FFR drop across the lesion.
- Below the FFR curve, the FFR drop per mm.

### E. Whole-image lesion detector (GPU; waits for the user's go-ahead)
- **Model:** U-Net (ResNet34 encoder, ImageNet start, as the segmentation model). Input: the
  frame through the shared pre-processing. Target: the expert boxes, filled. Loss: BCE + Dice.
  AMP; resumable.
- **Training data:**
  - CADICA development patients' annotated frames;
  - their lesion-free frames (every 4th, empty targets);
  - ARCADE stenosis train;
  - Kemerovo: patients split 50 / 50 (seed 0), the training half.
- **Validation** (early stopping and the operating point): ARCADE val, 5 CADICA development
  patients held out (seed 0), and 10 patients of Kemerovo's training half.
- **Detections:** connected components of the output ≥ t. An expert box is found when a
  detection overlaps it.
- **Implementation details, fixed before training** (2026-11-02):
  - images held at 512 px;
  - every CADICA box of every grade is a target;
  - Kemerovo frames taken every 4th per series (consecutive, nearly identical frames), for
    training, validation and test alike;
  - Kemerovo has 64 patients with boxes: 32 / 32.
  - Counts: train 3,566 images (4,838 boxes); validation 1,181 (1,479 boxes, 88 lesion-free
    frames); test 3,483 (4,424 boxes, 308 lesion-free frames).
  - Training: `configs/lesion_det.yaml` (as `configs/seg.yaml`, 60 epochs, patience 15,
    validation Dice for early stopping).
- **Operating point:** the highest t with validation sensitivity ≥ 95% of boxes. False alarms:
  per lesion-free frame, and per annotated frame (an upper bound: not every lesion is boxed).
- [ ] **Shown in the app** (flags to review, each with "measure this", and B's path through
  the strongest flag) if, at t, false alarms are ≤ 2 per lesion-free validation frame.
  Otherwise a research option.
- [ ] **Test:**
  - sensitivity (target ≥ 90%) on CADICA test patients, ARCADE test and Kemerovo's test
    half (all second looks, except Kemerovo's half, which was a P4i test once);
  - false alarms per lesion-free frame of the CADICA test patients (fresh).

### Amendment 2 (2026-11-02, before any E or G model was evaluated): architectures
The user asked for other architectures. E and G each compare five encoders in the same U-Net,
with the same data and training settings (ImageNet start):
- ResNet34 (as registered);
- MiT-B2 (SegFormer's transformer encoder);
- DenseNet121;
- MobileNetV3-Large;
- EfficientNet-B4.

The choice uses validation data only:
- **E:** each encoder at its own operating point (95% validation sensitivity). The fewest
  detections per lesion-free validation frame wins. Within 0.05 of each other, the higher
  sensitivity wins. The mean of the five probability maps is a sixth candidate. If no
  candidate reaches 95%, the highest sensitivity at t = 0.5 wins.
- **G:** the highest validation mean vessel Dice.

The chosen model alone goes to E's or G's rule and test. Every candidate's validation numbers
are reported.

### F. End-diastolic frames from the images (needs data with ECG) [needs human]
- **Method to follow:** Ciusdel et al. 2020 (Comput Med Imaging Graph). Spatial and temporal
  convolutions, trained against the ECG: precision 97.4% and recall 96.9% for ED frames
  after training on 56,655 runs.
- **Missing data:** no public angiogram runs with an ECG are known to us; CADICA, ARCADE and
  Kemerovo have none. Synthetic beating-heart runs (P7) carry an ECG and can pre-train.
- **Blocked on the human:** a real evaluation needs de-identified DICOM runs with their ECG,
  from a hospital.

### G. Vessel identity: LAD, LCx, RCA (GPU; waits for the user's go-ahead)
- **Data:** ARCADE's SYNTAX labels (1,500 images, 6,816 segments; CC0).
- **Classes:** background, plus SYNTAX segments grouped into four vessels:
  - RCA: 1–4, 16, 16a–c;
  - left main: 5;
  - LAD: 6–10, 9a, 10a;
  - LCx: 11–15, 12a, 12b, 14a, 14b.
- **Model:** the segmentation model's architecture with 5 output classes, trained on ARCADE
  train (early stopping on val), resumable.
- [ ] **Reported:**
  - ARCADE test: Dice per vessel, and pixel accuracy on vessel pixels (published reference:
    95–98% on a private cohort, arXiv 2601.17429);
  - CADICA (external): agreement of the run's side with CADICA's left / right labels
    (`CADICAprojections.json`). A run is "right" when RCA pixels outnumber left-tree pixels on
    its analysis frame. Reference: CathAI 93–97%.
- [ ] **Used in the product** (each path named by its majority vessel; "LAD / LCx / RCA"
  buttons that choose the target vessel; B's candidates limited to the chosen vessel) if both:
  - ARCADE val vessel-pixel accuracy ≥ 90%;
  - CADICA development agreement ≥ 95%.

  Otherwise a research option.

### H. Frame rates (7.5, 10, 15, 30 fps)
- [ ] The synthetic cine evaluation (`vstent cine-synthetic`) also at 10 fps, CADICA's rate,
  on new seeds. ED frames and flow velocity reported per frame rate.
- [ ] Two steps count frames, not seconds: the 3-frame median of the contrast curve and the
  3-frame minimum heart period. If any frame rate's image-based ED result is ≥ 10 points
  below the others, both become times (0.2 s, 0.3 s) and are re-evaluated on further new
  seeds.
- F (a learned ED model) is the main route to better ED frames: image-based ED is within
  ±1 frame for 75–79% of beats at every rate now.

### I. Contrast along the target vessel (display; no FFR change)
- **Per frame, along the chosen path:** the share of the path filled with contrast, from P7's
  transit along the centreline. The frame slider shows "vessel filling / full / washing out"
  for the target vessel, not the whole image.
- **TIMI frame count** (Gibson 1996): the frames from contrast entering the path to reaching
  its end, corrected to 30 fps (× 30 / fps).
- [ ] Reported on synthetic runs, where the truth is known: TIMI frame count error per frame
  rate.

### J. AngioCAD: an external, segment-level test (CPU; after G)
- **Data:** AngioCAD (Zenodo 10.5281/zenodo.15826856; CC BY 4.0). 413 patients' cine frames,
  each patient's 15 coronary segments graded by experts. No FFR, no ECG. Download
  MD5-verified; extracted without running anything.
- [x] **Protocol** (written 2026-11-02 after inspecting the files, **before** any model was
  run on AngioCAD).
  - **Layout:**
    - `AngioCAD_Dataset/<ID>/<series>/frame_NNNN.png`: 413 patients, 2,711 series, 121,566
      frames; frame rate and pixel size unknown.
    - `AngioCAD_Labels.xlsx`: per patient, the right- and left-coronary series (ranges such as
      "1-2" and "3-9"), and 15 segments graded NL, 1-25, 26-50, 51-75, 76-90, 91-99, 100.
    - Three frames at the archive's volume boundaries could not be decoded (`docs/DATA.md`).
  - **Units:**
    - a patient's side: right = Prox / Mid / Dist RCA, PDA, PLB; left = LM, Prox / Mid / Dist
      LAD, diagonals, Prox / Mid / Dist LCX, OM;
    - a side is **significant** if any of its segments is ≥ 51% (51-75 or worse), **normal**
      if all are NL, and **mild** otherwise (reported separately);
    - total occlusions (100) are reported separately as well.
  - **The product on every series of a side:** its analysis frame, the automatic path, the
    product's QCA and physics (with every adopted change at run time). A side is
    **flagged** if any of its series has a marked lesion ≥ 50% or FFR ≤ 0.80.
  - **Reported:**
    1. sensitivity: significant sides flagged, by worst grade;
    2. false alarms: normal sides flagged;
    3. normal sides' series with FFR ≤ 0.80 (De Bruyne anchor ≈ 13%), an external replicate
       of C;
    4. with G: each series' predicted side against the labels (an external check of G);
    5. with E chosen: E's detections on the same analysis frames, as a second flag.

  There are no targets; this is the external description of where the product stands. Run
  once.

- [ ] Report: `reports/p4k.md` (A–C, E, G–J), each part as it finishes.

## P4m: Virtual stenting checked on real angiograms without FFR (added 2026-11-02; pre-registered before any code)

Why: the user's goal is the best virtual stenting with the data we have. Without FFR, two real
anchors remain: the experts' lesion boxes (where a stent belongs) and the published post-PCI
physiology.

**Set:** CADICA patients' lesions graded 70–98%, one box per lesion (the box on the frame
nearest the middle of the lesion's boxes).
- **Measurement:** the clinician-style path through the box (`guided.box_path`), the product's
  QCA and physics.
- **Planner:** the guided app's (solver backend, `configs/planner.yaml`).
- **Lesions included:** those whose pre-PCI FFR is ≤ 0.80 (a plan is proposed).

**Measures**
1. **The stent is on the experts' lesion:** the share of lesions whose deepest point inside the
   box lies under a planned stent. Target ≥ 90%.
2. **Post-PCI FFR:** the mean, and the share ≤ 0.80, of the best plan's predicted FFR.
   - Anchor: FFR-SEARCH, 0.91 ± 0.07 (Circulation 2019), with about 11% of patients having a
     lesion ≤ 0.80.
   - Targets: mean 0.88–0.94; share ≤ 0.80 at most 15%.
3. **Plans reaching 0.90:** reported.

**Arms**
- The product now.
- Every adopted change, each as an arm: P4k-C2 (the wire position) if adopted; lesion reader
  v2 (P4k-A) if adopted.

**Stages**
- Development patients first.
- A change becomes the product only by its own rule (C2, A). P4m judges the stenting result,
  it adopts nothing.
- Test patients reported.
- [ ] `vstent stent-check --stage dev|test` → `reports/stent_check[_dev].md`.
  - **Development, the product now** (`reports/stent_check_dev.md`):
    - stent on the experts' lesion: 77% (target 90%);
    - post-PCI FFR 0.62 ± 0.29 (FFR-SEARCH 0.91 ± 0.07), still ≤ 0.80 in 65%;
    - 13% of plans reach 0.90.
  - **With the wire position:** 79%; 0.79 ± 0.13; 38%.
  - Neither meets the targets: the residual loss after stenting is the diffuse artefact and
    the under-read lesions (C, C3).

## P4l: End-diastolic frames without an ECG (added 2026-11-02; pre-registered before any code)

Why: image-based ED frames are within ±1 frame for 75–79% of beats (synthetic, every frame
rate). Real runs rarely carry an ECG, and none of our real datasets does. The user asked for a
ground truth and a method that need no ECG (`docs/INVENTION_NOTES.md` §8).

**Ground truth without an ECG**
- [ ] [needs human] **Expert-labelled ED frames,** as in Meng et al. 2022:
  - an ED-labelling page in the app (scroll the frames, mark one ED frame per beat);
  - 60 runs (30 CADICA test patients, 30 AngioCAD), labelled by two readers;
  - their agreement within ±1 frame reported;
  - the consensus is the real test set.
- [ ] **Beat-to-beat reproducibility (no labels):** for each method's ED frames, the mean
  centreline distance between successive beats after an affine registration, relative to the
  same frames shifted by ±1. Lower is better. Reported on CADICA development runs.

**Methods** (all on the same runs)
- M0: the vessel-tree spread maxima (now).
- M1: two clocks (mechanical spread, length and motion, plus contrast-front kinetics while
  filling), fused per beat.
- M2: a spatio-temporal network (MobileNetV3 per frame, temporal convolutions) trained on
  synthetic runs with a known ECG at 7.5, 10, 15 and 30 fps.
- M3: M2 fine-tuned on real runs (CADICA development patients, AngioCAD) with the
  reproducibility and two-clock signals.

**M2 in detail** (fixed 2026-11-02, before any data or training):
- **Data:**
  - synthetic beating-heart cases (`synthetic_cine`) with 7.5, 10, 15 and 30 fps drawn
    equally; both runs of every case;
  - seeds: train 20000–20249, validation 20250–20299, test 20300–20359 (new);
  - frames through the shared pre-processing at 128 px, saved with the true ED frames (the
    frame nearest each R wave) and the frame rate.
- **Model:**
  - MobileNetV3-Small per frame (ImageNet start, one input channel), 64 features;
  - with the frame time, three dilated temporal convolutions (kernel 5; dilation 1, 2, 4),
    then a per-frame logit;
  - BCE (positive weight 5); AdamW, lr 1e-3, 30 epochs, batch 4 runs, seed 0; resumable;
  - the epoch with the best validation ±1-frame accuracy is kept.
- **ED frames:** per-frame probability peaks ≥ 0.5, at least 0.6 × the median gap between
  peaks apart.
- **Scored as P7:** true ED beats in the well-opacified frames found within ±1 frame, per
  frame rate.

**Choice and targets**
- **Choice:** the method with the best synthetic development ±1-frame accuracy among those
  whose real development reproducibility is no worse than M0's. Both are measured on
  development data only.
- **Targets (test):**
  - synthetic ±1 frame ≥ 95% at every frame rate;
  - expert-labelled real runs ±1 frame ≥ 90% (Meng et al.: 93%).

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
