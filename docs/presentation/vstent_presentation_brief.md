# vstent: Virtual FFR and Stent Planning from Coronary Angiograms

### Presentation brief (source document for building slides)

> **Research use only — not for clinical decisions.** vstent is a research prototype. It is not
> a medical device, it has not been validated against invasive FFR, and it must never be used
> to decide treatment for a real patient. Every slide built from this document should carry
> this notice.

> **Confidential: intellectual property.** Section 9 names methods that may be patentable. Do
> not present them outside the team, publish them or send them to partners until a patent
> attorney has advised on filing. For external audiences, use the benefit-level wording
> given there and leave out the technical recipe.

---

## 0. How to use this document

- **For the presenter:** every numbered section is one or two slides. Each section starts with
  a **Slide** box (title, key message, suggested figure) followed by the detail and speaker
  notes.
- **For Claude (building the deck):**
  - keep every number exactly as written; each comes from a generated report in the project
    repository;
  - keep the limitations and the research-use notice;
  - do not add claims of clinical accuracy, approvals, market sizes or partners that are not
    in this document;
  - figures are in `figures/`, named in each section and listed in Appendix B.
- **Suggested deck:** 16–20 slides.
  - Main deck: sections 1–8, 10 and 12.
  - Internal only: section 9 (IP), after the attorney has seen it.
  - Backup slides: section 11 and the appendices.

---

## 1. The one-line pitch

**Slide**
- Title: *Virtual FFR and stent planning from the angiogram you already have*
- Key message: from a standard coronary cine angiogram, vstent finds the vessels and
  narrowings, estimates FFR without a pressure wire, and plans the shortest stent that
  restores flow, in about a minute on an ordinary laptop.
- Figure: `figures/pipeline.png`

**Speaker notes:** FFR (fractional flow reserve) tells a cardiologist whether a narrowing
actually limits blood flow. Today it needs a pressure wire pushed into the artery and a drug
(adenosine) to dilate the vessels. vstent computes a virtual FFR from the X-ray movie (cine
angiogram) that is recorded in every catheterisation anyway, and lets the cardiologist try
virtual stents before placing a real one.

---

## 2. The clinical problem

**Slide**
- Title: *Seeing a narrowing is not knowing whether it matters*
- Key message: visual assessment of angiograms misjudges which lesions limit flow; the
  pressure wire (FFR) answers this but is used in a minority of cases.
- Figure: a simple bar chart can be drawn from the FAME numbers below (no image provided).

**Detail**
- Coronary artery disease narrows the heart's arteries. The decision to stent depends on
  whether a narrowing limits flow (ischaemia), not only on how tight it looks.
- **FFR** = pressure beyond the narrowing ÷ aortic pressure during maximal flow. Guidelines
  treat **FFR ≤ 0.80** as significant (ESC 2019 guidelines for chronic coronary syndromes;
  the FAME trials).
- **Visual grading misleads.** In the FAME study (Tonino et al., JACC 2010), the share of
  lesions with wire FFR ≤ 0.80 was:
  - about **35%** of lesions graded 50–70% by eye;
  - about **80%** of lesions graded 71–90%;
  - about **96%** of lesions graded 91–99%.

  So many "moderate" lesions do not need a stent, and some do.
- **Why the wire is underused:** extra time, cost of the wire, adenosine (side effects,
  contraindications) and the risk of instrumenting the artery.
- **Angiography-derived FFR** removes the wire: the vessel's 3D geometry and a flow model
  give the pressure drop. Commercial tools exist (section 8). vstent pursues the same goal
  and adds planning, uncertainty and robustness features.

---

## 3. What vstent does: the user's workflow

**Slide**
- Title: *Six steps, one screen*
- Key message: load a cine run → pick the heartbeat frame → check the vessel and lesions →
  read the virtual FFR → try a stent → download the report.
- Figures: `figures/synthetic_ed_frames.png`, `figures/synthetic_vessel_and_lesions.png`,
  `figures/synthetic_ffr_curve.png`, `figures/synthetic_stent.png`. Live screenshots of the
  app are better still (Appendix C).

**The guided app** (`vstent app`, runs in a web browser on the same computer):

1. **Study.**
   - Upload a cine angiogram (multi-frame DICOM, as exported by the X-ray system), or pick a
     case from the built-in library: 10 synthetic beating hearts and 382 real public
     angiogram runs (CADICA, CC BY 4.0).
   - *Random* and *New study* start over.
2. **Frame.**
   - The software finds the end-diastolic frames, when the heart is still and the arteries
     are longest: from the ECG markers in the file, the recorded ECG, or, without them, the
     images.
   - The user scrolls them like a film strip. ★ marks the recommended frame and ◆ frames
     where experts marked a lesion (on the public cases).
3. **Vessel and lesions.**
   - The vessels are segmented automatically and drawn in blue, with an automatic path
     along the main vessel in yellow.
   - Narrowings of 50% or more are circled and numbered, and each has a magnified close-up
     with the measured vessel edges, so the user can check them by eye.
   - The path can be redrawn with two clicks.
4. **Virtual FFR.**
   - One large number, coloured by the FFR guide, with its likely range.
   - The vessel is measured on up to five heartbeats and the median is reported; the app
     says how many heartbeats agreed.
   - A curve shows where along the vessel the pressure is lost.
5. **Stent.**
   - *Suggest a stent* returns the shortest stent (generic diameter × length catalogue)
     that brings the virtual FFR to 0.90 or more.
   - It is drawn on the vessel with the FFR before and after. The user can change diameter,
     length and position and sees the result at once.
6. **Report.** JSON and PNG downloads with every input, assumption, software version and
   model checksum (an audit trail).

**The FFR guide** (always visible in the app; from pressure-wire studies):

| Virtual FFR | Reading |
|---|---|
| above 0.80 | not significant: no stent needed |
| 0.76–0.80 | grey zone: significant, stent usually advised |
| 0.75 or below | significant: stent advised |
| 0.90 or above after a stent | good stent result |

---

## 4. How it works: the technology

**Slide (overview)**
- Title: *From pixels to pressure*
- Key message: AI finds the vessels; measurement and physics, not a black box, turn them into
  FFR; every step is checked.
- Figure: `figures/pipeline.png`

### 4.1 Reading the cine run
- The whole run is read, not one picture. The software tracks how much vessel is visible in
  each frame (the contrast curve): when the contrast arrives, when the arteries are full and
  when it washes out.
- **End-diastolic frames:**
  - from the X-ray system's ECG markers or the ECG waveform stored in the DICOM file;
  - otherwise from the images (the vessel tree is largest when the heart is relaxed).
- On synthetic beating hearts, ECG-based frames were within ±1 frame of the true
  end-diastole for 100% of beats.
- **Heart motion is followed** from frame to frame (image registration), so points placed
  on one frame can be found on the others.

### 4.2 Finding the vessels (AI segmentation)
- A U-Net neural network (ResNet-34 encoder) marks every vessel pixel. It was trained on the
  public ARCADE dataset (1,000 annotated angiograms) plus 1,000 synthetic angiograms rendered
  by our own X-ray simulator.
- Accuracy (Dice overlap with expert masks):
  - 0.79 on ARCADE's held-out test images;
  - 0.91 on rendered test images.
- It runs in ONNX Runtime on an ordinary CPU; no GPU is needed.
- Figures: `figures/segmentation_examples.png`, `figures/rendered_angiograms.png`.

### 4.3 Measuring the vessel (QCA)
- **Path:** the centreline between two points, automatic along the main vessel, or two
  clicks by the user.
- **Diameters:** where the image brightness changes fastest across the vessel
  (minimum-cost edge tracking, the classical QCA criterion). The absolute size is set by
  fitting a model of the X-ray imaging chain (blur, edge enhancement) to healthy segments.
- **Densitometry (new, section 6):** at tight lesions the width drops below the image
  resolution. The amount of contrast the vessel carries still shows the true lumen area.
- **Pixel size:** from the DICOM geometry (magnification-corrected); otherwise 0.30 mm is
  assumed and the user is warned.
- On synthetic test tubes, the worst-case median diameter error was 0.10 px and the
  worst-case stenosis error 0.7 points.

### 4.4 The physics: from geometry to FFR
- **Pressure loss along the vessel:**
  - **viscous friction**, which grows with length and with 1/radius⁴ (Poiseuille);
  - **expansion loss** where blood leaves a narrowing (Kt = 1.52, a published constant).
- **Flow:** set by aortic pressure, venous pressure and the resistance of the heart muscle's
  small vessels. That resistance is calibrated so a healthy vessel carries a typical
  maximal-flow velocity (0.35 m/s).
- Solving these together gives the FFR at every point along the vessel (a virtual pullback).
- **Side branches** can be included as a vessel tree (flow split by Murray's law).
- **Checks:**
  - a healthy vessel gives FFR 0.98;
  - a 50% and a 70% lesion give 0.90 and 0.62;
  - mass is conserved at every junction to 10⁻¹⁶;
  - about 50,000 vessels per second on one laptop.
- Figure: `figures/physics_ffr_vs_ds.png` (FFR against stenosis severity),
  `figures/vessel_tree.png`.

### 4.5 A learned surrogate (preparation for 3D simulation)
- A small neural network (an ensemble of 5) reproduces the physics solver's FFR curve: mean
  error 0.0013 FFR, 1.5 ms per vessel in ONNX.
- **Honest note:** the physics solver is already fast, so the surrogate gives no speed-up
  today. It exists to prove the training → ONNX → app pipeline. Its training labels can
  later come from full 3D blood-flow simulations, where a learned surrogate brings large
  speed-ups.
- Figure: `figures/surrogate_vs_solver.png`

### 4.6 Uncertainty and grey-zone warnings
- Every FFR comes with a **likely range**: the model's own error (calibrated intervals with
  89.7% coverage at the 90% target), plus the effect of uncertain inputs (pixel size, flow,
  pressures, measurement error), sampled by Monte Carlo.
- A result whose range crosses 0.80 is flagged as **uncertain**.
- Figure: `figures/uncertainty_coverage.png`

### 4.7 Several heartbeats instead of one frame
- One frame can mislead: noise, overlap and motion change the measurement.
- vstent measures the same vessel on **up to five end-diastolic frames**, with the points
  following the heart, and reports the **median**.
- If the heartbeats fall on both sides of 0.80, the result is flagged.
- Figure: `figures/synthetic_heartbeats.png`, `figures/repeatability.png`

### 4.8 Optional extras
- **Two views → 3D:** two projections at least 30° apart reconstruct the vessel in 3D. Length
  is then free of foreshortening: median length error 0.7%, centreline error 0.20 mm on
  synthetic cases. Figure: `figures/two_view_3d.png`.
- **Measured flow velocity:** the speed of the contrast front along the vessel, corrected
  for the heart cycle. On synthetic two-view runs the median error was 5.8%. It is off by
  default (not yet validated on patients).

### 4.9 The virtual stent and the planner
- A stent opens the lumen to its diameter over its length; the FFR is recomputed with the
  heart-muscle resistance kept fixed.
- **The planner:**
  - searches a generic catalogue of diameters and lengths, including two-stent plans;
  - returns the shortest plan that reaches the target FFR (0.90);
  - takes 0.84 s for 5,000 candidates.
- On 100 synthetic vessels, the target was reached in 84.
- Figures: `figures/synthetic_stent.png`, `figures/synthetic_ffr_curve.png`.

### 4.10 Built for validation and regulation from day one
- **Validation tool** (`vstent validate`):
  - runs on a hospital's de-identified case list;
  - **locks** the predictions with a cryptographic checksum before the wire values are
    opened;
  - computes the statistics a diagnostic-accuracy study reports (STARD 2015): AUC,
    sensitivity, specificity, Bland–Altman and patient-level bootstrap.
- A **study protocol and data specification** are drafted for hospitals.
- **Regulatory groundwork** (drafts for review by a qualified person), in the style of IEC
  62304 and ISO 14971:
  - 48 software requirements traced to code and tests;
  - 20 hazards with controls;
  - a software bill of materials.
- Every reported number is regenerated from code at a recorded software version: **no
  hand-typed results**.

---

## 5. Results so far: what is proven, and what is not

**Slide**
- Title: *Engineering proven; clinical accuracy not yet measured*
- Key message: every engineering target is met. Real-angiogram checks show where the product
  stands, including a known weakness being fixed. Agreement with the pressure wire still needs
  hospital data.
- Figures: `figures/synthetic_validation.png`, `figures/repeatability.png`,
  `figures/stenosis_grading.png`

### 5.1 Engineering targets (all met: 23 of 23 in the MVP report)
| Area | Result |
|---|---|
| Physics sanity | healthy 0.98; 50% lesion 0.90; 70% lesion 0.62; monotonic |
| Segmentation | Dice 0.79 (ARCADE test), 0.91 (rendered) |
| Surrogate | error 0.0013 FFR; ONNX 1.5 ms per vessel |
| Planner | 5,000 candidates in 0.84 s |
| App | every stent-control update < 300 ms (worst 233 ms) |
| Uncertainty | 89.7% coverage of 90% intervals |

### 5.2 End-to-end on synthetic patients (our own physics as the "truth")
A simulated study: 102 vessels in 60 synthetic patients, rendered as X-ray images, analysed
like a real study (locked predictions, the full statistics):

| Analysis | AUC (FFR ≤ 0.80) | Accuracy |
|---|---:|---:|
| Vessel outlines given (as an analyst would correct them) | 0.974 | 90.2% |
| Fully automatic segmentation | 0.938 | 86.5% (94% analysable) |

**Caution:** the "truth" here comes from the same physics, so these numbers test the image
pipeline, not clinical accuracy.

### 5.3 Real angiograms (public data, no pressure wire)
| Check | Result |
|---|---|
| Repeatability: one frame | FFR coefficient of variation 18.7% (not good enough) |
| Repeatability: median of 5 frames | **5.1%**, below QFR's published 9.4% between observers |
| Lesion severity against expert visual grades (CADICA) | weak agreement so far (weighted kappa 0.13 on test patients); severe lesions read too mild: being fixed (section 6) |
| Physiology assumptions against invasive measurements (DEFINE-FLOW, 455 lesions) | the assumed maximal-flow velocity (0.35 m/s) is inside the measured interquartile range for the more realistic Doppler factor (median 0.33 m/s, k = 0.8) but above it for k = 0.5 (median 0.21 m/s): flow is a real source of error |
| Physics against an FDA laboratory benchmark (nozzle) | losses at a gradual exit match (1.06×); at an abrupt exit the model gives 0.61× the measured loss |

Figures: `figures/repeatability.png`, `figures/stenosis_grading.png`,
`figures/fda_nozzle.png`.

---

## 6. Closing the accuracy gap: measuring tight lesions

**Slide**
- Title: *When a narrowing is thinner than the X-ray blur*
- Key message: in small vessels a very tight narrowing does not look thinner. It looks
  fainter. vstent now measures that fading (densitometry) as well as the edges.
- Figures: `figures/cadica_vessel_and_lesions.png` (the experts' box in cyan),
  `figures/cadica_ffr_curve.png`.

**The problem we found ourselves.**
- On 382 real angiogram runs (CADICA), lesions the experts graded 70–90% or 90–98% were
  measured at a median of only about 42–43% narrowing.
- So the tool called most of them not significant, where pressure-wire studies find most such
  lesions significant:

| Expert visual grade | Called FFR ≤ 0.80 by vstent (median of 5 frames) | FAME: wire FFR ≤ 0.80 |
|---|---:|---:|
| 50–70% | 10% | ~35% |
| 70–90% | 16% | ~80% |
| 90–98% | 23% | ~96% |

**Why.** A 90% narrowing in a 2 mm vessel is narrower than the image blur. Its two edges stay
a few pixels apart, so edge-based measurement reads it as moderate. What does change is how
much contrast passes through it: the vessel fades.

**The fix: densitometry.**
- The X-ray darkness summed across the vessel is proportional to the lumen's area, and blur
  does not change that sum.
- vstent now takes the narrower of the edge measurement and the area measurement at every
  point.
- Overlapping vessels only make the area reading larger, so they cannot create a false
  narrowing.

**How it is being tested (the method and pass marks were fixed in writing before any
real-image measurement):**
- developed on rendered lesions with a known true lumen, and on half of the real patients;
- adopted only if it beats the current method on agreement with the experts without
  inventing narrowings in healthy vessels;
- then tested once on the other half.

**Result:** *[to be filled in from the P4h report when it is generated: section 6 table of
median measured narrowing by expert grade, before and after; agreement (weighted kappa);
share of 70–98% lesions called significant.]*

*[Rendered lesions with a known lumen: the table of mean %DS error by lesion group for edges
and densitometry, from the regenerated `reports/lumen_synthetic.md`.]*

---

## 7. Benefits

**Slide**
- Title: *Why this matters to cath labs and patients*

| For | Benefit |
|---|---|
| Patients | no pressure wire, no adenosine; fewer unnecessary stents and fewer missed significant lesions (once validated) |
| Cardiologists | a physiology answer and a stent plan in one screen, during or after the procedure; virtual stents before real ones; a likely range and a stability check instead of a bare number |
| Hospitals | uses the standard cine DICOM already recorded; runs on an ordinary computer (no GPU, no cloud required, data stay on site); full audit trail |
| Health systems (India and similar settings) | wire-free physiology where pressure wires are too costly to use routinely; can review archived angiograms retrospectively |
| Research | open, testable pipeline; locked-prediction validation tool ready for multicentre studies |

---

## 8. Competitive landscape

**Slide**
- Title: *Angiography-derived FFR is proven as a category; we aim to add planning, robustness
  and access*

| Product / study | Approach | Published accuracy against wire FFR |
|---|---|---|
| QFR (Medis), FAVOR II China | 3D QCA from two views + contrast flow | AUC 0.96; accuracy 92.7% |
| QFR (Medis), FAVOR II Europe-Japan | same | AUC 0.92; accuracy 86.8% |
| FFRangio (CathWorks), pooled 5 studies | 3D from ≥ 2 views, network model | sensitivity 91%, specificity 94%, accuracy 93% |
| MPFFR (Medipixel), Korean multicentre registry, JAHA 2026, 599 vessels | AI, automated, two views | AUC 0.949; accuracy 88.1%; 12.5 s per vessel |
| AngioAI-QFR (Almazov Centre), arXiv 2025, 100 vessels | AI pipeline + virtual stenting | AUC 0.93; sensitivity 0.88; specificity 0.86; 93% fully automatic |
| **vstent** | AI + QCA + physics; one or two views; multi-heartbeat median; uncertainty; planner | **not yet measured against wire FFR** |

**Where vstent aims to differ (to be proven):**
- single-view operation, with two views optional;
- a stability check across heartbeats;
- calibrated uncertainty and grey-zone flags;
- a built-in stent planner (shortest stent to target);
- runs on a laptop CPU;
- an affordable, deployable product for emerging markets;
- a validation tool designed for pre-registered, locked studies.

**Speaker note:** the competitors' numbers come from their own published validations. vstent
must be validated the same way before any comparison of accuracy is claimed.

---

## 9. Novelty and intellectual property (internal only until an attorney advises)

**Slide (external version, benefit level only)**
- Title: *Our own methods*
- Key message: we have developed our own methods for (1) a stable, multi-heartbeat
  measurement, (2) choosing which frames to trust, (3) flow from the moving contrast and
  (4) planning stents with uncertainty. Patent applications are being considered.

**Internal list (do not present externally before filing advice):**
- multi-frame consensus with motion-tracked points and an instability flag;
- selecting frames on which the measured path is fully filled with contrast;
- heart-cycle-corrected contrast flow velocity with calibrated intervals;
- densitometric narrowing of edge-based lumens at sub-resolution lesions;
- planning with uncertainty-aware ranking.

**Caution:**
- The basic pipeline (segment, model FFR, virtual stent) is published by others and is not
  novel on its own.
- Indian patent law (section 3(k)) excludes computer programs per se. Claims would need to
  tie to a technical effect on medical imaging data.
- Ask a patent attorney before any public disclosure.

---

## 10. What is lacking (honest limitations)

**Slide**
- Title: *What we have not proven yet*

1. **No wire-FFR validation.** No angiogram with a matching pressure-wire measurement has
   been analysed yet, so the clinical accuracy is unknown. This is the next and most
   important step, and it needs hospital data.
2. **Lesion severity on real images.** Tight lesions were read too mild (section 6). The
   densitometry fix is being tested.
3. **Physics simplifications.**
   - The model is steady and one-dimensional.
   - It underestimates the loss at abrupt lesion exits (0.61× on the FDA benchmark).
   - It assumes a typical maximal flow unless flow is measured.
4. **Single view by default:** foreshortening and overlap; the two-view 3D mode needs
   suitable runs.
5. **Training data:** public datasets only (ARCADE, CADICA, rendered images); no data from
   the target hospitals yet.
6. **Automation:** the automatic path can choose the wrong vessel. Without an ECG in the
   file, following the vessel across heartbeats often fails, and the result then rests on
   one heartbeat (the app says so).
7. **Regulatory:** drafts only; no clearance (CDSCO, CE, FDA) has been sought.

---

## 11. Validation plan and what we need

**Slide**
- Title: *The next milestone: a blinded comparison with the pressure wire*

**Study design (protocol drafted):**
- retrospective, multicentre, STARD 2015;
- predictions **locked** before the wire values are opened;
- primary endpoint: AUC for wire FFR ≤ 0.80; secondary: sensitivity, specificity,
  Bland–Altman agreement, per-territory results, time per case.

**Data needed per vessel:**
- the whole cine run as exported (multi-frame DICOM, with the ECG if recorded); a second
  view is helpful;
- the wire FFR value and where the pressure sensor sat (marked on the frame);
- aortic pressure if available;
- data de-identified at the hospital; ethics approval.

**Size:**
- 30–50 vessels for a first feasibility look;
- a few hundred for a definitive accuracy study (the sample size depends on the expected
  AUC and the prevalence).

**Sources being pursued:**
- collaborating cardiologists' past FFR cases;
- groups that published angiogram + FFR cohorts and state that data are available on
  request (for example the Korean multicentre QFR registry, JAHA 2026);
- the investigators of the open physiology datasets DEFINE-FLOW and CONTRAST.

---

## 12. Roadmap

**Slide**
- Title: *From prototype to validated product*

| Horizon | Milestones |
|---|---|
| Now | lesion-severity fix tested on real angiograms; guided app; presentation |
| Next 3 months | first wire-FFR dataset (30–50 vessels); locked validation; contour correction in the app; densitometry refinements |
| 3–6 months | multicentre retrospective study; 3D-simulation labels for the surrogate; two-view workflow in hospitals |
| 6–12 months | quality-management system; regulatory submission planning (CDSCO India; CE mark; FDA 510(k) with predicate devices); pilot deployments |

**The ask (adapt to the audience):**
- clinical partners with FFR cases;
- data-sharing agreements;
- funding for the validation study and regulatory work.

---

## Appendix A. Glossary

| Term | Meaning |
|---|---|
| FFR | fractional flow reserve: distal pressure ÷ aortic pressure at maximal flow; ≤ 0.80 = significant |
| Virtual FFR | FFR computed from images and physics, without a wire |
| QFR | quantitative flow ratio, a commercial angiography-derived FFR (Medis); not our product |
| Cine angiogram | the X-ray movie of the arteries filling with contrast, recorded at 7.5–30 frames per second |
| End-diastole (ED) | the moment the heart is relaxed: arteries longest and still; the standard measurement frame |
| QCA | quantitative coronary angiography: measuring vessel diameters from the image |
| %DS | percent diameter stenosis: how much narrower the vessel is than its healthy reference |
| MLD | minimal lumen diameter |
| Densitometry | estimating the lumen from how much contrast (darkness) the vessel shows |
| Segmentation | marking which pixels belong to vessels (AI) |
| Surrogate | a neural network trained to reproduce a physics model's output |
| ONNX | a standard format for running neural networks on any computer |
| AUC | area under the ROC curve: 1.0 perfect, 0.5 chance |
| Coefficient of variation | spread ÷ mean: lower = more repeatable |
| Weighted kappa | agreement between two graders beyond chance; 0.4–0.6 moderate, > 0.6 good |

## Appendix B. Figures (in `figures/`)

| File | What it shows | Use in |
|---|---|---|
| `pipeline.png` | the eight processing steps | 1, 4 |
| `synthetic_ed_frames.png` | end-diastolic frames of a beating-heart run, ★ recommended | 3, 4.1 |
| `synthetic_vessel_and_lesions.png` | vessels (blue), path (yellow), lesion (red, numbered) | 3, 4.3 |
| `synthetic_lesion_closeup.png` | magnified lesion: plain and with measured edges | 3, 6 |
| `synthetic_ffr_curve.png` | virtual FFR along the vessel, before and after the stent (0.41 → 0.95) | 3, 4.9 |
| `synthetic_stent.png` | the suggested stent drawn on the vessel | 3, 4.9 |
| `synthetic_heartbeats.png` | FFR on five heartbeats and their median | 4.7 |
| `cadica_ed_frames.png` | a real run's end-diastolic frames (CADICA, CC BY 4.0) | 4.1 |
| `cadica_vessel_and_lesions.png` | a real lesion graded 90–98% by experts (cyan box) | 6 |
| `cadica_ffr_curve.png`, `cadica_heartbeats.png`, `cadica_stent.png` | the same real case through the workflow | 6 |
| `repeatability.png` | one frame 18.7% vs median of 5 frames 5.1%, against QFR 9.4% | 4.7, 5.3 |
| `segmentation_examples.png` | AI vessel segmentation on public angiograms (ARCADE, CC0) | 4.2 |
| `rendered_angiograms.png` | synthetic angiograms from our X-ray simulator | 4.2 |
| `physics_ffr_vs_ds.png` | FFR against stenosis severity from the physics model | 4.4 |
| `vessel_tree.png` | a vessel tree with side branches | 4.4 |
| `surrogate_vs_solver.png` | learned surrogate against the physics solver | 4.5 |
| `uncertainty_coverage.png` | calibrated uncertainty intervals | 4.6 |
| `two_view_3d.png` | two-view 3D reconstruction | 4.8 |
| `synthetic_validation.png` | the simulated validation study (ROC, Bland–Altman) | 5.2 |
| `stenosis_grading.png` | measured narrowing against expert grades (real angiograms) | 5.3, 6 |
| `fda_nozzle.png` | physics against the FDA benchmark nozzle | 5.3 |

Public images are from ARCADE (CC0) and CADICA (CC BY 4.0: Jiménez-Partinen A et al., Expert
Systems 2024;41:e13708, doi:10.17632/p9bpx9ctcv.5); credit CADICA on any slide that shows its
images.

## Appendix C. Screenshots to take from the live app

The app runs locally (`uv run vstent app`, then http://localhost:8501). Take screenshots
(Windows: Win + Shift + S) of:
1. the whole page after choosing *Synthetic 1* (all steps visible): the overview slide;
2. step 1 with the frame slider (★ visible): the frames slide;
3. step 2 with a lesion circled and its close-up: the lesions slide;
4. step 3 showing the large coloured FFR number and the guide in the sidebar: the result
   slide;
5. step 4 after *Suggest a stent*, showing before → after: the planning slide;
6. a CADICA case marked "worst lesion 90–98%", on a ◆ frame after *Through the expert box*:
   the real-data slide.

## Appendix D. References

- Tonino PAL et al. Angiographic versus functional severity of coronary artery stenoses in the
  FAME study. J Am Coll Cardiol 2010;55:2816–21.
- Knuuti J et al. 2019 ESC Guidelines for the diagnosis and management of chronic coronary
  syndromes. Eur Heart J 2020;41:407–77.
- Xu B et al. FAVOR II China. J Am Coll Cardiol 2017;70:3077–87.
- Westra J et al. FAVOR II Europe-Japan. J Am Heart Assoc 2018;7:e009603.
- Westra J et al. QREP: reproducibility of QFR. EuroIntervention 2022.
- Lee SH, Gim DH et al. Artificial intelligence-driven angiographic quantification of FFR
  (MPFFR). J Am Heart Assoc 2026;15:e049331.
- Kopanitsa G et al. AngioAI-QFR. arXiv:2512.09134 (2025).
- CathWorks. FFRangio pooled analysis of 5 prospective cohort studies (company publication).
- ARCADE dataset (CC0); CADICA dataset (CC BY 4.0); DEFINE-FLOW and CONTRAST (CC0, Johnson
  et al.); FDA benchmark nozzle (public domain).

---

*Prepared from the vstent repository; numbers from its generated reports. Research use
only.*
