# Invention notes

> **Research use only — not for clinical decisions.**
>
> **What this file is:** a dated technical record of ideas in this repository that *might* be
> new, kept as a starting point for a conversation with a patent attorney. **It is not a patent
> draft, not a novelty or freedom-to-operate opinion, and not legal advice.** No professional
> prior-art search has been done. For items 5–6 a first web search was made on 2026-10-31 (its
> hits are listed); for the others, each section lists the prior art that most obviously needs
> checking. Several of these ideas may well turn out to be known.
>
> **Disclosure caution:** in many jurisdictions (e.g. under the European Patent Convention)
> any public disclosure before filing (a public repository, a paper, a talk, a demo) can
> destroy novelty; others (e.g. the US) have only a limited grace period. Keep this repository
> private and get advice before showing it publicly.

Dates and commits below come from the git history of this repository. Evidence points to the
generated reports in `reports/`.

---

## 1. Monotone-increment surrogate head for FFR pullback curves

- **First implemented:** 2026-10-28, commit `a565305` (`src/vstent/surrogate/model.py`,
  `monotone_ffr_head`).
- **What it is:** a 1D U-Net predicts, for every centreline segment, an unconstrained value
  *z*. The head turns these into non-negative pressure-drop increments
  softplus(*z*) / (N − 1) and accumulates them, so that
  FFR(s₀) = 1 and FFR(s_{i+1}) ≤ FFR(s_i) **by construction**: a pullback curve that starts at
  1 and never increases, as physics requires for a vessel without pressure recovery.
  Conditioning on whole-vessel quantities (length, reference size, Pa, Pv, microvascular
  resistance) enters through FiLM layers and a pooled bottleneck summary.
- **Why it may matter:** the curve shape is guaranteed rather than learned, which removes a
  class of non-physical outputs; in planning, the distal value is consistent with the whole
  curve. On the synthetic test split it reproduces the reduced-order model with a distal MAE
  well inside the target (see `reports/surrogate_eval.md`).
- **Known gap / possible extension:** the lower physical bound FFR ≥ Pv / Pa is *not* enforced;
  the surrogate went below it once when extrapolating far outside its training range
  (`reports/qca_gallery.md`; the pipeline now warns). A bounded variant, e.g.
  FFR = Pv/Pa + (1 − Pv/Pa) · exp(−cumsum(softplus(z))), would enforce both bounds.
- **Prior art to check:** monotonic / shape-constrained neural networks; physics-informed
  networks for haemodynamics; machine-learning FFR from CT and angiography (e.g. published
  ML-based CT-FFR work and commercial angiography-derived FFR products); cumulative-sum output
  layers for monotone sequences in other fields.

## 2. Constrained virtual-stent plan optimiser with a learned evaluator and solver re-check

- **First implemented:** 2026-10-28, commit `49c9b40` (`src/vstent/planning/optimizer.py`,
  `src/vstent/planning/stent.py`).
- **What it is:** an enumerate-and-filter optimiser over a generic catalogue.
  - **Search:** diameter × length × start position on a fixed grid, restricted to stents that
    cover a lesion's minimal lumen diameter.
  - **Constraints, vectorised:**
    - the stent lies inside the vessel;
    - nominal diameter ≤ a factor × the reference diameter at the distal stent end;
    - the maximum diameter stenosis over both landing zones stays below a threshold.
  - **Objective (lexicographic):** the shortest total stent length that reaches the target
    distal FFR, then the highest FFR; if no plan reaches the target, the highest FFR.
  - **Two stents:** a pool of the best candidates *per stent length* is kept for each lesion,
    so short two-stent combinations are not crowded out by long single stents (this was an
    observed failure mode of FFR-only pools). Pairs without overlap are evaluated.
  - **Evaluation:** every candidate is scored in one batch by the learned surrogate (ONNX),
    with the microvascular resistance, Pa and Pv fixed at their pre-PCI values. The proposed
    plans are then **re-checked with the physics solver**, and both values are reported.
  - **Output:** the top plans and a Pareto front (total length vs post-PCI FFR).
- **Why it may matter:** it combines a fast learned evaluator with a physics re-check of only
  the few returned plans, and returns an interpretable, constraint-respecting short list.
  Evidence: `reports/planner_benchmark.md` (throughput, agreement between surrogate-chosen
  and solver-chosen plans).
- **Prior art to check:** virtual PCI / virtual stenting tools for CT-FFR and
  angiography-derived FFR (commercial planners and academic papers); stent sizing and
  landing-zone rules in QCA guidelines; surrogate-assisted optimisation with high-fidelity
  verification in engineering design.

## 3. Two-sided robust reference diameter with capping for 2D QCA before physics

- **First implemented:** 2026-10-28, commit `accebf7` (`src/vstent/imaging/qca.py`,
  `robust_reference`, `qca_from_centerline`).
- **What it is:** along a path traced on a 2D angiogram, diameters are measured on the
  segmentation probability map. Then:
  - **One-sided branch rule:** when one edge is far farther than the other, a side branch is
    assumed and twice the nearer edge distance is used.
  - **Reference diameter:** a Theil–Sen line refined on the points within a band of
    [0.85, 1.15] × the line, so that *both* narrowings (below) and overlaps or branches
    (above) are excluded.
  - **Capping:** diameters above the upper band are capped **before** the profile goes to the
    pressure-loss model. Otherwise a spike from an overlapping vessel would produce a fictitious
    narrowing-plus-expansion loss.
- **Why it may matter:** it makes a physics model usable on raw 2D segmentations with overlaps;
  on one ARCADE image it changed the reference from an implausible value to a plausible one
  (see `docs/DECISIONS.md` D21).
- **Prior art to check:** conventional QCA reference methods (interpolated and
  regression-based references in commercial QCA software); robust regression in vessel
  analysis. **This element is the most likely to be known.**

## 4. Planned (not implemented): sim-to-real training for angiogram → FFR

- **Recorded:** 2026-10-28 (Phase 2 backlog in `docs/PLAN.md`; nothing implemented).
- **Idea:**
  - Generate 3D coronary trees with steady-flow CFD labels.
  - Render synthetic angiograms from them (projection, contrast, noise, overlap).
  - Train models that go from angiogram (or angiogram-derived profiles) to FFR, using the
    reduced-order labels for pre-training and the CFD labels for fine-tuning (multi-fidelity).
  - Keep the same feature pipeline and ONNX deployment path that this MVP validates.
- **Status:** an idea only; record its date, and do not describe it publicly before advice.
- **Update 2026-11-01 (P8):** the segmentation part is implemented (commits 7c7b74f, dd76215,
  7bd8ede–cae089c; `reports/sim_to_real.md`). On real angiograms it made no measurable
  difference; it was adopted under a pre-registered non-inferiority rule (D53). **No novelty
  is claimed for it:** rendering vessel trees to train segmentation is established practice.
  Angiogram → FFR is deferred until there are CFD or wire-FFR labels (D53).
- **Prior art to check:** synthetic angiogram rendering, domain randomisation and sim-to-real
  in medical imaging; multi-fidelity surrogate modelling for CFD.

## 5. Imaging-chain model fit for the lumen in 2D angiography (P4c)

- **First implemented:** 2026-10-31, commit `d37915f` (`src/vstent/imaging/lumen_model.py`;
  option `lumen="model"` of `qca_from_centerline`); evidence `reports/lumen_synthetic.md`
  (commit `b71e60a`) and, on real angiograms, `reports/stenosis_grading.md`.
- **The problem it addresses:** edge-based QCA reads a lumen's edges where the blurred
  intensity profile is steepest. Below a few pixels the blur, not the lumen, sets that distance,
  so tight stenoses are under-read. X-ray systems also apply edge enhancement (unsharp masking),
  which shifts edges and, along the vessel, exaggerates the contrast deficit of a short lesion.
  On rendered vessels with clinical-like settings, edges never read more than ~40–75% diameter
  stenosis, however tight the lesion, and the FFR computed from them is far too high.
- **What it is:**
  - The image is resampled along the normals of the centreline (a straightened vessel).
  - A forward model predicts the straightened image:
    - exact cell-averaged chord through a circular lumen of radius r(z) centred at x0(z);
    - Beer–Lambert transmission exp(−μ · chord);
    - system blur (Gaussian, σ_b);
    - the system's unsharp-mask edge enhancement ((1 + a)·Y − a·Gauss(Y, σ_e));
    - the background from bands on both sides.
  - **The imaging-chain parameters (μ, σ_b, a, σ_e) are estimated from the clinical image
    itself, shared along the path**, jointly with r(z) and x0(z) per point, by robust
    (soft-L1) nonlinear least squares with a curvature penalty. No calibration of the X-ray
    system is needed.
  - The Jacobian is analytic and sparse, because blur and edge enhancement are linear
    separable filters. The fit takes about 1 s per vessel on a laptop CPU.
  - The deep segmentation only provides the centreline and start values, and the fitted lumen
    feeds the reduced-order FFR model and the virtual-stent planner.
- **Why it may matter (synthetic evidence, best case: the renderer shares the model's
  physics):**
  - lesions ≥ 85% DS read within 2.3 points vs 30.7 for edges
    (`reports/lumen_synthetic.md`, after D45);
  - the FFR error on those falls from 0.53 to 0.04;
  - a lumen narrowed only along the X-ray (invisible in width) is seen through its area:
    eccentric lumens are read within 5.5 points vs 19.7.
- **On real angiograms (CADICA, expert visual grades; `reports/stenosis_grading.md`,
  commit `2c93bbd`) the advantage is not shown:**
  - development patients: weighted kappa 0.25 vs 0.19 for the product's intensity edges, but
    AUC for ≥ 70% lesions 0.73 vs 0.82, so it did not replace them (D44, rule fixed before the
    results);
  - test patients: kappa 0.23 vs 0.14, AUC 0.74 vs 0.72 (nothing chosen on these);
  - median %DS of lesions graded 90–98%: 53 (edges 42) on the test patients, still far below
    the grade.

  Something in real images (scatter, background structure, overlapping vessels, the vendor's
  processing) is missing from the model. **For a filing, the evidence is synthetic only**;
  a pitch must not claim a real-angiogram advantage until a fresh test set shows one.
- **Possible claim elements (for the attorney):**
  - joint, along-path estimation of the imaging system's blur and edge-enhancement parameters
    from the angiogram being measured;
  - its use to recover sub-resolution and eccentric lumen areas;
  - feeding them into a physics-based FFR and virtual-stent computation.
- **Prior art found (first search, 2026-10-31), to be reviewed by a professional:**
  - Pappas & Lim: a parametric model of film density along lines perpendicular to the vessel,
    with background and imaging-system distortions, estimating diameter and centre per line
    (SPIE vol. 504, 1984; IEEE Trans. ASSP 36(9), 1988; MIT RLE Technical Report 528, 1987).
    **Closest known prior art**: model-based fitting per profile is not new.
  - Model-based estimation of arterial diameter from X-ray angiograms (Chan, MIT thesis,
    1996).
  - Small-vessel diameters overestimated because of the system's point-spread function, and a
    size-adaptive edge operator (Reddy et al., IEEE Trans. Med. Imaging, 1997).
  - A commercial QCA package that reverses unsharp masking by iterative deconvolution before
    edge detection (Philips automated coronary analysis, studied in 1995).
  - **US 10,580,141 B2** (Koninklijke Philips, granted 2020-03-03, listed as active):
    sub-resolution lumen diameters in CT from centreline intensity profiles (FWHM). It is a
    different modality, but its claims as summarised are broadly worded, so **a
    freedom-to-operate review is needed**.
  - Densitometric QCA (Reiber and others, 1980s–90s), known to fail on the tightest lesions
    and on uneven backgrounds.
- **What may still be new** (to be judged by an attorney):
  - estimating the edge-enhancement parameters jointly along the whole path from the clinical
    image, not from vendor calibration or deconvolution with known settings;
  - the exact cell-integrated chord for sub-pixel lumens;
  - the combination with automatic segmentation and FFR or virtual-stent planning.
- **India:** algorithms and computer programmes per se are excluded (Patents Act s. 3(k)), as
  are methods of diagnosis (s. 3(i)). Claims would need to be framed as a technical
  improvement of an X-ray imaging measurement system, not as a diagnostic method.

## 6. Mask-gap bridging with image-only measurement and a no-lumen flag (P4c)

- **First implemented:** 2026-10-31, commits `d37915f` and `cacd7bd`
  (`imaging/centerline.py` `bridge_gaps`; `imaging/qca.py` `recentre_gaps`,
  `_no_lumen_in_gaps`; `inference/pipeline.py` `lumen_warnings`).
- **The problem it addresses:** a segmentation network often breaks a vessel exactly at its
  tightest narrowing or occlusion, or smooths across it. Path tracing then fails, or measures
  the gap with diameters interpolated from the healthy vessel, so a total occlusion reads as
  open (P4b: 10 of 11 occlusions measured, median 6% DS).
- **What it is:**
  - Skeleton ends of different components that face each other across a short gap are joined
    from anchors a few pixels back (skipping the skeleton's end hook).
  - Bridged points are re-centred on the image's lumen and measured from the image only, never
    from the mask.
  - A bridged point without lumen edges gets no lumen.
  - A run of points below 1 px of lumen is reported as "critical stenosis or total
    occlusion", with the warning that FFR is not valid for an occluded vessel.
- **Prior art to check:** gap filling and minimal-path vessel tracking (e.g. Deschamps &
  Cohen, minimal paths, 2001), connectivity-preserving segmentation, deep-learning detection of
  chronic total occlusions. **Likely known as separate elements**; any value lies in the system
  combination.
- **On real angiograms it did not trigger:** none of CADICA's 11 total occlusions was read as
  critical (≥ 90% DS) by any method; 10 were measured as open. The segmentation draws a vessel
  through the occlusion, so there is no gap to bridge. It works on rendered gaps only, so
  occlusion detection needs a different signal (e.g. contrast washout over the cine frames).

## 7. Phase-corrected contrast flow velocity with a calibrated interval (P7)

- **First implemented:** 2026-11-02 (`src/vstent/imaging/flow.py`: `flow_time`,
  `phasic_slope`, `flow_velocity`, `combine_runs`; `src/vstent/imaging/motion.py`; decision
  D55); evidence `reports/cine_synthetic.md` (synthetic only).
- **The problem it addresses:** coronary flow is phasic (faster in diastole). A contrast front
  timed over part of a heart beat, as in frame-count velocity methods, gives the systolic or
  the diastolic speed depending on when the contrast was injected. On the P7 development
  cases the plain front speed was off by about 20% (median, 3D path).
- **The idea:**
  1. The ED frames come from the R-wave pointer, the ECG, or the images (vessel-tree
     spread).
  2. They give each frame's heart phase. Arrival times along the path are mapped to "flow
     time", which runs at the systolic speed for a fixed systole after each R wave and at a
     population diastolic / systolic ratio for the rest of the beat; one beat of flow time
     per beat.
  3. The cycle-average velocity is the robust slope of distance on flow time, with the
     points followed through the beat by registering only the fully opacified frames.
     Filling and washout frames take the transform of the same heart phase one or more beats
     away.
  4. A split-conformal interval, calibrated on separate cases, is widened for a single
     view's foreshortening, and the velocity is used only when guards pass.
  On the development cases the median error went from about 20% to about 6–10%.
- **Status:** a candidate for discussion only. **Prior art exists for the general idea:**
  Tar et al. (Cardiol J 2021;28:163–165) note that the frame count could in theory be
  corrected to the average flow over the heart cycle, and leave it as future work. Phasic
  coronary flow and ECG-gated frame selection are textbook. Any claim would have to rest on
  the specific combination (flow-time mapping from image- or ECG-derived phase, phase-locked
  motion transforms for partially filled frames, calibrated interval with foreshortening
  widening) and its technical effect.
- **Prior art to check:** US 11,308,621 ("Method and apparatus for acquiring blood flow volume
  and blood flow velocity of coronary artery"); the QFR patents of Pulse Medical / Medis
  (contrast frame count, FAVOR conversion; also a **freedom-to-operate** question for using
  the FAVOR Pilot relation in a product); automated TIMI frame counting; ECG-free cardiac
  phase detection (e.g. Ciusdel et al., Comput Med Imaging Graph 2020).
- **India:** s.3(k) excludes computer programs per se and mathematical methods; a claim would
  need to be framed around the imaging system and its measured technical effect.
- **Real-data support (2026-11-02, D56):** in DEFINE-FLOW's Doppler tracings (455 real
  lesions) the resting diastolic / systolic velocity ratio has a median of 1.73, so the phasic
  bias the method corrects exists in patients. It differs by artery (RCA 1.35, left coronary
  about 1.8), which points to an artery-specific prior.
- **Weakness:** the synthetic test draws its flow profiles from the same family the
  correction assumes; its benefit on real contrast transits still needs a phantom or
  Doppler-wire data recorded with angiograms.

---

## Not considered inventive (engineering choices)

The following are recorded only so they are not mistaken for claims:
- shared pre-processing for training, ONNX and the app;
- minimum-cost edge tracking along the vessel and side branches taken from the segmentation's
  outline (D45): the classical QCA edge criterion (Reiber and others, 1980s), corrected here;
- frame-level calibration of the imaging chain (D46): did not hold on real frames;
- the cine analysis frame from the opacification curve and contrast transit along the vessel
  (P4d, D47): automated frame selection and frame-count flow are established (e.g. QFR's
  contrast-flow variant, automated TIMI frame counting);
- regulatory documents generated from YAML and checked against the code, and the SOUP list /
  SBOM from the lock file (P8, D48): established docs-as-code and requirements-traceability
  practice;
- two-view 3D reconstruction (P5, D49): DICOM projection geometry, epipolar pairing by dynamic
  programming, triangulation, motion (shift) correction from the vessel itself, and
  ellipse-equivalent diameters are established in 3D QCA and angiography-derived FFR (e.g.
  QAngio XA 3D, CAAS, QFR); the magnification correction is textbook;
- end-diastolic frames from the DICOM R-wave pointer or ECG, and from the vessel tree's
  size in the images (P7): ECG gating and image-based cardiac-phase detection are established;
- affine registration of the vessel tree across a run (P7): standard image registration;
- the FAVOR Pilot conversion of resting to hyperemic velocity (P7): published by its authors
  (and a freedom-to-operate question, see 7);
- the scaled lumen (P5b, D50): an imaging-chain fit on healthy windows sets the absolute
  size of an edge-measured profile; model-based QCA (fitting blurred vessel profiles) and
  calibration of edge bias are established;
- chunked, parallel ONNX inference;
- resumable checkpoints;
- report provenance;
- the Streamlit app structure.
