# Retrospective validation of vstent against invasive FFR: protocol template

> **Research use only.** vstent is a research prototype, not a medical device, and must not be
> used for decisions about patients. This is a **template** for a partner site. It has not been
> reviewed by a clinician, a statistician or an ethics committee. Everything in
> **[brackets]** must be completed or confirmed by the study team, and the whole protocol must
> be approved by the responsible ethics committee before any patient data is used.

| | |
|---|---|
| Title | [Diagnostic accuracy of angiography-derived FFR computed by vstent against invasive wire FFR: a retrospective study] |
| Version, date | [0.1, date] |
| Principal investigator, site(s) | [name, hospital] |
| Sponsor / data controller | [ ] |
| Registration | [e.g. ClinicalTrials.gov or OSF, before the reference values are opened] |
| Software | vstent [version tag and git commit, frozen before analysis] |

## 1. Background

Fractional flow reserve (FFR) measured with a pressure wire is the invasive reference for the
functional significance of a coronary stenosis; FFR ≤ 0.80 is the conventional cut-off for
ischaemia. Angiography-derived FFR estimates it from the angiogram alone.

vstent computes FFR along a vessel from one angiographic frame:
- **Measurement:** vessel segmentation, a centreline between two points, and 2D QCA.
- **Physics:** a reduced-order flow model with a fixed hyperemic velocity. It models either a
  single path, or the path with its side branches (tree model).
- **Surrogate and uncertainty:** a learned surrogate with calibrated uncertainty intervals.

So far it has been tested only on synthetic data and on public angiograms without pressure
measurements. **Its agreement with wire FFR is unknown**; this study measures it.

## 2. Objectives

**Primary:** per vessel, the diagnostic performance of vstent's FFR for identifying wire FFR
≤ 0.80, measured as the area under the ROC curve (AUC).

**Secondary:**
1. Sensitivity, specificity, PPV, NPV and accuracy at the 0.80 cut-off.
2. Agreement with wire FFR: Bland–Altman bias and 95% limits of agreement, proportional bias,
   mean absolute difference and Pearson correlation.
3. Feasibility: the share of eligible vessels that can be analysed, and accuracy by intention
   to diagnose (non-analysable vessels counted as misclassified).
4. The other index tests: single path with the reduced-order solver, and single path with the
   surrogate.
5. The grey zone: accuracy in vessels whose uncertainty interval contains 0.80 and in those
   whose interval does not, and a hybrid strategy (wire only when flagged).
6. Results per vessel territory (LAD, LCX, RCA) and per calibration method.
7. [Inter- and intra-observer reproducibility on a random [20]% subset analysed twice.]
8. [Analysis time per vessel.]
9. [Classification against iFR ≤ 0.89 where iFR was measured instead of FFR.]

## 3. Design

- A retrospective, observational diagnostic-accuracy study, reported following STARD 2015.
- [Single-centre / multicentre.]
- [Consecutive] eligible patients from [period].
- Analysis is per vessel; several vessels per patient are allowed and handled as clusters.

## 4. Population

**Inclusion (all):**
- Adults (≥ 18 years) who had invasive coronary angiography with wire FFR [at maximal
  hyperemia with intravenous / intracoronary adenosine] in at least one vessel during
  [period].
- The angiographic cine run of the measured vessel is archived in DICOM.
- The pressure-sensor position can be identified on the angiogram (a frame with the sensor, or
  a documented distance).

**Exclusion — candidates to confirm with the investigators [confirm each]:**
- [Prior coronary bypass grafting with a graft to the target vessel.]
- [Target vessel with a chronic total occlusion, or the culprit vessel of an acute
  ST-elevation myocardial infarction.]
- [Left main or ostial right coronary lesions, if the catheter position prevents analysis.]
- [Wire-FFR measurements with pressure drift above [0.03] at pull-back, per site SOP.]

Image-quality problems (overlap, foreshortening, poor contrast) are **not** exclusion criteria
decided after looking at the analysis. Vessels that cannot be analysed are kept and counted as
failures (intention to diagnose). Any image-based exclusion must be pre-defined and decided
by a reviewer blinded to the wire values, before the index test is run.

## 5. Index test: vstent

- **Frozen software.** The version, git commit, models (SHA-256) and configurations
  (`configs/physics.yaml`, `configs/uncertainty.yaml`, `configs/validation.yaml`) are fixed and
  recorded before the first vessel is analysed. `vstent validate` records them automatically in
  `predictions_meta.json`.
- **Primary index test (pre-specified):** `tree_solver`. This is the distal FFR at the sensor
  position from the tree model with side branches (`configs/validation.yaml`,
  `primary_index_test`). Rationale: it is the most complete physics in the software and reduces
  to the single-path model when no side branch is found. [Confirm before registration.]
- **Positive when** the index test's FFR ≤ 0.80.
- **Analyst workflow per vessel** (`DATA_SPEC.md`):
  1. Choose the analysis frame: end-diastole, the projection with the least foreshortening and
     overlap of the measured segment.
  2. Mark the start point (proximal, [at the ostium / catheter tip]) and the end point at the
     pressure-sensor position.
  3. Give the pixel size from [catheter / isocentre] calibration and the mean aortic pressure
     at hyperemia.
  4. Optionally, correct the vessel contour, saved as a mask; corrected contours are recorded
     in the case list.
- **Analysts:** [number, training]. They are blinded to the wire values and to the treatment
  decision.

## 6. Reference standard: wire FFR

- Wire FFR = mean distal pressure / mean aortic pressure at maximal hyperemia, read at the
  sensor position. [Hyperemia: adenosine, dose and route per site SOP.]
- Positive when FFR ≤ 0.80.
- [Where only iFR was measured: positive when iFR ≤ 0.89, a secondary classification-only
  analysis.]
- Measurements excluded for drift (section 4) are marked in the reference file with the
  reason. They appear in the flow of vessels.

## 7. Blinding and locking

- **Who holds what:**
  - The case list (`cases.csv`) has no wire values.
  - The reference file (`reference.csv`) is held by [the data manager]. It is not given to the
    analysts.
- **Locking:** the analysts run `vstent validate --manifest cases.csv`. This writes
  `predictions.csv` and its SHA-256 (`predictions_meta.json`): the lock.
- **Unblinding and scoring:** after locking, [the statistician] runs
  `vstent validate --predictions predictions.csv --reference reference.csv`. Scoring refuses
  predictions that were changed after locking.
- **Fixed in advance:** the configuration that defines the analysis (`configs/validation.yaml`)
  is fixed [and registered] before the reference file is opened.

## 8. Sample size

Generated by `vstent sample-size` from the assumptions in `configs/validation.yaml`. Report:
`reports/validation_sample_size.md`.

Endpoints considered:
- Sensitivity and specificity to a given precision (Buderer 1996).
- The AUC above a null value (Hanley & McNeil 1982 variance).
- The limits of agreement to a given precision (Bland & Altman 1986).

The largest of these is then inflated for clustering (design effect) and for vessels that
cannot be analysed.

**The current assumptions are placeholders.** [Set the expected prevalence, sensitivity /
specificity, AUC, SD of differences, vessels per patient, intra-patient correlation and failure
rate with a statistician, from published angiography-derived FFR studies and the site's case
mix; record the final numbers here.]

## 9. Statistical analysis plan

- **Unit of analysis:** vessel. Patients with several vessels are handled by a patient-level
  cluster bootstrap (2,000 resamples, seed 0) for every interval. Analytic intervals treating
  vessels as independent are reported alongside. [A statistician may prefer GEE or the
  Obuchowski (1997) clustered AUC.]
- **Primary endpoint:** AUC with its 95% interval (DeLong 1988 standard error on the logit
  scale; cluster bootstrap). [Success criterion, pre-specified: the lower bound of the 95%
  interval exceeds [0.80].]
- **Secondary endpoints:** as in section 2. Proportions use Wilson intervals. Limits of
  agreement use the Bland & Altman (1986) standard error sqrt(3 s² / n).
- **Failures:**
  - A vessel the pipeline cannot analyse keeps its reason (`predictions.csv`). It is reported
    in the feasibility and intention-to-diagnose results.
  - Index-test values are not imputed.
- **Subgroups (descriptive):** vessel territory; calibration (given / assumed); [wire FFR in
  the grey zone 0.75–0.85].
- **Software:** `vstent.validation` at the frozen version. Every result is reproducible from
  the locked predictions, the reference file and `configs/validation.yaml`.

## 10. Data management and protection

- **De-identification:** done at the site before analysis, following the checklist in
  `DATA_SPEC.md` (DICOM PS3.15 Annex E as the reference).
- **Processing:** local only. No patient data or derived results enter the vstent source
  repository; the software refuses tracked locations.
- **Storage and access:** [location, access control, encryption, retention period, deletion]
  per the data agreement.

## 11. Ethics and approvals

- [Ethics committee / IRB approval, reference number.]
- [Consent: waiver for retrospective use of de-identified data, if granted.]
- [Data-sharing / data-processing agreement between the site and the research team.]
- [Data-protection assessment (e.g. GDPR DPIA in the EU), if required.]

## 12. Reporting

- **STARD 2015:**
  - the flow of vessels from the report (case list → excluded → eligible → analysable);
  - the 2 × 2 tables;
  - the intervals.
- **Negative or disappointing results are reported in full.** All pre-specified analyses are
  reported, including the grey-zone and per-territory results.
- [Publication policy, authorship.]

## 13. Known limitations of the index test

These should be discussed in the report:
- **Single view:** one angiographic projection, so the analysis is affected by foreshortening
  and overlap. Two-view 3D reconstruction is planned (P5).
- **Fixed flow:** the hyperemic velocity is the same for every patient. Patient-specific flow
  is planned (P7).
- **Pixel size:** it is assumed when not calibrated.
- **Model:** the physics is reduced-order; junction losses in the tree model are neglected.
- **Segmentation data:** the segmentation model was trained on one public dataset (ARCADE).

## 14. Amendments and deviations

Every change after registration is dated, justified and reported. Any change to
`configs/validation.yaml` after the reference values are opened is a protocol deviation.

## References

- Bossuyt PM, et al. STARD 2015: an updated list of essential items for reporting diagnostic
  accuracy studies. BMJ 2015;351:h5527.
- Buderer NM. Statistical methodology: I. Incorporating the prevalence of disease into the
  sample size calculation for sensitivity and specificity. Acad Emerg Med 1996;3:895–900.
- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating
  characteristic (ROC) curve. Radiology 1982;143:29–36.
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated
  receiver operating characteristic curves: a nonparametric approach. Biometrics
  1988;44:837–845.
- Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of
  clinical measurement. Lancet 1986;1:307–310.
- Obuchowski NA. Nonparametric analysis of clustered ROC curve data. Biometrics
  1997;53:498–502.
- Wilson EB. Probable inference, the law of succession, and statistical inference. J Am Stat
  Assoc 1927;22:209–212.
