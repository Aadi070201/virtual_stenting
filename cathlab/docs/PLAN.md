# vcathlab plan

> **Research use only.** Simulated data; never a substitute for validation on real patients.

The virtual cath lab produces angiography with full ground truth for vstent's models, first
for lesion detection. It is a separate app: it never imports vstent, and vstent reads its
output only through `docs/DATA_FORMAT.md`. Check in with the human after each step.

## C1: Virtual patient and one X-ray frame (2026-11-04)

- [x] Anatomy: a prolate-spheroid heart with the coronary arteries in its grooves (LM, LAD
  with two diagonals, circumflex with two obtuse marginals; RCA with PDA and posterolateral
  branch; left or right dominance), SYNTAX segment numbers, normal proximal diameters (Dodge
  1992) with Murray's law at take-offs; branches attached to their parent (gap 0 mm).
- [x] Lesions: severity (area-equivalent diameter stenosis), length, position in a segment,
  eccentric lumens with a fixed orientation (apparent severity depends on the view).
- [x] C-arm: standard projections with jitter, SID, SOD, pixel size at the isocentre
  0.18–0.40 mm, table pan that frames the injected tree.
- [x] X-ray: chord-based attenuation of elliptical lumens, added where vessels cross, joined
  where lumens touch; foreshortening; heart shadow; catheter (5–7 F) with contrast lumen and
  wall; background, blur, edge enhancement and noise (ARCADE-fitted ranges from vstent D39).
- [x] Labels: vessel mask, segment map, overlap map, catheter mask, per lesion: true and
  apparent severity, box, overlap share, foreshortening, pixel size at its depth.
- [x] `vcathlab frame`; tests (projection, tube width, crossings, lesion narrowing,
  eccentric lumen across views, connected trees).

## C2: Beating heart, ECG and cine runs

- [ ] Heart cycle (contraction and twist), breathing, contrast filling and washout, a
  synthetic ECG with the R waves known; the true end-diastolic frames.
- [ ] DICOM export (XA multi-frame with geometry and the ECG).

## C3: Dataset format and a lesion dataset

- [x] `docs/DATA_FORMAT.md` fixed; `vcathlab dataset --config …` writes images, labels and a
  manifest (config, seeds, git commit, hashes).
- [x] Lesion dataset v1 with hard cases: mild plaque (< 50%, unlabelled), crossings,
  foreshortened segments, eccentric lesions, a range of pixel sizes. Dataset card with the
  distributions.
- [x] vstent side: `vstent.data.cathlab` reads the format (checks format, version, SHA-256).

## C4: vstent's lesion detector with cath lab data (planned in vstent's PLAN)

- [ ] Pre-registered in vstent: train with real + simulated images; compare on real
  validation data; decide by a rule fixed in advance; test once.

## C5: Editor app

- [ ] Streamlit: choose a patient, place and grade lesions, set the view and heart rate; see
  the cine, the ECG and every label; export the scene.

## C6: Pressure and flow through the heartbeat (own physics)

- [ ] Aortic pressure waveform, phasic coronary flow (systolic compression), losses at the
  lesions with blood inertia; Pa(t), Pd(t), Q(t), FFR and iFR in step with the cine.
- [ ] Checked against published relations (FFR vs %DS, flow reserve); independent of vstent's
  solver.
