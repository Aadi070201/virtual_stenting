# vcathlab progress

> **Research use only.** Simulated data.

**2026-11-04**
- **C1:** virtual patient and one X-ray frame. Anatomical left and right trees in the heart's
  grooves (branches attached: gap 0 mm over 20 patients, 90 branches), eccentric lesions,
  standard views with table pan, additive attenuation at crossings, catheter, heart shadow,
  ARCADE-fitted imaging chain; full labels. About 1 s per frame. 6 tests.
- **C3:** format `vcathlab-frames` v1 (`docs/DATA_FORMAT.md`), `vcathlab dataset`, and
  vstent's reader. **Lesion dataset v1** (`data/lesion_v1/`, 609 MB, card next to it): 3,000 /
  300 / 300 images; train: 25% without a lesion, 4,499 lesions (2,040 mild < 50%, 1,071
  50–69%, 1,099 70–89%, 289 ≥ 90%); of the ≥ 50% lesions 16% on an overlap, 30%
  foreshortened, 37% eccentric; pixel size 0.18–0.40 mm. Rendered at commit 7252b72; its
  manifest says `dirty: True` because an untracked vstent report existed elsewhere in the
  repository (the cath lab's files were committed). The dirty check now looks at `cathlab/`
  only.

**Open issues**
- Realism gaps against real angiograms: no small branches (septals, atrial, conus), smooth
  backgrounds, uniform vessel walls (no diffuse disease or calcium), no motion blur, lighter
  contrast. Whether they matter is measured by C4 on real validation data, not by eye.
