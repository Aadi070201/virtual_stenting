# Validation data specification

> **Research use only.** vstent is a research prototype, not a medical device. This document
> specifies the data for a retrospective validation study (`PROTOCOL.md`). It is not legal or
> data-protection advice: the de-identification and data-sharing steps must be confirmed by the
> partner site's data-protection officer and ethics committee.

## The rule: patient data stays local

- **No patient data, and nothing derived from it, ever enters this repository** — not images,
  masks, case lists, reference values, predictions or per-vessel reports.
- Keep study data **outside the repository** (recommended, e.g. `D:/vstent_study/`) or in a
  git-ignored folder (`data/`, `runs/`).
- `vstent validate` enforces this: it refuses case lists, images, masks, reference files and
  output folders that lie in the repository's tracked files. The default output folder is
  `runs/validation/<case-list>-<time>/` (git-ignored).
- Processing runs on the computer where the data is: the partner's own machine if the
  agreement requires it. Nothing is uploaded anywhere; the software makes no network calls.
- Outputs (`predictions.csv`, `report.md`, `report.json`) carry pseudonymous case IDs. Treat
  them as study data: store and share them only as the data agreement allows.

## What is needed per vessel

| Item | Required | Notes |
|---|---|---|
| Cine angiogram (DICOM) showing the vessel | yes | The analysed frame is chosen by the analyst (end-diastole, least foreshortening and overlap). A second projection ≥ 30° apart is wanted for P5 (two-view 3D). |
| Pressure-sensor position | yes | Marked on the analysed frame as pixel coordinates (the end point). If only a distance from the ostium was recorded, note it in `calibration` and mark the point by that distance. |
| Wire FFR at the sensor, at maximal hyperemia | yes (or iFR) | The value on the recording; hyperemia agent and route recorded in the study database. |
| iFR | optional | Analysed as classification only (FFR and iFR are different quantities). |
| Mean aortic pressure Pa at hyperemia (mmHg) | recommended | From the guide catheter; used as the inlet pressure. Without it the default (90 mmHg) is used. |
| Pd, drift check (Pd/Pa at pull-back) | recommended | For excluding drifted measurements per the protocol (reference file, `excluded`). |
| Calibration | recommended | Pixel size at the vessel (mm/px), from catheter calibration or the isocentre; see below. Without it 0.30 mm/px is assumed and the case is flagged. |
| Vessel label | recommended | LAD, LCX, RCA (or a site-defined label); used for per-territory tables. |

### DICOM attributes to keep

These are needed for the analysis (now or in P5 / P7) and do not identify the patient. Tag
numbers were checked against pydicom's DICOM data dictionary.

| Attribute | Tag | Use |
|---|---|---|
| Pixel Data, Rows, Columns, Bits Stored, Photometric Interpretation | (7FE0,0010), (0028,0010), (0028,0011), (0028,0101), (0028,0004) | the image |
| Number of Frames | (0028,0008) | multi-frame cine runs; the `frame` column selects one |
| Frame Time / Cine Rate | (0018,1063) / (0018,0040) | P7: frame counting |
| Imager Pixel Spacing | (0018,1164) | pixel size at the detector (not at the vessel) |
| Distance Source to Detector | (0018,1110) | P5: magnification correction |
| Distance Source to Patient | (0018,1111) | P5: magnification correction |
| Estimated Radiographic Magnification Factor | (0018,1114) | P5: cross-check |
| Positioner Primary Angle / Secondary Angle | (0018,1510) / (0018,1511) | P5: projection geometry |

Pixel size: `ImagerPixelSpacing` is measured at the detector. The vessel is magnified by about
`DistanceSourceToDetector / DistanceSourceToPatient` (isocentre approximation), so the pixel
size at the vessel is smaller. `vstent validate` does **not** use `ImagerPixelSpacing`
automatically. Put the calibrated value in `mm_per_px` and the method in `calibration`
(e.g. `catheter 6F`, `isocentre`). P5 adds the magnification correction from the geometry tags.

## De-identification checklist

Done at the partner site, before any data leaves the clinical system. The DICOM standard's
Basic Application Level Confidentiality Profile (PS3.15 Annex E) is the reference; the site's
own de-identification tool and data-protection officer decide.

- [ ] Remove or replace patient identifiers: Patient's Name (0010,0010), Patient ID
      (0010,0020), Other Patient IDs (0010,1000), Patient's Birth Date (0010,0030).
- [ ] Remove or replace study identifiers: Accession Number (0008,0050); replace Study,
      Series and SOP Instance UIDs (0020,000D), (0020,000E), (0008,0018) with new UIDs,
      consistently within a patient.
- [ ] Remove people and places: Institution Name (0008,0080) and Address (0008,0081),
      Referring / Performing Physician's Name (0008,0090) / (0008,1050), Operators' Name
      (0008,1070), Station Name (0008,1010), Device Serial Number (0018,1000).
- [ ] Dates and times: remove, or shift consistently per patient as the site decides
      (Study Date (0008,0020), Study Time (0008,0030), Content Date (0008,0023)).
- [ ] Remove private tags and curves / overlays that may carry identifiers.
- [ ] **Burned-in text:** check Burned In Annotation (0028,0301) and look at the pixels. Names,
      dates or IDs written into the image corners must be masked or cropped (cropping changes
      the coordinates; mark points after cropping).
- [ ] Keep the geometry and frame attributes in the table above.
- [ ] Set Patient Identity Removed (0012,0062) = YES and record the method in
      De-identification Method (0012,0063).
- [ ] File and folder names carry only the pseudonymous IDs (no names, dates of birth or
      hospital numbers).
- [ ] The key linking pseudonyms to patients stays at the hospital and is never shared.
- [ ] Clinical variables, if any, are coarsened as agreed (e.g. age bands).

## The two files

Two CSV files (UTF-8, comma-separated, header row), so that the people running the software
never see the wire values (`PROTOCOL.md`, blinding). Templates with the header only:
`templates/cases_template.csv` and `templates/reference_template.csv`. Paths are relative to
the CSV's folder; an empty cell means "not given". Unknown or misspelt columns are refused.

### `cases.csv`: for the analysts (index test)

| Column | Required | Type, units | Meaning |
|---|---|---|---|
| `case_id` | yes | text, unique | pseudonymous vessel ID, e.g. `P017-LAD` |
| `patient_id` | yes | text | pseudonymous patient ID (vessels of one patient are clustered in the statistics) |
| `vessel` | no | text | LAD, LCX, RCA, … |
| `image` | yes | path | PNG / JPG / DICOM file |
| `frame` | no | integer ≥ 0 | frame of a multi-frame DICOM (0-based); default the first |
| `start_row`, `start_col` | yes | integer px | proximal start point (row, column) on that frame |
| `end_row`, `end_col` | yes | integer px | **pressure-sensor position** (row, column) |
| `mm_per_px` | no | mm per pixel, > 0 | calibrated pixel size at the vessel |
| `calibration` | no | text | how `mm_per_px` was obtained |
| `pa_mmhg` | no | mmHg | mean aortic pressure at hyperemia |
| `pv_mmhg` | no | mmHg | venous pressure (rarely measured; default 5) |
| `mask` | no | path | vessel mask (8-bit PNG, same size as the frame, vessel = 255); an analyst-corrected contour. Used instead of the automatic segmentation unless `--ignore-masks`. |

Both points must lie on the vessel: a point farther than the vessel's half-width + 5 px from
the segmented vessel is refused and the vessel is recorded as not analysable.

### `reference.csv`: held by the data manager until predictions are locked

| Column | Required | Type | Meaning |
|---|---|---|---|
| `case_id` | yes | text | as in `cases.csv` |
| `ref_type` | yes | `FFR` or `iFR` | the invasive index |
| `ref_value` | yes | number | the value at the sensor position |
| `excluded` | no | text | a reason (e.g. `drift 0.04`) removes the vessel from the accuracy analysis; it is listed in the report's flow of vessels |

## Running it

```bash
# 1. blinded: run the pipeline on every vessel and lock the predictions (SHA-256)
uv run vstent validate --manifest D:/vstent_study/cases.csv --out D:/vstent_study/run1
# 2. after locking: the statistician scores them against the reference
uv run vstent validate --predictions D:/vstent_study/run1/predictions.csv \
    --reference D:/vstent_study/reference.csv
```

The run records the software version, git commit, model and configuration hashes and the
case list's SHA-256 (`predictions_meta.json`). Scoring refuses predictions that changed after
they were locked. A dry run on synthetic stand-in cases: `uv run vstent validate-synthetic`
(`reports/validation_synthetic.md`).
