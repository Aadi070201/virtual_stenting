# Exported dataset format: `vcathlab-frames`, version 1

> **Research use only.** Simulated images; no patient data.

The only link between the virtual cath lab and its users (vstent). A reader needs NumPy and
JSON, nothing from `vcathlab`. A change that breaks a reader raises the version.

## Files in a dataset folder

| File | Content |
|---|---|
| `manifest.json` | `format` (`"vcathlab-frames"`), `version` (1), `disclaimer`, `name`, `created_utc`, `git_commit`, `git_dirty`, `config` (everything needed to render it again), `segment_codes`, and per split: `n`, `shards` (each: `file`, `first`, `count`, `sha256`), `records`, `records_sha256` |
| `<split>-<k>.npz` | compressed arrays of `count` images, in order from index `first` |
| `<split>.jsonl` | one JSON object per image, in index order |
| `card.md` | what was drawn (artery, views, pixel sizes, lesion severities, hard cases) |

Splits are `train`, `val` and `test`; every image is its own simulated patient, with seeds in
disjoint ranges, so no patient is in two splits.

## Arrays (each (N, H, W))

| Key | dtype | Meaning |
|---|---|---|
| `images` | uint8 | the X-ray frame (dark vessels on a lighter background) |
| `vessel` | uint8 | 1 where a contrast-filled artery covers ≥ half the pixel (catheter excluded) |
| `segments` | int16 | index into `segment_codes` (SYNTAX numbers, e.g. `"6"` = proximal LAD) of the segment with the longest chord; −1 elsewhere |
| `overlap` | uint8 | 1 where two arteries (or distant parts of one) cross in the projection |
| `catheter` | uint8 | 1 on the catheter's contrast lumen |

## Records (one JSON line per image)

- `index`, `seed`, `scene` (the full editable scene: artery, dominance, lesions, view,
  catheter size).
- `truth`: `artery`, `dominance`, `scale`, `view` (e.g. `"LAO 44° CRA 27°"`),
  `iso_pixel_mm` (pixel size at the isocentre), `detector_pixel_mm`, `sid_mm`, `sod_mm`,
  `heart_centre_pixel_mm`, `catheter_french`, `catheter_outer_mm`, `imaging` (drawn chain
  values).
- `lesions`: one object per lesion:
  - `segment`, `segment_name`, `vessel`;
  - `ds_pct` (true diameter stenosis of the area-equivalent lumen), `area_stenosis_pct`,
    `ref_mm`, `mld_mm`, `length_mm`, `eccentricity` (minor / major axis, 1 = round),
    `orientation_deg`;
  - in this view: `apparent_ds_pct` (projected width vs reference), `box_xywh` (x, y, w, h
    in pixels, or `null` if the lesion is outside the image), `inside_share`,
    `overlap_share`, `min_sine` (1 = in the image plane), `foreshortened`, `pixel_mm`
    (pixel size at the lesion's depth).

Every lesion is listed, mild ones too. Which count as positives (e.g. `ds_pct` ≥ 50 and
`inside_share` ≥ 0.5) is the reader's choice and must be stated where it is used.
