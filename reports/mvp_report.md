# vstent MVP report

> **Research use only — not for clinical decisions.** This research prototype is not a medical device, is not validated and must never be used to make decisions about real patients.

Assembled by `vstent mvp-report` from the generated report files in `reports/` (no hand-typed numbers). Each line links back to its source report; the provenance table lists the commit each one was generated at.

## What the prototype does

Angiogram → vessel segmentation (U-Net, ONNX) → centreline and 2D QCA → FFR pullback from a reduced-order physics model and a learned surrogate (ONNX) → virtual stent with instant post-PCI FFR → plan optimiser (shortest generic-catalogue plan reaching the target FFR) → Streamlit app and case report.

## Acceptance targets: 21 of 21 met

| Milestone | Area | Metric | Value | Target | Met |
|---|---|---|---|---|---|
| M1 | Physics | Healthy 3.0 mm × 50 mm vessel, distal FFR | 0.9819 | ≥ 0.95 | yes |
| M1 | Physics | 3.0 mm × 60 mm, 10 mm cosine lesion, 50% DS | 0.9028 | [0.84, 0.94] | yes |
| M1 | Physics | 3.0 mm × 60 mm, 10 mm cosine lesion, 70% DS | 0.6197 | [0.45, 0.70] | yes |
| M1 | Physics | Distal FFR strictly decreasing, DS 30% → 90% (5% steps) | -0.0074 | max step < 0 | yes |
| M1 | Physics | Batch solver throughput (AMD Ryzen 7 6800H with Radeon Graphics) | 49,759 profiles/s | ≥ 5,000 | yes |
| M2 | Synthetic data | train: FFR < 0.80 / grey zone | 53.3% / 18.4% | ≥ 30% / ≥ 15% | yes |
| M2 | Synthetic data | val: FFR < 0.80 / grey zone | 54.2% / 18.1% | ≥ 30% / ≥ 15% | yes |
| M2 | Synthetic data | test: FFR < 0.80 / grey zone | 53.8% / 17.8% | ≥ 30% / ≥ 15% | yes |
| M3 | Surrogate (test split) | Distal FFR MAE | 0.0013 | ≤ 0.015 | yes |
| M3 | Surrogate (test split) | 95% limits of agreement | [-0.0057, +0.0056] | within ±0.04 | yes |
| M3 | Surrogate (test split) | Agreement at 0.8 | 99.76% | ≥ 97% | yes |
| M4 | Surrogate ONNX | Parity: max abs(FFR_torch - FFR_onnx) over 1,000 test vessels, all points | 5.4e-07 | < 0.0001 | yes |
| M4 | Surrogate ONNX | Single-vessel latency, median (end-to-end) | 1.53 ms | < 10 ms | yes |
| M4 | Surrogate ONNX | Batch of 1,024 vessels, median (end-to-end) | 187.7 ms | < 300 ms | yes |
| M5 | Segmentation | Mean val Dice | 0.8325 | ≥ 0.75 (stretch 0.8) | yes |
| M5 | Segmentation ONNX | Parity max abs(prob torch − onnx) | 1.8e-05 | < 0.0001 | yes |
| M5 | QCA (synthetic tubes) | Median abs. diameter error (worst case) | 0.10 px | ≤ 0.7 px | yes |
| M5 | QCA (synthetic tubes) | MLD location error (worst case) | 1.9 px | ≤ 3 px | yes |
| M5 | QCA (synthetic tubes) | DS error (worst case) | 0.7 points | within ±7 points | yes |
| M6 | Planner | 5,000 candidates, surrogate backend (median) | 0.959 s | < 1 s | yes |
| M7 | App | Stent-control update, server time (max over modes) | 179 ms | < 300 ms | yes |

## Further results (no target)

- Synthetic dataset: 125,000 vessels generated in 8.4 s.
- Surrogate distal-FFR MAE: test 0.0013 vs gradient-boosting baseline 0.0342; out-of-distribution 0.0039 (no target).
- For context, the reduced-order solver computes 1,024 vessels in 18.1 ms, the ONNX surrogate in 188 ms: no speed-up is claimed.
- INT8 quantisation: distal MAE 0.0026 vs 0.0013, speed-up 0.99× → not kept.
- Segmentation Dice on the held-out test split: 0.7925 (no target).
- ARCADE gallery (qualitative): 20/20 images analysed end-to-end; point of maximum DS inside an annotated stenosis box in 8/20; 19 outside the surrogate's training range.
- Planner on 100 synthetic vessels: target reached 84 (solver) / 84 (surrogate); solver confirms 82/84 surrogate plans; same total length 81/84.
- App update medians: synthetic 50 ms, image 77 ms.
- Demo (synthetic tandem lesions, surrogate backend): distal FFR 0.705 → 0.976 (solver) with 3.25 × 12 mm at 14.5 mm + 3.25 × 12 mm at 43.0 mm; planning 0.37 s.

## Limitations

- Research prototype: not a medical device, not validated, never for decisions about real patients.
- Physics: steady, Newtonian, single-vessel reduced-order model (Poiseuille + expansion loss) without side branches; FFR is very sensitive to the assumed microvascular resistance and hyperemic velocity.
- The surrogate is trained and evaluated on synthetic vessels with labels from the same reduced-order model: its accuracy is agreement with that model, not with invasive FFR. It is not faster than the reduced-order solver; it validates the training → ONNX → app pipeline for Phase 2 3D-CFD labels.
- Segmentation: one public dataset (ARCADE) whose masks cover only the annotated SYNTAX segments; no external validation.
- QCA: single 2D projection (foreshortening, overlap), uncalibrated pixel size unless given, crossing vessels can look like narrowings; automatic paths run through branches.
- Planning: the stent only widens the lumen to its radius (no recoil, malapposition or side-branch effects); two-stent search is heuristic; overlapping stents are not modelled.
- No patient outcome, invasive FFR or clinical data were used anywhere.

## Phase 2 next steps (from docs/PLAN.md; not started)

- 3D CFD labels: coronary trees from ImageCAS / ASOCA, steady hyperemic simulations (SimVascular or OpenFOAM); multi-fidelity surrogate training
- Graph neural network / graph transformer surrogate for vessel trees with side branches; flow split by Murray's law
- Two-view 3D reconstruction from DICOM angles; foreshortening correction
- Cine DICOM: end-diastolic frame selection; TIMI frame-count flow estimate
- Sim-to-real: render synthetic angiograms from 3D trees to train angiogram → FFR end-to-end
- Calibrated uncertainty and conformal intervals; grey-zone flagging
- Retrospective clinical validation against wire FFR with a partner hospital (needs ethics approval; human-led)
- Regulatory groundwork: IEC 62304 software lifecycle docs, ISO 14971 risk file

## Sources and provenance

| Report | Generated at commit | Dirty | Created (UTC) |
|---|---|---|---|
| [Physics checks](physics_checks.md) | `1c4c48900f` | False | 2026-10-26T11:25:10+00:00 |
| [Dataset card](dataset_card.md) | `86ec53c506` | False | 2026-10-28T09:00:40+00:00 |
| [Surrogate evaluation](surrogate_eval.md) | `4e93e88935` | False | 2026-10-28T13:13:18+00:00 |
| [ONNX benchmark](onnx_benchmark.md) | `49c9b40fa6` | False | 2026-10-28T16:01:16+00:00 |
| [Segmentation evaluation](seg_eval.md) | `226fc57e72` | False | 2026-10-28T14:46:21+00:00 |
| [Segmentation ONNX](seg_onnx.md) | `20d7f6b33c` | False | 2026-10-28T14:47:02+00:00 |
| [QCA synthetic validation](qca_synthetic.md) | `24acff05b5` | False | 2026-10-28T16:37:38+00:00 |
| [QCA gallery](qca_gallery.md) | `6b388bc15b` | False | 2026-10-28T14:58:01+00:00 |
| [Planner benchmark](planner_benchmark.md) | `8c89eb0b9d` | False | 2026-10-28T16:02:19+00:00 |
| [App latency](app_latency.md) | `7c79f1136a` | False | 2026-10-28T16:48:15+00:00 |
| [Demo](demo/demo.md) | `686bdf9e72` | False | 2026-10-28T16:37:43+00:00 |

This summary: commit `21ebb47ad22092cac9ef4b863ec9d7de8d7c01aa` (dirty: False), vstent 0.1.0, 2026-10-28T16:49:34+00:00.
