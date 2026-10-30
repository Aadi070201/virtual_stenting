# vstent MVP report

> **Research use only — not for clinical decisions.** This research prototype is not a medical device, is not validated and must never be used to make decisions about real patients.

Assembled by `vstent mvp-report` from the generated report files in `reports/` (no hand-typed numbers). Each line links back to its source report; the provenance table lists the commit each one was generated at.

## What the prototype does

Angiogram → vessel segmentation (U-Net, ONNX) → centreline and 2D QCA → FFR pullback from a reduced-order physics model and a learned surrogate (ONNX) → virtual stent with instant post-PCI FFR → plan optimiser (shortest generic-catalogue plan reaching the target FFR) → Streamlit app and case report.

## Acceptance targets: 23 of 23 met

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
| M2 | Synthetic data | cal: FFR < 0.80 / grey zone | 53.3% / 18.5% | ≥ 30% / ≥ 15% | yes |
| M3 | Surrogate (test split) | Distal FFR MAE | 0.0013 | ≤ 0.015 | yes |
| M3 | Surrogate (test split) | 95% limits of agreement | [-0.0057, +0.0056] | within ±0.04 | yes |
| M3 | Surrogate (test split) | Agreement at 0.8 | 99.76% | ≥ 97% | yes |
| M4 | Surrogate ONNX | Parity: max abs(FFR_torch - FFR_onnx) over 1,000 test vessels, all points | 5.4e-07 | < 0.0001 | yes |
| M4 | Surrogate ONNX | Single-vessel latency, median (end-to-end) | 1.53 ms | < 10 ms | yes |
| M4 | Surrogate ONNX | Batch of 1,024 vessels, median (end-to-end) | 187.7 ms | < 300 ms | yes |
| M5 | Segmentation | Mean val Dice (plan target; val also chose the model) | 0.8325 | ≥ 0.75 (stretch 0.8) | yes |
| M5 | Segmentation | Mean test Dice (held out) | 0.7925 | ≥ 0.75 (stretch 0.8) | yes |
| M5 | Segmentation ONNX | Parity max abs(prob torch − onnx) | 1.8e-05 | < 0.0001 | yes |
| M5 | QCA (synthetic tubes) | Median abs. diameter error (worst case) | 0.10 px | ≤ 0.7 px | yes |
| M5 | QCA (synthetic tubes) | MLD location error (worst case) | 1.9 px | ≤ 3 px | yes |
| M5 | QCA (synthetic tubes) | DS error (worst case) | 0.7 points | within ±7 points | yes |
| M6 | Planner | 5,000 candidates, surrogate backend (median) | 0.838 s | < 1 s | yes |
| M7 | App | Stent-control update, server time (max over modes) | 159 ms | < 300 ms | yes |

## Further results (no target)

- Synthetic dataset: 135,000 vessels generated in 7.7 s.
- Surrogate distal-FFR MAE: test 0.0013 vs gradient-boosting baseline 0.0342; out-of-distribution 0.0039 (no target).
- For context, the reduced-order solver computes 1,024 vessels in 18.1 ms, the ONNX surrogate in 188 ms: no speed-up is claimed.
- INT8 quantisation: distal MAE 0.0026 vs 0.0013, speed-up 0.99× → not kept.
- ARCADE gallery (qualitative): 20/20 images analysed end-to-end; point of maximum DS inside an annotated stenosis box in 8/20; 19 outside the surrogate's training range.
- Planner on 100 synthetic vessels: target reached 84 (solver) / 83 (surrogate); solver confirms 81/83 surrogate plans; same total length 80/83.
- App update medians: synthetic 66 ms, image 113 ms.
- Demo (synthetic tandem lesions, surrogate backend): distal FFR 0.705 → 0.976 (solver) with 3.25 × 12 mm at 14.5 mm + 3.25 × 12 mm at 43.0 mm; planning 0.67 s.

## Limitations

- Research prototype: not a medical device, not validated, never for decisions about real patients.
- Physics: steady, Newtonian, single-vessel reduced-order model (Poiseuille + expansion loss) without side branches; FFR is very sensitive to the assumed microvascular resistance and hyperemic velocity.
- The surrogate is trained and evaluated on synthetic vessels with labels from the same reduced-order model: its accuracy is agreement with that model, not with invasive FFR. It is not faster than the reduced-order solver; it validates the training → ONNX → app pipeline for Phase 2 3D-CFD labels.
- Segmentation: one public dataset (ARCADE) whose masks cover only the annotated SYNTAX segments; no external validation.
- QCA: single 2D projection (foreshortening, overlap), uncalibrated pixel size unless given, crossing vessels can look like narrowings; automatic paths run through branches.
- Planning: the stent only widens the lumen to its radius (no recoil, malapposition or side-branch effects); two-stent search is heuristic; overlapping stents are not modelled.
- No patient outcome, invasive FFR or clinical data were used anywhere.

## Phase 2 (status from docs/PLAN.md)

- P1: Uncertainty and grey-zone flagging (done)
- P2: Vessel trees with side branches (done)
- P3: Tree surrogate (graph network) + ONNX (done)
- P4: Clinical-validation readiness [needs human: partner, ethics approval] (done)
- P5: Projection geometry and two-view 3D reconstruction (not started)
- P6: 3D CFD labels and multi-fidelity training [needs human: data registration, compute] (not started)
- P7: Cine angiography (not started)
- P8: Sim-to-real angiograms and regulatory groundwork [needs human: regulatory review] (not started)

## Sources and provenance

| Report | Generated at commit | Dirty | Created (UTC) |
|---|---|---|---|
| [Physics checks](physics_checks.md) | `1c4c48900f` | False | 2026-10-26T11:25:10+00:00 |
| [Dataset card](dataset_card.md) | `d2f2cc3a2b` | False | 2026-10-30T15:21:51+00:00 |
| [Surrogate evaluation](surrogate_eval.md) | `4e93e88935` | False | 2026-10-28T13:13:18+00:00 |
| [ONNX benchmark](onnx_benchmark.md) | `49c9b40fa6` | False | 2026-10-28T16:01:16+00:00 |
| [Segmentation evaluation](seg_eval.md) | `e245936699` | False | 2026-10-30T15:57:00+00:00 |
| [Segmentation ONNX](seg_onnx.md) | `20d7f6b33c` | False | 2026-10-28T14:47:02+00:00 |
| [QCA synthetic validation](qca_synthetic.md) | `186b932015` | False | 2026-10-30T10:26:02+00:00 |
| [QCA gallery](qca_gallery.md) | `6b388bc15b` | False | 2026-10-28T14:58:01+00:00 |
| [Planner benchmark](planner_benchmark.md) | `c95b317ba6` | False | 2026-10-30T19:12:07+00:00 |
| [App latency](app_latency.md) | `d69b1543d7` | False | 2026-10-30T19:11:52+00:00 |
| [Demo](demo/demo.md) | `413b3d9304` | False | 2026-10-30T16:16:15+00:00 |

This summary: commit `bcffc45c6e1d749eff60b28493f3de2a32ff879c` (dirty: False), vstent 0.1.0, 2026-10-30T23:06:27+00:00.
