# Validation study: sample size

> Research use only — not for clinical decisions. Planning numbers for the protocol template (`docs/validation/PROTOCOL.md`), not a statistical analysis plan.

**The assumptions below are placeholders.** They must be set with a statistician from published angiography-derived FFR studies and the partner site's case mix before the protocol is submitted. The numbers only follow from them (`vstent sample-size`, `configs/validation.yaml`).

## Assumptions

| Quantity | Value |
|---|---|
| Two-sided α; power | 0.05; 0.8 |
| Prevalence of wire FFR ≤ 0.80 (per vessel) | 0.35 |
| Expected sensitivity; specificity | 0.85; 0.85 |
| Precision d (half-width of their 95% interval) | ± 0.075 |
| Expected AUC; null AUC | 0.9; 0.8 |
| SD of (index − wire FFR) | 0.07 |
| Precision w of each limit of agreement | ± 0.03 |
| Vessels per patient m; intra-patient correlation (ICC) | 1.3; 0.1 |
| Vessels that cannot be analysed | 10% |

## Vessels needed per endpoint

| Endpoint | Method | Vessels |
|---|---|---:|
| Sensitivity to ± d | Buderer 1996 | 249 |
| Specificity to ± d | Buderer 1996 | 134 |
| AUC above the null value | Hanley & McNeil 1982 variance | 166 |
| Limits of agreement to ± w | Bland & Altman 1986 | 63 |

Governing endpoint: **Sensitivity to ± d** (249 vessels). With the design effect 1 + (m − 1) · ICC = 1.030: 257 vessels; allowing 10% that cannot be analysed: **286 vessels in about 220 patients**.

## How the total changes with the assumptions

Other assumptions as above; sensitivity = specificity.

| Prevalence | Sens. = spec. | Expected AUC | Governing endpoint | Vessels | Patients |
|---:|---:|---:|---|---:|---:|
| 0.25 | 0.80 | 0.85 | AUC above the null value | 1082 | 833 |
| 0.25 | 0.85 | 0.90 | Sensitivity to ± d | 400 | 308 |
| 0.25 | 0.90 | 0.93 | Sensitivity to ± d | 283 | 218 |
| 0.35 | 0.80 | 0.85 | AUC above the null value | 825 | 635 |
| 0.35 | 0.85 | 0.90 | Sensitivity to ± d | 286 | 220 |
| 0.35 | 0.90 | 0.93 | Sensitivity to ± d | 203 | 157 |
| 0.45 | 0.80 | 0.85 | AUC above the null value | 694 | 534 |
| 0.45 | 0.85 | 0.90 | Sensitivity to ± d | 223 | 172 |
| 0.45 | 0.90 | 0.93 | Sensitivity to ± d | 158 | 122 |

Per-vessel formulas assume independent vessels; the design effect is a simple correction for patients with several vessels. A statistician may prefer a clustered method (e.g. Obuchowski 1997 for the AUC).

Generated from commit `0aab188f7217ca7a25139f106f42a1b3f124227b` (dirty: False).
