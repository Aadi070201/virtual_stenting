# Risk management plan (draft)

> **Research use only — not for clinical decisions.** Draft regulatory groundwork for review by
> a qualified person; not regulatory advice. Version 0.1, 2026-11-01, not reviewed.

ISO 14971:2019 §4.4, with ISO/TR 24971:2020 and IEC/TR 80002-1 for software.

## 1. Scope

- **Product:** vstent, standalone software ([INTENDED_USE.md](INTENDED_USE.md)), all items of
  the analysis path ([ARCHITECTURE.md](ARCHITECTURE.md)).
- **Life-cycle phases:** design and development now; release, use and post-market once a
  product exists.
- **Basis:** the intended clinical use, not today's research use. In research use no patient
  decision is made, so no harm can follow from the output; the research-use restriction
  (SR-01) is what keeps it so.

## 2. Responsibilities

| Role | Who | Today |
|---|---|---|
| Top management (sets the risk policy, ISO 14971 §4.2) | The founder | Policy drafted here, not yet approved |
| Risk analysis and controls | The developer | Drafts in this folder |
| Independent review | A qualified regulatory or risk professional, and an interventional cardiologist for the clinical hazards | **Not yet appointed** |

Most code and documents in this repository were written by the developer with an AI coding
assistant. A release needs a review of the design, code, tests and this file by a qualified
person who is not the author.

## 3. Risk acceptability (draft policy)

Defined in `docs/regulatory/hazards.yaml` and rendered in
[HAZARD_ANALYSIS.md](HAZARD_ANALYSIS.md):

- **Severity** in five levels, from negligible to death, described in clinical outcomes.
- **Probability** in five levels per use. One use is one analysed vessel.
- **Risk index** = severity × probability: ≤ 4 acceptable; 5–9 to be reduced as far as
  possible and weighed against the benefit; ≥ 10 unacceptable.

The probability is the probability of **harm**. It assumes the clinician acts on the output
(the probability that a hazardous situation leads to harm is set to 1) unless a reason is
written down. **Where nothing is measured, it is set conservatively high**; it is lowered
only by evidence from a generated report, which the hazard then cites. Estimated values say
so.

## 4. Method

1. **Intended use and foreseeable misuse** (§5.2): [INTENDED_USE.md](INTENDED_USE.md)
   sections 2 and 5.
2. **Safety-related characteristics** (§5.3): the ISO/TR 24971 Annex A questions, answered
   for this software (section 5).
3. **Hazards and hazardous situations** (§5.4), from four sources:
   - the clinical use (wrong FFR in either direction, wrong length or size, wrong vessel or
     patient);
   - software causes (IEC/TR 80002-1: wrong input, units, SOUP defects, model files,
     failures);
   - **measured weaknesses** in the generated reports (severity compression on real
     angiograms, the occlusion flag on real occlusions, the physics against the FDA nozzle,
     extrapolation on real images);
   - use errors (display, sizing, misuse).
4. **Risk estimation and evaluation** (§5.5, §6) with the policy above.
5. **Risk control** (§7.1), in this order: inherent safety by design (e.g. the solver as the
   reference, a transit that cannot enter the FFR), protective measures in the software
   (refusals, warnings, flags), then information for safety (labelling). Every control is a
   requirement in `requirements.yaml`, with its verification.
6. **Verification of controls** (§7.2): tests and generated reports. `vstent regulatory` fails
   if a control's requirement, test, report or code no longer exists; the test suite runs it.
7. **Residual risk** (§7.3) per hazard, after the controls, with its basis.
8. **Benefit–risk and overall residual risk** (§7.4, §8): not possible until clinical
   performance against invasive FFR is known. The comparison will be with angiography alone
   (the common practice when no wire is used) and with the pressure wire (the reference).
9. **Changes** (IEC 62304 §7.4): every change that touches an item updates `hazards.yaml`
   and `requirements.yaml` in the same commit.

## 5. Safety-related characteristics (ISO/TR 24971 Annex A, short form)

| Question | Answer |
|---|---|
| Intended use, users, environment | INTENDED_USE.md |
| Does it measure? | It estimates lumen dimensions (mm, %DS) and FFR (dimensionless); accuracy against references is in the reports |
| Is it interpretive? | Yes: FFR and plans are model estimates the user interprets |
| Does it control therapy? | No; it informs a decision on revascularisation and simulates a stent |
| Machine learning? | Segmentation (U-Net) and the FFR surrogate; both fixed (no learning during use) |
| Inputs from other devices? | Angiography images (DICOM); quality and geometry vary between X-ray systems |
| Data and security | De-identified images, processed locally; no network during analysis (H-15) |
| Time-critical? | Used during or after a procedure; updates within 300 ms, planning within 1 s (SR-29, SR-30) |
| User interface | Web app and CLI; no usability evaluation yet (SR-42, H-19) |

## 6. Production and post-production (§10, once a product exists)

- Complaints, field reports and literature on angiography-derived FFR.
- SOUP: published anomalies and vulnerabilities of every SBOM component
  ([sbom.cdx.json](sbom.cdx.json)), reviewed at each lock update and periodically.
- Model performance in use: drift of image characteristics; out-of-range warnings counted.
- Changes to models only under a predetermined change control plan (FDA) or algorithm change
  protocol (CDSCO software guidance §12.4).

## 7. Risk management review (§9)

Before any clinical investigation and before any release: the completeness of this plan, the
hazard analysis and the verification of every control, signed by the reviewer of section 2.
