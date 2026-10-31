# What the project needs from you (the human)

> Research use only. Nothing here is legal, regulatory or medical advice; the points marked
> "confirm" need a qualified person.

Most of Phase 2 runs on this laptop without you. These items cannot be done by the code. They
are ordered by **how long they take to arrange**, because several take months and block later
milestones.

## 1. Now (minutes)

- **Push and merge.** `git push -u origin phase2`, then open a pull request `phase2 → master`
  on GitHub and merge it when you are happy (P1, P2 and the code-review fixes).
- **Keep the repository private.** It contains the invention notes (`docs/INVENTION_NOTES.md`).
- **Keep benchmarks on an idle machine.** Timing reports (`vstent plan-benchmark`,
  `vstent app-latency`) are only valid when the PC is free (no games, browser or chat busy).
  They were last re-measured idle on 2026-10-30 (planner 0.838 s, slowest app update 159 ms).
  Tell me when the PC is free if they need re-running.

## 2. Start early (months of lead time)

### Ground truth without a hospital partner (P4b, from 2026-10-31)

You told me a hospital partner is not possible now, so validation runs on **public data**, one
component at a time (`docs/PLAN.md`, P4b):
- stenosis grading against expert grades on real angiograms (CADICA, CC BY 4.0);
- the pressure physics against an FDA bench experiment (public domain).

What public data cannot do:
- **No paired data.** No openly downloadable dataset pairs angiograms with invasive FFR
  (searched 2026-10-31), so **agreement with FFR itself stays unmeasured**.
- **Approval still needs a clinical study.** Regulators (CDSCO in India, CE in the EU, FDA in
  the US) expect clinical performance data against the reference standard before a device like
  this can be sold. Public data builds the technical file and the pitch; the clinical study
  will still be needed.

**The one lead for real FFR ground truth: ICADD.** It is an Indian dataset from the group of
Prof. Santosh Ansumali (JNCASR, Bangalore): 70 patients with CT, invasive angiography and
invasive FFR (medRxiv 2023, doi:10.1101/2023.05.13.23289934). It has no public download.
- **Your action:** email the authors and request access for research. Ask:
  - which licence applies;
  - whether commercial use is allowed;
  - whether the angiograms are DICOM with the FFR wire position.
- **Note:** the same group works on CT-FFR, so they may also be a collaborator or a
  competitor; decide what you share.
- **With access:** `vstent validate` (P4) runs on it unchanged, locally.

### If a clinical partner becomes possible later (P4)

- **What:** an interventional cardiologist at a hospital that performs pressure-wire FFR (or
  iFR) and is willing to share **retrospective, de-identified** cases.
- **Approvals (confirm locally):**
  - the hospital's ethics committee / IRB;
  - a data-sharing or data-processing agreement;
  - data protection (in the EU: GDPR).
  These often take several months, so start early.
- **Data needed per vessel:**
  - the cine angiogram DICOM runs, ideally two projections ≥ 30° apart, with geometry tags;
  - the FFR value and where the wire sensor was (marked on a frame);
  - Pa and Pd if available;
  - nothing that identifies the patient.
- **Size:** `reports/validation_sample_size.md` computes it from planning assumptions. With
  the current **placeholder** assumptions it is 286 vessels in about 220 patients. A
  statistician must set the real assumptions.
- **Ready for your partner (P4):**
  - `docs/validation/PROTOCOL.md`, a protocol template to complete with the investigators;
  - `docs/validation/DATA_SPEC.md`, which covers the data needed, the DICOM attributes to keep,
    a de-identification checklist and the two CSV files.
- **The tool:** `vstent validate` runs locally, on the partner's side if needed. It locks the
  predictions before the wire values are opened.
- **Rule:** no patient data ever enters this repository; the tool refuses tracked locations.

### Your P4 checklist
1. Share `docs/validation/` with the clinical partner and a statistician.
2. With them, complete the protocol's bracketed items:
   - exclusions;
   - primary index test;
   - success criterion;
   - sample-size assumptions in `configs/validation.yaml`.

   Then re-run `vstent sample-size`.
3. Ethics approval and the data agreement at the partner site.
4. Register the protocol before the reference values are opened.
5. Tell me when de-identified data exists, and where. I never need to see it: the partner can
   run `vstent validate` on their machine and share only the report.

### A regulatory adviser (P8; also shapes the intended use now)

- **Why:** software that informs whether and how to stent is a medical device.
  - **EU (MDR, Rule 11):** such software is at least class IIa, and possibly IIb.
  - **US:** comparable angiography-derived FFR products have been cleared as class II devices.
  - Confirm both with an adviser.
- **Decide with them:**
  - the target market (EU / US first);
  - the intended-use statement;
  - when to start a quality management system (ISO 13485), with IEC 62304 (software) and
    ISO 14971 (risk).
- **What P8 provides:** draft documents (intended use, software safety class, development plan,
  SOUP list, hazard analysis) for them to review. Drafts are not approval.

## 3. Before P6 (3D CFD)

- **Register for the CT datasets:** ImageCAS and / or ASOCA (grand-challenge.org).
  - Accept their licences yourself, then tell me where the files are.
  - **Check whether the licence allows commercial use.** Non-commercial licences may allow
    research but not use in a product; confirm with your adviser.
- **Decide a compute budget for CFD:** hundreds of 3D simulations. The laptop can do a small
  pilot; a few hundred to a few thousand runs need a workstation or cloud CPUs. I will estimate
  the cost from a pilot before you commit money.

## 4. Before P7 (cine angiography)

- **Cine examples:** a handful of de-identified, licensed cine DICOM runs (ideally with ECG),
  for example from the clinical partner under the same agreement. Without them, P7 is built and
  tested on synthetic sequences only.

## 5. Company and IP (any time, before going public)

- **Patents:** talk to a patent attorney **before any public disclosure** (papers, preprints,
  conference talks, open demos, detailed investor decks). `docs/INVENTION_NOTES.md` lists the
  candidate ideas with dates and commits.
- **India-specific patent points (confirm with an Indian patent attorney):**
  - The Indian Patents Act excludes "a computer programme per se" and algorithms (section
    3(k)), and methods of diagnosis of humans (section 3(i)).
  - Claims for software that diagnoses therefore have to be drafted around a technical
    system and its technical effect. File where the claims are strongest, and use PCT for the
    global markets.
- **Data licences** (all recorded in `docs/DATA.md`):
  - ARCADE is CC0, and CADICA is CC BY 4.0 (cite it). The FDA nozzle data is public domain.
    Commercial use is fine for all three.
  - Datasets with non-commercial licences may be used for research only. I will flag each
    one before using it.
- **Better segmentation data:** the tree work showed that the segmentation model rarely finds
  small side branches, because ARCADE does not annotate them. Annotated angiograms from a
  partner, or P8's synthetic angiograms, would improve this more than further physics work.

## What does *not* need you

P5 (projection geometry on synthetic data), P7's synthetic part and the technical parts of P8
run on this laptop. P3 and P4's technical parts are done. I check in after each milestone.
