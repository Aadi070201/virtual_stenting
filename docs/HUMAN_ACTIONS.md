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

- **Why:** software that informs whether and how to stent is a medical device. Preliminary
  classes in the drafts (`docs/regulatory/INTENDED_USE.md`; confirm all with an adviser):
  - **India (CDSCO, MDR 2017):** class C, under CDSCO's software guidance of 2026 (D with a
    diagnosis claim). Class C and D licences come from the central authority (CDSCO).
  - **EU (MDR, Rule 11):** likely IIb (III arguable).
  - **US:** class II, 510(k), with cleared angiography-derived FFR products as precedents.
- **Ready for them (P8, done 2026-11-01):** `docs/regulatory/`:
  - intended use and classification;
  - IEC 62304 safety class (proposed C);
  - development plan and architecture;
  - risk management plan;
  - 43 requirements and 20 hazards, traced to tests;
  - SOUP list and SBOM.
  Start with its README. The drafts are not approval.
- **Decide with them:**
  - the target market (India first?) and the intended-use wording ("aid" vs "diagnose", and
    whether the stent plan is part of it); this sets the class;
  - when to start a quality management system (ISO 13485). CDSCO asks domestic manufacturers
    for an undertaking of compliance with the Fifth Schedule of MDR 2017;
  - for a study in India: ethics committee approval, a test licence (Form MD-13) and
    permission for a clinical investigation (Form MD-23).
- **Also needed (people, not code):**
  - an independent reviewer of design, code and tests (class C; today there is one
    developer);
  - an interventional cardiologist for usability testing (IEC 62366-1).

## 3. Before P6 (3D CFD)

- **Register for the CT datasets:** ImageCAS and / or ASOCA (grand-challenge.org).
  - Accept their licences yourself, then tell me where the files are.
  - **Check whether the licence allows commercial use.** Non-commercial licences may allow
    research but not use in a product; confirm with your adviser.
- **Decide a compute budget for CFD:** hundreds of 3D simulations. The laptop can do a small
  pilot; a few hundred to a few thousand runs need a workstation or cloud CPUs. I will estimate
  the cost from a pilot before you commit money.

## 4. Real angiography runs (to confirm P5 and P7)

- **Two-view DICOM (confirms P5):** two runs of the same vessel from different angles (≥ 30°
  apart), with the positioner angles and distances in the DICOM header. The best option is a
  phantom of known size; otherwise a few de-identified runs from a partner. Only real data can
  confirm the angle convention and the C-arm distances (D49); the synthetic test cannot. A
  phantom of known lumen sizes (or IVUS / OCT reference diameters from a partner) would also
  verify the absolute lumen size (P5b), which today is checked on rendered vessels only.
- **Cine examples with ECG (confirms P7's end-diastolic frames):** a handful of
  de-identified, licensed cine DICOM runs whose header carries the ECG waveform or the R-wave
  pointer, for example from the clinical partner under the same agreement. P7 reads both, but
  only on synthetic DICOM so far; the image-based end-diastolic frames were checked against the
  truth on synthetic runs only (CADICA has no ECG).
- **Flow ground truth (before the measured velocity can become a default, P7):** the measured
  contrast flow velocity is accurate on synthetic beating hearts (two views: 5.8% median
  error), but untested on real flow. Options, cheapest first:
  - a flow phantom with a pump of known flow rate, imaged with contrast injections at a few
    flow rates (a university biomedical engineering lab, or the C-arm vendor's application
    team);
  - Doppler-wire (e.g. ComboWire) velocities recorded by a partner during the same runs;
  - the published FAVOR / QFR studies' authors, for de-identified frame-count data.
- **Freedom to operate for the measured velocity:** the FAVOR Pilot conversion and
  frame-count velocity are used by QFR products, which are patented (Pulse Medical, Medis).
  Ask the patent attorney before the measured velocity is offered commercially
  (`docs/INVENTION_NOTES.md` §7).

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

P5 is done on synthetic data (real two-view DICOM is still needed to confirm it). P7's synthetic part and the technical parts of P8
run on this laptop. P3 and P4's technical parts are done. I check in after each milestone.
