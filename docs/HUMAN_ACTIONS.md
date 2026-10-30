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
- **Re-measure on an idle machine.** Benchmarks are timing-sensitive. When the PC is free (no
  games, browser or chat busy), tell me and I re-run `vstent plan-benchmark` and
  `vstent app-latency` (about 5 minutes). The last planner run (1.345 s, fail) was taken under
  load.

## 2. Start early (months of lead time)

### A clinical partner and ethics approval (P4)

This is the most important item for both the research and the product: there is **no
comparison with invasive FFR yet**, so nobody can say how accurate the tool is.

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
- **Size:** published validations of angiography-derived FFR used on the order of 100–300
  vessels. P4 will produce the protocol and the sample-size calculation for your partner to
  review.
- **Rule:** no patient data ever enters this repository. The validation tool will run locally,
  on the partner's side if needed.

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
- **Data licences:**
  - ARCADE is CC0, so commercial use is fine.
  - Keep `docs/DATA.md` up to date for every new dataset.
- **Better segmentation data:** the tree work showed that the segmentation model rarely finds
  small side branches, because ARCADE does not annotate them. Annotated angiograms from a
  partner, or P8's synthetic angiograms, would improve this more than further physics work.

## What does *not* need you

P3 (tree surrogate), P5 (projection geometry on synthetic data) and the technical parts of P4
and P8 run on this laptop. I check in after each milestone.
