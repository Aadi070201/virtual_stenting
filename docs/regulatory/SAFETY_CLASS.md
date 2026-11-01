# IEC 62304 software safety class (draft)

> **Research use only — not for clinical decisions.** Draft regulatory groundwork for review by
> a qualified person; not regulatory advice. Version 0.1, 2026-11-01, not reviewed.

**Proposed class: C** for the whole software system, for the intended use in
[INTENDED_USE.md](INTENDED_USE.md).

## 1. The rule (IEC 62304:2006 + A1:2015, §4.3)

- **Class A:** the software cannot contribute to a hazardous situation, or the resulting risk
  is acceptable after risk control measures **external** to the software.
- **Class B:** it can contribute to a hazardous situation whose risk is unacceptable after
  external risk controls, and the possible harm is a non-serious injury.
- **Class C:** as B, and the possible harm is death or serious injury.

## 2. Analysis

1. **Can the software contribute to a hazardous situation?** Yes. A wrong FFR estimate,
   diameter or lesion length can lead to a deferred revascularisation (H-01, H-03, H-04, H-06)
   or an unnecessary one (H-02), and a wrong plan to a mis-sized stent (H-09)
   ([HAZARD_ANALYSIS.md](HAZARD_ANALYSIS.md)).
2. **The worst possible harm:** myocardial infarction after a deferred revascularisation;
   dissection, perforation or peri-procedural infarction after an unnecessary one. Both are
   serious injuries; death cannot be excluded.
3. **Risk controls outside the software,** and how much they can be credited:
   - *The physician's judgement with the angiogram and the clinical picture.* Weak for this
     hazard: FFR is estimated precisely because the angiogram alone does not show the lesion's
     physiological significance, so a wrong estimate cannot be recognised by looking.
   - *An invasive pressure wire in the same room.* A strong control if the instructions
     require it for every estimate near the threshold; it is not credited because that would
     narrow the intended use.
   - *Guidelines and heart-team review.* Procedural, not specific to a wrong estimate.
4. **Result:** after the external controls the risk of H-01 and H-02 is not shown to be
   acceptable, and the possible harm is serious: **class C**.

**An argument for class B,** for the reviewer: if the intended use limited the software to
"inform" (for example triage before a pressure-wire measurement that is always made), the
external control would be the wire itself. This is not proposed now, because it removes the
product's main benefit (avoiding the wire).

## 3. Software items and segregation

The released software is one Python process. Its items ([ARCHITECTURE.md](ARCHITECTURE.md))
are not segregated from each other (no separate processes, memory protection or independent
checks of one item by another), so by §4.3 every item takes the system's class C.

Not part of the medical device software, and therefore not classified: the training, export,
evaluation, benchmark, data, validation, report and documentation tools (the "tools" rows of
ARCHITECTURE.md, section 3). They produce the released model files and the evidence, so
their validation falls under the quality system (ISO 13485 §4.1.6), and each released model
file is verified as an artefact (ONNX parity, evaluation reports). A release should ship the
analysis path without them (DEVELOPMENT_PLAN.md, section 9).

## 4. What class C requires, and where the repository stands

| Activity | IEC 62304 | Required for C | Today |
|---|---|---|---|
| Development plan | §5.1 | yes | [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md), draft |
| Requirements, including risk-control requirements | §5.2 | yes | [REQUIREMENTS.md](REQUIREMENTS.md), 42 requirements, references checked |
| Architecture; SOUP requirements; segregation | §5.3 | yes | [ARCHITECTURE.md](ARCHITECTURE.md), [SOUP.md](SOUP.md); no segregation |
| Detailed design of every unit and its interfaces | §5.4.2–5.4.4 | C only | **Gap:** docstrings state units and behaviour; no separate detailed design or design review |
| Unit verification with documented acceptance criteria | §5.5.2–5.5.5 | yes (extra criteria for C) | Partly: 305 automated tests; acceptance criteria per milestone in PLAN.md; §5.5.4's additional criteria (event sequence, data and control flow, resource allocation, fault handling, initialisation, self-diagnostics, memory, boundary conditions) not reviewed unit by unit |
| Integration and system testing | §5.6–5.7 | yes | Partly: pipeline, CLI and app tests; generated reports; no system test plan or test records per release |
| Release with known residual anomalies | §5.8 | yes | **Gap:** tag v0.1.0 exists; no release record or anomaly list |
| Maintenance plan | §6 | yes | Draft in DEVELOPMENT_PLAN.md |
| Software risk management, published SOUP anomalies | §7, §7.1.3 | yes | [HAZARD_ANALYSIS.md](HAZARD_ANALYSIS.md); **gap:** SOUP anomaly lists not reviewed |
| Configuration management, SOUP identified | §8 | yes | git, uv.lock, SBOM ([sbom.cdx.json](sbom.cdx.json)); models outside git (gap for a release) |
| Problem resolution | §9 | yes | **Gap:** no issue tracker with safety assessment |
