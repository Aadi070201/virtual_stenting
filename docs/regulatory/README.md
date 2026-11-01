# Regulatory groundwork (drafts)

> **Research use only — not for clinical decisions.** These are drafts for review by a
> qualified regulatory professional; they are not regulatory advice and have not been
> reviewed. vstent is a research prototype, not a medical device.

Version 0.1, 2026-11-01 (P8). The documents a hospital, a device company or a regulator asks
for first, written for the intended use of a future product and kept true to the code.

## Documents

| Document | Standard | Kind |
|---|---|---|
| [INTENDED_USE.md](INTENDED_USE.md): intended use, users, misuse, preliminary class (India, EU, US) | ISO 14971 §5.2; CDSCO MDR 2017; EU MDR Rule 11; 21 CFR 870.1415 | written |
| [SAFETY_CLASS.md](SAFETY_CLASS.md): software safety class C and what it requires | IEC 62304 §4.3 | written |
| [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md): life cycle, configuration, verification, release, maintenance | IEC 62304 §5.1, §6, §8, §9 | written |
| [ARCHITECTURE.md](ARCHITECTURE.md): software items, interfaces, data flow | IEC 62304 §5.3 | written |
| [RISK_MANAGEMENT_PLAN.md](RISK_MANAGEMENT_PLAN.md): scope, policy, method | ISO 14971 §4.4 | written |
| [REQUIREMENTS.md](REQUIREMENTS.md): 42 software requirements with code and verification | IEC 62304 §5.2 | generated |
| [HAZARD_ANALYSIS.md](HAZARD_ANALYSIS.md): 19 hazards, risk before and after the controls | ISO 14971 §5–§7 | generated |
| [TRACEABILITY.md](TRACEABILITY.md): hazard → control → test; requirement → code → test | ISO 14971 §7.2; IEC 62304 §7.3 | generated |
| [SOUP.md](SOUP.md) and [sbom.cdx.json](sbom.cdx.json): third-party software, CycloneDX SBOM | IEC 62304 §5.3.3, §8.1.2 | generated |

**Generated documents are checked.** They come from `requirements.yaml`, `hazards.yaml`,
`soup.yaml` and `uv.lock` (`uv run vstent regulatory`). The test suite fails if:
- a referenced file, function, test or report no longer exists;
- a hazard's control is not a requirement;
- the code imports a third-party package that is not declared and listed;
- a generated document differs from its sources.

So the trace cannot silently go out of date. To change them, edit the YAML and run
`vstent regulatory`.

## Main conclusions

1. **Intended use (proposed):** aid trained interventional cardiologists in assessing the
   functional significance of coronary stenoses from X-ray angiograms (QCA, FFR estimate,
   virtual stent).
2. **Preliminary class:**
   - **India: C** (critical situation × drive clinical management, CDSCO software guidance
     2026; D with a diagnosis claim);
   - **EU: IIb** likely (Rule 11; III arguable);
   - **US: class II, 510(k)** under 21 CFR 870.1415, with angiography-derived FFR products as
     precedents (FFRangio, QAngio XA 3D, CAAS vFFR, QFR).
3. **IEC 62304 class C:** a wrong estimate can contribute to serious injury, and the
   physician cannot detect it by looking at the angiogram.
4. **Residual risk:**
   - after today's controls, 13 of 19 hazards are still unacceptable for clinical use;
   - most are set by measured weaknesses: severity compression on real angiograms, the
     occlusion flag on real occlusions, a single projection, the physics' own error;
   - this is why the software is research use only, and each hazard names the evidence that
     would lower it.
5. **Strengths a reviewer will look for:**
   - 39 of 42 requirements verified by tests or generated reports;
   - every number reproducible from code;
   - pre-registered methods on held-out public data;
   - an SBOM from the lock file;
   - patient data kept out by design.

## Before a clinical investigation or a submission

What the code cannot do, ordered by lead time (details in `docs/HUMAN_ACTIONS.md`):

1. **A regulatory adviser** to review these drafts and confirm the class and route per
   market.
2. **A quality management system** (ISO 13485): document control, design control, supplier
   control, tool validation, complaint handling. CDSCO asks domestic manufacturers for an
   undertaking of compliance with the Fifth Schedule of MDR 2017.
3. **Clinical evidence against invasive FFR:**
   - ICADD, or a study under `docs/validation/PROTOCOL.md`;
   - in India: ethics committee approval and CDSCO permission for a clinical investigation
     (Form MD-23), plus a test licence (MD-13).
4. **Independent review** of design, code and tests (single developer today).
5. **Usability engineering** (IEC 62366-1) and **security activities** (IEC 81001-5-1), with
   interventional cardiologists and a threat model.
6. **Model change control:** a predetermined change control plan (FDA) or algorithm change
   protocol (CDSCO).

## Sources

The standards cited (IEC 62304:2006+A1:2015, ISO 14971:2019, ISO/TR 24971:2020, IEC/TR
80002-1, IEC 62366-1, IEC 81001-5-1, ISO 13485:2016) are paid documents and are referred to
by clause, not quoted. Regulatory sources with links are listed in
[INTENDED_USE.md](INTENDED_USE.md).
