# Intended use and preliminary classification (draft)

> **Research use only — not for clinical decisions.** Draft regulatory groundwork for review by
> a qualified person; not regulatory advice. Version 0.1, 2026-11-01, not reviewed.

## 1. Status today

vstent is a **research prototype**. It is not a medical device, is not validated against
invasive pressure measurements and must never be used to make decisions about real patients
(requirement SR-01). Everything below describes the **intended use of a future product**. It
is written now so that requirements, hazards and validation are planned against it (ISO 14971
§5.2, IEC 62304 §4.3).

## 2. Proposed intended use (draft)

> vstent is standalone software that analyses X-ray coronary angiograms to:
>
> - measure the lumen dimensions of a coronary vessel segment the user selects (quantitative
>   coronary angiography, QCA);
> - estimate, from a computational flow model, the fractional flow reserve (FFR) along that
>   segment;
> - simulate how the estimated FFR changes after a virtual stent the user places.
>
> The results are intended to aid trained interventional cardiologists, together with the
> angiogram and other clinical information, in assessing the functional significance of
> coronary stenoses in adults with stable coronary artery disease undergoing invasive coronary
> angiography. The software does not replace the physician's judgement, does not select a
> device and is not intended for use by people who are not clinicians.

The wording follows the existing angiography-derived FFR products (section 6). Two choices
in it must be settled with the clinical and regulatory advisers:

- "aid in assessing" (supports a decision) versus "diagnose" (determines it); this sets the
  class in every framework below;
- whether the virtual-stent output is part of the intended use, or is offered only as
  physiology information without any sizing claim.

## 3. Users, patients and environment

| Item | Draft |
|---|---|
| Intended users | Interventional cardiologists trained in the software; cath-lab staff may prepare the analysis for review |
| Patient population | Adults with suspected or known stable coronary artery disease; native coronary arteries; reference diameters within the validated range (today's synthetic training range: `configs/synth_v1.yaml`) |
| Proposed exclusions (to be set with clinicians) | Bypass grafts; left main and ostial lesions; in-stent restenosis; the culprit vessel of an acute myocardial infarction; chronic total occlusions; severe microvascular disease; images with severe overlap, foreshortening or poor opacification |
| Use environment | Catheterisation laboratory or an offline review workstation; local software, no cloud service |
| Inputs | X-ray angiography images or cine runs (DICOM or exported frames); start and end points on the vessel; aortic pressure if measured |
| Outputs | Diameter profile, %DS, lesion length; FFR pullback curve and distal FFR with intervals and a grey-zone flag; post-stent FFR for a virtual stent; plain-language warnings; a case report |

## 4. Principle of operation

Segmentation of the vessels (a U-Net in ONNX Runtime) → centreline between the user's points
→ diameter profile from the image intensity (QCA) → FFR from a reduced-order flow model
(Poiseuille viscous loss plus an expansion loss; `docs/PHYSICS.md`) and from a learned
surrogate of that model → intervals from conformal calibration and Monte Carlo over the inputs
→ virtual stent and plan optimiser. Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

## 5. Reasonably foreseeable misuse (ISO 14971 §5.2)

- Using the FFR estimate **instead of** a pressure wire where guidelines call for one.
- Taking the suggested stent as the device size to implant (hazard H-09).
- Using it in excluded anatomy or populations (H-17).
- Reading the contrast-front speed as a flow measurement (H-12).
- Analysing an image of another patient or from another run (H-13).
- Use by people without training in angiography.

## 6. Preliminary classification

### Significance and situation (IMDRF framework, used by CDSCO)

CDSCO's software guidance classifies standalone software by the significance of its
information and the state of the healthcare situation (CDSCO/MD/GD/MDSW/01/2026, section 7.1,
Table 2), the IMDRF framework:

| Situation \ significance | Treatment or diagnosis | Drive clinical management | Inform clinical management |
|---|---|---|---|
| Critical | D | C | B |
| Serious | C | B | A |
| Non-serious | B | A | A |

- **Significance:** the FFR estimate is used "to aid in diagnosis" and "to guide next
  diagnostics or next treatment interventions" (revascularisation or not): **drive clinical
  management**, in the guidance's words. If the claim were to diagnose (section 2), it would
  be **treatment or diagnosis**.
- **Situation:** coronary artery disease can be life-threatening, the decision is about a
  major therapeutic intervention (PCI) and the users are specialised: the guidance's criteria
  for a **critical** situation (section 7.1(b)(i)).
- **Reading:** critical × drive clinical management → **class C**; with a diagnosis claim
  → **class D**.

### By jurisdiction

| Jurisdiction | Rule | Preliminary class | Notes |
|---|---|---|---|
| India (CDSCO, Medical Devices Rules 2017) | First Schedule; CDSCO software guidance Table 2 | **C** (D with a diagnosis claim) | Class C and D: licences from the Central Licensing Authority. Before marketing: a test licence (Form MD-13, applied for in MD-12) and permission for a clinical investigation (Form MD-23) (guidance sections 11–12). Check CDSCO's published list of classified software, or request a classification on the CDSCO MD Online portal. |
| European Union (MDR 2017/745) | Annex VIII, Rule 11 | **IIb** likely (III arguable) | Rule 11: software providing information used for diagnostic or therapeutic decisions is IIa, IIb if the decisions may cause a serious deterioration of health or a surgical intervention, III if they may cause death or an irreversible deterioration. A deferred revascularisation can end in myocardial infarction, so III must be argued against with the notified body. The machine-learning parts also make it a high-risk AI system under the AI Act (to confirm). |
| United States (FDA) | 21 CFR 870.1415, coronary vascular physiologic simulation software device | **Class II**, 510(k) | Precedents (FDA 510(k) database, found 2026-11-01; verify each before citing): CathWorks FFRangio K182149 (product code QEK, 2018-12-19); Medis QAngio XA 3D K182611 (QHA, 2019-05-30); Pie Medical CAAS Workstation with vFFR K232147 (QHA, 2024-04-09); QFR Solutions QFR K243769 (QHA, 2025-04-04). Software documentation level likely "Enhanced" (FDA guidance on the content of premarket submissions for device software functions, 2023). |

The classification decides the conformity route and the size of the clinical evidence. It is
for the regulatory adviser to confirm (HUMAN_ACTIONS.md).

## 7. What the intended use asks of the evidence

- **Clinical performance:** agreement of the FFR estimate with invasive FFR (sensitivity,
  specificity, AUC at 0.80, Bland–Altman) in the intended population, per vessel and per
  patient. Protocol template: `docs/validation/PROTOCOL.md`. Today: no paired data.
- **Technical performance:** QCA against reference lumens (rendered vessels: met; real
  angiograms against expert grades: severity compressed, `reports/stenosis_grading.md`);
  physics against bench and CFD data (`reports/physics_nozzle.md`; CFD: P6).
- **Usability:** IEC 62366-1 with interventional cardiologists (SR-42, not started).

## Sources

- CDSCO, *Guidance Document on Medical Device Software*, CDSCO/MD/GD/MDSW/01/2026:
  <https://www.cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Guidance-document-on-Medical-Device-Software-under-MDR-2017.pdf>
- FDA 510(k) database entries:
  [K182149](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/pmn.cfm?ID=K182149),
  [K182611](https://www.accessdata.fda.gov/cdrh_docs/pdf18/K182611.pdf),
  [K232147](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K232147);
  K243769 and the product codes from database mirrors
  ([QHA](https://beudamed.com/fda_classifications/QHA),
  [QEK](https://fda.innolitics.com/productcode/QEK.md)), to be checked against the FDA database.
- Regulation (EU) 2017/745, Annex VIII, Rule 11.
- IMDRF/SaMD WG/N12, *Possible framework for risk categorization and corresponding
  considerations* (2014).
