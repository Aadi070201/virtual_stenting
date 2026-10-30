# Invention notes

> **Research use only — not for clinical decisions.**
>
> **What this file is:** a dated technical record of ideas in this repository that *might* be
> new, kept as a starting point for a conversation with a patent attorney. **It is not a patent
> draft, not a novelty or freedom-to-operate opinion, and not legal advice.** No prior-art
> search has been done; each section lists the prior art that most obviously needs checking,
> and several of these ideas may well turn out to be known.
>
> **Disclosure caution:** in many jurisdictions (e.g. under the European Patent Convention)
> any public disclosure before filing (a public repository, a paper, a talk, a demo) can
> destroy novelty; others (e.g. the US) have only a limited grace period. Keep this repository
> private and get advice before showing it publicly.

Dates and commits below come from the git history of this repository. Evidence points to the
generated reports in `reports/`.

---

## 1. Monotone-increment surrogate head for FFR pullback curves

- **First implemented:** 2026-10-28, commit `a565305` (`src/vstent/surrogate/model.py`,
  `monotone_ffr_head`).
- **What it is:** a 1D U-Net predicts, for every centreline segment, an unconstrained value
  *z*. The head turns these into non-negative pressure-drop increments
  softplus(*z*) / (N − 1) and accumulates them, so that
  FFR(s₀) = 1 and FFR(s_{i+1}) ≤ FFR(s_i) **by construction**: a pullback curve that starts at
  1 and never increases, as physics requires for a vessel without pressure recovery.
  Conditioning on whole-vessel quantities (length, reference size, Pa, Pv, microvascular
  resistance) enters through FiLM layers and a pooled bottleneck summary.
- **Why it may matter:** the curve shape is guaranteed rather than learned, which removes a
  class of non-physical outputs; in planning, the distal value is consistent with the whole
  curve. On the synthetic test split it reproduces the reduced-order model with a distal MAE
  well inside the target (see `reports/surrogate_eval.md`).
- **Known gap / possible extension:** the lower physical bound FFR ≥ Pv / Pa is *not* enforced;
  the surrogate went below it once when extrapolating far outside its training range
  (`reports/qca_gallery.md`; the pipeline now warns). A bounded variant, e.g.
  FFR = Pv/Pa + (1 − Pv/Pa) · exp(−cumsum(softplus(z))), would enforce both bounds.
- **Prior art to check:** monotonic / shape-constrained neural networks; physics-informed
  networks for haemodynamics; machine-learning FFR from CT and angiography (e.g. published
  ML-based CT-FFR work and commercial angiography-derived FFR products); cumulative-sum output
  layers for monotone sequences in other fields.

## 2. Constrained virtual-stent plan optimiser with a learned evaluator and solver re-check

- **First implemented:** 2026-10-28, commit `49c9b40` (`src/vstent/planning/optimizer.py`,
  `src/vstent/planning/stent.py`).
- **What it is:** an enumerate-and-filter optimiser over a generic catalogue.
  - **Search:** diameter × length × start position on a fixed grid, restricted to stents that
    cover a lesion's minimal lumen diameter.
  - **Constraints, vectorised:**
    - the stent lies inside the vessel;
    - nominal diameter ≤ a factor × the reference diameter at the distal stent end;
    - the maximum diameter stenosis over both landing zones stays below a threshold.
  - **Objective (lexicographic):** the shortest total stent length that reaches the target
    distal FFR, then the highest FFR; if no plan reaches the target, the highest FFR.
  - **Two stents:** a pool of the best candidates *per stent length* is kept for each lesion,
    so short two-stent combinations are not crowded out by long single stents (this was an
    observed failure mode of FFR-only pools). Pairs without overlap are evaluated.
  - **Evaluation:** every candidate is scored in one batch by the learned surrogate (ONNX),
    with the microvascular resistance, Pa and Pv fixed at their pre-PCI values. The proposed
    plans are then **re-checked with the physics solver**, and both values are reported.
  - **Output:** the top plans and a Pareto front (total length vs post-PCI FFR).
- **Why it may matter:** it combines a fast learned evaluator with a physics re-check of only
  the few returned plans, and returns an interpretable, constraint-respecting short list.
  Evidence: `reports/planner_benchmark.md` (throughput, agreement between surrogate-chosen
  and solver-chosen plans).
- **Prior art to check:** virtual PCI / virtual stenting tools for CT-FFR and
  angiography-derived FFR (commercial planners and academic papers); stent sizing and
  landing-zone rules in QCA guidelines; surrogate-assisted optimisation with high-fidelity
  verification in engineering design.

## 3. Two-sided robust reference diameter with capping for 2D QCA before physics

- **First implemented:** 2026-10-28, commit `accebf7` (`src/vstent/imaging/qca.py`,
  `robust_reference`, `qca_from_centerline`).
- **What it is:** along a path traced on a 2D angiogram, diameters are measured on the
  segmentation probability map. Then:
  - **One-sided branch rule:** when one edge is far farther than the other, a side branch is
    assumed and twice the nearer edge distance is used.
  - **Reference diameter:** a Theil–Sen line refined on the points within a band of
    [0.85, 1.15] × the line, so that *both* narrowings (below) and overlaps or branches
    (above) are excluded.
  - **Capping:** diameters above the upper band are capped **before** the profile goes to the
    pressure-loss model. Otherwise a spike from an overlapping vessel would produce a fictitious
    narrowing-plus-expansion loss.
- **Why it may matter:** it makes a physics model usable on raw 2D segmentations with overlaps;
  on one ARCADE image it changed the reference from an implausible value to a plausible one
  (see `docs/DECISIONS.md` D21).
- **Prior art to check:** conventional QCA reference methods (interpolated and
  regression-based references in commercial QCA software); robust regression in vessel
  analysis. **This element is the most likely to be known.**

## 4. Planned (not implemented): sim-to-real training for angiogram → FFR

- **Recorded:** 2026-10-28 (Phase 2 backlog in `docs/PLAN.md`; nothing implemented).
- **Idea:**
  - Generate 3D coronary trees with steady-flow CFD labels.
  - Render synthetic angiograms from them (projection, contrast, noise, overlap).
  - Train models that go from angiogram (or angiogram-derived profiles) to FFR, using the
    reduced-order labels for pre-training and the CFD labels for fine-tuning (multi-fidelity).
  - Keep the same feature pipeline and ONNX deployment path that this MVP validates.
- **Status:** an idea only; record its date, and do not describe it publicly before advice.
- **Prior art to check:** synthetic angiogram rendering, domain randomisation and sim-to-real
  in medical imaging; multi-fidelity surrogate modelling for CFD.

---

## Not considered inventive (engineering choices)

The following are recorded only so they are not mistaken for claims:
- shared pre-processing for training, ONNX and the app;
- chunked, parallel ONNX inference;
- resumable checkpoints;
- report provenance;
- the Streamlit app structure.
