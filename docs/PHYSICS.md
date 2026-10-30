# Physics: reduced-order FFR model

> **Research use only — not for clinical decisions.** This model is a research prototype and has
> not been validated against invasive FFR.

Code: [`src/vstent/physics/`](../src/vstent/physics/). Constants: [`configs/physics.yaml`](../configs/physics.yaml).
Tests: [`tests/test_physics.py`](../tests/test_physics.py). Generated checks: `reports/physics_checks.md`
(`vstent physics-report`).

## Scope

A single coronary vessel, described by its lumen radius r(s) along the centreline, under steady
hyperemic flow. The model returns the flow Q, the pressure loss ΔP(s) and the FFR pullback curve
FFR(s). *Vessel trees* (P2, below) extend the same equations to a vessel with side branches.

## Units

SI throughout: m, m², m³/s, Pa, Pa·s, kg/m³. Pressures are entered in mmHg only in the config
file (1 mmHg = 133.322 Pa) and converted on load. Coefficients: a in Pa·s/m³, b in Pa·s²/m⁶,
R_micro in Pa·s/m³.

## Parameters (defaults)

| Symbol | Config key | Default | Meaning |
|---|---|---|---|
| μ | `mu` | 3.5e-3 Pa·s | blood viscosity (Newtonian) |
| ρ | `rho` | 1060 kg/m³ | blood density |
| Pa | `pa_mmhg` | 90 mmHg | mean aortic pressure |
| Pv | `pv_mmhg` | 5 mmHg | venous pressure |
| Kt | `kt` | 1.52 | expansion loss coefficient (Young & Tsai) |
| v_hyp | `v_hyp` | 0.35 m/s | hyperemic reference velocity (as in fixed-flow QFR) |
| — | `expansion_threshold` | 1.05 | ignore diverging runs with A_max/A_min below this |
| k | `murray_exponent` | 2.33 | trees: healthy flow split ∝ diameter^k at each junction |

## Equations

Input: N points s_i with uniform spacing Δs, radii r_i, reference radii r_ref,i.

1. **Area:** A_i = π r_i².

2. **Viscous (Poiseuille) coefficient**, cumulative, trapezoid rule:

   a(s_k) = Σ_{i<k} (8μ/π) · ½ (r_i⁻⁴ + r_{i+1}⁻⁴) · Δs,  a(s_0) = 0.

3. **Expansion (separation) losses.** A *diverging run* is a maximal sequence of consecutive
   segments with A_{i+1} ≥ A_i, running from a local minimum A_min to the next local maximum
   A_max. If A_max/A_min ≥ threshold, the run contributes

   b_run = Kt · ρ/2 · (1/A_min − 1/A_max)².

   b_run is spread over the run's segments in proportion to each segment's decrease of 1/A²
   (weights 1/A_i² − 1/A_{i+1}², which sum to 1/A_min² − 1/A_max²), so the pullback curve stays
   continuous. Converging sections are lossless. b(s) is the cumulative sum, b(s_0) = 0.

4. **Reference vessel.** r_ref(s) is the known reference for synthetic data; otherwise
   `estimate_reference_radius`: least-squares line through r(s), exclude points with
   r < 0.85 × fit, refit; at most 5 iterations, stopping early when the included set no longer
   changes.

5. **Microvascular resistance (calibration).** Q_ref = v_hyp · π r_ref(0)²;
   R_micro = (Pa − Pv)/Q_ref − a_ref(L), where a_ref is the viscous coefficient of the reference
   vessel. So the stenosis-free reference vessel carries exactly Q_ref. R_micro is floored at
   1% of (Pa − Pv)/Q_ref (see *Safeguards*).

6. **Flow** (resistance mode). Q is the positive root of

   b(L)·Q² + (a(L) + R_micro)·Q − (Pa − Pv) = 0,

   computed in the cancellation-free form Q = 2(Pa − Pv) / (c + √(c² + 4 b(L)(Pa − Pv))),
   c = a(L) + R_micro. This is exact when b(L) = 0 and always positive and finite.

7. **Pressure and FFR:** ΔP(s) = a(s)·Q + b(s)·Q²; Pd(s) = Pa − ΔP(s); FFR(s) = Pd(s)/Pa;
   `ffr_distal` = FFR(L). In resistance mode the losses add up exactly:
   a(L)Q + b(L)Q² + R_micro·Q = Pa − Pv.

8. **Flow modes.** `resistance` (default, steps 5–6) and `fixed_velocity` (Q = Q_ref, for
   comparison only). Fixed velocity ignores the flow reduction a stenosis causes, so it
   overestimates the pressure loss in severe lesions; for very severe lesions it gives FFR < 0.
   Those values are reported as computed, not clipped.

## Properties guaranteed by construction

- FFR(0) = 1 exactly (a(0) = b(0) = 0).
- FFR(s) is non-increasing: a and b are cumulative sums of non-negative terms and Q > 0.
- In resistance mode, FFR(L) = (Pv + R_micro·Q)/Pa > 0, and Q < Q_ref whenever r ≤ r_ref
  everywhere with r < r_ref somewhere.

## Implementation notes

- `solve_ffr_batch` (NumPy, shape (B, N)) is the single implementation; `solve_ffr` runs a
  batch of one. Per-sample Pa, Pv, v_hyp and Δs are supported for dataset generation.
- Diverging runs are found without Python loops: run starts are flagged, a global run id is the
  cumulative count of starts over the flattened (B, N−1) array (runs cannot merge across rows
  because each row's first diverging segment is a start), and per-run sums use `np.bincount`.
- Default resolution is N = 256. For a 70% DS, 10 mm lesion the distal FFR at N = 256 differs
  from N = 4096 by < 0.005 (test).

## Safeguards

- **R_micro floor.** If the reference vessel's own viscous loss at Q_ref would exceed Pa − Pv
  (very long, very thin vessels), R_micro would be ≤ 0. It is floored at 1% of (Pa − Pv)/Q_ref
  and the result is flagged `r_micro_clamped`. This is a numerical guard, not a tuned constant.
- Inputs are validated: radii finite and positive, uniform spacing, Pv < Pa.

## Vessel trees (P2)

Code: [`src/vstent/physics/tree.py`](../src/vstent/physics/tree.py). Tests:
[`tests/test_tree.py`](../tests/test_tree.py).

1. **Tree.** Branch 0 is the root; its inlet is at Pa. Every other branch k starts at sample
   t_k of its parent branch (inside the parent, never at its ends) and ends in an outlet that
   drains to Pv through an outlet resistance R_k. Each branch keeps one continuous radius
   profile with its own reference, and its own a(s) and b(s) from steps 2–3.

2. **Segments.** The take-offs cut each branch into segments. The flow in a segment is the sum
   of the outlet flows downstream of it, so it is constant along the segment and drops at each
   take-off. A segment from sample i to j with flow Q loses
   (a(s_j) − a(s_i))·Q + (b(s_j) − b(s_i))·Q². A diverging run that spans a take-off keeps its
   full b_run; each part is weighted by its own flow. With one flow, this is exactly step 7.

3. **Calibration (healthy tree).** Q_ref = v_hyp · π r_ref,root(0)². In the stenosis-free tree
   (r = r_ref on every branch), the flow arriving at a junction splits between the side
   branch(es) starting there and the continuing parent in proportion to d^k (Murray's law,
   k = `murray_exponent`). d is the side branch's reference diameter at its start, and the
   parent's one sample after the take-off. The healthy pressures, with viscous loss only as in
   step 5, give R_j = (P_j − Pv)/Q_j at every outlet j, floored at 1% of (Pa − Pv)/Q_j. With
   one branch this is exactly step 5.

4. **Solve.** The unknowns are the outlet flows q. Every outlet satisfies
   Σ_{e on its path} (A_e Q_e + B_e Q_e²) + R_j q_j = Pa − Pv. Here A_e and B_e are the
   segment's increases of a and b, and Q = W q with W_ej = 1 if segment e feeds outlet j.
   These equations are the gradient of a strictly convex function of q, so the solution is
   unique. The Jacobian Wᵀ diag(A + 2BQ) W + diag(R) is symmetric positive definite. Newton
   starts from the linear solution (B = 0) and stops when the relative change of q is below
   1e-13 (about 5 iterations). Steps are halved only if an outlet flow would become
   non-positive.

5. **Pressure and FFR.** The pressure is integrated down each branch from its take-off
   pressure; FFR = P/Pa at every sample of every branch.

**Checked by tests:**
- a one-branch tree reproduces the single-vessel solver to 1e-12;
- flow is conserved and pressure is continuous at every junction;
- every outlet satisfies P − Pv = R·q;
- the healthy tree splits flow by Murray's law;
- a side branch upstream of a lesion raises distal main-vessel FFR;
- a side branch carrying almost no flow, taking off inside a lesion's expansion run, leaves the
  main vessel's FFR unchanged (within 2e-4).

**Additional limitations of trees:**
- Junction losses (flow division at bifurcations) are neglected.
- Murray's law sets the healthy flow split; real territories and microvascular resistances
  differ.
- A side branch's ostium starts at its parent's pressure, with no entrance loss.

## Assumptions and limitations

- **No side branches (single-vessel model).** Flow is conserved along the vessel; in reality flow
  drops at each branch, so distal losses are overestimated in branching vessels. The tree model
  (P2) removes this where the side branches are known.
- **Steady flow.** No pulsatility, no unsteady inertia, a single mean Pa.
- **Newtonian blood**, constant viscosity; rigid walls; no collateral flow.
- **2D-derived geometry.** Circular cross-sections are assumed; diameters come from a single
  angiographic projection, with foreshortening and eccentric lesions not corrected.
- **No entrance-length effects.** Poiseuille flow is assumed fully developed everywhere, which
  underestimates viscous losses just downstream of a narrowing.
- **Expansion loss depends only on A_min and A_max.** Lesion length and shape affect FFR only
  through the viscous term, so for severe lesions FFR is fairly insensitive to lesion length
  (see `reports/figures/ffr_vs_ds.png`). Converging (entrance) losses are neglected.
- **Noise.** Measurement noise can split one expansion into several diverging runs. The 1.05
  threshold suppresses tiny runs, but a split expansion yields a lower total loss than the
  unsplit one, because the sum of squares is less than the square of the sum.
- **Microvascular function.** All vessels share one hyperemic velocity v_hyp at the reference;
  microvascular disease is represented only through the sampled range of v_hyp (M2).

## References

- Murray CD. The physiological principle of minimum work. I. The vascular system and the cost of
  blood volume. *Proc Natl Acad Sci USA* 1926;12:207–214 (flow ∝ diameter³).
- Huo Y, Kassab GS. Intraspecific scaling laws of vascular trees. *J R Soc Interface*
  2012;9:190–200 (flow ∝ diameter^(7/3) ≈ 2.33, the default exponent here).

- Young DF, Tsai FY. Flow characteristics in models of arterial stenoses. I. Steady flow.
  *J Biomech* 1973;6:395–410 (origin of the Kt expansion-loss term).
- Tu S, et al. Diagnostic accuracy of fast computational approaches to derive fractional flow
  reserve from diagnostic coronary angiography: the international multicenter FAVOR pilot study.
  *JACC Cardiovasc Interv* 2016;9:2024–2035 (fixed-flow QFR).
