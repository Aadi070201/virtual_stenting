# Physics: reduced-order FFR model

> **Research use only — not for clinical decisions.** This model is a research prototype and has
> not been validated against invasive FFR.

Code: [`src/vstent/physics/`](../src/vstent/physics/). Constants: [`configs/physics.yaml`](../configs/physics.yaml).
Tests: [`tests/test_physics.py`](../tests/test_physics.py). Generated checks: `reports/physics_checks.md`
(`vstent physics-report`).

## Scope

A single coronary vessel without side branches, described by its lumen radius r(s) along the
centreline, under steady hyperemic flow. The model returns the flow Q, the pressure loss ΔP(s)
and the FFR pullback curve FFR(s).

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

## Assumptions and limitations

- **No side branches.** Flow is conserved along the vessel; in reality flow drops at each branch,
  so distal losses are overestimated in branching vessels.
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

- Young DF, Tsai FY. Flow characteristics in models of arterial stenoses. I. Steady flow.
  *J Biomech* 1973;6:395–410 (origin of the Kt expansion-loss term).
- Tu S, et al. Diagnostic accuracy of fast computational approaches to derive fractional flow
  reserve from diagnostic coronary angiography: the international multicenter FAVOR pilot study.
  *JACC Cardiovasc Interv* 2016;9:2024–2035 (fixed-flow QFR).
