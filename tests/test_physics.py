"""Physics sanity tests for the reduced-order solver (PLAN.md, M1).

If a band test fails, look for an implementation bug; do not change the constants.
"""

import time

import numpy as np
import pytest

from vstent.physics.geometry import VesselProfile, cosine_lesion_profile, cosine_narrowing
from vstent.physics.solver import (
    PhysicsParams,
    expansion_coefficient,
    flow_from_resistance,
    solve_ffr,
    solve_ffr_batch,
    viscous_coefficient,
)

MM = 1e-3
PARAMS = PhysicsParams()


def _uniform(length, diameter, n=256):
    s = np.linspace(0.0, length, n)
    r = np.full(n, diameter / 2.0)
    return VesselProfile(s, r, r)


def _lesion(severity, lesion_length=10 * MM, length=60 * MM, diameter=3.0 * MM, n=256):
    return cosine_lesion_profile(length, diameter, lesion_length, severity, n=n)


# --- spec tests -------------------------------------------------------------------------


def test_uniform_tube_poiseuille_fixed_flow():
    length, radius = 50 * MM, 1.5 * MM
    res = solve_ffr(_uniform(length, 2 * radius), PARAMS, mode="fixed_velocity")
    expected = 8 * PARAMS.mu * length * res.q / (np.pi * radius**4)
    assert res.dp[-1] == pytest.approx(expected, rel=5e-3)


def test_healthy_vessel_ffr_near_one():
    res = solve_ffr(_uniform(50 * MM, 3.0 * MM), PARAMS)
    assert res.ffr_distal >= 0.95


def test_distal_ffr_strictly_decreases_with_stenosis():
    severities = np.arange(0.30, 0.901, 0.05)
    ffr = [solve_ffr(_lesion(sev), PARAMS).ffr_distal for sev in severities]
    assert np.all(np.diff(ffr) < 0)


@pytest.mark.parametrize("severity", [0.0, 0.3, 0.5, 0.7, 0.9])
def test_ffr_starts_at_one_and_never_increases(severity):
    res = solve_ffr(_lesion(severity), PARAMS)
    assert res.ffr[0] == 1.0
    assert np.all(np.diff(res.ffr) <= 0)


def test_ffr_monotone_on_noisy_multi_lesion_vessels():
    rng = np.random.default_rng(0)
    n, length = 256, 70 * MM
    s = np.linspace(0, length, n)
    r_ref = np.linspace(1.75, 1.25, n) * MM
    for _ in range(20):
        r = r_ref * cosine_narrowing(s, 20 * MM, 8 * MM, rng.uniform(0.2, 0.8))
        r *= cosine_narrowing(s, 50 * MM, 12 * MM, rng.uniform(0.2, 0.8))
        r *= 1 + 0.03 * rng.standard_normal(n)
        res = solve_ffr(VesselProfile(s, r, r_ref), PARAMS)
        assert res.ffr[0] == 1.0
        assert np.all(np.diff(res.ffr) <= 0)


@pytest.mark.parametrize(
    ("severity", "low", "high"),
    [(0.5, 0.84, 0.94), (0.7, 0.45, 0.70)],
)
def test_sanity_bands_3mm_60mm_10mm_cosine(severity, low, high):
    res = solve_ffr(_lesion(severity), PARAMS)
    assert low <= res.ffr_distal <= high


@pytest.mark.parametrize("severity", [0.3, 0.5, 0.7, 0.9])
def test_stenosed_flow_below_reference_and_root_valid(severity):
    res = solve_ffr(_lesion(severity), PARAMS)
    assert np.isfinite(res.q) and res.q > 0
    assert res.q < res.q_ref
    # q solves b·Q² + (a + R)·Q − (Pa − Pv) = 0
    residual = res.b[-1] * res.q**2 + (res.a[-1] + res.r_micro) * res.q - (res.pa - res.pv)
    assert abs(residual) < 1e-9 * (res.pa - res.pv)


def test_restored_lesion_matches_healthy_vessel():
    n, length = 256, 60 * MM
    s = np.linspace(0, length, n)
    r_ref = np.linspace(1.75, 1.35, n) * MM  # tapered reference
    lesion = cosine_narrowing(s, 30 * MM, 10 * MM, 0.7)
    stenosed = VesselProfile(s, r_ref * lesion, r_ref)
    restored_r = stenosed.r.copy()
    seg = (s >= 24 * MM) & (s <= 36 * MM)  # lesion plus 1 mm each side
    restored_r[seg] = np.maximum(restored_r[seg], r_ref[seg])
    healthy = solve_ffr(VesselProfile(s, r_ref, r_ref), PARAMS).ffr_distal
    restored = solve_ffr(stenosed.with_radius(restored_r), PARAMS).ffr_distal
    assert solve_ffr(stenosed, PARAMS).ffr_distal < 0.8
    assert abs(restored - healthy) < 0.01


# --- component tests --------------------------------------------------------------------


def test_viscous_coefficient_tapered_tube_matches_integral():
    n, length = 256, 60 * MM
    s = np.linspace(0, length, n)
    r0, r1 = 1.75 * MM, 1.25 * MM
    r = np.linspace(r0, r1, n)
    k = (r1 - r0) / length
    exact = 8 * PARAMS.mu / np.pi * (r0**-3 - r1**-3) / (3 * k)  # ∫ r⁻⁴ ds for linear r
    a = viscous_coefficient(r, s[1] - s[0], PARAMS.mu)
    assert a[0] == 0.0
    assert a[-1] == pytest.approx(exact, rel=1e-3)


def test_sudden_expansion_matches_borda_carnot():
    r = np.array([1.0, 1.0, 1.0, 2.0, 2.0]) * MM
    a1, a2 = np.pi * (1.0 * MM) ** 2, np.pi * (2.0 * MM) ** 2
    b = expansion_coefficient(r, PARAMS.rho, PARAMS.kt, PARAMS.expansion_threshold)
    assert b[-1] == pytest.approx(PARAMS.kt * PARAMS.rho / 2 * (1 / a1 - 1 / a2) ** 2)
    assert np.all(np.diff(b) >= 0)


def test_converging_and_small_expansions_are_lossless():
    taper = np.linspace(2.0, 1.0, 50) * MM
    assert np.all(expansion_coefficient(taper, PARAMS.rho, PARAMS.kt, 1.05) == 0)
    small_bump = np.array([1.0, 1.0, 1.02, 1.02]) * MM  # A ratio 1.0404 < 1.05
    assert np.all(expansion_coefficient(small_bump, PARAMS.rho, PARAMS.kt, 1.05) == 0)


def test_expansion_runs_are_kept_separate_across_batch_rows():
    # row 0 ends diverging and row 1 starts diverging: the runs must not merge
    r = np.array([[1.0, 0.5, 1.0], [0.5, 1.0, 1.0]]) * MM
    b = expansion_coefficient(r, PARAMS.rho, PARAMS.kt, 1.05)
    for i in range(2):
        single = expansion_coefficient(r[i], PARAMS.rho, PARAMS.kt, 1.05)
        np.testing.assert_allclose(b[i], single, rtol=1e-12)


def test_flow_root_without_expansion_is_linear():
    q = flow_from_resistance(np.array(2.0), np.array(0.0), np.array(3.0), np.array(10.0))
    assert q == pytest.approx(2.0)


def test_reference_vessel_carries_reference_flow():
    res = solve_ffr(_uniform(60 * MM, 3.0 * MM), PARAMS)
    assert res.q == pytest.approx(res.q_ref, rel=1e-12)
    assert not res.r_micro_clamped
    losses = res.loss_breakdown
    assert sum(losses.values()) == pytest.approx(res.pa - res.pv, rel=1e-12)


def test_fixed_velocity_overestimates_loss_in_severe_lesion():
    prof = _lesion(0.7)
    assert solve_ffr(prof, PARAMS, "fixed_velocity").ffr_distal < solve_ffr(prof, PARAMS).ffr_distal


def test_grid_convergence():
    coarse = solve_ffr(_lesion(0.7, n=256), PARAMS).ffr_distal
    fine = solve_ffr(_lesion(0.7, n=4096), PARAMS).ffr_distal
    assert abs(coarse - fine) < 0.005


def test_batch_matches_single_with_per_sample_parameters():
    profiles = [_lesion(sev) for sev in (0.0, 0.4, 0.6, 0.8)]
    pa = np.array([70.0, 90.0, 100.0, 110.0]) * 133.322
    pv = np.array([2.0, 5.0, 6.0, 8.0]) * 133.322
    v_hyp = np.array([0.25, 0.35, 0.40, 0.45])
    batch = solve_ffr_batch(
        np.stack([p.r for p in profiles]),
        profiles[0].ds,
        np.stack([p.r_ref for p in profiles]),
        PARAMS,
        pa=pa,
        pv=pv,
        v_hyp=v_hyp,
    )
    for i, prof in enumerate(profiles):
        params_i = PARAMS.model_copy(
            update={"pa_mmhg": pa[i] / 133.322, "pv_mmhg": pv[i] / 133.322, "v_hyp": v_hyp[i]}
        )
        single = solve_ffr(prof, params_i)
        np.testing.assert_allclose(batch.ffr[i], single.ffr, rtol=1e-12, atol=1e-14)


def test_invalid_inputs_rejected():
    prof = _lesion(0.5)
    with pytest.raises(ValueError):
        solve_ffr(prof, PARAMS, mode="nonsense")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        solve_ffr_batch(prof.r, prof.ds, prof.r_ref[:-1], PARAMS)
    with pytest.raises(ValueError):
        PhysicsParams(pa_mmhg=5.0, pv_mmhg=10.0)


@pytest.mark.slow
def test_batch_throughput():
    rng = np.random.default_rng(1)
    n_batch, n = 10_000, 256
    s = np.linspace(0, 60 * MM, n)
    r_ref = np.full((n_batch, n), 1.5 * MM)
    centers = rng.uniform(15, 45, n_batch)[:, None] * MM
    sev = rng.uniform(0.2, 0.9, n_batch)[:, None]
    r = r_ref * cosine_narrowing(s[None, :], centers, 10 * MM, sev)
    solve_ffr_batch(r[:100], s[1], r_ref[:100], PARAMS)  # warm-up
    t0 = time.perf_counter()
    solve_ffr_batch(r, s[1], r_ref, PARAMS)
    rate = n_batch / (time.perf_counter() - t0)
    assert rate >= 5_000
