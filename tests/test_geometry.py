import numpy as np
import pytest

from vstent.physics.geometry import (
    VesselProfile,
    cosine_lesion_profile,
    cosine_narrowing,
    estimate_reference_radius,
    find_lesions,
    minimal_lumen_diameter,
    percent_diameter_stenosis,
)

MM = 1e-3


def test_profile_validation():
    s = np.linspace(0, 10 * MM, 11)
    r = np.full(11, 1.5 * MM)
    VesselProfile(s, r)
    with pytest.raises(ValueError, match="uniformly"):
        VesselProfile(s**2, r)
    with pytest.raises(ValueError, match="increasing"):
        VesselProfile(s[::-1], r)
    with pytest.raises(ValueError, match="positive"):
        VesselProfile(s, -r)
    with pytest.raises(ValueError, match="shape"):
        VesselProfile(s, r[:-1])
    with pytest.raises(ValueError, match="shape"):
        VesselProfile(s, r, r[:-1])


def test_profile_arrays_are_readonly_copies():
    s = np.linspace(0, 10 * MM, 11)
    r = np.full(11, 1.5 * MM)
    prof = VesselProfile(s, r)
    r[0] = 99.0
    assert prof.r[0] == 1.5 * MM
    with pytest.raises(ValueError):
        prof.r[0] = 1.0


def test_basic_properties_and_resample():
    s = np.linspace(0, 50 * MM, 101)
    r = np.linspace(2.0, 1.5, 101) * MM
    prof = VesselProfile(s, r, r)
    assert prof.length == pytest.approx(50 * MM)
    assert prof.ds == pytest.approx(0.5 * MM)
    np.testing.assert_allclose(prof.area, np.pi * r**2)
    res = prof.resample(256)
    assert res.n == 256
    assert res.s[0] == 0.0 and res.s[-1] == pytest.approx(50 * MM)
    np.testing.assert_allclose(res.r, np.linspace(2.0, 1.5, 256) * MM, rtol=1e-12)
    np.testing.assert_allclose(res.r_ref, res.r)


def test_reference_estimate_recovers_taper_through_lesion():
    n, length = 256, 60 * MM
    s = np.linspace(0, length, n)
    r_true = np.linspace(1.75, 1.40, n) * MM
    r = r_true * cosine_narrowing(s, 30 * MM, 10 * MM, 0.6)
    r_est = estimate_reference_radius(s, r)
    assert np.max(np.abs(r_est / r_true - 1)) < 0.02


def test_unknown_reference_uses_estimate():
    prof = cosine_lesion_profile(60 * MM, 3.0 * MM, 10 * MM, 0.6)
    unknown = VesselProfile(prof.s, prof.r)
    np.testing.assert_allclose(unknown.reference_radius(), prof.r_ref, rtol=0.02)


def test_percent_stenosis_at_throat():
    prof = cosine_lesion_profile(60 * MM, 3.0 * MM, 10 * MM, 0.6, n=257)  # throat on a grid point
    ds_pct = prof.percent_stenosis()
    assert ds_pct.max() == pytest.approx(60.0)
    assert percent_diameter_stenosis(1.0, 2.0) == pytest.approx(50.0)


def test_find_lesions_tandem():
    n, length = 401, 80 * MM
    s = np.linspace(0, length, n)
    r_ref = np.full(n, 1.5 * MM)
    r = r_ref * cosine_narrowing(s, 20 * MM, 10 * MM, 0.5)
    r *= cosine_narrowing(s, 55 * MM, 15 * MM, 0.7)
    lesions = find_lesions(s, r, r_ref)
    assert len(lesions) == 2
    first, second = lesions
    assert first.s_mld == pytest.approx(20 * MM)
    assert second.s_mld == pytest.approx(55 * MM)
    assert first.mld == pytest.approx(1.5 * MM)
    assert second.max_ds_pct == pytest.approx(70.0)
    assert first.length < 10 * MM
    assert first.start_idx < first.mld_idx < first.end_idx < second.start_idx


def test_no_lesion_in_healthy_vessel():
    prof = cosine_lesion_profile(60 * MM, 3.0 * MM, 10 * MM, 0.25)  # below the 30% threshold
    assert prof.lesions() == []


def test_mld():
    r = np.array([1.5, 1.2, 0.6, 1.0, 1.5]) * MM
    mld, idx = minimal_lumen_diameter(r)
    assert mld == pytest.approx(1.2 * MM)
    assert idx == 2
