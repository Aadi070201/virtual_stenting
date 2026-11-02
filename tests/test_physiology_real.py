"""P4e: the model's physiology assumptions against invasive measurements (DEFINE-FLOW format)."""

import numpy as np
import pytest

from vstent.data.physiology import DATASETS
from vstent.validation import physiology_real as pr

FS = pr.FS_HZ


def _tracing(period=0.8, systole=0.30, ratio=2.0, seconds=6.0, first_r=0.25):
    """A DEFINE-FLOW-like tracing: R waves every ``period``, a Doppler velocity of 10 cm/s in
    systole and ``ratio`` × 10 cm/s in diastole, recorded on every other sample."""
    t = np.arange(0.0, seconds, 1.0 / FS)
    r_times = np.arange(first_r, seconds, period)
    ecg = 0.05 * np.random.default_rng(0).standard_normal(t.size)
    for r in r_times:
        ecg += np.exp(-0.5 * ((t - r) / 0.012) ** 2) - 0.25 * np.exp(
            -0.5 * ((t - r - 0.03) / 0.012) ** 2
        )
    since = np.full(t.size, np.inf)
    for r in r_times:
        since = np.where(t >= r, np.minimum(since, t - r), since)
    ipv = np.where(since < systole, 10.0, 10.0 * ratio)
    ipv[since == np.inf] = 10.0
    ipv[1::2] = np.nan
    flags = np.zeros(t.size)
    flags[np.round(r_times * FS).astype(int)] = 1.0
    return {"time": t, "Pd": np.full(t.size, 70.0), "Pa": np.full(t.size, 90.0), "IPV": ipv,
            "ECG": ecg, "RWave": flags}  # fmt: skip


def test_beat_profiles_recover_the_phasic_step():
    tr = _tracing(period=0.8, systole=0.30, ratio=2.0)
    beats = pr.beat_profiles(tr, int(1.0 * FS), int(5.0 * FS))
    assert len(beats) >= 3
    assert np.median([b["ratio"] for b in beats]) == pytest.approx(2.0, rel=0.05)
    assert np.median([b["systole_s"] for b in beats]) == pytest.approx(0.30, abs=0.02)
    assert np.median([b["period_s"] for b in beats]) == pytest.approx(0.8, abs=0.01)
    # without an ECG the recorder's flags give the beats
    flat = {**tr, "ECG": np.zeros_like(tr["ECG"])}
    assert len(pr.beat_profiles(flat, int(1.0 * FS), int(5.0 * FS))) >= 3


def test_summary_rules():
    def lesion(ratio, vapv, ffr):
        return {"flow_id": "F", "lesion": "L", "vessel": "LAD", "ffr": ffr, "cfr": 2.0,
                "bapv": 0.2, "vapv": vapv, "fpa": 90.0, "rest_ratio": ratio,
                "rest_systole_s": 0.3, "hyperemia_ratio": ratio}  # fmt: skip

    ok = pr.summarise([lesion(r, 0.6, 0.8) for r in (1.8, 2.0, 2.2)])
    assert not ok["rules"]["phasic_prior_replaced"]
    # k × 0.6 / 0.8 = 0.375 (k 0.5) or 0.6 (k 0.8): 0.35 m/s is outside both IQRs
    assert ok["rules"]["fixed_velocity_revisit"]
    flat = pr.summarise([lesion(r, 0.5, 1.0) for r in (1.1, 1.2, 1.3)])
    assert flat["rules"]["phasic_prior_replaced"] and flat["rules"]["phasic_median_ratio"] == 1.2
    assert pr.vessel("7") == "LAD" and pr.vessel("2") == "RCA" and pr.vessel("") == "unknown"


def test_datasets_are_pinned_by_md5():
    for meta in DATASETS.values():
        assert meta["files"] and all(len(md5) == 32 for md5 in meta["files"].values())
        assert meta["record"].startswith("https://zenodo.org/records/")


def test_artery_velocity_rule():
    """P4k-C4: per-artery hyperemic velocity and the ≥ 10% rule."""
    from vstent.validation.physiology_real import artery_velocity

    les = ([{"vessel": "LAD", "vapv": 0.40, "ffr": 0.8}] * 4
           + [{"vessel": "RCA", "vapv": 0.30, "ffr": 0.8}] * 2)  # fmt: skip
    r = artery_velocity(les)
    assert r["arteries"]["left"]["median"] == pytest.approx(0.4)
    assert r["arteries"]["RCA"]["median"] == pytest.approx(0.3)
    assert r["pooled"] == pytest.approx(0.4) and r["use_per_artery"]  # RCA 25% lower
    same = artery_velocity([{"vessel": "LCx", "vapv": 0.4, "ffr": 0.8},
                            {"vessel": "RCA", "vapv": 0.41, "ffr": 0.8}])  # fmt: skip
    assert not same["use_per_artery"]
