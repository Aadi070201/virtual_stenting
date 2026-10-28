"""Gradient-boosting baseline: hand-crafted features -> distal FFR (docs/PLAN.md, M3).

Features are measured from the profile, as the app would measure them, not taken from the
synthetic sampling metadata.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from vstent.physics.geometry import LESION_THRESHOLD_PCT
from vstent.surrogate.features import RawInputs
from vstent.units import m_to_mm, pa_to_mmhg

BASELINE_FEATURES = ("max_ds_pct", "lesion_length_mm", "d0_mm", "length_mm", "pa_mmhg", "r_micro")


def hand_features(raw: RawInputs) -> np.ndarray:
    """(B, 6): max DS (%), total length with DS > 30% (mm), D0 (mm), L (mm), Pa (mmHg),
    R_micro (Pa·s/m³)."""
    r, r_ref = np.atleast_2d(raw.r), np.atleast_2d(raw.r_ref)
    ds_pct = 100.0 * (1.0 - r / r_ref)
    ds_mm = m_to_mm(np.asarray(raw.length)) / (r.shape[1] - 1)
    return np.stack(
        [
            ds_pct.max(axis=1),
            (ds_pct > LESION_THRESHOLD_PCT).sum(axis=1) * ds_mm,
            m_to_mm(2.0 * r_ref[:, 0]),
            m_to_mm(np.asarray(raw.length)),
            pa_to_mmhg(np.asarray(raw.pa)),
            np.asarray(raw.r_micro),
        ],
        axis=1,
    )


def fit_baseline(raw: RawInputs, ffr_distal: np.ndarray, seed: int = 0):
    """Fit a histogram gradient-boosting regressor on the hand features."""
    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.1, early_stopping=True, random_state=seed
    )
    return model.fit(hand_features(raw), ffr_distal)
