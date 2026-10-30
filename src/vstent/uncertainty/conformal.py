"""Split-conformal intervals for the surrogate's distal FFR (P1). Torch-free.

Calibrated once on a held-out split (``val``) with the deployed (ONNX) ensemble. Nonconformity
score per vessel, with ŷ the ensemble-mean distal FFR, σ the ensemble standard deviation of
distal FFR and y the solver's distal FFR:

- ``"normalised"``: |y − ŷ| / (σ + σ_floor), so intervals widen where the members disagree;
- ``"absolute"``:   |y − ŷ|, one width for every vessel.

q̂ is the ⌈(n + 1)·level⌉-th smallest calibration score; the interval is ŷ ± q̂·(σ + σ_floor)
(``"absolute"``: ŷ ± q̂), with the upper end capped at 1. For new vessels drawn from the same
distribution as the calibration split, the interval contains the solver's value with probability
≥ ``level`` on average. That guarantee does not extend to vessels outside the synthetic training
distribution (OOD, stented or image-derived), and it covers surrogate-vs-solver error only, not
the error of the reduced-order model or of its inputs.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

CONFORMAL_FILE = "conformal.json"
Score = Literal["normalised", "absolute"]


@dataclass(frozen=True)
class ConformalCalibration:
    """A fitted split-conformal interval for distal FFR (dimensionless)."""

    level: float
    score: Score
    q: float
    sigma_floor: float
    n_calibration: int
    split: str

    def scale(self, std: np.ndarray | float) -> np.ndarray:
        """σ + σ_floor for normalised scores, 1 for absolute scores."""
        std = np.asarray(std, dtype=float)
        return std + self.sigma_floor if self.score == "normalised" else np.ones_like(std)

    def interval(
        self, pred: np.ndarray | float, std: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) around the ensemble-mean distal FFR ``pred`` with spread ``std``."""
        pred = np.asarray(pred, dtype=float)
        half = self.q * self.scale(std)
        return pred - half, np.minimum(pred + half, 1.0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def scores(
    pred: np.ndarray, std: np.ndarray, true: np.ndarray, score: Score, sigma_floor: float
) -> np.ndarray:
    """Nonconformity scores of calibration vessels (see the module docstring)."""
    err = np.abs(np.asarray(true, float) - np.asarray(pred, float))
    if score == "normalised":
        return err / (np.asarray(std, float) + sigma_floor)
    if score == "absolute":
        return err
    raise ValueError(f"unknown score {score!r}")


def conformal_quantile(values: np.ndarray, level: float) -> float:
    """The ⌈(n + 1)·level⌉-th smallest value (``inf`` when that exceeds n)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    v = np.sort(np.asarray(values, dtype=float))
    k = math.ceil((v.size + 1) * level)
    return float("inf") if k > v.size else float(v[k - 1])


def fit_conformal(
    pred: np.ndarray,
    std: np.ndarray,
    true: np.ndarray,
    *,
    level: float = 0.90,
    score: Score = "normalised",
    sigma_floor: float = 0.0,
    split: str = "val",
) -> ConformalCalibration:
    """Calibrate on held-out vessels: ensemble mean / std and solver distal FFR, shape (n,)."""
    if score == "normalised" and sigma_floor <= 0 and np.any(np.asarray(std) <= 0):
        raise ValueError("normalised scores need sigma_floor > 0 when some std is zero")
    q = conformal_quantile(scores(pred, std, true, score, sigma_floor), level)
    return ConformalCalibration(level, score, q, sigma_floor, int(np.size(pred)), split)


def coverage(cal: ConformalCalibration, pred, std, true) -> dict[str, float]:
    """Empirical coverage and interval widths of ``cal`` on vessels with known solver FFR."""
    lo, hi = cal.interval(pred, std)
    true = np.asarray(true, dtype=float)
    inside = (true >= lo) & (true <= hi)
    width = hi - lo
    return {
        "n": int(true.size),
        "coverage": float(inside.mean()),
        "width_mean": float(width.mean()),
        "width_median": float(np.median(width)),
        "width_p95": float(np.percentile(width, 95)),
    }
