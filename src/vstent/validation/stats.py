"""Diagnostic-accuracy and agreement statistics for the validation analysis (P4).

Conventions: a vessel is *positive* (ischaemic) when its value is at or below the cut-off
(FFR ≤ 0.80); the index test scores a vessel by its predicted FFR, lower = more likely positive.
Confidence intervals are two-sided at ``level``. Patient-level clustering (several vessels per
patient) is handled by :func:`cluster_bootstrap`; the analytic intervals here treat vessels as
independent. References: Wilson (1927) score interval; DeLong et al. (1988) AUC variance;
Bland & Altman (1986) limits of agreement and their standard error sqrt(3 s² / n).
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np
from scipy import stats

LOA_Z = 1.96  # limits of agreement: bias ± 1.96 SD (conventional, independent of ``level``)


def z_value(level: float) -> float:
    """Two-sided standard-normal quantile for a ``level`` interval."""
    return float(stats.norm.ppf(0.5 + level / 2.0))


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of the proportion ``k`` / ``n`` (NaN for n = 0)."""
    if n == 0:
        return float("nan"), float("nan")
    z, p = z_value(level), k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def auc_delong(score: np.ndarray, positive: np.ndarray) -> tuple[float, float]:
    """Area under the ROC curve of ``score`` (higher = more likely positive; ties count half)
    and its DeLong standard error. NaN when either class is empty."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    x, y = score[positive], score[~positive]
    m, n = x.size, y.size
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    psi = (x[:, None] > y[None, :]) + 0.5 * (x[:, None] == y[None, :])
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    s10 = v10.var(ddof=1) if m > 1 else 0.0
    s01 = v01.var(ddof=1) if n > 1 else 0.0
    return float(psi.mean()), float(np.sqrt(s10 / m + s01 / n))


def auc_interval(auc: float, se: float, level: float = 0.95) -> tuple[float, float]:
    """Interval of an AUC from its standard error, on the logit scale (stays inside [0, 1]);
    a plain normal interval clipped to [0, 1] when the AUC is 0 or 1."""
    if not np.isfinite(auc) or not np.isfinite(se):
        return float("nan"), float("nan")
    z = z_value(level)
    if 0.0 < auc < 1.0 and se > 0:
        logit, se_l = np.log(auc / (1 - auc)), se / (auc * (1 - auc))
        lo, hi = logit - z * se_l, logit + z * se_l
        return float(1 / (1 + np.exp(-lo))), float(1 / (1 + np.exp(-hi)))
    return float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))


def roc_points(score: np.ndarray, positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(false-positive rate, true-positive rate) at every threshold, from (0, 0) to (1, 1)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    thresholds = np.unique(score)[::-1]
    pos, neg = max(positive.sum(), 1), max((~positive).sum(), 1)
    tpr = [0.0] + [float((positive & (score >= t)).sum() / pos) for t in thresholds]
    fpr = [0.0] + [float((~positive & (score >= t)).sum() / neg) for t in thresholds]
    return np.array(fpr), np.array(tpr)


def binary_metrics(
    test_positive: np.ndarray, ref_positive: np.ndarray, level: float = 0.95
) -> dict[str, float | int | tuple[float, float]]:
    """2 × 2 table, sensitivity, specificity, PPV, NPV and accuracy, each with its Wilson
    interval (``<name>_ci``)."""
    t, r = np.asarray(test_positive, bool), np.asarray(ref_positive, bool)
    tp, fp = int((t & r).sum()), int((t & ~r).sum())
    fn, tn = int((~t & r).sum()), int((~t & ~r).sum())
    out: dict[str, float | int | tuple[float, float]] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    for name, k, n in (("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp),
                       ("ppv", tp, tp + fp), ("npv", tn, tn + fn),
                       ("accuracy", tp + tn, tp + tn + fp + fn)):  # fmt: skip
        out[name] = k / n if n else float("nan")
        out[f"{name}_ci"] = wilson(k, n, level)
    return out


def bland_altman(pred: np.ndarray, ref: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Bias (mean of pred − ref), SD of the differences, 95% limits of agreement, the intervals
    of the bias and of each limit (t-based; limit SE = sqrt(3 SD² / n)), and the slope of the
    differences on the means (proportional bias) with its interval."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    d, mean = pred - ref, (pred + ref) / 2.0
    n = d.size
    if n < 3:
        raise ValueError("Bland–Altman needs at least 3 pairs")
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    t = float(stats.t.ppf(0.5 + level / 2.0, n - 1))
    se_bias, se_loa = sd / np.sqrt(n), sd * np.sqrt(3.0 / n)
    low, high = bias - LOA_Z * sd, bias + LOA_Z * sd
    fit = stats.linregress(mean, d)
    t2 = float(stats.t.ppf(0.5 + level / 2.0, n - 2))
    return {
        "n": n,
        "bias": bias,
        "sd": sd,
        "bias_ci": (bias - t * se_bias, bias + t * se_bias),
        "loa_low": low,
        "loa_high": high,
        "loa_low_ci": (low - t * se_loa, low + t * se_loa),
        "loa_high_ci": (high - t * se_loa, high + t * se_loa),
        "slope": float(fit.slope),
        "slope_ci": (float(fit.slope - t2 * fit.stderr), float(fit.slope + t2 * fit.stderr)),
    }


def pearson(x: np.ndarray, y: np.ndarray, level: float = 0.95) -> tuple[float, tuple[float, float]]:
    """Pearson r and its Fisher-z interval (NaN interval for n < 4)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    if x.size < 4 or not np.isfinite(r) or abs(r) >= 1:
        return r, (float("nan"), float("nan"))
    z, half = np.arctanh(r), z_value(level) / np.sqrt(x.size - 3)
    return r, (float(np.tanh(z - half)), float(np.tanh(z + half)))


def cluster_bootstrap(
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], dict[str, float]],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float, int]]:
    """Percentile intervals of ``statistic`` when whole clusters (patients) are resampled with
    replacement. ``statistic(indices)`` returns named values for the rows ``indices``; a
    non-finite value (e.g. an AUC with one class) skips that replicate for that name. Returns
    name → (low, high, replicates used)."""
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in ids]
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.integers(ids.size, size=ids.size)
        idx = np.concatenate([members[i] for i in pick])
        for name, v in statistic(idx).items():
            if np.isfinite(v):
                values[name].append(float(v))
    q = (0.5 - level / 2.0, 0.5 + level / 2.0)
    return {
        name: (float(np.quantile(v, q[0])), float(np.quantile(v, q[1])), len(v))
        for name, v in values.items()
    }
