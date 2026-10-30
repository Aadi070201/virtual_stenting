"""Grey-zone flags (P1): a distal FFR whose interval contains the decision threshold.

Two intervals are assessed separately, because they answer different questions:

- ``model``: the surrogate's split-conformal interval (surrogate-vs-solver error only);
- ``inputs``: the solver's Monte Carlo interval over the input assumptions (those only).

A case is flagged against 0.80 (pre-PCI); a plan against the planner's target (post-PCI).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from vstent.config import StrictModel, load_config
from vstent.uncertainty.conformal import Score
from vstent.uncertainty.inputs import (
    InputUncertainty,
    InputUncertaintyConfig,
    QCAErrorModel,
    percentile_interval,
)

PRE_PCI_THRESHOLD = 0.80
UNCERTAINTY_CONFIG = Path("configs/uncertainty.yaml")


class UncertaintyConfig(StrictModel):
    """``configs/uncertainty.yaml``: interval level, conformal settings, input Monte Carlo."""

    level: float = Field(0.90, gt=0, lt=1)
    pre_pci_threshold: float = Field(PRE_PCI_THRESHOLD, gt=0, lt=1)
    conformal_score: Score = "normalised"
    sigma_floor: float = Field(0.0002, ge=0)  # FFR units, added to the ensemble SD
    calibration_split: str = "cal"  # a split used for nothing else (DECISIONS D29)
    inputs: InputUncertaintyConfig = InputUncertaintyConfig()


def load_uncertainty(
    path: str | Path = UNCERTAINTY_CONFIG,
) -> tuple[UncertaintyConfig, QCAErrorModel | None]:
    """The uncertainty config (defaults if the file is missing) and the QCA error model it
    names (``None`` if that report is missing). Relative paths are taken from the working
    directory, as everywhere in the configs."""
    path = Path(path)
    cfg = load_config(path, UncertaintyConfig) if path.exists() else UncertaintyConfig()
    return cfg, QCAErrorModel.from_report(cfg.inputs.qca_error_report)


@dataclass(frozen=True)
class Interval:
    """A central interval of distal FFR (dimensionless)."""

    low: float
    high: float
    level: float

    def contains(self, x: float) -> bool:
        return self.low <= x <= self.high

    def text(self) -> str:
        return f"{self.low:.3f}–{self.high:.3f}"


@dataclass(frozen=True)
class Assessment:
    """A distal FFR, its intervals and the grey-zone flag against ``threshold``."""

    threshold: float
    solver: float
    surrogate: float | None = None
    model: Interval | None = None
    inputs: Interval | None = None

    @property
    def model_flag(self) -> bool:
        return self.model is not None and self.model.contains(self.threshold)

    @property
    def inputs_flag(self) -> bool:
        return self.inputs is not None and self.inputs.contains(self.threshold)

    @property
    def flagged(self) -> bool:
        """Grey zone: either interval contains the threshold."""
        return self.model_flag or self.inputs_flag

    @property
    def level_text(self) -> str:
        """The intervals' level, e.g. "90%", or "model 90%, inputs 95%" when they differ."""
        levels = {k: iv.level for k, iv in (("model", self.model), ("inputs", self.inputs)) if iv}
        if len(set(levels.values())) > 1:
            return ", ".join(f"{k} {v:.0%}" for k, v in levels.items())
        return f"{next(iter(levels.values())):.0%}" if levels else ""

    def reasons(self, what: str = "Distal FFR") -> list[str]:
        """Plain-language explanation of each flag."""
        out = []
        pct = f"{self.model.level:.0%}" if self.model else ""
        if self.model_flag:
            out.append(
                f"{what}: the surrogate's {pct} interval {self.model.text()} contains "
                f"{self.threshold:.2f} (surrogate-vs-solver error)"
            )
        if self.inputs_flag:
            pct = f"{self.inputs.level:.0%}"
            out.append(
                f"{what}: the {pct} interval over the input assumptions "
                f"{self.inputs.text()} contains {self.threshold:.2f}"
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        def iv(x: Interval | None):
            return None if x is None else [round(x.low, 4), round(x.high, 4)]

        return {
            "threshold": self.threshold,
            "model_level": None if self.model is None else self.model.level,
            "input_level": None if self.inputs is None else self.inputs.level,
            "model_interval": iv(self.model),
            "input_interval": iv(self.inputs),
            "grey_zone": self.flagged,
            "grey_zone_model": self.model_flag,
            "grey_zone_inputs": self.inputs_flag,
        }


def model_interval(runner, ffr_distal: float, std_distal: float | None) -> Interval | None:
    """A ``SurrogateRunner``'s calibrated interval of one distal FFR, at its calibration level
    (``None`` without an ensemble spread or a conformal calibration)."""
    if std_distal is None or runner.conformal is None:
        return None
    low, high = runner.conformal.interval(float(ffr_distal), float(std_distal))
    return Interval(float(low), float(high), runner.conformal.level)


def assessment(
    threshold: float,
    solver: float,
    surrogate: float | None = None,
    model: Interval | None = None,
    samples: np.ndarray | None = None,
    level: float = 0.90,
) -> Assessment:
    """An :class:`Assessment` with the input interval taken from Monte Carlo ``samples``."""
    inputs = None if samples is None else Interval(*percentile_interval(samples, level), level)
    return Assessment(threshold, solver, surrogate, model, inputs)


def plan_assessments(plans, mc: InputUncertainty | None, level: float) -> list[Assessment]:
    """Intervals and flags (against the planner target) of a ``PlanResult``'s plans; ``level``
    is that of the input intervals (model intervals keep their calibration level)."""
    out = []
    for pl in plans.plans:
        model = pl.model_interval
        samples = None if mc is None else mc.post_samples(pl.stents, plans.edge)
        solver = pl.ffr if pl.ffr_solver is None else pl.ffr_solver
        surrogate = pl.ffr if plans.backend == "surrogate" else None
        out.append(assessment(plans.target, solver, surrogate, model, samples, level))
    return out
