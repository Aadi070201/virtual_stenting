"""Sample size for the validation study (P4) → ``reports/validation_sample_size.md``.

The planning assumptions (``configs/validation.yaml``, ``sample_size``) are placeholders to be
set with a statistician from published angiography-derived FFR studies and the partner site's
case mix; the numbers here only follow from them. Three endpoints, per vessel:

- **sensitivity and specificity** estimated to ± ``ci_half_width`` (Buderer 1996):
  N = z² Se (1 − Se) / (d² · prevalence), and the same for Sp with (1 − prevalence);
- **AUC** above ``null_auc`` when it truly is ``expected_auc`` (two-sided α, power 1 − β),
  with the Hanley & McNeil (1982) variance
  V(A) = [A(1 − A) + (n₁ − 1)(Q₁ − A²) + (n₂ − 1)(Q₂ − A²)] / (n₁ n₂),
  Q₁ = A / (2 − A), Q₂ = 2A² / (1 + A), n₁ positives, n₂ negatives;
- **Bland–Altman limits of agreement** with a 95% interval of ± ``loa_ci_half_width``:
  limit SE ≈ sqrt(3 s² / n) (Bland & Altman 1986), so n = 3 (z s / w)².

The largest is inflated by the design effect 1 + (m − 1) · ICC for m vessels per patient, then
for vessels that cannot be analysed. Patients = vessels / m.
"""

import json
import math
from pathlib import Path
from typing import Any

from pydantic import Field
from scipy import stats

from vstent import DISCLAIMER
from vstent.config import StrictModel, git_info
from vstent.validation.stats import z_value


class SampleSizeConfig(StrictModel):
    """Planning assumptions (placeholders until set with a statistician)."""

    alpha: float = Field(0.05, gt=0, lt=1)  # two-sided
    power: float = Field(0.80, gt=0, lt=1)
    prevalence: float = Field(0.35, gt=0, lt=1)  # share of vessels with wire FFR <= 0.80
    expected_sensitivity: float = Field(0.85, gt=0, lt=1)
    expected_specificity: float = Field(0.85, gt=0, lt=1)
    ci_half_width: float = Field(0.075, gt=0, lt=0.5)
    expected_auc: float = Field(0.90, gt=0.5, lt=1)
    null_auc: float = Field(0.80, ge=0.5, lt=1)
    expected_sd_difference: float = Field(0.07, gt=0)  # SD of (index − wire FFR)
    loa_ci_half_width: float = Field(0.03, gt=0)
    vessels_per_patient: float = Field(1.3, ge=1)
    icc: float = Field(0.10, ge=0, lt=1)  # intra-patient correlation of the endpoint
    non_analysable_fraction: float = Field(0.10, ge=0, lt=1)


def n_sens_spec(value: float, half_width: float, share: float, alpha: float = 0.05) -> int:
    """Vessels so that a proportion ``value`` among a ``share`` of them is estimated to
    ± ``half_width`` (Buderer 1996)."""
    z = z_value(1 - alpha)
    return math.ceil(z * z * value * (1 - value) / (half_width**2 * share))


def hanley_mcneil_variance(auc: float, n_pos: float, n_neg: float) -> float:
    """Variance of the empirical AUC (Hanley & McNeil 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    return (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (
        n_pos * n_neg
    )


def n_auc(
    expected: float, null: float, prevalence: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Vessels for the AUC test (H0: ``null``; H1: ``expected``), positives : negatives as the
    prevalence."""
    za, zb = z_value(1 - alpha), float(stats.norm.ppf(power))
    ratio = (1 - prevalence) / prevalence
    for n_pos in range(3, 100_000):
        n_neg = n_pos * ratio
        se0 = math.sqrt(hanley_mcneil_variance(null, n_pos, n_neg))
        se1 = math.sqrt(hanley_mcneil_variance(expected, n_pos, n_neg))
        if za * se0 + zb * se1 <= expected - null:
            return math.ceil(n_pos / prevalence)
    raise ValueError("no sample size below 100,000 positives")


def n_loa(sd: float, half_width: float, alpha: float = 0.05) -> int:
    """Pairs so that each limit of agreement has a 1 − α interval of ± ``half_width``."""
    z = z_value(1 - alpha)
    return math.ceil(3 * (z * sd / half_width) ** 2)


def sample_size(cfg: SampleSizeConfig) -> dict[str, Any]:
    """Vessels per endpoint, the governing one, then the clustered and attrition-inflated
    totals (vessels and patients)."""
    endpoints = {
        "sensitivity": n_sens_spec(cfg.expected_sensitivity, cfg.ci_half_width,
                                   cfg.prevalence, cfg.alpha),
        "specificity": n_sens_spec(cfg.expected_specificity, cfg.ci_half_width,
                                   1 - cfg.prevalence, cfg.alpha),
        "auc": n_auc(cfg.expected_auc, cfg.null_auc, cfg.prevalence, cfg.alpha, cfg.power),
        "limits_of_agreement": n_loa(cfg.expected_sd_difference, cfg.loa_ci_half_width,
                                     cfg.alpha),
    }  # fmt: skip
    governing = max(endpoints, key=endpoints.get)
    design_effect = 1 + (cfg.vessels_per_patient - 1) * cfg.icc
    clustered = math.ceil(endpoints[governing] * design_effect)
    vessels = math.ceil(clustered / (1 - cfg.non_analysable_fraction))
    return {
        "endpoints": endpoints,
        "governing": governing,
        "design_effect": design_effect,
        "vessels_clustered": clustered,
        "vessels": vessels,
        "patients": math.ceil(vessels / cfg.vessels_per_patient),
    }


def sensitivity_grid(cfg: SampleSizeConfig) -> list[dict[str, Any]]:
    """Total vessels when prevalence and expected sensitivity / specificity / AUC vary."""
    rows = []
    for prev in (0.25, 0.35, 0.45):
        for se_sp, auc in ((0.80, 0.85), (0.85, 0.90), (0.90, 0.93)):
            c = cfg.model_copy(
                update={"prevalence": prev, "expected_sensitivity": se_sp,
                        "expected_specificity": se_sp, "expected_auc": auc}
            )  # fmt: skip
            r = sample_size(c)
            rows.append({"prevalence": prev, "sens_spec": se_sp, "auc": auc,
                         "governing": r["governing"], "vessels": r["vessels"],
                         "patients": r["patients"]})  # fmt: skip
    return rows


_LABELS = {
    "sensitivity": "Sensitivity to ± d",
    "specificity": "Specificity to ± d",
    "auc": "AUC above the null value",
    "limits_of_agreement": "Limits of agreement to ± w",
}


def write_sample_size_report(cfg: SampleSizeConfig, out_md: str | Path) -> dict[str, Any]:
    """Write ``out_md`` (+ ``.json``) and return the result."""
    git = git_info()
    res = sample_size(cfg)
    grid = sensitivity_grid(cfg)
    a = cfg
    lines = [
        "# Validation study: sample size",
        "",
        f"> {DISCLAIMER} Planning numbers for the protocol template "
        "(`docs/validation/PROTOCOL.md`), not a statistical analysis plan.",
        "",
        "**The assumptions below are placeholders.** They must be set with a statistician from "
        "published angiography-derived FFR studies and the partner site's case mix before the "
        "protocol is submitted. The numbers only follow from them "
        "(`vstent sample-size`, `configs/validation.yaml`).",
        "",
        "## Assumptions",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Two-sided α; power | {a.alpha}; {a.power} |",
        f"| Prevalence of wire FFR ≤ 0.80 (per vessel) | {a.prevalence} |",
        f"| Expected sensitivity; specificity | {a.expected_sensitivity}; "
        f"{a.expected_specificity} |",
        f"| Precision d (half-width of their {1 - a.alpha:.0%} interval) | ± {a.ci_half_width} |",
        f"| Expected AUC; null AUC | {a.expected_auc}; {a.null_auc} |",
        f"| SD of (index − wire FFR) | {a.expected_sd_difference} |",
        f"| Precision w of each limit of agreement | ± {a.loa_ci_half_width} |",
        f"| Vessels per patient m; intra-patient correlation (ICC) | {a.vessels_per_patient}; "
        f"{a.icc} |",
        f"| Vessels that cannot be analysed | {a.non_analysable_fraction:.0%} |",
        "",
        "## Vessels needed per endpoint",
        "",
        "| Endpoint | Method | Vessels |",
        "|---|---|---:|",
        f"| {_LABELS['sensitivity']} | Buderer 1996 | {res['endpoints']['sensitivity']} |",
        f"| {_LABELS['specificity']} | Buderer 1996 | {res['endpoints']['specificity']} |",
        f"| {_LABELS['auc']} | Hanley & McNeil 1982 variance | {res['endpoints']['auc']} |",
        f"| {_LABELS['limits_of_agreement']} | Bland & Altman 1986 | "
        f"{res['endpoints']['limits_of_agreement']} |",
        "",
        f"Governing endpoint: **{_LABELS[res['governing']]}** "
        f"({res['endpoints'][res['governing']]} vessels). With the design effect "
        f"1 + (m − 1) · ICC = {res['design_effect']:.3f}: {res['vessels_clustered']} vessels; "
        f"allowing {a.non_analysable_fraction:.0%} that cannot be analysed: "
        f"**{res['vessels']} vessels in about {res['patients']} patients**.",
        "",
        "## How the total changes with the assumptions",
        "",
        "Other assumptions as above; sensitivity = specificity.",
        "",
        "| Prevalence | Sens. = spec. | Expected AUC | Governing endpoint | Vessels | Patients |",
        "|---:|---:|---:|---|---:|---:|",
        *(
            f"| {r['prevalence']:.2f} | {r['sens_spec']:.2f} | {r['auc']:.2f} | "
            f"{_LABELS[r['governing']]} | {r['vessels']} | {r['patients']} |"
            for r in grid
        ),
        "",
        "Per-vessel formulas assume independent vessels; the design effect is a simple "
        "correction for patients with several vessels. A statistician may prefer a clustered "
        "method (e.g. Obuchowski 1997 for the AUC).",
        "",
        f"Generated from commit `{git['git_commit']}` (dirty: {git['git_dirty']}).",
        "",
    ]
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    payload = {"assumptions": cfg.model_dump(), **res, "grid": grid, **git}
    out_md.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return res
