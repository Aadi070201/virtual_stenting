"""Validation report (P4): markdown, JSON and a figure from scored predictions.

The layout follows the order of a diagnostic-accuracy paper (STARD 2015): flow of vessels, the
pre-specified primary analysis, the other index tests, the grey zone, per-territory results and
every vessel that could not be analysed.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER
from vstent.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    SURFACE,
    new_figure,
    save_figure,
    style_axes,
)
from vstent.validation.manifest import ReferenceRow
from vstent.validation.run import (
    INDEX_TESTS,
    MODE,
    ValidationConfig,
    _float,
    score_predictions,
)
from vstent.validation.stats import LOA_Z, auc_delong, roc_points

COLORS = {"single_solver": SERIES_1, "single_surrogate": SERIES_2, "tree_solver": SERIES_3}
SHORT = {"single_solver": "single (solver)", "single_surrogate": "single (surrogate)",
         "tree_solver": "tree (solver)"}  # fmt: skip


def ci_text(ci) -> str:
    if ci is None or not all(np.isfinite(ci[:2])):
        return "–"
    return f"{ci[0]:.3f}–{ci[1]:.3f}"


def _v(x, digits: int = 3) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _primary_table(res: dict[str, Any]) -> list[str]:
    boot = res.get("bootstrap", {})
    rows = [
        ("AUC (FFR ≤ cut-off)", res.get("auc"), res.get("auc_ci"), boot.get("auc")),
        ("Sensitivity", res.get("sensitivity"), res.get("sensitivity_ci"),
         boot.get("sensitivity")),
        ("Specificity", res.get("specificity"), res.get("specificity_ci"),
         boot.get("specificity")),
        ("Positive predictive value", res.get("ppv"), res.get("ppv_ci"), None),
        ("Negative predictive value", res.get("npv"), res.get("npv_ci"), None),
        ("Accuracy (analysable vessels)", res.get("accuracy"), res.get("accuracy_ci"),
         boot.get("accuracy")),
        ("Accuracy, intention to diagnose", res.get("accuracy_itd"), res.get("accuracy_itd_ci"),
         None),
        ("Feasibility (analysable / eligible)", res.get("feasibility"),
         res.get("feasibility_ci"), None),
    ]  # fmt: skip
    ba = res.get("bland_altman")
    if ba:
        rows += [
            ("Bias (index − wire)", ba["bias"], ba["bias_ci"], boot.get("bias")),
            ("SD of differences", ba["sd"], None, None),
            ("Lower limit of agreement", ba["loa_low"], ba["loa_low_ci"], None),
            ("Upper limit of agreement", ba["loa_high"], ba["loa_high_ci"], None),
            ("Mean absolute difference", res.get("mae"), None, boot.get("mae")),
            ("Pearson r", res.get("r"), res.get("r_ci"), None),
            ("Proportional bias (slope of differences on means)", ba["slope"], ba["slope_ci"],
             None),
        ]  # fmt: skip
    out = [
        "| Quantity | Estimate | Interval (vessels independent) | Interval (patient bootstrap) |",
        "|---|---:|---:|---:|",
    ]
    for name, est, ci, bci in rows:
        out.append(f"| {name} | {_v(est)} | {ci_text(ci)} | {ci_text(bci)} |")
    return out


def _tests_table(tests: dict[str, Any], eligible: int) -> list[str]:
    out = ["| Index test | Analysable | AUC (interval) | Sensitivity | Specificity | Accuracy | "
           "Accuracy, ITD | Bias | Limits of agreement | MAE |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for test, r in tests.items():
        ba = r.get("bland_altman")
        loa = f"{ba['loa_low']:.3f} to {ba['loa_high']:.3f}" if ba else "–"
        out.append(
            f"| {INDEX_TESTS[test]} | {r['n']} / {eligible} | {_v(r.get('auc'))} "
            f"({ci_text(r.get('auc_ci'))}) | {_v(r.get('sensitivity'))} | "
            f"{_v(r.get('specificity'))} | {_v(r.get('accuracy'))} | "
            f"{_v(r.get('accuracy_itd'))} | {_v(ba['bias']) if ba else '–'} | {loa} | "
            f"{_v(r.get('mae'))} |"
        )
    return out


def validation_markdown(
    summary: dict[str, Any],
    meta: dict[str, Any] | None,
    cfg: ValidationConfig,
    *,
    title: str,
    intro: list[str],
    figure: str | None = None,
) -> list[str]:
    """Report lines for a scored run (``summary`` from :func:`score_predictions`)."""
    flow = summary["flow"]
    lines = [f"# {title}", "", f"> {DISCLAIMER}", "", *intro, ""]
    if meta:
        models = meta.get("models") or {}
        model_text = "; ".join(
            f"{k}: `{v['dir']}` (SHA-256 {v['sha256'][:12]}…)" for k, v in models.items() if v
        )
        lines += [
            "## Run",
            "",
            f"- Predictions locked {meta['created_utc']}: `{meta['predictions_file']}`, SHA-256 "
            f"`{meta['predictions_sha256'][:16]}…` (verified unchanged before scoring).",
            f"- Software: vstent {meta['vstent_version']}, commit `{meta['git_commit']}` "
            f"(dirty: {meta['git_dirty']}). Manifest SHA-256 `{meta['manifest']['sha256'][:16]}…`.",
            f"- Models: {model_text or 'none'}. Vessel masks from the manifest used: "
            f"{meta['use_masks']}.",
            "",
        ]
    lines += [
        "## Flow of vessels",
        "",
        f"- In the case list: {flow['manifest']}",
        f"- Without a reference value: {len(flow['no_reference'])}",
        f"- Excluded by the reference file (e.g. wire drift): {len(flow['excluded'])}"
        + "".join(f"\n  - {e['case_id']}: {e['reason']}" for e in flow["excluded"]),
        f"- Eligible: **{flow['eligible']} vessels in {flow['eligible_patients']} patients**",
        "",
    ]
    if flow["reference_without_prediction"]:
        lines += [f"Reference values without a case in the list (ignored): "
                  f"{', '.join(flow['reference_without_prediction'])}", ""]  # fmt: skip
    ffr = summary["index_tests"].get("FFR", {})
    primary = cfg.primary_index_test
    if primary in ffr:
        r = ffr[primary]
        lines += [
            f"## Primary analysis: {INDEX_TESTS[primary]} vs wire FFR",
            "",
            f"Pre-specified: index test FFR ≤ {cfg.ffr_cutoff:.2f} vs wire FFR ≤ "
            f"{cfg.ffr_cutoff:.2f}, per vessel; {cfg.level:.0%} intervals. {r['n']} analysable "
            f"vessels in {r['n_patients']} patients; {r['n_positive']} positive by wire "
            f"(prevalence {r['prevalence']:.0%}). Patient bootstrap: {cfg.n_bootstrap} "
            f"resamples of whole patients (seed {cfg.seed}).",
            "",
            *_primary_table(r),
            "",
            f"2 × 2 table: TP {r.get('tp', 0)}, FP {r.get('fp', 0)}, FN {r.get('fn', 0)}, "
            f"TN {r.get('tn', 0)}. Intention to diagnose counts the "
            f"{len(r['failed'])} eligible vessel(s) the index test could not analyse as "
            "misclassified.",
            "",
        ]
    if ffr:
        lines += ["## All index tests vs wire FFR", "",
                  *_tests_table(ffr, summary["eligible_by_type"]["FFR"]), ""]  # fmt: skip
    if figure:
        lines += [f"![Validation figure]({figure})", ""]
    ifr = summary["index_tests"].get("iFR")
    if ifr:
        lines += [
            "## Index tests vs wire iFR (classification only)",
            "",
            f"Index test FFR ≤ {cfg.ffr_cutoff:.2f} against iFR ≤ {cfg.ifr_cutoff:.2f}. FFR and "
            "iFR are different quantities, so there is no Bland–Altman analysis.",
            "",
            *_tests_table(ifr, summary["eligible_by_type"]["iFR"]),
            "",
        ]
    gz = summary.get("grey_zone")
    if gz:
        lines += [
            "## Grey zone (P1 intervals, single path)",
            "",
            f"A vessel is flagged when its model or input interval contains "
            f"{cfg.ffr_cutoff:.2f}. Classification by the solver's FFR.",
            "",
            "| | Vessels | Accuracy (interval) |",
            "|---|---:|---:|",
            f"| Not flagged | {gz['n'] - gz['n_flagged']} | {_v(gz['accuracy_unflagged'])} "
            f"({ci_text(gz['accuracy_unflagged_ci'])}) |",
            f"| Flagged | {gz['n_flagged']} ({gz['share_flagged']:.0%}) | "
            f"{_v(gz['accuracy_flagged'])} ({ci_text(gz['accuracy_flagged_ci'])}) |",
            f"| All | {gz['n']} | {_v(gz['accuracy_all'])} |",
            f"| Hybrid: wire only when flagged | {gz['n']} | {_v(gz['accuracy_hybrid'])} "
            f"({ci_text(gz['accuracy_hybrid_ci'])}) |",
            "",
            f"The hybrid strategy would need the wire in {gz['share_flagged']:.0%} of vessels.",
            "",
        ]
    terr = summary.get("territories")
    if terr:
        lines += [
            f"## Per vessel territory ({SHORT[primary]})",
            "",
            "| Vessel | Analysable / eligible | Prevalence | AUC | Sensitivity | Specificity "
            "| Bias | MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for t in terr:
            if t["n"] == 0:
                lines.append(f"| {t['vessel']} | 0 / {t['n_eligible']} | – | – | – | – | – | – |")
                continue
            lines.append(
                f"| {t['vessel']} | {t['n']} / {t['n_eligible']} | {t['prevalence']:.0%} | "
                f"{_v(t['auc'])} | {_v(t['sensitivity'])} | {_v(t['specificity'])} | "
                f"{_v(t['bias'])} | {_v(t['mae'])} |"
            )
        lines.append("")
    failed = {}
    for test, r in ffr.items():
        for f in r["failed"]:
            failed.setdefault(MODE[test], {})[f["case_id"]] = f["error"]
    if failed:
        lines += ["## Vessels that could not be analysed", ""]
        for mode, items in failed.items():
            lines.append(f"**{mode} path mode** ({len(items)}):")
            lines.append("")
            reasons: dict[str, int] = {}
            for err in items.values():  # group messages that differ only in numbers
                key = re.sub(r"\d+(\.\d+)?", "#", err)[:120]
                reasons[key] = reasons.get(key, 0) + 1
            for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"- {n} × {reason}")
            lines.append("")
    return lines


def _arrays(rows, refs: list[ReferenceRow], test: str):
    ref_by = {r.case_id: r for r in refs}
    pred, ref = [], []
    for row in rows:
        r = ref_by.get(row["case_id"])
        if r is None or r.excluded or r.ref_type != "FFR" or row[f"{MODE[test]}_status"] != "ok":
            continue
        v = _float(row[test])
        if np.isfinite(v):
            pred.append(v)
            ref.append(r.ref_value)
    return np.array(pred), np.array(ref)


def validation_figure(
    rows, refs: list[ReferenceRow], cfg: ValidationConfig, tests: list[str], path, title: str
) -> None:
    """Predicted vs wire FFR and Bland–Altman (primary index test), ROC curves (all tests)."""
    primary = cfg.primary_index_test
    pred, ref = _arrays(rows, refs, primary)
    fig = new_figure(12.0, 4.2, title, f"Primary index test: {INDEX_TESTS[primary]}; "
                     f"{pred.size} analysable vessels with wire FFR")  # fmt: skip
    ax_s, ax_b, ax_r = fig.subplots(1, 3, gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.17)
    for ax in (ax_s, ax_b, ax_r):
        style_axes(ax)
    cut = cfg.ffr_cutoff
    if pred.size:
        lo = float(min(pred.min(), ref.min(), 0.4)) - 0.02
        ax_s.plot([lo, 1.0], [lo, 1.0], color=INK_MUTED, lw=1, ls="--", zorder=1)
        ax_s.axvline(cut, color=INK_MUTED, lw=0.8, ls=":")
        ax_s.axhline(cut, color=INK_MUTED, lw=0.8, ls=":")
        ax_s.scatter(ref, pred, s=14, color=COLORS[primary], alpha=0.8, lw=0, zorder=3)
        ax_s.set_xlim(lo, 1.01)
        ax_s.set_ylim(lo, 1.01)
        mean, d = (pred + ref) / 2, pred - ref
        bias, sd = d.mean(), d.std(ddof=1) if d.size > 1 else 0.0
        ax_b.scatter(mean, d, s=14, color=COLORS[primary], alpha=0.8, lw=0, zorder=3)
        low, high = bias - LOA_Z * sd, bias + LOA_Z * sd
        for y, label, is_bias in ((bias, f"bias {bias:+.3f}", True), (low, f"{low:+.3f}", False),
                                  (high, f"{high:+.3f}", False)):  # fmt: skip
            ax_b.axhline(y, color=INK_SECONDARY if is_bias else INK_MUTED, lw=1,
                         ls="-" if is_bias else "--")  # fmt: skip
            ax_b.annotate(label, xy=(0.01, y), xycoords=("axes fraction", "data"), fontsize=7,
                          color=INK_SECONDARY, va="bottom", ha="left",
                          bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none",
                                "alpha": 0.85})  # fmt: skip
    ax_s.set_xlabel("Wire FFR", fontsize=8, color=INK_SECONDARY)
    ax_s.set_ylabel("Index test FFR", fontsize=8, color=INK_SECONDARY)
    ax_s.set_title("Agreement (dotted: cut-off)", fontsize=9, color=INK_SECONDARY, loc="left")
    ax_b.set_xlabel("Mean of index test and wire FFR", fontsize=8, color=INK_SECONDARY)
    ax_b.set_ylabel("Index test − wire FFR", fontsize=8, color=INK_SECONDARY)
    ax_b.set_title("Bland–Altman (bias, 95% limits)", fontsize=9, color=INK_SECONDARY, loc="left")
    ax_r.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--")
    for test in tests:
        p, r = _arrays(rows, refs, test)
        pos = r <= cut
        if p.size == 0 or pos.all() or not pos.any():
            continue
        fpr, tpr = roc_points(-p, pos)
        auc = auc_delong(-p, pos)[0]
        ax_r.plot(fpr, tpr, color=COLORS[test], lw=1.8, drawstyle="steps-post",
                  label=f"{SHORT[test]}: AUC {auc:.3f}")  # fmt: skip
    ax_r.set_xlabel("1 − specificity", fontsize=8, color=INK_SECONDARY)
    ax_r.set_ylabel("Sensitivity", fontsize=8, color=INK_SECONDARY)
    ax_r.set_title(f"ROC, wire FFR ≤ {cut:.2f}", fontsize=9, color=INK_SECONDARY, loc="left")
    ax_r.legend(fontsize=7, frameon=False, loc="lower right")
    save_figure(fig, path)


def write_validation_report(
    rows,
    meta: dict[str, Any] | None,
    refs: list[ReferenceRow],
    cfg: ValidationConfig,
    md_path: str | Path,
    fig_path: str | Path,
    *,
    title: str,
    intro: list[str],
    extra: list[str] | None = None,
) -> dict[str, Any]:
    """Score, then write ``md_path`` (+ ``.json``) and the figure; returns the summary."""
    tests = (meta or {}).get("index_tests")
    summary = score_predictions(rows, refs, cfg, tests)
    tests = tests or [t for t in INDEX_TESTS if t in summary["index_tests"].get("FFR", {})]
    md_path, fig_path = Path(md_path), Path(fig_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    validation_figure(rows, refs, cfg, tests, fig_path, title)
    rel = Path(os.path.relpath(fig_path, md_path.parent)).as_posix()
    lines = validation_markdown(summary, meta, cfg, title=title, intro=intro, figure=rel)
    lines += extra or []
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    payload = {"summary": summary, "run": meta}
    md_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8", newline="\n"
    )
    return summary
