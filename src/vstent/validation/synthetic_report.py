"""``vstent validate-synthetic`` (P4): the validation tool end to end on synthetic stand-in
cases → ``reports/validation_synthetic.md``. Not clinical data, not a clinical validation.

The stand-in study (:mod:`vstent.validation.synthetic`) is generated into ``data/`` (git-ignored),
then predicted twice: with the true lumen masks of the manifest (the main analysis: QCA, physics
and the statistics, without the segmenter's domain gap on rendered images), and with the
automatic segmenter (feasibility and accuracy when segmentation is part of the index test).
Because the truth is known, the report adds what a real study cannot measure: QCA length and
MLD errors, side branches found, and the side-branch effect inside the reference itself.
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import stats

from vstent.config import git_info
from vstent.inference.onnx_runtime import (
    ONNX_FILE,
    SEG_ONNX_FILE,
    SegmentationRunner,
    SurrogateRunner,
)
from vstent.inference.pipeline import analyse_tree
from vstent.physics.solver import PhysicsParams
from vstent.uncertainty.greyzone import load_uncertainty
from vstent.validation.manifest import read_cases, read_reference, resolve
from vstent.validation.report import SHORT, ci_text, write_validation_report
from vstent.validation.run import (
    INDEX_TESTS,
    ValidationConfig,
    _float,
    model_info,
    read_predictions,
    run_predictions,
    score_predictions,
)
from vstent.validation.synthetic import SyntheticStudyConfig, generate_synthetic_study

REPORT = Path("reports/validation_synthetic.md")
FIGURE = Path("reports/figures/validation_synthetic.png")
PREDICTIONS_DIR = Path("reports/validation_synthetic")


def _median_iqr(x: np.ndarray, digits: int = 2) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "–"
    q = np.percentile(x, [25, 50, 75])
    return f"{q[1]:+.{digits}f} ({q[0]:+.{digits}f} to {q[2]:+.{digits}f})"


def _truth_section(rows, truth: list[dict[str, str]], cfg: ValidationConfig) -> list[str]:
    """What only synthetic data can show: QCA against the true geometry, and the reference's
    own side-branch effect."""
    by = {r["case_id"]: r for r in rows}
    t_ok = [t for t in truth if by[t["case_id"]]["single_status"] == "ok"]
    length = np.array([_float(by[t["case_id"]]["length_mm"]) - float(t["length_mm"]) for t in t_ok])
    mld = np.array([_float(by[t["case_id"]]["mld_mm"]) - float(t["mld_mm"]) for t in t_ok])
    tree_ok = [t for t in truth if by[t["case_id"]]["tree_status"] == "ok"]
    found = np.array([int(by[t["case_id"]]["n_side_branches"]) for t in tree_ok])
    true_n = np.array([int(t["n_side_branches"]) for t in tree_ok])
    tree_ffr = np.array([float(t["ffr_tree"]) for t in truth])
    alone = np.array([float(t["ffr_main_alone"]) for t in truth])
    cut = cfg.ffr_cutoff
    flips = int(((alone <= cut) != (tree_ffr <= cut)).sum())
    lines = [
        "## Against the known truth (synthetic only)",
        "",
        "Masks run, single-path QCA of the main vessel, median (interquartile range):",
        "",
        f"- Path length, measured − true: {_median_iqr(length, 1)} mm ({length.size} vessels). "
        "QCA trims one vessel diameter at each end of the path (M5), and pixel sizes are given "
        "with a 3% error or assumed (0.30 mm/px).",
        f"- Minimum lumen diameter, measured − true: {_median_iqr(mld, 2)} mm.",
    ]
    if found.size:
        lines.append(
            f"- Side branches found by the tree QCA: exactly the true number in "
            f"{np.mean(found == true_n):.0%} of {found.size} vessels; found − true "
            f"{_median_iqr((found - true_n).astype(float), 0)}. Branches shorter than 5 mm or "
            "thinner than 0.3 × their parent are not modelled by design (D35)."
        )
    lines += [
        "",
        "Inside the reference itself (true geometry, the reference's own pressures and "
        "velocity): the main vessel alone as a single path vs the tree with its side branches:",
        "",
        f"- Mean FFR difference, main alone − tree: {np.mean(alone - tree_ffr):+.3f} "
        f"(SD {np.std(alone - tree_ffr, ddof=1):.3f}); the classification at {cut:.2f} differs "
        f"in {flips} of {len(truth)} vessels.",
        "",
        "So a single-path index test is biased low against this reference by construction: "
        "the comparison of single-path and tree index tests above shows the effect of side "
        "branches *under this model*, not which is closer to real wire FFR.",
        "",
    ]
    by_cal = {}
    for t in truth:
        r = by[t["case_id"]]
        if r["tree_status"] != "ok":
            continue
        key = "assumed (0.30 mm/px)" if r["pixel_size_assumed"] == "True" else "given"
        err = _float(r[cfg.primary_index_test]) - float(t["ffr_tree"])
        by_cal.setdefault(key, []).append(err)
    if by_cal:
        lines += [
            f"Primary index test by pixel-size calibration ({SHORT[cfg.primary_index_test]}):",
            "",
            "| Pixel size | Vessels | Mean difference | Mean absolute difference |",
            "|---|---:|---:|---:|",
            *(
                f"| {k} | {len(v)} | {np.mean(v):+.3f} | {np.mean(np.abs(v)):.3f} |"
                for k, v in sorted(by_cal.items())
            ),
            "",
        ]
    return lines


def _error_budget(
    data: Path, rows, truth: list[dict[str, str]], physics: PhysicsParams, cfg: ValidationConfig
) -> list[str]:
    """Where the tree index test's error comes from (masks run): the full pipeline; measured
    geometry with the true pixel size, pressures and velocity (measurement error only); and the
    true geometry with the pipeline's fixed Pv and velocity (assumption error only)."""
    by = {r["case_id"]: r for r in rows}
    cases = {c.case_id: c for c in read_cases(data / "cases.csv")}
    cut = cfg.ffr_cutoff
    full, geometry, assumptions, details = [], [], [], []
    for t in truth:
        cid = t["case_id"]
        if by[cid]["tree_status"] != "ok":
            continue
        c, ref = cases[cid], float(t["ffr_tree"])
        mask = cv2.imread(str(resolve(data, c.mask)), cv2.IMREAD_GRAYSCALE) / 255.0
        true_inputs = physics.model_copy(update={"pa_mmhg": c.pa_mmhg,
                                                 "pv_mmhg": float(t["pv_mmhg"]),
                                                 "v_hyp": float(t["v_hyp"])})  # fmt: skip
        try:
            g = analyse_tree(mask, c.start, c.end, params=true_inputs,
                             mm_per_px=float(t["true_mm_per_px"]))  # fmt: skip
        except ValueError:
            continue
        full.append((_float(by[cid]["tree_solver"]), ref))
        geometry.append((float(g.result.ffr[0][-1]), ref))
        assumptions.append((float(t["ffr_tree_fixed_inputs"]), ref))
        main = g.qca.tree.branches[0]
        details.append({
            "case_id": cid, "error": geometry[-1][0] - ref, "ref": ref,
            "inlet_true": float(t["ref_inlet_mm"]),
            "inlet": 2e3 * float(main.reference_radius()[0]),
            "mld_true": float(t["mld_mm"]), "mld": g.qca.branches[0].mld_mm,
            "sides_true": int(t["n_side_branches"]), "sides": g.qca.tree.n_branches - 1,
        })  # fmt: skip
    if not full:
        return []

    def row(name: str, pairs) -> str:
        p, r = np.array(pairs).T
        d = p - r
        wrong = int(((p <= cut) != (r <= cut)).sum())
        return (f"| {name} | {d.mean():+.3f} | {d.std(ddof=1):.3f} | {np.abs(d).mean():.3f} | "
                f"{wrong} of {d.size} |")  # fmt: skip

    return [
        "### Error budget of the tree index test",
        "",
        "The same vessels (masks run), each error against the stand-in reference:",
        "",
        "| Inputs to the tree solver | Bias | SD | MAE | Misclassified at "
        f"{cut:.2f} |",
        "|---|---:|---:|---:|---:|",
        row("Full pipeline (measured geometry; fixed Pv and velocity; pixel size as given or "
            "assumed)", full),
        row("Measured geometry, true pixel size, Pv and velocity (measurement error only)",
            geometry),
        row("True geometry, the pipeline's fixed Pv and velocity (assumption error only)",
            assumptions),
        "",
        f"The fixed hyperemic velocity ({physics.v_hyp} m/s; the stand-in samples "
        "0.25–0.45 m/s) and Pv are the error P7 targets with a patient-specific flow estimate.",
        "",
        *_largest_errors(details),
    ]  # fmt: skip


def _largest_errors(details: list[dict[str, Any]], k: int = 6) -> list[str]:
    """The measurement-only errors' largest cases and what was mis-measured. The flow is set
    by the inlet reference diameter (Q = v_hyp · π r_ref(0)², M1), so its error matters most."""
    err = np.array([d["error"] for d in details])
    rel = np.array([d["inlet"] / d["inlet_true"] - 1.0 for d in details])
    rho = stats.spearmanr(rel, err).statistic if len(details) > 2 else float("nan")
    missed = sum(d["sides"] < d["sides_true"] for d in details)
    lines = [
        "Measurement error only: the inlet reference diameter sets the flow (Q = v_hyp · π "
        f"r_ref(0)², M1). Its relative error (measured / true − 1) is "
        f"{_median_iqr(rel * 100, 1)} %, and its Spearman correlation with the FFR error is "
        f"{rho:+.2f}. Side branches missed: {missed} of {len(details)} vessels (a missed "
        "branch's flow goes down the main vessel and lowers its FFR). Largest errors:",
        "",
        "| Vessel | Reference FFR | Error | Inlet reference, true → measured (mm) | "
        "MLD, true → measured (mm) | Side branches, true → found |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d in sorted(details, key=lambda d: -abs(d["error"]))[:k]:
        lines.append(
            f"| {d['case_id']} | {d['ref']:.3f} | {d['error']:+.3f} | {d['inlet_true']:.2f} → "
            f"{d['inlet']:.2f} | {d['mld_true']:.2f} → {d['mld']:.2f} | {d['sides_true']} → "
            f"{d['sides']} |"
        )
    return [*lines, ""]


def _segmenter_section(rows, meta, refs, cfg: ValidationConfig) -> list[str]:
    s = score_predictions(rows, refs, cfg, meta.get("index_tests"))
    ffr = s["index_tests"].get("FFR", {})
    lines = [
        "## With the automatic segmenter",
        "",
        "The same cases, with the vessel segmented by the ARCADE-trained model instead of taken "
        "from the true mask. The rendered images match ARCADE's vessel contrast and noise "
        "(D39), but not its anatomy: one vessel tree on a smooth background, without the "
        "catheter, the other coronary vessels, ribs or the diaphragm. The segmenter misses many "
        "of these vessels, so the numbers below show a domain gap between rendered and clinical "
        "images (to be closed by P8's rendered angiograms), not the segmenter's accuracy on "
        "clinical angiograms (`reports/seg_eval.md`).",
        "",
        "| Index test | Analysable / eligible | AUC | Accuracy | Accuracy, ITD | MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for test, r in ffr.items():
        lines.append(
            f"| {INDEX_TESTS[test]} | {r['n']} / {s['eligible_by_type']['FFR']} "
            f"({r['feasibility']:.0%}) | {r.get('auc', float('nan')):.3f} "
            f"({ci_text(r.get('auc_ci'))}) | {r.get('accuracy', float('nan')):.3f} | "
            f"{r['accuracy_itd']:.3f} | {r.get('mae', float('nan')):.3f} |"
        )
    reasons: dict[str, int] = {}
    for r in rows:
        if r["single_status"] != "ok":
            key = r["single_error"].split(":")[0] + ": " + (
                "point off the segmented vessel" if "nearest segmented vessel" in r["single_error"]
                else "no path between the points" if "no vessel path" in r["single_error"]
                else "no vessel found" if "no skeleton" in r["single_error"]
                else r["single_error"].split(": ", 1)[-1][:80]
            )  # fmt: skip
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        lines += ["", "Why single-path analysis failed:", ""]
        lines += [f"- {n} × {k}" for k, n in sorted(reasons.items(), key=lambda x: -x[1])]
    return [*lines, ""]


def run_synthetic_validation(
    study: SyntheticStudyConfig,
    cfg: ValidationConfig,
    physics: PhysicsParams,
    *,
    seg_dir: str | Path | None,
    surrogate_dir: str | Path | None,
    uncertainty_path: str | Path,
    data_dir: str | Path | None = None,
    out_md: str | Path = REPORT,
    fig_path: str | Path = FIGURE,
    predictions_dir: str | Path = PREDICTIONS_DIR,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Generate the stand-in study, predict (masks; segmenter if given), score and report."""
    git = git_info()  # before anything is written
    gen = generate_synthetic_study(study, physics, data_dir)
    data = Path(gen["out_dir"])
    ucfg, qca_error = load_uncertainty(uncertainty_path)
    surrogate = None if surrogate_dir is None else SurrogateRunner.from_dir(surrogate_dir)
    seg = None if seg_dir is None else SegmentationRunner.from_dir(seg_dir)
    models = {"segmentation": model_info(seg_dir, SEG_ONNX_FILE),
              "surrogate": model_info(surrogate_dir, ONNX_FILE)}  # fmt: skip
    common = dict(
        segmenter=seg, surrogate=surrogate, params=physics, uncertainty=ucfg, qca_error=qca_error,
        models=models, allow_tracked_output=True, git=git,
    )  # fmt: skip
    arms = [("masks", True)] + ([("segmenter", False)] if seg is not None else [])
    paths = {}
    for arm, use_masks in arms:
        cb = None if progress is None else (lambda k, n, _row, arm=arm: progress(arm, k, n))
        paths[arm] = run_predictions(data / "cases.csv", predictions_dir, use_masks=use_masks,
                                     name=f"predictions_{arm}", progress=cb, **common)  # fmt: skip
    if surrogate is not None:
        surrogate.close()
    refs = read_reference(data / "reference.csv")
    with (data / "truth.csv").open(encoding="utf-8", newline="") as f:
        truth = list(csv.DictReader(f))
    rows, meta = read_predictions(paths["masks"])
    extra = _truth_section(rows, truth, cfg) + _error_budget(data, rows, truth, physics, cfg)
    if "segmenter" in paths:
        seg_rows, seg_meta = read_predictions(paths["segmenter"])
        extra += _segmenter_section(seg_rows, seg_meta, refs, cfg)
    else:
        extra += ["## With the automatic segmenter", "",
                  "Not run: no exported segmentation model (`models/seg/`).", ""]  # fmt: skip
    intro = [
        "**Synthetic stand-in data, not a clinical validation.** This report shows that the "
        "validation tool (`vstent validate`, `docs/validation/`) runs end to end, from a case "
        "list to the statistics a study would report. Every number below comes from rendered "
        "synthetic cases (`vstent validate-synthetic`, `configs/validation_synthetic.yaml`) "
        "and says nothing about agreement with real wire FFR.",
        "",
        f"The study: {gen['n_patients']} synthetic patients with {gen['n_cases']} vessels (one "
        "to three each), rendered as 512 × 512 X-ray-like projections. Each vessel is a "
        'synthetic tree (P2 sampler; 0–3 side branches). The stand-in "wire FFR" is the tree '
        "solver's FFR at the end of the main vessel on the true geometry, with the patient's "
        "sampled Pa and Pv and the vessel's sampled hyperemic velocity. The case list gives "
        "the pipeline only what a study records: the image, the start point and the sensor "
        "position (each with a 1.5 px click error), the measured Pa, and for "
        f"{gen['calibrated_cases']} of {gen['n_cases']} vessels a calibrated pixel size (3% "
        "error); the others assume 0.30 mm/px. Pv and the velocity are the pipeline's fixed "
        "defaults. The main analysis uses the true lumen masks (standing in for "
        "analyst-corrected contours).",
        "",
        "What it measures: the pipeline's image → geometry → FFR error under its own physics, "
        "plus the effect of its fixed assumptions (velocity, Pv, pixel size). The reference "
        "comes from the same tree physics as the tree index test, so tree vs reference "
        "agreement is optimistic by construction.",
    ]
    summary = write_validation_report(
        rows, meta, refs, cfg, out_md, fig_path,
        title="Validation tool on synthetic stand-in cases", intro=intro, extra=extra,
    )  # fmt: skip
    return {"study": gen, "summary": summary, "predictions": {k: str(v) for k, v in paths.items()}}
