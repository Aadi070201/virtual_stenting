"""The validation run (P4): predictions on a case list, then scoring against the reference.

Two steps, so that the analysis can be locked before anyone sees the wire values
(``docs/validation/PROTOCOL.md``):

1. :func:`run_predictions` runs the pipeline on every vessel of ``cases.csv`` and writes
   ``predictions.csv`` with ``predictions_meta.json`` (software version, git commit, model and
   manifest hashes, and the SHA-256 of ``predictions.csv`` that locks it). A vessel that cannot be
   analysed is recorded with the reason; nothing is dropped.
2. :func:`score_predictions` joins the locked predictions with ``reference.csv`` and computes
   the accuracy and agreement statistics per index test (:data:`INDEX_TESTS`).

The pipeline is the one of ``vstent analyse``: segmentation (or the case's mask), single-path
QCA with the solver, the surrogate and the P1 grey-zone intervals, and tree QCA with the tree
solver. Data and results must stay outside the repository's tracked files
(:func:`vstent.validation.manifest.check_local_only`). Torch-free.
"""

import csv
import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import Field

from vstent import DISCLAIMER, __version__
from vstent.config import StrictModel, git_info
from vstent.imaging.image_io import read_image_file
from vstent.inference.pipeline import Segmenter, analyse_tree, analyse_vessel
from vstent.physics.solver import PhysicsParams
from vstent.uncertainty.greyzone import UncertaintyConfig, assessment
from vstent.uncertainty.inputs import InputUncertainty, Measurement, QCAErrorModel
from vstent.validation.manifest import (
    CaseRow,
    ReferenceRow,
    case_files,
    check_local_only,
    read_cases,
    resolve,
)
from vstent.validation.sample_size import SampleSizeConfig
from vstent.validation.stats import (
    auc_delong,
    auc_interval,
    binary_metrics,
    bland_altman,
    cluster_bootstrap,
    pearson,
    wilson,
)

VALIDATION_CONFIG = Path("configs/validation.yaml")
PREDICTIONS_NAME = "predictions"  # → predictions.csv + predictions_meta.json

IndexTest = Literal["single_solver", "single_surrogate", "tree_solver"]
INDEX_TESTS: dict[str, str] = {
    "single_solver": "single path, reduced-order solver",
    "single_surrogate": "single path, surrogate (ensemble mean)",
    "tree_solver": "tree with side branches, reduced-order solver",
}
MODE = {"single_solver": "single", "single_surrogate": "single", "tree_solver": "tree"}
PREDICTION_COLUMNS = (
    "case_id", "patient_id", "vessel", "single_status", "single_error", "tree_status",
    "tree_error", "single_solver", "single_surrogate", "tree_solver", "model_low", "model_high",
    "input_low", "input_high", "grey_zone", "length_mm", "mld_mm", "rvd_mm", "max_ds_pct",
    "ref_inlet_mm", "n_side_branches", "mm_per_px", "pixel_size_assumed", "mask_used",
    "seconds", "warnings",
)  # fmt: skip


class ValidationConfig(StrictModel):
    """``configs/validation.yaml``: the pre-specified analysis and the sample-size assumptions."""

    primary_index_test: IndexTest = "tree_solver"
    ffr_cutoff: float = Field(0.80, gt=0, lt=1)  # index test and wire FFR
    ifr_cutoff: float = Field(0.89, gt=0, lt=1)  # wire iFR (classification only)
    level: float = Field(0.95, gt=0, lt=1)
    n_bootstrap: int = Field(2000, ge=100)  # patient-level cluster bootstrap
    seed: int = 0
    sample_size: SampleSizeConfig = SampleSizeConfig()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _probability(c: CaseRow, base: Path, segmenter: Segmenter | None, use_masks: bool):
    """Vessel probability map of a case: its mask if given (and used), else the segmenter's."""
    image = read_image_file(resolve(base, c.image), frame=c.frame).image
    if use_masks and c.mask:
        m = cv2.imread(str(resolve(base, c.mask)), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise ValueError(f"{c.mask}: not a readable mask image")
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if m.shape != image.shape:
            raise ValueError(f"mask {m.shape} and image {image.shape} differ in size")
        return m.astype(np.float32) / float(np.iinfo(m.dtype).max), True
    if segmenter is None:
        raise ValueError("no segmentation model and no mask")
    return segmenter.predict(image), False


def predict_case(
    c: CaseRow,
    base: Path,
    *,
    segmenter: Segmenter | None,
    surrogate,
    params: PhysicsParams,
    uncertainty: UncertaintyConfig,
    qca_error: QCAErrorModel | None,
    use_masks: bool = True,
) -> dict[str, Any]:
    """One row of ``predictions.csv`` (see :data:`PREDICTION_COLUMNS`); failures are recorded
    with their reason, never raised."""
    t0 = time.perf_counter()
    row: dict[str, Any] = {k: "" for k in PREDICTION_COLUMNS}
    row.update(case_id=c.case_id, patient_id=c.patient_id, vessel=c.vessel)
    pressures = {k: v for k, v in (("pa_mmhg", c.pa_mmhg), ("pv_mmhg", c.pv_mmhg)) if v}
    p = params.model_copy(update=pressures)
    warnings: list[str] = []
    try:
        prob, mask_used = _probability(c, base, segmenter, use_masks)
    except Exception as exc:  # unreadable input: neither mode can run
        err = f"input: {type(exc).__name__}: {exc}"
        row.update(single_status="failed", single_error=err, tree_status="failed", tree_error=err)
        row["seconds"] = round(time.perf_counter() - t0, 3)
        return row
    row["mask_used"] = mask_used
    try:
        a = analyse_vessel(prob, c.start, c.end, params=p, surrogate=surrogate,
                           mm_per_px=c.mm_per_px)  # fmt: skip
        q = a.qca
        ref = q.profile.reference_radius()  # m; the inlet value sets the flow (M1)
        i_mld = int(np.argmin(q.profile.r))
        mc = InputUncertainty(
            q.profile, p, a.solver.pa, a.solver.pv,
            measurement=Measurement(q.mm_per_px, q.calibration_assumed),
            config=uncertainty.inputs, qca_error=qca_error,
        )  # fmt: skip
        pre = assessment(uncertainty.pre_pci_threshold, a.ffr_distal_solver,
                         a.ffr_distal_surrogate, a.model_interval, mc.pre_samples,
                         uncertainty.level)  # fmt: skip
        row.update(
            single_status="ok",
            single_solver=round(a.ffr_distal_solver, 4),
            single_surrogate="" if a.ffr_distal_surrogate is None
            else round(a.ffr_distal_surrogate, 4),
            model_low="" if pre.model is None else round(pre.model.low, 4),
            model_high="" if pre.model is None else round(pre.model.high, 4),
            input_low=round(pre.inputs.low, 4), input_high=round(pre.inputs.high, 4),
            grey_zone=pre.flagged,
            length_mm=round(float(q.s_mm[-1]), 2), mld_mm=round(q.mld_mm, 3),
            rvd_mm=round(2e3 * float(ref[i_mld]), 3), max_ds_pct=round(q.max_percent_stenosis, 1),
            ref_inlet_mm=round(2e3 * float(ref[0]), 3),
            mm_per_px=round(q.mm_per_px, 5), pixel_size_assumed=q.calibration_assumed,
        )  # fmt: skip
        warnings += a.warnings
    except Exception as exc:
        row.update(single_status="failed", single_error=f"{type(exc).__name__}: {exc}")
    try:
        t = analyse_tree(prob, c.start, c.end, params=p, mm_per_px=c.mm_per_px)
        row.update(tree_status="ok", tree_solver=round(float(t.result.ffr[0][-1]), 4),
                   n_side_branches=t.qca.tree.n_branches - 1)  # fmt: skip
        warnings += [f"tree: {w}" for w in t.warnings]
    except Exception as exc:
        row.update(tree_status="failed", tree_error=f"{type(exc).__name__}: {exc}")
    row["warnings"] = " | ".join(warnings)
    row["seconds"] = round(time.perf_counter() - t0, 3)
    return row


def run_predictions(
    manifest: str | Path,
    out_dir: str | Path,
    *,
    segmenter: Segmenter | None,
    surrogate,
    params: PhysicsParams,
    uncertainty: UncertaintyConfig,
    qca_error: QCAErrorModel | None,
    use_masks: bool = True,
    models: dict[str, dict[str, str] | None] | None = None,
    allow_tracked_output: bool = False,
    name: str = PREDICTIONS_NAME,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    git: dict[str, Any] | None = None,
) -> Path:
    """Predict every vessel of ``manifest`` and lock the result in ``out_dir`` as
    ``<name>.csv`` + ``<name>_meta.json``; returns the CSV path. ``models`` describes the model
    files (see :func:`model_info`); ``git`` the commit state, if taken before earlier outputs
    of the same run were written."""
    git = git or git_info()  # before writing anything
    manifest = Path(manifest)
    cases = read_cases(manifest)
    base = manifest.parent
    check_local_only([manifest, *case_files(cases, base)], "Case lists, images and masks")
    out_dir = Path(out_dir)
    if not allow_tracked_output:
        check_local_only([out_dir], "Validation output")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, c in enumerate(cases):
        rows.append(predict_case(c, base, segmenter=segmenter, surrogate=surrogate, params=params,
                                 uncertainty=uncertainty, qca_error=qca_error,
                                 use_masks=use_masks))  # fmt: skip
        if progress is not None:
            progress(k + 1, len(cases), rows[-1])
    path = out_dir / f"{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PREDICTION_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    meta = {
        "disclaimer": DISCLAIMER,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "vstent_version": __version__,
        **git,
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "n_cases": len(cases),
        },  # fmt: skip
        "use_masks": use_masks,
        "index_tests": [t for t in INDEX_TESTS if t != "single_surrogate" or surrogate is not None],
        "models": models or {},
        "physics": params.model_dump(mode="json"),
        "uncertainty": uncertainty.model_dump(mode="json"),
        "predictions_file": path.name,
        "predictions_sha256": sha256_file(path),
    }
    meta_path(path).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def meta_path(predictions: str | Path) -> Path:
    """``<name>_meta.json`` next to ``<name>.csv``."""
    p = Path(predictions)
    return p.with_name(p.stem + "_meta.json")


def model_info(path: str | Path | None, filename: str) -> dict[str, str] | None:
    """Directory, file name and SHA-256 of an exported model (``None`` if not used)."""
    if path is None:
        return None
    return {"dir": str(path), "file": filename, "sha256": sha256_file(Path(path) / filename)}


def read_predictions(path: str | Path) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Rows of a predictions CSV and its run metadata (``None`` if missing). Raises
    ``ValueError`` if the file no longer matches the SHA-256 recorded when it was written."""
    path = Path(path)
    meta_file = meta_path(path)
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else None
    if meta is not None and meta.get("predictions_sha256") != sha256_file(path):
        raise ValueError(f"{path} was changed after it was locked (SHA-256 differs from "
                         f"{meta_file.name})")  # fmt: skip
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f)), meta


def _float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _accuracy(
    pred: np.ndarray,
    ref: np.ndarray,
    patients: np.ndarray,
    ref_cutoff: float,
    cfg: ValidationConfig,
    agreement: bool,
) -> dict[str, Any]:
    """Classification at the cut-offs, AUC and (``agreement``) Bland–Altman, MAE and r, each
    with analytic and patient-level bootstrap intervals."""
    ref_pos = ref <= ref_cutoff
    out: dict[str, Any] = {
        "n": int(pred.size),
        "n_patients": int(np.unique(patients).size),
        "n_positive": int(ref_pos.sum()),
        "prevalence": float(ref_pos.mean()) if pred.size else float("nan"),
    }
    if pred.size == 0:
        return out
    auc, se = auc_delong(-pred, ref_pos)
    out.update(auc=auc, auc_se=se, auc_ci=auc_interval(auc, se, cfg.level))
    out.update(binary_metrics(pred <= cfg.ffr_cutoff, ref_pos, cfg.level))
    if agreement and pred.size >= 3:
        d = pred - ref
        out["bland_altman"] = bland_altman(pred, ref, cfg.level)
        out["mae"] = float(np.abs(d).mean())
        out["r"], out["r_ci"] = pearson(pred, ref, cfg.level)

    def stat(idx: np.ndarray) -> dict[str, float]:
        p, r = pred[idx], ref[idx]
        rp, tp = r <= ref_cutoff, p <= cfg.ffr_cutoff
        vals = {
            "auc": auc_delong(-p, rp)[0],
            "sensitivity": (tp & rp).sum() / rp.sum() if rp.any() else np.nan,
            "specificity": (~tp & ~rp).sum() / (~rp).sum() if (~rp).any() else np.nan,
            "accuracy": float((tp == rp).mean()),
        }
        if agreement:
            vals["bias"] = float((p - r).mean())
            vals["mae"] = float(np.abs(p - r).mean())
        return vals

    out["bootstrap"] = {
        k: [lo, hi, n]
        for k, (lo, hi, n) in cluster_bootstrap(
            patients, stat, cfg.n_bootstrap, cfg.seed, cfg.level
        ).items()
    }
    return out


def score_predictions(
    rows: list[dict[str, str]],
    refs: list[ReferenceRow],
    cfg: ValidationConfig,
    tests: list[str] | None = None,
) -> dict[str, Any]:
    """Accuracy and agreement of every index test against the reference (JSON-ready dict).

    Flow: vessels in the manifest → with a reference → not excluded (eligible) → analysable
    per index test. Intention-to-diagnose accuracy counts eligible vessels the index test could
    not analyse as misclassified. ``tests``: the index tests the run computed (its metadata);
    by default those with at least one value."""
    if tests is None:
        tests = [t for t in INDEX_TESTS if any(np.isfinite(_float(r[t])) for r in rows)]
    pred_by = {r["case_id"]: r for r in rows}
    ref_by = {r.case_id: r for r in refs}
    excluded = [{"case_id": r.case_id, "reason": r.excluded} for r in refs
                if r.case_id in pred_by and r.excluded]  # fmt: skip
    eligible = [cid for cid in pred_by if cid in ref_by and not ref_by[cid].excluded]
    out: dict[str, Any] = {
        "config": cfg.model_dump(mode="json", exclude={"sample_size"}),
        "flow": {
            "manifest": len(rows),
            "no_reference": sorted(set(pred_by) - set(ref_by)),
            "reference_without_prediction": sorted(set(ref_by) - set(pred_by)),
            "excluded": excluded,
            "eligible": len(eligible),
            "eligible_patients": len({pred_by[c]["patient_id"] for c in eligible}),
        },
        "index_tests": {},
        "eligible_by_type": {},
    }
    for ref_type, ref_cut in (("FFR", cfg.ffr_cutoff), ("iFR", cfg.ifr_cutoff)):
        ids = [c for c in eligible if ref_by[c].ref_type == ref_type]
        if not ids:
            continue
        out["eligible_by_type"][ref_type] = len(ids)
        per_test = {}
        for test in tests:
            values = np.array([_float(pred_by[c][test]) for c in ids])
            status = np.array([pred_by[c][f"{MODE[test]}_status"] == "ok" for c in ids])
            ok = status & np.isfinite(values)
            sel = [c for c, k in zip(ids, ok, strict=True) if k]
            pred = values[ok]
            ref = np.array([ref_by[c].ref_value for c in sel])
            patients = np.array([pred_by[c]["patient_id"] for c in sel])
            res = _accuracy(pred, ref, patients, ref_cut, cfg, agreement=ref_type == "FFR")
            failed = [{"case_id": c, "error": pred_by[c][f"{MODE[test]}_error"]}
                      for c, k in zip(ids, ok, strict=True) if not k]  # fmt: skip
            res["failed"] = failed
            n_all = len(ids)
            correct = res.get("tp", 0) + res.get("tn", 0)
            res["feasibility"] = len(sel) / n_all
            res["feasibility_ci"] = wilson(len(sel), n_all, cfg.level)
            res["accuracy_itd"] = correct / n_all
            res["accuracy_itd_ci"] = wilson(correct, n_all, cfg.level)
            per_test[test] = res
        out["index_tests"][ref_type] = per_test
        if ref_type == "FFR":
            out["grey_zone"] = _grey_zone(ids, pred_by, ref_by, cfg)
            out["territories"] = _territories(ids, pred_by, ref_by, cfg)
    return out


def _grey_zone(ids, pred_by, ref_by, cfg: ValidationConfig) -> dict[str, Any] | None:
    """P1 grey-zone flags (single path): accuracy of the solver's call with and without a flag,
    and a hybrid strategy where flagged vessels get the wire (counted as correct)."""
    sel = [c for c in ids if pred_by[c]["single_status"] == "ok" and pred_by[c]["grey_zone"]]
    if not sel:
        return None
    flag = np.array([pred_by[c]["grey_zone"] == "True" for c in sel])
    pred = np.array([_float(pred_by[c]["single_solver"]) for c in sel])
    ref = np.array([ref_by[c].ref_value for c in sel])
    correct = (pred <= cfg.ffr_cutoff) == (ref <= cfg.ffr_cutoff)
    n, nf = len(sel), int(flag.sum())
    k_unflagged, k_flagged = int(correct[~flag].sum()), int(correct[flag].sum())
    hybrid = k_unflagged + nf
    return {
        "n": n,
        "n_flagged": nf,
        "share_flagged": nf / n,
        "accuracy_unflagged": k_unflagged / (n - nf) if n > nf else float("nan"),
        "accuracy_unflagged_ci": wilson(k_unflagged, n - nf, cfg.level),
        "accuracy_flagged": k_flagged / nf if nf else float("nan"),
        "accuracy_flagged_ci": wilson(k_flagged, nf, cfg.level),
        "accuracy_all": float(correct.mean()),
        "accuracy_hybrid": hybrid / n,
        "accuracy_hybrid_ci": wilson(hybrid, n, cfg.level),
    }


def _territories(ids, pred_by, ref_by, cfg: ValidationConfig) -> list[dict[str, Any]]:
    """Primary index test per vessel label: n, prevalence, MAE, bias, sensitivity, specificity,
    AUC (when both classes occur)."""
    test = cfg.primary_index_test
    status = f"{MODE[test]}_status"
    rows = []
    for label in sorted({pred_by[c]["vessel"] for c in ids}):
        sel = [c for c in ids if pred_by[c]["vessel"] == label and pred_by[c][status] == "ok"
               and np.isfinite(_float(pred_by[c][test]))]  # fmt: skip
        n_all = sum(pred_by[c]["vessel"] == label for c in ids)
        if not sel:
            rows.append({"vessel": label or "(none)", "n": 0, "n_eligible": n_all})
            continue
        p = np.array([_float(pred_by[c][test]) for c in sel])
        r = np.array([ref_by[c].ref_value for c in sel])
        bm = binary_metrics(p <= cfg.ffr_cutoff, r <= cfg.ffr_cutoff, cfg.level)
        auc, _ = auc_delong(-p, r <= cfg.ffr_cutoff)
        rows.append({
            "vessel": label or "(none)", "n": len(sel), "n_eligible": n_all,
            "prevalence": float((r <= cfg.ffr_cutoff).mean()), "mae": float(np.abs(p - r).mean()),
            "bias": float((p - r).mean()), "sensitivity": bm["sensitivity"],
            "specificity": bm["specificity"], "auc": auc,
        })  # fmt: skip
    return rows
