"""P4: validation tool — statistics, sample size, manifests, stand-in cases, run, report, CLI."""

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps
from typer.testing import CliRunner

from vstent.cli import app
from vstent.data.synth_tree import TreeSamplingConfig, sample_tree
from vstent.imaging.synthetic import render_projection
from vstent.inference.onnx_runtime import SurrogateRunner
from vstent.physics.solver import PhysicsParams
from vstent.uncertainty.greyzone import UncertaintyConfig
from vstent.uncertainty.inputs import InputUncertaintyConfig
from vstent.validation import manifest as mf
from vstent.validation import stats
from vstent.validation.report import write_validation_report
from vstent.validation.run import (
    INDEX_TESTS,
    ValidationConfig,
    read_predictions,
    run_predictions,
    score_predictions,
)
from vstent.validation.sample_size import (
    SampleSizeConfig,
    hanley_mcneil_variance,
    n_auc,
    n_loa,
    n_sens_spec,
    sample_size,
)
from vstent.validation.synthetic import (
    SyntheticStudyConfig,
    _clear,
    generate_synthetic_study,
    layout_tree,
)

PARAMS = PhysicsParams()
FAST_UNCERTAINTY = UncertaintyConfig(inputs=InputUncertaintyConfig(n_samples=40))
IN_GIT = mf._repo_root() is not None


def small_study_config(n_patients: int = 3) -> SyntheticStudyConfig:
    return SyntheticStudyConfig(
        n_patients=n_patients, image_size=256, mm_per_px=(0.25, 0.25), stub_mm=3.0,
        continuation_mm=(3.0, 5.0), calibrated_fraction=1.0,
        sampling=TreeSamplingConfig(length_mm=(25.0, 32.0), d0_mm=(3.0, 3.5),
                                    side_count_probs=[0.5, 0.5, 0.0, 0.0, 0.0],
                                    side_length_mm=(8.0, 14.0)),
    )  # fmt: skip


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    out = tmp_path_factory.mktemp("study")
    summary = generate_synthetic_study(small_study_config(), PARAMS, out)
    return out, summary


# --- statistics ------------------------------------------------------------------------------


def test_wilson_matches_the_textbook_value():
    lo, hi = stats.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert all(math.isnan(v) for v in stats.wilson(0, 0))
    assert stats.wilson(10, 10)[1] == pytest.approx(1.0)


def test_auc_equals_mann_whitney_and_delong_se_matches_simulation():
    rng = np.random.default_rng(0)
    score = np.round(rng.normal(size=60), 1)  # ties included
    pos = rng.random(60) < 0.4
    auc, _ = stats.auc_delong(score, pos)
    u = sps.mannwhitneyu(score[pos], score[~pos]).statistic
    assert auc == pytest.approx(u / (pos.sum() * (~pos).sum()))
    # DeLong SE vs the spread of the AUC over repeated binormal samples (AUC ~0.85)
    aucs, ses = [], []
    for _ in range(300):
        x, y = rng.normal(1.47, 1, 40), rng.normal(0, 1, 40)
        a, se = stats.auc_delong(np.r_[x, y], np.r_[np.ones(40, bool), np.zeros(40, bool)])
        aucs.append(a)
        ses.append(se)
    assert np.mean(ses) == pytest.approx(np.std(aucs), rel=0.2)
    assert all(math.isnan(v) for v in stats.auc_delong(score, np.zeros(60, bool)))


def test_auc_interval_stays_in_range():
    lo, hi = stats.auc_interval(0.95, 0.03)
    assert 0 < lo < 0.95 < hi < 1
    assert stats.auc_interval(1.0, 0.0) == (1.0, 1.0)
    fpr, tpr = stats.roc_points(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 0, 1, 0], bool))
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)


def test_binary_metrics_on_a_known_table():
    test = np.array([1, 1, 1, 0, 0, 0, 0, 1], bool)
    ref = np.array([1, 1, 0, 1, 0, 0, 0, 1], bool)
    m = stats.binary_metrics(test, ref)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 3)
    assert m["sensitivity"] == 0.75 and m["specificity"] == 0.75 and m["accuracy"] == 0.75
    assert m["sensitivity_ci"] == stats.wilson(3, 4)


def test_bland_altman_limits_and_their_intervals():
    ref = np.linspace(0.5, 0.95, 50)
    d = np.random.default_rng(1).normal(0.02, 0.05, 50)
    ba = stats.bland_altman(ref + d, ref)
    assert ba["bias"] == pytest.approx(d.mean()) and ba["sd"] == pytest.approx(d.std(ddof=1))
    assert ba["loa_low"] == pytest.approx(d.mean() - 1.96 * d.std(ddof=1))
    t = sps.t.ppf(0.975, 49)
    se = d.std(ddof=1) * math.sqrt(3 / 50)
    assert ba["loa_high_ci"][1] - ba["loa_high"] == pytest.approx(t * se)
    with pytest.raises(ValueError):
        stats.bland_altman(ref[:2], ref[:2])
    r, ci = stats.pearson(ref, 2 * ref)
    assert r == pytest.approx(1.0) and all(math.isnan(v) for v in ci)


def test_cluster_bootstrap_resamples_whole_patients():
    clusters = np.repeat(np.arange(20), 2)  # two vessels per patient
    x = np.repeat(np.random.default_rng(2).normal(size=20), 2)  # identical within a patient

    def stat(idx):
        # every drawn vessel's partner must be drawn the same number of times
        counts = np.bincount(idx, minlength=40)
        return {"pairs": float(np.all(counts[0::2] == counts[1::2])), "mean": x[idx].mean()}

    out = stats.cluster_bootstrap(clusters, stat, n_boot=300, seed=0)
    assert out["pairs"][:2] == (1.0, 1.0) and out["mean"][2] == 300
    assert out["mean"][0] < x.mean() < out["mean"][1]
    assert stats.cluster_bootstrap(clusters, stat, 300, seed=0) == out  # seeded


# --- sample size -----------------------------------------------------------------------------


def test_sample_size_formulas_match_hand_calculations():
    # Buderer: 1.96² · 0.85 · 0.15 / (0.075² · 0.35) = 248.8 → 249
    assert n_sens_spec(0.85, 0.075, 0.35) == 249
    assert n_sens_spec(0.85, 0.075, 0.65) == 134
    assert n_loa(0.07, 0.03) == 63  # 3 · (1.96 · 0.07 / 0.03)² = 62.7
    # at AUC 0.5 Hanley–McNeil equals the exact Mann–Whitney null variance
    assert hanley_mcneil_variance(0.5, 30, 50) == pytest.approx(81 / (12 * 30 * 50))
    assert n_auc(0.90, 0.80, 0.35) < n_auc(0.85, 0.80, 0.35)
    res = sample_size(SampleSizeConfig())
    assert res["governing"] == "sensitivity" and res["design_effect"] == pytest.approx(1.03)
    assert res["vessels"] == math.ceil(math.ceil(249 * 1.03) / 0.9)
    assert res["patients"] == math.ceil(res["vessels"] / 1.3)


# --- manifests and the local-only guard ------------------------------------------------------


def _write(path: Path, rows: list[dict], columns=None) -> Path:
    columns = columns or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def test_manifest_parsing_and_errors(tmp_path):
    row = {
        "case_id": "A", "patient_id": "P1", "vessel": "LAD", "image": "a.png", "frame": "",
        "start_row": "10", "start_col": "20", "end_row": "30", "end_col": "40",
        "mm_per_px": "", "calibration": "", "pa_mmhg": "92", "pv_mmhg": "", "mask": "",
    }  # fmt: skip
    cases = mf.read_cases(_write(tmp_path / "c.csv", [row, {**row, "case_id": "B"}]))
    assert cases[0].mm_per_px is None and cases[0].pa_mmhg == 92 and cases[1].end == (30, 40)
    with pytest.raises(mf.ManifestError, match="unknown column"):
        mf.read_cases(_write(tmp_path / "u.csv", [{**row, "mm_per_pixel": "0.3"}]))
    with pytest.raises(mf.ManifestError, match="repeats line 2"):
        mf.read_cases(_write(tmp_path / "d.csv", [row, row]))
    with pytest.raises(mf.ManifestError, match="line 2, start_row"):
        mf.read_cases(_write(tmp_path / "n.csv", [{**row, "start_row": "-1"}]))
    refs = mf.read_reference(_write(tmp_path / "r.csv", [
        {"case_id": "A", "ref_type": "ffr", "ref_value": "0.78", "excluded": ""},
        {"case_id": "B", "ref_type": "iFR", "ref_value": "0.91", "excluded": "wire drift"},
    ]))  # fmt: skip
    assert refs[0].ref_type == "FFR" and refs[1].excluded == "wire drift"
    with pytest.raises(mf.ManifestError, match="ref_type"):
        bad = {"case_id": "A", "ref_type": "Pd/Pa", "ref_value": "0.8", "excluded": ""}
        mf.read_reference(_write(tmp_path / "r2.csv", [bad]))


@pytest.mark.skipif(not IN_GIT, reason="needs the git checkout")
def test_local_only_guard(tmp_path):
    root = mf._repo_root()
    assert mf.tracked_paths([root / "README.md", root / "data" / "x.csv", tmp_path]) == [
        (root / "README.md").resolve()
    ]
    with pytest.raises(ValueError, match="git-ignored"):
        mf.check_local_only([root / "reports" / "patients.csv"], "Validation output")


# --- stand-in cases --------------------------------------------------------------------------


def test_projection_transmission():
    cols = np.arange(10.0, 90.0, 0.25)
    tube = (np.stack([np.full_like(cols, 30.0), cols], axis=1), np.full(cols.size, 6.0))
    img = render_projection((60, 100), [tube], mm_per_px=0.25, mu_per_mm=0.1, background=0.6,
                            blur_sigma=0.0, noise_std=0.0)  # fmt: skip
    assert img[30, 50] == pytest.approx(0.6 * math.exp(-0.1 * 12 * 0.25), rel=2e-3)
    assert img[5, 50] == pytest.approx(0.6)


def test_layout_is_inside_the_image_and_branches_do_not_touch():
    cfg = small_study_config()
    rng = np.random.default_rng(3)
    tree = sample_tree(rng, cfg.sampling.model_copy(
        update={"side_count_probs": [0.0, 1.0, 0.0, 0.0, 0.0]}), 256).tree  # fmt: skip
    lay = layout_tree(rng, tree, 0.25, cfg)
    assert lay is not None and len(lay.tubes) == tree.n_branches == 2
    for pts, rad in lay.tubes:
        assert np.all(pts >= rad[:, None]) and np.all(pts <= cfg.image_size - 1 - rad[:, None])
    (main, r_main), (side, r_side) = lay.tubes
    take_off = side[0]
    far = np.linalg.norm(side - take_off, axis=1) > 2 * (r_main.max() + r_side[0] + 3)
    assert _clear(side[far], r_side[far], [(main, r_main)], cfg.clearance_px)
    # start and sensor are the main centreline's ends; the arc between them is its length
    i0 = np.argmin(np.linalg.norm(main - lay.start_px, axis=1))
    i1 = np.argmin(np.linalg.norm(main - lay.end_px, axis=1))
    arc = np.linalg.norm(np.diff(main[i0 : i1 + 1], axis=0), axis=1).sum()
    assert arc == pytest.approx(tree.branches[0].length / 0.25e-3, abs=1.0)


def test_generated_study(study, tmp_path):
    out, summary = study
    cases = mf.read_cases(out / "cases.csv")
    refs = mf.read_reference(out / "reference.csv")
    assert summary["n_cases"] == len(cases) == len(refs) >= 3
    assert {c.patient_id for c in cases} == {"SYN-001", "SYN-002", "SYN-003"}
    import cv2

    for c in cases:
        img = cv2.imread(str(out / c.image), cv2.IMREAD_GRAYSCALE).astype(float)
        mask = cv2.imread(str(out / c.mask), cv2.IMREAD_GRAYSCALE) > 127
        assert img[mask].mean() < img[~mask].mean() - 8  # vessels are dark (ARCADE-like contrast)
    assert all(0 < r.ref_value <= 1 for r in refs)
    again = generate_synthetic_study(small_study_config(), PARAMS, tmp_path / "again")
    assert (tmp_path / "again" / "cases.csv").read_text() == (out / "cases.csv").read_text()
    assert again["n_cases"] == summary["n_cases"]


# --- run, lock, score, report ----------------------------------------------------------------


@pytest.fixture(scope="module")
def predicted(study, tiny_ensemble_export, tmp_path_factory):
    out, _ = study
    rows = list(csv.DictReader((out / "cases.csv").open(encoding="utf-8")))
    broken = [
        {**rows[0], "case_id": "BAD-IMAGE", "image": "images/missing.png", "mask": ""},
        {**rows[0], "case_id": "OFF-VESSEL", "end_row": "2", "end_col": "2"},
    ]
    manifest = _write(out / "cases_with_failures.csv", rows + broken)
    surrogate = SurrogateRunner.from_dir(tiny_ensemble_export["out_dir"])
    run_dir = tmp_path_factory.mktemp("run")
    path = run_predictions(manifest, run_dir, segmenter=None, surrogate=surrogate, params=PARAMS,
                           uncertainty=FAST_UNCERTAINTY, qca_error=None)  # fmt: skip
    refs = [{"case_id": r["case_id"], "ref_type": r["ref_type"], "ref_value": r["ref_value"],
             "excluded": ""} for r in csv.DictReader((out / "reference.csv").open())]  # fmt: skip
    refs[-1]["excluded"] = "wire drift > 0.03"
    refs[-2]["ref_type"], refs[-2]["ref_value"] = "iFR", "0.85"
    refs += [
        {"case_id": "BAD-IMAGE", "ref_type": "FFR", "ref_value": "0.70", "excluded": ""},
        {"case_id": "OFF-VESSEL", "ref_type": "FFR", "ref_value": "0.90", "excluded": ""},
    ]
    reference = _write(run_dir / "reference.csv", refs)
    return path, reference, len(rows)


def test_predictions_record_failures_with_reasons(predicted):
    path, _, n = predicted
    rows, meta = read_predictions(path)
    by = {r["case_id"]: r for r in rows}
    assert len(rows) == n + 2 and meta["predictions_sha256"] and meta["use_masks"]
    assert meta["index_tests"] == list(INDEX_TESTS)
    good = [r for r in rows if r["case_id"].startswith("SYN")]
    assert all(r["single_status"] == "ok" and r["tree_status"] == "ok" for r in good)
    assert all(r["mask_used"] == "True" and r["model_low"] and r["grey_zone"] for r in good)
    assert by["BAD-IMAGE"]["tree_status"] == "failed"
    assert by["BAD-IMAGE"]["single_error"].startswith("input:")
    assert "nearest segmented vessel" in by["OFF-VESSEL"]["single_error"]


def test_locked_predictions_detect_changes(predicted, tmp_path):
    path, _, _ = predicted
    copy = tmp_path / "predictions.csv"
    copy.write_bytes(path.read_bytes() + b"\n")
    (tmp_path / "predictions_meta.json").write_bytes(
        path.with_name("predictions_meta.json").read_bytes()
    )
    with pytest.raises(ValueError, match="changed after it was locked"):
        read_predictions(copy)


def test_scoring_flow_and_intention_to_diagnose(predicted):
    path, reference, n = predicted
    rows, meta = read_predictions(path)
    cfg = ValidationConfig(n_bootstrap=200)
    s = score_predictions(rows, mf.read_reference(reference), cfg, meta["index_tests"])
    flow = s["flow"]
    assert flow["manifest"] == n + 2 and len(flow["excluded"]) == 1
    assert flow["eligible"] == n + 1 and s["eligible_by_type"] == {"FFR": n, "iFR": 1}
    for test in INDEX_TESTS:  # the 2 broken cases fail in both modes; n - 2 FFR cases remain
        r = s["index_tests"]["FFR"][test]
        assert r["n"] == n - 2
        assert {f["case_id"] for f in r["failed"]} == {"BAD-IMAGE", "OFF-VESSEL"}
    single = s["index_tests"]["FFR"]["single_solver"]
    assert single["accuracy_itd"] == pytest.approx((single["tp"] + single["tn"]) / n)
    assert single["feasibility"] == pytest.approx(single["n"] / n)
    assert "bland_altman" not in s["index_tests"]["iFR"]["single_solver"]


def test_report_files(predicted, tmp_path):
    path, reference, _ = predicted
    rows, meta = read_predictions(path)
    write_validation_report(rows, meta, mf.read_reference(reference),
                            ValidationConfig(n_bootstrap=200), tmp_path / "report.md",
                            tmp_path / "figures" / "v.png", title="T", intro=["x"])  # fmt: skip
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## Flow of vessels", "## Primary analysis", "## All index tests",
        "## Vessels that could not be analysed", "figures/v.png", "Research use",
    ):  # fmt: skip
        assert heading in md
    assert (tmp_path / "figures" / "v.png").stat().st_size > 10_000
    assert (tmp_path / "report.json").exists()


@pytest.mark.skipif(not IN_GIT, reason="needs the git checkout")
def test_run_refuses_tracked_output(study):
    out, _ = study
    target = mf._repo_root() / "reports" / "_validation_should_not_exist"
    with pytest.raises(ValueError, match="git-ignored"):
        run_predictions(out / "cases.csv", target, segmenter=None, surrogate=None, params=PARAMS,
                        uncertainty=FAST_UNCERTAINTY, qca_error=None)  # fmt: skip
    assert not target.exists()


# --- CLI -------------------------------------------------------------------------------------


def test_cli_validate_and_sample_size(study, dark_seg_dir, tiny_ensemble_export, tmp_path):
    out, _ = study
    runner = CliRunner()
    res = runner.invoke(app, ["validate"])
    assert res.exit_code == 2 and "either --manifest" in res.output
    res = runner.invoke(app, [
        "validate", "--manifest", str(out / "cases.csv"), "--reference",
        str(out / "reference.csv"), "--out", str(tmp_path / "v"), "--seg-model",
        str(dark_seg_dir), "--surrogate-model", tiny_ensemble_export["out_dir"],
    ])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "Predictions locked" in res.output and "Primary (" in res.output
    assert (tmp_path / "v" / "report.md").exists()
    # the blinded step alone, then scoring the locked predictions
    res = runner.invoke(app, ["validate", "--predictions", str(tmp_path / "v" / "predictions.csv"),
                              "--reference", str(out / "reference.csv")])  # fmt: skip
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["sample-size", "--out", str(tmp_path / "ss.md")])
    assert res.exit_code == 0 and "vessels in about" in res.output
    assert "placeholders" in (tmp_path / "ss.md").read_text(encoding="utf-8")


def test_synthetic_validation_end_to_end(dark_seg_dir, tiny_ensemble_export, tmp_path):
    from vstent.validation.synthetic_report import run_synthetic_validation

    out = run_synthetic_validation(
        small_study_config(2), ValidationConfig(n_bootstrap=100), PARAMS,
        seg_dir=dark_seg_dir, surrogate_dir=tiny_ensemble_export["out_dir"],
        uncertainty_path=tmp_path / "missing.yaml", data_dir=tmp_path / "data",
        out_md=tmp_path / "rep" / "v.md", fig_path=tmp_path / "rep" / "fig.png",
        predictions_dir=tmp_path / "rep" / "pred",
    )  # fmt: skip
    md = (tmp_path / "rep" / "v.md").read_text(encoding="utf-8")
    for text in ("Synthetic stand-in data, not a clinical validation", "Error budget",
                 "## With the automatic segmenter", "Against the known truth"):  # fmt: skip
        assert text in md
    assert set(out["predictions"]) == {"masks", "segmenter"}
