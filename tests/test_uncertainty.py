import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from vstent.config import load_config
from vstent.data.synth import SynthConfig
from vstent.inference.onnx_runtime import ONNX_FILE, SCREEN_ONNX_FILE, SurrogateRunner
from vstent.physics.geometry import cosine_lesion_profile
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.planning.stent import Stent
from vstent.surrogate.dataset import load_raw
from vstent.surrogate.export import export_ensemble
from vstent.surrogate.features import prepare_inputs
from vstent.surrogate.train import load_checkpoint
from vstent.uncertainty.conformal import (
    CONFORMAL_FILE,
    ConformalCalibration,
    conformal_quantile,
    coverage,
    fit_conformal,
)
from vstent.uncertainty.greyzone import Assessment, Interval, UncertaintyConfig
from vstent.uncertainty.inputs import (
    InputUncertainty,
    Measurement,
    QCAErrorModel,
    applicable_sources,
    correlated_noise,
    percentile_interval,
)

ROOT = Path(__file__).resolve().parents[1]
MM = 1e-3


# --- conformal ---------------------------------------------------------------------------


def test_conformal_quantile_rank():
    v = np.arange(1, 20, dtype=float)  # n = 19: ceil(20 · 0.9) = 18th smallest
    assert conformal_quantile(v, 0.90) == 18.0
    assert conformal_quantile(np.arange(1, 10.0), 0.90) == 9.0  # ceil(10 · 0.9) = 9 = n
    assert conformal_quantile(np.arange(1, 5.0), 0.90) == float("inf")  # too few
    with pytest.raises(ValueError):
        conformal_quantile(v, 1.0)


@pytest.mark.parametrize("score", ["normalised", "absolute"])
def test_conformal_coverage_on_exchangeable_data(score):
    rng = np.random.default_rng(0)

    def draw(n):
        std = rng.uniform(0.002, 0.02, n)
        pred = rng.uniform(0.5, 0.95, n)
        return pred, std, pred + std * rng.standard_normal(n)

    cal = fit_conformal(*draw(5000), level=0.9, score=score, sigma_floor=0.001)
    cov = coverage(cal, *draw(20000))
    assert 0.88 <= cov["coverage"] <= 0.92
    assert cal.n_calibration == 5000 and cal.split == "val"


def test_conformal_interval_capped_and_roundtrip(tmp_path):
    cal = ConformalCalibration(0.9, "normalised", 2.0, 0.001, 100, "val")
    lo, hi = cal.interval(np.array([0.99, 0.70]), np.array([0.01, 0.01]))
    np.testing.assert_allclose(lo, [0.99 - 0.022, 0.70 - 0.022])
    np.testing.assert_allclose(hi, [1.0, 0.722])
    cal.save(tmp_path / "c.json")
    assert ConformalCalibration.load(tmp_path / "c.json") == cal
    with pytest.raises(ValueError, match="sigma_floor"):
        fit_conformal(np.ones(3), np.zeros(3), np.ones(3), sigma_floor=0.0)


# --- ensemble export and runner ----------------------------------------------------------


def test_ensemble_export_files_and_metadata(tiny_ensemble_export):
    out = Path(tiny_ensemble_export["out_dir"])
    for name in (ONNX_FILE, SCREEN_ONNX_FILE, CONFORMAL_FILE, "norm_stats.json", "config.yaml"):
        assert (out / name).exists(), name
    proto = onnx.load(str(out / ONNX_FILE))
    assert [o.name for o in proto.graph.output] == ["ffr", "ffr_std"]
    props = {p.key: p.value for p in proto.metadata_props}
    assert props["ensemble_size"] == "2" and "Research use only" in props["disclaimer"]
    meta = json.loads((out / "export_meta.json").read_text(encoding="utf-8"))
    assert meta["parity"]["passed"] and meta["parity_screen"]["passed"]
    assert meta["parity"]["max_abs_diff_std"] < 1e-4
    assert len(meta["members"]) == 2 and meta["conformal"]["split"] == "val"


def test_ensemble_runner_mean_spread_and_screen(tiny_ensemble, tiny_ensemble_export, tiny_data):
    from vstent.surrogate.ensemble import member_dirs

    runner = SurrogateRunner.from_dir(tiny_ensemble_export["out_dir"], workers=2, chunk_size=5)
    raw, _ = load_raw(tiny_data, "test", 12)
    mean, std = runner.predict_with_spread(raw)
    assert runner.has_spread and runner.ensemble_size == 2 and runner.conformal is not None
    assert mean.shape == std.shape == (12, 256) and np.all(std >= 0)
    assert np.all(mean[:, 0] == 1.0) and np.all(np.diff(mean, axis=1) <= 1e-7)
    np.testing.assert_array_equal(runner.predict(raw), mean)

    xp, xg = prepare_inputs(raw, runner.stats)
    with torch.no_grad():
        members = [
            load_checkpoint(m)(torch.from_numpy(xp), torch.from_numpy(xg)).numpy()
            for m in member_dirs(tiny_ensemble)
        ]
    np.testing.assert_allclose(mean, np.mean(members, axis=0), atol=1e-5)
    np.testing.assert_allclose(std, np.std(members, axis=0, ddof=1), atol=1e-5)

    screen = runner.screen
    assert screen is not runner and not screen.has_spread and screen.screen is screen
    np.testing.assert_allclose(screen.predict(raw), members[0], atol=1e-5)
    lo, hi = runner.distal_interval(mean[:, -1], std[:, -1])
    assert np.all(lo <= mean[:, -1]) and np.all(hi >= np.minimum(mean[:, -1], 1.0))
    runner.close()


def test_single_model_runner_has_no_spread(tiny_export, tiny_data):
    runner = SurrogateRunner.from_dir(tiny_export["out_dir"])
    assert not runner.has_spread and runner.conformal is None and runner.screen is runner
    raw, _ = load_raw(tiny_data, "test", 2)
    with pytest.raises(ValueError, match="not an ensemble"):
        runner.predict_with_spread(raw)
    profile = cosine_lesion_profile(0.05, 3e-3, 0.01, 0.5)
    assert runner.predict_profile_with_spread(profile, 12000.0, 700.0, 3e10)[2] is None
    assert runner.distal_interval(0.8, 0.0) is None


def test_export_ensemble_rejects_unfinished(tiny_ensemble, tmp_path):
    import shutil

    from vstent.surrogate.ensemble import member_dirs

    copy = tmp_path / tiny_ensemble.name  # same name: member names follow the ensemble's
    shutil.copytree(tiny_ensemble, copy)
    (member_dirs(copy)[1] / "summary.json").unlink()
    with pytest.raises(ValueError, match="not finished"):
        export_ensemble(copy, tmp_path / "out")


# --- input uncertainty -------------------------------------------------------------------


def _case():
    return cosine_lesion_profile(60 * MM, 3.0 * MM, 12 * MM, 0.6), PhysicsParams()


def test_applicable_sources():
    qe = QCAErrorModel(0.1, 0.6)
    assert applicable_sources(Measurement(), qe) == ["v_hyp", "pa_pv"]
    assert applicable_sources(Measurement(0.3, True), qe) == [
        "pixel_size", "qca_diameter", "v_hyp", "pa_pv",
    ]  # fmt: skip
    assert applicable_sources(Measurement(0.2, False), qe)[0] == "qca_diameter"
    assert applicable_sources(Measurement(0.3, True), None) == ["pixel_size", "v_hyp", "pa_pv"]


def test_no_varied_source_reproduces_the_solver():
    prof, params = _case()
    mc = InputUncertainty(prof, params, params.pa, params.pv, measurement=Measurement(), sources=[])
    ref = solve_ffr_batch(prof.r, prof.ds, prof.reference_radius(), params).ffr_distal[0]
    np.testing.assert_allclose(mc.pre_samples, ref, rtol=1e-12)
    np.testing.assert_allclose(mc.post_samples([]), mc.pre_samples, rtol=1e-12)


def test_monte_carlo_ranges_and_reproducibility():
    prof, params = _case()
    m = Measurement(0.3, True)
    qe = QCAErrorModel(0.1, 0.6)
    a = InputUncertainty(prof, params, params.pa, params.pv, measurement=m, qca_error=qe)
    b = InputUncertainty(prof, params, params.pa, params.pv, measurement=m, qca_error=qe)
    np.testing.assert_array_equal(a.pre_samples, b.pre_samples)
    assert a.sources == ["pixel_size", "qca_diameter", "v_hyp", "pa_pv"]
    assert a.k.min() >= 0.9 and a.k.max() <= 1.1
    assert a.v_hyp.min() >= 0.25 and a.v_hyp.max() <= 0.45
    pa_mmhg = a.pa / 133.322
    assert pa_mmhg.min() >= 70 and pa_mmhg.max() <= 110 and np.all(a.pv < a.pa)
    lo, hi = percentile_interval(a.pre_samples, 0.9)
    assert lo < a.pre_samples.mean() < hi
    # one source alone uses the same random numbers as in the full run
    v_only = InputUncertainty(prof, params, params.pa, params.pv, measurement=m, sources=["v_hyp"])
    np.testing.assert_array_equal(v_only.v_hyp, a.v_hyp)
    assert np.all(v_only.k == 1.0)


def test_stent_raises_post_samples_and_narrows_spread():
    prof, params = _case()
    mc = InputUncertainty(prof, params, params.pa, params.pv, measurement=Measurement(0.3, True))
    post = mc.post_samples([Stent(20 * MM, 20 * MM, 3.0 * MM)])
    assert np.all(post > mc.pre_samples)
    assert np.std(post) < np.std(mc.pre_samples)


def test_correlated_noise_statistics():
    rng = np.random.default_rng(0)
    e = correlated_noise(rng, 400, 512, corr_half_points=6.0)
    assert e.std() == pytest.approx(1.0, abs=0.03)
    x = e[:, 100:400]
    acf6 = np.mean(x[:, :-6] * x[:, 6:]) / np.mean(x * x)
    assert acf6 == pytest.approx(0.5, abs=0.05)
    white = correlated_noise(rng, 2, 10, corr_half_points=0.2)
    assert white.shape == (2, 10)


def test_qca_error_model_from_report(tmp_path):
    assert QCAErrorModel.from_report(tmp_path / "missing.json") is None
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"error_model": {"sd_px": 0.2, "corr_half_px": 1.5}}))
    assert QCAErrorModel.from_report(path) == QCAErrorModel(0.2, 1.5)


# --- grey zone and config ----------------------------------------------------------------


def test_assessment_flags():
    a = Assessment(0.80, 0.82, 0.83, Interval(0.81, 0.85, 0.9), Interval(0.76, 0.88, 0.9))
    assert not a.model_flag and a.inputs_flag and a.flagged
    assert len(a.reasons()) == 1 and "0.760–0.880" in a.reasons()[0]
    d = a.to_dict()
    assert d["grey_zone"] and d["model_interval"] == [0.81, 0.85] and d["level"] == 0.9
    assert not Assessment(0.80, 0.70).flagged


def test_uncertainty_config_matches_sampling_ranges():
    cfg = load_config(ROOT / "configs" / "uncertainty.yaml", UncertaintyConfig)
    s = load_config(ROOT / "configs" / "synth_v1.yaml", SynthConfig).sampling
    half = {
        k: (hi - lo) / 2 for k, (lo, hi) in (("v", s.v_hyp), ("pa", s.pa_mmhg), ("pv", s.pv_mmhg))
    }
    assert cfg.inputs.v_hyp_halfwidth == pytest.approx(half["v"])
    assert cfg.inputs.pa_halfwidth_mmhg == pytest.approx(half["pa"])
    assert cfg.inputs.pv_halfwidth_mmhg == pytest.approx(half["pv"])
    assert PhysicsParams().v_hyp == pytest.approx(np.mean(s.v_hyp))


# --- report ------------------------------------------------------------------------------


def test_uncertainty_report_smoke(tiny_ensemble_export, tiny_export, tmp_path):
    from vstent.planning.optimizer import PlannerConfig, StentCatalog
    from vstent.uncertainty.inputs import InputUncertaintyConfig
    from vstent.uncertainty.report import write_uncertainty_report

    cfg = UncertaintyConfig(
        inputs=InputUncertaintyConfig(
            n_samples=40, qca_error_report=str(ROOT / "reports" / "qca_synthetic.json")
        )
    )
    report = write_uncertainty_report(
        tiny_ensemble_export["out_dir"],
        tiny_export["out_dir"],
        tmp_path,
        config=cfg,
        params=PhysicsParams(),
        planner=PlannerConfig(n_plans=2),
        catalog=StentCatalog(diameters_mm=[2.5, 3.0], lengths_mm=[12, 18]),
        n_vessels=8,
        n_plans=2,
        device="cpu",
    )
    assert set(report["conformal"]["splits"]) == {"val", "test", "ood"}
    assert len(report["targets"]) == 2  # no app_latency.json in tmp_path
    assert report["inputs"]["population"]["n"] == 8
    assert {p["setting"] for p in report["inputs"]["presets"]} == {"synthetic", "image"}
    md = (tmp_path / "uncertainty.md").read_text(encoding="utf-8")
    assert "Research use only" in md and "Neither covers the error of the reduced-order" in md
    for fig in ("uncertainty_coverage", "uncertainty_inputs"):
        assert (tmp_path / "figures" / f"{fig}.png").exists()
