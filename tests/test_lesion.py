"""P4i: learned lesion reading (patches, training in folds, ONNX export, depth rescaling)."""

import json

import numpy as np
import pytest

from vstent.data import lesion_data
from vstent.imaging import lesion_patch as lp
from vstent.imaging import lumen_report as lr
from vstent.imaging.centerline import smooth_and_resample
from vstent.imaging.qca import qca_from_centerline


def _qca(case):
    cl = smooth_and_resample(np.round(case["points"][::8]))
    return qca_from_centerline(case["mask"], cl, image=case["image"], lumen="edges",
                               mm_per_px=case["imaging"]["mm_per_px"])  # fmt: skip


def test_patch_candidates_and_rescaling():
    case = lr.render_lesion(12.0, 0.7, "circular", 6200)
    q = _qca(case)
    cands = lp.candidates(q)
    assert cands and abs(q.centerline.s[cands[0].k] - q.centerline.s[-1] / 2) < 20
    patch, scalars = lp.lesion_patch(case["image"], case["mask"], q.centerline, cands[0])
    assert patch.shape == (2, lp.PATCH_ALONG, lp.PATCH_ACROSS) and patch.dtype == np.float32
    assert scalars.shape == (lp.N_SCALARS,) and scalars[0] == pytest.approx(cands[0].ds_pct / 100)
    # the dark lesion core sits in the middle row band of the contrast channel
    assert patch[0, :, 12:20].mean() > patch[0, :, :4].mean()
    new, pairs = lp.apply_learned(q, case["image"], case["mask"],
                                  lambda p, s: np.full(len(p), 85.0))  # fmt: skip
    assert new.lumen_method == "learned" and len(pairs) == len(cands)
    assert new.max_percent_stenosis == pytest.approx(85.0, abs=1.0)
    measured = np.array([c.ds_pct for c in cands])
    same, _ = lp.apply_learned(q, case["image"], case["mask"], lambda p, s: measured)
    assert np.allclose(same.profile.r, q.profile.r)


def _fake_data(root, n_patients=6, per=6):
    rng = np.random.default_rng(0)
    for name, n, graded in (("cadica", n_patients * per, True), ("arcade_train", 30, False)):
        patches = rng.normal(0, 0.1, (n, 2, lp.PATCH_ALONG, lp.PATCH_ACROSS)).astype(np.float32)
        scalars = rng.uniform(0, 1, (n, lp.N_SCALARS)).astype(np.float32)
        samples = []
        for i in range(n):
            ds = [10.0, 35.0, 60.0, 80.0, 94.0][i % 5] if graded else float("nan")
            if graded:
                patches[i, 0, 28:36, 12:20] += ds / 100  # learnable signal
            samples.append({"source": name, "split": "dev" if graded else "train",
                            "patient": f"p{i // per + 1}" if graded else f"a{i}",
                            "track": f"t{i}", "ds_target": ds,
                            "category": {10.0: "p0_20", 35.0: "p20_50", 60.0: "p50_70",
                                         80.0: "p70_90", 94.0: "p90_98"}.get(ds, ""),
                            "sig_target": float(ds >= 50) if graded else float(i % 2)})  # fmt: skip
            samples[-1]["patch"], samples[-1]["scalars"] = patches[i], scalars[i]
        lesion_data.save(samples, name, root)


def test_training_in_folds_resume_and_export(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    from vstent.imaging import lesion_model as lm
    from vstent.inference.lesion_runtime import LesionRunner

    _fake_data(tmp_path / "data")
    cfg = lm.LesionTrainConfig(folds=2, epochs=2, batch_size=16, data_dir=str(tmp_path / "data"),
                               runs_dir=str(tmp_path / "runs"))  # fmt: skip
    ckpt = tmp_path / "ckpt"
    # stop during the first fold, then resume to the end
    orig = lm.stop_requested
    lm.stop_requested = lambda d: True
    try:
        run = lm.train_lesion(cfg, checkpoint_root=ckpt)
    finally:
        lm.stop_requested = orig
    assert not (run / lm.OOF_FILE).exists()
    run = lm.resume_lesion(run.name, checkpoint_root=ckpt)
    oof = json.loads((run / lm.OOF_FILE).read_text(encoding="utf-8"))["samples"]
    assert len(oof) == 36 and all(0 <= s["oof_ds"] <= 100 for s in oof)
    assert {s["fold"] for s in oof} == {0, 1}
    meta = lm.export_lesion(run, tmp_path / "models")
    assert meta["parity_passed"] and meta["members"] == 2
    runner = LesionRunner.from_dir(tmp_path / "models" / run.name)
    p, s, _ = lesion_data.load("cadica", tmp_path / "data")
    ds, prob = runner.predict_both(p[:4], s[:4])
    assert ds.shape == (4,) and np.all((prob >= 0) & (prob <= 1))
    assert runner.predict(p[:0], s[:0]).shape == (0,)


def test_lesion_reading_stages(tmp_path):
    pytest.importorskip("torch")
    from vstent.imaging import lesion_model as lm
    from vstent.inference.lesion_runtime import LesionRunner
    from vstent.validation import lesion_reading as lread

    data = tmp_path / "data"
    _fake_data(data)
    p, s, m = lesion_data.load("cadica", data)
    for name in ("arcade_val", "arcade_test"):
        lesion_data.save([{**x, "patch": p[i], "scalars": s[i], "measured_ds": 40.0}
                          for i, x in enumerate(m)], name, data)  # fmt: skip
    test = [{**x, "split": "test", "measured_ds": 30.0, "patch": p[i], "scalars": s[i]}
            for i, x in enumerate(m)]  # fmt: skip
    dev = [{**x, "measured_ds": 30.0, "patch": p[i], "scalars": s[i]} for i, x in enumerate(m)]
    lesion_data.save(dev + test, "cadica", data)
    cfg = lm.LesionTrainConfig(folds=2, epochs=1, batch_size=16, data_dir=str(data),
                               runs_dir=str(tmp_path / "runs"))  # fmt: skip
    run = lm.train_lesion(cfg, checkpoint_root=tmp_path / "ckpt")
    lm.export_lesion(run, tmp_path / "models")
    runner = LesionRunner.from_dir(tmp_path / "models" / run.name)
    d = lread.dev_stage(run, runner, data)
    assert set(d["rule"]) == {"kappa_higher", "auc_ge70_higher", "median_0_20_le_30",
                              "arcade_sensitivity_higher", "arcade_fp_within_margin"}  # fmt: skip
    assert d["adopted"] == all(d["rule"].values())
    t = lread.test_stage(runner, data)
    assert t["cadica_test"]["learned"]["n_measured"] > 0 and "arcade_test" in t
    sections = [*lread.dev_sections(d), *lread.test_sections(t)]
    git = {"git_commit": "c", "git_dirty": False}
    lread.write_report(tmp_path, lread.REPORT, "x", sections, {"dev": d, "test": t}, git, "m")
    assert "Decision" in (tmp_path / "lesion_reading.md").read_text(encoding="utf-8")
