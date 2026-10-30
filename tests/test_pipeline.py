import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from vstent.cli import app
from vstent.imaging.centerline import longest_path_endpoints
from vstent.imaging.seg_eval import evaluate_seg_run
from vstent.imaging.seg_export import export_seg_run
from vstent.imaging.seg_train import SegConfig, SegModelConfig, SegTrainConfig, train_seg
from vstent.imaging.synthetic import render_tube, sine_centerline, stenosis_diameter
from vstent.inference.figures import plot_analysis
from vstent.inference.onnx_runtime import SegmentationRunner, SurrogateRunner
from vstent.inference.pipeline import analyse_image, analyse_vessel, range_warnings
from vstent.physics.geometry import cosine_lesion_profile
from vstent.units import mmhg_to_pa


def _tube_prob(shape=(240, 240)):
    pts, s = sine_centerline((0.0, 0.0), 180.0, angle_deg=15.0, amplitude_px=15.0)
    pts = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2 + np.array(shape) / 2
    d = stenosis_diameter(s, 12.0, 0.6, s[-1] / 2, 30.0)
    return render_tube(shape, pts, s, d, blur_sigma=1.0, noise_std=0.02)


@pytest.fixture(scope="module")
def tiny_seg_run(tiny_arcade, tmp_path_factory):
    root = tmp_path_factory.mktemp("seg")
    cfg = SegConfig(
        raw_dir=str(tiny_arcade),
        runs_dir=str(root / "runs"),
        model=SegModelConfig(encoder="resnet18", encoder_weights=None, image_size=64),
        train=SegTrainConfig(epochs=1, batch_size=4, device="cpu", workers=0),
    )
    return train_seg(cfg, run_dir=root / "runs" / "tiny_seg", checkpoint_root=root / "ckpt")


def test_export_seg_parity_and_runner(tiny_seg_run, tmp_path):
    meta = export_seg_run(
        tiny_seg_run, tmp_path / "model", tmp_path / "reports", parity_images=4, latency_runs=2
    )
    assert meta["parity"]["passed"] and meta["parity"]["n"] == 4
    assert (tmp_path / "reports" / "seg_onnx.md").exists()
    runner = SegmentationRunner.from_dir(tmp_path / "model")
    assert runner.image_size == 64
    prob = runner.predict(np.zeros((100, 80), np.uint8))
    assert prob.shape == (100, 80) and prob.dtype == np.float32
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_eval_seg_report(tiny_seg_run, tmp_path):
    report = evaluate_seg_run(tiny_seg_run, tmp_path, device="cpu")
    assert set(report["splits"]) == {"val", "test"} and report["splits"]["val"]["n"] == 4
    md = (tmp_path / "seg_eval.md").read_text(encoding="utf-8")
    assert "Research use only" in md and "Mean val Dice" in md and "Mean test Dice" in md
    assert [t["metric"].split(" (")[0] for t in report["targets"]] == [
        "Mean val Dice",
        "Mean test Dice",
    ]
    for name in ("seg_examples.png", "seg_dice_hist.png", "seg_training.png"):
        assert (tmp_path / "figures" / name).exists()
    assert json.loads((tiny_seg_run / "eval.json").read_text(encoding="utf-8"))["splits"]


def test_runner_rejects_non_segmentation_model(tiny_export):
    with pytest.raises(ValueError, match="not a vstent segmentation model"):
        SegmentationRunner(Path(tiny_export["out_dir"]) / "surrogate.onnx")


def test_analyse_vessel_solver_and_surrogate(tiny_export, tmp_path):
    tube = _tube_prob()
    runner = SurrogateRunner.from_dir(tiny_export["out_dir"])
    res = analyse_vessel(tube.image, tube.points[0], tube.points[-1], surrogate=runner)
    runner.close()
    assert res.qca.calibration_assumed
    assert any("not calibrated" in w for w in res.warnings)
    assert res.solver.ffr[0] == pytest.approx(1.0) and np.all(np.diff(res.solver.ffr) <= 1e-12)
    assert 0.0 < res.ffr_distal_solver < 1.0
    assert res.surrogate_ffr.shape == (256,) and res.surrogate_s[-1] == pytest.approx(
        res.qca.profile.s[-1]
    )
    assert abs(res.qca.max_percent_stenosis - 60.0) <= 7.0
    img = (tube.image * 255).astype(np.uint8)
    plot_analysis(res, img, tmp_path / "a.png", boxes=[(1, 2, 3, 4)])
    assert (tmp_path / "a.png").stat().st_size > 0


def test_analyse_image_uses_segmenter():
    tube = _tube_prob()

    class Fake:
        def predict(self, image):
            return tube.image

    blank = np.zeros((240, 240), np.uint8)
    res = analyse_image(blank, tube.points[0], tube.points[-1], segmenter=Fake())
    assert res.surrogate_ffr is None and res.ffr_distal_surrogate is None
    assert res.prob is tube.image


def test_range_warnings():
    long = cosine_lesion_profile(length=0.1, ref_diameter=3e-3, lesion_length=0.01, severity=0.5)
    msgs = range_warnings(long, mmhg_to_pa(93.0), mmhg_to_pa(5.0))
    assert len(msgs) == 1 and "vessel length 100.0 mm" in msgs[0]
    ok = cosine_lesion_profile(length=0.05, ref_diameter=3e-3, lesion_length=0.01, severity=0.5)
    assert range_warnings(ok, mmhg_to_pa(93.0), mmhg_to_pa(5.0)) == []
    assert len(range_warnings(ok, mmhg_to_pa(150.0), mmhg_to_pa(5.0))) == 1


def test_longest_path_endpoints_on_tube():
    tube = _tube_prob()
    a, b = longest_path_endpoints(tube.image >= 0.5)
    ends = {tuple(np.round(tube.points[0])), tuple(np.round(tube.points[-1]))}
    for p in (a, b):
        assert min(np.hypot(p[0] - e[0], p[1] - e[1]) for e in ends) < 12  # within ~1 diameter


def test_cli_analyse_end_to_end(dark_seg_dir, tiny_export, tmp_path):
    tube = _tube_prob()
    image = ((1.0 - 0.6 * tube.image) * 255).astype(np.uint8)  # dark vessel on bright background
    path = tmp_path / "angio.png"
    cv2.imwrite(str(path), image)
    args = [
        "analyse",
        str(path),
        "--seg-model",
        str(dark_seg_dir),
        "--surrogate-model",
        tiny_export["out_dir"],
        "--figure",
        str(tmp_path / "fig.png"),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Auto-selected path" in result.output and "Distal FFR: solver" in result.output
    assert "Research use only" in result.output and (tmp_path / "fig.png").exists()
    r, c = np.round(tube.points[0]).astype(int)
    r2, c2 = np.round(tube.points[-1]).astype(int)
    explicit = args[:2] + ["--start", str(r), str(c), "--end", str(r2), str(c2)] + args[2:6]
    assert CliRunner().invoke(app, explicit).exit_code == 0


def test_cli_analyse_without_models(tmp_path, monkeypatch):
    physics = Path("configs/physics.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    cv2.imwrite(str(tmp_path / "x.png"), np.zeros((32, 32), np.uint8))
    result = CliRunner().invoke(app, ["analyse", "x.png", "--physics", str(physics)])
    assert result.exit_code == 1 and "No exported model" in result.output


def test_qca_gallery_on_fake_arcade(tiny_arcade, dark_seg_dir, tiny_export, tmp_path):
    from vstent.inference.gallery import _inside, qca_gallery
    from vstent.physics.solver import PhysicsParams

    assert list(_inside(np.array([[5.0, 5.0], [50.0, 50.0]]), [(0, 0, 10, 10)])) == [True, False]
    report = qca_gallery(
        tiny_arcade,
        dark_seg_dir,
        tiny_export["out_dir"],
        params=PhysicsParams(),
        n_images=3,
        out_dir=tmp_path / "figs",
        report_path=tmp_path / "gallery.md",
    )
    s = report["summary"]
    assert s["n_images"] == 3 and s["n_analysed"] + s["n_failed"] == 3
    assert (tmp_path / "gallery.md").exists() and (tmp_path / "gallery.json").exists()
    assert len(list((tmp_path / "figs").glob("*.png"))) == s["n_analysed"]


def test_cli_analyse_with_plan(dark_seg_dir, tiny_export, tmp_path):
    tube = _tube_prob()
    path = tmp_path / "angio.png"
    cv2.imwrite(str(path), ((1.0 - 0.6 * tube.image) * 255).astype(np.uint8))
    args = [
        "analyse",
        str(path),
        "--seg-model",
        str(dark_seg_dir),
        "--surrogate-model",
        tiny_export["out_dir"],
        "--plan",
        "--mm-per-px",
        "0.25",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Stent plans (surrogate" in result.output


def test_cli_and_app_give_the_segmenter_the_same_pixels(
    dark_seg_dir, tiny_export, tmp_path, monkeypatch
):
    """One loader: a 12-bit image stored as a 16-bit PNG reaches the model as the app sees it
    (percentile-stretched uint8), not divided by 65535."""
    from vstent.imaging.image_io import read_image
    from vstent.inference import onnx_runtime

    tube = _tube_prob()
    path = tmp_path / "angio16.png"
    cv2.imwrite(str(path), ((1.0 - 0.6 * tube.image) * 4095).astype(np.uint16))
    seen = []
    predict = onnx_runtime.SegmentationRunner.predict

    def spy(self, image):
        seen.append(image)
        return predict(self, image)

    monkeypatch.setattr(onnx_runtime.SegmentationRunner, "predict", spy)
    args = ["analyse", str(path), "--seg-model", str(dark_seg_dir),
            "--surrogate-model", tiny_export["out_dir"]]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    expected = read_image(path.read_bytes(), path.name).image
    assert seen[0].dtype == np.uint8 and np.array_equal(seen[0], expected)
