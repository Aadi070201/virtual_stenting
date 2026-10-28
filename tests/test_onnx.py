import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from typer.testing import CliRunner

from vstent.cli import app
from vstent.inference.benchmark import run_benchmark
from vstent.inference.onnx_runtime import ONNX_FILE, SurrogateRunner
from vstent.physics.geometry import cosine_lesion_profile
from vstent.surrogate.dataset import load_raw
from vstent.surrogate.export import EXPORT_META, OPSET, export_run, sha256
from vstent.surrogate.features import NORM_STATS, RawInputs, prepare_inputs
from vstent.surrogate.model import ModelConfig
from vstent.surrogate.train import SurrogateConfig, TrainConfig, load_checkpoint, train_surrogate
from vstent.units import mmhg_to_pa

SMALL_MODEL = ModelConfig(base_channels=8, levels=3, groups=4, cond_dim=16)


@pytest.fixture(scope="module")
def tiny_run(tiny_data, tmp_path_factory):
    """A small surrogate trained for 2 epochs on the tiny dataset (CPU)."""
    root = tmp_path_factory.mktemp("onnx")
    cfg = SurrogateConfig(
        data_dir=str(tiny_data),
        runs_dir=str(root / "runs"),
        model=SMALL_MODEL,
        train=TrainConfig(epochs=2, batch_size=32, device="cpu"),
    )
    return train_surrogate(cfg, run_dir=root / "runs" / "tiny", checkpoint_root=root / "ckpt")


@pytest.fixture(scope="module")
def tiny_export(tiny_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "tiny"
    return export_run(tiny_run, out)


def test_export_writes_model_dir_with_provenance(tiny_export):
    out = Path(tiny_export["out_dir"])
    for name in (ONNX_FILE, NORM_STATS, "config.yaml", EXPORT_META):
        assert (out / name).exists(), name
    meta = json.loads((out / EXPORT_META).read_text(encoding="utf-8"))
    assert meta["opset"] == OPSET and meta["onnx_sha256"] == sha256(out / ONNX_FILE)
    assert meta["run_git_commit"] == tiny_export["run_git_commit"]
    proto = onnx.load(str(out / ONNX_FILE))
    assert {o.domain: o.version for o in proto.opset_import}[""] == OPSET
    props = {p.key: p.value for p in proto.metadata_props}
    assert props["n_points"] == "256" and props["feature_version"] == "1"
    assert "Research use only" in props["disclaimer"]
    batch_dims = [i.type.tensor_type.shape.dim[0].dim_param for i in proto.graph.input]
    assert batch_dims == ["batch", "batch"]


def test_parity_with_torch(tiny_export):
    par = tiny_export["parity"]
    assert par["passed"] and par["max_abs_diff"] < 1e-4
    assert par["n"] == 32  # the whole tiny test split


@pytest.mark.parametrize("n", [1, 7, 33])
def test_runner_dynamic_batch_and_chunking(tiny_export, tiny_data, n):
    raw, _ = load_raw(tiny_data, "train", n)
    model_dir = tiny_export["out_dir"]
    serial = SurrogateRunner.from_dir(model_dir, workers=1, chunk_size=None)
    chunked = SurrogateRunner.from_dir(model_dir, workers=3, chunk_size=5)
    a, b = serial.predict(raw), chunked.predict(raw)
    chunked.close()
    assert a.shape == b.shape == (n, 256) and a.dtype == np.float32
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert np.all(a[:, 0] == 1.0)
    assert np.all(np.diff(a, axis=1) <= 0)


def test_runner_on_profile_matches_torch(tiny_run, tiny_export):
    profile = cosine_lesion_profile(
        length=0.05, ref_diameter=3e-3, lesion_length=0.012, severity=0.6, n=100
    )
    pa, pv, r_micro = mmhg_to_pa(93.0), mmhg_to_pa(5.0), 3.0e10
    runner = SurrogateRunner.from_dir(tiny_export["out_dir"])
    s, ffr = runner.predict_profile(profile, pa, pv, r_micro)
    assert s.shape == ffr.shape == (256,)
    assert s[0] == profile.s[0] and s[-1] == pytest.approx(profile.s[-1])
    xp, xg = prepare_inputs(RawInputs.from_profile(profile, pa, pv, r_micro), runner.stats)
    with torch.no_grad():
        ref = load_checkpoint(tiny_run)(torch.from_numpy(xp), torch.from_numpy(xg)).numpy()[0]
    np.testing.assert_allclose(ffr, ref, atol=1e-5)


def test_runner_rejects_mismatched_stats(tiny_export, tmp_path):
    stats = json.loads((Path(tiny_export["out_dir"]) / NORM_STATS).read_text(encoding="utf-8"))
    stats["n_points"] = 128
    bad = tmp_path / "stats.json"
    bad.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match="n_points"):
        SurrogateRunner(Path(tiny_export["out_dir"]) / ONNX_FILE, bad)


def test_raw_inputs_subset(tiny_data):
    raw, _ = load_raw(tiny_data, "test", 10)
    assert len(raw) == 10
    one = raw.subset(3)
    assert one.r.shape == (1, 256) and one.length.shape == (1,)
    np.testing.assert_array_equal(one.r[0], raw.r[3])
    assert len(raw.subset(slice(2, 6))) == 4
    assert len(raw.subset(np.arange(10) % 2 == 0)) == 5
    assert len(raw.subset(np.array([0, 0, 1]))) == 3


def test_benchmark_report(tiny_export, tmp_path):
    report = run_benchmark(
        tiny_export["out_dir"],
        tmp_path,
        export_meta=tiny_export,
        batch_size=40,
        single_runs=5,
        batch_runs=2,
    )
    assert [t["passed"] for t in report["targets"]][0]  # parity
    assert set(report["timings"]) == {"default", "one_call"}
    assert report["timings"]["one_call"]["settings"]["chunk_size"] is None
    assert report["solver"]["batch_size"] == 32
    q = report["int8"]
    assert q["n"] == 32 and q["int8_bytes"] > 0
    assert q["kept"] == (q["accuracy_ok"] and q["speedup"] >= 1.1)
    md = (tmp_path / "onnx_benchmark.md").read_text(encoding="utf-8")
    assert "Research use only" in md and "not** claimed to be faster" in md
    assert "INT8 kept:" in md
    assert json.loads((tmp_path / "onnx_benchmark.json").read_text(encoding="utf-8"))


@pytest.mark.slow
def test_cli_export_surrogate(tiny_run, tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "export-surrogate",
            "--run",
            str(tiny_run),
            "--out",
            str(tmp_path / "model"),
            "--report-dir",
            str(tmp_path / "reports"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "model" / ONNX_FILE).exists()
    assert (tmp_path / "reports" / "onnx_benchmark.md").exists()
