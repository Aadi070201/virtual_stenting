from pathlib import Path

import pytest

from vstent.config import load_config
from vstent.data.synth import SynthConfig, generate_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """A tiny synthetic dataset (train 96, val 32, test 32, ood 16) built from synth_v1."""
    cfg = load_config(ROOT / "configs" / "synth_v1.yaml", SynthConfig)
    sizes = {"train": 96, "val": 32, "test": 32, "ood": 16}
    splits = {k: v.model_copy(update={"n": sizes[k]}) for k, v in cfg.splits.items()}
    out = tmp_path_factory.mktemp("synth")
    generate_dataset(cfg.model_copy(update={"splits": splits}), out_dir=out, workers=1)
    return out
