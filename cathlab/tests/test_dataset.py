"""The exported dataset: files, checksums, records and arrays agree."""

import hashlib
import json

import numpy as np

from vcathlab.dataset import FORMAT, DatasetConfig, seeds, write_dataset


def test_dataset_files_records_and_checksums(tmp_path):
    cfg = DatasetConfig(out_dir=str(tmp_path / "d"), n={"train": 3, "val": 2}, shard=2, workers=2)
    m = write_dataset(cfg, log=lambda _m: None)
    root = tmp_path / "d"
    assert m["format"] == FORMAT and m["version"] == 1 and "Research use only" in m["disclaimer"]
    assert set(seeds(cfg, "train")).isdisjoint(seeds(cfg, "val"))
    for split, n in (("train", 3), ("val", 2)):
        info = m["splits"][split]
        recs = [json.loads(x) for x in (root / info["records"]).read_text("utf-8").splitlines()]
        assert [r["index"] for r in recs] == list(range(n))
        total = 0
        for sh in info["shards"]:
            path = root / sh["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sh["sha256"]
            with np.load(path) as z:
                assert z["images"].shape == (sh["count"], 512, 512)
                assert z["images"].dtype == np.uint8 and z["segments"].dtype == np.int16
                total += sh["count"]
        assert total == n
        for r in recs:
            for les in r["lesions"]:
                assert 0 <= les["ds_pct"] < 100 and (
                    les["box_xywh"] is None or len(les["box_xywh"]) == 4
                )
    assert (root / "card.md").read_text("utf-8").startswith("# Dataset card")
