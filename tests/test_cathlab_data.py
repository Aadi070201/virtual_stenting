"""Reading the virtual cath lab's exported format (vcathlab-frames v1) without importing it."""

import hashlib
import json

import numpy as np
import pytest

from vstent.data import cathlab


def _dataset(root, n=3, version=1):
    root.mkdir()
    imgs = np.arange(n * 4 * 4, dtype=np.uint8).reshape(n, 4, 4)
    np.savez_compressed(root / "train-000.npz", images=imgs)
    lesions = [
        [{"box_xywh": [1, 1, 2, 2], "ds_pct": 70.0, "inside_share": 1.0},
         {"box_xywh": [0, 0, 1, 1], "ds_pct": 30.0, "inside_share": 1.0}],  # mild: unmarked
        [{"box_xywh": None, "ds_pct": 80.0, "inside_share": 0.0}],  # outside the image
        [],
    ]  # fmt: skip
    recs = "".join(json.dumps({"index": i, "lesions": lesions[i]}) + "\n" for i in range(n))
    (root / "train.jsonl").write_text(recs, encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    shard = {"file": "train-000.npz", "first": 0, "count": n, "sha256": sha(root / "train-000.npz")}
    split = {"n": n, "records": "train.jsonl", "records_sha256": sha(root / "train.jsonl"),
             "shards": [shard]}  # fmt: skip
    manifest = {"format": "vcathlab-frames", "version": version, "splits": {"train": split}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return imgs


def test_load_split_with_the_positive_rule(tmp_path):
    imgs = _dataset(tmp_path / "d")
    images, boxes, records = cathlab.load_split(tmp_path / "d", "train")
    assert np.array_equal(images, imgs) and len(records) == 3
    assert boxes == [[(1, 1, 2, 2)], [], []]
    mild_too = cathlab.load_split(tmp_path / "d", "train", min_ds=20)[1][0]
    assert mild_too == [(1, 1, 2, 2), (0, 0, 1, 1)]
    assert len(cathlab.load_split(tmp_path / "d", "train", max_images=2)[0]) == 2


def test_wrong_version_or_changed_file_is_refused(tmp_path):
    _dataset(tmp_path / "v2", version=2)
    with pytest.raises(ValueError, match="not a vcathlab-frames v1"):
        cathlab.read_manifest(tmp_path / "v2")
    _dataset(tmp_path / "d")
    (tmp_path / "d" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        cathlab.load_split(tmp_path / "d", "train")
