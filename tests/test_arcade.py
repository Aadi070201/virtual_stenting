import hashlib
import json
import zipfile

import pytest

from vstent.data.arcade import EXTRACTED_MARKER, SOURCE_FILE, fetch_arcade, md5sum, safe_extract


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_fetch_downloads_verifies_and_extracts(tmp_path):
    src = _zip(tmp_path / "src.zip", {"arcade/syntax/train/annotations/train.json": "{}"})
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    raw = tmp_path / "raw"
    messages = []
    source = fetch_arcade(
        raw,
        url=src.as_uri(),
        expected_md5=md5,
        expected_size=src.stat().st_size,
        log=messages.append,
    )
    assert source["md5"] == md5 and md5sum(raw / "arcade.zip") == md5
    assert (raw / "arcade" / "syntax" / "train" / "annotations" / "train.json").exists()
    assert (raw / EXTRACTED_MARKER).exists()
    assert json.loads((raw / SOURCE_FILE).read_text(encoding="utf-8"))["licence"].startswith("CC0")

    messages.clear()
    fetch_arcade(raw, url=src.as_uri(), expected_md5=md5, log=messages.append)
    assert any("already present" in m for m in messages)
    assert not any(m.startswith("Downloading") for m in messages)


def test_checksum_mismatch_leaves_no_files(tmp_path):
    src = _zip(tmp_path / "src.zip", {"a.txt": "x"})
    raw = tmp_path / "raw"
    with pytest.raises(ValueError, match="MD5"):
        fetch_arcade(raw, url=src.as_uri(), expected_md5="0" * 32, expected_size=None)
    assert not (raw / "arcade.zip").exists()
    assert not (raw / "arcade.zip.part").exists()


def test_safe_extract_rejects_path_traversal(tmp_path):
    bad = _zip(tmp_path / "bad.zip", {"../evil.txt": "x"})
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
