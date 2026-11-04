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


class _Resp:
    def __init__(self, data, status):
        self.data, self.status, self.headers, self.pos = data, status, {}, 0

    def read(self, n):
        block, self.pos = self.data[self.pos : self.pos + n], self.pos + n
        return block

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("honours_range", [True, False])
def test_download_resumes_an_interrupted_part(tmp_path, monkeypatch, honours_range):
    from vstent.data import arcade

    data = bytes(range(256)) * 40
    seen = []

    def urlopen(request, timeout):
        rng = request.get_header("Range")
        seen.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng and honours_range else 0
        return _Resp(data[start:], 206 if start else 200)

    monkeypatch.setattr(arcade.urllib.request, "urlopen", urlopen)
    dest = tmp_path / "f.zip"
    (tmp_path / "f.zip.part").write_bytes(b"\xff" * 7 if not honours_range else data[:3000])
    arcade.download("http://x/f.zip", dest, expected_md5=hashlib.md5(data).hexdigest(),
                    expected_size=len(data))  # fmt: skip
    assert dest.read_bytes() == data and seen == ["bytes=3000-" if honours_range else "bytes=7-"]
