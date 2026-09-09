"""Installer archive verification and extraction boundaries, without network access."""

import hashlib
import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Installer build environment requires Python 3.12+"
)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "installer_runtime_builder", ROOT / "scripts/build-bundled-runtime.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def archive(path, name, *, link=None):
    with tarfile.open(path, "w:gz") as bundle:
        item = tarfile.TarInfo(name)
        if link:
            item.type = tarfile.SYMTYPE
            item.linkname = link
            bundle.addfile(item)
        else:
            item.size = 5
            bundle.addfile(item, io.BytesIO(b"hello"))


def test_official_archive_is_reused_only_when_its_hash_matches(tmp_path, monkeypatch):
    content = b"checked archive fixture"
    sha = hashlib.sha256(content).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / (sha + ".tar.gz")).write_bytes(content)
    monkeypatch.setattr(
        builder, "urlopen", lambda *_a, **_k: pytest.fail("No network needed for verified cache")
    )
    result = builder.fetch_archive({"url": "unused", "sha256": sha, "bytes": len(content)}, cache)
    assert result.read_bytes() == content


def test_corrupt_download_is_not_installed_or_left_as_a_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "urlopen", lambda *_a, **_k: io.BytesIO(b"wrong"))
    source = {
        "url": "https://github.com/astral-sh/python-build-standalone/releases/download/fixture/file.tar.gz",
        "sha256": hashlib.sha256(b"right").hexdigest(),
        "bytes": 5,
    }
    with pytest.raises(ValueError, match="checksum"):
        builder.fetch_archive(source, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name,link",
    [("../escape", None), ("python/../../../escape", None), ("python/link", "../../escape")],
)
def test_python_archive_rejects_traversal_and_escaping_links(tmp_path, name, link):
    path = tmp_path / "archive.tar.gz"
    archive(path, name, link=link)
    destination = tmp_path / "extract"
    destination.mkdir()
    with pytest.raises((ValueError, tarfile.FilterError)):
        builder.extract_archive(path, destination)
    assert not (tmp_path / "escape").exists()


def test_normal_archive_structure_is_preserved(tmp_path):
    path = tmp_path / "archive.tar.gz"
    archive(path, "python/lib/example.txt")
    builder.extract_archive(path, tmp_path / "extract")
    assert (tmp_path / "extract/python/lib/example.txt").read_text() == "hello"
