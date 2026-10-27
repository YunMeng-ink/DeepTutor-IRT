"""Memoized file digest used by the parse cache, checkpoints and KB dedupe."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from deeptutor.services.file_digest import file_sha256, reset_file_digest_cache


@pytest.fixture(autouse=True)
def _clean_memo():
    reset_file_digest_cache()
    yield
    reset_file_digest_cache()


def _bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_mtime_ns + 10_000_000, stat.st_mtime_ns + 10_000_000))


def test_digest_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"%PDF-1.4 payload" * 1000)

    assert file_sha256(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_repeated_reads_reuse_the_memo(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"content")

    first = file_sha256(target)

    calls = {"count": 0}
    import deeptutor.services.file_digest as module

    original = module._hash_file

    def counting(path: Path) -> str:
        calls["count"] += 1
        return original(path)

    monkeypatch.setattr(module, "_hash_file", counting)

    assert file_sha256(target) == first
    assert file_sha256(target) == first
    assert calls["count"] == 0


def test_modified_file_is_rehashed(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"version one")

    first = file_sha256(target)
    target.write_bytes(b"version two")
    _bump_mtime(target)

    second = file_sha256(target)
    assert second != first
    assert second == hashlib.sha256(b"version two").hexdigest()


def test_same_size_rewrite_with_new_mtime_is_rehashed(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"aaaa")
    first = file_sha256(target)

    target.write_bytes(b"bbbb")
    _bump_mtime(target)

    assert file_sha256(target) != first


def test_missing_file_raises_like_open(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing.pdf")


def test_cache_is_bounded(tmp_path: Path) -> None:
    import deeptutor.services.file_digest as module

    for index in range(module._MAX_ENTRIES + 5):
        target = tmp_path / f"doc-{index}.bin"
        target.write_bytes(f"content-{index}".encode())
        file_sha256(target)

    assert len(module._digests) <= module._MAX_ENTRIES
