"""Process-wide memoized SHA-256 over file contents.

One indexing run hashes the same document several times: the parse-cache key,
the MinerU slice-checkpoint id, the page-evidence digest and the knowledge
base's duplicate-detection hash. On a scanned textbook that is hundreds of
megabytes read several times per operation, purely to recompute a value that
cannot have changed.

The memo is keyed by the file's identity stamp — resolved path, size,
modification time in nanoseconds and inode — so a modified file misses the
cache and is re-hashed, while repeated reads of the same version within one
process are free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading

_READ_CHUNK = 1 << 20  # 1 MiB
_MAX_ENTRIES = 64

_lock = threading.Lock()
_digests: dict[tuple[object, ...], str] = {}
_insertion_order: list[tuple[object, ...]] = []


def file_sha256(path: Path) -> str:
    """Return the full lowercase SHA-256 hexdigest of ``path``'s contents."""
    target = Path(path)
    try:
        stat = target.stat()
        key: tuple[object, ...] | None = (
            str(target.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            getattr(stat, "st_ino", 0),
        )
    except OSError:
        # Let the open() below surface the real failure to the caller.
        key = None

    if key is not None:
        with _lock:
            cached = _digests.get(key)
        if cached is not None:
            return cached

    digest = _hash_file(target)

    if key is not None:
        with _lock:
            if _digests.get(key) != digest:
                if len(_insertion_order) >= _MAX_ENTRIES:
                    _digests.pop(_insertion_order.pop(0), None)
                _digests[key] = digest
                _insertion_order.append(key)
    return digest


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_file_digest_cache() -> None:
    """Drop every memoized digest (tests, or after a bulk rewrite)."""
    with _lock:
        _digests.clear()
        _insertion_order.clear()


__all__ = ["file_sha256", "reset_file_digest_cache"]
