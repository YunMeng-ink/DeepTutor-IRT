"""Persistent, content-addressed embedding vector cache.

Re-indexing an unchanged corpus otherwise re-embeds every chunk: a 340-page
textbook is hundreds of provider round trips whose answers are identical to the
previous build. Vectors are keyed by ``(binding, model, dimensions, endpoint,
role, text digest)``, so switching any of those misses instead of silently
returning a vector produced by a different model. Storage is a bounded SQLite
file (float32 blobs, LRU eviction by ``last_used``) under the workspace's
``embedding_cache/`` directory — scoped per user/workspace like ``parse_cache``.

Every cache failure degrades to a miss. A broken or locked cache must never
fail an embedding run.
"""

from __future__ import annotations

from array import array
import hashlib
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ENTRIES = 50_000
_PRUNE_FRACTION = 0.1
_LOOKUP_CHUNK = 400
_DB_FILENAME = "vectors.sqlite3"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

_instances: dict[Path, "EmbeddingVectorCache"] = {}
_instances_lock = threading.Lock()


def cache_enabled() -> bool:
    """Whether the process should use the persistent vector cache.

    Explicit ``DEEPTUTOR_EMBEDDING_CACHE`` wins. Under pytest the default is
    off: tests assert exact provider call counts, and a cache shared across a
    test session would make them order-dependent.
    """
    raw = os.environ.get("DEEPTUTOR_EMBEDDING_CACHE", "").strip().lower()
    if raw in _FALSE_VALUES:
        return False
    if raw in _TRUE_VALUES:
        return True
    return not bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _max_entries() -> int:
    raw = os.environ.get("DEEPTUTOR_EMBEDDING_CACHE_MAX_ENTRIES", "").strip()
    try:
        return max(100, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_ENTRIES


def vector_cache_key(identity: str, text: str) -> str:
    """Digest one text under a caller-supplied model/endpoint identity."""
    return hashlib.sha256(f"{identity}\x00{text}".encode("utf-8")).hexdigest()


class EmbeddingVectorCache:
    """SQLite-backed vector cache for one workspace."""

    def __init__(self, path: Path, *, max_entries: int | None = None) -> None:
        self.path = Path(path)
        self.max_entries = int(max_entries or _max_entries())
        self._disabled = False
        self._disable_lock = threading.Lock()
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    # -- storage ---------------------------------------------------------

    def _disable(self, exc: BaseException) -> None:
        with self._disable_lock:
            if self._disabled:
                return
            self._disabled = True
        logger.warning("Embedding vector cache disabled after error at %s: %s", self.path, exc)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS vectors (
                            key TEXT PRIMARY KEY,
                            dim INTEGER NOT NULL,
                            vector BLOB NOT NULL,
                            created_at REAL NOT NULL,
                            last_used REAL NOT NULL
                        ) WITHOUT ROWID
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_vectors_last_used "
                        "ON vectors(last_used)"
                    )
                    conn.commit()
                    self._schema_ready = True
        return conn

    @staticmethod
    def _decode(dim: int, blob: bytes) -> list[float] | None:
        if not blob or dim <= 0 or len(blob) != dim * 4:
            return None
        values = array("f")
        values.frombytes(blob)
        if len(values) != dim:
            return None
        return values.tolist()

    @staticmethod
    def _encode(vector: list[float]) -> bytes:
        return array("f", vector).tobytes()

    # -- public API ------------------------------------------------------

    def lookup(self, identity: str, texts: list[str]) -> list[list[float] | None]:
        """Return cached vectors aligned with *texts*; ``None`` means a miss."""
        if not texts or self._disabled:
            return [None] * len(texts)
        keys = [vector_cache_key(identity, text) for text in texts]
        found: dict[str, list[float]] = {}
        stale: list[str] = []
        try:
            conn = self._connect()
            try:
                for start in range(0, len(keys), _LOOKUP_CHUNK):
                    chunk = keys[start : start + _LOOKUP_CHUNK]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT key, dim, vector FROM vectors WHERE key IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for key, dim, blob in rows:
                        vector = self._decode(int(dim), bytes(blob))
                        if vector is None:
                            stale.append(str(key))
                        else:
                            found[str(key)] = vector
                if found:
                    now = time.time()
                    hit_keys = list(found)
                    for start in range(0, len(hit_keys), _LOOKUP_CHUNK):
                        chunk = hit_keys[start : start + _LOOKUP_CHUNK]
                        placeholders = ",".join("?" for _ in chunk)
                        conn.execute(
                            f"UPDATE vectors SET last_used = ? WHERE key IN ({placeholders})",
                            [now, *chunk],
                        )
                if stale:
                    placeholders = ",".join("?" for _ in stale)
                    conn.execute(f"DELETE FROM vectors WHERE key IN ({placeholders})", stale)
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "no such table" in message:
                # File replaced underneath us; rebuild the schema next call.
                self._schema_ready = False
                return [None] * len(texts)
            if "locked" in message or "busy" in message:
                logger.debug("Embedding vector cache busy during lookup: %s", exc)
                return [None] * len(texts)
            self._disable(exc)
            return [None] * len(texts)
        except Exception as exc:
            self._disable(exc)
            return [None] * len(texts)
        return [found.get(key) for key in keys]

    def store(self, identity: str, texts: list[str], vectors: list[list[float]]) -> None:
        """Persist freshly embedded vectors, then bound the file size."""
        if not texts or len(texts) != len(vectors) or self._disabled:
            return
        rows: list[tuple[str, int, bytes, float, float]] = []
        now = time.time()
        for text, vector in zip(texts, vectors):
            if not vector:
                continue
            rows.append(
                (
                    vector_cache_key(identity, text),
                    len(vector),
                    self._encode(vector),
                    now,
                    now,
                )
            )
        if not rows:
            return
        try:
            conn = self._connect()
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO vectors (key, dim, vector, created_at, last_used) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                self._prune(conn)
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "no such table" in message:
                self._schema_ready = False
                return
            if "locked" in message or "busy" in message:
                logger.debug("Embedding vector cache busy during store: %s", exc)
                return
            self._disable(exc)
        except Exception as exc:
            self._disable(exc)

    def _prune(self, conn: sqlite3.Connection) -> None:
        count = int(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
        if count <= self.max_entries:
            return
        remove = max(1, int(count * _PRUNE_FRACTION))
        conn.execute(
            "DELETE FROM vectors WHERE key IN "
            "(SELECT key FROM vectors ORDER BY last_used ASC LIMIT ?)",
            (remove,),
        )
        logger.info(
            "Embedding vector cache pruned %d of %d entries (cap %d, %s)",
            remove,
            count,
            self.max_entries,
            self.path,
        )

    def close(self) -> None:
        """No persistent connection is held; present for lifecycle symmetry."""


def get_embedding_vector_cache() -> EmbeddingVectorCache | None:
    """The cache for the current user/workspace, or ``None`` when disabled."""
    if not cache_enabled():
        return None
    try:
        root = _cache_root()
    except Exception as exc:  # path services unavailable in standalone scripts
        logger.debug("Embedding vector cache unavailable: %s", exc)
        return None
    path = root / _DB_FILENAME
    with _instances_lock:
        cache = _instances.get(path)
        if cache is None:
            cache = EmbeddingVectorCache(path)
            _instances[path] = cache
        return cache


def _cache_root() -> Path:
    from deeptutor.multi_user.paths import get_current_path_service

    return get_current_path_service().get_embedding_cache_root()


def reset_embedding_vector_cache() -> None:
    """Drop in-process cache handles (files stay on disk)."""
    with _instances_lock:
        _instances.clear()


__all__ = [
    "EmbeddingVectorCache",
    "cache_enabled",
    "get_embedding_vector_cache",
    "reset_embedding_vector_cache",
    "vector_cache_key",
]