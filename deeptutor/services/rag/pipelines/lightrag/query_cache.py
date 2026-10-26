"""In-process cache of initialized LightRAG instances used for querying.

Every query used to build a fresh LightRAG instance, `initialize_storages()`
it (re-reading the persisted KV stores from disk) and `finalize_storages()`
it again. On a textbook-sized graph that is hundreds of milliseconds to
seconds of pure I/O per question, before any retrieval happens.

Instances are cached per published version and reused. Everything here runs
on the LightRAG worker loop: storages, queues and the owner-loop bridge are
all bound to the loop that created them, and version writes must never share
a live query instance, so indexing evicts the entry for its target root
before it builds its own.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
import threading
from typing import Any

from . import engine

logger = logging.getLogger(__name__)

# A desktop session queries one or two knowledge bases; keeping a couple of
# initialized instances warm covers that without pinning whole graphs in RAM.
MAX_CACHED_QUERY_INSTANCES = 2


@dataclass(frozen=True)
class QueryCacheKey:
    """Everything that makes a cached instance valid for one query."""

    root: str
    version_stamp: tuple[int, int] | None
    owner_loop: int
    # Resolved model bindings and initiating users. Usage and cache identity
    # are per model *and* per user, so neither may be shared across entries.
    identity: tuple[Any, ...]


_entries: OrderedDict[QueryCacheKey, Any] = OrderedDict()
_key_locks: dict[QueryCacheKey, asyncio.Lock] = {}
_registry_lock = threading.Lock()


def _key_lock(key: QueryCacheKey) -> asyncio.Lock:
    """One build per key; locks are created on the LightRAG worker loop."""
    with _registry_lock:
        lock = _key_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _key_locks[key] = lock
        return lock


async def acquire(
    key: QueryCacheKey,
    factory: Callable[[], Awaitable[Any]],
) -> Any:
    """Return the initialized instance for *key*, building it at most once."""
    async with _key_lock(key):
        with _registry_lock:
            cached = _entries.get(key)
            if cached is not None:
                _entries.move_to_end(key)
                return cached
        rag = await factory()
        with _registry_lock:
            _entries[key] = rag
            evicted = _evict_overflow_locked()
        await _shutdown_all(evicted)
        return rag


def _evict_overflow_locked() -> list[Any]:
    evicted: list[Any] = []
    while len(_entries) > MAX_CACHED_QUERY_INSTANCES:
        _key, rag = _entries.popitem(last=False)
        evicted.append(rag)
    return evicted


async def discard(key: QueryCacheKey, rag: Any) -> None:
    """Drop *rag* after a failed query so the next search starts clean."""
    with _registry_lock:
        removed = _entries.pop(key, None)
    if removed is not None:
        await _shutdown(rag)


async def evict_root(root: Any) -> None:
    """Finalize every cached instance backed by *root*.

    Called before an index write opens the same version directory: two live
    instances on one working directory would race the same storage files.
    """
    target = str(root)
    with _registry_lock:
        keys = [key for key in _entries if key.root == target]
        removed = [_entries.pop(key) for key in keys]
        for key in keys:
            _key_locks.pop(key, None)
    await _shutdown_all(removed)


async def clear() -> None:
    """Drop the whole cache (tests, configuration changes)."""
    with _registry_lock:
        removed = list(_entries.values())
        _entries.clear()
        _key_locks.clear()
    await _shutdown_all(removed)


def cached_roots() -> list[str]:
    """Roots with a live cached instance, newest use last."""
    with _registry_lock:
        return [key.root for key in _entries]


async def _shutdown_all(instances: list[Any]) -> None:
    for rag in instances:
        await _shutdown(rag)


async def _shutdown(rag: Any) -> None:
    """Finalize without losing the cleanup to a cancelled caller."""
    task = asyncio.ensure_future(_finalize(rag))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # The caller was cancelled; the instance is already out of the
        # registry and its cleanup keeps running on the worker loop.
        raise
    except BaseException:
        logger.warning("LightRAG query instance cleanup failed", exc_info=True)


async def _finalize(rag: Any) -> None:
    try:
        await engine.finalize(rag, cancel_pending=True)
    except Exception:
        logger.warning("LightRAG query instance cleanup failed", exc_info=True)


__all__ = [
    "MAX_CACHED_QUERY_INSTANCES",
    "QueryCacheKey",
    "acquire",
    "cached_roots",
    "clear",
    "discard",
    "evict_root",
]
