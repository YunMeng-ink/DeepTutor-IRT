"""Query instance cache for the local LightRAG pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from deeptutor.services.rag.pipelines.lightrag import query_cache
from deeptutor.services.rag.pipelines.lightrag.query_cache import (
    MAX_CACHED_QUERY_INSTANCES,
    QueryCacheKey,
)


class _FakeInstance:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture(autouse=True)
def _clean_cache():
    # Reset the module registry directly: an async fixture cannot run for the
    # plain sync tests in this file.
    query_cache._entries.clear()
    query_cache._key_locks.clear()
    yield
    query_cache._entries.clear()
    query_cache._key_locks.clear()


@pytest.fixture
def finalize_calls(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def finalize(rag: Any, *, cancel_pending: bool) -> None:
        calls.append(rag.name)

    monkeypatch.setattr(query_cache.engine, "finalize", finalize)
    return calls


def _key(root: str, *, stamp: tuple[int, int] | None = (1, 2), loop: int = 1) -> QueryCacheKey:
    return QueryCacheKey(root=root, version_stamp=stamp, owner_loop=loop, identity=("model",))


@pytest.mark.asyncio
async def test_acquire_builds_once_per_key(finalize_calls) -> None:
    built: list[str] = []

    async def factory() -> _FakeInstance:
        built.append("build")
        return _FakeInstance("one")

    first = await query_cache.acquire(_key("version-1"), factory)
    second = await query_cache.acquire(_key("version-1"), factory)

    assert first is second
    assert built == ["build"]
    assert finalize_calls == []


@pytest.mark.asyncio
async def test_different_stamp_cache_misses(finalize_calls) -> None:
    built: list[str] = []

    async def factory() -> _FakeInstance:
        built.append("build")
        return _FakeInstance(f"instance-{len(built)}")

    await query_cache.acquire(_key("version-1", stamp=(1, 2)), factory)
    await query_cache.acquire(_key("version-1", stamp=(3, 4)), factory)

    assert built == ["build", "build"]
    # Both stay within the LRU bound; the superseded one is released later
    # when it falls out of the cache (covered by the LRU test).
    assert finalize_calls == []


@pytest.mark.asyncio
async def test_owner_loop_and_identity_are_part_of_the_key() -> None:
    built: list[str] = []

    async def factory() -> _FakeInstance:
        built.append("build")
        return _FakeInstance("instance")

    await query_cache.acquire(_key("version-1", loop=1), factory)
    await query_cache.acquire(_key("version-1", loop=2), factory)

    assert built == ["build", "build"]


@pytest.mark.asyncio
async def test_concurrent_acquires_build_once(finalize_calls) -> None:
    built: list[str] = []

    async def factory() -> _FakeInstance:
        built.append("build")
        await asyncio.sleep(0.01)
        return _FakeInstance("one")

    results = await asyncio.gather(
        query_cache.acquire(_key("version-1"), factory),
        query_cache.acquire(_key("version-1"), factory),
    )

    assert results[0] is results[1]
    assert built == ["build"]


@pytest.mark.asyncio
async def test_lru_bounds_the_cache_and_finalizes_evictions(finalize_calls) -> None:
    async def factory(name: str):
        async def build() -> _FakeInstance:
            return _FakeInstance(name)

        return build

    for index in range(MAX_CACHED_QUERY_INSTANCES + 1):
        await query_cache.acquire(_key(f"version-{index}"), await factory(f"instance-{index}"))

    assert query_cache.cached_roots() == [
        f"version-{i}" for i in range(1, MAX_CACHED_QUERY_INSTANCES + 1)
    ]
    assert finalize_calls == ["instance-0"]


@pytest.mark.asyncio
async def test_evict_root_releases_only_that_version(finalize_calls) -> None:
    async def factory(name: str):
        async def build() -> _FakeInstance:
            return _FakeInstance(name)

        return build

    await query_cache.acquire(_key("version-1"), await factory("instance-1"))
    await query_cache.acquire(_key("version-2"), await factory("instance-2"))

    await query_cache.evict_root("version-1")

    assert query_cache.cached_roots() == ["version-2"]
    assert finalize_calls == ["instance-1"]


@pytest.mark.asyncio
async def test_discard_releases_the_failed_instance(finalize_calls) -> None:
    async def build() -> _FakeInstance:
        return _FakeInstance("broken")

    key = _key("version-1")
    rag = await query_cache.acquire(key, build)

    await query_cache.discard(key, rag)

    assert query_cache.cached_roots() == []
    assert finalize_calls == ["broken"]
    # A second discard for an already-dropped entry is a no-op.
    await query_cache.discard(key, rag)
    assert finalize_calls == ["broken"]


@pytest.mark.asyncio
async def test_finalize_failure_is_logged_not_raised(monkeypatch, caplog) -> None:
    async def boom(rag: Any, *, cancel_pending: bool) -> None:
        raise RuntimeError("storage already closed")

    monkeypatch.setattr(query_cache.engine, "finalize", boom)

    async def build() -> _FakeInstance:
        return _FakeInstance("one")

    key = _key("version-1")
    rag = await query_cache.acquire(key, build)
    with caplog.at_level("WARNING"):
        await query_cache.discard(key, rag)

    assert any("cleanup failed" in record.getMessage() for record in caplog.records)


# --- pipeline wiring -------------------------------------------------------


def _write_meta(root: Path, payload: dict[str, Any] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(
        json.dumps(payload or {"updated_at": "2026-01-01T00:00:00Z"}), encoding="utf-8"
    )


def test_version_stamp_tracks_meta_writes(tmp_path: Path) -> None:
    from deeptutor.services.rag.pipelines.lightrag.pipeline import _version_stamp

    root = tmp_path / "version-1"
    _write_meta(root)
    first = _version_stamp(root)
    assert first is not None

    _write_meta(root, {"updated_at": "2026-02-02T00:00:00Z", "padding": "x" * 64})
    second = _version_stamp(root)

    assert second is not None
    assert second != first
    assert _version_stamp(tmp_path / "missing") is None


@pytest.mark.asyncio
async def test_discard_leaves_a_replaced_entry_alone(finalize_calls) -> None:
    """A failed query must not tear down the instance that replaced it."""

    async def build(name: str):
        async def factory() -> _FakeInstance:
            return _FakeInstance(name)

        return factory

    key = _key("version-1")
    stale = await query_cache.acquire(key, await build("stale"))
    await query_cache.clear()  # simulates eviction; finalize recorded
    replacement = await query_cache.acquire(key, await build("replacement"))

    await query_cache.discard(key, stale)

    assert query_cache.cached_roots() == ["version-1"]
    assert "replacement" not in finalize_calls

    # The replacement still answers from the cache.
    async def failing_factory() -> _FakeInstance:
        raise AssertionError("replacement should have been cached")

    assert await query_cache.acquire(key, failing_factory) is replacement


@pytest.mark.asyncio
async def test_close_instance_releases_a_never_cached_instance(finalize_calls) -> None:
    rag = _FakeInstance("half-initialized")

    await query_cache.close_instance(rag)

    assert finalize_calls == ["half-initialized"]
    assert query_cache.cached_roots() == []
