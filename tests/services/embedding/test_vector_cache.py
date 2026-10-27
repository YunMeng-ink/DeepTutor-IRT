"""Persistent embedding vector cache behaviour and client integration."""

from __future__ import annotations

import asyncio

import pytest

from deeptutor.services.embedding import client as client_module
from deeptutor.services.embedding.client import EmbeddingClient
from deeptutor.services.embedding.config import EmbeddingConfig
from deeptutor.services.embedding.vector_cache import (
    EmbeddingVectorCache,
    vector_cache_key,
)


class _FakeAdapter:
    instances: list["_FakeAdapter"] = []
    SUPPORTS_INPUT_TYPE = False

    def __init__(self, config: dict) -> None:
        self.config = config
        self.calls = []
        _FakeAdapter.instances.append(self)

    async def embed(self, request):
        self.calls.append(request)
        return type(
            "Resp",
            (),
            {
                "embeddings": [
                    [float(len(text)), float(index)] for index, text in enumerate(request.texts)
                ],
            },
        )()


def _config(*, model: str = "text-embedding-3-small") -> EmbeddingConfig:
    base_url = "https://api.openai.com/v1/embeddings"
    return EmbeddingConfig(
        model=model,
        api_key="sk-test",
        base_url=base_url,
        effective_url=base_url,
        binding="openai",
        provider_name="openai",
        provider_mode="standard",
        dim=2,
        send_dimensions=True,
        batch_size=2,
        request_timeout=30,
    )


def _cache(tmp_path, *, max_entries: int = 100) -> EmbeddingVectorCache:
    return EmbeddingVectorCache(tmp_path / "vectors.sqlite3", max_entries=max_entries)


def test_cache_roundtrip_and_identity_miss(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.store("model-a|role", ["alpha", "beta"], [[0.5, 0.25], [1.0, 2.0]])

    hits = cache.lookup("model-a|role", ["alpha", "beta", "gamma"])

    assert hits[0] == pytest.approx([0.5, 0.25], rel=1e-6)
    assert hits[1] == pytest.approx([1.0, 2.0], rel=1e-6)
    assert hits[2] is None
    # Any identity component change (model, endpoint, role) misses.
    assert cache.lookup("model-b|role", ["alpha"]) == [None]


def test_cache_key_separates_texts_and_identities() -> None:
    assert vector_cache_key("m", "a") != vector_cache_key("m", "b")
    assert vector_cache_key("m", "a") != vector_cache_key("n", "a")


def test_cache_prunes_to_bound(tmp_path) -> None:
    cache = _cache(tmp_path, max_entries=3)
    cache.store("m", ["a", "b", "c"], [[1.0], [2.0], [3.0]])
    cache.store("m", ["d"], [[4.0]])

    hits = cache.lookup("m", ["a", "b", "c", "d"])

    assert sum(hit is not None for hit in hits) == 3


@pytest.mark.asyncio
async def test_client_reuses_cached_vectors(monkeypatch, tmp_path) -> None:
    cache = _cache(tmp_path)
    monkeypatch.setattr(client_module, "get_embedding_vector_cache", lambda: cache)
    monkeypatch.setattr(
        client_module, "_resolve_adapter_class", lambda _binding: _FakeAdapter
    )
    _FakeAdapter.instances = []

    client = EmbeddingClient(_config())
    first = await client.embed(["alpha", "beta"])
    second = await client.embed(["alpha", "beta"])

    assert second == first
    assert len(_FakeAdapter.instances[0].calls) == 1


@pytest.mark.asyncio
async def test_client_embeds_only_cache_misses(monkeypatch, tmp_path) -> None:
    cache = _cache(tmp_path)
    monkeypatch.setattr(client_module, "get_embedding_vector_cache", lambda: cache)
    monkeypatch.setattr(
        client_module, "_resolve_adapter_class", lambda _binding: _FakeAdapter
    )
    _FakeAdapter.instances = []
    client = EmbeddingClient(_config())
    cache.store(client._cache_identity(None), ["alpha"], [[9.0, 9.0]])

    vectors = await client.embed(["alpha", "beta"])

    adapter = _FakeAdapter.instances[0]
    assert [call.texts for call in adapter.calls] == [["beta"]]
    assert vectors[0] == [9.0, 9.0]
    assert vectors[1] == [float(len("beta")), 0.0]


@pytest.mark.asyncio
async def test_client_cache_misses_when_model_changes(monkeypatch, tmp_path) -> None:
    cache = _cache(tmp_path)
    monkeypatch.setattr(client_module, "get_embedding_vector_cache", lambda: cache)
    monkeypatch.setattr(
        client_module, "_resolve_adapter_class", lambda _binding: _FakeAdapter
    )
    _FakeAdapter.instances = []

    await EmbeddingClient(_config(model="model-a")).embed(["alpha"])
    await EmbeddingClient(_config(model="model-b")).embed(["alpha"])

    assert len(_FakeAdapter.instances[0].calls) == 1
    assert len(_FakeAdapter.instances[1].calls) == 1


def test_cache_failure_degrades_to_miss(tmp_path) -> None:
    cache = EmbeddingVectorCache(tmp_path / "not-a-dir" / "vectors.sqlite3")
    # Make the parent a file so SQLite cannot open the database.
    (tmp_path / "not-a-dir").write_text("blocked", encoding="utf-8")

    assert cache.lookup("m", ["alpha"]) == [None]
    cache.store("m", ["alpha"], [[1.0]])
    assert cache.lookup("m", ["alpha"]) == [None]


def test_cache_recreates_schema_after_database_file_is_deleted(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.store("m", ["alpha"], [[1.0, 2.0]])
    cache.path.unlink()

    assert cache.lookup("m", ["alpha"]) == [None]

    cache.store("m", ["alpha"], [[7.0, 8.0]])
    assert cache.lookup("m", ["alpha"])[0] == pytest.approx([7.0, 8.0])


@pytest.mark.asyncio
async def test_embedding_concurrency_path_still_reports_progress(monkeypatch, tmp_path) -> None:
    """A fully cached batch still advances the progress counter."""
    cache = _cache(tmp_path)
    monkeypatch.setattr(client_module, "get_embedding_vector_cache", lambda: cache)
    monkeypatch.setattr(
        client_module, "_resolve_adapter_class", lambda _binding: _FakeAdapter
    )
    _FakeAdapter.instances = []
    client = EmbeddingClient(_config())
    await client.embed(["alpha", "beta", "gamma", "delta"])
    reports: list[tuple[int, int]] = []

    await client.embed(
        ["alpha", "beta", "gamma", "delta"],
        progress_callback=lambda done, total: reports.append((done, total)),
    )

    assert reports == [(1, 2), (2, 2)]


def test_cache_can_be_disabled_by_env(monkeypatch) -> None:
    from deeptutor.services.embedding import vector_cache

    monkeypatch.setenv("DEEPTUTOR_EMBEDDING_CACHE", "0")
    assert vector_cache.cache_enabled() is False
    monkeypatch.setenv("DEEPTUTOR_EMBEDDING_CACHE", "1")
    assert vector_cache.cache_enabled() is True
    monkeypatch.delenv("DEEPTUTOR_EMBEDDING_CACHE")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert vector_cache.cache_enabled() is True
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    assert vector_cache.cache_enabled() is False


def test_lookup_is_thread_safe(tmp_path) -> None:
    import threading

    cache = _cache(tmp_path)
    cache.store("m", ["alpha"], [[1.0, 2.0]])
    results: list[list] = []
    errors: list[BaseException] = []

    def read() -> None:
        try:
            results.append(cache.lookup("m", ["alpha"]))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert all(result and result[0] == pytest.approx([1.0, 2.0]) for result in results)


def test_asyncio_thread_offload_roundtrip(tmp_path) -> None:
    cache = _cache(tmp_path)

    async def run() -> None:
        cache.store("m", ["alpha"], [[3.0, 4.0]])
        hits = await asyncio.to_thread(cache.lookup, "m", ["alpha"])
        assert hits[0] == pytest.approx([3.0, 4.0])

    asyncio.run(run())