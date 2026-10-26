"""Batch concurrency and connection reuse for the embedding client."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deeptutor.services.embedding.adapters.base import BaseEmbeddingAdapter
from deeptutor.services.embedding.client import EmbeddingClient
from deeptutor.services.embedding.config import EmbeddingConfig


class _TrackingAdapter:
    SUPPORTS_INPUT_TYPE = False

    instances: list["_TrackingAdapter"] = []

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        _TrackingAdapter.instances.append(self)

    async def embed(self, request):
        self.calls.append(list(request.texts))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # Later batches finish first, which also proves results are
            # reassembled in input order rather than completion order.
            delay = 0.05 if request.texts[0].startswith("first") else 0.01
            await asyncio.sleep(delay)
            return type(
                "Resp",
                (),
                {"embeddings": [[float(len(text))] for text in request.texts]},
            )()
        finally:
            self.in_flight -= 1


def _config(**overrides: Any) -> EmbeddingConfig:
    values: dict[str, Any] = {
        "model": "text-embedding-3-small",
        "api_key": "sk-test",
        "base_url": "https://api.openai.com/v1/embeddings",
        "effective_url": "https://api.openai.com/v1/embeddings",
        "binding": "openai",
        "provider_name": "openai",
        "provider_mode": "standard",
        "dim": 0,
        "batch_size": 1,
        "request_timeout": 30,
    }
    values.update(overrides)
    return EmbeddingConfig(**values)


@pytest.fixture
def adapter(monkeypatch) -> type[_TrackingAdapter]:
    _TrackingAdapter.instances = []
    monkeypatch.setattr(
        "deeptutor.services.embedding.client._resolve_adapter_class",
        lambda _b: _TrackingAdapter,
    )
    return _TrackingAdapter


@pytest.mark.asyncio
async def test_batches_overlap_and_keep_input_order(adapter) -> None:
    client = EmbeddingClient(_config())

    vectors = await client.embed(["first", "second", "third", "fourth"])

    recorded = adapter.instances[0]
    assert recorded.max_in_flight > 1
    # Vectors encode the text length, so order proves gather reassembly.
    assert vectors == [[5.0], [6.0], [5.0], [6.0]]


@pytest.mark.asyncio
async def test_concurrency_is_bounded_by_config(adapter) -> None:
    client = EmbeddingClient(_config(max_concurrent_batches=2))

    await client.embed([f"text-{i}" for i in range(6)])

    assert adapter.instances[0].max_in_flight == 2


@pytest.mark.asyncio
async def test_batch_delay_keeps_requests_serial(adapter) -> None:
    client = EmbeddingClient(_config(batch_delay=0.001))

    await client.embed(["first", "second", "third"])

    assert adapter.instances[0].max_in_flight == 1


@pytest.mark.asyncio
async def test_single_batch_never_creates_tasks(adapter) -> None:
    client = EmbeddingClient(_config(batch_size=10))

    vectors = await client.embed(["only"])

    assert vectors == [[4.0]]
    assert adapter.instances[0].calls == [["only"]]


@pytest.mark.asyncio
async def test_failure_cancels_siblings_and_raises_original_error(monkeypatch) -> None:
    cancelled = asyncio.Event()

    class _FailingAdapter(_TrackingAdapter):
        async def embed(self, request):
            self.calls.append(list(request.texts))
            if request.texts[0] == "boom":
                raise RuntimeError("batch exploded")
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return await super().embed(request)

    monkeypatch.setattr(
        "deeptutor.services.embedding.client._resolve_adapter_class",
        lambda _b: _FailingAdapter,
    )
    client = EmbeddingClient(_config())

    with pytest.raises(RuntimeError, match="batch exploded"):
        await client.embed(["boom", "slow"])

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_progress_stays_monotone_with_overlapping_batches(adapter) -> None:
    seen: list[tuple[int, int]] = []
    client = EmbeddingClient(_config())

    await client.embed(
        ["first", "second", "third", "fourth"],
        progress_callback=lambda done, total: seen.append((done, total)),
    )

    assert [done for done, _ in seen] == [1, 2, 3, 4]
    assert {total for _, total in seen} == {4}


@pytest.mark.asyncio
async def test_persistent_client_is_reused_within_one_loop() -> None:
    class _Adapter(BaseEmbeddingAdapter):
        async def embed(self, request):  # pragma: no cover - not exercised
            raise NotImplementedError

        def get_model_info(self) -> dict:
            return {}

    instance = _Adapter({"api_key": "k", "base_url": "https://example.test"})
    first = instance._persistent_client(timeout=10, verify=True)
    second = instance._persistent_client(timeout=10, verify=True)

    assert first is second
    await instance.aclose_clients()
    assert first.is_closed
    # A client requested after closing is a fresh, open one.
    third = instance._persistent_client(timeout=10, verify=True)
    assert third is not first
    assert not third.is_closed
    await instance.aclose_clients()


def test_persistent_client_is_scoped_per_event_loop() -> None:
    class _Adapter(BaseEmbeddingAdapter):
        async def embed(self, request):  # pragma: no cover - not exercised
            raise NotImplementedError

        def get_model_info(self) -> dict:
            return {}

    instance = _Adapter({"api_key": "k", "base_url": "https://example.test"})

    async def grab() -> Any:
        return instance._persistent_client(timeout=10, verify=True)

    first = asyncio.run(grab())
    second = asyncio.run(grab())

    assert first is not second
