"""Micro-benchmarks for the batch-1 performance work.

Run with the project interpreter:

    python scripts/bench_batch1.py

Each section measures the exact code path the optimization touched, so the
numbers are comparable across commits on the same machine.
"""

from __future__ import annotations

from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _report(label: str, samples: list[float], unit: str = "ms") -> None:
    scale = 1000.0 if unit == "ms" else 1.0
    value = statistics.median(samples) * scale
    print(f"{label:<58} {value:10.3f} {unit}")


# ---------------------------------------------------------------------------
# 1. Runtime settings load (mtime cache)
# ---------------------------------------------------------------------------


def bench_settings_load() -> None:
    from deeptutor.services.config.runtime_settings import RuntimeSettingsService

    with tempfile.TemporaryDirectory() as tmp:
        service = RuntimeSettingsService(Path(tmp) / "settings", process_env={})
        service.load_system(include_process_overrides=False)  # warm: create file

        samples = []
        for _ in range(500):
            start = time.perf_counter()
            service.load_system(include_process_overrides=False)
            samples.append(time.perf_counter() - start)
        _report("settings.load_system (cache hit)", samples)


def bench_settings_miss() -> None:
    """Cost of the read+normalize work the cache now skips."""
    from deeptutor.services.config import runtime_settings as module

    with tempfile.TemporaryDirectory() as tmp:
        service = module.RuntimeSettingsService(Path(tmp) / "settings", process_env={})
        path = service.path_for("system")
        service.load_system(include_process_overrides=False)
        payload = path.read_text(encoding="utf-8")

        samples = []
        for _ in range(500):
            path.write_text(payload, encoding="utf-8")
            with service._cache_lock:
                service._payload_cache.pop("system", None)
            start = time.perf_counter()
            service.load_system(include_process_overrides=False)
            samples.append(time.perf_counter() - start)
        _report("settings.load_system (forced cache miss)", samples)


# ---------------------------------------------------------------------------
# 2. Embedding batch scheduling
# ---------------------------------------------------------------------------


def bench_embedding_batches() -> None:
    import asyncio

    from deeptutor.services.embedding import client as client_module
    from deeptutor.services.embedding.config import EmbeddingConfig

    class _SleepyAdapter:
        SUPPORTS_INPUT_TYPE = False

        def __init__(self, _config: dict) -> None:
            pass

        async def embed(self, request):
            await asyncio.sleep(0.005)  # stands in for one HTTP round trip
            return type(
                "Resp",
                (),
                {"embeddings": [[0.1, 0.2] for _ in request.texts]},
            )()

    original = client_module._resolve_adapter_class
    client_module._resolve_adapter_class = lambda _binding: _SleepyAdapter
    try:
        config = EmbeddingConfig(
            model="bench",
            api_key="k",
            base_url="https://example.test/v1/embeddings",
            effective_url="https://example.test/v1/embeddings",
            binding="openai",
            batch_size=10,
        )
        client = client_module.EmbeddingClient(config)
        texts = [f"chunk-{index}" for index in range(400)]  # 40 batches

        async def run(concurrency: int) -> float:
            config.max_concurrent_batches = concurrency
            if concurrency == 1:
                config.batch_delay = 0.0001  # force the serial path
            else:
                config.batch_delay = 0.0
            start = time.perf_counter()
            await client.embed(texts)
            return time.perf_counter() - start

        serial = [asyncio.run(run(1)) for _ in range(3)]
        parallel = [asyncio.run(run(4)) for _ in range(3)]
        _report("embed 400 chunks / 40 batches (serial)", serial)
        _report("embed 400 chunks / 40 batches (concurrency=4)", parallel)
    finally:
        client_module._resolve_adapter_class = original


# ---------------------------------------------------------------------------
# 3. Coordinator event publishing
# ---------------------------------------------------------------------------


def bench_coordinator_publish() -> None:
    import asyncio

    from deeptutor.runtime.coordination.memory import MemoryCoordinator

    async def publish(count: int) -> float:
        coordinator = MemoryCoordinator()
        start = time.perf_counter()
        for index in range(count):
            await coordinator.publish_event(
                "turn-bench", {"type": "content", "content": f"chunk-{index}"}
            )
        return time.perf_counter() - start

    _report("coordinator.publish_event x1000", [asyncio.run(publish(1000))])
    _report("coordinator.publish_event x4000", [asyncio.run(publish(4000))])


# ---------------------------------------------------------------------------
# 4. Per-turn answer assembly (the old per-event path)
# ---------------------------------------------------------------------------


def bench_answer_assembly() -> None:
    from deeptutor.services.session._turn_runtime_shared import (
        _assemble_persisted_answer,
    )

    segments = [(None, "In a packet-switched network, data is divided into packets. " * 40)]
    samples = []
    for _ in range(300):
        start = time.perf_counter()
        _assemble_persisted_answer(segments, set())
        samples.append(time.perf_counter() - start)
    _report("assemble 15KB answer (once)", samples)

    # What the pre-fix executor did per event: 3,000 chunks x one assembly.
    total = sum(samples) / len(samples) * 3000
    print(f"{'  -> old per-turn cost over 3,000 events':<58} {total:10.2f}  s")


# ---------------------------------------------------------------------------
# 5. Context tokenization without a cache
# ---------------------------------------------------------------------------


def bench_token_count() -> None:
    from deeptutor.services.session.context_builder import count_tokens

    text = ("A router forwards packets between networks. " * 2000)[:80000]
    try:
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            count_tokens(text)
            samples.append(time.perf_counter() - start)
        _report("count_tokens on 80KB text", samples)
    except Exception as exc:  # tokenizer may be unavailable offline
        print(f"count_tokens skipped: {exc}")


def main() -> None:
    print("DeepTutor batch-1 micro-benchmarks\n")
    bench_settings_load()
    bench_settings_miss()
    print()
    bench_embedding_batches()
    print()
    bench_coordinator_publish()
    print()
    bench_answer_assembly()
    bench_token_count()


if __name__ == "__main__":
    main()
