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
    from deeptutor.services.session.context_builder import count_tokens, reset_token_count_cache

    text = ("A router forwards packets between networks. " * 2000)[:80000]
    try:
        # Loading tiktoken's BPE table is a one-time process cost (~100 ms);
        # keep it out of the per-call numbers.
        count_tokens("warm up the tokenizer")
        reset_token_count_cache()
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            count_tokens(text)
            samples.append(time.perf_counter() - start)
        _report("count_tokens on 80KB text, cold", [samples[0]])
        _report("count_tokens on 80KB text, memoized", samples[1:])
    except Exception as exc:  # tokenizer may be unavailable offline
        print(f"count_tokens skipped: {exc}")
    finally:
        reset_token_count_cache()


def bench_token_cache() -> None:
    from deeptutor.services.session.context_builder import (
        count_tokens,
        reset_token_count_cache,
    )

    text = ("A router forwards packets between networks. " * 2000)[:80000]
    reset_token_count_cache()
    count_tokens(text)  # warm

    samples = []
    for _ in range(50):
        start = time.perf_counter()
        count_tokens(text)
        samples.append(time.perf_counter() - start)
    _report("count_tokens 80KB (memoized)", samples)


def bench_turn_preview_cache() -> None:
    import asyncio
    import tempfile

    from deeptutor.services.session.sqlite_store import SQLiteSessionStore

    async def run(tmp: str) -> tuple[float, float]:
        store = SQLiteSessionStore(Path(tmp) / "sessions.db")
        session = await store.create_session("bench")
        turn = await store.create_turn(session["id"], capability="chat")
        events = [{"type": "content", "content": "x" * 200, "seq": seq} for seq in range(1, 401)]
        events.append({"type": "done", "content": "done", "seq": 401})
        await store.append_events(turn["id"], events)

        with store._connect() as conn:

            def measure() -> float:
                start = time.perf_counter()
                store._canonical_trace_preview_sync(conn, turn["id"], session["id"])
                return time.perf_counter() - start

            cold = measure()
            warm = statistics.median([measure() for _ in range(20)])
        return cold, warm

    with tempfile.TemporaryDirectory() as tmp:
        cold, warm = asyncio.run(run(tmp))
    print(f"{'turn preview (401 events), cold':<58} {cold * 1000:10.3f} ms")
    print(f"{'turn preview (401 events), cached':<58} {warm * 1000:10.3f} ms")


def bench_store_concurrency() -> None:
    import asyncio
    import tempfile

    from deeptutor.services.session.sqlite_store import SQLiteSessionStore

    async def run(tmp: str) -> float:
        store = SQLiteSessionStore(Path(tmp) / "sessions.db")

        def slow(value: int) -> int:
            time.sleep(0.02)
            return value

        start = time.perf_counter()
        await asyncio.gather(*(store._run(slow, index) for index in range(4)))
        return time.perf_counter() - start

    with tempfile.TemporaryDirectory() as tmp:
        elapsed = asyncio.run(run(tmp))
    print(f"{'4 store ops, 20 ms each (overlapped)':<58} {elapsed * 1000:10.3f} ms")
    print(f"{'  -> serial would be':<58} {80.0:10.3f} ms")


def bench_user_turn_count() -> None:
    import asyncio
    import tempfile

    from deeptutor.services.session.sqlite_store import SQLiteSessionStore

    async def run(tmp: str) -> tuple[float, float]:
        store = SQLiteSessionStore(Path(tmp) / "sessions.db")
        session = await store.create_session("bench")
        for index in range(200):
            await store.add_message(session["id"], "user", f"question {index}")
            await store.add_message(session["id"], "assistant", f"answer {index}")
        leaf = await store.add_message(session["id"], "user", "latest")

        start = time.perf_counter()
        await store.count_user_turns(session["id"], leaf)
        counted = time.perf_counter() - start

        start = time.perf_counter()
        await store.get_messages(session["id"])
        loaded = time.perf_counter() - start
        return counted, loaded

    with tempfile.TemporaryDirectory() as tmp:
        counted, loaded = asyncio.run(run(tmp))
    print(f"{'user-turn count (401 messages), one query':<58} {counted * 1000:10.3f} ms")
    print(f"{'  -> previous path (load all history)':<58} {loaded * 1000:10.3f} ms")


def bench_retry_schedule() -> None:
    from deeptutor.services.llm.factory import _build_retry_delays

    delays = _build_retry_delays(8, 5.0, True)
    print(f"{'retry waits, default config':<58} {len(delays)} attempts, {sum(delays):.0f} s total")


def bench_file_digest() -> None:
    import tempfile

    from deeptutor.services.file_digest import file_sha256, reset_file_digest_cache

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "big.bin"
        target.write_bytes(b"x" * (8 * 1024 * 1024))
        reset_file_digest_cache()

        samples = []
        for _ in range(6):
            start = time.perf_counter()
            file_sha256(target)
            samples.append(time.perf_counter() - start)
        _report("sha256 8 MiB file, first read", [samples[0]])
        _report("sha256 8 MiB file, memoized", samples[1:])


def main() -> None:
    print("DeepTutor performance micro-benchmarks")
    bench_settings_load()
    bench_settings_miss()
    print()
    bench_embedding_batches()
    print()
    bench_coordinator_publish()
    print()
    bench_answer_assembly()
    bench_token_count()
    print()
    print("batch 2")
    bench_token_cache()
    bench_turn_preview_cache()
    bench_store_concurrency()
    bench_user_turn_count()
    bench_retry_schedule()
    bench_file_digest()


if __name__ == "__main__":
    main()
