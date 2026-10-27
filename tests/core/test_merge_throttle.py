"""Cumulative text coalescing for subagent transcripts."""

from __future__ import annotations

import pytest

from deeptutor.capabilities.subagent.merge_throttle import MergeThrottle


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


@pytest.mark.asyncio
async def test_small_growth_is_deferred_then_flushed_in_full() -> None:
    emitted: list[tuple[str, dict]] = []
    clock = _Clock()

    async def emit(text, metadata):
        emitted.append((text, dict(metadata)))

    throttle = MergeThrottle(emit, clock=clock)
    await throttle.push("m", "a", {"n": 1})
    clock.value = 0.01
    await throttle.push("m", "ab", {"n": 1})
    clock.value = 0.02
    await throttle.push("m", "abc", {"n": 1})

    assert [text for text, _ in emitted] == ["a"]

    await throttle.flush()

    assert [text for text, _ in emitted] == ["a", "abc"]


@pytest.mark.asyncio
async def test_growth_at_threshold_is_sent_immediately() -> None:
    emitted: list[str] = []
    clock = _Clock()

    async def emit(text, _metadata):
        emitted.append(text)

    throttle = MergeThrottle(emit, min_growth_chars=8, clock=clock)
    await throttle.push("m", "start", {})
    clock.value = 0.01
    await throttle.push("m", "start" + "x" * 8, {})

    assert emitted == ["start", "start" + "x" * 8]


@pytest.mark.asyncio
async def test_elapsed_interval_sends_the_tail() -> None:
    emitted: list[str] = []
    clock = _Clock()

    async def emit(text, _metadata):
        emitted.append(text)

    throttle = MergeThrottle(emit, min_growth_chars=1000, max_interval_seconds=0.04, clock=clock)
    await throttle.push("m", "a", {})
    clock.value = 0.01
    await throttle.push("m", "ab", {})
    clock.value = 0.05
    await throttle.push("m", "abc", {})

    assert emitted == ["a", "abc"]


@pytest.mark.asyncio
async def test_duplicate_text_is_not_reemitted() -> None:
    emitted: list[str] = []

    async def emit(text, _metadata):
        emitted.append(text)

    throttle = MergeThrottle(emit)
    await throttle.push("m", "same", {})
    await throttle.push("m", "same", {})
    await throttle.flush()

    assert emitted == ["same"]


@pytest.mark.asyncio
async def test_non_prefix_update_flushes_pending_first() -> None:
    emitted: list[str] = []
    clock = _Clock()

    async def emit(text, _metadata):
        emitted.append(text)

    throttle = MergeThrottle(emit, clock=clock)
    await throttle.push("m", "a", {})
    clock.value = 0.01
    await throttle.push("m", "ab", {})
    await throttle.push("m", "replacement", {})

    assert emitted == ["a", "ab", "replacement"]


@pytest.mark.asyncio
async def test_flush_all_preserves_first_occurrence_order() -> None:
    emitted: list[tuple[str, str]] = []

    async def emit(text, metadata):
        emitted.append((metadata["id"], text))

    throttle = MergeThrottle(emit)
    await throttle.push("a", "a1", {"id": "a"})
    await throttle.push("b", "b1", {"id": "b"})
    await throttle.push("a", "a12", {"id": "a"})
    await throttle.push("b", "b12", {"id": "b"})
    await throttle.flush()

    assert emitted == [("a", "a1"), ("b", "b1"), ("a", "a12"), ("b", "b12")]