"""Token-level coalescing in the agentic streaming facade."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from deeptutor.runtime.agentic.client import (
    _coalescable_delta,
    _openai_stream_chunk,
    _ProviderOpenAIStream,
)


def _stream() -> _ProviderOpenAIStream:
    stream = _ProviderOpenAIStream(
        provider=SimpleNamespace(),
        messages=[],
        tools=None,
        model="test-model",
        max_tokens=10,
        temperature=0.0,
        reasoning_effort=None,
        tool_choice=None,
        extra_kwargs={},
    )
    stream._queue = asyncio.Queue()
    return stream


def _delta(chunk):
    return chunk.choices[0].delta


@pytest.mark.asyncio
async def test_content_chunks_coalesce_after_the_first() -> None:
    stream = _stream()
    for text in ("a", "b", "c", "d"):
        stream._queue.put_nowait(_openai_stream_chunk(content=text))
    stream._queue.put_nowait(None)

    first = await stream.__anext__()
    assert _delta(first).content == "a"
    merged = await stream.__anext__()
    assert _delta(merged).content == "bcd"
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_reasoning_and_content_keep_their_channels() -> None:
    stream = _stream()
    stream._queue.put_nowait(_openai_stream_chunk(reasoning_content="r1"))
    stream._queue.put_nowait(_openai_stream_chunk(content="c1"))
    stream._queue.put_nowait(_openai_stream_chunk(content="c2"))
    stream._queue.put_nowait(_openai_stream_chunk(reasoning_content="r2"))
    stream._queue.put_nowait(None)

    first = await stream.__anext__()
    assert _delta(first).reasoning_content == "r1"
    assert _delta(first).content is None
    second = await stream.__anext__()
    assert _delta(second).content == "c1"
    third = await stream.__anext__()
    assert _delta(third).content == "c2"
    fourth = await stream.__anext__()
    assert _delta(fourth).reasoning_content == "r2"
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_structured_chunks_are_never_coalesced() -> None:
    tool_call = SimpleNamespace(
        id="call_1",
        name="lookup",
        arguments={"q": "x"},
    )
    stream = _stream()
    stream._queue.put_nowait(_openai_stream_chunk(content="a"))
    stream._queue.put_nowait(_openai_stream_chunk(content="b"))
    stream._queue.put_nowait(_openai_stream_chunk(tool_call=tool_call))
    stream._queue.put_nowait(_openai_stream_chunk(finish_reason="tool_calls"))
    stream._queue.put_nowait(None)

    assert _delta(await stream.__anext__()).content == "a"
    assert _delta(await stream.__anext__()).content == "b"
    tool_chunk = await stream.__anext__()
    assert _delta(tool_chunk).tool_calls[0].function.name == "lookup"
    finish = await stream.__anext__()
    assert finish.choices[0].finish_reason == "tool_calls"
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_error_after_buffered_text_is_raised_after_the_text() -> None:
    stream = _stream()
    stream._queue.put_nowait(_openai_stream_chunk(content="a"))
    stream._queue.put_nowait(_openai_stream_chunk(content="b"))
    stream._queue.put_nowait(RuntimeError("provider failed"))
    stream._queue.put_nowait(None)

    assert _delta(await stream.__anext__()).content == "a"
    assert _delta(await stream.__anext__()).content == "b"
    with pytest.raises(RuntimeError, match="provider failed"):
        await stream.__anext__()


def test_coalescable_delta_rejects_structured_payloads():
    assert _coalescable_delta(_openai_stream_chunk(content="text")) == ("content", "text")
    assert _coalescable_delta(_openai_stream_chunk(reasoning_content="think")) == (
        "reasoning",
        "think",
    )
    assert _coalescable_delta(_openai_stream_chunk(finish_reason="stop")) is None
    assert _coalescable_delta(_openai_stream_chunk(usage={"total_tokens": 3})) is None
    assert (
        _coalescable_delta(
            _openai_stream_chunk(
                content="x",
                provider_specific_fields={"tool_args_preview": {"id": "c"}},
            )
        )
        is None
    )