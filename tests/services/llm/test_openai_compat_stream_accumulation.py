"""Live stream accumulation matches the chunk parser's final response."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from deeptutor.services.llm.provider_core.openai_compat_provider import (
    OpenAICompatProvider,
)
from deeptutor.services.provider_registry import find_by_name


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


class _FakeCompletions:
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(self.chunks)


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        reasoning=None,
        tool_calls=tool_calls or [],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


def _tool_delta(index: int, *, call_id: str = "", name: str = "", arguments: str = ""):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _provider(chunks: list) -> tuple[OpenAICompatProvider, _FakeCompletions]:
    provider = OpenAICompatProvider(
        api_key="test-key",
        api_base="https://gateway.example/v1",
        default_model="test-model",
        spec=find_by_name("custom"),
        provider_name="custom",
        wire_api="chat",
    )
    completions = _FakeCompletions(chunks)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return provider, completions


def test_chat_stream_accumulates_without_retaining_chunks() -> None:
    chunks = [
        _chunk(reasoning="think "),
        _chunk(reasoning="harder"),
        _chunk(content="Hello "),
        _chunk(content="world"),
        _chunk(
            tool_calls=[_tool_delta(0, call_id="call_1", name="lookup", arguments='{"q":')],
        ),
        _chunk(tool_calls=[_tool_delta(0, arguments='"x"}')], finish_reason="tool_calls"),
        _chunk(usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}),
    ]
    provider, completions = _provider(chunks)
    content_deltas: list[str] = []
    reasoning_deltas: list[str] = []
    tool_args: list[tuple[str, str, str]] = []

    async def on_content(text: str) -> None:
        content_deltas.append(text)

    async def on_reasoning(text: str) -> None:
        reasoning_deltas.append(text)

    async def on_tool_args(call_id: str, name: str, arguments: str) -> None:
        tool_args.append((call_id, name, arguments))

    response = asyncio.run(
        provider.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            on_content_delta=on_content,
            on_reasoning_delta=on_reasoning,
            on_tool_args_delta=on_tool_args,
        )
    )

    assert response.content == "Hello world"
    assert response.reasoning_content == "think harder"
    assert response.finish_reason == "tool_calls"
    assert response.usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert [call.name for call in response.tool_calls] == ["lookup"]
    assert response.tool_calls[0].arguments == {"q": "x"}
    assert content_deltas == ["Hello ", "world"]
    assert reasoning_deltas == ["think ", "harder"]
    assert tool_args[-1] == ("call_1", "lookup", '{"q":"x"}')
    assert completions.calls and completions.calls[0]["stream"] is True


def test_chat_stream_keeps_string_chunks_as_content() -> None:
    provider, _ = _provider(["plain ", "text"])

    response = asyncio.run(provider.chat_stream(messages=[{"role": "user", "content": "hi"}]))

    assert response.content == "plain text"


def test_live_accumulation_matches_parse_chunks() -> None:
    chunks = [
        _chunk(content="A"),
        _chunk(content="B"),
        _chunk(reasoning="R"),
        _chunk(
            tool_calls=[_tool_delta(0, call_id="c1", name="lookup", arguments="{}")],
            finish_reason="tool_calls",
        ),
    ]
    provider, _ = _provider(chunks)

    live = asyncio.run(provider.chat_stream(messages=[{"role": "user", "content": "hi"}]))
    parsed = provider._parse_chunks(chunks)

    assert live.content == parsed.content
    assert live.reasoning_content == parsed.reasoning_content
    assert live.finish_reason == parsed.finish_reason
    assert [(c.name, c.arguments) for c in live.tool_calls] == [
        (c.name, c.arguments) for c in parsed.tool_calls
    ]


def test_zero_usage_deltas_do_not_erase_final_counts() -> None:
    chunks = [
        _chunk(content="x", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        _chunk(usage={"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}),
    ]
    provider, _ = _provider(chunks)

    response = asyncio.run(provider.chat_stream(messages=[{"role": "user", "content": "hi"}]))

    assert response.usage == {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}