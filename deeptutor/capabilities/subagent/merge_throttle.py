"""Throttle cumulative subagent text before it enters the turn event stream.

Every local CLI backend reports an answer as *cumulative* text (each delta
carries everything so far), and the transcript renders the latest event under a
``merge_id``. Forwarding every delta therefore re-sends O(answer²) bytes and
persists one event per token for no visible benefit.

This throttle keeps that exact semantic — the last event still carries the full
text — while skipping intermediate prefixes: at most one event per
``min_growth_chars`` characters, or per ``max_interval_seconds``. Tool rows and
non-text channels are never merged, and all pending text is flushed before any
other event so transcript order is unchanged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import time
from typing import Any


class MergeThrottle:
    """Coalesce cumulative text events that share a merge id."""

    def __init__(
        self,
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
        *,
        min_growth_chars: int = 64,
        max_interval_seconds: float = 0.04,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._emit = emit
        self._min_growth = max(1, int(min_growth_chars))
        self._max_interval = max(0.0, float(max_interval_seconds))
        self._clock = clock
        self._last_text: dict[str, str] = {}
        self._last_sent_at: dict[str, float] = {}
        self._pending: dict[str, tuple[str, dict[str, Any]]] = {}

    async def push(
        self,
        merge_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """Send *text*, or remember it until it grows enough to be worth a frame."""
        previous = self._last_text.get(merge_id)
        if previous is not None and text == previous:
            return
        if previous is not None and text.startswith(previous):
            grown = len(text) - len(previous)
            elapsed = self._clock() - self._last_sent_at.get(merge_id, 0.0)
            if grown < self._min_growth and elapsed < self._max_interval:
                self._pending[merge_id] = (text, metadata)
                return
        pending = self._pending.get(merge_id)
        if pending is not None and text.startswith(pending[0]):
            # The new cumulative text supersedes the deferred prefix; sending
            # both would only add a frame the transcript immediately replaces.
            self._pending.pop(merge_id, None)
        else:
            await self.flush(merge_id)
        await self._send(merge_id, text, metadata)

    async def flush(self, merge_id: str | None = None) -> None:
        """Send the pending text for one merge id, or for every merge id."""
        keys = [merge_id] if merge_id is not None else list(self._pending)
        for key in keys:
            entry = self._pending.pop(key, None)
            if entry is None:
                continue
            text, metadata = entry
            await self._send(key, text, metadata)

    async def _send(self, merge_id: str, text: str, metadata: dict[str, Any]) -> None:
        self._last_text[merge_id] = text
        self._last_sent_at[merge_id] = self._clock()
        await self._emit(text, metadata)


__all__ = ["MergeThrottle"]