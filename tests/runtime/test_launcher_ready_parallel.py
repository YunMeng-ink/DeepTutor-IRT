"""Concurrent readiness waits for the launcher."""

from __future__ import annotations

import time

import pytest

from deeptutor.runtime import launcher


def _waits() -> list[dict]:
    return [
        {
            "name": "backend",
            "url": "http://127.0.0.1:1/",
            "process": None,
            "timeout": 5,
            "env_name": "X",
        },
        {
            "name": "frontend",
            "url": "http://127.0.0.1:2/",
            "process": None,
            "timeout": 5,
            "env_name": "Y",
        },
    ]


def test_services_wait_concurrently(monkeypatch) -> None:
    """Backend and frontend boots overlap instead of running one after the other."""
    calls: list[str] = []

    def _fake_wait(*, name: str, url: str, process, timeout: int, env_name: str, should_stop):
        calls.append(name)
        time.sleep(0.2)

    monkeypatch.setattr(launcher, "_wait_for_http", _fake_wait)

    start = time.monotonic()
    launcher._wait_for_services(waits=_waits(), should_stop=lambda: False)
    elapsed = time.monotonic() - start

    assert calls == ["backend", "frontend"]
    # Serial would take at least 0.4s; concurrent is about 0.2s.
    assert elapsed < 0.35


def test_first_failure_propagates_without_waiting_for_the_sibling(monkeypatch) -> None:
    def _fake_wait(*, name: str, url: str, process, timeout: int, env_name: str, should_stop):
        time.sleep(0.05 if name == "backend" else 5)
        if name == "backend":
            raise RuntimeError("backend exited")

    monkeypatch.setattr(launcher, "_wait_for_http", _fake_wait)

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="backend exited"):
        launcher._wait_for_services(waits=_waits(), should_stop=lambda: False)

    # The caller must not block on the sibling's full timeout.
    assert time.monotonic() - start < 1.0
