"""Memoization of the canonical users file."""

from __future__ import annotations

import json
import os
from pathlib import Path

from deeptutor.multi_user import identity


def _bump(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_mtime_ns + 10_000_000, stat.st_mtime_ns + 10_000_000))


def _read_calls(monkeypatch) -> dict[str, int]:
    calls = {"count": 0}
    original = identity._read_json

    def counting(path: Path):
        calls["count"] += 1
        return original(path)

    monkeypatch.setattr(identity, "_read_json", counting)
    return calls


def test_repeated_loads_hit_the_cache(mu_isolated_root, seed_user, monkeypatch) -> None:
    seed_user("alice", role="admin")
    identity.load_users()  # warm the memo

    calls = _read_calls(monkeypatch)

    first = identity.load_users()
    second = identity.load_users()

    assert first == second
    assert calls["count"] == 0


def test_cached_records_are_copies(mu_isolated_root, seed_user) -> None:
    """Callers get records they may mutate; the memo must stay intact."""
    seed_user("alice", role="admin")
    identity.load_users()

    records = identity.load_users()
    records["alice"]["role"] = "tampered"

    assert identity.get_user("alice")["role"] == "admin"


def test_write_invalidates_the_cache(mu_isolated_root, seed_user) -> None:
    seed_user("alice", role="admin")
    assert identity.get_user("bob") is None

    seed_user("bob")

    assert identity.get_user("bob") is not None


def test_external_file_change_is_picked_up(mu_isolated_root, seed_user) -> None:
    seed_user("alice", role="admin")
    assert identity.get_user("bob") is None

    users_file = identity.USERS_FILE
    payload = json.loads(users_file.read_text(encoding="utf-8"))
    payload["bob"] = {
        "id": "u_bob",
        "hash": "x",
        "role": "user",
        "created_at": "",
    }
    users_file.write_text(json.dumps(payload), encoding="utf-8")
    _bump(users_file)

    assert identity.get_user("bob") is not None


def test_env_bootstrap_admin_still_merges_on_every_call(mu_isolated_root) -> None:
    """The env admin is per-call state; it must never be served from the memo."""
    merged = identity.load_users("root", "hash-value")

    assert "root" in merged
    assert "root" not in identity.load_users()
