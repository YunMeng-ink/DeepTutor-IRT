from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deeptutor.services.config.knowledge_base_config import KnowledgeBaseConfigService
from deeptutor.services.config.model_catalog import ModelCatalogService
import deeptutor.services.config.runtime_settings as runtime_settings_module
from deeptutor.services.config.runtime_settings import RuntimeSettingsService


def _bump_mtime(path: Path) -> None:
    """Force a stamp change even on coarse-mtime filesystems."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_mtime_ns + 10_000_000, stat.st_mtime_ns + 10_000_000))


class TestRuntimeSettingsCache:
    def test_repeated_loads_hit_the_cache(self, tmp_path, monkeypatch) -> None:
        service = RuntimeSettingsService(tmp_path / "settings", process_env={})
        first = service.load_system(include_process_overrides=False)

        calls = {"count": 0}
        original = runtime_settings_module._json_object

        def counting(path: Path) -> dict:
            calls["count"] += 1
            return original(path)

        monkeypatch.setattr(runtime_settings_module, "_json_object", counting)
        second = service.load_system(include_process_overrides=False)
        third = service.load_system(include_process_overrides=False)

        assert second == first
        assert third == first
        assert calls["count"] == 0

    def test_external_file_change_invalidates_the_cache(self, tmp_path: Path) -> None:
        service = RuntimeSettingsService(tmp_path / "settings", process_env={})
        assert service.load_system(include_process_overrides=False)["backend_port"] == 8001

        path = service.path_for("system")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["backend_port"] = 9005
        path.write_text(json.dumps(payload), encoding="utf-8")
        _bump_mtime(path)

        assert service.load_system(include_process_overrides=False)["backend_port"] == 9005

    def test_save_refreshes_the_cache(self, tmp_path: Path) -> None:
        service = RuntimeSettingsService(tmp_path / "settings", process_env={})
        service.load_system(include_process_overrides=False)

        service.save_system({"backend_port": 9100})
        assert service.load_system(include_process_overrides=False)["backend_port"] == 9100

    def test_cached_loads_return_independent_copies(self, tmp_path: Path) -> None:
        service = RuntimeSettingsService(tmp_path / "settings", process_env={})
        first = service.load_system(include_process_overrides=False)
        first["backend_port"] = 1

        second = service.load_system(include_process_overrides=False)
        assert second["backend_port"] == 8001


class TestModelCatalogCache:
    def test_repeated_loads_hit_the_cache(self, tmp_path, monkeypatch) -> None:
        service = ModelCatalogService(tmp_path / "model_catalog.json")
        first = service.load()

        calls = {"count": 0}
        original = service._read_existing_catalog

        def counting() -> dict:
            calls["count"] += 1
            return original()

        monkeypatch.setattr(service, "_read_existing_catalog", counting)
        second = service.load()
        service.load()

        assert second == first
        assert calls["count"] == 0

    def test_save_refreshes_the_cache(self, tmp_path, monkeypatch) -> None:
        service = ModelCatalogService(tmp_path / "model_catalog.json")
        service.load()

        catalog = service.load()
        catalog["services"]["llm"]["active_profile_id"] = "cached-id"
        service.save(catalog)

        calls = {"count": 0}
        original = service._read_existing_catalog

        def counting() -> dict:
            calls["count"] += 1
            return original()

        monkeypatch.setattr(service, "_read_existing_catalog", counting)
        reloaded = service.load()

        assert reloaded["services"]["llm"]["active_profile_id"] == "cached-id"
        assert calls["count"] == 0


class TestKnowledgeBaseConfigCache:
    @staticmethod
    def _write_config(path: Path, default_kb: str | None) -> None:
        path.write_text(
            json.dumps({"defaults": {"default_kb": default_kb}, "knowledge_bases": {}}),
            encoding="utf-8",
        )

    def test_repeated_refreshes_short_circuit(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "kb_config.json"
        self._write_config(path, None)
        service = KnowledgeBaseConfigService(path)
        assert service.get_default_kb() is None

        calls = {"count": 0}
        original = service._load_config

        def counting() -> dict:
            calls["count"] += 1
            return original()

        monkeypatch.setattr(service, "_load_config", counting)
        assert service.get_default_kb() is None
        service.get_provider_mode("lightrag")

        assert calls["count"] == 0

    def test_external_change_forces_a_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "kb_config.json"
        self._write_config(path, None)
        service = KnowledgeBaseConfigService(path)
        assert service.get_default_kb() is None

        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["defaults"]["default_kb"] = "externally-set"
        path.write_text(json.dumps(payload), encoding="utf-8")
        _bump_mtime(path)

        assert service.get_default_kb() == "externally-set"
