"""Tests for the LightRAG multimodal-analysis JSON repair backstop.

LightRAG's analyze stage fail-fasts a whole document when one VLM response
misses a required field, even after its built-in retry. The DeepTutor vision
adapter repairs parseable-but-incomplete responses so a weak model cannot
cost an entire textbook.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from deeptutor.services.llm.config import LLMConfig
from deeptutor.services.rag.pipelines.lightrag.config import build_vision_model_func


def _vision_with_base(monkeypatch: pytest.MonkeyPatch, response: str):
    calls: list[dict] = []

    async def fake_base(prompt="", **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return response

    from deeptutor.services.llm import client as llm_client

    monkeypatch.setattr(
        llm_client,
        "build_model_func_for_config",
        lambda _config, *, allow_multimodal: fake_base,
    )
    vision = build_vision_model_func(
        llm_config=LLMConfig(model="fixture", binding="openai", api_key="offline-fixture")
    )
    return vision, calls


def _analyze(vision, response: str, monkeypatch: pytest.MonkeyPatch) -> tuple[object, list[dict]]:
    vision, calls = _vision_with_base(monkeypatch, response)
    result = asyncio.run(vision(prompt="analyze", image_inputs=[{"base64": "aGVsbG8="}]))
    return result, calls


def test_missing_name_and_type_are_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    response, _ = _analyze(
        None,
        json.dumps({"description": "a network topology diagram"}),
        monkeypatch,
    )

    parsed = json.loads(response)
    assert parsed["name"] == "figure"
    assert parsed["description"] == "a network topology diagram"
    assert isinstance(parsed["type"], str) and parsed["type"].strip()


def test_blank_description_is_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    response, _ = _analyze(
        None,
        json.dumps({"name": "拓扑图", "type": "Other", "description": "  "}),
        monkeypatch,
    )

    parsed = json.loads(response)
    assert parsed["name"] == "拓扑图"
    assert parsed["type"] == "Other"
    assert parsed["description"].strip()


def test_conforming_response_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = json.dumps({"name": "a", "type": "Other", "description": "b"})
    response, _ = _analyze(None, original, monkeypatch)

    assert response == original


def test_unparseable_response_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    original = '{"name": "a", "descript'
    response, _ = _analyze(None, original, monkeypatch)

    assert response == original


def test_non_json_response_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    original = "The image shows a router topology."
    response, _ = _analyze(None, original, monkeypatch)

    assert response == original
