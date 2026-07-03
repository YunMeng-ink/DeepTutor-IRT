"""Unit tests for the BKT engine (``deeptutor.learning.bkt``)."""

from __future__ import annotations

import pytest

from deeptutor.learning.bkt import (
    BKTEngine,
    BKTEngineDBN,
    BKTGlobalPriors,
    BKTParams,
    BKTUserOffsetTuner,
)
from deeptutor.learning.models import KnowledgeType

# ── BKTParams ───────────────────────────────────────────────────────────────


class TestBKTParams:
    def test_default_values(self):
        p = BKTParams()
        assert p.p_L0 == 0.1
        assert p.p_T == 0.1
        assert p.p_G == 0.2
        assert p.p_S == 0.1

    def test_custom_values(self):
        p = BKTParams(p_L0=0.5, p_T=0.3, p_G=0.15, p_S=0.05)
        assert p.p_L0 == 0.5
        assert p.p_G == 0.15

    def test_to_dict_roundtrip(self):
        p = BKTParams(p_L0=0.1, p_T=0.3, p_G=0.1, p_S=0.1)
        d = p.to_dict()
        assert d == {"p_L0": 0.1, "p_T": 0.3, "p_G": 0.1, "p_S": 0.1}

    def test_invalid_param_raises(self):
        with pytest.raises(ValueError):
            BKTParams(p_T=1.5)

    def test_negative_param_raises(self):
        with pytest.raises(ValueError):
            BKTParams(p_G=-0.1)


# ── BKTEngine ───────────────────────────────────────────────────────────────


class TestBKTEngine:
    def test_empty_correctness(self):
        engine = BKTEngine(BKTParams())
        assert engine.get_mastery() == pytest.approx(BKTParams().p_L0)

    def test_single_correct_increases(self):
        engine = BKTEngine(BKTParams(p_L0=0.5, p_T=0.1, p_G=0.2, p_S=0.1))
        engine.update(True)
        assert engine.get_mastery() == pytest.approx(0.836, abs=1e-3)

    def test_single_wrong_decreases(self):
        engine = BKTEngine(BKTParams(p_L0=0.5, p_T=0.1, p_G=0.2, p_S=0.1))
        engine.update(False)
        assert engine.get_mastery() == pytest.approx(0.200, abs=1e-3)

    def test_three_correct_exceeds_gate(self):
        engine = BKTEngine(BKTParams(p_L0=0.5, p_T=0.1, p_G=0.2, p_S=0.1))
        engine.update(True)
        engine.update(True)
        engine.update(True)
        assert engine.get_mastery() > 0.9

    def test_correct_then_wrong_decreases(self):
        engine = BKTEngine(BKTParams(p_L0=0.5, p_T=0.1, p_G=0.2, p_S=0.1))
        engine.update(True)
        after_correct = engine.get_mastery()
        engine.update(False)
        assert engine.get_mastery() < after_correct

    def test_wrong_then_correct_recovers(self):
        engine = BKTEngine(BKTParams(p_L0=0.5, p_T=0.1, p_G=0.2, p_S=0.1))
        engine.update(False)
        after_wrong = engine.get_mastery()
        engine.update(True)
        assert engine.get_mastery() > after_wrong

    def test_monotonic_correct_streak(self):
        engine = BKTEngine(BKTParams(p_L0=0.1, p_T=0.1, p_G=0.2, p_S=0.1))
        scores = []
        for _ in range(5):
            engine.update(True)
            scores.append(engine.get_mastery())
        assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))

    def test_mastery_bounded(self):
        engine = BKTEngine(BKTParams())
        for _ in range(50):
            engine.update(True)
        assert 0.0 <= engine.get_mastery() <= 1.0

    def test_memory_type_params(self):
        engine = BKTEngine(BKTGlobalPriors.get(KnowledgeType.MEMORY))
        engine.update(True)
        assert 0.0 <= engine.get_mastery() <= 1.0

    def test_concept_type_params(self):
        engine = BKTEngine(BKTGlobalPriors.get(KnowledgeType.CONCEPT))
        engine.update(True)
        assert 0.0 <= engine.get_mastery() <= 1.0

    def test_generic_priors_when_no_type(self):
        p = BKTGlobalPriors.get(None)
        assert p.p_L0 == 0.1
        assert p.p_T == 0.1

    def test_unknown_type_falls_back_to_generic(self):
        p = BKTGlobalPriors.get("unknown_type")
        assert p.p_L0 == 0.1

    def test_dbn_no_prerequisites_falls_back(self):
        engine = BKTEngine(BKTParams(p_L0=0.5, p_T=0.1))
        m1 = engine.update(True)
        engine2 = BKTEngine(BKTParams(p_L0=0.5, p_T=0.1))
        m2 = engine2.update(True, prerequisites=None)
        assert m1 == pytest.approx(m2, abs=1e-6)


# ── BKTEngineDBN ────────────────────────────────────────────────────────────


class TestBKTEngineDBN:
    def test_prerequisite_boosts_transition(self):
        engine = BKTEngineDBN(
            BKTParams(p_L0=0.1, p_T=0.1, p_G=0.2, p_S=0.1),
        )
        engine.update(True, prerequisites={"kp_prev": 0.9})
        score_with_prereq = engine.get_mastery()

        engine2 = BKTEngineDBN(BKTParams(p_L0=0.1, p_T=0.1, p_G=0.2, p_S=0.1))
        engine2.update(True, prerequisites={"kp_prev": 0.0})
        score_without_prereq = engine2.get_mastery()

        assert score_with_prereq > score_without_prereq

    def test_no_prerequisite_falls_back(self):
        engine = BKTEngineDBN(BKTParams(p_L0=0.5, p_T=0.1))
        engine.update(True, prerequisites=None)
        standard = BKTEngine(BKTParams(p_L0=0.5, p_T=0.1))
        standard.update(True)
        assert engine.get_mastery() == pytest.approx(standard.get_mastery(), abs=1e-6)

    def test_empty_prerequisites_dict_no_error(self):
        engine = BKTEngineDBN(BKTParams())
        engine.update(True, prerequisites={})
        assert 0.0 <= engine.get_mastery() <= 1.0


# ── BKTUserOffsetTuner ──────────────────────────────────────────────────────


class TestUserOffsetTuner:
    def test_apply_zero_offset_returns_global(self):
        global_params = BKTParams(p_T=0.3, p_G=0.1, p_S=0.1)
        result = BKTUserOffsetTuner.apply(global_params, {})
        assert result.p_T == 0.3
        assert result.p_G == 0.1
        assert result.p_S == 0.1

    def test_apply_positive_offset_on_p_T(self):
        global_params = BKTParams(p_T=0.1, p_G=0.2, p_S=0.1)
        result = BKTUserOffsetTuner.apply(global_params, {"p_T": 0.2})
        assert result.p_T == pytest.approx(0.3)

    def test_apply_negative_offset_on_p_G(self):
        global_params = BKTParams(p_T=0.1, p_G=0.2, p_S=0.1)
        result = BKTUserOffsetTuner.apply(global_params, {"p_G": -0.1})
        assert result.p_G == pytest.approx(0.1)

    def test_apply_offset_clamps_to_valid_range(self):
        global_params = BKTParams(p_G=0.2)
        result = BKTUserOffsetTuner.apply(global_params, {"p_G": 10.0})
        assert result.p_G <= 0.49
        assert result.p_G >= 0.01

    def test_update_correct_increases_p_T(self):
        offset = {}
        params = BKTParams(p_T=0.1, p_G=0.2, p_S=0.1)
        new_offset = BKTUserOffsetTuner.update(offset, True, 0.5, params)
        assert new_offset["p_T"] > 0.0

    def test_update_wrong_decreases_p_G(self):
        offset = {}
        params = BKTParams(p_T=0.1, p_G=0.2, p_S=0.1)
        new_offset = BKTUserOffsetTuner.update(offset, False, 0.5, params)
        assert new_offset["p_G"] < 0.0

    def test_update_clamps_offset_range(self):
        offset = {}
        params = BKTParams(p_T=0.5, p_G=0.5, p_S=0.5)
        for _ in range(200):
            offset = BKTUserOffsetTuner.update(offset, True, 0.99, params)
        assert all(-0.2 <= v <= 0.2 for v in offset.values())

    def test_update_persists_across_reconstruction(self):
        offset = {}
        params = BKTParams(p_T=0.1, p_G=0.2, p_S=0.1)
        for _ in range(10):
            offset = BKTUserOffsetTuner.update(offset, True, 0.8, params)
        assert offset["p_T"] != 0.0
        new_params = BKTUserOffsetTuner.apply(params, offset)
        assert new_params.p_T != params.p_T


# ── BKTGlobalPriors ────────────────────────────────────────────────────────


class TestGlobalPriors:
    def test_per_type_defaults(self):
        assert BKTGlobalPriors.get(KnowledgeType.MEMORY).p_T == 0.3
        assert BKTGlobalPriors.get(KnowledgeType.CONCEPT).p_G == 0.25
        assert BKTGlobalPriors.get(KnowledgeType.PROCEDURE).p_L0 == 0.05
        assert BKTGlobalPriors.get(KnowledgeType.DESIGN).p_T == 0.08

    def test_unknown_type_fallback(self):
        p = BKTGlobalPriors.get("nonexistent")
        assert p.p_L0 == 0.1

    def test_none_type_uses_generic(self):
        p = BKTGlobalPriors.get(None)
        assert p.p_L0 == 0.1
        assert p.p_G == 0.2


__all__ = [
    "TestBKTParams",
    "TestBKTEngine",
    "TestBKTEngineDBN",
    "TestUserOffsetTuner",
    "TestGlobalPriors",
]
