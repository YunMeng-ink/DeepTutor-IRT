"""Mastery scoring policy — BKT (Bayesian Knowledge Tracing) engine.

Replaces the previous recency-weighted accuracy with a standard BKT model,
optionally extended with DBN-BKT for skill prerequisite dependencies.

``compute_mastery`` maps a knowledge point's attempt history to a 0..1 mastery
score (the BKT posterior P(L)). When BKT parameters are supplied, the full
Bayesian update is replayed over the attempt sequence; otherwise the function
falls back to the legacy recency-weighted policy.

Callers (`LearningService.calculate_mastery`) need not change their signature.
"""

from __future__ import annotations

import os

from deeptutor.learning.bkt import BKTEngine, BKTGlobalPriors, BKTParams
from deeptutor.learning.models import KnowledgeType

# ── BKT 启用开关 ───────────────────────────────────────────────────────────
# 环境变量 BKT_ENABLED=0 可回退到旧的 recency-weighted 逻辑
_BKT_ENABLED: bool = os.environ.get("BKT_ENABLED", "1").lower() not in ("0", "false", "no")

# ── 旧逻辑保留（回退用） ───────────────────────────────────────────────────
_RECENCY_WEIGHTS: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 1.0)
_CONFIDENCE_CAP: dict[int, float] = {1: 0.5, 2: 0.8}


def _compute_mastery_legacy(correctness: list[bool]) -> float:
    """Legacy recency-weighted accuracy with low-confidence cap."""
    if not correctness:
        return 0.0
    recent = correctness[-len(_RECENCY_WEIGHTS) :]
    weights = _RECENCY_WEIGHTS[-len(recent) :]
    score = sum(w * (1.0 if c else 0.0) for w, c in zip(recent, weights, strict=True)) / sum(
        weights
    )
    return min(score, _CONFIDENCE_CAP.get(len(recent), 1.0))


def compute_mastery(
    correctness: list[bool],
    *,
    params: BKTParams | None = None,
    knowledge_type: KnowledgeType | str | None = None,
) -> float:
    """Return a 0..1 mastery score from attempt outcomes.

    Args:
        correctness: per-attempt correctness in chronological order.
        params: BKT parameters. If None, uses global priors for the
            given knowledge_type (or generic defaults).
        knowledge_type: used to select type-specific default params
            when ``params`` is None.

    When ``BKT_ENABLED`` env var is "0", falls back to the legacy
    recency-weighted accuracy policy.
    """
    if not correctness:
        return 0.0

    if not _BKT_ENABLED:
        return _compute_mastery_legacy(correctness)

    if params is None:
        params = BKTGlobalPriors.get(knowledge_type)

    engine = BKTEngine(params)
    for is_correct in correctness:
        engine.update(is_correct)
    return engine.get_mastery()


__all__ = ["compute_mastery"]
