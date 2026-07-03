"""Bayesian Knowledge Tracing (BKT) engine with DBN extension.

Provides standard BKT (4-parameter: p_L0, p_T, p_G, p_S) and an optional
DBN-BKT extension (prerequisite-aware transition probability) for mastery
scoring.

Reference: pyBKT (https://github.com/CAHLR/pyBKT) — the DBN extension models
skill dependencies as conditional transition probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deeptutor.learning.models import KnowledgeType


# Per-knowledge-type default priors.
_TYPE_PRIORS: dict[str, dict[str, float]] = {
    "memory": {"p_L0": 0.1, "p_T": 0.3, "p_G": 0.1, "p_S": 0.1},
    "concept": {"p_L0": 0.1, "p_T": 0.1, "p_G": 0.25, "p_S": 0.15},
    "procedure": {"p_L0": 0.05, "p_T": 0.2, "p_G": 0.15, "p_S": 0.1},
    "design": {"p_L0": 0.05, "p_T": 0.08, "p_G": 0.3, "p_S": 0.2},
}

_GENERIC_PRIORS: dict[str, float] = {"p_L0": 0.1, "p_T": 0.1, "p_G": 0.2, "p_S": 0.1}


@dataclass(frozen=True)
class BKTParams:
    """BKT model parameters for a single knowledge point."""

    p_L0: float = 0.1
    p_T: float = 0.1
    p_G: float = 0.2
    p_S: float = 0.1

    def __post_init__(self) -> None:
        for name in ("p_L0", "p_T", "p_G", "p_S"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name}={val} must be in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {"p_L0": self.p_L0, "p_T": self.p_T, "p_G": self.p_G, "p_S": self.p_S}


class BKTEngine:
    """Standard BKT engine with Bayes-rule update.

    Reconstructs the mastery posterior by replaying every observed attempt
    from the prior. Stateless across instantiations — all persistent state
    lives in the caller's storage (``LearningProgress.bkt_params``).
    """

    def __init__(self, params: BKTParams) -> None:
        self._params = params
        self._p_L: float = params.p_L0

    def update(
        self,
        is_correct: bool,
        prerequisites: dict[str, float] | None = None,
    ) -> float:
        """Apply one Bayes update and return the posterior P(L)."""
        p_L = self._p_L
        p_T = self._params.p_T
        p_G = self._params.p_G
        p_S = self._params.p_S

        if prerequisites is not None:
            p_T = self._dbn_adjust_p_T(p_T, prerequisites)

        if is_correct:
            p_L_given = (p_L * (1.0 - p_S)) / (p_L * (1.0 - p_S) + (1.0 - p_L) * p_G)
        else:
            p_L_given = (p_L * p_S) / (p_L * p_S + (1.0 - p_L) * (1.0 - p_G))

        self._p_L = p_L_given + (1.0 - p_L_given) * p_T
        return self._p_L

    def get_mastery(self) -> float:
        """Current posterior P(L)."""
        return self._p_L

    def get_params(self) -> BKTParams:
        """Current parameters used by this engine."""
        return self._params

    @staticmethod
    def _dbn_adjust_p_T(
        p_T: float,
        prerequisites: dict[str, float],
        alpha: float = 0.3,
    ) -> float:
        """Adjust transition probability based on prerequisite mastery.

        p_T_eff = p_T + alpha * max(prerequisite_mastery) * (1 - p_T)

        Args:
            p_T: base transition probability.
            prerequisites: map of prerequisite KP id -> current P(L).
            alpha: dependency strength (default 0.3).

        Returns:
            Clipped effective transition probability in [0, 1].
        """
        if not prerequisites:
            return p_T
        max_mastery = max(prerequisites.values())
        p_T_eff = p_T + alpha * max_mastery * (1.0 - p_T)
        return max(0.0, min(1.0, p_T_eff))


class BKTEngineDBN(BKTEngine):
    """DBN-BKT engine: prerequisite-aware transition probability.

    Accepts an optional ``skill_graph`` mapping knowledge point IDs to their
    prerequisite KP IDs. When ``update()`` is called with a ``prerequisites``
    dict (current P(L) for each prerequisite), the transition probability is
    boosted by the highest prerequisite mastery.

    Falls back to standard BKT when no prerequisites are supplied.
    """

    def __init__(
        self,
        params: BKTParams,
        skill_graph: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(params)
        self._skill_graph: dict[str, list[str]] = skill_graph or {}

    def update(
        self,
        is_correct: bool,
        prerequisites: dict[str, float] | None = None,
    ) -> float:
        return super().update(is_correct, prerequisites=prerequisites)


class BKTGlobalPriors:
    """Population-level default BKT parameters keyed by knowledge type."""

    @staticmethod
    def get(knowledge_type: KnowledgeType | str | None = None) -> BKTParams:
        if knowledge_type is None:
            return BKTParams(**_GENERIC_PRIORS)
        key = str(knowledge_type)
        # KnowledgeType enum: use .value directly to get "memory" not "KnowledgeType.MEMORY"
        try:
            key = knowledge_type.value  # type: ignore[union-attr]
        except AttributeError:
            key = str(knowledge_type).lower()
        priors = _TYPE_PRIORS.get(key)
        if priors is None:
            return BKTParams(**_GENERIC_PRIORS)
        return BKTParams(**priors)


# Offsets are parameter deltas from global priors, keyed by param name.
_OFFSET_LEARNING_RATE: float = 0.01
_OFFSET_CLAMP: tuple[float, float] = (-0.2, 0.2)
_PARAM_CLAMP: tuple[float, float] = (0.01, 0.99)


class BKTUserOffsetTuner:
    """Per-user parameter offset update via online gradient ascent.

    Keeps a lightweight per-KP offset dict ``{p_T: float, p_G: float, p_S: float}``
    that drifts from the global prior based on prediction error.
    """

    @staticmethod
    def apply(global_params: BKTParams, offset: dict[str, float]) -> BKTParams:
        """Return effective params = global + offset (clipped)."""
        return BKTParams(
            p_L0=max(0.01, min(0.99, global_params.p_L0 + offset.get("p_L0", 0.0))),
            p_T=max(0.01, min(0.99, global_params.p_T + offset.get("p_T", 0.0))),
            p_G=max(0.01, min(0.49, global_params.p_G + offset.get("p_G", 0.0))),
            p_S=max(0.01, min(0.49, global_params.p_S + offset.get("p_S", 0.0))),
        )

    @staticmethod
    def update(
        offset: dict[str, float],
        is_correct: bool,
        p_L_before: float,
        params: BKTParams,
    ) -> dict[str, float]:
        """Update offset based on prediction error.

        Args:
            offset: current offset dict (mutated copy).
            is_correct: actual outcome of the just-observed attempt.
            p_L_before: P(L) before this observation.
            params: current effective params.

        Returns:
            Updated offset dict with values clipped to [-0.2, 0.2].
        """
        p_L = max(0.01, min(0.99, p_L_before))
        p_G = max(0.01, min(0.49, params.p_G))
        p_S = max(0.01, min(0.49, params.p_S))

        if is_correct:
            p_obs = p_L * (1.0 - p_S) + (1.0 - p_L) * p_G
        else:
            p_obs = p_L * p_S + (1.0 - p_L) * (1.0 - p_G)

        error = (1.0 if is_correct else 0.0) - p_obs
        eta = _OFFSET_LEARNING_RATE

        p_T_grad = eta * error * (1.0 - p_L)
        p_G_grad = eta * error * (1.0 - p_L) * (-1.0 if is_correct else 1.0)
        p_S_grad = eta * error * p_L * (1.0 if is_correct else -1.0) * (-1.0)

        new_offset = dict(offset)
        new_offset["p_T"] = max(
            _OFFSET_CLAMP[0], min(_OFFSET_CLAMP[1], offset.get("p_T", 0.0) + p_T_grad)
        )
        new_offset["p_G"] = max(
            _OFFSET_CLAMP[0], min(_OFFSET_CLAMP[1], offset.get("p_G", 0.0) + p_G_grad)
        )
        new_offset["p_S"] = max(
            _OFFSET_CLAMP[0], min(_OFFSET_CLAMP[1], offset.get("p_S", 0.0) + p_S_grad)
        )
        return new_offset


__all__ = [
    "BKTParams",
    "BKTEngine",
    "BKTEngineDBN",
    "BKTGlobalPriors",
    "BKTUserOffsetTuner",
]
