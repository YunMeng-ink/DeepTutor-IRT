"""Mastery scoring policy — an IRT-3PL ability estimate, evidence-capped.

``compute_mastery`` maps a knowledge point's attempt history (and, when the
attempt carries them, the posed question's 3PL item parameters) to a 0..1
mastery score:

1. **Recency window** — only the most recent attempts count, each weighted so
   newer attempts matter more (recovery after early mistakes is rewarded).
2. **Ability estimate** — the window feeds a weighted-likelihood 3PL EAP
   estimate of the learner's ability ``theta`` (``irt.estimate_ability``).
   Difficulty and guessing enter here: three correct *hard* items estimate
   more ability than three correct *easy* ones, a correct guessable choice
   item is weaker evidence than a correct open answer, and a miss on an easy
   item is stronger counter-evidence than a miss on a hard one.
3. **Reference transform** — the mastery score is the model probability of
   answering the calibrated reference item (``irt.REFERENCE_ITEM``, a sharp
   medium item) correctly at the estimated ability, so the number stays
   readable as "expected accuracy on this objective's medium questions" and
   the existing 0.9 gate keeps its meaning.
4. **Confidence cap** — unchanged from the previous policy: with one attempt
   the score cannot exceed 0.5 and with two it cannot exceed 0.8, so a lucky
   answer or two can never declare a point mastered.

Calibration anchors (default medium items, guessing 0.1): one correct
-> 0.5 (cap), one partial-credit answer -> ~0.46 (above a miss, below the
one-attempt cap, because the guessing floor places a raw half-credit
response at a slightly negative ability), two correct -> 0.8 (cap), three
correct -> ~0.91 (clears the 0.9 gate), and a recent window of five correct
-> ~0.95. A single miss scores ~0.22 — low, but no longer a hard zero,
because under a Bayesian model one miss is thin evidence about ability, not
proof of none.

This is the one place the pedagogy math lives. The estimator may be swapped
again (a calibrated item bank, BKT, ...) by replacing ``compute_mastery``
alone; callers (``LearningService.calculate_mastery``) need not change.
"""

from __future__ import annotations

from typing import Sequence

from deeptutor.learning.irt import (
    IrtItem,
    estimate_ability,
    expected_score,
)

# Recency weights for the most recent attempts (oldest -> newest). Newer
# attempts count more, so recovery after early mistakes is rewarded. They act
# as likelihood weights in the ability estimate: the same evidence, weighted
# toward the present.
_RECENCY_WEIGHTS: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 1.0)

# Mastery cannot exceed this until enough attempts accumulate, so one or two
# correct answers cannot declare a point "mastered".
_CONFIDENCE_CAP: dict[int, float] = {1: 0.5, 2: 0.8}


def compute_mastery(
    correctness: Sequence[bool | float],
    items: Sequence[IrtItem | None] | None = None,
) -> float:
    """Return a 0..1 mastery score from a knowledge point's attempt outcomes.

    Args:
        correctness: per-attempt correctness or bounded partial credit, chronologically.
        items: matching 3PL item parameters per attempt (``None`` entries and
            a missing list fall back to the default medium item — the shape
            every attempt recorded before IRT scoring, and every caller that
            has no difficulty metadata, produces).
    """
    if not correctness:
        return 0.0
    recent = correctness[-len(_RECENCY_WEIGHTS) :]
    weights = _RECENCY_WEIGHTS[-len(recent) :]
    window_items = list(items[-len(recent) :]) if items is not None else []
    window_items += [None] * (len(recent) - len(window_items))
    clamped = [max(0.0, min(1.0, float(c))) for c in recent]
    theta = estimate_ability(clamped, window_items, weights)
    score = expected_score(theta)
    return min(score, _CONFIDENCE_CAP.get(len(recent), 1.0))


__all__ = ["compute_mastery"]
