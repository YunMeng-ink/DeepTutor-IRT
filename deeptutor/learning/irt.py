"""Three-parameter-logistic (3PL) IRT primitives for mastery scoring.

Pure math, no I/O and no model imports: everything here is deterministic and
bounded-cost, so :func:`deeptutor.learning.mastery.compute_mastery` can call it
on every graded answer without measurable overhead.

The 3PL response function

    P(correct | theta) = c + (1 - c) / (1 + exp(-a (theta - b)))

models the probability that a learner of ability ``theta`` answers an item
correctly, where the item is described by

* ``a`` — discrimination (how sharply the item separates abilities);
* ``b`` — difficulty, on the same logit scale as ``theta``;
* ``c`` — the guessing floor (a multiple-choice item can be answered right
  by guessing with probability ~1/#options).

Item parameters are not calibrated from response data — DeepTutor's questions
carry an LLM-assigned difficulty badge (``easy`` / ``medium`` / ``hard``) and a
question type, which :func:`item_from_labels` maps onto the ``b`` / ``c``
scale. The ability estimate and the reference-item mastery transform are
calibrated so the historical gate pacing survives the switch from
recency-weighted accuracy (three recency-strong correct answers on medium
items clear the 0.9 gate; one or two cannot) — see ``mastery.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Sequence

# Difficulty badge -> IRT b parameter (logit scale). With abilities on a
# standard-normal prior, ±1 logit puts easy/hard near the 16%/84% quantiles
# of the population — a deliberately wide but not absurd spacing for badges
# that are uncalibrated model judgements.
DIFFICULTY_B: dict[str, float] = {"easy": -1.0, "medium": 0.0, "hard": 1.0}

# Guessing floor by question type. A choice question is guessable (the exact
# floor comes from its option count), a short answer barely, an open answer
# effectively not.
GUESSING_BY_TYPE: dict[str, float] = {
    "choice": 0.25,
    "short": 0.10,
    "open": 0.05,
}
GUESSING_DEFAULT = 0.10
DEFAULT_DISCRIMINATION = 1.0

# Ability prior: theta ~ N(0, 1), the standard IRT population prior. It
# shrinks thin evidence toward the population mean, which is what keeps a
# single lucky answer from estimating a huge ability.
PRIOR_SD = 1.0

# Fixed EAP quadrature: 49 evenly spaced nodes over ±4.5 with trapezoid
# weights times the prior density. A fixed grid makes the estimate
# deterministic and its cost constant (nodes x window), unlike Newton-style
# MLE which diverges on all-correct / all-wrong histories.
_GRID_LO = -4.5
_GRID_HI = 4.5
_GRID_NODES = 49


def _build_grid() -> tuple[tuple[float, ...], tuple[float, ...]]:
    step = (_GRID_HI - _GRID_LO) / (_GRID_NODES - 1)
    thetas: list[float] = []
    weights: list[float] = []
    for index in range(_GRID_NODES):
        theta = _GRID_LO + index * step
        trapezoid = step if 0 < index < _GRID_NODES - 1 else step / 2.0
        prior = exp(-0.5 * (theta / PRIOR_SD) ** 2)
        thetas.append(theta)
        weights.append(trapezoid * prior)
    return tuple(thetas), tuple(weights)


_GRID_THETAS, _GRID_WEIGHTS = _build_grid()


@dataclass(frozen=True)
class IrtItem:
    """One scored question's 3PL parameters."""

    a: float = DEFAULT_DISCRIMINATION
    b: float = 0.0
    c: float = GUESSING_DEFAULT


# The yardstick the mastery score is read against: the model probability of
# answering this item correctly at the learner's estimated ability. Its
# discrimination (3.0 — a very sharp medium item) is the calibration constant
# that maps the ability scale onto the historical gate pacing: three
# recency-strong correct answers on medium items estimate enough ability to
# clear 0.9 on this item, one or two do not. c=0 because the yardstick is a
# constructed reference, not a guessable question.
REFERENCE_ITEM = IrtItem(a=3.0, b=0.0, c=0.0)


def irt_probability(theta: float, item: IrtItem) -> float:
    """3PL P(correct | theta), overflow-safe for any theta/item scale."""
    z = item.a * (theta - item.b)
    if z <= -35.0:
        return item.c
    if z >= 35.0:
        return 1.0
    return item.c + (1.0 - item.c) / (1.0 + exp(-z))


# Precomputed per-node log P / log(1-P) for the default item. Real windows
# are usually all-default (attempts recorded without difficulty metadata), so
# the ability estimate collapses to two dot products against this table.
_DEFAULT_ITEM = IrtItem()
_DEFAULT_KEY = (DEFAULT_DISCRIMINATION, 0.0, GUESSING_DEFAULT)
_DEFAULT_LOG_TABLE: tuple[tuple[float, float], ...] = tuple(
    (
        log(min(max(irt_probability(theta, _DEFAULT_ITEM), 1e-10), 1.0 - 1e-10)),
        log(1.0 - min(max(irt_probability(theta, _DEFAULT_ITEM), 1e-10), 1.0 - 1e-10)),
    )
    for theta in _GRID_THETAS
)


def item_from_labels(
    difficulty_label: str = "",
    question_type: str = "",
    option_count: int = 0,
) -> IrtItem:
    """Build an item from the metadata a posed question already carries.

    Unknown or missing values fall back to the medium/default item, so a
    question posed without a difficulty badge still scores — just without
    difficulty information.
    """
    b = DIFFICULTY_B.get(str(difficulty_label or "").strip().lower(), 0.0)
    qtype = str(question_type or "").strip().lower()
    c = GUESSING_BY_TYPE.get(qtype, GUESSING_DEFAULT)
    if qtype == "choice" and 2 <= option_count <= 6:
        c = min(0.35, 1.0 / option_count)
    return IrtItem(b=b, c=c)


def estimate_ability(
    outcomes: Sequence[float],
    items: Sequence[IrtItem | None],
    weights: Sequence[float],
) -> float:
    """Expected a posteriori (EAP) ability estimate over the fixed grid.

    ``outcomes`` are 0..1 correctness scores (partial credit allowed),
    ``items`` the matching 3PL parameters (``None`` -> default item), and
    ``weights`` recency weights that turn the likelihood into a weighted
    pseudo-likelihood — the formal version of "recent attempts count more".

    The window is first collapsed to sufficient statistics per unique item,
    so cost is O(window + nodes x distinct_items) with transcendental work
    only once per distinct item per node — a five-attempt window on mixed
    difficulty items costs the same as one on identical items.
    """
    # Sufficient statistics per item: weighted correct mass and weighted
    # incorrect mass. Every attempt on the same item shares one term.
    groups: dict[tuple[float, float, float], list[float]] = {}
    for outcome, item, weight in zip(outcomes, items, weights, strict=True):
        if weight <= 0.0:
            continue
        u = min(max(float(outcome), 0.0), 1.0)
        key = (item.a, item.b, item.c) if item is not None else _DEFAULT_KEY
        acc = groups.get(key)
        if acc is None:
            acc = groups[key] = [0.0, 0.0]
        acc[0] += weight * u
        acc[1] += weight * (1.0 - u)

    numer = 0.0
    denom = 0.0
    uniform_default = list(groups) == [_DEFAULT_KEY]
    for index, grid_weight in enumerate(_GRID_WEIGHTS):
        if uniform_default:
            w_correct, w_incorrect = groups[_DEFAULT_KEY]
            log_p, log_one_minus_p = _DEFAULT_LOG_TABLE[index]
            loglik = w_correct * log_p + w_incorrect * log_one_minus_p
        else:
            theta = _GRID_THETAS[index]
            loglik = 0.0
            for (a, b, c), (w_correct, w_incorrect) in groups.items():
                p = min(max(irt_probability(theta, IrtItem(a, b, c)), 1e-10), 1.0 - 1e-10)
                loglik += w_correct * log(p) + w_incorrect * log(1.0 - p)
        mass = grid_weight * exp(min(0.0, loglik))
        numer += _GRID_THETAS[index] * mass
        denom += mass
    if denom <= 0.0:
        return 0.0
    return numer / denom


def expected_score(theta: float, item: IrtItem = REFERENCE_ITEM) -> float:
    """Model probability of answering ``item`` correctly at ability ``theta``."""
    return irt_probability(theta, item)


__all__ = [
    "DEFAULT_DISCRIMINATION",
    "DIFFICULTY_B",
    "GUESSING_BY_TYPE",
    "GUESSING_DEFAULT",
    "IrtItem",
    "PRIOR_SD",
    "REFERENCE_ITEM",
    "estimate_ability",
    "expected_score",
    "irt_probability",
    "item_from_labels",
]
