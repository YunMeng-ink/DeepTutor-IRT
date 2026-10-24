"""Tests for the IRT-3PL mastery scoring model (``learning.irt`` + ``mastery``).

The policy contract these pin:

* **Anchors** — one correct -> 0.5 (cap), two -> 0.8 (cap), three -> >= 0.9
  (gate-clearing). Finite evidence never yields certainty 1.0.
* **Difficulty sensitivity** — the same correctness earns more mastery on
  harder items, and a miss on an easy item is stronger counter-evidence than
  a miss on a hard one. This is the scientific point of IRT scoring.
* **Guessing sensitivity** — a correct guessable choice answer is weaker
  evidence than a correct open answer.
* **Recency** — weighted likelihood keeps recovery ranked above decline.
* **Bounded cost** — the estimator is a fixed grid over a fixed window, so
  scoring stays microseconds regardless of history length.
"""

import math
import time

import pytest

from deeptutor.learning.irt import (
    DIFFICULTY_B,
    IrtItem,
    estimate_ability,
    expected_score,
    irt_probability,
    item_from_labels,
)
from deeptutor.learning.mastery import compute_mastery

# ── 3PL response function ──────────────────────────────────────────────────


def test_irt_probability_matches_the_3pl_formula():
    item = IrtItem(a=1.0, b=0.5, c=0.2)
    for theta in (-2.0, -0.5, 0.5, 2.0):
        expected = 0.2 + 0.8 / (1.0 + math.exp(-(theta - 0.5)))
        assert irt_probability(theta, item) == pytest.approx(expected, abs=1e-12)


def test_irt_probability_floor_is_the_guessing_parameter():
    item = IrtItem(c=0.25)
    assert irt_probability(-100.0, item) == pytest.approx(0.25)
    assert irt_probability(100.0, IrtItem(c=0.0)) == pytest.approx(1.0)
    # At the item's own difficulty the response sits midway between the
    # guessing floor and certainty.
    assert irt_probability(DIFFICULTY_B["hard"], IrtItem(b=DIFFICULTY_B["hard"], c=0.2)) == (
        pytest.approx(0.6)
    )


def test_irt_probability_is_monotone_in_theta():
    item = IrtItem(a=1.7, b=0.3, c=0.15)
    probs = [irt_probability(theta / 10.0, item) for theta in range(-30, 31)]
    assert probs == sorted(probs)


# ── item parameters from question metadata ─────────────────────────────────


def test_item_from_labels_maps_badges_to_difficulty():
    assert item_from_labels("hard") == IrtItem(b=DIFFICULTY_B["hard"])
    assert item_from_labels("EASY") == IrtItem(b=DIFFICULTY_B["easy"])
    assert item_from_labels("", "open") == IrtItem(c=0.05)
    assert item_from_labels("nonsense") == IrtItem()  # unknown -> medium default


def test_item_from_labels_derives_guessing_from_option_count():
    four_options = item_from_labels("medium", "choice", option_count=4)
    assert four_options.c == pytest.approx(0.25)
    two_options = item_from_labels("medium", "choice", option_count=2)
    assert two_options.c == pytest.approx(0.35)  # 1/2 clamped by the 0.35 cap
    fallback = item_from_labels("medium", "choice")  # no option count -> 4-ish
    assert fallback.c == pytest.approx(0.25)


# ── ability estimation ─────────────────────────────────────────────────────


def test_estimate_ability_uses_evidence_direction_and_stays_bounded():
    item = IrtItem()
    correct = estimate_ability([1.0, 1.0, 1.0], [item, item, item], [1.0, 1.0, 1.0])
    wrong = estimate_ability([0.0, 0.0, 0.0], [item, item, item], [1.0, 1.0, 1.0])
    assert correct > 0.0 > wrong
    assert abs(correct) < 4.5 and abs(wrong) < 4.5  # inside the fixed grid


def test_estimate_ability_harder_correct_answers_estimate_more_ability():
    medium = IrtItem(b=0.0)
    hard = IrtItem(b=1.0)
    on_medium = estimate_ability([1.0] * 3, [medium] * 3, [1.0] * 3)
    on_hard = estimate_ability([1.0] * 3, [hard] * 3, [1.0] * 3)
    assert on_hard > on_medium


def test_expected_score_reference_item_separates_gate_pacing():
    # The reference transform maps the N(0,1) ability scale onto mastery:
    # ability 0 (population mean) sits at 0.5, and the gate-clearing ability
    # sits near +0.75 — what three recency-strong correct medium answers
    # estimate.
    assert expected_score(0.0) == pytest.approx(0.5)
    assert expected_score(0.4) < 0.9
    assert expected_score(0.75) >= 0.9


# ── compute_mastery policy ─────────────────────────────────────────────────


def test_empty_history_is_zero():
    assert compute_mastery([]) == 0.0


def test_confidence_caps_bind_for_thin_evidence():
    assert compute_mastery([True]) == 0.5
    assert compute_mastery([True, True]) == 0.8
    assert compute_mastery([False, True]) < 0.5


def test_same_correctness_scores_higher_on_harder_items():
    hard = IrtItem(b=DIFFICULTY_B["hard"])
    medium = IrtItem(b=DIFFICULTY_B["medium"])
    easy = IrtItem(b=DIFFICULTY_B["easy"])
    on_hard = compute_mastery([True] * 3, [hard] * 3)
    on_medium = compute_mastery([True] * 3, [medium] * 3)
    on_easy = compute_mastery([True] * 3, [easy] * 3)
    assert on_hard > on_medium > on_easy
    # Acing easy questions must not clear the medium gate...
    assert on_easy < 0.9
    # ...while acing hard ones clears it with room to spare.
    assert on_hard > 0.9


def test_miss_on_easy_item_is_stronger_counter_evidence():
    easy = IrtItem(b=DIFFICULTY_B["easy"])
    hard = IrtItem(b=DIFFICULTY_B["hard"])
    assert compute_mastery([False] * 3, [easy] * 3) < compute_mastery([False] * 3, [hard] * 3)


def test_correct_guessable_answer_is_weaker_evidence():
    choice = IrtItem(c=0.25)
    open_answer = IrtItem(c=0.05)
    assert compute_mastery([True] * 3, [choice] * 3) < compute_mastery(
        [True] * 3, [open_answer] * 3
    )


def test_wrong_on_guessable_item_is_stronger_counter_evidence():
    choice = IrtItem(c=0.25)
    open_answer = IrtItem(c=0.05)
    assert compute_mastery([False] * 3, [choice] * 3) < compute_mastery(
        [False] * 3, [open_answer] * 3
    )


def test_mixed_items_are_read_in_attempt_order():
    hard = IrtItem(b=1.0)
    recovering = compute_mastery([False, True, True], [hard, hard, hard])
    declining = compute_mastery([True, True, False], [hard, hard, hard])
    assert recovering > declining


def test_none_items_fall_back_to_the_default_medium_item():
    defaults = compute_mastery([True, False, True])
    explicit = compute_mastery([True, False, True], [None, None, None])
    assert defaults == pytest.approx(explicit)


def test_partial_credit_lands_between_wrong_and_right():
    assert compute_mastery([False] * 3) < compute_mastery([0.5] * 3) < compute_mastery([True] * 3)


def test_out_of_range_correctness_is_clamped():
    assert compute_mastery([7.0]) == compute_mastery([True])
    assert compute_mastery([-3.0]) == compute_mastery([False])


def test_scores_are_deterministic():
    history = [1, 0, 1, 1, 0]
    assert [compute_mastery(history) for _ in range(3)] == [
        compute_mastery(history) for _ in range(3)
    ]


# ── bounded cost ───────────────────────────────────────────────────────────


def test_scoring_cost_stays_bounded_with_history_length():
    """Only the recency window feeds the estimate, and the estimate is a
    fixed grid — long histories must not cost more than short ones."""
    long_history = ([1, 0, 1] * 100) + [1, 1, 0, 1, 1]
    assert compute_mastery(long_history) == compute_mastery([1, 1, 0, 1, 1])

    start = time.perf_counter()
    for _ in range(200):
        compute_mastery(long_history)
    elapsed = time.perf_counter() - start
    # ~10 microseconds a call on the default-item fast path; the bound is
    # deliberately generous so slow CI machines do not flake.
    assert elapsed < 2.0


# ── service-level plumbing: posed-question metadata reaches the score ───────


def _service_with_kp(tmp_path):
    from deeptutor.learning.models import (
        KnowledgePoint,
        KnowledgeType,
        LearningModule,
        LearningProgress,
    )
    from deeptutor.learning.service import LearningService
    from deeptutor.learning.storage import LearningStore

    store = LearningStore(root=tmp_path)
    service = LearningService(store)
    progress = LearningProgress(book_id="book1")
    progress.modules = [
        LearningModule(
            id="m1",
            name="Module 1",
            order=0,
            knowledge_points=[
                KnowledgePoint(id="kp1", name="KP1", type=KnowledgeType.MEMORY, module_id="m1")
            ],
        )
    ]
    progress.current_module_id = "m1"
    progress.knowledge_types["kp1"] = KnowledgeType.MEMORY
    service.save(progress)
    return service


def _grade_three(service, difficulty: str, question_type: str = "short", options=()):
    from deeptutor.learning.models import PendingOption, PendingQuestion

    for index in range(3):
        service.register_question(
            "book1",
            PendingQuestion(
                question_id=f"q-{difficulty}-{index}",
                knowledge_point_id="kp1",
                module_id="m1",
                prompt=f"Question {index}",
                expected_answer="answer",
                question_type=question_type,
                options=[PendingOption(label=label, body=label) for label in options]
                if options
                else [],
                difficulty=difficulty,
            ),
            session_id="s1",
            turn_id=f"t-{index}",
        )
        progress, _interaction, _replayed = service.grade_interaction(
            "book1",
            answer="answer",
            question_id=f"q-{difficulty}-{index}",
            session_id="s1",
            turn_id=f"t-{index}",
        )
    return progress


def test_grade_interaction_captures_irt_item_params(tmp_path):
    service = _service_with_kp(tmp_path)
    progress = _grade_three(service, "hard", question_type="choice", options=("A", "B", "C", "D"))

    attempt = progress.quiz_attempts[-1]
    assert attempt.item_difficulty == pytest.approx(DIFFICULTY_B["hard"])
    assert attempt.item_guessing == pytest.approx(0.25)
    # The persisted aggregate keeps the parameters for later recomputation.
    stored = service.store.load("book1")
    assert stored.quiz_attempts[-1].item_difficulty == pytest.approx(1.0)


def test_hard_correct_answers_outscore_easy_correct_answers(tmp_path):
    hard_path = _grade_three(_service_with_kp(tmp_path / "a"), "hard")
    easy_path = _grade_three(_service_with_kp(tmp_path / "b"), "easy")

    hard_mastery = hard_path.mastery_levels["kp1"]
    easy_mastery = easy_path.mastery_levels["kp1"]
    assert hard_mastery >= 0.9 > easy_mastery


def test_legacy_attempts_without_item_params_still_score(tmp_path):
    """Attempts recorded before IRT scoring carry no item parameters; they
    fall back to the default medium item instead of breaking the estimate."""
    from deeptutor.learning.models import QuizAttempt
    from deeptutor.learning.service import LearningService

    service = _service_with_kp(tmp_path)
    progress = service.store.load("book1")
    for index in range(3):
        progress.quiz_attempts.append(
            QuizAttempt(question_id=f"legacy-{index}", knowledge_point_id="kp1", is_correct=True)
        )
    assert service.calculate_mastery(progress, "kp1") == pytest.approx(
        compute_mastery([True, True, True]), abs=1e-9
    )
