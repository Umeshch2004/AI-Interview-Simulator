"""
Top-level grader registry for validator discoverability.

The environment already has its grading logic in `env.tasks` and `env.graders`.
This module exposes three explicit task graders at the repository root so
OpenEnv validators can statically discover them without needing to infer paths
inside the package.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, Optional

from env.tasks import get_task as get_env_task

MIN_SCORE = 0.001
MAX_SCORE = 0.999


def _normalize_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = MIN_SCORE
    return max(MIN_SCORE, min(MAX_SCORE, value))


def _coerce_state_like(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "state") and callable(value.state):
        try:
            return _coerce_state_like(value.state())
        except Exception:
            return {}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump()
        except Exception:
            return {}
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return value.dict()
        except Exception:
            return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            return {}
    return {}


def _extract_score_from_state(state: Dict[str, Any]) -> Optional[float]:
    direct_keys = ("total_score", "average_score", "score", "reward")
    for key in direct_keys:
        value = state.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            nested = value.get("value", value.get("score", value.get("average_score")))
            if nested is not None:
                return float(nested)
        else:
            return float(value)

    history = state.get("history")
    if isinstance(history, Iterable) and not isinstance(history, (str, bytes, dict)):
        rewards = []
        for item in history:
            if not isinstance(item, dict):
                continue
            reward = item.get("reward")
            if reward is None:
                continue
            try:
                rewards.append(float(reward))
            except (TypeError, ValueError):
                continue
        if rewards:
            return sum(rewards) / len(rewards)

    return None


def _matches_task(task_id: str, state: Dict[str, Any]) -> bool:
    candidates = (
        state.get("task"),
        state.get("task_id"),
        state.get("difficulty"),
        state.get("base_task"),
        state.get("task_name"),
    )
    return any(candidate == task_id for candidate in candidates if candidate is not None)


def _grade_from_payload(
    task_id: str,
    *,
    answer: Optional[str],
    question: Optional[str],
    rubric: Optional[Dict[str, Any]],
    expected_concepts: Optional[Iterable[str]],
    category: str,
) -> Optional[float]:
    if not answer or not question or rubric is None or expected_concepts is None:
        return None

    reward, _ = get_env_task(task_id).grade(
        answer=answer,
        question=question,
        rubric=rubric,
        expected_concepts=list(expected_concepts),
        category=category,
    )
    return reward


def _grade_task(
    task_id: str,
    state: Any = None,
    reward: Optional[float] = None,
    *,
    answer: Optional[str] = None,
    question: Optional[str] = None,
    rubric: Optional[Dict[str, Any]] = None,
    expected_concepts: Optional[Iterable[str]] = None,
    category: str = "general",
    **_: Any,
) -> float:
    if reward is not None:
        return _normalize_score(reward)

    state_dict = _coerce_state_like(state)
    if state_dict:
        extracted = _extract_score_from_state(state_dict)
        if extracted is not None and (_matches_task(task_id, state_dict) or state_dict.get("task") is None):
            return _normalize_score(extracted)

    payload_score = _grade_from_payload(
        task_id,
        answer=answer,
        question=question,
        rubric=rubric,
        expected_concepts=expected_concepts,
        category=category,
    )
    if payload_score is not None:
        return _normalize_score(payload_score)

    return MIN_SCORE


def grade_easy(state: Any = None, reward: Optional[float] = None, **kwargs: Any) -> float:
    return _grade_task("easy", state=state, reward=reward, **kwargs)


def grade_medium(state: Any = None, reward: Optional[float] = None, **kwargs: Any) -> float:
    return _grade_task("medium", state=state, reward=reward, **kwargs)


def grade_hard(state: Any = None, reward: Optional[float] = None, **kwargs: Any) -> float:
    return _grade_task("hard", state=state, reward=reward, **kwargs)


class EasyGrader:
    def grade(self, state: Any = None, reward: Optional[float] = None, **kwargs: Any) -> float:
        return grade_easy(state=state, reward=reward, **kwargs)


class MediumGrader:
    def grade(self, state: Any = None, reward: Optional[float] = None, **kwargs: Any) -> float:
        return grade_medium(state=state, reward=reward, **kwargs)


class HardGrader:
    def grade(self, state: Any = None, reward: Optional[float] = None, **kwargs: Any) -> float:
        return grade_hard(state=state, reward=reward, **kwargs)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

GRADER_CLASSES = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
}

TASK_GRADER_PAIRS = [
    ("easy", grade_easy),
    ("medium", grade_medium),
    ("hard", grade_hard),
]

__all__ = [
    "EasyGrader",
    "MediumGrader",
    "HardGrader",
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "GRADERS",
    "GRADER_CLASSES",
    "TASK_GRADER_PAIRS",
]
