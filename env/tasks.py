"""
Task Definitions with Explicit Graders for AI Interview Simulator.

This module defines the 3 interview tasks (easy, medium, hard) with their
associated grader functions. Each task has a unique difficulty level and
grading rubric.
"""

from typing import Callable, Dict, Any, Tuple
from .graders import compute_reward

# ─── Task Type ────────────────────────────────────────────────────────────────
class Task:
    """A task represents a difficulty level with a grading function."""
    
    def __init__(self, name: str, difficulty: str, grader: Callable):
        self.name = name
        self.difficulty = difficulty
        self.grader = grader
    
    def grade(self, answer: str, question: str, rubric: Dict, 
              expected_concepts: list, category: str = "dsa") -> Tuple[float, Any]:
        """Grade an answer using this task's grader."""
        return self.grader(
            answer=answer,
            question=question,
            rubric=rubric,
            expected_concepts=expected_concepts,
            category=category,
            task=self.difficulty
        )


# ─── Grader Functions ─────────────────────────────────────────────────────────
def _easy_grader(answer: str, question: str, rubric: Dict, 
                 expected_concepts: list, category: str, task: str) -> Tuple[float, Any]:
    """Grader for easy difficulty questions."""
    reward, breakdown = compute_reward(
        answer=answer,
        question=question,
        rubric=rubric,
        expected_concepts=expected_concepts,
        category=category,
        task="easy",
        question_num=1,
        total_q=3,
    )
    return reward, breakdown


def _medium_grader(answer: str, question: str, rubric: Dict, 
                   expected_concepts: list, category: str, task: str) -> Tuple[float, Any]:
    """Grader for medium difficulty questions."""
    reward, breakdown = compute_reward(
        answer=answer,
        question=question,
        rubric=rubric,
        expected_concepts=expected_concepts,
        category=category,
        task="medium",
        question_num=1,
        total_q=4,
    )
    return reward, breakdown


def _hard_grader(answer: str, question: str, rubric: Dict, 
                 expected_concepts: list, category: str, task: str) -> Tuple[float, Any]:
    """Grader for hard difficulty questions with follow-up support."""
    reward, breakdown = compute_reward(
        answer=answer,
        question=question,
        rubric=rubric,
        expected_concepts=expected_concepts,
        category=category,
        task="hard",
        question_num=1,
        total_q=5,
    )
    return reward, breakdown


# ─── Task Registry ────────────────────────────────────────────────────────────
TASKS = {
    "easy": Task(
        name="Easy Interview - Python Basics & Data Structures",
        difficulty="easy",
        grader=_easy_grader,
    ),
    "medium": Task(
        name="Medium Interview - Algorithms & Problem Solving",
        difficulty="medium",
        grader=_medium_grader,
    ),
    "hard": Task(
        name="Hard Interview - System Design & Advanced Topics",
        difficulty="hard",
        grader=_hard_grader,
    ),
}


def get_task(difficulty: str) -> Task:
    """Get a task by difficulty level."""
    if difficulty not in TASKS:
        raise ValueError(f"Unknown task difficulty: {difficulty}. Choose from {list(TASKS.keys())}")
    return TASKS[difficulty]


def get_all_tasks() -> Dict[str, Task]:
    """Get all available tasks."""
    return TASKS.copy()
