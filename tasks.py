"""
Top-level task registry for validator discoverability.

The runtime environment uses `env.tasks`, but some validators look for a
repository-level `tasks.py` with explicit task-to-grader bindings. This file
keeps those bindings static and easy to introspect.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

TASKS: List[Dict[str, object]] = [
    {
        "id": "easy",
        "difficulty": "easy",
        "description": "Basic programming concepts and Python data structures.",
        "max_steps": 5,
        "grader": "graders:grade_easy",
        "grader_class": "graders:EasyGrader",
    },
    {
        "id": "medium",
        "difficulty": "medium",
        "description": "Algorithms, debugging, and complexity analysis.",
        "max_steps": 8,
        "grader": "graders:grade_medium",
        "grader_class": "graders:MediumGrader",
    },
    {
        "id": "hard",
        "difficulty": "hard",
        "description": "System design, advanced DSA, and behavioral follow-ups.",
        "max_steps": 15,
        "grader": "graders:grade_hard",
        "grader_class": "graders:HardGrader",
    },
]

TASKS_BY_ID = {task["id"]: task for task in TASKS}

TASK_GRADER_PAIRS = [
    (task["id"], task["grader"])
    for task in TASKS
]


def get_tasks() -> List[Dict[str, object]]:
    return deepcopy(TASKS)


def get_task(task_id: str) -> Dict[str, object]:
    try:
        return deepcopy(TASKS_BY_ID[task_id])
    except KeyError as exc:
        raise KeyError(f"Unknown task '{task_id}'. Expected one of {list(TASKS_BY_ID)}") from exc


__all__ = [
    "TASKS",
    "TASKS_BY_ID",
    "TASK_GRADER_PAIRS",
    "get_task",
    "get_tasks",
]
