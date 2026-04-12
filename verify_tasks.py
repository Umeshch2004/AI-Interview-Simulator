"""
Validator-focused verification for task and grader discoverability.

Run this before submission to confirm:
1. `openenv.yaml` declares at least 3 tasks with explicit graders.
2. Each manifest grader import resolves successfully.
3. Root-level `tasks.py` and `graders.py` expose the same 3 tasks.
4. Internal environment graders still execute correctly.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import yaml

from env import get_all_tasks as get_internal_tasks
from env import get_task as get_internal_task
from graders import GRADERS
from tasks import get_tasks as get_public_tasks

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "openenv.yaml"


TEST_SAMPLES = {
    "easy": {
        "question": "What is a dictionary in Python?",
        "answer": (
            "A dictionary is a key-value data structure that stores values under "
            "unique, hashable keys and supports fast lookup."
        ),
        "rubric": {"keywords": ["key", "value", "hash", "lookup"], "min_length": 30},
        "expected_concepts": ["key-value storage", "hash map", "unique keys", "fast retrieval"],
        "category": "dsa",
    },
    "medium": {
        "question": "How do you detect a cycle in a linked list?",
        "answer": (
            "Use Floyd's tortoise and hare algorithm with slow and fast pointers. "
            "If they meet, a cycle exists. The time complexity is O(n) and space is O(1)."
        ),
        "rubric": {"keywords": ["slow", "fast", "pointer", "floyd"], "min_length": 50},
        "expected_concepts": ["Floyd's cycle-finding algorithm", "two pointers", "O(1) space"],
        "category": "dsa",
    },
    "hard": {
        "question": "Design a distributed rate limiter for a public API like Twitter.",
        "answer": (
            "Use a token bucket stored in Redis for distributed coordination, shard by "
            "user or API key, and add regional replicas plus monitoring for failover."
        ),
        "rubric": {"keywords": ["token", "bucket", "redis", "distributed"], "min_length": 120},
        "expected_concepts": ["token bucket", "redis", "distributed cache", "high availability"],
        "category": "system_design",
    },
}


def _status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _load_manifest() -> Dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _resolve_grader(grader_ref: str) -> Tuple[Callable[..., float], Any]:
    if not isinstance(grader_ref, str) or ":" not in grader_ref:
        raise ValueError(f"Invalid grader reference: {grader_ref!r}")

    module_name, attr_name = grader_ref.split(":", 1)
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)

    if inspect.isclass(attr):
        instance = attr()
        if hasattr(instance, "grade") and callable(instance.grade):
            return instance.grade, attr
        raise TypeError(f"Grader class {grader_ref} has no callable grade() method")

    if not callable(attr):
        raise TypeError(f"Resolved grader {grader_ref} is not callable")

    return attr, attr


def _is_exclusive_unit_interval(value: float) -> bool:
    return 0.0 < float(value) < 1.0


def verify_manifest() -> bool:
    manifest = _load_manifest()

    print("=" * 72)
    print("OPENENV TASK / GRADER VALIDATION")
    print("=" * 72)

    required_top_level = ("spec_version", "app", "tasks")
    missing = [key for key in required_top_level if key not in manifest]
    ok = not missing
    print(f"[{_status(ok)}] Manifest required keys present: {required_top_level}")
    if missing:
        print(f"      Missing keys: {missing}")
        return False

    tasks = manifest.get("tasks") or []
    enough_tasks = len(tasks) >= 3
    print(f"[{_status(enough_tasks)}] Manifest task count: {len(tasks)}")
    if not enough_tasks:
        return False

    all_graders_resolve = True
    for task in tasks:
        task_id = task.get("id", "<missing-id>")
        grader_ref = task.get("grader")
        try:
            grader_fn, grader_obj = _resolve_grader(grader_ref)
            smoke_score = grader_fn(state={"task": task_id, "total_score": 0.77})
            score_ok = _is_exclusive_unit_interval(smoke_score)
            print(
                f"[{_status(score_ok)}] Manifest grader {task_id}: {grader_ref} -> "
                f"{getattr(grader_obj, '__name__', type(grader_obj).__name__)} "
                f"(smoke score={smoke_score:.3f})"
            )
            all_graders_resolve &= score_ok
        except Exception as exc:
            all_graders_resolve = False
            print(f"[FAIL] Manifest grader {task_id}: {grader_ref} -> {exc}")

    return all_graders_resolve


def verify_public_registry() -> bool:
    tasks = get_public_tasks()
    enough_tasks = len(tasks) >= 3
    print(f"[{_status(enough_tasks)}] Root tasks.py exports {len(tasks)} task definitions")
    if not enough_tasks:
        return False

    task_ids = {task["id"] for task in tasks}
    graders_present = task_ids == set(GRADERS)
    print(f"[{_status(graders_present)}] Root tasks.py and graders.py agree on task ids")
    if not graders_present:
        print(f"      task ids={sorted(task_ids)} grader ids={sorted(GRADERS)}")
        return False

    for task_id, grader in GRADERS.items():
        sample_score = grader(state={"task": task_id, "total_score": 0.66})
        score_ok = _is_exclusive_unit_interval(sample_score)
        print(f"[{_status(score_ok)}] Root grader {task_id}: sample score={sample_score:.3f}")
        if not score_ok:
            return False

    return True


def verify_internal_graders() -> bool:
    internal_tasks = get_internal_tasks()
    enough_tasks = len(internal_tasks) >= 3
    print(f"[{_status(enough_tasks)}] env.tasks exports {len(internal_tasks)} internal tasks")
    if not enough_tasks:
        return False

    all_ok = True
    for task_id, payload in TEST_SAMPLES.items():
        try:
            reward, _ = get_internal_task(task_id).grade(
                answer=payload["answer"],
                question=payload["question"],
                rubric=payload["rubric"],
                expected_concepts=payload["expected_concepts"],
                category=payload["category"],
            )
            score_ok = _is_exclusive_unit_interval(reward)
            print(f"[{_status(score_ok)}] Internal grader {task_id}: reward={reward:.3f}")
            all_ok &= score_ok
        except Exception as exc:
            all_ok = False
            print(f"[FAIL] Internal grader {task_id}: {exc}")

    return all_ok


def main() -> int:
    checks = [
        verify_manifest(),
        verify_public_registry(),
        verify_internal_graders(),
    ]

    passed = all(checks)
    print("=" * 72)
    print("READY FOR SUBMISSION" if passed else "VALIDATION FAILED")
    print("=" * 72)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
