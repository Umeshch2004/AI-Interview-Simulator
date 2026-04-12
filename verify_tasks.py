"""
Verification script to validate that all 3 tasks have graders properly configured.
Run this to verify submission completeness.
"""

from env import TASKS, get_task, get_all_tasks

def verify_tasks_with_graders():
    """Verify that all 3 tasks are properly configured with graders."""
    
    print("=" * 70)
    print("TASK VERIFICATION REPORT")
    print("=" * 70)
    
    all_tasks = get_all_tasks()
    
    # Check 1: Ensure we have at least 3 tasks
    num_tasks = len(all_tasks)
    print(f"\n✓ Number of tasks: {num_tasks}")
    
    if num_tasks < 3:
        print(f"  ✗ ERROR: Need at least 3 tasks, found {num_tasks}")
        return False
    
    # Check 2: Verify each task has a grader
    print(f"\n✓ Tasks with graders:")
    all_have_graders = True
    
    for difficulty, task in all_tasks.items():
        has_grader = callable(task.grader)
        status = "✓" if has_grader else "✗"
        print(f"  {status} {difficulty:10} : {task.name}")
        if not has_grader:
            all_have_graders = False
    
    if not all_have_graders:
        print(f"\n  ✗ ERROR: Not all tasks have graders!")
        return False
    
    # Check 3: Test grading with sample data
    print(f"\n✓ Grader functionality check:")
    
    test_samples = {
        "easy": {
            "question": "What is a dictionary in Python?",
            "answer": "A dictionary is a key-value data structure that allows you to store and retrieve data efficiently. Keys must be unique and hashable. For example, {'name': 'John', 'age': 30} is a dictionary with string keys.",
            "rubric": {"keywords": ["key", "value", "hash", "dict", "lookup"], "min_length": 30},
            "expected_concepts": ["key-value storage", "hash map", "O(1) lookup", "unique keys", "fast retrieval"],
            "category": "dsa",
        },
        "medium": {
            "question": "How do you detect a cycle in a linked list?",
            "answer": "You can use Floyd's cycle-finding algorithm with two pointers. Move one pointer (slow) one step at a time, and another (fast) two steps. If they meet at the same node, there's a cycle. Time complexity is O(n) and space is O(1).",
            "rubric": {"keywords": ["slow", "fast", "pointer", "floyd", "tortoise", "hare"], "min_length": 50},
            "expected_concepts": ["Floyd's cycle-finding algorithm", "Tortoise and Hare", "two pointers", "fast pointer", "slow pointer"],
            "category": "dsa",
        },
        "hard": {
            "question": "Design a distributed rate limiter for a public API like Twitter.",
            "answer": "Use a token bucket algorithm stored in Redis for distributed coordination. Each user/IP gets a bucket. Tokens are added at a fixed rate. For each request, consume one token. If no tokens available, reject the request. Use sliding window log as backup. Maintain separate buckets per region for scalability.",
            "rubric": {"keywords": ["token", "bucket", "redis", "sliding", "window", "distributed"], "min_length": 150},
            "expected_concepts": ["token bucket", "redis", "distributed cache", "sliding window", "load balancer", "high availability"],
            "category": "system_design",
        },
    }
    
    all_graders_work = True
    for difficulty, test_data in test_samples.items():
        try:
            task = get_task(difficulty)
            reward, breakdown = task.grade(
                answer=test_data["answer"],
                question=test_data["question"],
                rubric=test_data["rubric"],
                expected_concepts=test_data["expected_concepts"],
                category=test_data.get("category", "dsa"),
            )
            print(f"  ✓ {difficulty:10} grader works (reward: {reward:.3f})")
        except Exception as e:
            print(f"  ✗ {difficulty:10} grader failed: {str(e)}")
            all_graders_work = False
    
    if not all_graders_work:
        print(f"\n  ✗ ERROR: Some graders failed!")
        return False
    
    print("\n" + "=" * 70)
    print("✓ ALL CHECKS PASSED - Submission is ready!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_tasks_with_graders()
    exit(0 if success else 1)
