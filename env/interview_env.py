from __future__ import annotations
"""
AI Interview Simulator — Core Environment Logic.

Features:
  - Dynamic difficulty adaptation (rolling 3-answer window)
  - Follow-up question injection (hard mode, score 0.40–0.72)
  - Hybrid Gemini + heuristic reward (5 components)
  - Session UUID for reproducibility
  - Clean OpenEnv-compatible step/reset/state API
"""

import uuid
import random
import logging
from typing import Optional, Dict, Any, List, Tuple

from .models import Observation, Action, Reward, State, RewardBreakdown
from .engine import generate_interview_question, generate_followup
from .graders import compute_reward, generate_feedback
from .tasks import get_task

logger = logging.getLogger(__name__)

# Dynamic difficulty thresholds
STEP_UP_THRESHOLD   = 0.75   # sustained score → harder next question
STEP_DOWN_THRESHOLD = 0.40   # struggling      → easier next question
ROLLING_WINDOW      = 3      # questions to average

# Follow-up injection (hard mode only)
FOLLOWUP_LOW  = 0.40         # below this: skip to next (too hard)
FOLLOWUP_HIGH = 0.72         # above this: skip follow-up (nailed it)

# Difficulty ordering for adaptive adjustment
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


class InterviewEnv:
    def __init__(self, task: str = "easy"):
        if task not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid task: {task}. Choose from ['easy', 'medium', 'hard']")
        self.base_task = task
        self.session_id = str(uuid.uuid4())
        self._init_state()

    def _init_state(self):
        self.task = self.base_task
        self.total_q = {"easy": 3, "medium": 4, "hard": 5}.get(self.task, 3)
        self.current_idx = 0
        self.questions: List[Dict] = [generate_interview_question(self.task)]
        self.history: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self.done = False
        self.last_feedback = None
        self.last_reward = 0.0
        self.performance_history: List[float] = []
        self.current_difficulty = self.task
        self.follow_ups_injected = 0
        self.used_questions = {self.questions[0]["question"]}
        # Follow-up state: if not None, we're mid follow-up
        self._pending_followup: Optional[str] = None
        self._followup_parent_idx: Optional[int] = None

    # ─── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._init_state()
        logger.info(f"[{self.session_id}] Episode reset. task={self.task}")
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        answer = action.answer.strip()

        # ── Are we answering a follow-up? ──────────────────────────────
        if self._pending_followup is not None:
            return self._step_followup(answer)

        # ── Regular question ───────────────────────────────────────────
        current_q = self.questions[self.current_idx]
        reward_val, breakdown = compute_reward(
            answer=answer,
            question=current_q["question"],
            rubric=current_q["rubric"],
            expected_concepts=current_q["expected_concepts"],
            category=current_q.get("category", "dsa"),
            task=self.task,
            question_num=self.current_idx + 1,
            total_q=self.total_q,
        )

        self.total_reward += reward_val
        self.last_reward = reward_val
        self.performance_history.append(reward_val)
        feedback = generate_feedback(reward_val, breakdown, self.task)
        self.last_feedback = feedback

        entry = {
            "question": current_q["question"],
            "category": current_q.get("category", "general"),
            "answer": answer,
            "reward": reward_val,
            "breakdown": breakdown.dict(),
            "feedback": feedback,
            "is_followup": False,
        }
        self.history.append(entry)

        # ── Follow-up injection (hard mode) ────────────────────────────
        inject_followup = (
            self.task == "hard"
            and FOLLOWUP_LOW <= reward_val < FOLLOWUP_HIGH
        )
        if inject_followup:
            fu_q = generate_followup(answer, current_q["question"])
            self._pending_followup = fu_q
            self._followup_parent_idx = self.current_idx
            self.follow_ups_injected += 1
            obs = Observation(
                question=fu_q,
                difficulty=self.current_difficulty,
                category=current_q.get("category", "general"),
                question_number=self.current_idx + 1,
                total_questions=self.total_q,
                last_answer_feedback=feedback,
                last_reward=reward_val,
                remaining_questions=self.total_q - self.current_idx,
                follow_up_context=f"Follow-up on: {current_q['question'][:60]}...",
                performance_hint=self._performance_hint(),
            )
            reward_obj = Reward(value=reward_val, reason=feedback, breakdown=breakdown)
            info = self._build_info()
            return obs, reward_obj, False, info

        # ── Advance to next question ───────────────────────────────────
        self._adapt_difficulty()
        self.current_idx += 1
        if self.current_idx >= self.total_q:
            self.done = True
        else:
            # Dynamically pick a unique next question matching our adapted difficulty
            attempts = 0
            while attempts < 10:
                next_q = generate_interview_question(self.current_difficulty)
                if next_q["question"] not in self.used_questions:
                    break
                attempts += 1
            
            self.questions.append(next_q)
            self.used_questions.add(next_q["question"])

        obs = self._get_observation()
        reward_obj = Reward(value=reward_val, reason=feedback, breakdown=breakdown)
        return obs, reward_obj, self.done, self._build_info()

    def _step_followup(self, answer: str) -> Tuple[Observation, Reward, bool, Dict]:
        """Handle a follow-up question answer."""
        parent_q = self.questions[self._followup_parent_idx]
        reward_val, breakdown = compute_reward(
            answer=answer,
            question=self._pending_followup,
            rubric=parent_q["rubric"],
            expected_concepts=parent_q["expected_concepts"],
            category=parent_q.get("category", "dsa"),
            task=self.task,
            question_num=self.current_idx + 1,
            total_q=self.total_q,
        )
        # Follow-up contributes half weight to total
        followup_reward = reward_val * 0.5
        self.total_reward += followup_reward
        self.last_reward = reward_val
        self.performance_history.append(reward_val)
        feedback = generate_feedback(reward_val, breakdown, self.task)
        self.last_feedback = feedback

        self.history.append({
            "question": self._pending_followup,
            "category": parent_q.get("category", "general"),
            "answer": answer,
            "reward": reward_val,
            "breakdown": breakdown.dict(),
            "feedback": feedback,
            "is_followup": True,
        })

        # Clear follow-up state and advance
        self._pending_followup = None
        self._followup_parent_idx = None
        self._adapt_difficulty()
        self.current_idx += 1
        if self.current_idx >= self.total_q:
            self.done = True
        else:
            # Dynamically pick a unique next question matching our adapted difficulty
            attempts = 0
            while attempts < 10:
                next_q = generate_interview_question(self.current_difficulty)
                if next_q["question"] not in self.used_questions:
                    break
                attempts += 1
                
            self.questions.append(next_q)
            self.used_questions.add(next_q["question"])

        obs = self._get_observation()
        reward_obj = Reward(value=reward_val, reason=feedback, breakdown=breakdown)
        return obs, reward_obj, self.done, self._build_info()

    def state(self) -> State:
        return State(
            session_id=self.session_id,
            task=self.base_task,
            current_q_idx=self.current_idx,
            history=self.history,
            total_score=self.total_reward / max(len(self.history), 1),
            done=self.done,
            performance_history=self.performance_history,
            current_difficulty=self.current_difficulty,
            follow_ups_injected=self.follow_ups_injected,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _adapt_difficulty(self):
        """
        Adjust current_difficulty based on rolling 3-question average.
        Only changes the label logged in observations; base question pool is fixed per episode.
        """
        if len(self.performance_history) < ROLLING_WINDOW:
            return
        rolling_avg = sum(self.performance_history[-ROLLING_WINDOW:]) / ROLLING_WINDOW
        cur_idx = DIFFICULTY_ORDER.index(self.current_difficulty)

        if rolling_avg > STEP_UP_THRESHOLD and cur_idx < len(DIFFICULTY_ORDER) - 1:
            self.current_difficulty = DIFFICULTY_ORDER[cur_idx + 1]
            logger.info(f"[{self.session_id}] Difficulty ↑ → {self.current_difficulty} (avg={rolling_avg:.2f})")
        elif rolling_avg < STEP_DOWN_THRESHOLD and cur_idx > 0:
            self.current_difficulty = DIFFICULTY_ORDER[cur_idx - 1]
            logger.info(f"[{self.session_id}] Difficulty ↓ → {self.current_difficulty} (avg={rolling_avg:.2f})")

    def _performance_hint(self) -> str:
        if not self.performance_history:
            return None
        avg = sum(self.performance_history) / len(self.performance_history)
        if avg >= 0.75:
            return "You're performing excellently! Keep it up."
        elif avg >= 0.50:
            return "Good progress. Try to add more detail and examples."
        else:
            return "Keep going — try to cover the key concepts more fully."

    def _get_observation(self) -> Observation:
        if self.done or self.current_idx >= self.total_q:
            avg_score = self.total_reward / max(len(self.history), 1)
            return Observation(
                question="Interview complete. Thank you!",
                difficulty=self.current_difficulty,
                category="general",
                question_number=self.total_q,
                total_questions=self.total_q,
                last_answer_feedback=self.last_feedback,
                last_reward=self.last_reward,
                remaining_questions=0,
                performance_hint=f"Final score: {avg_score*100:.1f}%",
            )
        
        # If we have a pending follow-up, it takes precedence as the "current" question
        if self._pending_followup:
            parent_q = self.questions[self._followup_parent_idx]
            return Observation(
                question=self._pending_followup,
                difficulty=self.current_difficulty,
                category=parent_q.get("category", "general"),
                question_number=self.current_idx + 1,
                total_questions=self.total_q,
                last_answer_feedback=self.last_feedback,
                last_reward=self.last_reward,
                remaining_questions=self.total_q - self.current_idx,
                follow_up_context=f"Follow-up on: {parent_q['question'][:60]}...",
                performance_hint=self._performance_hint(),
            )

        q = self.questions[self.current_idx]
        remaining = self.total_q - self.current_idx
        return Observation(
            question=q["question"],
            difficulty=self.current_difficulty,
            category=q.get("category", "general"),
            question_number=self.current_idx + 1,
            total_questions=self.total_q,
            last_answer_feedback=self.last_feedback,
            last_reward=self.last_reward,
            remaining_questions=remaining,
            performance_hint=self._performance_hint(),
        )

    def _build_info(self) -> Dict:
        avg = self.total_reward / max(len(self.history), 1)
        # Final safety clamp for validator compliance
        avg = max(0.001, min(0.999, avg))
        return {
            "session_id": self.session_id,
            "total_reward": round(self.total_reward, 4),
            "average_score": round(avg, 4),
            "current_difficulty": self.current_difficulty,
            "history_length": len(self.history),
            "follow_ups_injected": self.follow_ups_injected,
        }