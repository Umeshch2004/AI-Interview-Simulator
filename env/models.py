from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class Observation(BaseModel):
    question: str
    difficulty: str                          # "easy", "medium", "hard"
    category: str = "general"               # "dsa", "system_design", "behavioral", "debugging"
    question_number: int
    total_questions: int
    last_answer_feedback: Optional[str] = None
    last_reward: float = 0.0
    remaining_questions: int
    follow_up_context: Optional[str] = None  # hint if follow-up was injected
    performance_hint: Optional[str] = None   # e.g. "You're doing great!"


class Action(BaseModel):
    answer: str


class RewardBreakdown(BaseModel):
    semantic_correctness: float = Field(0.0, description="Concept coverage vs expected concepts (0–0.4)")
    depth_and_detail: float = Field(0.0, description="Length, examples, code presence (0–0.2)")
    relevance: float = Field(0.0, description="Answer stays on topic (0–0.2)")
    clarity: float = Field(0.0, description="Structure and coherence heuristics (0–0.1)")
    followup_readiness: float = Field(0.0, description="Bonus for trade-off awareness in hard mode (0–0.1)")


class Reward(BaseModel):
    value: float = Field(..., gt=0.0, lt=1.0, description="Score strictly between 0 and 1 (exclusive)")
    reason: str
    breakdown: Optional[RewardBreakdown] = None


class State(BaseModel):
    session_id: str
    task: str
    current_q_idx: int
    history: List[Dict[str, Any]]
    total_score: float
    done: bool
    performance_history: List[float] = Field(default_factory=list)
    current_difficulty: str = "easy"
    follow_ups_injected: int = 0