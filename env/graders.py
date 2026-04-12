"""
Hybrid Reward Grader for AI Interview Simulator.

Architecture:
  1. LOCAL heuristics run instantly (no API call):
     - keyword coverage, length, code presence, structure signals
  2. GEMINI semantic call (once per question, cached per session):
     - semantic correctness + relevance

5-Component Reward (sums to 1.0):
  - semantic_correctness  : 0.0–0.40  (Gemini: concept coverage)
  - depth_and_detail      : 0.0–0.20  (local: length + examples + code)
  - relevance             : 0.0–0.20  (Gemini: on-topic check)
  - clarity               : 0.0–0.10  (local: structure heuristics)
  - followup_readiness    : 0.0–0.10  (local: signals for hard mode)
"""

from __future__ import annotations
import os
import re
import json
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)

# ─── Gemini Setup  (uses REST API directly — works on Python 3.7+) ───────────
import requests as _requests

_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_GEMINI_AVAILABLE = bool(_OPENROUTER_API_KEY)

# ─── Component Weights ────────────────────────────────────────────────────────
W_SEMANTIC    = 0.40
W_DEPTH       = 0.20
W_RELEVANCE   = 0.20
W_CLARITY     = 0.10
W_FOLLOWUP    = 0.10


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value between lo and hi (inclusive)."""
    return max(lo, min(hi, val))


def _clamp_exclusive(val: float, lo: float = 0.0, hi: float = 1.0, epsilon: float = 0.001) -> float:
    """Clamp value to be STRICTLY between lo and hi (exclusive).
    
    Used for final reward scores which must be in (0, 1) not [0, 1].
    If result would be lo or hi, nudge it inward by epsilon.
    """
    clamped = max(lo, min(hi, val))
    # Nudge inward if at boundaries
    if clamped <= lo:
        return lo + epsilon
    if clamped >= hi:
        return hi - epsilon
    return clamped


# ─── LOCAL: Depth & Detail Score (0→1) ───────────────────────────────────────
def _score_depth(answer: str, rubric: Dict) -> float:
    """
    Scores depth based on:
    - Answer length relative to expected minimum
    - Presence of code blocks / function definitions
    - Presence of examples ("for example", "e.g.", "such as")
    - List usage (structured answer)
    """
    text = answer.strip()
    score = 0.0

    min_len = rubric.get("min_length", 40)
    length_score = _clamp(len(text) / (min_len * 3))  # full score at 3× min
    score += length_score * 0.5

    # Code presence
    has_code = (
        "```" in text
        or "def " in text
        or "class " in text
        or "import " in text
        or re.search(r"\b(for|while|if)\b.+:", text) is not None
    )
    if rubric.get("code_required", False):
        score += 0.3 if has_code else 0.0
    else:
        score += 0.15 if has_code else 0.0

    # Example presence
    example_markers = ["for example", "e.g.", "for instance", "such as", "consider", "suppose"]
    if any(m in text.lower() for m in example_markers):
        score += 0.15

    # List / structure presence
    if re.search(r"(\n[-*•]|\d+\.)", text):
        score += 0.1

    return _clamp(score)


# ─── LOCAL: Clarity Score (0→1) ──────────────────────────────────────────────
def _score_clarity(answer: str) -> float:
    """
    Heuristic clarity signals:
    - Not too short (penalize < 15 chars)
    - Sentences end with punctuation
    - No excessive repetition
    """
    text = answer.strip()
    if len(text) < 15:
        return 0.1

    score = 0.5  # baseline

    # Proper sentence endings
    sentences = re.split(r'[.!?]', text)
    if len(sentences) >= 2:
        score += 0.2

    # Penalize very long run-on sentences (no punctuation > 200 chars)
    if not re.search(r'[.!?,;]', text[:200]):
        score -= 0.2

    # Penalize obvious hallucination starters
    hallucination_signals = ["i don't know", "i cannot", "i'm not sure", "i'm unable"]
    if any(h in text.lower() for h in hallucination_signals):
        score -= 0.3

    return _clamp(score)


# ─── LOCAL: Follow-up Readiness Score (0→1) ──────────────────────────────────
def _score_followup_readiness(answer: str, category: str) -> float:
    """
    Rewards answers that open discussion:
    - Mentions trade-offs
    - Uses hedging language ("however", "on the other hand")
    - Asks implicit questions ("this raises the question...")
    - DSA: mentions complexity
    - System design: mentions scaling concerns
    """
    text = answer.lower()
    score = 0.0

    tradeoff_markers = ["however", "trade-off", "trade off", "on the other hand",
                        "alternatively", "one downside", "one advantage", "but"]
    if any(m in text for m in tradeoff_markers):
        score += 0.4

    if category == "dsa":
        if re.search(r"o\([^)]+\)", text):  # O(n), O(log n), etc.
            score += 0.3
        if "space complexity" in text or "time complexity" in text:
            score += 0.2

    if category == "system_design":
        scale_markers = ["scale", "bottleneck", "throughput", "latency", "availability"]
        if any(m in text for m in scale_markers):
            score += 0.3

    # Discussion-opening phrases
    if any(p in text for p in ["raises the question", "worth noting", "important to consider"]):
        score += 0.1

    return _clamp(score)


# ─── LOCAL: Keyword Coverage (fallback for semantic) ─────────────────────────
def _score_keywords(answer: str, rubric: Dict) -> float:
    """Backup: fraction of rubric keywords found in answer."""
    keywords = rubric.get("keywords", [])
    if not keywords:
        return 0.5
    text = answer.lower()
    matched = sum(1 for kw in keywords if kw.lower() in text)
    return matched / len(keywords)


# ─── GEMINI: Semantic Correctness + Relevance ────────────────────────────────
def _call_gemini_grader(question: str, answer: str, expected_concepts: list) -> Tuple[float, float]:
    """
    Calls Gemini once per question. Returns:
      (semantic_score 0–1, relevance_score 0–1)
    Falls back to (0.5, 0.5) on any error.
    """
    if not _GEMINI_AVAILABLE:
        return 0.5, 0.5

    concepts_str = ", ".join(expected_concepts)
    prompt = f"""You are a strict but fair technical interview grader.

Question: {question}

Candidate Answer: {answer}

Expected Concepts (for this question): {concepts_str}

Grade the answer on TWO dimensions. Return ONLY valid JSON, no markdown, no explanation:

{{
  "semantic_correctness": <float 0.0-1.0>,
  "relevance": <float 0.0-1.0>,
  "brief_reason": "<one sentence>"
}}

Grading rules:
- semantic_correctness: What fraction of expected concepts does the answer correctly address?
  1.0 = covers all concepts correctly, 0.5 = covers half, 0.0 = completely wrong or blank
- relevance: Does the answer stay on topic and actually address the question asked?
  1.0 = directly answers the question, 0.5 = partially relevant, 0.0 = off-topic
- Be strict. Short vague answers should score 0.3-0.5 max.
- Answers with correct code examples deserve semantic_correctness >= 0.7
"""
    try:
        payload = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        headers = {
            "Authorization": f"Bearer {_OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "AI Simulator"
        }
        resp = _requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        raw = raw.rstrip("```").strip()
        data = json.loads(raw)
        sem = _clamp(float(data.get("semantic_correctness", 0.5)))
        rel = _clamp(float(data.get("relevance", 0.5)))
        logger.info("OpenRouter grader: sem=%.2f rel=%.2f reason=%s", sem, rel, data.get("brief_reason", ""))
        return sem, rel
    except Exception as e:
        logger.warning("OpenRouter grader error: %s. Using keyword fallback.", e)
        return None, None  # signal fallback



# ─── MAIN: Hybrid Reward Computation ─────────────────────────────────────────
def compute_reward(
    answer: str,
    question: str,
    rubric: Dict,
    expected_concepts: list,
    category: str = "dsa",
    task: str = "easy",
    question_num: int = 1,
    total_q: int = 5,
) -> Tuple[float, "RewardBreakdown"]:
    """
    Main hybrid grader entry point.
    Returns (total_reward: float[0,1], breakdown: RewardBreakdown)
    """
    from .models import RewardBreakdown

    # ── Local components (always run, instant) ──
    depth_raw    = _score_depth(answer, rubric)
    clarity_raw  = _score_clarity(answer)
    followup_raw = _score_followup_readiness(answer, category)

    # Only score follow-up readiness in hard mode
    followup_score = followup_raw * W_FOLLOWUP if task == "hard" else 0.0

    # ── Gemini semantic call ──
    sem_raw, rel_raw = _call_gemini_grader(question, answer, expected_concepts)

    if sem_raw is None:  # Gemini failed → keyword fallback
        kw_score = _score_keywords(answer, rubric)
        sem_score = kw_score * W_SEMANTIC
        rel_score = kw_score * W_RELEVANCE
    else:
        sem_score = sem_raw * W_SEMANTIC
        rel_score = rel_raw * W_RELEVANCE

    depth_score   = depth_raw   * W_DEPTH
    clarity_score = clarity_raw * W_CLARITY

    # ── Combined total ──
    total = sem_score + depth_score + rel_score + clarity_score + followup_score
    # Ensure total is strictly in (0, 1), not [0, 1] — validator requirement
    total = _clamp_exclusive(total, lo=0.0, hi=1.0)

    breakdown = RewardBreakdown(
        semantic_correctness=round(_clamp(sem_score, 0.0, 0.4), 4),
        depth_and_detail    =round(_clamp(depth_score, 0.0, 0.2), 4),
        relevance           =round(_clamp(rel_score, 0.0, 0.2), 4),
        clarity             =round(_clamp(clarity_score, 0.0, 0.1), 4),
        followup_readiness  =round(_clamp(followup_score, 0.0, 0.1), 4),
    )

    return round(total, 4), breakdown


def generate_feedback(reward: float, breakdown: "RewardBreakdown", task: str) -> str:
    """Generate human-readable feedback string from reward breakdown."""
    parts = []
    if breakdown.semantic_correctness < 0.2:
        parts.append("Try covering more of the key concepts.")
    elif breakdown.semantic_correctness >= 0.32:
        parts.append("Excellent concept coverage!")

    if breakdown.depth_and_detail < 0.08:
        parts.append("Your answer is too brief — add examples or code.")
    elif breakdown.depth_and_detail >= 0.15:
        parts.append("Good depth and detail.")

    if breakdown.relevance < 0.1:
        parts.append("Your answer drifted off-topic.")

    if breakdown.clarity < 0.05:
        parts.append("Try to structure your answer more clearly.")

    if task == "hard" and breakdown.followup_readiness >= 0.08:
        parts.append("Great — your answer shows awareness of trade-offs!")

    overall = f"Score: {reward*100:.1f}%"
    feedback = " | ".join(parts) if parts else "Keep up the good work!"
    return f"{overall} — {feedback}"