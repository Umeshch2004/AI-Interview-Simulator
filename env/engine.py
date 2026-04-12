import os
import json
import random
import requests
from typing import Dict, Any, List

# OpenRouter uses the standard OpenAI chat completions endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Robust static fallback to ensure the environment completely survives API rate-limits or network failures.
FALLBACK_QUESTIONS = {
    "easy": [
        {
            "question": "What is a dictionary in Python, and why is it useful?",
            "category": "dsa",
            "expected_concepts": ["key-value storage", "hash map", "O(1) lookup", "unique keys", "fast retrieval"],
            "rubric": {"keywords": ["key", "value", "hash", "dict", "lookup"], "min_length": 30},
            "follow_up_questions": []
        },
        {
            "question": "Explain the difference between a list and a tuple in Python.",
            "category": "dsa",
            "expected_concepts": ["mutability", "immutability", "performance", "hashable", "use cases"],
            "rubric": {"keywords": ["mutable", "immutable", "list", "tuple", "performance"], "min_length": 30},
            "follow_up_questions": []
        },
        {
            "question": "What is the purpose of a set in Python and when would you use it?",
            "category": "dsa",
            "expected_concepts": ["unique elements", "unordered", "membership testing", "mathematical operations", "performance"],
            "rubric": {"keywords": ["unique", "set", "unordered", "membership", "O(1)"], "min_length": 30},
            "follow_up_questions": []
        }
    ],
    "medium": [
        {
            "question": "How do you detect a cycle in a linked list?",
            "category": "dsa",
            "expected_concepts": ["Floyd's cycle-finding algorithm", "Tortoise and Hare", "two pointers", "fast pointer", "slow pointer"],
            "rubric": {"keywords": ["slow", "fast", "pointer", "floyd", "tortoise", "hare"], "min_length": 50},
            "follow_up_questions": []
        },
        {
            "question": "Implement a function to reverse a linked list.",
            "category": "dsa",
            "expected_concepts": ["pointer manipulation", "three pointers", "iteration", "edge cases", "time complexity"],
            "rubric": {"keywords": ["reverse", "pointer", "prev", "next", "current"], "min_length": 50},
            "follow_up_questions": []
        },
        {
            "question": "What is the difference between breadth-first search and depth-first search?",
            "category": "dsa",
            "expected_concepts": ["queue vs stack", "traversal order", "time complexity", "space complexity", "use cases"],
            "rubric": {"keywords": ["BFS", "DFS", "queue", "stack", "traversal"], "min_length": 50},
            "follow_up_questions": []
        }
    ],
    "hard": [
        {
            "question": "Design a distributed rate limiter for a public API like Twitter.",
            "category": "system_design",
            "expected_concepts": ["token bucket", "redis", "distributed cache", "sliding window", "load balancer", "high availability"],
            "rubric": {"keywords": ["token", "bucket", "redis", "sliding", "window", "distributed"], "min_length": 150},
            "follow_up_questions": []
        },
        {
            "question": "How would you design a URL shortening service like bit.ly?",
            "category": "system_design",
            "expected_concepts": ["hashing", "database design", "scalability", "collision handling", "caching"],
            "rubric": {"keywords": ["hash", "mapping", "database", "scale", "collision"], "min_length": 150},
            "follow_up_questions": []
        },
        {
            "question": "Design a real-time notification system for a chat application.",
            "category": "system_design",
            "expected_concepts": ["websockets", "message queues", "database", "scalability", "consistency"],
            "rubric": {"keywords": ["websocket", "queue", "notification", "real-time", "scale"], "min_length": 150},
            "follow_up_questions": []
        }
    ]
}

def generate_interview_question(difficulty: str) -> Dict[str, Any]:
    """
    Generates an interview question using the varied fallback question bank.
    Randomly selects from multiple questions per difficulty level.
    """
    # Use varied fallback questions to ensure question diversity
    questions = FALLBACK_QUESTIONS.get(difficulty, FALLBACK_QUESTIONS["easy"])
    selected_q = random.choice(questions).copy()
    selected_q["follow_up_questions"] = []  # Initialize empty
    return selected_q

def generate_followup(candidate_answer: str, original_question: str) -> str:
    """
    Synthesize exactly one follow-up question attacking a weakness or exploring depth in their answer.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Could you explain the memory constraints or trade-offs of your approach?"

    prompt = f"""
    You asked the candidate: {original_question}
    They answered: {candidate_answer}
    
    Write EXACTLY ONE short, deep follow-up question asking them to explain a trade-off, edge case, or alternative implementation to what they just said.
    Do not provide praise, context, or greetings. Output only the follow-up question.
    """

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 60
    }

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "AI Simulator"
        }
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text
    except Exception:
        return "Could you explain the memory constraints or trade-offs of your approach?"

def generate_candidate_answer(question: str, history: List[Dict[str, str]], category: str = "general") -> str:
    """
    Generate an AI candidate response for the UI's 'AI Mode'.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "I'm sorry, I cannot generate an answer right now. Please check my API configuration."

    history_str = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history[-3:]])
    
    prompt = f"""
    Role: Software Engineering Candidate
    Context: Technical Interview
    
    Interview History:
    {history_str}
    
    Current Question:
    {question}
    
    Instructions:
    Provide a clear, detailed, and technically accurate answer. Include code if appropriate.
    Answer directly and professionally. No greetings or meta-talk.
    """

    payload = {
        "model": "google/gemini-2.0-flash-001:free", # Updated stable free model ID
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "AI Simulator Chat"
        }
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Candidate AI Error: {str(e)}"
