---
title: Openenv Interview Simulator
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
---

# 🎯 AI Interview Simulator — OpenEnv Environment

> A production-grade reinforcement learning environment for training LLMs to ace technical interviews.
> Built for the **Meta PyTorch OpenEnv Hackathon**.

---

## 🚀 What is this?

The **AI Interview Simulator** is an OpenEnv-compatible environment where an LLM agent plays the role of a **technical interview candidate**. The environment asks questions from three difficulty tiers, evaluates responses using a **hybrid AI + heuristic grader**, and **dynamically adapts** interview difficulty based on performance.

This environment is ideal for:
- **RL training**: Reward-shaping with 5 components to teach LLMs to answer well
- **Benchmarking**: Evaluate LLM knowledge across DSA, system design, and behavioral domains
- **Interview prep tools / EdTech**: Plug into real products

---

## 🧠 Architecture

```
Agent (LLM candidate)
       │  answer (Action)
       ▼
┌─────────────────────────────────────────┐
│          Interview Environment           │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Question Bank                  │   │
│  │   easy (5) / medium (5) / hard(5)│   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Dynamic Difficulty Adapter     │   │
│  │   Rolling 3-answer window        │   │
│  │   >0.75 → harder, <0.40 → easier │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Hybrid Grader                  │   │
│  │   Gemini 1.5 Flash (semantic)    │   │
│  │   + Local heuristics             │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
       │  observation + reward (0–1)
       ▼
  RL Training Loop (TRL / torchforge / etc.)
```

---

## 🏆 Reward Function (5 Components, sums to 1.0)

| Component | Weight | Method | Description |
|---|---|---|---|
| `semantic_correctness` | **0.40** | Gemini 1.5 Flash | Concept coverage vs expected concepts |
| `depth_and_detail` | **0.20** | Local | Length, code blocks, examples |
| `relevance` | **0.20** | Gemini 1.5 Flash | Answer stays on topic |
| `clarity` | **0.10** | Local | Sentence structure, coherence |
| `followup_readiness` | **0.10** | Local | Trade-offs, complexity (hard only) |

**Hybrid design**: Local heuristics run instantly (no latency), Gemini is called once per question for semantic quality. Falls back to keyword matching if Gemini is unavailable.

---

## 📋 Tasks

| Task | Questions | Categories | Follow-ups | Max Steps |
|---|---|---|---|---|
| `easy` | 3 | DSA basics | ❌ | 5 |
| `medium` | 4 | DSA + algorithms | ❌ | 8 |
| `hard` | 5 | DSA + system design + behavioral | ✅ (score 0.4–0.72) | 15 |

---

## 🔌 API Reference

### `POST /reset?task=easy`
Start a new episode.

```json
{
  "observation": {
    "question": "What is a variable in programming?",
    "difficulty": "easy",
    "category": "dsa",
    "question_number": 1,
    "total_questions": 3,
    "last_answer_feedback": null,
    "last_reward": 0.0,
    "remaining_questions": 3,
    "follow_up_context": null,
    "performance_hint": null
  },
  "state": { ... }
}
```

### `POST /step`
Submit a candidate answer.

```json
// Request
{ "answer": "A variable is a named container that stores a value in memory..." }

// Response
{
  "observation": { ... },
  "reward": {
    "value": 0.82,
    "reason": "Score: 82.0% — Excellent concept coverage! Good depth and detail.",
    "breakdown": {
      "semantic_correctness": 0.35,
      "depth_and_detail": 0.18,
      "relevance": 0.18,
      "clarity": 0.09,
      "followup_readiness": 0.02
    }
  },
  "done": false,
  "info": {
    "session_id": "uuid-...",
    "total_reward": 0.82,
    "average_score": 0.82,
    "current_difficulty": "easy",
    "follow_ups_injected": 0
  }
}
```

### `GET /state`
Full session state with history.

### `GET /health`
Health check (required by HF Spaces).

---

## ⚡ Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key (server-side grader)
export GEMINI_API_KEY=your_key_here

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — run inference (hackathon-compliant proxy vars)
export API_BASE_URL=https://api.openai.com/v1   # or hackathon's LiteLLM proxy URL
export API_KEY=your_openai_or_proxy_key
export MODEL_NAME=gpt-4o-mini
export TASK_NAME=hard
python inference.py
```

### Docker

```bash
docker build -t interview-env .
docker run -e GEMINI_API_KEY=your_key -p 7860:7860 interview-env
```

### HF Spaces

```bash
pip install openenv-core
openenv push --repo-id your-username/interview-env
```

Set `GEMINI_API_KEY` as a Space Secret in the HF Space settings.

---

## 🔑 Environment Variables

### Server
| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Gemini API key for semantic grading |
| `PORT` | No | `7860` | Server port |

### Inference (hackathon-injected)
| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | **Yes** | — | LiteLLM proxy URL (injected by hackathon validator) |
| `API_KEY` | **Yes** | — | Auth key for the proxy (injected by hackathon validator) |
| `MODEL_NAME` | Yes | `gpt-4o-mini` | Model to use via proxy |
| `SPACE_URL` | No | `http://localhost:7860` | URL of the running environment |
| `TASK_NAME` | No | `easy` | Task difficulty: `easy` \| `medium` \| `hard` |
| `MAX_STEPS` | No | `15` | Max steps per episode |

---

## 📊 Sample Output

```
[START] task=hard model=meta-llama/Llama-3.1-8B-Instruct env=http://localhost:7860
[STEP] step=1 difficulty=hard
  Q: Design a URL shortening service like TinyURL...
  A: A URL shortening service needs several key components: a hash function to...
  reward=0.784 done=false
  [REWARD_BREAKDOWN] sem=0.340 depth=0.180 rel=0.180 clarity=0.084 followup=0.000
[STEP] step=2 difficulty=hard [FOLLOW-UP]
  Q: How would you handle 10 billion shortened URLs?
  A: For 10 billion URLs, I would use a distributed database like Cassandra...
  reward=0.712 done=false
  [REWARD_BREAKDOWN] sem=0.300 depth=0.160 rel=0.180 clarity=0.072 followup=0.000
...
[END] success=true steps=7 score=0.741 rewards=[0.784,0.712,0.698,0.801,0.753,0.620,0.819]
```

---

## 🏗️ Project Structure

```
AI_Interview_Simulator/
├── server/
│   ├── app.py          # FastAPI server (OpenEnv API: /reset, /step, /state, /health)
│   ├── __init__.py     # Exposes 'app' and main entrypoints
│   └── __main__.py     # CLI execution entrypoint
├── env/
│   ├── interview_env.py # Core environment logic
│   ├── models.py        # Pydantic: Observation, Action, Reward, RewardBreakdown, State
│   ├── tasks.py         # Question bank (15 questions, 5 per level)
│   ├── graders.py       # Hybrid grader (Gemini + local heuristics)
│   └── __init__.py
├── inference.py        # Inference script (HF Router LLM candidate)
├── openenv.yaml        # OpenEnv manifest
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
└── README.md           # This file
```

---

## 📄 License

MIT License — feel free to fork and extend!