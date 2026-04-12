import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graders import GRADERS
from env.interview_env import InterviewEnv
from env.models import Action, State
from tasks import get_task as get_public_task
from tasks import get_tasks as get_public_tasks
import gradio as gr
from .ui import demo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global env store (keyed by — single instance for simplicity, production would use sessions)
_env: InterviewEnv = None


class GraderRequest(BaseModel):
    task: str
    state: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    answer: Optional[str] = None
    question: Optional[str] = None
    rubric: Optional[Dict[str, Any]] = None
    expected_concepts: List[str] = Field(default_factory=list)
    category: str = "general"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    logger.info("AI Interview Simulator starting up...")
    _env = InterviewEnv(task="easy")
    yield
    logger.info("AI Interview Simulator shutting down.")


app = FastAPI(
    title="AI Interview Simulator",
    description=(
        "An OpenEnv-compatible reinforcement learning environment for technical interviews. "
        "The agent plays the role of a candidate and answers questions. "
        "Rewards are computed via a hybrid Gemini + heuristic grader (5-component, 0–1)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Mount Gradio UI
app = gr.mount_gradio_app(app, demo, path="/ui")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the Gradio UI by default."""
    return RedirectResponse(url="/ui")


@app.get("/health")
async def health():
    """Health check - required by Hugging Face Spaces."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/tasks")
async def list_tasks():
    """Return the statically declared task registry and grader bindings."""
    tasks = get_public_tasks()
    return {
        "count": len(tasks),
        "tasks": tasks,
    }


@app.post("/reset")
async def reset(task: str = "easy"):
    """
    Start a new interview episode.

    Parameters:
      - task: "easy" | "medium" | "hard"

    Returns initial observation and state.
    """
    global _env
    valid_tasks = ["easy", "medium", "hard"]
    if task not in valid_tasks:
        raise HTTPException(status_code=422, detail=f"Invalid task '{task}'. Choose from {valid_tasks}")
    _env = InterviewEnv(task=task)
    obs = _env.reset()
    return {
        "observation": obs.dict(),
        "state": _env.state().dict(),
    }


@app.post("/step")
async def step(action: Action):
    """
    Submit an answer to the current question.

    Body: { "answer": "<your answer>" }

    Returns:
      - observation: next question + metadata
      - reward: { value, reason, breakdown }
      - done: bool
      - info: { session_id, total_reward, average_score, current_difficulty, ... }
    """
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    if _env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new episode.")
    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        logger.exception("Error during step")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grader")
async def grade_task(request: GraderRequest):
    """
    Grade a task using the explicit top-level grader registry.

    The validator can call this endpoint with either:
    - `state` and optional `reward`
    - question-answer payload fields to run deterministic grading
    """
    if request.task not in GRADERS:
        raise HTTPException(status_code=404, detail=f"Unknown task '{request.task}'")

    if request.state is not None:
        state = request.state
    elif _env is not None:
        state = _env.state().dict()
    else:
        state = None

    grader = GRADERS[request.task]
    score = grader(
        state=state,
        reward=request.reward,
        answer=request.answer,
        question=request.question,
        rubric=request.rubric,
        expected_concepts=request.expected_concepts,
        category=request.category,
    )

    return {
        "task": get_public_task(request.task),
        "score": score,
    }


@app.get("/state")
async def get_state():
    """Return current environment state including full history and performance analytics."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _env.state().dict()


@app.get("/validate")
async def validate_environment():
    """Lightweight validator-friendly environment summary."""
    tasks = get_public_tasks()
    tasks_with_graders = [task for task in tasks if task.get("grader")]
    return {
        "valid": len(tasks_with_graders) >= 3,
        "task_count": len(tasks),
        "tasks_with_graders": len(tasks_with_graders),
        "app": "server.app:app",
        "tasks": tasks,
    }




def main():
    """Entry point for project scripts / console_scripts: run the ASGI server."""
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

def start():
    """Alias entrypoint used by `[project.scripts] start`.`"""
    main()


if __name__ == "__main__":
    main()
