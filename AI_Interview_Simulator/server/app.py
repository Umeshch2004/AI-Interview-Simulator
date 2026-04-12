import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env.interview_env import InterviewEnv
from env.models import Action, State

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global env store (keyed by — single instance for simplicity, production would use sessions)
_env: InterviewEnv = None


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
async def root():
    """Redirect researchers rendering the iframe to the Swagger API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Health check — required by Hugging Face Spaces."""
    return {"status": "ok", "version": "2.0.0"}


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


@app.get("/state")
async def get_state():
    """Return current environment state including full history and performance analytics."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _env.state().dict()




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
