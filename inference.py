import asyncio
import os
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
# ⚠️ CRITICAL: MUST use hackathon-provided environment variables.
# NOTE: Do NOT cache these at module load time — read from os.environ at call
#       time so that env vars injected by the hackathon validator are always
#       picked up even if they arrive after the module is imported.

# Environment configuration (non-LLM vars are safe to read at load time)
SPACE_URL = os.getenv("SPACE_URL", "http://localhost:7860").rstrip("/")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
TEMPERATURE = 0.7
TASK_ORDER = ("easy", "medium", "hard")

def _safe_score(val: float) -> float:
    """Ensure score is strictly in (0.001, 0.999) per validator requirement."""
    return max(0.001, min(0.999, val))


def _format_score(val: float, decimals: int = 3, clamp: bool = False) -> str:
    num = _safe_score(val) if clamp else float(val)
    return f"{num:.{decimals}f}"


def _format_reward(val: float) -> str:
    return f"{_safe_score(val):.3f}".rstrip("0").rstrip(".")


def _step_action_name(category: str, follow_up_context: Optional[str]) -> str:
    if follow_up_context:
        return "answer_followup"
    if category == "system_design":
        return "answer_system_design"
    if category == "behavioral":
        return "answer_behavioral"
    if category == "debugging":
        return "answer_debugging"
    return "answer"

# Warn early if LLM proxy vars are absent, but do NOT crash —
# the hackathon validator may inject them before the first LLM call.
# Expected script constraints:
if not os.getenv("API_BASE_URL"):
    print("[WARN] API_BASE_URL not set — using default", flush=True)
if not os.getenv("MODEL_NAME"):
    print("[WARN] MODEL_NAME not set — using default", flush=True)
if not os.getenv("HF_TOKEN") and not os.getenv("API_KEY"):
    print("[WARN] HF_TOKEN or API_KEY not set — will be required before LLM calls", flush=True)

# ─── Candidate Persona ────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a thoughtful software engineering candidate in a technical interview.
Your goal is to give clear, accurate, and well-structured answers.

Guidelines:
- If asked about code, include working code examples using triple backticks
- Explain your reasoning step by step
- Mention time/space complexity for algorithmic questions
- For system design, discuss scalability, trade-offs, and bottlenecks
- For behavioral questions, use the STAR method (Situation, Task, Action, Result)
- Be concise but thorough — aim for 3–6 sentences minimum
- Do NOT start with "I don't know" — always attempt an answer
""").strip()

def log_start(task: str, model: str, env: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    done_val = str(done).lower()
    err_val = "null" if error in (None, "") else error
    print(
        f"[STEP] step={step} action={action} reward={_format_reward(reward)} done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(_format_reward(r) for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={_format_score(score, 3, clamp=True)} rewards={rewards_str}",
        flush=True,
    )


def get_tasks_to_run() -> List[str]:
    """
    Run all tasks by default so the validator can observe easy/medium/hard in one
    inference pass. A comma-separated TASK_NAME or TASK env var still allows
    targeted local testing.
    """
    raw_filter = (os.getenv("TASK_NAME") or os.getenv("TASK") or "").strip()
    if not raw_filter:
        return list(TASK_ORDER)

    requested = [task.strip() for task in raw_filter.split(",") if task.strip()]
    valid = [task for task in requested if task in TASK_ORDER]

    if not valid:
        print(
            f"[WARN] Invalid task filter '{raw_filter}'. Falling back to all tasks: {', '.join(TASK_ORDER)}",
            flush=True,
        )
        return list(TASK_ORDER)

    return valid


# ─── LiteLLM Proxy LLM Call (Hackathon Compliant) ────────────────────────────
def get_llm_client() -> OpenAI:
    """Initialize OpenAI client with hackathon's LiteLLM proxy.
    
    ⚠️ Reads API_BASE_URL and API_KEY from os.environ at call time so that
    env vars injected by the hackathon validator are always honoured.
    """
    api_base_url = os.environ.get("API_BASE_URL")
    api_key      = os.environ.get("API_KEY") # Prioritize API_KEY as requested
    
    # Validation requirement: if not provided, fallback to HF_TOKEN but log it
    if not api_key:
        api_key = os.environ.get("HF_TOKEN")
    
    if not api_base_url or not api_key:
        raise ValueError(
            f"Missing required environment variables: API_BASE_URL={bool(api_base_url)}, API_KEY={bool(api_key)}"
        )
        
    return OpenAI(
        base_url=api_base_url,
        api_key=api_key,
    )

async def get_answer(
    question: str,
    history: List[dict],
    category: str,
    follow_up_context: Optional[str],
) -> str:
    """Call hackathon's LiteLLM proxy to generate a candidate answer.
    
    ⚠️ ALL LLM calls go through API_BASE_URL — NO direct API calls allowed!
    Wraps risky operations in try/except to provide graceful error handling.
    """
    # Read model name at call time (same reason as above)
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    # Build history context (last 3 Q&A pairs)
    history_block = ""
    try:
        for h in history[-3:]:
            history_block += f"\nQ: {h['question']}\nA: {h['answer'][:200]}\n"
    except (KeyError, TypeError) as e:
        print(f"[ERROR] Failed to build history context: {e}", flush=True)
        history_block = "(Error processing history)"

    context_note = ""
    if follow_up_context:
        context_note = f"\nNote: This is a follow-up question. Context: {follow_up_context}\n"

    user_prompt = textwrap.dedent(f"""
    {context_note}
    Question type: {category}

    Previous exchange (for context):
    {history_block if history_block else "(This is the first question)"}

    Current question:
    {question}

    Your answer:
    """).strip()

    # Run in thread pool because OpenAI client is synchronous.
    # Wrap risky operations (client init and API call) in try/except
    loop = asyncio.get_event_loop()
    
    try:
        client = get_llm_client()
    except ValueError as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}", flush=True)
        return "Error generating response: Missing API credentials"
    except Exception as e:
        print(f"[ERROR] Unexpected error initializing LLM client: {e}", flush=True)
        return "Error generating response: Client initialization failed"

    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=400,
            )
        )
        answer = response.choices[0].message.content.strip()
        return answer if answer else "I need more time to think about this."
    except AttributeError as e:
        print(f"[ERROR] Failed to parse API response: {e}", flush=True)
        return "Error generating response: Invalid API response"
    except Exception as e:
        print(f"[ERROR] LLM API call failed: {type(e).__name__}: {e}", flush=True)
        return "Error generating response: API call failed"


# ─── Main Loop ────────────────────────────────────────────────────────────────
async def run_episode(task: str, http: httpx.AsyncClient) -> float:
    """Run one full episode. Returns final average score."""
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    log_start(task=task, env="interview-env", model=model_name)

    rewards = []
    history = []
    steps   = 0
    done    = False

    try:
        # Reset environment
        reset_resp = await http.post(f"{SPACE_URL}/reset", params={"task": task})
        if reset_resp.status_code != 200:
            print(f"[ERROR] Reset failed for task='{task}': {reset_resp.text}", flush=True)
            return 0.0

        data = reset_resp.json()
        obs = data["observation"]

        while not done and steps < MAX_STEPS:
            question         = obs["question"]
            category         = obs.get("category", "general")
            follow_up_ctx    = obs.get("follow_up_context")

            if question == "Interview complete. Thank you!":
                break

            # Get LLM answer through hackathon's proxy
            try:
                answer = await get_answer(question, history, category, follow_up_ctx)
            except Exception as e:
                print(f"[ERROR] get_answer raised unexpected exception: {type(e).__name__}: {e}", flush=True)
                answer = "Error generating response"
            
            err_msg = None

            # Submit to environment
            try:
                step_resp = await http.post(f"{SPACE_URL}/step", json={"answer": answer})
            except Exception as e:
                print(f"[ERROR] Failed to submit answer: {type(e).__name__}: {e}", flush=True)
                done = True
                break
            
            if step_resp.status_code != 200:
                print(f"[ERROR] Step failed: {step_resp.text}", flush=True)
                err_msg = err_msg or f"HTTP {step_resp.status_code}"
                done = True
                break

            try:
                step_data = step_resp.json()
            except Exception as e:
                print(f"[ERROR] Failed to parse step response: {type(e).__name__}: {e}", flush=True)
                done = True
                break
            
            obs       = step_data.get("observation", {})
            reward    = step_data.get("reward", {})
            done      = step_data.get("done", False)
            info      = step_data.get("info", {})

            reward_val = reward.get("value", 0.0) if isinstance(reward, dict) else 0.0
            rewards.append(reward_val)
            history.append({"question": question, "answer": answer})
            steps += 1

            log_step(
                step=steps,
                action=_step_action_name(category, follow_up_ctx),
                reward=reward_val,
                done=done,
                error=err_msg,
            )
            
            if err_msg and answer == "Error generating response":
                break

        avg_score = sum(rewards) / len(rewards) if rewards else 0.0
        return avg_score

    finally:
        final_score = _safe_score(sum(rewards) / max(len(rewards), 1))
        final_succ  = final_score >= 0.5
        log_end(success=final_succ, steps=steps, score=final_score, rewards=[_safe_score(r) for r in rewards])


async def main():
    async with httpx.AsyncClient(timeout=60.0) as http:
        # Health check
        try:
            await http.get(f"{SPACE_URL}/health")
        except Exception as e:
            print(f"[WARN] Health check failed: {e}", flush=True)

        tasks_to_run = get_tasks_to_run()
        results = []

        for task_name in tasks_to_run:
            try:
                score = await run_episode(task=task_name, http=http)
                score = _safe_score(score)
                results.append((task_name, score))
            except Exception as e:
                print(f"[ERROR] Episode failed for task='{task_name}': {type(e).__name__}: {e}", flush=True)
                results.append((task_name, 0.001))

            print(f"      -> Avg score '{task_name}': {_format_score(results[-1][1], 4, clamp=True)}", flush=True)
            print("", flush=True)

        if results:
            average = _safe_score(sum(score for _, score in results) / len(results))
            print("\n===== FINAL SCORES =====", flush=True)
            for task_name, score in results:
                print(f"  {task_name}: {_format_score(score, 4, clamp=True)}", flush=True)
            print(f"  Average: {_format_score(average, 4, clamp=True)}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[CRITICAL] Fatal error in main(): {type(e).__name__}: {e}", flush=True)
        exit(0)
