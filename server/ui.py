import gradio as gr
import json
import logging
from typing import Optional
from env.interview_env import InterviewEnv
from env.models import Action

logger = logging.getLogger(__name__)

# Re-use the same environment logic as app.py
# In a real app, this should be session-based.
# For the hackathon demo, we use a single global instance for the UI.
_shared_env: Optional[InterviewEnv] = None

def get_env(task: str = "easy") -> InterviewEnv:
    global _shared_env
    if _shared_env is None:
        _shared_env = InterviewEnv(task=task)
    return _shared_env

def format_json(data):
    return json.dumps(data, indent=2)

def ui_reset(task: str):
    global _shared_env
    _shared_env = InterviewEnv(task=task)
    obs = _shared_env.reset()
    state = _shared_env.state().dict()
    
    reward_html = (
        "<div style='display: flex; gap: 20px; font-size: 1.2em; font-weight: bold;'>"
        "<div>Current Reward: <span style='color: #888;'>N/A</span></div>"
        "<div>Average: <span style='color: #4CAF50;'>0.0000</span></div>"
        "<div>Done: <span style='color: #f44336;'>False</span></div>"
        "</div>"
    )
    
    return (
        reward_html,
        "", # Clear answer
        "Environment reset. Initial question received.",
        format_json({"observation": obs.dict(), "state": state}),
        state.get("session_id", "N/A")
    )

def ui_step(answer: str):
    global _shared_env
    if _shared_env is None:
        return ui_reset("easy")
    
    if _shared_env.done:
        reward_html = (
            "<div style='display: flex; gap: 20px; font-size: 1.2em; font-weight: bold;'>"
            "<div>Current Reward: <span style='color: #888;'>N/A</span></div>"
            f"<div>Average: <span style='color: #4CAF50;'>{_shared_env.state().total_score:.4f}</span></div>"
            "<div>Done: <span style='color: #4CAF50;'>True</span></div>"
            "</div>"
        )
        return (
            reward_html,
            "",
            "Episode is already finished. Please Reset.",
            format_json(_shared_env.state().dict()),
            _shared_env.session_id
        )

    try:
        action = Action(answer=answer)
        obs, reward, done, info = _shared_env.step(action)
        state = _shared_env.state().dict()
        
        curr_reward = reward.value
        avg_score = info['average_score']
        
        # Color coding based on immediate performance
        curr_color = "#4CAF50" if curr_reward > 0.75 else "#FFC107" if curr_reward > 0.4 else "#f44336"
        avg_color = "#4CAF50" if avg_score > 0.6 else "#FFC107" if avg_score > 0.3 else "#f44336"
        done_color = "#f44336" if not done else "#4CAF50"
        
        reward_html = (
            "<div style='display: flex; gap: 20px; font-size: 1.2em; font-weight: bold;'>"
            f"<div>Current Reward: <span style='color: {curr_color};'>{curr_reward:.4f}</span></div>"
            f"<div>Average: <span style='color: {avg_color};'>{avg_score:.4f}</span></div>"
            f"<div>Done: <span style='color: {done_color};'>{done}</span></div>"
            "</div>"
        )
        
        status = f"Step complete. Reward: {curr_reward:.4f}. Feedback: {reward.reason[:100]}..."
        
        return (
            reward_html,
            "", # Clear answer box for next question
            status,
            format_json({"observation": obs.dict(), "reward": reward.dict(), "done": done, "info": info}),
            _shared_env.session_id
        )
    except Exception as e:
        logger.exception("UI Step Error")
        return (
            f"<div style='font-size: 1.2em; font-weight: bold;'>Error</div>",
            answer,
            f"Error: {str(e)}",
            format_json({"error": str(e)}),
            _shared_env.session_id if _shared_env else "N/A"
        )

def ui_get_state():
    global _shared_env
    if _shared_env is None:
        return "Not initialized", "{}"
    state = _shared_env.state().dict()
    return "State retrieved.", format_json(state)

# Custom CSS for the "OpenEnv" look
CSS = """
.container { max-width: 1200px; margin: auto; }
.sidebar { background-color: #0b0f19; padding: 20px; border-radius: 8px; }
.main-content { padding: 20px; }
.code-block { background-color: #161b22; color: #e6edf3; border-radius: 6px; padding: 10px; }
.metric-header { margin-bottom: 20px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=CSS) as demo:
    with gr.Row():
        # --- Sidebar ---
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### Quick Start")
            gr.Markdown("#### Connect to this environment")
            gr.Markdown("Connect from Python using `InterviewEnv`:")
            gr.Code(
                value="from interview_env import InterviewEnv\n\nwith InterviewEnv.from_env(\"Umeshch2004/AI-Interview-Simulator\") as env:\n    result = await env.step(\"Helpful answer\")",
                language="python"
            )
            
            gr.Markdown("Or connect directly to a running server:")
            gr.Code(
                value="env = InterviewEnv(base_url=\"http://localhost:8000\")",
                language="python"
            )
            
            gr.Markdown("#### Contribute to this environment")
            gr.Markdown("Submit improvements via pull request on the Hugging Face Hub.")
            gr.Code(
                value="openenv fork Umeshch2004/AI-Interview-Simulator --repo-type space",
                language="shell"
            )
            
            gr.Markdown("Then make your changes and submit a pull request:")
            gr.Code(
                value="cd <forked-repo>\nopenenv push Umeshch2004/AI-Interview-Simulator --create-pr",
                language="shell"
            )

        # --- Main Console ---
        with gr.Column(scale=2):
            reward_display = gr.HTML(
                value=(
                    "<div style='display: flex; gap: 20px; font-size: 1.2em; font-weight: bold;'>"
                    "<div>Current Reward: <span style='color: #888;'>0.0</span></div>"
                    "<div>Average: <span style='color: #888;'>0.0</span></div>"
                    "<div>Done: <span style='color: #888;'>False</span></div>"
                    "</div>"
                ),
                elem_classes=["metric-header"]
            )
            
            with gr.Group():
                answer_input = gr.Textbox(
                    label="Your Answer (Price Deltas)",
                    placeholder="Enter your response to the interview question...",
                    lines=3
                )
                
                with gr.Row():
                    difficulty_input = gr.Dropdown(
                        label="Task Difficulty (Restock Units)",
                        choices=["easy", "medium", "hard"],
                        value="easy"
                    )
                    session_id_display = gr.Textbox(
                        label="Session ID (Discount Pcts)",
                        value="Not started",
                        interactive=False
                    )
            
            with gr.Row():
                step_btn = gr.Button("Step", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
                state_btn = gr.Button("Get state")
            
            status_output = gr.Textbox(label="Status", interactive=False)
            
            json_output = gr.Code(
                label="Raw JSON response",
                language="json",
                lines=20
            )

    # Interactions
    reset_btn.click(
        fn=ui_reset,
        inputs=[difficulty_input],
        outputs=[reward_display, answer_input, status_output, json_output, session_id_display]
    )
    
    step_btn.click(
        fn=ui_step,
        inputs=[answer_input],
        outputs=[reward_display, answer_input, status_output, json_output, session_id_display]
    )
    
    state_btn.click(
        fn=ui_get_state,
        outputs=[status_output, json_output]
    )

if __name__ == "__main__":
    demo.launch()
