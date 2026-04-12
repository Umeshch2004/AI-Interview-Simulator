# 🚀 Setup Guide — AI Interview Simulator

Quick setup instructions for local development, HF Space deployment, and Docker.

---

## 📦 Local Development (Python 3.7+)

### 1. Clone & Install

```bash
git clone https://github.com/Umeshch2004/AI-Interview-Simulator.git
cd AI-Interview-Simulator
```

### 2. Create Virtual Environment

**With Python 3.7:**
```bash
python -m venv .venv37
.venv37\Scripts\activate  # Windows
source .venv37/bin/activate  # macOS/Linux
```

**With Python 3.10+:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.lock
```

### 4. Start the Server

```bash
python -m server
```

Server runs at: **http://localhost:7860**

---

## 🌐 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI (interactive API docs) |
| `/reset` | POST | Start new interview: `?task=easy\|medium\|hard` |
| `/step` | POST | Submit answer: `{"answer": "..."}` |
| `/state` | GET | Get full interview state & history |

### Example: Quick Test

```bash
# Health check
curl http://localhost:7860/health

# Start interview
curl -X POST http://localhost:7860/reset?task=easy

# Submit answer
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"answer": "A dictionary is a data structure..."}'
```

---

## 🤖 Run Inference with LLM

### 1. Get HF Token

1. Visit https://huggingface.co/settings/tokens
2. Create a **read** token
3. Copy the token

### 2. Run Inference

```bash
export HF_TOKEN="hf_your_token_here"
export SPACE_URL="http://localhost:7860"
export TASK_NAME="easy"  # or medium, hard
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

python inference.py
```

### Output Example:

```
[INFO] Server health: {'status': 'ok', 'version': '2.0.0'}
[START] task=easy model=meta-llama/Llama-3.1-8B-Instruct env=http://localhost:7860
[STEP] step=1 difficulty=easy
  Q: What is a dictionary in Python?
  A: A dictionary is a key-value data structure that...
  reward=0.830 done=false
  [REWARD_BREAKDOWN] sem=0.400 depth=0.160 rel=0.200 clarity=0.070 followup=0.000
[END] success=true steps=3 score=0.777 rewards=[0.830,0.820,0.680]
[SUMMARY] Final score for task='easy': 0.777
```

---

## 🐳 Docker Deployment

### Build

```bash
docker build -t interview-simulator .
```

### Run Locally

```bash
docker run -p 7860:7860 interview-simulator
```

### Push to Hugging Face

```bash
docker tag interview-simulator Umeshch2004/interview-simulator:latest
docker push Umeshch2004/interview-simulator:latest
```

---

## ☁️ Hugging Face Space Deployment

### Automatic (GitHub Sync)

1. Create Space at https://huggingface.co/new-space
2. Link your GitHub repo: `https://github.com/Umeshch2004/AI-Interview-Simulator`
3. Set SDK to **Docker**
4. HF auto-deploys on every push

### Manual

1. Create Space
2. In Space terminal:
   ```bash
   git clone <your-repo>
   cd AI-Interview-Simulator
   pip install -r requirements.lock
   ```

3. Create `app.py` at root:
   ```python
   from server import app
   
   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=7860)
   ```

4. Set **Python** SDK in Space config (not Docker)

---

## 🔧 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | (none) | Google Gemini API key for question generation |
| `PORT` | 7860 | Server port |
| `HF_TOKEN` | (none) | Hugging Face token for inference |
| `SPACE_URL` | http://localhost:7860 | Interview environment URL |
| `MODEL_NAME` | meta-llama/Llama-3.1-8B-Instruct | LLM model for inference |
| `TASK_NAME` | easy | Interview difficulty |

### Enable Question Generation

```bash
export GEMINI_API_KEY="your_gemini_key"
python -m server
```

If API key is missing, server uses **fallback questions** (no interruption).

---

## 📊 Project Structure

```
AI-Interview-Simulator/
├── server/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # CLI entry point
│   └── app.py               # FastAPI application
├── env/
│   ├── interview_env.py      # Core environment logic
│   ├── engine.py            # Question generation
│   ├── graders.py           # Reward computation
│   ├── models.py            # Pydantic models
│   └── client.py            # HTTP client
├── inference.py             # LLM agent script
├── pyproject.toml           # Project config & scripts
├── requirements.lock        # Pinned dependencies
├── uv.lock                  # UV lockfile
├── Dockerfile               # Container
├── openenv.yaml             # OpenEnv metadata
└── README.md                # Project overview
```

---

## 🐛 Troubleshooting

**Error: `ModuleNotFoundError: No module named 'server'`**
```bash
pip install -e .
```

**Error: `Attribute "app" not found in module "server"`**
- Ensure `server/__init__.py` exports `app`
- For HF Space, set build cmd to: `pip install -r requirements.lock`

**Port already in use:**
```bash
export PORT=8000
python -m server
```

**LLM inference fails (401):**
- Token is invalid or expired
- Check HF token at https://huggingface.co/settings/tokens

---

## 📚 Resources

- **OpenEnv Docs**: https://github.com/openenv-ai/openenv
- **FastAPI**: https://fastapi.tiangolo.com/
- **Hugging Face**: https://huggingface.co
- **PyTorch OpenEnv Hackathon**: (official hackathon repo)

---

## ✅ Verify Installation

```bash
# Test 1: Import check
python -c "from server import app; print('✓ Server imports OK')"

# Test 2: Health check (server running)
curl http://localhost:7860/health

# Test 3: Inference (with HF token)
export HF_TOKEN="your_token"
python inference.py
```

---

## 🚀 Next Steps

1. **Run locally** to verify setup
2. **Review `inference.py`** for LLM integration
3. **Deploy to HF Space** for public access
4. **Fine-tune** prompt engineering in `inference.py`

Happy interviewing! 🎯
