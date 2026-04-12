FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Gemini API key — set as a Hugging Face Space Secret, not hardcoded
ENV GEMINI_API_KEY=""
ENV PORT=7860

EXPOSE 7860

# Health check (HF Spaces requires the app to respond on /)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]