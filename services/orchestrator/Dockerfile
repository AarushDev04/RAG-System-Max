FROM python:3.11-slim
WORKDIR /app

# Install dependencies first (cached layer)
COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy orchestrator source files
COPY orchestrator/hf_client.py .
COPY orchestrator/cache.py .
COPY orchestrator/pipeline.py .
COPY orchestrator/main.py .

# Copy frontend — served at http://localhost:8000
COPY orchestrator/static/ ./static/

# 2 workers: one handles requests while the other awaits HF responses
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
