# ============================================================
# Root Dockerfile — orchestrator service
# This builds ONLY the orchestrator container.
# The full stack (all 9 containers) is started via:
#   docker compose up --build
# ============================================================

FROM python:3.11-slim
WORKDIR /app

# Install dependencies first — cached layer, only rebuilds
# when requirements.txt changes, not on every code edit
COPY services/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared HF inference client (used by all services)
COPY services/orchestrator/hf_client.py .

# Copy orchestrator source files
COPY services/orchestrator/cache.py .
COPY services/orchestrator/pipeline.py .
COPY services/orchestrator/main.py .

# Copy frontend — served at http://localhost:8000
COPY services/orchestrator/static/ ./static/

# Start with uvicorn — NOT python directly.
# uvicorn is the ASGI server that FastAPI requires.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]