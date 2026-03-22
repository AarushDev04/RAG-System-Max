FROM python:3.11-slim
WORKDIR /app

# Install dependencies first (cached layer — only rebuilds if requirements change)
COPY compression/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared HF client from the orchestrator folder
COPY orchestrator/hf_client.py .

# Copy compression service file
COPY compression/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
