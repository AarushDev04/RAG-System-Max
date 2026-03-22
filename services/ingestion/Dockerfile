FROM python:3.11-slim
WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared HF client from the orchestrator folder
COPY orchestrator/hf_client.py .

# Copy ingestion service files
COPY ingestion/pipeline.py .
COPY ingestion/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
