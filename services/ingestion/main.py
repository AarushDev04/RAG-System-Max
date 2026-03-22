"""
Ingestion Service — main.py
FastAPI entry point for the document ingestion pipeline.

Endpoints:
  POST /ingest   — run the full 6-step ingestion pipeline for a document
  GET  /health   — liveness probe
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import ingest_document

app = FastAPI(
    title="Ingestion Service",
    description="Runs the 6-step document ingestion pipeline (chunk, embed, BM25, KG, CLaRa, metadata).",
    version="1.0.0",
)


class IngestRequest(BaseModel):
    # Absolute path inside the container.
    # data/raw/ on your host maps to /app/data/raw/ inside the container
    # via the volume mount in docker-compose.yml.
    filepath: str


# ----------------------------------------------------------------
# POST /ingest
# ----------------------------------------------------------------
@app.post("/ingest")
async def ingest(req: IngestRequest):
    """
    Runs all six ingestion steps synchronously and returns when done.

    The orchestrator calls this via a BackgroundTask so it doesn't
    block the /query endpoint. Watch progress with:
      docker compose logs -f ingestion
    """
    try:
        result = await ingest_document(req.filepath)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"File not found inside container: {req.filepath}. "
                   "Make sure the file is in data/raw/ on your host machine.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result


# ----------------------------------------------------------------
# GET /health
# ----------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ingestion"}
