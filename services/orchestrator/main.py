"""
Orchestrator Service — main.py
FastAPI app wrapping the LangGraph RAG pipeline.

Endpoints:
  POST /query              — run the full RAG pipeline
  POST /ingest             — queue a document for ingestion
  GET  /health             — check all downstream services
  GET  /metrics            — Prometheus scrape target
  GET  /session/{id}       — fetch conversation history
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.responses import Response

from cache import CacheService
from hf_client import close_client
from pipeline import build_pipeline, run_pipeline

# ----------------------------------------------------------------
# Prometheus metrics  (replaces CloudWatch / X-Ray)
# ----------------------------------------------------------------
QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    ["query_type", "cache_type"],
)
LATENCY_HIST = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG pipeline latency in seconds",
)
ERROR_COUNTER = Counter(
    "rag_errors_total",
    "Total RAG pipeline errors",
)

# ----------------------------------------------------------------
# App-level singletons — initialised in lifespan, reused every request
# ----------------------------------------------------------------
cache_service: CacheService = CacheService()
_pipeline = None   # compiled LangGraph graph, built once at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline

    # Startup
    await cache_service.connect()
    _pipeline = build_pipeline(cache_service)   # compile graph once

    yield

    # Shutdown
    await cache_service.close()
    await close_client()   # close the shared httpx connection pool


app = FastAPI(
    title="Modern RAG System",
    description=(
        "Production-grade RAG with CLaRa compression and "
        "reasoning-aware retrieval. Free-tier HuggingFace inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------
class QueryRequest(BaseModel):
    query:      str
    session_id: str = "default"
    # Override auto-classification if you already know the query type:
    # simple_factoid | multi_hop | structured | global_analytic | latency_sensitive
    query_type: Optional[str] = None


class QueryResponse(BaseModel):
    answer:     str
    query_type: str
    cache_type: str    # "exact" | "semantic" | "miss"
    latency_ms: float
    telemetry:  list[str]


class IngestRequest(BaseModel):
    # Absolute path inside the ingestion container.
    # Files in data/raw/ on your host are mounted at /app/data/raw/ inside.
    filepath: str


class HealthResponse(BaseModel):
    status:   str    # "healthy" | "degraded"
    services: dict


# ----------------------------------------------------------------
# POST /query
# ----------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Main RAG endpoint. Runs the compiled LangGraph pipeline:
      cache_check → classify → retrieve → compress → generate → cache_store

    Cache hits short-circuit after the first node — expect < 50 ms.
    Cold-start queries (first HuggingFace call after idle) may take 30-60 s.
    """
    t0 = time.perf_counter()

    try:
        result = await run_pipeline(
            query=req.query,
            session_id=req.session_id,
            cache=cache_service,
            pipeline=_pipeline,   # reuse the pre-compiled graph
        )
    except Exception as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=500, detail=str(exc))

    latency_s  = time.perf_counter() - t0
    latency_ms = latency_s * 1000

    LATENCY_HIST.observe(latency_s)
    QUERY_COUNTER.labels(
        query_type=result["query_type"],
        cache_type=result["cache_type"],
    ).inc()

    # Persist both turns to Redis session store
    await asyncio.gather(
        cache_service.append_session(req.session_id, "user",      req.query),
        cache_service.append_session(req.session_id, "assistant", result["answer"]),
    )

    return QueryResponse(
        answer=result["answer"],
        query_type=result["query_type"],
        cache_type=result["cache_type"],
        latency_ms=round(latency_ms, 1),
        telemetry=result["telemetry"],
    )


# ----------------------------------------------------------------
# POST /ingest
# ----------------------------------------------------------------
@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Queues a document for ingestion and returns immediately.
    The ingestion service runs the full 6-step pipeline in the background:
      chunk → embed (Qdrant) → BM25 (Elasticsearch) → KG (Neo4j)
      → CLaRa encode (Qdrant) → metadata (PostgreSQL)

    Watch progress: docker compose logs -f ingestion
    """
    ingestion_url = os.getenv("INGESTION_SERVICE_URL", "http://ingestion:8001")

    async def _run() -> None:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{ingestion_url}/ingest",
                json={"filepath": req.filepath},
                timeout=300.0,
            )

    background_tasks.add_task(_run)
    return {"status": "ingestion_queued", "filepath": req.filepath}


# ----------------------------------------------------------------
# GET /health
# ----------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Pings every downstream service concurrently.
    Returns "healthy" only when all dependencies are reachable.
    """
    services: dict[str, str] = {}

    async def _check(name: str, url: str) -> None:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=3.0)
            services[name] = "ok" if r.status_code < 400 else "degraded"
        except Exception:
            services[name] = "unreachable"

    await asyncio.gather(
        _check("qdrant",        "http://qdrant:6333/healthz"),
        _check("elasticsearch", "http://elasticsearch:9200/_cluster/health"),
        _check("neo4j",         "http://neo4j:7474"),
        _check("retrieval",     "http://retrieval:8002/health"),
        _check("compression",   "http://compression:8003/health"),
    )

    overall = "healthy" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)


# ----------------------------------------------------------------
# GET /metrics
# ----------------------------------------------------------------
@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint. Grafana polls this every 15 s."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ----------------------------------------------------------------
# GET /session/{session_id}
# ----------------------------------------------------------------
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Returns the full conversation history for a session.
    Each entry: {"role": "user" | "assistant", "content": "..."}
    """
    history = await cache_service.get_session(session_id)
    return {"session_id": session_id, "history": history}


# ----------------------------------------------------------------
# POST /ingest-text
# Accepts raw text content from the browser frontend and writes it
# to data/raw/ inside the container, then triggers ingestion.
# This is the endpoint the chat UI calls when you drag-and-drop a file.
# ----------------------------------------------------------------
class IngestTextRequest(BaseModel):
    filename: str
    content:  str


@app.post("/ingest-text")
async def ingest_text_endpoint(req: IngestTextRequest, background_tasks: BackgroundTasks):
    """
    Writes raw text content to /app/data/raw/{filename} and queues ingestion.
    Called by the frontend when the user drops a file onto the ingest zone.
    Only .txt and .md files are safely handled this way — for PDFs use
    the volume-mount path approach via POST /ingest instead.
    """
    import re
    safe_name = re.sub(r"[^\w\.\-]", "_", req.filename)
    dest_path = f"/app/data/raw/{safe_name}"

    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(req.content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")

    ingestion_url = os.getenv("INGESTION_SERVICE_URL", "http://ingestion:8001")

    async def _run() -> None:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{ingestion_url}/ingest",
                json={"filepath": dest_path},
                timeout=300.0,
            )

    background_tasks.add_task(_run)
    return {"status": "ingestion_queued", "filepath": dest_path, "filename": safe_name}


# ----------------------------------------------------------------
# GET /  →  serve the chat frontend
# Static files from services/orchestrator/static/ are mounted here.
# Place index.html in that folder and it loads at http://localhost:8000
# ----------------------------------------------------------------
from fastapi.staticfiles import StaticFiles
import pathlib

_static_dir = pathlib.Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
