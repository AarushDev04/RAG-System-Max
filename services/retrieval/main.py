"""
Retrieval Service — main.py
FastAPI entry point for Layer 2 retrieval.

Endpoints:
  POST /retrieve   — classify query and return ranked chunks
  POST /classify   — classify query type only (used by orchestrator pipeline)
  GET  /health     — liveness probe
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, HTTPException
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from retrievers import DEFAULT_TOP_K, RetrievalRouter, classify_query

# ----------------------------------------------------------------
# Shared connection state — initialised once in lifespan
# ----------------------------------------------------------------
_qdrant: Optional[AsyncQdrantClient]  = None
_es:     Optional[AsyncElasticsearch] = None
_neo4j                                = None   # AsyncDriver — no public type exported
_router: Optional[RetrievalRouter]    = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qdrant, _es, _neo4j, _router

    _qdrant = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=6333,
    )
    _es = AsyncElasticsearch(
        [f"http://{os.getenv('ELASTIC_HOST', 'elasticsearch')}:9200"]
    )
    _neo4j = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        auth=(
            os.getenv("NEO4J_USER",     "neo4j"),
            os.getenv("NEO4J_PASSWORD", "ragpassword"),
        ),
    )
    _router = RetrievalRouter(_qdrant, _es, _neo4j)

    yield

    await _qdrant.close()
    await _es.close()
    await _neo4j.close()


app = FastAPI(
    title="Retrieval Service",
    description=(
        "Layer 2: Routes queries to the correct retriever "
        "(Hybrid, GraphRAG, HopRAG, SG-RAG, SelfQuery, CLaRa Memory) "
        "based on query type classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------
class RetrieveRequest(BaseModel):
    query:      str
    query_type: Optional[str] = None   # if None, auto-classified inside router
    top_k:      int = DEFAULT_TOP_K


class ClassifyRequest(BaseModel):
    query: str


# ----------------------------------------------------------------
# POST /retrieve
# ----------------------------------------------------------------
@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    """
    Classifies the query (unless query_type is provided), routes it to
    the right retriever, augments with CLaRa memory chunks concurrently,
    and returns the merged ranked list.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialised yet")

    try:
        query_type, chunks = await _router.route_and_retrieve(
            query=req.query,
            query_type=req.query_type,
            top_k=req.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query_type": query_type,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text":     c.text,
                "score":    c.score,
                "source":   c.source,
                "doc_id":   c.doc_id,
            }
            for c in chunks
        ],
    }


# ----------------------------------------------------------------
# POST /classify
# ----------------------------------------------------------------
@app.post("/classify")
async def classify(req: ClassifyRequest):
    """
    Classifies a query into one of five routing categories without
    running retrieval. Called by the orchestrator's classify_node.
    """
    try:
        query_type = await classify_query(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"query_type": query_type}


# ----------------------------------------------------------------
# GET /health
# ----------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "retrieval"}
