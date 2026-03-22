"""
Compression Service — main.py
Replaces: ECS Fargate (Compression Service) — LLMlingua-2 / LongRefiner
Uses:     CLaRa-7B-E2E via HuggingFace Serverless API (free tier)

Endpoints:
  POST /compress   — compress a retrieved context window before generation
  GET  /health     — liveness probe
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hf_client import clara_compress

app = FastAPI(
    title="Compression Service",
    description=(
        "Layer 3: Context compression using CLaRa-7B-E2E. "
        "Replaces LLMlingua-2 / LongRefiner from the architecture. "
        "Target compression ratio is 0.25 (4x), matching the architecture's "
        "CLaRa Memory Index spec of 4x-128x compression."
    ),
    version="1.0.0",
)


# ----------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------
class CompressRequest(BaseModel):
    context:          str
    query:            str
    # 0.25 = compress to 25% of original length (4x compression)
    # Lower = more aggressive. Don't go below 0.1 on free tier —
    # CLaRa needs enough tokens to preserve key facts.
    compression_ratio: float = 0.25


class CompressResponse(BaseModel):
    compressed_context: str
    original_tokens:    int   # word count of the input context
    compressed_tokens:  int   # word count of the compressed output
    actual_ratio:       float # compressed_tokens / original_tokens


# ----------------------------------------------------------------
# POST /compress
# ----------------------------------------------------------------
@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    """
    Compresses a retrieved context window to reduce token usage before
    passing it to the generation LLM (ChatQA).

    Called by the orchestrator at Flow 7/8 in the pipeline after all
    retrievers have returned their chunks and before LLM generation.

    The orchestrator skips this endpoint if the raw context is under
    200 words — short contexts don't benefit from compression and the
    HuggingFace round-trip latency isn't worth it.
    """
    if not req.context.strip():
        raise HTTPException(status_code=400, detail="context must not be empty")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not (0.05 <= req.compression_ratio <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="compression_ratio must be between 0.05 and 1.0",
        )

    try:
        compressed = await clara_compress(
            context=req.context,
            query=req.query,
            compression_ratio=req.compression_ratio,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CLaRa inference failed: {exc}")

    orig_tokens  = len(req.context.split())
    comp_tokens  = len(compressed.split())
    actual_ratio = comp_tokens / max(orig_tokens, 1)

    return CompressResponse(
        compressed_context=compressed,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        actual_ratio=round(actual_ratio, 3),
    )


# ----------------------------------------------------------------
# GET /health
# ----------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "compression"}
