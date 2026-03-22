"""
RAG Orchestrator Pipeline — pipeline.py
Replaces: Amazon ECS Fargate (Orchestrator Service) — Python / FastAPI / LangGraph

LangGraph node sequence:
  cache_check → [hit: respond] → classify → retrieve → compress → generate → cache_store → respond

Mirrors architecture flows 1-14 from the high-level system design diagram:
  Flow 1  : User query arrives
  Flow 2/3: Cache check (exact → semantic → miss)
  Flow 4  : Classification → retrieval routing
  Flow 5a-e: Retriever selection
  Flow 7/8: Context compression (CLaRa)
  Flow 9/10: LLM generation (ChatQA)
  Flow 11 : Answer → cache storage
  Flow 13 : Telemetry collected throughout
"""

import os
from functools import partial
from typing import Annotated, Optional
from operator import add

import httpx
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from cache import CacheService
from hf_client import chatqa_generate

# ----------------------------------------------------------------
# Service URLs — set via docker-compose environment
# ----------------------------------------------------------------
RETRIEVAL_URL   = os.getenv("RETRIEVAL_SERVICE_URL",   "http://retrieval:8002")
COMPRESSION_URL = os.getenv("COMPRESSION_SERVICE_URL", "http://compression:8003")

# Skip compression for short contexts — the HF round-trip isn't worth it
COMPRESSION_MIN_WORDS = int(os.getenv("COMPRESSION_MIN_WORDS", "200"))


# ----------------------------------------------------------------
# Graph state — passed between every node
# ----------------------------------------------------------------
class RAGState(TypedDict):
    query:              str
    session_id:         str
    query_type:         Optional[str]
    cache_hit:          bool
    cache_type:         str
    retrieved_chunks:   list[dict]
    raw_context:        Optional[str]
    compressed_context: Optional[str]
    answer:             Optional[str]
    # Annotated[list, add] means LangGraph merges telemetry by concatenation
    # instead of overwriting — each node appends without knowing what came before
    telemetry:          Annotated[list[str], add]


# ----------------------------------------------------------------
# Node 1: Cache check  (Flows 2 & 3)
# ----------------------------------------------------------------
async def _cache_check_node(state: RAGState, cache: CacheService) -> dict:
    cached_answer, cache_type = await cache.check(state["query"])
    if cached_answer:
        return {
            "cache_hit":  True,
            "cache_type": cache_type,
            "answer":     cached_answer,
            "telemetry":  [f"cache:{cache_type}"],
        }
    return {
        "cache_hit":  False,
        "cache_type": "miss",
        "telemetry":  ["cache:miss"],
    }


# ----------------------------------------------------------------
# Node 2: Query classification  (Flow 4)
# ----------------------------------------------------------------
async def _classify_node(state: RAGState) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{RETRIEVAL_URL}/classify",
            json={"query": state["query"]},
            timeout=30.0,
        )
        resp.raise_for_status()
        query_type = resp.json().get("query_type", "simple_factoid")

    return {
        "query_type": query_type,
        "telemetry":  [f"classify:{query_type}"],
    }


# ----------------------------------------------------------------
# Node 3: Retrieval  (Flows 5a-5e)
# ----------------------------------------------------------------
async def _retrieve_node(state: RAGState) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{RETRIEVAL_URL}/retrieve",
            json={
                "query":      state["query"],
                "query_type": state["query_type"],
                "top_k":      7,
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        data = resp.json()

    chunks  = data.get("chunks", [])
    raw_ctx = "\n\n".join(
        f"[{c['source']}] {c['text']}" for c in chunks
    )

    return {
        "retrieved_chunks": chunks,
        "raw_context":      raw_ctx,
        "telemetry":        [f"retrieved:{len(chunks)}"],
    }


# ----------------------------------------------------------------
# Node 4: Compression  (Flows 7 & 8 — Layer 3)
# ----------------------------------------------------------------
async def _compress_node(state: RAGState) -> dict:
    raw = state.get("raw_context") or ""

    # Skip compression for short contexts — not worth the latency
    if not raw or len(raw.split()) < COMPRESSION_MIN_WORDS:
        return {
            "compressed_context": raw,
            "telemetry":          ["compression:skipped"],
        }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{COMPRESSION_URL}/compress",
            json={
                "context":          raw,
                "query":            state["query"],
                "compression_ratio": 0.25,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        compressed = resp.json().get("compressed_context", raw)

    actual_ratio = len(compressed.split()) / max(len(raw.split()), 1)
    return {
        "compressed_context": compressed,
        "telemetry":          [f"compression:ratio={actual_ratio:.2f}"],
    }


# ----------------------------------------------------------------
# Node 5: LLM generation  (Flows 9 & 10)
# ----------------------------------------------------------------
async def _generate_node(state: RAGState) -> dict:
    # Prefer compressed context; fall back to raw if compression was skipped
    ctx = state.get("compressed_context") or state.get("raw_context") or ""

    answer = await chatqa_generate(
        context=ctx,
        question=state["query"],
        max_new_tokens=512,
        temperature=0.1,
    )

    return {
        "answer":    answer,
        "telemetry": ["generation:chatqa"],
    }


# ----------------------------------------------------------------
# Node 6: Cache store  (Flow 11)
# ----------------------------------------------------------------
async def _cache_store_node(state: RAGState, cache: CacheService) -> dict:
    answer = state.get("answer", "")
    # Only store if we have a real answer and this wasn't already a cache hit
    if answer and not state.get("cache_hit"):
        await cache.store(state["query"], answer)
        return {"telemetry": ["cache:stored"]}
    return {"telemetry": ["cache:store_skipped"]}


# ----------------------------------------------------------------
# Conditional edge: short-circuit on cache hit
# ----------------------------------------------------------------
def _route_after_cache(state: RAGState) -> str:
    return "respond" if state["cache_hit"] else "classify"


# ----------------------------------------------------------------
# Compiled graph — built once at module load, reused for every query.
#
# Building and compiling the graph is not free — it creates the
# node registry, validates edges, and sets up the state reducer.
# Doing this per-query would waste ~5-10ms per call and allocate
# unnecessary objects. We compile once and call ainvoke() repeatedly.
# ----------------------------------------------------------------
def build_pipeline(cache: CacheService) -> object:
    """
    Compile the LangGraph pipeline. Call this once at app startup
    and store the result. Pass the compiled graph to run_pipeline().
    """
    graph = StateGraph(RAGState)

    # Bind the cache instance to the nodes that need it via partial()
    graph.add_node("cache_check", partial(_cache_check_node, cache=cache))
    graph.add_node("classify",    _classify_node)
    graph.add_node("retrieve",    _retrieve_node)
    graph.add_node("compress",    _compress_node)
    graph.add_node("generate",    _generate_node)
    graph.add_node("cache_store", partial(_cache_store_node, cache=cache))
    graph.add_node("respond",     lambda s: s)   # passthrough terminal node

    graph.set_entry_point("cache_check")
    graph.add_conditional_edges("cache_check", _route_after_cache)
    graph.add_edge("classify",    "retrieve")
    graph.add_edge("retrieve",    "compress")
    graph.add_edge("compress",    "generate")
    graph.add_edge("generate",    "cache_store")
    graph.add_edge("cache_store", "respond")
    graph.add_edge("respond",     END)

    return graph.compile()


async def run_pipeline(
    query:      str,
    session_id: str,
    cache:      CacheService,
    pipeline:   object | None = None,
) -> dict:
    """
    Run the compiled RAG pipeline for one query.

    Args:
        query:      The user's question.
        session_id: Used to key the session store in Redis.
        cache:      The CacheService instance (shared across requests).
        pipeline:   Pre-compiled LangGraph graph. If None, a new one is
                    compiled (convenient for tests, avoid in production).

    Returns:
        {"answer": str, "query_type": str, "cache_type": str, "telemetry": list[str]}
    """
    if pipeline is None:
        pipeline = build_pipeline(cache)

    initial: RAGState = {
        "query":              query,
        "session_id":         session_id,
        "query_type":         None,
        "cache_hit":          False,
        "cache_type":         "miss",
        "retrieved_chunks":   [],
        "raw_context":        None,
        "compressed_context": None,
        "answer":             None,
        "telemetry":          [],
    }

    final = await pipeline.ainvoke(initial)

    return {
        "answer":     final.get("answer")     or "",
        "query_type": final.get("query_type") or "unknown",
        "cache_type": final.get("cache_type") or "miss",
        "telemetry":  final.get("telemetry")  or [],
    }
