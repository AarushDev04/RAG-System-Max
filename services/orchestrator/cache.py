"""
Cache Service — cache.py
Replaces: Amazon ElastiCache for Redis (Exact + Semantic cache + Session store)

Three layers:
  Exact cache    — SHA-256(query) → answer           O(1) lookup
  Semantic cache — embed(query) → cosine sim scan    O(n) scan, in-memory
  Session store  — session_id → conversation history  TTL-based
"""

import asyncio
import hashlib
import json
import os
from typing import Optional

import numpy as np
import redis.asyncio as aioredis

from hf_client import embed_single

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
REDIS_URL                = os.getenv("REDIS_URL",                "redis://localhost:6379")
EXACT_CACHE_TTL          = int(os.getenv("CACHE_TTL_SECONDS",    "3600"))
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SESSION_TTL              = 3600 * 4   # 4 hours

# Redis key namespaces
_NS_EXACT    = "cache:exact:"
_NS_SEMANTIC = "cache:semantic:"
_NS_SESSION  = "cache:session:"


class CacheService:
    """
    Thread-safe (asyncio-safe) cache service.
    Call connect() at app startup and close() at shutdown.
    """

    def __init__(self) -> None:
        self._redis:          Optional[aioredis.Redis] = None
        # In-memory list of {query, vec, answer} for semantic similarity scan.
        # Populated from Redis on startup so cache survives container restarts.
        self._semantic_store: list[dict]               = []

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    async def connect(self) -> None:
        self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        await self._reload_semantic_store()

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def _reload_semantic_store(self) -> None:
        """Restore the in-memory semantic store from Redis on startup."""
        if not self._redis:
            return
        keys = await self._redis.keys(f"{_NS_SEMANTIC}*")
        for key in keys:
            raw = await self._redis.get(key)
            if raw:
                try:
                    self._semantic_store.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass   # skip corrupted entries

    # ----------------------------------------------------------------
    # Exact cache — O(1), no embedding needed
    # ----------------------------------------------------------------
    async def get_exact(self, query: str) -> Optional[str]:
        key = _NS_EXACT + hashlib.sha256(query.encode()).hexdigest()
        return await self._redis.get(key)  # type: ignore[union-attr]

    async def set_exact(self, query: str, answer: str) -> None:
        key = _NS_EXACT + hashlib.sha256(query.encode()).hexdigest()
        await self._redis.setex(key, EXACT_CACHE_TTL, answer)  # type: ignore[union-attr]

    # ----------------------------------------------------------------
    # Semantic cache — embeds the query, scans in-memory store
    # ----------------------------------------------------------------
    async def get_semantic(self, query: str) -> Optional[str]:
        if not self._semantic_store:
            return None

        query_vec   = np.array(await embed_single(query))
        query_norm  = np.linalg.norm(query_vec)
        best_score  = 0.0
        best_answer = None

        for entry in self._semantic_store:
            cached_vec  = np.array(entry["vec"])
            cached_norm = np.linalg.norm(cached_vec)
            if cached_norm == 0 or query_norm == 0:
                continue
            score = float(np.dot(query_vec, cached_vec) / (query_norm * cached_norm))
            if score > best_score:
                best_score  = score
                best_answer = entry["answer"]

        return best_answer if best_score >= SEMANTIC_CACHE_THRESHOLD else None

    async def set_semantic(self, query: str, answer: str) -> None:
        vec   = await embed_single(query)
        key   = _NS_SEMANTIC + hashlib.sha256(query.encode()).hexdigest()
        entry = {"query": query, "vec": vec, "answer": answer}
        await self._redis.setex(  # type: ignore[union-attr]
            key,
            EXACT_CACHE_TTL * 4,   # semantic entries live 4x longer
            json.dumps(entry),
        )
        self._semantic_store.append(entry)

    # ----------------------------------------------------------------
    # Unified check — exact first (fast), then semantic (slower)
    # ----------------------------------------------------------------
    async def check(self, query: str) -> tuple[Optional[str], str]:
        """
        Returns (answer, cache_type).
        cache_type is "exact", "semantic", or "miss".
        """
        exact = await self.get_exact(query)
        if exact:
            return exact, "exact"

        semantic = await self.get_semantic(query)
        if semantic:
            return semantic, "semantic"

        return None, "miss"

    async def store(self, query: str, answer: str) -> None:
        """
        Persist a query→answer pair to both caches concurrently.
        Exact write is instant; semantic write embeds the query first.
        """
        await asyncio.gather(
            self.set_exact(query, answer),
            self.set_semantic(query, answer),
        )

    # ----------------------------------------------------------------
    # Session store — conversation history per session_id
    # ----------------------------------------------------------------
    async def get_session(self, session_id: str) -> list[dict]:
        raw = await self._redis.get(_NS_SESSION + session_id)  # type: ignore[union-attr]
        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []

    async def append_session(
        self,
        session_id: str,
        role:       str,
        content:    str,
    ) -> None:
        """Append one turn to the session history and reset the TTL."""
        history = await self.get_session(session_id)
        history.append({"role": role, "content": content})
        await self._redis.setex(  # type: ignore[union-attr]
            _NS_SESSION + session_id,
            SESSION_TTL,
            json.dumps(history),
        )
