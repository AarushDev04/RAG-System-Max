"""
Retrieval Service — retrievers.py
Implements all retrieval paths from the architecture diagram:

  Flow 5a  simple_factoid    → HybridRetriever    (Dense + BM25 + RRF rerank)
  Flow 5b  multi_hop         → GraphRAGRetriever   (Neo4j KG traversal)
  Flow 5c  structured        → HopRAGRetriever     (2-hop dense)
  Flow 5d  global_analytic   → SGRAGRetriever      (StepChain decomposition)
  Flow 5e  latency_sensitive → SelfQueryRetriever  (direct vector, no LLM)
  Always   +                 → CLaRaMemRetriever   (compressed memory index)

All retrievers return list[RetrievedChunk]. The RetrievalRouter selects
the right retriever based on query type, then augments with CLaRa memory
results concurrently using asyncio.gather.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from elasticsearch import AsyncElasticsearch
from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient

from hf_client import chatqa_generate, embed_single

# ----------------------------------------------------------------
# Constants — must match ingestion/pipeline.py exactly
# ----------------------------------------------------------------
QDRANT_DENSE_COLLECTION = "vector_index_chunks"
QDRANT_CLARA_COLLECTION = "clara_memory_index"
ELASTIC_INDEX           = "full_text_store"
DEFAULT_TOP_K           = 5


# ----------------------------------------------------------------
# Shared result type
# ----------------------------------------------------------------
@dataclass
class RetrievedChunk:
    chunk_id: str
    text:     str
    score:    float
    source:   str   # "hybrid" | "graph" | "hop_rag" | "sg_rag" | "self_query" | "clara_memory"
    doc_id:   str


# ----------------------------------------------------------------
# Query classifier
# ----------------------------------------------------------------
_CLASSIFY_PROMPT = """\
Classify the following query into exactly one of these categories:
- simple_factoid    : direct factual lookup with a single answer
- multi_hop         : requires connecting facts across multiple documents
- structured        : involves tables, lists, or structured data
- global_analytic   : requires summarising or analysing across many documents
- latency_sensitive : time-critical, needs the fastest possible response

Query: {query}

Reply with only the category name and nothing else:"""


async def classify_query(query: str) -> str:
    """
    Calls ChatQA to classify the query into one of five routing categories.
    Falls back to simple_factoid if the model returns an unexpected value.
    """
    raw = await chatqa_generate(
        context="",
        question=_CLASSIFY_PROMPT.format(query=query),
        max_new_tokens=16,
        temperature=0.0,
    )
    result = raw.strip().lower()
    valid  = {
        "simple_factoid",
        "multi_hop",
        "structured",
        "global_analytic",
        "latency_sensitive",
    }
    return result if result in valid else "simple_factoid"


# ----------------------------------------------------------------
# Flow 5a — Hybrid Retriever (Dense + BM25 + RRF rerank)
# ----------------------------------------------------------------
class HybridRetriever:
    """
    Runs dense vector search (Qdrant) and keyword search (Elasticsearch BM25)
    concurrently, then merges results using Reciprocal Rank Fusion (RRF).

    RRF score = sum of 1/(k + rank) across all lists. k=60 is the standard
    default from the original RRF paper (Cormack et al., 2009).
    """

    def __init__(self, qdrant: AsyncQdrantClient, es: AsyncElasticsearch) -> None:
        self.qdrant = qdrant
        self.es     = es

    async def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        query_vec = await embed_single(query)

        # Run dense + BM25 concurrently
        dense_task = self.qdrant.search(
            collection_name=QDRANT_DENSE_COLLECTION,
            query_vector=query_vec,
            limit=top_k * 2,
        )
        bm25_task = self.es.search(
            index=ELASTIC_INDEX,
            body={"query": {"match": {"text": query}}, "size": top_k * 2},
        )
        dense_hits, bm25_resp = await asyncio.gather(dense_task, bm25_task)

        dense_chunks = [
            RetrievedChunk(
                chunk_id=str(hit.id),
                text=hit.payload["text"],
                score=hit.score,
                source="dense",
                doc_id=hit.payload["doc_id"],
            )
            for hit in dense_hits
        ]
        bm25_chunks = [
            RetrievedChunk(
                chunk_id=hit["_id"],
                text=hit["_source"]["text"],
                score=hit["_score"],
                source="bm25",
                doc_id=hit["_source"]["doc_id"],
            )
            for hit in bm25_resp["hits"]["hits"]
        ]

        return self._rrf_rerank(dense_chunks, bm25_chunks, top_k)

    @staticmethod
    def _rrf_rerank(
        list_a: list[RetrievedChunk],
        list_b: list[RetrievedChunk],
        top_k:  int,
        k:      int = 60,
    ) -> list[RetrievedChunk]:
        scores:   dict[str, float]         = {}
        registry: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(list_a):
            scores[chunk.chunk_id]   = scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank + 1)
            registry[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(list_b):
            scores[chunk.chunk_id]   = scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank + 1)
            registry[chunk.chunk_id] = chunk

        fused = []
        for chunk_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            c = registry[chunk_id]
            fused.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                score=rrf_score,
                source="hybrid",
                doc_id=c.doc_id,
            ))
        return fused


# ----------------------------------------------------------------
# Flow 5b — GraphRAG / KG Retriever
# ----------------------------------------------------------------
class GraphRAGRetriever:
    """
    Extracts named entities from the query using ChatQA, then traverses
    the Neo4j knowledge graph to find related entities and their source
    chunks. Falls back gracefully if no entities are found.
    """

    def __init__(self, neo4j_driver, qdrant: AsyncQdrantClient) -> None:
        self.driver = neo4j_driver
        self.qdrant = qdrant

    async def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        entities_raw = await chatqa_generate(
            context="",
            question=f"List every named entity in this query as a comma-separated list: {query}",
            max_new_tokens=64,
            temperature=0.0,
        )
        entities = [e.strip() for e in entities_raw.split(",") if e.strip()]
        if not entities:
            return []

        chunks: list[RetrievedChunk] = []
        async with self.driver.session() as session:
            for entity in entities[:3]:
                result = await session.run(
                    """
                    MATCH (e:Entity)-[r:RELATION]-(related:Entity)
                    WHERE e.name =~ $pattern
                    RETURN e.name, r.type, related.name, r.chunk_id, r.doc_id
                    LIMIT 10
                    """,
                    pattern=f"(?i).*{entity}.*",
                )
                async for record in result:
                    chunk_id = record.get("r.chunk_id")
                    if not chunk_id:
                        continue

                    # Fetch the full chunk text from Qdrant
                    points = await self.qdrant.retrieve(
                        QDRANT_DENSE_COLLECTION,
                        ids=[chunk_id],
                        with_payload=True,
                    )
                    if not points:
                        continue

                    chunks.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        text=points[0].payload.get(
                            "text",
                            f"{record['e.name']} {record['r.type']} {record['related.name']}",
                        ),
                        score=1.0,
                        source="graph",
                        doc_id=record.get("r.doc_id") or "",
                    ))

        return chunks[:top_k]


# ----------------------------------------------------------------
# Flow 5c — HopRAG (2-hop dense retrieval)
# ----------------------------------------------------------------
class HopRAGRetriever:
    """
    2-hop retrieval:
      Hop 1: retrieve top-3 chunks for the original query.
      Bridge: ask ChatQA what follow-up question would help.
      Hop 2: retrieve top-k chunks for the bridge question.
    Deduplicates across both hops before returning.
    """

    def __init__(self, qdrant: AsyncQdrantClient) -> None:
        self.qdrant = qdrant

    async def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        # Hop 1
        vec1  = await embed_single(query)
        hits1 = await self.qdrant.search(
            QDRANT_DENSE_COLLECTION, query_vector=vec1, limit=3
        )
        ctx1 = " ".join(h.payload["text"] for h in hits1)

        # Generate bridging sub-question
        bridge_q = await chatqa_generate(
            context=ctx1,
            question=f"What follow-up question would best help answer: {query}",
            max_new_tokens=64,
            temperature=0.0,
        )

        # Hop 2
        vec2  = await embed_single(bridge_q)
        hits2 = await self.qdrant.search(
            QDRANT_DENSE_COLLECTION, query_vector=vec2, limit=top_k
        )

        # Deduplicate, preserving order (hits1 first, then hits2)
        seen:   set[str]            = set()
        chunks: list[RetrievedChunk] = []
        for h in hits1 + hits2:
            hid = str(h.id)
            if hid not in seen:
                seen.add(hid)
                chunks.append(RetrievedChunk(
                    chunk_id=hid,
                    text=h.payload["text"],
                    score=h.score,
                    source="hop_rag",
                    doc_id=h.payload["doc_id"],
                ))

        return chunks[:top_k]


# ----------------------------------------------------------------
# Flow 5d — SG-RAG / StepChain (global analytic)
# ----------------------------------------------------------------
class SGRAGRetriever:
    """
    Decomposes the query into up to 3 sub-questions using ChatQA,
    retrieves for each sub-question concurrently, then merges and
    deduplicates the results for a comprehensive context window.
    """

    def __init__(self, qdrant: AsyncQdrantClient) -> None:
        self.qdrant = qdrant

    async def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        decomposed_raw = await chatqa_generate(
            context="",
            question=(
                f"Break this analytical question into exactly 3 sub-questions "
                f"that together would answer it fully:\n{query}\n\n"
                "Output one sub-question per line, no numbering:"
            ),
            max_new_tokens=128,
            temperature=0.0,
        )
        sub_qs = [q.strip() for q in decomposed_raw.split("\n") if q.strip()][:3]
        if not sub_qs:
            sub_qs = [query]

        # Embed all sub-questions concurrently
        vecs = await asyncio.gather(*[embed_single(q) for q in sub_qs])

        # Search for each sub-question concurrently
        search_tasks = [
            self.qdrant.search(QDRANT_DENSE_COLLECTION, query_vector=vec, limit=3)
            for vec in vecs
        ]
        all_hits_nested = await asyncio.gather(*search_tasks)

        # Merge and deduplicate
        seen:   set[str]            = set()
        chunks: list[RetrievedChunk] = []
        for hits in all_hits_nested:
            for h in hits:
                hid = str(h.id)
                if hid not in seen:
                    seen.add(hid)
                    chunks.append(RetrievedChunk(
                        chunk_id=hid,
                        text=h.payload["text"],
                        score=h.score,
                        source="sg_rag",
                        doc_id=h.payload["doc_id"],
                    ))

        return chunks[:top_k]


# ----------------------------------------------------------------
# Flow 5e — SelfQuery Retriever (latency-sensitive)
# ----------------------------------------------------------------
class SelfQueryRetriever:
    """
    Pure vector lookup — no LLM calls, no reranking.
    Fastest path. Used when query_type == latency_sensitive.
    """

    def __init__(self, qdrant: AsyncQdrantClient) -> None:
        self.qdrant = qdrant

    async def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        vec  = await embed_single(query)
        hits = await self.qdrant.search(
            QDRANT_DENSE_COLLECTION, query_vector=vec, limit=top_k
        )
        return [
            RetrievedChunk(
                chunk_id=str(h.id),
                text=h.payload["text"],
                score=h.score,
                source="self_query",
                doc_id=h.payload["doc_id"],
            )
            for h in hits
        ]


# ----------------------------------------------------------------
# CLaRa Memory Retriever
# ----------------------------------------------------------------
class CLaRaMemRetriever:
    """
    Searches the CLaRa-compressed memory index in Qdrant.
    Always runs alongside the primary retriever (not instead of it)
    to surface compressed representations that the dense index may miss.
    """

    def __init__(self, qdrant: AsyncQdrantClient) -> None:
        self.qdrant = qdrant

    async def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
        vec  = await embed_single(query)
        hits = await self.qdrant.search(
            QDRANT_CLARA_COLLECTION, query_vector=vec, limit=top_k
        )
        return [
            RetrievedChunk(
                chunk_id=str(h.id),
                text=h.payload.get("encoded_text") or h.payload.get("original_text", ""),
                score=h.score,
                source="clara_memory",
                doc_id=h.payload.get("doc_id", ""),
            )
            for h in hits
        ]


# ----------------------------------------------------------------
# Retrieval Router
# ----------------------------------------------------------------
class RetrievalRouter:
    """
    Entry point for all retrieval. Classifies the query (unless the caller
    already did), dispatches to the appropriate retriever, then augments
    the results with CLaRa memory chunks — both running concurrently.
    """

    def __init__(
        self,
        qdrant:       AsyncQdrantClient,
        es:           AsyncElasticsearch,
        neo4j_driver,
    ) -> None:
        self.hybrid    = HybridRetriever(qdrant, es)
        self.graphrag  = GraphRAGRetriever(neo4j_driver, qdrant)
        self.hoprag    = HopRAGRetriever(qdrant)
        self.sgrag     = SGRAGRetriever(qdrant)
        self.selfquery = SelfQueryRetriever(qdrant)
        self.clara_mem = CLaRaMemRetriever(qdrant)

        self._route_map = {
            "simple_factoid":    self.hybrid,
            "multi_hop":         self.graphrag,
            "structured":        self.hoprag,
            "global_analytic":   self.sgrag,
            "latency_sensitive": self.selfquery,
        }

    async def route_and_retrieve(
        self,
        query:      str,
        query_type: Optional[str] = None,
        top_k:      int           = DEFAULT_TOP_K,
    ) -> tuple[str, list[RetrievedChunk]]:
        """
        1. Classify query if type not already known.
        2. Run primary retriever + CLaRa memory retriever concurrently.
        3. Merge, deduplicate, return.
        """
        if query_type is None:
            query_type = await classify_query(query)

        retriever = self._route_map.get(query_type, self.hybrid)

        # Primary + CLaRa run at the same time — genuine concurrency
        primary_chunks, clara_chunks = await asyncio.gather(
            retriever.retrieve(query, top_k),
            self.clara_mem.retrieve(query, top_k=2),
        )

        # Merge: primary chunks first, then any non-duplicate CLaRa chunks
        seen:   set[str]            = {c.chunk_id for c in primary_chunks}
        merged: list[RetrievedChunk] = list(primary_chunks)
        for c in clara_chunks:
            if c.chunk_id not in seen:
                merged.append(c)
                seen.add(c.chunk_id)

        return query_type, merged[: top_k + 2]
