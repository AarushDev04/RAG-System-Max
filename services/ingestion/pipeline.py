"""
Ingestion Pipeline — pipeline.py
Replaces: S3 Event → Step Functions → Lambda → ECS Fargate (Chunk & Embed)
          → ECS Fargate (KG Extraction) → SageMaker Batch Transform (CLaRa)

Six-step pipeline for every document:
  1. Parse & chunk          (sliding window, 512 words, 64-word overlap)
  2. Dense embed  → Qdrant  (vector_index_chunks collection)
  3. BM25 index   → Elasticsearch (full_text_store index)
  4. KG extract   → Neo4j   (Entity nodes + RELATION edges)
  5. CLaRa encode → Qdrant  (clara_memory_index collection)
  6. Metadata     → PostgreSQL (documents table)
"""

import os
import uuid
from pathlib import Path
from datetime import datetime

import asyncpg
from elasticsearch import AsyncElasticsearch
from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from hf_client import clara_encode_memory, chatqa_generate, embed_texts

# ----------------------------------------------------------------
# Constants — must match the retrieval service exactly
# ----------------------------------------------------------------
QDRANT_DENSE_COLLECTION = "vector_index_chunks"
QDRANT_CLARA_COLLECTION = "clara_memory_index"
ELASTIC_INDEX           = "full_text_store"
VECTOR_DIM              = 1024   # BAAI/bge-large-en-v1.5 output dimension
EMBED_BATCH_SIZE        = 32     # HuggingFace free tier safe batch size
KG_CHUNK_LIMIT          = 5     # max chunks sent for KG extraction per doc


# ----------------------------------------------------------------
# Step 1: Parse & chunk
# ----------------------------------------------------------------
def chunk_document(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_words: int = 16,
) -> list[dict]:
    """
    Sliding window word-level chunker.
    Returns a list of dicts: {chunk_id, text, word_start, word_end}.

    chunk_size=512 and overlap=64 gives good retrieval recall for most
    documents. For very long technical docs, try chunk_size=256.
    """
    words  = text.split()
    step   = chunk_size - overlap
    chunks = []

    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]
        if len(window) < min_words:
            break
        chunks.append({
            "chunk_id":   str(uuid.uuid4()),
            "text":       " ".join(window),
            "word_start": i,
            "word_end":   i + len(window),
        })

    return chunks


# ----------------------------------------------------------------
# Step 2: Dense embed → Qdrant
# ----------------------------------------------------------------
async def ingest_dense_vectors(
    qdrant: AsyncQdrantClient,
    doc_id: str,
    chunks: list[dict],
) -> None:
    # Create collection if it doesn't exist yet
    existing = {c.name for c in (await qdrant.get_collections()).collections}
    if QDRANT_DENSE_COLLECTION not in existing:
        await qdrant.create_collection(
            QDRANT_DENSE_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    # Embed in batches — HF free tier handles 32 at a time comfortably
    texts    = [c["text"] for c in chunks]
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_vecs = await embed_texts(texts[i : i + EMBED_BATCH_SIZE])
        all_vecs.extend(batch_vecs)

    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=vec,
            payload={
                "doc_id":     doc_id,
                "text":       chunk["text"],
                "word_start": chunk["word_start"],
                "word_end":   chunk["word_end"],
            },
        )
        for chunk, vec in zip(chunks, all_vecs)
    ]
    await qdrant.upsert(QDRANT_DENSE_COLLECTION, points=points)


# ----------------------------------------------------------------
# Step 3: BM25 → Elasticsearch
# ----------------------------------------------------------------
async def ingest_bm25(
    es: AsyncElasticsearch,
    doc_id: str,
    chunks: list[dict],
) -> None:
    # Create index with explicit BM25 similarity if it doesn't exist
    if not await es.indices.exists(index=ELASTIC_INDEX):
        await es.indices.create(
            index=ELASTIC_INDEX,
            body={
                "settings": {
                    "similarity": {"default": {"type": "BM25"}}
                },
                "mappings": {
                    "properties": {
                        "doc_id":     {"type": "keyword"},
                        "chunk_id":   {"type": "keyword"},
                        "text":       {"type": "text", "analyzer": "english"},
                        "word_start": {"type": "integer"},
                    }
                },
            },
        )

    # Build bulk body — alternating action + document lines
    ops = []
    for chunk in chunks:
        ops.append({"index": {"_index": ELASTIC_INDEX, "_id": chunk["chunk_id"]}})
        ops.append({
            "doc_id":     doc_id,
            "chunk_id":   chunk["chunk_id"],
            "text":       chunk["text"],
            "word_start": chunk["word_start"],
        })

    await es.bulk(body=ops)


# ----------------------------------------------------------------
# Step 4: KG extraction → Neo4j
# ----------------------------------------------------------------
async def extract_kg(
    neo4j_driver,
    doc_id: str,
    chunks: list[dict],
) -> None:
    """
    Asks ChatQA to extract (Subject, Relation, Object) triples from each
    chunk, then writes them as Entity nodes and RELATION edges in Neo4j.

    Limited to KG_CHUNK_LIMIT chunks per document to stay within HF
    free-tier rate limits. Production would use a dedicated NER/RE model.
    """
    async with neo4j_driver.session() as session:
        for chunk in chunks[:KG_CHUNK_LIMIT]:
            raw = await chatqa_generate(
                context=chunk["text"],
                question=(
                    "Extract all named entities and their relationships from the context. "
                    "Return each as a triple on its own line in this exact format: "
                    "(Subject, Relation, Object). Only include factual relationships."
                ),
                max_new_tokens=256,
            )

            for line in raw.split("\n"):
                line = line.strip().strip("()")
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    continue
                subj, rel, obj = parts
                if not subj or not rel or not obj:
                    continue

                await session.run(
                    """
                    MERGE (s:Entity {name: $subj})
                    MERGE (o:Entity {name: $obj})
                    MERGE (s)-[r:RELATION {type: $rel}]->(o)
                      ON CREATE SET r.doc_id = $doc_id, r.chunk_id = $chunk_id
                    """,
                    subj=subj,
                    rel=rel,
                    obj=obj,
                    doc_id=doc_id,
                    chunk_id=chunk["chunk_id"],
                )


# ----------------------------------------------------------------
# Step 5: CLaRa memory encode → Qdrant
# ----------------------------------------------------------------
async def ingest_clara_memory(
    qdrant: AsyncQdrantClient,
    doc_id: str,
    chunks: list[dict],
) -> None:
    """
    Encodes every other chunk through CLaRa and stores the compressed
    memory representations in a separate Qdrant collection.

    Sampling every other chunk (chunks[::2]) approximates the 4x
    compression ratio described in the architecture — we store half
    the chunks but each is compressed to ~25% of its original length.
    """
    existing = {c.name for c in (await qdrant.get_collections()).collections}
    if QDRANT_CLARA_COLLECTION not in existing:
        await qdrant.create_collection(
            QDRANT_CLARA_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    sampled       = chunks[::2]
    encoded_texts = []
    for chunk in sampled:
        enc = await clara_encode_memory(chunk["text"])
        encoded_texts.append(enc)

    if not encoded_texts:
        return

    # Embed the CLaRa-encoded representations
    all_vecs = []
    for i in range(0, len(encoded_texts), EMBED_BATCH_SIZE):
        batch_vecs = await embed_texts(encoded_texts[i : i + EMBED_BATCH_SIZE])
        all_vecs.extend(batch_vecs)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "doc_id":        doc_id,
                "original_text": chunk["text"],
                "encoded_text":  enc_text,
            },
        )
        for chunk, enc_text, vec in zip(sampled, encoded_texts, all_vecs)
    ]
    await qdrant.upsert(QDRANT_CLARA_COLLECTION, points=points)


# ----------------------------------------------------------------
# Step 6: Metadata → PostgreSQL
# ----------------------------------------------------------------
async def record_metadata(
    pool: asyncpg.Pool,
    doc_id: str,
    filename: str,
    num_chunks: int,
) -> None:
    await pool.execute(
        """
        INSERT INTO documents (doc_id, filename, num_chunks, ingested_at)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (doc_id) DO UPDATE
          SET ingested_at = EXCLUDED.ingested_at,
              num_chunks  = EXCLUDED.num_chunks
        """,
        doc_id,
        filename,
        num_chunks,
        datetime.utcnow(),
    )


# ----------------------------------------------------------------
# Master entry point
# ----------------------------------------------------------------
async def ingest_document(filepath: str) -> dict:
    """
    Runs all six ingestion steps sequentially for a single document.
    Opens and closes its own DB connections — safe to call concurrently
    for different documents.

    Returns:
        {"doc_id": str, "chunks": int, "status": "ingested"}
    """
    path   = Path(filepath)
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))
    text   = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_document(text)

    print(f"[Ingestion] Starting  doc_id={doc_id}  file={path.name}  chunks={len(chunks)}")

    # Open all connections once up front
    qdrant  = AsyncQdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
    es      = AsyncElasticsearch([f"http://{os.getenv('ELASTIC_HOST', 'localhost')}:9200"])
    neo4j   = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "ragpassword"),
        ),
    )
    pg_pool = await asyncpg.create_pool(
        os.getenv(
            "POSTGRES_DSN",
            "postgresql://raguser:ragpassword@localhost:5432/rag_metadata",
        )
    )

    try:
        await ingest_dense_vectors(qdrant, doc_id, chunks)
        print("[Ingestion] ✓ Dense vectors  → Qdrant")

        await ingest_bm25(es, doc_id, chunks)
        print("[Ingestion] ✓ BM25 index     → Elasticsearch")

        await extract_kg(neo4j, doc_id, chunks)
        print("[Ingestion] ✓ Knowledge graph → Neo4j")

        await ingest_clara_memory(qdrant, doc_id, chunks)
        print("[Ingestion] ✓ CLaRa memory   → Qdrant")

        await record_metadata(pg_pool, doc_id, path.name, len(chunks))
        print("[Ingestion] ✓ Metadata        → PostgreSQL")

    finally:
        # Always close connections, even if a step fails
        await qdrant.close()
        await es.close()
        await neo4j.close()
        await pg_pool.close()

    print(f"[Ingestion] Done  doc_id={doc_id}")
    return {"doc_id": doc_id, "chunks": len(chunks), "status": "ingested"}
