# RAG-System-Max
<div align="center">

# Modern RAG System

**Production-grade Retrieval-Augmented Generation with CLaRa compression and reasoning-aware retrieval**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1.5-FF6B35?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

*A full-stack RAG system implementing a 5-layer architecture with 7 intelligent retrieval paths, context compression, semantic caching, knowledge graph extraction, and a real-time chat interface — running entirely on free infrastructure.*

</div>

---

## Overview

This project is a complete implementation of a modern, production-grade RAG architecture. It is designed to demonstrate every component of an enterprise RAG pipeline — from document ingestion through multi-path retrieval, compression, LLM generation, and caching — without any cloud costs.

Heavy inference (LLM generation, embeddings, compression) is handled by the **HuggingFace Serverless Inference API** on the free tier. All other components run locally in Docker containers.

```
User query → Cache check → Query classification → Intelligent retrieval routing
          → Context compression → LLM generation → Cached response
```

---

## Architecture

The system implements a 5-layer architecture based on the Modern RAG System design with CLaRa and Reasoning-Aware Retrieval.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: Caching, Latency & Evaluation                         │
│  Redis (Exact Cache · Semantic Cache · Session Store)           │
│  Prometheus · Grafana                                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Generation & Orchestration                            │
│  FastAPI · LangGraph Pipeline · Query Classifier                │
│  Llama 3.1 8B Instruct (via HuggingFace)                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Compression & Context Shaping                         │
│  CLaRa compression (Llama 3.1 8B) · Context Assembler          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Retrieval & Reasoning                                 │
│  7 retrieval paths · RRF reranking · Query routing              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Data & Representation                                 │
│  Qdrant · Elasticsearch · Neo4j · PostgreSQL · Redis            │
└─────────────────────────────────────────────────────────────────┘
```

### Retrieval paths

The system automatically classifies each query and routes it to the optimal retriever:

| Query type | Retriever | Strategy |
|---|---|---|
| `simple_factoid` | **Hybrid** | Dense vectors + BM25 + Reciprocal Rank Fusion |
| `multi_hop` | **GraphRAG** | Neo4j knowledge graph traversal |
| `structured` | **HopRAG** | 2-hop dense retrieval with bridging questions |
| `global_analytic` | **SG-RAG** | StepChain query decomposition |
| `latency_sensitive` | **SelfQuery** | Direct vector lookup, no LLM overhead |
| always | **CLaRa Memory** | Compressed memory index (runs in parallel) |

### Technology stack

| Component | Technology | Role |
|---|---|---|
| Orchestration | Python · FastAPI · LangGraph | Pipeline execution |
| LLM & Compression | Llama 3.1 8B (HuggingFace) | Generation + CLaRa compression |
| Embeddings | BAAI/bge-large-en-v1.5 (HuggingFace) | 1024-dim dense vectors |
| Vector DB | Qdrant | Dense + CLaRa memory index |
| Full-text search | Elasticsearch | BM25 keyword retrieval |
| Graph DB | Neo4j | Knowledge graph / GraphRAG |
| Metadata store | PostgreSQL | Document metadata |
| Cache | Redis 7.2 | Exact + semantic cache + sessions |
| Monitoring | Prometheus + Grafana | Metrics and dashboards |

---

## Features

- **7 retrieval strategies** — automatically selected based on query classification
- **Reciprocal Rank Fusion** — merges dense and BM25 results for hybrid retrieval
- **CLaRa context compression** — reduces context to ~25% before generation
- **Semantic caching** — cosine similarity cache hits for near-duplicate queries
- **Knowledge graph extraction** — entities and relations extracted on ingestion
- **Session memory** — multi-turn conversation history via Redis
- **Real-time monitoring** — Prometheus metrics, Grafana dashboards
- **Chat UI** — dark terminal-aesthetic frontend served at `http://localhost:8000`
- **Swagger docs** — auto-generated API documentation at `http://localhost:8000/docs`
- **Zero cloud cost** — all inference via HuggingFace free tier

---

## Project structure

```
RAG System/
├── docker-compose.yml                  # Full stack definition
├── .env.example                        # Environment template
├── services/
│   ├── orchestrator/                   # Layer 4 — main entry point
│   │   ├── main.py                     # FastAPI app + all endpoints
│   │   ├── pipeline.py                 # LangGraph 6-node RAG graph
│   │   ├── cache.py                    # Exact + semantic + session cache
│   │   ├── hf_client.py               # HuggingFace inference client
│   │   ├── static/
│   │   │   └── index.html             # Chat UI (served at /)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── retrieval/                      # Layer 2 — all 7 retrievers
│   │   ├── retrievers.py              # HybridRetriever, GraphRAG, HopRAG, etc.
│   │   ├── main.py                    # /retrieve and /classify endpoints
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── compression/                    # Layer 3 — CLaRa compression
│   │   ├── main.py                    # /compress endpoint
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── ingestion/                      # Document ingestion pipeline
│       ├── pipeline.py                # 6-step ingestion pipeline
│       ├── main.py                    # /ingest endpoint
│       ├── Dockerfile
│       └── requirements.txt
├── infrastructure/
│   ├── postgres/
│   │   └── init.sql                   # Metadata schema
│   └── prometheus/
│       └── prometheus.yml             # Scrape config
└── tests/
    └── test_e2e.py                    # Full integration test suite
```

---

## Quickstart

### Prerequisites

- Docker Desktop with Compose v2
- A free HuggingFace account

### Step 1 — HuggingFace setup

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (role: Read)
3. Accept model terms at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Step 2 — Configure

```bash
git clone https://github.com/yourusername/rag-system
cd "RAG System"
cp .env.example .env
# Open .env and set: HF_API_TOKEN=hf_your_token_here
```

### Step 3 — Create the data directory

```bash
mkdir -p data/raw
```

### Step 4 — Start the stack

```bash
docker compose up --build
```

First build takes 5–10 minutes (downloading Docker images and Python packages). On subsequent runs use `docker compose up`.

### Step 5 — Open the UI

Navigate to **http://localhost:8000** in your browser.

---

## Usage

### Chat UI

Open `http://localhost:8000` for the full chat interface. Drag and drop `.txt` or `.md` files onto the left panel to ingest them, then ask questions.

### API

**Ingest a document**

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/app/data/raw/document.txt"}'
```

Place files in `data/raw/` on your host — they appear at `/app/data/raw/` inside the container.

**Query the system**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main argument of the document?",
    "session_id": "my-session"
  }'
```

Response:

```json
{
  "answer": "...",
  "query_type": "simple_factoid",
  "cache_type": "miss",
  "latency_ms": 4250.1,
  "telemetry": ["cache:miss", "classify:simple_factoid", "retrieved:7", "compression:ratio=0.24", "generation:chatqa"]
}
```

**Force a specific retrieval path**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarise all themes across the documents",
    "session_id": "my-session",
    "query_type": "global_analytic"
  }'
```

### Run the test suite

```bash
pip install httpx
python tests/test_e2e.py
```

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Chat UI | http://localhost:8000 | — |
| API docs (Swagger) | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Qdrant dashboard | http://localhost:6333/dashboard | — |
| Neo4j browser | http://localhost:7474 | neo4j / ragpassword |
| Elasticsearch | http://localhost:9200 | — |

---

## Ingestion pipeline

When a document is ingested it goes through 6 sequential steps:

```
1. Parse & chunk      → sliding window, 512 words, 64-word overlap
2. Dense embed        → BAAI/bge-large-en-v1.5 → Qdrant (vector_index_chunks)
3. BM25 index         → Elasticsearch (full_text_store)
4. KG extraction      → Llama 3.1 entity/relation extraction → Neo4j
5. CLaRa encode       → compressed memory representations → Qdrant (clara_memory_index)
6. Metadata           → PostgreSQL (documents table)
```

---

## LangGraph pipeline

Each query flows through a compiled LangGraph graph with 6 nodes:

```
cache_check ──hit──→ respond
     │
    miss
     ↓
  classify → retrieve → compress → generate → cache_store → respond
```

Cache hits return in under 50 ms. Cold-start queries (first HuggingFace call after idle) take 30–90 seconds while the model loads. Subsequent calls are fast.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HF_API_TOKEN` | required | HuggingFace API token |
| `HF_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `HF_CHATQA_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Generation model |
| `HF_CLARA_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Compression model |
| `HF_CHAT_BASE_URL` | `https://router.huggingface.co/v1` | HF chat completions URL |
| `HF_INFERENCE_BASE_URL` | `https://router.huggingface.co/hf-inference/models` | HF embedding URL |
| `CACHE_TTL_SECONDS` | `3600` | Exact cache entry lifetime |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity threshold for semantic cache |
| `COMPRESSION_MIN_WORDS` | `200` | Skip compression below this word count |
| `MAX_CONTEXT_TOKENS` | `4096` | Maximum context window size |

---

## Resetting the stack

```bash
# Stop all containers, keep data
docker compose down

# Full reset — wipes all database volumes
docker compose down -v
docker compose up --build
```

---

## Architecture reference

This implementation maps directly to the *Modern RAG Architecture with CLaRa and Reasoning-Aware Retrieval (Version 1.0)* system design. The AWS components from the production architecture are mapped to free local equivalents:

| Production (AWS) | This prototype |
|---|---|
| SageMaker Real-Time Endpoint | HuggingFace Serverless API |
| Amazon OpenSearch (vector) | Qdrant |
| Amazon OpenSearch (BM25) | Elasticsearch |
| Amazon Neptune | Neo4j |
| Amazon RDS PostgreSQL | PostgreSQL |
| Amazon ElastiCache Redis | Redis |
| ECS Fargate services | Docker containers |
| CloudWatch + X-Ray | Prometheus + Grafana |

The Python interface in `hf_client.py` mirrors what you would write for SageMaker endpoints — only the URL and auth header differ.

---

## License

MIT
