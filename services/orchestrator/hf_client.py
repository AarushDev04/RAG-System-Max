"""
HuggingFace Inference Client — hf_client.py

Embedding:  router.huggingface.co/hf-inference/models  (CPU, works fine)
LLM calls:  router.huggingface.co/v1/chat/completions  (OpenAI-compatible)

This file is copied into every service container that needs model inference.
Edit it only here — services/orchestrator/hf_client.py
The Dockerfiles pick it up from there.
"""

import asyncio
import os

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
_raw_token = os.environ.get("HF_API_TOKEN", "")
if not _raw_token:
    raise EnvironmentError(
        "HF_API_TOKEN is not set. "
        "Get a free token at https://huggingface.co/settings/tokens "
        "and add it to your .env file."
    )

HF_TOKEN        = _raw_token
EMBED_BASE_URL  = os.environ.get(
    "HF_INFERENCE_BASE_URL",
    "https://router.huggingface.co/hf-inference/models",
)
CHAT_BASE_URL   = os.environ.get(
    "HF_CHAT_BASE_URL",
    "https://router.huggingface.co/v1",
)
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type":  "application/json",
}

EMBEDDING_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
CHATQA_MODEL    = os.environ.get("HF_CHATQA_MODEL",    "meta-llama/Llama-3.1-8B-Instruct")
CLARA_MODEL = os.environ.get("HF_CLARA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
# ----------------------------------------------------------------
# Persistent HTTP client
# ----------------------------------------------------------------
_http_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=90.0)
    return _http_client

async def close_client() -> None:
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None

# ----------------------------------------------------------------
# Low-level POST with cold-start retry
# ----------------------------------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)
async def _post(url: str, payload: dict) -> dict | list:
    client = _get_client()
    resp   = await client.post(url, headers=HEADERS, json=payload)

    if resp.status_code == 503:
        body      = resp.json()
        wait_secs = float(body.get("estimated_time", 20))
        await asyncio.sleep(wait_secs)
        raise RuntimeError(f"Model loading, waited {wait_secs:.0f}s — retrying")

    resp.raise_for_status()
    return resp.json()

# ----------------------------------------------------------------
# Chat completions helper (used by both ChatQA and CLaRa)
# ----------------------------------------------------------------
async def _chat_complete(
    model:          str,
    messages:       list[dict],
    max_new_tokens: int   = 512,
    temperature:    float = 0.1,
) -> str:
    """
    Calls the OpenAI-compatible /v1/chat/completions endpoint.
    Works with any model available on the HF router.
    """
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_new_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    result = await _post(f"{CHAT_BASE_URL}/chat/completions", payload)
    return result["choices"][0]["message"]["content"].strip()

# ----------------------------------------------------------------
# 1. Embeddings  (unchanged — hf-inference CPU endpoint works fine)
# ----------------------------------------------------------------
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts with BAAI/bge-large-en-v1.5.
    Returns a list of 1024-dimensional float vectors.
    Keep batches at 32 or fewer to stay within free-tier limits.
    """
    if not texts:
        return []
    result = await _post(
        f"{EMBED_BASE_URL}/{EMBEDDING_MODEL}",
        {"inputs": texts},
    )
    if result and isinstance(result[0], float):
        return [result]
    return result

async def embed_single(text: str) -> list[float]:
    vecs = await embed_texts([text])
    return vecs[0]

# ----------------------------------------------------------------
# 2. ChatQA generation  (now uses /v1/chat/completions)
# ----------------------------------------------------------------
async def chatqa_generate(
    context:        str,
    question:       str,
    max_new_tokens: int   = 512,
    temperature:    float = 0.1,
) -> str:
    """
    Generate an answer using Llama-3.1-8B-Instruct via chat completions.
    When context is provided (RAG mode), it's injected as a system message.
    When context is empty (classification/extraction), system message is minimal.
    """
    if context.strip():
        messages = [
            {
                "role":    "system",
                "content": (
                    "You are a helpful assistant that answers questions based "
                    "strictly on the provided context. "
                    "If the context does not contain enough information, say so.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
        ]

    return await _chat_complete(
        model=CHATQA_MODEL,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

# ----------------------------------------------------------------
# 3. CLaRa compression  (now uses /v1/chat/completions)
# ----------------------------------------------------------------
async def clara_compress(
    context:           str,
    query:             str,
    compression_ratio: float = 0.25,
) -> str:
    """
    Compress a retrieved context window using Llama-3.2-3B-Instruct.
    Targets compression_ratio of original word count (0.25 = 4x compression).
    """
    target_words = max(int(len(context.split()) * compression_ratio), 32)
    messages = [
        {
            "role":    "system",
            "content": (
                "You are a text compression assistant. "
                "Compress the provided context to answer the query. "
                f"Target length: {target_words} words. "
                "Preserve all facts relevant to the query. "
                "Output only the compressed text, nothing else."
            ),
        },
        {
            "role":    "user",
            "content": f"Query: {query}\n\nContext to compress:\n{context}",
        },
    ]
    return await _chat_complete(
        model=CLARA_MODEL,
        messages=messages,
        max_new_tokens=int(target_words * 1.5),
        temperature=0.0,
    )

# ----------------------------------------------------------------
# 4. CLaRa memory encoding  (now uses /v1/chat/completions)
# ----------------------------------------------------------------
async def clara_encode_memory(document_chunk: str) -> str:
    """
    Encode a document chunk into a dense memory representation
    for storage in the CLaRa Memory Index in Qdrant.
    """
    messages = [
        {
            "role":    "system",
            "content": (
                "You are a memory encoder. Encode the following text into a "
                "dense memory representation that preserves all key facts, "
                "entities, and relationships. Output only the encoded text."
            ),
        },
        {"role": "user", "content": document_chunk},
    ]
    return await _chat_complete(
        model=CLARA_MODEL,
        messages=messages,
        max_new_tokens=256,
        temperature=0.0,
    )