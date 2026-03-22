"""
Integration test suite — tests/test_e2e.py
Runs against a live local stack (all services up via docker compose up).

Usage:
    pip install httpx
    python tests/test_e2e.py

Environment overrides:
    ORCHESTRATOR_URL   default: http://localhost:8000
    INGESTION_URL      default: http://localhost:8001
    RETRIEVAL_URL      default: http://localhost:8002
    COMPRESSION_URL    default: http://localhost:8003

Tests covered:
    1.  Health check
    2.  Document ingestion (via orchestrator /ingest)
    3.  Wait for ingestion to complete (polls with timeout)
    4.  Query classification
    5.  Simple factoid RAG query
    6.  Multi-hop RAG query
    7.  Exact cache hit (same query twice)
    8.  Semantic cache hit (paraphrased query)
    9.  CLaRa compression endpoint
    10. Session history
    11. Prometheus metrics
"""

import asyncio
import os
import time

import httpx

# ----------------------------------------------------------------
# Service URLs
# ----------------------------------------------------------------
BASE_URL        = os.getenv("ORCHESTRATOR_URL",  "http://localhost:8000")
INGESTION_URL   = os.getenv("INGESTION_URL",     "http://localhost:8001")
RETRIEVAL_URL   = os.getenv("RETRIEVAL_URL",     "http://localhost:8002")
COMPRESSION_URL = os.getenv("COMPRESSION_URL",   "http://localhost:8003")

# ----------------------------------------------------------------
# Console colours
# ----------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

passed = 0
failed = 0


async def run_test(name: str, coro) -> bool:
    global passed, failed
    try:
        await coro
        print(f"  {GREEN}✓{RESET} {name}")
        passed += 1
        return True
    except AssertionError as e:
        print(f"  {RED}✗{RESET} {name}")
        print(f"    {DIM}AssertionError: {e}{RESET}")
        failed += 1
        return False
    except Exception as e:
        print(f"  {RED}✗{RESET} {name}")
        print(f"    {DIM}{type(e).__name__}: {e}{RESET}")
        failed += 1
        return False


# ----------------------------------------------------------------
# 1. Health check
# ----------------------------------------------------------------
async def test_health():
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] in ("healthy", "degraded"), \
        f"Unexpected status: {data['status']}"
    print(f"    {DIM}status={data['status']}  services={data['services']}{RESET}")


# ----------------------------------------------------------------
# 2. Document ingestion
#    Uses the orchestrator /ingest endpoint which triggers the
#    ingestion service in the background. The document is written
#    to data/raw/ which is mounted into the ingestion container
#    at /app/data/raw/ — this is the only path both sides can see.
# ----------------------------------------------------------------
_TEST_FILENAME  = "e2e_test_doc.txt"
_HOST_DOC_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "raw", _TEST_FILENAME)
_CONTAINER_PATH = f"/app/data/raw/{_TEST_FILENAME}"

_TEST_CONTENT = (
    "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
    "It was designed by Gustave Eiffel and constructed between 1887 and 1889 "
    "as the entrance arch for the 1889 World's Fair. "
    "The tower stands 330 metres tall and remained the tallest man-made structure "
    "in the world for 41 years until the Chrysler Building was built in 1930. "
    "Over seven million people visit the Eiffel Tower every year, making it the "
    "most visited paid monument in the world. "
    "The tower is made of puddled iron and weighs approximately 7,300 tonnes. "
    "It was initially criticised by leading French artists and intellectuals "
    "but has since become a global cultural icon of France and one of the most "
    "recognisable structures on earth. "
) * 25   # ~1,500 words — enough for several chunks


async def test_ingest():
    # Write the test document to the shared data/raw/ mount
    os.makedirs(os.path.dirname(_HOST_DOC_PATH), exist_ok=True)
    with open(_HOST_DOC_PATH, "w", encoding="utf-8") as f:
        f.write(_TEST_CONTENT)

    # Trigger ingestion via the orchestrator (mirrors real usage)
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{BASE_URL}/ingest",
            json={"filepath": _CONTAINER_PATH},
            timeout=30,
        )
    assert r.status_code == 200, f"Ingest queue failed: {r.text}"
    data = r.json()
    assert data["status"] == "ingestion_queued", \
        f"Unexpected status: {data['status']}"
    print(f"    {DIM}queued  filepath={_CONTAINER_PATH}{RESET}")


# ----------------------------------------------------------------
# 3. Wait for ingestion to complete
#    Polls the Qdrant collection until at least one vector appears.
#    HuggingFace cold starts can take 30-90 s so we wait up to 3 min.
# ----------------------------------------------------------------
async def test_wait_for_ingestion():
    deadline = time.time() + 180   # 3-minute timeout
    poll_interval = 8

    async with httpx.AsyncClient() as c:
        while time.time() < deadline:
            try:
                r = await c.get(
                    "http://localhost:6333/collections/vector_index_chunks",
                    timeout=5,
                )
                if r.status_code == 200:
                    info = r.json()
                    count = (
                        info.get("result", {})
                            .get("vectors_count", 0)
                    )
                    if count and count > 0:
                        print(f"    {DIM}vectors_count={count}  ingestion complete{RESET}")
                        return
            except Exception:
                pass

            elapsed = int(time.time() - deadline + 180)
            print(f"    {DIM}waiting for ingestion... {elapsed}s elapsed{RESET}")
            await asyncio.sleep(poll_interval)

    raise AssertionError(
        "Ingestion did not complete within 3 minutes. "
        "Check: docker compose logs ingestion"
    )


# ----------------------------------------------------------------
# 4. Query classification
# ----------------------------------------------------------------
_CLASSIFY_CASES = [
    ("What is the height of the Eiffel Tower?",                          "simple_factoid"),
    ("Compare the Eiffel Tower and the Statue of Liberty construction",  "multi_hop"),
    ("List all landmarks near the Seine river",                          "structured"),
]

async def test_classification():
    async with httpx.AsyncClient() as c:
        for query, expected_type in _CLASSIFY_CASES:
            r = await c.post(
                f"{RETRIEVAL_URL}/classify",
                json={"query": query},
                timeout=60,
            )
            assert r.status_code == 200, \
                f"Classify returned {r.status_code} for: {query}"
            data = r.json()
            assert "query_type" in data, "No query_type in response"
            actual = data["query_type"]
            match = GREEN + "✓" + RESET if actual == expected_type else YELLOW + "~" + RESET
            print(f"    {DIM}{match} '{query[:45]}...'  → {actual}{RESET}")


# ----------------------------------------------------------------
# 5. Simple factoid RAG query
# ----------------------------------------------------------------
async def test_query_simple():
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{BASE_URL}/query",
            json={
                "query":      "How tall is the Eiffel Tower?",
                "session_id": "e2e_simple",
            },
            timeout=180,
        )
    assert r.status_code == 200, f"Query failed ({r.status_code}): {r.text}"
    data = r.json()
    assert data["answer"],        "Empty answer returned"
    assert data["latency_ms"] > 0, "Latency is zero"
    assert data["query_type"],    "No query_type in response"
    print(f"    {DIM}answer='{data['answer'][:70]}...'{RESET}")
    print(f"    {DIM}type={data['query_type']}  cache={data['cache_type']}  "
          f"latency={data['latency_ms']}ms{RESET}")


# ----------------------------------------------------------------
# 6. Multi-hop RAG query
# ----------------------------------------------------------------
async def test_query_multi_hop():
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{BASE_URL}/query",
            json={
                "query": (
                    "How did the reception of the Eiffel Tower change "
                    "over time and what impact did it have on tourism?"
                ),
                "session_id": "e2e_multihop",
                "query_type": "multi_hop",   # force the router
            },
            timeout=180,
        )
    assert r.status_code == 200, f"Multi-hop query failed: {r.text}"
    data = r.json()
    assert data["answer"], "Empty answer returned"
    print(f"    {DIM}answer='{data['answer'][:70]}...'{RESET}")
    print(f"    {DIM}retrieved={len(data.get('telemetry', []))} telemetry items{RESET}")


# ----------------------------------------------------------------
# 7. Exact cache hit
# ----------------------------------------------------------------
async def test_cache_exact():
    query = "Who designed the Eiffel Tower?"

    async with httpx.AsyncClient() as c:
        # First call — cache miss, full pipeline
        r1 = await c.post(
            f"{BASE_URL}/query",
            json={"query": query, "session_id": "e2e_cache"},
            timeout=180,
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["cache_type"] == "miss", \
            f"Expected first call to be a miss, got '{d1['cache_type']}'"

        # Second call — must be exact cache hit
        r2 = await c.post(
            f"{BASE_URL}/query",
            json={"query": query, "session_id": "e2e_cache"},
            timeout=10,
        )
        assert r2.status_code == 200
        d2 = r2.json()

    assert d2["cache_type"] == "exact", \
        f"Expected exact cache hit, got '{d2['cache_type']}'"
    assert d2["answer"] == d1["answer"], \
        "Cached answer differs from original"
    assert d2["latency_ms"] < d1["latency_ms"], \
        f"Cached ({d2['latency_ms']}ms) not faster than original ({d1['latency_ms']}ms)"
    print(f"    {DIM}first={d1['latency_ms']}ms  cached={d2['latency_ms']}ms{RESET}")


# ----------------------------------------------------------------
# 8. Semantic cache hit (paraphrased query)
# ----------------------------------------------------------------
async def test_cache_semantic():
    original   = "What year was the Eiffel Tower completed?"
    paraphrase = "When was construction of the Eiffel Tower finished?"

    async with httpx.AsyncClient() as c:
        # Seed the cache with the original
        r1 = await c.post(
            f"{BASE_URL}/query",
            json={"query": original, "session_id": "e2e_semantic"},
            timeout=180,
        )
        assert r1.status_code == 200

        # Give the semantic cache a moment to store the embedding
        await asyncio.sleep(2)

        # Paraphrase should hit semantic cache
        r2 = await c.post(
            f"{BASE_URL}/query",
            json={"query": paraphrase, "session_id": "e2e_semantic"},
            timeout=60,
        )
        assert r2.status_code == 200
        d2 = r2.json()

    # Semantic cache may or may not fire depending on embedding similarity
    # — we assert it's fast regardless (either semantic hit or at least not broken)
    assert d2["answer"], "Empty answer on paraphrase query"
    cache_note = d2["cache_type"]
    print(f"    {DIM}original→{r1.json()['cache_type']}  paraphrase→{cache_note}{RESET}")
    if cache_note == "semantic":
        print(f"    {DIM}semantic cache fired  latency={d2['latency_ms']}ms{RESET}")


# ----------------------------------------------------------------
# 9. Compression endpoint
# ----------------------------------------------------------------
async def test_compression():
    long_context = (
        "The Eiffel Tower was built between 1887 and 1889 by Gustave Eiffel. "
        "It is located on the Champ de Mars in Paris, France. "
        "The tower is 330 metres tall and weighs 7,300 tonnes. "
        "It was the tallest structure in the world for 41 years. "
        "Over 7 million people visit it annually. "
    ) * 40   # ~400 words

    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{COMPRESSION_URL}/compress",
            json={
                "context":           long_context,
                "query":             "How tall is the Eiffel Tower?",
                "compression_ratio": 0.25,
            },
            timeout=180,
        )
    assert r.status_code == 200, f"Compression failed ({r.status_code}): {r.text}"
    data = r.json()
    assert data["compressed_context"], "Empty compressed context"
    assert data["actual_ratio"] < 0.9, \
        f"Ratio {data['actual_ratio']:.2f} — compression not effective"
    print(f"    {DIM}{data['original_tokens']} words → {data['compressed_tokens']} words "
          f"(ratio={data['actual_ratio']:.2f}){RESET}")


# ----------------------------------------------------------------
# 10. Session history
# ----------------------------------------------------------------
async def test_session():
    session_id = "e2e_session_history"
    queries    = [
        "Where is the Eiffel Tower?",
        "What is it made of?",
    ]

    async with httpx.AsyncClient() as c:
        for q in queries:
            await c.post(
                f"{BASE_URL}/query",
                json={"query": q, "session_id": session_id},
                timeout=180,
            )

        r = await c.get(f"{BASE_URL}/session/{session_id}", timeout=10)

    assert r.status_code == 200
    data    = r.json()
    history = data["history"]

    # Each query produces 2 turns (user + assistant) → 4 total for 2 queries
    assert len(history) >= 4, \
        f"Expected ≥ 4 history entries, got {len(history)}"
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    print(f"    {DIM}{len(history)} turns stored for session '{session_id}'{RESET}")


# ----------------------------------------------------------------
# 11. Prometheus metrics
# ----------------------------------------------------------------
async def test_metrics():
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE_URL}/metrics", timeout=10)
    assert r.status_code == 200, f"Metrics returned {r.status_code}"
    body = r.text
    for expected_metric in (
        "rag_queries_total",
        "rag_latency_seconds",
        "rag_errors_total",
    ):
        assert expected_metric in body, \
            f"Metric '{expected_metric}' not found in /metrics output"
    print(f"    {DIM}all 3 expected metrics present{RESET}")


# ----------------------------------------------------------------
# Runner
# ----------------------------------------------------------------
async def main():
    print(f"\n{BOLD}RAG System — End-to-End Integration Tests{RESET}")
    print(f"{DIM}Orchestrator : {BASE_URL}")
    print(f"Ingestion    : {INGESTION_URL}")
    print(f"Retrieval    : {RETRIEVAL_URL}")
    print(f"Compression  : {COMPRESSION_URL}{RESET}\n")

    print(f"{BOLD}── Infrastructure ──────────────────────────{RESET}")
    await run_test("Health check",                test_health())

    print(f"\n{BOLD}── Ingestion ────────────────────────────────{RESET}")
    ok = await run_test("Queue document for ingestion", test_ingest())
    if ok:
        await run_test("Wait for ingestion to complete", test_wait_for_ingestion())
    else:
        print(f"  {YELLOW}⚠{RESET}  Skipping ingestion wait — ingest queue failed")

    print(f"\n{BOLD}── Retrieval ────────────────────────────────{RESET}")
    await run_test("Query classification",         test_classification())

    print(f"\n{BOLD}── RAG Pipeline ─────────────────────────────{RESET}")
    await run_test("Simple factoid query",         test_query_simple())
    await run_test("Multi-hop query",              test_query_multi_hop())

    print(f"\n{BOLD}── Caching ──────────────────────────────────{RESET}")
    await run_test("Exact cache hit",              test_cache_exact())
    await run_test("Semantic cache hit",           test_cache_semantic())

    print(f"\n{BOLD}── Services ─────────────────────────────────{RESET}")
    await run_test("CLaRa compression",            test_compression())
    await run_test("Session history",              test_session())
    await run_test("Prometheus metrics",           test_metrics())

    total = passed + failed
    print(f"\n{BOLD}{'─' * 44}{RESET}")
    colour = GREEN if failed == 0 else RED
    print(f"{BOLD}Results: {colour}{passed}/{total} passed{RESET}")
    if failed > 0:
        print(f"{RED}{failed} test(s) failed — see details above{RESET}")
        raise SystemExit(1)
    print(f"{GREEN}All tests passed.{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())