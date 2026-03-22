#!/usr/bin/env bash
# ============================================================
# RAG Prototype — Bootstrap Script
# Usage: ./scripts/bootstrap.sh [--expose]
#
# --expose   : tunnel port 8000 to the public internet via ngrok
#              so recruiters/interviewers can hit your live demo
# ============================================================
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"

log()  { echo -e "${GREEN}[bootstrap]${RESET} $*"; }
warn() { echo -e "${YELLOW}[warn]${RESET} $*"; }
die()  { echo -e "${RED}[error]${RESET} $*"; exit 1; }

# ----------------------------------------------------------------
# 0. Pre-flight checks
# ----------------------------------------------------------------
log "Checking dependencies..."
command -v docker   >/dev/null 2>&1 || die "Docker not found. Install from https://docs.docker.com/get-docker/"
command -v docker   >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 || \
  die "Docker Compose v2 not found. Update Docker Desktop."

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        warn ".env not found — copied from .env.example"
        warn "Edit .env and set HF_API_TOKEN before continuing."
        echo ""
        echo -e "  Get your free token at: ${BOLD}https://huggingface.co/settings/tokens${RESET}"
        echo ""
        read -p "Press ENTER after you've set HF_API_TOKEN in .env..."
    else
        die ".env.example not found. Are you in the project root?"
    fi
fi

# Validate token is set
source .env
if [ -z "${HF_API_TOKEN:-}" ] || [ "$HF_API_TOKEN" = "hf_your_token_here" ]; then
    die "HF_API_TOKEN is not set in .env. Get your free token at https://huggingface.co/settings/tokens"
fi

# ----------------------------------------------------------------
# 1. Create data directories
# ----------------------------------------------------------------
log "Creating data directories..."
mkdir -p data/raw data/processed

# ----------------------------------------------------------------
# 2. Pull and start all services
# ----------------------------------------------------------------
log "Starting all services with Docker Compose..."
docker compose pull --quiet
docker compose up -d --build

# ----------------------------------------------------------------
# 3. Wait for health checks
# ----------------------------------------------------------------
log "Waiting for services to be healthy..."

wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            log "  ✓ $name is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 3
    done
    warn "  ✗ $name did not become healthy in time (may still be starting)"
    return 1
}

wait_for_service "Qdrant"         "http://localhost:6333/healthz"
wait_for_service "Elasticsearch"  "http://localhost:9200/_cluster/health"
wait_for_service "PostgreSQL"     "http://localhost:5432" || true  # TCP check
wait_for_service "Redis"          "http://localhost:6379" || true  # TCP check
wait_for_service "Retrieval"      "http://localhost:8002/health"
wait_for_service "Compression"    "http://localhost:8003/health"
wait_for_service "Orchestrator"   "http://localhost:8000/health"

# ----------------------------------------------------------------
# 4. Verify with a quick smoke test
# ----------------------------------------------------------------
log "Running smoke test..."
HEALTH=$(curl -sf http://localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])" 2>/dev/null || echo "unreachable")
if [ "$HEALTH" = "healthy" ] || [ "$HEALTH" = "degraded" ]; then
    log "  ✓ Orchestrator health: $HEALTH"
else
    warn "  Orchestrator not responding yet — services may still be warming up"
fi

# ----------------------------------------------------------------
# 5. Optionally expose via ngrok
# ----------------------------------------------------------------
if [[ "${1:-}" == "--expose" ]]; then
    log "Setting up ngrok tunnel..."
    if ! command -v ngrok >/dev/null 2>&1; then
        warn "ngrok not found. Install from https://ngrok.com/download (free account)"
        warn "Then re-run: ./scripts/bootstrap.sh --expose"
    else
        ngrok http 8000 &
        sleep 3
        NGROK_URL=$(curl -sf http://localhost:4040/api/tunnels | \
            python3 -c "import sys,json; t=json.load(sys.stdin)['tunnels']; print([x for x in t if x['proto']=='https'][0]['public_url'])" 2>/dev/null || echo "")
        if [ -n "$NGROK_URL" ]; then
            log ""
            log "  ╔══════════════════════════════════════════════════╗"
            log "  ║  Public URL: ${BOLD}$NGROK_URL${RESET}"
            log "  ║  API Docs:   ${BOLD}$NGROK_URL/docs${RESET}"
            log "  ╚══════════════════════════════════════════════════╝"
            log ""
        fi
    fi
fi

# ----------------------------------------------------------------
# 6. Print summary
# ----------------------------------------------------------------
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║        RAG System is running locally         ║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║${RESET}  API (orchestrator)   http://localhost:8000  ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  API Docs (Swagger)   http://localhost:8000/docs ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Grafana dashboard    http://localhost:3000  ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Prometheus metrics   http://localhost:9090  ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Qdrant UI            http://localhost:6333  ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Neo4j browser        http://localhost:7474  ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Elasticsearch        http://localhost:9200  ${BOLD}║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║${RESET}  Ingest a doc:                               ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  curl -X POST http://localhost:8000/ingest   ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}    -d '{\"filepath\":\"/app/data/raw/doc.txt\"}' ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}                                              ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Run a query:                                ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  curl -X POST http://localhost:8000/query    ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}    -d '{\"query\":\"What is X?\",               ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}         \"session_id\":\"demo\"}'               ${BOLD}║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║${RESET}  Run tests:                                  ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  python tests/test_e2e.py                    ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}                                              ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Stop everything:                            ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  docker compose down -v                      ${BOLD}║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"