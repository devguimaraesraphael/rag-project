#!/bin/bash
set -e
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

ok()   { echo -e "${GREEN}  ✔  $1${RESET}"; }
info() { echo -e "${CYAN}  ▶  $1${RESET}"; }
warn() { echo -e "${YELLOW}  ⚠  $1${RESET}"; }
fail() { echo -e "${RED}  ✘  $1${RESET}"; exit 1; }
step() { echo -e "\n${BOLD}${CYAN}═══ $1 ${RESET}"; }

echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════╗"
echo -e "║   RAG Project — Startup             ║"
echo -e "╚══════════════════════════════════════╝${RESET}\n"

# ─── 1. Python ───────────────────────────────────────────────────────────────
step "1/5  Python Environment"

if [ ! -d ".venv" ]; then
  info "Creating virtual environment..."
  python3 -m venv .venv || fail "Failed to create venv. Install python3-venv: sudo apt install python3.12-venv"
  ok "Venv created"
else
  ok "Venv already exists"
fi

info "Installing dependencies..."
.venv/bin/pip install -r requirements.txt -q \
  && .venv/bin/pip install pytest -q \
  && ok "Dependencies installed" \
  || fail "Error installing dependencies"

# ─── 2. Qdrant ───────────────────────────────────────────────────────────────
step "2/5  Vector Database (Qdrant)"

if ! command -v docker &>/dev/null; then
  warn "Docker not found. Qdrant will not be started automatically."
  warn "Install Docker or start Qdrant manually before using the interface."
else
  if docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
    ok "Qdrant is already running"
  elif docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
    info "Restarting existing Qdrant container..."
    docker start qdrant > /dev/null
    ok "Qdrant started"
  else
    info "Pulling and starting Qdrant..."
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant > /dev/null
    ok "Qdrant started"
  fi

  # Wait for Qdrant to be ready
  info "Waiting for Qdrant to be ready..."
  for i in $(seq 1 15); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
      ok "Qdrant responding at http://localhost:6333"
      break
    fi
    sleep 1
    if [ "$i" -eq 15 ]; then
      warn "Qdrant took longer than expected. Continue anyway."
    fi
  done
fi

# ─── 3. Tests ────────────────────────────────────────────────────────────────
step "3/5  Unit Tests"

info "Running tests..."
if .venv/bin/python -m pytest tests/test_rag.py -v --tb=short 2>&1 | tee /tmp/rag_test_output.txt | grep -E "PASSED|FAILED|ERROR|passed|failed|error" ; then
  if grep -q "failed\|error" /tmp/rag_test_output.txt; then
    fail "Some tests failed. Check the errors above before continuing."
  else
    ok "All tests passed"
  fi
else
  fail "Error running tests."
fi

# ─── 4. Interface Web ─────────────────────────────────────────────────────────
step "4/5  Web Interface"

PORT="${PORT:-5000}"

# Kill previous process if exists
OLD_PID=$(lsof -ti tcp:$PORT 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
  info "Killing previous process on port $PORT (PID $OLD_PID)..."
  kill "$OLD_PID" 2>/dev/null || true
  sleep 1
fi

info "Starting Flask server on port $PORT..."
PORT=$PORT .venv/bin/python src/app.py > /tmp/rag_web.log 2>&1 &
WEB_PID=$!
echo $WEB_PID > /tmp/rag_web.pid

# Wait for server to start
for i in $(seq 1 10); do
  if curl -sf http://localhost:$PORT/ > /dev/null 2>&1; then
    ok "Web server running (PID $WEB_PID)"
    break
  fi
  sleep 1
  if [ "$i" -eq 10 ]; then
    echo ""
    warn "Server took too long to respond. Check: tail -f /tmp/rag_web.log"
  fi
done

# ─── 5. Abrir browser ─────────────────────────────────────────────────────────
step "5/5  Opening Interface"

URL="http://localhost:$PORT"
info "Opening $URL in browser..."

if command -v xdg-open &>/dev/null; then
  xdg-open "$URL" &>/dev/null &
elif command -v open &>/dev/null; then
  open "$URL"
else
  warn "Could not open browser automatically."
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}╔══════════════════════════════════════╗"
echo -e "║   All set!                          ║"
echo -e "╚══════════════════════════════════════╝${RESET}"
echo -e "\n  Web Interface : ${CYAN}${URL}${RESET}"
echo -e "  Qdrant UI    : ${CYAN}http://localhost:6333/dashboard${RESET}"
echo -e "  Web logs     : tail -f /tmp/rag_web.log"
echo -e "  Stop all     : bash scripts/stop.sh\n"

# Keep script active and show web logs
trap "kill $WEB_PID 2>/dev/null; echo -e '\n${YELLOW}Web server stopped.${RESET}\n'; exit 0" INT TERM
wait $WEB_PID
