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
step "1/5  Ambiente Python"

if [ ! -d ".venv" ]; then
  info "Criando ambiente virtual..."
  python3 -m venv .venv || fail "Falha ao criar venv. Instale python3-venv: sudo apt install python3.12-venv"
  ok "Venv criado"
else
  ok "Venv já existe"
fi

info "Instalando dependências..."
.venv/bin/pip install -r requirements.txt -q \
  && .venv/bin/pip install pytest -q \
  && ok "Dependências instaladas" \
  || fail "Erro ao instalar dependências"

# ─── 2. Qdrant ───────────────────────────────────────────────────────────────
step "2/5  Banco Vetorial (Qdrant)"

if ! command -v docker &>/dev/null; then
  warn "Docker não encontrado. Qdrant não será iniciado automaticamente."
  warn "Instale Docker ou inicie o Qdrant manualmente antes de usar a interface."
else
  if docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
    ok "Qdrant já está rodando"
  elif docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
    info "Reiniciando container Qdrant existente..."
    docker start qdrant > /dev/null
    ok "Qdrant iniciado"
  else
    info "Baixando e iniciando Qdrant..."
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant > /dev/null
    ok "Qdrant iniciado"
  fi

  # Aguarda Qdrant estar pronto
  info "Aguardando Qdrant ficar pronto..."
  for i in $(seq 1 15); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
      ok "Qdrant respondendo em http://localhost:6333"
      break
    fi
    sleep 1
    if [ "$i" -eq 15 ]; then
      warn "Qdrant demorou mais que o esperado. Continue mesmo assim."
    fi
  done
fi

# ─── 3. Testes ───────────────────────────────────────────────────────────────
step "3/5  Testes Unitários"

info "Executando testes..."
if .venv/bin/python -m pytest tests/test_rag.py -v --tb=short 2>&1 | tee /tmp/rag_test_output.txt | grep -E "PASSED|FAILED|ERROR|passed|failed|error" ; then
  if grep -q "failed\|error" /tmp/rag_test_output.txt; then
    fail "Alguns testes falharam. Verifique os erros acima antes de continuar."
  else
    ok "Todos os testes passaram"
  fi
else
  fail "Erro ao executar os testes."
fi

# ─── 4. Interface Web ─────────────────────────────────────────────────────────
step "4/5  Interface Web"

PORT="${PORT:-5000}"

# Mata processo anterior se existir
OLD_PID=$(lsof -ti tcp:$PORT 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
  info "Encerrando processo anterior na porta $PORT (PID $OLD_PID)..."
  kill "$OLD_PID" 2>/dev/null || true
  sleep 1
fi

info "Iniciando servidor Flask na porta $PORT..."
PORT=$PORT .venv/bin/python src/app.py > /tmp/rag_web.log 2>&1 &
WEB_PID=$!
echo $WEB_PID > /tmp/rag_web.pid

# Aguarda servidor subir
for i in $(seq 1 10); do
  if curl -sf http://localhost:$PORT/ > /dev/null 2>&1; then
    ok "Servidor web rodando (PID $WEB_PID)"
    break
  fi
  sleep 1
  if [ "$i" -eq 10 ]; then
    echo ""
    warn "Servidor demorou para responder. Verifique: tail -f /tmp/rag_web.log"
  fi
done

# ─── 5. Abrir browser ─────────────────────────────────────────────────────────
step "5/5  Abrindo Interface"

URL="http://localhost:$PORT"
info "Abrindo $URL no browser..."

if command -v xdg-open &>/dev/null; then
  xdg-open "$URL" &>/dev/null &
elif command -v open &>/dev/null; then
  open "$URL"
else
  warn "Não foi possível abrir o browser automaticamente."
fi

# ─── Resumo ───────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}╔══════════════════════════════════════╗"
echo -e "║   Tudo pronto!                       ║"
echo -e "╚══════════════════════════════════════╝${RESET}"
echo -e "\n  Interface Web : ${CYAN}${URL}${RESET}"
echo -e "  Qdrant UI     : ${CYAN}http://localhost:6333/dashboard${RESET}"
echo -e "  Logs web      : tail -f /tmp/rag_web.log"
echo -e "  Parar tudo    : bash scripts/stop.sh\n"

# Mantém script ativo e exibe logs da interface
trap "kill $WEB_PID 2>/dev/null; echo -e '\n${YELLOW}Servidor encerrado.${RESET}'; exit 0" INT TERM
wait $WEB_PID
