#!/bin/bash
cd "$(dirname "$0")/.."

echo "Encerrando RAG Project..."

# Para o servidor web
if [ -f /tmp/rag_web.pid ]; then
  PID=$(cat /tmp/rag_web.pid)
  kill "$PID" 2>/dev/null && echo "  ✔  Servidor web encerrado (PID $PID)" || echo "  ⚠  Processo $PID já não estava rodando"
  rm -f /tmp/rag_web.pid
fi

# Para qualquer processo na porta 5000
OLD=$(lsof -ti tcp:5000 2>/dev/null || true)
if [ -n "$OLD" ]; then
  kill $OLD 2>/dev/null && echo "  ✔  Porta 5000 liberada"
fi

# Para o Qdrant (Docker)
if command -v docker &>/dev/null && docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
  docker stop qdrant > /dev/null && echo "  ✔  Qdrant parado"
fi

echo "Tudo encerrado."
