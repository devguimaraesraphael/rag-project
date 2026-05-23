#!/bin/bash
cd "$(dirname "$0")/.."

echo "Stopping RAG Project..."

# Stop web server
if [ -f /tmp/rag_web.pid ]; then
  PID=$(cat /tmp/rag_web.pid)
  kill "$PID" 2>/dev/null && echo "  ✔  Web server stopped (PID $PID)" || echo "  ⚠  Process $PID was not running"
  rm -f /tmp/rag_web.pid
fi

# Stop any process on port 5000
OLD=$(lsof -ti tcp:5000 2>/dev/null || true)
if [ -n "$OLD" ]; then
  kill $OLD 2>/dev/null && echo "  ✔  Port 5000 freed"
fi

# Stop Qdrant (Docker)
if command -v docker &>/dev/null && docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
  docker stop qdrant > /dev/null && echo "  ✔  Qdrant stopped"
fi

echo "All stopped."
