#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PORT="${1:-5000}"
echo "Iniciando interface web na porta $PORT..."
echo "Acesse: http://localhost:$PORT"
.venv/bin/python src/app.py
