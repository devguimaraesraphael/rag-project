#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PORT="${1:-5000}"
echo "Starting web interface on port $PORT..."
echo "Access: http://localhost:$PORT"
.venv/bin/python src/app.py
