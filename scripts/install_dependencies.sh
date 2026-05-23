#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "Creating Python virtual environment..."
python3 -m venv .venv

echo "Installing project dependencies..."
.venv/bin/pip install -r requirements.txt -q

echo ""
echo "Dependencies installed successfully!"
echo "To activate the virtual environment: source .venv/bin/activate"
