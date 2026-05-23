#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "Criando ambiente virtual Python..."
python3 -m venv .venv

echo "Instalando dependências do projeto..."
.venv/bin/pip install -r requirements.txt -q

echo ""
echo "Dependências instaladas com sucesso!"
echo "Para ativar o ambiente virtual: source .venv/bin/activate"
