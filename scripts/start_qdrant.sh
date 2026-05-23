#!/bin/bash
set -e
echo "Iniciando Qdrant via Docker..."
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
echo "Qdrant disponível em http://localhost:6333"
