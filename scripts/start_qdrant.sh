#!/bin/bash
set -e
echo "Starting Qdrant via Docker..."
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
echo "Qdrant available at http://localhost:6333"
