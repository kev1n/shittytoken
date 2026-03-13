#!/usr/bin/env bash
set -euo pipefail

echo "=== ShittyToken Dev Setup ==="
echo ""

# Check prerequisites
command -v uv >/dev/null 2>&1 || { echo "Error: uv not found. Install from https://docs.astral.sh/uv/"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Error: docker not found."; exit 1; }

# Install Python dependencies
echo "[1/4] Installing dependencies..."
uv pip install -e ".[dev]"

# Create .env from example if missing
if [ ! -f .env ]; then
    echo "[2/4] Creating .env from .env.example..."
    cp .env.example .env
    echo "  Edit .env with your API keys before running."
else
    echo "[2/4] .env already exists, skipping."
fi

# Start infrastructure
echo "[3/4] Starting Docker services..."
make up

# Wait for health
echo "[4/4] Waiting for services..."
for i in $(seq 1 30); do
    if docker compose -f docker/docker-compose.yml exec -T redis redis-cli ping > /dev/null 2>&1 && \
       docker compose -f docker/docker-compose.yml exec -T postgres pg_isready -U shittytoken > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo ""
echo "=== Ready! ==="
echo ""
echo "  make seed    — Initialize knowledge graph"
echo "  make test    — Run tests"
echo "  make run     — Start orchestrator"
echo "  make router  — Start API gateway"
echo "  make web     — Start web UI"
echo ""
