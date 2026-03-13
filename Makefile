.PHONY: install dev test seed run up down logs clean

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

test:
	python -m pytest tests/ --ignore=tests/benchmark/test_full_benchmark.py --ignore=tests/knowledge/ -q

test-all:
	python -m pytest tests/ -q

seed:
	python -m shittytoken seed

run:
	python -m shittytoken run

up:
	docker compose -f docker/docker-compose.yml up -d neo4j

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f neo4j

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
