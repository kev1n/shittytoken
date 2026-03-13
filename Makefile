.PHONY: install dev test test-all seed run up down logs clean fmt lint check

# Package management (uv only)
install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"
	uv pip install -e ".[web]"

# Testing
test:
	python -m pytest tests/ --ignore=tests/benchmark/test_full_benchmark.py --ignore=tests/knowledge/ -q

test-all:
	python -m pytest tests/ -q

# Dev infrastructure
up:
	docker compose -f docker/docker-compose.yml up -d
	@echo "Waiting for services to be healthy..."
	@docker compose -f docker/docker-compose.yml exec -T neo4j cypher-shell -u neo4j -p shittytoken_dev "RETURN 1" > /dev/null 2>&1 && echo "  Neo4j: ready" || echo "  Neo4j: starting..."
	@docker compose -f docker/docker-compose.yml exec -T redis redis-cli ping > /dev/null 2>&1 && echo "  Redis: ready" || echo "  Redis: starting..."
	@docker compose -f docker/docker-compose.yml exec -T postgres pg_isready -U shittytoken > /dev/null 2>&1 && echo "  PostgreSQL: ready" || echo "  PostgreSQL: starting..."
	@echo ""
	@echo "Services:"
	@echo "  Neo4j Browser:  http://localhost:7474"
	@echo "  Redis:          redis://localhost:6379"
	@echo "  PostgreSQL:     postgresql://shittytoken:shittytoken_dev@localhost:5432/shittytoken"
	@echo "  Mock Worker:    http://localhost:8000"
	@echo ""
	@echo "Run 'make seed' to initialize the knowledge graph."

down:
	docker compose -f docker/docker-compose.yml down

down-clean:
	docker compose -f docker/docker-compose.yml down -v

logs:
	docker compose -f docker/docker-compose.yml logs -f

# Application
seed:
	python -m shittytoken seed

run:
	python -m shittytoken run

router:
	shittytoken-router

web:
	shittytoken-web

# Code quality
fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

lint:
	ruff check src/ tests/

check: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
