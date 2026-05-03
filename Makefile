.PHONY: up down test lint typecheck eval deploy-k8s clean install spacy-model help

# ── Variables ─────────────────────────────────────────────────────────────────────
PYTHON     := python3
PIP        := pip3
PYTEST     := pytest
COMPOSE    := docker compose -f infra/docker/docker-compose.yml
KUBECTL    := kubectl

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Installation ─────────────────────────────────────────────────────────────────

install: ## Install Python dependencies
	$(PIP) install -r requirements.txt

spacy-model: ## Download spaCy en_core_web_sm model
	$(PYTHON) -m spacy download en_core_web_sm

setup: install spacy-model ## Full local setup

# ── Infrastructure ────────────────────────────────────────────────────────────────

up: ## Start Neo4j + ChromaDB via docker compose
	$(COMPOSE) up -d
	@echo "Neo4j browser: http://localhost:7474 (neo4j/nexusrag123)"
	@echo "ChromaDB: http://localhost:8001"

down: ## Stop and remove containers
	$(COMPOSE) down

logs: ## Show container logs
	$(COMPOSE) logs -f

ps: ## Show running containers
	$(COMPOSE) ps

# ── API ───────────────────────────────────────────────────────────────────────────

run: ## Run API with hot reload (development)
	ANONYMIZED_TELEMETRY=False $(PYTHON) -m uvicorn src.api.main:app \
		--host 0.0.0.0 --port 8000 \
		--reload --reload-dir src --log-level info

run-prod: ## Run without reload (production-like)
	$(PYTHON) -m uvicorn src.api.main:app \
		--host 0.0.0.0 --port 8000 --workers 1 --log-level info

# ── Testing ───────────────────────────────────────────────────────────────────────

test: ## Run all unit + integration tests (CI-safe, no Ollama required)
	$(PYTEST) tests/unit tests/integration -v --tb=short \
		--cov=src --cov-report=term-missing --cov-report=xml:coverage.xml

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit -v --tb=short

test-integration: ## Run integration tests
	$(PYTEST) tests/integration -v --tb=short

test-e2e: ## Run E2E tests (requires Ollama running)
	OLLAMA_AVAILABLE=1 $(PYTEST) tests/e2e -v --tb=short

test-all: ## Run all tests including E2E
	OLLAMA_AVAILABLE=1 $(PYTEST) tests/ -v --tb=short \
		--cov=src --cov-report=term-missing

# ── Code quality ─────────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	ruff check src/ tests/ scripts/

lint-fix: ## Run ruff linter with auto-fix
	ruff check --fix src/ tests/ scripts/

typecheck: ## Run mypy type checker
	mypy src/ --ignore-missing-imports

format: ## Format code with ruff
	ruff format src/ tests/ scripts/

check: lint typecheck ## Run lint + typecheck

# ── Evaluation ────────────────────────────────────────────────────────────────────

eval: ## Run benchmark evaluation (requires running API)
	$(PYTHON) -c "\
import asyncio; \
from src.evaluation.benchmark import BenchmarkRunner; \
from src.evaluation.report_generator import ReportGenerator; \
async def main(): \
    runner = BenchmarkRunner(use_llm_judge=True); \
    report = await runner.run(); \
    gen = ReportGenerator(); \
    md, charts = gen.generate(report.__dict__, report.run_id); \
    print(f'Report: {md}'); \
    print(f'Charts: {charts}'); \
asyncio.run(main())"

eval-deterministic: ## Run deterministic-only eval (CI-safe)
	$(PYTHON) -c "\
import asyncio; \
from src.evaluation.benchmark import BenchmarkRunner; \
async def main(): \
    runner = BenchmarkRunner(use_llm_judge=False); \
    report = await runner.run(); \
    print(f'precision={report.mean_context_precision:.4f}'); \
    print(f'recall={report.mean_context_recall:.4f}'); \
    print(f'p95_latency={report.latency[\"p95\"]}ms'); \
    assert report.mean_context_precision >= 0.0, 'precision gate'; \
    print('All gates passed.'); \
asyncio.run(main())"

benchmark-create: ## Generate benchmark test cases from docs in data/raw/
	$(PYTHON) scripts/create_benchmark.py --docs data/raw --output data/benchmarks/test.json

# ── Ingestion ─────────────────────────────────────────────────────────────────────

ingest: ## Ingest all docs in data/raw/
	PYTHONPATH=$(PWD) $(PYTHON) scripts/run_ingest.py

# ── Docker ────────────────────────────────────────────────────────────────────────

docker-build: ## Build the API Docker image
	docker build -f infra/docker/Dockerfile.api -t nexus-rag-api:latest .

docker-run: ## Run the API in Docker (after docker-build)
	docker run --rm -p 8000:8000 \
		--env-file .env \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/reports:/app/reports \
		nexus-rag-api:latest

# ── Kubernetes ────────────────────────────────────────────────────────────────────

deploy-k8s: ## Deploy to Kubernetes
	$(KUBECTL) apply -f infra/k8s/

k8s-status: ## Check K8s deployment status
	$(KUBECTL) get pods -n nexus-rag

k8s-logs: ## Tail K8s pod logs
	$(KUBECTL) logs -f -l app=nexus-rag -n nexus-rag

# ── Monitoring ────────────────────────────────────────────────────────────────────

traces: ## View last 20 traces
	$(PYTHON) -c "\
from src.monitoring.tracer import get_tracer; \
import json; \
traces = get_tracer().read_traces(20); \
for t in traces: print(json.dumps({'id': t['trace_id'], 'query': t['query'][:50], 'status': t['status'], 'ms': t['total_duration_ms']}, indent=2))"

metrics: ## Show Prometheus metrics
	curl -s http://localhost:8000/metrics | grep nexus_rag

# ── Cleanup ───────────────────────────────────────────────────────────────────────

clean: ## Clean generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache coverage.xml .coverage
	rm -rf reports/traces/*.jsonl reports/charts/*.png reports/eval_runs/*.json 2>/dev/null || true
	@echo "Clean complete."

clean-data: ## Remove all vector/graph data (DESTRUCTIVE)
	rm -rf data/chroma data/cache
	@echo "WARNING: Vector store cleared. Re-ingest required."

ci-check: lint typecheck test ## Full CI check (lint + types + tests)
	@echo "CI checks passed ✓"