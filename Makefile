.PHONY: setup train serve test lint clean docker-up docker-down dbt-run

# ── Setup ──────────────────────────────────────────────────────────────
setup:
	pip install -e ".[dev,tracking]"
	pre-commit install
	@echo "✓ Environment ready"

# ── Data ───────────────────────────────────────────────────────────────
extract:
	python -m src.data.bigquery_extract

dbt-run:
	cd dbt && dbt run

dbt-test:
	cd dbt && dbt test

sync-features:
	python -m src.data.clickhouse_sync

# ── ML ─────────────────────────────────────────────────────────────────
cluster:
	python -m src.models.clustering

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluation

# ── Serving ────────────────────────────────────────────────────────────
serve-api:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-dashboard:
	streamlit run src/serving/streamlit_app.py --server.port 8501

serve-mcp:
	cd mcp_server && npm start

# ── Quality ────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v --tb=short

# ── Docker ─────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# ── Clean ──────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info
