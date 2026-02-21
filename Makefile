venv:
	uv sync --extra dev

ingest:
	uv run python scripts/ingest.py --data-path $(DATA_PATH)

run:
	uv run python main.py

db-stats:
	uv run python scripts/db_ops.py stats

db-search:
	uv run python scripts/db_ops.py search $(QUERY)

db-sources:
	uv run python scripts/db_ops.py list-sources

ruff-check:
	uv run ruff check

test:
	make ruff-check && uv run pytest

test-unit:
	make ruff-check && uv run pytest -m "not integration"

test-integration:
	make ruff-check && uv run pytest -m "integration"

test-html:
	make ruff-check && uv run pytest --html=docs/reports/test_report.html --self-contained-html

show-graph:
	uv run python scripts/show_graph.py