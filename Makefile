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

test:
	uv run pytest

test-unit:
	uv run pytest -m unit

test-html:
	uv run pytest --html=docs/reports/test_report.html --self-contained-html
