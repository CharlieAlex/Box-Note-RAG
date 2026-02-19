venv:
	uv sync --extra dev

ingest:
	uv run python scripts/ingest.py --data-path $(DATA_PATH)

db-stats:
	uv run python scripts/db_ops.py stats

db-search:
	uv run python scripts/db_ops.py search $(QUERY)

db-sources:
	uv run python scripts/db_ops.py list-sources
