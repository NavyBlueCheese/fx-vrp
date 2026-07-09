# Every target is a thin wrapper over `uv run`; on Windows without make, run the
# right-hand sides directly.

.PHONY: sync lint type test cov data data-sample scrape rates quality-report figures paper

sync:
	uv sync

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

type:
	uv run mypy --strict src

test:
	uv run pytest

cov:
	uv run pytest --cov=src/fxvrp --cov-report=term-missing --cov-fail-under=85

rates:
	uv run python scripts/fetch_rates.py

scrape:
	uv run python scripts/scrape_chains.py

# bounded, minutes-scale pull: rates + chains + the config sample window of ticks
data-sample: rates scrape
	uv run python scripts/ingest_ticks.py --sample

# the full 2007-2025 tick history; resumable, safe to interrupt and rerun
data: rates scrape
	uv run python scripts/ingest_ticks.py

quality-report:
	uv run python scripts/data_quality_report.py

figures:
	@echo "figures are produced from Phase 2 onwards (scripts/fig_*.py)"

paper:
	@echo "paper build lands in Phase 6 (paper/Makefile)"
