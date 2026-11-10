# Entry points for the common tasks. The work itself is done by the scripts in
# scripts/ and the package in src/; each target only calls them.
UV ?= uv
RUN = $(UV) run --frozen
PARTS = 0 1 2 3 4

.DEFAULT_GOAL := help
.PHONY: help sync sync-dev format check figures summaries reported-values manuscript

help:
	@echo "make sync             Install the locked analysis environment"
	@echo "make sync-dev         Install the locked environment with the development tools"
	@echo "make format           Format and lint-fix src/, scripts/ and tests/"
	@echo "make check            Run the CI checks: format check, lint, full test suite"
	@echo "make figures          Regenerate the main figures (needs the lab's processed data)"
	@echo "make summaries        Rebuild paper/reported/part{0..4}.json (parts 1-4 need the lab's processed data)"
	@echo "make reported-values  Write the summaries' values into the LaTeX macros and the docs"
	@echo "make manuscript       Build paper/main.pdf from the committed figures and macros"

# --inexact keeps the development tools that sync-dev installed.
sync:
	$(UV) sync --frozen --inexact

sync-dev:
	$(UV) sync --frozen --extra dev

format:
	$(RUN) ruff format src scripts tests
	$(RUN) ruff check --fix src scripts tests

check:
	$(RUN) ruff format --check src scripts tests
	$(RUN) ruff check src scripts tests
	$(RUN) pytest -q

figures:
	$(RUN) python scripts/figures/generate_all_figures.py

summaries:
	for part in $(PARTS); do \
		$(RUN) python scripts/reported/summarize_part.py --part $$part || exit 1; \
	done

reported-values:
	$(RUN) python scripts/reported/emit_reported_values.py

manuscript:
	$(MAKE) -C paper
