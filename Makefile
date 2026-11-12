# Entry points for the common tasks. The work itself is done by the scripts in
# scripts/ and the package in src/; each target only calls them.
UV ?= uv
RUN = $(UV) run --frozen
SUMMARIES = survey detection events behavior mpfc
# Extra arguments for release-check and refits, for example ARGS=--code-only or
# ARGS=--dry-run.
ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help sync sync-dev format check figures refits summaries reported-values release-check manuscript

help:
	@echo "make sync             Install the locked analysis environment"
	@echo "make sync-dev         Install the locked environment with the development tools"
	@echo "make format           Format and lint-fix src/, scripts/ and tests/"
	@echo "make check            Run the CI checks: format check, lint, full test suite"
	@echo "make figures          Regenerate the main figures (needs the lab's processed data)"
	@echo "make refits           Rerun the Part 3 fits and the figures that read them (needs the lab's processed data;"
	@echo "                       ARGS=--dry-run lists the runs, ARGS=--with-tables adds the table rebuilds)"
	@echo "make summaries        Rebuild paper/reported/<summary>.json (all but survey need the lab's processed data)"
	@echo "make reported-values  Write the summaries' values into the LaTeX macros and the docs"
	@echo "make release-check    Before a milestone: fail if a summary is stale or mis-tagged, or a cited value is pending"
	@echo "                       (ARGS=--code-only skips the input files, for a machine without the data)"
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

refits:
	$(RUN) python scripts/qa/run_refits.py $(ARGS)

summaries:
	for summary in $(SUMMARIES); do \
		$(RUN) python scripts/reported/summarize.py --summary $$summary || exit 1; \
	done

reported-values:
	$(RUN) python scripts/reported/emit_reported_values.py

release-check:
	$(RUN) python scripts/reported/check_release.py $(ARGS)

manuscript:
	$(MAKE) -C paper
