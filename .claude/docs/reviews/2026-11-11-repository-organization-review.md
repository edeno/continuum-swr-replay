# Repository organization review — 2026-11-11

Reviewed the checkout at `ed6789f8`, including the source tree, supported scripts,
documentation, test layout, and the route from scientific questions to reported
values. This is an organization assessment; the accompanying
[analysis-validity review](2026-11-11-analysis-validity-review.md) covers statistical
correctness. No analysis code or results were changed for this review.

## Assessment

The top-level organization is appropriate. The repository is reasonably easy to
enter, but tracing and modifying an individual analysis is less straightforward.
The main improvements should clarify ownership of calculations and shorten the
path to current instructions. A wholesale directory reorganization is unnecessary.

## What works

- `README.md` gives distinct routes for reading the paper, reproducing results,
  and finding figure code. `docs/reproduce.md` distinguishes a data-free manuscript
  build from summary regeneration and the full lab pipeline.
- `src/`, `scripts/`, `tests/`, `docs/`, and `paper/` have sensible roles. Scripts
  are grouped by domain; exploratory scripts and retired figures are marked.
- Shared event definitions, canonical tables, data paths, model specifications,
  and run infrastructure have explicit package homes.
- `paper/reported/*.json`, generated manuscript macros, question tags, and the
  release check provide a useful trace from a displayed value to its builder.
- Import boundaries, documentation paths, and the generated pipeline map are
  checked by tests. The test guide already prescribes an incremental migration
  toward tests that mirror the source tree.

## Priorities

### 1. Finish giving each scientific calculation one owner

This matters more than directory names. `analysis/` contains reusable statistics,
but scripts still implement some of the same operations. `TODO.md` already lists
the remaining copies. The earlier review confirmed that
`scripts/rate/plot_per_bin_rate_summary.py` computes a difference of AIC medians
where the summary uses paired differences, and selects a different cohort for
the reward-kernel plot. These are concrete consequences of duplicated logic.

Some run scripts also contain substantial reusable scientific work. For example,
`scripts/pfc/run_pfc_analysis_bandit.py` defines `score_session` and
`maximum_likelihood_comparison`; the content fit script implements cohort
selection and fitting orchestration. Script length alone is not a defect, but
these responsibilities make the boundary between library and command unclear.
`tests/README.md` even documents AST extraction as a workaround for testing prep
scripts that cannot conveniently be imported.

Recommendation: put cohort selection, model comparisons, uncertainty calculations,
and statistical aggregation in importable package functions. Have figures,
reports, and summary builders call those functions. Keep CLI parsing, job
selection, logging, and publishing in scripts. Preserve the existing result
schemas during extraction, and verify scientifically meaningful parity.

### 2. Make analysis domains easier to find inside the package

`analysis/replay/` contains rate models, content models, next-choice models,
stay/leave models, table builders, geometry, and summary statistics. By contrast,
`analysis/behavior/` contains behavioral-model covariate construction. A reader
looking for the stay/leave model must discover that it is in
`analysis/replay/learning.py`, despite its runner being under `scripts/behavior/`.

The largest reporting modules also span many questions: `reported/behavior.py`
has 3,118 lines and `reported/mpfc.py` has 2,438. These are measured source-line
counts, not evidence of incorrectness; the issue is the number of independent
analysis responsibilities a reviewer must navigate in one file.

Recommendation: use consistent domain names across source and scripts. As these
modules are changed, separate rate, content, choice, and stay/leave work behind
clear package interfaces, and split reporting builders by those same subjects.
Keep the five external summary names and their keys stable. Extend the existing
question index with direct analysis/runner links where useful; another parallel
catalog would add maintenance rather than clarity.

### 3. Separate current instructions from detailed reference and history

The entry README is short, but `docs/README.md` is 489 lines and
`scripts/README.md` is 535 lines. Both mix navigation with detailed descriptions
and move histories. The generated 1,167-line pipeline map is useful as a symbol
reference, but it is not a short route through an analysis.

There is also a specific mismatch: the documentation index describes
`docs/rate_models.md` as methods, explicitly “not results,” while that document
includes fitted results, comparisons, and historical interpretations. Its
generated value tags help keep numbers consistent; they do not establish that
surrounding interpretive prose remains current.

Recommendation: keep indexes short and link to dedicated method, artifact,
and migration references. Use `paper/reported/questions.md` for current numbers,
the manuscript for interpretation, and method documents for assumptions and
procedures. Keep explicitly historical explanations accessible in history docs.
If methods deliberately retain fitted examples, describe that scope accurately.

### 4. Expose the existing workflow runner as a workflow

`scripts/qa/run_refits.py` orchestrates the Part 3 scientific fits and dependent
figures. Its dependency selection, dry-run support, logs, and provenance checks
are useful infrastructure, but `qa/` is an unexpected place to find the main
refit workflow. `make help` exposes summaries and figures but no refit target.

The script index still describes this runner as Q2/Q3b/Q4, while its code and
the reproduction guide describe Part 3. The PFC script-index heading also uses
Q5 while the current paper outline calls this Part 4. Paths can resolve and
the symbol map can be current while these descriptions disagree.

Recommendation: expose the existing runner with a documented `make refits`
target or a clearly named workflow entry point, then use the current question
identifiers consistently. A new workflow engine is not needed for this change.

### 5. Keep durable scientific decisions visible outside agent history

The documentation policy appropriately separates implementation plans from
scientific documentation. However, it places both code reviews and methodological
audits under `.claude/docs/reviews/`, and some public method descriptions link
there for validation evidence. The development guide simultaneously calls this
tree “not scientific record.” This makes the authority of those audits unclear.

Recommendation: leave dated execution logs and review snapshots in their current
home. Promote accepted methodological decisions, unresolved limitations, and
validation evidence needed to interpret the paper into `docs/design/` or
`docs/known_issues/`, with links back to the supporting audit. The existing
reported-statistics choices document is a useful example of that pattern.

## Lower-priority cleanup

- Continue the existing policy of moving tests from the flat `tests/unit/`
  directory into the source-mirroring layout when the relevant module changes.
  A bulk test move offers little immediate scientific benefit.
- Add `reported/` explicitly to the documented dependency layers: it is a major
  architectural component but is absent from the numbered layer list in
  `docs/README.md`. State that scientific analysis does not depend on reporting.
- Preserve centralized path functions while improving artifact names gradually.
  The data tree's broad `replay_content/` name and `models*` folders are less
  descriptive than the code domains, but renaming stored outputs is a lower
  priority than fixing computational ownership and keeping provenance intact.

## Validation

Ran the existing architecture/documentation checks with the local environment:

```text
.venv/bin/python -m pytest -q \
  tests/unit/test_import_layering.py \
  tests/unit/test_no_new_script_imports.py \
  tests/unit/test_doc_paths_resolve.py \
  tests/unit/test_docs_have_no_retired_names.py \
  tests/unit/test_pipeline_map_is_current.py

22 passed in 2.13s
```

These establish the tested structural properties and reference consistency;
they do not establish statistical validity or the correctness of all prose.
The recommendations above come from inspecting responsibilities and tracing
specific results, in addition to those automated checks.
