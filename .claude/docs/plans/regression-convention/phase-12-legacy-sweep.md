# Phase 12 — Final sweep: nothing legacy left behind

[← back to PLAN.md](PLAN.md) · [overview](overview.md#what-happens-to-each-existing-family) · [legacy inventory](legacy-inventory.md)

Branch: `legacy-sweep` from master after Phases 10 and 11.

Removal is not done here. Each phase removes what it replaced (C13 and its "Retire what this phase
replaced" task). This phase takes the cross-cutting items no single phase owns, sends the ASK list to the
lead, and proves by search that nothing replaced is left.

**Inputs to read first:**

- [legacy-inventory.md](legacy-inventory.md): every section, its REMOVE, ASK and KEEP rows, and the
  section [Phase 12](legacy-inventory.md#phase-12).
- [shared-contracts.md#c13](shared-contracts.md#c13-no-legacy-left-behind).
- The PR descriptions of Phases 3 to 11 (their old-against-new comparisons and the ASK rows each listed).
- `scripts/workflows/run_refits.py:152-200` (`RUNS`); `src/continuum_swr_replay/reported/questions.py:320-390`
  (Part 3 questions, the dropped ones included); `TODO.md`; `docs/README.md`.

**Contracts referenced:** [C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures),
[C13](shared-contracts.md#c13-no-legacy-left-behind).

## Tasks

- **The ASK list to the lead**, as one list: every ASK row of the inventory still open, with the
  recommendation written there and what the family phases learned. Act on each answer here (remove; keep,
  with the reason written into the inventory's KEEP table; or move onto the selection function and the
  shared fitter).
- **Cross-cutting removals** (the [Phase 12 section](legacy-inventory.md#phase-12) of the inventory):
  shared helpers whose last caller left in an earlier phase; the old cohort selectors if a caller outlived
  its phase; the stop-clock constants and columns once their last reader is gone; then the docs:
  `docs/models_table.md` reduced to the convention's models (with
  `tests/repository/test_models_table_mirrors_specs.py` pointed at the new specs), `docs/rate_models.md`
  removed (its content lives in `docs/regression_models.md`), the doc sections the inventory lists,
  `TODO.md` entries about removed models, `docs/README.md`'s index and "Where results live" rows, and the
  question list (every replaced 3.x question `dropped`, with its reason).
- **Audit 1, the inventory**: for every REMOVE row, `git grep -n -w <name> -- src scripts tests docs` gives
  zero matches outside the dated records C13 exempts. Paste the commands and their empty output into the
  PR.
- **Audit 2, unreferenced code** in the touched packages (`analysis/replay_models/`,
  `analysis/event_tables/`, `analysis/event_definition/cohort_selection.py`, `analysis/behavior/covariates/`,
  `analysis/pfc/`, `analysis/regression/`, `reported/behavior/`, `reported/mpfc/`, `scripts/replay_models/`,
  `scripts/pfc/`, `scripts/tables/`): for each public function, class and module-level constant defined
  there (read with `ast`: top-level `def`, `class`, and `NAME =` without a leading underscore), count the
  references outside its own module and outside its own test file
  (`git grep -n -w <name> -- src scripts tests`, less those two paths). List every name with zero; review
  the list by hand; remove what nothing needs, or justify it in the PR (for example, a name a script reads
  through a string). The counting script lives in the phase scratch directory and is not committed.
- **Audit 3, words**: `git grep -n -i -E "legacy|deprecat|backward|compat|retired|obsolete|older |pre-|no longer" -- src scripts tests docs`
  (dated records exempt). Each hit is removed or justified in the PR (a justified hit names what it keeps
  and why, for example "pre-outcome" as a covariate's timing, or the canonical-vocabulary deny-list).
- **Audit 4, runs**: every entry of `RUNS` has a reader: its output is read by a summary builder in
  `src/continuum_swr_replay/reported/` or by a figure script (grep the output name). A run with no reader is
  removed, or is an ASK item.
- **Audit 5, exploratory**: no file under `scripts/exploratory/` imports a removed name (grep;
  `tests/repository/test_no_new_script_imports.py` passes); a file that served only a removed model was
  removed by its phase.
- **Release check**: `make release-check`; every cited value exists, and none is pending because of a
  removal.

## Deliberately not in this phase

- Any new analysis or model change.
- Deleting data files in Processed-Data.
- Removing an ASK item without the lead's answer.
- A removal a family phase should have made: if one is found, it is made here, and the PR says which phase
  missed it.

## Validation slice

| Test | Asserts |
| --- | --- |
| full suite | passes |
| `tests/repository/test_import_layering.py`, `test_produced_by_paths_exist.py`, `test_doc_paths_resolve.py`, `test_models_table_mirrors_specs.py`, `test_docs_have_no_retired_names.py`, `test_no_new_script_imports.py` | no reference to removed code or files remains |
| `tests/reported/test_manuscript_macros.py` | every macro the manuscript cites exists |
| audits 1 to 5 (outputs in the PR) | each empty or justified |
| `make release-check` | passes, or lists only pending values the lead has accepted |

## Fixtures

None new.

## Verification and runtime

Before: `make reported-values`; save the cited count and the macro list. After: the same count unless the
lead approved a text change. The audits take minutes.

## Rebuilt

`paper/reported/*.json` only if a value group leaves here; the emitted files; docs.

## Exit criteria

Every REMOVE row has zero matches; every ASK row has the lead's answer, acted on; audits 2 to 5 have no
unexplained hit; the release check passes.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
