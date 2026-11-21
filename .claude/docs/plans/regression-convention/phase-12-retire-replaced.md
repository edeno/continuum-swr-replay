# Phase 12 — Retire what the convention replaced

[← back to PLAN.md](PLAN.md) · [overview](overview.md#what-happens-to-each-existing-family)

Branch: `retire-replaced-models` from master after Phase 10 (and Phase 11 for the history models). Mostly
deletions. Two deletions happen earlier and are not repeated here: the older all-bins history model
(Phase 1) and the within-arm comparison (Phase 6b).

The rule (design section 7): nothing is removed before its replacement's values are in a summary, and a
value the manuscript cites is removed only with the lead's yes on the text change (C10). Remove only what
the design's table says is replaced; an analysis nothing cites is a question for the lead, not a removal.

**Inputs to read first:**

- [overview.md#what-happens-to-each-existing-family](overview.md#what-happens-to-each-existing-family).
- `scripts/workflows/run_refits.py:152-200` (`RUNS`); `reported/behavior/__init__.py:132-420`
  (`summarize`), `reported/behavior/rate.py`, `content.py`, `choice.py`, `coupling.py`, `history.py`,
  `stops.py`, `inputs.py`.
- `paper/*.tex` (which macros are cited); `make reported-values` output (cited count).
- `src/continuum_swr_replay/analysis/replay_models/model_specs.py`, `docs/models_table.md`,
  `tests/repository/test_models_table_mirrors_specs.py`.
- The callers of the old cohort selectors (`grep -rn "select_stop_cohort\|select_event_cohort" src scripts`):
  on 2026-11-20 they included `contrast_aggregation.py`, `rate/swr_contrast_rate.py`, `rate/stop_statistics.py`,
  `choice/retrospective.py`, `choice/choice_prediction.py`, `content/cohort.py`, `pfc/cohort.py`,
  `reported/behavior/__init__.py`, `reported/behavior/stops.py`, `scripts/pfc/measure_pfc_mark_context.py`,
  and several rate and choice runners.
- `src/continuum_swr_replay/reported/questions.py:320-358` (questions 3.1 to 3.8).

**Contracts referenced:** [C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures).

## Tasks

- **List cited values first**: for every value group to be removed, list the macros `paper/*.tex` cites.
  Send the list to the lead with the replacement value for each. Wait for the yes, per group.
- **Rate**: remove the per-bin ladder (`rate/per_bin_rate_ladder.py`, `rate/per_bin_rate_model.py`,
  `MODEL_PER_BIN_RATE`, `scripts/replay_models/rate/fit_per_bin_rate_models.py`, `plot_per_bin_rate_models.py`,
  `plot_per_bin_rate_summary.py`, and `plot_model_checking.py` if it checks only that ladder), the old
  per-stop family (`rate/per_stop_rate_model.py`, `MODEL_PER_STOP_RATE`, `fit_per_stop_rate_models.py`), the
  SWR-contrast rate ladder (`rate/swr_contrast_rate.py`, `fit_swr_contrast_rate.py`, its specs), the
  uncertainty rate model (`rate/uncertainty_rate.py`, `fit_uncertainty_rate.py`, `plot_uncertainty_rate.py`,
  `MODEL_UNCERTAINTY_RATE`), and, after Phase 11, the old history runners
  (`fit_duration_aware_history_rate.py`, `fit_cross_history_ridge_cv.py`) with the modules only they use.
  Then remove `per_bin_design.aggregate_to_timebins` only if no remaining caller needs it (the stop and
  time-bin table builders and `covariate_timing.py` referenced it on 2026-11-20; check each).
- **Content**: remove the content ladders (`MODEL_HIERARCHY*`, `MODEL_PATCH*`, the leaf-only helpers,
  `content/session_fits.py` functions only they use, `scripts/replay_models/content/fit_replay_content_models.py`,
  `fit_swr_contrast_content.py`, `generate_replay_content_figures.py` if it draws only them, the
  SWR-contrast figure `scripts/figures/plot_swr_contrast_coefs.py` and its `FIGURES` entry if the lead
  agrees), the per-term gate of `contrast_aggregation.py` if unused, and `fitting.fit_conditional_poisson`
  only if no remaining caller needs it (the Dirichlet-multinomial, estimator comparison, model checking,
  simulation and choice modules referenced it on 2026-11-20).
- **Choice, stay or leave, power**: the old fits and values once Phase 7's are in (keep the pooled GEE only
  if the lead keeps stay or leave and wants it).
- **mPFC**: the fixed-sigma bundles become the sensitivity path; remove the default-sigma code path only if
  the lead agrees.
- **Remaining cohort callers**: every remaining caller of `select_stop_cohort` / `select_event_cohort` that
  feeds a Figure 2 to 5 statistic moves to `select_visit_context` (for example `reported/behavior/stops.py`'s
  stop facts, `choice/retrospective.py`); then the old selectors are removed if nothing calls them.
- **Values, questions, runs, docs**: remove the old values from the builders; mark questions 3.1 to 3.8
  `dropped` with the reason "superseded by 3.11-3.17" (or as the lead words it); remove the old runs from
  `RUNS`; update `docs/models_table.md`, `docs/rate_models.md`, `docs/behavior_models.md`,
  `docs/README.md` ("Where results live"); regenerate `docs/regression_pipeline_map.md`; add rows to
  `docs/history/2026-module-moves.md`; delete the tests of removed code.
- **Stored fits on disk**: leave the old fit files in Processed-Data (no data deletion without the lead's
  yes); they are no longer read.
- **Release check**: `make release-check`; with the old values gone and the new ones rebuilt, any remaining
  pending cited value is listed for the lead.

## Deliberately not in this phase

- Any new analysis or model change.
- Deleting data files.
- Removing an analysis the design's table does not list as replaced.

## Validation slice

| Test | Asserts |
| --- | --- |
| full suite | passes after the deletions |
| `tests/repository/test_import_layering.py`, `test_produced_by_paths_exist.py`, `test_doc_paths_resolve.py`, `test_models_table_mirrors_specs.py`, `test_docs_have_no_retired_names.py` | no reference to removed code or files remains |
| `tests/reported/test_manuscript_macros.py` | every macro the manuscript cites still exists |
| `make release-check` (script) | its report lists no cited value that was removed |

## Fixtures

None new.

## Verification and runtime

Before: `make reported-values` and save the cited count and the list of macros. After: the cited count is
the same unless the lead approved a text change; every removed key is listed in the PR with its
replacement.

## Rebuilt

`paper/reported/behavior.json`, `mpfc.json` (if values leave), the emitted files; docs.

## Exit criteria

No code path of a replaced family remains; every Figure 2 to 5 statistic reads through the one selection
function; the manuscript's cited values all exist.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
