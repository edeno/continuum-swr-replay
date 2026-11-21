# Phase 11 — The event-history models on the shared fitter

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Branch: `history-models-convention` from master after Phase 10 (the design: "moved to the shared fitter
after Figure 4, since they already use at-risk time and a penalty"). CPU first; the ridge runner used a GPU,
so measure before choosing.

**Inputs to read first:**

- The design, section 7's row for the event-history models, and sections 4.3 and 4.4 (REML sets penalties,
  folds do not).
- `src/continuum_swr_replay/analysis/replay_models/rate/duration_aware_history.py:109`
  (`attach_event_ends`), `:169` (`at_risk_events`), `:201` (`build_design`), `:415`
  (`onset_offset_collinearity`), `:453` (`fit_pooled_history`);
  `scripts/replay_models/rate/fit_duration_aware_history_rate.py`.
- `src/continuum_swr_replay/analysis/replay_models/rate/ridge_history_cv.py:70-78` (folds, seed),
  `:220` (`assign_stop_folds`), `:383` (`fit_ridge_poisson_batch`), `:453` (`nested_cv`), `:518`
  (`ridge_kernels`), `:551` (`compare_cross_history`); `scripts/replay_models/rate/fit_cross_history_ridge_cv.py`.
- `src/continuum_swr_replay/analysis/replay_models/rate/history_kernel_statistics.py` and
  `reported/behavior/history.py:36` (`duration_aware_history_values`): the values these produce.
- `model_specs.py:1141-1250` (`MODEL_CROSS_HISTORY_RATE`, the duration-aware and long variants).
- Phase 5a's visit model and Phase 2's tools.

**Contracts referenced:** [C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C5](shared-contracts.md#c5-the-convention), [C7](shared-contracts.md#c7-the-fit-row-record).

## Tasks

- **Baseline**: save the `rate.duration_aware_history.*` and `rate.cross_history_ridge_cv.*` values and the
  stored kernels.
- **Design on the visit**: the self- and cross-history terms (onset- and end-anchored raised-cosine bases
  as now) added to Phase 5a's model A on the visit bins (C4 selection), with the end-anchored history counted
  from event ends as now.
- **Penalty by REML**: the history kernels become penalized smooths of lag (the convention's form for a
  peri-event lag), with one REML strength per kernel shared over sessions, replacing the per-animal nested
  cross-validation of the ridge sigma (`nested_cv`). The cross-history question ("does a recent event of one
  kind change the other kind's rate") is answered by the kernel with its simultaneous band and, as a
  held-out check, by the score summed over sessions on trial folds (Phase 2's fold function), not by a
  per-session held-out test.
- **Per session, then reduced**: per-session fits at the shared penalty, reduced session to day to animal,
  replacing the per-animal pooled fits with `C(session)` (the pooled form is stage 3; say so in the docs).
- **Runner** `scripts/replay_models/rate/fit_visit_history.py` (new), registered in `run_refits.py`; the
  two old runners are removed by this phase's retire task.
- **Summary values**: the self- and cross-history kernels' summaries per kind and difference (outside the
  declared family), with the collinearity record of onset- against end-anchored terms
  (`onset_offset_collinearity` measured the same concern).
- **Docs**: `docs/rate_models.md` (history on the visit model, REML in place of nested CV).

- **Retire what this phase replaced** (C13; the items are the REMOVE rows of
  [legacy-inventory.md#phase-11](legacy-inventory.md#phase-11)). In this PR, or in a second PR of this phase
  that merges right after it:
  1. Write the old-against-new comparison for the lead: each old value the section lists, read from the
     copy of the summary saved before the rebuild, beside its replacement from the rebuilt summary, with the
     reason for each change. Put it in the PR description.
  2. List the macros `paper/*.tex` cites from those old values (none of `behavior.json` or `mpfc.json` was
     cited on 2026-11-20; check again). Ask the lead about any that are; remove a cited value only after
     the yes.
  3. If any test of the shared fitter compares with `ridge_history_cv.fit_ridge_poisson`, first commit that fitter's coefficients on the test's data as a C13 fixture. Remove every REMOVE item of the section: modules, specs, runners, `RUNS` entries, plotters,
     value groups and their summary wiring, question-list entries (mark the replaced 3.x questions
     `dropped`, reason "superseded by" the new id), tests and doc sections.
  4. Check: `git grep -n -w <name> -- src scripts tests docs` gives no match for any removed symbol, file or
     column (dated records exempt, C13); `scripts/checks/write_pipeline_map.py` regenerated; one row per
     removal in `docs/history/2026-module-moves.md`; the full suite passes.
  The ASK rows of the section are not removed here; list them in the PR for the lead.

## Deliberately not in this phase

- The final audit (Phase 12).
- Adding history terms to the declared family.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/replay_models/rate/test_visit_history.py::test_planted_self_history_recovered` | a planted self-excitation kernel is recovered on simulated visits (the generator of `rate/duration_history_simulation.py`) |
| `test_visit_history.py::test_cross_history_null` | no cross effect planted: the cross kernel's band covers zero in at least 93 % of 100 data sets |
| `test_visit_history.py::test_end_anchored_history_from_event_ends` | the history columns count from event ends where the model says so |

## Fixtures

`rate/duration_history_simulation.py`'s simulator on synthetic visits.

## Verification and runtime

Simulated, then five sessions, then 242; compare kernel shapes with the baseline per kind; time the CPU run
before deciding on a GPU (C11 if one is used).

## Rebuilt

`models_convention/rate/history.parquet`; `paper/reported/behavior.json` and the emitted files.

## Exit criteria

History kernels on the visit model are stored with records and their values are in the summary.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
