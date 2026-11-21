# Phase 6b — Content robustness estimators on the new design; the within-arm comparison deleted

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#content-design)

Branch: `role-content-robustness` from master after Phase 6a. CPU (the Dirichlet-multinomial fitter uses
JAX; run it with `JAX_PLATFORMS=cpu` unless a measured run says otherwise).

**Inputs to read first:**

- The design, section 4.2 ("The count response and the Dirichlet-multinomial are robustness estimators on
  the same design") and section 7's table (within-arm comparison: "deleted when the content model's Q5
  values replace its values").
- Phase 6a's `analysis/replay_models/content/role_content.py` (the design) and its stored fits.
- `src/continuum_swr_replay/analysis/replay_models/content/dirichlet_multinomial.py:564`
  (`fit_dirichlet_multinomial`), `:728` (`cross_validate_dirichlet_multinomial`), `:1021`
  (`fit_sessions_dirichlet_multinomial`).
- `src/continuum_swr_replay/analysis/replay_models/content/estimator_comparison.py:109`
  (`estimator_rows`), `:370` (`summarize_estimators`), `:435` (`bootstrap_animals`);
  `scripts/replay_models/content/compare_content_estimators.py`.
- The within-arm comparison, to delete: `analysis/replay_models/content/uncertainty_content.py:705`
  (`within_arm_horserace`), `:853-897` (`HORSERACE_SUMMARY`, `summarize_horserace`);
  `scripts/replay_models/content/analyze_uncertainty_content.py:41-43`, `:324-371`;
  `scripts/figures/generate_uncertainty_content_figure.py:85`, `:113`, `:793-830` (`_plot_horserace`),
  `:943`; `reported/behavior/content.py:718` (`horserace_values`); `reported/behavior/__init__.py:37-38`,
  `:98`, `:303-312`; `reported/behavior/inputs.py` (`horserace_summary`); the tests that name it
  (`tests/analysis/replay_models/content/test_uncertainty_content.py`, `tests/analysis/replay_models/test_fitting.py`,
  `tests/reported/behavior/test_content.py`, `tests/scripts/figures/test_figure_script_fixes.py`).

**Contracts referenced:** [C5](shared-contracts.md#c5-the-convention), [C7](shared-contracts.md#c7-the-fit-row-record),
[C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures).

## Tasks

- **Count response**: the same role design with the per-arm decoded bin counts (`arm_count_hard`) as the
  grouped softmax response (`n_g` the event's bin count), its own pooled REML; estimands as in 6a.
- **Dirichlet-multinomial**: the same design matrix (Phase 6a's terms, the smooths at the main model's
  selected penalties, fixed; call, since the shared fitter has no Dirichlet-multinomial REML) fit with
  `dirichlet_multinomial.py`'s likelihood and `log phi = g0 + g1 * is_swr` as now. Adapt
  `fit_dirichlet_multinomial` to take a prebuilt design and a fixed quadratic penalty, or add a thin
  wrapper; do not fork it.
- **Estimator comparison** on the new design: `estimator_comparison.py` reads the three estimators' records
  (fraction, count, Dirichlet-multinomial) for the declared estimands; the bootstrap uses
  `shared_statistics.nested_bootstrap_draws` and `bootstrap_interval` (C5 rule 5), not the percentile
  interval it used. Rewrite `compare_content_estimators.py` to read them.
- **Delete the within-arm comparison** once Phase 6a's Q5 values are in `paper/reported/behavior.json`:
  first list the `horserace_*` macros `paper/*.tex` cites (C10); if any is cited, stop and ask the lead for the
  text change. Then remove `within_arm_horserace`, `summarize_horserace`, `HORSERACE_SUMMARY`, the script
  lines that build and write it, `_plot_horserace` and its panel in the uncertainty-content figure (redraw the
  figure without panel E, or ask the lead whether the figure stays), `horserace_values` and its summary
  wiring and input, its tests, and its mentions in docs (`docs/regression_models.md`,
  `docs/regression_pipeline_map.md` regenerated). Leave the quartile profiles of the same module: nothing
  retires them. Add a row to `docs/history/2026-module-moves.md`.
- **Summary values**: per declared content estimand, the three estimators' values side by side (as now for
  the old design), labeled as robustness.
- **Docs**: `docs/regression_models.md` and `docs/design/content_estimator_comparison.md` (the robustness estimators on the role design).

- **Retire what this phase replaced** (C13; the items are the REMOVE rows of
  [legacy-inventory.md#phase-6b](legacy-inventory.md#phase-6b)). In this PR, or in a second PR of this phase
  that merges right after it:
  1. Write the old-against-new comparison for the lead: each old value the section lists, read from the
     copy of the summary saved before the rebuild, beside its replacement from the rebuilt summary, with the
     reason for each change. Put it in the PR description.
  2. List the macros `paper/*.tex` cites from those old values (none of `behavior.json` or `mpfc.json` was
     cited on 2026-11-20; check again). Ask the lead about any that are; remove a cited value only after
     the yes.
  3. Remove every REMOVE item of the section: modules, specs, runners, `RUNS` entries, plotters,
     value groups and their summary wiring, question-list entries (mark the replaced 3.x questions
     `dropped`, reason "superseded by" the new id), tests and doc sections.
  4. Check: `git grep -n -w <name> -- src scripts tests docs` gives no match for any removed symbol, file or
     column (dated records exempt, C13); `scripts/checks/write_pipeline_map.py` regenerated; one row per
     removal in `docs/history/2026-module-moves.md`; the full suite passes.
  The ASK rows of the section are not removed here; list them in the PR for the lead.

## Deliberately not in this phase

- Stored outputs of the old estimator comparison stay on disk, unread.
- Q6 (Phase 7).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/replay_models/content/test_role_content_robustness.py::test_count_response_matches_fraction_when_one_bin` | with one bin per event the count and fraction fits are identical |
| `test_role_content_robustness.py::test_dm_fixed_penalty_design` | the Dirichlet-multinomial on the role design recovers a planted role slope and its `phi` on simulated counts (`content/simulation.py:449`, `simulate_dirichlet_multinomial_counts`) |
| `tests/analysis/replay_models/content/test_estimator_comparison.py` (updated) | the comparison reads the three records and uses the t-corrected nested bootstrap interval |
| `tests/repository/test_produced_by_paths_exist.py`, `test_doc_paths_resolve.py` | no reference to a deleted file remains |
| full suite | passes after the deletion |

## Fixtures

Synthetic counts from `content/simulation.py`; Phase 6a's synthetic helpers.

## Verification and runtime

- Before deleting: save the `horserace_*` values and the uncertainty-content figure; after: the Q5 values that
  replace them are in the summary (name their keys in the PR).
- The Dirichlet-multinomial fits are the slow part today (the runner uses 16 jobs); time one session and
  extrapolate; run in `tmux` (`csr-6b-dm`).

## Rebuilt

`models_convention/content_robustness/`; the estimator comparison outputs; the uncertainty-content figure
(if kept); `paper/reported/behavior.json` and the emitted files.

## Exit criteria

The three estimators' values exist for every declared content estimand; the within-arm comparison and its
values are gone and nothing references them.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
