# Phase 5a — Readability, low risk: narrative out of code, shims and dead code gone, one z-score path

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Findings addressed: D1, D4, D7, D8 (dead/test-only symbols), D5 (versions check, clocks wrappers).

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/model_specs.py:127-186, 200-212, 383-400, 469-480, 905-940, 1095-1110, 1264-1271` — narrative blocks; `docs/design/model_history.md:1-10` — where they go.
- `src/continuum_swr_replay/analysis/replay/tables.py` — the shim; importers: `prepare_data.py`, `analysis/replay/__init__.py:101`, `visualization/session_tables.py`, `scripts/decoding/regenerate_stop_table.py:32`, and every test importing `_EVENT_*` / `_STOP_*` from `.tables`.
- Dead symbols: `_helpers.py:104` (`immobility_bout_starts`), `retrospective.py:79` (`_PATCH_TO_ARM_SET`), `simulation.py:101-103` (`_BASE_BINARY`, `_CONTINUOUS_ARM`, `_EVENT_LEVEL`), `_spatial_bandit_rl.py:18` (`set_nan_at_max2`; note line 236 applies it — read that line before deleting: if the apply is a no-op, delete both; if not, keep both).
- Test-only surfaces: `covariates/history/visit_recency.py:287-306` (`*_per_arm` views), `covariates/history/reward_history.py:279-304` (`*_per_arm` views), `covariates/context.py:364-376` (`behavioral_location_at_times`), `covariates/clocks.py:141-152` (wrappers), `covariates/__init__.py` and `history/__init__.py` (`_x as _x` re-exports), `versions.py:40-66`.
- z-scoring sites: `scripts/content/fit_swr_contrast_content.py:131` (removed in phase 2a; confirm), `scripts/rate/fit_uncertainty_rate.py:600-613`, `scripts/rate/fit_swr_power_model.py:273-274`.
- `scripts/qa/freeze_regression_baseline.py`, `scripts/qa/compare_to_golden_baseline.py` — the parity gate.

**Contracts referenced:**

- [Standardization entry point](shared-contracts.md#standardization-entry-point) — enforced here.

**Designs referenced:** none.

## Tasks

- **Narrative out of `model_specs.py`.** Move the seven blocks listed above into
  `docs/design/model_history.md` under dated headings (verbatim), leaving in code a one-line pointer
  per rung ("reward-memory rung removed 2026-08-18; see model_history.md §…"). Fix the contradiction
  at 157-159 (last-rewarded family) to match the current state. Target: `model_specs.py` < 900 lines.
- **Delete the `tables.py` shim.** Point the four importers at `tables_event` / `tables_stop` /
  `tables_arm_patch` / `_tables_common`; tests import column tuples from `_tables_common`.
- **Delete strictly dead symbols** (list above) after re-running `grep -rn` for each across `src/ scripts/ tests/`.
- **Remove test-only compatibility surfaces.** Delete the four `*_per_arm` scalar views,
  `behavioral_location_at_times`, the two clock wrappers (rename `_trial_arrival_times` →
  `trial_arrival_times`, `_trial_outcome_times` → `trial_outcome_times` and update the four callers: `visualization/session_tables.py:86,505`, `scripts/figures/generate_event_gallery.py:91-92,599,639,732,745,1125`, `scripts/exploratory/_audit_covariate_accuracy.py:179,215`, `scripts/exploratory/_lagged_reward_kernel.py:25,53`
  in `visualization/session_tables.py` and `scripts/figures/generate_event_gallery.py`), and the
  `_x as _x` re-exports in both package `__init__`s. Rewrite the affected tests
  (`test_history_public_api.py`, `test_covariates.py`, `test_contfrag_arm_fraction_denominator.py`,
  `tests/benchmarks/bench_per_arm_helpers.py`) against the vectorized functions.
- **Trim `analysis/replay/__init__.__all__`** to names actually imported through the package (43 of 78
  are not); keep the star-import test passing.
- **Wire the version check.** Call `versions.assert_versions_compatible(covariate_schema_version=…)`
  inside `covariate_registry.require_covariate_schema` and
  `assert_versions_compatible(behavioral_context_cache_version=…)` inside
  `prepare_data.load_behavioral_context` when reading a cache; delete nothing else in `versions.py`.
- **One z-score path.** Replace the inline z-scoring in `fit_uncertainty_rate.py:600-613` and
  `fit_swr_power_model.py:273-274` with `attach_standardized(..., support_label="fit_cohort_all_rows")`
  and persist the reference JSON next to each output.
- **`event_immobility.py` drop count.** Have `filter_event_intervals_by_speed` (or the caller in
  `event_table.py`) return/log the number of intervals removed; delete the "Filtering is silent"
  caveat at lines 16-18.
- **Parity gate.** Before the first edit, run `scripts/qa/freeze_regression_baseline.py` for one
  cached session; after the last edit, `scripts/qa/compare_to_golden_baseline.py` must report
  byte-identical tables. Run the full unit suite.

## Deliberately not in this phase

- Renaming persisted values or columns (`in_transit` → `between_visits`, `is_prev_arm` →
  `is_last_trip_origin_arm`, `dot`, `tc`): revisit at the next full table regeneration, when the
  rename can ride the regeneration instead of forcing one.
- Structural moves (`_helpers` split, registry trim, CLI relocation): phase 5b.
- Any change to fitted outputs.

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest tests/unit -q` | passes; count of collected tests reported in the PR |
| `test_history_public_api.py` (rewritten) | the vectorized functions are importable from `covariates.history`; no `_x` names in `__all__` |
| `test_versions.py::test_require_schema_rejects_old_version` | a frame stamped with an old version raises through `require_covariate_schema` |
| `test_standardization.py::test_runner_references_written` | `fit_uncertainty_rate` smoke writes `uncertainty_rate.z_reference.json` with `support == "fit_cohort_all_rows"` |
| `compare_to_golden_baseline.py` (slow, real data) | one session byte-identical across all eight tables |
| `wc -l model_specs.py` | < 900 |

## Fixtures

- One cached real session for the parity gate (the same one `freeze_regression_baseline.py` already uses).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies. Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Every symbol listed for deletion is gone (grep evidence in the PR), and nothing scientific moved.
- The parity report is attached to the PR.
