# Phase 5b — Readability, structural: split `_helpers`, one schema module, trimmed registry, relocated CLIs

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Findings addressed: D2, D3, D5 (session keys, `is_stay`, event timing), D8 (CLIs, validators).

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/_helpers.py` (948 lines; function index: `nearest_well_distance_cm` 29, `immobility_bout_intervals` 118, `time_since_stop` 142, `hash_event_id` 265, `hash_stop_id` 296, `assign_bout` 328, `eq_indicator*` 343/360, `align_count_matrix` 391, `add_arm_fixed_effects` 514, `build_well_to_arm_map` 553, `attach_is_swr_to_onsets` 574, `raised_cosine_history_basis` 752, `aggregate_to_timebins` 783, `POPULATIONS` 939).
- `src/continuum_swr_replay/analysis/replay/model_specs.py:79` (`ARM_FE_PREDICTORS`), `:219-225` (`RECENCY_SPLINE_*`), `:597` (`N_REWARD_LAGS`), `:631-650` (`BIN_S`, `TMAX`, `SPLINE_KNOTS`, `N_HISTORY_*`, `STOP_COVARIATES`) — constants the builders import.
- Importers of those constants: `covariates/definitions.py:25`, `covariates/candidate_arm.py:22`, `covariates/history/visit_recency.py:10-13`, `covariates/history/value_uncertainty.py`, `_helpers.py` (for `ARM_FE_PREDICTORS`), `tables_stop.py:49`.
- `src/continuum_swr_replay/analysis/replay/covariate_registry.py:165-1027` (ledger), `:1030-1090` (derived views), `:1207-1425` (qvals ingestion helpers), `:1426-1540` (stamp/require); production callers: `qvals_table.py` (the ingestion quartet), `require_covariate_schema` ×42, `stamp_covariate_schema` ×12, `row_keys` ×2.
- `src/continuum_swr_replay/session_keys.py:25-28`; duplicates at `retrospective.py:95`, `timebin_tables.py:79`, `_helpers.py:27`, `table_contracts.py:22`, `scripts/content/fit_replay_content_models.py:134`, `prepare_data.py:456`.
- `is_stay` sites: `tables_event.py:284`, `tables_stop.py:292`, `shuffles.py:325`, `analysis/contfrag/tables.py:422, 913`.
- `src/continuum_swr_replay/analysis/replay/tables_event.py:467-566` — `_merge_classified_and_verify_timing` (100 lines re-deriving event timing).
- CLIs in `src/`: `analysis/replay/prepare_data.py` (`main`, argparse) and `analysis/behavior/qvals_table.py:965` (`main`).
- Validators in the content runner: `scripts/content/fit_replay_content_models.py:191-234, 235-414, 664-822`; existing home `analysis/replay/table_contracts.py` (300 lines).

**Contracts referenced:** none new; the phase must preserve every contract in [shared-contracts.md](shared-contracts.md).

**Designs referenced:** none.

## Tasks

- **Design-constant module.** Create `analysis/replay/design_constants.py` holding
  `ARM_FE_PREDICTORS`, `RECENCY_SPLINE_KNOTS/BOUNDS/COLUMNS`, `N_REWARD_LAGS`, `N_HISTORY_LAGS`,
  `N_HISTORY_BASIS`, `BIN_S`, `TMAX`, `SPLINE_KNOTS`, `TMAX_ARRIVAL`, `ARRIVAL_SPLINE_KNOTS`,
  `REWARD_HISTORY_N_LAGS`, `REWARD_HISTORY_N_BASIS`, `REWARD_HISTORY_COLUMNS`,
  `REWARD_HISTORY_INCOMPLETE`, and `raised_cosine_history_basis` (shared by the self-history and
  reward-history blocks). `model_specs.py` imports them (and keeps re-exporting the names in `__all__`
  so scripts are unchanged); builders import from `design_constants` so no builder imports
  `model_specs`. Move `OMISSION_KNOWN_LATENCY_SEC` from `analysis/constants.py` to
  `data_loaders/constants.py` so `data_loaders/trial_clocks.py` no longer imports from `analysis`.
- **Split `_helpers.py`.** New modules: `bouts.py` (`immobility_bout_intervals`, `time_since_stop`,
  `assign_bout`, `nearest_well_distance_cm`), `event_ids.py` (`hash_event_id`, `hash_stop_id`,
  `validate_event_ids`), `per_bin_design.py` (`aggregate_to_timebins`, `raised_cosine_history_basis`,
  `attach_is_swr_to_onsets`, `POPULATIONS`), keeping `_helpers.py` for the reducers, `eq_indicator*`,
  `align_count_matrix`, `add_arm_fixed_effects`, `build_well_to_arm_map`, and the alignment asserts.
  Update the 12 importers; `_helpers.py` < 400 lines. `hippocampus/event_table.py` imports from
  `bouts.py`/`event_ids.py` only.
- **Trim the registry.** Keep `_ROW_KEYS`, `row_keys`, `stamp_covariate_schema`,
  `require_covariate_schema`, `assert_unique_scientific_rows`, `finalize_covariate_schema`, the
  qvals ingestion quartet, and only the `Entry` rows those functions need (the qvals native →
  canonical set). Replace `assert_canonical_vocabulary`'s ledger-derived deny-list with a fixed
  `LEGACY_COLUMN_NAMES` frozenset (extract it once from the current ledger). Move the full ledger
  to `docs/design/2026-08-21-covariate-naming-scheme.md` as an appendix table. Target < 400 lines.
- **One session key.** Replace the six local definitions with `session_keys.SESSION_COLUMNS` /
  `SESSION_COLUMNS_DAY`; where a frame carries `day` instead of `session_date`, use
  `session_keys.session_key_frame`. Delete `_helpers._SESSION_COLS`.
- **One `is_stay`.** Add `stay_indicator(next_arm, current_arm) -> float` to `_helpers.py` (NaN
  when either is NaN) and use it at the five sites, including ContFrag.
- **Compute event timing once.** In `tables_event.py`, take `event_start_s`/`event_end_s`/
  `event_duration_s` from the classified table (`hippocampus/event_table.py`) and delete the
  re-derivation plus `_merge_classified_and_verify_timing`'s rename/compare/drop (467-566); keep a
  single `assert_event_label_alignment` call.
- **Relocate CLIs.** Move `prepare_data.main`/argparse to `scripts/decoding/prepare_replay_content_data.py`
  (the name `table_io.py:196` already points users to) and `qvals_table.main` to
  `scripts/behavior/build_qvals_table.py`; the modules keep only library functions. Update
  `docs/README.md`, `scripts/README.md`, and the two docstrings that cite the module invocation
  (`scripts/rate/fit_swr_contrast_rate.py:267-268`).
- **Relocate validators.** Move `_validate_requested_standardization`,
  `_validate_canonical_table_contract`, and `_validate_and_match_fit_results` from the content
  runner into `analysis/replay/table_contracts.py` as public functions with the same behaviour;
  the runner's science path (`_load_and_filter_cohort → _fit_content_ladder → fit_model_per_session`)
  should then read in under 200 lines.
- **Parity gate.** Same as phase 5a: freeze one session before, `compare_to_golden_baseline.py`
  after; additionally re-run one content smoke fit and diff its coefficients against the phase-2a
  smoke output (must be identical to 1e-12).

## Deliberately not in this phase

- Persisted-name renames (see phase 5a).
- Any change to a fitter or spec.
- Deleting `simulation.py` test-only symbols beyond phase 5a's list (the simulators are test infrastructure by design).

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest tests/unit -q` | passes |
| `test_import_layering.py` (new) | no module under `covariates/` imports `model_specs`; `data_loaders/` imports nothing from `analysis/`; `hippocampus/` imports nothing from `replay/` except `bouts`/`event_ids` |
| `test_session_keys.py::test_no_local_session_key_definitions` | grep-based: the six former sites are gone |
| `test_stay_indicator.py` | NaN propagation and equality semantics; ContFrag and Replay agree on a fixture |
| `test_covariate_registry.py` (rewritten) | qvals ingestion still canonicalizes the native fixture; legacy-name deny-list rejects a legacy column |
| `compare_to_golden_baseline.py` (slow) | byte-identical tables |
| content smoke coefficient diff (slow) | identical to 1e-12 |
| `wc -l` | `_helpers.py` < 400, `covariate_registry.py` < 400 |

## Fixtures

- The cached real session used by the parity gate; the phase-2a smoke output kept for the coefficient diff.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies. Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal are removed (`_merge_classified_and_verify_timing`, the ledger, `_SESSION_COLS`, the in-`src` CLIs).
- Parity and coefficient-diff evidence is attached to the PR.
