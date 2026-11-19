# Phase 2a — Content models: one draw per event, event-clustered SEs, length and outcome controls

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#event-clustered-sandwich-covariance)

Findings addressed: A1, A5 (reporting), A6 (restore base ladder), A7 (`_TSC` consistency), B1, B5, B6.

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/fitting.py:83-243` — `fit_conditional_poisson`; response parameter at 86, `_drop_zero_count_events` at 155, fit at 182, null llf at 209-214, output dict 218-243.
- `src/continuum_swr_replay/analysis/replay/fitting.py:246-300` — `fit_null_model_conditional` (`llf = -total * log(n_arms)`; equals `-n_events * log(n)` for fractions).
- `src/continuum_swr_replay/analysis/replay/tables_arm_patch.py:163-182` — `arm_fraction`; `:227-280` — patch builder (`patch_count_hard` at 277), where `patch_fraction` is added.
- `scripts/content/fit_replay_content_models.py:450-550` — `prepare_canonical_fit_request` (response column resolution); `:552-662` `fit_model_per_session`; `:1113-1196` `_resolve_run_config`; `:1296-1299` at-well filter; `:1376-1440` `_fit_content_ladder` (`n_categories` at 1402); `:986-1095` CLI flags.
- `scripts/content/fit_swr_contrast_content.py:120-231` — `_assemble_arm` (`_TSC` built at 221-222); `:233-280` `_assemble_patch`; `:366-452` `_run`; `:584-700` `main` (timing gate 595-600).
- `src/continuum_swr_replay/analysis/replay/model_specs.py:1106-1113` (`BASEc`, `_TSC`), `:1114-1257` (arm CS ladder), `:1258-1311` (patch ladder), `:1341-1363` (`LEAF_ONLY_*`, `leaf_only_predictors`), `:228-352` (`MODEL_HIERARCHY`).
- `src/continuum_swr_replay/analysis/replay/covariate_timing.py:64-166` — `DERIVED_COVARIATE_SOURCES`.
- `src/continuum_swr_replay/analysis/replay/covariates/definitions.py:111-420` — catalog entries (add `n_nonlocal_bins`-derived terms).
- `src/continuum_swr_replay/analysis/replay/model_comparison.py:101-180` — `summarize_comparison`.
- `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:80-95` — `std_factor`; `:146-175` — row fields.
- `src/continuum_swr_replay/analysis/replay/simulation.py:307-440` — `simulate_with_event_covariates`; `:471` — `simulate_autocorrelated_counts`.

**Contracts referenced:**

- [Fraction response](shared-contracts.md#fraction-response) — implemented here.
- [Content estimator result schema](shared-contracts.md#content-estimator-result-schema) — `bse_cluster_*`, `estimator`, `llf_includes_constants` added here.
- [Standardization entry point](shared-contracts.md#standardization-entry-point) — the SWR-content runner's `_z` is replaced here.
- [Fit provenance sidecar](shared-contracts.md#fit-provenance-sidecar) — wired into both content runners.

**Designs referenced:** [designs.md#event-clustered-sandwich-covariance](designs.md#event-clustered-sandwich-covariance).

## Tasks

- **Persist `patch_fraction`.** In `tables_arm_patch.build_event_patch_table` (after the
  `patch_count_hard` aggregation at 277) add
  `patch_fraction = arm_fraction_from_event_total(patch_count_hard, n_nonlocal_bins - n_undecoded_bins)`;
  add the column to the patch schema (`table_contracts.validate_event_patch_table` and the patch
  column order), add a catalog entry, and regenerate `event_patch_table.parquet` only
  (`prepare_data` builds it from the persisted event-arm table; confirm by reading
  `prepare_data._process_session` before choosing the regeneration path — if the patch table cannot
  be rebuilt alone, rebuild the arm+patch pair from cache with `--from-cache`; `event_id` does not change).
- **Event-clustered SEs.** Add `event_clustered_covariance` to `fitting.py` (design section 1) and
  emit `bse_cluster_<term>` from `fit_conditional_poisson`; set `estimator` and
  `llf_includes_constants=False` in its output (and in `fit_conditional_logit`,
  `fit_null_model_conditional*`). `contrast_aggregation.aggregate_contrast` prefers
  `bse_cluster_<term>` for `typical_session_se` when the column exists.
- **Fraction as the shipped response.** `fit_replay_content_models.py`: add
  `--response {fraction,hard_count}` (default `fraction`) resolved in
  `prepare_canonical_fit_request` to `arm_fraction` / `patch_fraction` or `arm_count_hard` /
  `patch_count_hard`; pass through `_ContentCohort.response_col`. Same flag in
  `fit_swr_contrast_content.main` (`_run(..., response_col=...)`). `freeze_common_cohort(..., drop_zero_response=True)`
  already removes events whose fraction is NaN/zero.
- **Reporting on the event scale.** In `summarize_comparison` keep `delta_aic_*` and the animal
  medians; drop `lrt_significant` and `child_better_aic` from `print_model_results` headline output
  (they remain in the per-session frame), and add `n_sessions_child_better_aic_gt2` only when
  `estimator == "dirichlet_multinomial"` or the response is a fraction (event-scale llf). Record the
  estimator string in every results frame and in `model_comparison.pkl`.
- **Event-length control in BASEc.** In `fit_swr_contrast_content._assemble_arm` (next to `_TSC`,
  221-222) build
  `log_n_nonlocal_bins_z = attach_standardized(ev, ["log_n_nonlocal_bins"], support_label="fit_cohort_all_rows")`
  from `np.log(ev["n_nonlocal_bins"])` and the products
  `log_n_nonlocal_bins_z_x_is_current_arm`, `log_n_nonlocal_bins_z_x_is_prev_arm`; patch
  analogs in `_assemble_patch`. In `model_specs.py` add both arm products and `_TSC` to `BASEc`
  (1106-1112) so every C/CS rung carries them (resolves the CS2/CS4/CS5 inconsistency), and the
  patch product plus the patch time term to `BASEc_PATCH`. Register the products in
  `DERIVED_COVARIATE_SOURCES` with source atoms `["n_nonlocal_bins", "is_current_arm"]`, and add
  `n_nonlocal_bins` to `definitions.py` (grain event, clock EVENT_RESPONSE, PRE_OUTCOME, producer
  `replay.tables_event.build_event_table`). Replace the runner-local `_z` (131) and its uses (183)
  with `attach_standardized`, persisting the reference as `<artifact-stem>.z_reference.json`.
- **Outcome strata.** Add `--outcome-stratum {all,rewarded,omitted}` to
  `fit_swr_contrast_content.main`; `rewarded` keeps `ev["is_rewarded_outcome"] == 1`, `omitted`
  keeps `== 0` (events with NaN outcome are excluded from both strata, and their count by SWR class
  is logged). Output files gain a `_rewarded` / `_omitted` suffix. The stratum is a cohort
  selection, so the timing gate at 595-600 is unchanged.
- **Leaf-only value chain.** Add `--leaf-only` to `fit_replay_content_models.py`: in
  `_load_and_filter_cohort` after the at-well filter (1299) keep `candidate_arm < LEAF_ONLY_N_CATEGORIES`,
  set `n_categories = 6` in `_fit_content_ladder` (1402), and map every rung's predictors through
  `leaf_only_predictors`. Output dir suffix `_leaf` (design section 4). Document in
  `model_specs.py:1326-1340` that Mv/Mu/Mvu are now compared to leaf-only M0–M4 fitted in the same
  run (`--leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu`).
- **Refit.** Back up `models/`, `models_patch/`, `models_swr_{T,F}/`, `models_patch_swr_{T,F}/`,
  `models_event_grain/`, and `models/swr_contrast_content_*` into `_pre_fix_backup_<YYYYMMDD>/`.
  Smoke with `--max-per-animal 1`, then in `tmux` session `content_refit` run: default arm ladder
  (restores `models/`), default patch ladder, `--swr-subset T|F` for both, the leaf-only belief run,
  and `fit_swr_contrast_content.py` for `all`, `rewarded`, `omitted`. Write provenance sidecars.
- **Before/after table.** Extend `scripts/qa/compare_rate_fits.py` (phase 1) into
  `scripts/qa/compare_fits.py` handling content frames: per (model, term) median coefficient, median
  `bse` and `bse_cluster`, animal sign counts, old vs new; save to
  `docs/audits/<date>-content-refit-comparison.md`.
- **Docs.** `docs/models_table.md:9-43` conventions: response = per-event fraction, SEs =
  event-clustered, note the count response is retained for the DM comparison; `:155-196` add
  `Mu`/`Mvu` and the `--leaf-only` chain. `docs/design/content_model_design.md`: one section on why
  the response changed (cite the review numbers). `covariate_dictionary.md`: entries for
  `patch_fraction`, `log_n_nonlocal_bins_z_x_*`.

## Deliberately not in this phase

- The Dirichlet-multinomial fitter and the three-way comparison (phase 2b); this phase produces the
  fraction fits that comparison consumes.
- Adding `is_rewarded_outcome × role` as a predictor: rejected in favor of strata because the
  timing registry would need a new availability class; revisit only if strata are underpowered.
- Ripple-power / session-time covariates (open in the review, not scheduled).
- Claims-table rewrite (phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_fitting.py::test_fraction_response_null_llf_is_event_scale` | with fractions summing to 1, `llf_null == -n_events * log(n_categories)` |
| `test_fitting.py::test_clustered_se_matches_model_se_for_one_draw` | on `simulate_with_event_covariates(L_e=1)`, `bse_cluster` within 5 % of `bse` |
| `test_fitting.py::test_clustered_se_matches_sampling_sd_for_sticky_bins` (slow) | on `simulate_autocorrelated_counts(rho=0.95)` converted to fractions, 200 replicates: `bse_cluster` within 20 % of the empirical SD of `coef` |
| `test_replay_tables.py::test_patch_fraction_sums_to_one` | `patch_fraction` sums to 1 per event with real-arm bins; NaN otherwise |
| `test_replay_content_model_selection.py::test_fraction_recovers_planted_effect` | planted β recovered within 3 clustered SE on fraction response |
| `test_fit_replay_content_models.py::test_subset_models_do_not_write_canonical_dir` | `--models M4,Mv` resolves to `models_M4-Mv`; writing to `models/` raises |
| `test_fit_replay_content_models.py::test_leaf_only_uses_six_categories` | `n_categories == 6`, arm FE 6/7 absent from predictors |
| `test_swr_contrast_content.py::test_outcome_stratum_selects_rows` | rewarded stratum contains only `is_rewarded_outcome == 1` events; NaN-outcome events excluded from both |
| `test_covariate_timing.py::test_length_products_resolve` | new product columns resolve to PRE_OUTCOME atoms |
| Refit smoke (slow) | all content runners complete with `--max-per-animal 1`; provenance sidecars present; comparison table generated |

## Fixtures

- `simulate_with_event_covariates` and `simulate_autocorrelated_counts` (existing).
- A small event-arm fixture with `n_nonlocal_bins` varying 10–200 and both outcome classes
  (extend the existing fixture in `tests/unit/test_replay_tables.py`).
- Real-data smoke gated on `PROCESSED_DATA_DIR`, marked `slow`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the runner-local `_z`; the count response is no longer the default anywhere).
- User-facing documentation listed as tasks is updated, not deferred.
