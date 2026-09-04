# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Source of the findings: `.claude/docs/reviews/2026-09-03-regression-pipeline-review.md`
(finding IDs A1–A7, B1–B9, C1–C6, D1–D9, E1–E9 are cited by phase files).

## Current codebase integration points

Rate families (phase 1, phase 3):

- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:135` — `fit_per_bin_rate_model`
  drops NaNs against the spec's own columns; becomes a fail-loud check that the design was frozen upstream.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:235` — `cross_validate_per_bin_rate_model`
  repeats the spec-specific dropna; same change.
- `src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:103` — `fit_per_stop_rate_model` spec-specific dropna; same change.
- `scripts/rate/fit_per_bin_rate_models.py:108-168` — `_fit_one_session` builds one design per population and loops the ladder; freeze goes between the two.
- `scripts/rate/fit_per_stop_rate_models.py:69-156` — `fit_per_stop_rate_hierarchy`; freeze per session before the model loop.
- `scripts/rate/fit_swr_contrast_rate.py:152-197` — `_fit` (`fit_status(res, "glm")` at 182 lacks `predictor_sd`); `:200-241` `_fit_one_session`; `:244-291` `_load` (qvals inner join, no location/outcome gate).
- `src/continuum_swr_replay/analysis/replay/model_comparison.py:53-56, 84-97` — equality guard on `n_events` only.
- `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:117-127` — `unstable = SE > 10|coef|` drop.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:783-935` — `aggregate_to_timebins`; onset binning at 876-882.
- `src/continuum_swr_replay/analysis/replay/tables_stop.py:136-147` — stop reward covariates frozen at `bout_starts`.
- `src/continuum_swr_replay/analysis/replay/covariates/context.py:615-624` — `_goal_arrival_time` / `_last_reward_time` written per timebin; no outcome-time twin.

Content family (phase 2a, 2b):

- `src/continuum_swr_replay/analysis/replay/fitting.py:83-243` — `fit_conditional_poisson` (response column parameter at 86; fit at 182; output dict 218-243).
- `src/continuum_swr_replay/analysis/replay/tables_arm_patch.py:163-182` — `arm_fraction` already persisted (count / real bins); patch builder at 227-280 has no fraction column.
- `scripts/content/fit_replay_content_models.py:552-662` (`fit_model_per_session`), `:1113-1196` (`_resolve_run_config`, output dir), `:1202-1373` (`_load_and_filter_cohort`, at-well filter at 1296-1299, `freeze_common_cohort` at 1312), `:1376-1440` (`_fit_content_ladder`), `:824-895` (`_publish_staged_directory`, whole-dir replace).
- `scripts/content/fit_swr_contrast_content.py:120-231` (`_assemble_arm`, fit-time `_z` at 131, `_TSC` at 221-222), `:281-343` (`_fit_session`), `:366-452` (`_run`, leaf-only branch at 391-398), `:584-700` (`main`, timing gate at 595-600).
- `src/continuum_swr_replay/analysis/replay/model_specs.py:1106-1113` (`BASEc`, `_TSC`), `:1114-1257` (arm CS ladder + key terms), `:1341-1357` (leaf-only rungs), `:277-352` (`MODEL_HIERARCHY`), `:109` (`BELIEF_DEPENDENT_CONTENT_MODELS`).
- `src/continuum_swr_replay/analysis/replay/covariate_timing.py:64-166` — `DERIVED_COVARIATE_SOURCES`; new product columns must be registered here.
- `src/continuum_swr_replay/analysis/replay/covariates/definitions.py` — catalog; new atoms must be added or `assert_predictor_timing` rejects them.
- `src/continuum_swr_replay/analysis/replay/simulation.py:307` — `simulate_with_event_covariates` (multinomial simulator to extend with a Dirichlet stage).

Provenance / tables (phase 3):

- `src/continuum_swr_replay/analysis/replay/prepare_data.py:1019-1020` — `load_contingency`; no `qvals_context_time == stop_start` check.
- `src/continuum_swr_replay/analysis/behavior/qvals_table.py:759-856` — `_broadcast_trial_covariates_to_stops` writes `qvals_context_time` and `outcome_known_time` per stop.
- `src/continuum_swr_replay/analysis/replay/_tables_common.py:206-247` — `_STOP_TABLE_BUILD_COLUMNS` / `_STOP_TABLE_COLUMNS` (new stop columns go here).
- `scripts/decoding/regenerate_stop_table.py` — rebuilds `stop_table` alone with the SWR split; `stop_id` unchanged.

Readability (phase 5a, 5b):

- `src/continuum_swr_replay/analysis/replay/model_specs.py:127-186, 200-212, 383-400, 469-480, 905-940, 1095-1110, 1264-1271` — narrative blocks.
- `src/continuum_swr_replay/analysis/replay/tables.py` (58 lines) — re-export shim; importers `prepare_data.py`, `analysis/replay/__init__.py:101`, `visualization/session_tables.py`, `scripts/decoding/regenerate_stop_table.py:32`.
- `src/continuum_swr_replay/analysis/replay/covariate_registry.py:165-1027` — rename ledger; `:1030-1090` derived views; `:1426-1540` stamp/require.
- `src/continuum_swr_replay/analysis/replay/versions.py:40-66` — `assert_versions_compatible`, no production caller.
- `src/continuum_swr_replay/analysis/replay/_helpers.py` — `immobility_bout_starts` (104, dead), `aggregate_to_timebins` (783), `raised_cosine_history_basis` (752), `attach_is_swr_to_onsets` (574), `POPULATIONS` (939), `add_arm_fixed_effects` (514), `_SESSION_COLS` (27).
- `src/continuum_swr_replay/session_keys.py:25-28` — canonical `SESSION_COLUMNS` / `SESSION_COLUMNS_DAY`; local duplicates at `retrospective.py:95`, `timebin_tables.py:79`, `_helpers.py:27`, `table_contracts.py:22`, `scripts/content/fit_replay_content_models.py:134`, `prepare_data.py:456`.
- `is_stay` implementations: `tables_event.py:284`, `tables_stop.py:292`, `shuffles.py:325`, `analysis/contfrag/tables.py:422, 913`.

Untouched: decoding (`scripts/decoding/run_decoding_bandit.py`, `analysis/hippocampus/data_aggregation.py`), event detection and classification (`analysis/hippocampus/event_table.py`, `event_detection.py`, `event_immobility.py`), the qvals ingestion adapter, ContFrag builders except where a shared helper moves, visualization.

## Scope and dependency policy

### Goals

- Every ladder comparison (ΔAIC, LRT, CV) is made on identical rows within a session, and for the
  shipped rate ladders those rows are every stop with positive immobility (no history-based drop).
- Reward history enters the rate models as a smooth three-feature kernel, with no window cutoff
  and no decay constant to justify.
- The shipped content estimator counts each event once; per-session uncertainty is event-clustered; ladder selection is on an event scale.
- The SWR-vs-non-SWR content contrasts control event length and are reported within outcome strata.
- A second content estimator (Dirichlet-multinomial) with an explicit concentration term is validated on simulation and compared with the shipped one on the same cohort.
- Family B carries an arrival-anchored time course and per-bin reward lags that respect the outcome boundary; fragmentation is documented.
- Every fitted artifact records the SHA-256 of the tables it was fit on; readers refuse or warn on mismatch.
- Docs and the claims table quote only numbers produced by the refits, with current predictor names.
- The regression pipeline reads as a catalog plus builders plus fitters, without migration ledgers, shims, or historical narrative in code.

### Non-Goals

- Re-running the decoder or changing event detection, thresholds, or the immobility rule.
- Redefining the stop as a well visit (renumbers `stop_id`; rejected at planning).
- A hierarchical/mixed model across animals (open question 3; report animal-level summaries only).
- Changing behavioral-model parameters (Beta prior/decay) or the qvals ingestion adapter.
- Renaming persisted column values or names that would force a full table regeneration (`in_transit`, `is_prev_arm`); listed as revisit triggers in phase 5a.
- Refitting the three sessions blocked by the NWB checksum problem.
- ContFrag ladders beyond keeping shared helpers importable.

### Dependency policy

No new dependencies. The Dirichlet-multinomial fitter uses `jax` (already a base dependency, CPU build) and `scipy.optimize`. No GPU is required for any phase.

## Metrics

- Phase 1: every persisted per-bin and per-stop fit has `n_bins`/`n_stops` equal across all rungs of a session × population (unit test on real pickle, formerly xfail), and the shipped ladder's cohort audit reports zero `n_drop_common_cohort_other`. `swr_contrast_rate_audited.pkl` has ≥ 1 row per (model, population, term) that is identified in ≥ 80 % of sessions.
- Phase 2a: per-session SE ratio (new / old) on `reward_fraction_candidate_arm_last10_visits` in the eight sessions listed in the review is within 4–8×; median ΔAIC(M3 vs M2) per session is within [−5, +10] rather than [10, 317].
- Phase 2b: on simulated DM data with φ ∈ {1, 3, 10, 100}, β recovered within 3 SE in ≥ 95 % of replicates and log φ within 0.3; the count-multinomial fit on the same data reports SEs smaller than the empirical sampling SD by the expected factor.
- Phase 3: `assert_qvals_aligned` passes on the live tables; every `models*/` output has a provenance sidecar whose table hashes match the live parquet files; arrival-anchored `M5a` KS-within-band is reported alongside `M5_time`.
- Phase 4: `grep` for the retired names listed in phase 4 returns zero hits outside `docs/design/model_history.md` and the registry.
- Phase 5a/5b: one real session rebuilt through `prepare_data` is byte-identical to the frozen golden baseline (`scripts/qa/compare_to_golden_baseline.py`); unit suite passes; `model_specs.py` < 900 lines; `covariate_registry.py` < 400 lines; `_helpers.py` < 400 lines.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The reward-history reparameterization changes the shipped Family-A/B models, so old and new coefficients are not term-for-term comparable | Phase 1 compares the joint reward step (ΔAIC vs parent) and the time-course KS before/after rather than lag coefficients; the fitted kernel is plotted from the three basis coefficients. |
| The structural indicator absorbs the epoch prefix and could mask a genuine early-session rate effect | It is a nuisance term on 1.3 % of stops; the time-since-arrival clock (phase 3) carries any real early-session structure. |
| Fraction response changes headline content coefficients | Phase 2a records the old-vs-new per-term medians in a comparison CSV before publishing; phase 2b's three-estimator report quantifies which conclusions are estimator-dependent. |
| DM concentration φ unidentified in small sessions | Per-session φ shared across events with an `is_swr` modifier; if `n_events < 40`, fix φ to the animal-pooled estimate; both paths reported. |
| Arrival-anchored spline extrapolates for late fragments | Bins with `time_since_last_well_arrival_s > 8 s` are dropped for the arrival variant only, and the retained share by SWR class is audited. |
| Refits take hours and collide with interactive use | Smoke on `--max-per-animal 1` first, extrapolate, run in `tmux`, document session names in the phase files. |
| Readability refactors silently change table values | Every 5a/5b PR must pass `compare_to_golden_baseline.py` on one real session and the full unit suite before merge. |

## Rollout Strategy

Replace in place, fail loud. No deprecation window and no compatibility aliases (matches the
finish-covariate-rename decision). Old fitted artifacts are moved to a dated
`_pre_fix_backup_<YYYYMMDD>/` directory inside `Processed-Data/replay_content/` before each refit
so before/after comparisons are possible; they are never read by production code. Regenerated
data and fits are not committed. Each phase merges independently; phases 2a and 3 each refit
their families, so the claims table is rewritten only in phase 4 after both have landed.

## Decisions taken at planning (2026-09-04)

1. **Reward-history block.** The ten lag dummies and the window-10 fraction are retired from every
   ladder. The only reward-history block is the three normalized raised-cosine features over the
   last ten known outcomes ([contract](shared-contracts.md#reward-history-kernel)). Nothing is
   dropped: undefined features are zero-filled and a single structural indicator
   `reward_history_incomplete` (fewer than two known outcomes; an unbroken ~15 s prefix of every
   epoch, 1.3 % of stops) enters as a nuisance main effect. Rationale recorded: a tenth-outcome
   cutoff was arbitrary and hard to justify in a paper; an exponential kernel needs a decay
   constant; a two-bump basis is near-collinear; the three-bump basis expresses a non-monotone
   kernel with two more parameters than the exponential and is defined from the first outcome for
   two of its three features.
2. **Fraction response.** `arm_fraction` (hard MAP bins) is the shipped response; `arm_fraction_posterior`
   is a sensitivity in the phase 2b comparison (r = 0.963 with the hard fraction; 18 % of events
   are one-hot only by argmax).
3. **Population inference.** Sign consistency over the five animals plus the animal-level
   bootstrap in phase 2b. No random-effects meta-analysis and no hierarchical GLM in this plan.
4. **Arrival-anchored spline.** Window [0, 8] s, interior knots 1, 2, 4, 6 s (87.6 % of at-well
   events retained, 88.7 % SWR / 86.8 % non-SWR; median event 3.7 s after arrival).

## Open Questions

1. Whether the `onset_lag` dummy self-history rungs survive is decided inside phase 3 from the
   refit CV status (see that phase's task list); not open for the planner.

## Estimated Effort

| Phase | Source LOC changed (est.) | Tests added/changed | Refit wall-clock (smoke → full) |
| --- | --- | --- | --- |
| 1 | ~350 | ~8 tests | Family A minutes; Family B and SWR-rate hours with `--workers 16` |
| 2a | ~500 | ~10 tests | content ladders: hours |
| 2b | ~600 (new module + simulator + comparison script) | ~8 tests | comparison: hours |
| 3 | ~400 | ~8 tests | stop-table regen + Family B refit: hours |
| 4 | docs only | 1 scan test | none |
| 5a | −600 net | ~6 tests changed | none (parity check only) |
| 5b | ~800 moved, −900 deleted | ~10 tests changed | none (parity check only) |
