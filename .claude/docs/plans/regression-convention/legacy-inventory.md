# Legacy inventory

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contract C13](shared-contracts.md#c13-no-legacy-left-behind)

Lead (2026-11-20): "Let's make sure to cleanup any legacy code." This file lists, by the phase that removes
it, every piece of legacy found in the areas this plan touches, searched in the `figure04-behavior-models`
checkout at `181d7f22` (src, scripts, tests, docs). Line numbers are of that checkout; re-find a name with
`git grep -n -w` before editing, since earlier phases move lines.

Classes:

- **REMOVE**: replaced by this plan; the phase named removes it in its "Retire what this phase replaced"
  task (C13). No question.
- **ASK**: nothing in this plan replaces it, or it serves an analysis outside the plan. A question for the
  lead, with a recommendation (the lead's rule: "an analysis nothing cites is a question for the lead, not a
  removal"). Collected in [the ASK list](#ask-list).
- **KEEP**: looks old, but is in use or is a guard; the reason is given.

Stored outputs in Processed-Data are never deleted; they are only no longer read. No value of
`paper/reported/behavior.json`, `mpfc.json` or `events.json` was cited by `paper/*.tex` on 2026-11-20
(only `survey` values are); each phase checks again before it removes a value group (C13).

"Callers" are the importers or readers found by `git grep` outside the item's own module and its tests.

Contents: [Phase 1](#phase-1) · [Phase 3](#phase-3) · [Phase 4](#phase-4) · [Phase 5a](#phase-5a) ·
[Phase 5b](#phase-5b) · [Phase 6a](#phase-6a) · [Phase 6b](#phase-6b) · [Phase 7](#phase-7) ·
[Phase 8](#phase-8) · [Phase 11](#phase-11) · [Cohort selectors and stop clocks](#cohort-selectors-and-stop-clocks) ·
[Phase 12](#phase-12) · [Scripts: exploratory and one-time](#scripts-exploratory-and-one-time) ·
[Docs](#docs) · [KEEP](#keep) · [ASK list](#ask-list) · [Counts](#counts)

## Phase 1

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 1-1 | `analysis/replay_models/rate/cross_history.py`, `scripts/replay_models/rate/fit_cross_history_rate.py`, `tests/analysis/replay_models/rate/test_cross_history.py`, run `rate.cross_history` (`scripts/workflows/run_refits.py:173`) and the doc mentions listed in Phase 1's first task | none reports it | nothing (lead: "ok delete") | REMOVE (in progress on `regression-fitter`; the task stays as written) |

## Phase 3

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 3-1 | `table_contracts.require_visit_context_labels` skips a table with no `visit_context` ("built before the label existed"; labels branch `9b460a90`) | every contract check | the label is required | REMOVE (the coordinator removes it on the labels branch; if master still has it, Phase 3 removes it) |
| 3-2 | `event_tables/stop_table.py:287-294` (`add_nonlocal_swr_split`): an event table with no `stop_id` or `is_swr` gives zero SWR counts instead of failing | `prepare_data.py`, the table build | `require_columns` | REMOVE |
| 3-3 | `event_tables/stop_table.py:321-323`: missing count columns filled with 0 after a groupby | same | an explicit `reindex(..., fill_value=0)` of both columns (a zero-count session is real, not old) | REMOVE the presence test, keep the zero fill |
| 3-4 | `event_tables/event_table.py:405-415` (`_attach_trajectory_rollups`): emits NaN trajectory columns "when `dot` predates the augmentation" | `build_event_table` | every decode frame carries the columns since the R11 re-decode; fail when they are absent | REMOVE (confirm on one epoch of each animal first) |
| 3-5 | `event_tables/uncertainty.py:207-214`: `__all__` re-exports `nonlocal_weight_sum` "from summaries" | grep callers of `uncertainty.nonlocal_weight_sum` | import from `event_tables/summaries.py` | REMOVE the re-export |
| 3-6 | `event_tables/event_table.py:191` comment "matches the legacy `prepare_session` aggregation contract" | — | state the contract itself | REMOVE (comment) |
| 3-7 | `behavior/covariates/history/value_uncertainty.py:346-348` comment "a stop table lacking it predates the outcome clock" | — | — | REMOVE (stale comment; the column is required) |
| 3-8 | `behavior/covariates/history/reward_history.py:14-16`: re-export of `reward_basis_from_lags` from `behavior.history_basis` | `per_bin_design.py` and the stop-table builder | import from `history_basis` | REMOVE the re-export |
| 3-9 | `replay_models/model_specs.py:45-56`: constants re-exported from `behavior.design_constants` "so `from ...model_specs import RATE_BIN_S` keeps working" | every rate module | import from `design_constants` | REMOVE the re-export (move the imports in the same phase; touches many files, so do it as its own commit) |
| 3-10 | `behavior/covariates/clocks.py:37-50` (`trial_outcome_times`): "Older or manually constructed frames lack" `outcome_known_time`, so `end_time` is used; the same fallback at `behavior/covariates/context.py:52`, `:567` | the table build | every trial frame carries `outcome_known_time` (`prepare_data.py:200-206` re-fetches a cache entry that lacks it); raise when absent | REMOVE (update the synthetic test frames that omit the field) |
| 3-11 | `scripts/tables/convert_tables_to_parquet.py`, `regenerate_stop_table.py` and the legacy-file warning of `prepare_replay_content_data.py:412-421` | see S-1, S-2, S-5 | — | REMOVE |

## Phase 4

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 4-1 | The event-evidence `trial_folds` and its helpers on master after the other session's merge (`analysis/event_evidence/tetrode_holdout.py`, from `swr-trial-folds`) | the held-out scoring | `analysis/regression/folds.py` (Phase 2); one implementation remains, the parity test becomes a test of it | REMOVE the duplicate |
| 4-2 | Any occupancy or position filter in a Figure 2 or 3 producer that duplicates the selection (none found in `scripts/figures/generate_figure02.py`, `generate_figure03.py`, `reported/events.py`, `scripts/event_properties/`, `scripts/event_evidence/`; recheck after the merge) | — | `select_visit_context` | REMOVE if found |

## Phase 5a

The per-bin rate ladder (Family B) and what serves only it.

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 5a-1 | `replay_models/rate/per_bin_rate_ladder.py` (whole; `fit_per_bin_rate_models` `:254`, `per_bin_cohort_records` `:388`) | `scripts/replay_models/rate/fit_per_bin_rate_models.py` | `rate/visit_rate.py` | REMOVE |
| 5a-2 | `replay_models/rate/per_bin_rate_model.py` (whole; `fit_per_bin_glm` `:293`, the statsmodels per-bin fit; `cross_validate_per_bin_rate_model` `:450` with its shuffled `KFold` `:509`; `per_bin_rate_gof` `:589`; `apply_arrival_window` `:157`) | per-bin ladder, `rate/swr_contrast_rate.py`, `rate/uncertainty_rate.py` | shared fitter | REMOVE after 5b removes the last two callers; if 5b has not merged, in 5b. Phase 1's check 1 gets its fixture first (C13) |
| 5a-3 | `MODEL_PER_BIN_RATE` (`model_specs.py:879-1083`) and its retired-rung comments (`:903-921`, `:657`) | ladder, plotters, `reported/behavior/rate.py` | the visit models' term lists | REMOVE |
| 5a-4 | `PerBinRateModelSpec.switch` (`model_specs.py:714-719`, the refusal `:806-813`) and `tests/analysis/replay_models/test_switch_flag_retired.py` | none sets it | — | REMOVE with the class (Phase 11, the last user) |
| 5a-5 | `rate/rate_fit_statistics.py` (whole) | `reported/behavior/rate.py`, the per-bin plotters | `rate/visit_rate_statistics.py` | REMOVE |
| 5a-6 | `rate/stop_statistics.py` (whole: stop cohort counts, dwell, onset time course within 4 s) | `reported/behavior/stops.py`, `plot_per_bin_rate_summary.py` | visit-grain descriptive values (Phase 3's `visit_clock_curves`) and 5a's values | REMOVE |
| 5a-7 | `reported/behavior/rate.py`: `per_bin_values` (`:81`), `per_bin_detail_values` (`:661`), `cohort_audit_values` (`:1170`, per-bin part), the constants `:57-79`; `reported/behavior/stops.py` (whole); the in-transit block of `reported/behavior/__init__.py:156-190`; `BehaviorInputs.per_bin`, `per_bin_all_stops`, `per_bin_in_transit`, `per_bin_audit`, `onset_table` (`reported/behavior/inputs.py:53-69`) | `summarize` | `reported/behavior/visit_rate.py` | REMOVE |
| 5a-8 | Runners and runs: `scripts/replay_models/rate/fit_per_bin_rate_models.py`; runs `rate.per_bin.at_well`, `rate.per_bin.all_stops`, `rate.per_bin.in_transit`, `rate.fig.per_bin_ladder`, `rate.fig.per_bin_summary`, `rate.fig.model_checking` (`run_refits.py:162-165`, `:178-180`) | `RUNS` | `rate.visit.*` | REMOVE |
| 5a-9 | Plotters `scripts/replay_models/rate/plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_model_checking.py` | runs above | Figure 4 (Phase 10) and 5a's spot-check plots | REMOVE |
| 5a-10 | Tests: `tests/analysis/replay_models/rate/test_per_bin_rate_ladder.py`, `test_per_bin_rate_model.py`, `test_per_bin_terms_frozen.py` (the per-bin part), `test_rate_fit_statistics.py`, `test_stop_statistics.py`, `test_rate_cohort_audits.py` (per-bin part); `tests/scripts/replay_models/rate/test_fit_per_bin_rate_models.py`, `test_plot_per_bin_rate_models.py`, `test_plot_per_bin_rate_summary.py`; `tests/reported/behavior/test_rate.py` (per-bin part), `test_stops.py` | — | 5a's tests | REMOVE |
| 5a-11 | Question entry 3.3 (`reported/questions.py:333`) | its values | 3.12 (Q2) and the kernel of model D | REMOVE (mark `dropped`) when its last value leaves (5b) |
| 5a-12 | Docs: `docs/rate_models.md:106-418` (Family B), `:491-615` (its point-process checks); `docs/models_table.md:213-264`; `docs/development.md:316-332` (the two families; rewrite to the visit model); `docs/covariate_dictionary.md:232-237`, `:453-499` (tc clock, per-bin reward history); `docs/known_issues/swr_mua_count_conventions.md:53-58` (the Family B bin-drop convention); `docs/README.md:186` row `models_per_bin_rate/` | — | `docs/regression_models.md` | REMOVE |

## Phase 5b

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 5b-1 | `rate/per_stop_rate_model.py` (whole; `sm.GLM` at `:195`, `:295`; `OFFSET_COL` `:331`; the absent-column allowances `:341`, `:351`), `MODEL_PER_STOP_RATE` (`model_specs.py:490`), `scripts/replay_models/rate/fit_per_stop_rate_models.py`, runs `rate.per_stop.*`, `per_stop_values` (`reported/behavior/rate.py:1228`), `BehaviorInputs.per_stop`, `per_stop_audit` | `summarize` | `rate/visit_total_rate.py` | REMOVE |
| 5b-2 | `rate/swr_contrast_rate.py` (whole), `rate/post_outcome.py` (whole; only the two contrast and uncertainty modules call it), `SWR_CONTRAST_RATE_G_SPECS`/`_S_SPECS` (`model_specs.py:1428-1429`), `SWR_CONTRAST_RATE_STOP_COVARS` (`:1351`), `_CONTRAST_HISTORY_LAGS` (`:1357`), `_g_specs`, `_s_specs`, `_BLOCKS`, `_block_formula`, `_block_cols` (`:1294-1427`), `scripts/replay_models/rate/fit_swr_contrast_rate.py`, runs `rate.swr_contrast.*`, `rate_contrast_audit_values` (`:1131`), `rate_contrast_separation_values` (`:1064`), `BehaviorInputs.rate_contrast` | `summarize`, `scripts/figures/plot_swr_contrast_coefs.py` (rate panels) | `rate/visit_rate_covariates.py` | REMOVE |
| 5b-3 | `rate/uncertainty_rate.py` (whole), `MODEL_UNCERTAINTY_RATE`, `MU_UNCERTAINTY_*` (`model_specs.py:1084-1105`), `scripts/replay_models/rate/fit_uncertainty_rate.py`, `plot_uncertainty_rate.py`, runs `rate.uncertainty.at_well`, `rate.fig.uncertainty` | none in `reported/` (the runner logs medians) | the MU covariates on model A | REMOVE |
| 5b-4 | `cohort_audit_values` (rest), `replay_models/cohort_audit.py` (whole) and its rate tables | rate and content ladders, `swr_power.py` | the fit-row record's cohort counts (C7) | REMOVE with its last caller (7) |
| 5b-5 | Tests: `tests/analysis/replay_models/rate/test_per_stop_rate_model.py`, `test_swr_contrast_rate.py`, `test_uncertainty_rate.py`, `test_post_outcome.py`; `tests/scripts/replay_models/rate/test_fit_per_stop_rate_models.py`, `test_swr_contrast_rate.py`, `test_fit_uncertainty_rate.py`, `test_plot_uncertainty_rate.py`; `tests/reported/behavior/test_rate.py` (rest) | — | 5b's tests | REMOVE |
| 5b-6 | Docs: `docs/rate_models.md:39-105` (Family A), `:419-490` (MU, both families); `docs/models_table.md:265-285`, `:349-366`, `:536-668` (G, S); `docs/covariate_dictionary.md:239-279` (rate rows of the per-family matrix, decision D3); `docs/design/cohort_and_timing_rules.md:410-441` (descriptive rungs, R1) | — | `docs/regression_models.md` | REMOVE |

## Phase 6a

The content ladders (arm, patch, contrast, leaf) and the hippocampal non-local event source's use of them.

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 6a-1 | Content specs in `model_specs.py`: `MODEL_HIERARCHY` (`:247`), `_M0_TO_M4`, `MODEL_PATCH` (`:367`), `MODEL_PATCH_SWITCH` (`:435`), the content `is_swr`-contrast specs (`SWR_CONTRAST_CONTENT_ARM_LADDER` `:1524`, `_PATCH_LADDER` `:1708`, the `BASEc*` sets `:1740-1743`, `LEAF_ONLY_CONTRAST_RUNGS` `:1774`, and their helpers `:1440-1794`, with the retired-rung comments `:194`, `:344`, `:1704-1706`), `leaf_only_predictors`, `rung_predictors`, `restrict_to_leaf_arms`, `leaf_only_cohort` (`:1795-1945`), `ComparisonRung` and `resolve_comparison_rungs` (`:1946-2071`), `default_content_model_names` (`:175`) | content runners, `reported/behavior/content.py` | `content/role_content.py` | REMOVE, except what ASK-1 (ContFrag) keeps: `MODEL_HIERARCHY_SWR`/`_MUA`, `MODEL_PATCH_SWR`/`_MUA` (`:363-423`) serve the ContFrag event sources |
| 6a-2 | `content/session_fits.py` (whole: `fit_model_per_session` `:125`, `fit_conditional_logit_robustness` `:287`, `fit_contrast_ladder` `:664`, `fit_content_ladder` `:837`, the patch fits `:963-1069`) | `scripts/replay_models/content/fit_replay_content_models.py`, `fit_swr_contrast_content.py` | role-content runner | REMOVE (ASK-1 for the ContFrag path) |
| 6a-3 | `content/cohort.py` (whole: `select_content_cohort` `:263`, `prepare_canonical_fit_request` `:99`, `content_cohort_record` `:511`, `contrast_fit_cohort` `:683`) | content runners, `dirichlet_multinomial.py`, `model_checking.py` | the selection function and the role design's cohort step | REMOVE (Dirichlet-multinomial caller moves in 6b) |
| 6a-4 | `content/model_comparison.py` (whole; the legacy-metadata branch `:186-200`), `content/model_checking.py` (whole), `content/fit_outputs.py` (whole: output subdirectories of the old ladders; `frame_estimator` `:290` tolerates frames without `estimator`) | runners, `reported/behavior/inputs.py`, `visualization/replay_content_plots.py`, `scripts/checks/freeze_regression_baseline.py` | the fit-row record and C9's layout | REMOVE (`resolve_contfrag_paths` `:38` moves to `analysis/contfrag/` if ASK-1 keeps ContFrag) |
| 6a-5 | `replay_models/contrast_aggregation.py` (whole: the per-term gate `MAX_UNIDENTIFIED_SHARE`/`MIN_SESSIONS_PER_TERM` `:94-97`, the older-bundle branches `:264-300`, `:386-400`, `:432`, `:465`) | `coefficient_statistics.py`, `rate_fit_statistics.py`, `swr_contrast_rate.py`, `estimator_comparison.py`, `session_fits.py`, `reported/behavior/__init__.py`, `content.py`, `common.py` | the support rule (C5) and `estimand_summary` | REMOVE in 6b (its last callers, the estimator comparison and `reported/behavior/common.py`, move there) |
| 6a-6 | `content/content_statistics.py`: `trunk_shares`, `trunk_share_statistics`, `current_patch_trunk_fraction` (`:47-196`) | `reported/behavior/content.py` (`trunk_values` `:194`) | the trunk role of the role model | REMOVE (descriptive trunk shares: ASK-6) |
| 6a-7 | `reported/behavior/content.py`: `_content_specs` (`:121`), `content_contrast_values` (`:326`), `breadth_derived_values` (`:426`, moves to 6b if it reads the Dirichlet-multinomial), `current_arm_paired_values` (`:260`), `leaf_values` (`:607`), `trunk_values` (`:194`, ASK-6); `BehaviorInputs.content_contrast_dir`, `leaf_dir`; `TAGS` rows `leaf_ladder`, `content_*` (`reported/behavior/common.py:20-35`) | `summarize` | `reported/behavior/role_content.py` | REMOVE |
| 6a-8 | Runners and runs: `scripts/replay_models/content/fit_replay_content_models.py` (hippocampal paths), `fit_swr_contrast_content.py`, `generate_replay_content_figures.py`; runs `content.arm_ladder.*`, `content.patch_ladder.*`, `content.swr_contrast.*` (`run_refits.py:185-216`) | `RUNS` | role-content runner | REMOVE (ASK-1 for the ContFrag runs) |
| 6a-9 | `scripts/figures/plot_swr_contrast_coefs.py` and `FIGURES["swr_contrast"]` (`reported/questions.py:133-136`) | values tagged `swr_contrast` | Figure 4 | REMOVE (its rate panels are already gone in 5b) |
| 6a-10 | `fitting.py`'s `fit_conditional_logit`, `fit_null_model_conditional_logit`, `conditional_logit_or_reason` (`:1148-1406`, the ConditionalLogit robustness path; legacy-input fallbacks `:1286-1292`, `:1314-1316`) | `session_fits.py` | none (the dominant-arm logit robustness is not in the convention) | REMOVE |
| 6a-11 | Recency within-event centering: `behavior/covariates/history/visit_recency.py:23` (`center_visit_recency_within_event`), `:98` (`spline_visit_recency_within_event`), `RECENCY_SPLINE_KNOTS`/`_BOUNDS`/`_COLUMNS` (`behavior/design_constants.py:66-73`), the columns `visit_recency_trials_event_leaf_centered*` (`candidate_arm.py` output block, `:200-230`) | content ladders, `content/simulation.py`, `pfc/marked_model_family.py`, `analysis/contfrag/tables.py`, `visualization/session_tables` | absolute recency smooth | REMOVE from replay_models; the columns and functions stay while ASK-1 (ContFrag) and Phase 8 (marked family) read them; Phase 8 moves the marked family to absolute recency; then REMOVE in Phase 12 if ASK-1 says ContFrag goes |
| 6a-12 | `event_tables/content_response.py`: `--response hard_count` "retired-count comparison" (`:17`, `RESPONSE_CHOICES` `:44`) | content runners, Dirichlet-multinomial | the count response is a named robustness estimator in 6b | KEEP the count column; REMOVE the retired-comparison wording and the runner flag in 6b |
| 6a-13 | Tests: `tests/analysis/replay_models/content/test_cohort.py`, `test_session_fits.py`, `test_model_comparison.py`, `test_content_statistics.py`, `test_cv_zero_count_filter.py`, `test_run_comparison.py`; `tests/analysis/replay_models/test_replay_content_*.py`, `test_recency_novelty_model_specs.py`, `test_swr_arm_baselines.py`, `test_audited_content_contrast.py`, `test_contrast_aggregation*.py` (6b), `test_model_specs.py` (content part); `tests/scripts/replay_models/content/test_fit_replay_content_models.py`, `test_swr_contrast_content.py`, `test_replay_content_runner_schema.py`, `test_runner_z_reference.py`, `test_provenance_labels.py`; `tests/reported/behavior/test_content.py` | — | 6a's tests | REMOVE (keep what ASK-1 keeps) |
| 6a-14 | Questions 3.1, 3.2, 3.6, 3.7 (`reported/questions.py:320-354`) | their values | 3.13 to 3.15, 3.17 | REMOVE (mark `dropped`) |
| 6a-15 | Docs: `docs/models_table.md:367-457`, `:669-833`, `:842-903`; `docs/design/content_model_design.md` (whole); `docs/covariate_dictionary.md` content rows of `:239-254`; `docs/README.md:64` and `:186-194` rows `models/`, `models_patch/`, `models_swr_{T,F}`, `models_swr_contrast/` | — | `docs/regression_models.md` | REMOVE (dated designs stay as records) |

## Phase 6b

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 6b-1 | Within-arm comparison: `content/uncertainty_content.py:705` (`within_arm_horserace`), `:853-897`; its script lines, figure panel, `horserace_values` (`reported/behavior/content.py:718`), `BehaviorInputs.horserace_summary` | `summarize`, `generate_uncertainty_content_figure.py` | Q5 values of 6a | REMOVE (lead: "ok delete") |
| 6b-2 | `content/estimator_comparison.py`: the old-design paths (`estimator_rows` `:109`, the `model_fallback` standard error `:125-181`, the percentile bootstrap) and its old-ladder inputs; `estimator_values` (`reported/behavior/content.py:495`); the `_dm` content-contrast runs `content.breadth_contrast.*`, `content.arm_ladder.dirichlet_multinomial` | `compare_content_estimators.py`, `summarize` | the comparison on the role design | REMOVE the old paths |
| 6b-3 | `content/dirichlet_multinomial.py`: its own cohort (`select_event_cohort` use), `concentration_cols_for_cohort` (`:865`), `animal_pooled_log_phi` (`:896`) if the role design does not use them | DM runner | the role design matrix | REMOVE what the new path does not call |
| 6b-4 | `contrast_aggregation.py` and `reported/behavior/common.py`'s gate note `_gate_note` (`:37`) | see 6a-5 | support rule | REMOVE |
| 6b-5 | Docs: `docs/design/content_estimator_comparison.md`; `docs/models_table.md:904-933`; `docs/README.md:65` | — | `docs/regression_models.md` | REMOVE (rewrite as the role-design comparison) |

## Phase 7

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 7-1 | `choice/choice_prediction.py` (whole: `build_choice_dataset` `:220` moves into `visit_choice.py` if reused, `fit_choice_model` `:442` with statsmodels `ConditionalLogit` `:468-520`, `predict_arm_fractions` `:67`, `build_residualized_choice_dataset` `:146`, `choice_swr_split_models_for_source` `:567`, `fit_choice_blocks` `:591`), `MODEL_CHOICE` (`model_specs.py:1251`), the choice block of `scripts/replay_models/content/compare_event_source_models.py`, `choice_values` (`reported/behavior/choice.py:41`), `BehaviorInputs.choice_fits` | `summarize` | `choice/visit_choice.py` | REMOVE (the rest of the event-source suite: ASK-3) |
| 7-2 | `choice/stay_leave.py`: `fit_is_stay_logistic_per_session` (`:405`, `sm.Logit` `:505`), `fit_is_stay_pooled_gee` (`:534`), `pooled_gee_role_summary` (`:819`), `cross_session_role_summary` (`:770`), `smallest_adjusted_p` (`:845`), `fit_stay_leave_roles` (`:687`), `fit_status_counts` (`:879`); `scripts/replay_models/choice/run_stay_leave.py`; runs `behavior.stay_leave.*`; `stay_leave_values` (`reported/behavior/choice.py:117`), `STAY_STRATA`; `BehaviorInputs.stay_leave_dir`; `scripts/checks/summarize_stage6_fdr.py` | `summarize` | `choice/visit_stay_leave.py` | REMOVE (`stop_replay_summary` `:151` moves to the visit grain if reused) |
| 7-3 | `coupling/swr_power.py` (whole: per-session z-scoring, `fit_power_model` `:210` by `lstsq`), `scripts/replay_models/coupling/fit_swr_power_model.py`, run `coupling.ripple_power`, `power_values` (`reported/behavior/coupling.py:29`), `BehaviorInputs.swr_power` | `summarize` | `coupling/visit_swr_power.py` | REMOVE |
| 7-4 | `fitting.py`: `fit_conditional_poisson` (`:677`) and its helpers (`:164-676`), `conditional_linear_predictor` (`:1012`), `fit_null_model_conditional` (`:1078`), `parse_unidentified_terms` (`:979`), `within_group_full_rank` (`:647`), `event_clustered_covariance` (`:180`), the diagnostics `:1440-1512`; then the module | `choice_prediction.py`, `dirichlet_multinomial.py` (`within_group_full_rank`), `model_checking.py`, `content/simulation.py`, `contrast_aggregation.py` | shared fitter; `within_group_full_rank` moves to `analysis/regression/` if the Dirichlet-multinomial still needs it | REMOVE after the C13 fixture for Phase 1's check 3 and check 1 (content part) |
| 7-5 | `replay_models/_fit_diagnostics.py` (statsmodels convergence and separation flags, the fallback `|coef| >= 10` rule `:51`, `:157`) | `coefficient_statistics.py`, `cohort_audit.py`, `fitting.py`, `contrast_aggregation.py`, `rate/duration_aware_history.py`, ... | the Newton decrement and support rules | REMOVE with its last caller (Phase 11) |
| 7-6 | `replay_models/common_cohort.py` (whole), `replay_models/cohort_audit.py` (whole), `replay_models/fit_comparability.py` (whole; AIC comparability of rungs) | per-stop, choice, content fits; `coefficient_statistics.py`, `estimator_comparison.py` | the cohort step and the fit-row record; information criteria between rungs are not used by the convention | REMOVE with the last caller (this phase, or 11 for `common_cohort` if the history runner uses it) |
| 7-7 | Tests: `tests/analysis/replay_models/choice/test_choice_prediction.py`, `test_predict_arm_fractions_nan.py`, `test_stay_leave.py`, `test_stay_leave_summaries.py`, `test_stop_replay_summary_patch_nan.py`, `test_fdr_nan_pvalue.py` (if stay-leave only); `tests/analysis/replay_models/coupling/test_swr_power.py`; `tests/analysis/replay_models/test_fitting.py`, `test_fit_error_handling.py`, `test_common_cohort.py`, `test_cohort_audit.py`, `test_fit_comparability.py`, `test_fit_diagnostics.py` (with their modules); `tests/scripts/replay_models/choice/test_run_stay_leave.py`; `tests/reported/behavior/test_choice.py`, `test_coupling.py` | — | 7's tests | REMOVE |
| 7-8 | Questions 3.4, 3.5, 3.8 (`reported/questions.py:338-358`) | their values | 3.16 (Q6) and the phase's descriptive values | REMOVE (mark `dropped`) |
| 7-9 | Docs: `docs/models_table.md:494-535`; `docs/design/cohort_and_timing_rules.md:190-195` (Stage 6); `docs/README.md` rows `models_per_event_swr_power`, `stage6_is_stay_*`; `scripts/README.md:90-91`, `:102-104` | — | `docs/regression_models.md` | REMOVE |

## Phase 8

From a search of `analysis/pfc/`, `reported/mpfc/`, `scripts/pfc/` and their tests.

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 8-1 | The statsmodels unit-fit backend: `FITTERS = ("jax", "statsmodels")` and `_check_fitter` (`analysis/pfc/fit.py:19-28`), the statsmodels branch of `fitter_provenance` (`:31-69`) and of `_fit_immobility_models` (`:207-289`, default `fitter="statsmodels"` at `:212`), `process_neuron(fitter="statsmodels")` default (`session.py:89`), the per-neuron branch of `_fit_session_neurons` (`session.py:1643-1720`), the fitter part of `_check_session_arguments` (`:1298-1325`), `run_session`'s "equivalence audit" docstring (`:874-889`), the runner flag `--fitter` with `choices=["jax"]` (`scripts/pfc/run_pfc_analysis_bandit.py:951-961`) | no production caller uses statsmodels; tests parametrize over both | the shared JAX fitter (one backend) | REMOVE |
| 8-2 | The batched ridge IRLS at fixed sigma: `DEFAULT_PENALTY_SIGMA` (`batched_glm.py:167`), `make_ridge_penalty` (`:677`), `implied_prior_sd` (`:740`), the statsmodels convergence rule and defaults (`:34-53`, `:153-157`, `:460-482`), `model_penalties` (`fit.py:112-138`), `fit_poisson_glms(penalty=None)` "reproduces statsmodels" (`:1652+`), `fit_poisson_designs` (`:1422`, tests only), `orthonormal_design_basis` (`:640`) | `session.py`, `scoring.py:202`, `marked_glm.py:27-36`, `:1409-1422`, `:1762`, `timing_calibration.py:34`, `:141`, `:175`, `load_population.py`, `generate_figure05.py:70`, `:540`, `:579`, `:878`, the runners | `analysis/regression/jax_poisson.py` with REML | REMOVE (fixture first, C13) |
| 8-3 | `penalty_sigma` parameters and flags: `session.py:94`, `:832`, `:886-889`; `--penalty-sigma` (`run_pfc_analysis_bandit.py:992`), `--penalty-sigma`/`--mark-penalty-sigma`/`--baseline-penalty-sigma` (`run_pfc_marked_kernels.py:746`, `run_pfc_marked_model_family.py:1201-1214`, `calibrate_pfc_kernel_timing.py:95`), `penalty_sigma_suffix` (`load_population.py:139-161`) | runners, Figure 5, tests | REML; a fixed penalty survives only as a named sensitivity model if the lead wants one (ASK-9) | REMOVE |
| 8-4 | Blocked folds: `heldout.assign_blocked_folds` (`:222`) and its callers `scoring.py:190`, `run_pfc_marked_kernels.py:560`, `run_pfc_marked_model_family.py:894`; `__init__.py:41` | — | trial folds (Phase 2) with a 250 ms margin | REMOVE |
| 8-5 | The stop cohort: `pfc/cohort.py:65` (`load_session_stop_cohort`), `stop_cohort`/`cohort_stops` in `session.py:829-872`, `event_sets.py:41`, `:64-68` ("reproducing the pre-cohort mask exactly"), `:98-108`, `:271`, `:310-313`, `:367-373`, `marked_glm.py:608-636`, `stop_context.py:545`, `:563-568`; `STOP_COHORTS` and `--stop-cohort all_stops` (`run_pfc_analysis_bandit.py:161`, `:923`); the `_allstops` bundle read by `generate_figure05.py:557`, `:861`; `measure_pfc_null_geometry.py:190-198` | 6 scripts, tests | in-visit bins and `select_visit_context` | REMOVE (`split_sorted_epochs`, `cohort.py:37`, KEEP: the epoch cohort) |
| 8-6 | Readers of older bundles in `load_population.py`: `RETIRED_SUFFIXES` (`:46-50`, `:173`), `LEGACY_FDR_TESTS` (`:127-136`), the `inference="legacy"` path (`:164-203`, `:327-350`, `:399-412`, `:527-529`, `:550-567`), "stale lag window from an older run" (`:421-439`); the matching writer guard `RETIRED_STATISTIC_SUFFIXES`/`_COLUMNS` and `drop_retired_statistics` (`run_pfc_analysis_bandit.py:187-188`, `:352-367`); the sidecar default "written before `event_set` existed" (`:344`) | `scripts/pfc/compare_pfc_cohorts.py:2604-2621` (`--before-dir`), tests | one bundle generation; an old bundle fails loudly | REMOVE (the shape check at `:421-439` becomes a raise) |
| 8-7 | `_validate_heldout_provenance` keys (`estimator="ridge_poisson_glm"`, `penalty_sigma_log_rate`, `n_folds`; `load_population.py:217-292`) and `reported/mpfc/population.py:79-80`, `:181-280` (`cohort_values` requires `stop_cohort == "at_well"`; `estimator_values` reports sigma, implied prior SDs, blocked folds) | `reported/mpfc`, tests (`tests/reported/mpfc/test_population.py:138-166`, `:235`) | the fit-row record (C7) | REMOVE / rewrite |
| 8-8 | `fit.py:292-377` (`_fit_diagnostics`, `_stack_fit_diagnostics`, the "fitter equivalence audit") and the runner's `--fit-diagnostics` | `session.py:1767` | the fit-row record | REMOVE |
| 8-9 | `scoring.maximum_likelihood_comparison` (`:385-465`, called at `:280`): an unpenalized refit to show where the fixed prior binds | `score_session` | REML reports edf; no fixed prior to audit | REMOVE |
| 8-10 | `stop_context.py`: its own Newton fitter (`:977-1130`), the day-clustered CR1 sandwich (`:1131`), `ESTIMATORS = ("conditional", "plugin")` with the check-only plugin path (`:109`, `:35-37`) | `scripts/pfc/run_pfc_stop_context.py:337`, `reported/mpfc/followups.py:259` | the shared fitter's grouped likelihood and the session to day to animal reduction (Phase 8 task) | REMOVE |
| 8-11 | `fit.py:141-204` (`test_theta_modulation`, statsmodels `fit_regularized`) | `process_neuron` | — | ASK-10 |
| 8-12 | Functions with no caller outside tests: `heldout.poisson_deviance` (`:317`), `marked_model_family.stop_level_permutation_weights` (`:1866`) | tests only | — | REMOVE with their tests (the deviance in the shared fitter replaces the first) |
| 8-13 | `event_set` option `with_swr_without_replay` (`event_sets.py:437`, `session.py:834`, `:905-911`, `marked_glm.py:38`, `:537`; runner `--event-set`, `--null-classes`) | `reported/mpfc/__init__.py:166-196`, `compare_pfc_event_set_bundles.py` | — | KEEP (an analysis the mPFC summary reads; refit on the new estimator) |
| 8-14 | `non_local_threshold` argument that only refuses a non-default value (`session.py:826`, `:845-851`, `:1318-1325`) | `diagnose_pfc_event_definition.py` | — | REMOVE (vestigial; the diagnostic script is ASK-8) |
| 8-15 | Stale comments: `session.py:100-102` (two strategies, three listed), `:858-869` ("returns a 7-tuple rather than a 5-tuple", "when fitter='statsmodels'"); `__init__.py:17`, `:41`, `:55`, `:86`; `load_population.py` "earlier estimator", "historical BH" (`:46-49`, `:127-128`, `:178`, `:331`, `:350`, `:550`); `marked_glm.py:308`, `:539` ("legacy models", "legacy ladder", meaning the unit ladder) | — | — | REMOVE / reword |
| 8-16 | Scripts that compare old generations: `scripts/pfc/compare_pfc_fitters.py` (statsmodels against JAX, retired-generation bundles only), `check_pfc_ridge_lambda0.py` ("archaeological": rebuilds an old basis), `compare_pfc_cohorts.py` (before/after the stop restriction, `inference="legacy"`; `--comparison cohort`), `diagnose_pfc_event_definition.py` (keeps the retired run-time event construction) | none in `RUNS` or `reported/` | — | REMOVE the first two (their subject is gone); ASK-8 for the other two |
| 8-17 | Tests of removed paths: `tests/analysis/pfc/test_glm_pipeline.py` (`TestFitImmobilityModels` `:955` and `TestProcessNeuronFitMask` `:2177` parametrized over statsmodels; `TestFitterEquivalence` `:1598-1769`; the `all_stops` tests `:1799`, `:1925`, `:1941`, `:2057`, `:2471`), `test_batched_glm.py` (`TestAgreementWithStatsmodels` `:71`, `:222`, `:260`, `:399`, `:575`), `test_heldout.py` (`TestAssignBlockedFolds` `:59`, statsmodels deviance `:173`), `test_pfc_load_population_nan_filter.py` (`:30`, `:241-400`), `test_cohort.py:56`, `test_prepare_session.py:87`, `:123`, `tests/scripts/pfc/test_compare_pfc_cohorts.py` | — | Phase 8's tests | REMOVE (`TestTheRetiredEventConstructionIsGone`, `test_glm_pipeline.py:2530`, KEEP as a guard until audit 3 of Phase 12 decides) |
| 8-18 | Docs: `docs/design/pfc_unit_glm_estimator.md:9-169`, `pfc_marked_event_kernels.md:20-49`, `:103-122`, `pfc_marked_model_family.md:16-63`, `:282-332`; `docs/design/cohort_and_timing_rules.md:97-207`, `:231-409`; `scripts/README.md:148` | — | the rewritten estimator docs | REMOVE / rewrite |

## Phase 11

| ID | Item | Callers | Replaced by | Class |
|---|---|---|---|---|
| 11-1 | `rate/duration_aware_history.py` (whole; `smf.glm` `:486`; `fit_pooled_history` `:453` with `C(session)`), `rate/ridge_history_cv.py` (whole; `fit_ridge_poisson` `:144`, the nested CV of sigma `:453`, `assign_stop_folds` `:220`), `rate/history_kernel_statistics.py`, `MODEL_CROSS_HISTORY_RATE`, `_DURATION_RATE`, `_LONG_RATE`, `_cross_history_spec`, `_duration_aware_spec` (`model_specs.py:1108-1250`), `PerBinRateModelSpec` (`:685`), `PerBinTerm` and the term helpers (`:553-684`), `longest_history_window_s` (`:1432`) | the two runners, `reported/behavior/history.py` | `rate/visit_history` (Phase 11) | REMOVE |
| 11-2 | `rate/duration_history_simulation.py` | `fit_duration_aware_history_rate.py`, `model_specs.py` (docstring) | the Phase 11 tests' simulator | KEEP if the new tests use it (then strip its spec imports); else REMOVE |
| 11-3 | `replay_models/per_bin_design.py` (whole: `aggregate_to_timebins` `:394`, `attach_is_swr_to_onsets` `:100`, `reward_kernel_weights` `:283`, `reconstruct_*_kernels` `:307-361`, `arrival_spanning_stop_ids` `:362`, `POPULATIONS` `:1030`, the "predates the arrival clocks" branch `:477`, the stop-bound fallback `:629`) | every old rate module; `analysis/behavior/covariates/definitions.py` and `covariate_timing.py` reference names | `replay_models/visit_design.py` (reuse moves there in Phase 3) | REMOVE |
| 11-4 | Design constants of the stop-bin design: `TMAX`, `SPLINE_KNOTS`, `TMAX_ARRIVAL`, `ARRIVAL_SPLINE_KNOTS`, `ARRIVAL_CLOCK_COLUMN`, `N_HISTORY_LAGS`, `N_HISTORY_BASIS`, `N_CROSS_LONG_*`, `REWARD_HISTORY_N_BASIS` (`behavior/design_constants.py:87-141`) | old rate modules | the visit design's term ranges | REMOVE those without a remaining caller (`REWARD_HISTORY_N_LAGS` stays: model D uses ten lags) |
| 11-5 | `scripts/replay_models/rate/fit_duration_aware_history_rate.py`, `fit_cross_history_ridge_cv.py`; runs `rate.duration_aware_history.*` and the ridge run; `duration_aware_history_values` (`reported/behavior/history.py:36`); `BehaviorInputs.duration_aware_pooled`, `cross_ridge_cv` | `summarize` | the visit history runner and values | REMOVE |
| 11-6 | `_fit_diagnostics.py`, `coefficient_statistics.py` (last callers: history statistics here; `reported/mpfc/population.py` moved in Phase 8) | — | `estimand_summary` | REMOVE |
| 11-7 | Tests: `tests/analysis/replay_models/rate/test_duration_aware_history.py`, `test_ridge_history_cv.py`, `test_history_kernel_statistics.py`, `test_per_bin_terms_frozen.py`, `test_per_bin_rate_helpers.py`, `tests/analysis/replay_models/test_switch_flag_retired.py`, `test_model_specs.py` (rate part), `test_coefficient_statistics.py`; `tests/scripts/replay_models/rate/test_fit_duration_aware_history_rate.py`; `tests/reported/behavior/test_history.py` | — | 11's tests | REMOVE |
| 11-8 | Docs: `docs/models_table.md:286-348`; `docs/design/duration_aware_history_simulation.md`; `docs/rate_models.md` (the file, now empty of current content); `docs/README.md:40`, `:48`; `scripts/README.md:82-88` | — | `docs/regression_models.md` | REMOVE |

## Cohort selectors and stop clocks

| ID | Item | Callers (2026-11-20) | Replaced by | Class |
|---|---|---|---|---|
| C-1 | `event_definition/cohort_selection.py`: `select_stop_cohort` (`:322`), `select_event_cohort` (`:378`), `STOP_COHORT_RULES` (`:217`), `_COHORT_STATES` (`:231`), `STOP_COHORTS` (`:239`), `STOP_COHORT_ARTIFACT_SUFFIX` (`:244`), `stop_cohort_suffix` (`:251`), `_occupancy_counts`, `_occupancy_event_counts`, `_cohort_mask` (`:285-321`) | `content/cohort.py` (6a), `contrast_aggregation.py` (6b), `rate/per_stop_rate_model.py` (5b), `rate/swr_contrast_rate.py` (5b), `rate/stop_statistics.py` (5a), `choice/choice_prediction.py` (7), `choice/retrospective.py` (ASK-2), `content/dirichlet_multinomial.py` (6b), `content/model_checking.py` (6a), `pfc/cohort.py` (8), `reported/behavior/__init__.py` (6a, trunk values), `reported/behavior/stops.py` (5a), runners `fit_per_bin_rate_models.py`, `fit_per_stop_rate_models.py`, `fit_swr_contrast_rate.py`, `fit_uncertainty_rate.py`, `fit_duration_aware_history_rate.py`, `fit_cross_history_ridge_cv.py`, `run_stay_leave.py`, `run_retrospective_alignment.py`, `plot_model_checking.py`, `plot_per_bin_rate_summary.py`, `scripts/pfc/run_pfc_analysis_bandit.py`, `scripts/pfc/measure_pfc_mark_context.py`, `scripts/figures/generate_figure05.py:62`, `:667`, `scripts/exploratory/_content_cohort_control.py` | `select_visit_context` (C4) | REMOVE in the phase that removes the last caller (11 if ASK-2 moves the retrospective analysis; otherwise 12); each phase moves or removes its callers |
| C-2 | The `--stop-cohort {at_well, in_transit, all_stops}` flags and the `_allstops` / `_intransit` sensitivity bundles of every runner | runners, `RUNS` | one call of `select_visit_context` with other labels when a group is wanted | REMOVE with each runner (C13: no flag selects an old cohort) |
| C-3 | `cohort_selection.py`: `stop_interval_mask` (`:479`), `events_in_stops` (`:520`) | `analysis/detector_comparison/features.py:53`, `:954` (Part 1-2), `pfc/event_sets.py:23-24`, `:107`, `:374-377` | — | KEEP (Part 1-2 use; not a cohort rule) |
| C-4 | `cohort_selection.py`: `freeze_common_cohort` (`:47`), `sorted_unit_ids` (`:41`) | `fitting.py`, `common_cohort.py` | the cohort step of each new family (it may call it) | KEEP if a new cohort step calls it; else REMOVE in 12 |
| C-5 | Stop-clock columns: `time_since_stop_start_s` (event table; `event_tables/_common.py`), `time_since_last_well_arrival_s` (event table), `t_since_stop_onset` (onset table), `last_well_arrival_time_s` (stop table, `stop_table.py:188`) | old rate and content models; `swr_power.py` (`PREDICTORS`); `analysis/contfrag/tables.py`; `analysis/detector_comparison/features.py` (`last_well_arrival_time_s`); `interactions.py:244-246` (`time_since_stop_start_s_x_*` unfitted products) | the visit clocks (Phase 3) | REMOVE `time_since_stop_start_s` products and every replay-model read in the family phases; the columns stay while ContFrag (ASK-1) and Part 1-2 (`last_well_arrival_time_s`: KEEP) read them |
| C-6 | Reward raised-cosine basis columns `reward_history_rcos_*` and `reward_history_incomplete` (stop table; `REWARD_HISTORY_COLUMNS`, `design_constants.py:89-94`) | per-bin, per-stop, contrast rate fits; `contrast_aggregation.py`; `analysis/contfrag/tables.py`; `visualization/session_tables`; `scripts/checks/audit_stop_table_reward_history.py`, `compare_to_golden_baseline.py` (its added-column list) | model D | REMOVE from the replay models in 5a/5b; the table columns are ASK-1 (ContFrag reads them) |
| C-7 | `event_tables/timebin_tables.py:158` (`build_stop_timebin_table`, the dense 4 ms stop-bin table; canonical name `stop_timebin` in `paths.py:340-349`) | `scripts/tables/build_timebin_tables.py`, `scripts/exploratory/_timeresolved_psth.py` only | the visit bin design (built on the fly) | ASK-4 |

## Phase 12

Cross-cutting items no family phase owns; Phase 12 removes them if they are still there.

| ID | Item | Class |
|---|---|---|
| 12-1 | Any helper of `replay_models/` left without a caller after Phases 5a to 11 (audit 2 of Phase 12 finds them) | REMOVE |
| 12-2 | The cohort selectors (C-1) if a caller outlived its phase | REMOVE |
| 12-3 | `replay_models/__init__.py:1-33` docstring (lists `per_bin_design`, `fitting`, `_fit_diagnostics`, `common_cohort`, `cohort_audit`, `contrast_aggregation`, `coefficient_statistics`, `fit_comparability`) | REMOVE (rewrite to what remains) |
| 12-4 | `docs/models_table.md` and `tests/repository/test_models_table_mirrors_specs.py`: reduced to the new specs | REMOVE the old blocks |
| 12-5 | `docs/development.md:126-130` (model_specs as the single source of all ladders), `:476-479` (the behavior statistics modules), `:511` (convention 9: AIC between rungs, `fit_comparability`) | REMOVE / rewrite |
| 12-6 | `docs/known_issues/reported_statistics_choices.md:26-54` (B rows of the old behavior statistics) | REMOVE the rows whose statistics are gone |
| 12-7 | `TODO.md:77-107`, `:121-147` entries about the removed models (content refits, the ridge fitter owner, the B choices, the MU and contrast ladders, Mvu, choice provenance) | REMOVE the entries the plan resolved; the rest are ASK-7 |
| 12-8 | `docs/README.md:23`, `:40`, `:46-48`, `:54`, `:64-65`, `:168-200` (index and "Where results live") | REMOVE / rewrite |
| 12-9 | `docs/design/cohort_and_timing_rules.md` (rules of the old cohort, the mPFC fixed sigma and blocked folds, the old audit tables) | REMOVE the file once nothing points to it (it is undated, so it is current documentation, not a record) |
| 12-10 | `paper/reported/removed_part*.md` | KEEP (records of removed values) |
| 12-11 | `docs/covariate_dictionary.md:382-452` ("F.1 Two clocks", "F.2 at-well cohort is not SWR-neutral"), `:511-518`, `:613`, `:626`, `:694-724` (stop clocks, the at-well cohort, `onset_lag`, `cross_hist_rcos_k`) | REMOVE / rewrite for the visit clocks |
| 12-12 | `docs/known_issues/current_arm_trunk_fallback.md:160-170` ("Guidance for legacy artifacts"), `:54-61` (pre-fix values in old pickles) | REMOVE once no reader reads those pickles |
| 12-13 | `scripts/README.md:17-18`, `:82-110`, `:128`, `:131`, `:148`; `tests/README.md:207`; one-line pointers in `docs/figure-pipeline.md:55`, `docs/reproduce.md:69`, `docs/data-lineage.md:35` | REMOVE / rewrite |
| 12-14 | `scripts/checks/summarize_cohort_audits.py`, `scripts/checks/freeze_regression_baseline.py` (reads old ladder outputs through `content/fit_outputs.py`), `scripts/checks/audit_stop_table_reward_history.py` (the raised-cosine basis) | REMOVE with their subjects, or rewrite for the new outputs (check each) |
| 12-15 | Dated designs that describe replaced models as current work (`docs/design/2026-07-13-content-rate-uncertainty-model-ladders.md`, `2026-06-27-state-space-point-process-event-model.md`) | KEEP (dated records); remove links that present them as current from `docs/README.md` |

## Scripts: exploratory and one-time

From a search of `scripts/replay_models/`, `scripts/pfc/`, `scripts/tables/`, `scripts/workflows/run_refits.py`
and `scripts/exploratory/`. Flags and compatibility branches of a runner leave with the runner (rows of the
family phases); the rows below are the ones that need their own decision.

| ID | Item | Callers | Phase | Class |
|---|---|---|---|---|
| S-1 | `scripts/tables/convert_tables_to_parquet.py` (one-time pickle to parquet migration; "absent (skipped)" `:77-78`) and `paths.py:384`'s `fmt="pkl"` reader that only it uses; the exemption in `tests/tables/test_table_io.py:309` | none in `RUNS`, `Makefile`, `docs/reproduce.md` | 3 | REMOVE |
| S-2 | `scripts/tables/regenerate_stop_table.py` (rebuilds only the stop table by joining the old event table; `run_refits.py:21-23` says it is not a run) and the error messages that point to it (`fit_per_stop_rate_models.py:148`, `:155`); tests `tests/scripts/tables/test_stop_table_regen_guards.py` | — | 3 (Phase 3 rebuilds the set; a partial rebuild is what the development guide forbids) | REMOVE |
| S-3 | `scripts/tables/refresh_contfrag_behavioral_covariates.py` (rebuilds ContFrag tables from a preserved bundle) | test only | — | ASK-1 |
| S-4 | `scripts/tables/write_regeneration_manifest.py`, `assemble_replay_content_shards.py`, `assemble_contfrag_shards.py` | `table_parity.py`, Phase 3's rebuild | — | KEEP (Phase 3 uses the manifest and may use the shards) |
| S-5 | `scripts/tables/prepare_replay_content_data.py:412-421` (warns about legacy `events.pkl`, `long_count.pkl`, `long_binary.pkl` in the output directory) and `scripts/replay_models/content/fit_replay_content_models.py:846-862` (names and refuses a legacy `long_count.pkl`) | — | 3 (the warning), 6a (the runner) | REMOVE |
| S-6 | `scripts/tables/prepare_replay_content_data.py:65-70` (`try/except ImportError` for datajoint with a placeholder error) | — | 3 | KEEP if the table build must run without datajoint installed (the `--from-cache` path); else REMOVE. Check `pyproject.toml` |
| S-7 | Runs whose output nothing reads (`run_refits.py`): `rate.per_stop.all_stops` (`:167`), `rate.swr_contrast.all_stops` (`:169`), `rate.duration_aware_history.at_well` (`:176`), `content.arm_ladder.swr_only`/`non_swr_only` (`:189-190`), `content.arm_ladder.dirichlet_multinomial` (`:191`), `content.patch_ladder.*` (`:192-194`), `content.breadth_contrast.rewarded`/`omitted` (`:202-203`), `content.event_source.*` (`:207-210`), `behavior.event_source_suite.swr`/`mua` (`:220-221`), `behavior.retrospective.*` (`:223-225`), `behavior.stay_leave.all_stops*` (`:228`, `:232-234`) | none | with their family (5b, 6a, 6b, 7, 11); ContFrag and retrospective runs: ASK-1, ASK-2 | REMOVE |
| S-8 | `rate.per_bin.in_transit` (`:165`) is optional and commented "cited nowhere", yet `reported/behavior/__init__.py:166-190` reads it | `summarize` | 5a | REMOVE (both) |
| S-9 | Plotters reachable only through their `_fig` runs, in no `FIGURES` entry: `plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_model_checking.py` (5a), `plot_uncertainty_rate.py` (5b, with its `--smoke` path), `content/generate_replay_content_figures.py` (6a); `content.fig.pipeline_data_flow` reads `models/M3.pkl` (`scripts/figures/generate_supp_pipeline_data_flow.py`) | `RUNS` | 5a, 5b, 6a | REMOVE (the pipeline figure: point it at the role model in 6a) |
| S-10 | Vestigial single-value flags: `--switch-window` (`fit_uncertainty_rate.py:209`), `--stop-cohort` of `run_retrospective_alignment.py:123` | — | 5b; ASK-2 | REMOVE |
| S-11 | `scripts/workflows/run_refits.py:246` `_SKIPPED_DIRS` (`_backup`, `_archive`, `_pre_`, `_staging`, `_golden`) | provenance check | — | KEEP (excludes backup directories from provenance; not an old path) |
| S-12 | Exploratory files that serve only replaced models: `_content_cohort_control.py` (and its `ALLOWED` entry, `tests/repository/test_no_new_script_imports.py:28`), `_content_reward_interactions.py`, `_covariate_ranking.py` (and the `covariate_ranking.pkl` mention, `docs/README.md:186`), `_cs5_response_by_design_factorial.py`, `_lagged_reward_kernel.py`, `_nonlocal_model_fit.py`, `_nonlocal_rate_fit.py`, `_timeresolved_psth.py`, `_verify_content_models.py`, `_audit_covariate_accuracy.py` (imports `replay_models.rate`) | none | the phase that removes the module each imports (6a, 5b, 5a, 7) | REMOVE |
| S-13 | Broken exploratory files: `_univariate_coefs.py` (imports `analysis.behavior.rate_model`, which does not exist, `:18`; reads the gone `models_rate/`), `_modelfit_summary.py` (reads the gone `models_rate/all_models.pkl`, `:98`) | none | 5a | REMOVE |
| S-14 | One-time migrations under `scripts/exploratory/`: `_augment_data_over_time.py` (pinned by `tests/scripts/exploratory/test_augment_discover.py`), `_verify_harmonization_parity.py` (pinned by `test_harmonization_parity_guard.py`) | tests | — | ASK-11 |
| S-15 | Other exploratory files (descriptive or for Figures 2 and 3: `_event_rasters.py`, `_fig2b_*.py`, `_peri_departure_rate.py`, `_periswitch_*.py`, `_session_raster.py`, `_reality_reward_test.py`, `_trunk_zscore_*.py`, `_content_patch_destination.py`, `_nonlocal_bout_arm_dots.py`, `_hpc_sorted_spikes.py`, `_statescript_contingency.py`, `_verify_covariate_invariants.py`, `_verify_qvals_invariants.py`) | none or tests | — | ASK-11 (outside the replaced families; `_nonlocal_bout_arm_dots.py` imports `model_specs` and breaks when 6a removes its names: fix or remove then) |
| S-16 | Compatibility branches in runners that stay: `fit_swr_contrast_content.py:263-267` (legacy sidecar key `q_arm_source`) leaves with the runner in 6a; `run_stay_leave.py:249-264` with the runner in 7; `generate_uncertainty_content_figure.py:95-105` with ASK-5; `summarize_pfc_marked_family.py:170-190` and `compare_pfc_marked_family_trees.py:64-69`, `:82` (defaults for older sidecars) | — | 8 for the two mPFC scripts | REMOVE (the mPFC ones in 8: refit, then require the fields) |

## Docs

The doc sections are listed with the phase that removes them (rows `*-12`, `*-15`, `5b-6`, `6b-5`, `7-9`,
`11-8`, `12-4` to `12-9`). Dated design files (`docs/design/<YYYY-MM-DD>-*.md`), `docs/design/model_history.md`
and `docs/history/` are records and are not edited, except where an undated doc points readers to them as
current. `docs/regression_pipeline_map.md` is generated: regenerate it, never edit it. `docs/behavior_models.md`
and `docs/behavior_model_covariates.md` describe the generative behavioral model and the qvals table, not the
regression models, and stay.

## KEEP

| Item | Why |
|---|---|
| `tables/covariate_registry.py:340` `LEGACY_COLUMN_NAMES` and `assert_canonical_vocabulary` (`:542`) | a deny-list that rejects old spellings in any table; a guard, not a compatibility path (outside the areas; listed for the lead) |
| `analysis/behavior/covariates/definitions.py:3` ("how legacy columns became canonical") | the docstring of the scientific catalog; reword "legacy" to "older" names in Phase 12's word audit or keep |
| `analysis/event_tables/prepare_data.py:195-215` (stale cache entries re-fetched from Spyglass) | cache invalidation by a version check; a database read, not an old analysis path |
| `analysis/event_tables/event_table.py:526-530` (an unmatched classified row) | a guard that catches a mis-join; not a tolerance |
| `analysis/event_properties/event_statistics.py:1069-1210` (`at_well` content place shares) | the decoded content's place on the track, not the stop cohort |
| `analysis/replay_models/covariate_timing.py` | the pre- and post-outcome timing check; the new cohort steps keep calling it |
| `analysis/replay_models/content/simulation.py` | the content simulator Phase 6a's simulations use (strip its `fit_conditional_poisson` mention at `:266` in 7) |
| `analysis/replay_models/content/dirichlet_multinomial.py` | the robustness estimator on the new design (6b) |
| `tests/repository/test_docs_have_no_retired_names.py` | a guard |
| `analysis/behavior/covariates/_spatial_bandit_rl.py:14-18`, `:346` (pandas-compat shims) | vendored behavioral-model code whose deviations from the source are listed in its docstring; the qvals table is not rebuilt by this plan |
| `analysis/behavior/covariates/spatial.py:28-34` (hop-count fallback when edges carry no `distance`) | synthetic test graphs have no edge distances; reword "Matches the legacy `predictors.compute_dist_to_arm` contract" in Phase 12's word audit |
| `analysis/behavior/covariates/interactions.py:229` `LEGACY_PRODUCT_NAMES` | ContFrag product names; follows ASK-1 |
| `scripts/tables/write_regeneration_manifest.py`, the shard assemblers | S-4 |
| `scripts/workflows/run_refits.py:246` `_SKIPPED_DIRS` | S-11 |
| `analysis/pfc/cohort.py:37` (`split_sorted_epochs`), the `event_set` option | 8-5, 8-13 |

## ASK list

| ID | Item | Why it is a question | Recommendation |
|---|---|---|---|
| ASK-1 | The ContFrag content and rate regressions: `MODEL_HIERARCHY_SWR`/`_MUA`, `MODEL_PATCH_SWR`/`_MUA` (`model_specs.py:363-423`), the ContFrag paths of `fit_replay_content_models.py` and `content/fit_outputs.py:38` (`resolve_contfrag_paths`), the ContFrag table columns they read (`analysis/contfrag/tables.py`: centered recency, `time_since_stop_start_s`, the reward raised-cosine basis), `docs/models_table.md:458-493` | the ContFrag side pipeline is outside this plan; nothing here replaces its models, and keeping them keeps much of the old content code alive | move the ContFrag event sources onto the role design (one more event source of the 6a runner) in Phase 6a, or retire the ContFrag regressions if no figure needs them; either way the old ladder code goes |
| ASK-2 | The retrospective-alignment analysis: `choice/retrospective.py` (with `choice/shuffles.py`), `event_tables/retrospective_metrics.py`, `scripts/replay_models/choice/run_retrospective_alignment.py`, read by `scripts/pfc/compare_pfc_cohorts.py`; it calls `select_event_cohort` and rejects legacy inputs (`retrospective.py:98`, `:137-139`, `:718`) | not one of the seven questions; no summary value reads it (check) | move it to `select_visit_context` and keep it as a descriptive control, or retire it with its table columns; recommend asking whether question 3.10's dropped null is still needed |
| ASK-3 | The rest of the event-source suite `scripts/replay_models/content/compare_event_source_models.py` (non-local, SWR and MUA sources compared) and `choice/switch_analysis.py` | its choice block is replaced (7-1); the source comparison is not in the plan | retire with the choice block unless the lead wants the source comparison; if kept, move it to the role design |
| ASK-4 | The dense stop time-bin table `stop_timebin` (C-7) | a canonical table only an exploratory script reads | retire the table, its builder and its canonical name in Phase 3's rebuild (the visit design replaces it); the event-onset table stays (Phase 3 reads it) |
| ASK-5 | `content/uncertainty_content.py`'s quartile profiles and `scripts/figures/generate_uncertainty_content_figure.py` (`FIGURES["uncertainty_content"]`) after the horse race leaves | a supplementary figure built on the old design; 6a's Q5 replaces its question | redraw it from the role model's uncertainty curves, or retire it with the figure entry |
| ASK-6 | The descriptive trunk shares (`content_statistics.trunk_shares` and `trunk_values`) | descriptive, not a model; the role model's trunk role gives a model-based share | keep as a descriptive value on the selection function (rewrite on `select_visit_context`) |
| ASK-7 | `TODO.md:121-147` "Open analyses from the retired notes" (property contrasts, per-arm value update, graded ripple-power content rung, marked point process) | open ideas, not code | the lead prunes the list; the plan removes only entries about removed models |
| ASK-8 | `scripts/pfc/compare_pfc_cohorts.py` (before/after the stop restriction; the one production caller of `inference="legacy"`) and `scripts/pfc/diagnose_pfc_event_definition.py` (keeps the retired run-time event construction) | audits of past changes; no summary reads them | retire both in Phase 8 (their subjects are gone once the bundles are refit); keep only if the lead wants the audit trail runnable |
| ASK-9 | A fixed-penalty sensitivity of the mPFC models (today `--penalty-sigma`, the `_sigma<v>` bundles, `scripts/pfc/part4_rate_threshold_sensitivity.py` and the marked-family prior options) | the convention sets penalties by REML; a prior sensitivity is not in the design | drop it; if a reviewer asks, a named sensitivity model with its own record |
| ASK-10 | `analysis/pfc/fit.py:141-204` (`test_theta_modulation`, a statsmodels unit fit of theta modulation during running) | not a peri-event unit model; the design does not name it | move it to the shared fitter in Phase 8 (one Poisson fitter in the repository); keep its definition |
| ASK-11 | `scripts/exploratory/` files outside the replaced families (S-14, S-15), and the two pinned one-time migrations `_augment_data_over_time.py`, `_verify_harmonization_parity.py` with their tests | exploratory one-offs nothing in this plan replaces | remove the two migrations and their tests (their migrations are done); ask whether the Figure 2b diagnostics are still wanted; move any still-wanted check into a test |
| ASK-12 | `scripts/tables/prepare_replay_content_data.py:65-70` (`try/except ImportError` for datajoint) | a fallback for a missing dependency | keep only if `--from-cache` must run without datajoint; otherwise remove (S-6) |

## Counts

Rows of the tables above (a row can hold several symbols of one module or family), by section and class:

| Section | REMOVE | ASK | KEEP |
|---|---|---|---|
| Phase 1 | 1 | 0 | 0 |
| Phase 3 | 11 | 0 | 0 |
| Phase 4 | 2 | 0 | 0 |
| Phase 5a | 12 | 0 | 0 |
| Phase 5b | 6 | 0 | 0 |
| Phase 6a | 15 | 0 | 0 |
| Phase 6b | 5 | 0 | 0 |
| Phase 7 | 9 | 0 | 0 |
| Phase 8 | 16 | 1 | 1 |
| Phase 11 | 8 | 0 | 0 |
| Cohort selectors and stop clocks | 5 | 1 | 1 |
| Phase 12 | 13 | 0 | 2 |
| Scripts | 11 | 3 | 2 |
| Total | 114 | 5 (12 ASK items in the [ASK list](#ask-list)) | 6 (15 rows in the [KEEP](#keep) table) |

Some REMOVE rows depend on an ASK answer (marked "ASK-n" in the row): ContFrag (ASK-1) decides how much of
6a-1, 6a-2, 6a-4, 6a-11, C-5 and C-6 can go.
