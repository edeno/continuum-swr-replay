# Regression pipeline code map

Every module that carries data from a decoded session to a fitted
regression, grouped by pipeline stage, with the public functions and
classes it defines (in definition order).

**Generated file — do not edit by hand.** Regenerate with
`uv run python scripts/qa/write_pipeline_map.py`;
`--check` fails when the tree and this file disagree.

Names are the anchor, not line numbers: grep the symbol. Private
symbols (`_name`) are omitted, but private *modules* are listed;
`__init__.py` re-export surfaces are omitted.

## Tables

Event, arm, patch, stop and time-bin tables built from the decoded session, plus the behavior-model table they join against.

### `src/continuum_swr_replay/analysis/behavior/_spatial_bandit_rl.py`

Beta-Bernoulli decision-variable covariates — VENDORED VERBATIM from Alison Comrie's `rl_helpers.py` (LorenFrankLab/SpatialBanditTask analysis code, /home/alison/Src/alison_spyglass/rl_helpers.py).

`set_nan_at_max2`, `add_ak_style_cols_to_RL_output`, `add_periswitch_cols`, `add_dvs_beta`, `add_switch_advantage_beta`, `add_stem_q_dvs_beta`, `make_full_beta_bernoulli_dv_df`

### `src/continuum_swr_replay/analysis/behavior/qvals_table.py`

Per-stop RL-model values via DataJoint — canonical, complete, timestamp-aligned.

`peri_switch_index`, `add_experienced_value_columns`, `add_beta_belief_entropy`, `canonicalize_behavior_model_columns`, `assign_stops_to_trials`, `build_animal`, `run_build_qvals_table`

### `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py`

Aggregation of analysis results over time.

`theta_power_zscore`, `validate_required_session_data_keys`, `validate_results_position_time_alignment`, `align_results_to_position`, `validate_dot_event_contract`, `get_data_over_time`, `load_data_over_time`, `get_event_types`, `compute_swr_overlap_class_counts`, `compute_swr_nl_event_counts`

### `src/continuum_swr_replay/analysis/hippocampus/event_analysis.py`

Analysis functions for manipulating and characterizing detected events.

`get_immobile_events`, `get_event_start_end_times`, `get_event_durations_ms`

### `src/continuum_swr_replay/analysis/hippocampus/event_detection.py`

Detection and analysis of neural events (ripples, non-local states).

`get_confident_times`, `positive_duration_interval_mask`, `event_strict_overlap_keys`, `event_strict_overlap_mask`, `get_overlap_times`, `classify_swr_overlap`, `merged_interval_duration`, `classify_swr_overlap_geometry`, `select_typical_event_by_class`

### `src/continuum_swr_replay/analysis/hippocampus/event_immobility.py`

Canonical whole-event immobility rules for neural event cohorts.

`EventDefinitionError`, `event_speeds_are_immobile`, `whole_event_immobility_mask`, `filter_event_labels_by_speed`, `validate_nonlocal_event_probability_contract`, `filter_event_intervals_by_speed`, `intervals_from_labels`, `build_speed_eligible_event_cohort`

### `src/continuum_swr_replay/analysis/hippocampus/event_table.py`

Canonical per-non-local-event classification table.

`build_classified_nonlocal_event_table`, `assert_drawn_class_matches_ripples`

### `src/continuum_swr_replay/analysis/hippocampus/hpc_raw_stats.py`

Raw-data quality statistics for hippocampal clusterless decoding.

`extract_discrete_transition_matrix`, `compute_hpc_raw_stats_single`, `check_transition_matrix_flags`, `apply_raw_stats_flags`

### `src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py`

Analysis of decoded posterior distributions.

`get_conditional_non_local_posterior`, `get_map_step_distance`, `step_distance_from_nodes`, `map_estimate_to_map_nodes`, `get_map_speed`, `get_posterior_stats`

### `src/continuum_swr_replay/analysis/hippocampus/quality_metrics.py`

Quality metrics for hippocampal decoding analysis.

`compute_local_decoding_error`, `compute_state_fraction_during_movement`, `compute_non_local_during_movement`, `compute_behavior_state_breakdown`, `compute_bout_nonlocal_dominance`, `compute_running_error_tail`, `compute_error_by_track_segment`, `compute_no_spike_firing_correspondence`, `compute_non_local_event_context`, `compute_local_state_collapse_details`, `compute_calibration_curve`, `compute_session_quality_summary`

### `src/continuum_swr_replay/analysis/hippocampus/result_loading.py`

Loading and managing saved decoding results.

`load_results`

### `src/continuum_swr_replay/analysis/hippocampus/triggered_analysis.py`

Event-triggered analysis utilities for extracting signal windows around events.

`get_signal_windows`

### `src/continuum_swr_replay/analysis/replay/_tables_common.py`

Shared prep helpers, the arm standardization column list, and canonical column-order constants for the per-grain table builders.

_No module-level public symbols._

### `src/continuum_swr_replay/analysis/replay/bouts.py`

Immobility bouts: where a stop begins and ends, and what falls inside one.

`nearest_well_distance_cm`, `immobility_bout_intervals`, `time_since_stop`, `assign_bout`

### `src/continuum_swr_replay/analysis/replay/cohort_selection.py`

Shared whole-unit complete-case cohort selection.

`freeze_common_cohort`, `stop_cohort_suffix`, `select_stop_cohort`, `select_event_cohort`, `stop_interval_mask`, `events_in_stops`

### `src/continuum_swr_replay/analysis/replay/event_ids.py`

Stable row identities: the ``event_id`` and ``stop_id`` hashes.

`hash_event_id`, `validate_event_ids`, `hash_stop_id`

### `src/continuum_swr_replay/analysis/replay/prepare_data.py`

Build the four canonical replay-content tables across all sessions.

`find_sessions`, `behavioral_context_cache_path`, `load_behavioral_context`, `populate_behavioral_context_cache`, `build_stop_table_scope`, `run_prepare_data`

### `src/continuum_swr_replay/analysis/replay/table_contracts.py`

Scientific table contracts beyond the generic covariate schema stamp.

`validate_event_arm_table`, `validate_event_patch_table`, `validate_requested_standardization`, `validate_canonical_table_contract`, `validate_and_match_fit_results`

### `src/continuum_swr_replay/analysis/replay/tables_arm_patch.py`

Canonical arm- and patch-grain builders: one row per (event, arm) / (event, patch).

`build_event_arm_table`, `build_event_patch_table`

### `src/continuum_swr_replay/analysis/replay/tables_event.py`

Canonical event-grain builders: one row per non-local event.

`build_event_table`, `build_event_sequence_table`

### `src/continuum_swr_replay/analysis/replay/tables_stop.py`

Canonical stop-grain builder: one row per stopping period.

`build_stop_table`, `add_nonlocal_swr_split`

### `src/continuum_swr_replay/analysis/replay/timebin_tables.py`

Within-stop time-binned tables (Part B).

`build_event_onset_table`, `build_stop_timebin_table`

## Covariates

Predictor construction: the timing clocks, the history blocks, the candidate-arm block, standardization, and the name registry.

### `src/continuum_swr_replay/analysis/replay/content_response.py`

Which per-event response the content ladders are fit on.

`resolve_response_column`, `default_response_for_estimator`, `resolve_response_choice`, `validate_fraction_response`, `renormalize_fraction_over_candidates`

### `src/continuum_swr_replay/analysis/replay/covariate_registry.py`

Covariate naming contract -- the runtime half of the 2026-08-21 rename.

`row_keys`, `Entry`, `derived_columns`, `canonical_names`, `RegistryError`, `validate`, `RenamePlan`, `plan_rename`, `apply_rename`, `canonicalize_numerical_paths`, `canonical_output_name`, `rename_dynamic_columns`, `assert_canonical_vocabulary`, `assert_unique_scientific_rows`, `stamp_covariate_schema`, `require_covariate_schema`, `finalize_covariate_schema`

### `src/continuum_swr_replay/analysis/replay/covariate_timing.py`

Per-covariate pre/post-outcome timing registry + model-build enforcement.

`timing_of`, `assert_predictor_timing`

### `src/continuum_swr_replay/analysis/replay/covariates/candidate_arm.py`

Shared derivation of candidate-arm predictors.

`broadcast_candidate_arms`, `attach_candidate_arm_covariates`

### `src/continuum_swr_replay/analysis/replay/covariates/clocks.py`

Canonical arrival and outcome-availability clocks for behavioral history.

`trial_arrival_times`, `trial_outcome_times`

### `src/continuum_swr_replay/analysis/replay/covariates/context.py`

Time-local behavioral location and completed-trial context.

`behavioral_location_from_port_experiences`, `summarize_behavioral_location`, `add_trial_context`, `trial_context_at_times`

### `src/continuum_swr_replay/analysis/replay/covariates/definitions.py`

Machine-readable scientific definitions for model-facing covariates.

`Availability`, `QueryClock`, `CovariateDefinition`, `definition_for`, `definitions_frame`

### `src/continuum_swr_replay/analysis/replay/covariates/history/_common.py`

Shared port-experience validation for the history covariates.

_No module-level public symbols._

### `src/continuum_swr_replay/analysis/replay/covariates/history/diagnostics.py`

Diagnostic per-animal fitted-alpha leaky-Q learner.

`q_all_arm`

### `src/continuum_swr_replay/analysis/replay/covariates/history/reward_history.py`

Reward-history covariates: rolling reward counts and recent reward rates over an arm's last visits, plus stop-level reward histories.

`rolling_reward_count_all_events`, `recent_reward_rate_all_events`, `recent_trial_reward_rate`, `lagged_trial_rewards`, `reward_history_basis_features`

### `src/continuum_swr_replay/analysis/replay/covariates/history/value_block.py`

Canonical qvals value-block and arm-uncertainty joins.

`load_contingency`, `attach_contingency_value_block`, `attach_arm_uncertainty`, `assert_qvals_aligned`

### `src/continuum_swr_replay/analysis/replay/covariates/history/value_uncertainty.py`

Value and uncertainty covariates computed from the behavioral stream: the canonical leaky-Q value history and the model-free behavioral-uncertainty covariates.

`canonical_port_value_history`, `canonical_trial_switch_history`, `add_uncertainty_covariates`

### `src/continuum_swr_replay/analysis/replay/covariates/history/visit_recency.py`

Visit-recency covariates: how recently (and whether ever) each arm was visited, at event time.

`center_visit_recency_within_event`, `spline_visit_recency_within_event`, `visit_recency_all_events`, `arm_never_visited_all_events`, `visit_recency_and_never_visited_all_events`

### `src/continuum_swr_replay/analysis/replay/spatial.py`

Graph-spatial helpers for per-arm predictors and QC metrics.

`dist_to_arm_from_well`, `arm_dist_rank`, `arm_adjacency`, `path_continuity`

### `src/continuum_swr_replay/analysis/replay/spatial_mapping.py`

Map decoded replay positions to spatial units (arms, patches).

`map_estimate_to_patch`, `map_estimate_to_arm`, `arm_to_well`, `well_to_arm_set`

### `src/continuum_swr_replay/analysis/replay/standardization.py`

Pooled-global standardization of continuous covariates (`<col>_z`).

`attach_standardized`, `assert_standardized_present`, `assert_standardized_matches_reference`, `z_reference_path`, `save_reference`, `load_reference`

### `src/continuum_swr_replay/analysis/replay/uncertainty.py`

Per-event decoder-confidence summaries from the arm-marginal posterior.

`posterior_peak_mean`, `posterior_entropy_mean`, `posterior_hpd_width`, `posterior_hpd_widths`

### `src/continuum_swr_replay/analysis/replay/uncertainty_content.py`

Model-light estimands for reward-belief uncertainty and replay content.

`assign_within_arm_quantiles`, `build_trial_leaf_table`, `hierarchical_quartile_profile`, `paired_quartile_contrasts`, `summarize_mu_coefficients`, `within_arm_horserace`, `summarize_horserace`

## Specs

The single source of truth for every model ladder, and the design constants the ladders share with the covariate builders.

### `src/continuum_swr_replay/analysis/replay/design_constants.py`

Design constants shared by the covariate builders and the model ladders.

_No module-level public symbols._

### `src/continuum_swr_replay/analysis/replay/model_specs.py`

Model definitions for replay content analysis.

`ModelSpec`, `default_content_model_names`, `PerBinRateModelSpec`, `leaf_only_predictors`

## Fitters

Estimators and the per-session fit machinery each runner calls.

### `src/continuum_swr_replay/analysis/replay/_fit_diagnostics.py`

Shared fit-diagnostics helpers: convergence, separation, warning capture.

`fit_status`, `separated_terms`, `capture_fit_warnings`, `count_fit_quality_warnings`

### `src/continuum_swr_replay/analysis/replay/_helpers.py`

Small shared primitives for the event-table builders.

`single_arm_from_well`, `arm_to_patch`, `stay_indicator_from_patches`, `stay_indicator`, `align_posterior_to_dot`, `safe_mode`, `safe_first`, `strict_first`, `strict_mean`, `strict_sum`, `eq_indicator`, `eq_indicator_or_nan`, `align_count_matrix`, `assert_event_label_alignment`, `assert_single_session_alignment`, `add_arm_fixed_effects`, `build_well_to_arm_map`

### `src/continuum_swr_replay/analysis/replay/choice_prediction.py`

Does replay content predict the next choice?

`predict_arm_fractions`, `build_residualized_choice_dataset`, `build_choice_dataset`, `fit_choice_model`

### `src/continuum_swr_replay/analysis/replay/dirichlet_multinomial.py`

Dirichlet-multinomial fitter for replay content over candidate categories.

`event_loglik`, `multinomial_log_constant`, `fit_dirichlet_multinomial`, `cross_validate_dirichlet_multinomial`, `concentration_cols_for_cohort`, `animal_pooled_log_phi`, `resolve_pooled_log_phi`, `fit_sessions_dirichlet_multinomial`

### `src/continuum_swr_replay/analysis/replay/fitting.py`

Per-session model fitting for the replay content model hierarchy.

`poisson_estimator_name`, `event_clustered_covariance`, `within_group_full_rank`, `fit_conditional_poisson`, `fit_null_model_conditional`, `fit_null_model_conditional_logit`, `fit_conditional_logit`

### `src/continuum_swr_replay/analysis/replay/per_bin_design.py`

The (stop x bin) point-process design and the bases it is built on.

`attach_is_swr_to_onsets`, `raised_cosine_history_basis`, `reward_basis_from_lags`, `reward_kernel_weights`, `reconstruct_reward_kernels`, `reconstruct_self_history_kernels`, `aggregate_to_timebins`

### `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py`

Per-bin conditional-intensity Poisson model for replay event onsets (Family B).

`freeze_per_bin_cohort`, `apply_arrival_window`, `require_frozen_per_bin_design`, `fit_per_bin_rate_model`, `cross_validate_per_bin_rate_model`, `per_bin_rate_gof`

### `src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py`

Poisson rate model for replay event occurrence.

`fit_per_stop_rate_model`, `cross_validate_per_stop_rate_model`

### `src/continuum_swr_replay/analysis/replay/shuffles.py`

Null-distribution generators for replay content QC.

`circular_time_shuffle`, `stop_label_shuffle`, `next_arm_circular_shift`

### `src/continuum_swr_replay/analysis/replay/simulation.py`

Synthetic-data generators for validating the conditional Poisson model.

`sample_arm_design`, `simulate_counts`, `simulate_with_event_covariates`, `simulate_dirichlet_multinomial_counts`, `simulate_autocorrelated_counts`, `SyntheticEnv`, `make_synthetic_env`, `simulate_posterior_bumps`, `simulate_hard_vs_soft_counts`, `make_bandit_track_graph`, `simulate_full_pipeline`

## Runners

Command-line entry points that fit or plot a ladder.

### `scripts/behavior/build_qvals_table.py`

Per-stop RL-model values via DataJoint — canonical, complete, timestamp-aligned.

`main`

### `scripts/content/analyze_content_occupancy.py`

Content occupancy of non-local events — where the decoded activity represents.

`parse_epoch_identifier`, `animal_from_epoch_id`, `find_sessions`, `session_content_distances`, `quadrant_fractions`, `arm_occupancy`, `analyze`, `render`, `main`

### `scripts/content/analyze_uncertainty_content.py`

Build trial-level reward-belief uncertainty/replay-content results.

`mu_inputs`, `build_parser`, `resolve_models_dir`, `main`

### `scripts/content/compare_content_estimators.py`

Fit three replay-CONTENT estimators on one at-well cohort and compare them.

`se_ratio_column`, `RungSpec`, `log_phi_figure_rung`, `resolve_rungs`, `ComparisonResult`, `unidentified_rows`, `n_unpaired_triples`, `summarize`, `bootstrap_animals`, `run_comparison`, `plot_coefficient_agreement`, `plot_log_phi_by_session`, `write_caption`, `main`

### `scripts/content/compare_event_source_models.py`

Run the replay-content analysis suite on one event source and save fits.

`fit_patch_hierarchy`, `fit_patch_switch`, `fit_per_stop_rate_hierarchy`, `fit_choice_blocks`, `run_switch_aligned`, `main`

### `scripts/content/fit_replay_content_models.py`

Fit the replay content model hierarchy (M0-M6) on pre-computed data.

`resolve_contfrag_paths`, `canonical_output_subdir`, `canonical_output_subdirs`, `partial_ladder_subdir`, `belief_chain_output_subdir`, `fit_n_categories`, `rung_predictors`, `restrict_to_leaf_arms`, `leaf_only_cohort`, `content_fit_inputs`, `prepare_canonical_fit_request`, `fit_model_per_session`, `scale_label`, `print_model_results`, `format_comparison_table`, `run_content_fit`, `main`

### `scripts/content/fit_swr_contrast_content.py`

Fit the beta-bernoulli replay-CONTENT ladder (arm + patch), is_swr version.

`resolve_concentration_cols`, `select_outcome_stratum`, `prepare_cohorts`, `main`

### `scripts/content/generate_replay_content_figures.py`

Generate all figures for the replay content analysis.

`build_parser`, `main`

### `scripts/content/quantify_swr_overlap_geometry.py`

Quantify the non-local/SWR overlap sub-cases on real data.

`quantify_session`, `print_report`, `main`

### `scripts/decoding/assemble_contfrag_shards.py`

Assemble disjoint per-animal ContFrag shards into one canonical bundle.

`assemble_contfrag_shards`, `main`

### `scripts/decoding/assemble_replay_content_shards.py`

Assemble disjoint Replay table shards into one canonical candidate bundle.

`assemble_replay_content_shards`, `assemble_replay_stop_shards`, `main`

### `scripts/decoding/build_timebin_tables.py`

Build the Part B within-stop time-binned tables over all sessions.

`build_one`, `validate`, `main`

### `scripts/decoding/convert_tables_to_parquet.py`

Convert the materialized canonical tables from pickle to parquet.

`convert`, `main`

### `scripts/decoding/prepare_contfrag_replay_content_data.py`

Build canonical replay-content tables from ContFrag decoding results.

`contfrag_table_basenames`, `find_sessions`, `main`

### `scripts/decoding/prepare_replay_content_data.py`

Build the four canonical replay-content tables across all sessions.

`main`

### `scripts/decoding/refresh_contfrag_behavioral_covariates.py`

Rebuild ContFrag tables while reusing immutable posterior aggregates.

`refresh_contfrag_behavioral_covariates`, `main`

### `scripts/decoding/regenerate_stop_table.py`

Regenerate ``stop_table.pkl`` with the new count columns.

`build_one`, `validate`, `main`

### `scripts/decoding/run_decoding_bandit.py`

Main script for running continuum neural decoding on spatial bandit task data.

`create_detector`, `perform_decoding`, `save_results_and_model`, `organize_results`, `generate_figurl`, `run_decode`, `main`

### `scripts/decoding/run_decoding_contfrag.py`

Run Continuous-Fragmented (ContFrag) decoding on spatial bandit task data.

`create_contfrag_detector`, `perform_decoding`, `save_results_and_model`, `organize_results`, `run_decode`, `main`

### `scripts/decoding/write_regeneration_manifest.py`

Validate candidate covariate bundles and write a release manifest.

`write_manifest`, `main`

### `scripts/rate/fit_cross_history_rate.py`

Bivariate per-bin conditional intensity: does one non-local stream's history drive the other's rate?

`fit_cross_history`, `main`

### `scripts/rate/fit_per_bin_rate_models.py`

Fit the per-bin conditional-intensity ladder (Family B) and persist it.

`fit_per_bin_rate_models`, `main`

### `scripts/rate/fit_per_stop_rate_models.py`

Fit the per-stop replay-rate hierarchy (R0-R1) and persist it.

`fit_per_stop_rate_hierarchy`, `main`

### `scripts/rate/fit_swr_contrast_rate.py`

Fit the per-bin non-local rate models, `is_swr` version (see docs/models_table.md).

`main`

### `scripts/rate/fit_swr_power_model.py`

Per-event SWR-coupling strength (ripple power) vs behavioral covariates.

`fit_power_model`, `main`

### `scripts/rate/fit_uncertainty_rate.py`

Fit the per-bin replay rate vs behavioral-uncertainty / value, by population.

`fit_uncertainty_rate`, `main`

### `scripts/rate/plot_model_checking.py`

Point-process model checking for the per-bin ladder (Family B).

`plot_gof_ladder`, `plot_example_quartet`, `build_parser`, `main`

### `scripts/rate/plot_per_bin_rate_models.py`

Plot the per-bin conditional-intensity ladder (Family B) from its tidy output.

`plot_reward_kernel`, `plot_selfhistory`, `build_parser`, `main`

### `scripts/rate/plot_per_bin_rate_summary.py`

Six-panel summary figure for the per-bin rate ladder (Family B).

`reward_kernel_rate_ratios`, `build_parser`, `main`

### `scripts/rate/plot_uncertainty_rate.py`

Plot replay rate vs uncertainty/value by population from the tidy fit output.

`report`, `plot`, `build_parser`, `main`

### `scripts/rate/summarize_per_bin_rate_models.py`

Summarize the per-bin conditional-intensity ladder (Family B) effects.

`section_baseline`, `section_coef`, `self_history_rate_ratios`, `section_kernel`, `reward_kernel_rate_ratios`, `section_reward_kernel`, `section_selfhistory`, `section_ladder`, `build_parser`, `main`

## Aggregation

Across-session and across-animal summaries, contrasts and model comparison over the per-session fits.

### `src/continuum_swr_replay/analysis/replay/cohort_audit.py`

Per-model cohort / drop-profile audit records.

`cohort_record`, `classify_drops`, `event_drop_buckets`, `stop_own_missing`, `drop_profile`, `escape_note_field`, `estimator_skip_note`, `records_from_common_cohort_provenance`, `write_cohort_audit`, `term_diagnostics`

### `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py`

Shared audited aggregation for the is_swr contrast ladders (rate + content).

`load_per_sd_scale`, `std_factor`, `separated_for_term`, `session_se`, `typical_session_se`, `pct_significant_sessions`, `aggregate_contrast`, `specs_from_ladder`

### `src/continuum_swr_replay/analysis/replay/learning.py`

Replay content × reward outcome → behavioral change.

`StopSummaryConfig`, `stop_replay_summary`, `fit_is_stay_logistic_per_session`, `fit_is_stay_pooled_gee`

### `src/continuum_swr_replay/analysis/replay/model_checking.py`

Model checking utilities for the replay content model hierarchy.

`cross_validate_conditional_poisson`, `compute_pearson_residuals`

### `src/continuum_swr_replay/analysis/replay/model_comparison.py`

Compare nested models via AIC, BIC, and likelihood ratio tests.

`compare_nested_models`, `summarize_comparison`, `run_full_hierarchy_comparison`

### `src/continuum_swr_replay/analysis/replay/retrospective.py`

Retrospective alignment of replay content with the next choice.

`at_well_event_cohort`, `arm_path_well_to_well`, `chosen_arm_fraction`, `chosen_arm_fraction_prev`, `ends_at_next`, `ends_at_prev`, `contains_path_to_next`, `contains_path_from_prev`, `chosen_patch_fraction`, `ends_at_next_patch`, `compute_retrospective_metrics`, `freeze_retrospective_cohort`, `retrospective_alignment_with_shuffle`

### `src/continuum_swr_replay/analysis/replay/summaries.py`

Per-event replay summaries from decoder outputs.

`posterior_to_arm_marginal`, `arm_map_from_posterior`, `n_nonlocal_bins`, `nonlocal_weight_sum`, `posterior_mass_total`, `arm_count_hard`, `arm_count_soft`, `arm_posterior_mass`, `arm_fraction_from_event_total`

### `src/continuum_swr_replay/analysis/replay/switch_analysis.py`

Switch-aligned timing analysis for patch-switch prediction.

`build_switch_aligned_table`, `build_control_aligned_table`, `stop_patch_replay`

### `src/continuum_swr_replay/analysis/replay/trajectory.py`

Per-event replay-trajectory rollups, shared by both event-table builders.

`brain_state_means`, `ahead_behind_summary`, `uncertainty_extra`, `trajectory_dynamics`, `compute_trajectory_rollups`

## Readers and QA

Artifact I/O, provenance gating, and the diagnostics that check the tables and the fits.

### `scripts/qa/analyze_event_spike_evidence.py`

Spike-evidence reality analysis for non-SWR non-local events.

`parse_epoch_identifier`, `animal_from_epoch_id`, `find_sessions`, `build_session_events`, `load_encoding`, `load_marks`, `event_likelihood`, `event_metrics`, `analyze_session`, `render_session_figures`, `summarize`, `per_animal_medians`, `main`

### `scripts/qa/audit_event_patch_fraction.py`

Audit a candidate event-patch table against its pre-change reference.

`sha256`, `assert_row_and_column_order`, `real_arm_bins`, `assert_fraction_definition`, `assert_matches_arm_fractions`, `main`

### `scripts/qa/audit_stop_table_reward_history.py`

Audit a candidate stop table against its pre-change reference.

`sha256`, `assert_row_and_column_order`, `incomplete_share`, `assert_prefix_per_session`, `main`

### `scripts/qa/check_data_availability.py`

Check data availability for all sessions in the spatial bandit task.

`check_position`, `check_spikes`, `check_ripple`, `check_theta`, `check_dios`, `check_session`, `main`

### `scripts/qa/check_fit_provenance.py`

Report whether every fitted artifact is still current for the tables it read.

`artifact_for_sidecar`, `find_sidecars`, `resolve_input`, `check_artifact`, `propagate_stale`, `format_line`, `main`

### `scripts/qa/check_pfc_ridge_lambda0.py`

Check that the penalised solver reproduces maximum likelihood at lambda = 0.

`audited_basis`, `audited_designs`, `rebuild_session`, `compare_session`, `build_report`, `main`

### `scripts/qa/compare_fits.py`

Compare a backed-up rate fit against the current one and write a markdown audit.

`delta_aic`, `n_bins_table`, `reward_quantities`, `arrival_clock_cv_table`, `arrival_clock_shape_table`, `cv_nan_share_table`, `arrival_cohort_table`, `outcome_boundary_shift_cohort`, `reward_boundary_shift_table`, `session_counts_by`, `rung_session_counts`, `cv_validity_table`, `per_stop_validity_table`, `contrast_fit_flag_table`, `term_separation_table`, `audited_row_table`, `audited_term_drop_table`, `drop_mechanism_table`, `build_report`, `ContentFraming`, `content_term_table`, `content_scale_table`, `content_rung_session_counts`, `content_delta_aic_table`, `content_separation_table`, `rung_sort_key`, `contrast_audited_table`, `contrast_design_delta_table`, `contrast_stratum_table`, `build_content_report`, `collect_content_inputs`, `run_content_mode`, `rate_rung_sort_key`, `cohort_size_table`, `reward_kernel_lag1`, `cohort_reward_rr_table`, `cohort_paired_swr_table`, `cohort_delta_aic_table`, `cohort_per_stop_delta_aic_table`, `cohort_arrival_shape_table`, `per_stop_source_note`, `cohort_burst_peak_table`, `cohort_arrival_retention_table`, `cohort_contrast_rung_table`, `cohort_provenance_table`, `cohort_provenance_note`, `in_transit_power_note`, `build_cohort_report`, `collect_cohort_inputs`, `run_cohort_contrast_mode`, `build_parser`, `main`

### `scripts/qa/compare_memoryless_to_reference.py`

Run the port and the reference's own arithmetic on one input, and compare.

`window_counts`, `overlap`, `shuffle_permutations`, `p_values`, `passes`, `compare`, `reference_row`, `fields_and_thresholds`, `compare_variants`, `decode_sweep_cell`, `sweep_totals`, `main`

### `scripts/qa/compare_pfc_cohorts.py`

Write the before/after audit of the mPFC unit-GLM stop cohort.

`read_sidecars`, `runner_code`, `provenance_table`, `bundle_sessions`, `session_cohort_table`, `coverage_summary`, `alignment_prose`, `gate_counts`, `significance_table`, `population_comparison`, `try_read_sidecars`, `same_events_precondition`, `event_count_agreement`, `shared_statistic_differences`, `matched_control_arrays`, `matched_control_significance`, `fitter_table`, `session_fit_counts_table`, `worker_time_table`, `curve_peaks`, `peak_shrinkage`, `shrinkage_table`, `implied_prior_per_lag`, `empirical_bayes_sigma`, `heldout_fraction_table`, `restrict_to_sessions`, `legacy_fdr_table`, `separated_column_table`, `engagement_overlap_table`, `null_window_table`, `fold_spread_table`, `sigma_sensitivity_table`, `estimator_provenance_table`, `rerun_sections`, `build_report`, `geometry_section`, `build_estimator_report`, `main`

### `scripts/qa/compare_pfc_fitters.py`

Write the equivalence audit of the batched JAX fitter against statsmodels.

`bundle_stems`, `load_bundle`, `compare_statistics`, `compare_modulations`, `compare_diagnostics`, `parse_time_file`, `build_report`, `main`

### `scripts/qa/compare_to_golden_baseline.py`

Diff a candidate replay_content root against a frozen golden baseline.

`main`

### `scripts/qa/compute_hpc_raw_stats.py`

Compute raw-data QC stats for hippocampal clusterless decoding.

`main`

### `scripts/qa/diagnose_pfc_event_definition.py`

Compare the PFC path's old, run-time event definition against the canonical one.

`build_old_event_sets`, `total_overlap_seconds`, `match_by_overlap`, `build_comparison`, `get_command_line_arguments`, `main`

### `scripts/qa/diagnose_trial_detection.py`

Diagnostic figures for trial-detection and predictor semantics.

`figure_trial_sanity`, `figure_predictor_collapse`, `main`

### `scripts/qa/freeze_regression_baseline.py`

Freeze the current canonical tables and model outputs as a read-only baseline.

`freeze`, `main`

### `scripts/qa/measure_pfc_null_geometry.py`

What the mPFC held-out null's windows look like, beside the real events.

`bundle_paths`, `session_of`, `read_sidecar`, `spacing_summary`, `draw_spacing_summary`, `session_window_sets`, `spacing_section`, `overlap_section`, `non_positive_section`, `excluded_section`, `main`

### `scripts/qa/memoryless_reference.py`

The Widloski & Foster detector as its MATLAB computes it, for a sorted 1-D track.

`ReferenceThresholds`, `decode_windows`, `center_and_spread`, `forward_jumps`, `runs_of_kept`, `merge_runs`, `dispersion_bins`, `ReferenceDetection`, `detect`, `set_aside_pairs`, `shuffle_statistics`, `shuffle_p_values`

### `scripts/qa/mutation_check.py`

Objective AST mutation check for value-critical builder logic.

`check`, `main`

### `scripts/qa/run_quality_metrics.py`

Compute decoding quality metrics across all sessions.

`maybe_merge_raw_stats`, `load_all_sessions`, `print_summary`, `plot_diagnostics`, `main`

### `scripts/qa/summarize_cohort_audits.py`

Concatenate and flag every ``*_cohort_audit.csv`` under the outputs dir.

`load_all`, `flag`, `main`

### `scripts/qa/summarize_stage6_fdr.py`

Print Stage 6's per-role cross-session FDR table for a saved bundle.

`main`

### `scripts/qa/sweep_memoryless_controls.py`

What each detector does with activity that is not one clean replay.

`usable_intervals`, `truth_path`, `covered_cm`, `detector_events`, `score_trial`, `jump_segments`, `trials`, `run_detectors`, `condition_rng`, `run_trials`, `run_no_replay`, `summarize`, `main`

### `scripts/qa/sweep_memoryless_heldout.py`

Does each detector's content reproduce in cells it never saw?

`folds`, `fold_models`, `trials`, `event_truth`, `matches_injection`, `permutations_for`, `score_events`, `run_fold`, `run_no_replay_fold`, `summarize`, `main`

### `scripts/qa/write_pipeline_map.py`

Generate the symbol-anchored regression pipeline code map.

`render`, `main`

### `src/continuum_swr_replay/analysis/replay/fit_provenance.py`

Sidecar records tying a fitted artifact to the tables it was fit on.

`fit_provenance_path`, `git_hash`, `record_inputs`, `write_fit_provenance`, `check_fit_provenance`, `require_fresh_inputs`

### `src/continuum_swr_replay/analysis/replay/posterior_io.py`

Load the conditional non-local posterior from a decoder results file.

`load_conditional_posterior`

### `src/continuum_swr_replay/analysis/replay/versions.py`

Centralized schema/version literals for the covariate pipeline.

`assert_versions_compatible`
