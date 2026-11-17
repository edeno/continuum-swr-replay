# Regression pipeline code map

Every module that carries data from a decoded session to a fitted
regression, grouped by pipeline stage, with the public functions and
classes it defines (in definition order).

**Generated file — do not edit by hand.** Regenerate with
`uv run python scripts/checks/write_pipeline_map.py`;
`--check` fails when the tree and this file disagree.

Names are the anchor, not line numbers: grep the symbol. Private
symbols (`_name`) are omitted, but private *modules* are listed;
`__init__.py` re-export surfaces are omitted.

## Tables

Event, arm, patch, stop and time-bin tables built from the decoded session, plus the behavior-model table they join against.

### `src/continuum_swr_replay/analysis/behavior/covariates/_spatial_bandit_rl.py`

Beta-Bernoulli decision-variable covariates — VENDORED from Alison Comrie's `rl_helpers.py` (LorenFrankLab/SpatialBanditTask analysis code, /home/alison/Src/alison_spyglass/rl_helpers.py).

`set_nan_at_max2`, `add_ak_style_cols_to_RL_output`, `add_periswitch_cols`, `add_dvs_beta`, `add_switch_advantage_beta`, `add_stem_q_dvs_beta`, `make_full_beta_bernoulli_dv_df`

### `src/continuum_swr_replay/analysis/behavior/covariates/qvals_table.py`

Per-stop RL-model values via DataJoint — canonical, complete, timestamp-aligned.

`peri_switch_index`, `add_experienced_value_columns`, `add_beta_belief_entropy`, `canonicalize_behavior_model_columns`, `assign_stops_to_trials`, `prepare_model_values`, `build_animal`, `stop_mapping_agreement`, `require_complete_mapping`, `standardize_qvals_table`, `write_qvals_outputs`

### `src/continuum_swr_replay/analysis/decode_quality/hpc_raw_stats.py`

Raw-data quality statistics for hippocampal clusterless decoding.

`extract_discrete_transition_matrix`, `compute_hpc_raw_stats_single`, `check_transition_matrix_flags`, `apply_raw_stats_flags`

### `src/continuum_swr_replay/analysis/decode_quality/posterior_analysis.py`

Analysis of decoded posterior distributions.

`get_conditional_non_local_posterior`, `load_conditional_posterior`, `get_map_step_distance`, `step_distance_from_nodes`, `map_estimate_to_map_nodes`, `track_distance_cm`, `normalized_replay_position_by_swr`, `get_map_speed`, `get_posterior_stats`

### `src/continuum_swr_replay/analysis/decode_quality/quality_metrics.py`

Quality metrics for hippocampal decoding analysis.

`compute_local_decoding_error`, `local_decoding_error_statistics`, `compute_state_fraction_during_movement`, `compute_non_local_during_movement`, `compute_behavior_state_breakdown`, `compute_bout_nonlocal_dominance`, `compute_running_error_tail`, `compute_error_by_track_segment`, `compute_no_spike_firing_correspondence`, `compute_non_local_event_context`, `compute_local_state_collapse_details`, `compute_calibration_curve`, `calibration_curve_statistics`, `session_state_fractions`, `state_fraction_statistics`, `session_decoder_quality`, `compute_session_quality_summary`

### `src/continuum_swr_replay/analysis/detection/artifacts/detect.py`

The artifact detector: its constants and its four rules.

`detector_constants`, `s_cores`, `t_ref_events`, `t_ca1_threshold`, `t_ca1_events`, `t_spk_events`, `t_cores`, `spike_ood`, `detect_artifacts`

### `src/continuum_swr_replay/analysis/detection/artifacts/features.py`

The artifact detector's features: S window spectra, J5 and L20 steps, spike lines.

`robust_scale`, `window_valid`, `window_centers`, `window_band_power`, `channel_specificity`, `level_jump`, `level_shift`, `robust_jump`, `reference_jump`, `persistent_step_count`, `nearest_mark_distance`, `spike_lines`

### `src/continuum_swr_replay/analysis/detection/detection_inputs.py`

What the replay detection runners read from a saved HPC decode and from each other.

`position_span_samples`, `marks_cache_dir`, `input_paths`, `marks_cache_path`, `load_marks`, `marks_sha256`, `decode_marks`, `carey_output_paths`, `memoryless_output_paths`

### `src/continuum_swr_replay/analysis/detection/nonlocal_events.py`

The paper's non-local event detector applied to a decode, on one time-indexed frame.

`theta_power_zscore`, `validate_required_session_data_keys`, `validate_results_position_time_alignment`, `align_results_to_position`, `validate_dot_event_contract`, `get_data_over_time`, `load_data_over_time`, `get_event_types`, `compute_swr_nl_event_counts`

### `src/continuum_swr_replay/analysis/detection/ripple_tetrodes.py`

Rule R-T: per epoch, keep the CA1 band tetrodes that ripple with the rest.

`TetrodeSelection`, `TetrodeSelectionError`, `check_one_channel_per_tetrode`, `select_ripple_tetrodes`

### `src/continuum_swr_replay/analysis/detection/simulated_run.py`

The simulated run of Figure 1, decoded by the switching state-space model.

`splice_remote_stationary`, `make_example_detector`, `estimate_state_space_model`, `SimulatedRun`, `DecodedSimulation`, `simulate_run`, `simulate_and_decode`

### `src/continuum_swr_replay/analysis/event_definition/analysed_time.py`

The analysed-time read point: a decode frame held to the one analysed-time rule.

`stamp_swr_source`, `require_swr_source`, `read_swr_table`, `publish_swr_table`, `excluded_intervals`, `swr_observed_stretches`, `swr_observed_mask`, `sampled_bounds`, `swr_event_labels`, `artifact_event_drops`, `read_artifact_event_drops`, `dropped_for_excluded_time`, `dropped_on_grid`, `with_swr_source`, `excluded_intervals_of`, `excluded_intervals_by_session`, `read_dot_with_swr_source`, `read_verified_dot_with_swr_source`, `analysed_bins`, `immobile_analysed_bins`, `swr_intervals`

### `src/continuum_swr_replay/analysis/event_definition/artifact_intervals.py`

The artifact run folder: one cores file per epoch, and the run's manifest.

`ArtifactRunError`, `EpochCores`, `cores_path`, `empty_cores`, `make_cores`, `validate_cores`, `write_cores`, `read_manifest`, `read_epoch_cores`, `epoch_artifact_intervals`

### `src/continuum_swr_replay/analysis/event_definition/artifact_masks.py`

The artifact mask: the detector's cores, padded from one table.

`core_class`, `artifact_intervals`

### `src/continuum_swr_replay/analysis/event_definition/classified_table.py`

Canonical per-non-local-event classification table.

`match_swr`, `build_classified_nonlocal_event_table`, `assert_drawn_class_matches_ripples`

### `src/continuum_swr_replay/analysis/event_definition/cohort_selection.py`

Shared whole-unit complete-case cohort selection.

`sorted_unit_ids`, `freeze_common_cohort`, `stop_cohort_suffix`, `select_stop_cohort`, `select_event_cohort`, `stop_interval_mask`, `events_in_stops`

### `src/continuum_swr_replay/analysis/event_definition/decode_masks.py`

The HPC decode's masks, from the paper's SWR source.

`require_hpc_decode_sources`, `require_saved_decode_masks`, `hpc_artifact_intervals`, `hpc_ripple_power_inputs`, `HpcDecodeMasks`, `hpc_decode_masks`

### `src/continuum_swr_replay/analysis/event_definition/eligibility.py`

One analysed-time rule for every event cohort, exposure, bin and window.

`DropReason`, `excluded_time`, `artifact_drop_reasons`, `overlaps_dropped_events`, `event_drop_reasons`, `label_drop_reasons`, `drop_labels`

### `src/continuum_swr_replay/analysis/event_definition/epoch_normalizers.py`

An epoch's run-speed baseline, shared by the decoders and the read path.

`run_speed_baseline`

### `src/continuum_swr_replay/analysis/event_definition/gallery_sample.py`

The event gallery's sample: each event's category and ambiguity, and the draw.

`event_catalog`, `stratified_sample`, `group_seed`, `select_events`

### `src/continuum_swr_replay/analysis/event_definition/immobility.py`

Canonical whole-event immobility rules for neural event cohorts.

`EventDefinitionError`, `require_time_index`, `require_canonical_speed_threshold`, `event_speeds_are_immobile`, `whole_event_immobility_mask`, `filter_event_labels_by_speed`, `validate_whole_event_immobility_contract`, `validate_nonlocal_event_probability_contract`, `filter_event_intervals_by_speed`, `intervals_from_labels`, `build_speed_eligible_event_cohort`

### `src/continuum_swr_replay/analysis/event_definition/nonlocal_event_rule.py`

The non-local event rule and the SWR-overlap classification of those events.

`nonlocal_probability`, `nonlocal_state_probability`, `get_confident_times`, `detect_nonlocal_events`, `positive_duration_interval_mask`, `strict_overlap_matrix`, `event_strict_overlap_keys`, `event_strict_overlap_mask`, `nonlocal_event_bins_by_swr`, `get_overlap_times`, `classify_swr_overlap`, `classify_swr_overlap_geometry`, `select_typical_event_by_class`

### `src/continuum_swr_replay/analysis/event_definition/swr_source.py`

The paper's one SWR source: the direct ripple_detection 2.0 Kay inventory.

`SwrSourceError`, `swr_inventory_dir`, `artifact_intervals_dir`, `swr_source_name`, `artifact_run_of`, `swr_epoch_id`, `swr_inventory_paths`, `inventory_manifest`, `swr_epoch_id_of_session`, `swr_source_files`, `swr_inventory_record`, `ripple_power_inputs`, `read_observed`, `artifact_intervals`, `EpochSource`, `epoch_source`, `read_swr_inventory`, `classify`, `swr_events`

### `src/continuum_swr_replay/analysis/event_evidence/spike_evidence.py`

Spike evidence of non-local events: does the prior-excluded spike likelihood favour the decoded location over the animal's?

`build_session_events`, `stored_likelihood_columns`, `stored_event_likelihood`, `window_spike_counts`, `event_metrics`

### `src/continuum_swr_replay/analysis/event_properties/brain_state_reference.py`

Theta, ripple and MUA levels of running and of quiet rest, the references for an event's.

`regime_masks`, `matched_window_means`, `brain_state_reference`

### `src/continuum_swr_replay/analysis/event_properties/class_or_continuum.py`

One class or two: do SWR-coupled and non-SWR non-local events differ by kind or by degree of ripple power?

`sarle_bimodality`, `dip_test`, `kde_modes`, `matched_unimodal_sample`, `matched_lognormal_sample`, `two_class_mixture_sample`, `common_support`, `StepFit`, `fit_step`, `overall_step_share`, `simulate_step_data`, `sustained_level`, `kay_like_intervals`, `kay_criterion`, `simulate_ripple_events`, `event_ripple_power`, `robust_sd`, `distribution_table`, `property_frame`, `design_rows`, `step_table`

### `src/continuum_swr_replay/analysis/event_properties/content_occupancy.py`

Where non-local content sits: each content time bin's track distance to the animal and to the nearest reward well.

`well_positions`, `content_distances`

### `src/continuum_swr_replay/analysis/event_properties/event_analysis.py`

Analysis functions for manipulating and characterizing detected events.

`get_immobile_events`, `get_event_start_end_times`, `get_event_durations_ms`

### `src/continuum_swr_replay/analysis/event_properties/event_statistics.py`

Summary statistics of the immobile non-local events.

`compute_swr_overlap_class_counts`, `overlap_class_statistics`, `boundary_offsets`, `boundary_offset_statistics`, `reference_medians`, `event_signal_statistics`, `power_change_statistics`, `rest_matched_theta_statistics`, `spike_evidence_statistics`, `peak_ripple_mode_statistics`, `threshold_step_statistics`, `most_steps`, `class_medians`, `n_animals_in_direction`, `quadrant_fractions`, `arm_occupancy_shares`, `content_occupancy_shares`, `animal_occupancy_shares`, `n_animals_non_swr_more_at_well`

### `src/continuum_swr_replay/analysis/event_properties/gamma_power.py`

Slow- and fast-gamma power of CA1 on the decode grid: z-scored per epoch, and its change against quiet rest in dB (``power_change_db``).

`band_power`, `zscore_on`, `event_means`, `samples_in_intervals`, `power_change_db`, `epoch_gamma`

### `src/continuum_swr_replay/analysis/event_properties/property_contrast.py`

Uncertainty of the SWR against non-SWR property contrasts: a nested bootstrap.

`duration_strata`, `matched_difference`, `bootstrap_contrasts`, `bootstrap_intervals`

### `src/continuum_swr_replay/analysis/event_properties/start_and_rest_theta.py`

Each non-local event's start distance, and the theta of duration-matched quiet rest.

`start_and_rest_theta`

### `src/continuum_swr_replay/analysis/event_properties/triggered_analysis.py`

Event-triggered analysis: signal windows around events, and their averages.

`get_signal_windows`, `compute_triggered_average_curves`

### `src/continuum_swr_replay/analysis/event_tables/_common.py`

Shared parts of the per-grain table builders.

`align_posterior_to_dot`, `safe_mode`, `first_row_floats`, `strict_mean`, `strict_sum`, `align_count_matrix`, `assert_event_label_alignment`, `assert_single_session_alignment`

### `src/continuum_swr_replay/analysis/event_tables/arm_patch_tables.py`

Canonical arm- and patch-grain builders: one row per (event, arm) / (event, patch).

`build_event_arm_table`, `build_event_patch_table`

### `src/continuum_swr_replay/analysis/event_tables/bouts.py`

Immobility bouts: where a stop begins and ends, and what falls inside one.

`nearest_well_distance_cm`, `immobility_bout_intervals`, `bout_exposure_s`, `summarize_bout_location`, `bout_start_context`, `time_since_stop`, `assign_bout`

### `src/continuum_swr_replay/analysis/event_tables/event_table.py`

Canonical event-grain builders: one row per non-local event.

`build_event_table`, `build_event_sequence_table`

### `src/continuum_swr_replay/analysis/event_tables/prepare_data.py`

The per-session replay-table build and the corpus-level steps around it.

`session_cache_dir`, `behavioral_context_cache_path`, `load_behavioral_context`, `build_session_tables`, `assemble_canonical_tables`, `standardize_event_arm`

### `src/continuum_swr_replay/analysis/event_tables/retrospective_metrics.py`

Per-event retrospective metrics: does an event's replay content align with the arm the animal visited next (or just left)?

`arm_path_well_to_well`, `chosen_arm_fraction`, `chosen_arm_fraction_prev`, `ends_at_next`, `ends_at_prev`, `contains_path_to_next`, `contains_path_from_prev`, `chosen_patch_fraction`, `ends_at_next_patch`, `compute_retrospective_metrics`

### `src/continuum_swr_replay/analysis/event_tables/stop_table.py`

Canonical stop-grain builder: one row per stopping period.

`build_stop_table`, `add_nonlocal_swr_split`

### `src/continuum_swr_replay/analysis/event_tables/table_contracts.py`

Scientific table contracts beyond the generic covariate schema stamp.

`event_bin_total_column`, `validate_event_arm_table`, `validate_event_patch_table`, `validate_requested_standardization`, `validate_canonical_table_contract`, `validate_and_match_fit_results`

### `src/continuum_swr_replay/analysis/event_tables/timebin_tables.py`

Within-stop time-binned tables (Part B).

`build_event_onset_table`, `build_stop_timebin_table`, `concat_session_timebin_tables`

### `src/continuum_swr_replay/tables/event_ids.py`

Stable row identities: the ``event_id`` and ``stop_id`` hashes.

`hash_event_id`, `validate_event_ids`, `hash_stop_id`

### `src/continuum_swr_replay/tables/session_keys.py`

The natural key for one recording session: (animal, session_date, epoch).

`session_key_set`, `n_sessions`, `session_key_frame`, `subset_max_per_animal`, `one_session`, `present_session_columns`, `require_session_unique_ids`

## Covariates

Predictor construction: the timing clocks, the history blocks, the candidate-arm block, standardization, and the name registry.

### `src/continuum_swr_replay/analysis/behavior/covariates/candidate_arm.py`

Shared derivation of candidate-arm predictors.

`broadcast_candidate_arms`, `attach_candidate_arm_covariates`, `eq_indicator`, `eq_indicator_or_nan`

### `src/continuum_swr_replay/analysis/behavior/covariates/clocks.py`

Canonical arrival and outcome-availability clocks for behavioral history.

`trial_arrival_times`, `trial_outcome_times`

### `src/continuum_swr_replay/analysis/behavior/covariates/context.py`

Time-local behavioral location and completed-trial context.

`behavioral_location_from_port_experiences`, `summarize_behavioral_location`, `add_trial_context`, `trial_context_at_times`

### `src/continuum_swr_replay/analysis/behavior/covariates/definitions.py`

Machine-readable scientific definitions for model-facing covariates.

`Availability`, `QueryClock`, `CovariateDefinition`, `definition_for`, `definitions_frame`

### `src/continuum_swr_replay/analysis/behavior/covariates/history/_common.py`

Shared port-experience validation for the history covariates.

_No module-level public symbols._

### `src/continuum_swr_replay/analysis/behavior/covariates/history/diagnostics.py`

Diagnostic per-animal fitted-alpha leaky-Q learner.

`q_all_arm`

### `src/continuum_swr_replay/analysis/behavior/covariates/history/reward_history.py`

Reward-history covariates: rolling reward counts and recent reward rates over an arm's last visits, plus stop-level reward histories.

`rolling_reward_count_all_events`, `recent_reward_rate_all_events`, `recent_trial_reward_rate`, `lagged_trial_rewards`, `reward_history_basis_features`

### `src/continuum_swr_replay/analysis/behavior/covariates/history/value_block.py`

Canonical qvals value-block and arm-uncertainty joins.

`load_contingency`, `attach_contingency_value_block`, `attach_arm_uncertainty`, `assert_qvals_aligned`

### `src/continuum_swr_replay/analysis/behavior/covariates/history/value_uncertainty.py`

Value and uncertainty covariates computed from the behavioral stream: the canonical leaky-Q value history and the model-free behavioral-uncertainty covariates.

`canonical_port_value_history`, `canonical_trial_switch_history`, `validate_uncertainty_inputs`, `add_uncertainty_covariates`

### `src/continuum_swr_replay/analysis/behavior/covariates/history/visit_recency.py`

Visit-recency covariates: how recently (and whether ever) each arm was visited, at event time.

`center_visit_recency_within_event`, `spline_visit_recency_within_event`, `visit_recency_all_events`, `arm_never_visited_all_events`, `visit_recency_and_never_visited_all_events`

### `src/continuum_swr_replay/analysis/behavior/covariates/interactions.py`

The interaction (product) design columns, each declared once with its factors.

`factor_source`, `interaction_product`, `attach_interactions`

### `src/continuum_swr_replay/analysis/behavior/covariates/spatial.py`

Graph-spatial helpers for per-arm predictors and QC metrics.

`dist_to_arm_from_well`, `arm_dist_rank`, `arm_adjacency`, `path_continuity`

### `src/continuum_swr_replay/analysis/behavior/spatial_mapping.py`

Map decoded replay positions to spatial units (arms, patches).

`map_estimate_to_patch`, `map_estimate_to_arm`, `arm_to_well`, `well_to_arm_set`, `single_arm_from_well`, `arms_of_wells`, `arm_to_patch`, `build_well_to_arm_map`, `stay_indicator_from_patches`, `stay_indicator`

### `src/continuum_swr_replay/analysis/event_tables/content_response.py`

Which per-event response the content ladders are fit on.

`resolve_response_column`, `default_response_for_estimator`, `resolve_response_choice`, `validate_fraction_response`, `renormalize_fraction_over_candidates`

### `src/continuum_swr_replay/analysis/event_tables/uncertainty.py`

Per-event decoder-confidence summaries from the arm-marginal posterior.

`posterior_peak_mean`, `posterior_entropy_mean`, `posterior_hpd_width`, `posterior_hpd_widths`

### `src/continuum_swr_replay/analysis/replay_models/content/uncertainty_content.py`

Model-light estimands for reward-belief uncertainty and replay content.

`assign_within_arm_quantiles`, `build_trial_leaf_table`, `hierarchical_quartile_profile`, `paired_quartile_contrasts`, `summarize_mu_coefficients`, `within_arm_horserace`, `summarize_horserace`, `restrict_to_analysis_cohort`, `analysis_cohort_counts`, `select_example_session`, `quartile_profiles_and_contrasts`

### `src/continuum_swr_replay/analysis/replay_models/covariate_timing.py`

Per-covariate pre/post-outcome timing registry + model-build enforcement.

`timing_of`, `assert_predictor_timing`

### `src/continuum_swr_replay/tables/covariate_registry.py`

Covariate naming contract -- the runtime half of the 2026-08-21 rename.

`row_keys`, `Entry`, `canonical_names`, `RegistryError`, `validate`, `assert_canonical_vocabulary`, `assert_unique_scientific_rows`, `stamp_covariate_schema`, `require_covariate_schema`

### `src/continuum_swr_replay/tables/standardization.py`

Pooled-global standardization of continuous covariates (`<col>_z`).

`attach_standardized`, `assert_standardized_present`, `assert_standardized_matches_reference`, `z_reference_path`, `save_reference`, `load_reference`

## Specs

The single source of truth for every model ladder, and the design constants the ladders share with the covariate builders.

### `src/continuum_swr_replay/analysis/behavior/design_constants.py`

Design constants shared by the covariate builders and the model ladders.

_No module-level public symbols._

### `src/continuum_swr_replay/analysis/replay_models/model_specs.py`

Model definitions for replay content analysis.

`ModelSpec`, `default_content_model_names`, `PerBinTerm`, `time_course_spline`, `PerBinRateModelSpec`, `longest_history_window_s`, `leaf_only_predictors`, `rung_predictors`, `restrict_to_leaf_arms`, `leaf_only_cohort`, `ComparisonRung`, `resolve_comparison_rungs`, `add_arm_fixed_effects`

## Fitters

Estimators and the per-session fit machinery each runner calls.

### `src/continuum_swr_replay/analysis/behavior/history_basis.py`

The raised-cosine history basis and the projection of a lag matrix onto it.

`raised_cosine_history_basis`, `reward_basis_from_lags`

### `src/continuum_swr_replay/analysis/replay_models/_fit_diagnostics.py`

Shared fit-diagnostics helpers: convergence, separation, warning capture, flag counts.

`fit_status`, `separated_terms`, `capture_fit_warnings`, `count_fit_quality_warnings`, `flagged_separated`, `fit_flag_counts`, `flagged_nonconverged`, `fold_flag_counts`

### `src/continuum_swr_replay/analysis/replay_models/choice/choice_prediction.py`

Does replay content predict the next choice?

`predict_arm_fractions`, `build_residualized_choice_dataset`, `build_choice_dataset`, `fit_choice_model`, `choice_swr_split_models_for_source`, `fit_choice_blocks`

### `src/continuum_swr_replay/analysis/replay_models/choice/shuffles.py`

Null-distribution generators for replay content QC.

`circular_time_shuffle`, `stop_label_shuffle`, `next_arm_circular_shift`

### `src/continuum_swr_replay/analysis/replay_models/content/dirichlet_multinomial.py`

Dirichlet-multinomial fitter for replay content over candidate categories.

`event_loglik`, `multinomial_log_constant`, `fit_dirichlet_multinomial`, `cross_validate_dirichlet_multinomial`, `concentration_cols_for_cohort`, `animal_pooled_log_phi`, `resolve_pooled_log_phi`, `fit_sessions_dirichlet_multinomial`

### `src/continuum_swr_replay/analysis/replay_models/content/simulation.py`

Synthetic-data generators for validating the conditional Poisson model.

`sample_arm_design`, `simulate_counts`, `simulate_with_event_covariates`, `simulate_dirichlet_multinomial_counts`, `simulate_autocorrelated_counts`, `SyntheticEnv`, `make_synthetic_env`, `simulate_posterior_bumps`, `simulate_hard_vs_soft_counts`, `make_bandit_track_graph`, `simulate_full_pipeline`

### `src/continuum_swr_replay/analysis/replay_models/fitting.py`

Per-session model fitting for the replay content model hierarchy.

`poisson_estimator_name`, `event_clustered_covariance`, `within_group_full_rank`, `fit_conditional_poisson`, `parse_unidentified_terms`, `conditional_linear_predictor`, `fit_null_model_conditional`, `fit_null_model_conditional_logit`, `fit_conditional_logit`, `conditional_logit_or_reason`

### `src/continuum_swr_replay/analysis/replay_models/per_bin_design.py`

The (stop x bin) point-process design and the bases it is built on.

`attach_is_swr_to_onsets`, `reward_kernel_weights`, `reconstruct_reward_kernels`, `reconstruct_self_history_kernels`, `arrival_spanning_stop_ids`, `aggregate_to_timebins`

### `src/continuum_swr_replay/analysis/replay_models/rate/duration_aware_history.py`

The duration-aware per-bin history design on real events, and its readouts.

`attach_event_ends`, `at_risk_events`, `build_design`, `stack_sessions`, `history_kernels`, `lag_support`, `onset_offset_collinearity`, `fit_pooled_history`

### `src/continuum_swr_replay/analysis/replay_models/rate/duration_history_simulation.py`

Simulated non-local events with duration, for the per-bin history design.

`HistoryKernel`, `Scenario`, `simulate_interval_streams`, `true_lag_kernel`, `fit_history_direction`, `summarize_direction`, `simulate_pooled_design`

### `src/continuum_swr_replay/analysis/replay_models/rate/per_bin_rate_model.py`

Per-bin conditional-intensity Poisson model for replay event onsets (Family B).

`freeze_per_bin_cohort`, `apply_arrival_window`, `require_frozen_per_bin_design`, `fit_per_bin_rate_model`, `fit_per_bin_glm`, `cross_validate_per_bin_rate_model`, `per_bin_rate_gof`

### `src/continuum_swr_replay/analysis/replay_models/rate/per_stop_rate_model.py`

Poisson rate model for replay event occurrence.

`fit_per_stop_rate_model`, `cross_validate_per_stop_rate_model`, `incomplete_stops`, `fit_per_stop_rate_hierarchy`, `fit_event_source_rate_hierarchy`

### `src/continuum_swr_replay/analysis/replay_models/rate/ridge_history_cv.py`

Ridge-penalised, cross-validated self vs self+cross history comparison.

`RidgeFit`, `ridge_design`, `fit_ridge_poisson`, `heldout_loglik`, `assign_stop_folds`, `fit_ridge_poisson_batch`, `nested_cv`, `ridge_kernels`, `compare_cross_history`

## Runners

Command-line entry points that fit or plot a ladder.

### `scripts/behavior/build_qvals_table.py`

Per-stop RL-model values via DataJoint — canonical, complete, timestamp-aligned.

`run_build_qvals_table`, `main`

### `scripts/decoding/_decode_runner.py`

The flow both decode runners share, and the record of what differs.

`save_results_and_model`, `organize_results`, `run_decode`, `parse_decode_args`

### `scripts/decoding/measure_decoder_quality.py`

Measure each analysed epoch's decoder quality: Local-state error, state fractions, calibration.

`measure_epoch`, `main`

### `scripts/decoding/run_decoding_bandit.py`

Main script for running continuum neural decoding on spatial bandit task data.

`generate_figurl`, `run_decode`, `main`

### `scripts/decoding/run_decoding_contfrag.py`

Run Continuous-Fragmented (ContFrag) decoding on spatial bandit task data.

`contfrag_detector_kwargs`, `run_decode`, `main`

### `scripts/detection/run_artifact_detection.py`

Detect LFP and spike artifacts in every epoch's CA1 data, on CPU.

`detector_constants`, `run_settings`, `reference_lfp`, `run_epoch`, `summary`, `register_epochs`, `build_parser`, `main`, `run`

### `scripts/detection/run_carey_detection.py`

Run Carey's candidate detector on every epoch's CA1 LFP and multiunit spikes.

`swr_run_inputs`, `observation_table`, `run_epoch`, `summary`, `run_settings`, `main`, `run`

### `scripts/tables/_table_runner.py`

The runner parts the table drivers share.

`find_sessions`, `find_contfrag_sessions`, `read_disjoint_shards`, `publish_replay_tables`, `publish_contfrag_tables`

### `scripts/tables/assemble_contfrag_shards.py`

Assemble disjoint per-animal ContFrag shards into one canonical bundle.

`assemble_contfrag_shards`, `main`

### `scripts/tables/assemble_replay_content_shards.py`

Assemble disjoint Replay table shards into one canonical candidate bundle.

`assemble_replay_content_shards`, `assemble_replay_stop_shards`, `main`

### `scripts/tables/build_timebin_tables.py`

Build the Part B within-stop time-binned tables over all sessions.

`build_one`, `validate`, `main`

### `scripts/tables/convert_tables_to_parquet.py`

Convert the materialized canonical tables from pickle to parquet.

`canonical_tables`, `convert`, `main`

### `scripts/tables/prepare_contfrag_replay_content_data.py`

Build canonical replay-content tables from ContFrag decoding results.

`build_contfrag_tables`, `build_parser`, `configure_paths`, `main`

### `scripts/tables/prepare_replay_content_data.py`

Build the four canonical replay-content tables across all sessions.

`populate_behavioral_context_cache`, `build_stop_table_scope`, `run_prepare_data`, `main`

### `scripts/tables/refresh_contfrag_behavioral_covariates.py`

Rebuild ContFrag tables while reusing immutable posterior aggregates.

`refresh_contfrag_behavioral_covariates`, `main`

### `scripts/tables/regenerate_stop_table.py`

Regenerate ``stop_table`` with the new count columns.

`build_one`, `validate`, `main`

### `scripts/tables/write_regeneration_manifest.py`

Validate candidate covariate bundles and write a release manifest.

`write_manifest`, `main`

## Aggregation

Across-session and across-animal summaries, contrasts and model comparison over the per-session fits.

### `src/continuum_swr_replay/analysis/event_tables/summaries.py`

Per-event replay summaries from decoder outputs.

`posterior_to_arm_marginal`, `arm_map_from_posterior`, `n_nonlocal_bins`, `nonlocal_weight_sum`, `posterior_mass_total`, `arm_count_hard`, `arm_count_soft`, `arm_posterior_mass`, `arm_fraction_from_event_total`

### `src/continuum_swr_replay/analysis/event_tables/trajectory.py`

Per-event replay-trajectory rollups, shared by both event-table builders.

`brain_state_means`, `ahead_behind_summary`, `uncertainty_extra`, `trajectory_dynamics`, `compute_trajectory_rollups`

### `src/continuum_swr_replay/analysis/replay_models/choice/retrospective.py`

Retrospective alignment of replay content with the next choice.

`at_well_event_cohort`, `freeze_retrospective_cohort`, `retrospective_alignment_with_shuffle`, `family_unit_counts`, `retrospective_status`, `select_event_subset`, `retrospective_alignment_by_session`

### `src/continuum_swr_replay/analysis/replay_models/choice/stay_leave.py`

Replay content × reward outcome → behavioral change.

`StopSummaryConfig`, `stop_replay_summary`, `fit_is_stay_logistic_per_session`, `fit_is_stay_pooled_gee`, `role_content_column`, `fit_stay_leave_roles`, `cross_session_role_summary`, `pooled_gee_role_summary`, `smallest_adjusted_p`, `fit_status_counts`

### `src/continuum_swr_replay/analysis/replay_models/choice/switch_analysis.py`

Switch-aligned timing analysis for patch-switch prediction.

`build_switch_aligned_table`, `build_control_aligned_table`, `stop_patch_replay`

### `src/continuum_swr_replay/analysis/replay_models/cohort_audit.py`

Per-model cohort / drop-profile audit records.

`is_qvals_sourced_column`, `is_qvals_gap_column`, `attach_has_qvals_row`, `qvals_gap_event_ids`, `cohort_record`, `fitted_cohort_record`, `classify_drops`, `event_drop_buckets`, `stop_own_missing`, `drop_profile`, `escape_note_field`, `estimator_skip_note`, `records_from_common_cohort_provenance`, `write_cohort_audit`, `term_diagnostics`

### `src/continuum_swr_replay/analysis/replay_models/content/model_checking.py`

Model checking utilities for the replay content model hierarchy.

`cross_validate_conditional_poisson`, `compute_pearson_residuals`

### `src/continuum_swr_replay/analysis/replay_models/content/model_comparison.py`

Compare nested models via AIC, BIC, and likelihood ratio tests.

`fit_metric_statistics`, `compare_nested_models`, `summarize_comparison`, `run_full_hierarchy_comparison`

### `src/continuum_swr_replay/analysis/replay_models/contrast_aggregation.py`

Shared audited aggregation for the is_swr contrast ladders (rate + content).

`std_factor`, `sd_covariate`, `separated_for_term`, `unidentified_for_term`, `TermSessionGate`, `gate_term_sessions`, `session_se`, `typical_session_se`, `pct_significant_sessions`, `gated_term_statistics`, `aggregate_contrast`, `specs_from_ladder`, `concentration_specs`, `audited_content_contrast`, `renormalized_rungs`, `contrast_fit_inputs`, `select_outcome_stratum`, `sd_reference`, `prepare_cohorts`, `assemble_arm`, `assemble_patch`, `prepare_session`

## Readers and QA

Artifact I/O, provenance gating, and the diagnostics that check the tables and the fits.

### `scripts/checks/_memoryless_reference.py`

The Widloski & Foster detector as its MATLAB computes it, for a sorted 1-D track.

`ReferenceThresholds`, `decode_windows`, `center_and_spread`, `forward_jumps`, `runs_of_kept`, `merge_runs`, `dispersion_bins`, `ReferenceDetection`, `detect`, `set_aside_pairs`, `shuffle_statistics`, `shuffle_p_values`

### `scripts/checks/_report.py`

Report helpers shared by the QA and PFC audit scripts: the run's command line, markdown tables.

_No module-level public symbols._

### `scripts/checks/audit_event_patch_fraction.py`

Audit a candidate event-patch table against its pre-change reference.

`assert_row_and_column_order`, `real_arm_bins`, `assert_fraction_definition`, `assert_matches_arm_fractions`, `main`

### `scripts/checks/audit_stop_table_reward_history.py`

Audit a candidate stop table against its pre-change reference.

`assert_row_and_column_order`, `incomplete_share`, `assert_prefix_per_session`, `main`

### `scripts/checks/check_data_availability.py`

Check data availability for all sessions in the spatial bandit task.

`check_position`, `check_spikes`, `check_ripple`, `check_theta`, `check_dios`, `check_session`, `main`

### `scripts/checks/check_fit_provenance.py`

Report whether every fitted artifact is still current for the tables it read.

`artifact_for_sidecar`, `find_sidecars`, `resolve_input`, `check_artifact`, `propagate_stale`, `format_line`, `main`

### `scripts/checks/compare_memoryless_to_reference.py`

Run the port and the reference's own arithmetic on one input, and compare.

`window_counts`, `overlap`, `shuffle_permutations`, `p_values`, `passes`, `compare`, `reference_row`, `fields_and_thresholds`, `compare_variants`, `decode_sweep_cell`, `sweep_totals`, `main`

### `scripts/checks/compare_to_golden_baseline.py`

Diff a candidate replay_content root against a frozen golden baseline.

`main`

### `scripts/checks/compute_hpc_raw_stats.py`

Compute raw-data QC stats for hippocampal clusterless decoding.

`main`

### `scripts/checks/diagnose_trial_detection.py`

Diagnostic figures for trial-detection and predictor semantics.

`figure_trial_sanity`, `figure_predictor_collapse`, `main`

### `scripts/checks/freeze_regression_baseline.py`

Freeze the current canonical tables and model outputs as a read-only baseline.

`freeze`, `main`

### `scripts/checks/mutation_check.py`

Objective AST mutation check for value-critical builder logic.

`check`, `main`

### `scripts/checks/run_quality_metrics.py`

Compute decoding quality metrics across all sessions.

`maybe_merge_raw_stats`, `load_all_sessions`, `print_summary`, `plot_diagnostics`, `main`

### `scripts/checks/summarize_cohort_audits.py`

Concatenate and flag every ``*_cohort_audit.csv`` under the outputs dir.

`load_all`, `flag`, `main`

### `scripts/checks/summarize_stage6_fdr.py`

Print Stage 6's per-role cross-session FDR table for a saved bundle.

`main`

### `scripts/checks/table_parity.py`

Regenerate the canonical tables into a scratch root and diff two roots.

`regen`, `frame_problems`, `diff`, `main`

### `scripts/checks/write_epoch_register_doc.py`

Regenerate the parts of ``docs/epoch_register.md`` that mirror the register.

`rule`, `reason_codes`, `counts`, `entries`, `render`, `main`

### `scripts/checks/write_pipeline_map.py`

Generate the symbol-anchored regression pipeline code map.

`render`, `main`

### `scripts/detection/sweep_memoryless_controls.py`

What each detector does with activity that is not one clean replay.

`main`

### `scripts/detector_comparison/_report.py`

What the two detection-comparison reports share.

`caption_with_pending`, `read_run`, `with_pending`, `per_session`

### `scripts/detector_comparison/add_state_shares.py`

Add each compared event's Local, No-Spike, non-local and uncertain shares to an event features run.

`shares_path`, `epoch_state_shares`, `main`

### `scripts/detector_comparison/report_event_features.py`

Report one event features run: its features per session and per animal.

`distribution`, `epoch_status`, `context_by_session`, `feature_tables`, `missing_by_session`, `never_missing`, `with_spikes_per_row`, `evidence_coverage_by_session`, `rejection_by_session`, `rejection_by_animal`, `memoryless_cohort_by_session`, `comparator_only_events`, `comparator_only_table`, `stratum_labels`, `report_tables`, `draw_figures`, `write_report`, `main`

### `scripts/detector_comparison/report_inventory_comparison.py`

Report one inventory comparison run: its counts per session and per animal.

`epoch_status`, `oriented_pairs`, `match_strength_by_session`, `match_strength_by_animal`, `swr_matrix_by_animal`, `matrix_population`, `paired_labels_by_session`, `definition_disagreement_by_session`, `labels_by_animal`, `report_tables`, `draw_figures`, `write_report`, `main`

### `scripts/detector_comparison/run_event_features.py`

Event features of every compared event, one epoch per task.

`memoryless_input_mismatch`, `epoch_tables`, `run_epoch`, `summary`, `run_settings`, `main`, `run`

### `scripts/detector_comparison/run_inventory_comparison.py`

Compare the replay-detection inventories, one epoch at a time.

`bounds_mismatch`, `run_epoch`, `summary`, `run_settings`, `main`, `run`

### `scripts/detector_comparison/sweep_memoryless_heldout.py`

Does each detector's content reproduce in cells it never saw?

`main`

### `scripts/event_evidence/analyze_event_spike_evidence.py`

Spike-evidence reality analysis for non-SWR non-local events.

`find_sessions`, `load_marks`, `events_with_map_in_left_out_bins`, `analyze_session`, `render_session_figures`, `summarize`, `per_animal_medians`, `main`

### `scripts/event_evidence/run_state_evidence.py`

Held-out spike evidence for each decoder state, on every canonical HPC decode.

`caveats`, `require_cpu`, `decode_files`, `decode_file_stats`, `reuse_reason`, `canonical_epochs`, `load_marks_and_training_mask`, `stored_log_likelihood`, `refit_encoding`, `relative_error`, `output_paths`, `LabelSet`, `label_probabilities`, `checked_halves`, `time_held_out_record`, `epoch_table`, `label_set_table`, `run_settings`, `run_epoch`, `run_epochs`, `figure_bouts`, `figure_calibration`, `figure_silence`, `write_csv`, `readme`, `summarize`, `build_parser`, `main`

### `scripts/event_evidence/split_state_evidence_by_swr.py`

Held-out spike evidence of Non-Local bouts, split by SWR coupling.

`label_epoch`, `main`

### `scripts/event_properties/bootstrap_event_properties.py`

Bootstrap the SWR against non-SWR property contrasts: class medians, their difference, and the duration-matched difference.

`load_events`, `main`

### `scripts/event_properties/measure_brain_state_reference.py`

Measure each epoch's theta, ripple and MUA levels during running and during quiet rest.

`main`

### `scripts/event_properties/measure_event_gamma.py`

Measure slow- and fast-gamma power during each non-local event, and during running and quiet rest.

`out_dir`, `epoch_gamma`, `epochs_to_measure`, `main`

### `scripts/event_properties/measure_event_start_and_rest_theta.py`

Measure each event's start distance and the theta-like share of duration-matched quiet rest.

`measure_epoch`, `main`

### `scripts/event_properties/measure_ripple_class_structure.py`

One class or two: do SWR-coupled and non-SWR non-local events differ by kind or by degree of ripple power?

`out_dir`, `measure_epoch`, `epoch_identifier`, `run_measure`, `load_events`, `distribution_simulation`, `plot_distributions`, `step_simulation`, `plot_property`, `spot_check_events`, `plot_spot_checks`, `run_analyze`, `main`

### `scripts/pfc/calibrate_pfc_kernel_timing.py`

What the mPFC kernel estimator does to the timing and size of a known response.

`parse_arguments`, `main`

### `scripts/pfc/check_pfc_ridge_lambda0.py`

Check that the penalised solver reproduces maximum likelihood at lambda = 0.

`audited_basis`, `audited_designs`, `rebuild_session`, `compare_session`, `build_report`, `main`

### `scripts/pfc/compare_pfc_cohorts.py`

Write the before/after audit of the mPFC unit-GLM stop cohort.

`read_sidecars`, `runner_code`, `provenance_table`, `bundle_sessions`, `session_cohort_table`, `coverage_summary`, `alignment_prose`, `gate_counts`, `significance_table`, `population_comparison`, `try_read_sidecars`, `same_events_precondition`, `event_count_agreement`, `shared_statistic_differences`, `matched_control_arrays`, `matched_control_significance`, `fitter_table`, `session_fit_counts_table`, `worker_time_table`, `curves_lacking_shrinkage_columns`, `peak_shrinkage`, `shrinkage_table`, `implied_prior_per_lag`, `empirical_bayes_sigma`, `heldout_fraction_table`, `restrict_to_sessions`, `legacy_fdr_table`, `separated_column_table`, `engagement_overlap_table`, `null_window_table`, `fold_spread_table`, `sigma_sensitivity_table`, `estimator_provenance_table`, `rerun_sections`, `build_report`, `geometry_section`, `build_estimator_report`, `main`

### `scripts/pfc/compare_pfc_event_set_bundles.py`

Does an extended-event-set mPFC bundle reproduce the canonical bundle where they overlap?

`compare_session`, `main`

### `scripts/pfc/compare_pfc_fitters.py`

Write the equivalence audit of the batched JAX fitter against statsmodels.

`bundle_stems`, `load_bundle`, `compare_statistics`, `compare_modulations`, `compare_diagnostics`, `parse_time_file`, `build_report`, `main`

### `scripts/pfc/compare_pfc_marked_family_trees.py`

How much do two marked-family bundle trees differ for the same sessions and models?

`compare_pair`, `main`

### `scripts/pfc/diagnose_pfc_event_definition.py`

Compare the PFC path's old, run-time event definition against the canonical one.

`build_old_event_sets`, `total_overlap_seconds`, `match_by_overlap`, `build_comparison`, `get_command_line_arguments`, `main`

### `scripts/pfc/measure_pfc_mark_context.py`

What the marked-kernel content mark is entangled with, session by session.

`load_current_arm_marks`, `group_context_table`, `summarize`, `build_report`, `main`

### `scripts/pfc/measure_pfc_null_geometry.py`

What the mPFC held-out null's windows look like, beside the real events.

`read_sidecar`, `spacing_summary`, `draw_spacing_summary`, `session_window_sets`, `spacing_section`, `overlap_section`, `non_positive_section`, `excluded_section`, `main`

### `scripts/pfc/part4_rate_threshold_sensitivity.py`

How much do Part 4's mPFC values depend on the firing-rate gate?

`gate_values`, `spikes_on_fitted_bins`, `raw_units`, `funnel`, `band_rows`, `main`

### `scripts/pfc/run_pfc_marked_kernels.py`

Fit content-marked mPFC replay kernels on the canonical at-well cohort.

`marked_artifact_path`, `marked_companion_paths`, `build_fit_contract`, `run_analysis`, `get_command_line_arguments`, `main`

### `scripts/pfc/run_pfc_marked_model_family.py`

Fit the recommended descriptive mPFC marked-kernel model family.

`available_marked_classes`, `model_slope_seed`, `resolve_model_names`, `family_artifact_path`, `family_companion_paths`, `build_fit_contract`, `run_analysis`, `get_command_line_arguments`, `main`

### `scripts/pfc/run_pfc_stop_context.py`

Stop-level context of the mPFC replay response, pooled across sessions (Part 4).

`default_output_dir`, `bundle_path`, `pfc_left_out`, `selected_sessions`, `current_table_hashes`, `cached_is_current`, `assemble`, `unit_table`, `fit`, `get_command_line_arguments`, `main`

### `scripts/pfc/summarize_pfc_marked_family.py`

Population summary of the mPFC marked-model family bundles.

`MixedContractError`, `load_bundles`, `joint_versus_marginal`, `plot_model`, `build_report`, `main`

### `scripts/workflows/run_refits.py`

Rerun the CPU model fits and the figures that read them, in dependency order.

`Run`, `select`, `missing_prerequisites`, `run_env`, `schedule`, `sidecar_times`, `written_artifacts`, `fit_roots`, `check_provenance`, `print_plan`, `main`

### `src/continuum_swr_replay/analysis/event_evidence/state_evidence/evidence.py`

Held-out evidence of every window, by tetrode half (see the package notes).

`held_out_evidence`, `window_evidence`

### `src/continuum_swr_replay/analysis/event_evidence/state_evidence/io.py`

Code versions and the decode's stamp, shared by the state-evidence runners.

`code_version`, `saved_decode_masks`

### `src/continuum_swr_replay/analysis/event_evidence/state_evidence/summary.py`

Summary statistics of the per-window table, over epochs nested in animals.

`sign_flip_p`, `stratified_bootstrap_ci`, `ols_slope_ci`, `epoch_summaries`, `epoch_summary_kind`, `epoch_slopes`, `nested_mean_intervals`, `AnimalTTest`, `animal_t_test`, `paired_bouts`, `bout_animal_summaries`, `bout_statistics`, `calibration_statistics`, `rate_calibration`, `coordinated_silence`, `summarize_state_evidence`, `split_non_local_by_swr`

### `src/continuum_swr_replay/analysis/event_evidence/state_evidence/terms.py`

Per-tetrode likelihood terms, and the encodings and decode files they come from.

`load_detector`, `encoding_inputs`, `load_encoding`, `encoding_comparison`, `event_likelihood`, `nearest_interior_bins`, `interior_bin_mask`, `subnormal_occupancy_bin_mask`, `TetrodeTerms`, `per_tetrode_terms`, `time_folds`, `training_spike_bins`, `running_only_masks`, `spikes_in_bins`

### `src/continuum_swr_replay/analysis/event_evidence/state_evidence/windows.py`

Speed regimes and the windows scored: bouts, matched windows, calibration windows.

`keyed_rng`, `regimes`, `analysed_regimes`, `state_bouts`, `matched_windows`, `calibration_windows`, `build_windows`, `tetrode_halves`, `null_partners`, `window_folds`

### `src/continuum_swr_replay/analysis/replay_models/content/fit_outputs.py`

Where content-ladder fits land, and how their numbers are read.

`resolve_contfrag_paths`, `canonical_output_subdir`, `canonical_output_subdirs`, `partial_ladder_subdir`, `belief_chain_output_subdir`, `scale_label`, `frame_estimator`

### `src/continuum_swr_replay/decoding/files.py`

The one naming rule for a decode's files.

`non_local_threshold_percent`, `data_over_time_suffix`, `DecodePaths`, `decode_paths`, `epoch_identifier_of`, `results_exist`

### `src/continuum_swr_replay/decoding/identifiers.py`

The epoch identifier: how a decode is named, built and read back.

`TrainingType`, `nwb_file_name`, `nwb_stem`, `session_name`, `parse_session_name`, `parse_epoch_name`, `parse_nwb_filename`, `EpochKey`, `format_epoch_identifier`, `decode_identifier`, `parse_epoch_identifier`

### `src/continuum_swr_replay/decoding/results.py`

The one reader of a decode's ``{epoch_identifier}_results.nc``, and of the clusterless detector it saved beside it (:func:`load_saved_detector`); :func:`load_results` reads an epoch's saved decode back whole: its results, fitted detector and environment.

`open_results`, `load_saved_detector`, `load_results`

### `src/continuum_swr_replay/tables/io.py`

On-disk format for the canonical analysis tables.

`umask_file_mode`, `publish_canonical`, `parquet_path`, `resolved_table_path`, `write_table`, `write_atomically`, `read_table`, `read_attrs`, `table_exists`

### `src/continuum_swr_replay/tables/parity.py`

Keyed table-to-table parity checks shared across the pipeline.

`diff_table`, `compare_table_roots`

### `src/continuum_swr_replay/tables/versions.py`

Centralized schema/version literals for the covariate pipeline.

`assert_versions_compatible`

## Unassigned

Modules under a mapped root that no stage pattern claims. A module landing here means `write_pipeline_map.py` needs a stage rule, not that the module is unused.

### `scripts/replay_models/choice/run_retrospective_alignment.py`

Stage 5B: per-session retrospective replay-content alignment.

`fit_input_paths`, `main`

### `scripts/replay_models/choice/run_stay_leave.py`

Stage 6: replay content × reward outcome → behavioral change.

`fit_input_paths`, `summary_path`, `fit_path`, `write_summary_provenance`, `log_cross_session_role_summary`, `main`

### `scripts/replay_models/content/analyze_content_occupancy.py`

Content occupancy of non-local events — where the decoded activity represents.

`find_sessions`, `session_content_distances`, `arm_occupancy`, `analyze`, `group_shares`, `render`, `main`

### `scripts/replay_models/content/analyze_uncertainty_content.py`

Build trial-level reward-belief uncertainty/replay-content results.

`mu_inputs`, `build_parser`, `resolve_models_dir`, `main`

### `scripts/replay_models/content/compare_content_estimators.py`

Fit three replay-CONTENT estimators on one at-well cohort and compare them.

`log_phi_figure_rung`, `plot_coefficient_agreement`, `plot_log_phi_by_session`, `write_caption`, `main`

### `scripts/replay_models/content/compare_event_source_models.py`

Run the replay-content analysis suite on one event source and save fits.

`run_switch_aligned`, `main`

### `scripts/replay_models/content/fit_replay_content_models.py`

Fit the replay content model hierarchy on pre-computed data.

`content_fit_inputs`, `print_model_results`, `format_comparison_table`, `run_content_fit`, `main`

### `scripts/replay_models/content/fit_swr_contrast_content.py`

Fit the beta-bernoulli replay-CONTENT ladder (arm + patch), is_swr version.

`main`

### `scripts/replay_models/content/generate_replay_content_figures.py`

Generate all figures for the replay content analysis.

`bundle_dir`, `all_models_input_paths`, `output_dir`, `build_parser`, `main`

### `scripts/replay_models/content/quantify_swr_overlap_geometry.py`

Quantify the non-local/SWR overlap sub-cases on real data.

`quantify_session`, `print_report`, `main`

### `scripts/replay_models/coupling/fit_swr_power_model.py`

Per-event SWR-coupling strength (ripple power) vs behavioral covariates.

`build_parser`, `main`

### `scripts/replay_models/rate/fit_cross_history_rate.py`

Bivariate per-bin conditional intensity: does one non-local stream's history drive the other's rate?

`main`

### `scripts/replay_models/rate/fit_cross_history_ridge_cv.py`

Ridge cross-validated comparison of self vs self+cross history, per animal.

`main`

### `scripts/replay_models/rate/fit_duration_aware_history_rate.py`

Duration-aware self- and cross-history of the per-bin event rate.

`main`

### `scripts/replay_models/rate/fit_per_bin_rate_models.py`

Fit the per-bin conditional-intensity ladder (Family B) and persist it.

`fit_input_paths`, `main`

### `scripts/replay_models/rate/fit_per_stop_rate_models.py`

Fit the per-stop replay-rate hierarchy (R0-R1) and persist it.

`fit_input_paths`, `main`

### `scripts/replay_models/rate/fit_swr_contrast_rate.py`

Fit the per-bin non-local rate models, `is_swr` version (see docs/models_table.md).

`fit_input_paths`, `main`

### `scripts/replay_models/rate/fit_uncertainty_rate.py`

Fit the per-bin replay rate vs behavioral-uncertainty / value, by population.

`fit_input_paths`, `main`

### `scripts/replay_models/rate/plot_model_checking.py`

Point-process model checking for the per-bin ladder (Family B).

`all_models_path`, `all_models_input_paths`, `arrival_retention_note`, `plot_gof_ladder`, `isi_autocorrelation_window`, `plot_rescaled_isi_autocorrelation`, `plot_example_quartet`, `build_parser`, `main`

### `scripts/replay_models/rate/plot_per_bin_rate_models.py`

Plot the per-bin conditional-intensity ladder (Family B) from its tidy output.

`all_models_path`, `all_models_input_paths`, `plot_reward_kernel`, `plot_selfhistory`, `build_parser`, `main`

### `scripts/replay_models/rate/plot_per_bin_rate_summary.py`

Six-panel summary figure for the per-bin rate ladder (Family B).

`all_models_path`, `all_models_input_paths`, `build_parser`, `main`

### `scripts/replay_models/rate/plot_uncertainty_rate.py`

Plot replay rate vs uncertainty/value by population from the tidy fit output.

`report`, `plot`, `build_parser`, `main`

### `src/continuum_swr_replay/analysis/replay_models/coefficient_statistics.py`

Across-session summaries of what the per-session fits wrote.

`fit_rows`, `require_one_fit_per_session`, `coefficient_statistics`, `coefficient_family_statistics`, `animal_median_bootstrap`, `gated_coefficient_statistics`, `gated_exp_median_ratio`, `paired_coefficient_statistics`, `gated_coefficient_difference_statistics`, `aic_difference_statistics`, `separated_fraction`, `known_fraction`, `fits_carrying`, `term_separation_statistics`, `rung_separation_range`, `audited_drop_statistics`, `group_range`

### `src/continuum_swr_replay/analysis/replay_models/common_cohort.py`

Fit a block of models per session on one frozen complete-case cohort.

`predictor_union`, `block_required_columns`, `block_table_name`, `hash_unit_ids`, `freeze_session_cohort`, `fit_per_session_common_cohort`, `assert_ladder_timing`

### `src/continuum_swr_replay/analysis/replay_models/content/cohort.py`

The events a content ladder is fit on, and where the others went.

`prepare_canonical_fit_request`, `CohortSelection`, `select_content_cohort`, `zero_response_event_ids`, `content_cohort_record`, `ContrastFitCohort`, `contrast_fit_cohort`

### `src/continuum_swr_replay/analysis/replay_models/content/content_statistics.py`

Summary statistics of what the non-local events represent.

`trunk_shares`, `trunk_share_statistics`, `current_patch_trunk_fraction`, `event_length_statistics`, `concentration_contrast_statistics`

### `src/continuum_swr_replay/analysis/replay_models/content/estimator_comparison.py`

How three content estimators agree, fit on one shared event set per session.

`se_ratio_column`, `estimator_rows`, `attach_delta_aic`, `unidentified_rows`, `identified_rows`, `n_unpaired_triples`, `wide_by_estimator`, `summarize_estimators`, `bootstrap_animals`, `n_fits_delta_aic_row_mismatch`, `ComparisonResult`, `shared_session_rows`, `drop_undefined_concentration`, `run_comparison`

### `src/continuum_swr_replay/analysis/replay_models/content/session_fits.py`

Fit the content rungs on every session of a cohort.

`fit_model_per_session`, `fit_conditional_logit_robustness`, `resolve_concentration_cols`, `fit_contrast_session`, `fit_contrast_ladder`, `fit_n_categories`, `fit_content_ladder`, `patch_models_for_source`, `patch_switch_models_for_source`, `fit_patch_hierarchy`, `fit_patch_switch`

### `src/continuum_swr_replay/analysis/replay_models/coupling/swr_power.py`

The per-event SWR-power model: an event's ripple power against the behavioral covariates.

`post_outcome_masks`, `balanced_post_outcome_selection`, `gate_events_after_outcome`, `fit_power_model`, `power_distribution_statistics`, `duration_power_correlation`, `assemble_power_frame`, `power_cohort_records`

### `src/continuum_swr_replay/analysis/replay_models/fit_comparability.py`

Which sessions of a rung pair can be compared by an information criterion.

`why_not_comparable`, `pair_rung_fits`

### `src/continuum_swr_replay/analysis/replay_models/rate/cross_history.py`

Bivariate per-bin conditional intensity: does one non-local stream's history drive the other's rate?

`fit_direction`, `fit_cross_history`

### `src/continuum_swr_replay/analysis/replay_models/rate/history_kernel_statistics.py`

Summary statistics of the event-history kernels of the duration-aware rate fits.

`lag_ms`, `lag_index`, `measurable_kernel`, `kernel_peak`, `self_history_statistics`, `self_history_range_statistics`, `cross_history_statistics`

### `src/continuum_swr_replay/analysis/replay_models/rate/per_bin_rate_ladder.py`

The per-bin rate ladder (Family B) fit over sessions, and its cohort audit.

`ladder_design`, `fit_ladder_session`, `fit_per_bin_rate_models`, `stop_level_design_columns`, `per_bin_cohort_records`

### `src/continuum_swr_replay/analysis/replay_models/rate/post_outcome.py`

What a per-bin rate model may see once the trial's outcome is known.

`mask_post_outcome_covariates`, `retained_time_course_support`, `gate_bins_after_outcome`

### `src/continuum_swr_replay/analysis/replay_models/rate/rate_fit_statistics.py`

Summary statistics of the per-bin event-rate ladder's fits.

`frozen_cohort_statistics`, `baseline_rate_statistics`, `baseline_rate_ratio_statistics`, `reward_gated_fits`, `reward_kernel_by_lag`, `reward_kernel_animal_medians`, `self_history_kernel_by_lag`, `reward_kernel_statistics`, `paired_lag_one_reward_kernels`, `paired_lag_one_kernel_statistics`, `ladder_fit_statistics`, `model_checking_examples`, `arrival_window_statistics`, `fitted_time_course`, `clock_shape_statistics`

### `src/continuum_swr_replay/analysis/replay_models/rate/stop_statistics.py`

Summary statistics of the immobility stops and of where event onsets fall in them.

`stop_cohort_counts`, `dwell_statistics`, `still_immobile_fraction`, `onset_rate_time_course`, `outcome_boundary_statistics`, `onset_time_statistics`, `reward_history_incomplete_statistics`

### `src/continuum_swr_replay/analysis/replay_models/rate/swr_contrast_rate.py`

The per-bin ``is_swr``-contrast rate ladders, fit over sessions, and their audit.

`primary_union`, `fit_post_outcome_cohort`, `fit_contrast_session`, `fit_contrast_sessions`, `join_contrast_qvals`, `home_cohort`, `contrast_cohort_records`, `covariate_sds`, `max_relative_sd_difference`, `contrast_term_diagnostics`, `cohort_sd_reference`, `aggregate_contrast_fits`

### `src/continuum_swr_replay/analysis/replay_models/rate/uncertainty_rate.py`

The per-bin replay rate vs behavioral uncertainty and value, by population.

`drop_exposure_sessions`, `fit_uncertainty_rate`, `uncertainty_cohort_records`, `join_uncertainty_qvals`, `standardize_rl_covariates`

### `src/continuum_swr_replay/decoding/encoding_model.py`

A saved decode's fitted clusterless encoding model, read back for reuse.

`load_clusterless_encoding_model`, `clusterless_encoding_model`, `check_marks_match_encoding`, `interior_bin_distances_cm`

### `src/continuum_swr_replay/decoding/fit.py`

The decode fit: a runner's record, its detector, the fit and the files it writes.

`DecodeRunner`, `algorithm_params`, `create_detector`, `perform_decoding`

### `src/continuum_swr_replay/tables/columns.py`

The required-column check, for the code that reads the canonical tables.

_No module-level public symbols._
