# Where moved code went (2026)

**History.** The package and the scripts were reorganized twice in 2026, in October and
in November. This file records where every moved, renamed or split module and script
went, for code outside this repository (notebooks, old scripts) and for outputs whose
provenance records an old path. It is not current instruction: the layout today is in
[../README.md](../README.md) ("Code layout") and [../../scripts/README.md](../../scripts/README.md).
There are no compatibility shims: an old import path raises `ModuleNotFoundError`.
The last section lists the dated audits the indexes once cited.

An old path is marked "(removed: ...)" so that `tests/repository/test_doc_paths_resolve.py`
does not look for it; every new path is checked.

## Package modules, 2026-11

The replay and behavior packages of `analysis/` became one package,
`analysis/behavior/`, with the subpackages `covariates/`, `tables/`, `rate/`,
`content/` and `choice/`, and the publication-figure helpers moved into
`visualization/`. Every module kept its file name. There are no compatibility
shims. Paths are under `src/continuum_swr_replay/`.

| Old module | New home |
|---|---|
| `analysis/behavior/_spatial_bandit_rl.py` (removed: moved) | `analysis/behavior/covariates/_spatial_bandit_rl.py` |
| `analysis/behavior/qvals_table.py` (removed: moved) | `analysis/behavior/covariates/qvals_table.py` |
| `analysis/replay/__init__.py` (removed: moved) | `analysis/behavior/__init__.py` |
| `analysis/replay/_fit_diagnostics.py` (removed: moved) | `analysis/behavior/_fit_diagnostics.py` |
| `analysis/replay/_helpers.py` (removed: moved) | `analysis/behavior/_helpers.py` |
| `analysis/replay/_tables_common.py` (removed: moved) | `analysis/behavior/tables/_tables_common.py` |
| `analysis/replay/bouts.py` (removed: moved) | `analysis/behavior/tables/bouts.py` |
| `analysis/replay/choice_prediction.py` (removed: moved) | `analysis/behavior/choice/choice_prediction.py` |
| `analysis/replay/coefficient_statistics.py` (removed: moved) | `analysis/behavior/coefficient_statistics.py` |
| `analysis/replay/cohort_audit.py` (removed: moved) | `analysis/behavior/cohort_audit.py` |
| `analysis/replay/content_response.py` (removed: moved) | `analysis/behavior/content_response.py` |
| `analysis/replay/content_statistics.py` (removed: moved) | `analysis/behavior/content/content_statistics.py` |
| `analysis/replay/contrast_aggregation.py` (removed: moved) | `analysis/behavior/contrast_aggregation.py` |
| `analysis/replay/covariate_timing.py` (removed: moved) | `analysis/behavior/covariate_timing.py` |
| `analysis/replay/covariates/__init__.py` (removed: moved) | `analysis/behavior/covariates/__init__.py` |
| `analysis/replay/covariates/candidate_arm.py` (removed: moved) | `analysis/behavior/covariates/candidate_arm.py` |
| `analysis/replay/covariates/clocks.py` (removed: moved) | `analysis/behavior/covariates/clocks.py` |
| `analysis/replay/covariates/context.py` (removed: moved) | `analysis/behavior/covariates/context.py` |
| `analysis/replay/covariates/definitions.py` (removed: moved) | `analysis/behavior/covariates/definitions.py` |
| `analysis/replay/covariates/history/__init__.py` (removed: moved) | `analysis/behavior/covariates/history/__init__.py` |
| `analysis/replay/covariates/history/_common.py` (removed: moved) | `analysis/behavior/covariates/history/_common.py` |
| `analysis/replay/covariates/history/diagnostics.py` (removed: moved) | `analysis/behavior/covariates/history/diagnostics.py` |
| `analysis/replay/covariates/history/reward_history.py` (removed: moved) | `analysis/behavior/covariates/history/reward_history.py` |
| `analysis/replay/covariates/history/value_block.py` (removed: moved) | `analysis/behavior/covariates/history/value_block.py` |
| `analysis/replay/covariates/history/value_uncertainty.py` (removed: moved) | `analysis/behavior/covariates/history/value_uncertainty.py` |
| `analysis/replay/covariates/history/visit_recency.py` (removed: moved) | `analysis/behavior/covariates/history/visit_recency.py` |
| `analysis/replay/covariates/interactions.py` (removed: moved) | `analysis/behavior/covariates/interactions.py` |
| `analysis/replay/design_constants.py` (removed: moved) | `analysis/behavior/design_constants.py` |
| `analysis/replay/dirichlet_multinomial.py` (removed: moved) | `analysis/behavior/content/dirichlet_multinomial.py` |
| `analysis/replay/duration_aware_history.py` (removed: moved) | `analysis/behavior/rate/duration_aware_history.py` |
| `analysis/replay/duration_history_simulation.py` (removed: moved) | `analysis/behavior/rate/duration_history_simulation.py` |
| `analysis/replay/fit_outputs.py` (removed: moved) | `analysis/behavior/content/fit_outputs.py` |
| `analysis/replay/fit_provenance.py` (removed: moved) | `runs/fit_provenance.py` (by way of `analysis/behavior/`) |
| `analysis/replay/fitting.py` (removed: moved) | `analysis/behavior/fitting.py` |
| `analysis/replay/history_kernel_statistics.py` (removed: moved) | `analysis/behavior/rate/history_kernel_statistics.py` |
| `analysis/replay/learning.py` (removed: moved) | `analysis/behavior/choice/learning.py` |
| `analysis/replay/model_checking.py` (removed: moved) | `analysis/behavior/content/model_checking.py` |
| `analysis/replay/model_comparison.py` (removed: moved) | `analysis/behavior/content/model_comparison.py` |
| `analysis/replay/model_specs.py` (removed: moved) | `analysis/behavior/model_specs.py` |
| `analysis/replay/per_bin_design.py` (removed: moved) | `analysis/behavior/per_bin_design.py` |
| `analysis/replay/per_bin_rate_model.py` (removed: moved) | `analysis/behavior/rate/per_bin_rate_model.py` |
| `analysis/replay/per_stop_rate_model.py` (removed: moved) | `analysis/behavior/rate/per_stop_rate_model.py` |
| `analysis/replay/prepare_data.py` (removed: moved) | `analysis/behavior/tables/prepare_data.py` |
| `analysis/replay/rate_fit_statistics.py` (removed: moved) | `analysis/behavior/rate/rate_fit_statistics.py` |
| `analysis/replay/retrospective.py` (removed: moved) | `analysis/behavior/choice/retrospective.py` |
| `analysis/replay/ridge_history_cv.py` (removed: moved) | `analysis/behavior/rate/ridge_history_cv.py` |
| `analysis/replay/shuffles.py` (removed: moved) | `analysis/behavior/choice/shuffles.py` |
| `analysis/replay/simulation.py` (removed: moved) | `analysis/behavior/content/simulation.py` |
| `analysis/replay/spatial.py` (removed: moved) | `analysis/behavior/covariates/spatial.py` |
| `analysis/replay/spatial_mapping.py` (removed: moved) | `analysis/behavior/spatial_mapping.py` |
| `analysis/replay/stop_statistics.py` (removed: moved) | `analysis/behavior/rate/stop_statistics.py` |
| `analysis/replay/summaries.py` (removed: moved) | `analysis/behavior/tables/summaries.py` |
| `analysis/replay/switch_analysis.py` (removed: moved) | `analysis/behavior/choice/switch_analysis.py` |
| `analysis/replay/table_contracts.py` (removed: moved) | `analysis/behavior/tables/table_contracts.py` |
| `analysis/replay/tables_arm_patch.py` (removed: moved) | `analysis/behavior/tables/tables_arm_patch.py` |
| `analysis/replay/tables_event.py` (removed: moved) | `analysis/behavior/tables/tables_event.py` |
| `analysis/replay/tables_stop.py` (removed: moved) | `analysis/behavior/tables/tables_stop.py` |
| `analysis/replay/timebin_tables.py` (removed: moved) | `analysis/behavior/tables/timebin_tables.py` |
| `analysis/replay/trajectory.py` (removed: moved) | `analysis/behavior/tables/trajectory.py` |
| `analysis/replay/uncertainty.py` (removed: moved) | `analysis/behavior/tables/uncertainty.py` |
| `analysis/replay/uncertainty_content.py` (removed: moved) | `analysis/behavior/content/uncertainty_content.py` |
| `figure_utilities.py` (removed: moved) | `visualization/figure_utilities.py` |

The sweep results the detection summary reads moved beside the summaries:

| Old path | New path |
|---|---|
| `docs/design/replay_detection_comparison/detection_controls_summary.csv` (removed: moved) | `paper/reported/inputs/detection/detection_controls_summary.csv` |
| `docs/design/replay_detection_comparison/detection_heldout_summary.csv` (removed: moved) | `paper/reported/inputs/detection/detection_heldout_summary.csv` |
| `docs/design/replay_detection_comparison/detection_sweep_summary.csv` (removed: moved) | `paper/reported/inputs/detection/detection_sweep_summary.csv` |
| `docs/design/replay_detection_comparison/memoryless_comparison_recording.json` (removed: moved) | `paper/reported/inputs/detection/memoryless_comparison_recording.json` |
| `docs/design/replay_detection_comparison/reference_check_summary.csv` (removed: moved) | `paper/reported/inputs/detection/reference_check_summary.csv` |

## Run plumbing moved to runs/, 2026-11-12

What a runner writes and reads about its own run (sidecars, manifests, epoch
filters) moved out of `analysis/` into `runs/`. Paths are under
`src/continuum_swr_replay/`.

| Old home | New home |
|---|---|
| `analysis/behavior/fit_provenance.py` (removed: moved) | `runs/fit_provenance.py` |
| `analysis/pfc/runner_support.filter_epoch_info`, `read_session_list` (removed: moved) | `runs/cli.py` |
| `analysis/pfc/runner_support.matching_sidecar`, `remove_bundle` (removed: moved) | `runs/fit_provenance.py` |
| `analysis/pfc/runner_support.is_nonempty_file` (removed: moved) | `runs/io.py` |
| `analysis/pfc/runner_support.json_safe` (removed) | `runs/io.json_ready` |
| `analysis/pfc/gpu.py` (removed: moved) | `runs/device.py` |
| `reported/mpfc/inputs.table_hashes` (removed: moved) | `runs/staleness.py` |
| `analysis/behavior/contrast_aggregation.load_per_sd_scale`, `load_fit_time_sd_reference` (removed: moved) | `runs/fit_provenance.py` |
| `analysis/behavior/covariates/qvals_table.run_build_qvals_table`, `_fetch_model_df` (removed: moved) | `scripts/behavior/build_qvals_table.py`; the table steps it calls stay in `qvals_table` (`prepare_model_values`, `build_animal`, `stop_mapping_agreement`, `require_complete_mapping`, `standardize_qvals_table`) |

## Calculations moved out of scripts, 2026-11

Later in November the calculations a script did itself moved into the package; the
scripts still exist and call them. Paths are under `src/continuum_swr_replay/analysis/`
unless they start with `scripts/`.

| Where the code was | Where it is |
|---|---|
| `behavior/content/model_comparison.py`: `why_not_comparable` and its column lists | `behavior/fit_comparability.py` (new) |
| `scripts/behavior/coupling/fit_swr_power_model.py`: the per-session fit, the post-outcome gate, the cohort record and the printed statistics | `behavior/coupling/swr_power.py` (new subpackage) |
| `scripts/behavior/rate/fit_swr_contrast_rate.py` (the post-outcome covariate mask) and `fit_uncertainty_rate.py` (the bin gate after the outcome) | `behavior/rate/post_outcome.py` (new): `mask_post_outcome_covariates`, `gate_bins_after_outcome`, `retained_time_course_support` |
| `scripts/behavior/content/fit_swr_contrast_content.py`: each rung's cohort and the per-session fit loop | `behavior/content/cohort.py` (`contrast_fit_cohort`) and `behavior/content/session_fits.py` (`fit_contrast_ladder`, `fit_contrast_session`, `resolve_concentration_cols`) |
| `scripts/behavior/content/fit_replay_content_models.py`: `_fit_conditional_logit_robustness` | `behavior/content/session_fits.py` (`fit_conditional_logit_robustness`, `LOGIT_SKIP_COLUMNS`) |
| `scripts/behavior/content/compare_content_estimators.py`: the per-fit rows, the AIC difference, the identifiability rule, the summary and the animal bootstrap | `behavior/content/estimator_comparison.py` (new) |
| `scripts/detection/_report.py` and `report_inventory_comparison.py`: `pool_sessions`, `summed`, `ratio`, `session_names` and the counts and cells tables | `inventory_comparison/pooling.py` (new) |
| `scripts/detection/measure_event_start_and_rest_theta.py`: `measure_epoch`'s computation | `hippocampus/start_and_rest_theta.py` (new): `start_and_rest_theta` |
| `scripts/detection/sweep_memoryless_detection.py` and `sweep_memoryless_controls.py`: their `summarize` functions | `hippocampus/memoryless_sequence/sweep_summaries.py` (new): `replay_sweep_summary`, `controls_summary` |
| `scripts/detection/sweep_memoryless_heldout.py`: its `summarize` function | `inventory_comparison/heldout.py` (`heldout_summary`) |

## Summary builders split by subject, 2026-11-12

The two largest summary builders became packages, one module per subject. Every
function kept its name and its body, and each value's `function` is unchanged;
the package's `summarize` and `default_inputs` are what
`scripts/reported/summarize.py` calls. Paths are under `src/continuum_swr_replay/`.

| Old module | New home |
|---|---|
| `reported/behavior.py` (removed: split) | `reported/behavior/__init__.py` (`summarize`); `reported/behavior/inputs.py` (`BehaviorInputs`, `default_inputs`, `fit_staleness`, `_read_per_bin_fits`); `reported/behavior/common.py` (`TAGS`, `_gate_note`, `_require`); `reported/behavior/rate.py` (`per_bin_values`, `per_bin_detail_values`, `per_stop_values`, `rate_contrast_separation_values`, `rate_contrast_audit_values`, `cohort_audit_values`); `reported/behavior/history.py` (`duration_aware_history_values`); `reported/behavior/stops.py` (`stop_table_values`); `reported/behavior/content.py` (`content_contrast_values`, `breadth_derived_values`, `estimator_values`, `leaf_values`, `horserace_values`, `trunk_values`, `current_arm_paired_values`); `reported/behavior/choice.py` (`choice_values`, `stay_leave_values`); `reported/behavior/coupling.py` (`power_values`) |
| `reported/mpfc.py` (removed: split) | `reported/mpfc/__init__.py` (`summarize`); `reported/mpfc/inputs.py` (`MpfcInputs`, `default_inputs`, `staleness`; `table_hashes`, since moved to `runs/staleness.py`; `STEM` and `read_sidecars`, since moved to `analysis/pfc/bundle_files.py` as `MPFC_STEM` and `read_sidecars`); `reported/mpfc/common.py` (`TAGS`, `ANIMAL_WORDS`, `CLASS_LABELS`); `reported/mpfc/population.py` (`cohort_values`, `estimator_values`, `prior_binding_values`, `called_fraction_values`, `null_geometry_values`, `overlap_and_effect_values`); `reported/mpfc/timing.py` (`timing_values`); `reported/mpfc/swr_without_replay.py` (`swr_without_replay_values`; `SWR_WITHOUT_REPLAY_CURVES`, since moved to `analysis/pfc/population_statistics.py`); `reported/mpfc/followups.py` (`marked_family_values`, `stop_context_values`, `followup_values`, `marked_family_staleness`, `stop_context_staleness`); `reported/mpfc/rate_cutoff.py` (`rate_cutoff_check_values`, `CHECK_FIRING_RATE_HZ`) |

## Package modules, 2026-10

The code-organization branch (`470a393..2e55b21b`) moved, renamed or split the
modules below. **There are no compatibility shims**: the old import paths raise
`ModuleNotFoundError`, so code outside this repo (notebooks, old scripts) must update
its imports. Paths are under `src/continuum_swr_replay/`; the import path is the file
path with `/` as `.` and no `.py`. An old path is marked "(removed: …)" so the
doc-path test does not look for it. The source of truth is
`git diff -M --name-status 470a393..2e55b21b -- src/`.

| Old module | New home |
|---|---|
| `analysis/hippocampus/swr_source.py` (removed: split) | `analysis/events/analysed_time.py` (the analysed-time read point: `excluded_intervals`, `with_swr_source`, `artifact_event_drops`, `swr_event_labels`, …) and `analysis/events/swr_source.py` (the SWR inventory read: `SWR_INVENTORY_RUN_ID`, `ARTIFACT_INTERVALS_RUN_ID`, `read_swr_inventory`, `swr_events`, `SwrSourceError`, …). `ARTIFACT_INTERVALS_ROOT` is now `paths.artifact_intervals_root()`; `SWR_INVENTORY_ROOT` is `paths.swr_inventory_root()`; `SWR_SOURCE_TABLES` is in `tables/io.py` |
| `analysis/hippocampus/artifacts/io.py` (removed: moved) | `analysis/events/artifact_intervals.py` |
| `analysis/hippocampus/artifacts/masks.py` (removed: split) | `analysis/events/artifact_masks.py` (`PADDING_S`, `core_class`, `artifact_intervals`, the signature sets); the generic interval functions (`merge_intervals`, `interval_mask`, `subtract_intervals`, `masked_seconds`, `pad_intervals`, `overlaps_intervals`, `covered_seconds`) are in `intervals.py` |
| `analysis/hippocampus/event_table.py` (removed: moved) | `analysis/events/classified_table.py` |
| `analysis/hippocampus/event_detection.py` (removed: moved) | `analysis/events/nonlocal_detection.py` |
| `analysis/hippocampus/event_eligibility.py` (removed: moved) | `analysis/events/eligibility.py` |
| `analysis/hippocampus/event_immobility.py` (removed: moved) | `analysis/events/immobility.py` |
| `analysis/hippocampus/epoch_normalizers.py` (removed: moved) | `analysis/events/epoch_normalizers.py` |
| `analysis/hippocampus/memoryless_sequence/statistics.py` (removed: renamed) | `analysis/hippocampus/memoryless_sequence/window_statistics.py` |
| `analysis/hippocampus/swr_inventories.py` (removed: renamed) | `analysis/hippocampus/swr_detectors.py` (it runs the four direct SWR detectors) |
| `analysis/replay/cohort_selection.py` (removed: moved) | `analysis/events/cohort_selection.py` |
| `analysis/inventory_comparison/intervals.py` (removed: moved) | `intervals.py` |
| `analysis/pfc/glm.py` (removed: split) | `analysis/pfc/design.py`, `analysis/pfc/event_sets.py`, `analysis/pfc/fit.py`, `analysis/pfc/matched_control.py`, `analysis/pfc/modulation.py` and `analysis/pfc/session.py` (`process_neuron`, `run_session`); `get_spikecount_per_time_bin` is in `intervals.py`. Git does not detect the rename: `git log --follow -M30% -- src/continuum_swr_replay/analysis/pfc/session.py` reaches the history of `glm.py` |
| `analysis/replay/posterior_io.py` (removed: moved) | `load_conditional_posterior` is in `analysis/hippocampus/posterior_analysis.py` |
| `analysis/replay/covariate_registry.py` (removed: moved) | `tables/covariate_registry.py` |
| `analysis/replay/event_ids.py` (removed: moved) | `tables/event_ids.py` |
| `analysis/replay/standardization.py` (removed: moved) | `tables/standardization.py` |
| `analysis/replay/versions.py` (removed: moved) | `tables/versions.py` |
| `decode_files.py` (removed: moved) | `decoding/files.py` |
| `decoding_transitions.py` (removed: moved) | `decoding/transitions.py` |
| `decoding_common.py` (removed: split) | `decoding/config.py` (`DecodingConfig`, `NonLocalDecodingConfig`); `decoding/files.py` (`results_exist`); `analysis/hippocampus/decode_masks.py` (`hpc_decode_masks`, `HpcDecodeMasks` and the other HPC decode-mask functions); `runs/batch.py` (`run_batch`, `setup_logger`, `release_nwb_files`, `raise_open_file_limit`); `runs/cli.py` (`parse_run_id`, `RUN_ID_PATTERN`, `add_common_args`, `filter_epoch_info`); `tables/io.py` (`write_atomically`); `runs/io.py` (`write_json`, `replace_outputs`); `runs/provenance.py` (`provenance_record`) |
| `table_io.py` (removed: moved) | `tables/io.py`; `sha256_file` is in `runs/io.py` |
| `table_parity.py` (removed: moved) | `tables/parity.py`; `BASENAME_TO_TABLE` and `CANONICAL_TABLE_BASENAMES` are in `paths.py` |
| `session_keys.py` (removed: moved) | `tables/session_keys.py` |
| `types.py` (removed: renamed) | `data_types.py` |
| `visualization/session_tables.py` (removed: split) | the `visualization/session_tables/` package, so the import path is unchanged: `plot_session_tables`, `register_panel` and the panel lists are in its `__init__.py`; the private panel code is in `visualization/session_tables/_common.py`, `visualization/session_tables/arm_covariates.py`, `visualization/session_tables/patch_covariates.py`, `visualization/session_tables/responses.py` and `visualization/session_tables/stop_covariates.py` |

Names that left a module that still exists:

- `COLORS` and `CONTFRAG_COLORS`: from `parameters.py` to `visualization/palettes.py`.
- `kay_version_check`: from `swr_inventories.py` (now
  `analysis/hippocampus/swr_detectors.py`) to `analysis/inventory_comparison/kay_version.py`.
- `parse_nwb_filename`, `parse_epoch_identifier`, `format_epoch_identifier`, `EpochKey`,
  `DEFAULT_SWR_DEFINITION`, `DEFAULT_TRAINING_TYPE` and `OTHER_RIPPLE_DETECTORS`: from
  `data_loaders/utils.py` to `decoding/identifiers.py`, which also took `decode_identifier`
  from `decoding/files.py` and `TrainingType` from `parameters.py` (2026-10-25).
- `DETECTOR_CONFIG` and `CONTFRAG_DETECTOR_CONFIG` (class names as strings) left
  `parameters.py`; `decoding/config.py`'s `nonlocal_detector_classes()` and
  `contfrag_classifier_classes()` return the classes themselves (2026-10-25).

Renamed functions: `inventory_comparison.adapters.swr_inventory` is now
`swr_detector_inventory` (it reads any one of the four SWR detectors' inventories).

New `src/` modules that took code out of `scripts/`: `analysis/hippocampus/detection_inputs.py`
(the detection runners' shared inputs), `analysis/behavior/content/fit_outputs.py` (the content
fitter's output naming), and functions added to `analysis/behavior/choice/learning.py` (Stage 6's
role summary) and `analysis/behavior/contrast_aggregation.py` (the contrast-content cohort
assembly). Script moves are listed below ("Scripts, 2026-10").

The detection-comparison and memoryless-sweep scripts stopped importing one another
(2026-10-25); what they shared moved here: `analysis/inventory_comparison/inventory_run.py`
(the inventory run's roster, comparisons, grid, decode path and table layout, taken from
`run_inventory_comparison.py`; its private `_speed_trace` is now `speed_trace`),
`analysis/inventory_comparison/feature_run.py` (the event features run's tables, from
`run_event_features.py`, and `read_inputs`, from `report_event_features.py`),
`event_populations` and `without_baseline_rows` added to
`analysis/inventory_comparison/populations.py`, `decode_marks` added to
`analysis/hippocampus/detection_inputs.py` (from `run_carey_detection.py`),
`visualization/inventory_examples.py` (from `generate_inventory_examples.py`; its private
`_label` is now `detector_label`), `analysis/hippocampus/memoryless_sequence/sweep.py`,
`sweep_controls.py` and `comparison_simulation.py` (from the two sweeps and
`generate_memoryless_comparison.py`), `runs/device.py` (`xla_flags` and its flags, from
`run_memoryless_detection.py`, and `default_jax_platforms`), `select_analysed_epochs` added to
`runs/cli.py`, `described_commit` added to `runs/provenance.py` (the sweeps' `git_commit`), and
`REMOTE_CM` added to `analysis/constants.py` (from `analyze_event_spike_evidence.py`).

## Scripts, 2026-11

`rate/`, `content/` and `coupling/` moved under `behavior/`, and `qa/` split
into `workflows/` (runners that run other scripts) and `checks/` (audits,
parity checks and doc writers). The two choice runners then moved from the top
of `behavior/` into `behavior/choice/`, beside `analysis/behavior/choice/`; the
fit sidecars they wrote before the move record the old path in `cli`. Nothing
is left at the old paths.

| Old path | New path |
|---|---|
| `scripts/behavior/run_retrospective_alignment.py` (removed: moved) | `scripts/behavior/choice/run_retrospective_alignment.py` |
| `scripts/behavior/run_stay_leave.py` (removed: moved) | `scripts/behavior/choice/run_stay_leave.py` |
| `scripts/content/analyze_content_occupancy.py` (removed: moved) | `scripts/behavior/content/analyze_content_occupancy.py` |
| `scripts/content/analyze_uncertainty_content.py` (removed: moved) | `scripts/behavior/content/analyze_uncertainty_content.py` |
| `scripts/content/compare_content_estimators.py` (removed: moved) | `scripts/behavior/content/compare_content_estimators.py` |
| `scripts/content/compare_event_source_models.py` (removed: moved) | `scripts/behavior/content/compare_event_source_models.py` |
| `scripts/content/fit_replay_content_models.py` (removed: moved) | `scripts/behavior/content/fit_replay_content_models.py` |
| `scripts/content/fit_swr_contrast_content.py` (removed: moved) | `scripts/behavior/content/fit_swr_contrast_content.py` |
| `scripts/content/generate_replay_content_figures.py` (removed: moved) | `scripts/behavior/content/generate_replay_content_figures.py` |
| `scripts/content/quantify_swr_overlap_geometry.py` (removed: moved) | `scripts/behavior/content/quantify_swr_overlap_geometry.py` |
| `scripts/coupling/fit_swr_power_model.py` (removed: moved) | `scripts/behavior/coupling/fit_swr_power_model.py` |
| `scripts/qa/_memoryless_reference.py` (removed: moved) | `scripts/checks/_memoryless_reference.py` |
| `scripts/qa/_report.py` (removed: moved) | `scripts/checks/_report.py` |
| `scripts/qa/audit_event_patch_fraction.py` (removed: moved) | `scripts/checks/audit_event_patch_fraction.py` |
| `scripts/qa/audit_stop_table_reward_history.py` (removed: moved) | `scripts/checks/audit_stop_table_reward_history.py` |
| `scripts/qa/check_data_availability.py` (removed: moved) | `scripts/checks/check_data_availability.py` |
| `scripts/qa/check_fit_provenance.py` (removed: moved) | `scripts/checks/check_fit_provenance.py` |
| `scripts/qa/compare_memoryless_to_reference.py` (removed: moved) | `scripts/checks/compare_memoryless_to_reference.py` |
| `scripts/qa/compare_to_golden_baseline.py` (removed: moved) | `scripts/checks/compare_to_golden_baseline.py` |
| `scripts/qa/compute_hpc_raw_stats.py` (removed: moved) | `scripts/checks/compute_hpc_raw_stats.py` |
| `scripts/qa/diagnose_trial_detection.py` (removed: moved) | `scripts/checks/diagnose_trial_detection.py` |
| `scripts/qa/freeze_regression_baseline.py` (removed: moved) | `scripts/checks/freeze_regression_baseline.py` |
| `scripts/qa/mutation_check.py` (removed: moved) | `scripts/checks/mutation_check.py` |
| `scripts/qa/run_quality_metrics.py` (removed: moved) | `scripts/checks/run_quality_metrics.py` |
| `scripts/qa/run_refits.py` (removed: moved) | `scripts/workflows/run_refits.py` |
| `scripts/qa/summarize_cohort_audits.py` (removed: moved) | `scripts/checks/summarize_cohort_audits.py` |
| `scripts/qa/summarize_stage6_fdr.py` (removed: moved) | `scripts/checks/summarize_stage6_fdr.py` |
| `scripts/qa/table_parity.py` (removed: moved) | `scripts/checks/table_parity.py` |
| `scripts/qa/write_epoch_register_doc.py` (removed: moved) | `scripts/checks/write_epoch_register_doc.py` |
| `scripts/qa/write_pipeline_map.py` (removed: moved) | `scripts/checks/write_pipeline_map.py` |
| `scripts/rate/fit_cross_history_rate.py` (removed: moved) | `scripts/behavior/rate/fit_cross_history_rate.py` |
| `scripts/rate/fit_cross_history_ridge_cv.py` (removed: moved) | `scripts/behavior/rate/fit_cross_history_ridge_cv.py` |
| `scripts/rate/fit_duration_aware_history_rate.py` (removed: moved) | `scripts/behavior/rate/fit_duration_aware_history_rate.py` |
| `scripts/rate/fit_per_bin_rate_models.py` (removed: moved) | `scripts/behavior/rate/fit_per_bin_rate_models.py` |
| `scripts/rate/fit_per_stop_rate_models.py` (removed: moved) | `scripts/behavior/rate/fit_per_stop_rate_models.py` |
| `scripts/rate/fit_swr_contrast_rate.py` (removed: moved) | `scripts/behavior/rate/fit_swr_contrast_rate.py` |
| `scripts/rate/fit_uncertainty_rate.py` (removed: moved) | `scripts/behavior/rate/fit_uncertainty_rate.py` |
| `scripts/rate/plot_model_checking.py` (removed: moved) | `scripts/behavior/rate/plot_model_checking.py` |
| `scripts/rate/plot_per_bin_rate_models.py` (removed: moved) | `scripts/behavior/rate/plot_per_bin_rate_models.py` |
| `scripts/rate/plot_per_bin_rate_summary.py` (removed: moved) | `scripts/behavior/rate/plot_per_bin_rate_summary.py` |
| `scripts/rate/plot_uncertainty_rate.py` (removed: moved) | `scripts/behavior/rate/plot_uncertainty_rate.py` |

## Scripts, 2026-10

`scripts/` was regrouped by pipeline on 2026-10-20. Nothing is left at the old paths;
outputs written before the move still record the old path in their provenance.
An old path is marked "(removed: moved)" so the doc-path test does not look for it.

| Old path | New path |
|---|---|
| `scripts/decoding/run_swr_detection.py` (removed: moved) | `scripts/detection/run_swr_detection.py` |
| `scripts/decoding/run_carey_detection.py` (removed: moved) | `scripts/detection/run_carey_detection.py` |
| `scripts/decoding/run_memoryless_detection.py` (removed: moved) | `scripts/detection/run_memoryless_detection.py` |
| `scripts/decoding/run_artifact_detection.py` (removed: moved) | `scripts/detection/run_artifact_detection.py` |
| `scripts/qa/run_inventory_comparison.py` (removed: moved) | `scripts/detection/run_inventory_comparison.py` |
| `scripts/qa/report_inventory_comparison.py` (removed: moved) | `scripts/detection/report_inventory_comparison.py` |
| `scripts/qa/run_event_features.py` (removed: moved) | `scripts/detection/run_event_features.py` |
| `scripts/qa/report_event_features.py` (removed: moved) | `scripts/detection/report_event_features.py` |
| `scripts/qa/run_state_evidence.py` (removed: moved) | `scripts/detection/run_state_evidence.py` |
| `scripts/qa/sweep_memoryless_detection.py` (removed: moved) | `scripts/detection/sweep_memoryless_detection.py` |
| `scripts/qa/sweep_memoryless_controls.py` (removed: moved) | `scripts/detection/sweep_memoryless_controls.py` |
| `scripts/qa/sweep_memoryless_heldout.py` (removed: moved) | `scripts/detection/sweep_memoryless_heldout.py` |
| `scripts/qa/analyze_event_spike_evidence.py` (removed: moved) | `scripts/detection/analyze_event_spike_evidence.py` |
| `scripts/decoding/prepare_replay_content_data.py` (removed: moved) | `scripts/tables/prepare_replay_content_data.py` |
| `scripts/decoding/prepare_contfrag_replay_content_data.py` (removed: moved) | `scripts/tables/prepare_contfrag_replay_content_data.py` |
| `scripts/decoding/regenerate_stop_table.py` (removed: moved) | `scripts/tables/regenerate_stop_table.py` |
| `scripts/rate/fit_swr_power_model.py` (removed: moved) | `scripts/behavior/coupling/fit_swr_power_model.py` |
| `scripts/decoding/build_timebin_tables.py` (removed: moved) | `scripts/tables/build_timebin_tables.py` |
| `scripts/decoding/assemble_replay_content_shards.py` (removed: moved) | `scripts/tables/assemble_replay_content_shards.py` |
| `scripts/decoding/assemble_contfrag_shards.py` (removed: moved) | `scripts/tables/assemble_contfrag_shards.py` |
| `scripts/decoding/convert_tables_to_parquet.py` (removed: moved) | `scripts/tables/convert_tables_to_parquet.py` |
| `scripts/decoding/write_regeneration_manifest.py` (removed: moved) | `scripts/tables/write_regeneration_manifest.py` |
| `scripts/decoding/refresh_contfrag_behavioral_covariates.py` (removed: moved) | `scripts/tables/refresh_contfrag_behavioral_covariates.py` |
| `scripts/qa/compare_pfc_cohorts.py` (removed: moved) | `scripts/pfc/compare_pfc_cohorts.py` |
| `scripts/qa/compare_pfc_fitters.py` (removed: moved) | `scripts/pfc/compare_pfc_fitters.py` |
| `scripts/qa/check_pfc_ridge_lambda0.py` (removed: moved) | `scripts/pfc/check_pfc_ridge_lambda0.py` |
| `scripts/qa/measure_pfc_null_geometry.py` (removed: moved) | `scripts/pfc/measure_pfc_null_geometry.py` |
| `scripts/qa/diagnose_pfc_event_definition.py` (removed: moved) | `scripts/pfc/diagnose_pfc_event_definition.py` |
| `scripts/qa/memoryless_reference.py` (removed: moved) | `scripts/checks/_memoryless_reference.py` (2026-10-25: a module, not a script) |

## Runners renamed, 2026-10-17

The two behaviour analyses left `scripts/pfc/` and took names for what they do. Their
output files keep the stage names.

| Old path | New path |
|---|---|
| `scripts/pfc/run_stage5b_retrospective.py` (removed: renamed) | `scripts/behavior/choice/run_retrospective_alignment.py` ("Stage 5B"; at the top of `scripts/behavior/` until 2026-11) |
| `scripts/pfc/run_stage6_learning.py` (removed: renamed) | `scripts/behavior/choice/run_stay_leave.py` ("Stage 6"; at the top of `scripts/behavior/` until 2026-11) |

## Data files moved, 2026-10-23

`qvals_table` and its standardization reference moved from `models_per_bin_rate/` to the
root of `replay_content/`, and the per-event ripple-power model to its own folder,
`models_per_event_swr_power/`. Nothing is left at the old `models_per_bin_rate/` paths
(the "Phase D disk move" entry of
[the code-organization plan](../../.claude/docs/plans/code-organization/PLAN.md)).

## Dated audits the docs once cited

These point-in-time audits are in `.claude/docs/reviews/`. The indexes and
`docs/rate_models.md` cited them until 2026-11-12; they no longer do, because the rule
each one checked is in a design doc and the current numbers are in `paper/reported/`.
A design doc may still link one as its dated source. Their numbers describe the data
of their date.

- [2026-09-06-rate-refit-arrival-clock.md](../../.claude/docs/reviews/2026-09-06-rate-refit-arrival-clock.md)
  — the rate refit that added the arrival clock; why the retired per-lag reward step
  was not a like-for-like comparison.
- [2026-09-12-rate-cohort-atwell.md](../../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md)
  — the rate ladders on the three stop cohorts (218 sessions), when the well-anchored
  cohort became the canonical one ([../design/cohort_and_timing_rules.md](../design/cohort_and_timing_rules.md)).
- [2026-09-13-stage5b-stage6-atwell.md](../../.claude/docs/reviews/2026-09-13-stage5b-stage6-atwell.md)
  — the retrospective alignment and the stay/leave analysis moved onto that cohort.
- [2026-09-14-pfc-atwell-cohort.md](../../.claude/docs/reviews/2026-09-14-pfc-atwell-cohort.md)
  and [2026-09-15-pfc-canonical-rerun.md](../../.claude/docs/reviews/2026-09-15-pfc-canonical-rerun.md)
  — the mPFC unit GLM moved onto that cohort, then rerun on the canonical event table.
- [2026-09-19-pfc-marked-family-summary.md](../../.claude/docs/reviews/2026-09-19-pfc-marked-family-summary.md)
  — the population summary of the 2026-09-19 marked-family corpus
  ([../design/pfc_marked_model_family.md](../design/pfc_marked_model_family.md), "Artifacts").
