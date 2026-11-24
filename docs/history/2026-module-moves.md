# Where moved code went (2026)

**History.** The package and the scripts were reorganized twice in 2026, in October and
in November. This file records where every moved, renamed or split module and script
went, for code outside this repository (notebooks, old scripts) and for outputs whose
provenance records an old path. It is not current instruction: the layout today is in
[../README.md](../README.md) ("Code layout") and [../../scripts/README.md](../../scripts/README.md).
There are no compatibility shims: an old import path raises `ModuleNotFoundError`.
The section "Dated audits the docs once cited" lists the dated audits the indexes once cited.

An old path is marked "(removed: ...)". `tests/repository/test_doc_paths_resolve.py`
skips this file (and the dated design specs, `docs/design/20*-*.md`), so neither the
old nor the new paths here are checked; they were correct when each section was
written.

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
| `analysis/behavior/tables/prepare_data.run_prepare_data`, `populate_behavioral_context_cache`, `build_stop_table_scope` and the session loop (removed: moved) | `scripts/tables/prepare_replay_content_data.py`; the per-session build and the corpus steps stay in `prepare_data` (`_process_session` renamed `build_session_tables`; `assemble_canonical_tables`, `standardize_event_arm`) |
| `analysis/behavior/tables/prepare_data.assemble_replay_content_shards`, `assemble_replay_stop_shards` (removed: moved) | `scripts/tables/assemble_replay_content_shards.py` |
| `analysis/behavior/tables/prepare_data.find_sessions`, `read_disjoint_shards`, `_publish_canonical_tables`; `analysis/contfrag/prepare_data.find_sessions`, `publish_contfrag_tables` (removed: moved) | `scripts/tables/_table_runner.py` (`find_sessions`, `read_disjoint_shards`, `publish_replay_tables`, `find_contfrag_sessions`, `publish_contfrag_tables`) |
| `analysis/contfrag/prepare_data.build_contfrag_tables`, `assemble_contfrag_shards`, `refresh_contfrag_behavioral_covariates` (removed: moved) | `scripts/tables/prepare_contfrag_replay_content_data.py`, `assemble_contfrag_shards.py`, `refresh_contfrag_behavioral_covariates.py`; the per-session builds stay in `contfrag/prepare_data` (`_process_session` renamed `build_contfrag_session_tables`, `_rebuild_session` renamed `rebuild_contfrag_session_tables`; the event-patch projection the three drivers repeated is `contfrag_event_patch_table`) |

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
| `analysis/hippocampus/swr_source.py` (removed: split) | `analysis/events/analysed_time.py` (the analyzed-time read point: `excluded_intervals`, `with_swr_source`, `artifact_event_drops`, `swr_event_labels`, …) and `analysis/events/swr_source.py` (the SWR inventory read: `SWR_INVENTORY_RUN_ID`, `ARTIFACT_INTERVALS_RUN_ID`, `read_swr_inventory`, `swr_events`, `SwrSourceError`, …). `ARTIFACT_INTERVALS_ROOT` is now `paths.artifact_intervals_root()`; `SWR_INVENTORY_ROOT` is `paths.swr_inventory_root()`; `SWR_SOURCE_TABLES` is in `tables/io.py` |
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

The two behavior analyses left `scripts/pfc/` and took names for what they do. Their
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

## Package modules, 2026-11-13

`analysis/behavior/` was split in three, in data-flow order: `analysis/behavior/` keeps
what the animal did (`covariates/`, `design_constants.py`, `spatial_mapping.py`, and the
new `history_basis.py`); `analysis/event_tables/` takes the canonical tables (the old
`tables/` subpackage and `content_response.py`); `analysis/replay_models/` takes the
Part 3 models (`rate/`, `content/`, `coupling/`, `choice/`) and the fitting machinery
they share. The stuttering table modules were renamed and `choice/learning.py` became
`choice/stay_leave.py`. The `analysis.behavior` package no longer re-exports names
(its PEP 562 table is gone): import each name from the module that defines it. There
are no compatibility shims. Paths are under `src/continuum_swr_replay/`.

| Old module | New home |
|---|---|
| `analysis/behavior/_fit_diagnostics.py` (removed: moved) | `analysis/replay_models/_fit_diagnostics.py` |
| `analysis/behavior/choice/__init__.py` (removed: moved) | `analysis/replay_models/choice/__init__.py` |
| `analysis/behavior/choice/choice_prediction.py` (removed: moved) | `analysis/replay_models/choice/choice_prediction.py` |
| `analysis/behavior/choice/learning.py` (removed: moved) | `analysis/replay_models/choice/stay_leave.py` |
| `analysis/behavior/choice/retrospective.py` (removed: moved) | `analysis/replay_models/choice/retrospective.py` |
| `analysis/behavior/choice/shuffles.py` (removed: moved) | `analysis/replay_models/choice/shuffles.py` |
| `analysis/behavior/choice/switch_analysis.py` (removed: moved) | `analysis/replay_models/choice/switch_analysis.py` |
| `analysis/behavior/coefficient_statistics.py` (removed: moved) | `analysis/replay_models/coefficient_statistics.py` |
| `analysis/behavior/cohort_audit.py` (removed: moved) | `analysis/replay_models/cohort_audit.py` |
| `analysis/behavior/common_cohort.py` (removed: moved) | `analysis/replay_models/common_cohort.py` |
| `analysis/behavior/content/__init__.py` (removed: moved) | `analysis/replay_models/content/__init__.py` |
| `analysis/behavior/content/cohort.py` (removed: moved) | `analysis/replay_models/content/cohort.py` |
| `analysis/behavior/content/content_statistics.py` (removed: moved) | `analysis/replay_models/content/content_statistics.py` |
| `analysis/behavior/content/dirichlet_multinomial.py` (removed: moved) | `analysis/replay_models/content/dirichlet_multinomial.py` |
| `analysis/behavior/content/estimator_comparison.py` (removed: moved) | `analysis/replay_models/content/estimator_comparison.py` |
| `analysis/behavior/content/fit_outputs.py` (removed: moved) | `analysis/replay_models/content/fit_outputs.py` |
| `analysis/behavior/content/model_checking.py` (removed: moved) | `analysis/replay_models/content/model_checking.py` |
| `analysis/behavior/content/model_comparison.py` (removed: moved) | `analysis/replay_models/content/model_comparison.py` |
| `analysis/behavior/content/session_fits.py` (removed: moved) | `analysis/replay_models/content/session_fits.py` |
| `analysis/behavior/content/simulation.py` (removed: moved) | `analysis/replay_models/content/simulation.py` |
| `analysis/behavior/content/uncertainty_content.py` (removed: moved) | `analysis/replay_models/content/uncertainty_content.py` |
| `analysis/behavior/content_response.py` (removed: moved) | `analysis/event_tables/content_response.py` |
| `analysis/behavior/contrast_aggregation.py` (removed: moved) | `analysis/replay_models/contrast_aggregation.py` |
| `analysis/behavior/coupling/__init__.py` (removed: moved) | `analysis/replay_models/coupling/__init__.py` |
| `analysis/behavior/coupling/swr_power.py` (removed: moved) | `analysis/replay_models/coupling/swr_power.py` |
| `analysis/behavior/covariate_timing.py` (removed: moved) | `analysis/replay_models/covariate_timing.py` |
| `analysis/behavior/fit_comparability.py` (removed: moved) | `analysis/replay_models/fit_comparability.py` |
| `analysis/behavior/fitting.py` (removed: moved) | `analysis/replay_models/fitting.py` |
| `analysis/behavior/model_specs.py` (removed: moved) | `analysis/replay_models/model_specs.py` |
| `analysis/behavior/per_bin_design.py` (removed: moved) | `analysis/replay_models/per_bin_design.py` |
| `analysis/behavior/rate/__init__.py` (removed: moved) | `analysis/replay_models/rate/__init__.py` |
| `analysis/behavior/rate/cross_history.py` (removed: moved) | `analysis/replay_models/rate/cross_history.py` |
| `analysis/behavior/rate/duration_aware_history.py` (removed: moved) | `analysis/replay_models/rate/duration_aware_history.py` |
| `analysis/behavior/rate/duration_history_simulation.py` (removed: moved) | `analysis/replay_models/rate/duration_history_simulation.py` |
| `analysis/behavior/rate/history_kernel_statistics.py` (removed: moved) | `analysis/replay_models/rate/history_kernel_statistics.py` |
| `analysis/behavior/rate/per_bin_rate_ladder.py` (removed: moved) | `analysis/replay_models/rate/per_bin_rate_ladder.py` |
| `analysis/behavior/rate/per_bin_rate_model.py` (removed: moved) | `analysis/replay_models/rate/per_bin_rate_model.py` |
| `analysis/behavior/rate/per_stop_rate_model.py` (removed: moved) | `analysis/replay_models/rate/per_stop_rate_model.py` |
| `analysis/behavior/rate/post_outcome.py` (removed: moved) | `analysis/replay_models/rate/post_outcome.py` |
| `analysis/behavior/rate/rate_fit_statistics.py` (removed: moved) | `analysis/replay_models/rate/rate_fit_statistics.py` |
| `analysis/behavior/rate/ridge_history_cv.py` (removed: moved) | `analysis/replay_models/rate/ridge_history_cv.py` |
| `analysis/behavior/rate/stop_statistics.py` (removed: moved) | `analysis/replay_models/rate/stop_statistics.py` |
| `analysis/behavior/rate/swr_contrast_rate.py` (removed: moved) | `analysis/replay_models/rate/swr_contrast_rate.py` |
| `analysis/behavior/rate/uncertainty_rate.py` (removed: moved) | `analysis/replay_models/rate/uncertainty_rate.py` |
| `analysis/behavior/tables/__init__.py` (removed: moved) | `analysis/event_tables/__init__.py` |
| `analysis/behavior/tables/_tables_common.py` (removed: moved) | `analysis/event_tables/_common.py` |
| `analysis/behavior/tables/bouts.py` (removed: moved) | `analysis/event_tables/bouts.py` |
| `analysis/behavior/tables/prepare_data.py` (removed: moved) | `analysis/event_tables/prepare_data.py` |
| `analysis/behavior/tables/summaries.py` (removed: moved) | `analysis/event_tables/summaries.py` |
| `analysis/behavior/tables/table_contracts.py` (removed: moved) | `analysis/event_tables/table_contracts.py` |
| `analysis/behavior/tables/tables_arm_patch.py` (removed: moved) | `analysis/event_tables/arm_patch_tables.py` |
| `analysis/behavior/tables/tables_event.py` (removed: moved) | `analysis/event_tables/event_table.py` |
| `analysis/behavior/tables/tables_stop.py` (removed: moved) | `analysis/event_tables/stop_table.py` |
| `analysis/behavior/tables/timebin_tables.py` (removed: moved) | `analysis/event_tables/timebin_tables.py` |
| `analysis/behavior/tables/trajectory.py` (removed: moved) | `analysis/event_tables/trajectory.py` |
| `analysis/behavior/tables/uncertainty.py` (removed: moved) | `analysis/event_tables/uncertainty.py` |

`analysis/behavior/_helpers.py` (removed: dissolved) was split by subject; each
function went to the lowest package that uses it:

| Function | New home |
|---|---|
| `single_arm_from_well`, `arms_of_wells`, `arm_to_patch`, `build_well_to_arm_map`, `stay_indicator_from_patches`, `stay_indicator` | `analysis/behavior/spatial_mapping.py` |
| `align_posterior_to_dot`, `safe_mode`, `first_row_floats`, `strict_mean`, `strict_sum`, `align_count_matrix`, `assert_event_label_alignment`, `assert_single_session_alignment` | `analysis/event_tables/_common.py` |
| `eq_indicator`, `eq_indicator_or_nan` | `analysis/behavior/covariates/candidate_arm.py` |
| `add_arm_fixed_effects` | `analysis/replay_models/model_specs.py` |

Two definitions moved down so that no package imports one after it:

- the per-event retrospective metrics (`compute_retrospective_metrics`, the five
  metrics and their `prev_arm` mirrors, `arm_path_well_to_well`), which the table
  build merges into the event table, from `analysis/replay_models/choice/retrospective.py`
  to `analysis/event_tables/retrospective_metrics.py`; the shuffle null stays;
- `raised_cosine_history_basis` and `reward_basis_from_lags`, which the reward-history
  covariates read, from `analysis/replay_models/per_bin_design.py` to
  `analysis/behavior/history_basis.py`.

## Scripts, 2026-11-13

The Part 3 runners moved from `scripts/behavior/` to `scripts/replay_models/`, beside
`analysis/replay_models/`; `scripts/behavior/` keeps `build_qvals_table.py` and
`fetch_spatial_bandit_behavior.sh`. The run names of `scripts/workflows/run_refits.py`
(`rate.per_bin.at_well`, `behavior.retrospective.all`, ...) did not change. Fit sidecars
written before the move record the old script path in `cli`.

| Old path | New path |
|---|---|
| `scripts/behavior/choice/run_retrospective_alignment.py` (removed: moved) | `scripts/replay_models/choice/run_retrospective_alignment.py` |
| `scripts/behavior/choice/run_stay_leave.py` (removed: moved) | `scripts/replay_models/choice/run_stay_leave.py` |
| `scripts/behavior/content/analyze_content_occupancy.py` (removed: moved) | `scripts/replay_models/content/analyze_content_occupancy.py` |
| `scripts/behavior/content/analyze_uncertainty_content.py` (removed: moved) | `scripts/replay_models/content/analyze_uncertainty_content.py` |
| `scripts/behavior/content/compare_content_estimators.py` (removed: moved) | `scripts/replay_models/content/compare_content_estimators.py` |
| `scripts/behavior/content/compare_event_source_models.py` (removed: moved) | `scripts/replay_models/content/compare_event_source_models.py` |
| `scripts/behavior/content/fit_replay_content_models.py` (removed: moved) | `scripts/replay_models/content/fit_replay_content_models.py` |
| `scripts/behavior/content/fit_swr_contrast_content.py` (removed: moved) | `scripts/replay_models/content/fit_swr_contrast_content.py` |
| `scripts/behavior/content/generate_replay_content_figures.py` (removed: moved) | `scripts/replay_models/content/generate_replay_content_figures.py` |
| `scripts/behavior/content/quantify_swr_overlap_geometry.py` (removed: moved) | `scripts/replay_models/content/quantify_swr_overlap_geometry.py` |
| `scripts/behavior/coupling/fit_swr_power_model.py` (removed: moved) | `scripts/replay_models/coupling/fit_swr_power_model.py` |
| `scripts/behavior/rate/fit_cross_history_rate.py` (removed: moved) | `scripts/replay_models/rate/fit_cross_history_rate.py` |
| `scripts/behavior/rate/fit_cross_history_ridge_cv.py` (removed: moved) | `scripts/replay_models/rate/fit_cross_history_ridge_cv.py` |
| `scripts/behavior/rate/fit_duration_aware_history_rate.py` (removed: moved) | `scripts/replay_models/rate/fit_duration_aware_history_rate.py` |
| `scripts/behavior/rate/fit_per_bin_rate_models.py` (removed: moved) | `scripts/replay_models/rate/fit_per_bin_rate_models.py` |
| `scripts/behavior/rate/fit_per_stop_rate_models.py` (removed: moved) | `scripts/replay_models/rate/fit_per_stop_rate_models.py` |
| `scripts/behavior/rate/fit_swr_contrast_rate.py` (removed: moved) | `scripts/replay_models/rate/fit_swr_contrast_rate.py` |
| `scripts/behavior/rate/fit_uncertainty_rate.py` (removed: moved) | `scripts/replay_models/rate/fit_uncertainty_rate.py` |
| `scripts/behavior/rate/plot_model_checking.py` (removed: moved) | `scripts/replay_models/rate/plot_model_checking.py` |
| `scripts/behavior/rate/plot_per_bin_rate_models.py` (removed: moved) | `scripts/replay_models/rate/plot_per_bin_rate_models.py` |
| `scripts/behavior/rate/plot_per_bin_rate_summary.py` (removed: moved) | `scripts/replay_models/rate/plot_per_bin_rate_summary.py` |
| `scripts/behavior/rate/plot_uncertainty_rate.py` (removed: moved) | `scripts/replay_models/rate/plot_uncertainty_rate.py` |

## Package modules, 2026-11-13 (second pass)

`analysis/hippocampus/` was dissolved into four packages, one per question, and two
packages were renamed: `analysis/events/` became `analysis/event_definition/` and
`analysis/inventory_comparison/` became `analysis/detector_comparison/`. The detectors
went to `detection/` (`data_aggregation.py`, which applies the paper's non-local event
rule to a decode, became `detection/nonlocal_events.py`), the decoder checks to
`decode_quality/`, the spike evidence to `event_evidence/` and the event statistics to
`event_properties/`; `decode_masks.py` joined `event_definition/`, and
`events/nonlocal_detection.py` became `event_definition/nonlocal_event_rule.py`.
`hippocampus/result_loading.py` was merged into the foundation `decoding/results.py`.
The `analysis.hippocampus` package no longer re-exports names (its PEP 562 table is
gone): import each name from the module that defines it. There are no compatibility
shims. Paths are under `src/continuum_swr_replay/`.

A pickle that holds an instance of a moved class names the class's old module. One
kind of stored file does: the memoryless detector's `*_memoryless_candidates.pkl`
(`CandidateSequences` and its config). Its reader,
`analysis/detection/memoryless_sequence/detect.py:load_candidates`, pointed the two
former module names at today's modules for the read until the memoryless rerun of
2026-11-14 (runs `e2331f976fd1-seed20260922` and its `-decoderencoding` control, written
under today's path), which the comparison and features runs now read; the shim was then
removed. The runs written before 2026-11-13 (`d0964cc76ed2-seed20260922` and older) keep
their candidate files under the former path, and nothing supported reads those files.

`nonlocal_probability` moved from `decoding/results.py` to
`analysis/event_definition/nonlocal_event_rule.py`, beside the new
`nonlocal_state_probability` (the same sum on the decoder's output), which the
non-local detector (`analysis/detection/nonlocal_events.py`) and five other callers
now use in place of an inline sum.

| Old module | New home |
|---|---|
| `analysis/events/__init__.py` (removed: moved) | `analysis/event_definition/__init__.py` |
| `analysis/events/analysed_time.py` (removed: moved) | `analysis/event_definition/analysed_time.py` |
| `analysis/events/artifact_intervals.py` (removed: moved) | `analysis/event_definition/artifact_intervals.py` |
| `analysis/events/artifact_masks.py` (removed: moved) | `analysis/event_definition/artifact_masks.py` |
| `analysis/events/classified_table.py` (removed: moved) | `analysis/event_definition/classified_table.py` |
| `analysis/events/cohort_selection.py` (removed: moved) | `analysis/event_definition/cohort_selection.py` |
| `analysis/events/eligibility.py` (removed: moved) | `analysis/event_definition/eligibility.py` |
| `analysis/events/epoch_normalizers.py` (removed: moved) | `analysis/event_definition/epoch_normalizers.py` |
| `analysis/events/gallery_sample.py` (removed: moved) | `analysis/event_definition/gallery_sample.py` |
| `analysis/events/immobility.py` (removed: moved) | `analysis/event_definition/immobility.py` |
| `analysis/events/nonlocal_detection.py` (removed: moved) | `analysis/event_definition/nonlocal_event_rule.py` |
| `analysis/events/swr_source.py` (removed: moved) | `analysis/event_definition/swr_source.py` |
| `analysis/hippocampus/__init__.py` (removed) | removed: its re-exports are gone; import from the defining module |
| `analysis/hippocampus/artifacts/__init__.py` (removed: moved) | `analysis/detection/artifacts/__init__.py` |
| `analysis/hippocampus/artifacts/detect.py` (removed: moved) | `analysis/detection/artifacts/detect.py` |
| `analysis/hippocampus/artifacts/features.py` (removed: moved) | `analysis/detection/artifacts/features.py` |
| `analysis/hippocampus/brain_state_reference.py` (removed: moved) | `analysis/event_properties/brain_state_reference.py` |
| `analysis/hippocampus/class_or_continuum.py` (removed: moved) | `analysis/event_properties/class_or_continuum.py` |
| `analysis/hippocampus/content_occupancy.py` (removed: moved) | `analysis/event_properties/content_occupancy.py` |
| `analysis/hippocampus/data_aggregation.py` (removed: moved) | `analysis/detection/nonlocal_events.py` |
| `analysis/hippocampus/decode_masks.py` (removed: moved) | `analysis/event_definition/decode_masks.py` |
| `analysis/hippocampus/detection_inputs.py` (removed: moved) | `analysis/detection/detection_inputs.py` |
| `analysis/hippocampus/event_analysis.py` (removed: moved) | `analysis/event_properties/event_analysis.py` |
| `analysis/hippocampus/event_statistics.py` (removed: moved) | `analysis/event_properties/event_statistics.py` |
| `analysis/hippocampus/gamma_power.py` (removed: moved) | `analysis/event_properties/gamma_power.py` |
| `analysis/hippocampus/hpc_raw_stats.py` (removed: moved) | `analysis/decode_quality/hpc_raw_stats.py` |
| `analysis/hippocampus/memoryless_sequence/__init__.py` (removed: moved) | `analysis/detection/memoryless_sequence/__init__.py` |
| `analysis/hippocampus/memoryless_sequence/comparison_simulation.py` (removed: moved) | `analysis/detection/memoryless_sequence/comparison_simulation.py` |
| `analysis/hippocampus/memoryless_sequence/config.py` (removed: moved) | `analysis/detection/memoryless_sequence/config.py` |
| `analysis/hippocampus/memoryless_sequence/detect.py` (removed: moved) | `analysis/detection/memoryless_sequence/detect.py` |
| `analysis/hippocampus/memoryless_sequence/epoch.py` (removed: moved) | `analysis/detection/memoryless_sequence/epoch.py` |
| `analysis/hippocampus/memoryless_sequence/event_table.py` (removed: moved) | `analysis/detection/memoryless_sequence/event_table.py` |
| `analysis/hippocampus/memoryless_sequence/likelihood.py` (removed: moved) | `analysis/detection/memoryless_sequence/likelihood.py` |
| `analysis/hippocampus/memoryless_sequence/recording_statistics.py` (removed: moved) | `analysis/detection/memoryless_sequence/recording_statistics.py` |
| `analysis/hippocampus/memoryless_sequence/scoring.py` (removed: moved) | `analysis/detection/memoryless_sequence/scoring.py` |
| `analysis/hippocampus/memoryless_sequence/shuffle.py` (removed: moved) | `analysis/detection/memoryless_sequence/shuffle.py` |
| `analysis/hippocampus/memoryless_sequence/simulation.py` (removed: moved) | `analysis/detection/memoryless_sequence/simulation.py` |
| `analysis/hippocampus/memoryless_sequence/sweep.py` (removed: moved) | `analysis/detection/memoryless_sequence/sweep.py` |
| `analysis/hippocampus/memoryless_sequence/sweep_controls.py` (removed: moved) | `analysis/detection/memoryless_sequence/sweep_controls.py` |
| `analysis/hippocampus/memoryless_sequence/sweep_replays.py` (removed: moved) | `analysis/detection/memoryless_sequence/sweep_replays.py` |
| `analysis/hippocampus/memoryless_sequence/sweep_statistics.py` (removed: moved) | `analysis/detection/memoryless_sequence/sweep_statistics.py` |
| `analysis/hippocampus/memoryless_sequence/sweep_summaries.py` (removed: moved) | `analysis/detection/memoryless_sequence/sweep_summaries.py` |
| `analysis/hippocampus/memoryless_sequence/window_statistics.py` (removed: moved) | `analysis/detection/memoryless_sequence/window_statistics.py` |
| `analysis/hippocampus/posterior_analysis.py` (removed: moved) | `analysis/decode_quality/posterior_analysis.py` |
| `analysis/hippocampus/property_contrast.py` (removed: moved) | `analysis/event_properties/property_contrast.py` |
| `analysis/hippocampus/quality_metrics.py` (removed: moved) | `analysis/decode_quality/quality_metrics.py` |
| `analysis/hippocampus/result_loading.py` (removed: moved) | `decoding/results.py` (`load_results`, `_load_saved_decode`) |
| `analysis/hippocampus/ripple_tetrodes.py` (removed: moved) | `analysis/detection/ripple_tetrodes.py` |
| `analysis/hippocampus/spike_evidence.py` (removed: moved) | `analysis/event_evidence/spike_evidence.py` |
| `analysis/hippocampus/start_and_rest_theta.py` (removed: moved) | `analysis/event_properties/start_and_rest_theta.py` |
| `analysis/hippocampus/state_evidence/__init__.py` (removed: moved) | `analysis/event_evidence/state_evidence/__init__.py` |
| `analysis/hippocampus/state_evidence/evidence.py` (removed: moved) | `analysis/event_evidence/state_evidence/evidence.py` |
| `analysis/hippocampus/state_evidence/io.py` (removed: moved) | `analysis/event_evidence/state_evidence/io.py` |
| `analysis/hippocampus/state_evidence/summary.py` (removed: moved) | `analysis/event_evidence/state_evidence/summary.py` |
| `analysis/hippocampus/state_evidence/terms.py` (removed: moved) | `analysis/event_evidence/state_evidence/terms.py` |
| `analysis/hippocampus/state_evidence/windows.py` (removed: moved) | `analysis/event_evidence/state_evidence/windows.py` |
| `analysis/hippocampus/swr_detectors.py` (removed: moved) | `analysis/detection/swr_detectors.py` |
| `analysis/hippocampus/triggered_analysis.py` (removed: moved) | `analysis/event_properties/triggered_analysis.py` |
| `analysis/inventory_comparison/__init__.py` (removed: moved) | `analysis/detector_comparison/__init__.py` |
| `analysis/inventory_comparison/adapters.py` (removed: moved) | `analysis/detector_comparison/adapters.py` |
| `analysis/inventory_comparison/availability.py` (removed: moved) | `analysis/detector_comparison/availability.py` |
| `analysis/inventory_comparison/cells.py` (removed: moved) | `analysis/detector_comparison/cells.py` |
| `analysis/inventory_comparison/components.py` (removed: moved) | `analysis/detector_comparison/components.py` |
| `analysis/inventory_comparison/epoch_comparison.py` (removed: moved) | `analysis/detector_comparison/epoch_comparison.py` |
| `analysis/inventory_comparison/evidence.py` (removed: moved) | `analysis/detector_comparison/evidence.py` |
| `analysis/inventory_comparison/example_windows.py` (removed: moved) | `analysis/detector_comparison/example_windows.py` |
| `analysis/inventory_comparison/feature_run.py` (removed: moved) | `analysis/detector_comparison/feature_run.py` |
| `analysis/inventory_comparison/features.py` (removed: moved) | `analysis/detector_comparison/features.py` |
| `analysis/inventory_comparison/heldout.py` (removed: moved) | `analysis/detector_comparison/heldout.py` |
| `analysis/inventory_comparison/heldout_sweep.py` (removed: moved) | `analysis/detector_comparison/heldout_sweep.py` |
| `analysis/inventory_comparison/inventory_run.py` (removed: moved) | `analysis/detector_comparison/inventory_run.py` |
| `analysis/inventory_comparison/kay_version.py` (removed: moved) | `analysis/detector_comparison/kay_version.py` |
| `analysis/inventory_comparison/nonlocal_comparison.py` (removed: moved) | `analysis/detector_comparison/nonlocal_comparison.py` |
| `analysis/inventory_comparison/overlap.py` (removed: moved) | `analysis/detector_comparison/overlap.py` |
| `analysis/inventory_comparison/pooling.py` (removed: moved) | `analysis/detector_comparison/pooling.py` |
| `analysis/inventory_comparison/populations.py` (removed: moved) | `analysis/detector_comparison/populations.py` |
| `analysis/inventory_comparison/recount.py` (removed: moved) | `analysis/detector_comparison/recount.py` |
| `analysis/inventory_comparison/schema.py` (removed: moved) | `analysis/detector_comparison/schema.py` |
| `analysis/inventory_comparison/selection.py` (removed: moved) | `analysis/detector_comparison/selection.py` |
| `analysis/inventory_comparison/swr_matrix.py` (removed: moved) | `analysis/detector_comparison/swr_matrix.py` |

## Scripts, 2026-11-13 (second pass)

`scripts/detection/` was split by the package each script drives. It keeps the
detector runners (`run_artifact_detection.py`, `run_swr_detection.py`,
`run_carey_detection.py`, `run_memoryless_detection.py`) and the memoryless sweeps
(`sweep_memoryless_detection.py`, `sweep_memoryless_controls.py`); the rest went to
`scripts/detector_comparison/`, `scripts/event_evidence/` and
`scripts/event_properties/`. Outputs written before the move record the old script path
in `produced_by`.

| Old path | New path |
|---|---|
| `scripts/detection/_report.py` (removed: moved) | `scripts/detector_comparison/_report.py` |
| `scripts/detection/add_state_shares.py` (removed: moved) | `scripts/detector_comparison/add_state_shares.py` |
| `scripts/detection/analyze_event_spike_evidence.py` (removed: moved) | `scripts/event_evidence/analyze_event_spike_evidence.py` |
| `scripts/detection/bootstrap_event_properties.py` (removed: moved) | `scripts/event_properties/bootstrap_event_properties.py` |
| `scripts/detection/measure_brain_state_reference.py` (removed: moved) | `scripts/event_properties/measure_brain_state_reference.py` |
| `scripts/detection/measure_event_gamma.py` (removed: moved) | `scripts/event_properties/measure_event_gamma.py` |
| `scripts/detection/measure_event_start_and_rest_theta.py` (removed: moved) | `scripts/event_properties/measure_event_start_and_rest_theta.py` |
| `scripts/detection/measure_ripple_class_structure.py` (removed: moved) | `scripts/event_properties/measure_ripple_class_structure.py` |
| `scripts/detection/report_event_features.py` (removed: moved) | `scripts/detector_comparison/report_event_features.py` |
| `scripts/detection/report_inventory_comparison.py` (removed: moved) | `scripts/detector_comparison/report_inventory_comparison.py` |
| `scripts/detection/run_event_features.py` (removed: moved) | `scripts/detector_comparison/run_event_features.py` |
| `scripts/detection/run_event_spike_evidence.sh` (removed: moved) | `scripts/event_evidence/run_event_spike_evidence.sh` |
| `scripts/detection/run_inventory_comparison.py` (removed: moved) | `scripts/detector_comparison/run_inventory_comparison.py` |
| `scripts/detection/run_state_evidence.py` (removed: moved) | `scripts/event_evidence/run_state_evidence.py` |
| `scripts/detection/split_state_evidence_by_swr.py` (removed: moved) | `scripts/event_evidence/split_state_evidence_by_swr.py` |
| `scripts/detection/sweep_memoryless_heldout.py` (removed: moved) | `scripts/detector_comparison/sweep_memoryless_heldout.py` |

## Decoding, 2026-11-14

Two moves into `decoding/`, neither changing behavior.

| Old place | New place |
|---|---|
| `scripts/decoding/_decode_runner.py`: `DecodeRunner`, `algorithm_params`, `create_detector`, `perform_decoding` | `decoding/fit.py` |
| `analysis/detection/memoryless_sequence/likelihood.py`: `CLUSTERLESS_LIKELIHOOD_KEYS`, `load_clusterless_encoding_model`, `clusterless_encoding_model`, `check_marks_match_encoding`, `interior_bin_distances_cm` | `decoding/encoding_model.py` |

`run_decode`, `organize_results`, `save_results_and_model` and `parse_decode_args` stay in
`scripts/decoding/_decode_runner.py`: they read `analysis.event_definition` (the HPC
decode's masks and SWR source) and `tables` (atomic writes), which `decoding/` may not
import. The memoryless detector's grid, windows and likelihood stay in its
`likelihood.py`.

## The memoryless comparison on Figure 1's run, 2026-11-16

The simulated comparison of the two replay detectors reads Figure 1's run and model. Its
values changed: every sweep was re-run.

| Old place | New place |
|---|---|
| `memoryless_sequence/sweep.py`: `build_recording`'s own recording (25 cells) and plain fit | `simulated_run.simulate_run` and `simulated_run.estimate_state_space_model` (30 cells, EM, the real decodes' settings) |
| `memoryless_sequence/sweep.py`: `non_local_intervals`; `recording_statistics.state_space_intervals` | `event_definition/nonlocal_event_rule.py`: `detect_nonlocal_events`; the labeled recording's dict carries `state_space_intervals` |
| `comparison_simulation`: `CONTINUOUS_OUTBOUND_EVENT`, `CONTINUOUS_INBOUND_EVENT`; reported values `recording_outbound_*`, `recording_inbound_*` | `SWEEP_EVENT` (the run has one sweep); `recording_sweep_*` |
| `scripts/figures/generate_figure01.py`: colors, sizes, the decoded-position panel, the row layout | `visualization/simulated_run_panels.py`, shared with `generate_memoryless_comparison.py` |
| `memoryless_sequence/simulation.py`: `stray_window_spikes`, `StrayWindowEvent`; `recording_statistics`: `stray_window_dispersions`, `outside_pieces`; reported values `recording_stray_window_*` | removed with the stray-window event of the labeled recording; the real-data pilot's outside-window values (`pilot_*`) remain |
| `generate_memoryless_comparison.py`: the burst, carving and held-out panels and `check_controls_claims`, `check_heldout_claims` | removed from the figure; the sweeps and their reported values remain |

## Figure 1's simulation, 2026-11-16

Figure 1's simulation and decode moved out of its script, and the non-local event rule became
one function. The comparison simulation's results are unchanged.

| Old place | New place |
|---|---|
| `scripts/figures/generate_figure01.py`: `make_example_detector`, the simulation and decode inside `generate_figure` | `analysis/detection/simulated_run.py` (`simulate_and_decode`, `DecodedSimulation`) |
| `analysis/detection/memoryless_sequence/comparison_simulation.py`: `DECODE_BIN_STRIDE`, `REMOTE_STATIONARY_CM`, `REMOTE_STATIONARY_S`, `splice_remote_stationary` | `analysis/detection/simulated_run.py`; `comparison_simulation` still exports them |
| `analysis/detection/nonlocal_events.py`: `get_confident_times` followed by `build_speed_eligible_event_cohort`, written out in the detector | `analysis/event_definition/nonlocal_event_rule.py`: `detect_nonlocal_events` |

| `scripts/decoding/run_decoding_bandit.py`: `nonlocal_detector_kwargs` | `decoding/transitions.py` |

Figure 1's recording also changed: the sweep that ends at the animal is in the first pause, and
a remote place held still is in the third (it was two sweeps). Its detector is now built from
the real decodes' settings (`nonlocal_detector_kwargs`, `NonLocalDecodingConfig`), where it had a
demo transition setting of its own.

## Simulated controls are events, not bursts, 2026-11-17

The two simulated controls of the memoryless comparison were called bursts; "event" in this
project already means what a detector reports, so the detectors' output on them is now a report.

| Old name | New name |
|---|---|
| `simulation.placeless_burst`, `simulation.jumping_burst`; conditions `"placeless_burst"`, `"jumping_burst"` | `placeless_event`, `jumping_event`; `"placeless_event"`, `"jumping_event"` |
| controls summary block `"bursts"` | `"control_events"` |
| `sweep_controls.BURST_DURATION_S` | `CONTROL_EVENT_DURATION_S` |
| `sweep_statistics.burst_share_range`, `placeless_burst_statistics` | `control_event_share_range`, `placeless_event_statistics` |

## A simulated event is when its spikes are, 2026-11-17

A simulated event's start and end were the bounds it was drawn in. A sparse event's first spike
can come well after its nominal start, and no detector can report what has not fired, so the
truth is now the span of the event's own spikes:

- Figure 1's run (`simulated_run.SimulatedRun.patterns`): each pattern from its first to its last
  spike, the silence from the last spike before it to the first after. The windows the patterns
  were drawn in are kept as `pattern_windows`; the training mask and the sweeps' placements still
  read those, so no fitted model or sweep recording changed.
- The comparison's recording (`comparison_simulation.simulate_and_decode`): the two added replays'
  rows of `event_times` are their spike spans (`simulation.added_spike_span`).
- The replay sweep: a report's onset and offset errors are against the replay's own first and
  last spike, computed where the report is scored (`sweep_replays.run_cell`); `duration_s` on a
  row is still the nominal duration of the grid. A pooled block (`timing_pooled`) gives the errors' quartiles,
  extremes and tail over the replays both detectors report.
- The replay sweep runs each combination up the track and down it (`sweep_replays.planned_cells`,
  `DIRECTIONS`; 9,912 cells). The animal rests at the top of the track wherever a replay is placed,
  so an upward replay ends at its edge nearest the animal and a downward one starts there; with
  upward replays alone the sliding window's long errors looked like a property of offsets.
- Each replay's placement and firing noise are drawn from generators seeded by the replay's own
  name (`sweep_replays.cell_key`, `SWEEP_SEED`), which also keys its shuffle. Before, the firing
  noise was seeded by the replicate's index alone, so the 9,912 replays shared 56 draws, and
  placements came from one generator in grid order. Every sweep value moved slightly.
- The sweep's cells run in parallel processes (`--n-workers`, 24 by default): most of a cell's time
  is JAX compiling the state-space decode for that cell's window. A row does not depend on the
  process count.

## The simulated comparison's code by experiment, 2026-11-17

| Old place | New place |
|---|---|
| `memoryless_sequence/sweep_statistics.py`: the controls' statistics (`controls_value`, `no_replay_statistics`, `control_event_share_range`, `parts_by_level`, `carving_statistics`, `part_recovery_statistics`) | `memoryless_sequence/controls_statistics.py` |
| `memoryless_sequence/sweep_statistics.py`: the held-out check's statistics (`pooled_heldout_rate`, `heldout_fold_statistics`, `placeless_event_statistics`, `lowest_estimator_agreement`, `heldout_event_classes`) | `detector_comparison/heldout_statistics.py` |
| `scripts/figures/generate_memoryless_comparison.py`, `paper/figures/memoryless_comparison/` | `scripts/figures/generate_supp_memoryless_comparison.py`, `paper/figures/supp_memoryless_comparison/` (tracked; Supplementary Figure S1 in `paper/main.tex`) |

Removed with the values that read them: `sweep_statistics.sweep_timing_statistics` and
`sweep_by_duration`, and the replay sweep summary's blocks `timing_by_duration`,
`median_duration_ratio_by_duration`, `by_path` and `median_by_distance_from_animal`. The pooled
timing block and the error distribution replace the first two; nothing read the last two.

The controls and the held-out check answer questions of their own in the question list (1.2 and
1.3); they were tagged to question 1.1 with the comparison figure's values.

## The replay sweep draws its replays, 2026-11-17

A swept replay's spikes were sampled at random from each cell's firing rate, so 280 of 9,912
replays had no spike at all and many had one or two. They are now drawn as Figure 1's patterns
are (`simulation.draw_replay`, `DrawnReplay`): spikes evenly spaced from the replay's start to
its end, at Figure 1's sweep's rate (`DRAWN_SPIKE_RATE_HZ`), each fired by the cell whose place
field the replay is passing. A replay's first and last spike are its start and end, and every
replay has at least five spikes. The sweep loses its firing-rate axis (`sweep.drawn_replay_grid`;
3,304 cells), and replicates differ only in where in the run the replay is placed. The four
replays added to the comparison's recording are drawn the same way. The controls and the
held-out check still sample their spikes (`sweep.replay_grid`, `inject_replay`).

## American spellings, 2026-11-19

Names took their American spellings (`scripts/checks/american_spelling.py` holds the word
list; `tests/repository/test_american_spelling.py` keeps the repository American). Every
module kept its place; one file was renamed. Paths are under `src/continuum_swr_replay/`.

| Old name | New name |
|---|---|
| `analysis/event_definition/analysed_time.py` (removed: renamed) | `analysis/event_definition/analyzed_time.py` |
| `analysed_time.analysed_bins`, `immobile_analysed_bins` | `analyzed_time.analyzed_bins`, `immobile_analyzed_bins` |
| `epoch_register.analysed_epochs`, `analysed_epoch_ids`, `analysed_decode_files`, `analysed_animals` | `analyzed_epochs`, `analyzed_epoch_ids`, `analyzed_decode_files`, `analyzed_animals` |
| `epoch_register.Status.ANALYSED`; the register's status `analysed` | `Status.ANALYZED`; `analyzed` |
| `runs.cli.select_analysed_epochs` | `select_analyzed_epochs` |
| `state_evidence.windows.analysed_regimes` | `analyzed_regimes` |
| `detector_comparison.features.neighbour_gaps` | `neighbor_gaps` |
| mpfc summary key `analysed_epochs` (macro `\MpfcAnalysedEpochs`) | `analyzed_epochs` (`\MpfcAnalyzedEpochs`) |
| `run_retrospective_alignment.py` provenance key `n_sessions_analysed` | `n_sessions_analyzed` |
| `compare_pfc_cohorts.py` columns `peak_penalised`, `median_abs_peak_penalised` | `peak_penalized`, `median_abs_peak_penalized` |
| `class_or_continuum.StepFit.n_labelled`: the `n_labelled` column of `ripple_class_structure/step_fits.csv` and `step_simulation.csv` | `n_labeled` (files written before keep `n_labelled`; nothing reads the column by name) |
| `run_memoryless_detection.py` left-out reason `NOT_ANALYSED` (`"not an analysed epoch in the epoch register"` in run manifests) | `NOT_ANALYZED` (`"not an analyzed epoch ..."`) |
| keyword arguments `state_evidence.windows.null_partners(centre_bins=)`, `pfc.modulation.prior_binds(penalised_peak=)` | `center_bins=`, `penalized_peak=` |

The PFC cohort records keep their key `n_analysed`: the records on disk carry it, and
`reported/mpfc/population.py` reads it.

## The all-bins cross-history model removed, 2026-11-20

The older cross-history runner, which fit `MODEL_CROSS_HISTORY_RATE` per session with every bin of a
stop at risk, was removed; nothing reported it. The rungs stay in `model_specs.py`: the duration-aware
model fits them as its onset-only comparison. Its stored output,
`models_per_bin_rate/cross_history_models.pkl`, is left on disk and nothing reads it. The module path is under
`src/continuum_swr_replay/`.

| Old path | New home |
|---|---|
| `analysis/replay_models/rate/cross_history.py` (removed: retired) | none |
| `scripts/replay_models/rate/fit_cross_history_rate.py` (removed: retired) | none |
| `tests/analysis/replay_models/rate/test_cross_history.py` (removed: retired) | none |

## The well visit in the tables, 2026-11-20

The well visit (first nose-poke in to final poke-out) became the tables' unit: the stop,
event and event × arm tables gained the visit columns (`docs/covariate_dictionary.md`,
"The well visit and its clocks"). What that replaced was removed. Paths are under
`src/continuum_swr_replay/` unless they start with `scripts/`.

| Removed | Replaced by |
|---|---|
| the dense stop time-bin table `stop_timebin_table.parquet` (canonical name `stop_timebin`, its row key in `tables/covariate_registry.py`) (removed: replaced by the visit bins) | the rate models' visit bins (`analysis/replay_models/visit_design.visit_timebins`), built per session from the stop and event tables; the file in `Processed-Data/replay_content/` is kept, no longer read |
| `analysis/event_tables/timebin_tables.build_stop_timebin_table`, `concat_session_timebin_tables`, `ONSETS_IN_EXCLUDED_BINS_ATTR` (removed: replaced by the visit bins) | `build_event_onset_table` stays; `scripts/tables/build_timebin_tables.py` writes only the event-onset table |
| `scripts/exploratory/_timeresolved_psth.py` (removed: its table is gone) | none |
| `scripts/tables/regenerate_stop_table.py` (removed: the tables are rebuilt as one set) | `scripts/tables/prepare_replay_content_data.py` |
| `scripts/tables/convert_tables_to_parquet.py` and the `fmt` parameter of `paths.table_path` (removed: the one-time pickle migration is done) | `table_path` names only the parquet file; the pickles stay under `_pickle_archive/` |
| `analysis.event_tables.uncertainty.nonlocal_weight_sum` (removed: re-export) | `analysis.event_tables.summaries.nonlocal_weight_sum` |
| `analysis.behavior.covariates.reward_basis_from_lags`, `analysis.behavior.covariates.history.reward_basis_from_lags` (removed: re-exports) | `analysis.behavior.history_basis.reward_basis_from_lags` |
| `analysis.replay_models.per_bin_design.arrival_spanning_stop_ids` (removed: moved) | `analysis.replay_models.visit_design.arrival_spanning_stop_ids`, beside the outcome-boundary shift and the bin coverage the two designs share |
| the rate and history constants re-exported by `analysis.replay_models.model_specs` (`RATE_BIN_S`, `TMAX`, `TMAX_ARRIVAL`, `N_HISTORY_LAGS`, `N_HISTORY_BASIS`, `N_REWARD_LAGS`, `REWARD_HISTORY_*`, `RECENCY_SPLINE_*`, `ARRIVAL_*`, `ARM_FE_PREDICTORS`, `SPLINE_KNOTS`, `N_CROSS_LONG_*`) (removed: re-exports) | `analysis.behavior.design_constants` |

## The supplementary figures named for their main figures, 2026-11-22

Each supplementary figure in the paper is the supplement of one main figure and is named for
it (`docs/development.md`, convention 11); its id in `reported/questions.py` is `SN`, a panel
`SNb`.

| Old place | New place |
|---|---|
| `scripts/figures/generate_supp_memoryless_comparison.py`, `paper/figures/supp_memoryless_comparison/`, figure id `memoryless_comparison` | `scripts/figures/generate_supp_figure01.py`, `paper/figures/supp_figure01/`, figure id `S1` (the supplement to Figure 1) |
| `scripts/figures/generate_supp_heldout_tetrode.py`, `paper/figures/supp_heldout_tetrode/`, figure id `heldout_tetrode` | `scripts/figures/generate_supp_figure02.py`, `paper/figures/supp_figure02/`, figure id `S2`, values tagged by panel `S2a` to `S2d` (the supplement to Figure 2) |
| `scripts/figures/generate_supp_event_gamma.py`, `paper/figures/supp_event_gamma/`, figure id `event_gamma` | `scripts/figures/generate_supp_figure03.py`, `paper/figures/supp_figure03/`, figure id `S3` (the supplement to Figure 3) |
| `tests/scripts/figures/test_generate_supp_memoryless_comparison.py`, `test_generate_supp_heldout_tetrode.py` | `test_generate_supp_figure01.py`, `test_generate_supp_figure02.py` |

## The per-time-bin counts by signal, 2026-11-22

The counts behind Figure 2i were extended to MUA, counted in the same pass and by
non-local probability, so that every count is a sum over one table.

| Old place | New place |
|---|---|
| `scripts/event_properties/measure_nonlocal_by_ripple_power.py`, `Processed-Data/nonlocal_by_ripple_power/` (`paths.NONLOCAL_BY_RIPPLE_POWER_DIR`) | `scripts/event_properties/measure_nonlocal_by_signal.py`, `Processed-Data/nonlocal_by_signal/` (`paths.NONLOCAL_BY_SIGNAL_DIR`): per epoch, signal (`ripple_power`, `mua`), signal bin and probability bin |
| `analysis.event_properties.class_or_continuum.rest_power_counts`, `nonlocal_share_by_power`, `RIPPLE_POWER_EDGES` | `analysis.event_properties.nonlocal_by_signal.rest_signal_counts`, `count_by_signal_and_probability`, `RIPPLE_POWER_EDGES` (with `MUA_EDGES`, `PROBABILITY_EDGES`, `nonlocal_counts`) |
| `analysis.event_properties.event_statistics.animal_power_counts` (columns `power_low`, `power_high`) | `event_statistics.animal_signal_counts` (columns `signal_low`, `signal_high`) |
| `draw_share_by_power` in `scripts/figures/generate_figure02.py` | `visualization.signal_share_panels.draw_share_by_signal`, which Figure 2's supplement draws MUA with |
| `reported.events.EventsInputs.nonlocal_by_ripple_power` | `EventsInputs.nonlocal_by_signal` |
| `scripts/exploratory/_trunk_zscore_timebin.py` (marked temporary) | removed |

## The declared-penalty check on the simulation engine, 2026-11-23

The declared-penalty check became a scenario list of the simulation engine
(`analysis/regression/simulation.py`), run by `scripts/checks/run_simulations.py`. Its
own drawing, fitting and reduction code was removed. In `scripts/checks/simulate_declared_penalty.py`:

| Old place | New place |
|---|---|
| `Session` (removed: replaced) | `cohort` returns each session's rows by key |
| `simulate` (removed: replaced) | `expected_counts` (the scenario's expected outcome); `regression.simulation.draw_data_set` draws |
| `fit_contrasts`, `summarize_data_set`, `run_data_set`, `run` (removed: replaced) | `regression.simulation.run` |
| `data_seed`, `bootstrap_seeds` (removed: replaced) | `regression.simulation.data_seed`, `regression.simulation.bootstrap_seeds` |
| `summarize` (removed: replaced) | `table`, from the engine's tables |
| `report` (removed: replaced) | the runner prints the table and its warnings (`run_simulations.warnings`) |
| `OUTPUT_NAME` (removed: replaced) | the runner writes `<model>.parquet` |

## The held-out tetrode run is one run, 2026-11-23

The stored held-out tetrode run in `Processed-Data/event_tetrode_holdout/` was replaced
on 2026-11-23 by the run that also holds the extra scores (the HPD length, the
shuffled-event reference, the rate and mark parts, the place information). That run was
made with the same settings and null seed 0, and until then it was in
`Processed-Data/event_tetrode_holdout_extra/`, a folder name that no longer exists. The
old run was deleted. The two runs had the same rows, and their shared columns differed by
run-to-run numerical noise of the GPU pass: the largest row difference in `gain` was
0.044 nats, and the cohort sum of `gain` differed by 0.76 nats in 5.65 million. The two
windows outputs that record the run's files, `Processed-Data/event_nonlocal_restriction_windows/`
and `Processed-Data/event_nonlocal_restriction_swr_allowed_windows/`, were made again.

| Old place | New place |
|---|---|
| `paths.EVENT_TETRODE_HOLDOUT_EXTRA_DIR` (removed) | `paths.EVENT_TETRODE_HOLDOUT_DIR` |
| `score_tetrode_holdout.py --extra-scores` (removed) | every run computes the extra scores; `--null-seed` stays |
| `tetrode_holdout.require_extra_columns`, `EXTRA_COLUMNS` (removed) | `tetrode_holdout.require_run_columns`, `RUN_COLUMNS` |
| `reported.events.require_trial_folds`, `require_tetrode_holdout_folds`, `read_tetrode_holdout` | `analysis.event_evidence.heldout_runs` |

## The restriction run is one run, 2026-11-24

The restriction run in `Processed-Data/event_nonlocal_restriction/` is, since 2026-11-24,
the run that scores all four decoders (local-only, SWR-only, full and SWR-allowed). It was
made with `score_nonlocal_restriction.py --from-stored-fit` from the fold-fit run of the
three decoders: each fold's decoder rebuilt from that run's records, no EM. Its header
records `settings.fit_source: stored` and `inputs.stored_fit`, the deleted run's folder name
and manifest digest, kept as the record of where the fit came from; every fold's fit facts
(transitions, initial distribution, convergence, the checks) are in this run's own records.
The run therefore cannot be resumed or gathered again under its own header; a decoder is
added later with `--from-stored-fit Processed-Data/event_nonlocal_restriction --output-dir
<new folder>`. The three-decoder run and its two derived outputs
(`event_nonlocal_restriction_heldout_events/`, `event_nonlocal_restriction_windows/`) were
deleted; the four-decoder run's derived outputs took their names
(they were `event_nonlocal_restriction_swr_allowed_heldout_events/` and
`event_nonlocal_restriction_swr_allowed_windows/`, folder names that no longer exist).

The three decoders' scores in this run differ from the deleted run's by about 3e-5 nats per
bin (each fold's `stored_score_difference`), which moves the full model's non-local
probability across 0.8 in a few bins: the held-out events are 88,370 instead of 88,373
(238 of 242 epochs have the same count). In the events summary 225 values moved, 16 by more
than 1e-4 relative, the largest by 1.9e-3 (an interval bound); no formatted value changed.
The 42 values `restriction_swr_allowed_run_*` and `restriction_heldout_swr_allowed_run_*`
(the second run's copies of three-decoder values and its held-out event counts) were
removed. `event_bin_facts`' records keep the deleted run's record digest
(`epoch_inputs.restriction_record`), which nothing checks.

| Old place | New place |
|---|---|
| `paths.EVENT_NONLOCAL_RESTRICTION_SWR_ALLOWED_DIR`, `..._SWR_ALLOWED_HELDOUT_EVENTS_DIR`, `..._SWR_ALLOWED_WINDOWS_DIR` (removed) | `paths.EVENT_NONLOCAL_RESTRICTION_DIR`, `..._HELDOUT_EVENTS_DIR`, `..._WINDOWS_DIR` |
| `EventsInputs.nonlocal_restriction_swr_allowed`, `nonlocal_restriction_swr_allowed_heldout_events` (removed) | `EventsInputs.nonlocal_restriction`, `nonlocal_restriction_heldout_events` |
| `REFUSED_FOUR` in `scripts/figures/generate_supp_figure02.py` (removed) | panel l reads panel i's runs |

## Figure 2's code, 2026-11-24

The figure scripts of Figure 2 and its supplement were given one shape (inputs read
first, then drawn), and what they share moved into shared modules. Paths are under
`src/continuum_swr_replay/` unless they start with `scripts/` or `tests/`.

| Old place | New place |
|---|---|
| `visualization/simulated_run_panels.py`: `LABEL_SIZE`, `SMALL_SIZE`, `PANEL_LETTER_SIZE`, `set_publication_style`, `MAX_FIGURE_SIZE_CM`, `TEXT_SIZE_LIMITS_PT`, `LINE_WIDTH_LIMITS_PT`, `check_journal_limits`, `drawn_extent`, `drawn_texts`, `TEXT_OVERLAP_PT`, `overlapping_texts`, `overlapping_panels`, `check_placement` (removed: moved) | `visualization/publication_style.py` |
| `tests/visualization/test_simulated_run_panels.py`: the journal-limit and placement tests (removed: moved) | `tests/visualization/test_publication_style.py` |
| `reported/events.py`: `EventsInputs`, `default_inputs`, `_input_staleness` (now `input_staleness`), `event_table_staleness`, `require_fold_fit`, `run_staleness`, `HeldoutRestriction`, `restriction_inputs_note`, `restriction_input_staleness`, `checked_reference`, `read_checked`, `HELDOUT_EVENTS_RERUN`, `HELDOUT_RESTRICTION_RERUN`, `heldout_restriction_refusal`, `read_heldout_restriction` (removed: moved; `reported.events.default_inputs` is the same function, for `summarize.py`) | `reported/inputs.py`, with `output_staleness`, `participation_staleness`, `read_figure_restriction` and `panel_refusal`, which the figures and the summary share |
| `scripts/figures/generate_figure02.py`: `generate_figure(nwb_file_name, epoch_name, data_dir, use_cache)` and `--data-dir` (removed) | `read_inputs(nwb_file_name, epoch_name, use_cache)`, then `generate_figure(inputs)`; the data root is `$CSR_PROCESSED_DATA_DIR` |
| `scripts/figures/generate_figure02.py`: `BOTTOM_LETTERS`, `BOTTOM_PANELS`, `BOTTOM_ROWS`, `BOTTOM_COLUMNS`, `read_participation(folder, ...)`, `restriction_refusal`; the panel names `traces`, `ripple_power` (removed) | `LAYOUT`, `PANELS`, `LETTERS`, `LETTER_ROWS`, `LETTER_COLUMNS`, `read_participation(inputs, ...)`, `reported.inputs.panel_refusal`; `event_traces`, `nonlocal_ripple_power` |
| `scripts/figures/generate_supp_figure02.py`: `read_inputs()`'s keys `calibration`, `no_spike`, `probability`, `distance`, `sweep`, `timing`, `mua`, `mua_states`, `split`, `outside`, `parts`; `REFUSED` (removed) | one key per panel, its name in `LAYOUT` (`nonlocal_mua` and `outside_distance` hold two tables each); `reported.inputs.panel_refusal` |
| `tests/scripts/figures/test_figure_script_fixes.py`: Figure 2's tests, its shared helpers and the `offline` fixture (removed: moved) | `tests/scripts/figures/test_generate_figure02.py`, `tests/support/_figure02_tables.py`, `tests/scripts/figures/conftest.py` |
| `test_the_figure02_panel_e_label_lies_inside_the_figure` (removed) | none: it drew a proxy of a layout the figure no longer has |
| `scripts/figures/generate_figure02.py`: `CLASS_ORDER`, `CLASS_COLORS`, `CLASS_LABELS`, `C_NO_SWR`, `DECODER_TIMES`, `DECODER_OUTSIDE`; `generate_supp_figure02.py`: `TRACE_CLASSES`, `TRACE_LABELS`, `OUTSIDE_LABEL`, `TRIGGER_WINDOW_S`; `generate_figure03.py` and `generate_supp_figure03.py`: `KINDS`; `visualization/figure_panels.py`: `_EVENT_CLASSES` (removed) | `visualization/palettes.py`: `EVENT_KINDS`, `OVERLAP_CLASSES`, `SIGNAL_LABELS`; `visualization/heldout_gain_panels.py`: `DIFFERENCE_LABELS`, `EVENTS_LABELS`; `analysis/event_properties/triggered_analysis.py`: `WINDOW_KINDS`, `TRIGGERED_WINDOW_S` |
| `scripts/figures/generate_figure02.py`: its render cache (`_compute_figure_inputs`, `_figure02_cache_path`, the four fingerprint functions, `FIGURE02_CACHE_VERSION`, `Processed-Data/intermediates/*_figure02_*_cache.joblib`), `--force-recompute` and `frame_event_counts` (removed) | `Processed-Data/event_trigger_windows/` (`scripts/event_properties/measure_trigger_windows.py`), read by `reported.inputs.read_trigger_windows`; `triggered_analysis.frame_event_counts`; the example session is loaded at each render (`read_session`) |
| `scripts/figures/generate_supp_figure02.py`: `read_frame_traces`, `_epoch_keys`, `_frame_epoch`, `--workers`, `NO_SPIKE_WORKERS`, `NO_SPIKE_SIGNAL` (removed) | `read_traces`, from `event_trigger_windows/` (`triggered_analysis.stored_no_spike_trace`, `stored_animal_window_medians`, `NO_SPIKE_SIGNAL`) |
| `scripts/figures/generate_figure02.py`: `EXAMPLE_NWB_FILE`, `EXAMPLE_EPOCH`; the example session read with `load_data` at each render (removed) | `parameters.EXAMPLE_NWB_FILE`, `EXAMPLE_EPOCH`; `Processed-Data/example_session/<epoch_id>/` (`scripts/decoding/store_example_session.py`), read by `reported.inputs.read_example_session`; the position, SWRs and track graph from the decode frame and the decode's environment |

## The per-animal shares by signal, 2026-11-24

The per-animal table by ripple power and MUA holds shares since the averaging rule
(each epoch's share, then the animal's mean over its epochs), so its functions were
renamed. `nonlocal_by_signal.nonlocal_counts` still returns per-epoch counts and keeps
its name.

| Old place | New place |
|---|---|
| `analysis.event_properties.event_statistics.animal_signal_counts` | `event_statistics.animal_signal_shares` |
| `analysis.event_properties.nonlocal_by_signal.animal_nonlocal_counts` | `nonlocal_by_signal.animal_nonlocal_shares` |
| `scripts/figures/generate_supp_figure02.py`: `read_mua_counts` | `read_mua_shares` |

## The regression fitter's test-only code, 2026-11-24

Public code of `analysis/regression/` that only the package's own tests used was removed
(the lead, 2026-11-24: "yes remove"), and the padded rows got one container. No result
changed: the stored reference outputs (`tests/analysis/regression/data/numpy_reference/`)
pass unedited. Paths are under `src/continuum_swr_replay/analysis/regression/`.

| Old place | Why it went, and what replaces it | Commit |
|---|---|---|
| `Likelihood.loglik`, `score_eta`, `information`, `dinformation`, `d2information` (removed) | only tests called them; the kernels call the families' `_jax_*` functions, which the tests now call through `tests/analysis/regression/conftest.py:family_kernel`; `weighted_design` stays | e6a115fd |
| `record.write_records`, `read_records`, `records_frame`, `JSON_FIELDS`, `describe` (removed) | only `test_record.py` used the parquet storage and the words of the causes; `FitRecord` and `NoFit` stay | 1bfcb448 |
| `support.terms_failing_share` (removed) | its 20 % rule was never applied by a fit; `constants.MAX_FAILING_SHARE` stays (the penalty search and `visit_rate.third_clock_decision` read it) | dcd7dbd4 |
| `terms.prior_sd_from_log_penalty` (removed) | only a test called it; the prior SD of a log penalty `rho` is `exp(-rho / 2)` | c82dd554 |
| `reml.PenaltySearch.n_grid_not_finite`, `share_without_fit` (removed) | only tests read them; `sum(search.grid_failures.values())`, and the share of `search.set_aside` not `fitted` | c82dd554 |
| `reml.SetAsideSession.running_away` (removed) | only tests read it; `reference.fit.running_away` | c82dd554 |
| `runtime.require_device` (removed) | a one-line wrapper; `in_float64` calls `runs.device.require_one_jax_device` | c82dd554 |
| `SetAsideSession(index, name, reference, refits, left_out, reason)` | `SetAsideSession(index, name, reference, refit)`, with `refits`, `left_out` and `reason` properties of the `SessionRefit` | 8debf964 |
| `likelihoods.Rows`, `likelihoods.Groups` (removed: moved) | `padding.Rows`, `padding.Groups` (`likelihoods` registers `Groups` as a JAX pytree) | afe5b9d8 |
| `padding.PaddedRows`, `padding.PaddedGroups` (removed) | `pad_rows` and `pad_groups` return `padding.Rows`: the offset, or the `Groups`, as its `aux` | afe5b9d8 |

## The pinned runs, 2026-11-24

The constants that pin which stored run of each analysis is the canonical one moved to
one block of `src/continuum_swr_replay/paths.py`, beside the roots they select, with one
naming, `<WHAT>_RUN`. No value changed. Code that a test patches reads a pin as
`paths.<PIN>` when called, so a test patches `paths`, not `swr_source`.

| Old place | New place |
|---|---|
| `analysis/event_definition/swr_source.py`: `SWR_INVENTORY_RUN_ID` | `paths.SWR_INVENTORY_RUN` |
| `analysis/event_definition/swr_source.py`: `ARTIFACT_INTERVALS_RUN_ID` | `paths.ARTIFACT_INTERVALS_RUN` |
| `analysis/detector_comparison/feature_run.py`: `EVENT_FEATURES_RUN` | `paths.EVENT_FEATURES_RUN` |
| `analysis/detector_comparison/feature_run.py`: `STATE_SPACE_FEATURES_RUN` | `paths.STATE_SPACE_FEATURES_RUN` |
| `paths.STATE_EVIDENCE_RUN` (among the folder names) | `paths.STATE_EVIDENCE_RUN` (in the pinned-run block) |
| `reported/detection.py`: `MEMORYLESS_PILOT_RUN` | `paths.MEMORYLESS_PILOT_RUN` |
