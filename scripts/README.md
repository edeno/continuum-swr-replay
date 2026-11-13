# Scripts index

Analysis scripts grouped by domain. Run them from the repository root, so `scripts` is
importable: `uv run python scripts/replay_models/rate/fit_per_bin_rate_models.py`. Scripts
import the package (`continuum_swr_replay`) and read and write under the data root
(`$CSR_PROCESSED_DATA_DIR`, else `Processed-Data/`). Each script's docstring and
`--help` give its options, inputs and outputs; this index says what it is for.

**Only `_`-prefixed modules are importable by other scripts.** Shared code belongs in
the package, or in a `_`-prefixed module beside the scripts that share it
(`decoding/_decode_runner.py`, `detection/_report.py`, `checks/_report.py`, `checks/_memoryless_reference.py`). A script that is not `_`-prefixed is
never imported by another script or by `src/`; tests may import any script.
[`tests/repository/test_no_new_script_imports.py`](../tests/repository/test_no_new_script_imports.py)
enforces this for scripts, and
[`tests/repository/test_import_layering.py`](../tests/repository/test_import_layering.py) for `src/`.
The first test's `ALLOWED` table is the current exception list, and it only
shrinks. One entry is left: `exploratory/_content_cohort_control.py`, which runs
`replay_models/content/fit_swr_contrast_content.py` with its cohort rule bypassed.

A runner that computes on the CPU says so first thing in its `main`
(`default_jax_platforms("cpu")`), not when imported.

The exploratory scripts that unit tests pin (`_verify_harmonization_parity.py`,
`_statescript_contingency.py`, `_augment_data_over_time.py`,
`_trunk_zscore_extract.py`, `_trunk_zscore_timebin.py`) stay in `exploratory/` as
documented exceptions to "unsupported one-off": their leading underscore already makes
them importable.

## `decoding/` — the decoders
- `run_decoding_bandit.py` — continuum decoding for the bandit task (HPC/mPFC/OFC).
- `run_decoding_contfrag.py` — decoding for the ContFrag (SWR/MUA) event sources.
- `_decode_runner.py` — not a script: the flow both decode runners share, imported as `scripts.decoding._decode_runner`.
- `measure_decoder_quality.py` — each analysed epoch's Local-state decoding error, state fractions and calibration curve, which the events summary reads.

## `detection/` — event detectors and the detection comparison
- `_report.py` — not a script: what the two detection-comparison reports share, imported as `scripts.detection._report`.
- `run_artifact_detection.py` — the artifact mask of every epoch's CA1 data (`docs/design/2026-10-09-artifact-detection.md`).
- `run_swr_detection.py` — the four SWR inventories (Kay, Karlsson, Yu, Zugaro) for every epoch, on shared CA1 inputs.
- `run_carey_detection.py` — Carey's candidate detector, the fifth SWR definition, for every analysed epoch.
- `run_memoryless_detection.py` — the Widloski & Foster sliding-window detector on each HPC epoch's saved decode.
- `run_inventory_comparison.py` — compares the nine event inventories of each analysed epoch on one substrate.
- `report_inventory_comparison.py` — the counts of one comparison run, per session and per animal, with figures.
- `run_event_features.py` — the features of every compared event of the nine inventories.
- `add_state_shares.py` — adds each compared event's decoder-state shares to an event features run.
- `report_event_features.py` — the features of one event features run, per session and per animal, with figures.
- `run_state_evidence.py` — held-out spike evidence for each decoder state on every canonical HPC decode.
- `analyze_event_spike_evidence.py` — the spike-evidence test that non-SWR non-local events favour their decoded location.
- `run_event_spike_evidence.sh [NJOBS]` — the run of it the events summary reads, one process per epoch (run it in tmux).
- `split_state_evidence_by_swr.py` — held-out spike evidence of Non-Local bouts, split by SWR coupling.
- `measure_ripple_class_structure.py` — do SWR-coupled and non-SWR non-local events differ by kind or by degree of ripple power.
- `measure_brain_state_reference.py`, `measure_event_gamma.py`, `measure_event_start_and_rest_theta.py` — each epoch's running and rest reference levels, each event's gamma power, start distance and rest theta.
- `bootstrap_event_properties.py` — bootstrap of the SWR against non-SWR property contrasts.
- `sweep_memoryless_detection.py` — injected replays scored by both detectors; writes `paper/reported/inputs/detection/detection_sweep_summary.csv`.
- `sweep_memoryless_controls.py`, `sweep_memoryless_heldout.py` — the detectors on activity that is not one clean replay, and held-out content; write `detection_{controls,heldout}_summary.csv` beside it.

## `tables/` — the event, stop and time-bin tables
- `prepare_replay_content_data.py` — build the canonical replay-content tables (event / event-arm / event-patch / event-sequence / stop).
- `regenerate_stop_table.py` — rebuild the per-stop immobility table.
- `build_timebin_tables.py` — build the per-(stop×bin) design tables.
- `prepare_contfrag_replay_content_data.py` — assemble ContFrag content tables.
- `assemble_replay_content_shards.py`, `assemble_contfrag_shards.py` — assemble per-animal table shards into one bundle.
- `refresh_contfrag_behavioral_covariates.py` — rebuild the ContFrag tables, reusing a preserved bundle's posterior aggregates.
- `write_regeneration_manifest.py` — validate candidate covariate bundles and write the release manifest.
- `convert_tables_to_parquet.py` — the one-time pickle-to-parquet migration of the tables.
- `_table_runner.py` — not a script: the session selection, shard reading and publishing the drivers above share.

## `behavior/` — what the animal did and what the behavioral model says
- `fetch_spatial_bandit_behavior.sh` — pull the canonical SpatialBanditTask behavior CSVs.
- `build_qvals_table.py` — build the per-stop behavioral-model covariate table (`qvals_table`).

The Part 3 model runners are under `replay_models/`, one folder per subpackage of
`analysis/replay_models/`.

## `replay_models/rate/` — event-rate models and their plots
- `fit_per_bin_rate_models.py`, `fit_per_stop_rate_models.py` — Family B / A rate ladders ([../docs/rate_models.md](../docs/rate_models.md)).
- `fit_swr_contrast_rate.py` — the `is_swr`-contrast per-bin ladder (G0–G8 / S0–S10).
- `fit_uncertainty_rate.py`, `fit_cross_history_rate.py` — MU / cross-history rate models.
- `fit_cross_history_ridge_cv.py` — ridge cross-validated comparison of self vs self+cross history, per animal.
- `fit_duration_aware_history_rate.py` — self- and cross-history refit on the at-risk design, per session or pooled per animal.
- `plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_model_checking.py`, `plot_uncertainty_rate.py` — rate figures; each docstring names the panels that are reported values.

## `replay_models/coupling/` — SWR-coupling strength of the events
- `fit_swr_power_model.py` — each non-local event's ripple power against the behavioral covariates.

## `replay_models/content/` — replay content / choice models
- `fit_replay_content_models.py` — arm/patch content hierarchy (M0–M7 / P0–P4), per `is_swr` split.
- `fit_swr_contrast_content.py` — the HMM-harmonized content `is_swr`-contrast ladder (C / CP).
- `compare_event_source_models.py` — choice (clogit) model comparison.
- `compare_content_estimators.py` — three content estimators on one shared event set ([../docs/design/content_estimator_comparison.md](../docs/design/content_estimator_comparison.md)).
- `analyze_content_occupancy.py` — where content sits relative to the animal and the reward wells; `--all` writes the table the events summary reads.
- `quantify_swr_overlap_geometry.py`, `generate_replay_content_figures.py`.
- `analyze_uncertainty_content.py` — reward-belief uncertainty profiles, content rates and response sensitivities.

## `replay_models/choice/` — content and the choice that follows
- `run_retrospective_alignment.py` ("Stage 5B") — does content during a stop match the arm visited next.
- `run_stay_leave.py` ("Stage 6") — content × reward outcome → stay or leave.

## `figures/` — publication figures
- `generate_figure01.py`, `generate_figure02.py`, `generate_figure03.py`, `generate_figure04.py`, `generate_all_figures.py` (orchestrator).
- `generate_uncertainty_content_figure.py` — the uncertainty/content story and its diagnostic figures.
- `generate_supp_*.py`, `generate_event_gallery.py`, `run_event_gallery.sh`.
- `plot_swr_contrast_coefs.py` — per-animal coefficient forest plots for the `is_swr` rate and content contrast ladders.
- `generate_memoryless_comparison.py` — supplemental figure: the Widloski & Foster detector against the state-space detector in simulation.
- `generate_inventory_examples.py` — seeded example events from each cell of one inventory comparison run.
- `generate_event_feature_examples.py` — seeded comparator-only and ours-only example events from one event features run.

## `workflows/` — runners that run other scripts
- `run_refits.py` — reruns the CPU fits for Part 3 and the figures that read them, in dependency order (`make refits`; `--dry-run` lists the runs).

## `checks/` — parity, audits, checks and doc writers
- `_report.py` — not a script: the command line and markdown table several reports share, imported as `scripts.checks._report`.
- `check_data_availability.py`, `diagnose_trial_detection.py`, `compute_hpc_raw_stats.py`, `run_quality_metrics.py`.
- `_memoryless_reference.py` — not a script: an independent re-implementation of the Widloski & Foster reference's sorted-cell arithmetic.
- `compare_memoryless_to_reference.py` — the memoryless port beside that reference; writes `paper/reported/inputs/detection/reference_check_summary.csv`.
- `check_fit_provenance.py` — which fitted artifacts under `models*/` are stale for the tables they read.
- `audit_event_patch_fraction.py`, `audit_stop_table_reward_history.py` — keyed-parity audits of a candidate table against its reference.
- `freeze_regression_baseline.py`, `compare_to_golden_baseline.py` — freeze a golden baseline, and diff a regenerated root against it.
- `table_parity.py` — rebuild every canonical table into a scratch root, and diff two roots table by table.
- `mutation_check.py` — AST mutation check on the covariate builders (`docs/design/builder_test_floor.md`).
- `summarize_cohort_audits.py` — rolls the per-run `*_cohort_audit.csv` files into one table.
- `write_pipeline_map.py` — regenerates `docs/regression_pipeline_map.md`; `--check` fails on drift.
- `write_epoch_register_doc.py` — rewrites the generated parts of `docs/epoch_register.md`; `--check` fails on drift.
- `summarize_stage6_fdr.py` — Stage 6's per-role cross-session FDR table for any saved stay/leave fit.

## `pfc/` — PFC responses to the events (Part 4) and their audits
- `run_pfc_analysis_bandit.py` — the mPFC unit GLMs ([../docs/design/pfc_unit_glm_estimator.md](../docs/design/pfc_unit_glm_estimator.md)).
- `run_pfc_marked_kernels.py` — the content-marked kernels ([../docs/design/pfc_marked_event_kernels.md](../docs/design/pfc_marked_event_kernels.md)).
- `run_pfc_marked_model_family.py` — the descriptive marked model family ([../docs/design/pfc_marked_model_family.md](../docs/design/pfc_marked_model_family.md)).
- `summarize_pfc_marked_family.py` — the marked-family bundles summarized unit → session → animal.
- `run_pfc_stop_context.py` — does the mPFC response to replay change with the stop's context.
- `compare_pfc_cohorts.py` — the mPFC unit-GLM before/after audit of two bundle trees.
- `compare_pfc_event_set_bundles.py` — does an extended-event-set mPFC bundle reproduce the canonical one where they overlap.
- `compare_pfc_marked_family_trees.py` — how much two marked-family bundle trees differ for the same sessions and models.
- `measure_pfc_null_geometry.py` — what the held-out null's windows look like beside the real events.
- `measure_pfc_mark_context.py` — what the marked kernels' content mark is entangled with.
- `calibrate_pfc_kernel_timing.py` — how far the kernel estimator moves a response's onset and peak, in simulation.
- `part4_rate_threshold_sensitivity.py` — Part 4's per-unit values at several firing-rate gates, without a refit.
- `diagnose_pfc_event_definition.py` — the retired run-time event construction against the canonical event table.
- `check_pfc_ridge_lambda0.py` — the audit that the penalised solver at zero penalty is the maximum-likelihood one.
- `compare_pfc_fitters.py` — **history**: the audit of the batched JAX fitter against per-neuron statsmodels.

## `reported/` — the reported values

Every number the project reports is read from a summary file, named for what it holds
(`paper/reported/<summary>.json`: survey, detection, events, behavior, mpfc).
- `summarize.py --summary NAME` — writes that summary from outputs the analyses already wrote
  (`continuum_swr_replay.reported.<name>`), with a record of the code that built it; refits nothing.
  It writes into `paper/reported/` only from committed code; `--output-dir` makes a trial build.
- `check_release.py` — before a milestone: fails on a summary that is missing, stale or mis-tagged,
  or on a pending value the manuscript cites (`make release-check`).
- `emit_reported_values.py` — writes `paper/reported_values.tex` (one LaTeX macro per value) and rewrites the
  number inside every `<!--v:<summary>.key-->…<!--/v-->` marker in the markdown under `paper/` and `docs/`
  (markers in code are left alone). Every number it changes is listed in `paper/reported/changes.md`.

## `exploratory/` — unsupported one-offs
Scratch scripts (`_*.py`): diagnostics, ad-hoc plots, verification checks. Not part of the
audited pipeline; outputs go to `.claude/research/*_outputs/`. May be stale. The
exceptions are the five that unit tests pin (listed at the top of this file), which
must keep passing their tests.

## Moved scripts

Where each moved script went (2026-10 and 2026-11) is recorded in
[../docs/history/2026-module-moves.md](../docs/history/2026-module-moves.md).
