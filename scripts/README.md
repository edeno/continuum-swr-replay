# Scripts index

Analysis scripts grouped by domain. Run them from the repository root, so `scripts` is
importable: `uv run python scripts/replay_models/rate/fit_per_bin_rate_models.py`. Scripts
import the package (`continuum_swr_replay`) and read and write under the data root
(`$CSR_PROCESSED_DATA_DIR`, else `Processed-Data/`). Each script's docstring and
`--help` give its options, inputs and outputs; this index says what it is for.

**Only `_`-prefixed modules are importable by other scripts.** Shared code belongs in
the package, or in a `_`-prefixed module beside the scripts that share it
(`decoding/_decode_runner.py`, `detector_comparison/_report.py`, `checks/_report.py`, `checks/_memoryless_reference.py`). A script that is not `_`-prefixed is
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
`_trunk_zscore_extract.py`) stay in `exploratory/` as
documented exceptions to "unsupported one-off": their leading underscore already makes
them importable.

## `decoding/` — the decoders
- `run_decoding_bandit.py` — continuum decoding for the bandit task (HPC/mPFC/OFC).
- `run_decoding_contfrag.py` — decoding for the ContFrag (SWR/MUA) event sources.
- `_decode_runner.py` — not a script: the flow both decode runners share, imported as `scripts.decoding._decode_runner`.
- `measure_decoder_quality.py` — each analyzed epoch's Local-state decoding error, state fractions (the immobile ones inside the well visits) and calibration curve, which the events summary reads.

## `detection/` — the event detectors (Part 1)
- `run_artifact_detection.py` — the artifact mask of every epoch's CA1 data (`docs/design/2026-10-09-artifact-detection.md`).
- `run_swr_detection.py` — the four SWR inventories (Kay, Karlsson, Yu, Zugaro) for every epoch, on shared CA1 inputs.
- `run_carey_detection.py` — Carey's candidate detector, the fifth SWR definition, for every analyzed epoch.
- `run_memoryless_detection.py` — the Widloski & Foster sliding-window detector on each HPC epoch's saved decode.
- `sweep_memoryless_detection.py` — injected replays scored by both detectors; writes `paper/reported/inputs/detection/detection_sweep_summary.csv`.
- `sweep_memoryless_controls.py` — the detectors on activity that is not one clean replay; writes `detection_controls_summary.csv` beside it.

## `detector_comparison/` — the inventories compared (Part 1)
- `_report.py` — not a script: what the two detection-comparison reports share, imported as `scripts.detector_comparison._report`.
- `run_inventory_comparison.py` — compares the nine event inventories of each analyzed epoch on one substrate.
- `report_inventory_comparison.py` — the counts of one comparison run, per session and per animal, with figures.
- `run_event_features.py` — the features of every compared event of the nine inventories.
- `add_state_shares.py` — adds each compared event's decoder-state shares to an event features run.
- `measure_exposure_in_well_visits.py` — each epoch's non-local detector exposure, rebuilt from an inventory comparison run's tables, and its part inside the well visits (the events summary's left-out rates).
- `report_event_features.py` — the features of one event features run, per session and per animal, with figures.
- `sweep_memoryless_heldout.py` — held-out content of both detectors on simulated data; writes `detection_heldout_summary.csv` beside the sweep summaries.

## `event_evidence/` — are the events real (Part 2)
- `run_state_evidence.py` — held-out spike evidence for each decoder state on every canonical HPC decode.
- `summarize_state_calibration_in_visits.py` — the held-out state calibration re-summarized over the immobile windows that start inside a well visit, from the stored windows of `run_state_evidence.py` (Figure 2's supplement; nothing decoded again).
- `split_state_evidence_by_swr.py` — held-out spike evidence of Non-Local bouts, split by SWR coupling, over the bouts that start inside a well visit (it records `cohort_selection.ANALYZED_SUBSET` and the bouts kept and left out in its provenance).
- `analyze_event_spike_evidence.py` — the spike-evidence test that non-SWR non-local events favor their decoded location.
- `run_event_spike_evidence.sh [NJOBS]` — the run of it the events summary reads, one process per epoch (run it in tmux). It scores every event; the summary reads the events that start inside a well visit and the quiet windows whose first sample is inside one.
- `score_nonlocal_restriction.py` — per-bin held-out scores of three decoders that allow the non-local states never, only inside SWRs, or always, on every analyzed epoch: for each fold the place-field encodings are fit on the decode's training bins of the other folds (more than 1 s from the fold), the transitions and initial conditions by EM with the fold's bins missing, and the fold's bins are scored under that fit (GPU; needs `--gpus`).
- `measure_event_participation.py` — tetrode participation and spike rate of every canonical non-local event against one Local-rest control window of its length inside one well visit of the same epoch (CPU; the held-out tetrode and window runs read its controls).
- `score_event_likelihood_advantage.py` — the decoder's stored spike likelihood at each canonical event's decoded positions against the animal's position, per spike of the participation run's count (CPU).
- `score_tetrode_holdout.py` — the held-out tetrode score of every canonical non-local event and its Local-rest control: each tetrode left out of the decoder in turn and scored under the other tetrodes' position distribution against local coding, with encodings held out in folds of whole trials balanced by destination well (GPU; needs `--gpus`).
- `measure_bin_facts.py` — each decode bin's spikes over every tetrode and whether it is inside a well visit (the visits from the session cache only, whose entries the run records), on every epoch of the restriction run (CPU; reads the lab's database, read only).
- `sum_nonlocal_restriction_windows.py` — those per-bin scores summed over each canonical non-local event and its Local-rest control, with each window's spike count and its sums over the bins outside SWRs (CPU; `--restriction-dir` reads another restriction run).
- `detect_heldout_events.py` — the non-local events detected by the canonical rule from the restriction run's held-out state probabilities (Figure 2j reads them), with the restriction scores summed over each, which held-out and canonical events share a bin, and each epoch's bins, spikes and scores by category, well visit and held-out event window; each epoch checks that the rule on the canonical decode's own probabilities gives its canonical labels (CPU; `--restriction-dir` reads another restriction run).

## `event_properties/` — what the events are like (Part 2)
Each is over the events that start inside a well visit and the time inside one
(`docs/figure-pipeline.md`, "The data subset of Figures 2 and 3"), and records
`cohort_selection.ANALYZED_SUBSET` in its manifest; `measure_content_along_arms.py`
and the `measure` stage of `measure_ripple_class_structure.py` measure every event,
and their readers select.
- `measure_ripple_class_structure.py` — do SWR-coupled and non-SWR non-local events differ by kind or by degree of ripple power.
- `measure_brain_state_reference.py`, `measure_event_gamma.py`, `measure_event_start_and_rest_theta.py` — each epoch's running and rest reference levels, each event's gamma power, start distance and rest theta.
- `sweep_event_rule.py` — each epoch's non-local events, and those with no SWR, at other settings of the event rule (the sensitivity check of Figure 2's supplement; the canonical events are not changed).
- `measure_nonlocal_by_signal.py` — each epoch's immobile time bins by ripple power, by MUA and by non-local probability (Figure 2i and the MUA panel of Figure 2's supplement).
- `bootstrap_event_properties.py` — bootstrap of the SWR against non-SWR property contrasts.

## `tables/` — the event, stop and time-bin tables
- `prepare_replay_content_data.py` — build the canonical replay-content tables (event / event-arm / event-patch / event-sequence / stop).
- `build_timebin_tables.py` — build the event-onset table.
- `prepare_contfrag_replay_content_data.py` — assemble ContFrag content tables.
- `assemble_replay_content_shards.py`, `assemble_contfrag_shards.py` — assemble per-animal table shards into one bundle.
- `refresh_contfrag_behavioral_covariates.py` — rebuild the ContFrag tables, reusing a preserved bundle's posterior aggregates.
- `write_regeneration_manifest.py` — validate candidate covariate bundles and write the release manifest.
- `_table_runner.py` — not a script: the session selection, shard reading and publishing the drivers above share.

## `behavior/` — what the animal did and what the behavioral model says
- `fetch_spatial_bandit_behavior.sh` — pull the canonical SpatialBanditTask behavior CSVs.
- `build_qvals_table.py` — build the per-stop behavioral-model covariate table (`qvals_table`).

The Part 3 model runners are under `replay_models/`, one folder per subpackage of
`analysis/replay_models/`.

## `replay_models/rate/` — event-rate models and their plots
- `fit_per_bin_rate_models.py`, `fit_per_stop_rate_models.py` — Family B / A rate ladders ([../docs/rate_models.md](../docs/rate_models.md)).
- `fit_swr_contrast_rate.py` — the `is_swr`-contrast per-bin ladder (G0–G8 / S0–S10).
- `fit_uncertainty_rate.py` — the MU rate models.
- `fit_cross_history_ridge_cv.py` — ridge cross-validated comparison of self vs self+cross history, per animal.
- `fit_duration_aware_history_rate.py` — self- and cross-history refit on the at-risk design, per session or pooled per animal.
- `plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_model_checking.py`, `plot_uncertainty_rate.py` — rate figures; each docstring names the panels that are reported values.
- `describe_visit_clocks.py` — model-free event-start rates per at-risk second on the three visit clocks, per kind and animal, and the stops by how they end; writes `models_convention/descriptive/visit_clock_curves.parquet`.

## `replay_models/coupling/` — SWR-coupling strength of the events
- `fit_swr_power_model.py` — each non-local event's ripple power against the behavioral covariates.

## `replay_models/content/` — replay content / choice models
- `fit_replay_content_models.py` — arm/patch content hierarchy (M0–M7 / P0–P4), per `is_swr` split.
- `fit_swr_contrast_content.py` — the HMM-harmonized content `is_swr`-contrast ladder (C / CP).
- `compare_event_source_models.py` — choice (clogit) model comparison.
- `compare_content_estimators.py` — three content estimators on one shared event set ([../docs/design/content_estimator_comparison.md](../docs/design/content_estimator_comparison.md)).
- `analyze_content_occupancy.py` — where content sits relative to the animal and the reward wells, with its own plots (wells placed at the animal's median stopped position on each leaf arm). The events summary reads `scripts/event_properties/measure_content_along_arms.py`'s output instead, whose wells are the track graph's end points.
- `quantify_swr_overlap_geometry.py`, `generate_replay_content_figures.py`.
- `analyze_uncertainty_content.py` — reward-belief uncertainty profiles, content rates and response sensitivities.

## `replay_models/choice/` — content and the choice that follows
- `run_retrospective_alignment.py` ("Stage 5B") — does content during a stop match the arm visited next.
- `run_stay_leave.py` ("Stage 6") — content × reward outcome → stay or leave.

## `figures/` — publication figures
- `generate_figure01.py`, `generate_figure02.py`, `generate_figure03.py`, `generate_figure05.py`, `generate_all_figures.py` (orchestrator).
- `generate_uncertainty_content_figure.py` — the uncertainty/content story and its diagnostic figures.
- `generate_supp_*.py`, `generate_event_gallery.py`, `run_event_gallery.sh`.
- `plot_swr_contrast_coefs.py` — per-animal coefficient forest plots for the `is_swr` rate and content contrast ladders.
- `generate_supp_figure01.py` — supplemental figure: the Widloski & Foster detector against the state-space detector in simulation.
- `generate_supp_figure02.py` — Figure 2's supplement (S2), ten panels in four rows: the decoder's states calibrated by held-out spikes and MUA in No-Spike; the event rule's sensitivity and the timing table; non-local time against MUA, the HPD coverage, the held-out tetrode gain split by bins with and without spikes, and shared content; the events' peak ripple power and the gain outside SWRs away from the SWRs (reads every analyzed epoch's decode frame with `--workers` processes, 4 by default).
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
- `compare_regression_to_mgcv.py` (with `compare_regression_to_mgcv.R`) — the shared regression fitter beside mgcv on four synthetic data sets; rewrites the fixtures in `tests/analysis/regression/data/mgcv/` ([../docs/design/regression_fitter.md](../docs/design/regression_fitter.md)).
- `compare_regression_to_neurospatial.py` — run with neurospatial's own interpreter: its REML penalty selection on one synthetic term, written to `tests/analysis/regression/data/neurospatial/`.
- `simulate_interval_coverage.py` — the coverage of the animal-balanced interval when the draws resample animals then sessions (the convention's rule), animals then days then sessions, or animals only, in simulation; writes the table to `--out` (`--quick` for a smoke run).
- `run_simulations.py` — runs a model's scenario list through the simulation engine (`analysis/regression/simulation.py`: the outcome drawn from a planted truth on the model's rows, the real fit, estimands and reduction); `--model` names an entry of its `MODELS`; writes `<model>.parquet` with the seed, the commit and the declared penalties to `--output-dir` (by default `models_convention/simulations/`; `--quick` for a smoke run, `--reference` for a fixture).
- `simulate_declared_penalty.py` — the scenario list of the declared-penalty rule for a per-session smooth (run by `run_simulations.py --model declared_penalty`): the penalty that gives a target median edf (4, 6, 8), then the bias, spread, interval coverage, interval-excludes-zero and day-level signed-rank rejection rates (and a reference t test over the five animals) of the animal-balanced curve contrast for a planted rise and a flat curve, with the sessions each estimate used; one shared curve, no session, day or animal effect; writes `declared_penalty.parquet`.
- `check_stored_folds_reproduce.py` — reads only: the held-out runs' stored folds are the fold rule's (`analysis/regression/folds.py`) from the session cache's trials, with the counts their manifests record, and the held-out tetrode run records the same fold counts as the restriction run; exits non-zero on any difference (`--workers`, `--epochs`).
- `compare_regression_to_current_fitters.py` — the shared fitter without a penalty beside the current per-bin and content fitters on three real sessions.
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
- `check_pfc_ridge_lambda0.py` — the audit that the penalized solver at zero penalty is the maximum-likelihood one.
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
