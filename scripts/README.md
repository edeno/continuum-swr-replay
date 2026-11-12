# Scripts index

Analysis scripts grouped by domain. Run from the repo root, e.g.
`uv run python scripts/behavior/rate/fit_per_bin_rate_models.py`. Scripts import the package
(`continuum_swr_replay`) and read/write `Processed-Data/`. Run from the repo root so
`scripts` is importable.

**Only `_`-prefixed modules are importable by other scripts.** Shared code belongs in
the package, or in a `_`-prefixed module beside the scripts that share it
(`decoding/_decode_runner.py`, `detection/_report.py`, `checks/_report.py`, `checks/_memoryless_reference.py`). A script that is not `_`-prefixed is
never imported by another script or by `src/`; tests may import any script.
[`tests/repository/test_no_new_script_imports.py`](../tests/repository/test_no_new_script_imports.py)
enforces this for scripts, and
[`tests/repository/test_import_layering.py`](../tests/repository/test_import_layering.py) for `src/`.
The first test's `ALLOWED` table is the current exception list, and it only
shrinks. One entry is left: `exploratory/_content_cohort_control.py`, which runs
`behavior/content/fit_swr_contrast_content.py` with its cohort rule bypassed.

The detection-comparison and memoryless-sweep families once imported each other; their
shared code is now in the package (2026-10-25): the inventory run's roster, grid and table
layout in `analysis/inventory_comparison/inventory_run.py`, the event features run's tables
and inputs in `analysis/inventory_comparison/feature_run.py` (its populations in
`populations.py`), a decode's marks (`decode_marks`) in `analysis/hippocampus/detection_inputs.py`,
the example figures' draw, inputs, recomputation and figure in
`visualization/inventory_examples.py`, the sweep's simulated recording and detector runs in
`analysis/hippocampus/memoryless_sequence/sweep.py`, the controls' conditions in
`memoryless_sequence/sweep_controls.py`, the comparison figure's simulation in
`memoryless_sequence/comparison_simulation.py`, the epoch selection in
`runs/cli.py` (`select_analysed_epochs`), the device choice and XLA flags in
`runs/device.py`, the sweeps' commit in `runs/provenance.py` (`described_commit`) and the
remote distance in `analysis/constants.py` (`REMOTE_CM`). A runner that computes on the CPU
says so first thing in its `main` (`default_jax_platforms("cpu")`), not when imported.

The exploratory scripts that unit tests pin (`_verify_harmonization_parity.py`,
`_statescript_contingency.py`, `_augment_data_over_time.py`,
`_trunk_zscore_extract.py`, `_trunk_zscore_timebin.py`) stay in `exploratory/` as
documented exceptions to "unsupported one-off": their leading underscore already makes
them importable.

## `decoding/` — the decoders
- `run_decoding_bandit.py` — continuum decoding for the bandit task (HPC/mPFC/OFC).
- `run_decoding_contfrag.py` — decoding for the ContFrag (SWR/MUA) event sources.
- `_decode_runner.py` — not a script: the flow both decode runners share (the HPC
  masks and SWR source, fit, save, resume), imported as `scripts.decoding._decode_runner`;
  each runner's `RUNNER` record holds what differs. Only the NonLocal runner keeps an
  HPC decode's per-bin log-likelihood (`hpc_log_likelihood`).

## `detection/` — event detectors and the detection comparison
- `_report.py` — not a script: what the two detection-comparison reports share (the
  pending marking of the memoryless inventories, caption and quantile wording,
  `read_run`, which reads a run's `ok` epochs through `feature_run.read_manifest` and
  `ok_epochs`, pooling sessions within an animal), imported as `scripts.detection._report`.
- `run_artifact_detection.py` — the artifact mask: LFP and spike artifacts in every
  epoch's CA1 data, on CPU (S blocks, reference and CA1 steps, synchronous spike
  lines; `docs/design/2026-10-09-artifact-detection.md`).
- `run_swr_detection.py` — the four SWR inventories (Kay, Karlsson, Yu and Zugaro, from
  `ripple_detection` at its pinned commit) for every epoch, on shared CA1 inputs: the
  stored ripple band, the referenced raw LFP behind it, and `load_data`'s speed and time
  span (refresh design, Stage 3). Per epoch it checks that the raw LFP re-filtered as
  Spyglass filtered it reproduces the stored band (a mismatch stops the epoch), chooses
  Zugaro's channel by buzcode's `bz_GetBestRippleChan` rule, records Yu's threshold
  diagnostics, and checks Kay against the canonical `SWR_event`. `--animal`, `--date` and
  `--epoch` narrow the epochs, which run on a dask `LocalCluster` of `--n-workers`
  processes, each logging to `logs/{epoch_id}_swr.log`; only the main process writes
  `manifest.json`. A failing epoch is recorded and the rest run; a rerun skips the epochs
  already `ok` unless `--overwrite` is given.
- `run_carey_detection.py` — Carey's candidate detector (`Carey_candidate_detector`, ripple power and
  multiunit activity jointly), the fifth SWR definition, for every analysed epoch, into
  `replay_detection_comparison/carey/<run_id>/`. It reads the SWR runner's inputs: the stored ripple band
  of the tetrodes rule R-T kept, cut to the position span, the head speed and the artifact mask. It
  refuses an epoch whose band hashes differ from the SWR run's, or whose marks are not the ones the
  decode's encoding model was fitted on, and writes nothing for it. Each tetrode's marks are binned
  onto the LFP times as one unit. The detector's normalization leaves out the artifact mask
  (`normalization_mask`), as the other four SWR detectors' does. Four adaptations of the published
  detector are recorded in the manifest: a tetrode's multiunit train stands in for a unit; the package's
  Hilbert ripple score, not the published spectral one; the stored 150–250 Hz band averaged over the
  kept tetrodes, not one 140–250 Hz channel; no theta restriction. One event table and one
  observed-stretches table per epoch, plus `manifest.json`. `--animal`, `--date` and `--epoch` narrow the
  epochs, which run on a dask `LocalCluster` of `--n-workers` processes, each logging to
  `logs/{epoch_id}_carey.log`; a failing epoch is recorded and the rest run, and a rerun skips the
  epochs already `ok` unless `--overwrite` is given. Built and tested; it ran on all 242 epochs
  (run `fa6537e48bcf`). Reads a decode's marks as the memoryless runner does
  (`detection_inputs.decode_marks`).
- `run_memoryless_detection.py` — the Widloski & Foster sliding-window detector
  on each HPC epoch's saved decode, with its encoding-pair shuffle; one row per
  candidate sequence (spec 02). Loads the marks from the Spyglass database
  unless `--cache-dir` already holds them; a failing epoch is recorded and the
  rest run; `run.log` and `manifest.json` sit beside the outputs. `--date` and
  `--epoch` narrow `--epoch-id`, `--animal` or `--all`. With
  `--cuda_visible_devices` the epochs run in parallel on a dask
  `LocalCUDACluster`, one epoch per GPU at a time, each logging to
  `logs/{epoch_id}_memoryless.log`; only the main process writes the manifest.
- `run_inventory_comparison.py` — compares nine inventories of each analysed epoch on one substrate:
  the five SWR definitions (`kay`, `karlsson`, `yu`, `zugaro`, `carey`), `mua`, `state_space`, and the
  Widloski detector under two cohorts (`memoryless`, cohort `published`; `memoryless_criteria`, cohort
  `replay_criteria`). Takes the Carey run (`--carey-run-id`) and the memoryless run
  (`--memoryless-run-id`); the SWR run is the one `swr_source` pins. Per epoch it writes, as one set,
  the inventories, observation tables, every overlap pair, directional agreement, time coverage,
  components, the fragment recount, the three cells, the SWR label matrix (three sequence cohorts
  under five SWR definitions) and excluded time by reason, to
  `replay_detection_comparison/inventory/<run_id>/`. A run with no usable record reads `unavailable`,
  never empty; a failed epoch writes nothing. `--animal`, `--date`, `--epoch` or `--epoch-id` narrow
  the epochs, which run one at a time; a rerun skips the epochs already `ok` unless `--overwrite` is
  given. Built and tested on simulated epochs; ran on all 242 analysed epochs (run `afeb897d850f`). The Widloski columns
  are pending the Phase 2 pilot checks.
- `report_inventory_comparison.py` — the counts of one comparison run, per session and per animal
  (sums of counts and seconds, never averages of proportions; `unavailable` sessions listed, not
  pooled; `unknown` never counted as `unmatched` or non-SWR). Tables go to `<inventory run>/report/`
  with a `report.md`; figures to `paper/figures/inventory_comparison/<run_id>/` with `captions.md`.
  Counts only, no verdicts; every Widloski row and caption is marked pending the Phase 2 pilot checks.
- `run_event_features.py` — the event features of every compared event of the nine inventories, on the
  decode grid by one rule (no baseline windows: dropped by the lead, 2026-10-18). Takes the comparison run
  (`--inventory-run-id`); the memoryless run is the one its manifest names. Per epoch it rebuilds the
  pairwise memberships from the stored tables and fails unless they reproduce the stored `directional`
  and `cells`. It then writes, as one set, the `features` table (timing, engagement, one ripple and MUA
  trace, `ss_*` and `ms_*` spatial features, context, and the common spike-evidence score), the
  `pair_membership` table and the `rejection` table (the memoryless rejection stage of each
  state-space event), to `replay_detection_comparison/event_features/<run_id>/`. An epoch the
  memoryless run has no record for has `ms_*` missing and its memberships `unknown`. `--animal`,
  `--date`, `--epoch` or `--epoch-id` narrow the epochs; `--n-workers` sets the worker processes; a
  rerun skips the epochs already `ok` unless `--overwrite` is given. Built and tested on simulated
  epochs; the one-epoch smoke ran (run `68f51dabdb3c`), and the pilot ran on 5 epochs (run `86ed2e7fa90a`: 4 `ok`, and peanut, which failed there on a bug at its last window, re-checked `ok` under `f8a6984ae781`); the full run ran on all 242 epochs (run `30cf1cc432e9`, all `ok`). The
  Widloski columns and every `ms_*` feature are pending the Phase 2 pilot checks.
- `add_state_shares.py` — adds each compared event's Local, No-Spike, non-local and uncertain shares (the
  share of its decode-grid rows with that state above 0.8; uncertain when none is) to an event features
  run, without rerunning it: `{epoch_id}_state_shares.parquet` beside the run's tables and a
  `state_shares_manifest.json`. It refuses an epoch unless the decode's sha256 and grid are the run's and
  the recomputed non-local share equals the stored `ss_frac_rows_nonlocal` exactly. Ran on run
  `30cf1cc432e9` (242 `ok`, about a minute with 8 workers).
- `report_event_features.py` — the features of one event features run, per session and per animal
  (sums of counts, never averages of proportions; quantiles of the events; `unknown` never counted as
  `unmatched`; the evidence beside the spikes per row, with no baseline). Tables go to `<features run>/report/` with a `report.md`;
  figures to `paper/figures/event_features/<run_id>/` with `captions.md`. Counts only, no verdicts;
  every Widloski row and caption is marked pending the Phase 2 pilot checks.
- `run_state_evidence.py` — held-out spike evidence for each decoder state (Local, No-Spike,
  Non-Local Continuous / Fragmented) on every canonical HPC decode: confident bouts against
  matched random windows, and calibration curves over random 20 ms windows, scored by a
  tetrode half the choice did not see (10 seeded half-partitions x 2 directions), primarily
  under a 2-fold time-held-out running-only refit (alternating 30 s blocks, each window scored
  by the fold without its centre bin), and secondarily under a running-only refit on every
  running bin and the decode's own encoding. Method and caveats: the package
  `src/continuum_swr_replay/analysis/hippocampus/state_evidence/`. Per epoch it checks that the
  reloaded marks refit the stored encoding exactly, that the per-tetrode terms reproduce
  `results.nc` and the library's likelihood, and that no event bin is in a running-only fit,
  then writes a per-window parquet and a validation JSON; then the summary CSVs, three figures,
  `manifest.json` (a recorded reason for every skipped, failed or recomputed epoch) and a
  `README.md`. `--epochs`, `--decode-dir` (default `Processed-Data/`; without `--epochs`, every
  `*_HPC` epoch with a decode file there), `--output-dir` (default
  `Processed-Data/_refresh_<date>/state_evidence/`, the only place written), `--n-workers`
  dask CPU workers, `--window-level-stats`; a rerun reuses an epoch only when its commit, run
  settings and decode files (size, mtime) all match, unless `--overwrite`.
- `analyze_event_spike_evidence.py` — the spike-evidence reality check: does each
  non-SWR non-local event's prior-excluded spike likelihood favour its decoded
  location over the animal's, above a matched immobility baseline. Its closing
  summary prints `event_statistics.spike_evidence_statistics`, the function the
  events summary reports from.
- `run_event_spike_evidence.sh [NJOBS]` — the run of it that the events summary and
  `measure_ripple_class_structure.py` read: every event of every analysed epoch
  (`--n-sample 1000000 --likelihood stored`), one process per epoch, written to
  `Processed-Data/event_spike_evidence/<epoch>/`. An epoch whose metrics exist is
  skipped. Run it in tmux.
- `sweep_memoryless_detection.py` — injects replays of chosen path, duration, rate and
  distance into a simulated recording and scores both detectors by one rule; writes the raw
  table to `Processed-Data/` and the aggregates to
  `paper/reported/inputs/detection/detection_sweep_summary.csv` (about six minutes).
- `sweep_memoryless_controls.py`, `sweep_memoryless_heldout.py` — what each detector
  does with activity that is not one clean replay (no replay, place-less bursts,
  jumps and chains), and whether its content reproduces in held-out cells; they write
  `paper/reported/inputs/detection/detection_{controls,heldout}_summary.csv`.

## `tables/` — the event, stop and time-bin tables
- `prepare_replay_content_data.py` — build the canonical replay-content tables
  (event / event-arm / event-patch / event-sequence / stop) across all sessions.
- `regenerate_stop_table.py` — rebuild the per-stop immobility table.
- `build_timebin_tables.py` — build the per-(stop×bin) design tables.
- `prepare_contfrag_replay_content_data.py` — assemble ContFrag content tables.
- `assemble_replay_content_shards.py`, `assemble_contfrag_shards.py` — assemble
  disjoint per-animal table shards into one canonical candidate bundle.
- `refresh_contfrag_behavioral_covariates.py` — rebuild the ContFrag tables, reusing
  the immutable posterior aggregates of a preserved bundle.
- `write_regeneration_manifest.py` — validate candidate covariate bundles and write
  the release manifest.
- `convert_tables_to_parquet.py` — the one-time pickle-to-parquet migration of the
  materialized tables.

## `behavior/rate/` — event-rate models + their plots
- `fit_per_bin_rate_models.py`, `fit_per_stop_rate_models.py` — Family B / A rate ladders.
- `fit_swr_contrast_rate.py` — the `is_swr`-contrast per-bin ladder (G0–G8 / S0–S10).
- `fit_uncertainty_rate.py`, `fit_cross_history_rate.py` — MU / cross-history rate models.
- `fit_duration_aware_history_rate.py` — self- and cross-history refit on the at-risk design (bins inside an
  ongoing event dropped; history from event onsets and ends; `MODEL_CROSS_HISTORY_DURATION_RATE`), all four
  populations. `--mode session` fits each session (`models_per_bin_rate/duration_aware_history_models.pkl`);
  `--mode pooled` stacks each animal's sessions with a session fixed effect, for kernels a session is too sparse
  to measure (`duration_aware_history_pooled.pkl`). Each output has its provenance sidecar.
- `plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_model_checking.py`, `plot_uncertainty_rate.py` — rate figures.
  Each number they draw or print is a function of `analysis/behavior/` (`rate_fit_statistics`, `stop_statistics`,
  `coefficient_statistics`), read over the sessions and stops the behavior summary reads it over.

## `behavior/coupling/` — SWR-coupling strength of the events
- `fit_swr_power_model.py` — each non-local event's ripple power (`ripple_power_z`) against the behavioral covariates: how strongly an event is SWR-coupled, beside the rate (how often) and content (what) layers.

## `behavior/content/` — replay content / choice models
- `fit_replay_content_models.py` — arm/patch content hierarchy (M0–M7 / P0–P4), per `is_swr` split.
- `fit_swr_contrast_content.py` — the HMM-harmonized content `is_swr`-contrast ladder (C / CP).
- `compare_event_source_models.py` — choice (clogit) model comparison.
- `compare_content_estimators.py` — conditional Poisson (counts), conditional Poisson
  (fractions, clustered SE) and Dirichlet-multinomial fit on one shared event set,
  with equalized AIC, an animal bootstrap, and `paper/figures/content_estimators/`.
- `analyze_content_occupancy.py` — where content sits relative to the animal and the
  reward wells, per session; `--all` writes the table the events summary reads,
  `Processed-Data/content_occupancy/content_occupancy_per_session.csv`.
- `quantify_swr_overlap_geometry.py`, `generate_replay_content_figures.py`.
- `analyze_uncertainty_content.py` — trial×well reward-belief uncertainty profiles,
  absolute content rates, response sensitivities, and cohort audit tables.

## `behavior/`
- `fetch_spatial_bandit_behavior.sh` — pull the canonical SpatialBanditTask behavior CSVs.
- `build_qvals_table.py` — build the per-stop behavioral-model covariate table
  (`qvals_table`) from the shared DataJoint RL-model and trial-timing schemas.
- `run_retrospective_alignment.py` (was `pfc/run_stage5b_retrospective.py`, "Stage 5B") —
  does content during a stop match the arm visited next, against a visit-context shuffle
  (Part 3, the events relate to behavior). `at_well` only, because its metrics need a current arm.
- `run_stay_leave.py` (was `pfc/run_stage6_learning.py`, "Stage 6") — content × reward
  outcome → stay or leave (Part 3); `--stop-cohort all_stops` writes the `_allstops`
  sensitivity bundle. Output files keep their `stage5b_*` / `stage6_*` names.

## `figures/` — publication figures
- `generate_figure01.py … generate_figure04.py`, `generate_all_figures.py` (orchestrator).
  `generate_figure04.py` takes `--stop-cohort` and describes the bundle of that name:
  `at_well` (default) is the canonical population, `all_stops` reads the `_allstops`
  sensitivity bundle and refits the example neuron unrestricted.
- `generate_uncertainty_content_figure.py` — ground-up uncertainty/content story
  and diagnostic figures from the supported content-analysis artifacts.
- `generate_supp_*.py`, `generate_event_gallery.py`, `run_event_gallery.sh`.
- `plot_swr_contrast_coefs.py` — per-animal coefficient forest plots for the is_swr
  rate + content contrast ladders (standardized effect sizes) → `paper/figures/swr_contrast/`.
  Each row draws the sessions the per-term gate keeps (`contrast_aggregation.gate_term_sessions`,
  the audited table's sessions; a term the gate refuses is drawn empty, "not reported"), each
  animal's median, and the median over animals with a 95% whisker: the 2.5th and 97.5th
  percentiles of a bootstrap over the animals' medians (`coefficient_statistics.animal_median_bootstrap`).
- `generate_memoryless_comparison.py` — supplemental figure: the ported Widloski & Foster
  sliding-window detector against the state-space detector on the Figure 1 simulation, with
  the sweep summary below; every verdict in its titles and caption is read from the data,
  the labeled recording's verdicts and numbers from `memoryless_sequence/recording_statistics.py`
  (written to `paper/reported/inputs/detection/memoryless_comparison_recording.json`)
  and the sweeps' panels and checks from `memoryless_sequence/sweep_statistics.py`, the
  functions the detection summary reports from → `paper/figures/memoryless_comparison/`.
- `generate_inventory_examples.py` — seeded draws from each cell of one inventory comparison run
  (`--run-id`), each event on one time axis: spikes, the state-space posterior, the Widloski window
  posteriors (recomputed and checked against the run's stored candidates; a figure that disagrees is
  not written, and the manifest records why), the gate masks, speed, the ripple band and every inventory's events. Draws only
  `matched` or `unmatched` events, never `unknown`. → `paper/figures/inventory_comparison/<run_id>/examples/`,
  with `captions.md` and `manifest.json`. Captions are marked pending the Phase 2 pilot checks.
- `generate_event_feature_examples.py` — seeded draws of comparator-only events (each detector's events
  unmatched by `state_space`) and ours-only events (`state_space` events of the exactly-one cell) from one
  event features run (`--run-id`), in the form of `generate_inventory_examples.py`. Each event is in one of two
  descriptive subsets by its spike evidence within its population, detector and animal (high: at or above the
  group's 75th percentile; low: at or below its 25th), or in neither. The title gives the evidence, the spikes
  per row, and the `ss_disagreement` (comparator-only) or the
  rejection stage in each memoryless cohort (ours-only). A figure whose recomputed memoryless windows are
  not the stored ones is not written, and the manifest records why. →
  `paper/figures/event_features/<run_id>/examples/`, with `captions.md` and `manifest.json`. Captions are
  marked pending the Phase 2 pilot checks. Built and tested; not yet run on a real features run.
- `archive/` — retired figure scripts (not linted; kept for provenance).

## `checks/` — parity, audits, checks and doc writers (`workflows/` holds `run_refits.py`)
- `_report.py` — not a script: the run's command line (`_invocation`) and the markdown
  table (`_md`) that several QA and PFC reports share, imported as `scripts.checks._report`.
- `run_refits.py` — reruns the CPU fits for Part 3 (the events relate to behavior) and the figures that read them,
  in dependency order. Each run is pinned to its own CPUs from `--cpus` (default: all this
  process may use), and OMP/MKL/OpenBLAS/numba threads are capped per run. Runs are named
  `<question>.<analysis>[.<variant>]` (`rate.swr_contrast.at_well`); `--only
  rate,content.swr_contrast` selects by name or prefix and adds everything that depends on
  the selection; `--dry-run` lists the runs under their question. The uncited sensitivity runs
  are opt-in (`--include-optional`, or name them in full). One log and JSON record
  per run (a "blocked" record when a dependency failed) go under `logs/refits_<time>/`, then
  `check_fit_provenance.py` checks every artifact the runs wrote. Set
  `CSR_PROCESSED_DATA_DIR` to read and write another data root. `--with-tables` first
  rebuilds the tables the selected runs read, as `tables.*` runs in the order their code
  requires: `prepare_replay_content_data.py --stop-only` → `build_qvals_table.py --write` →
  `prepare_replay_content_data.py`, with `build_timebin_tables.py` and the two ContFrag
  table builds beside them. The stop-only bootstrap refuses to replace a complete stop
  table, so use it on a data root without one. It does not run the artifact and SWR
  detection or the decodes (`--dry-run` lists them in order as prerequisites, and a
  preflight checks they are on disk before a table run), the Part 1 and Part 2 checks or the PFC unit
  GLM.
- `check_data_availability.py`, `diagnose_trial_detection.py`, `compute_hpc_raw_stats.py`,
  `run_quality_metrics.py`.
- `_memoryless_reference.py` — not a script: the Widloski & Foster reference's sorted-cell
  arithmetic, re-implemented apart from the port, imported as `scripts.checks._memoryless_reference`.
- `compare_memoryless_to_reference.py` — runs the memoryless port beside
  `_memoryless_reference.py`, an independent re-implementation of the Widloski & Foster
  reference's sorted-cell arithmetic, on the Figure 1 simulation step by step and on
  every sweep cell; writes
  `paper/reported/inputs/detection/reference_check_summary.csv` (about ten
  minutes; `--n-seeds 1` for a one-minute check).
- `check_fit_provenance.py` — sweeps the whole `models*/` tree and reports which
  fitted artifacts are stale for the tables they read, from the SHA-256 sidecars
  each runner writes.
- `audit_event_patch_fraction.py`, `audit_stop_table_reward_history.py` — keyed-parity
  audits of a candidate event-patch / stop table against its pre-change reference:
  every pre-existing column must be unchanged and only the declared new columns may
  appear.
- `freeze_regression_baseline.py`, `compare_to_golden_baseline.py` — freeze the
  canonical tables and model outputs as a read-only golden baseline (with a sha256
  manifest), and diff a regenerated root against it.
- `table_parity.py` — `regen` rebuilds every canonical table (replay, time-bin and
  ContFrag) into a new scratch root with the builders and order of the 2026-10-11
  rebuild, for the whole cohort or for a few epochs (`--sessions`, by running each
  builder against a copy of the epoch register with the other epochs excluded);
  `diff` compares two roots table by table on each row key, exactly, and reports
  every table (`--restrict` cuts a full reference down to the candidate's sessions).
- `mutation_check.py` — AST mutation check on the covariate builders; the measured
  floor is recorded in `docs/design/builder_test_floor.md`.
- `summarize_cohort_audits.py` — rolls the per-run `*_cohort_audit.csv` files into
  one table.
- `write_pipeline_map.py` — regenerates `docs/regression_pipeline_map.md`, the
  symbol-anchored map of the modules carrying a decoded session to a fitted
  regression, read out of the source with `ast`; `--check` fails on drift, as
  `tests/scripts/checks/test_pipeline_map_is_current.py` does.
- `summarize_stage6_fdr.py` — prints Stage 6's per-role cross-session FDR table for
  any saved `stage6_is_stay_logistic*.pkl`, using the runner's own statistic, so a
  before/after comparison across populations reads one implementation.

## `pfc/` — PFC responses to the events (Part 4, the events relate to mPFC) and their audits
- `run_pfc_analysis_bandit.py` takes `--stop-cohort`: the well-anchored stop cohort the
  hippocampal rate and content ladders are read on is the default (`at_well`, canonical
  artifact names), and `all_stops` publishes an unrestricted sensitivity bundle under
  `_allstops` names. It fits the unit GLMs under a Gaussian prior on the
  event-basis coefficients and scores them out of fold: `--penalty-sigma` (the prior
  standard deviation on the log-rate scale; another value writes a `_sigma<value>`
  sensitivity bundle and draws no null), `--n-folds` (blocked over whole immobility
  runs, with runs whose ±250 ms kernel supports overlap joined into one block before
  assignment), `--n-control-draws` with `--fold-seed` / `--control-draw-seed` (derived
  from the session key unless given, so a lone refit reproduces the batch's split and
  null). The canonical batch is **σ = 3**, **K = 5** and **R = 1,000** — a plus-one
  p-value floor of 1/1,001 — and every requested draw must be usable for a unit's
  p-value to be reported; the 20-draw floor is for deliberately shortened
  diagnostics. It takes `--gpu N` / `--allow-cpu` to say which card it may take on a shared
  host. A batch runs as one process per card — `--n_workers 1 --gpu N --sessions
  <partition file>` — because one card handed to several workers is refused, and the
  same card handed twice to one process is ignored.
- `run_pfc_marked_kernels.py` fits the content-marked extension of those unit GLMs
  (`docs/design/pfc_marked_event_kernels.md`) on the same session assembly (fit mask,
  analysed-time rule, canonical event sets): per replay class an ordinary kernel, a
  log-duration nuisance kernel and a content kernel, all jointly, scored out of fold
  against a mark-label shuffle. It takes `--scientific-mark`
  (`current_arm_fraction`, the default, or `current_arm_fraction_posterior`),
  `--nuisance-marks` (log event duration by default; pass the flag with no values to
  drop it), `--n-shuffles` with `--shuffle-seed` (default 100, giving 1/101 tail
  resolution — 0 skips the reference, and any other value must be at least the 20-draw
  floor), `--design-batch-size` for how many permuted designs are scored at once, and
  `--event-arm-table` for the table the mark is read from. It shares
  `--penalty-sigma`, `--n-folds`, `--fold-seed`, `--brain-area`, the session
  selectors (`--animal` / `--date` / `--epoch` / `--sessions` / `--max-sessions`, over
  the epoch register's analysed epochs), `--gpu` / `--allow-cpu`, `--output-dir`
  (default `pfc_marked_kernels/` under the processed-data root) and `--overwrite`
  with the canonical runner. It has no `--stop-cohort`: it always fits the
  well-anchored cohort. `--no-baseline` drops the per-stop well and
  reward-outcome baseline columns (bundle contract 2 carries them by default).
  **No bundle has been produced yet.**
- `run_pfc_marked_model_family.py` fits the recommended descriptive extension
  without decoding mPFC activity or computing p-values. Its nine default core
  families jointly characterize spatial role, content-weighted uncertainty and
  value, reward memory/recency, trajectory geometry, within-stop timing,
  switch/reward context, and graded ripple/MUA/theta coupling. Pass
  `--include-interactions` for the three current-arm/uncertainty interactions
  with exploration, reward, and ripple power, or select explicit `--models`.
  Every target is compared out of fold with the otherwise identical joint model
  lacking only that target's temporal block. Session loading and folds are
  shared across models; the six compact core models share the 155-column tier,
  while the three content horse-races and three interactions share the 221-column
  tier for JAX compilation reuse. Wide-design microbatches are bounded by
  the measured single-mark memory workload. It supports the same session/GPU,
  penalty, fold, output, overwrite, and event-arm-table options as the
  single-mark runner plus `--event-table`, `--mark-penalty-sigma` and
  `--min-marked-events` (default 10 per class), and `--no-baseline` (drop the
  per-stop well and reward-outcome baseline columns). `--model-set followup`
  fits the follow-up models instead (replay content, event strength, stop
  context, ripple power per class and pooled; `docs/design/pfc_marked_model_family.md`),
  with `--tier {joint,marginal,all}` for their generated single-target marginal
  tier, `--event-set with_swr_without_replay` to add the SWRs without replay to
  the ripple-power models, and `--slope-permutations` (default 30 for the
  follow-up set, 0 otherwise) with `--slope-permutation-seed` (a base; each
  model's seed also depends on its name) for the residualized shuffled-mark
  reference of every slope kernel, drawn after the observed fit passes its
  convergence gate. A class a session cannot carry for a model is dropped from
  that model and named in the sidecar. The follow-up set requires an
  explicit `--output-dir` (use `Processed-Data/pfc_response_followups/marked_family`).
  Recommended-set output goes to `pfc_marked_model_family/` under the
  processed-data root; the bundles there are the 71-session corpus of
  2026-09-19 (`.claude/docs/reviews/2026-09-19-pfc-marked-family-corpus.md`), fit
  before the analysed-time rule under bundle contract 1 -- a default-directory
  rerun under contract 2 removes them as stale, so do not point a run there.
- `summarize_pfc_marked_family.py` aggregates those bundles unit → session →
  animal (fraction of units with a positive held-out gain, population mean
  slope kernels with pre- and post-onset means, all animal-balanced; when the
  bundles carry nulls, the max-statistic call against the shuffled-mark
  reference with its plus-one p, and the per-lag 95 % band, each joint target
  beside its marginal fit, one figure per model); no per-unit p-values. The
  statistics are `analysis/pfc/marked_family_statistics.py`'s. It
  refuses a directory that mixes bundle contracts, event sets or model sets. `--bundle-dir`,
  `--out <markdown>`, `--figure-dir`, `--blowup` (default 1000),
  `--summary-parquet` (default `<bundle-dir>/summary.parquet`, one row per
  model, tier, class, target and statistic).
- `run_pfc_stop_context.py` — does the mPFC response to replay change with the
  stop's context (Part 4)? `assemble` builds, per session with a canonical mPFC
  bundle, the unit × replay-event table (spikes in `[0, +200) ms` after onset,
  the stop's event-free fitted baseline, `is_rewarded_outcome`,
  `next_trip_switches_patch`, overlapping events) on the Figure 4 session
  assembly and caches it under
  `Processed-Data/pfc_response_followups/stop_context/sessions/` (`--workers`
  for parallel sessions, CPU only). `fit` pools all sessions per class × response
  sign × unit set (called / all) and writes `results.csv`: Poisson model with
  per-unit intercepts, the stop baseline estimated jointly (conditional) or
  plugged in, session-clustered CIs, per-animal and leave-one-animal-out fits
  (`analysis/pfc/stop_context.py`, whose docstring states the model and the two
  estimators).
- `compare_pfc_cohorts.py` — writes the mPFC unit-GLM before/after audit: per-session
  bins and events on either side of the stop cohort (from the bundles' own provenance
  sidecars), the decode/position alignment trim, the leading-edge coverage diagnostic,
  and the Figure 4 population loaded from both bundle trees. When the earlier tree
  also carries sidecars it audits a **rerun** as well: whether the two trees describe
  the same events, the unit-by-unit differences against the estimator's gate
  tolerances (per quantity and per session), the matched non-event control's tests,
  the estimator/device/fit counts per card and per session, and — given
  `--worker-time <batch>/worker*.time` — the wall clock each worker of the batch
  cost. When the after tree's sidecars name the penalised estimator it writes a
  different report instead (`--comparison` forces either): what the prior did to each
  unit's peak modulation against the same block fit unpenalised, split by whether that
  fit ran away; how many units carried a basis column with no spikes under it; what
  the two inferences call, per class and animal; how steady each nested pair's
  held-out difference is between folds; and the peaks' sensitivity to the prior,
  against whatever `_sigma<value>` bundles are on disk (the report names how many
  sessions those are). `--geometry <report.md>` folds in the null-geometry
  measurements beside the p-values they qualify.
- `measure_pfc_null_geometry.py` — measures what the mPFC held-out null's windows
  look like beside the real events, and changes nothing: the inter-onset spacing of
  the shipped matched control and of the null's own draws (rebuilt draw by draw from
  their recorded seeds, for the sessions named on the command line) against the real
  classes'; how much of the engagement null's draws land on the shipped control they
  are entered beside; how many units called at raw p < .05 have no positive held-out
  improvement of their own; and how many events of each class had no legal within-bout
  control and so entered neither side of its test. It and `compare_pfc_cohorts.py`
  load the population at the summary's firing-rate cutoff
  (`analysis/pfc/load_population.py`'s `MIN_FIRING_RATE_HZ`, none) unless
  `--min-firing-rate` gives another.
- `measure_pfc_mark_context.py` — measures what the marked kernels' content mark
  (`current_arm_fraction`) is entangled with, session by session and per replay
  class, over the well-anchored events of the canonical `event_arm_table`: its
  correlation with log event duration and with session time, and the share of its
  variance the occupied arm explains. Reported for the reader of the label-shuffle
  reference, not acted on. `--event-arm-table`, `--out <markdown>`.
- `calibrate_pfc_kernel_timing.py` — how far the shipped kernel estimator (basis,
  σ = 3 prior, fitter) moves the onset and peak of a response of known start: plants
  boxcar and alpha responses in simulated Poisson units, fits them as `run_session`
  does, and writes `Processed-Data/pfc_response_followups/timing_calibration.json`,
  which Part 4's timing values read. CPU only (`JAX_PLATFORMS=cpu`); `--quick` for a
  two-scenario check.
- `part4_rate_threshold_sensitivity.py` — recomputes Part 4's per-unit mPFC values
  (called fractions, overlap, split, effect sizes, SWRs without replay) with
  `reported/mpfc.py`'s own functions at firing-rate gates of none (> 0 Hz on the fitted
  bins), 0.1, 0.5 and 1 Hz — the bundles hold every unit, so no refit — with the
  loader's drop counts per gate and the called fractions per rate band. Writes
  `part4_rate_threshold_sensitivity{,_long}.csv` under
  `.claude/research/pfc_response_followups_outputs/` (`--out-dir`).
- `diagnose_pfc_event_definition.py` — reports, per session, what the mPFC unit GLM's
  retired run-time event construction yielded against the canonical event table it now
  reads: the two event counts, how many events change SWR class once the lists are
  matched by overlap, the start-time shifts, and how many reclassifications are
  partial-overlap cases the two split rules may legitimately disagree on.
- `check_pfc_ridge_lambda0.py` — writes the audit that the penalised solver with
  its penalty set to zero is still the maximum-likelihood one: it rebuilds each
  gate session's design *as it stood when the reference was fit* (the basis that
  dropped only patsy's first boundary function) from that session's own fit mask,
  event sets and seeded control draw, fits it unpenalised, and compares against
  the stored statsmodels diagnostics. Cells whose maximum-likelihood estimate
  does not exist are reported rather than gated — they are the ones the prior was
  introduced for.
- `compare_pfc_fitters.py` — **history.** Wrote the equivalence audit of the batched
  JAX Poisson fitter against the per-neuron statsmodels IRLS, from two runs of the
  same sessions (`--fitter jax` / `--fitter statsmodels`, both with
  `--fit-diagnostics`): every numeric column of the shipped statistics, the five
  modulation time courses, the best-model and NaN patterns that must be identical, the
  per-model parameters and fit criteria, and the wall clock of both runs. The unit
  runner no longer offers the unpenalised per-neuron backend, so this compares two
  archived trees rather than two runs anyone can make now.

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

## Moved scripts (2026-11)

`rate/`, `content/` and `coupling/` moved under `behavior/`, and `qa/` split
into `workflows/` (runners that run other scripts) and `checks/` (audits,
parity checks and doc writers). Nothing is left at the old paths.

| Old path | New path |
|---|---|
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

## Moved scripts (2026-10)

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
