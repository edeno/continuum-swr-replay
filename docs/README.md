# Documentation index

Every documentation artifact has **one home by type**. This index states the scheme so
it stops drifting; add new docs to the matching home.

## Where each doc TYPE lives

| type | home | what it is |
|---|---|---|
| catalog | `docs/models_table.md` | every regression model in R notation (mirrors `model_specs.py`) |
| definitions | `docs/covariate_dictionary.md`, `docs/behavior_model_covariates.md` | canonical covariate definitions; the `qvals_table` data dictionary |
| reference (models) | `docs/behavior_models.md` | the SpatialBanditTask generative models (Beta-Bernoulli, HMM, Q-learners) that produce the behavioral covariates |
| methods and results | `docs/rate_models.md` | how the two event-rate model families work, and what their fits show: the values it quotes are generated markers filled from `paper/reported/` |
| results | `paper/reported/questions.md` (generated) | every reported value, under the question it answers, in the paper's outline |
| reference | `docs/task_description.md`, `docs/cached_trials_for_testing.md`, `docs/known_issues/` | task/data background |
| cohort | `docs/epoch_register.md` | which run epochs are analysed, and the reason for every other one (mirrors `data_loaders/epoch_register.py`) |
| **design spec** (scientific/methodological) | **`docs/design/`** | analysis designs — human-discoverable for collaborators/reviewers |
| implementation plan (agent process) | `.claude/docs/plans/` | phased execution plans |
| review (code/methods output) | `.claude/docs/reviews/` | point-in-time review reports, and the dated audits (before/after refit comparisons, regeneration and parity checks) |
| exploratory notes + figures | `.claude/research/` | scratch analyses; `*_outputs/` figures |
| code map (generated) | `docs/regression_pipeline_map.md` | every pipeline module and the public symbols it defines, by stage |
| model spec (code) | `src/continuum_swr_replay/analysis/replay/model_specs.py` | single source of truth for all model ladders |

Rule of thumb: **published/scientific → `docs/`; agent-process/scratch → `.claude/`.**

## `docs/` contents

- [../paper/reported/questions.md](../paper/reported/questions.md) — *start here*: every question of
  the paper, by part, with the values that answer it (generated from
  `src/continuum_swr_replay/reported/questions.py` and the summaries).
- [reproduce.md](reproduce.md) — *how to run*: what each level of rebuild needs, the full
  rebuild in order (from the cohort to the PDF), the decoders, and what is committed.
- [figure-pipeline.md](figure-pipeline.md) — *figures*: each figure's script, inputs,
  output and tests.
- [data-lineage.md](data-lineage.md) — *data*: where each stage reads and writes, from the NWB
  files to the summaries.
- [development.md](development.md) — *development guide*: the task and data model, the
  code layout, the analysis conventions and the reported-numbers rule. `CLAUDE.md`
  imports it.
- [models_table.md](models_table.md) — *catalog*: every model in R formula notation.
- [covariate_dictionary.md](covariate_dictionary.md) — *definitions*: covariate semantics across families.
- [behavior_model_covariates.md](behavior_model_covariates.md) — *definitions*: `qvals_table.parquet` data dictionary.
- [behavior_models.md](behavior_models.md) — *reference*: the generative behavioral models (Beta-Bernoulli, HMM, Q-learners) whose latent variables the above columns hold.
- [rate_models.md](rate_models.md) — *methods*: the two event-rate families (A per-stop, B per-bin).
- [task_description.md](task_description.md) — *reference*: behavioral task / recording setup.
- [epoch_register.md](epoch_register.md) — *cohort*: the exclusion rule, its reason
  codes, and every excluded or pending epoch with its evidence; every stage reads the
  register (`data_loaders/epoch_register.py`), and the audit checks it.
- [cached_trials_for_testing.md](cached_trials_for_testing.md) — *reference*: test-fixture trial cache.
- [regression_pipeline_map.md](regression_pipeline_map.md) — *code map*: every module
  from decoded session to fitted regression, grouped by stage, with the public symbols
  each defines. Generated — regenerate with `scripts/qa/write_pipeline_map.py`.
- [design/](design/) — *design specs*: per-analysis design documents.
  - [design/model_history.md](design/model_history.md) — *design journal*: why each
    replay-content model rung carries the predictors it does (reversed reasoning
    and retracted numbers included). The machine-readable catalog is
    `analysis/replay/model_specs.py`; this is the record of how it got that way.
  - [design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md) —
    *design spec*: the covariate causal-hygiene rules every fit runner enforces
    and the `<run>_cohort_audit.csv` schema (drop buckets + reconciliation
    invariant).
  - [design/builder_test_floor.md](design/builder_test_floor.md) — *reference*: the
    measured value-level test coverage of the covariate builders and the target,
    kept honest by `scripts/qa/mutation_check.py`.
  - [design/2026-05-27-figure01-revision-design.md](design/2026-05-27-figure01-revision-design.md)
    — the Figure 1 revision: the No-Spike-wins scenario, spikes shown entering the model,
    and the corrected transition labels.
  - [design/2026-06-23-replay-trajectory-metrics-design.md](design/2026-06-23-replay-trajectory-metrics-design.md)
    — per-event replay trajectory metrics, defined identically for the non-local and
    ContFrag events; pending an implementation plan.
  - [design/2026-06-25-non-swr-event-reality-design.md](design/2026-06-25-non-swr-event-reality-design.md)
    — the spike-evidence test that non-SWR non-local events are real, from the likelihood
    alone with the prior removed.
  - [design/2026-06-27-state-space-point-process-event-model.md](design/2026-06-27-state-space-point-process-event-model.md)
    — a state-space point-process GLM (Czanner et al. 2008) of the event onset train,
    extending Family B; not implemented.
  - [design/2026-07-13-content-rate-uncertainty-model-ladders.md](design/2026-07-13-content-rate-uncertainty-model-ladders.md)
    — proposed Aim-1 content-allocation, replay-occurrence, event-size, and
    marginal-content ladders in R formula notation.
  - [design/2026-08-21-covariate-naming-scheme.md](design/2026-08-21-covariate-naming-scheme.md)
    — proposed predictor-column names across the four canonical tables,
    code-verified; also flags the duplicate and dead columns the pass found.
  - [design/2026-10-03-full-pipeline-refresh-design.md](design/2026-10-03-full-pipeline-refresh-design.md)
    — the October full re-decode, comparison inventories and downstream regeneration; done
    and audited 2026-10-12. `scripts/detection/run_swr_detection.py` cites it as its
    `DESIGN_SPEC`.
  - [design/2026-10-04-decoder-transition-prior.md](design/2026-10-04-decoder-transition-prior.md)
    — decision R10: the decoders' discrete-transition prior (setting P30-B), its sources
    and sensitivity; the values live in `decoding/transitions.py`.
  - [design/2026-10-09-swr-source.md](design/2026-10-09-swr-source.md) — the paper's
    one SWR source: the direct ripple_detection 2.0 Kay inventory, its comparison with
    the stored `RippleTimesV1` events, the interface every SWR consumer reads, and the
    artifact hook.
  - [design/2026-10-09-artifact-detection.md](design/2026-10-09-artifact-detection.md) —
    the LFP and spike artifact detector: its four signatures (S, T-ref, T-ca1, T-spk),
    thresholds placed by one rule with their evidence, the one mask and its padding,
    validation against rule-defined and model-reviewed SWR sets, field precedents, the
    Spyglass comparison, and known limits.
  - [design/2026-10-09-swr-tetrode-selection.md](design/2026-10-09-swr-tetrode-selection.md)
    — rule R-T, the per-epoch choice of the CA1 tetrodes all four SWR detectors read: the
    events at the ports (study C1), the j16 tetrodes that do not ripple with CA1, the rule
    with its effect and blind checks, why the reference veto and the field's noise-channel
    form are not used, the visit-onset numbers, and the limits.
  - [design/2026-11-10-questions-and-numbers-design.md](design/2026-11-10-questions-and-numbers-design.md)
    — one question list with each question's numbers, generated from the
    summaries; interpretation moved into the manuscript's Results; the documents it
    retires and where each section goes; the inventory's findings and the lead's
    decisions.
  - [design/content_model_design.md](design/content_model_design.md) — the `is_swr`
    content ladder, parallel to the per-bin rate ladder, on the beta-Bernoulli covariates;
    implemented in `scripts/content/fit_swr_contrast_content.py`.
  - [design/content_estimator_comparison.md](design/content_estimator_comparison.md)
    — the retired count multinomial vs the shipped per-event fractions vs a
    Dirichlet-multinomial, on one 218-session cohort: what each assumes,
    simulation recovery, which conclusions are estimator-dependent, and the
    `log_phi_is_swr` concentration result. Its values answer questions 3.1, 3.6 and 3.7.
  - [design/pfc_marked_event_kernels.md](design/pfc_marked_event_kernels.md)
    — the content-marked mPFC replay kernels: the `current_arm_fraction` mark
    and its within-session/class scaling, the jointly fit ordinary + duration +
    content kernels, the nested held-out comparison per replay class, and why
    the mark-label shuffle is a descriptive reference rather than a calibrated
    conditional test.
  - [design/pfc_marked_model_family.md](design/pfc_marked_model_family.md)
    — the recommended no-mPFC-decoder extension: nine core marked temporal
    models for spatial role, uncertainty/value/reward memory, trajectory
    geometry, within-stop timing, switch/reward context and graded coupling,
    plus three optional content interactions; exact leaf-only aggregation,
    adjustment sets, missingness, held-out comparison and GPU-memory contracts.
  - [design/replay_detection_comparison/](design/replay_detection_comparison/) —
    *design spec set*: running several published replay-detection methods on the same
    recordings and comparing which events each finds. Start at
    [overview.md](design/replay_detection_comparison/overview.md), which carries the
    governing rule (harmonize substrate, never method internals), the seven detection
    inventories, the clusterless scope boundary, and the cross-cutting decisions the
    methods section will draw on. Component specs:
    [01](design/replay_detection_comparison/01-yu-swr-detector.md) SWR detection with a
    noise-percentile threshold (Yu et al. 2017);
    [02](design/replay_detection_comparison/02-memoryless-sequence.md) the Widloski &
    Foster (2025) sliding-window sequence detector, including the clusterless
    encoding-pair shuffle;
    [03](design/replay_detection_comparison/03-inventory-comparison.md) the N-way overlap
    accounting and the cohort × SWR-definition matrix;
    [04](design/replay_detection_comparison/04-event-features.md) symmetric event
    characterization and selection diagnostics at default settings, with threshold
    robustness deferred to a later cost pilot and limited sweeps;
    [05](design/replay_detection_comparison/05-validation.md) the parked validation
    studies needed for performance claims. Broad refits and robustness studies do not
    block the initial descriptive report.
    [memoryless_simulation_controls.md](design/replay_detection_comparison/memoryless_simulation_controls.md)
    extends the simulated comparison of spec 02 with timing, false positives, multi-part
    events and a held-out evidence test.
    [detector_comparison.md](design/replay_detection_comparison/detector_comparison.md)
    tabulates every detection method considered, with a verified DOI for every work
    cited;
    the lab's <!--v:survey.papers-->57<!--/v-->-paper survey of detection and decoding settings is
    [../paper/tables/table_s1_literature_parameters.csv](../paper/tables/table_s1_literature_parameters.csv)
    (Supplementary Table S1; the paper's one copy, audited against each paper's methods,
    supplement and released code), and
    [literature_parameter_corrections.md](design/replay_detection_comparison/literature_parameter_corrections.md)
    records the first round of corrections (2026-09-21). Reference source vendored at
    [reference/widloski_foster_2025/](../reference/widloski_foster_2025/) and
    [reference/yu_2017_franklab/](../reference/yu_2017_franklab/).
    Status: the memoryless sequence detector of spec 02, its significance shuffle,
    its per-epoch runner, its simulated-replay sweep and supplemental figure are
    implemented and tested (`analysis/hippocampus/memoryless_sequence/`,
    `scripts/detection/run_memoryless_detection.py`); the inventory comparison's
    accounting of spec 03 is implemented and tested on synthetic inventories
    (`analysis/inventory_comparison/`). Not yet done: the real-data pilot, the
    comparison's per-detector adapters and runner, the Yu detector's project
    wiring, and specs 04–05.
- [known_issues/](known_issues/) — *reference*: documented data/pipeline issues.
  - [reported_statistics_choices.md](known_issues/reported_statistics_choices.md) —
    *open*: choices the reported statistics make that have not been decided (which
    function, what it does, what is open); write each decision there.
  - [2026-10-12-full-redecode-r11.md](known_issues/2026-10-12-full-redecode-r11.md) — the
    October full re-decode (R11): what changed, the backups, and what stays stale until rerun.
  - [three_sessions_blocked_by_nwb_checksum.md](known_issues/three_sessions_blocked_by_nwb_checksum.md)
    — *resolved 2026-08-19*: three sessions once failed an upstream NWB checksum;
    their files were rewritten. The cohort is now the register's 242 analysed epochs.
  - [wilbur20210404_missing_ca1_ripples.md](known_issues/wilbur20210404_missing_ca1_ripples.md)
    — wilbur20210404 has no CA1 ripple extraction, but it is a depletion session,
    never in the cohort; its register entry is `DEPLETION_TASK`.
  - [swr_mua_count_conventions.md](known_issues/swr_mua_count_conventions.md) — don't
    cross-substitute the stop-level vs bin-level `n_swr_events`/`n_mua_events` count.
  - [current_arm_trunk_fallback.md](known_issues/current_arm_trunk_fallback.md) —
    historical diagnosis of the destination fallback; fixed by explicit
    behavioral-location labels and past-only context (regeneration pending).
  - [contains_path_length_confound.md](known_issues/contains_path_length_confound.md) —
    `contains_path_to_next` is 2.8× harder for cross-patch than within-patch trips
    by construction; don't pool across trip types.
  - [degenerate_mua_events.md](known_issues/degenerate_mua_events.md) — *open*: a few
    ContFrag `MUA_event` labels span several seconds (up to ~23 s) across a stop, run and
    stop; worked around, upstream cause not found.
  - [nonlocal_minimum_duration_rounding.md](known_issues/nonlocal_minimum_duration_rounding.md) —
    the non-local 15 ms minimum is really 7 samples (12 ms span), 1.1 % of events;
    fix at the next full rebuild.

## Code layout

`src/continuum_swr_replay/` is layered. Code may import from its own layer and the
layers below it, never from a layer above:

1. **Foundation**: `data_loaders/`, `data_types.py`, `decoding/`, `intervals.py`,
   `parameters.py`, `paths.py`, `runs/`, `tables/`. A foundation module imports only
   other foundation modules.
2. **`analysis/events/`**, the analysed-time rule and the event substrate. It imports
   only the foundation, `analysis/constants.py` (which itself imports only the
   foundation), `analysis/shared_statistics.py` and itself. `analysis/shared_statistics.py` holds
   the statistics that know nothing of the task, for every `analysis/` package: the
   signed-rank test, the Benjamini-Hochberg correction and the animal-balanced
   summary over sessions. It imports nothing from the package.
3. **The other `analysis/` packages** (`hippocampus/`, `replay/`, `pfc/`, `behavior/`,
   `contfrag/`, `inventory_comparison/`). `hippocampus/` never imports `pfc/` or
   `inventory_comparison/`. No `analysis/` code imports `visualization/`.
4. **`visualization/`** (with `figure_utilities.py`) and **`reported/`** (the summary
   builders), side by side, then **`scripts/`** on top. `reported/` imports the
   foundation and `analysis/` (it names and describes the statistics that `analysis/`
   computes); nothing in `analysis/` or `visualization/` imports `reported/`, so the
   analysis does not depend on the reporting. Nothing in `src/` imports `scripts/`.

There are no package cycles. `tests/unit/test_import_layering.py` enforces all of
this except the rules for `reported/`, which hold today but are not tested (it also keeps `data_loaders/` from importing `analysis/`, `hippocampus/` from
importing `replay/`, and the covariate builders from importing `model_specs.py`). It
has no allowlist: any violation fails.

### Moved modules (2026-10)

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
(the detection runners' shared inputs), `analysis/replay/fit_outputs.py` (the content
fitter's output naming), and functions added to `analysis/replay/learning.py` (Stage 6's
role summary) and `analysis/replay/contrast_aggregation.py` (the contrast-content cohort
assembly). Script moves are listed in [scripts/README.md](../scripts/README.md)
("Moved scripts (2026-10)").

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

## Figure outputs

- **`paper/figures/<topic>/`** — publication figures only, in topic subdirs (main-text
  `figure0N/`, `supp_*`, and analysis topics like `rate/`, `reality/`, `quality_metrics/`).
  No loose files at the top level. Generated by `scripts/figures/*` and the `scripts/*/plot_*.py`.
  Scripts spell the directory as `continuum_swr_replay.paths.figure_dir(topic)`, which sits
  beside the data root, so a run on another root (`CSR_PROCESSED_DATA_DIR`, e.g. a
  `run_refits` staging run) writes its figures there, not over the paper's.
- **`.claude/research/*_outputs/`** — exploratory / scratch figures (from `scripts/exploratory/_*.py`),
  not results-of-record.

## Where results live

Three tiers:

1. **Result DATA** (`.parquet`/`.pkl`/`.nc`, gitignored) under `Processed-Data/replay_content/`:

   The canonical tables (events, event-arms, event-patches, sequences, stops) are
   built by `scripts/tables/prepare_replay_content_data.py`, and `qvals_table` by
   `scripts/behavior/build_qvals_table.py`. All of them sit at the root, beside
   their standardization references (`standardization_reference.json`,
   `qvals_standardization_reference.json`). They
   are **parquet**; read and write them through `continuum_swr_replay.tables.io`
   (`read_table` / `write_table`), never `pd.read_pickle`; `read_table`,
   `read_attrs` and `table_exists` refuse a `.pkl` path. Fitted-model artifacts
   (`all_models.pkl` and friends) hold live statsmodels objects and stay pickled.
   Superseded table pickles are kept under `_pickle_archive/`; nothing reads them.
   Code spells these locations through `continuum_swr_replay.paths`
   (`replay_content_dir()`, `model_dir(key, root)`, `table_path(name, root)`), not inline.

   | dir | producer | holds |
   |---|---|---|
   | `models_per_bin_rate/` | `scripts/rate/fit_per_bin_rate_models.py`, `fit_swr_contrast_rate.py`, `fit_uncertainty_rate.py`, `fit_cross_history_rate.py` | the per-bin rate fits: `all_models.pkl` (Family-B ladder), `swr_contrast_rate.pkl`, `uncertainty_rate.pkl`, `cross_history_models.pkl`; plus three exploratory one-offs nothing supported reads (`contingency_table.pkl`, `contingency_csv_alpha.json`, `covariate_ranking.pkl`) |
   | `models_per_event_swr_power/` | `scripts/coupling/fit_swr_power_model.py` | the per-event ripple-power model `swr_power_model.pkl`, its `.z_reference_by_session.json` and its two audits |
   | `models_per_stop_rate/` | `scripts/rate/fit_per_stop_rate_models.py` | Family-A per-stop ladder |
   | `models/`, `models_patch/` | `scripts/content/fit_replay_content_models.py` (arm / patch) | the default content ladders — arm `M0`, `M0a`, `M1`–`M6`; patch `P0`–`P3` — on the at-well events, plus the opt-in value/uncertainty rungs (`Mv`, `Mu`, `Mvu`, `Pv`) when named with `--models` |
   | `models_swr_{T,F}/`, `models_patch_swr_{T,F}/` | same, `--swr-subset T\|F` | content split by SWR (T) / non-SWR (F) |
   | `models_dm/` | same, `--estimator dirichlet_multinomial` | the M0–M6 arm ladder refit as a Dirichlet-multinomial (count response + fitted within-event concentration); the matching contrast bundles `swr_contrast_content_{arm,patch,audited}_dm{,_rewarded,_omitted,_len}.pkl` sit in `models_swr_contrast/` beside their conditional twins |
   | `content_estimator_comparison/` | `scripts/content/compare_content_estimators.py` | the three-estimator comparison on one shared cohort — `content_estimator_comparison.parquet`, `_summary.csv`, `content_estimator_bootstrap.csv`, `_skips.csv`. Read with [design/content_estimator_comparison.md](design/content_estimator_comparison.md) |

   Since 2026-10-23 `qvals_table` and its reference sit at the root and the power model in
   its own folder; nothing is left at the old `models_per_bin_rate/` paths (the move is the
   "Phase D disk move" entry of
   [the code-organization plan](../.claude/docs/plans/code-organization/PLAN.md)).

   The tables are built from the upstream runs below. Each location is a
   `continuum_swr_replay.paths` function, resolved when called under the data root
   `paths.processed_data_dir()`: `$CSR_PROCESSED_DATA_DIR` when set, else the
   checkout's `Processed-Data/`. That variable is the one place a run or a test
   moves every root; no library module or supported script binds one at import. (The old upper-case names,
   `parameters.PROCESSED_DATA_DIR` and `paths.REPLAY_CONTENT_DIR` and the rest,
   still resolve for notebooks and `scripts/exploratory/`; assigning one moves
   nothing, so to redirect a run set `$CSR_PROCESSED_DATA_DIR`.) A run folder's
   `manifest.json` names its inputs and settings.

   | location | `paths.` | producer | holds |
   |---|---|---|---|
   | `Processed-Data/` (root) | `processed_data_dir()` | `scripts/decoding/run_decoding_bandit.py`, `run_decoding_contfrag.py` | the per-epoch decodes: `*_results.nc`, `*_detector.pkl`, `*_environment.pkl`, `*_data_over_time.pkl` (file names: `decoding/files.py:decode_paths`) |
   | `artifact_intervals/<run_id>/` | `artifact_intervals_root()` | `scripts/detection/run_artifact_detection.py` | the artifact intervals each epoch's excluded time is built from |
   | `replay_detection_comparison/swr/<run_id>/` | `swr_inventory_root()` | `scripts/detection/run_swr_detection.py` | the SWR inventories (Kay and the other detectors) |
   | `replay_detection_comparison/memoryless/<run_id>/` | `memoryless_root()` | `scripts/detection/run_memoryless_detection.py` | the memoryless sequence detector's events |
   | `replay_detection_comparison/carey/<run_id>/` | `carey_root()` | `scripts/detection/run_carey_detection.py` | the Carey detector's events |
   | `replay_detection_comparison/inventory/<run_id>/` | `inventory_comparison_root()` | `scripts/detection/run_inventory_comparison.py` | the detectors' inventories compared |
   | `replay_detection_comparison/event_features/<run_id>/` | `event_features_root()` | `scripts/detection/run_event_features.py` | per-event features for the comparison |
   | `contfrag_replay_content/` | `contfrag_replay_content_dir()` | `scripts/tables/prepare_contfrag_replay_content_data.py` | the replay-content tables built from the Continuous-Fragmented decodes |

   Both rate families are fit on the **well-anchored stop cohort** — the same
   population the content family is fit on at the event grain — and an
   unsuffixed rate artifact is that cohort. The sensitivity bundles carry the
   cohort in the filename: `*_allstops` (every immobility stop) and
   `*_intransit` (in-transit stops only, thin — 50 of 218 sessions). Every
   fitted row and provenance sidecar records its `stop_cohort`. See
   [design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md) and
   [2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md).

   (These dir names are terse — a descriptive rename and un-mix of the content directories
   has been considered and deferred; the names above are the ones the runners write.)

   Each **primary** fit runner (content base/contrast, rate contrast, Family A/B) also
   writes a **`<run>_cohort_audit.csv`** (one row per model × population; retained-N and
   mutually-exclusive drop buckets, reconciling `n_in − Σdrops == n_modeled`; coverage
   boundary in [models_table.md](models_table.md#cohort-audit-tables)) and the contrast
   runners a `<run>_term_diagnostics.csv`
   (per-term quasi-separation counts). `scripts/qa/summarize_cohort_audits.py` stacks and
   flags them; the schema is documented in [models_table.md](models_table.md#cohort-audit-tables).

   The **PFC** unit bundles and the two behaviour analyses (retrospective alignment and
   stay/leave) write outside that table — the unit bundles at the
   `Processed-Data/` root, the two behaviour analyses in `replay_content/` beside the
   hippocampal fits:

   | artifact | producer | holds |
   |---|---|---|
   | `Processed-Data/{epoch_identifier}_mPFC_modulation.pkl` + five `*_modulation.npy` | `scripts/pfc/run_pfc_analysis_bandit.py` | per-unit penalised-GLM statistics — held-out deviance and separated-column count per model, each nested pair's held-out difference (total and per fold), an empirical p-value per event class and for the engagement comparison, the best model by held-out deviance, and per curve the penalised and unpenalised peak with that peak's own standard error — plus the five peri-event modulation time courses (`SWR`, `no_SWR`, `non_local_SWR`, `non_local_no_SWR`, and `matched_control`, the duration-matched non-event baseline). All five are read into the Figure 4 population by `analysis/pfc/load_population.py`; the first four are NaN-free by its drop rule, while the matched-control curve may carry NaN and is reduced with `nanmean` |
   | `replay_content/stop_replay_summary.pkl`, `stage6_is_stay_{logistic,pooled_gee}{,_swr,_non_swr,_mua}.pkl` | `scripts/behavior/run_stay_leave.py` | the per-stop replay summary and the stay/leave models (question 3.5) |
   | `replay_content/stage5b_retrospective{,_swr_only,_non_swr_only}.pkl` | `scripts/behavior/run_retrospective_alignment.py` | the retrospective/prospective event-grain metrics against their visit-context shuffle |
   | `pfc_marked_kernels/{epoch_identifier}_{mark}_marked_kernels.pkl` + `.lag_time.npy`, `.{class}.typical.npy`, `.{class}.mark_slope.npy`, `.mark_label_shuffle_reference.npz`, `.provenance.json` | `scripts/pfc/run_pfc_marked_kernels.py` | the content-marked kernels of [design/pfc_marked_event_kernels.md](design/pfc_marked_event_kernels.md). One bundle per session × mark (`current_arm_fraction`, or `current_arm_fraction_posterior` as a sensitivity): the pickle holds the per-neuron statistics (held-out deviance of the joint full model and of each class's reduced model, the class's held-out improvement, the label-shuffle tail fraction and usable-shuffle count, convergence flags); the `.npy` companions hold the reported lag axis and, per replay class (`non_local_SWR`, `non_local_no_SWR`), the typical-event kernel and the content mark-slope kernel; the `.npz` holds that class's shuffled held-out improvements; the sidecar the provenance. **No bundle exists on disk** |
   | `pfc_marked_model_family/{epoch_identifier}_{model}_marked_family.pkl` + `.lag_time.npy`, `.{class}.typical.npy`, `.{class}.{target}.slope.npy`, `.provenance.json` | `scripts/pfc/run_pfc_marked_model_family.py` | the no-p-value descriptive family in [design/pfc_marked_model_family.md](design/pfc_marked_model_family.md): one bundle per session × eligible model, with all recorded units, target-specific blocked-held-out deviance changes, typical-event and one-SD target kernels, exact mark scales/ranks/device/timing in provenance, and sidecar-only markers for ineligible or no-unit sessions. The bundles on disk are the **2026-09-19 corpus batch** (71 sessions × 12 models, 816 bundles + 36 ineligible markers, git 4d69b83, base σ 3 / mark σ 0.5), fit before the analysed-time rule and the current cohort: [2026-09-19-pfc-marked-family-corpus.md](../.claude/docs/reviews/2026-09-19-pfc-marked-family-corpus.md), population summary (unit → session → animal, `scripts/pfc/summarize_pfc_marked_family.py`, figures under `paper/figures/pfc_marked_family/`) in [2026-09-19-pfc-marked-family-summary.md](../.claude/docs/reviews/2026-09-19-pfc-marked-family-summary.md); the prior choice and why a family `typical` curve is not the Figure 4 kernel are in [2026-09-18-pfc-marked-family-smoke.md](../.claude/docs/reviews/2026-09-18-pfc-marked-family-smoke.md) |

   All five are fit on the **well-anchored stop cohort**, the same population as the
   rate and content ladders, and an unsuffixed PFC artifact is that cohort; `_allstops`
   is the unrestricted sensitivity bundle (unit bundles and Stage 6 only — Stage 5B has
   one rule at the event grain). **Only Stage 6 has shipped an `_allstops` bundle**;
   the unit-GLM one has never been run — produce it with
   `run_pfc_analysis_bandit.py --stop-cohort all_stops` (a full <!--v:mpfc.sessions-->75<!--/v-->-session refit), which
   is the only way to compare the unit fits across populations on artifacts rather than
   on the unit test that pins the unrestricted path. A bundle fit at a prior other than
   the canonical one is written under a `_sigma<value>` name beside it and draws no
   null — it exists for its modulation curves. Every PFC artifact carries a `*.provenance.json`
   sidecar recording `stop_cohort`, the stop-table SHA-256, and the bins and events
   before and after the restriction. The unit bundles' sidecar also states **which
   events the fit describes**: `is_swr_definition`, the SHA-256 of the canonical event
   table and of the session's cached decode frame
   (`canonical_event_table_sha256`, `data_over_time_sha256`), the cohort-restricted
   `n_events_total` / `n_events_swr` / `n_events_non_swr`, and the matched control's
   `matched_control_seed` / `n_matched_control` / `n_matched_control_unplaced` /
   `matched_control_placement_method`;
   `run_pfc_analysis_bandit.py` refuses to skip a bundle that does not carry the
   current values. It also states the **estimator and the inference**: `estimator`
   (`ridge_poisson_glm`), `inference` (`blocked_heldout_deviance`),
   `penalty_sigma_log_rate` with the `implied_prior_sd` it means for a curve,
   `n_basis_per_class`, `n_folds` / `fold_seed` / `fold_sizes` / `events_per_fold`,
   `n_control_draws` / `control_draw_seed` / `null_usable_draws` /
   `min_usable_null_draws` / `null_windows` (per class: the events total, the fixed
   matchable subset actually tested, the excluded remainder, the tested source indices
   and how many windows each draw placed), `empirical_pvalue_method` (`plus_one`),
   `control_null_event_policy` (`fixed_jointly_matchable_subset`) and
   `engagement_null_overlap` (how much of the engagement null's draws land on the shipped
   control). The
   canonical bundles carry **σ = <!--v:mpfc.penalty_sigma-->3<!--/v-->** (implied per-lag sd **<!--v:mpfc.prior_sd_per_lag-->2.07<!--/v-->**, a rate multiplier
   of ×<!--v:mpfc.prior_multiplier_per_lag_low-->0.017<!--/v-->–×<!--v:mpfc.prior_multiplier_per_lag_high-->57<!--/v--> at 95 %; sustained sd **<!--v:mpfc.prior_sd_sustained-->0.90<!--/v-->**, ×<!--v:mpfc.prior_multiplier_sustained_low-->0.17<!--/v-->–×<!--v:mpfc.prior_multiplier_sustained_high-->5.9<!--/v-->), **K = <!--v:mpfc.folds-->5<!--/v-->** folds and
   **R = <!--v:mpfc.null_draws-->1,000<!--/v-->** null draws — read from the sidecars into `paper/reported/mpfc.json`. The finite-sample Monte Carlo p-value is **`(r + 1)/(R + 1)`**,
   so with <!--v:mpfc.null_draws-->1,000<!--/v--> usable draws its minimum is **1/<!--v:mpfc.min_pvalue_denominator-->1,001<!--/v--> ≈ <!--v:mpfc.min_pvalue-->0.0010<!--/v-->**; zero is impossible.
   Draws that fail to place every matched window are excluded, and **every
   requested draw must be usable** for a unit's p-value to be reported — an
   `R = 1,000` label always means a denominator of 1,000, and the per-unit
   effective counts are in both the table and the sidecar (`null_usable_draws`,
   `min_usable_null_draws`). The 20-draw floor governs deliberately shortened
   diagnostics only. Unit fractions at raw p < 0.05 are
   descriptive previews only; population inference aggregates through session and
   animal. A bundle that does not state all of these is refit rather than skipped,
   and the population loader also refuses it before figure generation — the
   canonical bundle name is unsuffixed, so nothing else tells a penalised bundle from
   the maximum-likelihood one that held the same name. It also records what the fits
   themselves did: `n_model_fits` (neuron × model cells that produced a fit at all),
   `n_nonconverged_model_fits` and `n_neurons_with_nonconverged_model` (fits that ran
   to `maxiter` without settling, and how many neurons carry at least one). A fit that
   never settled is non-estimable for the unit that owns it — every statistic reading
   it is NaN — so those counts are the only record of how much was dropped.

   **The current <!--v:mpfc.sessions-->75<!--/v--> at-well unit bundles are the ridge/held-out estimator refit on the
   R11 re-decode**, at git `110c9ae` on 2026-10-11/12 (logs in `logs/pfc_refresh2_20261011/`).
   The estimator's first batch, at `db01df0` (2026-09-16/17), replaced the
   maximum-likelihood tree, which is kept whole at
   `Processed-Data/_pre_ridge_backup_20260915/`. That before/after is audited in
   [2026-09-18-pfc-ridge-heldout.md](../.claude/docs/reviews/2026-09-18-pfc-ridge-heldout.md):
   the two inferences call comparable fractions of a comparable population, which are **not**
   comparable test by test. In the current bundles (`paper/reported/mpfc.json`),
   non-converged fits are **<!--v:mpfc.nonconverged_percent-->0.26<!--/v-->%** of model fits (<!--v:mpfc.nonconverged_model_fits-->432<!--/v--> of <!--v:mpfc.model_fits-->166,968<!--/v-->,
   over <!--v:mpfc.nonconverged_neurons-->48<!--/v--> neurons). The prior binds on
   <!--v:mpfc.prior_binds_percent_min-->2.6<!--/v-->–<!--v:mpfc.prior_binds_percent_max-->4.9<!--/v-->% of well-posed units per curve. The null's geometry, by the rules of
   `scripts/pfc/measure_pfc_null_geometry.py`: the engagement null's draws land on the
   shipped control they are entered beside on a median <!--v:mpfc.engagement_null_overlap_median-->11 %<!--/v--> of their windows, <!--v:mpfc.class_called_flat_fraction_min@pctnum-->4<!--/v-->–<!--v:mpfc.class_called_flat_fraction_max-->8 %<!--/v--> of
   the units called at raw p < .05 for an event class have no positive held-out improvement of their own,
   and <!--v:mpfc.unmatchable_percent_class_min-->0.27<!--/v-->–<!--v:mpfc.unmatchable_percent_class_max-->0.60<!--/v-->% of a class's events have no legal within-bout control and enter neither
   side of its test. Figure 4 is regenerated from these bundles; question 4.1's numbers are read from
   `paper/reported/mpfc.json`. A one-session sensitivity pair at σ = 2 and σ = 4.5
   (`j1620210710_02_r1`) was fit beside the first batch under `_sigma2` / `_sigma4p5`; it was
   not refit on the R11 re-decode and is no longer in `Processed-Data/`.
   **The replay-split curves were swapped in every bundle written before 2026-09-15** —
   `non_local_SWR_modulation` held the non-SWR kernel and vice versa — so any per-class
   modulation magnitude quoted from the split model before that date, including from the
   backup tree, is mislabelled; whole-model comparisons never read a block and are
   unaffected. See
   [design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md),
   [2026-09-14-pfc-atwell-cohort.md](../.claude/docs/reviews/2026-09-14-pfc-atwell-cohort.md) and
   [2026-09-13-stage5b-stage6-atwell.md](../.claude/docs/reviews/2026-09-13-stage5b-stage6-atwell.md).
   The earlier canonical-events rerun is in
   [2026-09-15-pfc-canonical-rerun.md](../.claude/docs/reviews/2026-09-15-pfc-canonical-rerun.md).

   The batched fitter runs on GPU through the project's `cuda` extra, and the decoders'
   GPU cluster needs `dask-cuda` and `cuda-bindings` (which dask-cuda imports without
   declaring) from the `gpu` extra. These runs installed `jax-cuda12-plugin[with-cuda]==0.8.0`,
   `dask-cuda==25.10.0` and `cuda-bindings==13.1.1` by hand, when `uv.lock` had no entries
   for the extras; it now locks those versions, so `uv sync --frozen --inexact --extra gpu`
   installs the same GPU environment.

2. **Results**: the reported values (`paper/reported/<summary>.json`, listed in
   `paper/reported/questions.md`).

3. **Figures**: `paper/figures/<topic>/` (publication) vs `.claude/research/*_outputs/` (exploratory) — see above.
