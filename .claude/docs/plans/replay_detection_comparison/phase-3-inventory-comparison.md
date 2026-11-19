# Phase 3 — Eight inventories on one substrate: adapters, runner, report, examples

[← back to PLAN.md](PLAN.md) · [spec 03](../../../../docs/design/replay_detection_comparison/03-inventory-comparison.md) ·
[overview](../../../../docs/design/replay_detection_comparison/overview.md) ·
[detector table](../../../../docs/design/replay_detection_comparison/detector_comparison.md)

Written 2026-10-17 from the lead's brief (`.superpowers/sdd/PLAN/phase3-plan-brief.md`). The spec wins
where this file and the spec disagree. Status lives in [PLAN.md](PLAN.md), not here.

## What this phase delivers

The spec's five deliverables (03 §Deliverables) for the 242 analyzed epochs, at the retained default
settings, over eight inventories:

| # | detector name in the tables | source |
|---|---|---|
| 1 | `kay` | the direct Kay inventory, run `056eee0dcb5a` (D7: this is inventory 1, the paper's SWR source) |
| 2 | `karlsson` | the same run |
| 3 | `yu` | the same run |
| 7 | `zugaro` | the same run |
| 8 | `carey` | new: `Carey_candidate_detector`, package defaults (Task 1, Task 2) |
| 4 | `mua` | the `MUA_event` labels of each canonical decode frame |
| 5 | `state_space` | `build_classified_nonlocal_event_table` on each canonical decode frame |
| 6 | `memoryless` | the Widloski table, run `d0964cc76ed2-seed20260922`, cohort `published` |

Long is dropped (D1): every tetrode targets stratum pyramidale (`docs/task_description.md:159-161`), and
Long needs a stratum radiatum channel.

**No claim about the Widloski column is made from this phase** until the deferred Phase 2 pilot checks
(null validation, two sensitivities; PLAN.md "Phase 2 pilot") are done. Phase 3 does not wait for them.
The report and the examples carry that note wherever the `memoryless` column appears.

## Inputs to read first

- `src/continuum_swr_replay/analysis/inventory_comparison/` — the accounting, built and tested. The functions
  this phase calls, by module: `schema` (`SessionIdentity` :79, `inventory_from_intervals` :268,
  `from_classified_nonlocal_table` :351, `from_memoryless_table` :402, `combine_inventories` :199);
  `availability` (`sample_runs` :97, `still_intervals` :172, `observation_table` :215,
  `missing_run_table` :255, `coverage_from_observations` :268); `overlap` (`Coverage` :140,
  `remove_bridging_events` :308, `overlap_pairs` :352, `event_matches` :403, `flag_pairs` :457,
  `directional_agreement` :488, `time_coverage` :547); `components.overlap_components` :44;
  `recount.merge_fragments` :70 with `FRAGMENT_GAPS_S` and `not_still_s` (the times of the samples that were
  not still, `availability.not_still_times`; no merged event holds one); `cells` (`membership_against` :77,
  `no_detector_time` :140); `swr_matrix` (`swr_labels` :143, `swr_label_matrix` :211, `paired_labels` :294,
  `definition_disagreement` :338). The worked end-to-end epoch is
  `tests/unit/test_inventory_comparison_end_to_end.py`.
- `scripts/decoding/run_swr_detection.py` — the four SWR inventories. Per-epoch steps in the module docstring
  (lines 1-120); `output_paths` :239, `observation_table` :329 (the `swr_observed` file, scopes
  `stored_band` and `zugaro`), `array_sha256` :396, `json_ready` :403, `epoch_artifact_intervals` :535,
  `_detect` :622 (span cut :664-675, R-T :680-705, the `input_sha256` record :785-789), `run_epochs` :1102,
  `run_settings` :1170, `run_lock` :1222. The run folder holds `{epoch_id}_{kay,karlsson,yu,zugaro}_events.parquet`
  (indexed `event_number` 1..n, columns `start_time`, `end_time`, `duration`, ...), `{epoch_id}_swr_observed.parquet`,
  and `manifest.json` with each output's sha256 and each epoch's kept `electrode_ids` and `input_sha256`.
- `src/continuum_swr_replay/analysis/hippocampus/swr_inventories.py` — `DETECTOR_SETTINGS` :522,
  `drop_overlapping_events` :646, `run_swr_detectors` :683, `gap_free_stretches` :273.
- `src/continuum_swr_replay/analysis/hippocampus/swr_source.py` — `SWR_INVENTORY_RUN_ID` :172,
  `SWR_INVENTORY_ROOT` :188, `ARTIFACT_INTERVALS_RUN_ID` :191, `swr_source_files` :381 (Kay only),
  `ripple_power_inputs` :448 (kept electrodes, span, artifact mask), `artifact_intervals` :524,
  `excluded_intervals` :562, `artifact_event_drops` :915, `read_artifact_event_drops` :959,
  `dropped_for_excluded_time` :967, `excluded_intervals_of` :1127, `read_dot_with_swr_source` :1192.
- `src/continuum_swr_replay/analysis/hippocampus/event_eligibility.py` — the one analyzed-time rule:
  `DropReason`, `EXCLUDED_TIME_REASONS`, `excluded_time` :97, `overlaps_dropped_events` :157,
  `event_drop_reasons` :198.
- `src/continuum_swr_replay/analysis/hippocampus/event_table.py:186` — `build_classified_nonlocal_event_table`.
- `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:612-630` — the decode labels only
  speed-eligible non-local and MUA events (`build_speed_eligible_event_cohort`), so those two inventories hold
  no moving events.
- `scripts/decoding/run_memoryless_detection.py` — `input_paths` :284 (the decode's `_detector.pkl` and
  `_80_data_over_time.pkl` under `PROCESSED_DATA_DIR`), `load_marks` :423 (the decode's marks, cached under
  `replay_detection_comparison/memoryless/marks_cache/`), `observation_table` :497 (kinds `decoded_windows`,
  `swr_labels`); `check_marks_match_encoding` is in `memoryless_sequence/likelihood.py:217`.
- The Carey detector at the pinned package (`68e1784`):
  `.venv/lib/python3.11/site-packages/ripple_detection/detectors/_carey.py:466`. It has **no
  `normalization_mask`** (its joint score is z-scored over every valid sample, :797), its state rule is
  "contained in a low-speed interval" (:833), and `theta_lfp` defaults to None.
- `src/continuum_swr_replay/analysis/pfc/glm.py:48` — `get_spikecount_per_time_bin`, the MUA binning rule
  (`time[i] <= s < time[i+1]`), the same rule `get_multiunit_hse` uses inline.

## Shared conventions for every task

- **Speed trace.** Every speed-eligibility read and the shared still mask use one trace: the canonical
  decode frame's `head_speed` on its 500 Hz grid, as `swr_source.swr_events` and the classified table read it.
- **Epoch bounds** for every observation table are the decode grid's first and last time (equal to the
  position span the SWR runner cut to; check it).
- **Dropped events.** Every event the analyzed-time rule drops (reasons `artifact`, `near_artifact`,
  `overlaps_dropped_event`) leaves its cohort, keeps its reason in a `set_aside` table, and its span is
  removed from its own detector's observed time, as `remove_bridging_events` does for bridging events, so a
  dropped event never makes another detector's event read `unmatched` (open question Q10).
- **Outputs** go under `PROCESSED_DATA_DIR` (`parameters.py:84`; `CSR_PROCESSED_DATA_DIR` overrides it). In
  the phase-3 worktree, `Processed-Data` is a symlink to the main tree's, so a run writes the main tree's data.
  Tables are written with `table_io.write_table` (parquet with `attrs`).
- **Names.** No docstring, test or module names a plan, phase or task number. The one exception is the
  deferred-check phrase "pending the Phase 2 pilot checks", which marks every row, caption and figure that
  shows a memoryless inventory.

## Tasks

### Task 1 — Carey on one epoch's shared inputs (package code, no IO)

**Unblocked 2026-10-17** (see "Answers from the lead"; Q1–Q4 were the artifact mask, time grid, ripple channels and theta). Write the tests for the parts the
answers do not change first.

- **Files.** Modify `src/continuum_swr_replay/analysis/hippocampus/swr_inventories.py`. Tests in a new
  `tests/unit/test_carey_inventory.py`.
- **Consumes.** `ripple_detection.Carey_candidate_detector`; `get_spikecount_per_time_bin`
  (`analysis/pfc/glm.py:48`); `gap_free_stretches` (:273); `drop_overlapping_events` (:646);
  `ripple_detection.minimum_sample_count`.
- **Produces.**
  - `CAREY_SETTINGS`: a read-only mapping beside `DETECTOR_SETTINGS`, not inside it. Adding `carey` to
    `DETECTOR_SETTINGS` would change the SWR runner's outputs, its settings gate and `run_swr_detectors`.
    Every keyword set explicitly to the package default (`speed_threshold` 4.0, `low_threshold` 1,
    `high_threshold` 3, `minimum_duration` 0.020, `minimum_active_units` 5, `spike_cap` 2, the three
    smoothing sigmas, `baseline_cap` 4, `state_merge_gap` and `minimum_state_duration` 0.050,
    `maximum_duration` None, `threshold_method` `"zscore"`), and no theta channel unless Q4 says otherwise.
  - `tetrode_spike_counts(spike_times, time)`: shape `(n_time, n_tetrodes)`, one column per tetrode of the
    decode's marks (adaptation 1: a tetrode's multiunit train is one "unit"), binned by the MUA rule on the
    grid Q2 chooses. A spike inside a timestamp gap (a step over 1.5 median steps) is not counted on the
    sample before the gap.
  - `run_carey_detector(time, ripple_band, spike_counts, speed, sampling_frequency, *, unobserved)`:
    Carey with `CAREY_SETTINGS`, the artifact mask handled as Q1 decides, then
    `drop_overlapping_events` on the mask, as the other four. Returns the package's event table (indexed
    `event_number` 1..n) or the exception it raised, as `run_swr_detectors` does.
  - `carey_observed_stretches(time, ripple_band, spike_counts, unobserved)`: the stretches Carey can detect
    in, by the rule it splits at (non-finite rows, timestamp gaps, stretches shorter than its minimum
    duration dropped), less the mask. No edge trimming for its kernels, as for Kay, whose smoothing is not
    trimmed either.
- **Tests first** (simulated, `ripple_detection.simulate.simulate_session`, as the package docstring example):
  - `CAREY_SETTINGS` equals the package signature's defaults (read with `inspect.signature`), so "package
    default" cannot drift;
  - binning: a spike at a sample time lands in that sample; a spike at `time[-1]` lands in the last; a spike
    in a gap is not counted; the column order is the marks' order;
  - a simulated session with five ripple-plus-burst events gives events, every one with
    `n_active_units >= 5`;
  - an event across a NaN stretch or a timestamp gap is not found, and the observed stretches exclude both;
  - an event overlapping the mask is dropped and the rest are renumbered 1..n;
  - fewer tetrodes than `minimum_active_units`, or no spikes, returns the package's error, not an empty table.
- **Accepts.** Tests pass; `ruff` and `mypy` clean on the module.

### Task 2 — The Carey runner, beside the SWR runner

- **Files.** Create `scripts/decoding/run_carey_detection.py`. Tests in `tests/unit/test_run_carey_detection.py`
  with fetchers injected, as `tests/unit/test_run_swr_detection.py` does for the SWR runner.
- **Consumes.** From `run_swr_detection`: `epoch_id`, `array_sha256`, `json_ready`, `write_json`,
  `epoch_artifact_intervals`, `run_epochs` (its `task` parameter exists for another runner), `run_lock`,
  `RUN_ID_PATTERN`. From the package: `get_ripple_band_inputs`, `get_epoch_position_info`,
  `swr_source.ripple_power_inputs` (the R-T kept electrodes and span of the SWR source run),
  `epochs_to_run`, `filter_epoch_info`, `provenance_record`. The marks: `load_marks` and `input_paths` from
  `run_memoryless_detection` (the decode's marks, from its cache when present), checked with
  `check_marks_match_encoding` against the decode's encoding model. Task 1's functions.
- **Produces**, under `PROCESSED_DATA_DIR/replay_detection_comparison/carey/<run_id>/` (run id: the first 12
  characters of the commit, as the SWR runner):
  - `{epoch_id}_carey_events.parquet`, the package's table with `attrs` stamped as the SWR runner stamps them;
  - `{epoch_id}_carey_observed.parquet`, columns `scope` (`"carey"`), `start_s`, `end_s`, the SWR runner's
    layout;
  - `manifest.json`: settings (`CAREY_SETTINGS`, the package commit, the SWR run id whose tetrodes and inputs
    it read, the artifact run, the marks rule, the grid) and one record per epoch (status, `n_events`, the
    kept electrodes, the number of multiunit tetrodes, the input hashes, the marks' sha256, wall clock,
    worker peak RSS, each output's path and sha256). Written by the main process only, atomically, after
    each epoch. A rerun skips `ok` epochs unless `--overwrite`, and refuses other settings.
- **Per epoch:** read the stored band; cut it to the position span; subset to the SWR run's kept electrodes;
  **refuse the epoch unless `array_sha256` of the time and of the kept band equal that run's
  `input_sha256["time"]` and `["ripple_band"]`** (the same inputs, byte for byte); interpolate `head_speed`
  onto the LFP times as `_detect` does; read the artifact mask; read and check the decode's marks; run
  Task 1; write.
- **Tests first:** an injected epoch writes both files and a manifest record; a band hash that differs
  from the SWR run's fails the epoch with that reason and writes nothing; marks that do not match the
  encoding model fail the epoch; a rerun skips an `ok` epoch and refuses changed settings; a detector error is
  recorded as `failed` with the error.
- **Accepts.** Tests pass; `--help` runs; the one-epoch smoke (runbook R1) passes its gates.

### Task 3 — Review item: an event between the last still sample and the first moving one

The spec review of 2026-09-22 found that an event ending after the last still sample of a run and before the
first moving one is speed-eligible (its only samples are still) yet outside exposure, so it reads `unknown`.
The same holds at a run's start.

- **Files.** Modify `availability.still_intervals` (:172). Update `tests/unit/test_inventory_availability.py`
  and the hand-worked exposure in `tests/unit/test_inventory_comparison_end_to_end.py` (its still intervals
  become `[0, 20.x]` and `[24.y, 60]` to the neighboring samples).
- **Change** (pending Q9): each still run reaches out to the neighboring samples on either side, when they
  continue the run's stretch (no timestamp gap, finite time). The event rule reads only the samples inside an
  event, so an event is speed-eligible exactly when it lies inside such an interval and no sample inside it is
  moving or unknown. Exposure grows by at most one sample interval (2 ms) at each end of a still run.
- **Tests first:** the review's case (event ending between the last still and first moving sample) is
  `within_shared_exposure` and `unmatched` when nothing overlaps it; the mirrored case at a run's start; a run
  bounded by a timestamp gap or a NaN speed does not reach across it; an event containing a moving sample is
  still ineligible (set aside by `combine_inventories`, never counted).
- **Accepts.** The availability, overlap, cells and end-to-end tests pass with the new exposures.

### Task 4 — Review item: excluded time by reason, and set-aside spans

The spec asks for "the exposure duration and excluded time by reason for each comparison"; the observation
tables carry a reason per stretch, but nothing joins them to `Coverage`.

- **Files.** Modify `availability.py` and `overlap.py`. Tests in the existing
  `test_inventory_availability.py` and `test_inventory_overlap.py`.
- **Produces.**
  - `set_aside_events(coverage, events)` in `overlap.py`, beside `remove_bridging_events`: removes each given
    event's span from its own detector's `observed` and `exposure`. The runner passes the dropped events
    (Shared conventions). `remove_bridging_events` is unchanged.
  - `exclusion_by_reason(observations, coverage, still, set_aside, *, comparisons)` in `availability.py`:
    for each epoch and each named comparison (a set of detectors), the epoch's seconds outside the shared
    exposure, split by reason, first match wins, in this order: excluded time; each detector's own
    unavailable reason from its observation table (`"<detector>: <reason>"`); a set-aside span
    (`"<detector>: <drop or bridging reason>"`); speed not still (`moving`, `speed_unknown`). One row per
    reason plus one `shared_exposure` row, so the rows sum to the epoch span. Columns:
    `animal, day, epoch, comparison, reason, seconds`.
  - The comparisons the runner names: every pair, `all` (the eight), and each SWR-matrix population
    (`state_space` or `memoryless` with the four SWR definitions).
- **Tests first:** on the end-to-end epoch, the rows sum to the epoch span and `shared_exposure` equals
  `total_duration` of the intersected exposures; a stretch missing for two detectors is counted once, under
  the first; a dropped event's span reads under its reason and its detector's events no longer count as
  `unmatched` against it; a missing run (`missing_run_table`) reads as its reason for the whole epoch.
- **Accepts.** Tests pass; every existing accounting test passes unchanged.

### Task 5 — One adapter per inventory

- **Files.** Create `src/continuum_swr_replay/analysis/inventory_comparison/adapters.py` (the accounting
  modules do no IO, so the readers of recorded outputs go here); list it in the package `__init__` docstring.
  Tests in `tests/unit/test_inventory_adapters.py`, with the SWR source fixtures
  (`tests/unit/_swr_inventory_fixtures.py`: `write_swr_inventory`, `write_artifact_run`, `use_swr_source`).
- **Each adapter** returns three tables for one epoch: the inventory (common schema, moving events kept with
  their flag), its observation table (`observation_table`, or `missing_run_table` when the run has no usable
  record), and its set-aside events (key, bounds, reason). Each builds its coverage from its own inputs:

  | adapter | events | dropped events | observed |
  |---|---|---|---|
  | `swr_inventory(run_dir, detector, ...)` for `kay`, `karlsson`, `yu`, `zugaro` | `{epoch_id}_{detector}_events.parquet`, checked against the manifest's sha256, through `inventory_from_intervals` (native key `event_number`, native duration the table's `duration`) | `event_drop_reasons` with the artifact mask, then `overlaps_dropped_events` against `dropped_for_excluded_time(artifact_event_drops(dot))`, the one rule's second pass | the `swr_observed` file, scope `stored_band` (`zugaro` for Zugaro) |
  | `carey_inventory(run_dir, ...)` | `{epoch_id}_carey_events.parquet`, checked against its manifest | as the SWR row | `{epoch_id}_carey_observed.parquet` |
  | `mua_inventory(dot, ...)` | `intervals_from_labels(dot, "MUA_event")` through `inventory_from_intervals` | `artifact_event_drops` rows with `event_col == "MUA_event"` and a reason | `sample_runs` on the decode grid, valid outside the artifact mask (`interval_mask`) |
  | `state_space_inventory(dot, ...)` | `from_classified_nonlocal_table(build_classified_nonlocal_event_table(dot))` | `artifact_event_drops` rows with `event_col == "non_local_event"` and a reason | `sample_runs` on the decode grid, valid where the non-local probability is finite and outside the mask |
  | `memoryless_inventory(run_dir, ...)` | `from_memoryless_table(..., cohort="published")` (Q7) | rows the cohort flag selects with a stored excluded-time `artifact_drop_reason`; the stored `overlaps_dropped_event` is not read, and the second pass is made again on grid-sampled bounds, as the SWR row's | the observed file's `decoded_windows` stretches |

  `dot` is the canonical decode frame read once through `read_dot_with_swr_source`; `artifact_event_drops`
  is read once from the same pickle (`read_artifact_event_drops`). The SWR-file reader repeats
  `swr_source_files`' manifest and sha256 check for any detector and run folder; `swr_source` itself is
  not changed, as it is the paper's SWR source.
- **Tests first** (simulated inventories written to a temp run folder):
  - each adapter's inventory passes `validate_inventory`, empty ones included, with the schema's dtypes;
  - native counts equal the source file's rows; a moving event stays with its flag;
  - an event in excluded time is in `set_aside` with `artifact` or `near_artifact`; a Karlsson event
    overlapping a Kay SWR dropped for excluded time is set aside with `overlaps_dropped_event`;
  - a changed file (sha256 mismatch), a missing epoch or a failed record yields `missing_run_table` with the
    reason, not an empty observed inventory;
  - Kay through this adapter keeps exactly the events `swr_source.with_swr_source` keeps in the same frame
    (the paper's cohort, by key);
  - the memoryless adapter refuses an unshuffled table (the existing `from_memoryless_table` check surfaces).
- **Accepts.** Tests pass; on one real epoch (runbook R3) the kept Kay, MUA and state-space counts equal the
  decode frame's speed-eligible kept events.

### Task 6 — The per-epoch runner

- **Files.** Create `scripts/qa/run_inventory_comparison.py`. Tests in
  `tests/integration/test_run_inventory_comparison.py` (one simulated epoch, on disk, end to end).
- **Consumes.** Task 5's adapters; `still_intervals` on the decode frame's `head_speed`;
  `coverage_from_observations`; `set_aside_events`; `combine_inventories`; `remove_bridging_events`;
  `overlap_pairs`, `event_matches`, `flag_pairs`, `directional_agreement`, `time_coverage`;
  `overlap_components` over the eight without `memoryless_criteria` (its events and their pairs left out);
  `merge_fragments` (detector `state_space`, every `FRAGMENT_GAPS_S`, `sample_interval_s` 0.002, and
  `not_still_s` from `not_still_times` on the decode frame's `head_speed`) followed by `overlap_pairs` and
  `directional_agreement` on each recount; `membership_against` for the three cells (exactly one detector,
  each against the other seven; SWRs `kay, karlsson, yu, zugaro` against `state_space, memoryless`;
  sequences `state_space, memoryless, memoryless_criteria` against `kay, karlsson, yu, zugaro, carey, mua`,
  see Q6); `no_detector_time` over the eight; `swr_labels`,
  `swr_label_matrix`, `paired_labels`, `definition_disagreement` (cohorts `state_space, memoryless`;
  definitions `kay, karlsson, yu, zugaro`, see Q6); `exclusion_by_reason`.
- **Produces**, under `PROCESSED_DATA_DIR/replay_detection_comparison/inventory/<run_id>/`, per epoch
  `{epoch_id}_<table>.parquet` for: `inventory` (eligible and moving, with `component_id`), `set_aside`
  (dropped and bridging, with reason), `observations`, `exclusion_by_reason`, `counts` (per detector: native,
  eligible, set aside, own exposure seconds, rate = eligible events wholly inside own exposure per exposure
  second), `pairs` (flagged), `directional`, `time_coverage`, `components`, `recount_directional` (with
  `gap_s`), `cells` (with a `cell` column), `no_detector_time`, `swr_labels`, `swr_matrix`, `paired_labels`,
  `definition_disagreement`. These are one atomic set per epoch: written together, never partly.
  `manifest.json` records the input runs (SWR, Carey, memoryless, artifact), `swr_source_name()`, the decode
  files' sha256, the detector roster and cohort, and per epoch the status, the counts, wall clock and peak
  RSS. Rerun, skip and settings-gate rules as the other runners.
- **Runs** sequentially, one epoch at a time, CPU only, with `--animal`, `--date`, `--epoch`, `--epoch-id`
  and `--overwrite`, holding `run_lock` on its folder. No parallel code: if the smoke (R3) shows the
  sequential run would take more than about 4 h, the executor runs five pinned processes, one per animal,
  each under its own run id (one run id cannot have two writers).
- **Tests first:** the simulated epoch on disk reproduces the hand-worked exposures of the end-to-end test
  (after Task 3); an epoch whose memoryless run is missing writes every table with `memoryless` unavailable
  and its cells `unknown`, not `unmatched`; event keys are unique across two epochs written by one run; every
  table's dtypes survive a write and read; a rerun skips `ok` epochs.
- **Accepts.** Tests pass; the one-epoch smoke (R3) passes its gates.

### Task 7 — Per-session and per-animal report

- **Files.** Create `scripts/qa/report_inventory_comparison.py`; figures go to
  `paper/figures/inventory_comparison/`, tables to `<inventory run>/report/`. Tests in
  `tests/unit/test_report_inventory_comparison.py` on synthetic per-epoch tables.
- **Consumes.** The per-epoch tables of one inventory run, concatenated.
- **Produces** (spec 03 §Reporting): per-session values with sessions nested in animal, and animal-level
  summaries that pool by summing counts (`n_matched` over `n_eligible`), never by averaging proportions;
  native duration distributions per detector (`interval_span_s` and the native duration, both shown), before
  any duration-stratified view; native counts beside eligible counts, set-aside counts by reason, skipped and
  failed epochs; exposure and excluded time by reason per comparison; directional agreement with overlap
  length and IoU distributions; time coverage; the recount against the native count; the three cells with
  `unknown` reported apart; no-detector time; the SWR matrix with its denominators and the paired-label and
  definition-disagreement tables; component sizes and spans. A `memoryless` column is marked "pending the
  Phase 2 pilot checks" in every table and caption. The report states counts; it writes no claim.
- **Tests first:** an animal summary of two sessions equals the summed counts, not the mean of proportions;
  an unavailable session is excluded from a pooled proportion and listed; an `unknown` membership never
  enters an `unmatched` count.
- **Accepts.** Tests pass; the report on the five pilot epochs (R4) is read by the lead.

### Task 8 — Sampled examples from each cell

- **Files.** Create `scripts/figures/generate_inventory_examples.py`; figures to
  `paper/figures/inventory_comparison/examples/`. Tests in `tests/unit/test_inventory_examples.py` for the
  sampler only.
- **Sampling rule** (stated in the figure captions and the run's manifest): for each cell (each detector's
  exactly-one events; SWRs matched by neither decoder; sequences matched by no triggered inventory) and each
  animal, draw 2 events with membership `unmatched` and, as counterexamples in the other direction, 2 events of
  the same source detector with membership `matched` by the same set, by `numpy.random.default_rng` seeded
  from a fixed run seed and the (cell, animal) pair; fewer if the cell has fewer. Events with `unknown`
  membership are never drawn.
- **Each panel set**, on one time axis around the event: spikes by tetrode (the decode's marks); the
  state-space posterior (the decode's `results.nc`); the memoryless window posteriors, recomputed for the
  plotted span with the `memoryless_sequence` functions on the decode's encoding model, and checked against
  the stored candidate's statistics; gate masks (non-local probability over 0.8, the memoryless per-window
  gates from `{epoch_id}_memoryless_candidates.pkl`, speed below 4 cm/s, excluded time shaded); speed; the
  ripple band of the kept tetrodes; every inventory's events as bars.
- **Tests first:** the sampler is deterministic for a seed, draws only `unmatched` (or `matched` for
  counterexamples), never `unknown`, and returns fewer when the cell is small.
- **Accepts.** One example per cell renders on a pilot epoch; the recomputed memoryless statistics equal the
  stored ones.

### Task 9 — Docs that ship with the code

- `docs/analysis_map.md`: Q1 rows for `decoding/run_carey_detection.py`, `qa/run_inventory_comparison.py`,
  `qa/report_inventory_comparison.py`, `figures/generate_inventory_examples.py`; remove "Goal 4's full detector
  comparison" from "No script" once the run is done.
- `scripts/README.md`: entries for the four scripts.
- `docs/design/replay_detection_comparison/03-inventory-comparison.md` header status and
  `detector_comparison.md`'s Carey row status, once the runs are done.
- `PLAN.md`'s phase row and status line after each stop point.

## Runbook for the real run

Machine: breeze (`breeze.cin.ucsf.edu`; same `/home` and `/cumulus`; Spyglass reachable). CPU only. Before
each launch, and again before each full run:

```bash
nproc; uptime; free -g; df -h /cumulus
ps -eo user,pcpu,pmem,args --sort=-pcpu | head -20   # other users' load
```

Pick a core range no other user's job is on, and pin with `taskset -c <range>`. Set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` so each worker uses one core. Run inside
`tmux`; the session names are below. Run from the phase-3 worktree with its `.venv`, from a clean commit
(the run id is the commit). The Carey and comparison runners' settings gate keys, as the SWR runner's
does, on the inputs and rules (Carey's also on the `ripple_detection` package commit), not the repository
commit: a rerun under one `--run-id` from another commit is accepted, and each epoch's record holds the
commit it ran at. `PROCESSED_DATA_DIR` is the main tree's `Processed-Data` through the symlink;
set `CSR_PROCESSED_DATA_DIR` only to write elsewhere.

**R0. Before any run.** Q1–Q4, Q7, Q9 and Q10 answered and their tasks merged into the phase branch; the suite
green; `swr_source.SWR_INVENTORY_RUN_ID` is `056eee0dcb5a` and `ARTIFACT_INTERVALS_RUN_ID` is
`fbbff144a123`.

**R1. Carey smoke, one epoch.** tmux `carey_smoke`.

```bash
taskset -c <range> .venv/bin/python scripts/decoding/run_carey_detection.py \
    --animal chimi --date 20200225 --epoch 02_r1 --n-workers 1
```

Gates: the epoch is `ok`; its band hashes matched the SWR run's; the marks matched the encoding model; every
event has `n_active_units >= 5` and lies in the observed stretches; look at five events by eye against the
Kay events of the same epoch. Record wall clock and worker peak RSS from the manifest. For scale: the SWR
runner's run `056eee0dcb5a` took a median 61 s per epoch (max 366 s, 5.9 h summed over 255 epochs) at a median
6.1 GiB worker peak (max 8.8 GiB); Carey reads the same LFP and adds the marks. **Stop point:** report the
smoke's numbers and the five events to the lead.

**R2. Carey, all 242 epochs.** tmux `carey_full`. `--n-workers` = min(pinned cores, free GiB / (smoke peak RSS
x 1.5)). Extrapolate before launching: 242 x smoke wall clock / n_workers. Gates: every epoch `ok`, or each
failure explained; event counts per epoch printed beside Kay's; no epoch with zero events unexplained.
**Stop point:** report the counts per animal to the lead before the comparison.

**R3. Comparison smoke, one epoch.** tmux `inventory_smoke`.

```bash
taskset -c <one core> .venv/bin/python scripts/qa/run_inventory_comparison.py \
    --carey-run-id <R2 run id> --memoryless-run-id d0964cc76ed2-seed20260922 \
    --epoch-id chimi20200225_02_r1_HPC
```

Gates, by hand: each detector's native count equals its source file's rows; kept Kay, MUA and state-space
counts equal the decode frame's kept events; `exclusion_by_reason` sums to the epoch span for every
comparison; the `kay` versus `state_space` directional proportion is close to the classified table's
`is_swr` fraction (not equal: different denominators; explain the gap); spot-check five `matched` and five
`unmatched` pairs on a plot. Record wall clock and peak RSS. Extrapolate: 242 x the smoke's wall clock. If
over about 4 h, split into five pinned processes by `--animal`, each under its own run id, and say so in
the manifest notes.

**R4. Pilot, five epochs.** tmux `inventory_pilot`. The five Phase 2 pilot epochs (PLAN.md "Phase 2 pilot":
one per animal). Then the report (Task 7) on them. **Stop point:** the lead reads the pilot report before the
full run.

**R5. Full run, 242 epochs.** tmux `inventory_full`. Gates: every epoch `ok`; the manifest's input runs are
the ones named; the roster has the eight detectors in every epoch (a missing run reads `unavailable`, never
absent).

**R6. Report and examples.** tmux `inventory_report`. Task 7 on the full run, then Task 8. **Stop point:**
the lead reviews the report and examples. Phase 4 starts from these tables.

Outputs, all under `PROCESSED_DATA_DIR/replay_detection_comparison/`: `carey/<run_id>/` (R1, R2),
`inventory/<run_id>/` with `report/` (R3–R6); figures in `paper/figures/inventory_comparison/`. Logs under
`logs/` in the working directory.

## Deliberately not in this phase

- The deferred eligibility and encoding controls (03 §Deferred): native speed gates, running-only encoding,
  reference-channel ripple power.
- Event features and the common spike-evidence diagnostic (Phase 4, spec 04); threshold robustness (Phase 5).
- The Phase 2 pilot checks (null validation, two sensitivities). Deferred by the lead; no Widloski claim
  until they are done.
- Any change to the `ripple_detection` package (Q1 option c needs the lead first), to the paper's SWR source
  (`swr_source.py`), or to run `056eee0dcb5a`.
- Spyglass writes of any kind; every read goes through the existing read paths.
- Long (D1, dropped).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_carey_inventory.py` | settings equal the package defaults; MUA binning rule; events need 5 active tetrodes; gaps and NaN stretches unobserved; mask overlap dropped and renumbered |
| `test_run_carey_detection.py` | outputs and manifest written; a band hash mismatch or marks mismatch fails the epoch and writes nothing; rerun skip and settings gate |
| `test_inventory_availability.py` (updated) | still runs reach the neighboring samples, never across a gap or NaN; `exclusion_by_reason` partitions the epoch |
| `test_inventory_overlap.py` (updated) | the review's boundary event is `unmatched`, not `unknown`; `set_aside_events` makes a dropped event's span unobserved |
| `test_inventory_comparison_end_to_end.py` (updated) | the hand-worked exposures under the new still rule |
| `test_inventory_adapters.py` | schema and dtypes for every adapter; native counts; drop reasons including `overlaps_dropped_event`; missing or changed runs read unavailable; Kay equals the paper's cohort |
| `tests/integration/test_run_inventory_comparison.py` (slow) | one simulated epoch on disk, every table written and readable; a missing run gives `unknown`, not `unmatched` |
| `test_report_inventory_comparison.py` | pooling sums counts; unavailable sessions listed, not pooled; `unknown` never counted |
| `test_inventory_examples.py` | sampler deterministic, draws only `unmatched` or `matched`, never `unknown` |
| R1, R3 (real data, manual) | the gates listed in the runbook |

## Answers from the lead (2026-10-17)

The lead agreed with every recommendation below. These rulings bind the tasks; the questions are kept under
them as asked.

- **Q1.** Use the package's new `normalization_mask` on `Carey_candidate_detector` (the lead's commit
  `9766fd4`; the project pins it from `e1decf2`). Pass the artifact mask's complement as the other four
  detectors are run. Task 1 is no longer blocked.
- **Q2.** The LFP's 1000 Hz grid, with each tetrode's spikes binned onto it.
- **Q3.** The stored 150–250 Hz band of the R-T kept tetrodes, averaged as the package does.
- **Q4.** No theta restriction (the package default). The 4 cm/s speed rule gates.
- **Q5.** Yes: record Q3 and Q4 as adaptations 3 and 4.
- **Q6.** Carey is a fifth SWR definition in the SWR label matrix, and also in the "triggered" set.
- **Q7.** Both cohorts, `published` and `replay_criteria`.
- **Q8.** Native counts equal eligible counts for MUA and state-space, with a note saying why.
- **Q9.** Agreed: still runs reach the neighboring samples.
- **Q10.** Agreed: a dropped event's span leaves its own detector's coverage.
- **Q11.** Accepted: the Carey runner imports the SWR runner's helpers.

## Open questions for the lead

Carey input choices that neither the spec nor the other detectors' runs settle:

- **Q1. The artifact mask in Carey's normalization.** The other four leave the mask out of their
  normalization (`normalization_mask`) and drop events that touch it. Carey has no `normalization_mask`. Options:
  (a) set the band and the spike counts to NaN inside the mask, so Carey's blocks split there and its
  normalizations leave it out (its kernels then restart at each mask edge); (b) run on the unmasked inputs and
  drop overlapping events, so its z-score includes artifact time, unlike the others; (c) add
  `normalization_mask` to the package (a package change, which needs your approval).
- **Q2. Carey's time grid.** The LFP's 1000 Hz samples, with each tetrode's spikes binned onto them by the MUA
  rule (the SWR detectors' grid), or the decode's 500 Hz grid (the MUA detector's grid) with the band
  resampled.
- **Q3. Carey's ripple channels.** The stored 150–250 Hz band of the R-T kept tetrodes, averaged as the package
  averages channels (what Kay, Karlsson and Yu read), or one channel as the original (140–250 Hz, one
  tetrode; for example Zugaro's chosen channel, which is read from the raw LFP, not the stored band).
- **Q4. Carey's theta restriction.** The package default has none; the original excludes high-theta periods,
  which needs a theta channel (one of the CA1 raw channels would do).
- **Q5.** If Q3 or Q4 departs from the original, record those as adaptations 3 and 4 beside your two?

Other choices:

- **Q6. Carey's place in the cells and the matrix.** This plan puts Carey in the "triggered" set for the
  neither-ripple-nor-burst cell (it is LFP × rate triggered), and leaves it out of the SWR inventories of the
  "SWRs matched by neither decoder" cell and out of the SWR matrix, which the spec defines with four
  definitions. Keep, or add Carey as a fifth matrix column?
- **Q7. The memoryless cohort.** `published` only (the paper's definition), or also `replay_criteria` as a
  second named inventory?
- **Q8. Native counts for MUA and state-space.** The decode labels only speed-eligible events, so their native
  counts equal their eligible counts and their moving events are not stored. Report that with a note, or
  recompute the native MUA events from the cached marks (`get_multiunit_hse`) and check its eligible events
  equal the labels?
- **Q9. Review item 2's fix.** Still runs reach the neighboring samples (Task 3): exposure grows by at most
  2 ms at each end of a still run. Agree?
- **Q10. Dropped events' spans.** Remove each dropped event's span from its own detector's observed time, as
  the spec does for bridging events, so a Kay SWR dropped near an artifact does not make a Karlsson event read
  `unmatched` against Kay. Agree?
- **Q11. Script imports.** `scripts/README.md` says scripts do not import each other; the Carey runner would
  import the SWR runner's helpers (as `run_artifact_detection.py` already does) and the memoryless runner's
  `load_marks`. Accept, or move those helpers into the package first?

## Review

Before merging, dispatch an independent reviewer (Opus) against the diff. Confirm every task is implemented as
specified; the "not in this phase" list is honored; tests exercise the asserted behavior; no docstring or test
names a plan or phase; the docs of Task 9 are updated; and the R1/R3 gates were met on real data and recorded.
No merge into `master` or `memoryless-sequence-detection` without the lead's approval.
