# Phase 4 — Event features, the common spike-evidence diagnostic, and reasons for disagreement

[← back to PLAN.md](PLAN.md) · [spec 04](../../../../docs/design/replay_detection_comparison/04-event-features.md) ·
[spec 03](../../../../docs/design/replay_detection_comparison/03-inventory-comparison.md) ·
[overview](../../../../docs/design/replay_detection_comparison/overview.md) ·
[detector table](../../../../docs/design/replay_detection_comparison/detector_comparison.md)

Written 2026-10-18 from the lead's brief (`.superpowers/sdd/PLAN/phase4-plan-brief.md`). The spec wins
where this file and the spec disagree. Status lives in [PLAN.md](PLAN.md), not here.

## What this phase delivers

Spec 04 §"Initial report: default settings", for the 242 analysed epochs, at the retained default settings,
over the nine inventories of Phase 3 run `afeb897d850f` (`kay`, `karlsson`, `yu`, `zugaro`, `carey`, `mua`,
`state_space`, `memoryless`, `memoryless_criteria`). It answers Q1 of `docs/analysis_map.md` ("Are the
decoded non-local events real, including those outside SWRs?"), for every inventory alike.

1. **Per-event features**, the same rule on every inventory's intervals: timing, population engagement,
   LFP and MUA on one common trace each, spatial features in two namespaces (`ss_*`, `ms_*`), and context.
   How often each feature is missing is reported.
2. **The common spike-evidence diagnostic** on every compared event and on matched baseline windows, with
   spike counts, valid-row coverage and the unscorable intervals counted.
3. **The populations**: shared events, each method's unmatched events, and sequences matched by no SWR or
   MUA event, from Phase 3's membership, with unknown memberships kept apart.
4. **Reasons for disagreement**: the memoryless detector's rejection stage for every ours-only event (from
   its stored gates and candidates), with unknown comparisons kept apart; and the comparator-only checks.
5. **The report**, per session and per animal, with native populations first and duration strata beside
   them, and **sampled examples** by 03's rule.

Boundaries need nothing new. Phase 3's report already holds the signed onset and offset differences
(`boundary_offsets_*`), the agreement under stricter match rules (`match_strength_*`), one-to-many matches
(the `pairs` table) and the fragment-merge recount (`recount_*`). The Phase 4 report links to them.

**No claim about the Widloski columns** (`memoryless`, `memoryless_criteria`, and every `ms_*` feature, which
reads that detector's window posteriors) until the deferred Phase 2 pilot checks are done. Every such row,
column, caption and figure is marked "pending the Phase 2 pilot checks".

Not in scope: threshold robustness (Phase 5, spec 04 §"Planned follow-up"), trajectory-specific likelihood
(the spec allows it; none is reported, so its "evaluate on all intervals" rule has nothing to apply to),
held-out evidence (`inventory_comparison/heldout.py`, spec 05), and every control under 04 §"Other
follow-ups".

## Inputs to read first

- **The Phase 3 run**, `Processed-Data/replay_detection_comparison/inventory/afeb897d850f/`: 242 epochs `ok`;
  per epoch `{epoch_id}_<table>.parquet` for every table of `run_inventory_comparison.TABLE_DTYPES`
  (`inventory`, `set_aside`, `observations`, `pairs`, `directional`, `cells`, ...), read with
  `table_io.read_table` (:205); `manifest.json` (input runs: SWR `056eee0dcb5a`, Carey `fa6537e48bcf`,
  memoryless `d0964cc76ed2-seed20260922`, artifacts `fbbff144a123`; each epoch's decode sha256); `report/`.
  Phase 4 reads these tables and does not re-run Phase 3.
- `scripts/qa/run_inventory_comparison.py`: `ROSTER`, `DETECTORS`, `CELL_DETECTORS`, `CELLS`,
  `SWR_DEFINITIONS`, `SEQUENCE_COHORTS`, `TRIGGERED`, `MEMORYLESS_CRITERIA`, `SAMPLE_INTERVAL_S`, `MAX_GAP_S`,
  `decode_path` (:305), `output_path` (:592), and `compare_epoch` (:428), whose order of building the
  coverage Task 4 repeats. The runner computes `event_matches` (per event and other detector) but does not
  write it.
- `src/continuum_swr_replay/analysis/inventory_comparison/`: `availability` (`still_intervals` :231,
  `coverage_from_observations` :358), `overlap` (`Coverage` :160, `set_aside_events` :372, `event_matches`
  :462, `directional_agreement` :547, `MATCH_DTYPES`), `cells.membership_against` (:77), `intervals`
  (`intersect`, `subtract`, `union`, `contains`, `total_duration`), `schema.INVENTORY_DTYPES` (:44),
  `recount.FRAGMENT_GAPS_S` (:35).
- The spike-evidence code to reuse, `scripts/qa/analyze_event_spike_evidence.py`: `REMOTE_CM` (100 cm), its
  backend in `analysis/hippocampus/state_evidence/terms.py` (`load_encoding` :93, `encoding_inputs` :71,
  `nearest_interior_bins` :189, `interior_bin_mask` :225, `UNDERFLOW_OCCUPANCY_FLOOR` :256,
  `subnormal_occupancy_bin_mask` :283), and the library call it wraps,
  `non_local_detector.likelihoods.clusterless_kde.predict_clusterless_kde_log_likelihood(is_local=False)`.
  The script's own statistic (`dll_animal`, MAP against the animal) and its baseline (low non-local
  probability) are **not** the spec's; see "Conflicts" below.
- The grid convention of the memoryless likelihood, `memoryless_sequence/likelihood.py`:
  `clusterless_epoch_log_likelihood` (:328; row k holds the spikes in `[t_k, t_{k+1})`),
  `_drop_the_unreachable_row` (:440; the grid's last sample has no row), `interior_bin_distances_cm` (:415),
  `check_marks_match_encoding` (:217), `load_clusterless_encoding_model` (:151).
- The marks: `run_carey_detection.decode_marks` (:197), which reads `run_memoryless_detection.load_marks`
  (:423) from `replay_detection_comparison/memoryless/marks_cache/`; spikes on a grid by
  `swr_inventories.tetrode_spike_counts` (:799), the MUA rule of `pfc/glm.get_spikecount_per_time_bin` (:48).
- The memoryless run, `Processed-Data/replay_detection_comparison/memoryless/d0964cc76ed2-seed20260922/`:
  `{epoch_id}_memoryless_candidates.pkl` holds `detect.CandidateSequences` (:604: `runs`, `duration_s`,
  `dispersion_cm`, `is_replay`, `decisions`, `assembly`, `midpoint_s`, `config`), with `WindowDecisions`
  (:154: `center_index`, `spread_cm`, `jump_cm`, `gates`), `WindowGates` (:114: `posterior_usable`,
  `speed_eligible`, `spread_below_threshold`, `jump_below_threshold`, `has_center`, `kept`) and
  `SequenceAssembly` (:195: `runs`, `pieces`, `pre_merge_runs`, `kept`). `{epoch_id}_memoryless_events.parquet`
  has one row per candidate (`first_window`, `last_window`, `is_replay`, `whole_event_speed_eligible`,
  `shuffle_testable`, `shuffle_pass`, `p_spread`, `p_jump`, `n_shuffles`, `artifact_drop_reason`). Gate order:
  spec 02 §"Data flow, per epoch" steps 4-10. `statistics.dispersion_cm` (:278), `windows_holding` (:44),
  `NO_CENTER` (:32). The undecoded windows are computed as `generate_inventory_examples.load_epoch` (:337)
  computes `missing_windows`.
- The decode frame (`run_inventory_comparison.decode_path`, read through `swr_source.read_dot_with_swr_source`),
  on its 500 Hz grid: `Local_prob`, `No-Spike_prob`, `Non-Local_Continuous_prob`, `Non-Local_Fragmented_prob`,
  `map_estimate`, `hpd` (HPD spatial coverage, cm; `posterior_analysis.get_posterior_stats`),
  `linear_position`, `track_segment_id`, `patch_id`, `head_speed`, `SWR_zscore` (the Kay consensus over the
  tetrodes rule R-T kept, z-scored with the artifact mask left out, NaN inside it; nearest-sampled onto the
  grid; `swr_source.ripple_power_inputs` :448), `MUA_zscore` (the multiunit rate z-score, the mask left out of
  its normalization).
- Context: `Processed-Data/replay_content/stop_table.parquet` (`stop_id`, `stop_start`, `stop_end`, `at_well`,
  `occupancy_state`, `occupancy_well_id`, `current_patch`, `is_rewarded_outcome`, `last_well_arrival_time_s`;
  keyed by `animal`, `session_date`, `epoch`). An event belongs to the stop holding its onset, as
  `replay/cohort_selection.events_in_stops` (:533) and `stop_interval_mask` (:491) assign it.
- `hippocampus/event_detection.get_confident_times` (:48): our events are runs of non-local probability above
  0.8 lasting at least 15 ms (`minimum_duration=0.015`).
- `state_evidence/windows.keyed_rng` (:31), the seeded stream every draw here uses.
- The report and examples to reuse: `scripts/qa/report_inventory_comparison.py` (`read_run` :347,
  `with_pending` :423, `pool_sessions` :459, `_distribution` :509, `PENDING`, `_caption`) and
  `scripts/figures/generate_inventory_examples.py` (`draw_examples` :247, `load_epoch` :337,
  `recompute_windows` :415, `draw_example` :623, `detector_color` :175, `_track_key` :191,
  `ANIMAL_POSITION_COLOR`, `MEMORYLESS_COLOR`).
- Runner plumbing: `run_swr_detection` `RUN_ID_PATTERN` (:224), `json_ready` (:403), `write_json` (:907),
  `run_epochs` (:1102), `run_lock` (:1222); `decoding_common.provenance_record`, `write_atomically`;
  `table_io.write_table` (:156), `sha256_file` (:53).

## Shared conventions for every task

- **One grid.** Every interval of every inventory is read on the decode grid. Its **rows** are the grid
  samples with `event_start_s <= t <= event_end_s`. Every per-interval feature (spikes, likelihood,
  posterior, LFP, MUA, speed, position) is read over those rows and no others. An interval with no row
  (shorter than a sample, off the grid) is unscorable for every row-based feature and counted.
- **Spikes of a row.** Row k holds the spikes with `t_k <= s < t_{k+1}`, the rule of the decode, of
  `clusterless_epoch_log_likelihood` and of `tetrode_spike_counts`. The grid's last sample is never a
  row (`_drop_the_unreachable_row`): an interval ending on it has one row fewer, and one holding only it
  has no row (`no_row`). Six compared events of four real epochs end on it (`features._grid_rows`).
- **The compared events** are the `inventory` rows with `whole_event_speed_eligible` true, of all nine
  inventories (moving events and set-aside events are not compared, as in Phase 3). `memoryless` and
  `memoryless_criteria` share their published events, interval for interval.
- **Namespaces.** Posterior-derived features carry `ss_` (the state-space decode frame) or `ms_` (the
  memoryless run's window statistics), and each is computed for every inventory's events. No `cf_*`
  features (open question Q1). Features with no prefix (engagement, LFP, MUA, evidence, context) read no
  posterior.
- **Missing stays missing.** A feature that cannot be computed is NaN (or NA), never zero or a fill; its
  count is reported (Task 8).
- **Pooling.** Counts and seconds are summed per animal over its available sessions, then divided
  (`pool_sessions`); proportions are never averaged. Distributions are reported as quantiles
  (`_distribution`), per session and pooled per animal.
- **Outputs** go under `PROCESSED_DATA_DIR / "replay_detection_comparison" / "event_features" / <run_id>/`
  (`parameters.py:84`; `CSR_PROCESSED_DATA_DIR` overrides it), figures under
  `paper/figures/event_features/<run_id>/`. In the worktree `Processed-Data` is a symlink to the main
  tree's.
- **Names.** No docstring, test or module names a plan, phase or task number, except the phrase "pending the
  Phase 2 pilot checks".
- **Read-only inputs.** Spyglass, the `ripple_detection` package, `swr_source.py`, and every earlier run
  are not changed.

## Tasks

### Task 1 — Interval rows, engagement, LFP and MUA, timing, context

- **Files.** Create `src/continuum_swr_replay/analysis/inventory_comparison/features.py`; list it in the
  package `__init__` docstring. Tests in `tests/unit/test_inventory_features.py`.
- **Consumes.** `tetrode_spike_counts`; `intervals.contains`; `cohort_selection.stop_interval_mask`.
- **Produces.**
  - `interval_rows(time, starts, ends) -> (first_row, n_rows)`: arrays, `n_rows == 0` where no sample lies
    in the closed interval.
  - `RowTraces`: a frozen dataclass of per-row arrays for one epoch, each shape `(n_time,)` unless noted:
    `time`, `spike_counts` `(n_time, n_tetrodes)`, `ripple_z` (`SWR_zscore`), `mua_z` (`MUA_zscore`),
    `linear_position`, `track_segment_id`, `patch_id`, and Task 2's `evidence` and Task 3's state columns.
    `RowTraces.from_frame(dot, spike_times)` builds it.
  - `interval_features(traces, intervals)`: one row per interval, read from cumulative sums over its rows:
    `first_row`, `n_rows`; `n_spikes`, `n_active_tetrodes`, `n_tetrodes`, `spike_rate_hz`
    (`n_spikes / (n_rows * SAMPLE_INTERVAL_S)`); `ripple_z_max`, `ripple_z_mean`, `mua_z_max`, `mua_z_mean`
    (NaN when any row is NaN, so a masked row is never skipped silently); `linear_position_cm`,
    `track_segment_id`, `patch_id` at the middle row.
  - `neighbour_gaps(inventory)`: `previous_gap_s`, `next_gap_s`, end-to-start, to the nearest compared event
    of the same inventory in the same epoch; NaN at the epoch's first and last event.
  - `stop_context(intervals, stops)`: by onset, `stop_id`, `at_well`, `occupancy_state`, `occupancy_well_id`,
    `current_patch`, `is_rewarded_outcome`, `time_in_stop_s` (onset − `stop_start`), `time_since_arrival_s`
    (onset − `last_well_arrival_time_s`); NA when the onset is in no stop.
  - `DURATION_STRATA_S = (0.0, 0.050, 0.100, 0.200, 0.500, inf)` and `duration_stratum(interval_span_s)`
    (Q6).
- **Tests first** (synthetic grid at 500 Hz, hand-placed spikes):
  - an interval between two samples has no rows; a bound exactly on a sample includes it;
  - a spike at `t_k` counts in row k, a spike just before `t_{k+1}` in row k, and one at `t_{k+1}` in row
    k+1; active tetrodes counts columns with a spike;
  - **identical intervals from two inventories give identical rows of features** (bit for bit);
  - a NaN `SWR_zscore` row makes `ripple_z_*` NaN for every interval that holds it, and only those;
  - gaps: three events give NaN, two gaps, NaN; a different inventory's event does not enter;
  - an onset on a stop's closed bound is in that stop; an onset in no stop gives NA context.
- **Accepts.** Tests pass; `ruff` and `mypy` clean on the module.

### Task 2 — The common spike-evidence diagnostic

- **Files.** Create `src/continuum_swr_replay/analysis/inventory_comparison/evidence.py`. Tests in
  `tests/unit/test_inventory_evidence.py`, on `tests/unit/_memoryless_clusterless_epoch.build_simulated_epoch`
  (a real clusterless KDE encoding model on a Y track).
- **Consumes.** `encoding_inputs`, `nearest_interior_bins`, `interior_bin_mask`,
  `subnormal_occupancy_bin_mask`, `interior_bin_distances_cm`, the library likelihood; `REMOTE_CM` imported
  from `analyze_event_spike_evidence` and recorded in the manifest.
- **Produces.**
  - `log_likelihood_blocks(time, rows, linear_position, spike_times, spike_waveform_features, encargs)`:
    yields `(first_row, L)` per contiguous run of the requested rows, `L` shape `(n_rows, n_interior_bins)`,
    computed by one library call on the run padded by one sample before and two after, the pads dropped,
    so row k holds exactly the spikes in `[t_k, t_{k+1})` whatever the blocking. Long runs are split into
    blocks of a fixed number of rows to bound memory.
  - `remote_evidence(L, animal_bin, remote, usable)`: per row, `max over remote usable bins of L −
    L[animal_bin]`; NaN where the animal's position is not finite, its bin is not usable, or no usable bin is
    remote. `remote` is `interior_bin_distances_cm > REMOTE_CM` (the script's "above"), `usable` is
    `~subnormal_occupancy_bin_mask` (bins where a CPU rebuild is not the decode).
  - `epoch_row_evidence(...)`: the per-row evidence over the requested rows (NaN elsewhere), the arrays
    `RowTraces.evidence` holds.
  - `interval_evidence(traces, intervals)`: `evidence_dll` (mean over the valid rows), `evidence_n_valid_rows`,
    `evidence_valid_fraction` (valid over `n_rows`), `evidence_unscorable` (`"no_row"`, `"no_valid_row"` or
    `""`). Spike counts come from Task 1 over the same rows.
- **Rows computed.** The union of every compared event's rows and the rows of the `all` comparison's shared
  exposure (the baseline pool, Task 5). Nothing else is decoded.
- **Tests first:**
  - block rows equal one call on the whole simulated grid (row k from `[t_k, t_{k+1})`) to float tolerance,
    for blocks cut at a spike, beside a spike and at a window boundary (`BOUNDARY_SAMPLES`);
  - an interval's score does not depend on where the blocks are cut, nor on the other intervals requested;
  - a remote burst (spikes from fields over 100 cm from the animal) scores above zero; spikes from fields at
    the animal score below zero; a silent interval is scored (silence carries likelihood) and reported with
    zero spikes;
  - a row with NaN position, and a row whose animal bin is flagged subnormal, are invalid; coverage reports
    them; an interval with only such rows is `no_valid_row`;
  - a flagged bin never wins the remote maximum;
  - junction placeholder bins never enter (`interior_bin_mask` raises on a mismatch).
- **Accepts.** Tests pass; on the smoke epoch (R1) the evidence of every published memoryless event equals
  that of its `memoryless_criteria` twin exactly.

### Task 3 — Posterior features in two namespaces, and the comparator-only checks

- **Files.** Modify `features.py`. Tests in `tests/unit/test_inventory_features.py`.
- **Consumes.** The decode frame's state columns, `map_estimate`, `hpd`; the stored `CandidateSequences`;
  `statistics.dispersion_cm`, `NO_CENTER`, `windows_holding`; `nearest_interior_bins`,
  `interior_bin_distances_cm`.
- **Produces**, for every interval of every inventory:
  - `ss_features(traces, intervals, bin_distance_cm, map_bin, animal_bin)`: `ss_local_prob_mean`,
    `ss_no_spike_prob_mean`, `ss_continuous_prob_mean`, `ss_fragmented_prob_mean`, `ss_nonlocal_prob_max`,
    `ss_frac_rows_nonlocal` (non-local above `DEFAULT_NON_LOCAL_THRESHOLD`), `ss_extent_cm`
    (`dispersion_cm` of the rows' MAP bins), `ss_distance_cm_mean` (graph distance, MAP bin to animal bin),
    `ss_hpd_cm_mean`; and, for the comparator-only checks, `ss_n_runs_nonlocal` and
    `ss_longest_run_nonlocal_s` (runs of rows above 0.8 inside the interval; a run that formed no event is
    shorter than `get_confident_times`' 15 ms or crosses the interval's bound) and
    `ss_nearest_event_gap_s` (end-to-start gap to the nearest compared `state_space` event; NaN for
    `state_space` rows).
  - `ss_disagreement`: for events of any other inventory, `near` when `ss_nearest_event_gap_s <=
    max(FRAGMENT_GAPS_S)` (50 ms, the recount's widest gap), else `subthreshold` when `ss_n_runs_nonlocal > 0`,
    else `sustained`. Read only for events unmatched by `state_space` (Task 4); it describes, it does not
    decide.
  - `ms_features(candidates, intervals, undecoded, bin_distance_cm, animal_bin_at_window)`: over the windows
    whose midpoint (`midpoint_s`) lies in the interval: `ms_n_windows`, `ms_n_windows_decoded` (not
    undecoded), `ms_n_windows_with_center` (`gates.has_center`), `ms_spread_cm_mean` (`spread_cm` where the
    posterior was usable), `ms_extent_cm` (`dispersion_cm` of the centers), `ms_distance_cm_mean` (center to
    the animal's bin at the window's midpoint sample), `ms_frac_windows_kept` (`gates.kept`). The stored
    centers are after the speed gate, so a window whose midpoint speed is not below 4 cm/s has none: missing,
    not recomputed (Q10).
- **Tests first:**
  - identical intervals give identical `ss_*` and `ms_*`;
  - `ss_extent_cm` and `ms_extent_cm` equal `dispersion_cm` computed by hand on a three-bin path, and both
    use one distance matrix;
  - a MAP on a junction gap maps to a real bin (`nearest_interior_bins`); a junction bin never enters;
  - an interval shorter than the 4 ms stride has `ms_n_windows == 0` and every other `ms_*` NaN;
  - a comparator event 40 ms from our event reads `near`, one holding a 10 ms run above 0.8 reads
    `subthreshold`, one with probability below 0.8 throughout reads `sustained`;
  - an undecoded window (holding a masked sample) is counted apart and enters no mean.
- **Accepts.** Tests pass.

### Task 4 — The populations, from Phase 3's tables

- **Files.** Add `stored_coverage(observations, set_aside, still)` to `availability.py`; tests in
  `tests/unit/test_inventory_availability.py` and the integration test of Task 7.
- **Consumes.** `coverage_from_observations`, `set_aside_events`, `event_matches`, `directional_agreement`,
  `membership_against`; the runner's `still_intervals` call on the decode frame's `head_speed`
  (`DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC`, `MAX_GAP_S`).
- **Produces.** `stored_coverage` rebuilds an epoch's coverage from its stored `observations` and `set_aside`
  tables (dropped and bridging rows alike; subtracting a bridging span as `set_aside_events` does equals
  `remove_bridging_events`' subtraction) and the still intervals. The runner (Task 7) then recomputes
  `event_matches` over the stored compared events and pairs, with `ROSTER` as the roster, and writes it as
  `pair_membership`. The populations read from it and from the stored `cells`:

  | population | rows |
  |---|---|
  | `all` | every compared event of the detector |
  | `matched_by:<B>` / `unmatched_by:<B>` | pairwise membership against B; `unknown_by:<B>` counted apart, with why (B unavailable in the epoch, or the event not wholly inside the pair's shared exposure) |
  | `exactly_one_detector`, `swr_matched_by_no_decoder`, `sequence_matched_by_no_trigger` | the stored `cells`, membership `unmatched`; `matched` is the reference beside it, `unknown` counted apart |
  | `baseline` | Task 5's windows |

  "Shared events" are `matched_by:<B>`; "each method's unmatched events" are `unmatched_by:<B>` and the
  exactly-one cell; "sequences matched by no SWR or MUA event" are `sequence_matched_by_no_trigger` (Carey
  counts as an SWR definition, Q6 of Phase 3).
- **Tests first:** on the integration test's simulated epoch, the rebuilt coverage equals the coverage
  `compare_epoch` built, the recomputed `directional_agreement` equals the stored `directional` table, and
  `membership_against` over the recomputed matches equals the stored `cells`, all exactly; a set-aside event's
  span reads unobserved for its own detector only.
- **Accepts.** Tests pass; R1's gate below.

### Task 5 — Baseline windows

- **Files.** Create `src/continuum_swr_replay/analysis/inventory_comparison/selection.py` (baselines, Task 6's
  rejection stages). Tests in `tests/unit/test_inventory_selection.py`.
- **Consumes.** `intervals` arithmetic; the `all` comparison's shared exposure from Task 4's coverage;
  `keyed_rng`; Task 2's per-row evidence; `nearest_interior_bins`, `interior_bin_distances_cm`.
- **Produces.**
  - `baseline_pool(coverage, events)`: the `all` comparison's shared exposure (all nine inventories'
    exposures intersected: observed, outside excluded time, finite speed below 4 cm/s) less the union of
    **every** event of every inventory (compared, moving and set aside). The low non-local probability rule
    of the QA script is not used.
  - `matched_baselines(pool, events, traces, animal_bin, *, seed)`: for each compared event, candidate
    windows of exactly the event's `n_rows` consecutive grid rows lying wholly inside one pool interval.
    Level `location`: the animal's bin at the window's middle row within `LOCATION_TOLERANCE_CM` (10 cm)
    graph distance of the event's; used when at least `MIN_MATCHED` (20) such windows exist. Otherwise level
    `duration_only` (same epoch and length), when at least `MIN_MATCHED` exist; otherwise `failed`. Up to
    `N_BASELINES` (100) windows are drawn without replacement by `keyed_rng(BASELINE_SEED, event_key)`.
    Returns per event `baseline_level`, `baseline_n`, `baseline_median_dll`, `baseline_q95_dll` and
    `evidence_rank` (share of the event's baseline scores below its own), and the first draw of each event as
    a window (`matched_event_key`, bounds), which Task 7 runs through Tasks 1-3 as the `baseline` population.
- **Tests first:**
  - no window overlaps any event of any inventory, moving and set-aside ones included, or leaves the shared
    exposure; a window never spans a timestamp gap;
  - every window has the event's row count; a location-level window's animal bin is within 10 cm;
  - an event longer than every pool interval is `failed`; one whose location has 5 windows falls back to
    `duration_only` and says so;
  - draws are deterministic for a seed and key, and adding another event does not change an event's draws;
  - `evidence_rank` is 1 when the event beats every baseline and 0 when it beats none.
- **Accepts.** Tests pass; R1 reports the levels per detector.

### Task 6 — Rejection reasons for ours-only activity

- **Files.** Modify `selection.py`. Tests in `tests/unit/test_inventory_selection.py`.
- **Consumes.** `CandidateSequences` and the memoryless events table of the epoch; Task 4's
  `pair_membership` (state-space events against `memoryless` and against `memoryless_criteria`); the
  undecoded windows (`windows_holding` on the artifact mask, as `load_epoch`).
- **Produces** `rejection_stages(events, membership, candidates, candidate_table, undecoded, *, cohort)`: one
  row per state-space compared event and cohort (`memoryless`, `memoryless_criteria`). Over the windows whose
  midpoint lies in the event: `n_windows`, `n_undecoded`, `n_posterior_unusable`, `n_speed_ineligible`,
  `n_spread_failed`, `n_jump_failed`, `n_kept`, `n_kept_in_runs` (kept windows inside a `pre_merge_runs`
  run; the rest were single-window runs the detector dropped). Over the candidates overlapping the event
  (strict, positive duration, on the candidate's `event_start_s`/`event_end_s`): `n_candidates`, and flags
  `any_duration_failed` (`duration_s` not above 100 ms), `any_dispersion_failed` (not above 12 cm),
  `any_replay`, `any_replay_not_speed_eligible`, `any_shuffle_failed`, `any_shuffle_untestable`,
  `any_in_cohort`. Every reason is kept as its own count or flag. One exclusive `stage`, the furthest the
  detector got, in spec 02's order: `no_window`, `no_kept_window`, `no_run`, `no_candidate`, `duration`,
  `dispersion`, `speed`, then for `memoryless` `shuffle_failed` or `shuffle_untestable`, and `in_cohort`.
  An event whose membership is `unknown` gets stage `unknown` and `unknown_reason`, and no rejection reason
  (unknown comparisons are not rejections).
- **Tests first** (synthetic `WindowDecisions`, `SequenceAssembly` and candidate tables built with the
  detector's own `gate_windows`, `assemble_sequences` and `replay_mask` so gates and stages agree with the
  pipeline, spec 04 §Testing):
  - a window failing only spread counts under spread and stage `no_kept_window`;
  - a single kept window between rejected ones is `no_run`;
  - a lone 60 ms run is `no_candidate`; two such runs merged across a short gap form a candidate and reach
    `duration` or beyond;
  - a stationary candidate is `dispersion`; a moving-speed replay candidate is `speed`; an unshuffled or
    failing one is `shuffle_failed`;
  - an event matched by the cohort is `in_cohort` and an `unmatched` event never is;
  - an `unknown` event has no rejection reason.
- **Accepts.** Tests pass; R1 gate: for every state-space event, `stage == "in_cohort"` exactly when its
  membership against that cohort is `matched`.

### Task 7 — The per-epoch runner

- **Files.** Create `scripts/qa/run_event_features.py`. Tests in `tests/integration/test_run_event_features.py`,
  on the simulated epoch on disk of `tests/integration/test_run_inventory_comparison.py` (its fixtures
  `write_decode`, `write_carey_run`, `write_memoryless_run`, run first through the comparison runner), plus a
  fitted encoding model and marks for Task 2.
- **Consumes.** Tasks 1-6; the Phase 3 tables of `--inventory-run-id`; the memoryless run of the inventory
  run's manifest; `decode_marks` with `check_marks_match_encoding`; the stop table; `run_epochs`,
  `run_lock`, `json_ready`, `write_json`, `RUN_ID_PATTERN`.
- **Per epoch:** refuse the epoch unless the decode's sha256 equals the one the inventory run recorded; read
  the frame, the stored tables, the candidates, the marks (checked against the encoding model) and the stop
  rows; rebuild the coverage and the memberships (Task 4) and **fail the epoch unless they reproduce the
  stored `directional` and `cells`**; compute the rows (Task 2), the baselines (Task 5), the features of
  every compared event and every baseline window (Tasks 1-3), the rejection stages (Task 6).
- **Produces**, under `PROCESSED_DATA_DIR/replay_detection_comparison/event_features/<run_id>/` (run id: the
  first 12 characters of the commit), one atomic set per epoch:
  - `{epoch_id}_features.parquet`: the schema columns of every compared event and baseline window (detector
    `baseline`, `matched_event_key` naming its event), `duration_stratum`, and every feature of Tasks 1-3 and
    5, with declared dtypes held for an empty epoch;
  - `{epoch_id}_pair_membership.parquet`: `event_matches`, `MATCH_DTYPES`;
  - `{epoch_id}_rejection.parquet`: Task 6;
  - `manifest.json`: the inventory, memoryless, SWR, Carey and artifact run ids; the decode, stop-table and
    marks sha256; `REMOTE_CM`, the grid rule, the baseline constants and seed; tracking confidence recorded
    as unavailable (Q5); per epoch the status, counts (events per detector, scorable, unscorable by reason,
    baseline levels), wall clock and worker peak RSS. Main process writes it, atomically, after each epoch.
    A rerun skips `ok` epochs unless `--overwrite` and refuses other settings.
- **Runs** with `--inventory-run-id`, `--animal`, `--date`, `--epoch`, `--epoch-id`, `--n-workers`,
  `--overwrite`; dask `LocalCluster`, one epoch per worker, as the Carey runner. The device is the
  environment's (`JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`); the runner adds no flag for it.
- **Tests first:** the simulated epoch writes all three tables and a manifest record; its memberships
  reproduce the stored tables; a changed decode sha256 fails the epoch and writes nothing; an epoch whose
  memoryless run is missing writes features with `ms_*` NaN, memberships `unknown` and stages `unknown`;
  dtypes survive a write and read; a rerun skips `ok` epochs and refuses changed settings.
- **Accepts.** Tests pass; `--help` runs; R1's gates pass.

### Task 8 — Per-session and per-animal report

- **Files.** Create `scripts/qa/report_event_features.py`; tables to `<features run>/report/`, figures to
  `paper/figures/event_features/<run_id>/`. Tests in `tests/unit/test_report_event_features.py` on synthetic
  per-epoch tables.
- **Consumes.** The run's per-epoch tables; `pool_sessions`, `_distribution`, `with_pending`, `_caption`,
  `PENDING` from the Phase 3 report.
- **Produces** (each table per session and per animal):
  - `features_<group>_by_*` for the groups timing, engagement, lfp_mua, evidence, ss, ms, context: quantiles
    of each feature by detector and population (Task 4's), native populations; the pairwise populations at
    session level for B in `state_space` and `memoryless`, and for every B at animal level;
  - `features_<group>_by_stratum_by_*`: the same by `duration_stratum`, beside the native tables;
  - `missing_by_*`: per detector, population and feature, `n` and `n_missing`;
  - `evidence_coverage_by_*`: per detector, events, scorable, unscorable by reason, rows and valid rows
    summed, spikes summed;
  - `baseline_matching_by_*`: events by `baseline_level`, per detector;
  - `evidence_vs_baseline_by_*`: per detector and population, scorable events, events above their
    `baseline_q95_dll`, events at or below their `baseline_median_dll` (counts; no verdict);
  - `rejection_by_*`: per cohort, the exclusive `stage` counts and every reason's count, `unknown` apart with
    its reasons; `memoryless_cohort_by_*`: criteria and published counts, the shuffle's failures and
    untestable candidates (02's before and after);
  - `comparator_only_by_*`: per detector, its events unmatched by `state_space` by `ss_disagreement`, with
    their `ss_*` composition and evidence counts;
  - `report.md`, which lists every table with its caption, links Phase 3's `boundary_offsets_*`,
    `match_strength_*` and `recount_*` for the boundaries, and states what is unknown (tracking confidence;
    the Widloski columns pending). It states counts and writes no claim.
  - Figures: feature distributions by population for the two decoders' pairs, per animal; evidence against
    baseline per detector; rejection stages per animal; comparator-only classes per animal; evidence by
    duration stratum. Event colors from `parameters.COLORS` by family (`detector_color`); memoryless in
    `MEMORYLESS_COLOR` and marked †.
- **Tests first:** an animal row of two sessions equals the summed counts, not the mean of proportions; an
  `unknown` membership never enters an `unmatched` population; a missing feature is counted in `missing_by_*`
  and absent from its quantiles; every `ms_*` and memoryless row is marked pending; baseline rows never
  enter a detector's populations.
- **Accepts.** Tests pass; the report on the pilot (R2) is read by the lead.

### Task 9 — Sampled examples

- **Files.** Create `scripts/figures/generate_event_feature_examples.py`; figures to
  `paper/figures/event_features/<run_id>/examples/`. Modify `generate_inventory_examples.draw_examples` to
  take the column and the role mapping it draws by (defaults unchanged, so `test_inventory_examples.py`
  passes as it is). Tests in `tests/unit/test_event_feature_examples.py` for the sampler's inputs only.
- **Sampling rule** (03's, stated in captions and the manifest): per population, source detector and animal,
  2 events of the target subset and 2 counterexamples of the other subset, by `draw_examples` with a new run
  seed (`EXAMPLE_SEED = 20261018`); fewer when a subset is smaller; `unknown` never drawn.
  - **Comparator-only** (each detector of `CELL_DETECTORS` but `state_space`, its events `unmatched_by:state_space`):
    target "evidence above baseline" (`evidence_dll > baseline_q95_dll`), counterexamples "evidence at or
    below baseline" (`evidence_dll <= baseline_median_dll`).
  - **Ours-only** (`state_space` events of the exactly-one cell): target "evidence at or below baseline",
    counterexamples "evidence above baseline".
  - The two labels are descriptive (Q3); the caption says an excess over baseline is not a calibrated
    declaration of replay (spec 04).
- **Each figure** is `draw_example`'s panel set (spikes, both posteriors with the animal's linear position in
  magenta, `bone_r`, the linearized track by `plot_linear_track_graph_strip` with the decode's `edge_order`
  and `edge_spacing` and arms labeled, the gates, speed, ripple band, every inventory's bars) through
  `load_epoch` and `recompute_windows` (a figure whose recomputation disagrees is not written), with the
  event's `evidence_dll`, its baseline median and q95, `ss_disagreement` or the rejection `stage`, in the
  caption.
- **Tests first:** the sampler draws only the named subsets, deterministically, never `unknown`, fewer when
  small; Phase 3's sampler test passes unchanged.
- **Accepts.** One example per population renders on a pilot epoch.

### Task 10 — Docs that ship with the code

- `docs/analysis_map.md`: Q1 rows for `qa/run_event_features.py`, `qa/report_event_features.py` and
  `figures/generate_event_feature_examples.py`. Claim 23 (the fraction of SWRs with local content) stays
  under "No script" unless the lead decides the `ss_*` composition of the SWR inventories supports it.
- `scripts/README.md`: entries for the three scripts.
- `docs/design/replay_detection_comparison/04-event-features.md` header status, and `overview.md`'s spec
  table row, once the run is done. Words only; no number in the specs changes.
- `PLAN.md`'s phase row and status line after each stop point.

## Runbook for the real run

Machine: breeze (96 cores, 4 TB RAM; same `/home` and `/cumulus`). CPU first. Before each launch:

```bash
nproc; uptime; free -g; df -h /cumulus
ps -eo user,pcpu,pmem,args --sort=-pcpu | head -20   # other users' load
nvidia-smi                                           # only if GPU is needed
```

Pick a core range no other job is on and pin with `taskset -c <range>`; set `JAX_PLATFORMS=cpu`,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Run inside `tmux` from the worktree's `.venv`,
from a clean commit (the run id is the commit), under `/usr/bin/time -v`. The settings gate keys on the
inputs and rules, not the commit.

**GPU only if needed:** if R1's extrapolation exceeds about 8 h with the cores free, stop and ask the lead.
If agreed, use only idle cards, never 0, 4 or 8, with `CUDA_VISIBLE_DEVICES=<card>` and without
`JAX_PLATFORMS=cpu`; first rerun the smoke epoch on the GPU under another run id and compare its
`evidence_dll` with the CPU run's (they differ only in float rounding; the subnormal bins are left out on
both).

**R0. Before any run.** Q1-Q10 answered and the tasks merged into the phase branch; the suite green; the
inventory run `afeb897d850f` has 242 `ok` epochs; `swr_source.SWR_INVENTORY_RUN_ID` is `056eee0dcb5a` and
`ARTIFACT_INTERVALS_RUN_ID` is `fbbff144a123`.

**R1. Smoke, one epoch.** tmux `features_smoke`.

```bash
JAX_PLATFORMS=cpu taskset -c <range> /usr/bin/time -v .venv/bin/python scripts/qa/run_event_features.py \
    --inventory-run-id afeb897d850f --epoch-id chimi20200225_02_r1_HPC --n-workers 1
```

Gates: the epoch is `ok`; its recomputed memberships reproduced the stored `directional` and `cells`; the
features table has one row per compared event of each detector (the Phase 3 `counts.n_eligible`) and one per
baseline window; every published memoryless event's features equal its criteria twin's; no baseline window
overlaps an event or leaves the shared exposure; `stage == "in_cohort"` exactly for matched events; the
unscorable intervals and baseline levels per detector are printed. By eye: five ours-only and five
comparator-only events against Phase 3's example panels. Record wall clock, peak RSS and CPU % (`time -v`),
and the share of the wall clock spent in the likelihood (logged). Extrapolate: 242 x wall clock /
n_workers, with n_workers = min(free cores / measured cores per worker, free GiB / (peak RSS x 1.5)).
**Stop point:** report the numbers to the lead.

**R2. Pilot, five epochs.** tmux `features_pilot`. The five Phase 2 pilot epochs (PLAN.md "Phase 2 pilot",
one per animal), then the report (Task 8) and one example per population (Task 9) on them. **Stop point:**
the lead reads the pilot report.

**R3. Full run, 242 epochs.** tmux `features_full`. Gates: every epoch `ok`, or each failure explained; the
manifest's input runs are the ones named; per detector, features rows equal Phase 3's `n_eligible` summed.

**R4. Report and examples.** tmux `features_report`. Task 8 on the full run, then Task 9. **Stop point:** the
lead reviews the report and examples.

Outputs: `PROCESSED_DATA_DIR/replay_detection_comparison/event_features/<run_id>/` with `report/`; figures in
`paper/figures/event_features/<run_id>/` and `examples/`. Logs under `logs/`.

**Expected cost (an extrapolation, to be replaced by R1).** The likelihood dominates. Rows decoded: the
shared still exposure, 47 % of the 86.4 h of epochs (Phase 3's exposure, 144.8 ks), plus the events outside
it. The one CPU measurement of this library call here is the state-evidence run on
`chimi20200221_08_r4_HPC` (2026-10-17): per-tetrode terms for 159,107 spikes in 215 s, about 740 spikes/s
for one worker. An epoch has a few hundred thousand marks (chimi20200225_02_r1, about 358,000), of which the
still rows hold some part (not measured). That gives about 3-6 min per epoch on one worker, 12-24
worker-hours for 242 epochs, about 1-2 h on 12 workers, at about 7 GiB per worker (the Carey and
state-evidence runs' peaks, 6-7 GiB). The Phase 3 runner's 9.6 s median per epoch bounds the reading of the
frame and tables. The report should take minutes; the examples (about 160 figures) cost what Phase 3's do
per figure.

## Deliberately not in this phase

- Threshold robustness (Phase 5), and every follow-up control of 04 and 03 (native speed gates,
  running-only encoding, reference-channel ripple power, leave-one-tetrode-out).
- The Phase 2 pilot checks (deferred by the lead); no Widloski claim.
- `cf_*` features (Q1), trajectory-specific likelihood, held-out evidence.
- Tracking confidence (Q5).
- Any change to Phase 3's runner, its tables or run `afeb897d850f`, to `swr_source.py`, the
  `ripple_detection` package (which needs the lead first), or Spyglass.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_inventory_features.py` | closed-interval rows; the spike rule per row; identical intervals give identical features; NaN rows propagate; neighbour gaps; stop by onset; `ss_*`/`ms_*` on hand-worked paths; comparator-only classes |
| `test_inventory_evidence.py` | block rows equal one whole call; scores independent of blocking; remote above zero, local below; invalid rows and flagged bins; unscorable reasons |
| `test_inventory_availability.py` (updated) | `stored_coverage` equals the runner's coverage |
| `test_inventory_selection.py` | baseline pool and matching levels, deterministic draws, rank; rejection stages agree with synthetic gate failures built by the detector's own functions; unknown is not a rejection |
| `tests/integration/test_run_event_features.py` (slow) | one simulated epoch end to end; memberships reproduce the stored tables; a missing memoryless run gives `unknown`; reruns and settings gate |
| `test_report_event_features.py` | pooling sums counts; `unknown` never pooled as `unmatched`; missing counted; pending marks; baseline rows apart |
| `test_event_feature_examples.py` | sampler draws only the named subsets, deterministically |
| R1, R2 (real data, manual) | the gates listed in the runbook |

## Conflicts between the spec, the built code and the decisions

1. **"All seven detectors' shared observation exposure"** (04, baseline windows) predates Carey and Long's
   drop. The baseline pool uses the `all` comparison of the nine inventories (eight detectors;
   `memoryless_criteria` shares `memoryless`' coverage).
2. **The spec's statistic is not the QA script's.** The script scores `dll_animal`, the MAP against the
   animal, and uses `REMOTE_CM` to group events by mean `|ahead_behind_distance|`. The spec takes the best
   remote real bin, with `REMOTE_CM` as the remote-bin cutoff. Task 2 builds the spec's statistic on the
   script's backend. Its scores will not reproduce the script's numbers.
3. **The script's binning.** `event_likelihood` bins spikes on each window's own axis, so its last row holds
   no spike and a row's spikes depend on the call. Task 2 bins on the decode grid instead, so identical
   intervals give identical scores.
4. **No stored memoryless posteriors.** The run keeps per-window centers (after the speed gate), spreads and
   gates, not the posteriors. `ms_*` reads those; recomputing posteriors for every interval would repeat the
   decode (Q10).
5. **Tracking confidence** (04, context) is in neither the decode frame nor the stop table (Q5).
6. **Boundaries** (04) are already reported by Phase 3 (`boundary_offsets`, `match_strength`, the `pairs`
   table, `recount`); Phase 4 links them and adds no table.
7. **Phase 3 writes no `event_matches`**, which the pairwise populations need. Task 4 recomputes it from the
   stored tables and refuses an epoch unless it reproduces the stored `directional` and `cells`.
8. **"Supported" and "questionable"** (04, examples) are not defined in the spec; Task 9's definitions are
   Q3.

## Open questions for the lead

Each with the recommended answer.

- **Q1. `cf_*` features?** Recommend **no**. ContFrag decodes the same spikes with the same encoding model,
  without Local or No-Spike states, and serves the content tables (Q2-Q4), not the selection question here.
  `ss_*` gives our model's reading and `ms_*` the comparator's; a third reading answers no question of 04.
  If wanted later it is one more namespace on the same engine, from the ContFrag `data_over_time`.
- **Q2. Baseline constants.** Recommend `N_BASELINES = 100` per event, `LOCATION_TOLERANCE_CM = 10` (graph
  distance of the animal's bin at the middle row), `MIN_MATCHED = 20` for a level, and the fallback to
  `duration_only` reported per event. The per-row evidence makes extra windows nearly free.
- **Q3. Example subsets.** Recommend "evidence above baseline" = above the event's own `baseline_q95_dll`,
  and "at or below baseline" = at or below its `baseline_median_dll`, labelled as descriptions, not
  verdicts.
- **Q4. Ours-only for the rejection reasons.** Recommend state-space events `unmatched_by` each memoryless
  cohort, pairwise (not only the exactly-one cell), both cohorts reported.
- **Q5. Tracking confidence.** Recommend leaving it unavailable and saying so in the report; reading DLC
  likelihoods would need a Spyglass fetch per epoch.
- **Q6. Duration strata.** Recommend 0-50, 50-100, 100-200, 200-500 and over 500 ms of `interval_span_s`,
  beside the native populations.
- **Q7. The common ripple trace.** Recommend the decode frame's `SWR_zscore` (the Kay consensus of the R-T
  tetrodes, normalized outside the artifact mask), on the 500 Hz grid like every other feature. It is
  nearest-sampled from 1 kHz, so a peak can be one sample off; re-reading the stored band per epoch would
  avoid that at extra cost.
- **Q8. Device and workers.** Recommend CPU, about 12 pinned workers on breeze after R1; GPU only if R1
  extrapolates beyond about 8 h.
- **Q9. The `near` rule.** Recommend 50 ms, the recount's widest gap, so boundary differences are read on
  the scale Phase 3 already uses.
- **Q10. `ms_*` from stored windows.** Recommend the stored statistics; a window without a center is missing
  and counted.

## Review

Before merging, dispatch an independent reviewer (Opus) against the diff. Confirm every task is implemented
as specified; the "not in this phase" list is honored; tests exercise the asserted behavior; no docstring or
test names a plan or phase; the docs of Task 10 are updated; and the R1 gates were met on real data and
recorded. No merge into `master` or `memoryless-sequence-detection` without the lead's approval.
