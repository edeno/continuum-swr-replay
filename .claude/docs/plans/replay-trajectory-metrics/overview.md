# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

Tier-1 (per-timebin, per-session `data_over_time`):

- **Two Tier-1 builders, not one.** The Non-Local and ContFrag scenarios have
  separate `data_over_time` builders (and separate `*_data_over_time.pkl`
  outputs, ~218 each). New per-timebin columns must be added to **both**:
  - `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py` —
    `get_data_over_time` (Non-Local, 4 state probs).
  - `src/continuum_swr_replay/analysis/contfrag/data_aggregation.py` —
    `get_contfrag_data_over_time` (ContFrag, 2 state probs; mirrors the HPC
    builder).
- `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:333-380` —
  `get_data_over_time` assembles the per-timebin DataFrame. New columns are
  added as keys in this dict. **Already present** (do not re-add):
  `Local_prob`, `No-Spike_prob`, `Non-Local_Continuous_prob`,
  `Non-Local_Fragmented_prob`, `map_estimate`, `map_speed`,
  `ahead_behind_distance`, `hpd`, `theta_phase`, `theta_power_zscore`,
  `SWR_zscore`, `MUA_zscore`.
- `src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py:121-261` —
  `get_map_speed`: the track-graph shortest-path machinery (`nx.shortest_path_length(..., weight="distance")`
  at line ~187, bin→node mapping at ~174-179). `step_distance` reuses this.
- `src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py:264-369` —
  `get_posterior_stats`: returns the per-timebin arrays consumed above.
- `scripts/run_decoding_bandit.py:179-201` — `save_results_and_model` writes
  `{epoch_identifier}_{threshold_int}_data_over_time.pkl`. Regeneration runs
  through this path.
- Raw signals available in the `data` dict (for the raw counterparts):
  `data["ripple_consensus_trace"]`, `data["multiunit_firing_rate"][brain_area]`,
  `data["theta_power"]` (created in `data_loaders/lfp.py`). Time-aligned to
  decode bins via the `_nearest` helper at `data_aggregation.py:312-322`.

Tier-3 (per-event rollups, event tables):

- `src/continuum_swr_replay/analysis/replay/tables.py:485-527` — section 7 of
  `build_event_table`, where conditional-posterior rollups are keyed by event
  and assigned. New trajectory rollups follow this exact pattern.
- `src/continuum_swr_replay/analysis/replay/tables.py:1351-1418` —
  `_event_table_columns`: the canonical Non-Local column list. New columns are
  appended here.
- `src/continuum_swr_replay/analysis/replay/uncertainty.py:28-204` — the
  rollup pattern to mirror (`_weighted_mean_by_event`, `posterior_peak_mean`,
  etc.).
- `src/continuum_swr_replay/analysis/replay/spatial.py:120-192` —
  `arm_adjacency` + `path_continuity` (exist; merged in by Phase 2).
- `src/continuum_swr_replay/analysis/replay/retrospective.py:429-468` —
  `compute_retrospective_metrics` (exists; merged in by Phase 2; `prev_arm`
  mirror added).
- `src/continuum_swr_replay/analysis/contfrag/tables.py:210-449` —
  `build_contfrag_event_table`; `_summarize_event_state` at 163-207;
  `_contfrag_event_table_columns` at 862-909.
- `src/continuum_swr_replay/analysis/replay/spatial_mapping.py:18-173` —
  `map_estimate_to_arm`/`_to_patch`, `arm_to_well`, `well_to_arm_set`.
- `src/continuum_swr_replay/data_loaders/constants.py:152-178` —
  `WELL_PATCH_MAPPING`, `TRACK_SEGMENT_TO_PATCH`. Trunk arms {6,7,8} also in
  `retrospective.py:79` (`_PATCH_TO_TRUNK_ARM`).

Left alone: the decoder/detector, the posterior computation, the event-label
windowing, and the existing event-table columns (all new columns are additive).

## Scope and dependency policy

### Goals

- One per-event row, per scenario, carrying the full trajectory-metric set
  from the spec, with **identical definitions** across Non-Local and ContFrag.
- Per-timebin signals computed once in Tier 1; rollups are pure functions of
  the Tier-1 columns plus the track graph.
- Merge the already-written `path_continuity` and retrospective metrics into
  the canonical tables rather than leaving them standalone.

### Non-Goals

- No new decoder runs — every metric derives from existing posteriors.
- No sequence/replay score or its shuffle significance, no
  `radius_of_gyration`, no experienced-vs-shortcut path — explicitly deferred
  (see spec "Considered and deferred").
- No changes to event-label windowing or the SWR-overlap classification.

### Dependency policy

No new dependencies. Uses existing `numpy`, `pandas`, `networkx`, `xarray`.

## Metrics

- **Correctness:** each rollup matches a hand-computed value on the simulated
  fixture (exact for counts/fractions; `rtol=1e-9` for continuous distances).
- **Parity:** a test asserts the trajectory-metric columns produced by
  `build_event_table` and `build_contfrag_event_table` have identical names
  and dtypes and are produced by the same shared functions.
- **Regeneration parity:** for one real session, all pre-existing
  `data_over_time` columns are bit-identical before/after Phase 1 (only new
  columns added).
- **Runtime:** Phase 1 per-session aggregation overhead < ~10% over current
  (step_distance reuses the already-computed all-pairs distance dict).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| `step_distance` / `path_length` inflated by single-bin MAP jumps | Report robust order statistics (median/p90/max); `path_length` interpreted alongside `continuous_fraction`; no thresholding (jumps are fragmented content). |
| Junction bins (`edge_id == -1`) corrupting arm-based rollups | Reuse the established junction-exclusion pattern from `map_estimate_to_arm` and the existing `n_junction_bins` handling. |
| Non-Local vs ContFrag rollups silently drifting | Single shared module (designs.md); parity test enforces identical column sets. |
| Regeneration changes existing columns unintentionally | Phase 1 captures a baseline `data_over_time` and asserts existing columns unchanged. |
| `data_over_time` weighting differs (Non-Local weights by `nonlocal_prob`; ContFrag uses uniform weights) | Rollup functions take an explicit `weights` argument; each table passes its own convention, as the existing posterior rollups already do. |

## Rollout Strategy

All new columns are additive; existing consumers are untouched. Phase 1
regenerates the 218-session `data_over_time` pkls in place after the
single-session smoke test confirms feasibility and column parity. No feature
flag, no deprecation window (CLAUDE.md default; confirmed additive-only).

## Open Questions

1. **`run_speed_baseline` definition** (denominator of `compression_factor`).
   Current best-answer: per-session 90th-percentile of the animal's head speed
   (`position_info` speed) — a robust, threshold-free "typical run speed."
   Computed as a per-session scalar column in Phase 1. Revisit if the executor
   or user prefers median-during-locomotion.
2. **`pause_fraction` bin unit.** Uses `PLACE_BIN_SIZE_1D = 2.0` cm
   (`parameters.py:90`) as the "stays within one bin" distance. Confirm this is
   the right place-bin reference for the 1D bandit environment.

## Estimated Effort

- Phase 1: ~120 LOC (one step-distance helper, ~4 new `data_over_time`
  columns, one per-session scalar) + a regeneration run.
- Phase 2: ~400 LOC (new `trajectory.py` rollup module + wiring into two table
  builders + column-list updates + `prev_arm` retrospective mirror) + tests.
