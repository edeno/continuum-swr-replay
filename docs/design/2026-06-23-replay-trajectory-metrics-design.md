# Replay Trajectory Metrics — Design Spec

**Date:** 2026-06-23
**Status:** Design — pending implementation plan

## Goal

Characterize replay trajectories during immobility with a shared set of
per-event summary statistics that answer five questions:

1. **What is the replay doing?** — trajectory content and dynamics
2. **What is the animal doing?** — behavioral context
3. **What is its brain state?** — population / LFP state
4. **What is the uncertainty?** — decode confidence
5. **What is interesting from the replay literature?** — derived,
   hypothesis-driven measures

The metrics apply to **both** event scenarios with **identical
definitions** so the two are directly comparable:

- **Non-Local** scenario — events from the non-local detector
  (`build_event_table` in `analysis/replay/tables_event.py`)
- **ContFrag** scenario — SWR/MUA-detected events
  (`build_contfrag_event_table` in `analysis/contfrag/tables.py`)

## Analysis grain

Two layers:

- **Per-timebin** signals computed once and stored alongside the existing
  `data_over_time` / `event_sequence_table` per-timebin data.
- **Per-event rollup** — a canonical scalar summary per event, merged into
  the event tables. Rollups consume the per-timebin layer; they are not
  recomputed from the posterior.

## Design principles

- **Minimize hard thresholds.** Prefer robust order statistics (median,
  percentiles), topological adjacency (binary track-graph relations), and
  state-conditioning over arbitrary distance cutoffs.
- **Jumps are content, not artifacts.** A single-bin teleport across the
  track is exactly what the Fragmented state captures. We do not threshold
  jumps away; we condition on state and report robust statistics.
- **Distance is along the track graph**, never Euclidean. Use the existing
  `nx.shortest_path_length(..., weight="distance")` primitive.
- **Reuse existing primitives.** Replay speed, ahead/behind distance, HPD,
  path continuity, and the retrospective metrics already exist; this work
  rolls them up and merges them in rather than reinventing them.
- **The only threshold is the existing project-wide `non_local_threshold`
  (0.8)** for "what counts as non-local." No new thresholds are introduced.

## Reusable existing primitives

| Primitive | Location | Grain | Notes |
|---|---|---|---|
| `get_map_speed` | `analysis/hippocampus/posterior_analysis.py:121` | per-timebin | replay speed, track-graph, Gaussian-smoothed |
| `get_ahead_behind_distance` | `non_local_detector.analysis.distance1D` | per-timebin | signed cm, animal→decoded along track |
| `maximum_a_posteriori_estimate` | `non_local_detector.analysis.posterior` | per-timebin | decoded MAP linear position |
| `get_HPD_spatial_coverage` | `non_local_detector.model_checking` | per-timebin | 95% HPD spatial coverage (cm) |
| `map_estimate_to_arm` | `analysis/replay/spatial_mapping.py:60` | per-timebin | linear position → arm (junctions excluded) |
| `path_continuity` | `analysis/replay/spatial.py:154` | per-event | arm-adjacency fraction (exists, not yet merged) |
| `posterior_peak_mean` / `_entropy_mean` / `_hpd_width` | `analysis/replay/uncertainty.py:48-150` | per-event | confidence summaries |
| retrospective metrics | `analysis/replay/retrospective.py` | per-event | `ends_at_next`, `contains_path_to_next`, `chosen_arm_fraction` (exist, not yet merged) |

`step_distance` and the track-graph shortest-path distance matrix are the
only new per-timebin building blocks required.

## Metric catalog

Legend: **E** = exists today, **R** = robust rollup of an existing
per-timebin signal, **N** = new.

### 1. What is the replay doing? (trajectory dynamics)

**Per-timebin (compute once):**

- `map_estimate` (E) — decoded MAP linear position
- `decoded_arm` (E) — arm 0–8, junctions excluded
- `map_speed` (E) — replay speed (cm/s, track-graph, smoothed)
- `ahead_behind_distance` (E) — signed cm, animal→decoded
- `step_distance` (N) — track-graph |Δ map_estimate| between consecutive bins

**Per-event rollup:**

- `start_position`, `end_position` (N) — first / last in-event `map_estimate`
- `start_arm`, `end_arm` (N) — first / last in-event `decoded_arm`
- `net_displacement` (N) — track-graph distance(start_position, end_position)
- `path_length` (N) — Σ `step_distance`
- `replay_speed_median`, `replay_speed_p90` (R) — robust to >10⁴ cm/s jump bins
- `replay_speed_continuous_median` (R) — median `map_speed` over Continuous-state bins only (clean smooth-trajectory speed)
- `tortuosity` (N) — `path_length / net_displacement` (read alongside `continuous_fraction`)
- `step_distance_median`, `step_distance_p90`, `step_distance_max` (N) — jumpiness distribution, threshold-free
- `path_continuity` (E) — arm-adjacency fraction; merge in
- `spatial_extent` (N) — graph diameter of visited place bins (decoded range)
- `n_unique_arms` (E)
- `n_reversals` (N) — count of direction reversals along the trajectory
  (out-and-back vs. one-way); distinct from `tortuosity` (magnitude) and
  `ab_slope` (net direction)
- `n_patches_visited` (N) — count of distinct patches the decoded arms touch
  (via `TRACK_SEGMENT_TO_PATCH`)
- `crosses_trunk` (N) — whether the trajectory leaves the current patch and
  traverses a trunk arm (6/7/8) into another patch
- `pause_fraction` (N) — fraction of in-event steps where the decoded position
  stays within ~one place bin (`step_distance` <= place_bin_size); the
  place-bin size is a principled unit, not a tuned threshold

### 2. What is the animal doing? (behavioral context)

Mostly existing:

- `current_arm`, `current_patch`, animal linear position (E)
- `max_speed`, `mean_speed`, `start_speed`, `end_speed` (E) — **animal** head speed
- `time_since_stop`, `time_since_port_entry`, `time_since_reward` (E)
- `prev_arm`, `next_arm`, `last_rewarded_arm` (E)
- `is_after_switch`, `is_after_omission`, `is_after_reward` (E)
- `orientation` (N) — head orientation at event onset (cheap; already used internally for ahead/behind)

### 3. What is its brain state? (population / LFP)

- `local_prob_mean`, `continuous_prob_mean`, `fragmented_prob_mean`,
  `nospike_prob_mean` (N for Non-Local, partial E for ContFrag) — per-event
  means of all four state posteriors. **The Non-Local event table currently
  has none of these.**
- `nonlocal_fraction` (N) — fraction of bins non-local (w_t > 0.8)
- `nospike_fraction` (N) — fraction of bins in the No-Spike (silent) state
- `mua_rate` (N), `mua_rate_z` (N) — multiunit rate over the event, raw and
  per-session z-scored
- `ripple_power` (N), `ripple_power_z` (N) — consensus ripple amplitude over
  the event, raw and per-session z-scored
- `theta_power` (N), `theta_z` (N) — theta-band power over the event, raw and
  per-session z-scored; distinguishes attentive vs. quiescent immobility
- `is_swr`, `overlap_class`, `overlap_subclass`, `matched_swr_*` (E)

Z-scores use a per-session baseline (mean/std over the full epoch time axis),
computed at the per-timebin tier before event slicing so the baseline is not
contaminated by event-only selection.

### 4. What is the uncertainty? (decode confidence)

- `posterior_peak_mean`, `posterior_entropy_mean` (E)
- `posterior_hpd_width_50`, `_80`, `_95` (E) — HPD width in arm count
- `hpd_coverage_cm` (R) — per-event rollup of the per-timebin spatial HPD coverage
- `n_junction_bins` (E)
- `multimodal_fraction` (N) — fraction of bins whose 95% HPD set spans
  **non-graph-adjacent** arms (topological, threshold-free)

`confident_fraction` was considered and **dropped**: peak/entropy already
carry confidence continuously, and a "confident" cutoff would need an
arbitrary threshold.

### 5. What is interesting from the replay literature?

- **Remote vs. local** (Karlsson & Frank 2009): `mean_abs_ahead_behind` (N),
  `remote_fraction` = fraction of bins where `decoded_arm != current_arm` (N)
- **Reach / sweep distance** (Pfeiffer & Foster 2013): `max_ahead`,
  `max_behind`, `max_abs_ahead_behind` (N) — farthest the decoded content
  projects from the animal
- **Termination at a well** (N): `terminates_at_well` — whether the trajectory
  ends at a degree-1 well terminal vs. mid-arm or junction; complements
  `ends_at_next` (which arm) by capturing whether the endpoint is a reward site
- **Direction — geometric** (Foster & Wilson 2006; Diba & Buzsáki 2007):
  `ab_slope` (N) = least-squares slope of `ahead_behind_distance` over the
  event; `is_outbound` (N) = sign of `ab_slope` (positive = sweeping away
  from animal, negative = returning). Always defined.
- **Direction — path-relative** (reuse retrospective machinery):
  - prospective (upcoming path): `ends_at_next`, `contains_path_to_next`,
    `chosen_arm_fraction` (E) — merge in
  - retrospective (just-taken path): mirror set toward `prev_arm` (N)
- **Compression factor** (N): `replay_speed_median` / animal's typical run
  speed (per-session baseline)
- **Initiation locality** (N): track-graph distance from `start_position` to
  the animal's position — does the trajectory begin near the animal then
  move away
- **Sequenceness vs. fragmentation**: `continuous_fraction` + `path_continuity`
  read together (E)

## Direction definitions (resolved)

Two complementary, non-redundant metrics:

- **Geometric** — `ab_slope` / `is_outbound`: sign and magnitude of the
  trend in `ahead_behind_distance`. Answers "is decoded content moving
  outward or returning," always defined regardless of where on the track the
  trajectory sits.
- **Path-relative** — prospective and retrospective alignment via the
  existing retrospective metrics (and a `prev_arm` mirror). Answers "does the
  content match where the animal is going / came from," which is the
  behaviorally meaningful axis for the bandit task.

## Threshold resolution (resolved)

| Candidate metric | Resolution |
|---|---|
| `jump_fraction` | Dropped. Replaced by `step_distance` {median, p90, max} (threshold-free) + `path_continuity`. |
| `multimodal_fraction` | Redefined topologically (non-adjacent arms in HPD set) — threshold-free. Kept. |
| `confident_fraction` | Dropped. `posterior_peak_mean` / `_entropy_mean` cover confidence continuously. |
| replay speed jump artifacts | Use median / p90 order statistics (robust); optionally Continuous-state-only speed. No cutoff. |
| `tortuosity` | Kept continuous; interpreted alongside `continuous_fraction`. |
| "non-local" cutoff | Existing project-wide `non_local_threshold = 0.8`. No new threshold. |

## Storage architecture (resolved)

Three tiers, one source of truth per tier, separated by what each signal is a
function of:

- **Tier 1 — per-timebin, per-session → `data_over_time`** (source of truth).
  Signals defined on the full session time axis, independent of event
  labeling, computed once per session at aggregation time:
  - existing: `map_estimate`, `map_speed`, `ahead_behind_distance`, `hpd`
  - new: `step_distance`, the four per-bin state probabilities
    (`local`/`continuous`/`fragmented`/`nospike`), `ripple_power`(+`_z`),
    `mua_rate`(+`_z`), `theta_power`(+`_z`)
- **Tier 2 — per-event per-timebin → `event_sequence_table`** (a view, not a
  recomputation). The in-event slice keyed by `event_id`, carrying whichever
  Tier-1 columns the rollups consume. Both scenarios slice the *same* Tier-1
  signals, which is what keeps their definitions from drifting.
- **Tier 3 — per-event scalars → event tables** (the rollups). The summary
  columns, merged into `build_event_table` and `build_contfrag_event_table`.

Rationale: computing per-timebin signals once on the session axis avoids
recomputation drift between scenarios, lets `ripple_power`/`mua_rate` z-score
against a well-defined per-session baseline, and keeps the event tables thin.
Cost: adding Tier-1 columns requires regenerating the `data_over_time` pkls
(a one-time pipeline rerun).

## Scope and structure

- Compute the new per-timebin signals (`step_distance`, per-bin state probs,
  ripple/MUA + z-scores) in Tier 1 alongside the existing per-timebin pipeline.
- Add per-event rollup functions, mirroring the structure of
  `analysis/replay/uncertainty.py`, returning scalar columns.
- Merge `path_continuity` and the retrospective metrics into the canonical
  event tables (they exist but are currently standalone).
- Apply identical definitions in both `build_event_table`
  (`analysis/replay/tables.py`) and `build_contfrag_event_table`
  (`analysis/contfrag/tables.py`); factor shared logic so the two scenarios
  cannot drift.

## Out of scope

- Cell-assembly / coactivity replay measures (clusterless decoding).
- `nominal_reward_prob` and contingency-block predictors (deferred elsewhere).
- New decoder runs — all metrics derive from existing posteriors.

## Considered and deferred

Recorded so their absence is a decision, not an oversight:

- **Sequence / replay score** (rank-order or weighted correlation of decoded
  position vs. time) and its **shuffle-based significance** — the canonical
  replay-coherence measures. Deferred to a follow-up; `path_continuity` and
  `continuous_fraction` serve as coherence proxies in the first pass.
- **`radius_of_gyration`** — dispersion of decoded positions about their
  centroid; `spatial_extent` covers extent for now.
- **Experienced vs. novel/shortcut path** (Gupta et al. 2010) — needs
  cumulative behavioral occupancy history; deferred.
- **`replay_speed_std`** — speed variability; dropped as niche.

## Open items deferred to the implementation plan

- Naming convention to keep Non-Local and ContFrag columns identical.
- Whether the `data_over_time` pkl regeneration is a separate migration step
  or folded into the metric build.

## Known caveats (2026-06-24)

- **`path_length` / `step_distance_max` / `_p90` include one event-boundary
  step.** `step_distance` is the session-level per-bin MAP displacement, so the
  first in-event bin's step crosses the event boundary (its predecessor lies
  just before the event). Matches `path_length = Σ step_distance`; events are
  mostly contiguous so it is one step out of ~10–25. Refinement (mask the first
  in-event step, or re-derive distances strictly between consecutive in-event
  bins) deferred.
- **Canonical tables predate the `pause_fraction` NaN-aware fix.** The
  `replay_content/*.pkl` regenerated 2026-06-24 carry the pre-fix
  `pause_fraction` (NaN `step_distance` bins counted as not-paused). This only
  differs for events with NaN in-event `step_distance` bins (rare — junctions
  never win the MAP argmax inside confident events). Re-regenerate to pick up
  the fix when convenient.
