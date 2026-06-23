# Designs — shared rollup module `analysis/replay/trajectory.py`

[← back to PLAN.md](PLAN.md)

This is the single home for the per-event rollup logic that **both**
`build_event_table` and `build_contfrag_event_table` call (Phase 2). The
scientific definitions live in the
[spec](../../../../docs/superpowers/specs/2026-06-23-replay-trajectory-metrics-design.md);
this file is the implementation contract.

## Module contract

Every rollup family is a pure function over per-timebin arrays already
filtered to in-event rows, grouped by `event_id`, mirroring
`uncertainty._weighted_mean_by_event` (`uncertainty.py:28-45`): it returns
`(unique_ids_sorted, {column_name: per_event_array})` so the caller assigns
columns exactly as section 7 of `build_event_table` does
(`tables.py:485-527`). `unique_ids_sorted` MUST match `np.unique(event_id)`,
which equals the `groupby(sort=True)` order the table builders already assert
against.

Weighting is explicit: callers pass `weights` (Non-Local → `nonlocal_prob`;
ContFrag → `np.ones`), exactly as the existing posterior rollups do. Metrics
that are intrinsically unweighted (counts, fractions, order statistics over
the trajectory) ignore `weights`.

```python
# analysis/replay/trajectory.py

import numpy as np
import pandas as pd
import networkx as nx

#: returned by each family; caller assigns one DataFrame column per key
RollupResult = tuple[np.ndarray, dict[str, np.ndarray]]
```

## Helpers (non-obvious)

### `_linear_to_node(linear_pos, env)`

Maps a MAP linear position (cm) to the nearest **real** place-bin-center node,
excluding junction bins (`edge_id == -1`) — same exclusion as
`map_estimate_to_arm` (`spatial_mapping.py:60-95`). Used by `net_displacement`,
`spatial_extent`, and `initiation_distance`.

```python
def _linear_to_node(linear_pos: float, nodes_df: pd.DataFrame) -> int | None:
    """Nearest real-bin node_id to a linear position; None if linear_pos is NaN.

    nodes_df is env.place_bin_centers_nodes_df_, pre-filtered to edge_id != -1.
    """
    if not np.isfinite(linear_pos):
        return None
    idx = (nodes_df["linear_position"].to_numpy() - linear_pos).__abs__().argmin()
    return int(nodes_df["node_id"].to_numpy()[idx])
```

### `_graph_dist(node_a, node_b, dist_lookup)`

`dist_lookup` is the all-pairs dict built once per call via
`dict(nx.shortest_path_length(env.track_graph_with_bin_centers_edges_, weight="distance"))`
— the same object `get_map_speed` builds (`posterior_analysis.py:185-188`).
Returns `np.nan` if either node is `None`.

## Rollup families

### `trajectory_dynamics(...) -> RollupResult`

Inputs (per in-event timebin, aligned): `map_estimate`, `decoded_arm`,
`step_distance` (from Tier 1), `map_speed`, `continuous_prob`, `time`,
`event_id`; plus `env`, `run_speed_baseline` (per-session scalar column),
`place_bin_size`.

Per event (group by `event_id`, in time order):

| Column | Definition |
| --- | --- |
| `start_position`, `end_position` | first / last finite `map_estimate` |
| `start_arm`, `end_arm` | first / last finite `decoded_arm` |
| `net_displacement` | `_graph_dist(_linear_to_node(start_position), _linear_to_node(end_position))` |
| `path_length` | `np.nansum(step_distance)` over in-event bins |
| `tortuosity` | `path_length / net_displacement` (NaN if `net_displacement == 0`) |
| `replay_speed_median`, `replay_speed_p90` | `np.nanmedian` / `np.nanpercentile(.,90)` of `map_speed` |
| `replay_speed_continuous_median` | median `map_speed` over bins where `continuous_prob` is the modal non-local state (see note) |
| `step_distance_median/_p90/_max` | order statistics of `step_distance` |
| `spatial_extent` | max pairwise `_graph_dist` over the set of visited real-bin nodes |
| `n_reversals` | count of sign changes in the signed along-track increment (see code) |
| `pause_fraction` | `mean(step_distance <= place_bin_size)` over in-event steps |
| `n_patches_visited` | `len(unique(TRACK_SEGMENT_TO_PATCH[a] for a in decoded_arm if finite))` |
| `crosses_trunk` | any decoded_arm in {6,7,8} **and** `n_patches_visited > 1` |
| `terminates_at_well` | `end_arm` is a leaf arm, i.e. `arm_to_well(end_arm) is not NaN` |
| `compression_factor` | `replay_speed_median / run_speed_baseline` |
| `initiation_distance` | `_graph_dist(_linear_to_node(start_position), animal_node)`, animal_node from the event's `track_segment_id`/position |

`n_reversals` (the one non-trivial loop):

```python
def _n_reversals(map_estimate: np.ndarray, dist_lookup, nodes_df) -> int:
    nodes = [_linear_to_node(p, nodes_df) for p in map_estimate]
    nodes = [n for n in nodes if n is not None]
    if len(nodes) < 3:
        return 0
    # signed step = +graph_dist if linear_position increased, else -graph_dist
    signs = np.sign(np.diff([nodes_df.set_index("node_id")
                             .loc[n, "linear_position"] for n in nodes]))
    signs = signs[signs != 0]
    return int(np.sum(np.diff(signs) != 0))
```

Note on `replay_speed_continuous_median`: "continuous bins" = in-event bins
where `Non-Local_Continuous_prob > Non-Local_Fragmented_prob`. Both columns are
already in `data_over_time`.

### `brain_state_means(...) -> RollupResult`

Plain per-event means of existing `data_over_time` columns (no graph logic):

| Column | Source column / definition |
| --- | --- |
| `local_prob_mean` | mean `Local_prob` |
| `continuous_prob_mean` | mean `Non-Local_Continuous_prob` |
| `fragmented_prob_mean` | mean `Non-Local_Fragmented_prob` |
| `nospike_prob_mean` | mean `No-Spike_prob` |
| `nonlocal_fraction` | `mean(nonlocal_prob > non_local_threshold)` |
| `nospike_fraction` | `mean(No-Spike_prob == modal state)` (argmax over 4 states) |
| `ripple_power`, `ripple_power_z` | mean `ripple_power` (raw, Phase 1) / mean `SWR_zscore` |
| `mua_rate`, `mua_rate_z` | mean `mua_rate` (raw, Phase 1) / mean `MUA_zscore` |
| `theta_power`, `theta_z` | mean `theta_power` (raw, Phase 1) / mean `theta_power_zscore` |

ContFrag already computes `continuous_prob_mean` / `fragmented_prob_mean` via
`_summarize_event_state` (`contfrag/tables.py:163-207`). Phase 2 replaces that
bespoke pair with this shared family so all four states are produced
identically; `dominant_state` / `continuous_fraction` (ContFrag-only, not in
the shared set) stay where they are.

### `ahead_behind_summary(...) -> RollupResult`

Inputs: `ahead_behind_distance`, `decoded_arm`, `current_arm` (per-event
scalar broadcast), `time`, `event_id`.

| Column | Definition |
| --- | --- |
| `mean_abs_ahead_behind` | `nanmean(abs(ahead_behind_distance))` |
| `max_ahead` | `nanmax(ahead_behind_distance)` (clip to ≥0; else NaN) |
| `max_behind` | `-nanmin(ahead_behind_distance)` (clip to ≥0) |
| `max_abs_ahead_behind` | `nanmax(abs(ahead_behind_distance))` |
| `remote_fraction` | `mean(decoded_arm != current_arm)` over finite decoded_arm |
| `ab_slope` | least-squares slope of `ahead_behind_distance` vs `time` (NaN if <2 finite points) |
| `is_outbound` | `ab_slope > 0` |

`ab_slope` uses `np.polyfit(t, ab, 1)[0]` on finite pairs per event.

### `uncertainty_extra(...) -> RollupResult`

| Column | Definition |
| --- | --- |
| `hpd_coverage_cm` | per-event `nanmean` of the Tier-1 `hpd` column |
| `multimodal_fraction` | fraction of in-event bins whose 95% HPD arm-set spans **non-adjacent** arms, using `arm_adjacency(track_graph)` (`spatial.py:120-151`); threshold-free (adjacency is topological) |

`multimodal_fraction` consumes the per-bin arm posterior already marginalized
in section 7 of `build_event_table` (`tables.py:506-508`); for ContFrag the
same marginalization at `contfrag/tables.py` ~398-427 feeds it.

## Merge-in functions (exist; wire in Phase 2)

- `path_continuity(decoded_arms, arm_adjacency(track_graph))` →
  `spatial.py:154-192`. Compute per event over the in-event `decoded_arm`
  sequence; assign as column `path_continuity`.
- `compute_retrospective_metrics(event_table, event_arm_table, event_sequence_table, include_patch=True)`
  → `retrospective.py:429-468`. Returns one row per `event_id` with
  `chosen_arm_fraction`, `ends_at_next`, `contains_path_to_next`,
  `chosen_patch_fraction`, `ends_at_next_patch`. Merge on `event_id`.
- **`prev_arm` mirror (new):** add a parallel set toward the just-taken path —
  `ends_at_prev`, `contains_path_from_prev`, `chosen_arm_fraction_prev` —
  by reusing the same internal path helpers (`arm_path_well_to_well`,
  `_PATCH_TO_TRUNK_ARM` at `retrospective.py:79`) with `prev_arm`/`prev_well`
  substituted for `next_arm`/`next_well`. Keep it in `retrospective.py` next
  to the prospective versions.
