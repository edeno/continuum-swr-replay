# Detailed Refactoring Changes

## Phase 2.1: Extract DIO Loading Helper

### File: `src/continuum_swr_replay/data_loaders/events.py`

**Before** (lines 131-179): 48 lines of nearly identical code repeated 3 times

**After**: Extract helper function (adds ~15 lines, removes ~33 lines net savings)

```python
def _load_dio_event_type(
    nwb_file_name: str, pattern: str, valid_times: tuple[float, float]
) -> pd.DataFrame:
    """Load and process a specific type of DIO event.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    pattern : str
        SQL LIKE pattern for event names (e.g., "Poke%", "Pump%", "Light%")
    valid_times : tuple[float, float]
        (start_time, end_time) tuple defining valid time range

    Returns
    -------
    pd.DataFrame
        Processed DIO events with start/end times and event numbers
    """
    events = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]  # type: ignore[misc]
        for nwb_file in (
            DIOEvents & {"nwb_file_name": nwb_file_name} & f'dio_event_name LIKE "{pattern}"'
        ).fetch_nwb()
    }
    return pd.concat(
        [
            convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in events.items()
        ],
        axis=0,
    )


def load_dios(nwb_file_name: str, valid_times: tuple[float, float]) -> DIOEventData:
    """Load DIO events (beam breaks, pump events, lights)."""
    beam_breaks = _load_dio_event_type(nwb_file_name, "Poke%", valid_times)
    pump_events = _load_dio_event_type(nwb_file_name, "Pump%", valid_times)
    light_events = _load_dio_event_type(nwb_file_name, "Light%", valid_times)

    return {
        "beam_breaks": beam_breaks,
        "pump_events": pump_events,
        "light_events": light_events,
    }
```

**Impact**:
- Code reduction: ~33 lines (41% reduction)
- Maintainability: Single location for DIO loading logic
- Performance: No change (same operations)

---

## Phase 2.2: Unify Detector Configuration

### Files:
- `scripts/run_decoding_bandit.py` (lines 162-166)
- `src/continuum_swr_replay/parameters.py` (lines 95-99)

**Change**: Remove duplication in run_decoding_bandit.py, import from parameters.py

**Before** in run_decoding_bandit.py:
```python
DETECTOR_CLASSES = {
    "HPC": (NonLocalClusterlessDetector, "clusterless"),
    "mPFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
    "OFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
}
```

**After** in run_decoding_bandit.py:
```python
from continuum_swr_replay.parameters import DETECTOR_CONFIG

# Map detector config to actual classes
DETECTOR_CLASSES = {
    brain_area: (
        NonLocalClusterlessDetector if config[0] == "NonLocalClusterlessDetector"
        else NonLocalSortedSpikesDetector,
        config[1]
    )
    for brain_area, config in DETECTOR_CONFIG.items()
}
```

**Impact**:
- Eliminates duplicate definition
- Single source of truth in parameters.py
- Easier to maintain and extend

---

## Phase 3.1: Vectorize Reward Detection (HIGH IMPACT)

### File: `src/continuum_swr_replay/data_loaders/trials.py` (lines 250-256)

**Current Algorithm** (O(n_visits × n_pumps)):
```python
well_visits["is_reward"] = [
    pump_events.start_time.between(row.start_time, row.end_time).any()
    for row in well_visits.itertuples()
]
```

**Optimized Algorithm** (O(n_visits + n_pumps)):
```python
# Create interval index for efficient containment checks
well_visit_intervals = pd.IntervalIndex.from_arrays(
    well_visits['start_time'].values,
    well_visits['end_time'].values,
    closed='both'
)

# Check which well visit contains each pump event
pump_to_visit_mask = well_visit_intervals.contains(pump_events['start_time'].values[:, None])

# Mark visits that have at least one pump event
visit_has_reward = pump_to_visit_mask.any(axis=0)
well_visits["is_reward"] = visit_has_reward
```

**Complexity Analysis**:
- Old: For each of ~1000 visits, check all ~500 pump events = 500,000 comparisons
- New: Build interval index O(n_visits) + vectorized containment O(n_pumps × log(n_visits)) ≈ 4,500 operations

**Expected Speedup**: 100-300x  (from ~2.0s to ~0.01s)

---

## Phase 3.2: Optimize Distance Computation (HIGH IMPACT)

### File: `src/continuum_swr_replay/data_loaders/task_variables.py` (lines 185-221)

**Current Algorithm** (O(n_timepoints × graph_operations)):
```python
def dist_to_goal(position_info, trial, track_graph, track_segment_id_to_edge):
    return np.array([
        dist_to_goal_per_time(
            row.track_segment_id, row.projected_x_position, row.projected_y_position,
            trial.to_well, track_graph, track_segment_id_to_edge
        )
        for row in position_info.itertuples()
    ])
```

Each `dist_to_goal_per_time` call:
1. Copies the entire track graph
2. Adds a new node for current position
3. Computes shortest path

**Optimized Algorithm** (O(n_segments × shortest_path + n_timepoints)):

```python
def precompute_segment_distances(track_graph, goal_node_id, track_segment_id_to_edge):
    """Precompute distance from each segment midpoint to goal.

    Returns dict mapping segment_id -> (distance_from_start, distance_from_end)
    """
    segment_distances = {}

    for segment_id, (node1, node2) in track_segment_id_to_edge.items():
        # Distance if approaching from node1 side
        dist_from_node1 = nx.shortest_path_length(
            track_graph, source=node1, target=goal_node_id, weight="distance"
        )
        # Distance if approaching from node2 side
        dist_from_node2 = nx.shortest_path_length(
            track_graph, source=node2, target=goal_node_id, weight="distance"
        )
        # Edge length
        edge_length = track_graph.edges[(node1, node2)]["distance"]

        segment_distances[segment_id] = (dist_from_node1, dist_from_node2, edge_length)

    return segment_distances


def dist_to_goal_vectorized(position_info, trial, track_graph, track_segment_id_to_edge):
    """Vectorized distance computation using precomputed segment distances."""
    # Precompute once per trial
    segment_distances = precompute_segment_distances(
        track_graph, trial.to_well, track_segment_id_to_edge
    )

    distances = np.zeros(len(position_info))

    for idx, row in enumerate(position_info.itertuples()):
        segment_id = row.track_segment_id
        dist_node1, dist_node2, edge_length = segment_distances[segment_id]

        # Get position along edge
        node1, node2 = track_segment_id_to_edge[segment_id]
        node1_pos = np.array(track_graph.nodes[node1]["pos"])
        projected_pos = np.array([row.projected_x_position, row.projected_y_position])

        # Distance from start of segment
        dist_from_start = np.linalg.norm(projected_pos - node1_pos)

        # Interpolate between the two endpoint distances
        # If closer to node1: use dist_node1 + dist_from_start
        # If closer to node2: use dist_node2 + (edge_length - dist_from_start)
        dist_via_node1 = dist_node1 + dist_from_start
        dist_via_node2 = dist_node2 + (edge_length - dist_from_start)

        # Take minimum (shortest path)
        distances[idx] = min(dist_via_node1, dist_via_node2)

    return distances
```

**Complexity Analysis**:
- Old: 5000 timepoints × (graph copy + shortest path) = 5000 × ~10ms = 50s
- New: 12 segments × shortest path + 5000 × (interpolation) = 0.12s + 0.01s = 0.13s

**Expected Speedup**: 50-100x (from ~5.0s to <0.1s per trial)

---

## Phase 3.3: Optimize Spike Loading Queries

### File: `src/continuum_swr_replay/data_loaders/spikes.py` (lines 209-240)

**Current Approach**: Try-except with redundant query

**Before**:
```python
try:
    restriction = {..., "sorter_params_name": "clusterless_fixed", ...}
    marks = (UnitMarks & restriction).fetch_dataframe()
    # ... process marks ...
except ValueError:
    restriction = {..., "sorter_params_name": "default_clusterless", ...}
    marks = (UnitMarks & restriction).fetch_dataframe()
    # ... identical processing ...
```

**After** (Option 1 - Check existence first):
```python
restriction_base = {
    "nwb_file_name": nwb_file_name,
    "curation_id": 0,
    "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
    "preproc_params_name": "franklab_tetrode_hippocampus",
    "team_name": "ac_em_xs",
    "sorter": "clusterless_thresholder",
    "mark_param_name": "default",
}

# Try first parameter name
restriction = {**restriction_base, "sorter_params_name": "clusterless_fixed"}
if len(UnitMarks & restriction) > 0:
    marks = (UnitMarks & restriction).fetch_dataframe()
else:
    # Fall back to second parameter name
    restriction = {**restriction_base, "sorter_params_name": "default_clusterless"}
    marks = (UnitMarks & restriction).fetch_dataframe()

marks = [(mark.index.to_numpy(), mark.to_numpy()) for mark in marks]
spike_times_tuple, spike_waveform_features_tuple = zip(*marks, strict=False)
spike_times = list(spike_times_tuple)
spike_waveform_features = list(spike_waveform_features_tuple)
```

**Impact**:
- Eliminates redundant full data fetch for half of sessions
- Expected speedup: 2x for affected sessions (from ~4s to ~2s)

---

## Verification Requirements

For each change, we must verify:

1. **Exact Output Match**:
   - Boolean columns: Exact equality
   - Float columns: `np.allclose(rtol=1e-10, atol=1e-12)`
   - Integer columns: Exact equality
   - Index: Exact match

2. **Performance Improvement**:
   - Measure time before and after
   - Document speedup ratio
   - Verify no performance regression anywhere

3. **Edge Cases**:
   - Empty DataFrames
   - Single element arrays
   - NaN values
   - Boundary conditions

## Rollback Strategy

- Each phase is a separate git commit
- Can revert individual commits without affecting others
- Test each phase independently before proceeding
- Keep detailed log in REFACTORING_LOG.md
