"""Component-level timing to isolate performance bottlenecks.

Tests individual functions that will be optimized:
- DIO event loading
- Reward detection
- Distance computation
- Spike loading
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

# Test configuration
NWB_FILE_NAME = "senor20201028_.nwb"
EPOCH_NAME = "08_r4"


def time_component(name: str, func, *args, **kwargs):
    """Time a component and print result."""
    print(f"  {name}...", end=" ", flush=True)
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.4f}s")
    return result, elapsed


def test_dio_loading():
    """Test DIO event loading timing."""
    from continuum_swr_replay.data_loaders.events import load_dios
    from continuum_swr_replay.data_loaders.position import get_position_info
    from spyglass.common import PositionIntervalMap

    print("\n1. DIO Event Loading:")

    # Get position interval for time bounds
    position_interval_name = (
        PositionIntervalMap
        & {"nwb_file_name": NWB_FILE_NAME, "interval_list_name": EPOCH_NAME}
    ).fetch1("position_interval_name")

    position_data, _ = time_component(
        "Position data",
        get_position_info,
        NWB_FILE_NAME,
        EPOCH_NAME,
        position_interval_name
    )

    valid_times = (position_data["position_info"].index.to_numpy()[[0, -1]])

    dio_events, dio_time = time_component(
        "DIO events",
        load_dios,
        NWB_FILE_NAME,
        valid_times
    )

    print(f"    Beam breaks: {len(dio_events['beam_breaks'])}")
    print(f"    Pump events: {len(dio_events['pump_events'])}")
    print(f"    Light events: {len(dio_events['light_events'])}")

    return dio_time


def test_reward_detection():
    """Test reward detection timing."""
    from continuum_swr_replay.data_loaders import load_data

    print("\n2. Reward Detection (in well visits):")

    # Load minimal data needed
    data = load_data(NWB_FILE_NAME, EPOCH_NAME)

    # Extract components
    beam_breaks = data["beam_breaks"]
    pump_events = data["pump_events"]
    position_info = data["position_info"]
    track_graph = data["track_graph"]

    # Import and time the well visits creation
    from continuum_swr_replay.data_loaders.trials import make_well_visits_df

    well_visits, wv_time = time_component(
        "Well visits (includes reward detection)",
        make_well_visits_df,
        beam_breaks,
        pump_events,
        position_info,
        track_graph
    )

    print(f"    Total visits: {len(well_visits)}")
    print(f"    Rewarded: {well_visits['is_reward'].sum()}")

    # Isolate reward detection timing
    print("\n  Timing just the reward detection logic:")
    from continuum_swr_replay.data_loaders.trials import get_poke_in_poke_out_times, get_run_intervals, get_start_end_node_id

    # Recreate well_visits up to reward detection
    poke_times = get_poke_in_poke_out_times(beam_breaks)
    poke_times = pd.concat((
        poke_times,
        pd.DataFrame({
            "start_time": [position_info.index[0], position_info.index[-1] - 1e-4],
            "end_time": [position_info.index[0], position_info.index[-1] - 1e-4],
            "duration": [0.0, 0.0],
        }, index=[0, 0]),
    )).sort_values("start_time")
    poke_times = poke_times.assign(
        **get_start_end_node_id(position_info, track_graph, poke_times)
    ).set_index(["from_well", "to_well"])

    run_times = get_run_intervals(poke_times)
    run_times = run_times.assign(
        **get_start_end_node_id(position_info, track_graph, run_times)
    ).set_index(["from_well", "to_well"])

    well_visits_partial = pd.concat((poke_times, run_times)).sort_values("start_time").reset_index()
    well_visits_partial["is_run"] = well_visits_partial["from_well"] != well_visits_partial["to_well"]

    # Now time just the reward detection
    print("    Reward detection only...", end=" ", flush=True)
    start = time.perf_counter()
    is_reward = [
        pump_events.start_time.between(row.start_time, row.end_time).any()
        for row in well_visits_partial.itertuples()
    ]
    reward_time = time.perf_counter() - start
    print(f"{reward_time:.4f}s")

    print(f"    Well visits: {len(well_visits_partial)}, Pump events: {len(pump_events)}")
    print(f"    Operations: {len(well_visits_partial) * len(pump_events)}")

    return reward_time


def test_distance_computation():
    """Test distance to goal computation timing."""
    from continuum_swr_replay.data_loaders import load_data

    print("\n3. Distance to Goal Computation:")

    # Load data
    data = load_data(NWB_FILE_NAME, EPOCH_NAME)

    position_info = data["position_info"]
    trials = data["trials"]
    track_graph = data["track_graph"]

    from continuum_swr_replay.data_loaders.task_variables import dist_to_goal

    # Get first trial
    trial = trials.iloc[0]
    trial_pos = position_info.loc[trial.start_time:trial.end_time]

    # Build track_segment_id_to_edge mapping
    track_segment_id_to_edge = {
        track_graph.edges[edge]["edge_id"]: edge for edge in track_graph.edges
    }

    # Time distance computation for one trial
    dist_arr, dist_time = time_component(
        f"Distance computation (trial 0, {len(trial_pos)} timepoints)",
        dist_to_goal,
        trial_pos,
        trial,
        track_graph,
        track_segment_id_to_edge
    )

    print(f"    Mean distance: {np.mean(dist_arr):.2f}")
    print(f"    Time per point: {dist_time / len(trial_pos) * 1000:.3f}ms")

    # Estimate total time for all trials
    total_timepoints = sum(len(position_info.loc[t.start_time:t.end_time]) for t in trials.itertuples())
    estimated_total = dist_time / len(trial_pos) * total_timepoints
    print(f"    Estimated total for all trials ({len(trials)}): {estimated_total:.2f}s")

    return dist_time, estimated_total


def test_spike_loading():
    """Test spike loading timing."""
    from continuum_swr_replay.data_loaders.spikes import get_spike_data

    print("\n4. Spike Data Loading:")

    spike_data, spike_time = time_component(
        "Spike loading (all brain areas)",
        get_spike_data,
        NWB_FILE_NAME
    )

    for brain_area, times in spike_data["spike_times"].items():
        total_spikes = sum(len(st) for st in times)
        print(f"    {brain_area}: {len(times)} units, {total_spikes} spikes")

    return spike_time


def main():
    """Run all component timing tests."""
    print("=" * 80)
    print("COMPONENT-LEVEL TIMING ANALYSIS")
    print("=" * 80)
    print(f"NWB File: {NWB_FILE_NAME}")
    print(f"Epoch: {EPOCH_NAME}")

    results = {}

    try:
        results["dio_loading"] = test_dio_loading()
    except Exception as e:
        print(f"  ERROR: {e}")

    try:
        results["reward_detection"] = test_reward_detection()
    except Exception as e:
        print(f"  ERROR: {e}")

    try:
        results["distance_computation"] = test_distance_computation()
    except Exception as e:
        print(f"  ERROR: {e}")

    try:
        results["spike_loading"] = test_spike_loading()
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    for component, timing in results.items():
        if isinstance(timing, tuple):
            print(f"{component}: {timing[0]:.4f}s (estimated total: {timing[1]:.2f}s)")
        else:
            print(f"{component}: {timing:.4f}s")

    return results


if __name__ == "__main__":
    main()
