"""Baseline test script for data loading refactoring verification.

This script captures the current outputs and performance metrics before refactoring.
It will be used to verify that refactored code produces identical results.

Test case: senor20201028_.nwb, epoch 08_r4
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders import load_data

# Test configuration
NWB_FILE_NAME = "senor20201028_.nwb"
EPOCH_NAME = "08_r4"
BASELINE_DIR = Path("test_outputs/baseline")
BASELINE_DIR.mkdir(parents=True, exist_ok=True)


def time_function(func, *args, **kwargs):
    """Time a function call and return result and elapsed time."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def save_baseline_data(data: dict, filename: str):
    """Save baseline data to pickle file."""
    filepath = BASELINE_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved baseline: {filepath}")


def summarize_data(data: dict) -> dict:
    """Create summary of data for comparison."""
    summary = {}

    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            summary[key] = {
                "type": "DataFrame",
                "shape": value.shape,
                "columns": list(value.columns),
                "dtypes": {col: str(dtype) for col, dtype in value.dtypes.items()},
                "index_name": value.index.name,
                "head": value.head(3).to_dict(),
                "tail": value.tail(3).to_dict(),
            }
        elif isinstance(value, dict):
            summary[key] = {
                "type": "dict",
                "keys": list(value.keys()),
                "nested_summary": {k: summarize_value(v) for k, v in value.items()},
            }
        elif isinstance(value, list):
            summary[key] = {
                "type": "list",
                "length": len(value),
                "element_types": [type(v).__name__ for v in value],
            }
        else:
            summary[key] = summarize_value(value)

    return summary


def summarize_value(value):
    """Summarize a single value."""
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": value.shape,
            "dtype": str(value.dtype),
            "mean": float(np.mean(value)) if value.size > 0 else None,
            "std": float(np.std(value)) if value.size > 0 else None,
        }
    elif isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": value.shape,
            "columns": list(value.columns),
        }
    elif isinstance(value, pd.Series):
        return {
            "type": "Series",
            "length": len(value),
            "dtype": str(value.dtype),
        }
    elif isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
        }
    else:
        return {
            "type": type(value).__name__,
            "value": str(value)[:100],
        }


def main():
    """Run baseline data loading and capture results."""
    print("=" * 80)
    print("BASELINE DATA LOADING TEST")
    print("=" * 80)
    print(f"NWB File: {NWB_FILE_NAME}")
    print(f"Epoch: {EPOCH_NAME}")
    print()

    # Full data load timing
    print("Loading full data...")
    data, full_load_time = time_function(load_data, NWB_FILE_NAME, EPOCH_NAME)
    print(f"✓ Full load_data() time: {full_load_time:.3f}s")
    print()

    # Save full data
    save_baseline_data(data, "full_data.pkl")

    # Create and save summary
    summary = summarize_data(data)
    save_baseline_data(summary, "data_summary.pkl")

    # Extract key components for verification
    print("Extracting key components for verification...")
    verification_data = {}

    # Position info
    if "position_info" in data:
        verification_data["position_info"] = data["position_info"].copy()
        print(f"  - position_info: {data['position_info'].shape}")

    # Well visits and trials (for reward detection verification)
    if "well_visits" in data:
        verification_data["well_visits"] = data["well_visits"].copy()
        print(f"  - well_visits: {data['well_visits'].shape}")
        print(f"    Rewards: {data['well_visits']['is_reward'].sum()} / {len(data['well_visits'])}")

    if "trials" in data:
        verification_data["trials"] = data["trials"].copy()
        print(f"  - trials: {data['trials'].shape}")
        print(f"    Rewarded trials: {data['trials']['is_reward'].sum()} / {len(data['trials'])}")

    # Task variables (for distance computation verification)
    if "task_variables" in data:
        verification_data["task_variables"] = data["task_variables"].copy()
        print(f"  - task_variables: {data['task_variables'].shape}")
        dist_to_goal = data["task_variables"]["dist_to_goal"]
        print(f"    dist_to_goal: mean={dist_to_goal.mean():.2f}, std={dist_to_goal.std():.2f}")

    # Spike data (for spike loading verification)
    if "spike_times" in data:
        verification_data["spike_times"] = data["spike_times"]
        total_spikes = sum(sum(len(st) for st in times) for times in data["spike_times"].values())
        print(f"  - spike_times: {len(data['spike_times'])} brain areas, {total_spikes} total spikes")
        for brain_area, times in data["spike_times"].items():
            print(f"    {brain_area}: {len(times)} units")

    if "spike_waveform_features" in data:
        verification_data["spike_waveform_features"] = data["spike_waveform_features"]

    # DIO events
    if "beam_breaks" in data:
        verification_data["beam_breaks"] = data["beam_breaks"].copy()
        print(f"  - beam_breaks: {len(data['beam_breaks'])} events")

    if "pump_events" in data:
        verification_data["pump_events"] = data["pump_events"].copy()
        print(f"  - pump_events: {len(data['pump_events'])} events")

    if "light_events" in data:
        verification_data["light_events"] = data["light_events"].copy()
        print(f"  - light_events: {len(data['light_events'])} events")

    # Save verification data
    save_baseline_data(verification_data, "verification_data.pkl")

    # Performance baseline
    performance = {
        "full_load_time": full_load_time,
        "nwb_file_name": NWB_FILE_NAME,
        "epoch_name": EPOCH_NAME,
    }
    save_baseline_data(performance, "performance.pkl")

    print()
    print("=" * 80)
    print("BASELINE CAPTURE COMPLETE")
    print("=" * 80)
    print(f"Baseline data saved to: {BASELINE_DIR}")
    print()
    print("Summary of loaded data:")
    print(f"  - Keys: {list(data.keys())}")
    print(f"  - Total load time: {full_load_time:.3f}s")

    return data, verification_data, performance


if __name__ == "__main__":
    main()
