"""Verification script to compare refactored code outputs against baseline.

This script loads the baseline data and compares it against newly loaded data
to ensure refactoring has not changed computational results.
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
RESULTS_DIR = Path("test_outputs/refactored")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline_data(filename: str):
    """Load baseline data from pickle file."""
    filepath = BASELINE_DIR / filename
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_results(data: dict, filename: str):
    """Save refactored results to pickle file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved results: {filepath}")


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, name: str, rtol=1e-10, atol=1e-12) -> bool:
    """Compare two DataFrames with detailed reporting."""
    try:
        # Check shapes
        if df1.shape != df2.shape:
            print(f"  ✗ {name}: Shape mismatch - baseline {df1.shape} vs new {df2.shape}")
            return False

        # Check columns
        if not df1.columns.equals(df2.columns):
            print(f"  ✗ {name}: Column mismatch")
            print(f"    Baseline: {list(df1.columns)}")
            print(f"    New: {list(df2.columns)}")
            return False

        # Check index
        if not df1.index.equals(df2.index):
            print(f"  ✗ {name}: Index mismatch")
            return False

        # Check values column by column
        all_match = True
        for col in df1.columns:
            if df1[col].dtype == np.bool_ or df2[col].dtype == np.bool_:
                # Boolean columns must match exactly
                if not (df1[col] == df2[col]).all():
                    diff_count = (~(df1[col] == df2[col])).sum()
                    print(f"  ✗ {name}.{col}: {diff_count} boolean mismatches")
                    all_match = False
            elif pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                # Numeric columns use np.allclose
                if not np.allclose(df1[col].values, df2[col].values, rtol=rtol, atol=atol, equal_nan=True):
                    max_diff = np.nanmax(np.abs(df1[col].values - df2[col].values))
                    print(f"  ✗ {name}.{col}: Max difference {max_diff:.2e} (rtol={rtol})")
                    all_match = False
            else:
                # Other types (strings, objects) must match exactly
                if not (df1[col] == df2[col]).all():
                    diff_count = (~(df1[col] == df2[col])).sum()
                    print(f"  ✗ {name}.{col}: {diff_count} value mismatches")
                    all_match = False

        if all_match:
            print(f"  ✓ {name}: Exact match")
        return all_match

    except Exception as e:
        print(f"  ✗ {name}: Comparison error - {e}")
        return False


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str, rtol=1e-10, atol=1e-12) -> bool:
    """Compare two numpy arrays."""
    try:
        if arr1.shape != arr2.shape:
            print(f"  ✗ {name}: Shape mismatch - baseline {arr1.shape} vs new {arr2.shape}")
            return False

        if not np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True):
            max_diff = np.nanmax(np.abs(arr1 - arr2))
            print(f"  ✗ {name}: Max difference {max_diff:.2e} (rtol={rtol})")
            return False

        print(f"  ✓ {name}: Exact match (within tolerance)")
        return True

    except Exception as e:
        print(f"  ✗ {name}: Comparison error - {e}")
        return False


def compare_spike_data(baseline: dict, new: dict, name: str) -> bool:
    """Compare spike time dictionaries."""
    try:
        if set(baseline.keys()) != set(new.keys()):
            print(f"  ✗ {name}: Brain area keys mismatch")
            print(f"    Baseline: {set(baseline.keys())}")
            print(f"    New: {set(new.keys())}")
            return False

        all_match = True
        for brain_area in baseline.keys():
            baseline_times = baseline[brain_area]
            new_times = new[brain_area]

            if len(baseline_times) != len(new_times):
                print(f"  ✗ {name}.{brain_area}: Unit count mismatch - {len(baseline_times)} vs {len(new_times)}")
                all_match = False
                continue

            for i, (bt, nt) in enumerate(zip(baseline_times, new_times)):
                if not np.allclose(bt, nt, rtol=1e-10, atol=1e-12):
                    print(f"  ✗ {name}.{brain_area}[{i}]: Spike times mismatch")
                    all_match = False

        if all_match:
            print(f"  ✓ {name}: Exact match")
        return all_match

    except Exception as e:
        print(f"  ✗ {name}: Comparison error - {e}")
        return False


def main():
    """Run verification test."""
    print("=" * 80)
    print("REFACTORING VERIFICATION TEST")
    print("=" * 80)
    print(f"NWB File: {NWB_FILE_NAME}")
    print(f"Epoch: {EPOCH_NAME}")
    print()

    # Load baseline data
    print("Loading baseline data...")
    try:
        baseline_data = load_baseline_data("verification_data.pkl")
        baseline_perf = load_baseline_data("performance.pkl")
        print(f"✓ Baseline load time: {baseline_perf['full_load_time']:.3f}s")
    except FileNotFoundError:
        print("ERROR: Baseline data not found. Run test_baseline.py first.")
        return False
    print()

    # Load new data
    print("Loading data with refactored code...")
    start = time.perf_counter()
    new_data = load_data(NWB_FILE_NAME, EPOCH_NAME)
    new_load_time = time.perf_counter() - start
    print(f"✓ New load time: {new_load_time:.3f}s")
    print()

    # Compare performance
    speedup = baseline_perf['full_load_time'] / new_load_time
    time_saved = baseline_perf['full_load_time'] - new_load_time
    print("Performance Comparison:")
    print(f"  Baseline: {baseline_perf['full_load_time']:.3f}s")
    print(f"  New:      {new_load_time:.3f}s")
    print(f"  Speedup:  {speedup:.2f}x")
    print(f"  Saved:    {time_saved:.3f}s ({time_saved/baseline_perf['full_load_time']*100:.1f}%)")
    print()

    # Compare outputs
    print("Comparing outputs...")
    all_passed = True

    # Position info
    if "position_info" in baseline_data:
        passed = compare_dataframes(
            baseline_data["position_info"],
            new_data["position_info"],
            "position_info"
        )
        all_passed = all_passed and passed

    # Well visits (critical for reward detection)
    if "well_visits" in baseline_data:
        passed = compare_dataframes(
            baseline_data["well_visits"],
            new_data["well_visits"],
            "well_visits"
        )
        all_passed = all_passed and passed

        # Extra check on is_reward column
        baseline_rewards = baseline_data["well_visits"]["is_reward"]
        new_rewards = new_data["well_visits"]["is_reward"]
        if not (baseline_rewards == new_rewards).all():
            print(f"  ✗ CRITICAL: is_reward column mismatch!")
            print(f"    Baseline: {baseline_rewards.sum()} rewards")
            print(f"    New: {new_rewards.sum()} rewards")
            all_passed = False

    # Trials
    if "trials" in baseline_data:
        passed = compare_dataframes(
            baseline_data["trials"],
            new_data["trials"],
            "trials"
        )
        all_passed = all_passed and passed

    # Task variables (critical for distance computation)
    if "task_variables" in baseline_data:
        passed = compare_dataframes(
            baseline_data["task_variables"],
            new_data["task_variables"],
            "task_variables"
        )
        all_passed = all_passed and passed

        # Extra check on dist_to_goal
        baseline_dist = baseline_data["task_variables"]["dist_to_goal"].values
        new_dist = new_data["task_variables"]["dist_to_goal"].values
        if not np.allclose(baseline_dist, new_dist, rtol=1e-10, atol=1e-12, equal_nan=True):
            max_diff = np.nanmax(np.abs(baseline_dist - new_dist))
            print(f"  ✗ CRITICAL: dist_to_goal mismatch! Max diff: {max_diff:.2e}")
            all_passed = False

    # Spike data
    if "spike_times" in baseline_data:
        passed = compare_spike_data(
            baseline_data["spike_times"],
            new_data["spike_times"],
            "spike_times"
        )
        all_passed = all_passed and passed

    if "spike_waveform_features" in baseline_data:
        passed = compare_spike_data(
            baseline_data["spike_waveform_features"],
            new_data["spike_waveform_features"],
            "spike_waveform_features"
        )
        all_passed = all_passed and passed

    # DIO events
    if "beam_breaks" in baseline_data:
        passed = compare_dataframes(
            baseline_data["beam_breaks"],
            new_data["beam_breaks"],
            "beam_breaks"
        )
        all_passed = all_passed and passed

    if "pump_events" in baseline_data:
        passed = compare_dataframes(
            baseline_data["pump_events"],
            new_data["pump_events"],
            "pump_events"
        )
        all_passed = all_passed and passed

    if "light_events" in baseline_data:
        passed = compare_dataframes(
            baseline_data["light_events"],
            new_data["light_events"],
            "light_events"
        )
        all_passed = all_passed and passed

    print()
    print("=" * 80)
    if all_passed:
        print("✓ VERIFICATION PASSED - All outputs match!")
        print("=" * 80)
        print()
        print(f"Refactoring successful:")
        print(f"  - Correctness: ✓ Verified")
        print(f"  - Performance: {speedup:.2f}x speedup ({time_saved:.3f}s saved)")
        return True
    else:
        print("✗ VERIFICATION FAILED - Outputs do not match!")
        print("=" * 80)
        print()
        print("Please review the differences above and revert changes.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
