"""Fast targeted test for reward detection verification.

Only loads minimal data needed to verify reward detection logic.
"""

import pickle
import time
from pathlib import Path

import numpy as np

from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.types import SessionDataKeys

# Test configuration
NWB_FILE_NAME = "senor20201028_.nwb"
EPOCH_NAME = "08_r4"
BASELINE_DIR = Path("test_outputs/baseline")

# Load only what we need for reward detection testing
MINIMAL_INCLUDE = frozenset({
    SessionDataKeys.POSITION_INFO,
    SessionDataKeys.BEAM_BREAKS,
    SessionDataKeys.PUMP_EVENTS,
    SessionDataKeys.WELL_VISITS,
    SessionDataKeys.TRIALS,
})


def load_baseline_verification_data():
    """Load baseline verification data."""
    filepath = BASELINE_DIR / "verification_data.pkl"
    with open(filepath, "rb") as f:
        return pickle.load(f)


def compare_reward_detection(baseline, new_data):
    """Compare reward detection results."""
    print("\nComparing reward detection...")

    # Compare well_visits is_reward column
    baseline_rewards = baseline["well_visits"]["is_reward"]
    new_rewards = new_data["well_visits"]["is_reward"]

    if (baseline_rewards == new_rewards).all():
        print(f"  ✓ well_visits['is_reward']: Exact match ({baseline_rewards.sum()}/{len(baseline_rewards)} rewards)")
    else:
        mismatches = (~(baseline_rewards == new_rewards)).sum()
        print(f"  ✗ well_visits['is_reward']: {mismatches} mismatches!")
        print(f"    Baseline: {baseline_rewards.sum()} rewards")
        print(f"    New: {new_rewards.sum()} rewards")
        return False

    # Compare trials is_reward column
    baseline_trial_rewards = baseline["trials"]["is_reward"]
    new_trial_rewards = new_data["trials"]["is_reward"]

    if (baseline_trial_rewards == new_trial_rewards).all():
        print(f"  ✓ trials['is_reward']: Exact match ({baseline_trial_rewards.sum()}/{len(baseline_trial_rewards)} rewarded trials)")
    else:
        mismatches = (~(baseline_trial_rewards == new_trial_rewards)).sum()
        print(f"  ✗ trials['is_reward']: {mismatches} mismatches!")
        print(f"    Baseline: {baseline_trial_rewards.sum()} rewarded trials")
        print(f"    New: {new_trial_rewards.sum()} rewarded trials")
        return False

    # Compare shapes and structure
    if baseline["well_visits"].shape != new_data["well_visits"].shape:
        print(f"  ✗ well_visits shape mismatch: {baseline['well_visits'].shape} vs {new_data['well_visits'].shape}")
        return False

    if baseline["trials"].shape != new_data["trials"].shape:
        print(f"  ✗ trials shape mismatch: {baseline['trials'].shape} vs {new_data['trials'].shape}")
        return False

    print("  ✓ All shapes match")

    return True


def main():
    """Run fast reward detection verification."""
    print("=" * 80)
    print("FAST REWARD DETECTION VERIFICATION")
    print("=" * 80)
    print(f"NWB File: {NWB_FILE_NAME}")
    print(f"Epoch: {EPOCH_NAME}")
    print(f"Loading: {len(MINIMAL_INCLUDE)} data types (minimal)")
    print()

    # Load baseline
    print("Loading baseline data...")
    baseline = load_baseline_verification_data()
    print("✓ Baseline loaded")

    # Load new data with selective loading
    print("\nLoading data with refactored code (selective loading)...")
    start = time.perf_counter()
    new_data = load_data(NWB_FILE_NAME, EPOCH_NAME, include=MINIMAL_INCLUDE)
    load_time = time.perf_counter() - start
    print(f"✓ Data loaded in {load_time:.2f}s")

    # Compare
    passed = compare_reward_detection(baseline, new_data)

    print("\n" + "=" * 80)
    if passed:
        print("✓ VERIFICATION PASSED - Reward detection is correct!")
        print("=" * 80)
        return True
    else:
        print("✗ VERIFICATION FAILED - Reward detection has errors!")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
