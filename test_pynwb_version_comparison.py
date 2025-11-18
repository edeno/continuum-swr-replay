"""Quick test to compare pynwb 2.7 vs 2.8 performance.

This test specifically measures NWB file I/O and object construction overhead.
"""

import time
import subprocess
import sys

# Test configuration
NWB_FILE_NAME = "senor20201028_.nwb"
EPOCH_NAME = "08_r4"


def get_pynwb_version():
    """Get currently installed pynwb version."""
    import pynwb
    return pynwb.__version__


def run_load_test():
    """Run a single load test and return the time."""
    from continuum_swr_replay.data_loaders import load_data

    print(f"Testing with pynwb {get_pynwb_version()}...")

    start = time.perf_counter()
    data = load_data(NWB_FILE_NAME, EPOCH_NAME)
    elapsed = time.perf_counter() - start

    print(f"✓ Loaded {len(data)} data types in {elapsed:.2f}s")
    return elapsed


def main():
    """Compare pynwb 2.7 vs 2.8 performance."""
    print("=" * 80)
    print("PyNWB VERSION COMPARISON TEST")
    print("=" * 80)
    print(f"NWB File: {NWB_FILE_NAME}")
    print(f"Epoch: {EPOCH_NAME}")
    print()

    current_version = get_pynwb_version()
    print(f"Current pynwb version: {current_version}")

    current_time = run_load_test()

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"PyNWB {current_version}: {current_time:.2f}s")

    # Compare to baseline (pynwb 2.8.3 with our optimizations)
    baseline_time = 310.8  # From Phase 3.2 results

    if current_version.startswith("2.7"):
        print(f"Baseline (pynwb 2.8.3): {baseline_time:.2f}s")
        improvement = baseline_time - current_time
        percent = (improvement / baseline_time) * 100
        print()
        if improvement > 0:
            print(f"✓ IMPROVEMENT: {improvement:.2f}s faster ({percent:.1f}% speedup)")
            print(f"   This suggests pynwb 2.7 reduces deep copy overhead")
        else:
            print(f"✗ NO IMPROVEMENT: {-improvement:.2f}s slower")
            print(f"   Deep copy issue may not be the bottleneck")

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
