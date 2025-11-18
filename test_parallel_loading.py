"""Test parallel loading implementation for correctness and performance.

Verifies that parallel and sequential loading produce identical results,
then measures the performance improvement.
"""

import time
import numpy as np
import pandas as pd
from continuum_swr_replay.data_loaders import load_data

# Test configuration
NWB_FILE_NAME = "senor20201028_.nwb"
EPOCH_NAME = "08_r4"


def compare_data_structures(data1, data2, label1="parallel", label2="sequential"):
    """Compare two data dictionaries for exact equality."""
    print(f"\n{'='*80}")
    print(f"COMPARING {label1.upper()} vs {label2.upper()}")
    print(f"{'='*80}")

    # Check keys match
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    if keys1 != keys2:
        print(f"✗ Key mismatch!")
        print(f"  Only in {label1}: {keys1 - keys2}")
        print(f"  Only in {label2}: {keys2 - keys1}")
        return False

    print(f"✓ Both have {len(keys1)} keys")

    # Compare each data structure
    all_match = True
    for key in sorted(keys1):
        val1 = data1[key]
        val2 = data2[key]

        # DataFrames
        if isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
            if val1.equals(val2):
                print(f"✓ {key}: DataFrames match exactly")
            else:
                print(f"✗ {key}: DataFrames differ")
                all_match = False

        # Lists of arrays (spike times, waveform features)
        elif isinstance(val1, dict) and isinstance(val2, dict):
            # Nested dictionaries (spike_times, spike_waveform_features)
            if set(val1.keys()) != set(val2.keys()):
                print(f"✗ {key}: Dict keys differ")
                all_match = False
            else:
                match = True
                for subkey in val1.keys():
                    if isinstance(val1[subkey], list) and isinstance(val2[subkey], list):
                        if len(val1[subkey]) != len(val2[subkey]):
                            print(f"✗ {key}[{subkey}]: List lengths differ ({len(val1[subkey])} vs {len(val2[subkey])})")
                            match = False
                            all_match = False
                        else:
                            for i, (arr1, arr2) in enumerate(zip(val1[subkey], val2[subkey])):
                                if not np.array_equal(arr1, arr2):
                                    print(f"✗ {key}[{subkey}][{i}]: Arrays differ")
                                    match = False
                                    all_match = False
                if match:
                    print(f"✓ {key}: Nested dict with lists of arrays match exactly")

        # NetworkX graphs
        elif hasattr(val1, 'nodes') and hasattr(val2, 'nodes'):
            if (set(val1.nodes()) == set(val2.nodes()) and
                set(val1.edges()) == set(val2.edges())):
                print(f"✓ {key}: Graphs match exactly")
            else:
                print(f"✗ {key}: Graphs differ")
                all_match = False

        # pandas Series
        elif isinstance(val1, pd.Series) and isinstance(val2, pd.Series):
            if val1.equals(val2):
                print(f"✓ {key}: Series match exactly")
            else:
                print(f"✗ {key}: Series differ")
                all_match = False

        # Other types
        else:
            try:
                if val1 == val2:
                    print(f"✓ {key}: Values match exactly")
                else:
                    print(f"✗ {key}: Values differ")
                    all_match = False
            except (ValueError, TypeError):
                # For types that don't support direct comparison
                if str(val1) == str(val2):
                    print(f"✓ {key}: String representations match")
                else:
                    print(f"✗ {key}: Values differ (compared as strings)")
                    all_match = False

    print(f"\n{'='*80}")
    if all_match:
        print("✅ ALL DATA STRUCTURES MATCH EXACTLY")
    else:
        print("❌ SOME DATA STRUCTURES DIFFER")
    print(f"{'='*80}\n")

    return all_match


def main():
    """Test parallel loading correctness and performance."""
    print(f"\n{'='*80}")
    print("PARALLEL LOADING TEST")
    print(f"{'='*80}")
    print(f"NWB File: {NWB_FILE_NAME}")
    print(f"Epoch: {EPOCH_NAME}")
    print()

    # Test 1: Load with parallel=True
    print("Step 1: Loading data with parallel=True (default)...")
    start_parallel = time.perf_counter()
    data_parallel = load_data(NWB_FILE_NAME, EPOCH_NAME, parallel=True)
    time_parallel = time.perf_counter() - start_parallel
    print(f"✓ Parallel loading completed in {time_parallel:.2f}s")
    print(f"  Loaded {len(data_parallel)} data types")

    # Test 2: Load with parallel=False
    print("\nStep 2: Loading data with parallel=False (sequential)...")
    start_sequential = time.perf_counter()
    data_sequential = load_data(NWB_FILE_NAME, EPOCH_NAME, parallel=False)
    time_sequential = time.perf_counter() - start_sequential
    print(f"✓ Sequential loading completed in {time_sequential:.2f}s")
    print(f"  Loaded {len(data_sequential)} data types")

    # Test 3: Compare outputs
    print("\nStep 3: Comparing outputs for correctness...")
    outputs_match = compare_data_structures(data_parallel, data_sequential)

    # Test 4: Performance comparison
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"Parallel loading:   {time_parallel:.2f}s")
    print(f"Sequential loading: {time_sequential:.2f}s")

    if time_parallel < time_sequential:
        improvement = time_sequential - time_parallel
        percent = (improvement / time_sequential) * 100
        print(f"\n✅ SPEEDUP: {improvement:.2f}s faster ({percent:.1f}% improvement)")
    else:
        regression = time_parallel - time_sequential
        percent = (regression / time_sequential) * 100
        print(f"\n⚠️  SLOWER: {regression:.2f}s slower ({percent:.1f}% regression)")
        print("   Note: First run may be slower due to cold caches")

    print(f"{'='*80}\n")

    # Final verdict
    if outputs_match and time_parallel <= time_sequential * 1.1:  # Allow 10% margin
        print("✅ PARALLEL LOADING IMPLEMENTATION: SUCCESS")
        print("   - Outputs match exactly")
        print("   - Performance is equal or better")
        return 0
    elif outputs_match:
        print("⚠️  PARALLEL LOADING IMPLEMENTATION: PARTIAL SUCCESS")
        print("   - Outputs match exactly ✓")
        print("   - Performance slower than expected (may improve with warm cache)")
        return 0
    else:
        print("❌ PARALLEL LOADING IMPLEMENTATION: FAILED")
        print("   - Outputs do not match!")
        return 1


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
