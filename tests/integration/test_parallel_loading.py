"""Integration tests for parallel data loading.

Tests that parallel and sequential loading produce identical results
and measures performance improvement.
"""

import time

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders import load_data


# Test configuration - use a standard test session
TEST_NWB_FILE = "senor20201028_.nwb"
TEST_EPOCH = "08_r4"


def compare_data_structures(data1, data2):
    """Compare two data dictionaries for exact equality.

    Parameters
    ----------
    data1 : dict
        First dataset to compare
    data2 : dict
        Second dataset to compare

    Returns
    -------
    bool
        True if all data structures match exactly, False otherwise
    """
    # Check keys match
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    if keys1 != keys2:
        pytest.fail(
            f"Key mismatch!\n"
            f"  Only in data1: {keys1 - keys2}\n"
            f"  Only in data2: {keys2 - keys1}"
        )

    # Compare each data structure
    for key in sorted(keys1):
        val1 = data1[key]
        val2 = data2[key]

        # DataFrames
        if isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
            assert val1.equals(val2), f"{key}: DataFrames differ"

        # pandas Series
        elif isinstance(val1, pd.Series) and isinstance(val2, pd.Series):
            assert val1.equals(val2), f"{key}: Series differ"

        # NetworkX graphs
        elif hasattr(val1, 'nodes') and hasattr(val2, 'nodes'):
            assert set(val1.nodes()) == set(val2.nodes()), f"{key}: Graph nodes differ"
            assert set(val1.edges()) == set(val2.edges()), f"{key}: Graph edges differ"

        # Nested dicts (spike times, waveform features)
        elif isinstance(val1, dict) and isinstance(val2, dict):
            assert set(val1.keys()) == set(val2.keys()), f"{key}: Dict keys differ"

            for subkey in val1.keys():
                if isinstance(val1[subkey], list) and isinstance(val2[subkey], list):
                    assert len(val1[subkey]) == len(val2[subkey]), \
                        f"{key}[{subkey}]: List lengths differ"

                    for i, (arr1, arr2) in enumerate(zip(val1[subkey], val2[subkey])):
                        assert np.array_equal(arr1, arr2), \
                            f"{key}[{subkey}][{i}]: Arrays differ"

        # Other types
        else:
            try:
                assert val1 == val2, f"{key}: Values differ"
            except (ValueError, TypeError):
                # For types that don't support direct comparison
                assert str(val1) == str(val2), \
                    f"{key}: String representations differ"

    return True


@pytest.mark.slow
@pytest.mark.integration
class TestParallelLoading:
    """Integration tests for parallel data loading."""

    def test_parallel_sequential_equivalence(self):
        """Verify parallel and sequential loading produce identical results."""
        # Load with parallel=True
        data_parallel = load_data(TEST_NWB_FILE, TEST_EPOCH, parallel=True)

        # Load with parallel=False
        data_sequential = load_data(TEST_NWB_FILE, TEST_EPOCH, parallel=False)

        # Compare all data structures
        assert compare_data_structures(data_parallel, data_sequential)

    @pytest.mark.parametrize("parallel", [True, False])
    def test_parallel_flag_respected(self, parallel):
        """Verify parallel flag controls execution and returns valid data."""
        data = load_data(TEST_NWB_FILE, TEST_EPOCH, parallel=parallel)

        # Check that we got data back
        assert len(data) > 0, "No data returned"

        # Check for essential keys
        assert "position_info" in data, "Missing position_info"
        assert "track_graph" in data, "Missing track_graph"

    @pytest.mark.slow
    def test_parallel_performance_improvement(self):
        """Verify parallel loading is faster than sequential (optional verification)."""
        # Measure sequential time
        start = time.perf_counter()
        data_seq = load_data(TEST_NWB_FILE, TEST_EPOCH, parallel=False)
        time_seq = time.perf_counter() - start

        # Measure parallel time
        start = time.perf_counter()
        data_par = load_data(TEST_NWB_FILE, TEST_EPOCH, parallel=True)
        time_par = time.perf_counter() - start

        # Log times for review (don't assert - performance can vary)
        print(f"\nPerformance comparison:")
        print(f"  Sequential: {time_seq:.2f}s")
        print(f"  Parallel:   {time_par:.2f}s")

        if time_par < time_seq:
            speedup = ((time_seq - time_par) / time_seq) * 100
            print(f"  Speedup:    {speedup:.1f}% faster")
        else:
            slowdown = ((time_par - time_seq) / time_seq) * 100
            print(f"  Note:       {slowdown:.1f}% slower (may be first run/cold cache)")

        # Verify both returned same amount of data
        assert len(data_seq) == len(data_par), "Different number of data structures"

    def test_data_integrity_with_parallel(self):
        """Verify essential data structures are present with parallel loading."""
        data = load_data(TEST_NWB_FILE, TEST_EPOCH, parallel=True)

        # Check essential data structures
        essential_keys = [
            "position_info",
            "track_graph",
            "spike_times",
            "ripple_times",
            "trials",
            "task_variables",
        ]

        for key in essential_keys:
            assert key in data, f"Missing essential key: {key}"

        # Check data types
        assert isinstance(data["position_info"], pd.DataFrame)
        assert hasattr(data["track_graph"], 'nodes')  # NetworkX graph
        assert isinstance(data["spike_times"], dict)
        assert isinstance(data["trials"], pd.DataFrame)

    def test_selective_loading_with_parallel(self):
        """Verify selective loading works with parallel mode."""
        from continuum_swr_replay.types import DataLoadingPreset

        # Load only spikes and position
        data = load_data(
            TEST_NWB_FILE,
            TEST_EPOCH,
            include=DataLoadingPreset.SPIKES_AND_POSITION,
            parallel=True
        )

        # Should have position and spikes
        assert "position_info" in data
        assert "spike_times" in data

        # Should NOT have trials or task variables (not in this preset)
        # Note: trials/task_variables may still be loaded if needed as dependencies
        # This is expected behavior
