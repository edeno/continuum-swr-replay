"""Profile data loading performance to identify bottlenecks.

Uses Python's built-in cProfile to get detailed timing information.
"""

import cProfile
import pstats
import time
from io import StringIO

from continuum_swr_replay.data_loaders import load_data

# Test configuration
NWB_FILE_NAME = "senor20201028_.nwb"
EPOCH_NAME = "08_r4"


def profile_load_data():
    """Profile the complete data loading pipeline."""
    print("=" * 80)
    print("DATA LOADING PERFORMANCE PROFILER")
    print("=" * 80)
    print(f"NWB File: {NWB_FILE_NAME}")
    print(f"Epoch: {EPOCH_NAME}")
    print()

    # Create profiler
    profiler = cProfile.Profile()

    # Profile the load_data function
    print("Loading data with profiling...\n")
    start_time = time.perf_counter()
    profiler.enable()
    data = load_data(NWB_FILE_NAME, EPOCH_NAME)
    profiler.disable()
    elapsed = time.perf_counter() - start_time

    print(f"\n✓ Data loaded successfully in {elapsed:.2f}s ({len(data)} data types)")

    # Save profiling stats to file
    profiler.dump_stats("profile_data_loading.prof")
    print("\nProfile saved to: profile_data_loading.prof")

    # Print top time-consuming functions
    print("\n" + "=" * 80)
    print("TOP TIME-CONSUMING FUNCTIONS")
    print("=" * 80)

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    ps.print_stats(50)  # Top 50 functions by cumulative time

    print(s.getvalue())

    print("\n" + "=" * 80)
    print("TOP FUNCTIONS BY TIME-PER-CALL")
    print("=" * 80)

    s2 = StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.strip_dirs()
    ps2.sort_stats("time")
    ps2.print_stats(30)  # Top 30 functions by internal time

    print(s2.getvalue())

    # Print summary statistics for key data loading components
    print("\n" + "=" * 80)
    print("KEY COMPONENT ANALYSIS")
    print("=" * 80)

    s3 = StringIO()
    ps3 = pstats.Stats(profiler, stream=s3)
    ps3.strip_dirs()

    # Look for specific data loading functions
    patterns = [
        "get_position_info",
        "get_spike_data",
        "get_hpc_marks",
        "load_dios",
        "make_trials_df",
        "make_well_visits_df",
        "get_task_variables",
        "dist_to_goal",
        "fetch",  # DataJoint fetch operations
    ]

    for pattern in patterns:
        print(f"\n--- Functions matching '{pattern}' ---")
        s_pattern = StringIO()
        ps_pattern = pstats.Stats(profiler, stream=s_pattern)
        ps_pattern.strip_dirs()
        ps_pattern.sort_stats("cumulative")
        ps_pattern.print_stats(pattern)
        output = s_pattern.getvalue()
        # Only print if we found matches
        if "function calls" in output:
            print(output)

    print("\n" + "=" * 80)
    print("\nDetailed stats available in: profile_data_loading.prof")
    print("View with: uv run python -m pstats profile_data_loading.prof")
    print("\nCommon pstats commands:")
    print("  sort cumulative  - Sort by cumulative time")
    print("  sort time        - Sort by internal time")
    print("  stats 20         - Show top 20 functions")
    print("  stats <pattern>  - Show functions matching pattern")

    return data


if __name__ == "__main__":
    try:
        data = profile_load_data()
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
