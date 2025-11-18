# Parallel Data Loading Plan

## Objective

Implement concurrent loading of independent data components to reduce total load time from ~310s to ~160-210s (30-45% improvement).

## Current Architecture

Sequential loading (lines 182-282 in `bandit_task.py`):

```
1. Position data (always loaded) - required for time bounds
2. Spike data - 635s (67.7% of load time)
3. Ripple data - 90.5s (9.6%)
4. Theta data
5. Multiunit data (depends on spike_times)
6. DIO events - 42.6s (4.5%)
7. Well visits (depends on DIO)
8. Trials (depends on well visits)
9. Task variables (depends on trials)
10. Electrode group info
```

## Parallelization Strategy

### Phase 1: Independent Components (Run in Parallel)

These have no dependencies and can run concurrently:

```python
with ProcessPoolExecutor(max_workers=4) as executor:
    spike_future = executor.submit(get_spike_data, nwb_file_name)
    ripple_future = executor.submit(get_ripple, nwb_file_name, epoch_name, ripple_detector_name)
    theta_future = executor.submit(get_theta, nwb_file_name, epoch_name)
    dio_future = executor.submit(load_dios, nwb_file_name, valid_times)
    electrode_future = executor.submit(get_electrode_group_info, nwb_file_name)

    # Wait for all to complete
    spike_data = spike_future.result()
    ripple_data = ripple_future.result()
    theta_data = theta_future.result()
    dio_events = dio_future.result()
    electrode_info = electrode_future.result()
```

**Expected speedup**: 635s / 4 workers ≈ 160s (assuming good parallelism)

### Phase 2: Dependent Components (Run Sequentially)

After Phase 1 completes:

```python
# Filter spike times (depends on position_info.index from Phase 0)
spike_times, waveform_features = filter_spike_times(...)

# Multiunit (depends on spike_times)
multiunit_data = get_multiunit(spike_times, position_info)

# Well visits (depends on DIO events)
well_visits = make_well_visits_df(dio_events["beam_breaks"], dio_events["pump_events"], ...)

# Trials (depends on well visits)
trials = make_trials_df_from_well_visits(well_visits)

# Task variables (depends on trials)
task_variables = make_task_variables(position_info, trials, track_graph)
```

## Implementation Considerations

### Thread vs Process Parallelism

**Use ProcessPoolExecutor** because:
- DataJoint/PyMySQL connections are **not thread-safe**
- Each process gets its own database connection (avoids connection conflicts)
- HDF5 file handles are not shared across processes (thread-safe)
- Serialization overhead is acceptable given long I/O times (~300s)

**Threading Issues Encountered**:
- ThreadPoolExecutor caused `AssertionError: Result length not requested length`
- PyMySQL connection objects cannot be safely shared across threads
- DataJoint table queries from multiple threads cause connection corruption
- Switching to ProcessPoolExecutor resolves these issues

### Optimal Worker Count

Based on profiling:
- Spike data: 635s (67.7%)
- Ripple data: 90.5s (9.6%)
- DIO events: 42.6s (4.5%)
- Theta data: ~30s (estimated)
- Electrode info: <1s

**Recommendation**: 4 workers
- Worker 1: Spike data (longest)
- Worker 2: Ripple data
- Worker 3: DIO events
- Worker 4: Theta data

With perfect parallelism: max(635, 90.5, 42.6, 30) = 635s
With realistic 70% efficiency: 635s * 0.7 = 445s total for Phase 1

### Error Handling

```python
try:
    result = future.result(timeout=600)  # 10 minute timeout
except TimeoutError:
    logger.error(f"Data loading timed out for {component}")
    raise
except Exception as e:
    logger.error(f"Error loading {component}: {e}")
    raise
```

### Backward Compatibility

- Preserve exact same API
- No changes to function signature
- Add optional `parallel: bool = True` parameter
- Default to parallel loading, allow fallback to sequential

## Expected Performance

| Scenario | Sequential Time | Parallel Time | Speedup |
|----------|----------------|---------------|---------|
| **Best case** (100% parallel efficiency) | 310s | 160s | 1.94x |
| **Realistic** (70% efficiency) | 310s | 210s | 1.48x |
| **Worst case** (I/O bottleneck) | 310s | 250s | 1.24x |

**Target**: 160-210s (30-45% improvement over current 310s)

## Risks and Mitigation

### Risk 1: Database Connection Limits
**Issue**: Multiple threads querying DataJoint simultaneously
**Mitigation**:
- DataJoint has connection pooling
- 4 threads is conservative (well below typical limits)
- Monitor connection count during testing

### Risk 2: Memory Pressure
**Issue**: Loading multiple large datasets concurrently
**Mitigation**:
- Threads share memory (no duplication)
- Peak memory = max(component sizes), not sum
- Monitor memory usage during testing

### Risk 3: HDF5 Thread Safety
**Issue**: HDF5 has thread-safety limitations
**Mitigation**:
- Each thread opens its own file handle
- PyNWB/HDMF handle thread safety internally
- Test thoroughly with verification suite

### Risk 4: Non-deterministic Errors
**Issue**: Race conditions or threading bugs
**Mitigation**:
- Comprehensive verification testing
- Run multiple iterations to catch non-deterministic failures
- Fall back to sequential if errors occur

## Implementation Steps

1. **Create parallel loading function** (`load_data_parallel`)
2. **Add `parallel: bool = True` parameter to `load_data()`**
3. **Implement Phase 1 parallel execution**
4. **Implement Phase 2 sequential execution**
5. **Add comprehensive error handling and logging**
6. **Run verification tests** (ensure exact output match)
7. **Performance testing** (measure actual speedup)
8. **Document in docstring** (note about parallel loading)

## Testing Strategy

### Correctness Verification
```bash
# Run with parallel loading
data_parallel = load_data(nwb_file_name, epoch_name, parallel=True)

# Run with sequential loading
data_sequential = load_data(nwb_file_name, epoch_name, parallel=False)

# Compare all outputs (should be identical)
compare_outputs(data_parallel, data_sequential)
```

### Performance Testing
```bash
# Measure sequential
time_sequential = timeit(lambda: load_data(..., parallel=False))

# Measure parallel
time_parallel = timeit(lambda: load_data(..., parallel=True))

# Calculate speedup
speedup = time_sequential / time_parallel
print(f"Speedup: {speedup:.2f}x ({time_sequential - time_parallel:.1f}s saved)")
```

### Stress Testing
- Run on all 109 sessions to ensure robustness
- Test with different `include` parameter configurations
- Monitor system resources (CPU, memory, network)

## Code Location

**File**: `src/continuum_swr_replay/data_loaders/bandit_task.py`
**Function**: `load_data()` (lines 44-282)
**New code**: ~50-100 lines for parallel execution logic

## Success Criteria

- ✅ All verification tests pass (exact output match)
- ✅ Load time reduced by 30-45% (100-150s saved)
- ✅ No increase in memory usage
- ✅ Backward compatible (existing code works unchanged)
- ✅ Comprehensive error handling
- ✅ Documentation updated

## Future Enhancements

1. **Adaptive worker count**: Adjust based on available CPU cores
2. **Progress reporting**: Show which components are loading
3. **Caching**: Store loaded data to avoid reloading same session
4. **Batch loading**: Load multiple sessions efficiently
