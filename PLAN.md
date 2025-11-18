# Data Loading Optimization and Refactoring Plan

## Objective

Optimize data loading performance while maintaining exact computational correctness. All changes must be verified to produce identical outputs.

## Test Case

- **NWB File**: `senor20201028_.nwb`
- **Epoch**: `08_r4`
- **Verification**: Compare all outputs before/after with `np.allclose()` and `pd.testing.assert_frame_equal()`

## Refactoring Tasks

### Phase 1: Setup and Baseline (COMPLETED ✓)

- [ ] Create baseline test script to capture current outputs
- [ ] Measure current performance for each component
- [ ] Save baseline results for comparison

### Phase 2: Low-Risk Refactorings (Code Quality)

**Priority**: Medium | **Risk**: Low | **Impact**: Maintainability

#### Task 2.1: Extract DIO Loading Helper

- **File**: `src/continuum_swr_replay/data_loaders/events.py`
- **Lines**: 131-179
- **Change**: Extract `_load_dio_event_type()` helper function
- **Verification**: Compare `beam_breaks`, `pump_events`, `light_events` DataFrames
- **Expected Impact**: No performance change, 33% fewer lines

#### Task 2.2: Unify Detector Configuration

- **Files**:
  - `src/continuum_swr_replay/parameters.py` (line 95-99)
  - `scripts/run_decoding_bandit.py` (line 162-166)
- **Change**: Use single `DETECTOR_CONFIG` from parameters.py
- **Verification**: Ensure detector creation still works
- **Expected Impact**: Eliminates duplicate definition

### Phase 3: High-Priority Performance Optimizations

#### Task 3.1: Vectorize Reward Detection (HIGH IMPACT)

- **File**: `src/continuum_swr_replay/data_loaders/trials.py`
- **Lines**: 250-256
- **Current Complexity**: O(n_visits × n_pumps) = ~500,000 operations
- **Target Complexity**: O(n_visits + n_pumps) = ~1,500 operations
- **Change**: Use interval-based vectorized operations
- **Verification**:
  - Compare `well_visits['is_reward']` column (exact boolean match)
  - Check trial-level `is_reward` aggregation
- **Expected Speedup**: 100-300x
- **Risk**: Medium (different algorithm, must verify correctness)

#### Task 3.2: Optimize Distance Computation (HIGH IMPACT)

- **File**: `src/continuum_swr_replay/data_loaders/task_variables.py`
- **Lines**: 209-221 (dist_to_goal function)
- **Current Complexity**: O(n_timepoints × graph_copy_cost)
- **Target Complexity**: O(n_segments × shortest_path + n_timepoints)
- **Change**: Precompute distance lookup table per segment
- **Verification**:
  - Compare `dist_to_goal` array with `np.allclose(rtol=1e-10)`
  - Verify `task_variables['dist_to_goal']` DataFrame column
- **Expected Speedup**: 50-100x
- **Risk**: Medium (algorithmic change)

#### Task 3.3: Optimize Spike Loading Queries (MEDIUM IMPACT)

- **File**: `src/continuum_swr_replay/data_loaders/spikes.py`
- **Lines**: 209-240 (get_hpc_marks function)
- **Change**: Check for parameter name existence before fetching, or use OR query
- **Verification**:
  - Compare spike_times lists (exact match)
  - Compare spike_waveform_features arrays (exact match)
  - Check both parameter name scenarios
- **Expected Speedup**: 2x for affected sessions (eliminate redundant query)
- **Risk**: Low (same data, different query strategy)

### Phase 4: Additional Optimizations (If Time Permits)

#### Task 4.1: Optimize Position Interpolation

- **File**: `src/continuum_swr_replay/data_loaders/position.py`
- **Lines**: 106-115
- **Change**: Use scipy.interpolate directly without DataFrame overhead
- **Expected Speedup**: 2-5x
- **Risk**: Medium

## Verification Strategy

### Per-Task Verification

1. **Create test fixture**: Save original outputs to pickle
2. **Run modified code**: Capture new outputs
3. **Compare outputs**:
   - DataFrames: `pd.testing.assert_frame_equal(check_exact=False, rtol=1e-10)`
   - Arrays: `np.allclose(rtol=1e-10, atol=1e-12)`
   - Booleans: Exact match
4. **Performance test**: Time function before/after with same inputs

### Integration Testing

- Run full `load_data()` pipeline with test case
- Compare all returned dictionary keys
- Verify all downstream computations (trials, task_variables)

### Acceptance Criteria

- ✓ All outputs match within numerical precision (1e-10 relative tolerance)
- ✓ Performance improvement measured and documented
- ✓ No regressions in other test cases
- ✓ Code passes existing tests (if any)

## Rollback Plan

- Git commits are atomic per task
- Each task has verification before proceeding
- Can revert individual commits if issues found
- Maintain REFACTORING_LOG.md with results

## Performance Targets

| Task | Component | Current Time | Target Time | Expected Speedup |
|------|-----------|--------------|-------------|------------------|
| 3.1 | Reward Detection | ~2.0s | <0.01s | 100-300x |
| 3.2 | Distance Computation | ~5.0s | <0.1s | 50-100x |
| 3.3 | Spike Loading | ~4.0s | ~2.0s | 2x |
| **Total** | **load_data()** | **~60-90s** | **~50-75s** | **1.2-1.5x** |

## Risk Assessment

### Low Risk

- DIO helper extraction (code deduplication only)
- Detector config unification (no logic change)
- Spike loading optimization (same query, different strategy)

### Medium Risk

- Reward detection vectorization (algorithm change)
- Distance computation optimization (algorithm change)

### Mitigation

- Comprehensive verification for each change
- Test with multiple sessions beyond baseline
- Keep original code as comments for reference
- Atomic commits for easy rollback

## Implementation Order

1. **Phase 1**: Create baseline test and measurements ← START HERE
2. **Phase 2.1**: DIO helper (safe, improves code clarity)
3. **Phase 2.2**: Unify detector config (safe)
4. **Phase 3.3**: Spike loading (low risk, medium impact)
5. **Phase 3.1**: Reward detection (medium risk, high impact)
6. **Phase 3.2**: Distance computation (medium risk, high impact)
7. **Phase 4.1**: Position interpolation (if time allows)

## Success Metrics

- [ ] All verification tests pass
- [ ] Total data loading time reduced by >15%
- [ ] Code maintainability improved (fewer duplicated lines)
- [ ] No computational correctness regressions
- [ ] Documentation updated

## Notes

- Focus on correctness over speed
- If any verification fails, revert and investigate
- Document all findings in REFACTORING_LOG.md
- Consider edge cases: empty data, single element arrays, NaN handling
