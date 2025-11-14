# Implementation Plan for CODE_REVIEW.md Issues

**Date Created**: 2025-11-14
**Total Estimated Time**: ~18 hours over 2 weeks
**Based on**: CODE_REVIEW.md senior engineer review

---

## Overview

This plan addresses critical issues and high-priority improvements identified in the code review. The plan is organized into 3 phases by priority, with a test-first approach for refactoring work.

**Scope Adjustments**:

- ✅ Keep and fix `generate_figure03.py` (not removing)
- ✅ No deprecation cycle needed (breaking changes OK)
- ✅ Return new objects for filtering (functional style)
- ✅ Write tests BEFORE refactoring
- ✅ Keep PFC module as placeholder
- ❌ No Pydantic validation (deferred)
- ❌ No dependency pinning (deferred)
- ❌ No integration test expansion (deferred)

---

## Phase 1: Critical Blocking Issues (Must Fix Immediately)

**Time Estimate: 1 hour**
**Priority: CRITICAL**

### 1.1 Fix Import Error in run_decoding_bandit.py

**File**: `scripts/run_decoding_bandit.py:13`
**Issue**: Imports from non-existent module `visualization.visualization`

**Fix**:

```python
# CHANGE FROM:
from continuum_swr_replay.visualization.visualization import (
    create_interactive_1D_decoding_figurl,
)

# CHANGE TO:
from continuum_swr_replay.visualization.figurl import (
    create_interactive_1D_decoding_figurl,
)
```

**Time**: 5 minutes
**Verification**: `uv run python scripts/run_decoding_bandit.py --help`

### 1.2 Fix Syntax Error in generate_figure03.py

**File**: `scripts/generate_figure03.py:2`
**Issue**: Invalid syntax preventing script execution

**Fix**: Complete the incomplete script implementation (inspect and fix syntax errors)

**Time**: 15 minutes
**Verification**: `uv run ruff check scripts/generate_figure03.py`

### 1.3 Fix API Typo (Breaking Change)

**Files**: `src/continuum_swr_replay/visualization/plots.py` (lines 908, 930, 970)
**Issue**: Parameter `edege_linewidth` should be `edge_linewidth`

**Fix**:

- Rename parameter to `edge_linewidth` (find/replace in 3 functions)
- Update all docstrings
- Update any usage in codebase

**Functions affected**:

- `plot_graph_as_1D()` - line 908
- `plot_graph_as_1D()` - line 930 (appears multiple times)
- `plot_graph_as_1D()` - line 970

**Time**: 20 minutes
**Breaking Change**: Yes, but better to fix early
**Verification**: `uv run ruff check src/` and `uv run pytest tests/`

---

## Phase 2: High Priority Type Safety Improvements

**Time Estimate: 10-11 hours**
**Priority: HIGH**

### 2.1 Add Type Hints to Data Loader Modules

**Goal**: Achieve >90% type coverage across codebase

#### 2.1.1 Add Types to lfp.py

**File**: `src/continuum_swr_replay/data_loaders/lfp.py`
**Issue**: 5 functions missing type annotations

**Tasks**:

- Define return TypedDicts for `get_theta()`, `get_ripple()`, etc.
- Add numpy array type annotations (`NDArray[np.float64]`)
- Add pandas DataFrame type hints
- Document all parameters and return types

**Time**: 2 hours
**Deliverable**: Full type coverage for LFP module
**Verification**: `uv run mypy src/continuum_swr_replay/data_loaders/lfp.py`

#### 2.1.2 Add Types to events.py

**File**: `src/continuum_swr_replay/data_loaders/events.py`
**Issue**: 2 functions missing type annotations

**Tasks**:

- Add type hints to all functions
- Consider using dataclass for event structures
- Add pandas DataFrame type hints

**Time**: 1 hour
**Deliverable**: Full type coverage for events module
**Verification**: `uv run mypy src/continuum_swr_replay/data_loaders/events.py`

#### 2.1.3 Add Types to trials.py

**File**: `src/continuum_swr_replay/data_loaders/trials.py`
**Issue**: 5 functions missing type annotations

**Tasks**:

- Add pandas DataFrame type hints with proper return types
- Add type hints for all function parameters
- Document return structures

**Time**: 2 hours
**Deliverable**: Full type coverage for trials module
**Verification**: `uv run mypy src/continuum_swr_replay/data_loaders/trials.py`

#### 2.1.4 Add Types to task_variables.py

**File**: `src/continuum_swr_replay/data_loaders/task_variables.py`
**Issue**: 5 functions missing type annotations (lines 71, 119, 153)

**Tasks**:

- Add NetworkX graph type hints (`nx.Graph`)
- Add pandas DataFrame type hints
- Add numpy array type annotations
- Type all function parameters and returns

**Time**: 2 hours
**Deliverable**: Full type coverage for task variables module
**Verification**: `uv run mypy src/continuum_swr_replay/data_loaders/task_variables.py`

#### 2.1.5 Fix Remaining Mypy Errors

**Files with errors**:

- `utils.py:74` - nested function `_count_exposure`
- `animation.py:452`
- `interactive.py:181, 269`
- `figurl.py:201, 637, 659, 667`

**Tasks**:

- Add type hints to nested functions
- Fix incompatible type assignments
- Add return type annotations

**Time**: 1 hour
**Deliverable**: Clean mypy output (<5 errors)
**Verification**: `uv run mypy src/`

### 2.2 Replace dict[str, Any] with TypedDict

**File**: `src/continuum_swr_replay/data_loaders/bandit_task.py`
**Issue**: `load_data()` returns untyped `dict[str, Any]`

**Fix**:

```python
# CHANGE FROM:
def load_data(
    nwb_file_name: str,
    epoch_name: str,
    ripple_detector_name: str = "Kay",
) -> dict[str, Any]:
    ...

# CHANGE TO:
from ..analysis.hippocampus.types import SessionData

def load_data(
    nwb_file_name: str,
    epoch_name: str,
    ripple_detector_name: str = "Kay",
) -> SessionData:
    ...
```

**Benefits**:

- Better IDE autocomplete
- Type safety for all downstream code
- Catches missing/wrong keys at type-check time

**Time**: 1 hour
**Verification**: Verify return statement matches TypedDict structure, run mypy

### 2.3 Refactor In-Place Modifications to Return New Objects

**Time**: 3 hours
**IMPORTANT**: Write tests FIRST before refactoring!

#### 2.3.1 Add Tests for Current Behavior (1.5 hours)

**Files to test**:

1. `filter_spike_times()` in `src/continuum_swr_replay/data_loaders/spikes.py:356-393`
2. `filter_ripple_lfp_time()` in `src/continuum_swr_replay/data_loaders/lfp.py`

**Test cases to cover**:

- Normal operation (multiple spike trains, within/outside bounds)
- Empty dictionaries
- Single element arrays
- Boundary conditions (exact start/end times)
- Edge case: all spikes filtered out
- Edge case: no spikes filtered

**Test file**: `tests/unit/test_data_loaders.py` (or create new file)

**Deliverable**: Green test suite documenting expected behavior
**Verification**: `uv run pytest tests/unit/test_data_loaders.py -v`

#### 2.3.2 Refactor to Functional Style (1.5 hours)

**Approach**:

```python
# OLD (in-place modification):
def filter_spike_times(
    spike_times: dict[str, list[NDArray[np.float64]]],
    spike_waveform_features: dict[str, list[NDArray[np.float64]]],
    position_time: NDArray[np.float64],
) -> None:
    """Filter spike times to match position data time bounds (modifies in place)."""
    for key in spike_times.keys():
        spike_times[key] = [filtered]  # modifies argument!

# NEW (functional style):
def filter_spike_times(
    spike_times: dict[str, list[NDArray[np.float64]]],
    spike_waveform_features: dict[str, list[NDArray[np.float64]]],
    position_time: NDArray[np.float64],
) -> tuple[dict[str, list[NDArray[np.float64]]], dict[str, list[NDArray[np.float64]]]]:
    """Filter spike times to match position data time bounds (returns new dicts)."""
    filtered_times = {}
    filtered_features = {}
    for key in spike_times.keys():
        filtered_times[key] = [filtered]  # new dict
    return filtered_times, filtered_features
```

**Tasks**:

1. Refactor `filter_spike_times()` to return new objects
2. Refactor `filter_ripple_lfp_time()` to return new objects
3. Update all call sites to use returned values
4. Update docstrings to reflect new behavior
5. Ensure all tests still pass

**Deliverable**: Functional-style filtering functions
**Verification**: All existing tests pass + new tests pass

---

## Phase 3: Medium Priority Refactoring

**Time Estimate: 7 hours**
**Priority: MEDIUM**

### 3.1 Extract Long Functions in figurl.py

**File**: `src/continuum_swr_replay/visualization/figurl.py`
**Issue**: `create_interactive_1D_decoding_figurl()` is 194 lines long

**Refactor Plan**:

Extract view creation helpers:

```python
def _create_speed_view(
    speed: NDArray[np.float64],
    time: NDArray[np.float64],
    ...
) -> pn.Column:
    """Create speed visualization panel.

    Parameters
    ----------
    speed : NDArray[np.float64]
        Speed data in cm/s
    time : NDArray[np.float64]
        Time points in seconds
    ...

    Returns
    -------
    pn.Column
        Panel column with speed visualization
    """
    # Extract ~40 lines of speed plotting code
    ...

def _create_head_direction_view(...) -> pn.Column:
    """Create head direction visualization panel."""
    # Extract ~30 lines of head direction code
    ...

def _create_ripple_view(...) -> pn.Column:
    """Create ripple detection visualization panel."""
    # Extract ~40 lines of ripple plotting code
    ...

def _create_posterior_view(...) -> pn.Column:
    """Create posterior decoding visualization panel."""
    # Extract ~50 lines of posterior plotting code
    ...
```

**Main function becomes**:

```python
def create_interactive_1D_decoding_figurl(...):
    """Create interactive 1D decoding visualization (now ~60 lines)."""
    # Setup and validation
    ...

    # Create views using helpers
    speed_view = _create_speed_view(...)
    direction_view = _create_head_direction_view(...)
    ripple_view = _create_ripple_view(...)
    posterior_view = _create_posterior_view(...)

    # Compose final layout
    ...
```

**Benefits**:

- Easier to test individual components
- Better readability
- Reusable view components

**Time**: 4 hours
**Verification**: Function still produces same output, existing tests pass

### 3.2 Centralize Constants

**File**: `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:8`
**Issue**: `MAX_SPEED_THRESHOLD_CM_PER_SEC = 4.0` should be in central config

**Fix**:

1. Add to `src/continuum_swr_replay/parameters.py`:

   ```python
   # Speed thresholds
   MAX_SPEED_THRESHOLD_CM_PER_SEC: float = 4.0
   ```

2. Update import in `data_aggregation.py`:

   ```python
   from ...parameters import MAX_SPEED_THRESHOLD_CM_PER_SEC
   ```

3. Remove local definition

**Time**: 1 hour
**Verification**: Tests pass, constant accessible from central location

### 3.3 Use Dataclasses for Parameters

**File**: `src/continuum_swr_replay/parameters.py`
**Issue**: Plain dict `decoding_parameters = {...}` lacks type safety

**Refactor**:

```python
from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True)
class DecodingParameters:
    """Parameters for neural decoding analysis.

    Attributes
    ----------
    place_bin_size : float
        Spatial bin size in cm for position discretization
    waveform_std : float
        Standard deviation for waveform feature likelihood in µV
    position_std : float
        Standard deviation for position likelihood in cm
    block_size : int
        Block size for GPU memory management
    """
    place_bin_size: float = 2.0
    waveform_std: float = 24.0
    position_std: float = 6.0
    block_size: int = 4096

# Singleton instance
DECODING_PARAMS: Final = DecodingParameters()
```

**Migration**:

- Update all references from `decoding_parameters['key']` to `DECODING_PARAMS.key`
- Search codebase for `decoding_parameters[` and replace

**Benefits**:

- Immutable configuration (frozen=True)
- Type checking for parameters
- Better IDE autocomplete
- Validation at instantiation

**Time**: 2 hours
**Verification**: All tests pass, no runtime errors, mypy passes

---

## Execution Strategy

### Sprint 1 (Days 1-2): Critical Fixes

**Time: 1 hour**

**Tasks**:

1. Fix import error in `run_decoding_bandit.py` (5 min)
2. Fix syntax error in `generate_figure03.py` (15 min)
3. Fix typo `edege_linewidth` → `edge_linewidth` (20 min)
4. Run full test suite to ensure nothing broken (20 min)
5. Commit: "Fix critical blocking issues from code review"

**Deliverable**: All scripts executable, no blocking issues

**Verification Checklist**:

- [ ] `uv run python scripts/run_decoding_bandit.py --help` works
- [ ] `uv run ruff check scripts/` passes
- [ ] `uv run ruff check src/` passes
- [ ] `uv run pytest tests/` passes
- [ ] Git diff shows only expected changes

### Sprint 2 (Week 1, Days 3-4): Type Hints - Part 1

**Time: 5 hours**

**Tasks**:

1. Add types to `lfp.py` (2 hours)
   - Define TypedDicts for return values
   - Add all function annotations
   - Run mypy after completion
   - Commit: "Add type hints to lfp.py"

2. Add types to `events.py` (1 hour)
   - Add all function annotations
   - Run mypy after completion
   - Commit: "Add type hints to events.py"

3. Add types to `trials.py` (2 hours)
   - Add pandas DataFrame type hints
   - Add all function annotations
   - Run mypy after completion
   - Commit: "Add type hints to trials.py"

**Deliverable**: 3 modules fully typed

**Verification Checklist**:

- [ ] `uv run mypy src/continuum_swr_replay/data_loaders/lfp.py` passes
- [ ] `uv run mypy src/continuum_swr_replay/data_loaders/events.py` passes
- [ ] `uv run mypy src/continuum_swr_replay/data_loaders/trials.py` passes
- [ ] `uv run pytest tests/` passes

### Sprint 3 (Week 1, Days 5-6): Type Hints - Part 2

**Time: 4 hours**

**Tasks**:

1. Add types to `task_variables.py` (2 hours)
   - Add NetworkX and pandas type hints
   - Run mypy after completion
   - Commit: "Add type hints to task_variables.py"

2. Fix remaining mypy errors (1 hour)
   - Fix all 11 mypy errors across visualization modules
   - Run full mypy check
   - Commit: "Fix remaining mypy errors"

3. Replace `dict[str, Any]` with `SessionData` TypedDict (1 hour)
   - Update `load_data()` return type
   - Verify return structure matches TypedDict
   - Commit: "Use TypedDict for load_data return type"

**Deliverable**: >90% type coverage, clean mypy output

**Verification Checklist**:

- [ ] `uv run mypy src/` has <5 errors
- [ ] Type coverage >90% (estimate via mypy report)
- [ ] `uv run pytest tests/` passes
- [ ] IDE autocomplete works for `load_data()` return

### Sprint 4 (Week 2, Days 7-8): Refactoring with Tests

**Time: 3 hours**

**Tasks**:

1. Write tests for `filter_spike_times()` (45 min)
   - Test normal operation
   - Test edge cases
   - Ensure tests pass with current implementation
   - Commit: "Add comprehensive tests for spike filtering"

2. Write tests for `filter_ripple_lfp_time()` (45 min)
   - Test normal operation
   - Test edge cases
   - Ensure tests pass with current implementation
   - Commit: "Add tests for LFP time filtering"

3. Refactor to return new objects (1.5 hours)
   - Refactor `filter_spike_times()` to functional style
   - Refactor `filter_ripple_lfp_time()` to functional style
   - Update all call sites
   - Ensure all tests still pass
   - Commit: "Refactor in-place modifications to functional style"

**Deliverable**: Functional-style filtering with comprehensive tests

**Verification Checklist**:

- [ ] New tests pass before refactoring
- [ ] All tests pass after refactoring
- [ ] No in-place modifications in refactored functions
- [ ] Docstrings updated to reflect new behavior
- [ ] `uv run mypy src/` passes

### Sprint 5 (Week 2, Days 9-10): Code Organization

**Time: 7 hours**

**Tasks**:

1. Extract helper functions in `figurl.py` (4 hours)
   - Extract `_create_speed_view()`
   - Extract `_create_head_direction_view()`
   - Extract `_create_ripple_view()`
   - Extract `_create_posterior_view()`
   - Update main function to use helpers
   - Ensure output unchanged
   - Commit: "Extract view creation helpers in figurl.py"

2. Centralize `MAX_SPEED_THRESHOLD` (1 hour)
   - Add constant to `parameters.py`
   - Update import in `data_aggregation.py`
   - Remove local definition
   - Run tests
   - Commit: "Centralize MAX_SPEED_THRESHOLD constant"

3. Convert parameters to dataclass (2 hours)
   - Create `DecodingParameters` dataclass
   - Update all references in codebase
   - Run full test suite
   - Commit: "Convert decoding parameters to immutable dataclass"

**Deliverable**: Well-organized, maintainable codebase

**Verification Checklist**:

- [ ] `create_interactive_1D_decoding_figurl()` is <100 lines
- [ ] All constants in `parameters.py`
- [ ] `DECODING_PARAMS.place_bin_size` works (not dict access)
- [ ] `uv run mypy src/` passes
- [ ] `uv run pytest tests/` passes
- [ ] `uv run ruff check src/` passes

---

## Testing Strategy

After each sprint, run the full verification checklist:

### Verification Checklist

```bash
# 1. Check code style
uv run ruff check src/

# 2. Check type hints
uv run mypy src/

# 3. Run all tests
uv run pytest tests/

# 4. Verify git status
git status  # Should show only expected changes
```

### Test Coverage Goals

From code review, maintain these coverage targets:

- Critical modules (data_loaders, analysis): >80%
- Analysis modules: >60%
- Visualization: >40%
- Overall: >50%

---

## Deferred Items (Not in Current Scope)

### Explicitly Removed from Plan

- ❌ **Deprecation cycle for typo fix** - Breaking change is acceptable
- ❌ **Remove generate_figure03.py** - Keep and fix instead
- ❌ **Remove/implement PFC module** - Keep as placeholder for future work
- ❌ **Add Pydantic validation** - Not needed for current scope
- ❌ **Pin dependency versions** - Can be done later if needed
- ❌ **Expand integration tests** - Holding off per user request
- ❌ **Add pre-commit hooks** - Can add in future sprint

### May Revisit Later

- Add Pydantic for runtime validation (8 hours)
- Pin dependency versions in pyproject.toml (1 hour)
- Add pre-commit hooks for CI/CD (1 hour)
- Expand integration test coverage (2+ hours)

---

## Success Criteria

### Phase 1 Success (Critical Fixes)

- [ ] All import statements work correctly
- [ ] No syntax errors in any scripts
- [ ] API typos fixed in plots.py
- [ ] All tests passing
- [ ] Scripts executable without errors

### Phase 2 Success (Type Safety)

- [ ] Mypy reports <5 errors across entire codebase
- [ ] Type hint coverage >90%
- [ ] `load_data()` returns typed `SessionData`
- [ ] Filtering functions return new objects (functional style)
- [ ] All tests passing

### Phase 3 Success (Refactoring)

- [ ] No functions >150 lines
- [ ] All constants centralized in `parameters.py`
- [ ] Parameters use frozen dataclass
- [ ] Code is well-organized and maintainable
- [ ] All tests passing

### Overall Success

- [ ] **All tests passing** throughout all phases
- [ ] **Clean mypy output** (<5 errors)
- [ ] **Clean ruff output** (no style violations)
- [ ] **Code review grade improved** from 4/5 to 4.5/5
- [ ] **Codebase is publication-ready**

---

## Progress Tracking

Track progress by updating this section after each sprint:

- [ ] Sprint 1: Critical Fixes (1 hour)
- [ ] Sprint 2: Type Hints Part 1 (5 hours)
- [ ] Sprint 3: Type Hints Part 2 (4 hours)
- [ ] Sprint 4: Refactoring with Tests (3 hours)
- [ ] Sprint 5: Code Organization (7 hours)

**Total Progress**: 0/5 sprints completed (0/18 hours)

---

## Notes

- Always run tests after each change
- Commit after each logical unit of work
- Write descriptive commit messages
- If any sprint takes significantly longer than estimated, reassess approach
- Keep CLAUDE.md updated if architectural patterns change
- Update CODE_REVIEW.md with "RESOLVED" notes as issues are fixed

---

**Plan Status**: Ready to Execute
**Next Action**: Begin Sprint 1 (Phase 1: Critical Fixes)
