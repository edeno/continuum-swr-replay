# Implementation Tasks Checklist

**Date Created**: 2025-11-14
**Based on**: PLAN.md and CODE_REVIEW.md
**Total Estimated Time**: ~18 hours over 2 weeks

---

## Overview

This document provides a detailed task checklist for implementing the code review improvements. Check off tasks as you complete them to track progress.

**Quick Stats**:

- Total Milestones: 5
- Total Tasks: ~60
- Current Progress: 0/60 tasks completed (0%)

---

## Milestone 1: Critical Blocking Issues ⏱️ 1 hour

**Goal**: Fix all import errors, syntax errors, and API typos
**Status**: ⬜ Not Started

### Tasks

#### 1.1 Fix Import Error in run_decoding_bandit.py (5 min)

- [x] Open `scripts/run_decoding_bandit.py`
- [x] Navigate to line 13
- [x] Change `from continuum_swr_replay.visualization.visualization import` to `from continuum_swr_replay.visualization.figurl import`
- [x] Save file
- [x] Verify: Run `uv run python scripts/run_decoding_bandit.py --help`
- [x] Confirm script runs without import error

#### 1.2 Fix Syntax Error in generate_figure03.py (15 min)

- [x] Open `scripts/generate_figure03.py`
- [x] Inspect line 2 and identify syntax error
- [x] Fix syntax error (complete incomplete code or fix invalid statement)
- [x] Save file
- [x] Verify: Run `uv run ruff check scripts/generate_figure03.py`
- [x] Confirm no syntax errors reported

#### 1.3 Fix API Typo: edege_linewidth → edge_linewidth (20 min)

- [x] Open `src/continuum_swr_replay/visualization/plots.py`
- [x] Navigate to line 908 (first occurrence in `plot_graph_as_1D`)
- [x] Change parameter `edege_linewidth` to `edge_linewidth`
- [x] Update docstring for this parameter
- [x] Navigate to line 930 (second occurrence)
- [x] Change parameter `edege_linewidth` to `edge_linewidth`
- [x] Update docstring
- [x] Navigate to line 970 (third occurrence)
- [x] Change parameter `edege_linewidth` to `edge_linewidth`
- [x] Update docstring
- [x] Search codebase for any usage of `edege_linewidth` and update
- [x] Save file

#### 1.4 Verification & Commit (20 min)

- [x] Run `uv run ruff check src/`
- [x] Confirm no errors
- [x] Run `uv run ruff check scripts/`
- [x] Confirm no errors
- [x] Run `uv run pytest tests/`
- [x] Confirm all tests pass
- [x] Review git diff to ensure only expected changes
- [x] Stage changes: `git add scripts/run_decoding_bandit.py scripts/generate_figure03.py src/continuum_swr_replay/visualization/plots.py`
- [x] Commit: `git commit -m "Fix critical blocking issues from code review"`

**Milestone 1 Complete**: ✅

---

## Milestone 2: Type Hints - Data Loaders Part 1 ⏱️ 5 hours

**Goal**: Add type hints to lfp.py, events.py, and trials.py
**Status**: ✅ Complete

### Tasks

#### 2.1 Add Type Hints to lfp.py (2 hours)

- [x] Open `src/continuum_swr_replay/data_loaders/lfp.py`
- [x] Review all functions and identify missing type annotations
- [x] Define TypedDict for `get_theta()` return value at top of file
- [x] Define TypedDict for `get_ripple()` return value
- [x] Define TypedDict for other function returns as needed
- [x] Add type hints to `get_theta()` function signature
- [x] Add type hints to `get_ripple()` function signature
- [x] Add type hints to `get_ripple_consensus_trace()` function signature
- [x] Add type hints to `filter_ripple_lfp_time()` function signature
- [x] Add type hints to any other functions missing them
- [x] Add `NDArray[np.float64]` for numpy arrays
- [x] Add `pd.DataFrame` for pandas DataFrames
- [x] Update all docstrings to match new type signatures
- [x] Save file
- [x] Run `uv run mypy src/continuum_swr_replay/data_loaders/lfp.py`
- [x] Fix any mypy errors reported
- [x] Run `uv run pytest tests/`
- [x] Confirm tests pass
- [x] Stage and commit: `git commit -m "Add type hints to lfp.py"`

#### 2.2 Add Type Hints to events.py (1 hour)

- [x] Open `src/continuum_swr_replay/data_loaders/events.py`
- [x] Review all functions and identify missing type annotations
- [x] Consider defining dataclass for event structures
- [x] Add type hints to all function parameters
- [x] Add type hints to all function returns
- [x] Add `pd.DataFrame` and `NDArray` annotations as appropriate
- [x] Update all docstrings to match new type signatures
- [x] Save file
- [x] Run `uv run mypy src/continuum_swr_replay/data_loaders/events.py`
- [x] Fix any mypy errors reported
- [x] Run `uv run pytest tests/`
- [x] Confirm tests pass
- [x] Stage and commit: `git commit -m "Add type hints to events.py"`

#### 2.3 Add Type Hints to trials.py (2 hours)

- [x] Open `src/continuum_swr_replay/data_loaders/trials.py`
- [x] Review all 5 functions missing type annotations
- [x] Add type hints for pandas DataFrame returns
- [x] Add type hints for all function parameters
- [x] Add type hints for all local variables where helpful
- [x] Update all docstrings to match new type signatures
- [x] Save file
- [x] Run `uv run mypy src/continuum_swr_replay/data_loaders/trials.py`
- [x] Fix any mypy errors reported
- [x] Run `uv run pytest tests/`
- [x] Confirm tests pass
- [x] Stage and commit: `git commit -m "Add type hints to trials.py"`

**Milestone 2 Complete**: ✅

---

## Milestone 3: Type Hints - Data Loaders Part 2 ⏱️ 4 hours

**Goal**: Complete type coverage and fix all mypy errors
**Status**: ⬜ Not Started

### Tasks

#### 3.1 Add Type Hints to task_variables.py (2 hours)

- [x] Open `src/continuum_swr_replay/data_loaders/task_variables.py`
- [x] Import NetworkX types: `import networkx as nx`
- [x] Add type hint to function at line 71
- [x] Add type hint to function at line 119
- [x] Add type hint to function at line 153
- [x] Add type hints to remaining 2 functions
- [x] Add `nx.Graph` type hints for graph parameters
- [x] Add `pd.DataFrame` type hints for DataFrame returns
- [x] Add `NDArray[np.float64]` for numpy arrays
- [x] Update all docstrings to match new type signatures
- [x] Save file
- [x] Run `uv run mypy src/continuum_swr_replay/data_loaders/task_variables.py`
- [x] Fix any mypy errors reported
- [x] Run `uv run pytest tests/`
- [x] Confirm tests pass
- [x] Stage and commit: `git commit -m "Add type hints to task_variables.py"`

#### 3.2 Fix Remaining Mypy Errors (1 hour)

- [x] Open `src/continuum_swr_replay/data_loaders/utils.py`
- [x] Navigate to line 74 (nested function `_count_exposure`)
- [x] Add type hints to nested function
- [x] Save and verify with mypy
- [x] Open `src/continuum_swr_replay/visualization/animation.py`
- [x] Navigate to line 452
- [x] Add missing type annotation
- [x] Save and verify with mypy
- [x] Open `src/continuum_swr_replay/visualization/interactive.py`
- [x] Navigate to line 181 and add type annotation
- [x] Navigate to line 269 and add type annotation
- [x] Save and verify with mypy
- [x] Open `src/continuum_swr_replay/visualization/figurl.py`
- [x] Navigate to line 201 and fix incompatible type assignment
- [x] Navigate to line 637 and add type annotation
- [x] Navigate to line 659 and add type annotation
- [x] Navigate to line 667 and add type annotation
- [x] Save and verify with mypy
- [x] Run `uv run mypy src/` (full codebase check)
- [x] Confirm <5 errors total
- [x] Stage and commit: `git commit -m "Fix remaining mypy errors across visualization modules"`

#### 3.3 Replace dict[str, Any] with TypedDict (1 hour)

- [x] Open `src/continuum_swr_replay/data_loaders/bandit_task.py`
- [x] Add import: `from ..analysis.hippocampus.types import SessionData`
- [x] Locate `load_data()` function signature
- [x] Change return type from `dict[str, Any]` to `SessionData`
- [x] Review function return statement
- [x] Verify all returned keys match SessionData TypedDict
- [x] Add any missing keys or fix mismatched types
- [x] Update function docstring Returns section
- [x] Save file
- [x] Run `uv run mypy src/continuum_swr_replay/data_loaders/bandit_task.py`
- [x] Fix any type errors
- [x] Test IDE autocomplete on `load_data()` return value
- [x] Run `uv run pytest tests/`
- [x] Confirm tests pass
- [x] Stage and commit: `git commit -m "Use SessionData TypedDict for load_data return type"`

**Milestone 3 Complete**: ✅

---

## Milestone 4: Refactor to Functional Style ⏱️ 3 hours

**Goal**: Write tests first, then refactor in-place modifications to return new objects
**Status**: 🔄 In Progress

### Tasks

#### 4.1 Write Tests for filter_spike_times() (45 min)

- [x] Create or open `tests/unit/test_data_loaders.py`
- [x] Import `filter_spike_times` from `continuum_swr_replay.data_loaders.spikes`
- [x] Write test: `test_filter_spike_times_normal_operation()`
  - [x] Create sample spike_times dict with multiple brain areas
  - [x] Create sample spike_waveform_features dict
  - [x] Create sample position_time array
  - [x] Call filter_spike_times()
  - [x] Assert spikes outside time bounds are removed
  - [x] Assert spikes within bounds are kept
- [x] Write test: `test_filter_spike_times_empty_dict()`
  - [x] Test with empty input dicts
- [x] Write test: `test_filter_spike_times_all_filtered()`
  - [x] Test where all spikes are outside bounds
- [x] Write test: `test_filter_spike_times_none_filtered()`
  - [x] Test where all spikes are within bounds
- [x] Write test: `test_filter_spike_times_boundary_conditions()`
  - [x] Test spikes exactly at start/end times
- [x] Run `uv run pytest tests/unit/test_data_loaders.py::test_filter_spike_times -v`
- [x] Confirm all tests pass with current implementation
- [x] Stage and commit: `git commit -m "Add comprehensive tests for filter_spike_times"`

#### 4.2 Write Tests for filter_ripple_lfp_time() (45 min)

- [x] Open `tests/unit/test_lfp.py` (created comprehensive test file)
- [x] Import `filter_ripple_lfp_time` from `continuum_swr_replay.data_loaders.lfp`
- [x] Write test: `test_filter_to_position_bounds()` (normal operation)
  - [x] Create sample ripple LFP data
  - [x] Create sample position_time array
  - [x] Call filter_ripple_lfp_time()
  - [x] Assert times outside bounds are removed
- [x] Write test: `test_empty_dataframes()` (empty data)
  - [x] Test with empty input
- [x] Write test: `test_boundary_conditions_exact_start_end()` (boundary conditions)
  - [x] Test times exactly at boundaries
- [x] Write test: `test_all_data_outside_bounds()` (all data filtered)
- [x] Write test: `test_no_data_filtered_all_within_bounds()` (no data filtered)
- [x] Write test: `test_preserves_data_structure()` (structure preservation)
- [x] Write test: `test_modifies_in_place()` (current in-place behavior)
- [x] Run `uv run pytest tests/unit/test_lfp.py -v`
- [x] Confirm all tests pass with current implementation (7/7 passed)
- [x] Stage and commit: `git commit -m "test(M4): Add comprehensive tests for filter_ripple_lfp_time"`

#### 4.3 Refactor filter_spike_times() to Functional Style (45 min)

- [x] Open `src/continuum_swr_replay/data_loaders/spikes.py`
- [x] Navigate to `filter_spike_times()` function (line ~356)
- [x] Change return type from `None` to `tuple[dict[str, list[NDArray[np.float64]]], dict[str, list[NDArray[np.float64]]]]`
- [x] Create new empty dicts: `filtered_times = {}` and `filtered_features = {}`
- [x] Change all in-place modifications to populate new dicts
- [x] Add return statement: `return filtered_times, filtered_features`
- [x] Update docstring to remove "modifies in place" and add Returns section
- [x] Save file
- [x] Find all call sites of `filter_spike_times()` in codebase
- [x] Update call site in bandit_task.py to capture return values
- [x] Update all 8 test cases in test_spikes.py
- [x] Save all modified files
- [x] Run `uv run pytest tests/unit/test_spikes.py::TestSpikeFiltering -v`
- [x] Confirm all tests still pass (8/8 passed)
- [x] Run `uv run pytest tests/`
- [x] Confirm all tests pass (161/161 passed)
- [x] Run `uv run mypy src/`
- [x] Confirm no type errors
- [x] Stage and commit: `git commit -m "refactor(M4): Refactor filter_spike_times to functional style"`

#### 4.4 Refactor filter_ripple_lfp_time() to Functional Style (45 min)

- [ ] Open `src/continuum_swr_replay/data_loaders/lfp.py`
- [ ] Navigate to `filter_ripple_lfp_time()` function
- [ ] Change return type from `None` to appropriate return type
- [ ] Create new data structures instead of modifying in place
- [ ] Add return statement
- [ ] Update docstring to remove "modifies in place" and add Returns section
- [ ] Save file
- [ ] Find all call sites of `filter_ripple_lfp_time()` in codebase
- [ ] Update each call site to capture return values
- [ ] Save all modified files
- [ ] Run `uv run pytest tests/unit/test_data_loaders.py::test_filter_ripple_lfp_time -v`
- [ ] Confirm all tests still pass
- [ ] Run `uv run pytest tests/`
- [ ] Confirm all tests pass
- [ ] Run `uv run mypy src/`
- [ ] Confirm no new type errors
- [ ] Stage and commit: `git commit -m "Refactor filter_ripple_lfp_time to return new objects"`

**Milestone 4 Complete**: ⬜

---

## Milestone 5: Code Organization & Refactoring ⏱️ 7 hours

**Goal**: Extract long functions, centralize constants, use dataclasses
**Status**: ⬜ Not Started

### Tasks

#### 5.1 Extract Helper Functions in figurl.py (4 hours)

##### 5.1.1 Plan Extraction (30 min)

- [ ] Open `src/continuum_swr_replay/visualization/figurl.py`
- [ ] Navigate to `create_interactive_1D_decoding_figurl()` (line with function start)
- [ ] Read through all 194 lines
- [ ] Identify distinct view creation sections:
  - [ ] Speed view (~lines ?)
  - [ ] Head direction view (~lines ?)
  - [ ] Ripple view (~lines ?)
  - [ ] Posterior view (~lines ?)
- [ ] Note parameters needed for each helper
- [ ] Note return types for each helper

##### 5.1.2 Extract _create_speed_view() (45 min)

- [ ] Create new function `_create_speed_view()` before main function
- [ ] Add type hints for all parameters
- [ ] Add return type: `-> pn.Column`
- [ ] Add comprehensive docstring with Parameters and Returns
- [ ] Move speed visualization code into this function
- [ ] Test function works independently
- [ ] Update main function to call `speed_view = _create_speed_view(...)`
- [ ] Verify output unchanged

##### 5.1.3 Extract _create_head_direction_view() (45 min)

- [ ] Create new function `_create_head_direction_view()`
- [ ] Add type hints for all parameters
- [ ] Add return type: `-> pn.Column`
- [ ] Add comprehensive docstring
- [ ] Move head direction visualization code into this function
- [ ] Update main function to call helper
- [ ] Verify output unchanged

##### 5.1.4 Extract _create_ripple_view() (45 min)

- [ ] Create new function `_create_ripple_view()`
- [ ] Add type hints for all parameters
- [ ] Add return type: `-> pn.Column`
- [ ] Add comprehensive docstring
- [ ] Move ripple visualization code into this function
- [ ] Update main function to call helper
- [ ] Verify output unchanged

##### 5.1.5 Extract _create_posterior_view() (45 min)

- [ ] Create new function `_create_posterior_view()`
- [ ] Add type hints for all parameters
- [ ] Add return type: `-> pn.Column`
- [ ] Add comprehensive docstring
- [ ] Move posterior visualization code into this function
- [ ] Update main function to call helper
- [ ] Verify output unchanged

##### 5.1.6 Verify and Commit (30 min)

- [ ] Verify main function is now <100 lines
- [ ] Run `uv run mypy src/continuum_swr_replay/visualization/figurl.py`
- [ ] Fix any type errors
- [ ] Run `uv run pytest tests/`
- [ ] Confirm all tests pass
- [ ] Check git diff to review changes
- [ ] Stage and commit: `git commit -m "Extract view creation helpers in figurl.py"`

#### 5.2 Centralize MAX_SPEED_THRESHOLD Constant (1 hour)

- [ ] Open `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py`
- [ ] Locate `MAX_SPEED_THRESHOLD_CM_PER_SEC = 4.0` (line ~8)
- [ ] Copy the constant definition and its comment
- [ ] Open `src/continuum_swr_replay/parameters.py`
- [ ] Add constant in appropriate section with comment
- [ ] Save `parameters.py`
- [ ] Return to `data_aggregation.py`
- [ ] Remove local constant definition
- [ ] Add import: `from ...parameters import MAX_SPEED_THRESHOLD_CM_PER_SEC`
- [ ] Save file
- [ ] Search codebase for other uses of this constant
- [ ] Update any other files to import from `parameters.py`
- [ ] Run `uv run pytest tests/`
- [ ] Confirm tests pass
- [ ] Run `uv run mypy src/`
- [ ] Confirm no errors
- [ ] Stage and commit: `git commit -m "Centralize MAX_SPEED_THRESHOLD constant in parameters.py"`

#### 5.3 Convert Parameters to Dataclass (2 hours)

##### 5.3.1 Create DecodingParameters Dataclass (45 min)

- [ ] Open `src/continuum_swr_replay/parameters.py`
- [ ] Add imports: `from dataclasses import dataclass` and `from typing import Final`
- [ ] Locate `decoding_parameters` dict
- [ ] Create `DecodingParameters` dataclass before the dict:

  ```python
  @dataclass(frozen=True)
  class DecodingParameters:
      """Parameters for neural decoding analysis."""
      place_bin_size: float = 2.0
      waveform_std: float = 24.0
      position_std: float = 6.0
      block_size: int = 4096
  ```

- [ ] Add comprehensive docstring with Attributes section
- [ ] Create singleton instance: `DECODING_PARAMS: Final = DecodingParameters()`
- [ ] Keep old dict for now (for backward compatibility during migration)
- [ ] Save file
- [ ] Run `uv run mypy src/continuum_swr_replay/parameters.py`
- [ ] Fix any type errors

##### 5.3.2 Update All References (1 hour)

- [ ] Search codebase for `decoding_parameters['place_bin_size']`
- [ ] Replace with `DECODING_PARAMS.place_bin_size`
- [ ] Search codebase for `decoding_parameters['waveform_std']`
- [ ] Replace with `DECODING_PARAMS.waveform_std`
- [ ] Search codebase for `decoding_parameters['position_std']`
- [ ] Replace with `DECODING_PARAMS.position_std`
- [ ] Search codebase for `decoding_parameters['block_size']`
- [ ] Replace with `DECODING_PARAMS.block_size`
- [ ] Search for any remaining `decoding_parameters[` usage
- [ ] Update all found instances
- [ ] Save all modified files

##### 5.3.3 Remove Old Dict and Verify (15 min)

- [ ] Open `src/continuum_swr_replay/parameters.py`
- [ ] Remove old `decoding_parameters = {...}` dict
- [ ] Save file
- [ ] Run `uv run mypy src/`
- [ ] Confirm no type errors
- [ ] Run `uv run pytest tests/`
- [ ] Confirm all tests pass
- [ ] Run `uv run ruff check src/`
- [ ] Confirm no errors
- [ ] Stage and commit: `git commit -m "Convert decoding parameters to immutable dataclass"`

**Milestone 5 Complete**: ⬜

---

## Final Verification Checklist

Before marking the project as complete, verify all success criteria:

### Phase 1 Success (Critical Fixes)

- [ ] All import statements work correctly
- [ ] No syntax errors in any scripts
- [ ] API typos fixed in plots.py
- [ ] All tests passing
- [ ] Scripts executable without errors

### Phase 2 Success (Type Safety)

- [ ] Mypy reports <5 errors across entire codebase
- [ ] Type hint coverage >90% (estimated)
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

- [ ] **All tests passing** - Run `uv run pytest tests/`
- [ ] **Clean mypy output** - Run `uv run mypy src/` (<5 errors)
- [ ] **Clean ruff output** - Run `uv run ruff check src/` (no violations)
- [ ] **No blocking issues remain**
- [ ] **Code review grade improved** from 4/5 to 4.5/5
- [ ] **Codebase is publication-ready**

---

## Progress Summary

Track your overall progress here:

### Milestones

- [x] Milestone 1: Critical Blocking Issues (4/4 task groups) ✅
- [x] Milestone 2: Type Hints - Data Loaders Part 1 (3/3 task groups) ✅
- [x] Milestone 3: Type Hints - Data Loaders Part 2 (3/3 task groups) ✅
- [ ] Milestone 4: Refactor to Functional Style (1/4 task groups) 🔄
- [ ] Milestone 5: Code Organization & Refactoring (0/3 task groups)

### Time Tracking

- [x] Sprint 1: Critical Fixes - 1 hour (Milestone 1) ✅
- [x] Sprint 2: Type Hints Part 1 - 2 hours actual (Milestone 2) ✅
- [x] Sprint 3: Type Hints Part 2 - 1.5 hours actual (Milestone 3) ✅
- [ ] Sprint 4: Refactoring with Tests - 3 hours (Milestone 4)
- [ ] Sprint 5: Code Organization - 7 hours (Milestone 5)

**Total Progress**: 3/5 milestones completed (~4.5/18 hours)
**Completion**: 60%

---

## Notes

- ✅ Check off tasks as you complete them
- 📝 Add notes below any task if you encounter issues
- ⏱️ Track actual time spent vs. estimates
- 🔄 If a task takes >50% longer than estimated, reassess approach
- 💾 Commit after each logical group of tasks
- 🧪 Always run tests before committing

---

## Next Actions

**Immediate Next Task**: Start Milestone 1, Task 1.1 (Fix import error in run_decoding_bandit.py)

**Command to begin**: Open `scripts/run_decoding_bandit.py` and navigate to line 13.

---

**Status**: Ready to Execute
**Last Updated**: 2025-11-14
