# Scripts Refactoring Tasks

This file tracks the implementation of PLAN.md. Each task should be marked complete `[x]` when finished and verified.

**Reference:** See PLAN.md for detailed specifications and rationale.

---

## Milestone 1: Event Analysis Module (Week 1) ✅ COMPLETE

**Goal:** Create `src/continuum_swr_replay/analysis/hippocampus/event_analysis.py` with utilities for filtering and characterizing detected events.

**Completed:** 2025-11-19

### Implementation

- [x] Create `src/continuum_swr_replay/analysis/hippocampus/event_analysis.py`
- [x] Implement `get_immobile_events()` with validation
  - [x] Add parameter validation (speed_threshold > 0, method in allowed values)
  - [x] Add column existence checks
  - [x] Implement "endpoints", "max", "mean" filtering methods
- [x] Implement `get_event_start_end_times()`
  - [x] Convert event labels to start/end time intervals
  - [x] Add duration column calculation
  - [x] Filter out non-events (event_col == 0)
- [x] Implement `get_event_durations_ms()` convenience function
  - [x] Validate 'duration' column exists
  - [x] Convert seconds to milliseconds

### Testing

- [x] Create `tests/analysis/hippocampus/test_event_analysis.py`
- [x] Test `get_immobile_events()`:
  - [x] Test endpoints method filters correctly
  - [x] Test max method filters correctly
  - [x] Test mean method filters correctly
  - [x] Test empty data handling
  - [x] Test invalid parameters raise ValueError
- [x] Test `get_event_start_end_times()`:
  - [x] Test label-to-interval conversion
  - [x] Test duration calculation accuracy
  - [x] Test non-events are excluded
  - [x] Test empty data returns empty DataFrame
- [x] Test `get_event_durations_ms()`:
  - [x] Test unit conversion (seconds to ms)
  - [x] Test missing column raises ValueError
- [x] Verify >80% code coverage: 21/21 tests passing (100%)

### Code Quality

- [x] Run ruff: All checks passed
- [x] Run mypy: No issues found
- [x] Update `__init__.py` to export new functions
- [x] Applied scientific-code-reviewer agent and fixed all issues

**Acceptance Criteria:** ✅ Module created, all functions implemented with docstrings and type hints, tests pass with >80% coverage, linting passes.

---

## Milestone 2: Event-Triggered Analysis Module (Week 1-2) ✅ COMPLETE

**Goal:** Create `src/continuum_swr_replay/analysis/hippocampus/triggered_analysis.py` for extracting signal windows around events.

**Completed:** 2025-11-19

### Implementation

- [x] Create `src/continuum_swr_replay/analysis/hippocampus/triggered_analysis.py`
- [x] Implement `get_signal_windows()`:
  - [x] Add comprehensive input validation (sampling_frequency, window_sec, required columns)
  - [x] Implement window parameter calculation
  - [x] Add bounds checking with `np.clip()` for searchsorted indices
  - [x] Add out-of-range event warnings
  - [x] Handle empty sessions with logging
  - [x] Support event_id_col for MultiIndex coordinates
  - [x] Return properly structured xr.DataArray with dimensions (event, time, signal)

### Testing

- [x] Create `tests/analysis/hippocampus/test_triggered_analysis.py`
- [x] Test `get_signal_windows()`:
  - [x] Test normal window extraction
  - [x] Test events near data boundaries (partial windows)
  - [x] Test events outside data range (NaN handling)
  - [x] Test empty events returns empty DataArray with proper structure
  - [x] Test invalid sampling_frequency raises ValueError
  - [x] Test missing columns raise ValueError
  - [x] Test event_id_col creates proper MultiIndex
  - [x] Test dimension validation (event, time, signal)
- [x] Create fixtures with mock event data
- [x] Verify >80% code coverage (14/14 tests passing = 100%)

### Code Quality

- [x] Run ruff: `uv run ruff check src/continuum_swr_replay/analysis/hippocampus/triggered_analysis.py`
- [x] Run mypy: `uv run mypy src/continuum_swr_replay/analysis/hippocampus/triggered_analysis.py`
- [x] Update `__init__.py` to export new functions
- [x] Applied scientific-code-reviewer agent and addressed all critical issues

**Acceptance Criteria:** ✅ Module created, function implemented with robust edge case handling, comprehensive tests pass with >80% coverage, linting passes.

---

## Milestone 3: Analysis Constants (Week 2) ✅ COMPLETE

**Goal:** Create `src/continuum_swr_replay/analysis/constants.py` with shared constants.

**Completed:** 2025-11-19

### Implementation

- [x] Create `src/continuum_swr_replay/analysis/constants.py`
- [x] Add event detection thresholds:
  - [x] `DEFAULT_NON_LOCAL_THRESHOLD = 0.8`
  - [x] `DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC = 4.0`
  - [x] `DEFAULT_SAMPLING_FREQUENCY_HZ = 500.0`
- [x] Add histogram bin definitions:
  - [x] `BINS_DURATION_MS`
  - [x] `BINS_PROBABILITY`
  - [x] `BINS_DISTANCE_CM`
  - [x] `BINS_HPD_CM`
  - [x] `BINS_SPEED_CM_PER_SEC`
- [x] Add event-triggered analysis constants:
  - [x] `DEFAULT_WINDOW_EVENT_TRIGGERED_SEC = 0.25`
- [x] Add module docstring referencing `parameters.py` pattern

### Testing

- [x] Test constants are importable: `from continuum_swr_replay.analysis.constants import BINS_DURATION_MS`
- [x] Verify constants have expected types and values

**Acceptance Criteria:** ✅ Constants module created, all constants defined, importable from analysis subpackage.

---

## Milestone 4: Baseline Figure Generation (Week 2)

**Goal:** Generate baseline figures from current scripts for regression testing.

### Baseline Creation

- [ ] Run `python scripts/generate_figure02.py` and save outputs to `tests/baselines/figure02/`
- [ ] Run `python scripts/generate_figure03.py` and save outputs to `tests/baselines/figure03/`
- [ ] Run `python scripts/generate_figure04.py` and save outputs to `tests/baselines/figure04/`
- [ ] Document baseline generation process in `tests/baselines/README.md`

**Acceptance Criteria:** Baseline figures generated and saved for comparison during refactoring.

---

## Milestone 5: Refactor generate_figure03.py (Week 3)

**Goal:** Reduce figure03 from 1018 lines to ~300 lines using new utilities.

**Dependencies:** Milestones 1, 2, 3 must be complete.

### Refactoring

- [ ] Create backup: `cp scripts/generate_figure03.py scripts/generate_figure03.py.backup`
- [ ] Update imports to use new modules:
  - [ ] Import from `analysis.hippocampus.event_analysis`
  - [ ] Import from `analysis.hippocampus.triggered_analysis`
  - [ ] Import from `analysis.constants`
- [ ] Replace `get_event_start_end()` with `get_event_start_end_times()`
- [ ] Replace `_get_signal_windows()` with `get_signal_windows()`
- [ ] Replace speed filtering with `get_immobile_events()`
- [ ] Replace duration calculation with `get_event_durations_ms()`
- [ ] Add CLI arguments:
  - [ ] `--output-dir`
  - [ ] `--format`
  - [ ] `--dpi`
  - [ ] `--data-dir`
- [ ] Add module docstring
- [ ] Organize code into panel-generating functions

### Verification

- [ ] Run refactored script: `python scripts/generate_figure03.py --output-dir /tmp/test_figures`
- [ ] Compare outputs with baseline (visual inspection or data comparison)
- [ ] Verify script is <350 lines: `wc -l scripts/generate_figure03.py`
- [ ] Test CLI arguments work correctly

**Acceptance Criteria:** Figure03 refactored to <350 lines, uses library functions, generates identical output to baseline, CLI functional.

---

## Milestone 6: Refactor generate_figure02.py (Week 3-4)

**Goal:** Break monolithic 2015-line script into 4-5 focused scripts of ~200-300 lines each.

**Dependencies:** Milestones 1, 2, 3 must be complete.

### Analysis

- [ ] Review figure02 structure and identify logical breakpoints
- [ ] Map line ranges to new scripts:
  - [ ] Lines 1-283 → `generate_figure02a_distributions.py`
  - [ ] Lines 284-669 → `generate_figure02b_triggered_averages.py`
  - [ ] Lines 670-1229 → `generate_figure02c_overlaps.py`
  - [ ] Lines 1230-2015 → `generate_figure02d_timing.py`

### Implementation

- [ ] Create `scripts/generate_figure02a_distributions.py`:
  - [ ] Extract probability distribution plots
  - [ ] Use constants from `analysis.constants`
  - [ ] Add CLI arguments
  - [ ] Add docstring
- [ ] Create `scripts/generate_figure02b_triggered_averages.py`:
  - [ ] Extract event-triggered average plots
  - [ ] Use `get_signal_windows()`
  - [ ] Add CLI arguments
  - [ ] Add docstring
- [ ] Create `scripts/generate_figure02c_overlaps.py`:
  - [ ] Extract overlap analysis plots
  - [ ] Use event analysis utilities
  - [ ] Add CLI arguments
  - [ ] Add docstring
- [ ] Create `scripts/generate_figure02d_timing.py`:
  - [ ] Extract timing relationship plots
  - [ ] Use event analysis utilities
  - [ ] Add CLI arguments
  - [ ] Add docstring
- [ ] Rename original: `mv scripts/generate_figure02.py scripts/generate_figure02_old.py`

### Verification

- [ ] Run each new script and verify outputs
- [ ] Compare with baseline figure02 outputs
- [ ] Verify each script is <350 lines
- [ ] Test CLI arguments for all scripts

**Acceptance Criteria:** Figure02 split into 4-5 scripts, each <350 lines, uses library functions, generates identical outputs to baseline, all CLIs functional.

---

## Milestone 7: Refactor generate_figure04.py (Week 4)

**Goal:** Reduce figure04 from 291 lines to ~150 lines using library functions.

**Dependencies:** Visualization utilities may need to be added first.

### Implementation

- [ ] Create backup: `cp scripts/generate_figure04.py scripts/generate_figure04.py.backup`
- [ ] Assess if GLM plotting functions should be extracted to visualization module
- [ ] If yes, create `src/continuum_swr_replay/visualization/glm_plots.py`:
  - [ ] Implement `plot_raster_relative_to_event()`
  - [ ] Implement `plot_glm_modulation()`
  - [ ] Add tests
- [ ] Update figure04 to use library functions
- [ ] Add CLI arguments for session selection
- [ ] Add module docstring

### Verification

- [ ] Run refactored script with different sessions
- [ ] Compare output with baseline
- [ ] Verify script is <200 lines
- [ ] Test CLI arguments work

**Acceptance Criteria:** Figure04 refactored to <200 lines, GLM plotting functions extracted if reusable, generates identical output, CLI functional.

---

## Milestone 8: CLI Standardization (Week 5)

**Goal:** Ensure all figure scripts have consistent CLI interface.

### Implementation

- [ ] Review all figure scripts for CLI consistency
- [ ] Ensure all have:
  - [ ] `--output-dir` argument
  - [ ] `--format` argument (pdf, png, svg)
  - [ ] `--dpi` argument
  - [ ] `--data-dir` argument (where applicable)
- [ ] Update `generate_all_figures.py` if it exists
- [ ] Document CLI usage in script docstrings

### Verification

- [ ] Test `--help` for all scripts shows consistent format
- [ ] Test each CLI argument works correctly
- [ ] Verify output directories are created if missing

**Acceptance Criteria:** All figure scripts have standardized CLI with consistent argument names and behavior.

---

## Milestone 9: Regression Testing (Week 5)

**Goal:** Automated regression tests to ensure figures remain identical after refactoring.

### Implementation

- [ ] Create `tests/integration/test_figure_regression.py`
- [ ] Implement data-level comparison for figure outputs
- [ ] Add tests for each refactored figure script
- [ ] Document regression testing procedure

### Testing

- [ ] Run regression tests: `uv run pytest tests/integration/test_figure_regression.py`
- [ ] Verify all tests pass
- [ ] Document any acceptable differences (if any)

**Acceptance Criteria:** Regression tests implemented, passing, and documented.

---

## Milestone 10: Documentation Updates (Week 5-6)

**Goal:** Update project documentation to reflect new structure.

### Updates

- [ ] Update CLAUDE.md:
  - [ ] Add section on figure generation
  - [ ] Document new analysis modules
  - [ ] Update examples to use new utilities
- [ ] Update module docstrings with usage examples:
  - [ ] `event_analysis.py`
  - [ ] `triggered_analysis.py`
- [ ] Create/update `tests/README.md`:
  - [ ] Explain test organization
  - [ ] Document baseline generation
  - [ ] Document regression testing
- [ ] Review and update inline code comments

### Verification

- [ ] Read through updated documentation for clarity
- [ ] Verify examples in docstrings are correct
- [ ] Check for broken cross-references

**Acceptance Criteria:** All documentation updated, examples tested, no broken references.

---

## Milestone 11: Final Verification (Week 6)

**Goal:** Comprehensive verification before marking refactoring complete.

### Code Quality

- [ ] Run full test suite: `uv run pytest`
- [ ] Verify >80% overall test coverage
- [ ] Run ruff on all modified files: `uv run ruff check src/ scripts/`
- [ ] Run mypy on all modified files: `uv run mypy src/`
- [ ] Check for any TODO/FIXME comments

### Functionality

- [ ] Generate all figures with refactored scripts
- [ ] Visual comparison with baseline figures
- [ ] Verify figure quality (resolution, colors, labels)
- [ ] Test CLI arguments across all scripts

### Cleanup

- [ ] Remove backup files (`.backup` suffixes)
- [ ] Remove any temporary test files
- [ ] Archive old monolithic scripts if needed
- [ ] Update `.gitignore` if necessary

### Documentation

- [ ] Review PLAN.md for any missed items
- [ ] Review TASKS.md completion status
- [ ] Update CHANGELOG.md if it exists
- [ ] Create summary report of changes

**Acceptance Criteria:** All tests passing, all scripts functional, documentation complete, code quality checks pass, figures verified.

---

## Progress Summary

Track overall progress here:

- [x] Milestone 1: Event Analysis Module ✅ (2025-11-19)
- [x] Milestone 2: Event-Triggered Analysis Module ✅ (2025-11-19)
- [x] Milestone 3: Analysis Constants ✅ (2025-11-19)
- [ ] Milestone 4: Baseline Figure Generation
- [ ] Milestone 5: Refactor generate_figure03.py
- [ ] Milestone 6: Refactor generate_figure02.py
- [ ] Milestone 7: Refactor generate_figure04.py
- [ ] Milestone 8: CLI Standardization
- [ ] Milestone 9: Regression Testing
- [ ] Milestone 10: Documentation Updates
- [ ] Milestone 11: Final Verification

**Target Completion:** Week 6
**Current Status:** Milestones 1-3 complete (3/11)

---

## Notes

- **Dependencies:** Some milestones have prerequisites. Check "Dependencies" section before starting.
- **Testing:** Each implementation milestone includes integrated testing. Do not skip tests.
- **Code Review:** After each milestone, consider having code reviewed before proceeding.
- **Fallback:** If a milestone takes longer than expected, see PLAN.md "Fallback Strategy" section.
- **Questions:** Refer to PLAN.md for detailed specifications and architectural decisions.

---

## Tracking Legend

- `[ ]` - Not started
- `[x]` - Complete
- `[~]` - In progress (use if needed for multi-day tasks)
- `[?]` - Blocked or needs clarification

**Last Updated:** 2025-11-19 (Milestones 1-2 complete)
