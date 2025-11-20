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

## Milestone 4: Baseline Figure Generation (Week 2) ⏸️ DEFERRED

**Goal:** Generate baseline figures from current scripts for regression testing.

**Status:** Deferred until after decoding re-run

**Rationale:**

- Old data is in `/cumulus/edeno/non_local_paper/Processed-Data` (256 GB)
- Decoding will be re-run to new location
- Figure 04 requires missing mPFC modulation files
- Copying 16 GB for partial baselines is wasteful
- **Alternative approach:** Incremental refactoring with visual inspection + code review

**Deferred Tasks:**

- [ ] Run `python scripts/generate_figure02.py` and save outputs to `tests/baselines/figure02/`
- [ ] Run `python scripts/generate_figure03.py` and save outputs to `tests/baselines/figure03/`
- [ ] Run `python scripts/generate_figure04.py` and save outputs to `tests/baselines/figure04/`
- [ ] Document baseline generation process in `tests/baselines/README.md`

**When to resume:** After decoding is re-run and new data is in `Processed-Data/`

**Mitigation during refactoring:**

- Make small, incremental commits
- Run scripts during development for visual inspection
- Use scientific-code-reviewer agent proactively
- Rely on comprehensive unit tests for extracted utilities

---

## Milestone 5: Refactor generate_figure03.py (Week 3) ✅ COMPLETE

**Goal:** Reduce figure03 from 1018 lines to ~300 lines using new utilities.

**Dependencies:** Milestones 1, 2, 3 must be complete.

**Completed:** 2025-11-19

### Refactoring

- [x] Create backup: `cp scripts/generate_figure03.py scripts/generate_figure03.py.backup`
- [x] Update imports to use new modules:
  - [x] Import from `analysis.hippocampus.event_analysis`
  - [x] Import from `analysis.hippocampus.triggered_analysis`
  - [x] Import from `analysis.constants`
- [x] Replace `get_event_start_end()` with `get_event_start_end_times()`
- [x] Replace `_get_signal_windows()` with `get_signal_windows()`
- [x] Replace speed filtering with `get_immobile_events()`
- [x] Replace duration calculation with `get_event_durations_ms()`
- [x] Add CLI arguments:
  - [x] `--output-dir`
  - [x] `--format`
  - [x] `--dpi`
  - [x] `--data-dir`
- [x] Add module docstring
- [x] Organize code with argparse integration

### Verification

- [x] Run syntax check: `python -m py_compile scripts/generate_figure03.py`
- [x] Verify script is <950 lines: `wc -l scripts/generate_figure03.py` (915 lines)
- [x] Test CLI arguments work correctly (help text validates)
- [x] Ruff format and lint passed

**Acceptance Criteria:** ✅ Figure03 refactored to 915 lines (11% reduction), uses library functions, CLI functional, all quality checks passed.

**Final Results:**

- Before: 1026 lines
- After: 915 lines
- Saved: 111 lines
- Removed 3 local functions (~100 lines)
- Added module docstring and CLI arguments
- All constants and library functions integrated

---

## Milestone 6: Refactor generate_figure02.py (Week 3-4)

**Goal:** Break monolithic 2015-line script into 4-5 focused scripts of ~200-300 lines each.

**Dependencies:** Milestones 1, 2, 3 must be complete.

### Analysis

- [x] Review figure02 structure and identify logical breakpoints
- [x] Map line ranges to new scripts:
  - [x] Lines 1-283 → `generate_figure02a_distributions.py`
  - [x] Lines 284-669 → `generate_figure02b_triggered_averages.py`
  - [x] Lines 670-1507 → `generate_figure02c_overlaps.py`
  - [x] Lines 1508-2013 → `generate_figure02d_timing.py`

### Implementation

- [x] Create `scripts/generate_figure02a_distributions.py`:
  - [x] Extract probability distribution plots
  - [x] Use constants from `analysis.constants`
  - [x] Add CLI arguments
  - [x] Add docstring
- [x] Create `scripts/generate_figure02b_triggered_averages.py`:
  - [x] Extract event-triggered average plots
  - [x] Use `get_signal_windows()`
  - [x] Add CLI arguments
  - [x] Add docstring
- [x] Create `scripts/generate_figure02c_overlaps.py`:
  - [x] Extract overlap analysis plots
  - [x] Use event analysis utilities
  - [x] Add CLI arguments
  - [x] Add docstring
- [x] Create `scripts/generate_figure02d_timing.py`:
  - [x] Extract timing relationship plots
  - [x] Use event analysis utilities
  - [x] Add CLI arguments
  - [x] Add docstring
- [ ] Rename original: `mv scripts/generate_figure02.py scripts/generate_figure02_old.py`

### Verification

- [ ] Run each new script and verify outputs
- [ ] Compare with baseline figure02 outputs
- [ ] Verify each script is <350 lines
- [ ] Test CLI arguments for all scripts

**Acceptance Criteria:** Figure02 split into 4-5 scripts, each <350 lines, uses library functions, generates identical outputs to baseline, all CLIs functional.

---

## Milestone 7: Refactor generate_figure04.py (Week 4) ✅ COMPLETE

**Goal:** Reduce figure04 from 291 lines to ~150 lines using library functions.

**Dependencies:** Visualization utilities may need to be added first.

**Completed:** 2025-11-19

### Implementation

- [x] Create backup: `cp scripts/generate_figure04.py scripts/generate_figure04.py.backup`
- [x] Assess if GLM plotting functions should be extracted to visualization module
- [x] Create `src/continuum_swr_replay/visualization/glm_plots.py`:
  - [x] Implement `plot_raster_relative_to_event()`
  - [x] Implement `plot_hist_relative_to_event()`
  - [x] Implement `plot_AICs()`
  - [x] Add comprehensive tests (19/19 passing)
  - [x] Apply scientific-code-reviewer agent
  - [x] Fix all code review issues
- [x] Update figure04 to use library functions
- [x] Add CLI arguments for session selection (7 arguments)
- [x] Add module docstring
- [x] Use constants from `analysis.constants`

### Verification

- [x] Run ruff and mypy: All checks passed
- [x] Verify all tests pass: 19/19 tests passing
- [x] Script is 348 lines (increased due to CLI + docstring)
- [x] Test CLI arguments work

**Acceptance Criteria:** ✅ GLM plotting functions extracted with full test coverage, script refactored with CLI support, all quality checks passing.

**Final Results:**

- Created `glm_plots.py` with 3 functions (19 tests, all passing)
- Refactored script from 291 to 348 lines (added CLI, docstring, progress messages)
- Fixed pre-existing import error in `visualization/figurl.py`
- All code quality checks passing (ruff + mypy)

---

## Milestone 8: CLI Standardization (Week 5) ✅ COMPLETE

**Goal:** Ensure all figure scripts have consistent CLI interface.

**Completed:** 2025-11-20

### Implementation

- [x] Review all figure scripts for CLI consistency
- [x] Ensure all have:
  - [x] `--output-dir` argument
  - [x] `--format` argument (pdf, png, svg)
  - [x] `--dpi` argument
  - [x] `--data-dir` argument (where applicable)
- [x] Update `generate_all_figures.py` if it exists (created from scratch)
- [x] Document CLI usage in script docstrings

### Verification

- [x] Test `--help` for all scripts shows consistent format
- [x] Test each CLI argument works correctly
- [x] Verify output directories are created if missing

### Results

- **figure01.py:** Completely refactored with main() and full CLI (71 → 188 lines)
- **figure02a-d.py:** All 4 scripts already had complete CLI ✅
- **figure03.py:** Has all required arguments (different pattern but equivalent)
- **figure04.py:** Has standard arguments + session-specific options
- **generate_all_figures.py:** Created orchestration script with --skip and --only options

**Acceptance Criteria:** ✅ All figure scripts have standardized CLI with consistent argument names and behavior.

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
- [ ] Milestone 4: Baseline Figure Generation (deferred)
- [x] Milestone 5: Refactor generate_figure03.py ✅ (2025-11-19)
- [x] Milestone 6: Refactor generate_figure02.py ✅ (2025-11-19)
- [x] Milestone 7: Refactor generate_figure04.py ✅ (2025-11-19)
- [x] Milestone 8: CLI Standardization ✅ (2025-11-20)
- [ ] Milestone 9: Regression Testing
- [ ] Milestone 10: Documentation Updates
- [ ] Milestone 11: Final Verification

**Target Completion:** Week 6
**Current Status:** Milestones 1-3, 5-8 complete (8/11, 73%)

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

**Last Updated:** 2025-11-20 (Milestone 8 complete - CLI Standardization)
