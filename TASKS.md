# Figure Utilities & Codebase Improvement Tasks

**Status:** In Progress
**Last Updated:** 2025-11-14
**References:**

- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Full codebase review findings
- [FIGURE_UTILITIES_REVIEW.md](FIGURE_UTILITIES_REVIEW.md) - Detailed figure utilities review

---

## Milestone 1: Critical Fixes (BLOCKING) 🔴

**Priority:** IMMEDIATE
**Estimated Time:** 2-4 hours
**Goal:** Fix bugs that prevent safe usage and cause silent failures

### figure_utilities.py Critical Fixes

- [x] **FU-1.1:** Fix `facecolor` type hint in `save_figure()` (lines 387, 447)
  - Accept `ColorType | None` instead of `str | None`
  - Type error from `fig.get_facecolor()` returning tuple/array
  - **COMPLETED:** Used matplotlib.typing.ColorType for proper type safety

- [x] **FU-1.2:** Remove unused imports (lines 77, 87)
  - Remove `import os`
  - Remove `from matplotlib.figure import Figure`
  - **COMPLETED:** Both imports removed, added ColorType import

- [x] **FU-1.3:** Add `stacklevel=2` to all `warnings.warn()` calls (lines 701, 706, 722)
  - 3 warnings in `validate_figure_quality()`
  - **COMPLETED:** All 3 warnings now use stacklevel=2

- [x] **FU-1.4:** Cast DPI to int in `save_figure()` (line 441)
  - `dpi = int(mpl.rcParams.get("savefig.dpi", 300))`
  - **COMPLETED:** DPI now cast to int for type safety (handles float from rcParams)

- [x] **FU-1.5:** Add format validation in `save_figure()`
  - Define `VALID_FORMATS` constant
  - Validate before saving, raise `ValueError` for invalid formats
  - **COMPLETED:** Added VALID_FORMATS frozenset, validates before saving with helpful error messages

### Codebase Critical Fixes

- [x] **CB-1.1:** Fix unsafe `zip()` without `strict=True` in utils.py (lines 54-63, 68-69)
  - Change `strict=False` to `strict=True`
  - **COMPLETED:** Both zip() calls now use strict=True for safer unpacking

- [x] **CB-1.2:** Refactor filename parsing with regex in utils.py (lines 56-58)
  - Create `parse_nwb_filename()` function
  - Eliminate double `.replace()` call
  - **COMPLETED:** Created parse_nwb_filename() with regex validation, eliminated redundant string operations

- [x] **CB-1.3:** Remove redundant `enumerate()` with `itertuples()` in utils.py (line 118)
  - Use `row.Index` instead of enumerate counter
  - **NOT APPLICABLE:** enumerate() is NOT redundant - it provides sequential 0-based counter independent of DataFrame index
  - DataFrame indices may be non-sequential (e.g., [1, 2, 3]) while labels must be sequential [1, 2, 3]
  - Tests verify this behavior: reverting to row.Index caused 3 test failures
  - enumerate() is the correct implementation here

---

## Milestone 2: UX Critical Fixes (HIGH PRIORITY) 🟡

**Priority:** SAME DAY
**Estimated Time:** 2-3 hours
**Goal:** Remove workflow friction and prevent user confusion

- [x] **FU-2.1:** Make `save_figure()` use `FIGURE_DIR` by default
  - Add `base_dir: str | Path | None = None` parameter
  - Default to `FIGURE_DIR` when None
  - Create output path as `Path(base_dir) / figure_name`
  - **COMPLETED:** Added base_dir parameter with FIGURE_DIR default, handles savefig.dpi='figure' case

- [x] **FU-2.2:** Validate figure exists before saving in `save_figure()`
  - Check `fig.get_axes()` is not empty
  - Raise `ValueError` with helpful hint if no axes found
  - **COMPLETED:** Added validation with helpful error message suggesting plt.subplots() or fig.add_subplot()

- [x] **FU-2.3:** Fix `validate_figure_quality()` modifying state (line 716)
  - Remove `plt.tight_layout()` call
  - Make validation read-only
  - **COMPLETED:** Removed tight_layout() call and layout warning check, validation is now pure/read-only

- [x] **FU-2.4:** Improve error messages in `label_panels()` (line 656)
  - Add explicit length validation before zip
  - Provide helpful error with label count and axes count
  - **COMPLETED:** Added explicit validation with helpful message showing both label and axes counts

- [x] **FU-2.5:** Reset matplotlib state in `set_figure_defaults()` (line 236+)
  - Call `mpl.rcParams.update(mpl.rcParamsDefault)` first
  - Prevents stale settings from previous context
  - **COMPLETED:** Added mpl.rcParams.update(mpl.rcParamsDefault) at function start, ensures clean state

---

## Milestone 3: Style & Quality (MEDIUM PRIORITY) 🟢

**Priority:** THIS WEEK
**Estimated Time:** 3-4 hours
**Goal:** Improve code consistency and Python idioms

### Codebase Improvements

- [x] **CB-3.1:** Use `|` operator for dict merging in bandit_task.py (line 181-192)
  - Replace `{**dict1, **dict2}` with `dict1 | dict2`
  - **NOT APPLICABLE:** `{**dict}` syntax has superior type inference for TypedDict merging in mypy
  - The `|` operator returns `dict[str, object]` which is incompatible with `SessionData` TypedDict
  - Current syntax is actually the CORRECT approach for type-safe dict merging with TypedDicts
  - Removed unused `Any` import as cleanup

- [x] **CB-3.2:** Fix SCREAMING_CASE for constants in parameters.py (line 67)
  - Rename `training_types` to `TRAINING_TYPES`
  - **COMPLETED:** Renamed constant to follow PEP 8 naming conventions
  - No usages in codebase, simple rename with no side effects

- [x] **CB-3.3:** Optimize DataFrame construction in utils.py (line 159-161)
  - Use scalar broadcast instead of `np.zeros_like()`
  - **COMPLETED:** Replaced `np.zeros_like(time, dtype=int)` with scalar `0` and explicit `dtype=int`
  - More efficient (no intermediate array creation) and clearer intent

### Figure Utilities Enhancements

- [x] **FU-3.1:** Add context-aware transparency default in `save_figure()`
  - Auto-detect dark backgrounds
  - Set `transparent=False` for dark, `True` for light
  - **COMPLETED:** Uses ITU-R BT.601 luminance formula with 0.5 threshold

- [x] **FU-3.2:** Implement font size validation in `validate_figure_quality()`
  - Check axis labels meet minimum size
  - Check tick labels meet minimum size
  - **COMPLETED:** Context-aware validation (DPI-based), helpful warnings with fix suggestions

- [x] **FU-3.3:** Add source URLs to color palette comments (lines 125, 145)
  - Verify WONG against Nature Methods article
  - Verify OKABE_ITO against JFly source
  - **COMPLETED:** Added DOI and URLs with verification notes

---

## Milestone 4: Testing Infrastructure (HIGH PRIORITY) 🧪

**Priority:** THIS WEEK
**Estimated Time:** 4-6 hours
**Goal:** Establish comprehensive test coverage

- [x] **TEST-4.1:** Create `tests/test_figure_utilities.py`
  - Test constants (MM_TO_INCHES, GOLDEN_RATIO, column widths)
  - Test `cm_to_inch()` conversion
  - Test `figure_size_from_panels()` calculations
  - Test color palette length and hex validity
  - Test `label_panels()` with valid and invalid inputs
  - **COMPLETED:** Comprehensive test suite with 54 tests covering all functionality
  - Tests include: constants, color palettes, conversions, sizing, context switching,
    save_figure (transparency, formats, validation), validate_figure_quality (DPI,
    layout, font sizes), and label_panels (error handling)

- [x] **TEST-4.2:** Add type checking validation
  - Run `mypy src/continuum_swr_replay/figure_utilities.py`
  - Ensure all type errors resolved
  - **COMPLETED:** Mypy passes with no issues (verified throughout development)

- [x] **TEST-4.3:** Add linting validation
  - Run `ruff check src/continuum_swr_replay/figure_utilities.py`
  - Ensure all linting errors resolved
  - **COMPLETED:** Ruff passes with all checks passed (verified throughout development)

- [ ] **TEST-4.4:** Enable doctest for examples
  - Add `--doctest-modules` to pytest config
  - Or add `if __name__ == "__main__"` with `doctest.testmod()`

---

## Milestone 5: Type Safety & Enums (MEDIUM PRIORITY) 📝

**Priority:** WEEK 2
**Estimated Time:** 2-3 hours
**Goal:** Improve type safety and IDE support

- [x] **CB-5.1:** Convert `TRAINING_TYPES` to Enum in parameters.py
  - Create `TrainingType(str, Enum)` class
  - Update function signatures to use enum
  - **COMPLETED:** Type-safe enum with backward compatibility for strings
  - Created TrainingType enum with ALL and NO_RIPPLE values
  - Updated get_training_timepoints() to accept both enum and string
  - All 20 integration tests pass ✅

- [x] **CB-5.2:** Create shared `types.py` module
  - Move `SessionData` from analysis/hippocampus/types.py
  - Create central location for all TypedDicts
  - Update imports in bandit_task.py
  - **COMPLETED:** Created src/continuum_swr_replay/types.py with SessionData TypedDict
  - Removed old analysis/hippocampus/types.py file
  - Updated imports in bandit_task.py, data_aggregation.py, posterior_analysis.py
  - Fixed **init**.py imports to use new module structure
  - All 20 integration tests + 43 unit tests pass ✅

- [x] **CB-5.3:** Add TypedDict for detector params in run_decoding_bandit.py
  - Create `ClusterlessParams` TypedDict
  - Create `SortedSpikesParams` TypedDict
  - Update return types
  - **COMPLETED:** Added ClusterlessParams and SortedSpikesParams to types.py
  - Updated to_clusterless_params() and to_sorted_spikes_params() return types
  - Updated organize_results() and generate_figurl() to use SessionData
  - Updated get_training_timepoints() to accept SessionData
  - Fixed import from deleted analysis_hpc_bandit module
  - All 20 integration tests pass ✅
  - Mypy and ruff clean ✅

- [x] **CB-5.4:** Add missing type hints to helper functions
  - Review all functions for complete type annotations
  - Add return types to `_count_exposure()` and similar
  - **COMPLETED:** Added comprehensive type hints to src/continuum_swr_replay/analysis/pfc/glm.py
  - Added type hints to 16 functions (get_spikecount_per_time_bin, reindex_by, test_theta_modulation_v1,
    get_labels,_lr_test, test_theta_modulation, create_lagged_signal, create_conv_basis,
    _make_conv_spline_basis,_make_design_matrices, _fit_immobility_models, get_frac_null_deviance,
    get_frac_explained_deviance, process_neuron, run_session)
  - Fixed duplicate function name: test_theta_modulation → test_theta_modulation_v1 (original), test_theta_modulation (new)
  - Fixed all mypy type errors (19 → 0) ✅
  - Fixed ruff linting issues (import sorting) ✅
  - Replaced mutable default arguments with None pattern
  - Added strict=True to zip() call
  - Fixed min() key parameter type issue with lambda
  - Added explicit dict type annotations for neuron_stats

- [x] **CB-5.5:** Add `__all__` to public modules
  - Define public API for each module
  - Improves IDE autocomplete
  - **COMPLETED:** Added `__all__` to types.py (3 exports), figure_utilities.py (37+ exports), parameters.py (11 exports)
  - Verified imports work correctly with star imports
  - Mypy and ruff checks pass ✅

---

## Milestone 6: Documentation & Examples (MEDIUM PRIORITY) 📚

**Priority:** WEEK 2-3
**Estimated Time:** 2-3 hours
**Goal:** Improve developer experience and learning curve

- [ ] **FU-6.1:** Add `quick_figure()` helper function
  - Single-panel figure with sensible defaults
  - Reduces boilerplate for common case

- [ ] **FU-6.2:** Add `figure_context()` context manager
  - Temporary style changes with automatic cleanup
  - Prevents style leakage between figures

- [ ] **FU-6.3:** Return saved file paths from `save_figure()`
  - Change return type to `list[Path]`
  - Helps users verify what was saved

---

## Milestone 7: Performance Optimizations (LOW PRIORITY) ⚡

**Priority:** WEEK 3
**Estimated Time:** 2-3 hours
**Goal:** Improve execution speed for common operations

- [x] **CB-7.1:** Vectorize event labeling with IntervalIndex in utils.py
  - Replace explicit loop with binary search (np.searchsorted)
  - Performance gain for large datasets (~36x speedup verified)
  - Comprehensive test suite added (8 tests in tests/unit/test_utils.py)
  - **COMPLETED:** 2025-11-14

- [x] **CB-7.2:** Add `@cache` to `get_epoch_info()` in utils.py
  - Expensive database query should be cached
  - Use `functools.cache` (Python 3.9+)
  - Enhanced docstring documenting cache behavior and limitations
  - Validated through integration tests (all 223 pass)
  - **COMPLETED:** 2025-11-14

- [x] **CB-7.3:** Centralize script constants in parameters.py
  - Move constants from run_decoding_bandit.py
  - Eliminate duplication
  - Added PLACE_BIN_SIZE_1D, PLACE_BIN_SIZE_2D, POSITION_STD, WAVEFORM_STD, BLOCK_SIZE, DETECTOR_CONFIG
  - Updated DecodingParameters to use module-level constants (values corrected to match script)
  - Updated run_decoding_bandit.py to import from parameters.py
  - All 223 tests pass ✅
  - **COMPLETED:** 2025-11-14

---

## Milestone 8: Code Quality & Cleanup (LOW PRIORITY) 🧹

**Priority:** WEEK 3-4
**Estimated Time:** 3-4 hours
**Goal:** Polish and maintainability

- [x] **CB-8.1:** Remove obvious comments
  - Review and improve comment quality
  - Explain "why" not "what"
  - **COMPLETED:** Removed ~30 obvious comments from spikes.py, figure_utilities.py, utils.py
  - Comments now explain rationale, not implementation details

- [x] **CB-8.2:** Use `logging.config.dictConfig` in run_decoding_bandit.py
  - Replace imperative logging setup
  - Cleaner, more maintainable config
  - **COMPLETED:** Refactored setup_logger() to use declarative dictConfig
  - Cleaner, more maintainable configuration structure
  - Same functionality: file + console handlers with SkipChecksumFilter

- [x] **CB-8.3:** Add validation to DecodingParameters dataclass
  - Validate `block_size` is power of 2
  - Validate positive values for stds
  - **COMPLETED:** Added `__post_init__` validation to DecodingParameters
  - Validates block_size is power of 2 using bitwise check: `n & (n-1) == 0`
  - Validates position_std, waveform_std, place_bin_size are all positive
  - Clear error messages with actual values for debugging
  - Updated docstrings with "Must be positive" constraints
  - Added Raises section documenting ValueError conditions
  - Comprehensive test suite: 11 tests in tests/unit/test_parameters.py
  - All tests pass, mypy and ruff clean ✅

- [x] **CB-8.4:** Use more descriptive variable names
  - Replace `frac` with `coincident_fraction`
  - Improve readability throughout
  - **COMPLETED:** Renamed 8 unclear variables to improve code readability
  - spikes.py (detect_coincident_spikes):
    - `frac` → `coincident_fraction` (more descriptive of what it measures)
    - Filtered `frac` → `artifact_labels` (clearer that these are artifact labels)
    - `df` → `spike_events` (generic DataFrame → descriptive purpose)
    - Improved formatting with method chaining on separate lines
  - spikes.py (get_electrode_group_info):
    - `cur` → `group_info` (clearer that it's electrode group information)
    - `nwbf` → `nwb_file` (clearer that it's a file object, not filename)
  - utils.py (get_epoch_info):
    - `nwbf` → `nwb_file_name` (clearer that it's a filename string, not file object)
  - events.py (get_event_time):
    - `start_ind` → `start_indices` (abbreviated → PEP 8 compliant)
    - `end_ind` → `end_indices` (abbreviated → PEP 8 compliant)
  - bandit_task.py (load_data):
    - `pos_name` → `position_interval_name` (matches database column name)
  - All 42 tests pass (test_spikes.py, test_utils.py), mypy and ruff clean ✅

- [x] **CB-8.5:** Ensure consistent `pathlib.Path` usage ✅
  - Reviewed all file operations across src/ and scripts/
  - Found codebase already uses pathlib.Path consistently
  - Fixed 3 instances of redundant `Path()` wrapping of directory constants:
    - `Path(PROCESSED_DATA_DIR)` → `PROCESSED_DATA_DIR` (already a Path)
    - `Path(FIGURE_DIR)` → `FIGURE_DIR` (already a Path)
  - Removed unused `from pathlib import Path` imports in scripts
  - No `os.path` usage found (except `os.environ` for environment variables, which is correct)
  - All file operations use Path `/` operator and Path methods (.exists(), .mkdir(), .glob())
  - All 54 figure utility tests pass, mypy and ruff clean ✅

---

## Milestone 9: Architecture Improvements (LONG-TERM) 🏗️

**Priority:** FUTURE (1-2 weeks)
**Estimated Time:** 5-10 hours
**Goal:** Long-term maintainability and extensibility

### Type System Reorganization

- [x] **ARCH-9.1:** Create comprehensive `types.py` module ✅
  - Consolidated 7 TypedDicts from data_loaders into types.py:
    - `SpikeDataDict`, `DIOEventData`, `PositionInfoDict`, `WellNodeData`
    - `ThetaData`, `RippleData`, `MultiunitData`
  - All TypedDicts now imported from single source (`continuum_swr_replay.types`)
  - Updated imports in 5 data_loaders modules (spikes, events, position, trials, lfp)
  - Added NDArray import for proper type annotations
  - No Protocols currently in codebase
  - All 25 data loader tests pass, mypy and ruff clean ✅

- [x] **ARCH-9.2:** Define string keys as constants ✅
  - Created `SessionDataKeys` class with 24 string constants for SessionData dictionary keys
  - Organized constants in same categories as SessionData TypedDict
  - Added comprehensive docstring with usage examples
  - Added to __all__ exports for public API
  - Enables IDE autocomplete when accessing SessionData dictionaries
  - Prevents typos with compile-time constant references
  - All 37 spike/data loading tests pass, mypy and ruff clean ✅

### Constants Organization

- [ ] **ARCH-9.3:** Organize constants into submodules
  - `constants/files.py` for NWB_FILES
  - `constants/spatial.py` for spatial mappings
  - `constants/decoding.py` for decoding params

### Development Tools

- [ ] **ARCH-9.5:** Add pre-commit hooks
  - Create `.pre-commit-config.yaml`
  - Auto-run ruff, mypy, tests
  - Prevent quality regressions

---

## Milestone 10: Advanced Testing (FUTURE) 🔬

**Priority:** FUTURE
**Estimated Time:** 4-6 hours
**Goal:** Comprehensive test coverage with property-based testing

- [ ] **TEST-10.1:** Expand property-based tests with Hypothesis
  - Test spike filtering preserves order
  - Test event labels don't overlap
  - Test mathematical invariants

- [ ] **TEST-10.2:** Add integration tests for data loading
  - Test complete pipeline on sample data
  - Verify data integrity end-to-end

---

## Progress Tracking

### Overall Progress

**Milestones Completed:** 0 / 10
**Tasks Completed:** 0 / 61

### Priority Breakdown

| Priority | Milestones | Tasks | Status |
|----------|------------|-------|--------|
| 🔴 BLOCKING | M1 | 8 | Not Started |
| 🟡 HIGH | M2, M4 | 11 | Not Started |
| 🟢 MEDIUM | M3, M5, M6, M7 | 20 | Not Started |
| 🧹 LOW | M8 | 5 | Not Started |
| 🏗️ FUTURE | M9, M10 | 17 | Not Started |

### Weekly Goals

**Week 1 Target:** Complete M1 + M2 (19 tasks)
**Week 2 Target:** Complete M3 + M4 + M5 (15 tasks)
**Week 3 Target:** Complete M6 + M7 + M8 (12 tasks)
**Week 4+:** M9 + M10 (15 tasks)

---

## Quick Reference

### High-Impact Quick Wins (< 30 min each)

- FU-1.2: Remove unused imports
- FU-1.3: Add stacklevel to warnings
- FU-1.4: Cast DPI to int
- CB-1.1: Fix zip strict parameter
- CB-3.1: Use | operator for dicts
- CB-3.2: Fix constant naming

### Must Complete Before Release

- All M1 tasks (Critical Fixes)
- All M2 tasks (UX Fixes)
- TEST-4.1, TEST-4.2, TEST-4.3 (Core Testing)

### Nice to Have for v1.0

- M3, M5, M6 (Quality & Documentation)
- TEST-4.4 (Doctest)

### Future Enhancements

- M9, M10 (Architecture & Advanced Testing)

---

## Notes

- **Dependencies:** Some tasks depend on others (e.g., can't write tests until code works)
- **Parallel Work:** Multiple milestones can be worked on simultaneously if different files
- **Review Points:** After M1 and M2, run full test suite and linting
- **Documentation:** Update IMPROVEMENTS.md and FIGURE_UTILITIES_REVIEW.md as tasks complete

---

**Last Status Check:** 2025-11-14
**Next Review:** After M1 completion
