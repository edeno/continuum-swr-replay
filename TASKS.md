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

- [ ] **FU-1.4:** Cast DPI to int in `save_figure()` (line 441)
  - `dpi = int(mpl.rcParams.get("savefig.dpi", 300))`

- [ ] **FU-1.5:** Add format validation in `save_figure()`
  - Define `VALID_FORMATS` constant
  - Validate before saving, raise `ValueError` for invalid formats

### Codebase Critical Fixes

- [ ] **CB-1.1:** Fix unsafe `zip()` without `strict=True` in utils.py (lines 54-63, 68-69)
  - Change `strict=False` to `strict=True`

- [ ] **CB-1.2:** Refactor filename parsing with regex in utils.py (lines 56-58)
  - Create `parse_nwb_filename()` function
  - Eliminate double `.replace()` call

- [ ] **CB-1.3:** Remove redundant `enumerate()` with `itertuples()` in utils.py (line 118)
  - Use `row.Index` instead of enumerate counter

---

## Milestone 2: UX Critical Fixes (HIGH PRIORITY) 🟡

**Priority:** SAME DAY
**Estimated Time:** 2-3 hours
**Goal:** Remove workflow friction and prevent user confusion

- [ ] **FU-2.1:** Make `save_figure()` use `FIGURE_DIR` by default
  - Add `base_dir: str | Path | None = None` parameter
  - Default to `FIGURE_DIR` when None
  - Create output path as `Path(base_dir) / figure_name`

- [ ] **FU-2.2:** Validate figure exists before saving in `save_figure()`
  - Check `fig.get_axes()` is not empty
  - Raise `ValueError` with helpful hint if no axes found

- [ ] **FU-2.3:** Fix `validate_figure_quality()` modifying state (line 716)
  - Remove `plt.tight_layout()` call
  - Make validation read-only

- [ ] **FU-2.4:** Improve error messages in `label_panels()` (line 656)
  - Add explicit length validation before zip
  - Provide helpful error with label count and axes count

- [ ] **FU-2.5:** Reset matplotlib state in `set_figure_defaults()` (line 236+)
  - Call `mpl.rcParams.update(mpl.rcParamsDefault)` first
  - Prevents stale settings from previous context

---

## Milestone 3: Style & Quality (MEDIUM PRIORITY) 🟢

**Priority:** THIS WEEK
**Estimated Time:** 3-4 hours
**Goal:** Improve code consistency and Python idioms

### Codebase Improvements

- [ ] **CB-3.1:** Use `|` operator for dict merging in bandit_task.py (line 181-192)
  - Replace `{**dict1, **dict2}` with `dict1 | dict2`

- [ ] **CB-3.2:** Fix SCREAMING_CASE for constants in parameters.py (line 67)
  - Rename `training_types` to `TRAINING_TYPES`

- [ ] **CB-3.3:** Optimize DataFrame construction in utils.py (line 114-116)
  - Use scalar broadcast instead of `np.zeros_like()`

### Figure Utilities Enhancements

- [ ] **FU-3.1:** Add context-aware transparency default in `save_figure()`
  - Auto-detect dark backgrounds
  - Set `transparent=False` for dark, `True` for light

- [ ] **FU-3.2:** Implement font size validation in `validate_figure_quality()`
  - Check axis labels meet minimum size
  - Check tick labels meet minimum size

- [ ] **FU-3.3:** Add source URLs to color palette comments (lines 125, 145)
  - Verify WONG against Nature Methods article
  - Verify OKABE_ITO against JFly source

---

## Milestone 4: Testing Infrastructure (HIGH PRIORITY) 🧪

**Priority:** THIS WEEK
**Estimated Time:** 4-6 hours
**Goal:** Establish comprehensive test coverage

- [ ] **TEST-4.1:** Create `tests/test_figure_utilities.py`
  - Test constants (MM_TO_INCHES, GOLDEN_RATIO, column widths)
  - Test `cm_to_inch()` conversion
  - Test `figure_size_from_panels()` calculations
  - Test color palette length and hex validity
  - Test `label_panels()` with valid and invalid inputs

- [ ] **TEST-4.2:** Add type checking validation
  - Run `mypy src/continuum_swr_replay/figure_utilities.py`
  - Ensure all type errors resolved

- [ ] **TEST-4.3:** Add linting validation
  - Run `ruff check src/continuum_swr_replay/figure_utilities.py`
  - Ensure all linting errors resolved

- [ ] **TEST-4.4:** Enable doctest for examples
  - Add `--doctest-modules` to pytest config
  - Or add `if __name__ == "__main__"` with `doctest.testmod()`

---

## Milestone 5: Type Safety & Enums (MEDIUM PRIORITY) 📝

**Priority:** WEEK 2
**Estimated Time:** 2-3 hours
**Goal:** Improve type safety and IDE support

- [ ] **CB-5.1:** Convert `training_types` to Enum in parameters.py
  - Create `TrainingType(str, Enum)` class
  - Update function signatures to use enum

- [ ] **CB-5.2:** Create shared `types.py` module
  - Move `SessionData` from analysis/hippocampus/types.py
  - Create central location for all TypedDicts
  - Update imports in bandit_task.py

- [ ] **CB-5.3:** Add TypedDict for detector params in run_decoding_bandit.py
  - Create `ClusterlessParams` TypedDict
  - Create `SortedSpikesParams` TypedDict
  - Update return types

- [ ] **CB-5.4:** Add missing type hints to helper functions
  - Review all functions for complete type annotations
  - Add return types to `_count_exposure()` and similar

- [ ] **CB-5.5:** Add `__all__` to public modules
  - Define public API for each module
  - Improves IDE autocomplete

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

- [ ] **CB-7.1:** Vectorize event labeling with IntervalIndex in utils.py
  - Replace explicit loop with pandas IntervalIndex
  - Performance gain for large datasets

- [ ] **CB-7.2:** Add `@cache` to `get_epoch_info()` in utils.py
  - Expensive database query should be cached
  - Use `functools.cache` (Python 3.9+)

- [ ] **CB-7.3:** Centralize script constants in parameters.py
  - Move constants from run_decoding_bandit.py
  - Eliminate duplication

---

## Milestone 8: Code Quality & Cleanup (LOW PRIORITY) 🧹

**Priority:** WEEK 3-4
**Estimated Time:** 3-4 hours
**Goal:** Polish and maintainability

- [ ] **CB-8.1:** Remove obvious comments
  - Review and improve comment quality
  - Explain "why" not "what"

- [ ] **CB-8.2:** Use `logging.config.dictConfig` in run_decoding_bandit.py
  - Replace imperative logging setup
  - Cleaner, more maintainable config

- [ ] **CB-8.3:** Add validation to DecodingParameters dataclass
  - Validate `block_size` is power of 2
  - Validate positive values for stds

- [ ] **CB-8.4:** Use more descriptive variable names
  - Replace `frac` with `coincident_fraction`
  - Improve readability throughout

- [ ] **CB-8.5:** Ensure consistent `pathlib.Path` usage
  - Review all file operations
  - Use Path methods instead of string concatenation

---

## Milestone 9: Architecture Improvements (LONG-TERM) 🏗️

**Priority:** FUTURE (1-2 weeks)
**Estimated Time:** 5-10 hours
**Goal:** Long-term maintainability and extensibility

### Type System Reorganization

- [ ] **ARCH-9.1:** Create comprehensive `types.py` module
  - All TypedDicts in one place
  - All Protocols defined
  - Import from single source

- [ ] **ARCH-9.2:** Define string keys as constants
  - Create `SessionDataKeys` class
  - Prevent typos with autocomplete

### Constants Organization

- [ ] **ARCH-9.3:** Organize constants into submodules
  - `constants/files.py` for NWB_FILES
  - `constants/spatial.py` for spatial mappings
  - `constants/decoding.py` for decoding params

### Configuration Management

- [ ] **ARCH-9.4:** Consider Pydantic for configuration
  - Type-safe settings with validation
  - Environment variable support
  - Better error messages

### Development Tools

- [ ] **ARCH-9.5:** Add pre-commit hooks
  - Create `.pre-commit-config.yaml`
  - Auto-run ruff, mypy, tests
  - Prevent quality regressions

- [ ] **ARCH-9.6:** Add CLI framework (click/typer)
  - Better command-line interface for scripts
  - Auto-generated help text
  - Type-safe arguments

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

- [ ] **TEST-10.3:** Add neuroscience-specific test fixtures
  - Sample spike times
  - Sample position data
  - Sample task events

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
