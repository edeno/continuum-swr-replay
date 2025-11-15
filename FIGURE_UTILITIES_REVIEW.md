# Figure Utilities Review Summary

**File:** `src/continuum_swr_replay/figure_utilities.py`
**Date:** 2025-11-14
**Reviewers:** scientific-code-reviewer + scientific-ux-reviewer

---

## Executive Summary

The enhanced `figure_utilities.py` (66 → 726 lines) provides excellent foundations for publication-quality scientific figures, but requires **critical fixes before release**.

**Code Quality:** 7.5/10 - Well-structured but has type safety and testing gaps
**UX Quality:** NEEDS_POLISH - Good API design with sharp edges in common workflows

**Overall Status:** ⚠️ **REQUEST_CHANGES** - Must fix 9 critical issues (4 code + 5 UX)

**Estimated effort to production-ready:** 4-6 hours focused work

---

## 🔴 Critical Issues (MUST FIX)

### Code Review Criticals

#### 1. **Type Safety Violation in `save_figure()`** [Line 447]

**Issue:** `fig.get_facecolor()` returns tuple/array, but parameter typed as `str | None`

```python
# Current (WRONG):
def save_figure(figure_name: str, *, facecolor: str | None = None, ...):
    if facecolor is None:
        fig = plt.gcf()
        facecolor = fig.get_facecolor()  # ❌ Type error

# Fix:
def save_figure(
    figure_name: str,
    *,
    facecolor: str | tuple[float, ...] | None = None,  # ✅ Accept matplotlib color types
    ...
):
    ...
```

#### 2. **Unused Imports** [Lines 77, 87]

```python
import os  # ❌ UNUSED - remove
from matplotlib.figure import Figure  # ❌ UNUSED - remove
```

#### 3. **Missing `stacklevel` in `warnings.warn()`** [Lines 701, 706, 722]

```python
# Current:
warnings.warn(f"Low DPI ({fig.dpi})...")  # ❌ Wrong stack level

# Fix:
warnings.warn(f"Low DPI ({fig.dpi})...", stacklevel=2)  # ✅
```

#### 4. **No Test Coverage**

Definition of Done requires tests. Need `tests/test_figure_utilities.py` with:

- Constant validation (MM_TO_INCHES, GOLDEN_RATIO, column widths)
- Function tests (cm_to_inch, figure_size_from_panels, label_panels)
- Color palette integrity (no duplicates, valid hex)
- Example validation (docstrings run without error)

### UX Review Criticals

#### 5. **`save_figure()` doesn't use `FIGURE_DIR` by default** [Line 384-465]

**Problem:** Users must manually prepend `FIGURE_DIR` to every path

```python
# Current workflow (friction):
save_figure(f"{FIGURE_DIR}/figure_1")  # User must remember

# Expected workflow:
save_figure("figure_1")  # Should auto-save to FIGURE_DIR

# Fix:
def save_figure(
    figure_name: str,
    *,
    base_dir: str | Path | None = None,  # None = use FIGURE_DIR
    ...
):
    if base_dir is None:
        base_dir = FIGURE_DIR
    output_path = Path(base_dir) / figure_name
```

#### 6. **No validation that figure exists before saving** [Line 384-465]

**Problem:** Silently creates blank PDF if no figure created

```python
# Add to save_figure():
fig = plt.gcf()

if not fig.get_axes():
    raise ValueError(
        f"Cannot save figure '{figure_name}': no axes found.\n"
        f"HINT: Did you forget to call plt.subplots() or create a plot?"
    )
```

#### 7. **`validate_figure_quality()` modifies figure state** [Line 716]

**Problem:** Validation function should be read-only, but calls `plt.tight_layout()`

```python
# Remove this line entirely:
plt.tight_layout()  # ❌ Modifies figure during "validation"

# Just check settings, don't modify:
if not fig.get_constrained_layout():
    warnings.warn(...)
```

#### 8. **`label_panels()` has unclear error on length mismatch** [Line 656]

```python
# Add explicit validation before zip:
if labels is None:
    labels = list(string.ascii_lowercase[:len(axes)])

if len(labels) != len(axes):
    raise ValueError(
        f"Label/axes count mismatch: {len(labels)} labels for {len(axes)} axes.\n"
        f"HINT: Provided labels: {labels}\n"
        f"HINT: Number of axes: {len(axes)}"
    )

for ax, label in zip(axes, labels):  # No strict=True needed now
    ...
```

#### 9. **Context switching doesn't reset matplotlib state** [Line 187]

**Problem:** Switching contexts may leave stale settings

```python
def set_figure_defaults(context: Literal[...]) -> None:
    # RESET to clean state first
    mpl.rcParams.update(mpl.rcParamsDefault)

    if context == "paper":
        ...
```

---

## 🟡 High Priority Issues (SHOULD FIX)

### 10. **Integer type cast missing for DPI** [Line 441]

```python
if dpi is None:
    dpi = int(mpl.rcParams.get("savefig.dpi", 300))  # ✅ Cast to int
```

### 11. **Color palette scientific accuracy not verified** [Lines 125-162]

Verify WONG and OKABE_ITO values match published sources:

- Wong: Nature Methods 8(6) 2011
- Okabe-Ito: JFly Data Depository

Add source URLs in comments:

```python
WONG: list[str] = [
    # Source: https://www.nature.com/articles/nmeth.1618
    # Verified against Bang Wong (2011) Nature Methods 8(6):441
    "#000000",  # Black
    ...
]
```

### 12. **`validate_figure_quality()` doesn't check font sizes**

Docstring claims to validate fonts, but doesn't. Implement:

```python
def validate_figure_quality() -> bool:
    fig = plt.gcf()
    passed = True

    # Check font sizes for paper context
    min_fontsize = PAPER_FONT_MIN if fig.dpi > 300 else PRES_BODY / 2

    for ax in fig.get_axes():
        if ax.xaxis.label.get_fontsize() < min_fontsize:
            warnings.warn(f"X-axis label too small", stacklevel=2)
            passed = False

    return passed
```

### 13. **`transparent=True` default breaks dark backgrounds**

Context-aware transparency:

```python
def save_figure(..., transparent: bool | None = None):
    if transparent is None:
        # Dark backgrounds should not be transparent
        current_facecolor = fig.get_facecolor()
        is_dark = mpl.colors.to_rgba(current_facecolor)[0] < 0.5
        transparent = not is_dark
```

### 14. **Add format validation**

```python
VALID_FORMATS = ["pdf", "png", "svg", "eps", "jpg", "jpeg", "tif", "tiff"]

def save_figure(...):
    if formats is None:
        formats = ["pdf", "png"]

    invalid = [fmt for fmt in formats if fmt.lower() not in VALID_FORMATS]
    if invalid:
        raise ValueError(
            f"Invalid format(s): {invalid}. Must be one of: {VALID_FORMATS}"
        )
```

---

## 🟢 Nice to Have (CONSIDER)

### 15. **Add complete workflow example to module docstring**

```python
"""
Examples
--------
Complete paper figure workflow:

>>> from continuum_swr_replay.figure_utilities import *
>>> import matplotlib.pyplot as plt
>>>
>>> # Configure for paper
>>> set_figure_defaults("paper")
>>>
>>> # Create multi-panel figure
>>> figsize = figure_size_from_panels(nrows=2, ncols=2, panel_width_cm=4.0)
>>> fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
>>>
>>> # Plot with colorblind-safe colors
>>> for ax, color in zip(axes.flat, WONG[1:]):
...     ax.plot(data, color=color, linewidth=1.5)
>>>
>>> # Label and save
>>> label_panels(axes.flatten())
>>> save_figure("figure_1")  # Auto-saves to FIGURE_DIR
"""
```

### 16. **Add `quick_figure()` helper**

Reduce boilerplate for single panels:

```python
def quick_figure(
    context: Literal["paper", "presentation", "presentation_dark"] = "paper",
    width: float | None = None,
    aspect_ratio: float = GOLDEN_RATIO,
    **subplot_kwargs
) -> tuple[Figure, Axes]:
    """Create single-panel figure with sensible defaults."""
    set_figure_defaults(context)

    if width is None:
        width = ONE_COLUMN if context == "paper" else SLIDE_WIDTH

    height = width * aspect_ratio

    return plt.subplots(
        figsize=(width, height),
        constrained_layout=True,
        **subplot_kwargs
    )
```

### 17. **Add context manager for style isolation**

```python
@contextmanager
def figure_context(context: Literal[...]):
    """Temporarily set figure defaults."""
    old_params = mpl.rcParams.copy()
    try:
        set_figure_defaults(context)
        yield
    finally:
        mpl.rcParams.update(old_params)

# Usage:
with figure_context("paper"):
    fig, ax = plt.subplots(...)
    save_figure("paper_fig")
# Settings auto-restored
```

### 18. **Return saved file paths**

```python
def save_figure(...) -> list[Path]:
    """
    Returns
    -------
    list of Path
        Paths to saved files
    """
    saved_files = []

    for fmt in formats:
        output_file = Path(f"{figure_name}.{fmt}")
        plt.savefig(...)
        saved_files.append(output_file.absolute())

    return saved_files
```

---

## ✅ Excellent Aspects (Keep These!)

1. **Comprehensive documentation** - NumPy-style docstrings with examples
2. **Colorblind-safe palettes** - WONG and OKABE_ITO with citations
3. **Three contexts** - paper/presentation/presentation_dark covers 95% of use cases
4. **Typography constants** - Clear standards for Nature/Science requirements
5. **Multi-panel support** - `figure_size_from_panels()` and `label_panels()` are brilliant
6. **Dimension constants** - ONE_COLUMN, TWO_COLUMN, GOLDEN_RATIO well-documented
7. **Dark mode support** - Thoughtful presentation_dark context
8. **Keyword-only args** - Forces named parameters to prevent mistakes

---

## 📊 Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Lines of Code | 726 | - | ✅ |
| Documentation Coverage | 100% | 100% | ✅ |
| Type Hint Coverage | ~95% | 100% | ⚠️ |
| Test Coverage | 0% | >80% | ❌ |
| Linting Errors | 5 | 0 | ⚠️ |
| Type Errors | 1 | 0 | ❌ |
| Quality Score | 7.5/10 | 8.5/10 | ⚠️ |

---

## 🎯 Priority-Ordered Fix List

### Immediate (1-2 hours)

1. ✅ Fix `facecolor` type hint in `save_figure()`
2. ✅ Remove unused imports (`os`, `Figure`)
3. ✅ Add `stacklevel=2` to all warnings
4. ✅ Cast DPI to int
5. ✅ Add format validation

### Same Day (2-3 hours)

6. ✅ Integrate with `FIGURE_DIR` by default
7. ✅ Validate figure exists before saving
8. ✅ Fix `validate_figure_quality()` to be read-only
9. ✅ Improve `label_panels()` error messages
10. ✅ Reset matplotlib state in `set_figure_defaults()`

### This Week (3-4 hours)

11. 🧪 Create comprehensive test suite
12. 📚 Verify color palette values against sources
13. 🔧 Implement font size validation
14. 🎨 Add context-aware transparency
15. 📖 Add complete workflow example

### Nice to Have

16. 🚀 Add `quick_figure()` helper
17. 🔒 Add `figure_context()` context manager
18. 📁 Return saved file paths

---

## 🧪 Required Tests

Create `tests/test_figure_utilities.py`:

```python
import pytest
import numpy as np
from continuum_swr_replay.figure_utilities import *

class TestConstants:
    def test_mm_to_inches(self):
        assert np.isclose(MM_TO_INCHES, 1.0 / 25.4)

    def test_column_widths(self):
        assert np.isclose(ONE_COLUMN, 89.0 / 25.4)
        assert np.isclose(TWO_COLUMN, 178.0 / 25.4)

    def test_golden_ratio(self):
        assert np.isclose(GOLDEN_RATIO, (np.sqrt(5) - 1) / 2)

class TestColorPalettes:
    def test_wong_length(self):
        assert len(WONG) == 8

    def test_wong_valid_hex(self):
        for color in WONG:
            assert color.startswith("#")
            assert len(color) == 7

    def test_okabe_ito_length(self):
        assert len(OKABE_ITO) == 8

class TestFunctions:
    def test_cm_to_inch(self):
        assert np.isclose(cm_to_inch(2.54), 1.0)
        assert np.isclose(cm_to_inch(10), 3.937, rtol=1e-3)

    def test_figure_size_from_panels(self):
        width, height = figure_size_from_panels(
            nrows=2, ncols=2,
            panel_width_cm=4.5,
            panel_height_cm=4.5
        )
        assert width > 0
        assert height > 0

    def test_label_panels_length_mismatch(self):
        fig, axes = plt.subplots(2, 2)
        with pytest.raises(ValueError, match="mismatch"):
            label_panels(axes.flatten(), labels=['a', 'b'])  # 4 axes, 2 labels
```

---

## 📝 Next Steps

### Today

1. Fix all 9 critical issues (items 1-9 above)
2. Run `ruff check` and `mypy` to verify no errors
3. Create basic test file

### This Week

4. Implement font size validation
5. Add color palette verification
6. Write comprehensive tests
7. Add workflow example

### Future

8. Consider neuroscience-specific helpers (spatial plots, raster styling)
9. Integrate with existing `parameters.py` colors
10. Add pre-commit hooks to prevent regressions

---

## 🏆 Overall Assessment

**This is high-quality work** with excellent design decisions. The module provides real value for creating publication-quality neuroscience figures. The issues found are **fixable in a few hours** and don't require architectural changes.

**Strengths:**

- Well-organized, comprehensive documentation
- Thoughtful API design (contexts, keyword args, defaults)
- Scientific rigor (colorblind palettes, typography standards)
- Excellent multi-panel support

**Weaknesses:**

- Type safety gaps (easy fix)
- Missing tests (required before release)
- UX friction in common workflows (path handling, validation)

**Recommendation:** Implement the 9 critical fixes, add tests, then this is **production-ready code** that researchers will appreciate using.

---

**Last updated:** 2025-11-14
**Status:** ⚠️ NEEDS_FIXES - See priority list above
