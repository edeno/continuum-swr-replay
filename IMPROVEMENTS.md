# Code Improvements Checklist

## Raymond Hettinger-Style Code Review Findings

This document lists all identified improvements for the continuum-swr-replay codebase, organized by priority and impact.

---

## 🔴 Critical Issues (High Priority - Fix First)

### C1. Fix Unsafe `zip()` Calls Without `strict=True`

**File:** `src/continuum_swr_replay/data_loaders/utils.py:54-63, 68-69`
**Issue:** Using `strict=False` explicitly disables length mismatch detection
**Impact:** Silent data corruption if file lists are inconsistent
**Fix:**

```python
# Change from:
animal, date = zip(*[...], strict=False)
# To:
animal, date = zip(*[...], strict=True)
```

### C2. Inefficient String Manipulation with Double `.replace()`

**File:** `src/continuum_swr_replay/data_loaders/utils.py:56-58`
**Issue:** Same operation `file_name.replace("_.nwb", "")` called twice
**Impact:** Performance waste, error-prone
**Fix:**

```python
# Use regex-based parser
import re

def parse_nwb_filename(file_name: str) -> tuple[str, int]:
    """Extract animal name and date from NWB filename.

    Example: 'chimi20200212_.nwb' -> ('chimi', 20200212)
    """
    match = re.match(r"^(\w+?)(\d{8})_.nwb$", file_name)
    if not match:
        raise ValueError(f"Invalid NWB filename format: {file_name}")
    animal, date_str = match.groups()
    return animal, int(date_str)

animals_dates = [parse_nwb_filename(f) for f in file_names]
animal, date = zip(*animals_dates, strict=True)
```

---

## 🟡 Performance & Style Issues (Medium Priority)

### M1. Use Modern Dict Merging with `|` Operator

**File:** `src/continuum_swr_replay/data_loaders/bandit_task.py:181-192`
**Issue:** Using `**` unpacking instead of Python 3.9+ `|` operator
**Impact:** Readability and slight performance
**Fix:**

```python
# Change from:
return {
    **position_data,
    **spike_data,
    # ...
}

# To:
return (
    position_data
    | spike_data
    | ripple_data
    | get_theta(nwb_file_name, epoch_name)
    | get_multiunit(spike_data["spike_times"], position_data["position_info"])
    | dio_events
    | {
        "electrode_group_info": electrode_group_info,
        "well_visits": well_visits,
        "trials": trials_df,
        "task_variables": task_variables_df,
    }
)
```

### M2. Convert String Constants to Enum

**File:** `src/continuum_swr_replay/parameters.py:67-70`
**Issue:** Using bare list for `training_types` - no type safety
**Impact:** Easy to introduce typos, no IDE autocomplete
**Fix:**

```python
from enum import Enum

class TrainingType(str, Enum):
    """Types of training data selection for decoder."""
    ALL = "all"
    NO_RIPPLE = "no_ripple"

    def __str__(self) -> str:
        return self.value

# Update function signatures:
def get_training_timepoints(
    data: dict[str, Any],
    training_type: TrainingType = TrainingType.NO_RIPPLE
) -> np.ndarray:
    if training_type == TrainingType.ALL:
        # ...
    elif training_type == TrainingType.NO_RIPPLE:
        # ...
```

### M3. Fix Circular Import with `TYPE_CHECKING`

**File:** `src/continuum_swr_replay/data_loaders/bandit_task.py:27-34`
**Issue:** `SessionData` TypedDict in wrong module (analysis instead of types)
**Impact:** Backwards dependency, confusing architecture
**Fix:**

```
# Create new shared types module:
src/continuum_swr_replay/
├── types.py              # NEW - shared type definitions
├── data_loaders/
│   └── bandit_task.py    # imports from ..types
└── analysis/
    └── hippocampus/
        └── types.py      # Move SessionData here → to ../../types.py
```

```python
# data_loaders/bandit_task.py
from ..types import SessionData  # Clean, direct import

def load_data(...) -> SessionData:
    ...
```

### M4. Use SCREAMING_CASE for Module-Level Constants

**File:** `src/continuum_swr_replay/parameters.py:67`
**Issue:** `training_types` should be `TRAINING_TYPES` per PEP 8
**Impact:** Style consistency
**Fix:**

```python
# Change:
training_types = ["all", "no_ripple"]
# To:
TRAINING_TYPES = ["all", "no_ripple"]  # Or use Enum (M2)
```

### M5. Inefficient DataFrame Construction

**File:** `src/continuum_swr_replay/data_loaders/utils.py:114-116`
**Issue:** Creating intermediate numpy array unnecessarily
**Impact:** Minor performance waste
**Fix:**

```python
# Change from:
event_labels = pd.DataFrame(
    np.zeros_like(time, dtype=int), index=time, columns=["event_number"]
)

# To:
event_labels = pd.DataFrame(
    0,  # Scalar broadcasts to fill
    index=time,
    columns=["event_number"],
    dtype=int
)
```

### M6. Vectorize Event Labeling with IntervalIndex

**File:** `src/continuum_swr_replay/data_loaders/utils.py:99-122`
**Issue:** Using explicit loop for interval-based labeling (slow for many events)
**Impact:** Performance on large datasets
**Fix:**

```python
def get_labels(times: pd.DataFrame, time: pd.Index) -> pd.DataFrame:
    """Create event labels using IntervalIndex (vectorized)."""
    # Create interval index from start/end times
    intervals = pd.IntervalIndex.from_arrays(
        times['start_time'],
        times['end_time'],
        closed='both'
    )

    # Vectorized lookup
    event_labels = pd.DataFrame(
        0,
        index=time,
        columns=['event_number'],
        dtype=int
    )

    for i, interval in enumerate(intervals, start=1):
        mask = time.to_series().between(interval.left, interval.right)
        event_labels.loc[mask, 'event_number'] = i

    return event_labels
```

### M7. Use TypedDict for Method Return Types

**File:** `scripts/run_decoding_bandit.py:70-83`
**Issue:** Return type `dict[str, Any]` is too vague
**Impact:** No type checking on returned dictionary structure
**Fix:**

```python
from typing import TypedDict

class ClusterlessParams(TypedDict):
    position_std: float
    waveform_std: float
    block_size: int

class SortedSpikesParams(TypedDict):
    position_std: float
    block_size: int

@dataclass(frozen=True)
class DecodingConfig:
    # ...

    def to_clusterless_params(self) -> ClusterlessParams:
        return {
            "position_std": self.position_std,
            "waveform_std": self.waveform_std,
            "block_size": self.block_size,
        }

    def to_sorted_spikes_params(self) -> SortedSpikesParams:
        return {
            "position_std": self.position_std,
            "block_size": self.block_size,
        }
```

### M8. Centralize Script Constants in `parameters.py`

**File:** `scripts/run_decoding_bandit.py:45-59`
**Issue:** Constants duplicated between `parameters.py` and scripts
**Impact:** Risk of inconsistency
**Fix:**

```python
# Move to parameters.py:
PLACE_BIN_SIZE_1D: Final = 2.0
PLACE_BIN_SIZE_2D: Final = 3.5
POSITION_STD: Final = float(np.sqrt(12.5))
WAVEFORM_STD: Final = 24.0
BLOCK_SIZE: Final = 2**10

DETECTOR_CLASSES: Final[dict[str, tuple[type, str]]] = {
    "HPC": (NonLocalClusterlessDetector, "clusterless"),
    "mPFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
    "OFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
}

# Import in script:
from continuum_swr_replay.parameters import (
    PLACE_BIN_SIZE_1D,
    DETECTOR_CLASSES,
    # ...
)
```

---

## 🟢 Minor Improvements (Low Priority - Nice to Have)

### L1. Remove Obvious Comments

**File:** Various
**Issue:** Comments that restate what code does
**Example:**

```python
# ❌ Bad:
# Concatenate all spike times
concat_spike_times = np.concatenate(spike_times)

# ✅ Good:
# Create single array for vectorized closeness detection
concat_spike_times = np.concatenate(spike_times)
```

### L2. Add Doctest Validation

**File:** `src/continuum_swr_replay/data_loaders/bandit_task.py:86-143`
**Issue:** Beautiful examples in docstrings, but are they tested?
**Fix:**

```python
# Add to bottom of modules with examples:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Or in pytest.ini / pyproject.toml:
[tool.pytest.ini_options]
addopts = "--doctest-modules"
```

### L3. Use `logging.config.dictConfig` for Cleaner Setup

**File:** `scripts/run_decoding_bandit.py:91-150`
**Issue:** Imperative logging setup is verbose
**Fix:**

```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(message)s',
            'datefmt': '%d-%b-%y %H:%M:%S'
        },
    },
    'filters': {
        'skip_checksum': {'()': SkipChecksumFilter},
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(log_path),
            'mode': 'w',
            'formatter': 'standard',
            'filters': ['skip_checksum'],
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'filters': ['skip_checksum'],
        },
    },
    'loggers': {
        name_logfile: {
            'handlers': ['file', 'console'],
            'level': level,
            'propagate': False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
return logging.getLogger(name_logfile)
```

### L4. Define String Keys as Constants

**File:** Throughout codebase
**Issue:** Magic strings for dict keys (`data["spike_times"]`, etc.)
**Fix:**

```python
# In types.py or constants.py:
class SessionDataKeys:
    """String keys for SessionData dictionary."""
    SPIKE_TIMES = "spike_times"
    SPIKE_WAVEFORM_FEATURES = "spike_waveform_features"
    POSITION_INFO = "position_info"
    TRACK_GRAPH = "track_graph"
    RIPPLE_TIMES = "ripple_times"
    # ... etc

# Usage:
from continuum_swr_replay.types import SessionDataKeys as SDK

spike_times = data[SDK.SPIKE_TIMES]  # Autocomplete + typo prevention
```

### L5. Organize Constants into Submodules

**File:** `src/continuum_swr_replay/data_loaders/constants.py`
**Issue:** Single 212-line file with different types of constants
**Fix:**

```
constants/
├── __init__.py        # Re-export everything
├── files.py           # NWB_FILES
├── spatial.py         # WELL_PATCH_MAPPING, TRACK_SEGMENT_*, etc.
└── decoding.py        # Decoding-related constants
```

### L6. Add Type Hints to All Functions

**File:** Various
**Issue:** Some helper functions missing return type hints
**Example:**

```python
# Before:
def _count_exposure(df: pd.DataFrame):  # Missing return type
    df["exposure"] = np.arange(len(df)) + 1
    return df

# After:
def _count_exposure(df: pd.DataFrame) -> pd.DataFrame:
    df["exposure"] = np.arange(len(df)) + 1
    return df
```

### L7. Use `pathlib.Path` Consistently

**File:** `src/continuum_swr_replay/parameters.py:13-15`
**Issue:** Mixing Path and string operations
**Fix:**

```python
# Already doing this well! Just ensure consistency:
ROOT_DIR: Final[Path] = Path(__file__).parent.parent
PROCESSED_DATA_DIR: Final[Path] = ROOT_DIR / "Processed-Data"
FIGURE_DIR: Final[Path] = ROOT_DIR / "figures"

# Ensure all file operations use Path methods:
with open(PROCESSED_DATA_DIR / "file.txt") as f:  # ✅
# Not: open(str(PROCESSED_DATA_DIR) + "/file.txt")  # ❌
```

### L8. Add `__all__` to Public Modules

**File:** All modules in `src/continuum_swr_replay/`
**Issue:** No explicit public API definition
**Fix:**

```python
# At top of each module:
__all__ = [
    "load_data",
    "get_epoch_info",
    "get_training_timepoints",
]

# Benefits:
# - Clear public API
# - Helps IDEs with autocomplete
# - from module import * works correctly (though rarely used)
```

### L9. Use `@cache` for Expensive Pure Functions

**File:** `src/continuum_swr_replay/data_loaders/utils.py:18-96`
**Issue:** `get_epoch_info()` queries database every time
**Fix:**

```python
from functools import cache

@cache  # Python 3.9+
def get_epoch_info() -> pd.DataFrame:
    """Get information about all available epochs (cached)."""
    # ... expensive database query
    return epoch_info
```

### L10. Add Validation to Dataclass Fields

**File:** `src/continuum_swr_replay/parameters.py:18-60`
**Issue:** No validation that `block_size` is power of 2, etc.
**Fix:**

```python
from dataclasses import dataclass, field

def _validate_block_size(value: int) -> int:
    """Ensure block_size is a power of 2."""
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"block_size must be power of 2, got {value}")
    return value

@dataclass(frozen=True)
class DecodingParameters:
    """Parameters for neural decoding analysis."""

    place_bin_size: float = 2.0
    waveform_std: float = 24.0
    position_std: float = 6.0
    block_size: int = field(default=4096)

    def __post_init__(self):
        # Validate block_size is power of 2
        _validate_block_size(self.block_size)

        # Validate positive values
        if self.place_bin_size <= 0:
            raise ValueError("place_bin_size must be positive")
        if self.waveform_std <= 0:
            raise ValueError("waveform_std must be positive")
        if self.position_std <= 0:
            raise ValueError("position_std must be positive")
```

### L11. Use More Descriptive Variable Names

**File:** Various
**Issue:** Single-letter or abbreviated names in comprehensions
**Example:**

```python
# Before:
frac = (
    df.loc[df.labels > 0].groupby("labels").sort_group_id.nunique() / n_sort_groups
)

# After:
coincident_fraction = (
    df.loc[df.labels > 0]
    .groupby("labels")
    .sort_group_id.nunique()
    / n_sort_groups
)
```

### L12. Add Property-Based Tests with Hypothesis

**File:** `tests/property/test_mathematical_invariants.py` (expand)
**Issue:** Great start, but could test more invariants
**Ideas:**

```python
from hypothesis import given
import hypothesis.strategies as st

@given(st.lists(st.floats(min_value=0, max_value=100)))
def test_spike_filtering_preserves_order(spike_times):
    """Filtered spikes should maintain temporal order."""
    filtered, _ = detect_coincident_spikes([np.array(spike_times)])
    if filtered:
        assert all(np.diff(filtered[0]) >= 0), "Spike times not ordered"

@given(st.lists(st.floats(min_value=0, max_value=1000)))
def test_event_labels_no_overlap(event_times):
    """Each time point should have at most one event label."""
    # ... test that labels don't overlap
```

---

## 📋 Architecture Improvements (Long-term)

### A1. Create Shared Type Definitions Module

**Current:** Types scattered across modules
**Proposed:** Single source of truth for types
**Structure:**

```
src/continuum_swr_replay/
├── types.py              # All TypedDicts, Protocols
│   ├── SessionData
│   ├── SpikeDataDict
│   ├── ClusterlessParams
│   ├── SortedSpikesParams
│   └── ...
```

### A4. Add Pre-commit Hooks

**File:** `.pre-commit-config.yaml` (create)
**Purpose:** Ensure code quality automatically

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

---

## 📊 Summary Statistics

| Priority | Count | Estimated Effort |
|----------|-------|------------------|
| 🔴 Critical | 3 | 2-4 hours |
| 🟡 Medium | 8 | 1-2 days |
| 🟢 Low | 12 | 3-5 days |
| 📋 Architecture | 4 | 1-2 weeks |
| **Total** | **27** | **~2-3 weeks** |

---

## 🎯 Recommended Implementation Order

### Week 1: Critical Fixes

- [ ] C1: Fix `strict=False` in zip calls
- [ ] C2: Refactor filename parsing with regex
- [ ] C3: Remove redundant enumerate + itertuples
- [ ] M1: Use `|` for dict merging
- [ ] M4: Fix SCREAMING_CASE for constants

### Week 2: Type Safety & Enums

- [ ] M2: Convert training_types to Enum
- [ ] M3: Move SessionData to shared types.py
- [ ] M7: Add TypedDict for detector params
- [ ] L6: Add missing type hints
- [ ] L8: Add `__all__` to public modules

### Week 3: Performance & Testing

- [ ] M5: Optimize DataFrame construction
- [ ] M6: Vectorize event labeling
- [ ] L2: Enable doctest validation
- [ ] L9: Add `@cache` to expensive functions
- [ ] L12: Expand property-based tests

### Week 4: Polish & Architecture

- [ ] M8: Centralize constants in parameters.py
- [ ] L1: Remove obvious comments
- [ ] L5: Organize constants into submodules
- [ ] A1: Create shared types module
- [ ] A4: Add pre-commit hooks

---

## 📖 References

- [PEP 8: Style Guide](https://peps.python.org/pep-0008/)
- [PEP 484: Type Hints](https://peps.python.org/pep-0484/)
- [Raymond Hettinger Talks](https://pyvideo.org/speaker/raymond-hettinger.html)
- [Effective Python by Brett Slatkin](https://effectivepython.com/)
- [Python Type Checking Guide](https://realpython.com/python-type-checking/)

---

*Last updated: 2025-11-14*
*Review by: Claude (channeling Raymond Hettinger)*
