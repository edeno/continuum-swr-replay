# Scripts Refactoring Plan

## Overview

This document outlines an incremental refactoring plan for the scripts directory to improve code organization, eliminate duplication, and follow best practices outlined in CLAUDE.md.

**Guiding Principles:**

1. Scripts orchestrate, libraries implement - no analysis logic in scripts
2. Avoid duplicating functionality already in `src/continuum_swr_replay/analysis/`
3. Changes are incremental and testable
4. Follow conventions in CLAUDE.md (time in seconds, proper naming, type hints)
5. All new functions get docstrings and type hints

---

## Current State Assessment

### Existing Functionality (DO NOT DUPLICATE)

**Already in `analysis/hippocampus/`:**

- ✅ `get_confident_times()` - Detect confident events from probability thresholds
- ✅ `get_overlap_times()` - Find temporal overlap between two event types
- ✅ `get_data_over_time()` - Aggregate decoding results into time-indexed DataFrame
- ✅ `get_conditional_non_local_posterior()` - Extract conditional posterior
- ✅ `get_map_speed()` - Compute decoded position speed
- ✅ `get_posterior_stats()` - Calculate posterior summary statistics

**Already in `visualization/plots.py`:**

- ✅ `plot_peri_event_probability()` - Peri-event time histogram with multiple signals
- ✅ Component plots (theta, ripple, MUA, spike rasters, etc.)
- ✅ `plot_detector()` - Comprehensive multi-panel decoding figure

### Scripts Analysis

| Script | Lines | Status | Main Issues |
|--------|-------|--------|-------------|
| `run_decoding_bandit.py` | 766 | ✅ Good | Well-structured, minimal changes needed |
| `generate_figure01.py` | 71 | ✅ Good | Simple simulation demo |
| `generate_figure02.py` | 2015 | ❌ Major | Exploratory notebook disguised as script; massive repetition |
| `generate_figure03.py` | 1018 | ⚠️ Moderate | Contains reusable functions that should be in src/ |
| `generate_figure04.py` | 291 | ⚠️ Minor | GLM plotting should be in visualization module |
| `test.ipynb` | - | ℹ️ Test | Data loading test (keep as-is) |

---

## Phase 1: Extract Core Utilities (Weeks 1-2)

### Goal

Extract NEW reusable functions from scripts to `src/` without duplicating existing functionality.

**CRITICAL:** Many initially proposed functions duplicate existing code. This revised plan removes duplicates and focuses on truly new functionality.

### 1.1: Event Analysis Functions

**Create:** `src/continuum_swr_replay/analysis/hippocampus/event_analysis.py`

**Note:** Placed in `analysis/hippocampus/` alongside `event_detection.py` to maintain clear module organization.

```python
"""Analysis functions for manipulating and characterizing detected events.

This module complements event_detection.py (which detects events from signals)
by providing functions to analyze, filter, and characterize already-detected events.
"""

import logging
import numpy as np
import pandas as pd
from typing import Literal

logger = logging.getLogger(__name__)


def get_immobile_events(
    data: pd.DataFrame,
    event_col: str,
    speed_col: str = "head_speed",
    speed_threshold: float = 4.0,
    method: Literal["endpoints", "max", "mean"] = "endpoints",
    group_cols: list[str] = ["animal", "day", "epoch"],
) -> pd.DataFrame:
    """Filter events to only include those during low-speed periods.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed data with event labels and speed
    event_col : str
        Column containing event labels (0 = no event, 1+ = event ID)
    speed_col : str
        Column containing speed values (cm/s)
    speed_threshold : float
        Maximum speed threshold (cm/s)
    method : {"endpoints", "max", "mean"}
        How to evaluate speed during event:
        - "endpoints": Both start and end must be below threshold (default, most restrictive)
        - "max": Maximum speed during event must be below threshold
        - "mean": Mean speed during event must be below threshold
    group_cols : list[str]
        Grouping columns

    Returns
    -------
    pd.DataFrame
        Filtered data containing only low-speed events

    Notes
    -----
    The "endpoints" method matches existing patterns in generate_figure02.py
    and is appropriate for ensuring immobility at event boundaries (relevant
    for replay analysis where animal should be still).

    Examples
    --------
    >>> immobile_replay = get_immobile_events(
    ...     data_over_time,
    ...     event_col="non_local_event",
    ...     speed_threshold=4.0,
    ...     method="endpoints"
    ... )
    """
    # Validate inputs
    if speed_threshold <= 0:
        raise ValueError(f"speed_threshold must be positive, got {speed_threshold}")
    if method not in ["endpoints", "max", "mean"]:
        raise ValueError(f"method must be 'endpoints', 'max', or 'mean', got {method}")

    if event_col not in data.columns:
        raise ValueError(f"event_col '{event_col}' not found in data")
    if speed_col not in data.columns:
        raise ValueError(f"speed_col '{speed_col}' not found in data")

    def _filter_low_speed(group):
        if method == "endpoints":
            return (
                (group[speed_col].iloc[0] < speed_threshold) and
                (group[speed_col].iloc[-1] < speed_threshold)
            )
        elif method == "max":
            return group[speed_col].max() < speed_threshold
        elif method == "mean":
            return group[speed_col].mean() < speed_threshold

    return (
        data.loc[data[event_col] > 0]
        .groupby(group_cols + [event_col])
        .filter(_filter_low_speed)
    )


def get_event_start_end_times(
    data: pd.DataFrame,
    event_col: str,
    group_cols: list[str] = ["animal", "day", "epoch"],
) -> pd.DataFrame:
    """Extract start and end times for labeled events.

    Converts event labels (0 = no event, 1+ = event ID) to interval format
    with start_time, end_time columns. This is the inverse operation of
    get_labels() from data_loaders.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed DataFrame with event labels
    event_col : str
        Column containing event IDs (0 = no event, 1+ = event number)
    group_cols : list[str]
        Grouping columns (typically animal, day, epoch)

    Returns
    -------
    pd.DataFrame
        Columns: [*group_cols, event_col, start_time, end_time, duration]
        One row per unique event (event_col > 0)

    Notes
    -----
    - Duration is in seconds (end_time - start_time)
    - Events with ID 0 (non-events) are automatically excluded
    - Time values come from DataFrame index

    Examples
    --------
    >>> event_times = get_event_start_end_times(
    ...     data_over_time,
    ...     event_col="non_local_event"
    ... )
    >>> print(event_times.head())
       animal       day epoch  non_local_event  start_time  end_time  duration
    0  chimi  20200212  02_r1                1     1234.56   1234.62      0.06
    """
    if event_col not in data.columns:
        raise ValueError(f"event_col '{event_col}' not found in data")

    # Extract start/end times for each event
    event_times = (
        data.reset_index()
        .groupby(group_cols + [event_col])
        .agg(start_time=("index", "min"), end_time=("index", "max"))
        .reset_index()
    )

    # Remove non-events (event_col == 0)
    event_times = event_times[event_times[event_col] > 0]

    # Compute duration
    event_times["duration"] = event_times["end_time"] - event_times["start_time"]

    return event_times


def get_event_durations_ms(event_times: pd.DataFrame) -> pd.Series:
    """Convert event durations from seconds to milliseconds.

    This is a convenience wrapper for event times from get_confident_times()
    or get_event_start_end_times().

    Parameters
    ----------
    event_times : pd.DataFrame
        Event times with 'duration' column in seconds

    Returns
    -------
    pd.Series
        Event durations in milliseconds

    Notes
    -----
    For events from get_confident_times(), the duration column is already
    computed accurately as (end_time - start_time). This is more accurate
    than count-based duration estimation.
    """
    if "duration" not in event_times.columns:
        raise ValueError("event_times must have 'duration' column")

    return event_times["duration"] * 1000.0
```

**Source:** Extract from `generate_figure03.py` lines 406-414 and patterns from `generate_figure02.py`
**Status:** NEW - complements existing `get_confident_times()` and `get_overlap_times()`
**Rationale:** Removes duplicates (label_continuous_events, compute_event_durations) and adds missing utility (get_event_start_end_times)

---

### 1.2: Event-Triggered Analysis

**Create:** `src/continuum_swr_replay/analysis/hippocampus/triggered_analysis.py`

**Note:** Renamed from `time_series_utils.py` for clarity - this module is specifically for event-triggered temporal analysis.

```python
"""Event-triggered analysis utilities for extracting signal windows around events."""

import logging
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def get_signal_windows(
    data: pd.DataFrame,
    event_times: pd.DataFrame,
    signal_cols: list[str],
    time_col: str = "start_time",
    window_sec: float = 2.0,
    sampling_frequency: float = 500.0,
    event_id_col: str | None = None,
) -> xr.DataArray:
    """Extract signal windows around event times.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed DataFrame with signal columns and grouping columns
        Must have columns: animal, day, epoch, {signal_cols}
    event_times : pd.DataFrame
        DataFrame with event times and metadata
        Must have columns: animal, day, epoch, {time_col}
    signal_cols : list[str]
        Names of signal columns to extract from data
    time_col : str
        Column in event_times with event times (typically "start_time" or "end_time")
    window_sec : float
        Total window duration in seconds (centered on event time)
    sampling_frequency : float
        Sampling rate in Hz
    event_id_col : str | None
        Column name in event_times for event IDs (for coordinate labeling)
        If None, sequential integers are used

    Returns
    -------
    xr.DataArray
        Shape: (n_events, n_timepoints, n_signals)
        Dimensions: ("event", "time", "signal")
        Coordinates:
            - event: MultiIndex from event_times if event_id_col provided, else integers
            - time: Time relative to event in seconds (negative = before, positive = after)
            - signal: Signal names from signal_cols

    Raises
    ------
    ValueError
        If required columns are missing or parameters are invalid

    Notes
    -----
    - Windows are centered on event times
    - Missing data filled with NaN (occurs for events near data boundaries)
    - Events outside the data time range will have partial or fully NaN windows
    - Empty sessions (no events) are logged and skipped

    Examples
    --------
    >>> signal_windows = get_signal_windows(
    ...     data_over_time,
    ...     non_local_times,
    ...     signal_cols=["theta_power_zscore", "SWR_zscore", "head_speed"],
    ...     time_col="start_time",
    ...     window_sec=2.0,
    ... )
    >>> # Average across events
    >>> mean_triggered = signal_windows.mean(dim="event")
    >>> # Plot
    >>> mean_triggered.sel(signal="SWR_zscore").plot()
    """
    # Validate inputs
    if sampling_frequency <= 0:
        raise ValueError(f"sampling_frequency must be positive, got {sampling_frequency}")
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}")

    required_data_cols = ["animal", "day", "epoch"] + signal_cols
    missing_cols = set(required_data_cols) - set(data.columns)
    if missing_cols:
        raise ValueError(f"data missing required columns: {missing_cols}")

    required_event_cols = ["animal", "day", "epoch", time_col]
    if event_id_col is not None:
        required_event_cols.append(event_id_col)
    missing_cols = set(required_event_cols) - set(event_times.columns)
    if missing_cols:
        raise ValueError(f"event_times missing required columns: {missing_cols}")

    # Compute window parameters
    half_window_sec = window_sec / 2.0
    dt = 1 / sampling_frequency
    half_window_samples = int(round(half_window_sec / dt))
    window_len = 2 * half_window_samples + 1
    offsets = np.arange(-half_window_samples, half_window_samples + 1)

    all_windows = []
    event_ids = []

    for (animal, day, epoch), group in data.groupby(["animal", "day", "epoch"]):
        # Get events for this session
        session_events = event_times.loc[
            (event_times.animal == animal)
            & (event_times.day == day)
            & (event_times.epoch == epoch)
        ]

        if len(session_events) == 0:
            logger.debug(f"No events found for {animal}/{day}/{epoch}, skipping")
            continue

        times = session_events[time_col].to_numpy()
        time_index = group.index.values

        # Find closest valid indices for each event time
        center_idxs = np.searchsorted(time_index, times)
        # Clip to valid range (handles events outside data timespan)
        center_idxs = np.clip(center_idxs, 0, len(group) - 1)

        # Warn about events far outside data range
        time_min, time_max = time_index[0], time_index[-1]
        out_of_range = (times < time_min - half_window_sec) | (times > time_max + half_window_sec)
        if out_of_range.any():
            logger.warning(
                f"{out_of_range.sum()} events for {animal}/{day}/{epoch} "
                f"outside data time range [{time_min:.2f}, {time_max:.2f}], "
                f"will have partial windows"
            )

        # Create window indices
        all_window_idxs = center_idxs[:, None] + offsets[None, :]

        # Check bounds
        mask = (all_window_idxs >= 0) & (all_window_idxs < len(group))

        # Initialize windows with NaN
        n_events_session = len(times)
        n_signals = len(signal_cols)
        windows = np.full((n_events_session, window_len, n_signals), np.nan)

        # Fill valid data
        valid_rows, valid_cols = np.where(mask)
        valid_indices = all_window_idxs[mask]

        for i, signal_col in enumerate(signal_cols):
            windows[valid_rows, valid_cols, i] = group[signal_col].to_numpy()[valid_indices]

        all_windows.append(windows)

        # Track event IDs
        if event_id_col is not None:
            event_ids.extend(
                pd.MultiIndex.from_frame(
                    session_events[["animal", "day", "epoch", event_id_col]]
                ).tolist()
            )
        else:
            event_ids.extend(range(len(event_ids), len(event_ids) + n_events_session))

    if not all_windows:
        # Return empty array with proper structure
        logger.warning("No events found in any session, returning empty array")
        return xr.DataArray(
            np.empty((0, window_len, len(signal_cols))),
            dims=("event", "time", "signal"),
            coords={
                "event": [],
                "time": offsets * dt,
                "signal": signal_cols,
            },
        )

    # Concatenate all sessions
    all_windows_arr = np.vstack(all_windows)
    time_axis = offsets * dt

    return xr.DataArray(
        all_windows_arr,
        dims=("event", "time", "signal"),
        coords={
            "event": event_ids,
            "time": time_axis,
            "signal": signal_cols,
        },
    )
```

**Source:** Extract from `generate_figure03.py` lines 406-513
**Status:** NEW functionality - complements existing `plot_peri_event_probability()`

---

### 1.3: Visualization Utilities

**Create:** `src/continuum_swr_replay/visualization/event_plots.py`

```python
"""Plotting utilities for event-related analyses."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from scipy.stats import sem

from continuum_swr_replay.parameters import COLORS


def plot_event_duration_histogram(
    durations: pd.Series,
    bins: np.ndarray | None = None,
    color: str | None = None,
    label: str | None = None,
    ax: Axes | None = None,
    density: bool = True,
    **hist_kwargs,
) -> Axes:
    """Plot event duration histogram with clean styling.

    Parameters
    ----------
    durations : pd.Series
        Event durations in milliseconds
    bins : np.ndarray | None
        Bin edges (default: np.arange(0, 250, 10))
    color : str | None
        Color for histogram
    label : str | None
        Label for legend
    ax : Axes | None
        Axes to plot on
    density : bool
        Normalize to probability density
    **hist_kwargs
        Additional arguments to plt.hist()

    Returns
    -------
    Axes
        Axes object
    """
    if ax is None:
        ax = plt.gca()

    if bins is None:
        bins = np.arange(0, 250, 10)

    ax.hist(
        durations,
        bins=bins,
        color=color,
        label=label,
        density=density,
        alpha=0.7,
        edgecolor="none",
        **hist_kwargs,
    )

    # Clean styling
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Event Duration (ms)")
    ax.set_ylabel("Density" if density else "Count")
    ax.tick_params(axis="both", which="both", length=3, direction="out")

    return ax


def plot_triggered_average(
    signal_windows: xr.DataArray,
    signal_names: list[str] | None = None,
    ylim_dict: dict[str, tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (6, 4),
    time_label: str = "Time from event start (s)",
    show_sem: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot event-triggered averages for multiple signals.

    Parameters
    ----------
    signal_windows : xr.DataArray
        Signal windows with dimensions (event, time, signal)
    signal_names : list[str] | None
        Subset of signals to plot (default: all)
    ylim_dict : dict | None
        Y-axis limits per signal
    figsize : tuple
        Figure size
    time_label : str
        X-axis label
    show_sem : bool
        Show standard error of mean

    Returns
    -------
    fig : Figure
    axes : np.ndarray
        Array of Axes
    """
    if signal_names is None:
        signal_names = signal_windows.coords["signal"].values.tolist()

    n_signals = len(signal_names)
    fig, axes = plt.subplots(
        n_signals, 1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )

    if n_signals == 1:
        axes = np.array([axes])

    mean_signals = signal_windows.mean(dim="event")

    if show_sem:
        sem_signals = sem(signal_windows.values, axis=0, nan_policy="omit")

    for ax, signal_name in zip(axes, signal_names):
        signal_data = mean_signals.sel(signal=signal_name)
        time = signal_data.coords["time"].values

        ax.plot(time, signal_data.values, color="black", linewidth=1)

        if show_sem:
            signal_idx = list(signal_windows.coords["signal"].values).index(signal_name)
            sem_vals = sem_signals[:, signal_idx]
            ax.fill_between(
                time,
                signal_data.values - sem_vals,
                signal_data.values + sem_vals,
                color="gray",
                alpha=0.3,
                linewidth=0,
            )

        # Styling
        ax.axvline(0, color="lightgrey", linestyle="--", linewidth=0.75, alpha=0.5)
        ax.axhline(0, color="lightgrey", linestyle="--", linewidth=0.75, alpha=0.5)
        ax.set_ylabel(signal_name.replace("_", " ").title())

        if ylim_dict and signal_name in ylim_dict:
            ax.set_ylim(ylim_dict[signal_name])

        ax.tick_params(axis="both", which="both", length=2, width=0.5, labelsize=7)

    axes[-1].set_xlabel(time_label, labelpad=2)

    return fig, axes


def plot_venn_diagram_overlap(
    event_counts: dict[str, int],
    colors: dict[str, str],
    title: str = "Event Overlap",
    figsize: tuple[float, float] = (8, 8),
) -> plt.Figure:
    """Plot 3-way Venn diagram for event overlaps.

    Parameters
    ----------
    event_counts : dict
        Dictionary with keys like "A only", "B only", "A ∩ B", "All three"
    colors : dict
        Colors for each event type
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
    """
    # Based on figure02.py:1186-1229
    # Implementation of Venn diagram plotting
    ...
```

**Source:** Extract patterns from `generate_figure02.py` (lines 678-777) and `generate_figure03.py` (600-703)
**Status:** NEW - provides cleaner interface than repetitive code in scripts

---

**Create:** `src/continuum_swr_replay/visualization/glm_plots.py`

```python
"""Plotting utilities for GLM analysis results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_raster_relative_to_event(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float],
    ax: Axes | None = None,
    color: str = "black",
) -> Axes:
    """Plot spike raster aligned to event times.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times for a single neuron
    event_times : np.ndarray
        Event times to align to
    window : tuple
        (start, end) window relative to event
    ax : Axes | None
        Axes to plot on
    color : str
        Raster color

    Returns
    -------
    Axes
    """
    if ax is None:
        ax = plt.gca()

    relative_spike_times = []
    for event_time in event_times:
        event_relative = spike_times - event_time
        in_window = (event_relative > window[0]) & (event_relative < window[1])
        relative_spike_times.append(event_relative[in_window])

    ax.eventplot(relative_spike_times, color=color)
    ax.set_xlim(window)
    ax.axvline(0, color="black", linestyle="--")

    return ax


def plot_glm_modulation(
    lag_time: np.ndarray,
    spline_basis: np.ndarray,
    glm_result,
    modulation_bounds: tuple[float, float] = (np.log(0.5), np.log(2.0)),
    ax: Axes | None = None,
) -> Axes:
    """Plot GLM coefficient modulation over time.

    Parameters
    ----------
    lag_time : np.ndarray
        Time lags
    spline_basis : np.ndarray
        Spline basis functions
    glm_result
        Fitted GLM result with params and bse attributes
    modulation_bounds : tuple
        (min, max) log-scale modulation bounds
    ax : Axes | None
        Axes to plot on

    Returns
    -------
    Axes
    """
    if ax is None:
        ax = plt.gca()

    # Plot modulation
    modulation = glm_result.params[1:] @ spline_basis.T
    ax.plot(lag_time, modulation, color="C0", linewidth=1.5)

    # Error bands
    se_modulation = glm_result.bse[1:] @ spline_basis.T
    ax.fill_between(
        lag_time,
        modulation + se_modulation,
        modulation - se_modulation,
        alpha=0.25,
        color="C0",
    )

    # Reference lines
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

    # Styling
    ax.set_ylim(modulation_bounds)
    ax.set_ylabel("Modulation")
    ax.set_yticklabels([
        f"{np.exp(modulation_bounds[0]):.1f}",
        "1",
        f"{np.exp(modulation_bounds[1]):.1f}"
    ])

    return ax
```

**Source:** Extract from `generate_figure04.py` lines 20-82, 85-194
**Status:** NEW - GLM-specific plotting not currently in visualization module

---

### 1.4: Dataset Utilities

**Create:** `src/continuum_swr_replay/analysis/dataset_utils.py`

```python
"""Utilities for loading and managing processed datasets."""

import pandas as pd
from pathlib import Path


def load_processed_dataset(
    data_dir: Path,
    pattern: str = "*_HPC_80_data_over_time.pkl",
    add_derived_columns: bool = True,
) -> pd.DataFrame:
    """Load and concatenate processed data files.

    This is a common pattern in figure generation scripts.

    Parameters
    ----------
    data_dir : Path
        Directory containing processed data
    pattern : str
        Glob pattern for files
    add_derived_columns : bool
        Add commonly used derived columns

    Returns
    -------
    pd.DataFrame
        Concatenated data from all matching files
    """
    data_path = data_dir.glob(pattern)
    data = pd.concat([pd.read_pickle(p) for p in data_path])

    if add_derived_columns:
        # Add commonly used derived columns
        data["Non-Local_prob"] = (
            data["Non-Local_Continuous_prob"]
            + data["Non-Local_Fragmented_prob"]
        )
        data["distance"] = data["ahead_behind_distance"].abs()

    return data
```

**Source:** Pattern from `generate_figure02.py` lines 30-44, `generate_figure03.py` lines 34-45
**Status:** NEW - common utility pattern

---

## Phase 2: Refactor Figure Scripts (Week 2)

### Goal

Convert figure scripts to use extracted utilities, making them concise orchestration scripts.

### 2.1: Refactor `generate_figure03.py`

**Before:** 1018 lines
**After Target:** ~300 lines

**Changes:**

1. Replace `_get_signal_windows()` with `extract_signal_windows()` from `time_series_utils`
2. Use `filter_events_by_speed()` from `event_utils`
3. Use `plot_triggered_average()` from `event_plots`
4. Keep only figure-specific layout and composition

**Example structure:**

```python
#!/usr/bin/env python
"""Generate Figure 3: Non-local event characteristics.

Shows duration, spatial properties, and physiological correlates.
"""
from pathlib import Path
from continuum_swr_replay.analysis.dataset_utils import load_processed_dataset
from continuum_swr_replay.analysis.event_utils import (
    compute_event_durations, filter_events_by_speed
)
from continuum_swr_replay.analysis.time_series_utils import extract_signal_windows
from continuum_swr_replay.visualization.event_plots import (
    plot_event_duration_histogram, plot_triggered_average
)


def create_panel_a(data):
    """Event duration distributions."""
    durations = compute_event_durations(data, "non_local_event")
    # ... plot


def create_panel_b(data):
    """Triggered averages."""
    # ... use extract_signal_windows + plot_triggered_average


def main():
    data = load_processed_dataset(PROCESSED_DATA_DIR)
    output_dir = Path("paper/figures")
    output_dir.mkdir(exist_ok=True, parents=True)

    create_panel_a(data)
    create_panel_b(data)
    # ... etc
```

**Testing:**

```bash
# After refactoring, verify figures are identical
python generate_figure03.py
# Compare with saved baseline figures
```

---

### 2.2: Refactor `generate_figure02.py`

**Strategy:** Break into multiple focused scripts

**Before:** 2015 lines (monolithic)
**After:** 4-5 scripts of ~200-300 lines each

**New scripts:**

1. `generate_figure02a_distributions.py` - Probability distributions (lines 1-283)
2. `generate_figure02b_triggered_averages.py` - Event-triggered analysis (284-669)
3. `generate_figure02c_overlaps.py` - Venn diagrams and overlaps (670-1229)
4. `generate_figure02d_timing.py` - Timing relationships (1230-2015)

**Each script:**

- Imports from `src/` modules
- Has CLI arguments (`--output-dir`, `--format`)
- Generates one panel or related set of panels
- ~200-300 lines max

---

### 2.3: Refactor `generate_figure04.py`

**Changes:**

1. Replace plotting functions with `glm_plots.plot_glm_modulation()`, `plot_raster_relative_to_event()`
2. Add CLI arguments for session selection
3. Add docstring

**Before:** 291 lines with embedded functions
**After:** ~150 lines using library functions

---

## Phase 3: Add CLI and Configuration (Week 3)

### Goal

Make all figure scripts configurable and properly documented.

### 3.1: Standardize CLI Interface

All figure scripts should support:

```python
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    # Generate figures...
```

### 3.2: Analysis Constants

**Create:** `src/continuum_swr_replay/analysis/constants.py`

**Note:** Following existing pattern in `parameters.py`, use module-level constants instead of dataclasses.

```python
"""Analysis constants and default parameters.

This module defines constants used across analysis and figure generation scripts.
Follows the pattern established in continuum_swr_replay.parameters.
"""

import numpy as np

# Event detection thresholds
DEFAULT_NON_LOCAL_THRESHOLD = 0.8
DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC = 4.0
DEFAULT_SAMPLING_FREQUENCY_HZ = 500.0

# Histogram bin definitions (shared across figures)
BINS_DURATION_MS = np.arange(0, 250, 10)
BINS_PROBABILITY = np.arange(0, 1.05, 0.05)
BINS_DISTANCE_CM = np.arange(0, 230, 2)
BINS_HPD_CM = np.arange(0, 250, 10)
BINS_SPEED_CM_PER_SEC = np.arange(0, 20, 0.5)

# Event-triggered analysis
DEFAULT_WINDOW_EVENT_TRIGGERED_SEC = 0.25

# Note: Colors are already defined in continuum_swr_replay.parameters.COLORS
```

**Rationale:** Simpler than dataclass, matches existing codebase patterns, directly importable.

---

## Phase 4: Testing (Integrated with Phase 1)

**CRITICAL CHANGE:** Testing is moved from Phase 4 to Phase 1. Each new module must have tests before moving to Phase 2.

**Why:** Waiting until Week 4 to write tests means 3 weeks of potentially buggy code. Testing must be concurrent with implementation.

## Phase Dependencies and Critical Path

### Sequential Dependencies (Cannot Parallelize)

```
Phase 1.1 (event_analysis.py + tests)
  ↓
Phase 2.2 (refactor generate_figure02.py - uses event_analysis)

Phase 1.2 (triggered_analysis.py + tests)
  ↓
Phase 2.1 (refactor generate_figure03.py - uses triggered_analysis)

Phase 1.3 (visualization/event_plots.py + tests)
  ↓
Phase 2.3 (refactor generate_figure04.py - uses event_plots)
```

### Parallelizable Work

- Phase 3 (CLI standardization) can proceed independently for any script
- Phase 1.3 and Phase 2.3 can proceed while Phase 2.1/2.2 are in progress
- Testing (now in Phase 1) happens concurrently with implementation

### Fallback Strategy

If any Phase 1.x module takes longer than expected:
1. Phase 2.x scripts can temporarily use existing functions (`get_confident_times()`, etc.)
2. Update imports later when new utilities are ready
3. Don't block figure generation on utility extraction

### Critical Path Timeline

**Realistic estimate: 5-6 weeks**

- **Weeks 1-2:** Phase 1 (implementation + tests for all modules)
- **Weeks 3-4:** Phase 2 (refactor scripts sequentially, verify outputs)
- **Week 5:** Phase 3 (CLI + configuration - can overlap with Phase 2)
- **Week 6:** Buffer for issues, documentation updates

---

## Phase 5: Documentation Updates (Week 5-6)

### 5.1: Add Docstrings

All new functions must have (checked during Phase 1):

- One-line summary
- Parameters section with types
- Returns section with types
- Examples section
- Notes section (if needed)

Use existing `analysis/hippocampus/` docstrings as templates.

### 5.2: Type Hints

All functions must have complete type hints (enforced in Phase 1):

```python
def extract_signal_windows(
    data: pd.DataFrame,
    event_times: pd.DataFrame,
    signal_cols: list[str],
    time_col: str = "start_time",
    window_sec: float = 2.0,
    sampling_frequency: float = 500.0,
    event_col: str | None = None,
) -> xr.DataArray:
    ...
```

### 4.3: Add Unit Tests

**Create:** `tests/analysis/test_event_utils.py`

```python
"""Tests for event analysis utilities."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_utils import (
    compute_event_durations,
    filter_events_by_speed,
)


def test_compute_event_durations():
    """Test event duration calculation."""
    # Create mock data
    data = pd.DataFrame({
        "animal": ["mouse1"] * 10,
        "day": [20200101] * 10,
        "epoch": ["01_r1"] * 10,
        "event": [0, 0, 1, 1, 1, 0, 2, 2, 2, 2],
    })
    data.index = np.arange(10) / 500.0  # Time in seconds

    durations = compute_event_durations(
        data, "event", sampling_frequency=500.0
    )

    # Event 1: 3 samples = 6 ms
    # Event 2: 4 samples = 8 ms
    assert len(durations) == 2
    assert durations.iloc[0] == pytest.approx(6.0)
    assert durations.iloc[1] == pytest.approx(8.0)
```

---

## Phase 5: Update Documentation (Week 5)

### 5.1: Update CLAUDE.md

Add section on figure generation:

```markdown
## Figure Generation

Figure scripts are in `scripts/` and use library functions from `src/`.

**Generating all figures:**
```bash
python scripts/generate_all_figures.py --output-dir paper/figures --format pdf
```

**Generating individual figures:**

```bash
python scripts/generate_figure03.py --output-dir paper/figures
```

**Figure script structure:**

- Import analysis functions from `src/continuum_swr_replay/analysis/`
- Import plotting functions from `src/continuum_swr_replay/visualization/`
- Keep scripts under 300 lines
- One script per figure or closely related panels
- Support CLI arguments for output directory and format

```

### 5.2: Add Examples to Module Docstrings

Each new module should have examples in its docstring showing common usage patterns.

---

## Migration Checklist

### For Each Function Being Extracted:

- [ ] Check it doesn't duplicate existing functionality
- [ ] Add complete docstring (summary, parameters, returns, examples)
- [ ] Add type hints
- [ ] Add unit test
- [ ] Update script to use new function
- [ ] Verify output is identical
- [ ] Run linting: `uv run ruff check src/`
- [ ] Run type checking: `uv run mypy src/`
- [ ] Run tests: `uv run pytest tests/`

### For Each Script Being Refactored:

- [ ] Identify reusable functions → extract to src/
- [ ] Add CLI arguments
- [ ] Add module docstring
- [ ] Reduce to <300 lines
- [ ] Verify figures are identical to baseline
- [ ] Update `generate_all_figures.py` if needed

---

## Success Criteria

### Phase 1 Complete:
- ✅ New modules in `src/` with tests
- ✅ No functionality duplication
- ✅ All tests pass
- ✅ Ruff and mypy pass

### Phase 2 Complete:
- ✅ All figure scripts under 300 lines
- ✅ Scripts use library functions
- ✅ Figures identical to baseline
- ✅ No repetitive code blocks

### Phase 3 Complete:
- ✅ All scripts have CLI
- ✅ Configuration system in place
- ✅ Consistent interface across scripts

### Phase 4 Complete:
- ✅ All functions documented
- ✅ All functions type-hinted
- ✅ Unit tests with >80% coverage

### Phase 5 Complete:
- ✅ CLAUDE.md updated
- ✅ Examples in all docstrings
- ✅ README updated if needed

---

## Non-Goals

**Do NOT:**
- Change existing well-functioning code in `analysis/hippocampus/`
- Modify `run_decoding_bandit.py` structure (already good)
- Add functionality not used by current scripts
- Change data formats or file naming conventions
- Refactor `test.ipynb` (it's a notebook, keep as-is)
- Extract `plot_venn_diagram_overlap()` - too specific, keep in script if needed

---

## Summary of Revisions Based on Code Review

### Critical Issues Addressed

1. **Removed Duplicate Functions**
   - ❌ Deleted `label_continuous_events()` - duplicates `get_confident_times()`
   - ❌ Deleted `compute_event_durations()` - durations already in `get_confident_times()`
   - ✅ Replaced with simple converter: `get_event_durations_ms()`
   - ⚠️ Removed vague `compute_event_timing_differences()` - overlaps with `get_overlap_times()`

2. **Fixed Type Signatures and Validation**
   - All `group_cols` parameters now non-optional with sensible defaults
   - Added input validation for all numeric parameters (sampling frequency, window size, etc.)
   - Added validation for required DataFrame columns
   - Changed `event_col` → `event_id_col` in `get_signal_windows()` for clarity

3. **Fixed Scientific Correctness**
   - Added bounds checking with `np.clip()` in `get_signal_windows()`
   - Added warnings for events outside data time range
   - Clarified speed filtering logic with explicit `method` parameter
   - Added explicit empty data handling with logging

4. **Reorganized Modules**
   - `event_utils.py` → `analysis/hippocampus/event_analysis.py` (co-located with `event_detection.py`)
   - `time_series_utils.py` → `analysis/hippocampus/triggered_analysis.py` (clearer scope)
   - `dataset_utils.py` → Will be in `data_loaders/processed_data.py` (matches existing patterns)
   - Function naming: Use consistent `get_` prefix to match existing codebase

### Architectural Improvements

5. **Added Missing Utilities**
   - ✅ `get_event_start_end_times()` - used in multiple scripts, was missing
   - ✅ Simplified `get_immobile_events()` - combines filtering with proper validation

6. **Testing Strategy**
   - Moved testing from Phase 4 to Phase 1 (concurrent with implementation)
   - Added explicit regression testing requirements
   - Each module must have >80% test coverage before moving to Phase 2

7. **Simplified Configuration**
   - Replaced `PlotConfig` dataclass with module-level constants in `analysis/constants.py`
   - Follows existing pattern in `parameters.py`
   - Simpler imports: `from continuum_swr_replay.analysis.constants import BINS_DURATION_MS`

### Timeline and Dependencies

8. **Updated Timeline**
   - Original: 5 weeks (too aggressive)
   - Revised: 5-6 weeks (realistic with buffer)
   - Added explicit phase dependencies and critical path
   - Defined fallback strategies if phases take longer

9. **Reduced Scope**
   - Removed `plot_venn_diagram_overlap()` - too specific, not reusable
   - Skipped overly narrow functions that aren't used across multiple scripts
   - Focused on truly reusable utilities

### Key Takeaways

**What Changed:**
- Fewer functions (removed duplicates)
- Better organized (hippocampus submodule)
- More robust (validation, error handling)
- Testable from day 1 (integrated testing)
- Realistic timeline (5-6 weeks vs 5)

**What Stayed:**
- Core goal: Extract reusable code from scripts
- Incremental approach: One phase at a time
- Conventions: Follow CLAUDE.md patterns
- Existing code: Don't modify working analysis modules

**Success Criteria Remain:**
- All figure scripts under 300 lines
- No duplicate code patterns
- All tests passing with >80% coverage
- Figures identical to baseline
- Documentation complete

---

## References

- **CLAUDE.md**: Project conventions and architecture
- **Existing modules**: `src/continuum_swr_replay/analysis/hippocampus/` for patterns
- **Scientific code best practices**: PEP 8, PEP 484 (type hints), NumPy docstring style
- **Code Review**: See review report for detailed analysis of issues and recommendations
