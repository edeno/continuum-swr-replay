# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuroscience research project analyzing continuum replay during Sharp Wave Ripples (SWR) in a spatial bandit task. The project processes and analyzes neural recordings from multiple brain regions:

- **HPC** (Hippocampus CA1): Uses clusterless decoding with spike waveform features
- **mPFC** (Medial Prefrontal Cortex): Uses sorted units
- **OFC** (Orbitofrontal Cortex): Uses sorted units

Data is stored in NWB (Neurodata Without Borders) format and managed through the Spyglass neuroscience data pipeline.

## Development Setup

This project uses **UV** for package management (not pip/conda):

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Install in development mode
uv pip install -e .

# Install with dev dependencies
uv sync --extra dev
```

### Code Quality Tools

When dev dependencies are installed:

- **ruff** - Fast Python linter and formatter (combines black, flake8, and more)
- **mypy** - Type checking
- **pytest** - Testing (testpaths: `tests/`)

Run these tools directly:

```bash
# Lint and check code
uv run ruff check src/

# Format code (ruff also handles formatting)
uv run ruff format src/

# Type checking
uv run mypy src/

# Run tests
uv run pytest
```

## Project Structure

```
src/continuum_swr_replay/
├── data_loaders/          # Data loading from Spyglass/NWB
│   └── load_data_bandit.py   # Main data loader for spatial bandit task
├── analysis/              # Analysis modules
│   ├── hippocampus/      # Hippocampus-specific analysis
│   └── pfc/              # Prefrontal cortex analysis
└── visualization/         # Visualization utilities

scripts/                   # Analysis scripts
paper/                     # Paper materials (data, figures, manuscript)
```

## Key Architecture Concepts

### Data Loading Pipeline (`load_data_bandit.py`)

The main entry point is `load_data()` which orchestrates loading all data types for a given session and epoch:

```python
load_data(nwb_file_name: str, epoch_name: str, ripple_detector_name: str = "Kay")
```

**Data flow:**

1. **Position data** - Loads DLC-tracked position, linearizes to track graph, adds task-relevant spatial variables
2. **Spike data** - Loads clusterless marks for HPC and sorted units for mPFC/OFC, filters coincident spikes
3. **LFP data** - Loads ripple-band filtered LFP, theta phase/power, computes ripple consensus trace
4. **Task events** - Loads DIO events (beam breaks, pump/reward, lights), constructs trials and well visits
5. **Task variables** - Computes derived variables (time/distance to goal, path progress, reward omissions, patch changes)

Returns a dictionary containing all data types as pandas DataFrames or lists, indexed by time.

### Spatial Bandit Task Structure

The task uses a multi-well track environment with:

- **Wells** (nodes 0-5) grouped into 3 **patches** (1-3)
- **Track segments** (edges) connecting wells, represented as a NetworkX graph
- **Trials**: Runs from one well to another, may include reward at destination
- **Well visits**: Poke-in/poke-out intervals at wells
- **Patch changes**: Trials that move between different patches

Key mappings:

- `WELL_PATCH_MAPPING`: Maps well IDs to patch IDs
- `TRACK_SEGMENT_TO_PATCH`: Maps track segment IDs to patch IDs
- `TRACK_SEGMENT_TO_EDGE_ID`: Maps track segments to graph edge IDs

### Data Integration with Spyglass

This project relies on the Spyglass DataJoint pipeline. Key tables used:

- `Session`, `Nwbfile` - Session metadata
- `PositionOutput`, `IntervalPositionInfo` - Position tracking data
- `TrackGraph`, `IntervalLinearizedPosition` - Spatial linearization
- `CuratedSpikeSorting`, `UnitMarks` - Spike data
- `LFPBandV1`, `LFPOutput` - LFP signals
- `RippleTimesV1` - Detected ripple events
- `DIOEvents` - Digital I/O events (beam breaks, pumps)

When modifying data loading, respect the DataJoint query patterns and restriction syntax.

### Spike Data Processing

- **HPC**: Uses clusterless thresholding with waveform feature extraction. Automatically detects and removes coincident spikes across tetrodes (likely artifacts).
- **mPFC/OFC**: Uses sorted units from MountainSort4. Filters by electrode group location based on targeted_location metadata.

Spike times are organized as lists of arrays: `spike_times[brain_area][sort_group_index]`

### Spatial Variables

Position is tracked in multiple coordinate systems:

- **Raw position**: `head_position_x`, `head_position_y` (2D Cartesian)
- **Projected position**: `projected_x_position`, `projected_y_position` (projected onto track graph)
- **Linear position**: `linear_position` (1D along track)
- **Track segment ID**: Which edge of the graph the animal is on

Task-relevant derived variables:

- `path_progress`: 0-1 normalized progress along current trial path
- `dist_to_goal`: Distance remaining to goal well
- `time_to_goal`: Time elapsed since trial start
- `turn_direction`: Sequence of left/right turns in trial (e.g., "left-right-left")

## Common Data Analysis Patterns

### Loading data for a session

```python
from continuum_swr_replay.data_loaders.load_data_bandit import load_data, get_epoch_info

# Get all available epochs
epoch_info = get_epoch_info()

# Load data for specific session
data = load_data(
    nwb_file_name="chimi20200212_.nwb",
    epoch_name="02_r1",
    ripple_detector_name="Kay"
)

# Access different data types
position_info = data["position_info"]  # pd.DataFrame, time-indexed
spike_times = data["spike_times"]  # dict[brain_area] -> list[np.ndarray]
ripple_times = data["ripple_times"]  # pd.DataFrame with start/end times
trials = data["trials"]  # pd.DataFrame of trial metadata
task_variables = data["task_variables"]  # pd.DataFrame, time-indexed
```

### Working with spike data

```python
# HPC spikes include waveform features
hpc_spike_times = data["spike_times"]["HPC"]
hpc_waveform_features = data["spike_waveform_features"]["HPC"]

# Iterate over tetrodes/sort groups
for tetrode_idx, (times, features) in enumerate(zip(hpc_spike_times, hpc_waveform_features)):
    # times: np.ndarray of spike times in seconds
    # features: np.ndarray of shape (n_spikes, n_features)
    pass

# PFC spikes (sorted units)
if "mPFC" in data["spike_times"]:
    mpfc_spike_times = data["spike_times"]["mPFC"]
    # List of arrays, one per unit
```

### Time alignment

All data is time-aligned and can be indexed/sliced consistently:

```python
# All time-indexed DataFrames use seconds as the index
trial = data["trials"].iloc[0]
trial_position = data["position_info"].loc[trial.start_time:trial.end_time]
trial_task_vars = data["task_variables"].loc[trial.start_time:trial.end_time]
```

## Dependencies and JAX

This project uses **JAX** for machine learning and requires CUDA 12 for GPU acceleration:

- JAX is installed with CUDA 12 support: `jax[cuda12]`
- Consider using GPU-aware code when implementing new analyses
- Dask is available for parallel processing across CPU cores
- Optional: `dask-cuda` for GPU-based Dask operations

## Animal and Session Information

The dataset includes recordings from 5 animals across multiple sessions:

- **chimi**: 29 sessions (Feb-Mar 2020)
- **j16**: 16 sessions (Jul 2021)
- **peanut**: 16 sessions (Nov-Dec 2020)
- **senor**: 26 sessions (Oct-Nov 2020)
- **wilbur**: 22 sessions (Mar-Apr 2021)

All sessions use the spatial bandit task configuration. See `NWB_FILES` in `load_data_bandit.py` for the complete list.

## Running Decoding Analysis

The main analysis workflow is in `scripts/run_decoding_bandit.py`, which performs continuum decoding using non-local detection algorithms.

### Decoding Workflow

1. **Load data**: Use `load_data()` to get all session data
2. **Set up environment**: Create `Environment` with place bins and track graph
3. **Select training data**: Exclude ripple times or use all data (via `get_training_timepoints()`)
4. **Perform decoding**: Use detector-specific methods:
   - `NonLocalClusterlessDetector` for HPC (uses waveform features)
   - `NonLocalSortedSpikesDetector` for mPFC/OFC (uses sorted spikes)
5. **Organize results**: Compute derived measures (ahead/behind distance, HPD)
6. **Save outputs**: Results as NetCDF (.nc), models as pickle (.pkl)
7. **Generate figurl**: Interactive visualization URL (optional)

### Key Decoding Parameters

```python
# Place binning
place_bin_size = 2.0  # cm for 1D, 3.5 cm for 2D

# Decoding parameters (HPC clusterless)
position_std = np.sqrt(12.5)  # ~3.5 cm
waveform_std = 24.0
block_size = 1024  # for GPU processing

# Training options
training_type = "no_ripple"  # or "all"
non_stationary = False  # if True, uses speed as transition covariate
non_local_threshold = 0.8  # threshold for classifying non-local events
```

### Command-Line Usage

```bash
# Decode all sessions for an animal
uv run python scripts/run_decoding_bandit.py --animal chimi --brain_area HPC

# Decode specific session
uv run python scripts/run_decoding_bandit.py --animal chimi --date 20200212 --epoch 02_r1

# With non-stationary model and figurl
uv run python scripts/run_decoding_bandit.py --animal chimi --non_stationary --create_figurl

# Overwrite existing results
uv run python scripts/run_decoding_bandit.py --animal chimi --overwrite
```

### GPU and Distributed Computing

The script uses Dask with CUDA for GPU-accelerated parallel processing:

```python
# Automatically sets up LocalCUDACluster
# Configure visible GPUs via command line
--cuda_visible_devices "0,1,2,3"  # Use specific GPUs

# Set XLA memory fraction (done automatically in script)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
```

Multiple sessions are processed in parallel using `dask.delayed` and `dask.compute`.

### Output Files

Results are saved to `PROCESSED_DATA_DIR` (defined in `src.parameters`):

- `{epoch_identifier}_results.nc` - Decoding results (xarray Dataset)
- `{epoch_identifier}_detector.pkl` - Fitted detector model
- `{epoch_identifier}_environment.pkl` - Environment configuration
- `{epoch_identifier}_{threshold}_data_over_time.pkl` - Derived time series data

Logs are written to `logs/{epoch_identifier}.log` with git hash for reproducibility.

## Important Conventions

1. **Time units**: Always in seconds (never milliseconds)
2. **Epoch naming**: Format is `"{epoch_number}_r{replicate}"` (e.g., "02_r1")
3. **Epoch identifiers**: `{animal}{date}_{epoch}_{brain_area}` (e.g., "chimi20200212_02_r1_HPC")
4. **Brain area names**: Use "HPC", "mPFC", "OFC" (exact capitalization matters)
5. **Interval naming**: Spyglass uses " noPrePostTrialTimes" suffix for trimmed intervals
6. **Track graph**: NetworkX graph where nodes are wells (0-5) and edges are track segments
7. **Index/time alignment**: Use `.loc[start:end]` for time-based slicing, not `.iloc`

## Notes on Data Quality

- Not all sessions have recordings from all brain areas (check keys in `spike_times` dict)
- Coincident spike detection removes likely artifacts (>33% of tetrodes firing within 40μs)
- Position data uses DLC tracking when available, falls back to older tracking methods
- Some sessions may lack certain interval types (wrapped in try/except in data loading)
- Check for empty spike data before running decoding (`if not data["spike_times"]`)

## Known Issues and Fixes

### Import Path Updates (Fixed ✅)
The import paths in the codebase have been updated from the old `from src.X` pattern to the correct package structure `from continuum_swr_replay.X`. All modules now use proper imports:

- ✅ `continuum_swr_replay.data_loaders.load_data_bandit`
- ✅ `continuum_swr_replay.analysis.hippocampus.analysis_hpc_bandit`
- ✅ `continuum_swr_replay.visualization.visualization`
- ✅ `continuum_swr_replay.parameters`

### Critical Correctness Issues (Fixed ✅)

Five critical correctness issues that could cause crashes or incorrect results have been fixed:

1. **Division by zero in z-score calculation** ([load_data_bandit.py:679-687](src/continuum_swr_replay/data_loaders/load_data_bandit.py#L679-L687))
   - Now checks if std is 0 or NaN before division
   - Returns zeros if firing rate is constant

2. **Division by zero in path progress** ([load_data_bandit.py:1021-1025](src/continuum_swr_replay/data_loaders/load_data_bandit.py#L1021-L1025))
   - Validates max distance before normalization
   - Returns zeros if start and end nodes are identical

3. **Empty array handling** ([load_data_bandit.py:446-448](src/continuum_swr_replay/data_loaders/load_data_bandit.py#L446-L448))
   - `_detect_coincident_spikes()` now checks for empty spike times
   - Returns empty lists safely instead of crashing

4. **Unsafe dictionary access** ([run_decoding_bandit.py:326-328](scripts/run_decoding_bandit.py#L326-L328))
   - Validates brain area exists in spike data before access
   - Logs warning and skips gracefully if data missing

5. **Assertion replaced with proper error handling** ([load_data_bandit.py:1162-1168](src/continuum_swr_replay/data_loaders/load_data_bandit.py#L1162-L1168))
   - Replaced `assert` with explicit `ValueError` with detailed message
   - Ensures data validation works even with Python optimization flags

### External Dependencies
- **kachery packages** (`kachery`, `kachery-client`, `kachery-cloud>=0.4.0`) are now included in dependencies for Spyglass database access and figurl visualization support.
