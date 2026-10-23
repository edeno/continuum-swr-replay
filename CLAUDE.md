# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuroscience research project analyzing continuum replay during Sharp Wave Ripples (SWR) in a spatial bandit task. The project processes and analyzes neural recordings from multiple brain regions:

- **HPC** (Hippocampus CA1): Uses clusterless decoding with spike waveform features
- **mPFC** (Medial Prefrontal Cortex): Uses sorted units
- **OFC** (Orbitofrontal Cortex): Uses sorted units

Data is stored in NWB (Neurodata Without Borders) format and managed through the Spyglass neuroscience data pipeline.

For a detailed description of the behavioral task, training protocol,
implant, and recording setup, see
[docs/task_description.md](docs/task_description.md).

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
├── data_loaders/          # Data loading from Spyglass/NWB (load_data in bandit_task.py)
├── analysis/              # Analysis modules
│   ├── behavior/          #   behavioral-model covariates (qvals_table)
│   ├── contfrag/          #   Continuous-Fragmented classifier analysis
│   ├── events/            #   analysed-time rule, SWR source, event eligibility, classified event table, cohorts
│   ├── hippocampus/       #   decode masks and aggregation, posterior/quality metrics, SWR inventories, artifact detector
│   ├── inventory_comparison/ # event inventories from different detectors compared
│   ├── pfc/               #   mPFC/OFC unit analysis
│   ├── replay/            #   replay content/rate tables, model ladders, fitters
│   └── constants.py       #   analysis constants and defaults
├── decoding/              # decode configs, transitions, file names, the results.nc reader
├── runs/                  # run infrastructure: provenance, CLI, batch loops
├── tables/                # canonical tables: schema, row ids, parquet I/O, atomic writes, parity
├── visualization/         # Plotting utilities; palettes.py holds COLORS
├── figure_utilities.py    # publication-figure helpers
├── intervals.py           # interval algebra: overlap conventions, masks, spike binning
├── parameters.py          # global parameters, PROCESSED_DATA_DIR
├── paths.py               # output locations, canonical table names, table_path()
└── data_types.py          # shared type definitions

scripts/                   # Analysis scripts, grouped by domain — see scripts/README.md
├── decoding/  detection/  tables/  rate/  coupling/  content/  behavior/  figures/  qa/  pfc/
└── exploratory/           #   unsupported one-offs (_*.py)
paper/figures/<topic>/     # Publication figures in topic subdirs (no loose files)
docs/                      # Published reference — see docs/README.md for the doc-type→home scheme
├── design/               #   scientific/methodological design specs (human-discoverable)
└── known_issues/
.claude/docs/              # Agent-process docs (not scientific record)
├── plans/                #   phased implementation plans
└── reviews/              #   code/methods review reports and point-in-time audits
.claude/research/          # Exploratory scratch analyses + *_outputs/ figures
tests/                     # pytest suite — see tests/README.md
```

> Documentation homes are indexed in [docs/README.md](docs/README.md): published/scientific
> docs live in `docs/` (incl. design specs in `docs/design/`); agent-process/scratch under `.claude/`.
> The import layering rules and the 2026-10 old→new module map are in [docs/README.md](docs/README.md) ("Code layout").
> **All model ladders (rate, content, is_swr-contrast) are defined in one place —
> `src/continuum_swr_replay/analysis/replay/model_specs.py`** (the single source of truth;
> `docs/models_table.md` mirrors it, its rung sets enforced by
> `tests/unit/test_models_table_mirrors_specs.py` — the formulas themselves are not
> pinned by that test). Where fitted outputs / figures land is in
> [docs/README.md](docs/README.md) ("Where results live" / "Figure outputs").

## Key Architecture Concepts

### Data Loading Pipeline (`bandit_task.py`)

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

The task uses a "three-legged spider" track with 6 wells organized into 3
patches. The track graph has:

- **10 nodes**: 6 wells (nodes 0-5, all degree-1 terminals) + 3 patch
  junctions (nodes 6, 7, 8) + 1 center junction (node 9)
- **9 edges (arms)**:
  - **Leaf arms** (6 total): each connects one well to its patch junction
    - arm 0: well 0 ↔ patch junction 6
    - arm 1: well 1 ↔ patch junction 6
    - arm 2: well 2 ↔ patch junction 7
    - arm 3: well 3 ↔ patch junction 7
    - arm 4: well 4 ↔ patch junction 8
    - arm 5: well 5 ↔ patch junction 8
  - **Trunk arms** (3 total): each connects a patch junction to the center
    - arm 6: patch junction 6 ↔ center junction 9
    - arm 7: patch junction 7 ↔ center junction 9
    - arm 8: patch junction 8 ↔ center junction 9

**IMPORTANT consequence**: each well is degree-1, so it touches exactly ONE
arm. `well_to_arm_set(well_i)` returns `{i}` for i in 0..5. Predictors
derived from wells (`is_prev`, `is_next`, `is_last_rewarded`) mark exactly
one of the nine arms per event — not two. The animal's `track_segment_id`
when stopped at a well also equals that same single arm, so `is_current`
collapses with `well_to_arm_set(prev_trial_to_well)`.

Task-level primitives:

- **Wells** (nodes 0-5) grouped into 3 **patches** (1-3)
- **Trials**: Runs from one well to another, may include reward at destination
- **Well visits**: Poke-in/poke-out intervals at wells
- **Patch changes**: Trials that move between different patches

Key mappings:

- `WELL_PATCH_MAPPING` (wells → patches): `{0:1, 1:1, 2:2, 3:2, 4:3, 5:3}`
- `TRACK_SEGMENT_TO_PATCH` (arms → patches): leaf arms match their well's
  patch; trunk arms carry the patch label of the leaves they join.
  `{0:1, 1:1, 6:1, 2:2, 3:2, 7:2, 4:3, 5:3, 8:3}`
- `TRACK_SEGMENT_TO_EDGE_ID`: Maps track segments to graph edge IDs

**Junction place bins**: The non_local_detector creates placeholder place
bins at track-graph junctions with `edge_id = -1` (interpolation artifacts
from `edge_spacing > 0`, not real spatial positions). Any MAP-to-arm or
posterior-to-arm computation MUST exclude these bins before argmax or
normalization — otherwise the MAP can silently land on a junction bin and
return NaN. See `map_estimate_to_arm` in
`analysis/replay/spatial_mapping.py` for the canonical
nearest-neighbor-over-real-bins pattern.

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

### Canonical per-event classification table

The single source of truth for a non-local event's timing and its
SWR-overlap classification is
`analysis/events/classified_table.py:build_classified_nonlocal_event_table`.
It produces one row per non-local event over the *full* event-label
window: the label's probability contract is **validated, not re-filtered**
(a violation raises `EventDefinitionError` rather than silently dropping
bins), so no non-local event is clipped. From that window it derives
timing, per-event speed summaries, peak non-local probability, the 3-way
`overlap_class` (and 6-way `overlap_subclass`), and the matched SWR.

**One analysed-time rule.** *Excluded time* is the artifact mask
(`analysis/events/artifact_masks.py`) widened by 0.3 s on each side
(`analysis/events/analysed_time.py:excluded_intervals`; the rule is in
`analysis/events/eligibility.py`). Nothing inside it is analysed.
An event of any kind (non-local, SWR, MUA, ContFrag label, Widloski), an
exposure second, a time bin with its history window, and a peri-event window
are used only when they lie wholly outside excluded time. An event must also
be immobile as a whole: every sampled speed finite and below 4 cm/s
(`analysis/events/immobility.py`). A kept event that overlaps a dropped SWR or MUA
event is dropped too, because its overlap class is unknown.

The rule is applied once, where a decode frame is read
(`analysis/events/analysed_time.py:with_swr_source`), so every table
inherits it. Each dropped event keeps a reason
(`analysis/events/analysed_time.py:artifact_event_drops`), and kept events
keep their labels and `event_id`. A stop's `immobility_duration_s` is its
span less the excluded seconds in it, so it is no longer
`stop_end - stop_start`.

Inputs use the mask itself, not excluded time: the decoders' missing bins,
the SWR detectors' normalization, and the ripple-power covariate
(`SWR_zscore`, the Kay consensus over the tetrodes rule R-T keeps). With no
artifact run configured, the rule changes nothing.

Event-level `is_swr` means strict positive-duration overlap, not the old
inclusive `any SWR_event > 0`. (The stop-level `is_swr_stop` flag in
`build_stop_table` deliberately still uses the inclusive per-timebin rule —
the two flags do not share a convention.) Because of that dual convention
the stop-level `n_SWR`/`n_MUA` totals still differ slightly from the
bin/onset counts, so the count must **not** be cross-substituted between
the stop-level and bin-level tables — see
[docs/known_issues/swr_mua_count_conventions.md](docs/known_issues/swr_mua_count_conventions.md).
The previously recorded Δ82/Δ250 magnitudes predate the unified speed
gating and **must be re-derived** before being cited.

The replay-content `build_event_table` in
`analysis/replay/tables_event.py` builds on
top of this. The classified columns are merged in on `event_id`; the
classified pass's timing columns arrive renamed `label_event_start` /
`_end` / `_duration` / `_midpoint_time`, are checked bit-for-bit against
the builder pass's canonical `event_start_s` / `event_end_s` /
`event_duration_s` / `event_midpoint_s`, and are then dropped — the shipped
table carries only the `event_*_s` names. `label_event_*` survives solely
in `covariate_registry.py`, as a legacy-spelling alias resolving to those
same canonical columns.

Changing the eligible cohort renumbers non-local labels, so `event_id`
hashes shift. `event_table`, `event_arm_table`, `event_sequence_table`,
`stop_table`, and the qvals/stop joins must therefore be regenerated as
one atomic set — a partial regeneration mis-joins silently on `event_id`.
The artifact rule drops events without renumbering the rest, so kept
`event_id`s do not shift; the atomic set is still regenerated together,
because dropped events leave every table.

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

### Replay event-rate models (two families)

Two parallel families estimate replay event *rate* (how often events occur):
**Family A** (`MODEL_PER_STOP_RATE`, per-stop Poisson with a `log(immobility)`
offset — the literature-comparable layer) and **Family B**
(`MODEL_PER_BIN_RATE`, the per-bin conditional-intensity ladder with a spline
time course + reward kernel + self-history — the confound-free layer). See
[docs/rate_models.md](docs/rate_models.md) for which family to cite for what,
the M0–M9 ladder, and the point-process model-checking. Runners:
`scripts/rate/fit_per_stop_rate_models.py`, `scripts/rate/fit_per_bin_rate_models.py`;
plotter: `scripts/rate/plot_per_bin_rate_models.py`.

### Loading data for a session

```python
from continuum_swr_replay.data_loaders import load_data, get_epoch_info

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

This project uses **JAX** for machine learning. The base install is
**CPU-only** so the package builds on every platform (including
macOS arm64, where `jax-cuda12-plugin` has no wheel).

- Default: `pip install` / `uv sync` resolves plain `jax`.
- GPU: install the `cuda` extra to pull in `jax[cuda12]` —
  ``uv sync --extra cuda``. Linux-only.
- The legacy `gpu` extra also bundles `dask-cuda` and
  `cuda-bindings` for GPU-aware Dask: ``uv sync --extra gpu``.
- Consider using GPU-aware code when implementing new analyses.
- Dask is available for parallel processing across CPU cores.

## Animal and Session Information

The dataset includes recordings from 5 animals. Each NWB file can contain multiple epochs (e.g., `02_r1`, `04_r2`, `06_r3`). The cohort is the epoch register's analysed epochs (`data_loaders/epoch_register.csv`, read by `analysed_epochs()`); `docs/epoch_register.md` gives each excluded epoch's reason. After the R11 swap: 242 epochs in 45 NWB files.

| Animal | NWB Files | Epochs | Dates |
|--------|-----------|--------|-------|
| chimi | 14 | 46 | Feb-Mar 2020 |
| j16 | 8 | 58 | Jul 2021 |
| peanut | 6 | 44 | Nov-Dec 2020 |
| senor | 10 | 49 | Oct-Nov 2020 |
| wilbur | 7 | 45 | Mar-Apr 2021 |

See `NWB_FILES` in `data_loaders/constants.py` for the complete file list.

## Running Decoding Analysis

The main analysis workflow is in `scripts/decoding/run_decoding_bandit.py`, which performs continuum decoding using non-local detection algorithms.

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
uv run python scripts/decoding/run_decoding_bandit.py --animal chimi --brain_area HPC

# Decode specific session
uv run python scripts/decoding/run_decoding_bandit.py --animal chimi --date 20200212 --epoch 02_r1

# With non-stationary model and figurl
uv run python scripts/decoding/run_decoding_bandit.py --animal chimi --non_stationary --create_figurl

# Overwrite existing results
uv run python scripts/decoding/run_decoding_bandit.py --animal chimi --overwrite
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

Results are saved to `PROCESSED_DATA_DIR` (defined in `continuum_swr_replay.parameters`):

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
6. **Track graph**: NetworkX graph of 10 nodes, where nodes 0–5 are wells (6–8 patch junctions, 9 the center) and edges are track segments
7. **Index/time alignment**: Use `.loc[start:end]` for time-based slicing, not `.iloc`

## Notes on Data Quality

- Not all sessions have recordings from all brain areas (check keys in `spike_times` dict)
- Coincident spike detection removes likely artifacts (>33% of tetrodes firing within 40μs)
- Position data uses DLC tracking when available, falls back to older tracking methods
- Some sessions may lack certain interval types (wrapped in try/except in data loading)
- Check for empty spike data before running decoding (`if not data["spike_times"]`)
- Artifacts: one detector (`analysis/hippocampus/artifacts/`, run `analysis/events/swr_source.py:ARTIFACT_INTERVALS_RUN_ID`) masks sustained ripple-band noise (S blocks), step artifacts and synchronous spike lines; see `docs/design/2026-10-09-artifact-detection.md`. An epoch with an S block is left out whole (epoch register reason `SUSTAINED_NOISE`).
