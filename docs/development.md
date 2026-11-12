# Development guide

How this repository is organized and the rules for working in it: the task and
data model, the code layout, the analysis conventions, and how reported numbers
are produced. `CLAUDE.md` imports this file, so the assistant works from the same
rules; edit them here.

## Project Overview

This is a neuroscience research project analyzing continuum replay during Sharp Wave Ripples (SWR) in a spatial bandit task. The project processes and analyzes neural recordings from multiple brain regions:

- **HPC** (Hippocampus CA1): Uses clusterless decoding with spike waveform features
- **mPFC** (Medial Prefrontal Cortex): Uses sorted units
- **OFC** (Orbitofrontal Cortex): Uses sorted units

Data is stored in NWB (Neurodata Without Borders) format and managed through the Spyglass neuroscience data pipeline.

For a detailed description of the behavioral task, training protocol,
implant, and recording setup, see
[docs/task_description.md](task_description.md).

### Replay terminology

**Replay includes stationary representations as well as trajectories.** The project
lead explicitly clarified this on 2026-11-12. Stationary refers to the represented
location remaining stable, not to the animal's movement. Do not restrict replay
to moving or sequential representations. Use "trajectory replay" or "sequential
replay" for a narrower analysis or comparator that requires a trajectory. Validate
spatial content in stationary events as well as trajectories; absence of movement
in represented space does not by itself make an event a non-replay control.

## Development Setup

This project uses **uv** for package management (not pip/conda). Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the
repository root:

```bash
make sync-dev   # the locked environment (uv.lock) with the development tools
make check      # what CI runs: format check, lint, the full test suite
```

`make help` lists every target. `make sync` installs the analysis environment
without the development tools; the GPU build is `uv sync --frozen --inexact --extra gpu`
(Linux only). Commands run inside the environment with `uv run --frozen`, which
uses the locked versions and does not re-solve them.

### Code Quality Tools

- **ruff** formats and lints `src/`, `scripts/` and `tests/` (`make format`).
  CI fails on unformatted code. Notebooks and the vendored `.claude/skills/`
  are not formatted.
- **pre-commit** runs ruff and the basic file checks on each commit. Install the
  hook once with `uv run pre-commit install`.
- **pytest** runs the suite in `tests/` (`uv run --frozen pytest`). The full
  suite takes about 8 minutes on an Apple-silicon laptop; run a single file
  while working. On macOS, 15 tests of the CPU-pinning runners
  (`test_run_memoryless_detection.py`, `test_run_refits.py`) are skipped because
  `os.sched_getaffinity` exists only on Linux; CI runs on Linux, where they run.
- **mypy** is configured but does not pass yet, so neither CI nor pre-commit
  runs it (see `TODO.md`).

```bash
uv run --frozen pytest tests/scripts/checks/test_epoch_register_doc.py
uv run --frozen ruff check src scripts tests
uv run --frozen mypy src/
```

## Project Structure

```
src/continuum_swr_replay/
├── data_loaders/          # Data loading from Spyglass/NWB (load_data in bandit_task.py)
├── analysis/              # Analysis modules
│   ├── behavior/          #   the events and the task: covariates/, tables/, rate/, content/, choice/, and the shared model ladders, fitters and fit statistics
│   ├── contfrag/          #   Continuous-Fragmented classifier analysis
│   ├── events/            #   analysed-time rule, SWR source, event eligibility, classified event table, cohorts
│   ├── hippocampus/       #   decode masks and aggregation, posterior/quality metrics, event statistics, SWR inventories, artifact detector
│   ├── inventory_comparison/ # event inventories from different detectors compared
│   ├── pfc/               #   mPFC/OFC unit analysis
│   ├── constants.py       #   analysis constants and defaults
│   └── shared_statistics.py #  statistics that know nothing of the task
├── decoding/              # decode configs and detector classes, transitions, identifiers and file names, the results.nc reader
├── runs/                  # run infrastructure: provenance, CLI, batch loops, BLAS thread pin (threads), JAX device choice (device)
├── tables/                # canonical tables: schema, row ids, parquet I/O, atomic writes, parity
├── visualization/         # Plotting utilities; palettes.py holds COLORS, figure_utilities.py the publication-figure helpers
├── intervals.py           # interval algebra: overlap conventions, masks, spike binning
├── reported/              # reported numbers: the summary builders (survey, detection, events, behavior, mpfc), the question list, values, the code record, rounding, the macro emitter, Table S1
├── parameters.py          # global parameters
├── paths.py               # data roots, read at call time from $CSR_PROCESSED_DATA_DIR (processed_data_dir(), ...), canonical table names, table_path()
└── data_types.py          # shared type definitions

scripts/                   # Analysis scripts, grouped by domain — see scripts/README.md
├── decoding/  detection/  tables/  behavior/{rate,content,coupling}/  figures/  workflows/  checks/  pfc/  reported/
└── exploratory/           #   unsupported one-offs (_*.py)
paper/                     # The manuscript — see paper/README.md
├── main.tex  main.pdf     #   LaTeX source and the built PDF (make manuscript)
├── reported/              #   <summary>.json summaries every reported number is read from
│   └── inputs/            #     the tracked input files a summary reads (today the detection sweep summaries)
├── reported_values.tex    #   generated macros, one per reported number (make reported-values)
├── tables/                #   Supplementary Table S1 and the literature survey CSV it is built from
└── figures/<topic>/       #   Publication figures in topic subdirs (no loose files)
Makefile                   # Common tasks: make help
docs/                      # Published reference — see docs/README.md for the doc-type→home scheme
├── design/               #   scientific/methodological design specs (human-discoverable)
├── history/              #   where moved code and data went
└── known_issues/
.claude/docs/              # Agent-process history (plans, reviews, dated audits); a reader's evidence is in docs/
├── plans/                #   phased implementation plans
└── reviews/              #   code/methods review reports and point-in-time audits
.claude/research/          # Exploratory scratch analyses + *_outputs/ figures
tests/                     # pytest suite — see tests/README.md
```

> Documentation homes are indexed in [docs/README.md](README.md): published/scientific
> docs live in `docs/` (incl. design specs in `docs/design/`); agent-process/scratch under `.claude/`.
> The import layering rules are in [docs/README.md](README.md) ("Code layout"); where moved
> modules and scripts went is in [docs/history/2026-module-moves.md](history/2026-module-moves.md).
> **All model ladders (rate, content, is_swr-contrast) are defined in one place —
> `src/continuum_swr_replay/analysis/behavior/model_specs.py`** (the single source of truth;
> `docs/models_table.md` mirrors it, its rung sets enforced by
> `tests/repository/test_models_table_mirrors_specs.py` — the formulas themselves are not
> pinned by that test). Where fitted outputs / figures land is in
> [docs/README.md](README.md) ("Where results live" / "Figure outputs").

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
`analysis/behavior/spatial_mapping.py` for the canonical
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

**The analyses only read the lab's database.** `load_data`, `get_epoch_info` and the
other loaders fetch from Spyglass; nothing in `src/` or `scripts/` inserts, populates,
deletes, creates a schema or exports. Keep it that way. Never write to the lab's
Spyglass database without the explicit approval, from the person who owns the data, of
that specific step; a script that must write is a lab operation, documented on its own,
not part of the analysis. `tests/repository/test_no_database_writes.py` fails when code here
calls a DataJoint write method.

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
[docs/known_issues/swr_mua_count_conventions.md](known_issues/swr_mua_count_conventions.md).
The previously recorded Δ82/Δ250 magnitudes predate the unified speed
gating and **must be re-derived** before being cited.

The replay-content `build_event_table` in
`analysis/behavior/tables/tables_event.py` builds on
top of this. The classified columns are merged in on `event_id`
(`_merge_classified_event_facts`). Event timing is derived once, by the
classified pass: its `event_start` / `event_end` / `event_duration` /
`event_midpoint_time` are renamed to the canonical `event_start_s` /
`event_end_s` / `event_duration_s` / `event_midpoint_s` and merged in; the
builder pass does not recompute them. The shipped table carries only the
`event_*_s` names. The old `label_event_*` spellings survive solely in
`tables/covariate_registry.py:LEGACY_COLUMN_NAMES`, the deny-list that
`assert_canonical_vocabulary` uses to reject them.

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
time course + reward kernel + self-history — the confound-free layer). Family B
is fit on at-risk bins only (a bin inside an ongoing event of the outcome's
detector is dropped, self-history counted from event ends), so its rates are
the rate at which events start while none is ongoing; Family A's are events per
second of stop. See
[docs/rate_models.md](rate_models.md) for which family to cite for what,
the M0–M9 ladder, and the point-process model-checking. Runners:
`scripts/behavior/rate/fit_per_stop_rate_models.py`, `scripts/behavior/rate/fit_per_bin_rate_models.py`;
plotter: `scripts/behavior/rate/plot_per_bin_rate_models.py`.

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
trial_position = data["position_info"].loc[trial.movement_start_time:trial.end_time]
trial_task_vars = data["task_variables"].loc[trial.movement_start_time:trial.end_time]
```

## Dependencies and JAX

This project uses **JAX** for machine learning. The base install is
**CPU-only** so the package builds on every platform (including
macOS arm64, where `jax-cuda12-plugin` has no wheel).

- Default: `pip install` / `uv sync` resolves plain `jax`.
- GPU: the `cuda` extra adds `jax-cuda12-plugin` (the CUDA build of JAX), and
  the `gpu` extra adds it with `dask-cuda` and `cuda-bindings`, which the
  decoders' GPU cluster needs: ``uv sync --frozen --inexact --extra gpu``.
  Linux-only; `uv.lock` locks both extras.
- Consider using GPU-aware code when implementing new analyses.
- Dask is available for parallel processing across CPU cores.

## Animal and Session Information

The dataset includes recordings from 5 animals. Each NWB file can contain multiple epochs (e.g., `02_r1`, `04_r2`, `06_r3`). The cohort is the epoch register's analysed epochs (`data_loaders/epoch_register.csv`, read by `analysed_epochs()`); `docs/epoch_register.md` gives each excluded epoch's reason. The register analyses 242 epochs in 45 NWB files; the table counts those, and its dates are
the recording days of the analysed epochs (`tests/repository/test_documented_values.py` checks both).

| Animal | NWB Files | Epochs | Dates |
|--------|-----------|--------|-------|
| chimi | 14 | 46 | Feb 2020 |
| j16 | 8 | 58 | Jul 2021 |
| peanut | 6 | 44 | Nov-Dec 2020 |
| senor | 10 | 49 | Oct-Nov 2020 |
| wilbur | 7 | 45 | Mar-Apr 2021 |

See `NWB_FILES` in `data_loaders/constants.py` for the complete file list.

## Running Decoding Analysis

How to run the decoders, their settings and their outputs are in
[reproduce.md](reproduce.md#decoding).

## Reported numbers

Every number the project reports is read from a summary file, `paper/reported/<summary>.json`, named for
what it holds: `survey`, `detection`, `events`, `behavior`, `mpfc` (`scripts/reported/summarize.py`, then
`scripts/reported/emit_reported_values.py`). The survey summary holds the literature numbers the
introduction quotes, computed from the replay-study survey of Supplementary Table S1
(`continuum_swr_replay.reported.survey`); it needs no lab data.

Which question each value answers, and which figure shows it, is written once, in the question list
(`src/continuum_swr_replay/reported/questions.py`). It organizes the writing by part: a question need
not have a figure. A question's id is its place in the outline (`2d.2`: section 2d, its second question).
Each builder declares its values' tags in one table (`TAGS`), and a summary whose values name a question
or figure the list does not know is not written; a summary written before the list changed is still
emitted, and its mis-tagged values are listed in `paper/reported/questions.md` until it is rebuilt. `make reported-values` writes that file: every
question in the paper's outline, with the values that answer it, their macros, and the values by figure.

Each value records the builder function that made it (`function`; "Builder function" in `questions.md`),
so a macro leads to its code in two steps: `\EventsRippleMedianNonSwr` is made in `brain_state_values` of
`reported/events.py`, which reads it from the `analysis/` function that computes it
(`event_statistics.event_signal_statistics`). A builder makes its values through
`reported.values.value_factory`, which is given that function. Whether the manuscript cites a value is not recorded anywhere: it is read from the macros
`paper/main.tex` and the other `.tex` files under `paper/` use. `make reported-values` prints how many
values are cited, and `make release-check` fails when a cited value is pending.

A summary in `paper/reported/` is built from committed files: `summarize.py` writes nothing there when a
file of the checkout that the build used (its code, the package's data files, an input the repository
tracks, `uv.lock`) differs from the commit; `--output-dir` elsewhere makes a trial build. Each summary
records those files, so `make release-check` can say when one has changed since. It also fails when
`paper/reported_values.tex` is not what the summaries give.

A builder names and describes values; it does not compute them. The statistics are functions in
`analysis/`, each returning named numbers (a dict, or a table with one row per group), tested there on
tables small enough to check by hand, so that a figure showing a statistic calls the function the text's
number comes from. The statistics that know nothing of the task are in `analysis/shared_statistics.py`: the
signed-rank p (NaN below six non-zero differences), the Benjamini-Hochberg correction (a missing p is
left out of the family) and the animal-balanced summary over sessions; every other module calls those.
Each builder's statistics:

- events: `analysis/hippocampus/event_statistics.py`, `analysis/inventory_comparison/nonlocal_comparison.py`;
- mpfc: `analysis/pfc/population_statistics.py`, `analysis/pfc/timing.py`, and
  `analysis/pfc/marked_family_statistics.py` (through the marked-family summary
  `scripts/pfc/summarize_pfc_marked_family.py` writes);
- detection: `analysis/hippocampus/memoryless_sequence/sweep_statistics.py`, `recording_statistics.py`
  and `event_table.py`;
- behavior: `analysis/behavior/coefficient_statistics.py` (a per-session fit summarized over sessions and
  animals; the other modules build on it), `rate/rate_fit_statistics.py`, `rate/stop_statistics.py`,
  `content/content_statistics.py`, `rate/history_kernel_statistics.py`, and the stay/leave summaries in
  `choice/learning.py`.

A builder may count the size of an input it reads (its epochs, its animals) and copy a statistic a file
already holds. Figure 4 and the `scripts/pfc/` reports call these functions; the scripts and figures that
still compute a reported statistic themselves are listed in `TODO.md`.

- When stating a result number — in a doc, a caption or a chat reply — read it from a summary file and
  name the file and the key. If the number is in no summary, say it is not computed; do not compute it
  inline or quote it from memory.
- In a doc, write a reported number inside a marker, `<!--v:<summary>.key-->…<!--/v-->`, and let the
  emitter fill it. Rounding happens only in `continuum_swr_replay.reported.formatting`.
- A value with a `pending` note is not quoted as a finding. A value read from an output that records
  other tables than the current ones (a stale fit, a stale per-event measure) is kept and marked pending
  with the reason, and `make release-check` fails when the manuscript cites a pending value. A builder
  refuses to build when an input cannot give the value at all: it does not cover the cohort (an epoch,
  an event or a unit has no row), it is an older design, or its record cannot be read. Not every input
  records the tables it was made from; those have no staleness check
  (`docs/known_issues/reported_statistics_choices.md`, entry A3).
- A number quoted from a cited paper (for example, another study's share of replays) is not a result
  of this project and has no summary key: write it as text beside its citation. A number this project
  computes, even from published sources (the survey summary's literature numbers), is a reported number.

## Important Conventions

1. **Time units**: Always in seconds (never milliseconds)
2. **Epoch naming**: Format is `"{epoch_number}_r{replicate}"` (e.g., "02_r1")
3. **Epoch identifiers**: `{animal}{date}_{epoch}_{brain_area}` (e.g., "chimi20200212_02_r1_HPC"), built and parsed only by `decoding/identifiers.py`
4. **Brain area names**: Use "HPC", "mPFC", "OFC" (exact capitalization matters)
5. **Interval naming**: Spyglass uses " noPrePostTrialTimes" suffix for trimmed intervals
6. **Track graph**: NetworkX graph of 10 nodes, where nodes 0–5 are wells (6–8 patch junctions, 9 the center) and edges are track segments
7. **Index/time alignment**: Use `.loc[start:end]` for time-based slicing, not `.iloc`
8. **Data roots**: every root is read at call time from `$CSR_PROCESSED_DATA_DIR` through `paths`. In tests that variable is the one patch point (the `processed_data_root` fixture); the suite starts with a sentinel root that cannot be read or written
9. **AIC and BIC differences**: the child rung's minus the parent rung's, within session; negative means the child is better (`analysis/behavior/coefficient_statistics.aic_difference_statistics`, `analysis/behavior/content/model_comparison.compare_nested_models`). Both compare a session only when its two fits can be compared (`analysis/behavior/fit_comparability.why_not_comparable`: the same numbers of events, bins and stops, the same clock and stop cohort, likelihoods on the same constants). Likelihood-ratio statistics and held-out gains keep their own signs

## Notes on Data Quality

- Not all sessions have recordings from all brain areas (check keys in `spike_times` dict)
- Coincident spike detection removes likely artifacts (>33% of tetrodes firing within 40μs)
- Position data uses DLC tracking when available, falls back to older tracking methods
- Some sessions may lack certain interval types or brain areas: the loaders count rows and skip the step when there are none, and an absent area raises `spikes.AreaNotRecordedError`, which `get_spike_data` skips
- Check for empty spike data before running decoding (`if not data["spike_times"]`)
- Artifacts: one detector (`analysis/hippocampus/artifacts/`, run `analysis/events/swr_source.py:ARTIFACT_INTERVALS_RUN_ID`) masks sustained ripple-band noise (S blocks), step artifacts and synchronous spike lines; see `docs/design/2026-10-09-artifact-detection.md`. An epoch with an S block is left out whole (epoch register reason `SUSTAINED_NOISE`).
