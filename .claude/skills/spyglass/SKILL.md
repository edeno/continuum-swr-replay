---
name: spyglass
description: Analyze neuroscience data using the Spyglass data analysis framework (DataJoint + NWB). Use when working with neural recordings, spike sorting, position tracking, LFP analysis, or decoding tasks. Trigger when user mentions Spyglass, DataJoint tables, NWB files, neural data analysis, or specific pipelines (position, LFP, spike sorting, decoding).
---

# Spyglass

## Overview

Spyglass is a data analysis framework for neuroscience that facilitates storage, analysis, and sharing of neural data using DataJoint and the NWB (Neurodata Without Borders) format. This skill helps analyze neural recordings including spike sorting, position tracking, LFP analysis, and decoding.

## Role & Core Directives

- **Read-only access**: Never generate code that writes to the database (`insert`, `populate`, etc.)
- **Teaching approach**: Always show code, output, and explain (≤200 words unless asked)
- **Best practices**: Use PEP8, descriptive variable names, prefer DataJoint commands over raw SQL
- **Environment**: Assume Jupyter environment with remote NWB files; `fetch_nwb()` works for files in `AnalysisNwbfileKachery`

## Core Concepts

### DataJoint Operators

- **`&`** (restriction): Filter rows by key dict or SQL condition
- **`*`** (join): Combine tables on matching primary/foreign keys
- **`.proj()`**: Rename/drop attributes before join
- **`.fetch()` / `.fetch1()`**: Materialize query results
- **`.aggr()`**: On-the-fly aggregation

### Spyglass-Specific Operators

- **`<<`** (upstream): Restrict by ancestor attribute (searches up dependency chain)
- **`>>`** (downstream): Restrict by descendant attribute (searches down)
- **`.restrict_by()`**: Explicit long-distance restrict with direction
- **Merge table methods**:
  - `.merge_restrict()`: Union of all part tables
  - `.merge_get_part()`: Get the part table containing specific key
  - `.merge_fetch()`: Fast cross-part fetch
  - `.merge_view()`: Preview combined data structure

### Merge Tables

Merge tables consolidate results from different pipeline versions (e.g., `PositionOutput`, `LFPOutput`, `SpikeSortingOutput`). Each has part tables for specific implementations. Always use `merge_id` to reference specific entries, never type it manually.

**Standard workflow**:

1. Build restriction key with friendly fields (e.g., `nwb_file_name`, `interval_list_name`)
2. Use `merge_get_part(key)` to find which part table contains the data
3. Fetch `merge_id` from the result: `merge_key = result.fetch1("KEY")`
4. Restrict merge table with `merge_id`: `(MergeTable & merge_key).fetch1_dataframe()`

## Quick Start

```python
from spyglass.common import Session, IntervalList
from spyglass.position import PositionOutput
from spyglass.lfp import LFPOutput
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

# Find sessions
nwb_file_name = "j1620210710_.nwb"
Session & {"nwb_file_name": nwb_file_name}

# Get position data
key = {
    "nwb_file_name": nwb_file_name,
    "interval_list_name": "pos 1 valid times",
    "trodes_pos_params_name": "default",
}
merge_key = (PositionOutput.merge_get_part(key)).fetch1("KEY")
position_data = (PositionOutput & merge_key).fetch1_dataframe()
```

## Common Tasks

### 1. Finding Available Data

**List sessions**:

```python
Session.fetch()
```

**Find position data for session**:

```python
PositionOutput.merge_get_part({"nwb_file_name": nwb_file}, multi_source=True)
```

**Preview merge table structure**:

```python
PositionOutput.merge_view()
```

### 2. Table Joining

When connecting two tables:

1. Use `.describe()` or `.heading` to view primary/foreign keys
2. Look for direct foreign keys; if none, check `.parents()` / `.children()`
3. Chain joins, confirming keys at each step
4. Test with `.fetch(limit=1, as_dict=True)`

### 3. Fetching Data

**Large tables**: Always limit rows

```python
(IntervalList & restriction).fetch(limit=10)
```

**Get specific columns**:

```python
(Session & key).fetch('session_start_time', limit=3)
```

**As DataFrame**:

```python
(PositionOutput & merge_key).fetch1_dataframe()
```

### 4. Working with Spike Data

```python
# Get spike times using merge_id
spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})

# Or use friendly keys with get_restricted_merge_ids
ids = SpikeSortingOutput.get_restricted_merge_ids(friendly_key)
```

### 5. LFP Analysis

```python
key = {
    "nwb_file_name": nwb_file,
    "lfp_electrode_group_name": "lfp_tets_j16",
    "target_interval_list_name": "02_r1 noPrePostTrialTimes",
    "filter_name": "LFP 0-400 Hz",
    "filter_sampling_rate": 30000,
}
merge_key = (LFPOutput.merge_get_part(key)).fetch1("KEY")
lfp_data = (LFPOutput & merge_key).fetch1_dataframe()
```

## Communication Style

Use this pattern when explaining:

1. Analogy → formal term → one-sentence definition
2. Use markdown for structure
3. Keep explanations ≤200 words unless user requests more
4. Always show working code examples

## References

For detailed information, read these reference files:

- **[references/api_commands.md](references/api_commands.md)**: Complete API reference with all DataJoint and Spyglass commands
- **[references/common_tables.md](references/common_tables.md)**: Essential tables, their keys, and purposes
- **[references/workflows.md](references/workflows.md)**: Step-by-step workflows for common analysis tasks
