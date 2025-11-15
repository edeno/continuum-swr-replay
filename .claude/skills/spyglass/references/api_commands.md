# Spyglass API Reference

Complete reference of DataJoint and Spyglass commands for querying and analyzing neural data.

## DataJoint Core Commands

### Restriction (`&`)

Filter rows by key dictionary or SQL condition.

```python
# By key dictionary
Session & {'nwb_file_name': 'j1620210710_.nwb'}

# By SQL condition
Session & 'session_start_time > "2021-07-01"'

# Multiple conditions
Session & {'subject_id': 'J16'} & 'session_start_time > "2021-07-01"'
```

### Join (`*`)

Combine tables on matching primary/foreign keys.

```python
# Simple join
Session * Subject

# With projection
(Session * Subject).proj(session_date='session_start_time')
```

### Projection (`.proj()`)

Rename or drop attributes before join.

```python
# Rename attribute
Session.proj(session_date='session_start_time')

# Compute new attribute
Session.proj(month='MONTH(session_start_time)')

# Drop attributes (by omission)
Session.proj('nwb_file_name', 'subject_id')  # Only keeps these
```

### Fetch (`.fetch()` / `.fetch1()`)

Materialize query results as arrays, dicts, or DataFrames.

```python
# Fetch all rows
(Session & key).fetch()

# Fetch specific attributes
(Session & key).fetch('nwb_file_name', 'session_start_time')

# Fetch as dictionaries
(Session & key).fetch(as_dict=True)

# Fetch single row
(Session & key).fetch1()

# Fetch with limit
(IntervalList & key).fetch('valid_times', limit=10)
```

### Aggregation (`.aggr()`)

On-the-fly aggregation across joined tables.

```python
# Count entries
Spikes.aggr(IntervalList, n='count(*)')

# Average, sum, etc.
Spikes.aggr(IntervalList, avg_rate='avg(firing_rate)', n='count(*)')
```

### Union (`dj.U()`)

Union of two or more relations.

```python
task_or_sleep = dj.U(
    (IntervalList & 'interval_list_name="task"'),
    (IntervalList & 'interval_list_name="sleep"')
)
```

### OR-Restriction (`dj.AndList()`)

Restrict using list of dictionaries (OR operation).

```python
files = Session & dj.AndList([
    {'nwb_file_name': 'fileA.nwb'},
    {'nwb_file_name': 'fileB.nwb'}
])
```

### Table Inspection

```python
# View schema and primary keys
PositionOutput.describe()

# View all columns as dict
Session.heading

# View dependencies
Session.parents()
Session.children()
```

## Spyglass-Specific Commands

### Upstream Restriction (`<<`)

Restrict by ancestor attribute (searches up dependency chain).

```python
# Find all PositionOutput entries for a session
PositionOutput() << "nwb_file_name = 'j1620210710_.nwb'"
```

### Downstream Restriction (`>>`)

Restrict by descendant attribute (searches down dependency chain).

```python
# Find all sessions that have specific position parameters
Session() >> 'trodes_pos_params_name="default"'
```

### Explicit Restriction (`.restrict_by()`)

Explicit long-distance restrict with direction control.

```python
# Upstream
PositionOutput().restrict_by(
    "nwb_file_name = 'j1620210710_.nwb'",
    direction="up"
)

# Downstream
Session().restrict_by(
    'trodes_pos_params_name="default"',
    direction="down"
)
```

## Merge Table Commands

### Union View (`.merge_restrict()`)

Show combined data from all part tables.

```python
# List all entries across all parts
PositionOutput.merge_restrict()

# With restriction
PositionOutput.merge_restrict({'nwb_file_name': nwb_file})
```

### Get Part Table (`.merge_get_part()`)

Find which part table contains specific key.

```python
# Single result
part = PositionOutput.merge_get_part(key)

# Multiple possible sources
parts = PositionOutput.merge_get_part(key, multi_source=True)
```

### Cross-Part Fetch (`.merge_fetch()`)

Fast fetch across part tables.

```python
data = PositionOutput.merge_fetch(
    {'nwb_file_name': 'j1620210710_.nwb'}
)
```

### Preview Structure (`.merge_view()`)

View available columns and structure.

```python
PositionOutput.merge_view()
```

## NWB File Commands

### Fetch NWB File (`.fetch_nwb()`)

Load raw or analysis NWB as h5py.File object.

```python
# Load NWB file
nwb_file = (LFPOutput & part_key).fetch_nwb()

# Access data
data = nwb_file['processing']['LFP']['data'][:]
```

## DataFrame Commands

### Basic DataFrame (`.fetch1_dataframe()`)

Fetch first matching row as pandas DataFrame.

```python
position_df = (PositionOutput & part_key).fetch1_dataframe()
```

### Pose DataFrame (`.fetch_pose_dataframe()`)

Fetch pose keypoint data as DataFrame.

```python
# All bodyparts
pose_df = (PositionOutput & part_key).fetch_pose_dataframe()

# Specific bodypart
nose_df = (PositionOutput & part_key).fetch_pose_dataframe(
    bodypart='nose'
)
```

### Decoder Results (`fetch_results`)

Return decoder results as dict/array.

```python
results = (DecodingOutput & part_key).fetch_results()
```

## Spike Sorting Commands

### Get Merge IDs (`get_restricted_merge_ids`)

Map friendly keys to merge_ids.

```python
# Get merge_ids for specific criteria
merge_ids = SpikeSortingOutput.get_restricted_merge_ids(key)
```

### Fetch Spike Times (`.get_spike_times()`)

Get spike times for specific merge_id.

```python
spike_times = SpikeSortingOutput().get_spike_times(
    {"merge_id": merge_id}
)
```

### Fetch Spike Data (`.fetch_spike_data()`)

Get spike times and unit IDs from SortedSpikesGroup.

```python
spike_times, unit_ids = SortedSpikesGroup.fetch_spike_data(
    {
        "nwb_file_name": nwb_file,
        "unit_filter_params_name": "all_units",
        "sorted_spikes_group_name": "HPC_02_r1_clusterless",
    },
    return_unit_ids=True,
)
```

## Best Practices

1. **Always limit large queries**: Use `limit=` parameter to avoid overwhelming memory
2. **Use friendly keys first**: Start with readable keys (like `nwb_file_name`), then get `merge_id`
3. **Preview before fetching**: Use `.fetch(limit=1)` or `merge_view()` to check structure
4. **Check table relationships**: Use `.describe()`, `.parents()`, `.children()` when joining
5. **Prefer DataJoint over SQL**: Use restriction operators instead of raw SQL queries
