# Spyglass Workflows

Step-by-step workflows for common analysis tasks in Spyglass.

## Workflow 1: Exploring Available Data

### Step 1: List All Sessions

```python
from spyglass.common import Session

# View all sessions
Session.fetch(limit=20)

# View sessions for specific subject
Session & {'subject_id': 'J16'}
```

### Step 2: Inspect Session Details

```python
# Get full session info
session_info = (Session & {'nwb_file_name': nwb_file}).fetch1()

# Check experimenters
from spyglass.common import Session
Session.Experimenter & {'nwb_file_name': nwb_file}

# Check recording devices
Session.DataAcquisitionDevice & {'nwb_file_name': nwb_file}
```

### Step 3: Find Available Intervals

```python
from spyglass.common import IntervalList

# All intervals for session
IntervalList & {'nwb_file_name': nwb_file}

# Get specific interval times
(IntervalList & {
    'nwb_file_name': nwb_file,
    'interval_list_name': '02_r1'
}).fetch('valid_times')
```

### Step 4: Preview Analysis Results

```python
from spyglass.position import PositionOutput

# See what position data exists
PositionOutput.merge_view()

# Restrict to specific session
PositionOutput.merge_restrict({'nwb_file_name': nwb_file})

# Find which pipeline versions available
PositionOutput.merge_get_part(
    {'nwb_file_name': nwb_file},
    multi_source=True
)
```

## Workflow 2: Fetching Position Data

### Step 1: Build Restriction Key

Identify the fields needed to uniquely specify the data:

```python
key = {
    'nwb_file_name': 'j1620210710_.nwb',
    'interval_list_name': 'pos 1 valid times',
    'trodes_pos_params_name': 'default'
}
```

### Step 2: Get Part Table and merge_id

```python
from spyglass.position import PositionOutput

# Find which part table has this data
part = PositionOutput.merge_get_part(key)

# Get the merge_id
merge_key = part.fetch1("KEY")
print(merge_key)  # {'merge_id': '...'}
```

### Step 3: Fetch Data

```python
# Fetch as DataFrame
position_df = (PositionOutput & merge_key).fetch1_dataframe()

# View structure
print(position_df.columns)
print(position_df.head())

# Extract specific columns
xy_data = position_df[['position_x', 'position_y']]
```

### Step 4: Plot Position

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(
    position_df['position_x'],
    position_df['position_y'],
    'b-',
    alpha=0.5,
    linewidth=0.5
)
plt.xlabel('X Position (cm)')
plt.ylabel('Y Position (cm)')
plt.title(f"Position trajectory: {key['nwb_file_name']}")
plt.axis('equal')
plt.show()
```

## Workflow 3: LFP Analysis

### Step 1: Identify Electrode Groups

```python
from spyglass.common import ElectrodeGroup

# View available electrode groups
ElectrodeGroup & {'nwb_file_name': nwb_file}

# Get details for specific group
group_info = (ElectrodeGroup & {
    'nwb_file_name': nwb_file,
    'electrode_group_name': 'lfp_tets_j16'
}).fetch1()
```

### Step 2: Check Available Filters

```python
from spyglass.common import FirFilterParameters

# Common LFP filters
FirFilterParameters & {'filter_name': 'LFP 0-400 Hz'}

# Theta filters
FirFilterParameters & 'filter_name LIKE "Theta%"'

# All available filters
FirFilterParameters.fetch('filter_name')
```

### Step 3: Build LFP Key

```python
from spyglass.lfp import LFPOutput

key = {
    'nwb_file_name': nwb_file,
    'lfp_electrode_group_name': 'lfp_tets_j16',
    'target_interval_list_name': '02_r1 noPrePostTrialTimes',
    'filter_name': 'LFP 0-400 Hz',
    'filter_sampling_rate': 30000
}
```

### Step 4: Fetch LFP Data

```python
# Get merge_id
merge_key = LFPOutput.merge_get_part(key).fetch1("KEY")

# Fetch data
lfp_df = (LFPOutput & merge_key).fetch1_dataframe()

# View structure
print(f"Shape: {lfp_df.shape}")
print(f"Sampling rate: {1/(lfp_df.index[1] - lfp_df.index[0])} Hz")
print(f"Duration: {lfp_df.index[-1] - lfp_df.index[0]} seconds")
```

### Step 5: Plot LFP

```python
import matplotlib.pyplot as plt

# Plot first 2 seconds of first 4 channels
duration = 2.0  # seconds
channels = lfp_df.columns[:4]

fig, axes = plt.subplots(len(channels), 1, figsize=(12, 8), sharex=True)
for ax, channel in zip(axes, channels):
    data = lfp_df[channel]
    mask = lfp_df.index < lfp_df.index[0] + duration
    ax.plot(lfp_df.index[mask], data[mask])
    ax.set_ylabel(f'{channel}\n(μV)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
```

## Workflow 4: Spike Sorting Analysis

### Step 1: Find Available Spike Sorting Results

```python
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

# Preview available results
SpikeSortingOutput.merge_view()

# For specific session
SpikeSortingOutput.merge_restrict({'nwb_file_name': nwb_file})
```

### Step 2: Get Spike Times

**Method A: Using merge_id directly**
```python
# If you know the merge_id
spike_times = SpikeSortingOutput().get_spike_times(
    {"merge_id": "0164f4ef-8f78-c9a7-d50e-72c699bbbffc"}
)
```

**Method B: Using friendly keys**
```python
# Build key with readable fields
key = {
    'nwb_file_name': nwb_file,
    'sort_interval_name': '02_r1',
    'sorter': 'mountainsort4',
    'curation_id': 1
}

# Get merge_ids
merge_ids = SpikeSortingOutput.get_restricted_merge_ids(key)

# Fetch spike times
for merge_id in merge_ids:
    spike_times = SpikeSortingOutput().get_spike_times(
        {"merge_id": merge_id}
    )
```

### Step 3: Work with Spike Groups

```python
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

# Fetch spike data with unit IDs
spike_times, unit_ids = SortedSpikesGroup.fetch_spike_data(
    {
        'nwb_file_name': nwb_file,
        'unit_filter_params_name': 'all_units',
        'sorted_spikes_group_name': 'HPC_02_r1_clusterless'
    },
    return_unit_ids=True
)
```

### Step 4: Compute Firing Rates

```python
import numpy as np

# Get interval boundaries
interval_times = (IntervalList & {
    'nwb_file_name': nwb_file,
    'interval_list_name': '02_r1'
}).fetch1('valid_times')

# Compute firing rate for each unit
for unit_id, spikes in zip(unit_ids, spike_times):
    # Count spikes in interval
    n_spikes = np.sum(
        (spikes >= interval_times[0, 0]) & 
        (spikes <= interval_times[-1, 1])
    )
    
    # Compute duration
    duration = np.sum(interval_times[:, 1] - interval_times[:, 0])
    
    # Firing rate
    firing_rate = n_spikes / duration
    
    print(f"Unit {unit_id}: {firing_rate:.2f} Hz")
```

## Workflow 5: Decoding Analysis

### Step 1: Find Decoding Results

```python
from spyglass.decoding import DecodingOutput

# See what's available
DecodingOutput.merge_view()

# For specific session
DecodingOutput.merge_restrict({'nwb_file_name': nwb_file})
```

### Step 2: Fetch Decoder Results

```python
# Build key
key = {
    'nwb_file_name': nwb_file,
    'decoding_params_name': 'default_decoding',
    # ... other specific keys
}

# Get merge_id
merge_key = DecodingOutput.merge_get_part(key).fetch1("KEY")

# Fetch results
results = (DecodingOutput & merge_key).fetch_results()
```

### Step 3: Analyze Results

```python
# Typically results contain:
# - posterior: P(position | spikes) over time
# - position: actual position
# - time: timestamps

posterior = results['posterior']  # Shape: (n_time, n_position_bins)
actual_position = results['position']
time = results['time']

# Compute decoded position (MAP estimate)
decoded_position = np.argmax(posterior, axis=1)
```

### Step 4: Visualize Decoding

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot posterior
axes[0].imshow(
    posterior.T,
    aspect='auto',
    extent=[time[0], time[-1], 0, posterior.shape[1]],
    origin='lower',
    cmap='hot'
)
axes[0].set_ylabel('Position Bin')
axes[0].set_title('Posterior Probability')

# Plot actual vs decoded
axes[1].plot(time, actual_position, 'b-', label='Actual', alpha=0.7)
axes[1].plot(time, decoded_position, 'r-', label='Decoded', alpha=0.7)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Position')
axes[1].legend()

plt.tight_layout()
plt.show()
```

## Workflow 6: Joining Tables

### Example: Find All Position Data for Sessions with Specific Experimenters

```python
from spyglass.common import Session, LabMember
from spyglass.position import PositionOutput

# Step 1: Find sessions by experimenter
sessions = (
    Session.Experimenter &
    {'lab_member_name': 'Kyu Hyun Lee'}
).fetch('nwb_file_name')

# Step 2: Find position data for these sessions
position_entries = []
for session in sessions:
    entries = PositionOutput.merge_restrict(
        {'nwb_file_name': session}
    ).fetch()
    position_entries.extend(entries)

print(f"Found {len(position_entries)} position analyses")
```

### Example: Find LFP Data for Specific Brain Region

```python
from spyglass.common import ElectrodeGroup, BrainRegion
from spyglass.lfp import LFPOutput

# Step 1: Find electrode groups in region
electrode_groups = (
    ElectrodeGroup &
    (BrainRegion & {'region_name': 'CA1'})
).fetch('KEY')

# Step 2: Find LFP data for these groups
for group_key in electrode_groups:
    lfp_data = LFPOutput.merge_restrict(group_key)
    if len(lfp_data):
        print(f"Found LFP for: {group_key}")
```

## Workflow 7: Upstream/Downstream Navigation

### Find All Downstream Results from a Session

```python
from spyglass.position import PositionOutput

# All position outputs for this session
position_data = PositionOutput() << f"nwb_file_name = '{nwb_file}'"

print(f"Found {len(position_data)} position results")
```

### Find Sessions That Have Specific Parameters

```python
from spyglass.common import Session

# Find sessions that were processed with specific parameters
sessions = Session() >> 'trodes_pos_params_name="default"'

print(f"Found {len(sessions)} sessions with default position parameters")
```

## Common Patterns

### Pattern 1: Batch Processing

```python
# Process multiple sessions
sessions = Session & {'subject_id': 'J16'}

results = []
for session_key in sessions.fetch('KEY'):
    # Get data
    key = {**session_key, 'interval_list_name': 'task'}
    try:
        merge_key = PositionOutput.merge_get_part(key).fetch1("KEY")
        data = (PositionOutput & merge_key).fetch1_dataframe()
        results.append(data)
    except:
        print(f"No data for {session_key}")
        continue
```

### Pattern 2: Safe Fetching with Error Handling

```python
def safe_fetch_position(key):
    """Safely fetch position data with error handling"""
    try:
        part = PositionOutput.merge_get_part(key)
        if len(part) == 0:
            print(f"No position data for {key}")
            return None
        merge_key = part.fetch1("KEY")
        return (PositionOutput & merge_key).fetch1_dataframe()
    except Exception as e:
        print(f"Error fetching {key}: {e}")
        return None
```

### Pattern 3: Incremental Exploration

```python
# Start broad
IntervalList & {'nwb_file_name': nwb_file}

# Add constraints incrementally
intervals = (
    IntervalList &
    {'nwb_file_name': nwb_file} &
    'interval_list_name LIKE "%r1%"'
)

# Preview before fetching
intervals.fetch(limit=5)
```

## Troubleshooting

### No Results Found

```python
# Check if session exists
assert len(Session & {'nwb_file_name': nwb_file}) > 0

# Check if interval exists  
assert len(IntervalList & key) > 0

# Check what's available
PositionOutput.merge_view()
```

### Multiple Results

```python
# If merge_get_part returns multiple
parts = PositionOutput.merge_get_part(key, multi_source=True)
for part in parts:
    merge_key = part.fetch1("KEY")
    print(f"Found in {part.table_name}: {merge_key}")
```

### Join Not Working

```python
# Check primary/foreign keys
Table1.describe()
Table2.describe()

# Check for common keys
set(Table1.heading.names) & set(Table2.heading.names)

# Use parents/children to find path
Table1.parents()
Table2.children()
```
