# Spyglass Common Tables Reference

Essential tables in the Spyglass schema with their primary keys and purposes.

## Common Schema Tables

### Session and File Management

#### `common_nwbfile.Nwbfile`

**Primary Key**: `nwb_file_name`

Registry of every raw NWB file. All pipelines anchor here.

```python
from spyglass.common import Nwbfile
Nwbfile & {'nwb_file_name': 'j1620210710_.nwb'}
```

#### `common_nwbfile.AnalysisNwbfile`

**Primary Key**: `analysis_file_abs_path`

Tracks derived NWB files (filtered LFP, decoding results, etc.).

#### `common_session.Session`

**Primary Key**: `nwb_file_name`

One row per recording session. First thing you typically restrict on.

**Key Fields**:

- `subject_id`: Animal identifier
- `session_start_time`: Timestamp
- `session_description`: Text description
- `institution_name`: FK to Institution
- `lab_name`: FK to Lab

**Part Tables**:

- `Session.Experimenter`: Maps sessions to experimenters
- `Session.DataAcquisitionDevice`: Lists hardware used

```python
from spyglass.common import Session
Session & {'nwb_file_name': nwb_file}
```

### Time Intervals

#### `common_interval.IntervalList`

**Primary Key**: `nwb_file_name`, `interval_list_name`

Time windows (task, sleep, artifact) that gate every analysis.

**Key Fields**:

- `valid_times`: Array of [start, stop] timestamps
- `interval_list_name`: Descriptive name (e.g., "02_r1", "sleep")

```python
from spyglass.common import IntervalList
(IntervalList & {
    'nwb_file_name': nwb_file,
    'interval_list_name': '02_r1'
}).fetch('valid_times')
```

### Subject and Lab

#### `common_subject.Subject`

**Primary Key**: `subject_id`

Animal metadata, auto-populated from NWB.

**Key Fields**:

- `sex`: M/F/U
- `species`: Taxonomy
- `genotype`: Genetic info
- `subject_description`: Additional notes

#### `common_lab.LabMember`

**Primary Key**: `lab_member_name`

People registry used in Session.Experimenter and permissions.

#### `common_lab.LabTeam`

**Primary Key**: `team_name`

Groups members so collaborators can manage their own data.

**Part Table**: `LabTeamMember` links members to teams.

#### `common_lab.Institution`

**Primary Key**: `institution_name`

Lookup table referenced in `Session.institution_name`.

#### `common_lab.Lab`

**Primary Key**: `lab_name`

Lookup table referenced in `Session.lab_name`.

### Hardware and Devices

#### `common_device.DataAcquisitionDevice`

**Primary Key**: `data_acquisition_device_name`

Amplifier/digitizer catalog.

**Key Fields**:

- `device_manufacturer`: Company name
- `device_model`: Model number
- `amplifier_type`: E.g., "Intan"

#### `common_device.CameraDevice`

**Primary Key**: `camera_name`

Camera hardware referenced by TaskEpoch.

#### `common_device.ProbeType`

**Primary Key**: `probe_type`

Defines shank count, site spacing, manufacturer.

**Key Fields**:

- `probe_type`: E.g., "128c-4s8mm6cm-20um-40um-sl"
- `num_shanks`: Number of shanks
- `contact_side_numbering`: Site layout

#### `common_device.Probe`

**Primary Key**: `probe_id`

Physical probe instances linked to sessions.

### Ephys Recording

#### `common_ephys.ElectrodeGroup`

**Primary Key**: `nwb_file_name`, `electrode_group_name`

Groups channels on a probe.

**Key Fields**:

- `target_brain_region`: Anatomical region
- `targeted_x`, `targeted_y`, `targeted_z`: Coordinates

#### `common_ephys.Electrode`

**Primary Key**: `nwb_file_name`, `electrode_id`

Channel-level metadata (coordinates, region).

**Key Fields**:

- `probe_shank`: Shank number
- `probe_electrode`: Site on shank
- `region_id`: FK to BrainRegion
- `x`, `y`, `z`: Electrode coordinates

#### `common_ephys.Raw`

**Primary Key**: `nwb_file_name`, `interval_list_name`, `raw_object_id`

Entry point for raw ElectricalSeries upstream of LFP and spikes.

### Filtering

#### `common_filter.FirFilterParameters`

**Primary Key**: `filter_name`, `filter_sampling_rate`

Library of standard FIR kernels (theta, gamma, ripple).

**Key Fields**:

- `filter_type`: E.g., "bandpass"
- `filter_low_stop`, `filter_low_pass`: Lower frequencies
- `filter_high_pass`, `filter_high_stop`: Upper frequencies

### Position Tracking

#### `common_position.IntervalPositionInfo`

**Primary Key**: `nwb_file_name`, `interval_list_name`

Links raw pose series to analysis epochs.

### Sensors and I/O

#### `common_sensors.SensorData`

**Primary Key**: `nwb_file_name`, `sensor_data_object_id`

Generic analog/IMU channels.

#### `common_dio.DIOEvents`

**Primary Key**: `dio_event_name`

TTL pulses and sync lines for behavior timestamping.

### Task and Behavior

#### `common_task.Task`

**Primary Key**: `task_name`

Lookup of behavioral task definitions.

#### `common_task.TaskEpoch`

**Primary Key**: `nwb_file_name`, `epoch`

Maps each epoch to Task, CameraDevice, and IntervalList.

**Key Fields**:

- `task_name`: FK to Task
- `task_environment`: E.g., "linear track"
- `camera_name`: FK to CameraDevice

### Brain Regions

#### `common_region.BrainRegion`

**Primary Key**: `region_id`

Standardized anatomical labels.

**Key Fields**:

- `region_name`: E.g., "CA1", "PFC"
- `subregion_name`: More specific location

## Analysis Pipeline Tables

### Position

#### `position.PositionOutput` (merge master)

**Primary Key**: `merge_id`

Final XY/theta trajectories.

**Part Tables**:

- `TrodesPosV1`: LED-based tracking
- `DLCPosV1`: DeepLabCut tracking

```python
from spyglass.position import PositionOutput

# Find position data
key = {
    'nwb_file_name': nwb_file,
    'interval_list_name': 'pos 1 valid times',
    'trodes_pos_params_name': 'default'
}
merge_key = PositionOutput.merge_get_part(key).fetch1("KEY")
position_df = (PositionOutput & merge_key).fetch1_dataframe()
```

### LFP

#### `lfp.LFPOutput` (merge master)

**Primary Key**: `merge_id`

Band-limited LFP data.

**Main Part**: `LFPV1`

```python
from spyglass.lfp import LFPOutput

key = {
    'nwb_file_name': nwb_file,
    'lfp_electrode_group_name': 'lfp_tets_j16',
    'target_interval_list_name': '02_r1 noPrePostTrialTimes',
    'filter_name': 'LFP 0-400 Hz',
    'filter_sampling_rate': 30000
}
merge_key = LFPOutput.merge_get_part(key).fetch1("KEY")
lfp_df = (LFPOutput & merge_key).fetch1_dataframe()
```

### Spike Sorting

#### `spikesorting.SpikeSortingOutput` (merge master)

**Primary Key**: `merge_id`

Curated spike times.

**Main Part**: `CurationV1`

```python
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

# Get spike times
spike_times = SpikeSortingOutput().get_spike_times(
    {"merge_id": merge_id}
)

# Or use friendly keys
merge_ids = SpikeSortingOutput.get_restricted_merge_ids(key)
```

### Decoding

#### `decoding.DecodingOutput` (merge master)

**Primary Key**: `merge_id`

Decoder predictions.

**Part Tables**:

- `ClusterlessDecodingV1`: Clusterless decoding
- `SortedSpikesDecodingV1`: Sorted spikes decoding

```python
from spyglass.decoding import DecodingOutput

results = (DecodingOutput & part_key).fetch_results()
```

### Linearization

#### `linearization.LinearizationOutput` (merge master)

**Primary Key**: `merge_id`

Position projected onto track.

### Ripple Detection

#### `ripple.RippleTimesOutput` (merge master)

**Primary Key**: `merge_id`

Detected ripple events.

## Working with Merge Tables

All merge tables follow the same pattern:

1. **Build restriction key** with friendly fields
2. **Get part table** containing the data:

   ```python
   part = MergeTable.merge_get_part(key)
   ```

3. **Fetch merge_id**:

   ```python
   merge_key = part.fetch1("KEY")
   ```

4. **Restrict and fetch data**:

   ```python
   data = (MergeTable & merge_key).fetch1_dataframe()
   ```

## Table Joining Protocol

When connecting two tables:

1. Use `.describe()` or `.heading` to view primary/foreign keys
2. Look for direct foreign keys
3. If none, check for linker tables via `.parents()` / `.children()`
4. Chain joins, confirming keys at each step
5. Smoke-test: `.fetch(limit=1, as_dict=True)`

## Best Practices

- **Always start with Session or Nwbfile**: These are the root tables
- **Restrict by interval**: Use IntervalList to gate time windows
- **Use merge_view() first**: Preview structure before fetching
- **Limit large queries**: Add `limit=` parameter to avoid memory issues
- **Check relationships**: Use `.describe()` when joining unfamiliar tables
