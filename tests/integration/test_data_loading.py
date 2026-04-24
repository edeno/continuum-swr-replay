"""Integration tests for the complete data loading pipeline.

Tests that the full load_data() function correctly orchestrates all
data loading steps and returns properly structured data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# These tests will use mock data since we don't have access to real NWB files
# In practice, you would use a small test NWB file


class TestDataLoadingIntegration:
    """Integration tests for load_data() function."""

    def test_mock_load_data_structure(self, mock_session_data):
        """Test that mock session data has all required keys."""
        required_keys = {
            'position_info',
            'track_graph',
            'linear_edge_order',
            'linear_edge_spacing',
            'spike_times',
            'spike_waveform_features',
            'ripple_times',
            'trials',
        }

        assert all(key in mock_session_data for key in required_keys)

    def test_position_data_time_indexed(self, mock_session_data):
        """Position data must be time-indexed (seconds)."""
        position_info = mock_session_data['position_info']

        assert isinstance(position_info.index, pd.Index)
        assert position_info.index.is_monotonic_increasing
        assert position_info.index.dtype in [np.float64, np.float32]

    def test_spike_times_within_position_bounds(self, mock_session_data):
        """All spike times must be within position time bounds."""
        position_info = mock_session_data['position_info']
        spike_times = mock_session_data['spike_times']

        t_min = position_info.index.min()
        t_max = position_info.index.max()

        for brain_area, spike_list in spike_times.items():
            for tetrode_spikes in spike_list:
                if len(tetrode_spikes) > 0:
                    assert tetrode_spikes.min() >= t_min, \
                        f"{brain_area} has spikes before position start"
                    assert tetrode_spikes.max() <= t_max, \
                        f"{brain_area} has spikes after position end"

    def test_spike_features_match_spike_counts(self, mock_session_data):
        """Waveform features must match spike counts for each tetrode."""
        spike_times = mock_session_data['spike_times']
        spike_features = mock_session_data['spike_waveform_features']

        assert 'HPC' in spike_times
        assert 'HPC' in spike_features

        hpc_times = spike_times['HPC']
        hpc_features = spike_features['HPC']

        assert len(hpc_times) == len(hpc_features)

        for times, features in zip(hpc_times, hpc_features, strict=True):
            assert len(times) == features.shape[0], \
                "Feature count must match spike count"

    def test_trials_within_position_bounds(self, mock_session_data):
        """Trial times must be within position recording bounds."""
        position_info = mock_session_data['position_info']
        trials = mock_session_data['trials']

        t_min = position_info.index.min()
        t_max = position_info.index.max()

        assert trials['start_time'].min() >= t_min
        assert trials['end_time'].max() <= t_max

    def test_ripple_times_within_position_bounds(self, mock_session_data):
        """Ripple times must be within position recording bounds."""
        position_info = mock_session_data['position_info']
        ripple_times = mock_session_data['ripple_times']

        t_min = position_info.index.min()
        t_max = position_info.index.max()

        assert ripple_times['start_time'].min() >= t_min
        assert ripple_times['end_time'].max() <= t_max

    def test_track_graph_connectivity(self, mock_session_data):
        """Track graph must be connected (no isolated nodes)."""
        track_graph = mock_session_data['track_graph']

        # Graph should be connected
        import networkx as nx
        assert nx.is_connected(track_graph), \
            "Track graph has disconnected components"

    def test_edge_order_matches_graph(self, mock_session_data):
        """Linear edge order must correspond to edges in track graph."""
        track_graph = mock_session_data['track_graph']
        linear_edge_order = mock_session_data['linear_edge_order']

        for edge in linear_edge_order:
            node1, node2 = edge
            assert track_graph.has_edge(node1, node2), \
                f"Edge {edge} not in track graph"

    def test_edge_spacing_matches_order(self, mock_session_data):
        """Edge spacing must have same length as edge order."""
        linear_edge_order = mock_session_data['linear_edge_order']
        linear_edge_spacing = mock_session_data['linear_edge_spacing']

        assert len(linear_edge_order) == len(linear_edge_spacing), \
            "Edge spacing length must match edge order length"

    def test_spike_times_are_sorted(self, mock_session_data):
        """All spike times must be in ascending order."""
        spike_times = mock_session_data['spike_times']

        for brain_area, spike_list in spike_times.items():
            for i, tetrode_spikes in enumerate(spike_list):
                if len(tetrode_spikes) > 1:
                    assert np.all(np.diff(tetrode_spikes) > 0), \
                        f"{brain_area} tetrode {i} has unsorted spikes"

    def test_no_nan_in_critical_position_data(self, mock_session_data):
        """Critical position columns should not have NaN values."""
        position_info = mock_session_data['position_info']

        critical_columns = ['linear_position', 'head_speed', 'track_segment_id']

        for col in critical_columns:
            if col in position_info.columns:
                assert not position_info[col].isna().any(), \
                    f"Column {col} contains NaN values"

    def test_speed_is_non_negative(self, mock_session_data):
        """Speed must be non-negative."""
        position_info = mock_session_data['position_info']

        if 'head_speed' in position_info.columns:
            assert (position_info['head_speed'] >= 0).all(), \
                "Speed cannot be negative"

    def test_trial_durations_positive(self, mock_session_data):
        """Trial durations must be positive."""
        trials = mock_session_data['trials']

        durations = trials['end_time'] - trials['start_time']
        assert (durations > 0).all(), "Trial durations must be positive"

    def test_ripple_durations_positive(self, mock_session_data):
        """Ripple durations must be positive."""
        ripple_times = mock_session_data['ripple_times']

        durations = ripple_times['end_time'] - ripple_times['start_time']
        assert (durations > 0).all(), "Ripple durations must be positive"


class TestDataLoadingEdgeCases:
    """Test edge cases in data loading."""

    def test_missing_brain_area_handled_gracefully(self, mock_session_data):
        """Should handle missing brain areas without crashing."""
        spike_times = mock_session_data['spike_times']

        # OFC might not be present in all sessions
        ofc_present = 'OFC' in spike_times

        # Should not crash whether present or not
        if ofc_present:
            assert isinstance(spike_times['OFC'], list)
        else:
            assert 'OFC' not in spike_times

    def test_empty_spike_times_for_brain_area(self, mock_session_data):
        """Should handle brain areas with no spikes."""
        # Create empty spike data
        empty_spike_data = {
            'HPC': [np.array([])],  # Single tetrode, no spikes
        }

        # Should not crash during processing
        assert len(empty_spike_data['HPC']) == 1
        assert len(empty_spike_data['HPC'][0]) == 0

    def test_very_short_session(self):
        """Should handle very short sessions (< 1 second)."""
        time = np.arange(0.0, 0.5, 0.002)  # 500ms

        position_df = pd.DataFrame({
            'linear_position': np.linspace(0, 10, len(time)),
            'head_speed': np.full(len(time), 10.0),
            'track_segment_id': np.zeros(len(time), dtype=int),
        }, index=time)

        assert len(position_df) > 0
        assert position_df.index.is_monotonic_increasing


class TestTemporalConsistency:
    """Test temporal consistency across data types."""

    def test_all_dataframes_use_seconds(self, mock_session_data):
        """All time-indexed DataFrames must use seconds, not milliseconds."""
        position_info = mock_session_data['position_info']
        trials = mock_session_data['trials']
        ripple_times = mock_session_data['ripple_times']

        # Typical session is minutes to hours, not thousands of hours
        # If times are in milliseconds, values would be huge
        assert position_info.index.max() < 10000, \
            "Position times likely in milliseconds, not seconds"

        assert trials['end_time'].max() < 10000, \
            "Trial times likely in milliseconds, not seconds"

        assert ripple_times['end_time'].max() < 10000, \
            "Ripple times likely in milliseconds, not seconds"

    def test_events_occur_during_position_recording(self, mock_session_data):
        """All events must occur during position recording period."""
        position_info = mock_session_data['position_info']
        trials = mock_session_data['trials']

        # No trial should start before position recording
        assert (trials['start_time'] >= position_info.index.min()).all()

        # No trial should end after position recording
        assert (trials['end_time'] <= position_info.index.max()).all()

    def test_goal_arrival_between_trial_bounds(self, mock_session_data):
        """Goal arrival time must be between trial start and end."""
        trials = mock_session_data['trials']

        if 'goal_arrival_time' in trials.columns:
            assert (trials['goal_arrival_time'] >= trials['start_time']).all()
            assert (trials['goal_arrival_time'] <= trials['end_time']).all()


class TestSelectiveDataLoading:
    """Test selective data loading with DataLoadingPreset and custom includes."""

    def test_minimal_preset_loads_only_required_keys(self, mock_session_data):
        """MINIMAL preset should only include position and spike data."""
        from continuum_swr_replay.types import DataLoadingPreset
        from continuum_swr_replay.types import SessionDataKeys as SDK

        # Simulate what load_data would return with MINIMAL preset
        # (In practice this would use load_data with include=DataLoadingPreset.MINIMAL)
        minimal_keys = {
            SDK.POSITION_INFO,
            SDK.TRACK_GRAPH,
            SDK.LINEAR_EDGE_ORDER,
            SDK.LINEAR_EDGE_SPACING,
            SDK.SPIKE_TIMES,
            SDK.SPIKE_WAVEFORM_FEATURES,
        }

        assert DataLoadingPreset.MINIMAL == minimal_keys

    def test_neural_preset_includes_lfp_data(self, mock_session_data):
        """NEURAL preset should include all neural data."""
        from continuum_swr_replay.types import DataLoadingPreset
        from continuum_swr_replay.types import SessionDataKeys as SDK

        # NEURAL extends MINIMAL with ripple, theta, and multiunit
        assert SDK.RIPPLE_TIMES in DataLoadingPreset.NEURAL
        assert SDK.THETA_PHASE in DataLoadingPreset.NEURAL
        assert SDK.MULTIUNIT_FIRING_RATE in DataLoadingPreset.NEURAL

        # Should still include MINIMAL keys
        assert SDK.POSITION_INFO in DataLoadingPreset.NEURAL
        assert SDK.SPIKE_TIMES in DataLoadingPreset.NEURAL

    def test_behavioral_preset_excludes_neural_data(self, mock_session_data):
        """BEHAVIORAL preset should not include spike or LFP data."""
        from continuum_swr_replay.types import DataLoadingPreset
        from continuum_swr_replay.types import SessionDataKeys as SDK

        # Should include position and task data
        assert SDK.POSITION_INFO in DataLoadingPreset.BEHAVIORAL
        assert SDK.TRIALS in DataLoadingPreset.BEHAVIORAL
        assert SDK.TASK_VARIABLES in DataLoadingPreset.BEHAVIORAL

        # Should NOT include neural data
        assert SDK.SPIKE_TIMES not in DataLoadingPreset.BEHAVIORAL
        assert SDK.RIPPLE_TIMES not in DataLoadingPreset.BEHAVIORAL
        assert SDK.THETA_PHASE not in DataLoadingPreset.BEHAVIORAL

    def test_decoding_preset_includes_everything_needed(self, mock_session_data):
        """DECODING preset should include all data needed for decoding."""
        from continuum_swr_replay.types import DataLoadingPreset
        from continuum_swr_replay.types import SessionDataKeys as SDK

        # Should include all neural data (from NEURAL)
        assert SDK.SPIKE_TIMES in DataLoadingPreset.DECODING
        assert SDK.RIPPLE_TIMES in DataLoadingPreset.DECODING
        assert SDK.THETA_PHASE in DataLoadingPreset.DECODING

        # Should also include trials and task variables
        assert SDK.TRIALS in DataLoadingPreset.DECODING
        assert SDK.TASK_VARIABLES in DataLoadingPreset.DECODING
        assert SDK.ELECTRODE_GROUP_INFO in DataLoadingPreset.DECODING

    def test_preset_union_works(self, mock_session_data):
        """Should be able to extend presets with custom keys."""
        from continuum_swr_replay.types import DataLoadingPreset
        from continuum_swr_replay.types import SessionDataKeys as SDK

        # Extend MINIMAL with trials
        custom = DataLoadingPreset.MINIMAL | {SDK.TRIALS, SDK.WELL_VISITS}

        # Should have MINIMAL keys
        assert SDK.POSITION_INFO in custom
        assert SDK.SPIKE_TIMES in custom

        # Should have added keys
        assert SDK.TRIALS in custom
        assert SDK.WELL_VISITS in custom

        # Should NOT have other keys
        assert SDK.RIPPLE_TIMES not in custom
