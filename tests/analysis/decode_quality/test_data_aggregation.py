"""Tests for data aggregation (get_data_over_time).

Tests the primary function that bridges decoding results to downstream analysis,
combining event detection, posterior statistics, and behavioral data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.decode_quality.data_aggregation import (
    get_data_over_time,
    validate_dot_event_contract,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_time():
    """500 Hz time array, 2 seconds."""
    return np.arange(0.0, 2.0, 0.002)


@pytest.fixture
def mock_position_info(mock_time):
    """Position data matching the time array."""
    n = len(mock_time)
    return pd.DataFrame(
        {
            "head_position_x": np.linspace(0, 100, n),
            "head_position_y": np.zeros(n),
            "head_speed": np.full(n, 2.0),  # Below the 4 cm/s immobility threshold
            "head_orientation": np.zeros(n),
            "linear_position": np.linspace(0, 100, n),
            "projected_x_position": np.linspace(0, 100, n),
            "projected_y_position": np.zeros(n),
            "track_segment_id": np.zeros(n, dtype=int),
        },
        index=mock_time,
    )


@pytest.fixture
def mock_results(mock_time):
    """Minimal xarray Dataset mimicking decoding results.

    Creates state probabilities and posterior distributions
    with the structure expected by get_data_over_time.
    """
    n_time = len(mock_time)
    n_position = 50
    states = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]

    # State probabilities: mostly local
    probs = np.zeros((n_time, len(states)))
    probs[:, 0] = 0.7  # Local
    probs[:, 1] = 0.1  # No-Spike
    probs[:, 2] = 0.15  # Non-Local Continuous
    probs[:, 3] = 0.05  # Non-Local Fragmented

    # Create a non-local event (prob > 0.8) in the middle
    event_slice = slice(400, 430)  # 30 samples = 60ms
    probs[event_slice, 0] = 0.05
    probs[event_slice, 1] = 0.05
    probs[event_slice, 2] = 0.7
    probs[event_slice, 3] = 0.2

    acausal_state_probs = xr.DataArray(
        probs,
        dims=["time", "states"],
        coords={"time": mock_time, "states": states},
    )

    # Posterior: uniform over position for simplicity
    # Need state dim with "Non-Local Continuous" and "Non-Local Fragmented"
    posterior_states = ["Non-Local Continuous", "Non-Local Fragmented"]
    posterior = np.ones((n_time, len(posterior_states), n_position)) / n_position
    # Give non-local continuous most of the mass
    posterior[:, 0, :] *= 0.8
    posterior[:, 1, :] *= 0.2

    acausal_posterior = xr.DataArray(
        posterior,
        dims=["time", "state", "position"],
        coords={
            "time": mock_time,
            "state": posterior_states,
            "position": np.linspace(0, 100, n_position),
        },
    )

    return xr.Dataset(
        {
            "acausal_state_probabilities": acausal_state_probs,
            "acausal_posterior": acausal_posterior,
        }
    )


@pytest.fixture
def mock_session_data_for_aggregation(mock_position_info, mock_time):
    """Session data dict with all keys required by get_data_over_time."""
    n = len(mock_time)
    return {
        "position_info": mock_position_info,
        "ripple_times": pd.DataFrame(
            {
                "start_time": [0.8, 1.5],
                "end_time": [0.85, 1.55],
                "duration": [0.05, 0.05],
            }
        ),
        "hse_times": {
            "HPC": pd.DataFrame(
                {
                    "start_time": [0.8],
                    "end_time": [0.85],
                    "duration": [0.05],
                }
            )
        },
        "theta_phase": pd.Series(
            np.sin(2 * np.pi * 8 * mock_time),
            index=mock_time,
            name="theta_phase",
        ),
        "theta_power": pd.Series(
            np.ones(n) + np.random.RandomState(42).randn(n) * 0.1,
            index=mock_time,
            name="theta_power",
        ),
        "zscored_ripple_consensus_trace": pd.Series(
            np.random.RandomState(43).randn(n),
            index=mock_time,
            name="zscored_ripple_consensus_trace",
        ),
        "ripple_consensus_trace": pd.Series(
            np.abs(np.random.RandomState(45).randn(n)),
            index=mock_time,
            name="ripple_consensus_trace",
        ),
        "multiunit_rate_zscore": {
            "HPC": pd.DataFrame(
                {"firing_rate": np.random.RandomState(44).randn(n)},
                index=mock_time,
            )
        },
        "multiunit_firing_rate": {
            "HPC": pd.DataFrame(
                {"firing_rate": np.abs(np.random.RandomState(46).randn(n))},
                index=mock_time,
            )
        },
        "track_graph": _make_simple_track_graph(),
    }


def _make_simple_track_graph():
    """Create a minimal track graph for posterior stats."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    nx.set_edge_attributes(graph, {(0, 1): 50.0, (1, 2): 50.0}, "distance")
    graph.nodes[0]["pos"] = (0, 0)
    graph.nodes[1]["pos"] = (50, 0)
    graph.nodes[2]["pos"] = (100, 0)
    graph.edges[(0, 1)]["edge_id"] = 0
    graph.edges[(1, 2)]["edge_id"] = 1
    return graph


@pytest.fixture
def mock_environment():
    """Minimal Environment mock for posterior stats.

    Uses a mock object since constructing a real Environment
    requires the full non_local_detector pipeline.
    """
    from unittest.mock import MagicMock

    env = MagicMock()
    # These attributes are accessed by get_posterior_stats
    n_bins = 50
    env.edge_spacing = np.array([0.0])
    env.place_bin_edges_ = np.linspace(0, 100, n_bins + 1).reshape(1, -1)
    env.is_track_interior_ = np.ones(n_bins, dtype=bool)

    # For MAP speed computation
    track_graph = _make_simple_track_graph()
    env.track_graph_with_bin_centers_edges_ = track_graph
    env.place_bin_centers_nodes_df_ = pd.DataFrame(
        {"node_id": np.zeros(n_bins, dtype=int)}
    )
    return env


# ============================================================================
# Tests: Input Validation
# ============================================================================


class TestGetDataOverTimeValidation:
    """Test input validation for get_data_over_time."""

    def test_missing_required_keys_raises_key_error(
        self, mock_results, mock_environment
    ):
        """Missing data keys should raise KeyError with helpful message."""
        incomplete_data = {
            "position_info": pd.DataFrame({"head_speed": [1.0]}, index=[0.0]),
            # Missing: ripple_times, hse_times, theta_phase, etc.
        }

        with pytest.raises(KeyError, match="Missing required data keys"):
            get_data_over_time(
                incomplete_data,
                mock_results,
                mock_environment,
                "chimi20200212_.nwb",
                "02_r1",
            )

    def test_missing_single_key_names_it(
        self, mock_session_data_for_aggregation, mock_results, mock_environment
    ):
        """Error message should name the specific missing key."""
        data = dict(mock_session_data_for_aggregation)
        del data["theta_phase"]

        with pytest.raises(KeyError, match="theta_phase"):
            get_data_over_time(
                data,
                mock_results,
                mock_environment,
                "chimi20200212_.nwb",
                "02_r1",
            )

    def test_error_message_mentions_preset(self, mock_results, mock_environment):
        """Error should mention DataLoadingPreset.DECODING for guidance."""
        data = {"position_info": pd.DataFrame({"head_speed": [1.0]}, index=[0.0])}

        with pytest.raises(KeyError, match="DataLoadingPreset.DECODING"):
            get_data_over_time(
                data,
                mock_results,
                mock_environment,
                "chimi20200212_.nwb",
                "02_r1",
            )

    def test_all_required_keys_present_does_not_raise_validation(
        self, mock_session_data_for_aggregation
    ):
        """Complete data dict passes the key validator with no error.

        Targets the extracted validator
        :func:`validate_required_session_data_keys` directly so the
        test exercises only the validation contract — no need to
        instantiate the full aggregation pipeline (which would
        require a real ``Environment``) and no need to swallow a
        bag of broad downstream exceptions that could mask
        unrelated regressions in the aggregator. The validator is
        the same one ``get_data_over_time`` calls, so this is a
        production-path test, not a re-typed snippet.
        """
        from continuum_swr_replay.analysis.decode_quality.data_aggregation import (
            validate_required_session_data_keys,
        )

        validate_required_session_data_keys(
            mock_session_data_for_aggregation,
            required_keys=[
                "position_info",
                "ripple_times",
                "hse_times",
                "theta_phase",
                "theta_power",
                "zscored_ripple_consensus_trace",
                "multiunit_rate_zscore",
            ],
        )


# ============================================================================
# Tests: Producer output columns (step_distance / run_speed_baseline)
# ============================================================================


class TestGetDataOverTimeOutputColumns:
    """The aggregator's new per-session run-speed baseline must fail loud on
    degenerate input rather than emit a NaN column.

    (A full output-column assertion for ``step_distance`` / ``run_speed_baseline``
    would need a richer ``Environment`` fixture than the minimal MagicMock used
    here; ``step_distance`` is covered directly in ``test_posterior_analysis``.)
    """

    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test_run_speed_baseline_all_nan_head_speed_raises(
        self, mock_session_data_for_aggregation, mock_results, mock_environment
    ):
        """All-NaN head speed must fail loud, not emit a NaN baseline column."""
        data = dict(mock_session_data_for_aggregation)
        pos = data["position_info"].copy()
        pos["head_speed"] = np.nan
        data["position_info"] = pos
        with pytest.raises(ValueError, match="run_speed_baseline is not finite"):
            get_data_over_time(
                data, mock_results, mock_environment, "chimi20200212_.nwb", "02_r1"
            )


# ============================================================================
# validate_dot_event_contract — table-regen input-contract guard
# ============================================================================


def _clean_dot() -> pd.DataFrame:
    """Single session, time-sorted, one contiguous run per event label."""
    return pd.DataFrame(
        {
            "animal": ["chimi"] * 6,
            "day": [20200212] * 6,
            "epoch": ["02_r1"] * 6,
            "non_local_event": [0, 1, 1, 0, 2, 2],  # two contiguous events
        },
        index=np.arange(6, dtype=float),
    )


def test_validate_dot_contract_clean_passes():
    assert validate_dot_event_contract(_clean_dot()) == {}


def test_validate_dot_contract_rejects_multiple_sessions():
    dot = _clean_dot()
    dot.loc[3:, "epoch"] = "04_r2"  # two epochs in one frame
    with pytest.raises(ValueError, match="multiple_epoch|input-contract"):
        validate_dot_event_contract(dot)


def test_validate_dot_contract_rejects_unsorted_time():
    dot = _clean_dot()
    dot.index = [0.0, 1.0, 2.0, 1.5, 4.0, 5.0]  # time goes backwards once
    with pytest.raises(ValueError, match="unsorted_time|input-contract"):
        validate_dot_event_contract(dot)


def test_validate_dot_contract_rejects_duplicate_time():
    dot = _clean_dot()
    dot.index = [0.0, 1.0, 2.0, 2.0, 4.0, 5.0]
    with pytest.raises(ValueError, match="unsorted_time|input-contract"):
        validate_dot_event_contract(dot)


@pytest.mark.parametrize("bad_time", [np.nan, np.inf, -np.inf])
def test_validate_dot_contract_rejects_nonfinite_time(bad_time):
    dot = _clean_dot()
    index = dot.index.to_numpy(copy=True)
    index[2] = bad_time
    dot.index = index
    with pytest.raises(ValueError, match="nonfinite_time|input-contract"):
        validate_dot_event_contract(dot)


def test_validate_dot_contract_rejects_noncontiguous_label():
    dot = _clean_dot()
    # label 1 split by a gap (rows 1 and 3) -> non-contiguous event
    dot["non_local_event"] = [0, 1, 0, 1, 2, 2]
    with pytest.raises(ValueError, match="noncontiguous|input-contract"):
        validate_dot_event_contract(dot)
