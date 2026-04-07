"""Unit tests for decoding quality metrics.

Tests each quality metric function with synthetic data_over_time DataFrames
that have known properties, verifying correct computation of error metrics,
state fractions, event context classification, and calibration curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.quality_metrics import (
    compute_calibration_curve,
    compute_error_by_track_segment,
    compute_local_decoding_error,
    compute_local_state_collapse_details,
    compute_no_spike_firing_correspondence,
    compute_non_local_during_movement,
    compute_non_local_event_context,
    compute_session_quality_summary,
    compute_state_fraction_during_movement,
)

N_BINS = 1000
SAMPLING_FREQUENCY = 500.0


@pytest.fixture
def perfect_session() -> pd.DataFrame:
    """Session with perfect local decoding: Local_prob=1, error=0."""
    n = N_BINS
    time = np.arange(n) / SAMPLING_FREQUENCY
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "Local_prob": np.ones(n),
            "No-Spike_prob": np.zeros(n),
            "Non-Local_Continuous_prob": np.zeros(n),
            "Non-Local_Fragmented_prob": np.zeros(n),
            "ahead_behind_distance": np.zeros(n),
            "map_estimate": np.linspace(0, 100, n),
            "linear_position": np.linspace(0, 100, n),
            "head_speed": np.full(n, 10.0),
            "track_segment_id": np.repeat([0, 1, 2, 3, 4], n // 5),
            "hpd": np.full(n, 5.0),
            "non_local_event": np.zeros(n, dtype=int),
            "SWR_event": np.zeros(n, dtype=int),
            "MUA_event": np.zeros(n, dtype=int),
            "theta_phase": rng.uniform(0, 2 * np.pi, n),
            "theta_power_zscore": rng.normal(0, 1, n),
            "SWR_zscore": rng.normal(0, 1, n),
            "MUA_zscore": rng.normal(0, 1, n),
            "animal": "mouse1",
            "day": 20200101,
            "epoch": "02_r1",
        },
        index=time,
    )


@pytest.fixture
def collapsed_session() -> pd.DataFrame:
    """Session where local state collapses during movement."""
    n = N_BINS
    time = np.arange(n) / SAMPLING_FREQUENCY

    local_prob = np.ones(n) * 0.9
    local_prob[200:600] = 0.1  # collapse during bins 200-599

    nonlocal_prob = 1.0 - local_prob
    error = np.zeros(n)
    error[200:600] = 50.0  # high error during collapse

    return pd.DataFrame(
        {
            "Local_prob": local_prob,
            "No-Spike_prob": np.zeros(n),
            "Non-Local_Continuous_prob": nonlocal_prob * 0.7,
            "Non-Local_Fragmented_prob": nonlocal_prob * 0.3,
            "ahead_behind_distance": error,
            "map_estimate": np.linspace(0, 100, n),
            "linear_position": np.linspace(0, 100, n),
            "head_speed": np.full(n, 10.0),
            "track_segment_id": np.repeat([0, 1, 2, 3, 4], n // 5),
            "hpd": np.where(local_prob > 0.5, 5.0, 50.0),
            "non_local_event": np.zeros(n, dtype=int),
            "SWR_event": np.zeros(n, dtype=int),
            "MUA_event": np.zeros(n, dtype=int),
            "theta_phase": np.zeros(n),
            "theta_power_zscore": np.zeros(n),
            "SWR_zscore": np.zeros(n),
            "MUA_zscore": np.zeros(n),
            "animal": "mouse2",
            "day": 20200102,
            "epoch": "02_r1",
        },
        index=time,
    )


@pytest.fixture
def session_with_events() -> pd.DataFrame:
    """Session with non-local events in different neural contexts."""
    n = N_BINS
    time = np.arange(n) / SAMPLING_FREQUENCY

    non_local_event = np.zeros(n, dtype=int)
    swr_event = np.zeros(n, dtype=int)
    theta_power = np.zeros(n)

    # Event 1: during SWR (bins 100-120)
    non_local_event[100:121] = 1
    swr_event[100:121] = 1

    # Event 2: during high theta (bins 300-320)
    non_local_event[300:321] = 2
    theta_power[300:321] = 3.0

    # Event 3: neither (bins 500-520)
    non_local_event[500:521] = 3

    return pd.DataFrame(
        {
            "Local_prob": np.where(non_local_event > 0, 0.1, 0.9),
            "No-Spike_prob": np.full(n, 0.05),
            "Non-Local_Continuous_prob": np.where(non_local_event > 0, 0.7, 0.05),
            "Non-Local_Fragmented_prob": np.where(non_local_event > 0, 0.15, 0.0),
            "ahead_behind_distance": np.zeros(n),
            "map_estimate": np.linspace(0, 100, n),
            "linear_position": np.linspace(0, 100, n),
            "head_speed": np.full(n, 2.0),  # immobile
            "track_segment_id": np.repeat([0, 1, 2, 3, 4], n // 5),
            "hpd": np.full(n, 5.0),
            "non_local_event": non_local_event,
            "SWR_event": swr_event,
            "MUA_event": np.zeros(n, dtype=int),
            "theta_phase": np.full(n, np.pi),  # all at pi
            "theta_power_zscore": theta_power,
            "SWR_zscore": np.zeros(n),
            "MUA_zscore": np.zeros(n),
            "animal": "mouse3",
            "day": 20200103,
            "epoch": "02_r1",
        },
        index=time,
    )


class TestComputeLocalDecodingError:
    def test_perfect_session_has_zero_error(self, perfect_session):
        result = compute_local_decoding_error(perfect_session)
        assert len(result) == 1
        assert result["median_error_cm"].iloc[0] == pytest.approx(0.0)
        assert result["p90_error_cm"].iloc[0] == pytest.approx(0.0)

    def test_collapsed_session_excludes_bad_bins(self, collapsed_session):
        result = compute_local_decoding_error(collapsed_session, local_prob_threshold=0.5)
        assert result["median_error_cm"].iloc[0] == pytest.approx(0.0)
        assert result["n_local_bins"].iloc[0] < N_BINS

    def test_higher_threshold_fewer_bins(self, collapsed_session):
        low = compute_local_decoding_error(collapsed_session, local_prob_threshold=0.3)
        high = compute_local_decoding_error(collapsed_session, local_prob_threshold=0.8)
        assert high["n_local_bins"].iloc[0] <= low["n_local_bins"].iloc[0]

    def test_empty_data_returns_empty_df(self):
        empty = pd.DataFrame(
            columns=[
                "Local_prob", "ahead_behind_distance",
                "animal", "day", "epoch",
            ]
        )
        result = compute_local_decoding_error(empty)
        assert len(result) == 0


class TestComputeStateFractionDuringMovement:
    def test_perfect_session_local_fraction_is_one(self, perfect_session):
        result = compute_state_fraction_during_movement(
            perfect_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert result["local_fraction"].iloc[0] == pytest.approx(1.0)
        assert result["max_collapse_duration_sec"].iloc[0] == pytest.approx(0.0)
        assert result["n_collapse_episodes"].iloc[0] == 0

    def test_collapsed_session_detects_collapse(self, collapsed_session):
        result = compute_state_fraction_during_movement(
            collapsed_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        # local_fraction is mean of Local_prob during movement: (0.9*600 + 0.1*400)/1000 = 0.58
        assert result["local_fraction"].iloc[0] < 1.0
        assert result["max_collapse_duration_sec"].iloc[0] > 0
        assert result["n_collapse_episodes"].iloc[0] >= 1
        # Collapse spans bins 200-599: elapsed time = (599-200)/500 + 1/500 = 0.8 sec
        assert result["max_collapse_duration_sec"].iloc[0] == pytest.approx(0.8)

    def test_immobile_session_excluded(self, session_with_events):
        """Session where animal is immobile should have no moving bins."""
        result = compute_state_fraction_during_movement(
            session_with_events, speed_threshold=4.0,
        )
        assert len(result) == 0


class TestComputeNonLocalDuringMovement:
    def test_no_nonlocal_during_movement(self, perfect_session):
        result = compute_non_local_during_movement(
            perfect_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert result["n_nonlocal_movement_events"].iloc[0] == 0
        assert result["nonlocal_movement_fraction"].iloc[0] == pytest.approx(0.0)

    def test_detects_nonlocal_during_movement(self, collapsed_session):
        """Collapsed session has high non-local prob during movement."""
        result = compute_non_local_during_movement(
            collapsed_session, non_local_threshold=0.5,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert result["n_nonlocal_movement_events"].iloc[0] >= 1
        assert result["nonlocal_movement_fraction"].iloc[0] > 0


class TestComputeErrorByTrackSegment:
    def test_perfect_session_all_segments_zero_error(self, perfect_session):
        result = compute_error_by_track_segment(perfect_session)
        assert (result["median_error_cm"] == 0.0).all()
        assert len(result) == 5  # 5 segments

    def test_segment_fractions_sum_to_one(self, perfect_session):
        result = compute_error_by_track_segment(perfect_session)
        assert result["segment_fraction"].sum() == pytest.approx(1.0)


class TestComputeNoSpikeFiringCorrespondence:
    def test_anticorrelated_nospike_and_mua(self):
        """When no-spike is high and MUA is low, correlation should be negative."""
        n = 1000
        rng = np.random.default_rng(42)
        mua = rng.normal(0, 1, n)
        nospike = -mua + rng.normal(0, 0.1, n)
        nospike = np.clip(nospike, 0, 1)

        data = pd.DataFrame({
            "No-Spike_prob": nospike,
            "MUA_zscore": mua,
            "animal": "mouse1",
            "day": 20200101,
            "epoch": "02_r1",
        })
        result = compute_no_spike_firing_correspondence(data)
        assert result["nospike_mua_rho"].iloc[0] < 0

    def test_positive_correlation_detected(self):
        """Positive correlation should be flagged."""
        n = 1000
        rng = np.random.default_rng(42)
        mua = rng.normal(0, 1, n)
        nospike = mua + rng.normal(0, 0.1, n)
        nospike = np.clip(nospike, 0, 1)

        data = pd.DataFrame({
            "No-Spike_prob": nospike,
            "MUA_zscore": mua,
            "animal": "mouse1",
            "day": 20200101,
            "epoch": "02_r1",
        })
        result = compute_no_spike_firing_correspondence(data)
        assert result["nospike_mua_rho"].iloc[0] > 0


class TestComputeNonLocalEventContext:
    def test_classifies_swr_theta_neither(self, session_with_events):
        result = compute_non_local_event_context(
            session_with_events, theta_power_threshold=2.0,
        )
        assert result["n_non_local_events"].iloc[0] == 3
        assert result["frac_swr"].iloc[0] == pytest.approx(1 / 3)
        assert result["frac_theta"].iloc[0] == pytest.approx(1 / 3)
        assert result["frac_neither"].iloc[0] == pytest.approx(1 / 3)

    def test_fractions_sum_to_one(self, session_with_events):
        result = compute_non_local_event_context(session_with_events)
        total = result["frac_swr"].iloc[0] + result["frac_theta"].iloc[0] + result["frac_neither"].iloc[0]
        assert total == pytest.approx(1.0)

    def test_no_events_returns_empty(self, perfect_session):
        result = compute_non_local_event_context(perfect_session)
        assert len(result) == 0


class TestComputeLocalStateCollapseDetails:
    def test_no_collapse_in_perfect_session(self, perfect_session):
        result = compute_local_state_collapse_details(
            perfect_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert len(result) == 0

    def test_detects_collapse_episode(self, collapsed_session):
        result = compute_local_state_collapse_details(
            collapsed_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert len(result) >= 1
        episode = result.iloc[0]
        assert episode["duration_sec"] > 0
        assert episode["mean_error_cm"] > 0
        assert episode["classification"] in ("uncertain", "confidently_wrong")

    def test_collapse_classified_as_uncertain_when_hpd_high(self, collapsed_session):
        """Collapse region has hpd=50, rest has hpd=5, so session median ~ 5.
        Collapse hpd (50) > median (5) -> uncertain."""
        result = compute_local_state_collapse_details(
            collapsed_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert result.iloc[0]["classification"] == "uncertain"


class TestComputeCalibrationCurve:
    def test_perfect_session_monotonic(self, perfect_session):
        result = compute_calibration_curve(perfect_session)
        assert result["is_monotonic_decreasing"].iloc[0] is True or result["is_monotonic_decreasing"].iloc[0] == True  # noqa: E712

    def test_negative_slope_for_good_decoder(self):
        """Higher Local_prob should mean lower error."""
        n = 10000
        rng = np.random.default_rng(42)
        local_prob = rng.uniform(0, 1, n)
        error = (1 - local_prob) * 50 + rng.normal(0, 1, n)

        data = pd.DataFrame({
            "Local_prob": local_prob,
            "ahead_behind_distance": error,
            "animal": "mouse1",
            "day": 20200101,
            "epoch": "02_r1",
        })
        result = compute_calibration_curve(data)
        assert result["calibration_slope"].iloc[0] < 0


class TestComputeSessionQualitySummary:
    def test_perfect_session_no_flags(self, perfect_session):
        result = compute_session_quality_summary(
            perfect_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert len(result) == 1
        assert result["n_flags"].iloc[0] == 0

    def test_collapsed_session_has_collapse_detected(self, collapsed_session):
        """Collapsed session should show collapse in state fractions even if
        flags don't trigger (local fraction may still be > 0.5 overall)."""
        result = compute_session_quality_summary(
            collapsed_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        # local_fraction < 1 because 400/1000 bins are collapsed
        assert result["local_fraction"].iloc[0] < 1.0
        assert result["max_collapse_duration_sec"].iloc[0] > 0

    def test_multi_session(self, perfect_session, collapsed_session):
        combined = pd.concat([perfect_session, collapsed_session])
        result = compute_session_quality_summary(
            combined, sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert len(result) == 2

    def test_output_has_continuous_metrics(self, perfect_session):
        result = compute_session_quality_summary(
            perfect_session, sampling_frequency=SAMPLING_FREQUENCY,
        )
        expected_cols = [
            "median_error_cm", "local_fraction", "nospike_mua_rho",
            "calibration_slope", "n_flags",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
