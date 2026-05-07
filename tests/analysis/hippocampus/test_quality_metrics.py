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
        assert result["n_events"].iloc[0] == 3
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

    def test_one_event_with_nan_theta_phase_does_not_poison_session_stats(self):
        """Regression: a single event with all-NaN ``theta_phase`` rows
        used to propagate NaN through the per-event circular mean and
        then through the session-level
        ``np.mean(np.exp(1j * phases))``, returning
        ``theta_mrl=NaN`` and ``theta_mean_angle=NaN`` even when the
        other theta events had clean phase samples. ``_nearest`` in
        ``data_aggregation`` returns NaN when no theta sample is within
        tolerance, so this is a routine occurrence — and it silently
        masked otherwise-valid theta locking.

        Post-fix ``_circular_mean`` drops non-finite phases, and the
        session-level reduction filters to events with finite per-event
        mean phase before applying the ``>= 3`` threshold.
        """
        rows = []
        for ev_id in (1, 2, 3, 4):
            for _ in range(5):
                phase = np.nan if ev_id == 4 else (0.5 + 0.05 * (ev_id - 1))
                rows.append(
                    {
                        "animal": "a",
                        "day": 1,
                        "epoch": "e1",
                        "non_local_event": ev_id,
                        "SWR_event": 0,
                        "theta_power_zscore": 1.5,
                        "theta_phase": phase,
                    }
                )
        df = pd.DataFrame(rows)
        result = compute_non_local_event_context(
            df,
            event_col="non_local_event",
            theta_power_threshold=0.5,
        )
        # All four events classify as theta (non-SWR + theta_power > 0.5).
        assert result["frac_theta"].iloc[0] == pytest.approx(1.0)
        # The three surviving phases are all close, so MRL should be ~1.
        assert np.isfinite(result["theta_mrl"].iloc[0]), (
            f"theta_mrl={result['theta_mrl'].iloc[0]} — one event's "
            "missing theta_phase poisoned the session circular mean."
        )
        assert result["theta_mrl"].iloc[0] > 0.99
        assert np.isfinite(result["theta_mean_angle"].iloc[0])

    def test_per_event_circular_mean_drops_internal_nans(self):
        """Within a single event, NaN phase samples (e.g. some bins
        had no theta sample within tolerance) must not poison the
        per-event ``circular_mean_theta_phase``."""
        rows = []
        # Event 1: 5 samples, two are NaN — should average over the
        # three finite samples.
        finite_phases = [0.5, 0.55, 0.6]
        sample_phases = [finite_phases[0], np.nan, finite_phases[1], np.nan, finite_phases[2]]
        for phase in sample_phases:
            rows.append(
                {
                    "animal": "a",
                    "day": 1,
                    "epoch": "e1",
                    "non_local_event": 1,
                    "SWR_event": 0,
                    "theta_power_zscore": 1.5,
                    "theta_phase": phase,
                }
            )
        # Three more clean events so the session reduction has the
        # required ``>= 3`` finite events.
        for ev_id in (2, 3, 4):
            for _ in range(5):
                rows.append(
                    {
                        "animal": "a",
                        "day": 1,
                        "epoch": "e1",
                        "non_local_event": ev_id,
                        "SWR_event": 0,
                        "theta_power_zscore": 1.5,
                        "theta_phase": 0.55,
                    }
                )
        df = pd.DataFrame(rows)
        result = compute_non_local_event_context(
            df,
            event_col="non_local_event",
            theta_power_threshold=0.5,
        )
        assert np.isfinite(result["theta_mrl"].iloc[0]), (
            "Per-event circular mean must drop NaN phase samples "
            "instead of returning NaN for the whole event."
        )
        assert result["theta_mrl"].iloc[0] > 0.99

    def test_boundary_touch_swr_does_not_inflate_frac_swr(self):
        """Regression: a SWR that only touches a non-local event at
        a boundary timestamp shares zero positive duration. The
        previous label-based ``has_swr=("SWR_event", lambda x: (x > 0).any())``
        aggregator counted that as overlap because ``get_labels``
        writes inclusive bin labels, inflating ``frac_swr``. Strict
        positive-duration semantics must classify it as theta- or
        neither-associated, not SWR-associated."""
        # NL labels [0.0, 1.0]; SWR labels [1.0, 2.0]; share t=1.0.
        df = pd.DataFrame(
            {
                "animal": ["a"] * 5,
                "day": [1] * 5,
                "epoch": ["e1"] * 5,
                "non_local_event": [1, 1, 1, 0, 0],
                "SWR_event": [0, 0, 2, 2, 2],
                "MUA_event": [0, 0, 0, 0, 0],
                "head_speed": [0.0] * 5,
                "theta_phase": [np.nan] * 5,
                "theta_power_zscore": [0.0] * 5,
            },
            index=[0.0, 0.5, 1.0, 1.5, 2.0],
        )
        out = compute_non_local_event_context(df)
        assert out["frac_swr"].iloc[0] == 0.0, (
            f"frac_swr={out['frac_swr'].iloc[0]} — boundary-touch SWR "
            "must not be classified as SWR-associated."
        )

    def test_all_nan_theta_phases_yield_nan_session_stats(self):
        """Sanity: if every theta event has only NaN phase samples,
        the session circular stats remain NaN (the gate fails because
        no finite per-event phase exists, regardless of event count)."""
        rows = []
        for ev_id in (1, 2, 3, 4):
            for _ in range(5):
                rows.append(
                    {
                        "animal": "a",
                        "day": 1,
                        "epoch": "e1",
                        "non_local_event": ev_id,
                        "SWR_event": 0,
                        "theta_power_zscore": 1.5,
                        "theta_phase": np.nan,
                    }
                )
        df = pd.DataFrame(rows)
        result = compute_non_local_event_context(
            df,
            event_col="non_local_event",
            theta_power_threshold=0.5,
        )
        assert np.isnan(result["theta_mrl"].iloc[0])
        assert np.isnan(result["theta_mean_angle"].iloc[0])


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

    def test_nan_error_bin_does_not_poison_slope(self):
        """Regression: a probability bin that contains only rows with
        NaN ``ahead_behind_distance`` (e.g. timepoints outside the
        linearized track) used to emit a row with
        ``median_error_cm = NaN`` and ``n_bins_in_group = 0``. That
        row then poisoned ``stats.linregress``, leaving every bin's
        ``calibration_slope`` NaN and ``is_monotonic_decreasing``
        False — so a session with otherwise usable calibration bins
        lost the slope diagnostic, and
        ``compute_session_quality_summary``'s
        ``flag_poor_calibration`` silently never tripped. The fix
        drops NaN-error rows before bin aggregation so each emitted
        bin reflects actual decoded errors.
        """
        df = pd.DataFrame(
            {
                "animal": ["a"] * 8,
                "day": [1] * 8,
                "epoch": ["e"] * 8,
                # Two finite-error bins (low and mid Local_prob);
                # the high bin contains only NaN-error rows.
                "Local_prob": [0.05, 0.10, 0.40, 0.60, 0.85, 0.95, 0.90, 0.92],
                "ahead_behind_distance": [
                    42.0, 40.0, 12.0, 10.0,
                    np.nan, np.nan, np.nan, np.nan,
                ],
            }
        )
        out = compute_calibration_curve(df, n_bins=3)
        # The all-NaN-error high bin must be dropped, leaving exactly
        # the two finite bins.
        assert len(out) == 2
        # Slope must be finite (and negative for a good calibration).
        slope = float(out["calibration_slope"].iloc[0])
        assert np.isfinite(slope), (
            f"calibration_slope={slope} — NaN-error bin still poisons "
            "the linregress over the surviving finite bins."
        )
        assert slope < 0
        # The two surviving bins are monotonically decreasing
        # (41 cm at low p → 11 cm at mid p), so the diagnostic must
        # report True.
        assert bool(out["is_monotonic_decreasing"].iloc[0]) is True


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
