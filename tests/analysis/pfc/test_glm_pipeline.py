"""Tests for the GLM fitting pipeline.

Tests the core signal processing and model fitting functions used to analyze
PFC neuron modulation during hippocampal replay events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.glm import (
    _fit_immobility_models,
    _make_design_matrices,
    create_conv_basis,
    create_lagged_signal,
    get_frac_explained_deviance,
    get_frac_null_deviance,
    process_neuron,
    test_theta_modulation as run_theta_modulation_test,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sampling_frequency():
    return 500.0


@pytest.fixture
def time_axis(sampling_frequency):
    """10-second epoch at 500 Hz."""
    return np.arange(0, 10, 1.0 / sampling_frequency)


@pytest.fixture
def simple_event_times():
    """Two non-overlapping events for each event type."""
    return {
        "SWR": pd.DataFrame({
            "start_time": [1.0, 5.0],
            "end_time": [1.1, 5.1],
        }),
        "non_local": pd.DataFrame({
            "start_time": [1.0, 5.0, 7.0],
            "end_time": [1.1, 5.1, 7.1],
        }),
        "non_local_SWR": pd.DataFrame({
            "start_time": [1.0, 5.0],
            "end_time": [1.1, 5.1],
        }),
        "non_local_no_SWR": pd.DataFrame({
            "start_time": [7.0],
            "end_time": [7.1],
        }),
    }


@pytest.fixture
def position_info(time_axis):
    """Synthetic position info: immobile most of the time, running briefly."""
    speed = np.ones(len(time_axis)) * 2.0  # mostly immobile
    speed[2000:2500] = 10.0  # running from 4s to 5s
    return pd.DataFrame(
        {"head_speed": speed},
        index=time_axis,
    )


@pytest.fixture
def is_immobile(position_info):
    return position_info.head_speed < 4.0


@pytest.fixture
def theta_phase(time_axis):
    """Synthetic theta phase: 8 Hz oscillation."""
    phase = np.mod(2 * np.pi * 8 * time_axis, 2 * np.pi) - np.pi
    return pd.Series(phase, index=time_axis)


@pytest.fixture
def poisson_spike_times(time_axis):
    """Synthetic Poisson spike train at ~5 Hz."""
    rng = np.random.default_rng(42)
    dt = np.median(np.diff(time_axis))
    rate = 5.0  # Hz
    spikes = time_axis[rng.random(len(time_axis)) < rate * dt]
    return spikes


@pytest.fixture
def spike_count(poisson_spike_times, time_axis):
    from continuum_swr_replay.analysis.pfc.glm import get_spikecount_per_time_bin
    return get_spikecount_per_time_bin(poisson_spike_times, time_axis)


# ---------------------------------------------------------------------------
# create_lagged_signal
# ---------------------------------------------------------------------------


class TestCreateLaggedSignal:
    def test_output_shape(self):
        signal = np.array([0, 0, 1, 0, 0], dtype=float)
        lags = np.array([-1, 0, 1])
        result = create_lagged_signal(signal, lags)
        assert result.shape == (5, 3)

    def test_zero_lag_is_identity(self):
        signal = np.array([1, 2, 3, 4, 5], dtype=float)
        lags = np.array([0])
        result = create_lagged_signal(signal, lags)
        np.testing.assert_array_equal(result[:, 0], signal)

    def test_positive_lag_shifts_forward(self):
        signal = np.array([1, 0, 0, 0, 0], dtype=float)
        lags = np.array([2])
        result = create_lagged_signal(signal, lags)
        # Shift by 2 moves the spike to index 0+2=2... but shift(2) moves
        # the value DOWN by 2 rows (pandas convention)
        expected = np.array([0, 0, 1, 0, 0], dtype=float)
        np.testing.assert_array_equal(result[:, 0], expected)

    def test_out_of_bounds_filled_with_zero(self):
        signal = np.array([0, 0, 0, 0, 1], dtype=float)
        lags = np.array([2])
        result = create_lagged_signal(signal, lags)
        # The 1 at index 4 shifted by 2 would go to index 6, which is OOB
        # So only zeros remain in the valid range
        assert result[-1, 0] == 0  # shifted out
        assert result.sum() == 0 or result[-3, 0] == 0  # no data leaked

    def test_default_lags(self):
        signal = np.zeros(300)
        result = create_lagged_signal(signal)
        # Default is np.arange(-100, 101) = 201 lags
        assert result.shape == (300, 201)


# ---------------------------------------------------------------------------
# create_conv_basis
# ---------------------------------------------------------------------------


class TestCreateConvBasis:
    def test_output_shapes(self, sampling_frequency):
        n_time = 1000
        count = np.zeros(n_time)
        count[500] = 1  # single event at midpoint
        lags = np.arange(-50, 51)
        knots = np.arange(-0.08, 0.08, 0.02)[1:]

        conv_basis, spline_basis, lag_time = create_conv_basis(
            count, lags, knots, sampling_frequency
        )

        n_lags = len(lags)
        n_basis = spline_basis.shape[1]

        assert spline_basis.shape[0] == n_lags
        assert lag_time.shape == (n_lags,)
        assert conv_basis.shape == (n_time, n_basis)

    def test_lag_time_in_seconds(self, sampling_frequency):
        lags = np.arange(-50, 51)
        knots = np.arange(-0.08, 0.08, 0.02)[1:]
        count = np.zeros(200)

        _, _, lag_time = create_conv_basis(count, lags, knots, sampling_frequency)

        np.testing.assert_allclose(lag_time, lags / sampling_frequency)

    def test_zero_input_gives_zero_output(self, sampling_frequency):
        count = np.zeros(200)
        lags = np.arange(-20, 21)
        knots = np.array([-0.02, 0.0, 0.02])

        conv_basis, _, _ = create_conv_basis(count, lags, knots, sampling_frequency)

        np.testing.assert_array_equal(conv_basis, 0)

    def test_spline_basis_rows_sum_positive(self, sampling_frequency):
        """B-spline basis should have non-negative partition of unity."""
        count = np.zeros(200)
        lags = np.arange(-50, 51)
        knots = np.arange(-0.08, 0.08, 0.02)[1:]

        _, spline_basis, _ = create_conv_basis(count, lags, knots, sampling_frequency)

        # B-spline basis functions should be non-negative
        assert np.all(spline_basis >= -1e-10)


# ---------------------------------------------------------------------------
# _make_design_matrices
# ---------------------------------------------------------------------------


class TestMakeDesignMatrices:
    def test_output_structure(self, simple_event_times, position_info):
        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )

        expected_models = {
            "constant", "SWR", "non_local", "non_local_no_SWR",
            "non_local_SWR", "non_local_SWR + non_local_no_SWR",
            "SWR + non_local_no_SWR",
        }
        assert set(dm.keys()) == expected_models

    def test_constant_model_is_ones(self, simple_event_times, position_info):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        n_time = len(position_info)
        np.testing.assert_array_equal(dm["constant"], np.ones((n_time, 1)))

    def test_design_matrix_shapes_consistent(self, simple_event_times, position_info):
        dm, spline_basis, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        n_time = len(position_info)
        n_basis = spline_basis.shape[1]

        for name, matrix in dm.items():
            assert matrix.shape[0] == n_time, f"{name} has wrong n_time"
            # constant has 1 col, single-event has 1+n_basis, dual has 1+2*n_basis
            assert matrix.shape[1] >= 1, f"{name} has no columns"

        assert dm["SWR"].shape[1] == 1 + n_basis
        assert dm["SWR + non_local_no_SWR"].shape[1] == 1 + 2 * n_basis

    def test_empty_times_raises(self, position_info):
        with pytest.raises(ValueError, match="non-empty"):
            _make_design_matrices({}, position_info)


# ---------------------------------------------------------------------------
# _fit_immobility_models
# ---------------------------------------------------------------------------


class TestFitImmobilityModels:
    def test_returns_results_for_all_models(
        self, simple_event_times, position_info, is_immobile, spike_count
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        results, models = _fit_immobility_models(is_immobile, dm, spike_count)

        assert set(results.keys()) == set(dm.keys())
        assert set(models.keys()) == set(dm.keys())

    def test_results_have_aic(
        self, simple_event_times, position_info, is_immobile, spike_count
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        results, _ = _fit_immobility_models(is_immobile, dm, spike_count)

        for name, result in results.items():
            assert hasattr(result, "aic"), f"{name} result missing AIC"
            assert hasattr(result, "deviance"), f"{name} result missing deviance"
            assert np.isfinite(result.aic), f"{name} AIC is not finite"

    def test_more_complex_model_has_lower_or_equal_deviance(
        self, simple_event_times, position_info, is_immobile, spike_count
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        results, _ = _fit_immobility_models(is_immobile, dm, spike_count)

        # Nested: constant ⊂ SWR ⊂ SWR + non_local_no_SWR
        assert results["SWR"].deviance <= results["constant"].deviance + 1e-6
        assert (
            results["SWR + non_local_no_SWR"].deviance
            <= results["SWR"].deviance + 1e-6
        )


# ---------------------------------------------------------------------------
# Deviance fraction helpers
# ---------------------------------------------------------------------------


class TestDevianceFractions:
    def test_frac_null_deviance_positive_for_nested_models(
        self, simple_event_times, position_info, is_immobile, spike_count
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        results, _ = _fit_immobility_models(is_immobile, dm, spike_count)

        frac = get_frac_null_deviance(
            results["SWR + non_local_no_SWR"],
            results["SWR"],
            results["constant"],
        )
        assert frac >= -1e-10  # non-negative for nested models

    def test_frac_explained_deviance_bounded(
        self, simple_event_times, position_info, is_immobile, spike_count
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        results, _ = _fit_immobility_models(is_immobile, dm, spike_count)

        frac = get_frac_explained_deviance(
            results["SWR + non_local_no_SWR"],
            results["SWR"],
            results["constant"],
        )
        assert -1e-6 <= frac <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# test_theta_modulation
# ---------------------------------------------------------------------------


class TestThetaModulation:
    def test_returns_four_values(self, theta_phase, is_immobile, spike_count):
        lr, p, mod, phase = run_theta_modulation_test(
            theta_phase, is_immobile, spike_count,
        )
        assert np.isfinite(lr)
        assert 0 <= p <= 1
        assert mod >= 0
        assert -np.pi <= phase <= np.pi

    def test_unmodulated_neuron_has_low_modulation(self, is_immobile, time_axis):
        """A neuron with uniform firing should show weak theta modulation."""
        rng = np.random.default_rng(123)
        # Constant rate, no theta modulation
        spike_count = rng.poisson(0.01, len(time_axis))
        theta_phase = pd.Series(
            np.mod(2 * np.pi * 8 * time_axis, 2 * np.pi) - np.pi,
            index=time_axis,
        )

        lr, p, mod, _ = run_theta_modulation_test(
            theta_phase, is_immobile, spike_count,
        )
        # p-value should be high (not significant)
        assert p > 0.01 or mod < 0.5


# ---------------------------------------------------------------------------
# process_neuron (end-to-end)
# ---------------------------------------------------------------------------


class TestProcessNeuron:
    def test_returns_expected_structure(
        self, poisson_spike_times, is_immobile, theta_phase,
        time_axis, simple_event_times, position_info, sampling_frequency,
    ):
        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times, position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )

        stats, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod = process_neuron(
            poisson_spike_times, is_immobile, theta_phase,
            time_axis, dm, spline_basis, lag_time,
        )

        # stats should be a dict with firing rates
        assert "run_avg_firing_rate" in stats
        assert "immobility_avg_firing_rate" in stats
        assert stats["run_avg_firing_rate"] >= 0
        assert stats["immobility_avg_firing_rate"] >= 0

    def test_modulation_arrays_have_correct_shape(
        self, poisson_spike_times, is_immobile, theta_phase,
        time_axis, simple_event_times, position_info, sampling_frequency,
    ):
        lags = np.arange(-50, 51)
        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times, position_info,
            sampling_frequency=sampling_frequency,
            lags=lags,
        )

        _, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod = process_neuron(
            poisson_spike_times, is_immobile, theta_phase,
            time_axis, dm, spline_basis, lag_time,
        )

        n_lags = len(lags)
        assert swr_mod.shape == (n_lags,)
        assert no_swr_mod.shape == (n_lags,)
        assert nl_swr_mod.shape == (n_lags,)
        assert nl_no_swr_mod.shape == (n_lags,)

    def test_stats_contain_likelihood_ratio_tests(
        self, poisson_spike_times, is_immobile, theta_phase,
        time_axis, simple_event_times, position_info, sampling_frequency,
    ):
        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times, position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )

        stats, *_ = process_neuron(
            poisson_spike_times, is_immobile, theta_phase,
            time_axis, dm, spline_basis, lag_time,
        )

        # Check key LR tests exist
        assert "SWR vs. constant: lr" in stats
        assert "SWR vs. constant: pval" in stats
        assert "min_aic_model" in stats

        # p-values should be valid
        for key, val in stats.items():
            if key.endswith(": pval"):
                assert 0 <= val <= 1, f"{key} = {val} out of range"
