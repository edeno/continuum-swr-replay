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
)
from continuum_swr_replay.analysis.pfc.glm import (
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
        # The 1 at index 4 shifted by 2 would land at index 6 — OOB,
        # so the column must be all-zero. Pin the *whole* expected
        # vector: a regression that leaks the value into any in-range
        # bin (e.g. via wrap-around or fill-with-edge) would still
        # have most slots equal to zero, so an
        # ``any-bin-is-zero``-style assertion silently passes.
        expected = np.zeros(5, dtype=float)
        np.testing.assert_array_equal(result[:, 0], expected)

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

    def test_absent_event_class_marks_dependent_outputs_as_unavailable(
        self, poisson_spike_times, is_immobile, theta_phase,
        time_axis, position_info, sampling_frequency,
    ):
        """Regression: ``_make_design_matrices`` builds spline
        regressors for every event class regardless of whether that
        class has any events. An empty event table leaves its spline
        columns all-zero, so the design matrix is rank-deficient
        against the intercept; statsmodels still produces finite
        zero parameters and finite degenerate LR / fractional-deviance
        values, and ``process_neuron``'s NaN-params check doesn't
        catch them.

        Post-fix: rank-deficient designs are detected up front. Any
        LR test or fractional deviance involving the affected models
        is NaN; modulation arrays sourced from a rank-deficient
        design are NaN; and AIC selection skips those models.
        """
        # Empty non_local_no_SWR. Other event classes are populated.
        event_times = {
            "SWR": pd.DataFrame({"start_time": [1.0, 5.0],
                                 "end_time": [1.1, 5.1]}),
            "non_local": pd.DataFrame({"start_time": [1.0, 5.0],
                                       "end_time": [1.1, 5.1]}),
            "non_local_SWR": pd.DataFrame({"start_time": [1.0, 5.0],
                                           "end_time": [1.1, 5.1]}),
            "non_local_no_SWR": pd.DataFrame(
                columns=["start_time", "end_time"]
            ),
        }
        dm, spline_basis, lag_time = _make_design_matrices(
            event_times, position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )
        stats, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod = process_neuron(
            poisson_spike_times, is_immobile, theta_phase,
            time_axis, dm, spline_basis, lag_time,
        )

        # Every test that touches the absent event class must be NaN.
        for affected in (
            "non_local_no_SWR vs. constant",
            "SWR + non_local_no_SWR vs. constant",
            "SWR + non_local_no_SWR vs. SWR",
            "SWR + non_local_no_SWR vs. non_local_no_SWR",
            "non_local_SWR + non_local_no_SWR vs. constant",
            "non_local_SWR + non_local_no_SWR vs. non_local_SWR",
            "non_local_SWR + non_local_no_SWR vs. non_local_no_SWR",
            "non_local_SWR + non_local_no_SWR vs. non_local",
        ):
            for suffix in (": lr", ": pval", ": frac_null_dev",
                           ": frac_explained_dev"):
                key = affected + suffix
                assert np.isnan(stats[key]), (
                    f"{key} = {stats[key]} but the model uses an empty "
                    "event class — the LR/dev statistic is not "
                    "estimable and must be NaN."
                )

        # Tests that don't touch the absent class stay finite.
        for unaffected in (
            "SWR vs. constant",
            "non_local vs. constant",
            "non_local_SWR vs. constant",
        ):
            assert np.isfinite(stats[unaffected + ": pval"])

        # Modulation curves sourced from rank-deficient designs are
        # all-NaN; the corresponding max-mod stats are NaN.
        assert np.all(np.isnan(swr_mod))
        assert np.all(np.isnan(no_swr_mod))
        assert np.all(np.isnan(nl_no_swr_mod))
        for stat in (
            "SWR_max_mod", "SWR_max_mod_time",
            "non_local_no_SWR_max_mod", "non_local_no_SWR_max_mod_time",
            "non_local_no_SWR_split_max_mod",
            "non_local_no_SWR_split_max_mod_time",
        ):
            assert np.isnan(stats[stat])
        # The replay-split model also depends on non_local_no_SWR, so
        # its non_local_SWR modulation curve is NaN too.
        assert np.all(np.isnan(nl_swr_mod))
        assert np.isnan(stats["non_local_SWR_max_mod"])
        assert np.isnan(stats["non_local_SWR_max_mod_time"])

        # AIC selectors must skip non-estimable models. With
        # non_local_no_SWR absent, "SWR + non_local_no_SWR" and
        # "non_local_SWR + non_local_no_SWR" drop out; the remaining
        # candidates are constant / SWR / non_local on the SWR side
        # and constant / non_local on the replay-split side.
        assert stats["min_aic_model"] in {"constant", "SWR", "non_local"}
        assert stats["min_aic_model_replay_split"] in {"constant", "non_local"}

    def test_non_nested_swr_vs_non_local_lr_test_is_not_emitted(
        self, poisson_spike_times, is_immobile, theta_phase,
        time_axis, simple_event_times, position_info, sampling_frequency,
    ):
        """Regression: ``("SWR + non_local_no_SWR", "non_local")`` is
        NOT a nested model pair in general. The full design carries
        ``SWR`` (= ``non_local_SWR + SWR_only``) plus
        ``non_local_no_SWR``; the reduced design carries ``non_local``
        (= ``non_local_SWR + non_local_no_SWR``). Recovering
        ``non_local`` from the full requires ``SWR_only == 0`` (no
        SWR-only events), which is the empty special case. Outside
        that case, the chi-square LR statistic has no valid null
        distribution and ``frac_*_dev`` no longer represents "unique
        deviance explained by the added regressors."

        The fix drops that comparison from ``process_neuron``'s
        ``test_cases`` so the invalid LR / fractional-deviance
        columns are no longer emitted. The valid sibling comparison
        ``("non_local_SWR + non_local_no_SWR", "non_local")`` (which
        IS nested) and the SWR-controlled comparisons against
        ``"SWR"`` and ``"non_local_no_SWR"`` are preserved.
        """
        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times, position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )
        stats, *_ = process_neuron(
            poisson_spike_times, is_immobile, theta_phase,
            time_axis, dm, spline_basis, lag_time,
        )

        invalid_test = "SWR + non_local_no_SWR vs. non_local"
        for suffix in (": lr", ": pval", ": frac_null_dev", ": frac_explained_dev"):
            key = invalid_test + suffix
            assert key not in stats, (
                f"non-nested comparison {invalid_test!r} still emitted "
                f"{key!r} — chi-square LR has no valid null distribution "
                "and 'unique deviance explained' loses its interpretation "
                "when SWR-only events exist."
            )

        # Sibling nested comparisons must still be reported.
        for valid_key in (
            "non_local_SWR + non_local_no_SWR vs. non_local: pval",
            "SWR + non_local_no_SWR vs. SWR: pval",
            "SWR + non_local_no_SWR vs. non_local_no_SWR: pval",
        ):
            assert valid_key in stats, (
                f"valid nested comparison {valid_key!r} was dropped by "
                "the fix — only the non-nested non_local comparison "
                "should be removed."
            )
