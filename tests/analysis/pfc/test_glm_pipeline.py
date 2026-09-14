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
    make_matched_control_times,
    process_neuron,
)
from continuum_swr_replay.analysis.pfc.glm import (
    test_theta_modulation as run_theta_modulation_test,
)
from continuum_swr_replay.analysis.replay.cohort_selection import select_stop_cohort

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
    """Two non-overlapping events for each event type.

    ``matched_control`` carries one duration-matched non-event window per
    ``non_local`` event, in immobile time (the run window is 4-5 s) and clear
    of every real event, as :func:`make_matched_control_times` draws them.
    """
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
        "matched_control": pd.DataFrame(
            {
                "start_time": [2.0, 6.0, 8.0],
                "end_time": [2.1, 6.1, 8.1],
            }
        ),
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
            "matched_control", "matched_control + non_local_no_SWR",
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
        # Adding nested predictors can only explain *more* null deviance,
        # so the fraction must be non-negative. A function returning a
        # constant (e.g. 0.0) for any input would pass ``>= 0`` — pin
        # finiteness so a NaN/inf regression also trips here.
        assert np.isfinite(frac)
        assert frac >= -1e-10

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
        # Bounded fraction must be finite, in [0, 1], and (under the seeded
        # event fixture) strictly between 0 and 1 — a degenerate output
        # locked to 0 or 1 indicates broken nested-model accounting.
        assert np.isfinite(frac)
        assert -1e-6 <= frac <= 1.0 + 1e-6
        assert 0.0 < frac < 1.0


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
        # A function returning (p=0, mod=0) would pass the prior
        # disjunction (mod < 0.5 is True). Require *both* a non-significant
        # p-value AND a modest modulation depth so a degenerate output is
        # caught.
        assert p > 0.01 and mod < 0.5, f"p={p:.3f}, mod={mod:.3f}"


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

        stats, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod, mc_mod = process_neuron(
            poisson_spike_times, is_immobile, theta_phase,
            time_axis, dm, spline_basis, lag_time,
        )

        # Both rates must be strictly positive — the prior ``>= 0`` check
        # passed silently for a function returning constant 0. Magnitudes
        # are dominated by Poisson noise (the fixture's run window is only
        # 1 s of a 10 s epoch), so don't pin a specific Hz value.
        assert "run_avg_firing_rate" in stats
        assert "immobility_avg_firing_rate" in stats
        assert np.isfinite(stats["run_avg_firing_rate"])
        assert np.isfinite(stats["immobility_avg_firing_rate"])
        assert stats["run_avg_firing_rate"] > 0
        assert stats["immobility_avg_firing_rate"] > 0

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

        _, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod, mc_mod = process_neuron(
            poisson_spike_times, is_immobile, theta_phase,
            time_axis, dm, spline_basis, lag_time,
        )

        n_lags = len(lags)
        for arr in (swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod, mc_mod):
            assert arr.shape == (n_lags,)
            # A function returning all-NaN/all-zero arrays would pass the
            # shape check alone.
            assert np.isfinite(arr).all()

        # The four modulation arrays correspond to distinct event subsets
        # in the fixture; a regression that duplicates one array across
        # all four outputs would slip past shape-only checks.
        assert not np.array_equal(swr_mod, no_swr_mod)
        assert not np.array_equal(nl_swr_mod, nl_no_swr_mod)

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
            "matched_control": pd.DataFrame(
                {"start_time": [2.0, 6.0], "end_time": [2.1, 6.1]}
            ),
        }
        dm, spline_basis, lag_time = _make_design_matrices(
            event_times, position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )
        stats, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod, mc_mod = process_neuron(
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


# ---------------------------------------------------------------------------
# Stop-cohort restriction (fit mask and event sets)
# ---------------------------------------------------------------------------


@pytest.fixture
def session_stops():
    """One session's two disjoint stops: one well-anchored, one in transit.

    Both sit in immobile stretches of the ``position_info`` fixture (which runs
    only between 4 s and 5 s), so what the cohort drops is the stop rule and not
    the speed rule.
    """
    return pd.DataFrame(
        {
            "stop_id": ["s0", "s1"],
            "stop_start": [1.5, 6.0],
            "stop_end": [3.0, 7.0],
            "occupancy_state": ["at_well", "in_transit"],
        }
    )


@pytest.fixture
def cohort_stops(session_stops):
    """The kept stops, cut from ``session_stops`` by the shared cohort rule."""
    selected, _ = select_stop_cohort(session_stops, cohort="at_well")
    assert list(selected["stop_id"]) == ["s0"], (
        "the in-transit stop must be dropped by the cohort rule, or the tests "
        "below are not testing the restriction they claim to"
    )
    return selected


class TestBuildSessionFitMask:
    def test_fit_mask_is_immobility_inside_the_cohort_stop(
        self, position_info, is_immobile, cohort_stops
    ):
        from continuum_swr_replay.analysis.pfc.glm import build_session_fit_mask

        returned_immobile, fit_mask, counts = build_session_fit_mask(
            position_info, cohort_stops=cohort_stops, stop_cohort="at_well"
        )

        inside = (position_info.index >= 1.5) & (position_info.index <= 3.0)
        expected = is_immobile & pd.Series(inside, index=position_info.index)
        pd.testing.assert_series_equal(fit_mask, expected, check_names=False)
        pd.testing.assert_series_equal(
            returned_immobile, is_immobile, check_names=False
        )
        # The 1.5 s stop at 500 Hz is a small slice of the 10 s epoch: the
        # restriction must actually bite, not pass everything through.
        assert counts["n_bins_fit"] == int(expected.sum())
        assert counts["n_bins_fit"] < counts["n_bins_immobile"]
        assert counts["n_bins_immobile"] == int(is_immobile.sum())
        assert counts["n_bins_total"] == len(position_info)
        assert counts["stop_cohort"] == "at_well"
        # The dropped in-transit stop (6-7 s) is immobile throughout, so its
        # bins are excluded by the stop rule alone.
        in_dropped_stop = (position_info.index >= 6.0) & (position_info.index <= 7.0)
        assert is_immobile.to_numpy()[in_dropped_stop].all()
        assert not fit_mask.to_numpy()[in_dropped_stop].any()

    def test_all_stops_reproduces_the_immobility_mask_bit_for_bit(
        self, position_info, is_immobile
    ):
        from continuum_swr_replay.analysis.pfc.glm import build_session_fit_mask

        _, fit_mask, counts = build_session_fit_mask(
            position_info, stop_cohort="all_stops"
        )

        pd.testing.assert_series_equal(fit_mask, is_immobile, check_names=False)
        assert counts["n_bins_fit"] == counts["n_bins_immobile"]

    def test_a_cohort_without_stops_is_refused(self, position_info):
        from continuum_swr_replay.analysis.pfc.glm import build_session_fit_mask

        with pytest.raises(ValueError, match="cohort_stops"):
            build_session_fit_mask(position_info, stop_cohort="at_well")


def _canonical_events(event_windows, is_swr):
    """The session rows ``load_session_canonical_events`` would return."""
    return pd.DataFrame(
        {
            "event_id": np.arange(len(event_windows), dtype=np.uint64),
            "start_time": [start for start, _ in event_windows],
            "end_time": [end for _, end in event_windows],
            "is_swr": list(is_swr),
            "occupancy_state": ["at_well"] * len(event_windows),
        }
    )


def _swr_intervals(ripple_windows):
    """The intervals ``load_session_swr_intervals`` would return."""
    return pd.DataFrame(
        {
            "start_time": [start for start, _ in ripple_windows],
            "end_time": [end for _, end in ripple_windows],
        },
        dtype=float,
    )


def _synthetic_session(position_info, event_windows, ripple_windows):
    """A minimal ``(data, results)`` pair with non-local probability set high
    inside ``event_windows`` and ripples spanning ``ripple_windows``."""
    import xarray as xr

    time = position_info.index.to_numpy(dtype=float)
    prob = np.zeros((len(time), 3))
    prob[:, 0] = 1.0  # Local
    for start, end in event_windows:
        inside = (time >= start) & (time <= end)
        prob[inside, 0] = 0.0
        prob[inside, 1] = 1.0  # Non-Local Continuous
    results = xr.Dataset(
        {
            "acausal_state_probabilities": (
                ("time", "states"),
                prob,
            )
        },
        coords={
            "time": time,
            "states": [
                "Local",
                "Non-Local Continuous",
                "Non-Local Fragmented",
            ],
        },
    )
    data = {
        "position_info": position_info,
        "ripple_times": pd.DataFrame(
            {
                "start_time": [start for start, _ in ripple_windows],
                "end_time": [end for _, end in ripple_windows],
            }
        ),
    }
    return data, results


class TestBuildSessionEventSets:
    # Three replay events and three ripples, all in immobile stretches: one of
    # each inside the kept stop (1.5-3.0 s), one of each inside the dropped
    # in-transit stop (6.0-7.0 s), and one of each outside every stop.
    EVENT_WINDOWS = [(2.0, 2.2), (6.2, 6.4), (8.0, 8.2)]
    RIPPLE_WINDOWS = [(2.0, 2.15), (6.2, 6.35), (8.0, 8.15)]
    IS_SWR = [True, True, True]

    def test_only_events_starting_inside_a_cohort_stop_survive(
        self, position_info, cohort_stops
    ):
        from continuum_swr_replay.analysis.pfc.glm import build_session_event_sets

        data, results = _synthetic_session(
            position_info, self.EVENT_WINDOWS, self.RIPPLE_WINDOWS
        )

        event_times, counts = build_session_event_sets(
            data,
            results,
            canonical_events=_canonical_events(self.EVENT_WINDOWS, self.IS_SWR),
            swr_intervals=_swr_intervals(self.RIPPLE_WINDOWS),
            cohort_stops=cohort_stops,
            stop_cohort="at_well",
        )

        assert counts["n_nonlocal_before"] == 3
        assert counts["n_nonlocal_after"] == 1
        assert counts["n_ripple_before"] == 3
        assert counts["n_ripple_after"] == 1
        np.testing.assert_allclose(
            event_times["non_local"].start_time.to_numpy(), [2.0], atol=0.01
        )
        np.testing.assert_allclose(
            event_times["SWR"].start_time.to_numpy(), [2.0], atol=0.01
        )
        # The kept replay event overlaps the kept ripple, so the SWR split is
        # taken over the restricted sets, not the full ones.
        assert len(event_times["non_local_SWR"]) == 1
        assert len(event_times["non_local_no_SWR"]) == 0

    def test_all_stops_keeps_every_speed_eligible_event(self, position_info):
        from continuum_swr_replay.analysis.pfc.glm import (
            build_session_event_sets,
            build_session_fit_mask,
        )

        data, results = _synthetic_session(
            position_info, self.EVENT_WINDOWS, self.RIPPLE_WINDOWS
        )
        _, fit_mask, _ = build_session_fit_mask(position_info, stop_cohort="all_stops")

        event_times, counts = build_session_event_sets(
            data,
            results,
            canonical_events=_canonical_events(self.EVENT_WINDOWS, self.IS_SWR),
            swr_intervals=_swr_intervals(self.RIPPLE_WINDOWS),
            stop_cohort="all_stops",
            fit_mask=fit_mask,
        )

        assert counts["n_nonlocal_after"] == counts["n_nonlocal_before"] == 3
        assert counts["n_ripple_after"] == counts["n_ripple_before"] == 3
        assert len(event_times["non_local_SWR"]) == 3
        # Nothing is dropped, so nothing can have been dropped over fitted bins.
        assert counts["n_nonlocal_dropped_overlapping_fit"] == 0
        assert counts["n_ripple_dropped_overlapping_fit"] == 0

    def test_a_cohort_without_stops_is_refused(self, position_info):
        from continuum_swr_replay.analysis.pfc.glm import build_session_event_sets

        data, results = _synthetic_session(
            position_info, self.EVENT_WINDOWS, self.RIPPLE_WINDOWS
        )

        with pytest.raises(ValueError, match="cohort_stops"):
            build_session_event_sets(
                data,
                results,
                canonical_events=_canonical_events(self.EVENT_WINDOWS, self.IS_SWR),
                swr_intervals=_swr_intervals(self.RIPPLE_WINDOWS),
                stop_cohort="at_well",
            )

    def test_the_canonical_is_swr_column_decides_the_split(
        self, position_info, cohort_stops
    ):
        # A ripple that begins exactly where the event ends. The retired split
        # (inclusive interval overlap against ``ripple_times``) called such an
        # event SWR-coupled; the canonical rule requires positive-duration
        # overlap and calls it non-SWR. The column, not the intervals, decides.
        from continuum_swr_replay.analysis.pfc.glm import build_session_event_sets

        boundary_windows = [(2.0, 2.2), (2.4, 2.6)]
        boundary_ripples = [(2.2, 2.3), (2.4, 2.5)]
        data, results = _synthetic_session(
            position_info, boundary_windows, boundary_ripples
        )

        event_times, counts = build_session_event_sets(
            data,
            results,
            canonical_events=_canonical_events(boundary_windows, [False, True]),
            swr_intervals=_swr_intervals(boundary_ripples),
            cohort_stops=cohort_stops,
            stop_cohort="at_well",
        )

        assert counts["n_nonlocal_swr"] == 1
        assert counts["n_nonlocal_no_swr"] == 1
        np.testing.assert_allclose(
            event_times["non_local_no_SWR"].start_time.to_numpy(), [2.0]
        )
        np.testing.assert_allclose(
            event_times["non_local_SWR"].start_time.to_numpy(), [2.4]
        )
        # The SWR regressor keeps every interval: only the replay events are
        # split by the column.
        assert len(event_times["SWR"]) == 2

    def test_the_event_set_names_are_the_four_the_designs_are_keyed_on(
        self, position_info, cohort_stops
    ):
        from continuum_swr_replay.analysis.pfc.glm import build_session_event_sets

        data, results = _synthetic_session(
            position_info, self.EVENT_WINDOWS, self.RIPPLE_WINDOWS
        )

        event_times, _ = build_session_event_sets(
            data,
            results,
            canonical_events=_canonical_events(self.EVENT_WINDOWS, self.IS_SWR),
            swr_intervals=_swr_intervals(self.RIPPLE_WINDOWS),
            cohort_stops=cohort_stops,
            stop_cohort="at_well",
        )

        assert set(event_times) == {
            "non_local_no_SWR",
            "non_local_SWR",
            "non_local",
            "SWR",
        }

    def test_an_event_outside_the_aligned_frame_is_counted_and_dropped(
        self, position_info
    ):
        # The canonical events were built on the decode's own axis. One past
        # the end of the aligned position frame has no bins to be modelled on,
        # so it is dropped -- and counted, because it means the two axes
        # disagree over more than an empty tail.
        from continuum_swr_replay.analysis.pfc.glm import build_session_event_sets

        data, results = _synthetic_session(
            position_info, self.EVENT_WINDOWS, self.RIPPLE_WINDOWS
        )
        last = float(position_info.index[-1])
        windows = [*self.EVENT_WINDOWS, (last + 1.0, last + 1.2)]

        _, counts = build_session_event_sets(
            data,
            results,
            canonical_events=_canonical_events(windows, [True, True, True, False]),
            swr_intervals=_swr_intervals(self.RIPPLE_WINDOWS),
            stop_cohort="all_stops",
        )

        assert counts["n_events_outside_frame"] == 1
        assert counts["n_swr_intervals_outside_frame"] == 0
        assert counts["n_nonlocal_before"] == 3

    def test_a_dropped_event_at_a_stops_leading_edge_is_counted(
        self, position_info, cohort_stops, is_immobile
    ):
        # An event beginning 0.1 s before the kept stop starts is dropped by the
        # onset rule, but its +/-0.25 s basis would have covered fitted bins
        # inside the stop. That is the gap the diagnostic exists to size.
        from continuum_swr_replay.analysis.pfc.glm import (
            build_session_event_sets,
            build_session_fit_mask,
        )

        leading_edge_windows = [(1.4, 1.45), (8.0, 8.2)]
        leading_edge_ripples = [(1.4, 1.45)]
        data, results = _synthetic_session(
            position_info, leading_edge_windows, leading_edge_ripples
        )
        _, fit_mask, _ = build_session_fit_mask(
            position_info, cohort_stops=cohort_stops, stop_cohort="at_well"
        )

        _, counts = build_session_event_sets(
            data,
            results,
            canonical_events=_canonical_events(leading_edge_windows, [True, False]),
            swr_intervals=_swr_intervals(leading_edge_ripples),
            cohort_stops=cohort_stops,
            stop_cohort="at_well",
            fit_mask=fit_mask,
            basis_half_window_s=0.25,
        )

        # Both replay events are dropped; only the leading-edge one is within
        # a basis half-window of a fitted bin (the 8.0 s one is not).
        assert counts["n_nonlocal_after"] == 0
        assert counts["n_nonlocal_dropped_overlapping_fit"] == 1
        assert counts["n_ripple_dropped_overlapping_fit"] == 1
        # With nothing kept, a fraction of the kept count is undefined, not zero.
        assert counts["n_nonlocal_dropped_overlapping_fit_frac_of_kept"] is None

    def test_the_diagnostic_is_none_when_no_fit_mask_is_given(
        self, position_info, cohort_stops
    ):
        from continuum_swr_replay.analysis.pfc.glm import build_session_event_sets

        data, results = _synthetic_session(
            position_info, self.EVENT_WINDOWS, self.RIPPLE_WINDOWS
        )

        _, counts = build_session_event_sets(
            data,
            results,
            canonical_events=_canonical_events(self.EVENT_WINDOWS, self.IS_SWR),
            swr_intervals=_swr_intervals(self.RIPPLE_WINDOWS),
            cohort_stops=cohort_stops,
            stop_cohort="at_well",
        )

        assert counts["n_nonlocal_dropped_overlapping_fit"] is None
        assert counts["n_ripple_dropped_overlapping_fit"] is None


class TestCountEventsCoveringFitBins:
    def test_only_onsets_within_the_window_of_a_fitted_bin_count(self):
        from continuum_swr_replay.analysis.pfc.glm import (
            count_events_covering_fit_bins,
        )

        times = np.arange(0.0, 10.0, 0.002)
        fit_mask = (times >= 5.0) & (times < 6.0)

        # 4.80 is 0.20 s before the first fitted bin (inside the window);
        # 4.70 is 0.30 s before it (outside); 6.10 is 0.10 s after the last.
        counted = count_events_covering_fit_bins(
            np.array([4.70, 4.80, 5.5, 6.10, 9.0]),
            times,
            fit_mask,
            basis_half_window_s=0.25,
        )

        assert counted == 3

    def test_no_onsets_counts_zero(self):
        from continuum_swr_replay.analysis.pfc.glm import (
            count_events_covering_fit_bins,
        )

        times = np.arange(0.0, 1.0, 0.002)

        assert (
            count_events_covering_fit_bins(
                np.array([]), times, np.ones_like(times, dtype=bool)
            )
            == 0
        )

    def test_an_empty_fit_mask_counts_zero(self):
        from continuum_swr_replay.analysis.pfc.glm import (
            count_events_covering_fit_bins,
        )

        times = np.arange(0.0, 1.0, 0.002)

        assert (
            count_events_covering_fit_bins(
                np.array([0.5]), times, np.zeros_like(times, dtype=bool)
            )
            == 0
        )


class TestProcessNeuronFitMask:
    def test_default_fit_mask_reproduces_the_immobility_fit_bit_for_bit(
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        simple_event_times,
        position_info,
        sampling_frequency,
    ):
        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )

        default_stats, *default_mods = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
        )
        explicit_stats, *explicit_mods = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
            fit_mask=is_immobile,
        )

        assert default_stats == explicit_stats
        for default_mod, explicit_mod in zip(default_mods, explicit_mods, strict=True):
            np.testing.assert_array_equal(default_mod, explicit_mod)
        # With no stop restriction the fitted sample IS every immobile bin, so
        # the redefined column and its unrestricted companion must agree exactly.
        assert (
            default_stats["immobility_avg_firing_rate"]
            == default_stats["immobility_avg_firing_rate_all_immobile"]
        )

    def test_a_narrower_fit_mask_changes_the_fitted_sample(
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        simple_event_times,
        position_info,
        sampling_frequency,
    ):
        from continuum_swr_replay.analysis.pfc.glm import get_spikecount_per_time_bin

        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )
        # Keep the immobile bins around the two SWR events only.
        inside = pd.Series(
            (time_axis >= 0.5) & (time_axis <= 6.0), index=position_info.index
        )
        fit_mask = is_immobile & inside

        stats, *_ = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
            fit_mask=fit_mask,
        )

        # The reported immobility rate describes the fitted bins, so it equals
        # the mean over the narrowed mask -- not over every immobile bin.
        spike_count = get_spikecount_per_time_bin(poisson_spike_times, time_axis)
        expected = spike_count[fit_mask].mean() * sampling_frequency
        assert stats["immobility_avg_firing_rate"] == pytest.approx(expected)
        # Running statistics still use every moving bin, cohort or not.
        expected_run = spike_count[~is_immobile].mean() * sampling_frequency
        assert stats["run_avg_firing_rate"] == pytest.approx(expected_run)
        # The companion column keeps the unrestricted immobility rate, so a
        # bundle's cohort is readable from the table: the two now differ.
        expected_all = spike_count[is_immobile].mean() * sampling_frequency
        assert stats["immobility_avg_firing_rate_all_immobile"] == pytest.approx(
            expected_all
        )
        assert (
            stats["immobility_avg_firing_rate"]
            != stats["immobility_avg_firing_rate_all_immobile"]
        )


class TestAlignmentIsIdentityOnMatchedAxes:
    """A session whose decode and position axes already match must be
    untouched by the shared-timestamp alignment on the PFC load path: same
    events, same design matrices, same fitted coefficients, bit for bit."""

    EVENT_WINDOWS = [(2.0, 2.2), (6.2, 6.4), (8.0, 8.2)]
    RIPPLE_WINDOWS = [(2.0, 2.15), (6.2, 6.35), (8.0, 8.15)]
    IS_SWR = [True, True, True]

    @classmethod
    def _fit(cls, data, results, position_info, sampling_frequency, spike_count):
        from continuum_swr_replay.analysis.pfc.glm import (
            build_session_event_sets,
            build_session_fit_mask,
        )

        _, fit_mask, _ = build_session_fit_mask(position_info, stop_cohort="all_stops")
        event_times, counts = build_session_event_sets(
            data,
            results,
            canonical_events=_canonical_events(cls.EVENT_WINDOWS, cls.IS_SWR),
            swr_intervals=_swr_intervals(cls.RIPPLE_WINDOWS),
            stop_cohort="all_stops",
            fit_mask=fit_mask,
        )
        event_times["matched_control"] = make_matched_control_times(
            event_times, fit_mask, np.random.default_rng(0)
        )
        design_matrices, _, _ = _make_design_matrices(
            event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-25, 26, 1),
        )
        glm_results, _ = _fit_immobility_models(fit_mask, design_matrices, spike_count)
        return event_times, design_matrices, glm_results, counts

    def test_fitted_numbers_are_unchanged_by_the_alignment_step(
        self, position_info, sampling_frequency, spike_count, monkeypatch
    ):
        import continuum_swr_replay.analysis.hippocampus.data_aggregation as agg

        data, results = _synthetic_session(
            position_info, self.EVENT_WINDOWS, self.RIPPLE_WINDOWS
        )

        with_alignment = self._fit(
            data, results, position_info, sampling_frequency, spike_count
        )

        # Bypass the alignment step entirely and refit: on matched axes the two
        # paths must agree exactly, not approximately.
        monkeypatch.setattr(
            agg,
            "align_results_to_position",
            lambda results, position_info, **kwargs: (results, position_info, {}),
        )
        without_alignment = self._fit(
            data, results, position_info, sampling_frequency, spike_count
        )

        events_a, matrices_a, glms_a, counts_a = with_alignment
        events_b, matrices_b, glms_b, counts_b = without_alignment

        assert counts_a == counts_b
        assert events_a.keys() == events_b.keys()
        for key in events_a:
            pd.testing.assert_frame_equal(
                events_a[key].reset_index(drop=True),
                events_b[key].reset_index(drop=True),
                check_exact=True,
            )
        assert matrices_a.keys() == matrices_b.keys()
        for key in matrices_a:
            assert np.array_equal(matrices_a[key], matrices_b[key])
        assert glms_a.keys() == glms_b.keys()
        for key in glms_a:
            assert np.array_equal(glms_a[key].params, glms_b[key].params)
            assert glms_a[key].llf == glms_b[key].llf


class TestTheRetiredEventConstructionIsGone:
    """The run-time event construction must not survive anywhere on the PFC
    path. It reads a separately detected ripple population and re-detects
    non-local events, which is exactly the drift from the content and rate
    families that reading the canonical table removes. Its only home is
    ``scripts/qa/diagnose_pfc_event_definition.py``, which exists to compare
    the two."""

    RETIRED = ('data["ripple_times"]', "get_overlap_times", "get_confident_times")

    def _sources(self):
        from pathlib import Path

        import continuum_swr_replay.analysis.pfc as pfc_package

        pfc_dir = Path(pfc_package.__file__).parent
        repo = Path(pfc_package.__file__).resolve().parents[4]
        figure = repo / "scripts" / "figures" / "generate_figure04.py"
        assert figure.exists(), f"expected the Figure 4 script at {figure}"
        return [*sorted(pfc_dir.glob("*.py")), figure]

    def test_no_pfc_source_rebuilds_its_own_events(self):
        offenders = [
            (path, token)
            for path in self._sources()
            for token in self.RETIRED
            if token in path.read_text()
        ]

        assert offenders == []
