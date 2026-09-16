"""Tests for the GLM fitting pipeline.

Tests the core signal processing and model fitting functions used to analyze
PFC neuron modulation during hippocampal replay events.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from patsy import dmatrix

from continuum_swr_replay.analysis.pfc.glm import (
    _fit_immobility_models,
    _fit_theta_models,
    _make_design_matrices,
    _prefit_session,
    build_theta_design,
    convolve_event_basis,
    event_basis_functions,
    event_onset_counts,
    fit_models_batched,
    fitter_provenance,
    get_frac_explained_deviance,
    get_frac_null_deviance,
    is_separated,
    make_matched_control_times,
    model_penalties,
    peak_log_modulation,
    peak_modulation_se,
    prior_binds,
    process_neuron,
    shipped_fit_counts,
)
from continuum_swr_replay.analysis.pfc.glm import (
    test_theta_modulation as run_theta_modulation_test,
)
from continuum_swr_replay.analysis.replay.cohort_selection import select_stop_cohort


def spike_counts_of(spike_times, time_axis):
    """The binned counts ``process_neuron`` derives from the same spike times."""
    from continuum_swr_replay.analysis.pfc.glm import get_spikecount_per_time_bin

    return get_spikecount_per_time_bin(spike_times, np.asarray(time_axis))


#: Prior standard deviation of an event-basis coefficient, on the log-rate
#: scale, for the tests that fit the penalised estimator. The fixtures here are
#: ten seconds of a ~5 Hz Poisson train with three events, so every event class
#: has spline columns with no spikes under them and the unpenalised fits run
#: away (coefficients past 100, a deviance above the constant model's, half the
#: ladder at ``maxiter``); the prior is what makes them estimable.
PENALTY_SIGMA = 2.0

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
        "SWR": pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "end_time": [1.1, 5.1],
            }
        ),
        "non_local": pd.DataFrame(
            {
                "start_time": [1.0, 5.0, 7.0],
                "end_time": [1.1, 5.1, 7.1],
            }
        ),
        "non_local_SWR": pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "end_time": [1.1, 5.1],
            }
        ),
        "non_local_no_SWR": pd.DataFrame(
            {
                "start_time": [7.0],
                "end_time": [7.1],
            }
        ),
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
# The peri-event basis and its convolution
# ---------------------------------------------------------------------------


class TestEventBasisFunctions:
    """The peri-event kernel's basis: zero at both ends, one in the middle."""

    def test_the_kernel_is_zero_at_both_ends_of_the_fit_window(
        self, sampling_frequency
    ):
        """Neither edge is free.

        A clamped cubic B-spline has one function equal to one at each edge of
        the window. Dropping only the first -- which is what patsy's
        ``bs(...) - 1`` does -- pins the kernel to zero at the left edge and
        leaves the right edge free to run off, so the same modulation gets a
        boundary artefact at one end and not the other.
        """
        lags = np.arange(-125, 126)

        spline_basis, _, _ = event_basis_functions(lags, sampling_frequency)

        assert np.all(spline_basis[0] == 0.0)
        assert np.all(spline_basis[-1] == 0.0)
        # Any coefficient vector, not just the basis functions one at a time.
        rng = np.random.default_rng(0)
        kernels = spline_basis @ rng.normal(size=(spline_basis.shape[1], 20))
        np.testing.assert_array_equal(kernels[0], 0.0)
        np.testing.assert_array_equal(kernels[-1], 0.0)

    def test_the_basis_sums_to_one_between_the_outer_knots(
        self, sampling_frequency
    ):
        """So a flat modulation is representable, with equal coefficients."""
        lags = np.arange(-125, 126)

        spline_basis, lag_time, knots = event_basis_functions(
            lags, sampling_frequency
        )

        interior = (lag_time >= knots[0]) & (lag_time <= knots[-1])
        np.testing.assert_allclose(spline_basis[interior].sum(axis=1), 1.0, atol=1e-12)
        assert np.all(spline_basis >= -1e-12)

    def test_the_window_that_sums_to_one_covers_the_reported_lags(
        self, sampling_frequency
    ):
        """The +/-250 ms fit window's flat span is the +/-200 ms report window."""
        spline_basis, lag_time, knots = event_basis_functions(
            np.arange(-125, 126), sampling_frequency
        )

        assert knots[0] == pytest.approx(-0.20)
        assert knots[-1] == pytest.approx(0.20)
        assert spline_basis.shape == (251, len(knots) + 2)

    def test_the_basis_is_two_functions_narrower_than_the_full_one(
        self, sampling_frequency
    ):
        """Stated, because it changes every design's width and every fit."""
        lags = np.arange(-125, 126)
        lag_time = lags / sampling_frequency
        knots = np.arange(lag_time[0], lag_time[-1], 0.050)[1:]
        full = np.asarray(
            dmatrix(
                "bs(lag_time, knots=knots, include_intercept=True) - 1",
                {"lag_time": lag_time, "knots": knots},
            )
        )

        spline_basis, _, _ = event_basis_functions(lags, sampling_frequency)

        assert full.shape[1] == len(knots) + 4
        assert spline_basis.shape[1] == len(knots) + 2
        np.testing.assert_allclose(spline_basis, full[:, 1:-1])


#: Prior the planted-kernel recovery is read at -- the shipped default.
PLANTED_KERNEL_SIGMA = 3.0


@pytest.fixture(scope="module")
def planted_kernel_session():
    """Units whose rate really is driven by a known peri-event kernel.

    Three modulation amplitudes x two baseline rates, all sharing one design:
    a 300 s epoch with 300 events, the kernel a 30 ms bump 20 ms after onset,
    projected into the basis so it is exactly representable (50 ms knots cannot
    carry a 30 ms bump, and the recoverable truth is the projection, not the
    bump). A seventh unit fires nowhere near an event, so its unpenalised fit
    separates.
    """
    rng = np.random.default_rng(20260916)
    sampling_frequency = 500.0
    lags = np.arange(-125, 126)
    spline_basis, lag_time, _ = event_basis_functions(lags, sampling_frequency)
    n_time = int(300 * sampling_frequency)
    time = np.arange(n_time) / sampling_frequency

    onsets = np.sort(rng.uniform(1.0, 295.0, size=300))
    events = pd.DataFrame({"start_time": onsets, "end_time": onsets + 0.08})
    convolved = convolve_event_basis(
        event_onset_counts(events, time), spline_basis, lags
    )
    design = np.column_stack([np.ones(n_time), convolved])

    shape = np.exp(-0.5 * ((lag_time - 0.02) / 0.03) ** 2)
    cells, counts = [], []
    for amplitude in (0.5, 1.5, 3.0):
        coefficients = np.linalg.lstsq(spline_basis, amplitude * shape, rcond=None)[0]
        curve = spline_basis @ coefficients
        planted = curve[np.argmax(np.abs(curve))]
        for rate in (2.0, 20.0):
            cells.append(
                {"amplitude": amplitude, "rate": rate, "planted_peak": planted}
            )
            counts.append(
                rng.poisson(
                    np.exp(np.log(rate / sampling_frequency) + convolved @ coefficients)
                )
            )

    # The separated unit: spikes everywhere except under the kernel's support.
    covered = convolved.sum(axis=1) > 0
    silent = rng.poisson(np.exp(np.log(5.0 / sampling_frequency)), size=n_time)
    silent[covered] = 0
    counts.append(silent)

    from continuum_swr_replay.analysis.pfc.batched_glm import (
        fit_poisson_glms,
        make_ridge_penalty,
    )

    counts = np.column_stack(counts).astype(float)
    penalty = make_ridge_penalty(design.shape[1], PLANTED_KERNEL_SIGMA)
    return {
        "design": design,
        "counts": counts,
        "cells": cells,
        "spline_basis": spline_basis,
        "lag_time": lag_time,
        # Fit once: every test below reads the same two fits.
        "penalised": fit_poisson_glms(design, counts, penalty=penalty),
        "unpenalised": fit_poisson_glms(design, counts),
    }


class TestPlantedKernelRecovery:
    """The estimator has to return the kernel that generated the spikes.

    Agreement with a reference solver says the arithmetic is right; this says
    the answer is. The prior shrinks towards no modulation, so a recovered peak
    is expected to come back *at most* the planted one plus sampling noise, and
    at least the planted one times the ridge factor ``tau**2 / (tau**2 + s**2)``
    -- ``tau`` the prior standard deviation of the kernel at one lag, ``s`` the
    fit's own standard error there. On a design with this many events that
    factor is above 0.99, which is the point of the prior: it is meant to bind
    only where the data say nothing.
    """

    SIGMA = PLANTED_KERNEL_SIGMA

    @staticmethod
    def fits(session):
        return session["penalised"], session["unpenalised"]

    def peaks(self, fit, session):
        peak = peak_log_modulation(
            fit.params[:, 1:], session["spline_basis"], session["lag_time"]
        )
        se = peak_modulation_se(
            fit.covariance[:, 1:, 1:], session["spline_basis"], peak.index
        )
        return peak, se

    def test_every_planted_amplitude_and_rate_is_recovered(
        self, planted_kernel_session
    ):
        from continuum_swr_replay.analysis.pfc.batched_glm import implied_prior_sd

        penalised, _ = self.fits(planted_kernel_session)
        peak, se = self.peaks(penalised, planted_kernel_session)
        window = np.abs(planted_kernel_session["lag_time"]) <= 0.2
        tau = implied_prior_sd(
            self.SIGMA, planted_kernel_session["spline_basis"], window
        )["per_lag"]

        for unit, cell in enumerate(planted_kernel_session["cells"]):
            planted = cell["planted_peak"]
            ridge_factor = tau**2 / (tau**2 + se[unit] ** 2)
            assert ridge_factor > 0.99, cell
            # Shrunk towards zero by at most the ridge factor, and free to
            # scatter by the fit's own standard error either way.
            assert planted * ridge_factor - 3 * se[unit] <= peak.value[unit], cell
            assert peak.value[unit] <= planted + 3 * se[unit], cell
            assert peak.lag[unit] == pytest.approx(0.02, abs=0.02), cell

    def test_a_higher_rate_recovers_the_same_kernel_more_precisely(
        self, planted_kernel_session
    ):
        penalised, _ = self.fits(planted_kernel_session)
        peak, se = self.peaks(penalised, planted_kernel_session)

        for low, high in ((0, 1), (2, 3), (4, 5)):  # 2 Hz then 20 Hz, per amplitude
            assert se[high] < se[low]
            planted = planted_kernel_session["cells"][high]["planted_peak"]
            assert abs(peak.value[high] - planted) <= max(3 * se[high], 0.1)

    def test_the_recovered_amplitudes_are_ordered_like_the_planted_ones(
        self, planted_kernel_session
    ):
        penalised, _ = self.fits(planted_kernel_session)
        peak, _ = self.peaks(penalised, planted_kernel_session)

        n_cells = len(planted_kernel_session["cells"])
        for rate_offset in (0, 1):
            small, middle, large = peak.value[:n_cells][rate_offset::2]
            assert small < middle < large

    def test_the_prior_barely_moves_a_well_sampled_fit(self, planted_kernel_session):
        """Where the data speak, the prior does not: binding is the exception."""
        penalised, unpenalised = self.fits(planted_kernel_session)
        penalised_peak, _ = self.peaks(penalised, planted_kernel_session)
        mle_peak, mle_se = self.peaks(unpenalised, planted_kernel_session)

        n_cells = len(planted_kernel_session["cells"])
        binds = prior_binds(
            penalised_peak.value[:n_cells], mle_peak.value[:n_cells], mle_se[:n_cells]
        )

        assert not binds.any()
        assert not is_separated(unpenalised.params[:n_cells, 1:]).any()

    def test_the_separated_unit_gets_a_finite_curve_instead_of_a_runaway(
        self, planted_kernel_session
    ):
        penalised, unpenalised = self.fits(planted_kernel_session)
        penalised_peak, _ = self.peaks(penalised, planted_kernel_session)
        mle_peak, _ = self.peaks(unpenalised, planted_kernel_session)

        separated = -1
        assert is_separated(unpenalised.params[:, 1:])[separated]
        assert mle_peak.value[separated] < -10.0  # a rate multiplier of 5e-5
        assert np.isfinite(penalised_peak.value[separated])
        # Suppression, said at a scale a rate can have: at most a few percent
        # of baseline, not zero.
        assert -8.0 < penalised_peak.value[separated] < -1.0


class TestPeakLogModulation:
    """The curve-level read of a fit: where the kernel peaks, and how well."""

    @staticmethod
    def basis():
        spline_basis, lag_time, _ = event_basis_functions(np.arange(-125, 126), 500.0)
        return spline_basis, lag_time

    def test_it_finds_the_planted_peak(self):
        spline_basis, lag_time = self.basis()
        # A bump at +20 ms, built in the basis so it is exactly representable.
        shape = np.exp(-0.5 * ((lag_time - 0.02) / 0.03) ** 2)
        coefficients = np.linalg.lstsq(spline_basis, 1.5 * shape, rcond=None)[0]

        peak = peak_log_modulation(coefficients, spline_basis, lag_time)

        # Against the curve the basis can actually carry: 50 ms knots smooth a
        # 30 ms bump, so the representable peak is a little under the 1.5 asked
        # for, and that is what a fit could recover.
        curve = spline_basis @ coefficients
        assert peak.value[0] == pytest.approx(np.abs(curve).max(), rel=1e-12)
        assert peak.value[0] == pytest.approx(1.34, abs=0.05)
        assert peak.lag[0] == pytest.approx(0.02, abs=0.006)
        assert spline_basis[peak.index[0]] @ coefficients == pytest.approx(
            peak.value[0]
        )

    def test_a_negative_excursion_is_the_peak_when_it_is_the_largest(self):
        spline_basis, lag_time = self.basis()
        shape = -np.exp(-0.5 * ((lag_time + 0.05) / 0.03) ** 2)
        coefficients = np.linalg.lstsq(spline_basis, 2.0 * shape, rcond=None)[0]

        peak = peak_log_modulation(coefficients, spline_basis, lag_time)

        assert peak.value[0] < -1.5
        assert peak.lag[0] == pytest.approx(-0.05, abs=0.006)

    def test_the_peak_is_taken_over_the_reported_window_only(self):
        """A kernel that runs away outside +/-200 ms is not reported as a peak.

        The models are fit on +/-250 ms precisely so the ends are outside what
        is read; a peak over the fit window would report the discarded part.
        """
        spline_basis, lag_time = self.basis()
        coefficients = np.zeros(spline_basis.shape[1])
        coefficients[0] = 5.0  # the first basis function, which lives at -250 ms
        coefficients[5] = 1.0

        peak = peak_log_modulation(coefficients, spline_basis, lag_time)

        assert abs(peak.lag[0]) <= 0.2
        assert peak.value[0] < 5.0

    def test_units_are_read_independently(self):
        spline_basis, lag_time = self.basis()
        shape = np.exp(-0.5 * ((lag_time - 0.02) / 0.03) ** 2)
        one = np.linalg.lstsq(spline_basis, 1.0 * shape, rcond=None)[0]
        two = np.linalg.lstsq(spline_basis, -2.0 * shape, rcond=None)[0]

        peak = peak_log_modulation(np.row_stack([one, two]), spline_basis, lag_time)

        assert peak.value[0] > 0 > peak.value[1]
        assert abs(peak.value[1]) == pytest.approx(2 * peak.value[0], rel=0.01)

    def test_a_unit_with_no_fit_comes_back_nan(self):
        spline_basis, lag_time = self.basis()
        coefficients = np.full(spline_basis.shape[1], np.nan)

        peak = peak_log_modulation(coefficients, spline_basis, lag_time)

        assert np.isnan(peak.value[0])
        assert np.isnan(peak.lag[0])
        assert peak.index[0] == -1

    def test_a_window_that_excludes_every_lag_is_refused(self):
        spline_basis, lag_time = self.basis()

        with pytest.raises(ValueError, match="no lag inside"):
            peak_log_modulation(
                np.zeros(spline_basis.shape[1]),
                spline_basis,
                lag_time,
                half_window_s=-1.0,
            )


class TestPeakModulationSe:
    def test_it_is_the_basis_quadratic_form_not_the_diagonal(self):
        """The off-diagonal terms matter: the coefficients are correlated."""
        spline_basis, lag_time = self.basis()
        rng = np.random.default_rng(0)
        root = rng.normal(size=(spline_basis.shape[1], spline_basis.shape[1])) * 0.1
        covariance = (root @ root.T)[None]
        index = np.array([120])

        actual = peak_modulation_se(covariance, spline_basis, index)

        at_peak = spline_basis[120]
        assert actual[0] == pytest.approx(np.sqrt(at_peak @ covariance[0] @ at_peak))
        diagonal_only = np.sqrt(at_peak**2 @ np.diag(covariance[0]))
        assert actual[0] != pytest.approx(diagonal_only)

    def test_it_matches_a_fitted_curves_own_uncertainty(self):
        """Against the standard error of the same linear functional, from a fit."""
        rng = np.random.default_rng(5)
        spline_basis, lag_time = self.basis()
        n_time = 20_000
        onsets = np.sort(rng.uniform(1.0, 38.0, size=150))
        events = pd.DataFrame({"start_time": onsets, "end_time": onsets + 0.08})
        time = np.arange(n_time) / 500.0
        convolved = convolve_event_basis(
            event_onset_counts(events, time), spline_basis, np.arange(-125, 126)
        )
        design = np.column_stack([np.ones(n_time), convolved])
        counts = rng.poisson(np.exp(np.log(0.02) + convolved.sum(axis=1) * 0.5))

        from continuum_swr_replay.analysis.pfc.batched_glm import fit_poisson_glms

        fit = fit_poisson_glms(design, counts.astype(float))
        peak = peak_log_modulation(fit.params[:, 1:], spline_basis, lag_time)
        actual = peak_modulation_se(fit.covariance[:, 1:, 1:], spline_basis, peak.index)

        at_peak = spline_basis[peak.index[0]]
        expected = np.sqrt(at_peak @ fit.covariance[0, 1:, 1:] @ at_peak)
        assert actual[0] == pytest.approx(expected)
        assert 0.0 < actual[0] < 1.0

    def test_a_unit_without_a_peak_has_no_error(self):
        spline_basis, _ = self.basis()
        covariance = np.eye(spline_basis.shape[1])[None]

        assert np.isnan(peak_modulation_se(covariance, spline_basis, np.array([-1]))[0])

    basis = staticmethod(TestPeakLogModulation.basis)


class TestPriorBinds:
    def test_it_is_a_statement_about_the_curve_not_a_coefficient(self):
        penalised = np.array([1.0, 1.0, 1.0])
        mle = np.array([1.05, 2.0, 1.0])
        mle_se = np.array([0.5, 0.5, 0.5])

        binds = prior_binds(penalised, mle, mle_se)

        # Moved by less than the data can resolve -> not binding; more -> yes.
        assert not binds[0]
        assert binds[1]
        assert not binds[2]

    def test_a_unit_without_a_maximum_likelihood_peak_does_not_count(self):
        binds = prior_binds(
            np.array([1.0, 1.0]), np.array([np.nan, 1.0]), np.array([0.1, np.nan])
        )

        assert not binds.any()

    def test_is_separated_finds_the_runaway_coefficients(self):
        params = np.array([[0.5, -1.0, 2.0], [0.5, -40.0, 2.0], [np.nan, 1.0, 1.0]])

        separated = is_separated(params)

        assert separated.tolist() == [False, True, True]


class TestConvolveEventBasis:
    """The convolution columns, against the lagged-matrix product they replace."""

    @staticmethod
    def lagged_matrix_product(signal, spline_basis, lags):
        """The construction this replaces: 201 shifted copies, then a product.

        ``pandas.Series.shift(lag)`` moves a value DOWN by ``lag`` rows, so
        column ``j`` holds ``signal[t - lags[j]]`` and a positive lag reads as
        time after an onset. Written out here rather than imported, so the test
        pins the arithmetic and not the implementation.
        """
        frame = pd.DataFrame(signal)
        lagged = np.concatenate(
            [frame.shift(int(lag), fill_value=0).to_numpy() for lag in lags], axis=1
        )
        return lagged @ spline_basis

    def test_it_equals_the_lagged_matrix_product(self, sampling_frequency):
        lags = np.arange(-125, 126)
        spline_basis, _, _ = event_basis_functions(lags, sampling_frequency)
        rng = np.random.default_rng(3)
        signal = np.zeros(20_000)
        signal[rng.choice(20_000, 200, replace=False)] = 1.0

        actual = convolve_event_basis(signal, spline_basis, lags)

        expected = self.lagged_matrix_product(signal, spline_basis, lags)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_an_onset_puts_the_kernel_after_it_at_positive_lags(
        self, sampling_frequency
    ):
        """The sign convention, pinned on a single onset.

        Column ``b`` at time ``t`` must carry ``B_b(t - onset)``: the response
        at lag ``L`` after the onset is the basis evaluated at ``+L``.
        """
        lags = np.arange(-50, 51)
        spline_basis, _, _ = event_basis_functions(lags, sampling_frequency)
        signal = np.zeros(400)
        signal[200] = 1.0

        actual = convolve_event_basis(signal, spline_basis, lags)

        for offset, lag_index in ((0, 50), (20, 70), (-20, 30)):
            np.testing.assert_allclose(
                actual[200 + offset], spline_basis[lag_index], atol=1e-12
            )

    def test_two_onsets_add(self, sampling_frequency):
        lags = np.arange(-20, 21)
        spline_basis, _, _ = event_basis_functions(lags, sampling_frequency)
        first, second = np.zeros(300), np.zeros(300)
        first[100] = 1.0
        second[110] = 1.0

        both = convolve_event_basis(first + second, spline_basis, lags)

        np.testing.assert_allclose(
            both,
            convolve_event_basis(first, spline_basis, lags)
            + convolve_event_basis(second, spline_basis, lags),
            atol=1e-12,
        )

    def test_no_onsets_gives_no_signal(self, sampling_frequency):
        lags = np.arange(-20, 21)
        spline_basis, _, _ = event_basis_functions(lags, sampling_frequency)

        actual = convolve_event_basis(np.zeros(300), spline_basis, lags)

        np.testing.assert_array_equal(actual, 0.0)

    def test_an_onset_near_the_edge_is_not_wrapped(self, sampling_frequency):
        """An onset whose window runs off the start contributes only what fits."""
        lags = np.arange(-20, 21)
        spline_basis, _, _ = event_basis_functions(lags, sampling_frequency)
        signal = np.zeros(300)
        signal[2] = 1.0

        actual = convolve_event_basis(signal, spline_basis, lags)

        expected = self.lagged_matrix_product(signal, spline_basis, lags)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert np.all(actual[-1] == 0.0)


class TestMakeDesignMatrices:
    def test_output_structure(self, simple_event_times, position_info):
        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )

        expected_models = {
            "constant",
            "SWR",
            "non_local",
            "non_local_no_SWR",
            "non_local_SWR",
            "non_local_SWR + non_local_no_SWR",
            "SWR + non_local_no_SWR",
            "matched_control",
            "matched_control + non_local_no_SWR",
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


def _fit_ladder(is_immobile, design_matrices, spike_count, fitter):
    """The ladder as it is shipped under each backend.

    The batched backend fits the penalised estimator; statsmodels has no
    penalised path and fits the maximum-likelihood one, which is the only
    reason both appear here.
    """
    return _fit_immobility_models(
        is_immobile,
        design_matrices,
        spike_count,
        fitter=fitter,
        penalties=(
            model_penalties(design_matrices, PENALTY_SIGMA)
            if fitter == "jax"
            else None
        ),
    )


@pytest.mark.parametrize("fitter", ["statsmodels", "jax"])
class TestFitImmobilityModels:
    def test_returns_results_for_all_models(
        self, simple_event_times, position_info, is_immobile, spike_count, fitter
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        results, models = _fit_ladder(is_immobile, dm, spike_count, fitter)

        assert set(results.keys()) == set(dm.keys())
        # The batched backend has no per-unit statsmodels model object, and
        # nothing downstream reads one; a neuron it hands back to statsmodels
        # comes with the full set, as the statsmodels path always does. Either
        # way the model objects are all-or-nothing.
        assert set(models.keys()) in (set(), set(dm.keys()))
        if fitter == "statsmodels":
            assert set(models.keys()) == set(dm.keys())

    def test_results_have_aic(
        self, simple_event_times, position_info, is_immobile, spike_count, fitter
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        results, _ = _fit_ladder(is_immobile, dm, spike_count, fitter)

        for name, result in results.items():
            assert hasattr(result, "aic"), f"{name} result missing AIC"
            assert hasattr(result, "deviance"), f"{name} result missing deviance"
            assert np.isfinite(result.aic), f"{name} AIC is not finite"

    def test_more_complex_model_has_lower_or_equal_deviance(
        self, dense_session, fitter
    ):
        # On the well-posed fixture: a nested model can only lower the
        # deviance at the optimum, and neither an unpenalised fit that never
        # settled nor a prior that binds hard has an optimum to say that of.
        results, _ = _fit_ladder(
            dense_session["fit_mask"],
            dense_session["design_matrices"],
            dense_session["spike_count"],
            fitter,
        )

        # Nested: constant ⊂ SWR ⊂ SWR + non_local_no_SWR
        assert results["SWR"].deviance <= results["constant"].deviance + 1e-6
        assert (
            results["SWR + non_local_no_SWR"].deviance <= results["SWR"].deviance + 1e-6
        )


# ---------------------------------------------------------------------------
# Deviance fraction helpers
# ---------------------------------------------------------------------------


class TestDevianceFractions:
    """Read off the well-posed fixture, under the estimator that is shipped.

    The fractions are differences of deviances between nested fits; on a
    fixture where the maximum-likelihood fit does not exist, an unpenalised
    "fit" can leave the larger model with the larger deviance and the fractions
    then describe the solver rather than the data.
    """

    def test_frac_null_deviance_positive_for_nested_models(self, dense_session):
        results, _ = _fit_ladder(
            dense_session["fit_mask"],
            dense_session["design_matrices"],
            dense_session["spike_count"],
            "jax",
        )

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

    def test_frac_explained_deviance_bounded(self, dense_session):
        results, _ = _fit_ladder(
            dense_session["fit_mask"],
            dense_session["design_matrices"],
            dense_session["spike_count"],
            "jax",
        )

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
            theta_phase,
            is_immobile,
            spike_count,
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
            theta_phase,
            is_immobile,
            spike_count,
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

        stats, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod, mc_mod = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
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
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        simple_event_times,
        position_info,
        sampling_frequency,
    ):
        lags = np.arange(-50, 51)
        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=lags,
        )

        _, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod, mc_mod = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
            fitter="jax",
            penalty_sigma=PENALTY_SIGMA,
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

        stats, *_ = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
            fitter="jax",
            penalty_sigma=PENALTY_SIGMA,
        )

        # Check key LR tests exist
        assert "SWR vs. constant: lr" in stats
        assert "SWR vs. constant: pval" in stats
        assert "min_aic_model" in stats

        # p-values should be valid
        for key, val in stats.items():
            if key.endswith(": pval"):
                assert 0 <= val <= 1, f"{key} = {val} out of range"

    def test_one_models_failed_fit_leaves_the_neurons_other_models_alone(
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        simple_event_times,
        position_info,
        sampling_frequency,
    ):
        """A fit that produced nothing usable is non-estimable, not fatal.

        The ladder is nine fits of nine designs. One of them coming back with
        NaN parameters says nothing about the other eight, and failing the
        neuron would discard them -- and would do it silently, the neuron
        coming back with firing rates and no statistics at all.
        """
        import dataclasses

        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )
        fits, _ = _fit_ladder(is_immobile, dm, spike_counts_of(poisson_spike_times,
                                                              time_axis), "jax")
        broken = dict(fits)
        broken["SWR"] = dataclasses.replace(
            fits["SWR"], params=np.full_like(fits["SWR"].params, np.nan)
        )

        stats, swr_mod, no_swr_mod, *_ = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
            prefitted_immobility=broken,
        )

        # Every comparison that reads the broken fit is unavailable ...
        assert np.isnan(stats["SWR vs. constant: lr"])
        assert np.isnan(stats["SWR vs. constant: frac_null_dev"])
        assert np.isnan(stats["SWR + non_local_no_SWR vs. SWR: lr"])
        assert stats["min_aic_model"] != "SWR"
        # ... and nothing else is: the other models were fit on their own.
        assert np.isfinite(stats["non_local vs. constant: lr"])
        assert np.isfinite(stats["non_local_SWR + non_local_no_SWR vs. non_local: lr"])
        assert np.isfinite(stats["matched_control + non_local_no_SWR vs. "
                                 "matched_control: lr"])
        assert np.isfinite(swr_mod).all()
        assert np.isfinite(no_swr_mod).all()
        assert np.isfinite(stats["immobility_avg_firing_rate"])

    def test_a_model_that_never_converged_is_marked_non_estimable(
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        simple_event_times,
        position_info,
        sampling_frequency,
    ):
        """A fit that ran to maxiter is not an estimate, whatever it returned."""
        import dataclasses

        dm, spline_basis, lag_time = _make_design_matrices(
            simple_event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )
        fits, _ = _fit_ladder(is_immobile, dm, spike_counts_of(poisson_spike_times,
                                                              time_axis), "jax")
        broken = dict(fits)
        broken["non_local"] = dataclasses.replace(fits["non_local"], converged=False)

        stats, *_ = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
            prefitted_immobility=broken,
        )

        assert np.isnan(stats["non_local vs. constant: lr"])
        assert stats["min_aic_model"] != "non_local"
        assert stats["min_aic_model_replay_split"] != "non_local"
        assert np.isfinite(stats["SWR vs. constant: lr"])

    def test_absent_event_class_marks_dependent_outputs_as_unavailable(
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        position_info,
        sampling_frequency,
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
            "SWR": pd.DataFrame({"start_time": [1.0, 5.0], "end_time": [1.1, 5.1]}),
            "non_local": pd.DataFrame(
                {"start_time": [1.0, 5.0], "end_time": [1.1, 5.1]}
            ),
            "non_local_SWR": pd.DataFrame(
                {"start_time": [1.0, 5.0], "end_time": [1.1, 5.1]}
            ),
            "non_local_no_SWR": pd.DataFrame(columns=["start_time", "end_time"]),
            "matched_control": pd.DataFrame(
                {"start_time": [2.0, 6.0], "end_time": [2.1, 6.1]}
            ),
        }
        dm, spline_basis, lag_time = _make_design_matrices(
            event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )
        stats, swr_mod, no_swr_mod, nl_swr_mod, nl_no_swr_mod, mc_mod = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
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
            for suffix in (": lr", ": pval", ": frac_null_dev", ": frac_explained_dev"):
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
            "SWR_max_mod",
            "SWR_max_mod_time",
            "non_local_no_SWR_max_mod",
            "non_local_no_SWR_max_mod_time",
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
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        simple_event_times,
        position_info,
        sampling_frequency,
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
            simple_event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )
        stats, *_ = process_neuron(
            poisson_spike_times,
            is_immobile,
            theta_phase,
            time_axis,
            dm,
            spline_basis,
            lag_time,
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


@pytest.fixture
def dense_session():
    """A 120 s epoch with enough events and spikes that no fit separates.

    Every spline column then has hundreds of spikes in its support, which is
    the regime a real session is in and the regime the batched fitter is for.
    """
    sampling_frequency = 500.0
    time = np.arange(0, 120, 1.0 / sampling_frequency)
    position_info = pd.DataFrame({"head_speed": np.full(len(time), 2.0)}, index=time)
    fit_mask = position_info.head_speed < 4.0

    rng = np.random.default_rng(11)
    onsets = np.sort(rng.uniform(1.0, 118.0, size=60))
    event_times = {
        "non_local": pd.DataFrame({"start_time": onsets, "end_time": onsets + 0.08}),
        "non_local_SWR": pd.DataFrame(
            {"start_time": onsets[::2], "end_time": onsets[::2] + 0.08}
        ),
        "non_local_no_SWR": pd.DataFrame(
            {"start_time": onsets[1::2], "end_time": onsets[1::2] + 0.08}
        ),
        "SWR": pd.DataFrame(
            {"start_time": onsets[::2] - 0.01, "end_time": onsets[::2] + 0.07}
        ),
        "matched_control": pd.DataFrame(
            {"start_time": onsets + 0.4, "end_time": onsets + 0.48}
        ),
    }
    lags = np.arange(-50, 51)
    design_matrices, spline_basis, lag_time = _make_design_matrices(
        event_times,
        position_info,
        sampling_frequency=sampling_frequency,
        lags=lags,
    )
    # ~20 Hz, so every basis column's support holds hundreds of spikes.
    spike_count = rng.poisson(20.0 / sampling_frequency, size=len(time))
    spike_times = np.repeat(time, spike_count) + rng.uniform(
        0.0, 1.0 / sampling_frequency, size=int(spike_count.sum())
    )
    return {
        "design_matrices": design_matrices,
        "spline_basis": spline_basis,
        "lag_time": lag_time,
        "spike_count": spike_count,
        "spike_times": np.sort(spike_times),
        "fit_mask": fit_mask,
        "theta_phase": pd.Series(
            np.mod(2 * np.pi * 8 * time, 2 * np.pi) - np.pi, index=time
        ),
        "time": time,
    }


class TestFitterEquivalence:
    """Unpenalised, the batched backend must not move a single shipped number.

    ``process_neuron`` is the unit of comparison because it is what the
    pipeline calls: everything the bundle carries -- the LR tests, the
    fractional deviances, the AIC selections and the modulation curves -- is
    derived from the fits, so comparing its outputs compares the fits through
    every transformation they are read through.

    The comparison is run on a well-posed session, because that is where there
    is anything to compare: where a spline column's support holds no spikes the
    maximum-likelihood estimate is at infinity and the two solvers disagree
    about a number neither of them means. That regime is what the prior is for,
    and statsmodels has no penalised path to check it against.
    """

    #: Far tighter than the real-data audit's gate; on a synthetic session the
    #: two backends differ only by the order float64 sums are accumulated in.
    ATOL = 1e-8

    def test_process_neuron_outputs_are_the_same_under_either_fitter(
        self, dense_session
    ):
        arguments = (
            dense_session["spike_times"],
            dense_session["fit_mask"],
            dense_session["theta_phase"],
            dense_session["time"],
            dense_session["design_matrices"],
            dense_session["spline_basis"],
            dense_session["lag_time"],
        )

        sm_stats, *sm_mods = process_neuron(*arguments, fitter="statsmodels")
        jax_stats, *jax_mods = process_neuron(*arguments, fitter="jax")

        assert set(sm_stats) == set(jax_stats)
        for key, expected in sm_stats.items():
            actual = jax_stats[key]
            if isinstance(expected, str) or expected is None:
                assert actual == expected, key
            elif np.isnan(expected):
                assert np.isnan(actual), key
            else:
                assert abs(actual - expected) < self.ATOL, key

        for expected, actual in zip(sm_mods, jax_mods, strict=True):
            np.testing.assert_allclose(actual, expected, atol=self.ATOL, rtol=0)

    def test_the_batched_path_is_taken_and_agrees_on_a_well_posed_session(
        self, dense_session
    ):
        """The same comparison, read off the fits rather than off the outputs.

        The maximum-likelihood fit exists only where no spline column's support
        is empty of spikes; this fixture has enough events and enough spikes
        that every model has one, which is the regime the two backends can be
        compared in at all.
        """
        from continuum_swr_replay.analysis.pfc.batched_glm import fit_poisson_glms

        design_matrices = dense_session["design_matrices"]
        spike_count = dense_session["spike_count"]
        fit_mask = dense_session["fit_mask"]
        counts = spike_count[np.asarray(fit_mask, dtype=bool)]
        for name, design in design_matrices.items():
            batched = fit_poisson_glms(design[np.asarray(fit_mask, dtype=bool)], counts)
            assert batched.converged.all(), (
                f"{name} did not converge; the fixture is not well posed"
            )

        expected, _ = _fit_immobility_models(
            fit_mask, design_matrices, spike_count, fitter="statsmodels"
        )
        actual, models = _fit_immobility_models(
            fit_mask, design_matrices, spike_count, fitter="jax"
        )
        assert models == {}, "the batched path was not taken"

        for name in design_matrices:
            np.testing.assert_allclose(
                actual[name].params,
                expected[name].params,
                atol=self.ATOL,
                rtol=0,
                err_msg=name,
            )
            assert abs(actual[name].deviance - expected[name].deviance) < self.ATOL
            assert abs(actual[name].llf - expected[name].llf) < self.ATOL
            assert abs(actual[name].aic - expected[name].aic) < self.ATOL

    def test_theta_modulation_is_the_same_under_either_fitter(
        self, theta_phase, is_immobile, spike_count
    ):
        expected = run_theta_modulation_test(
            theta_phase, is_immobile, spike_count, fitter="statsmodels"
        )
        actual = run_theta_modulation_test(
            theta_phase, is_immobile, spike_count, fitter="jax"
        )

        for expected_value, actual_value in zip(expected, actual, strict=True):
            assert abs(actual_value - expected_value) < self.ATOL

    def test_the_regularized_theta_fit_is_statsmodels_under_both(
        self, theta_phase, is_immobile, spike_count
    ):
        """``theta_mod`` comes from the elastic net, whichever IRLS ran.

        statsmodels' ``fit_regularized(alpha=1e-20, L1_wt=0)`` lands ~1e-4 from
        the unregularized MLE, so substituting one for the other would move a
        shipped column by more than the audit's tolerance. Both paths must call
        the same estimator for it.
        """
        _, _, modulation, preferred_phase = run_theta_modulation_test(
            theta_phase, is_immobile, spike_count, fitter="jax"
        )
        joint_mask, design = build_theta_design(theta_phase, is_immobile)
        regularized = sm.GLM(
            spike_count[joint_mask], design, family=sm.families.Poisson()
        ).fit_regularized(alpha=1e-20, L1_wt=0.0)

        assert modulation == pytest.approx(
            np.hypot(regularized.params[1], regularized.params[2])
        )
        assert preferred_phase == pytest.approx(
            np.arctan2(regularized.params[2], regularized.params[1])
        )

    def test_prefitted_results_are_used_rather_than_refit(
        self, simple_event_times, position_info, is_immobile, spike_count
    ):
        """A session fits every neuron at once and hands each one its slice."""
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        prefitted, _ = _fit_immobility_models(
            is_immobile, dm, spike_count, fitter="jax"
        )

        results, models = _fit_immobility_models(
            is_immobile, dm, spike_count, prefitted=prefitted
        )

        assert models == {}
        assert all(results[name] is prefitted[name] for name in dm)

    def test_missing_prefitted_models_are_refused(
        self, simple_event_times, position_info, is_immobile, spike_count
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        prefitted, _ = _fit_immobility_models(
            is_immobile, dm, spike_count, fitter="jax"
        )
        del prefitted["SWR"]

        with pytest.raises(ValueError, match="missing models"):
            _fit_immobility_models(is_immobile, dm, spike_count, prefitted=prefitted)

    def test_an_unknown_fitter_is_refused(
        self, simple_event_times, position_info, is_immobile, spike_count
    ):
        dm, _, _ = _make_design_matrices(
            simple_event_times, position_info, lags=np.arange(-50, 51)
        )
        with pytest.raises(ValueError, match="fitter must be one of"):
            _fit_immobility_models(is_immobile, dm, spike_count, fitter="nemos")

    def test_the_provenance_record_names_the_backend(self):
        record = fitter_provenance("jax")
        assert record["fitter"] == "jax"
        assert record["fitter_tol"] == 1e-8
        assert record["jax_backend"] in {"cpu", "gpu", "tpu"}
        assert record["jax_version"]

        record = fitter_provenance("statsmodels")
        assert record["fitter"] == "statsmodels"
        assert record["fitter_tol"] == 1e-8
        assert record["jax_backend"] is None
        assert record["jax_version"] is None


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
    @pytest.mark.parametrize("fitter", ["statsmodels", "jax"])
    def test_default_fit_mask_reproduces_the_immobility_fit_bit_for_bit(
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        simple_event_times,
        position_info,
        sampling_frequency,
        fitter,
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
            fitter=fitter,
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
            fitter=fitter,
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
        ).times
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


class TestTheBatchedFitterRefusesAnUnconfiguredDevice:
    """A library caller cannot reach the GPU without going through the policy.

    The runner picks one card and bounds what JAX takes of it. Anything that
    imports the fitter directly -- a notebook, another script -- would
    otherwise get JAX's default on a shared host: every visible card, most of
    each preallocated, and other people's jobs evicted with no message.
    """

    @pytest.fixture
    def unconfigured(self, monkeypatch):
        from continuum_swr_replay.analysis.pfc import gpu as gpu_module

        monkeypatch.setattr(gpu_module, "_CONFIGURED", None)
        # The suite pins CPU, which is a legitimate way to fit; lift it so the
        # process looks like one that could reach a card.
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    def test_fit_models_batched_refuses(self, unconfigured):
        from continuum_swr_replay.analysis.pfc.gpu import UnconfiguredDeviceError

        design = np.ones((32, 2))
        design[:, 1] = np.arange(32)
        with pytest.raises(UnconfiguredDeviceError, match="configure_jax_gpu"):
            fit_models_batched({"constant": design}, np.ones((32, 3)))

    def test_the_batched_theta_fit_refuses(self, unconfigured):
        from continuum_swr_replay.analysis.pfc.gpu import UnconfiguredDeviceError

        with pytest.raises(UnconfiguredDeviceError, match="configure_jax_gpu"):
            _fit_theta_models(np.ones((32, 3)), np.ones((32, 2)), fitter="jax")

    def test_a_cpu_pinned_process_still_fits(self, monkeypatch):
        from continuum_swr_replay.analysis.pfc import gpu as gpu_module

        monkeypatch.setattr(gpu_module, "_CONFIGURED", None)
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        design = np.ones((64, 1))
        results = fit_models_batched({"constant": design}, np.ones((64, 2)))

        assert results["constant"].params.shape == (2, 1)


class TestShippedFitCounts:
    """What a session records about fits that never settled.

    A GLM that runs to ``maxiter`` still reports an LR test and an AIC, so the
    fact that it did not converge leaves no trace in the bundle. These counts
    are the only place it survives the run, which is why they are taken over
    the results that are actually read -- after the hand-back, so a neuron the
    batched fitter declined is counted on statsmodels' flag, not on its own.
    """

    class _Fit:
        def __init__(self, converged: bool) -> None:
            self.converged = converged

    def _neuron(self, *converged: bool) -> dict:
        return {f"m{i}": self._Fit(flag) for i, flag in enumerate(converged)}

    def test_converged_fits_count_as_fits_and_not_as_failures(self):
        counts = shipped_fit_counts(
            [self._neuron(True, True, True), self._neuron(True, True, True)]
        )

        assert counts == {
            "n_model_fits": 6,
            "n_nonconverged_model_fits": 0,
            "n_neurons_with_nonconverged_model": 0,
        }

    def test_failures_are_counted_per_fit_and_per_neuron(self):
        # Two failures on one neuron is two fits but one neuron: the two
        # denominators answer different questions and must not be conflated.
        counts = shipped_fit_counts(
            [self._neuron(True, False, False), self._neuron(True, True, True)]
        )

        assert counts["n_model_fits"] == 6
        assert counts["n_nonconverged_model_fits"] == 2
        assert counts["n_neurons_with_nonconverged_model"] == 1

    def test_a_neuron_with_no_fit_at_all_enters_neither_total(self):
        # A response with no spikes has no fit at all, so that neuron has no
        # results; counting it as neurons x models would invent fits.
        counts = shipped_fit_counts([None, self._neuron(True, False)])

        assert counts["n_model_fits"] == 2
        assert counts["n_nonconverged_model_fits"] == 1
        assert counts["n_neurons_with_nonconverged_model"] == 1

    def test_a_result_with_no_convergence_flag_is_read_as_converged(self):
        # ``fit_regularized`` results carry no flag; absence is not failure.
        class Bare:
            pass

        counts = shipped_fit_counts([{"m0": Bare()}])

        assert counts["n_model_fits"] == 1
        assert counts["n_nonconverged_model_fits"] == 0

    def test_a_real_prefit_reports_counts_consistent_with_its_fits(
        self,
        poisson_spike_times,
        is_immobile,
        theta_phase,
        time_axis,
        simple_event_times,
        position_info,
        sampling_frequency,
    ):
        design_matrices, _, _ = _make_design_matrices(
            simple_event_times,
            position_info,
            sampling_frequency=sampling_frequency,
            lags=np.arange(-50, 51),
        )

        *_, counts = _prefit_session(
            [poisson_spike_times],
            np.asarray(time_axis),
            is_immobile,
            is_immobile,
            design_matrices,
            theta_phase,
            logger=logging.getLogger("test_shipped_fit_counts"),
        )

        assert counts["n_model_fits"] in (0, len(design_matrices))
        assert (
            0 <= counts["n_nonconverged_model_fits"] <= max(counts["n_model_fits"], 0)
        )
        assert counts["n_neurons_with_nonconverged_model"] <= 1
