"""The batched Poisson GLM fitter against the statsmodels fits it replaces.

Every check here is value-level against ``statsmodels.api.GLM(...).fit()``,
because that is the whole contract: the batched fitter exists to be faster, not
to be different. The tolerances are far below the ones the real-data
equivalence audit is gated on.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import statsmodels.api as sm

from continuum_swr_replay.analysis.pfc.batched_glm import (
    DEFAULT_MAXITER,
    DEFAULT_PENALTY_SIGMA,
    DEFAULT_TOL,
    BatchedGLMResults,
    fit_poisson_glms,
    implied_prior_sd,
    jax_backend,
    make_ridge_penalty,
)

# Well below the audit's gate (params 1e-5, deviance/llf 1e-6): the two fitters
# solve the same equations from the same starting point with the same stopping
# rule, so agreement is at the level of float64 arithmetic, not of the model.
ATOL = 1e-8


def statsmodels_fit(design, counts):
    """The per-unit fit the batched fitter must reproduce."""
    return sm.GLM(counts, design, family=sm.families.Poisson()).fit()


@pytest.fixture
def planted():
    """Poisson counts for five units from a shared six-column design."""
    rng = np.random.default_rng(20260914)
    n_time, n_features, n_units = 4000, 6, 5
    design = np.column_stack(
        [np.ones(n_time), rng.normal(size=(n_time, n_features - 1)) * 0.4]
    )
    coefficients = np.column_stack(
        [
            np.linspace(-2.5, -0.5, n_units),
            rng.normal(size=(n_units, n_features - 1)) * 0.3,
        ]
    )
    counts = rng.poisson(np.exp(design @ coefficients.T)).astype(float)
    return design, counts, coefficients


class TestAgreementWithStatsmodels:
    def test_params_deviance_llf_and_aic_match_per_unit(self, planted):
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts)

        for unit in range(counts.shape[1]):
            expected = statsmodels_fit(design, counts[:, unit])
            actual = batched.for_unit(unit)
            np.testing.assert_allclose(
                actual.params, expected.params, atol=ATOL, rtol=0
            )
            assert abs(actual.deviance - expected.deviance) < ATOL
            assert abs(actual.llf - expected.llf) < ATOL
            assert abs(actual.aic - expected.aic) < ATOL

    def test_degrees_of_freedom_match_statsmodels(self, planted):
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts)
        expected = statsmodels_fit(design, counts[:, 0])

        actual = batched.for_unit(0)
        assert actual.df_resid == expected.df_resid
        assert actual.df_model == expected.df_model
        assert actual.nobs == design.shape[0]

    def test_iteration_count_and_convergence_flag_match(self, planted):
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts)

        for unit in range(counts.shape[1]):
            expected = statsmodels_fit(design, counts[:, unit])
            actual = batched.for_unit(unit)
            assert actual.converged is True
            assert actual.converged == expected.converged
            # Same starting values, same step, same rule: the same number of
            # IRLS steps, not merely the same answer.
            assert actual.n_iter == expected.fit_history["iteration"]

    def test_the_planted_coefficients_are_recovered(self, planted):
        design, counts, coefficients = planted
        batched = fit_poisson_glms(design, counts)
        # A sanity floor under the statsmodels comparison: both could agree on
        # a wrong answer, so the estimates must also be near the truth.
        np.testing.assert_allclose(batched.params, coefficients, atol=0.15)

    def test_a_single_unit_may_be_passed_as_a_1d_array(self, planted):
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts[:, 2])

        assert batched.n_units == 1
        np.testing.assert_allclose(
            batched.for_unit(0).params,
            statsmodels_fit(design, counts[:, 2]).params,
            atol=ATOL,
            rtol=0,
        )

    def test_standard_errors_are_the_information_at_the_fitted_parameters(
        self, planted
    ):
        """``sqrt(diag((X' W X)^+))`` at the parameters that were returned.

        The single-neuron figure panel draws a +/-1 SE band, so the batched
        path has to carry standard errors. They are exact against a direct
        dense evaluation, and agree with statsmodels to ~1e-7 rather than to
        ATOL -- statsmodels reports the covariance of its *last weighted
        least-squares step*, whose weights are one IRLS iteration behind the
        parameters it returns. The same 1e-7 separates statsmodels from the
        dense evaluation, so the gap is statsmodels' lag and not this
        fitter's arithmetic.
        """
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts)

        for unit in range(counts.shape[1]):
            actual = batched.for_unit(unit)
            fitted_mean = np.exp(design @ actual.params)
            dense = np.sqrt(
                np.diag(np.linalg.pinv(design.T @ (design * fitted_mean[:, None])))
            )
            np.testing.assert_allclose(actual.bse, dense, atol=1e-12, rtol=0)

            expected = statsmodels_fit(design, counts[:, unit])
            np.testing.assert_allclose(actual.bse, expected.bse, atol=1e-6, rtol=0)

    def test_batching_does_not_change_a_units_answer(self, planted):
        """A unit fit alone and the same unit fit in a batch give one answer.

        The convergence rule is per unit, so a unit that has converged must be
        frozen rather than carried along by the units still iterating -- it
        must take the same number of steps and land in the same place. (The
        agreement is to float64 noise, not bit-for-bit: the batched cross
        products are one matrix multiply over all units, which does not
        accumulate in the same order as a single-column one.)
        """
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts)

        for unit in range(counts.shape[1]):
            alone = fit_poisson_glms(design, counts[:, unit]).for_unit(0)
            together = batched.for_unit(unit)
            np.testing.assert_allclose(
                alone.params, together.params, atol=1e-12, rtol=0
            )
            assert abs(alone.deviance - together.deviance) < 1e-9
            assert alone.n_iter == together.n_iter

    def test_the_answer_does_not_depend_on_the_time_chunking(self, planted):
        design, counts, _ = planted
        coarse = fit_poisson_glms(design, counts, chunk_rows=4096)
        fine = fit_poisson_glms(design, counts, chunk_rows=1024)

        np.testing.assert_allclose(coarse.params, fine.params, atol=ATOL, rtol=0)


class TestRankDeficientDesign:
    """An event class with no events leaves its spline columns all-zero.

    statsmodels solves such a design through a pseudo-inverse and returns
    finite parameters; the rank-deficiency gate downstream is what routes the
    affected statistics to NaN. The batched fitter must behave the same way --
    raising, or returning NaN, would send a neuron down the failure path
    instead.
    """

    @pytest.fixture
    def deficient(self):
        rng = np.random.default_rng(7)
        n_time = 3000
        informative = rng.normal(size=n_time) * 0.4
        design = np.column_stack(
            [
                np.ones(n_time),
                informative,
                np.zeros(n_time),  # an event class with no events
                rng.normal(size=n_time) * 0.3,
                informative,  # a duplicate of an earlier column
            ]
        )
        counts = rng.poisson(
            np.exp(design @ np.array([-1.0, 0.3, 0.0, -0.2, 0.0]))
        ).astype(float)
        return design, counts

    def test_rank_is_below_the_number_of_features(self, deficient):
        design, counts = deficient
        batched = fit_poisson_glms(design, counts)

        assert batched.rank == np.linalg.matrix_rank(design)
        assert batched.rank < batched.n_features

    def test_params_match_the_statsmodels_pinv_solution(self, deficient):
        design, counts = deficient
        batched = fit_poisson_glms(design, counts)
        expected = statsmodels_fit(design, counts)

        actual = batched.for_unit(0)
        assert np.all(np.isfinite(actual.params))
        np.testing.assert_allclose(actual.params, expected.params, atol=1e-7, rtol=0)
        # The minimum-norm solution splits a duplicated column's effect evenly
        # and leaves an all-zero column at exactly zero.
        assert actual.params[2] == pytest.approx(0.0, abs=1e-12)
        assert actual.params[1] == pytest.approx(actual.params[4], abs=1e-9)

    def test_standard_errors_follow_the_pinv_solution(self, deficient):
        design, counts = deficient
        batched = fit_poisson_glms(design, counts)
        expected = statsmodels_fit(design, counts)

        actual = batched.for_unit(0)
        np.testing.assert_allclose(actual.bse, expected.bse, atol=1e-6, rtol=0)
        # A column the design does not span has an exactly zero parameter, and
        # no uncertainty about it either.
        assert actual.bse[2] == pytest.approx(0.0, abs=1e-12)

    def test_aic_counts_the_rank_not_the_columns(self, deficient):
        design, counts = deficient
        batched = fit_poisson_glms(design, counts)
        expected = statsmodels_fit(design, counts)

        actual = batched.for_unit(0)
        assert abs(actual.aic - expected.aic) < ATOL
        # statsmodels' aic is -2 llf + 2 (df_model + 1) = -2 llf + 2 rank, so
        # on a rank-deficient design it is NOT -2 llf + 2 n_features.
        assert actual.aic == pytest.approx(-2 * actual.llf + 2 * actual.rank)
        assert actual.aic != pytest.approx(-2 * actual.llf + 2 * actual.n_features)


class TestUnitsWithoutSpikes:
    def test_statsmodels_refuses_an_all_zero_response(self):
        """The behaviour the batched fitter has to reproduce per unit."""
        design = np.column_stack([np.ones(500), np.linspace(-1, 1, 500)])
        with pytest.raises(ValueError, match="deviance function returned a nan"):
            statsmodels_fit(design, np.zeros(500))

    def test_an_all_zero_unit_comes_back_nan_and_the_batch_survives(self):
        rng = np.random.default_rng(3)
        n_time = 2000
        design = np.column_stack([np.ones(n_time), rng.normal(size=n_time) * 0.5])
        spiking = rng.poisson(np.exp(design @ np.array([-1.0, 0.4]))).astype(float)
        counts = np.column_stack([np.zeros(n_time), spiking])

        batched = fit_poisson_glms(design, counts)

        silent = batched.for_unit(0)
        assert np.all(np.isnan(silent.params))
        assert np.isnan(silent.deviance)
        assert np.isnan(silent.llf)
        assert np.isnan(silent.aic)
        assert silent.converged is False

        # The spiking unit in the same batch is untouched by its neighbour.
        alive = batched.for_unit(1)
        np.testing.assert_allclose(
            alive.params, statsmodels_fit(design, spiking).params, atol=ATOL, rtol=0
        )


class TestNonConvergence:
    def test_a_unit_that_hits_maxiter_is_reported_unconverged(self, planted):
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts, maxiter=1)

        assert not batched.converged.any()
        assert (batched.n_iter == 1).all()
        # The parameters are still the one step that was taken, not NaN.
        assert np.all(np.isfinite(batched.params))

    def test_an_empty_design_reports_every_unit_as_undefined(self):
        batched = fit_poisson_glms(np.empty((0, 3)), np.empty((0, 2)))

        assert batched.rank == 0
        assert batched.nobs == 0
        assert np.all(np.isnan(batched.params))
        assert not batched.converged.any()


class TestSeparation:
    """A design column whose support holds no spikes, with and without a prior.

    Unpenalised, that coefficient's maximum-likelihood estimate is at ``-inf``
    and whatever a solver reports is an artefact of where it stopped. The
    prior's whole job is to make the fit finite there, and to leave it alone
    everywhere else.
    """

    @pytest.fixture
    def separating(self):
        """A column whose support holds no spikes: the MLE is at -inf."""
        rng = np.random.default_rng(5)
        n_time = 4000
        indicator = np.zeros(n_time)
        indicator[100:200] = 1.0
        design = np.column_stack([np.ones(n_time), indicator])
        counts = rng.poisson(0.05, size=n_time).astype(float)
        counts[100:200] = 0.0
        return design, counts

    def test_the_unpenalised_fit_runs_away(self, separating):
        design, counts = separating
        batched = fit_poisson_glms(design, counts)

        # Not an error, and not NaN -- just a number with no meaning: a rate
        # multiplier of 1e-10 on the bins the column covers.
        assert batched.params[0, 1] < -15.0

    def test_the_prior_makes_the_separated_coefficient_finite(self, separating):
        design, counts = separating
        penalty = make_ridge_penalty(design.shape[1], sigma=2.0)

        batched = fit_poisson_glms(design, counts, penalty=penalty)

        assert batched.converged[0]
        assert np.all(np.isfinite(batched.params))
        # "at most a small fraction of baseline", which is what the prior has
        # to say about a column with no spikes under it -- not -infinity, and
        # not zero either.
        assert -3.0 < batched.params[0, 1] < -2.0
        assert batched.n_iter[0] < DEFAULT_MAXITER

    def test_the_intercept_still_carries_the_baseline_rate(self, separating):
        """The prior is on the basis columns; the intercept is left free."""
        design, counts = separating
        penalty = make_ridge_penalty(design.shape[1], sigma=2.0)

        batched = fit_poisson_glms(design, counts, penalty=penalty)

        # Every bin outside the indicator's support keeps its own mean rate,
        # to within the sliver of it the now-finite indicator coefficient
        # accounts for (exp(-2.2) x 100 bins out of 4000).
        outside = counts[design[:, 1] == 0.0]
        assert batched.params[0, 0] == pytest.approx(np.log(outside.mean()), abs=5e-3)


class TestPenalty:
    """The Gaussian prior, against a dense penalised Newton solve."""

    @staticmethod
    def dense_penalized_fit(design, counts, penalty, maxiter=200, tol=1e-10):
        """Penalised IRLS in the design's own columns, in dense NumPy.

        The estimator the batched solver must reproduce, written the obvious
        way: no orthonormal basis, no chunking, no batching -- solve
        ``(X' W X + Lambda) beta = X' W z`` on the working response until the
        penalised deviance settles.
        """
        penalty_matrix = np.diag(penalty)
        mean = (counts + counts.mean()) / 2.0
        eta = np.log(mean)
        beta = np.zeros(design.shape[1])
        previous = np.inf
        for _ in range(maxiter):
            weight = mean
            working = eta + (counts - mean) / mean
            beta = np.linalg.solve(
                design.T @ (design * weight[:, None]) + penalty_matrix,
                design.T @ (weight * working),
            )
            eta = design @ beta
            mean = np.exp(eta)
            ratio = np.where(counts > 0, counts / mean, 1.0)
            deviance = 2.0 * np.sum(counts * np.log(ratio) - (counts - mean))
            penalized = deviance + beta @ penalty_matrix @ beta
            if abs(penalized - previous) < tol:
                break
            previous = penalized
        return beta

    def test_a_zero_penalty_is_the_unpenalised_fit_bit_for_bit(self, planted):
        design, counts, _ = planted
        unpenalised = fit_poisson_glms(design, counts)

        zeros = fit_poisson_glms(design, counts, penalty=np.zeros(design.shape[1]))

        assert np.array_equal(zeros.params, unpenalised.params)
        assert np.array_equal(zeros.deviance, unpenalised.deviance)
        assert np.array_equal(zeros.llf, unpenalised.llf)
        assert np.array_equal(zeros.bse, unpenalised.bse)
        assert np.array_equal(zeros.n_iter, unpenalised.n_iter)

    def test_the_penalised_fit_matches_a_dense_newton_reference(self, planted):
        design, counts, _ = planted
        penalty = make_ridge_penalty(design.shape[1], sigma=2.0)

        batched = fit_poisson_glms(design, counts, penalty=penalty)

        for unit in range(counts.shape[1]):
            expected = self.dense_penalized_fit(design, counts[:, unit], penalty)
            np.testing.assert_allclose(
                batched.params[unit], expected, atol=ATOL, rtol=0
            )

    def test_the_penalty_shrinks_only_the_penalised_columns(self, planted):
        """A column with lambda 0 is fit exactly as it was; its neighbours are not.

        Not a tautology: the penalty is stated in the design's own columns but
        applied in the orthonormal basis, where a wrong transform would leak
        shrinkage into the free directions.
        """
        design, counts, _ = planted
        penalty = np.zeros(design.shape[1])
        penalty[-1] = 10.0

        batched = fit_poisson_glms(design, counts, penalty=penalty)
        unpenalised = fit_poisson_glms(design, counts)

        assert np.all(np.abs(batched.params[:, -1]) < np.abs(unpenalised.params[:, -1]))
        for unit in range(counts.shape[1]):
            expected = self.dense_penalized_fit(design, counts[:, unit], penalty)
            np.testing.assert_allclose(
                batched.params[unit], expected, atol=ATOL, rtol=0
            )

    def test_an_intercept_only_design_is_untouched_by_a_zero_penalty(self, planted):
        _, counts, _ = planted
        constant = np.ones((counts.shape[0], 1))

        penalised = fit_poisson_glms(constant, counts, penalty=np.zeros(1))
        plain = fit_poisson_glms(constant, counts)

        assert np.array_equal(penalised.params, plain.params)
        np.testing.assert_allclose(
            penalised.params[:, 0], np.log(counts.mean(axis=0)), atol=1e-10
        )

    def test_standard_errors_are_the_penalised_posterior_mode(self, planted):
        design, counts, _ = planted
        penalty = make_ridge_penalty(design.shape[1], sigma=2.0)

        batched = fit_poisson_glms(design, counts, penalty=penalty)

        for unit in range(counts.shape[1]):
            fitted_mean = np.exp(design @ batched.params[unit])
            dense = np.sqrt(
                np.diag(
                    np.linalg.pinv(
                        design.T @ (design * fitted_mean[:, None]) + np.diag(penalty)
                    )
                )
            )
            np.testing.assert_allclose(batched.bse[unit], dense, atol=1e-12, rtol=0)

    def test_a_rank_deficient_design_is_still_the_minimum_norm_solution(self):
        """An all-zero column stays at exactly zero under the prior too."""
        rng = np.random.default_rng(11)
        n_time = 3000
        design = np.column_stack(
            [np.ones(n_time), rng.normal(size=n_time) * 0.4, np.zeros(n_time)]
        )
        counts = rng.poisson(np.exp(design @ np.array([-1.0, 0.3, 0.0]))).astype(float)

        batched = fit_poisson_glms(
            design, counts, penalty=make_ridge_penalty(3, sigma=2.0)
        )

        assert batched.params[0, 2] == pytest.approx(0.0, abs=1e-12)
        assert batched.bse[0, 2] == pytest.approx(0.0, abs=1e-12)

    def test_the_covariance_is_the_penalised_inverse_information(self, planted):
        """The curve helpers need the off-diagonal terms, not just ``bse``."""
        design, counts, _ = planted
        penalty = make_ridge_penalty(design.shape[1], sigma=2.0)

        batched = fit_poisson_glms(design, counts, penalty=penalty)

        for unit in range(counts.shape[1]):
            fitted_mean = np.exp(design @ batched.params[unit])
            dense = np.linalg.pinv(
                design.T @ (design * fitted_mean[:, None]) + np.diag(penalty)
            )
            np.testing.assert_allclose(
                batched.covariance[unit], dense, atol=1e-12, rtol=0
            )
            # And ``bse`` is its diagonal, so the two can never disagree.
            np.testing.assert_allclose(
                batched.bse[unit], np.sqrt(np.diag(batched.covariance[unit])), rtol=0,
                atol=1e-15,
            )

    def test_a_failed_unit_has_no_covariance_either(self):
        design = np.column_stack([np.ones(600), np.linspace(-1, 1, 600)])

        batched = fit_poisson_glms(design, np.zeros(600))

        assert np.all(np.isnan(batched.covariance[0]))

    def test_make_ridge_penalty_leaves_the_intercept_free(self):
        penalty = make_ridge_penalty(5, sigma=2.0)

        np.testing.assert_allclose(penalty, [0.0, 0.25, 0.25, 0.25, 0.25])

    def test_a_malformed_penalty_is_refused(self, planted):
        design, counts, _ = planted
        with pytest.raises(ValueError, match="one value per design column"):
            fit_poisson_glms(design, counts, penalty=np.zeros(design.shape[1] - 1))
        with pytest.raises(ValueError, match="non-negative"):
            fit_poisson_glms(design, counts, penalty=-np.ones(design.shape[1]))
        with pytest.raises(ValueError, match="sigma must be positive"):
            make_ridge_penalty(3, sigma=0.0)


class TestResultsContract:
    def test_for_unit_exposes_what_the_pipeline_reads(self, planted):
        design, counts, _ = planted
        unit = fit_poisson_glms(design, counts).for_unit(0)

        for attribute in (
            "params",
            "llf",
            "deviance",
            "aic",
            "converged",
            "df_resid",
            "df_model",
        ):
            assert hasattr(unit, attribute), attribute
        assert isinstance(unit, type(unit))
        assert unit.params.shape == (design.shape[1],)

    def test_batched_arrays_are_numpy_and_indexed_by_unit(self, planted):
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts)

        assert isinstance(batched, BatchedGLMResults)
        assert isinstance(batched.params, np.ndarray)
        assert batched.params.shape == (counts.shape[1], design.shape[1])
        for array in (batched.llf, batched.deviance, batched.aic):
            assert array.shape == (counts.shape[1],)

    def test_the_defaults_are_the_statsmodels_defaults(self):
        assert DEFAULT_MAXITER == 100
        assert DEFAULT_TOL == 1e-8

    def test_mismatched_row_counts_are_refused(self, planted):
        design, counts, _ = planted
        with pytest.raises(ValueError, match="rows"):
            fit_poisson_glms(design, counts[:-1])


class TestBackend:
    def test_the_backend_is_named_for_the_provenance_record(self):
        assert jax_backend() in {"cpu", "gpu", "tpu"}

    def test_the_fit_runs_on_cpu(self, planted):
        """Forced onto CPU, so the suite is the same on a GPU machine."""
        design, counts, _ = planted
        with jax.default_device(jax.devices("cpu")[0]):
            batched = fit_poisson_glms(design, counts)

        np.testing.assert_allclose(
            batched.for_unit(0).params,
            statsmodels_fit(design, counts[:, 0]).params,
            atol=ATOL,
            rtol=0,
        )

    def test_float64_is_enabled(self, planted):
        design, counts, _ = planted
        assert jax.config.read("jax_enable_x64")
        assert fit_poisson_glms(design, counts).params.dtype == np.float64


class TestImpliedPriorSd:
    """What a per-coefficient sigma says about the curve it shapes.

    Two readings, and they differ by a factor of ``sqrt(n_basis)``: a sigma
    picked to look loose on one coefficient is tight on a modulation held
    across the window. The audit quotes these rather than restating them.
    """

    @staticmethod
    def basis(n_basis=11, n_lags=251):
        """A partition of unity, as the event basis is."""
        from continuum_swr_replay.analysis.pfc.glm import event_basis_functions

        spline_basis, lag_time, _ = event_basis_functions(
            np.arange(-125, 126), 500.0
        )
        assert spline_basis.shape == (n_lags, n_basis)
        return spline_basis, lag_time

    def test_the_sustained_reading_is_sigma_over_root_n_basis(self):
        spline_basis, _ = self.basis()

        implied = implied_prior_sd(3.0, spline_basis)

        assert implied["sustained"] == pytest.approx(3.0 / np.sqrt(11))

    def test_the_per_lag_reading_is_sigma_times_the_basis_norm(self):
        spline_basis, lag_time = self.basis()
        window = np.abs(lag_time) <= 0.2

        implied = implied_prior_sd(3.0, spline_basis, window)

        norms = np.sqrt((spline_basis[window] ** 2).sum(axis=1))
        assert implied["per_lag"] == pytest.approx(
            3.0 * np.sqrt((norms**2).mean())
        )
        assert implied["per_lag_min"] == pytest.approx(3.0 * norms.min())
        assert implied["per_lag_max"] == pytest.approx(3.0 * norms.max())

    def test_a_sustained_modulation_is_the_tighter_of_the_two(self):
        """The asymmetry the default sigma was chosen for."""
        spline_basis, lag_time = self.basis()
        window = np.abs(lag_time) <= 0.2

        implied = implied_prior_sd(DEFAULT_PENALTY_SIGMA, spline_basis, window)

        assert implied["sustained"] < DEFAULT_PENALTY_SIGMA < implied["per_lag"] * 2
        # The gap between the two readings is a property of the basis alone:
        # sqrt(n_basis) times the basis norm, about 2.3 on this one.
        norms = np.sqrt((spline_basis[window] ** 2).sum(axis=1))
        assert implied["per_lag"] / implied["sustained"] == pytest.approx(
            np.sqrt(11 * (norms**2).mean())
        )

    def test_the_default_prior_admits_the_modulations_the_corpus_shows(self):
        """x0.02 to x57 at one lag, x0.17 to x5.9 held across the window.

        Rat mPFC units are reported to move by at most ~10x above baseline and
        to near silence below, so the prior has to hold that comfortably at one
        lag and still bound a sustained response.
        """
        spline_basis, lag_time = self.basis()
        window = np.abs(lag_time) <= 0.2

        implied = implied_prior_sd(DEFAULT_PENALTY_SIGMA, spline_basis, window)

        assert implied["per_lag_multiplier_high"] > 20.0
        assert implied["per_lag_multiplier_low"] < 0.05
        assert 3.0 < implied["sustained_multiplier_high"] < 10.0
        assert implied["sustained_multiplier_low"] == pytest.approx(
            1.0 / implied["sustained_multiplier_high"]
        )

    def test_it_scales_linearly_in_sigma(self):
        spline_basis, _ = self.basis()

        loose = implied_prior_sd(4.5, spline_basis)
        tight = implied_prior_sd(1.5, spline_basis)

        assert loose["per_lag"] == pytest.approx(3.0 * tight["per_lag"])
        assert loose["sustained"] == pytest.approx(3.0 * tight["sustained"])

    def test_a_non_positive_sigma_is_refused(self):
        spline_basis, _ = self.basis()

        with pytest.raises(ValueError, match="sigma must be positive"):
            implied_prior_sd(0.0, spline_basis)

    def test_the_default_is_the_documented_one(self):
        assert DEFAULT_PENALTY_SIGMA == 3.0
