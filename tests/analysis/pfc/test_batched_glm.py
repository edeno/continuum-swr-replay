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
    DEFAULT_TOL,
    BatchedGLMResults,
    fit_poisson_glms,
    jax_backend,
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
    """Where the two fitters part company, and how that is reported.

    statsmodels solves each IRLS step as a least squares on the weighted
    design; this fitter solves the normal equations in an orthonormal basis.
    They agree while the fitted means stay within a few orders of magnitude of
    one another. A design column with no spikes in its support sends its
    coefficient to -inf, the fitted means spread past what a Gram matrix can
    carry, and the two solvers take different steps. The fitter has to *say*
    so rather than quietly report a near-miss.
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

    def test_a_separating_fit_is_flagged_for_a_reference_fit(self, separating):
        design, counts = separating
        batched = fit_poisson_glms(design, counts)

        assert batched.needs_reference_fit[0]
        assert batched.for_unit(0).needs_reference_fit is True

    def test_a_well_posed_fit_is_not_flagged(self, planted):
        design, counts, _ = planted
        batched = fit_poisson_glms(design, counts)

        assert not batched.needs_reference_fit.any()

    def test_a_unit_without_spikes_is_not_offered_for_a_reference_fit(self):
        """statsmodels refuses it too, so there is no reference to fall back to."""
        design = np.column_stack([np.ones(800), np.linspace(-1, 1, 800)])
        batched = fit_poisson_glms(design, np.zeros(800))

        assert not batched.converged[0]
        assert not batched.needs_reference_fit[0]


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
