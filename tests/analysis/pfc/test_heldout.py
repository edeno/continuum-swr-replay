"""Blocked folds, out-of-fold deviance, and the control-draw null.

Value-level throughout: the fold assignment is checked against the blocks it
was built from, the held-out score against the model that generated the data,
and the null against the uniform distribution a valid p-value has to have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from continuum_swr_replay.analysis.pfc.batched_glm import (
    BatchedGLMResults,
    make_ridge_penalty,
)
from continuum_swr_replay.analysis.pfc.glm import (
    convolve_event_basis,
    event_basis_functions,
    event_onset_counts,
)
from continuum_swr_replay.analysis.pfc.heldout import (
    assign_blocked_folds,
    control_design,
    control_draw_null,
    empirical_pvalue,
    heldout_deviance,
    null_draw_seeds,
    poisson_deviance,
)

SAMPLING_FREQUENCY = 500.0


def bouts_mask(run_lengths, gap=40):
    """A fit mask with the given contiguous True runs, separated by gaps."""
    pieces = []
    for length in run_lengths:
        pieces.append(np.ones(length, dtype=bool))
        pieces.append(np.zeros(gap, dtype=bool))
    mask = np.concatenate(pieces)
    time = np.arange(len(mask)) / SAMPLING_FREQUENCY
    return pd.Series(mask, index=time)


def runs_of(mask):
    flat = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], flat, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2)


class TestAssignBlockedFolds:
    def test_no_run_is_split_across_folds(self):
        mask = bouts_mask([300, 120, 450, 90, 200, 610, 75, 260])

        folds = assign_blocked_folds(mask, n_folds=5, seed=7)

        assert folds.shape == (int(np.asarray(mask).sum()),)
        position = 0
        for start, stop in runs_of(mask):
            length = stop - start
            block = folds[position : position + length]
            assert len(set(block.tolist())) == 1, "a run was split across folds"
            position += length

    def test_every_fold_is_used(self):
        mask = bouts_mask([300, 120, 450, 90, 200, 610, 75, 260])

        folds = assign_blocked_folds(mask, n_folds=5, seed=7)

        assert set(folds.tolist()) == set(range(5))

    def test_fold_sizes_are_within_one_run_of_each_other(self):
        """The bound assigning each run to the lightest fold gives.

        Every seed, not one: a rule that happened to balance for the seed the
        test picked would be worth nothing.
        """
        run_lengths = [300, 120, 450, 90, 200, 610, 75, 260, 380, 140, 520, 95]
        mask = bouts_mask(run_lengths)

        for seed in range(25):
            sizes = np.bincount(assign_blocked_folds(mask, n_folds=5, seed=seed))
            assert sizes.max() - sizes.min() <= max(run_lengths), seed

    def test_on_a_session_shaped_mask_the_folds_are_within_a_few_percent(self):
        """The bound is one bout, which is small next to a session's fold."""
        rng = np.random.default_rng(0)
        run_lengths = rng.integers(200, 4000, size=200).tolist()
        mask = bouts_mask(run_lengths, gap=50)

        sizes = np.bincount(assign_blocked_folds(mask, n_folds=5, seed=3))

        assert (sizes.max() - sizes.min()) < 0.10 * sizes.mean()

    def test_the_assignment_is_deterministic_in_the_seed(self):
        mask = bouts_mask([300, 120, 450, 90, 200, 610, 75, 260])

        first = assign_blocked_folds(mask, n_folds=5, seed=3)
        again = assign_blocked_folds(mask, n_folds=5, seed=3)
        other = assign_blocked_folds(mask, n_folds=5, seed=4)

        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_folds_are_in_the_row_order_of_the_masked_design(self):
        """Fold ``i`` labels the ``i``-th fit bin, not the ``i``-th time bin."""
        mask = bouts_mask([10, 10, 10, 10])
        design_rows = np.flatnonzero(np.asarray(mask))

        folds = assign_blocked_folds(mask, n_folds=2, seed=1)

        assert folds.shape == design_rows.shape

    def test_too_few_runs_for_the_folds_is_refused(self):
        mask = bouts_mask([100, 100])

        with pytest.raises(ValueError, match="fewer than the"):
            assign_blocked_folds(mask, n_folds=5, seed=0)

    def test_a_single_fold_is_refused(self):
        mask = bouts_mask([100, 100, 100])

        with pytest.raises(ValueError, match="at least 2"):
            assign_blocked_folds(mask, n_folds=1, seed=0)

    def test_runs_with_overlapping_kernel_support_stay_in_one_fold(self):
        values = np.concatenate(
            [
                np.ones(100, dtype=bool),
                np.zeros(10, dtype=bool),
                np.ones(100, dtype=bool),
                np.zeros(100, dtype=bool),
                np.ones(100, dtype=bool),
                np.zeros(10, dtype=bool),
                np.ones(100, dtype=bool),
            ]
        )
        mask = pd.Series(values, index=np.arange(len(values)) / SAMPLING_FREQUENCY)

        folds = assign_blocked_folds(mask, n_folds=2, seed=3, support_radius_bins=10)

        assert np.unique(folds[:200]).size == 1
        assert np.unique(folds[200:]).size == 1
        assert folds[0] != folds[-1]

    def test_too_few_support_separated_blocks_is_refused(self):
        mask = bouts_mask([100, 100, 100], gap=10)

        with pytest.raises(ValueError, match="support-separated blocks"):
            assign_blocked_folds(mask, n_folds=2, seed=0, support_radius_bins=10)


class TestPoissonDeviance:
    def test_it_is_zero_at_a_perfect_fit(self):
        counts = np.array([[0.0], [1.0], [3.0]])

        assert poisson_deviance(counts, counts + 1e-300)[0] == pytest.approx(0.0)

    def test_a_silent_bin_contributes_twice_the_fitted_mean(self):
        counts = np.zeros((4, 1))
        mean = np.full((4, 1), 0.25)

        assert poisson_deviance(counts, mean)[0] == pytest.approx(2.0)

    def test_it_matches_statsmodels_deviance_on_a_fitted_model(self):
        import statsmodels.api as sm

        rng = np.random.default_rng(2)
        design = np.column_stack([np.ones(500), rng.normal(size=500)])
        counts = rng.poisson(np.exp(design @ np.array([-1.0, 0.4]))).astype(float)
        fit = sm.GLM(counts, design, family=sm.families.Poisson()).fit()

        actual = poisson_deviance(counts[:, None], fit.fittedvalues[:, None])

        assert actual[0] == pytest.approx(fit.deviance)


@pytest.fixture
def event_session():
    """A session of immobility bouts with events, and units driven by them.

    Unit 0 responds to ``non_local`` events, unit 1 is a flat Poisson train.
    """
    rng = np.random.default_rng(20260915)
    mask = bouts_mask([2000] * 12, gap=200)
    time = np.asarray(mask.index, dtype=float)
    fit_rows = np.asarray(mask, dtype=bool)

    onsets = time[np.flatnonzero(fit_rows)[::400]][1:]
    events = pd.DataFrame({"start_time": onsets, "end_time": onsets + 0.08})

    lags = np.arange(-50, 51)
    spline_basis, lag_time, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    convolved = convolve_event_basis(
        event_onset_counts(events, time), spline_basis, lags
    )
    # A bump peaking just after onset, worth ~e^1.2 at its peak.
    kernel = 1.2 * np.exp(-0.5 * ((lag_time - 0.02) / 0.03) ** 2)
    coefficients = np.linalg.lstsq(spline_basis, kernel, rcond=None)[0]

    baseline = np.log(10.0 / SAMPLING_FREQUENCY)
    driven = rng.poisson(np.exp(baseline + convolved @ coefficients))
    flat = rng.poisson(np.exp(baseline), size=len(time))
    counts = np.column_stack([driven, flat]).astype(float)

    constant = np.ones((len(time), 1))
    return {
        "fit_mask": mask,
        "time": time,
        "events": events,
        "counts": counts,
        "lags": lags,
        "spline_basis": spline_basis,
        "designs": {
            "constant": constant[fit_rows],
            "events": np.concatenate([constant, convolved], axis=1)[fit_rows],
        },
        "fit_counts": counts[fit_rows],
    }


class TestHeldoutDeviance:
    def test_the_generating_model_predicts_held_out_spikes_better(self, event_session):
        folds = assign_blocked_folds(event_session["fit_mask"], n_folds=5, seed=5)
        penalty = make_ridge_penalty(
            event_session["designs"]["events"].shape[1], sigma=2.0
        )

        constant = heldout_deviance(
            event_session["designs"]["constant"],
            event_session["fit_counts"],
            None,
            folds,
        )
        events = heldout_deviance(
            event_session["designs"]["events"],
            event_session["fit_counts"],
            penalty,
            folds,
        )

        # Unit 0 was generated with the event kernel; unit 1 was not.
        assert events.total[0] < constant.total[0]
        assert events.total[1] > constant.total[1] - 1.0

    def test_a_regressor_the_data_does_not_use_does_not_help_on_average(
        self, event_session
    ):
        """In-sample deviance always falls; out of fold it need not, and here
        it does not."""
        folds = assign_blocked_folds(event_session["fit_mask"], n_folds=5, seed=5)
        design = event_session["designs"]["events"]
        penalty = make_ridge_penalty(design.shape[1], sigma=2.0)

        heldout = heldout_deviance(design, event_session["fit_counts"], penalty, folds)

        assert heldout.total[1] >= (
            heldout_deviance(
                event_session["designs"]["constant"],
                event_session["fit_counts"],
                None,
                folds,
            ).total[1]
            - 1.0
        )

    def test_the_per_fold_deviances_sum_to_the_total(self, event_session):
        folds = assign_blocked_folds(event_session["fit_mask"], n_folds=5, seed=5)

        heldout = heldout_deviance(
            event_session["designs"]["constant"],
            event_session["fit_counts"],
            None,
            folds,
        )

        assert heldout.per_fold.shape == (5, 2)
        np.testing.assert_allclose(heldout.per_fold.sum(axis=0), heldout.total)

    def test_each_fold_is_scored_on_a_fit_that_never_saw_it(self, event_session):
        """The held-out fold's deviance is the one a training fit gives it.

        Read directly: refit on the other folds here, predict the held-out
        fold, and compare -- if the scored fit had seen the fold, or the folds
        were rotated by one, the numbers would not line up.
        """
        from continuum_swr_replay.analysis.pfc.batched_glm import fit_poisson_glms

        folds = assign_blocked_folds(event_session["fit_mask"], n_folds=5, seed=5)
        design = event_session["designs"]["events"]
        counts = event_session["fit_counts"]
        penalty = make_ridge_penalty(design.shape[1], sigma=2.0)

        heldout = heldout_deviance(design, counts, penalty, folds)

        for fold in range(5):
            held = folds == fold
            fitted = fit_poisson_glms(design[~held], counts[~held], penalty=penalty)
            expected = poisson_deviance(
                counts[held], np.exp(design[held] @ fitted.params.T)
            )
            np.testing.assert_allclose(heldout.per_fold[fold], expected, rtol=1e-12)

    def test_the_penalty_is_part_of_the_estimator_being_scored(self, event_session):
        """Scoring a penalised and an unpenalised estimator gives different
        numbers, so the penalty cannot be quietly dropped from the training
        fit."""
        folds = assign_blocked_folds(event_session["fit_mask"], n_folds=5, seed=5)
        design = event_session["designs"]["events"]
        counts = event_session["fit_counts"]

        penalised = heldout_deviance(
            design, counts, make_ridge_penalty(design.shape[1], sigma=0.1), folds
        )
        unpenalised = heldout_deviance(design, counts, None, folds)

        assert not np.allclose(penalised.total, unpenalised.total)

    def test_a_unit_with_no_spikes_scores_nan_and_leaves_the_others_alone(
        self, event_session
    ):
        folds = assign_blocked_folds(event_session["fit_mask"], n_folds=5, seed=5)
        counts = event_session["fit_counts"].copy()
        silent = np.zeros((counts.shape[0], 1))
        counts = np.column_stack([silent, counts])

        heldout = heldout_deviance(
            event_session["designs"]["constant"], counts, None, folds
        )

        assert np.isnan(heldout.total[0])
        assert np.all(np.isfinite(heldout.total[1:]))

    def test_mismatched_row_counts_are_refused(self, event_session):
        folds = assign_blocked_folds(event_session["fit_mask"], n_folds=5, seed=5)

        with pytest.raises(ValueError, match="they must agree"):
            heldout_deviance(
                event_session["designs"]["constant"][:-1],
                event_session["fit_counts"],
                None,
                folds,
            )

    def test_a_nonconverged_training_fit_invalidates_that_units_score(
        self, monkeypatch
    ):
        def finite_but_nonconverged(design, counts, **kwargs):
            n_units = np.atleast_2d(counts).shape[1]
            return BatchedGLMResults(
                params=np.zeros((n_units, design.shape[1])),
                llf=np.zeros(n_units),
                deviance=np.zeros(n_units),
                aic=np.zeros(n_units),
                converged=np.array([False, True]),
                n_iter=np.full(n_units, 100),
                rank=design.shape[1],
                nobs=len(design),
                n_features=design.shape[1],
            )

        monkeypatch.setattr(
            "continuum_swr_replay.analysis.pfc.batched_glm.fit_poisson_glms",
            finite_but_nonconverged,
        )
        design = np.ones((20, 1))
        counts = np.ones((20, 2))
        folds = np.repeat([0, 1], 10)

        scored = heldout_deviance(design, counts, None, folds)

        assert np.isnan(scored.total[0])
        assert np.isfinite(scored.total[1])

    def test_a_nonfinite_heldout_prediction_invalidates_the_score(self, monkeypatch):
        def overflowing_fit(design, counts, **kwargs):
            n_units = np.atleast_2d(counts).shape[1]
            return BatchedGLMResults(
                params=np.full((n_units, design.shape[1]), 1_000.0),
                llf=np.zeros(n_units),
                deviance=np.zeros(n_units),
                aic=np.zeros(n_units),
                converged=np.ones(n_units, dtype=bool),
                n_iter=np.ones(n_units, dtype=int),
                rank=design.shape[1],
                nobs=len(design),
                n_features=design.shape[1],
            )

        monkeypatch.setattr(
            "continuum_swr_replay.analysis.pfc.batched_glm.fit_poisson_glms",
            overflowing_fit,
        )
        design = np.ones((20, 1))
        counts = np.ones((20, 1))
        folds = np.repeat([0, 1], 10)

        scored = heldout_deviance(design, counts, None, folds)

        assert np.isnan(scored.total[0])


class TestEmpiricalPvalue:
    def test_it_uses_the_finite_sample_plus_one_correction(self):
        null = np.array([[1.0], [2.0], [3.0], [4.0]])

        assert empirical_pvalue(np.array([2.5]), null, min_usable_draws=1)[
            0
        ] == pytest.approx(3 / 5)
        assert empirical_pvalue(np.array([0.0]), null, min_usable_draws=1)[
            0
        ] == pytest.approx(1.0)
        assert empirical_pvalue(np.array([5.0]), null, min_usable_draws=1)[
            0
        ] == pytest.approx(1 / 5)

    def test_a_tie_counts_as_the_null_being_as_large(self):
        null = np.array([[1.0], [2.0]])

        assert empirical_pvalue(np.array([2.0]), null, min_usable_draws=1)[
            0
        ] == pytest.approx(2 / 3)

    def test_draws_that_failed_are_left_out_of_the_denominator(self):
        null = np.array([[1.0], [np.nan], [3.0]])

        assert empirical_pvalue(np.array([2.0]), null, min_usable_draws=1)[
            0
        ] == pytest.approx(2 / 3)

    def test_a_unit_with_no_usable_draw_is_nan(self):
        null = np.full((3, 1), np.nan)

        assert np.isnan(empirical_pvalue(np.array([2.0]), null, min_usable_draws=1)[0])

    def test_fewer_than_the_reporting_floor_is_unavailable(self):
        null = np.arange(19.0)[:, None]

        assert np.isnan(empirical_pvalue(np.array([2.0]), null)[0])


@pytest.fixture
def null_session():
    """Immobility bouts with events, and units that ignore them entirely."""
    rng = np.random.default_rng(4242)
    mask = bouts_mask([1500] * 14, gap=150)
    time = np.asarray(mask.index, dtype=float)
    fit_rows = np.asarray(mask, dtype=bool)

    onsets = time[np.flatnonzero(fit_rows)[::300]][1:]
    events = pd.DataFrame({"start_time": onsets, "end_time": onsets + 0.08})
    event_times = {
        "non_local": events,
        "non_local_SWR": events.iloc[::2].reset_index(drop=True),
        "non_local_no_SWR": events.iloc[1::2].reset_index(drop=True),
        "SWR": events.iloc[::2].reset_index(drop=True),
    }
    # Every unit is a flat Poisson train: nothing in the data is locked to an
    # event, so an event-shaped regressor is exchangeable with a control.
    counts = rng.poisson(
        np.exp(np.log(12.0 / SAMPLING_FREQUENCY)), size=(len(time), 24)
    ).astype(float)
    lags = np.arange(-50, 51)
    spline_basis, _, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    return {
        "fit_mask": mask,
        "event_times": event_times,
        "counts": counts,
        "lags": lags,
        "spline_basis": spline_basis,
    }


class TestNullDrawSeeds:
    def test_the_seeds_are_distinct_and_none_is_the_session_seed(self):
        seeds = null_draw_seeds(control_seed=123456789, n_draws=50)

        assert len(seeds) == 50
        assert len(set(seeds.tolist())) == 50
        assert 123456789 not in seeds.tolist()

    def test_they_are_derived_from_the_session_seed(self):
        """Reproducible for a session refit on its own, and different between
        sessions -- two sessions must not share a null draw."""
        first = null_draw_seeds(11, 8)
        again = null_draw_seeds(11, 8)
        other = null_draw_seeds(12, 8)

        np.testing.assert_array_equal(first, again)
        assert not set(first.tolist()) & set(other.tolist())

    def test_a_negative_count_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            null_draw_seeds(1, -1)


class TestControlDrawNull:
    def test_each_class_gets_its_own_count_matched_draws(self, null_session):
        """The point of the per-class null: a class with half the onsets is
        stood in for by half as many windows."""
        folds = assign_blocked_folds(null_session["fit_mask"], n_folds=5, seed=2)
        seeds = [101, 102, 103]

        null = control_draw_null(
            null_session["event_times"],
            null_session["fit_mask"],
            null_session["counts"],
            folds,
            rng_seeds=seeds,
            spline_basis=null_session["spline_basis"],
            lags=null_session["lags"],
            penalty_sigma=2.0,
        )

        n_units = null_session["counts"].shape[1]
        for name, events in null_session["event_times"].items():
            assert null.improvement[name].shape == (len(seeds), n_units)
            assert null.n_events[name] == len(events)
            assert 0 < null.n_tested[name] <= len(events)
            # Every window placed, and never more than the class has events.
            assert np.all(null.n_placed[name] <= null.n_tested[name])
            assert np.all(null.n_placed[name] > 0)
        # The halved class really is drawn half as often.
        assert null.n_events["non_local_no_SWR"] == null.n_events["non_local"] // 2
        assert (
            null.n_placed["non_local_no_SWR"].max() < null.n_placed["non_local"].min()
        )
        assert null.engagement.shape[0] == 0

    def test_observed_and_null_use_one_fixed_matchable_event_subset(self, null_session):
        """An unmatchable real event is excluded from both sides, not only null."""
        event_times = dict(null_session["event_times"])
        events = event_times["non_local"]
        # 3.1 s falls in the first moving gap, so this source event cannot be
        # matched within a fitted bout. It remains a real event in the
        # all-event descriptive model but cannot enter the randomisation test.
        outside = pd.DataFrame({"start_time": [3.1], "end_time": [3.18]})
        event_times["non_local"] = pd.concat(
            [events, outside], ignore_index=True
        ).sort_values("start_time", ignore_index=True)
        folds = assign_blocked_folds(null_session["fit_mask"], n_folds=5, seed=2)

        null = control_draw_null(
            event_times,
            null_session["fit_mask"],
            null_session["counts"][:, :2],
            folds,
            rng_seeds=[101, 102],
            subset_seed=77,
            spline_basis=null_session["spline_basis"],
            lags=null_session["lags"],
            penalty_sigma=2.0,
            classes=["non_local"],
        )

        assert null.n_events["non_local"] == len(events) + 1
        assert null.n_tested["non_local"] == len(events)
        assert 3.1 not in (
            event_times["non_local"]
            .iloc[null.tested_source_indices["non_local"]]
            .start_time.to_numpy()
        )
        assert np.all(null.n_placed["non_local"] == len(events))
        assert np.isfinite(null.improvement["non_local"]).all()

        tested = event_times["non_local"].iloc[null.tested_source_indices["non_local"]]
        fit_rows = np.asarray(null_session["fit_mask"], dtype=bool)
        fit_counts = null_session["counts"][fit_rows, :2]
        constant = np.ones((fit_rows.sum(), 1))
        design = control_design(
            tested,
            null_session["fit_mask"],
            null_session["spline_basis"],
            null_session["lags"],
        )
        expected = heldout_deviance(constant, fit_counts, None, folds).total
        expected -= heldout_deviance(
            design,
            fit_counts,
            make_ridge_penalty(design.shape[1], 2.0),
            folds,
        ).total
        np.testing.assert_allclose(null.observed_improvement["non_local"], expected)

    def test_a_classs_draws_have_that_classs_durations(self, null_session):
        """Duration matching, read off the drawn windows themselves."""
        from continuum_swr_replay.analysis.pfc.glm import (
            MATCHED_CONTROL_SOURCE_CLASS,
            make_matched_control_times,
        )

        events = null_session["event_times"]["non_local_no_SWR"]
        draw = make_matched_control_times(
            {**null_session["event_times"], MATCHED_CONTROL_SOURCE_CLASS: events},
            null_session["fit_mask"],
            np.random.default_rng(5),
        )
        drawn = draw.times

        np.testing.assert_allclose(
            (drawn.end_time - drawn.start_time).to_numpy(),
            (events.end_time - events.start_time).iloc[draw.source_indices].to_numpy(),
            atol=1e-12,
        )

    def test_two_draws_differ_and_a_repeated_seed_does_not(self, null_session):
        folds = assign_blocked_folds(null_session["fit_mask"], n_folds=5, seed=2)
        arguments = {
            "spline_basis": null_session["spline_basis"],
            "lags": null_session["lags"],
            "penalty_sigma": 2.0,
            "classes": ["non_local"],
        }

        first = control_draw_null(
            null_session["event_times"],
            null_session["fit_mask"],
            null_session["counts"][:, :2],
            folds,
            rng_seeds=[101, 202],
            **arguments,
        )
        again = control_draw_null(
            null_session["event_times"],
            null_session["fit_mask"],
            null_session["counts"][:, :2],
            folds,
            rng_seeds=[101],
            **arguments,
        )

        np.testing.assert_allclose(
            first.improvement["non_local"][0], again.improvement["non_local"][0]
        )
        assert not np.allclose(
            first.improvement["non_local"][0], first.improvement["non_local"][1]
        )

    def test_a_repeated_seed_is_refused(self, null_session):
        folds = assign_blocked_folds(null_session["fit_mask"], n_folds=5, seed=2)

        with pytest.raises(ValueError, match="distinct"):
            control_draw_null(
                null_session["event_times"],
                null_session["fit_mask"],
                null_session["counts"][:, :1],
                folds,
                rng_seeds=[7, 7],
                spline_basis=null_session["spline_basis"],
                lags=null_session["lags"],
                classes=["non_local"],
            )

    def test_an_incompletely_placed_draw_is_not_a_null_replicate(self, null_session):
        folds = assign_blocked_folds(null_session["fit_mask"], n_folds=5, seed=2)

        null = control_draw_null(
            null_session["event_times"],
            null_session["fit_mask"],
            null_session["counts"][:, :2],
            folds,
            rng_seeds=[101],
            spline_basis=null_session["spline_basis"],
            lags=null_session["lags"],
            classes=["non_local"],
            max_attempts=0,
        )

        assert null.n_placed["non_local"].tolist() == [0]
        assert np.isnan(null.improvement["non_local"]).all()

    def test_the_engagement_null_replaces_the_tested_regressor(self, null_session):
        folds = assign_blocked_folds(null_session["fit_mask"], n_folds=5, seed=2)
        baseline = null_session["event_times"]["non_local_no_SWR"]

        null = control_draw_null(
            null_session["event_times"],
            null_session["fit_mask"],
            null_session["counts"][:, :3],
            folds,
            rng_seeds=[11, 12],
            spline_basis=null_session["spline_basis"],
            lags=null_session["lags"],
            penalty_sigma=2.0,
            classes=["non_local_no_SWR"],
            baseline_control=baseline,
        )

        assert null.engagement.shape == (2, 3)
        assert np.all(np.isfinite(null.engagement))

    @pytest.mark.slow
    def test_the_p_values_are_uniform_for_classes_of_different_sizes(
        self, null_session
    ):
        """The property that makes the null usable, on units built to have none.

        Run for two classes whose onset counts differ by a factor of two: a
        null that was not count-matched would be uniform for one of them at
        best. Each unit's "observed" statistic is one draw's improvement and
        the null is the other draws'; under exchangeability the rank of the
        observed among ``R`` draws is uniform, so the p-values must be too. The
        threshold is well clear of what ``R = 40`` draws and 24 units can
        resolve (the statistic cannot go below ``1 / (2 R)`` by discreteness
        alone).
        """
        folds = assign_blocked_folds(null_session["fit_mask"], n_folds=5, seed=2)
        seeds = list(range(9000, 9041))  # one observed draw + R = 40 null draws

        null = control_draw_null(
            null_session["event_times"],
            null_session["fit_mask"],
            null_session["counts"],
            folds,
            rng_seeds=seeds,
            spline_basis=null_session["spline_basis"],
            lags=null_session["lags"],
            penalty_sigma=2.0,
            classes=["non_local", "non_local_no_SWR"],
        )

        assert null.n_events["non_local"] > 1.5 * null.n_events["non_local_no_SWR"]
        for class_name in ("non_local", "non_local_no_SWR"):
            draws = null.improvement[class_name]
            pvalues = empirical_pvalue(draws[0], draws[1:])

            assert np.all(np.isfinite(pvalues)), class_name
            ks = stats.kstest(pvalues, "uniform").statistic
            assert ks < 0.30, f"{class_name}: p-values not uniform, KS = {ks:.3f}"
            assert 0.2 < pvalues.mean() < 0.8, class_name
