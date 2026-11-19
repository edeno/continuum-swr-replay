"""Held-out scoring of one session, on a small synthetic session."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc import scoring
from continuum_swr_replay.analysis.pfc.batched_glm import DEFAULT_PENALTY_SIGMA
from continuum_swr_replay.analysis.pfc.event_sets import EVENT_CLASSES
from continuum_swr_replay.analysis.pfc.heldout import (
    BEST_MODEL_CANDIDATES,
    ENGAGEMENT_PAIR,
    MIN_USABLE_NULL_DRAWS,
)
from continuum_swr_replay.analysis.pfc.modulation import (
    CURVE_SOURCES,
    NESTED_MODEL_PAIRS,
)

SAMPLING_FREQUENCY = 500.0


def _synthetic_context(seed=0, n_units=3):
    """A small session with the nine designs, three bouts and real events."""
    from continuum_swr_replay.analysis.pfc import (
        make_design_matrices,
        make_matched_control_times,
    )

    rng = np.random.default_rng(seed)
    n_time = 30_000
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    mask = np.zeros(n_time, dtype=bool)
    for start, stop in ((500, 10_000), (11_000, 20_000), (21_000, 29_500)):
        mask[start:stop] = True
    fit_mask = pd.Series(mask, index=time)

    def windows(onsets):
        return pd.DataFrame({"start_time": onsets, "end_time": onsets + 0.08})

    swr_onsets = time[np.arange(600, 29_000, 900)]
    non_local_onsets = time[np.arange(700, 29_000, 700)]
    is_swr = np.zeros(len(non_local_onsets), dtype=bool)
    is_swr[::2] = True
    event_times = {
        "SWR": windows(swr_onsets),
        "non_local": windows(non_local_onsets),
        "non_local_SWR": windows(non_local_onsets[is_swr]),
        "non_local_no_SWR": windows(non_local_onsets[~is_swr]),
    }
    event_times["matched_control"] = make_matched_control_times(
        event_times, fit_mask, np.random.default_rng(7)
    ).times

    position_info = pd.DataFrame({"head_speed": np.zeros(n_time)}, index=time)
    lags = np.arange(-25, 26)
    designs, spline_basis, lag_time = make_design_matrices(
        event_times,
        position_info,
        sampling_frequency=SAMPLING_FREQUENCY,
        lags=lags,
    )
    counts = rng.poisson(0.01, size=(n_time, n_units)).astype(float)
    return {
        "fit_mask": fit_mask,
        "counts": counts,
        "designs": designs,
        "spline_basis": spline_basis,
        "lags": lags,
        "lag_time": lag_time,
        "event_times": event_times,
        "artifact_dropped": pd.DataFrame(
            {"start_time": np.empty(0), "end_time": np.empty(0)}
        ),
        "n_basis_per_class": int(spline_basis.shape[1]),
    }


def _synthetic_curves(context, n_report_lags=201):
    """Stand-in shipped curves, one per bundle array, in the units' order."""
    n_units = context["counts"].shape[1]
    curve = np.zeros((n_report_lags, n_units))
    curve[n_report_lags // 2] = 0.5
    return dict.fromkeys(CURVE_SOURCES, curve)


def test_a_basis_column_with_no_spikes_under_it_is_counted():
    design = np.column_stack(
        [np.ones(6), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]
    )
    counts = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [3.0, 3.0],
        ]
    )

    separated = scoring.separated_column_counts(design, counts)

    # Unit 0 has a spike under the first basis column and none under the
    # second; unit 1's only spikes fall where neither column has support.
    np.testing.assert_array_equal(separated, [1, 2])


def test_every_fold_holds_some_of_the_rarest_class():
    from continuum_swr_replay.analysis.pfc.heldout import assign_blocked_folds

    context = _synthetic_context()
    folds = assign_blocked_folds(context["fit_mask"], 3, seed=5)

    per_class = scoring.events_per_fold(context, folds)

    assert set(per_class) == set(context["event_times"])
    for name, counts in per_class.items():
        assert len(counts) == 3
        assert sum(counts) <= len(context["event_times"][name])
    assert all(count > 0 for count in per_class["non_local_no_SWR"])


def test_the_scored_table_says_what_each_model_bought_out_of_fold(caplog):
    context = _synthetic_context()

    statistics, diagnostics = scoring.score_session(
        context,
        modulation_curves=_synthetic_curves(context),
        penalty_sigma=DEFAULT_PENALTY_SIGMA,
        n_folds=3,
        fold_seed=5,
        n_control_draws=4,
        control_draw_seed=6,
        logger=logging.getLogger("test"),
    )

    assert len(statistics) == context["counts"].shape[1]
    for model in context["designs"]:
        assert np.isfinite(statistics[f"{model}: heldout_deviance"]).all()
        assert (statistics[f"{model}: separated_columns"] >= 0).all()

    # A nested pair's per-fold differences sum to the total it reports.
    for full, reduced in NESTED_MODEL_PAIRS:
        stem = scoring.pair_name(full, reduced)
        per_fold = np.stack(statistics[f"{stem}: heldout_delta_dev_folds"].to_numpy())
        assert per_fold.shape == (len(statistics), 3)
        np.testing.assert_allclose(
            per_fold.sum(axis=1),
            statistics[f"{stem}: heldout_delta_dev"].to_numpy(),
            rtol=1e-8,
            atol=1e-9,
        )

    # This fixture has two event onsets outside the fit mask. They are excluded
    # from the matched test, but cannot contribute to the fit-mask design, so
    # its statistic still equals the separately reported all-event difference.
    null_pairs = [
        (event_class, scoring.pair_name(event_class, "constant"))
        for event_class in EVENT_CLASSES
    ]
    null_pairs.append(("engagement", scoring.pair_name(*ENGAGEMENT_PAIR)))
    for _, stem in null_pairs:
        np.testing.assert_allclose(
            statistics[f"{stem}: matched_null_observed_delta_dev"],
            statistics[f"{stem}: heldout_delta_dev"],
        )

    # Four draws exercise the null integration but are deliberately below the
    # reporting floor, so the columns exist and say "unavailable" rather than
    # turning a coarse rank into evidence.
    for column in (
        "SWR: p_heldout",
        "non_local: p_heldout",
        "non_local_no_SWR: p_heldout",
        "non_local_SWR: p_heldout",
        f"{scoring.pair_name(*ENGAGEMENT_PAIR)}: p_heldout",
    ):
        values = statistics[column].to_numpy(dtype=float)
        assert np.isnan(values).all()

    usable_columns = {
        "SWR": "SWR: n_usable_null_draws",
        "non_local": "non_local: n_usable_null_draws",
        "non_local_no_SWR": "non_local_no_SWR: n_usable_null_draws",
        "non_local_SWR": "non_local_SWR: n_usable_null_draws",
        "engagement": (f"{scoring.pair_name(*ENGAGEMENT_PAIR)}: n_usable_null_draws"),
    }
    for name, column in usable_columns.items():
        assert column in statistics
        np.testing.assert_array_equal(
            statistics[column].to_numpy(),
            diagnostics["null_usable_draws"][name],
        )

    assert set(statistics["min_heldout_model"]) <= set(
        BEST_MODEL_CANDIDATES["min_heldout_model"]
    )
    assert sum(diagnostics["fold_sizes"]) == int(np.asarray(context["fit_mask"]).sum())
    assert diagnostics["non_estimable_models"] == []
    assert diagnostics["fold_support_radius_bins"] == 25
    assert all(
        record["n_required_draws"] == MIN_USABLE_NULL_DRAWS
        for record in diagnostics["null_windows"].values()
    )
    assert all(
        record["n_events_total"] - record["n_events_tested"]
        == record["n_events_excluded_unmatchable"]
        for record in diagnostics["null_windows"].values()
    )
    assert {
        name: record["n_events_excluded_unmatchable"]
        for name, record in diagnostics["null_windows"].items()
    } == {
        "non_local_no_SWR": 0,
        "non_local_SWR": 2,
        "non_local": 2,
        "SWR": 2,
    }


def test_the_null_avoids_the_events_the_session_dropped(monkeypatch):
    """The null's controls avoid what ``run_session``'s shipped control avoids:
    the events the analyzed-time rule dropped, handed back in the context."""
    context = _synthetic_context()
    context["artifact_dropped"] = pd.DataFrame(
        {"start_time": [3.0, 30.0], "end_time": [3.4, 30.2]}
    )
    seen = {}
    real = scoring.control_draw_null

    def spy(*args, **kwargs):
        seen["avoid"] = kwargs.get("avoid")
        return real(*args, **kwargs)

    monkeypatch.setattr(scoring, "control_draw_null", spy)
    scoring.score_session(
        context,
        modulation_curves=_synthetic_curves(context),
        penalty_sigma=DEFAULT_PENALTY_SIGMA,
        n_folds=3,
        fold_seed=5,
        n_control_draws=2,
        control_draw_seed=6,
        logger=logging.getLogger("test"),
    )

    assert seen["avoid"] is context["artifact_dropped"]


def test_the_best_model_is_the_one_with_the_lowest_heldout_deviance():
    context = _synthetic_context()

    statistics, _ = scoring.score_session(
        context,
        modulation_curves=_synthetic_curves(context),
        penalty_sigma=DEFAULT_PENALTY_SIGMA,
        n_folds=3,
        fold_seed=5,
        n_control_draws=0,
        control_draw_seed=6,
        logger=logging.getLogger("test"),
    )

    assert "SWR: p_heldout" not in statistics.columns
    for column, candidates in BEST_MODEL_CANDIDATES.items():
        deviances = np.column_stack(
            [statistics[f"{name}: heldout_deviance"] for name in candidates]
        )
        expected = [candidates[index] for index in deviances.argmin(axis=1)]
        assert list(statistics[column]) == expected


def test_every_curve_is_read_from_its_own_classs_block():
    """`CURVE_SOURCES` must name the design block the curve is named after.

    The replay-split design is concatenated `[constant, non_local_no_SWR,
    non_local_SWR]`, the reverse of the order the model's name lists the
    classes; reading it by the name swapped both of that model's curves in
    every bundle written before 2026-09-15. This compares the block against
    the single-class design of the class the curve is named for, so the
    mapping cannot drift again.
    """
    context = _synthetic_context()
    designs = context["designs"]
    n_basis = context["n_basis_per_class"]
    single_class = {
        "SWR": "SWR",
        "no_SWR": "non_local_no_SWR",
        "non_local_SWR": "non_local_SWR",
        "non_local_no_SWR": "non_local_no_SWR",
        "matched_control": "matched_control",
    }

    for curve, (model, block) in CURVE_SOURCES.items():
        start = 1 + block * n_basis
        np.testing.assert_array_equal(
            designs[model][:, start : start + n_basis],
            designs[single_class[curve]][:, 1:],
            err_msg=f"{curve} does not read {single_class[curve]}'s block",
        )


def test_a_nonconverged_mle_is_not_used_for_the_prior_audit(monkeypatch):
    from continuum_swr_replay.analysis.pfc.batched_glm import BatchedGLMResults

    n_time, n_units, n_basis = 20, 2, 2
    wanted = {model for model, _ in CURVE_SOURCES.values()}
    designs = {
        model: np.ones((n_time, 1 + n_basis * (1 + int(" + " in model))), dtype=float)
        for model in wanted
    }

    def finite_but_nonconverged(design, counts, **kwargs):
        n_features = design.shape[1]
        return BatchedGLMResults(
            params=np.zeros((n_units, n_features)),
            llf=np.zeros(n_units),
            deviance=np.zeros(n_units),
            aic=np.zeros(n_units),
            converged=np.array([False, True]),
            n_iter=np.array([100, 2]),
            rank=n_features,
            nobs=len(design),
            n_features=n_features,
            covariance=np.tile(np.eye(n_features), (n_units, 1, 1)),
        )

    monkeypatch.setattr(scoring, "fit_poisson_glms", finite_but_nonconverged)
    columns = scoring.maximum_likelihood_comparison(
        designs,
        np.ones(n_time, dtype=bool),
        np.ones((n_time, n_units)),
        {
            "spline_basis": np.eye(n_basis),
            "lag_time": np.array([-0.1, 0.1]),
        },
        {curve: np.zeros((n_basis, n_units)) for curve in CURVE_SOURCES},
        logging.getLogger("test"),
    )

    for curve in CURVE_SOURCES:
        assert columns[f"{curve}: mle_converged"].tolist() == [False, True]
        assert np.isnan(columns[f"{curve}: peak_log_modulation_mle"][0])
        assert np.isfinite(columns[f"{curve}: peak_log_modulation_mle"][1])
        assert not columns[f"{curve}: prior_binds"][0]


def _three_class_context(design: np.ndarray, n_units: int = 2) -> dict:
    """A context holding only the three-class design: two basis functions read
    on three lags, of which -0.3 s is outside the reported window. The last two
    bins are outside the fit mask."""
    from continuum_swr_replay.analysis.pfc.modulation import THREE_CLASS_MODEL

    fit_mask = np.ones(len(design), dtype=bool)
    fit_mask[-2:] = False
    return {
        "fit_mask": fit_mask,
        "designs": {THREE_CLASS_MODEL: design},
        "counts": np.ones((len(design), n_units)),
        "spline_basis": np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        "lag_time": np.array([-0.3, 0.0, 0.1]),
    }


def test_the_extended_curves_read_their_own_blocks_of_the_three_class_fit(
    monkeypatch,
):
    """Block 2 (SWRs without replay) and block 1 (SWR-coupled replay) of the
    three-class fit, on the reported lags only; a unit whose fit did not
    converge gets a NaN curve, and only the fit-mask rows are fit."""
    from continuum_swr_replay.analysis.pfc.batched_glm import BatchedGLMResults

    design = np.random.default_rng(0).normal(size=(12, 7))
    params = np.array(
        [
            [0.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [0.0, 7.0, 7.0, 2.0, 4.0, 1.0, 3.0],
        ]
    )

    def known_fit(design, counts, **kwargs):
        assert len(design) == len(counts) == 10
        return BatchedGLMResults(
            params=params,
            llf=np.zeros(2),
            deviance=np.zeros(2),
            aic=np.zeros(2),
            converged=np.array([False, True]),
            n_iter=np.array([100, 2]),
            rank=7,
            nobs=len(design),
            n_features=7,
            covariance=None,
        )

    monkeypatch.setattr(scoring, "fit_poisson_glms", known_fit)
    curves = scoring.extended_curves(_three_class_context(design), 3.0)

    # lags 0.0 and 0.1: 0.5 * b0 + 0.5 * b1, then b1
    np.testing.assert_allclose(curves["SWR_no_non_local"][:, 1], [2.0, 3.0])
    np.testing.assert_allclose(curves["non_local_SWR_3class"][:, 1], [3.0, 4.0])
    for curve in curves.values():
        assert curve.shape == (2, 2)
        assert np.isnan(curve[:, 0]).all()


def test_a_rank_deficient_three_class_design_gives_nan_curves():
    """No SWR without replay: its block is all zero, so no curve is fit."""
    design = np.random.default_rng(0).normal(size=(12, 7))
    design[:, 5:] = 0.0

    curves = scoring.extended_curves(_three_class_context(design, n_units=3), 3.0)

    assert set(curves) == {"SWR_no_non_local", "non_local_SWR_3class"}
    for curve in curves.values():
        assert curve.shape == (2, 3)
        assert np.isnan(curve).all()
