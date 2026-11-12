"""Contracts of the three-estimator content comparison's fit.

The comparison exists to say whether the retired count model, the shipped
fraction model and the Dirichlet-multinomial disagree, so what is pinned here is
what makes such a comparison readable: every (session, rung, term, estimator)
appears exactly once, the three estimators are fit on ONE shared event set per
(session, rung), the likelihood constants are equalized before any AIC is
differenced, and the animal bootstrap reports an interval about each median.

The cohort is simulated with a planted Dirichlet concentration (phi = 3), so the
within-event timebins are genuinely correlated and the count model's standard
errors are known to be too small -- which is the quantitative claim the
comparison is supposed to surface. The shared-event rule itself is checked on
tables small enough to read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.content import estimator_comparison as ec
from continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial import (
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial import (
    LOG_PHI_INTERCEPT,
    multinomial_log_constant,
)
from continuum_swr_replay.analysis.behavior.contrast_aggregation import MIN_EVENTS
from continuum_swr_replay.analysis.behavior.fitting import (
    CONDITIONAL_POISSON_COUNT,
    CONDITIONAL_POISSON_FRACTION,
    fit_conditional_poisson,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    N_ARM_CATEGORIES,
    ComparisonRung,
)
from continuum_swr_replay.tables.io import read_table, write_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from tests.support._content_tables import (
    COMPARISON_N_EVENTS_PER_SESSION,
    estimator_comparison_cohort,
)

SC = list(SESSION_COLUMNS)

BASE_RUNG = ComparisonRung(
    name="B_spatial",
    predictors=("is_current_arm", "is_prev_arm"),
    parent=None,
    leaf_only=False,
    key_term="is_current_arm",
)
REWARD_RUNG = ComparisonRung(
    name="B_reward",
    predictors=(
        "is_current_arm",
        "is_prev_arm",
        "reward_fraction_candidate_arm_last10_visits",
    ),
    parent="B_spatial",
    leaf_only=False,
    key_term="reward_fraction_candidate_arm_last10_visits",
)


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    """Two sessions from two animals, both carrying SWR and non-SWR events."""
    return estimator_comparison_cohort()


@pytest.fixture(scope="module")
def one_rung(cohort) -> ec.ComparisonResult:
    return ec.run_comparison(
        cohort, (BASE_RUNG,), n_jobs=2, n_bootstrap=200, seed=ec.BOOTSTRAP_SEED
    )


@pytest.fixture(scope="module")
def two_rungs(cohort) -> ec.ComparisonResult:
    return ec.run_comparison(
        cohort,
        (BASE_RUNG, REWARD_RUNG),
        n_jobs=2,
        n_bootstrap=200,
        seed=ec.BOOTSTRAP_SEED,
    )


# ── the shared event set, by hand ───────────────────────────────────────────


def _events(responses: dict[int, tuple[list[float], list[float]]]) -> pd.DataFrame:
    """Three candidate arms per event: ``{event_id: (fractions, counts)}``."""
    rows = []
    for event_id, (fractions, counts) in responses.items():
        for arm, (fraction, count) in enumerate(zip(fractions, counts, strict=True)):
            rows.append(
                {
                    "event_id": event_id,
                    "candidate_arm": arm,
                    "is_current_arm": float(arm == 0),
                    "arm_fraction": fraction,
                    "arm_count_hard": count,
                }
            )
    return pd.DataFrame(rows)


def test_the_shared_event_set_is_the_intersection_of_the_two_responses():
    usable = {e: ([0.5, 0.5, 0.0], [2.0, 2.0, 0.0]) for e in range(MIN_EVENTS)}
    # Event 100 has fraction mass but no counts, event 101 counts but no
    # fraction: each survives one response's rules and neither is shared.
    frame = _events(
        {
            **usable,
            100: ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            101: ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        }
    )
    rows, info = ec.shared_session_rows(
        frame,
        ["is_current_arm"],
        fraction_col="arm_fraction",
        count_col="arm_count_hard",
    )
    assert info["n_events_in"] == MIN_EVENTS + 2
    assert info["n_events_fraction"] == MIN_EVENTS + 1
    assert info["n_events_count"] == MIN_EVENTS + 1
    assert info["n_events_shared"] == MIN_EVENTS
    assert info["reason"] == ""
    assert set(rows["event_id"]) == set(usable)
    # Every candidate row of a shared event is kept.
    assert len(rows) == 3 * MIN_EVENTS


def test_a_session_with_too_few_shared_events_is_skipped():
    # Each response keeps MIN_EVENTS events, enough on its own, but only
    # MIN_EVENTS - 1 of them are the same events.
    frame = _events(
        {
            **{e: ([0.5, 0.5, 0.0], [2.0, 2.0, 0.0]) for e in range(MIN_EVENTS - 1)},
            100: ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            101: ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        }
    )
    rows, info = ec.shared_session_rows(
        frame,
        ["is_current_arm"],
        fraction_col="arm_fraction",
        count_col="arm_count_hard",
    )
    assert rows is None
    assert info["n_events_fraction"] == MIN_EVENTS
    assert info["n_events_count"] == MIN_EVENTS
    assert info["n_events_shared"] == MIN_EVENTS - 1
    assert info["reason"] == "too_few_shared"


def test_a_response_with_too_few_events_names_its_own_reason():
    frame = _events(
        {e: ([0.5, 0.5, 0.0], [2.0, 2.0, 0.0]) for e in range(MIN_EVENTS - 1)}
    )
    rows, info = ec.shared_session_rows(
        frame,
        ["is_current_arm"],
        fraction_col="arm_fraction",
        count_col="arm_count_hard",
    )
    assert rows is None
    assert info["n_events_shared"] == 0
    assert info["reason"] not in {"", "too_few_shared"}


def test_an_event_with_a_missing_concentration_covariate_leaves_whole():
    frame = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 2, 3, 3],
            "is_swr": [True, True, np.nan, np.nan, False, False],
        }
    )
    kept, n_dropped = ec.drop_undefined_concentration(frame, ["is_swr"])
    assert n_dropped == 1
    assert list(kept["event_id"]) == [1, 1, 3, 3]
    # No concentration design, nothing to drop.
    unchanged, n_none = ec.drop_undefined_concentration(frame, [])
    assert n_none == 0
    assert unchanged is frame


def test_a_fit_whose_parent_saw_other_events_is_counted_once_per_estimator():
    rows = pd.DataFrame(
        {
            "animal": ["a"] * 4,
            "session_date": ["d"] * 4,
            "epoch": ["e"] * 4,
            "model": ["B"] * 4,
            "estimator": ["count", "count", "dm", "fraction"],
            "term": ["x", "y", "x", "x"],
            "delta_aic_row_mismatch": [True, True, False, True],
        }
    )
    assert ec.n_fits_delta_aic_row_mismatch(rows) == 2
    assert ec.n_fits_delta_aic_row_mismatch(rows.iloc[:0]) == 0


# ── the comparison on the simulated cohort ──────────────────────────────────


def test_the_simulated_cohort_carries_both_swr_values(cohort):
    per_session = cohort.groupby(SC, sort=False)["is_swr"].nunique()
    assert (per_session == 2).all()
    assert (
        cohort.groupby(SC, sort=False)["event_id"]
        .nunique()
        .eq(COMPARISON_N_EVENTS_PER_SESSION)
        .all()
    )


def test_each_session_rung_term_estimator_combination_appears_exactly_once(one_rung):
    coefficients = one_rung.coefficients
    assert not coefficients.duplicated([*SC, "model", "term", "estimator"]).any()
    counts = coefficients.groupby([*SC, "model", "estimator"]).size().unstack()
    n_predictors = len(BASE_RUNG.predictors)
    assert (counts[CONDITIONAL_POISSON_COUNT] == n_predictors).all()
    assert (counts[CONDITIONAL_POISSON_FRACTION] == n_predictors).all()
    # The Dirichlet-multinomial adds log_phi_intercept and log_phi_is_swr.
    assert (counts[DIRICHLET_MULTINOMIAL] == n_predictors + 2).all()


def test_the_three_estimators_are_fit_on_one_shared_event_set(one_rung):
    per_fit = one_rung.coefficients.groupby([*SC, "model"])["n_events"].nunique()
    assert (per_fit == 1).all()
    assert one_rung.coefficients["n_events"].eq(COMPARISON_N_EVENTS_PER_SESSION).all()
    assert one_rung.coefficients["n_events_shared_dropped"].eq(0).all()


def test_the_fraction_estimator_alone_reports_clustered_standard_errors(one_rung):
    by_estimator = one_rung.coefficients.groupby("estimator")["se_source"].unique()
    assert list(by_estimator[CONDITIONAL_POISSON_FRACTION]) == ["event_clustered"]
    assert list(by_estimator[CONDITIONAL_POISSON_COUNT]) == ["model"]
    assert list(by_estimator[DIRICHLET_MULTINOMIAL]) == ["model"]


def test_only_the_fraction_estimator_is_flagged_as_a_quasi_scale(one_rung):
    scales = one_rung.coefficients.groupby("estimator")["scale"].unique()
    assert list(scales[CONDITIONAL_POISSON_FRACTION]) == [ec.QUASI_SCALE]
    assert list(scales[CONDITIONAL_POISSON_COUNT]) == [ec.LIKELIHOOD_SCALE]
    assert list(scales[DIRICHLET_MULTINOMIAL]) == [ec.LIKELIHOOD_SCALE]


def test_the_conditional_fits_carry_the_multinomial_constant_into_their_likelihood(
    cohort, one_rung
):
    """A conditional llf plus ``multinomial_log_constant`` is what is stored."""
    key = tuple(cohort[SC].drop_duplicates().iloc[0])
    rows = cohort[
        np.logical_and.reduce([cohort[c] == v for c, v in zip(SC, key, strict=True)])
    ]
    direct = fit_conditional_poisson(
        rows,
        predictors=list(BASE_RUNG.predictors),
        response_col=ec.COUNT_RESPONSE,
        event_col="event_id",
        n_categories=N_ARM_CATEGORIES,
    )
    constant = multinomial_log_constant(rows[ec.COUNT_RESPONSE], rows["event_id"])
    stored = one_rung.coefficients[
        np.logical_and.reduce(
            [one_rung.coefficients[c] == v for c, v in zip(SC, key, strict=True)]
        )
        & (one_rung.coefficients["estimator"] == CONDITIONAL_POISSON_COUNT)
    ].iloc[0]
    assert stored["llf_equalized"] == pytest.approx(direct["llf"] + constant, rel=1e-10)
    assert stored["aic_equalized"] == pytest.approx(
        2 * direct["n_params"] - 2 * (direct["llf"] + constant), rel=1e-10
    )
    # The Dirichlet-multinomial likelihood already carries the constant, so its
    # equalized value is its own llf.
    dm = one_rung.coefficients[
        (one_rung.coefficients["estimator"] == DIRICHLET_MULTINOMIAL)
    ].iloc[0]
    assert dm["aic_equalized"] == pytest.approx(
        2 * dm["n_params"] - 2 * dm["llf_equalized"], rel=1e-12
    )


def test_overdispersion_widens_the_dirichlet_multinomial_standard_errors(one_rung):
    """With phi = 3 planted, the count model's SEs are the optimistic ones."""
    row = one_rung.summary[
        (one_rung.summary["model"] == BASE_RUNG.name)
        & (one_rung.summary["term"] == "is_current_arm")
    ].iloc[0]
    # The column names spell out WHICH standard error each side is: the fraction
    # ratio is a sandwich over a model SE, the DM's is model over model, and the
    # two are not the same comparison.
    assert row["median_se_ratio_dm_model_over_count_model"] > 1.0
    assert row["median_se_ratio_fraction_cluster_over_count_model"] > 1.0
    # The mean structure is shared, so the point estimates agree even where the
    # uncertainty does not.
    assert row["median_coef_dm"] == pytest.approx(row["median_coef_count"], abs=0.25)


def test_the_planted_concentration_shift_is_recovered_with_a_consistent_sign(one_rung):
    row = one_rung.summary[one_rung.summary["model"] == BASE_RUNG.name].iloc[0]
    assert row["dm_log_phi_is_swr_n_animals"] == 2
    assert row["dm_log_phi_is_swr_n_animals_neg"] == 2
    assert row["dm_log_phi_is_swr_animal_median"] < 0.0


def test_the_bootstrap_reports_an_interval_about_each_median(one_rung):
    bootstrap = one_rung.bootstrap
    assert not bootstrap.duplicated(["model", "term", "estimator"]).any()
    assert bootstrap["n_animals"].eq(2).all()
    assert bootstrap["seed"].eq(ec.BOOTSTRAP_SEED).all()
    # two animals: the median +/- t(0.975, 1) sqrt(2) SD(draws)
    half = 12.7062047362 * np.sqrt(2) * bootstrap["bootstrap_sd"]
    np.testing.assert_allclose(bootstrap["low"], bootstrap["statistic"] - half)
    np.testing.assert_allclose(bootstrap["high"], bootstrap["statistic"] + half)


def test_the_bootstrap_is_reproducible_from_its_seed(one_rung):
    repeated = ec.bootstrap_animals(
        one_rung.coefficients, n_draws=200, seed=ec.BOOTSTRAP_SEED
    )
    pd.testing.assert_frame_equal(repeated, one_rung.bootstrap)


def test_the_comparison_table_round_trips_through_parquet(one_rung, tmp_path):
    path = write_table(one_rung.coefficients, tmp_path / "comparison")
    restored = read_table(path)
    assert len(restored) == len(one_rung.coefficients)
    assert not restored.duplicated([*SC, "model", "term", "estimator"]).any()


def test_the_summary_has_one_row_per_rung_and_term(two_rungs):
    summary = two_rungs.summary
    assert not summary.duplicated(["model", "term"]).any()
    expected = set(
        map(tuple, two_rungs.coefficients[["model", "term"]].drop_duplicates().values)
    )
    assert set(map(tuple, summary[["model", "term"]].values)) == expected


def test_the_summary_reports_the_sessions_each_term_lost_to_identifiability(two_rungs):
    """A session that cannot inform a term (NaN coefficient) leaves that term's
    medians, and the summary says how many did."""
    coefficients = two_rungs.coefficients.copy()
    term = "is_current_arm"
    first = coefficients[
        (coefficients["model"] == BASE_RUNG.name) & (coefficients["term"] == term)
    ][SC].iloc[0]
    hit = (
        (coefficients["model"] == BASE_RUNG.name)
        & (coefficients["term"] == term)
        & (coefficients[SC] == first).all(axis=1)
        & (coefficients["estimator"] == CONDITIONAL_POISSON_FRACTION)
    )
    coefficients.loc[hit, "coef"] = np.nan

    summary = ec.summarize_estimators(coefficients).set_index(["model", "term"])

    before = two_rungs.summary.set_index(["model", "term"])
    row = summary.loc[(BASE_RUNG.name, term)]
    assert row["n_sessions_unidentified"] == (
        before.loc[(BASE_RUNG.name, term), "n_sessions_unidentified"] + 1
    )
    assert (
        row["n_sessions_count"]
        == before.loc[(BASE_RUNG.name, term), "n_sessions_count"] - 1
    )


def test_delta_aic_is_formed_only_against_a_parent_fit_on_the_same_events(two_rungs):
    coefficients = two_rungs.coefficients
    parentless = coefficients[coefficients["model"] == BASE_RUNG.name]
    assert parentless["delta_aic_vs_parent"].isna().all()

    child = coefficients[coefficients["model"] == REWARD_RUNG.name]
    assert child["delta_aic_vs_parent"].notna().all()
    assert not child["delta_aic_row_mismatch"].any()
    # The reward predictor is in the generative truth, so the count model --
    # which treats an event's 32 timebins as 32 independent observations --
    # prefers the larger model decisively on every session.
    per_session = child.drop_duplicates([*SC, "estimator"]).pivot(
        index=SC, columns="estimator", values="delta_aic_vs_parent"
    )
    assert (per_session[CONDITIONAL_POISSON_COUNT] < 0).all()
    # The Dirichlet-multinomial spends the same evidence at the event scale, so
    # its support for the same true predictor is far weaker on every session --
    # the count model's confidence is the overdispersion it ignores.
    assert (
        per_session[DIRICHLET_MULTINOMIAL] > per_session[CONDITIONAL_POISSON_COUNT]
    ).all()


def test_both_rungs_are_compared_on_the_same_sessions(two_rungs):
    per_rung = two_rungs.coefficients.groupby("model")[SC].apply(
        lambda frame: frozenset(map(tuple, frame.drop_duplicates().values))
    )
    assert per_rung.nunique() == 1
    assert two_rungs.skips.empty


def test_the_persisted_rows_carry_the_scale_the_rule_is_read_on(one_rung):
    """``sd`` and ``separated_terms`` travel with the coefficients."""
    coefficients = one_rung.coefficients
    predictors = coefficients[coefficients["term"].isin(BASE_RUNG.predictors)]
    assert predictors["sd"].notna().all()
    assert (predictors["sd"] > 0).all()
    # The concentration MODIFIER carries the SD of its own event-level design
    # column, so the effect-scale rule applies to it like any other term; the
    # intercept's column is a constant with no spread and stays flag-only.
    intercept = coefficients[coefficients["term"] == LOG_PHI_INTERCEPT]
    assert not intercept.empty
    assert intercept["sd"].isna().all()
    modifier = coefficients[coefficients["term"] == ec.LOG_PHI_IS_SWR]
    assert not modifier.empty
    assert (modifier["sd"] > 0).all()
    assert "separated_terms" in coefficients.columns
