"""Contracts of the three-estimator content comparison.

The comparison exists to say whether the retired count model, the shipped
fraction model and the Dirichlet-multinomial disagree, so what is pinned here is
what makes such a comparison readable: every (session, rung, term, estimator)
appears exactly once, the three estimators are fit on ONE shared event set per
(session, rung), the likelihood constants are equalized before any AIC is
differenced, and the animal bootstrap reports the requested percentiles.

The cohort is simulated with a planted Dirichlet concentration (phi = 3), so the
within-event timebins are genuinely correlated and the count model's standard
errors are known to be too small -- which is the quantitative claim the
comparison is supposed to surface.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.content import estimator_comparison as ec
from continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial import (
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial import (
    multinomial_log_constant,
)
from continuum_swr_replay.analysis.behavior.content.simulation import (
    simulate_dirichlet_multinomial_counts,
)
from continuum_swr_replay.analysis.behavior.fitting import (
    CONDITIONAL_POISSON_COUNT,
    CONDITIONAL_POISSON_FRACTION,
    fit_conditional_poisson,
)
from continuum_swr_replay.tables.io import read_table, write_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from scripts.behavior.content import compare_content_estimators as mod

SC = list(SESSION_COLUMNS)

#: Planted concentration. Small enough that each event's timebins are strongly
#: correlated, so an estimator that treats them as independent understates its
#: uncertainty.
PHI_TRUE = 3.0
LOG_PHI_IS_SWR_TRUE = -0.5
N_EVENTS_PER_SESSION = 60
BINS_PER_EVENT = 32

BETA_TRUE = {
    "is_current_arm": 1.0,
    "is_prev_arm": -0.5,
    "reward_fraction_candidate_arm_last10_visits": 0.4,
}

BASE_RUNG = mod.RungSpec(
    name="B_spatial",
    predictors=("is_current_arm", "is_prev_arm"),
    parent=None,
    leaf_only=False,
    key_term="is_current_arm",
)
REWARD_RUNG = mod.RungSpec(
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


def _session(animal: str, session_date: str, seed: int, offset: int) -> pd.DataFrame:
    frame = simulate_dirichlet_multinomial_counts(
        BETA_TRUE,
        log_phi_intercept=float(np.log(PHI_TRUE)),
        log_phi_is_swr=LOG_PHI_IS_SWR_TRUE,
        n_events=N_EVENTS_PER_SESSION,
        L_e=BINS_PER_EVENT,
        seed=seed,
    ).copy()
    # ``attrs["p_true"]`` is an array and pandas compares attrs element-wise on
    # concat; drop it rather than work around the comparison.
    frame.attrs = {}
    frame["event_id"] = frame["event_id"] + offset
    frame["animal"] = animal
    frame["session_date"] = session_date
    frame["epoch"] = "02_r1"
    return frame


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    """Two sessions from two animals, both carrying SWR and non-SWR events."""
    return pd.concat(
        [
            _session("alpha", "20200101", seed=11, offset=0),
            _session("bravo", "20200102", seed=22, offset=10_000),
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def one_rung(cohort) -> mod.ComparisonResult:
    return mod.run_comparison(
        cohort, (BASE_RUNG,), n_jobs=2, n_bootstrap=200, seed=ec.BOOTSTRAP_SEED
    )


@pytest.fixture(scope="module")
def two_rungs(cohort) -> mod.ComparisonResult:
    return mod.run_comparison(
        cohort,
        (BASE_RUNG, REWARD_RUNG),
        n_jobs=2,
        n_bootstrap=200,
        seed=ec.BOOTSTRAP_SEED,
    )


def test_the_simulated_cohort_carries_both_swr_values(cohort):
    per_session = cohort.groupby(SC, sort=False)["is_swr"].nunique()
    assert (per_session == 2).all()
    assert (
        cohort.groupby(SC, sort=False)["event_id"]
        .nunique()
        .eq(N_EVENTS_PER_SESSION)
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
    assert one_rung.coefficients["n_events"].eq(N_EVENTS_PER_SESSION).all()
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
        response_col=mod.COUNT_RESPONSE,
        event_col="event_id",
        n_categories=mod.N_ARM_CATEGORIES,
    )
    constant = multinomial_log_constant(rows[mod.COUNT_RESPONSE], rows["event_id"])
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


def test_the_bootstrap_reports_the_requested_percentiles(one_rung):
    bootstrap = one_rung.bootstrap
    assert not bootstrap.duplicated(["model", "term", "estimator"]).any()
    for column in ("p2_5", "p50", "p97_5"):
        assert column in bootstrap.columns
    assert (bootstrap["p2_5"] <= bootstrap["p50"]).all()
    assert (bootstrap["p50"] <= bootstrap["p97_5"]).all()
    assert bootstrap["n_animals"].eq(2).all()
    assert bootstrap["seed"].eq(ec.BOOTSTRAP_SEED).all()
    current = bootstrap[
        (bootstrap["term"] == "is_current_arm")
        & (bootstrap["estimator"] == DIRICHLET_MULTINOMIAL)
    ].iloc[0]
    assert current["p2_5"] <= current["statistic"] <= current["p97_5"]


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


@pytest.mark.slow
def test_figures_and_caption_are_written(two_rungs, tmp_path):
    agreement = mod.plot_coefficient_agreement(
        two_rungs.coefficients,
        (BASE_RUNG, REWARD_RUNG),
        tmp_path / "coef_agreement.png",
    )
    log_phi = mod.plot_log_phi_by_session(
        two_rungs.coefficients,
        two_rungs.bootstrap,
        tmp_path / "log_phi_by_session.png",
        rung=BASE_RUNG.name,
    )
    caption = mod.write_caption(tmp_path / "content_estimators_caption.md")
    assert agreement.stat().st_size > 0
    assert log_phi.stat().st_size > 0
    assert "seed 0" in caption.read_text()


def test_resolve_rungs_pulls_in_the_parent_a_rung_is_differenced_against():
    selected = mod.resolve_rungs(["CS5_armvalue"])
    names = [rung.name for rung in selected]
    assert names == ["C2_base_swr_leaf", "CS5_armvalue"]
    # The parent carries everything but the two value terms, so the rung's
    # delta_aic prices only those.
    parent, rung = selected
    assert set(rung.predictors) - set(parent.predictors) == {
        "reward_belief_mean_candidate_arm_preoutcome",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    }
    assert set(parent.predictors) <= set(rung.predictors)
    # A leaf-only rung must be differenced against a leaf-only parent: a
    # six-category likelihood and a nine-arm one are not comparable.
    assert all(rung.leaf_only for rung in selected)
    assert selected[-1].n_categories == 6


def test_resolve_rungs_refuses_an_unknown_name():
    with pytest.raises(ValueError, match="unknown rung"):
        mod.resolve_rungs(["not_a_rung"])


def test_every_comparison_rung_names_a_parent_on_its_own_candidate_set():
    catalog = {rung.name: rung for rung in mod.COMPARISON_RUNGS}
    for rung in mod.COMPARISON_RUNGS:
        if rung.parent is None:
            continue
        assert catalog[rung.parent].leaf_only == rung.leaf_only
        assert set(catalog[rung.parent].predictors) <= set(rung.predictors)


def test_the_persisted_rows_carry_the_scale_the_rule_is_read_on(one_rung):
    """``sd`` and ``separated_terms`` travel with the coefficients."""
    coefficients = one_rung.coefficients
    predictors = coefficients[coefficients["term"].isin(BASE_RUNG.predictors)]
    assert predictors["sd"].notna().all()
    assert (predictors["sd"] > 0).all()
    # The concentration MODIFIER carries the SD of its own event-level design
    # column, so the effect-scale rule applies to it like any other term; the
    # intercept's column is a constant with no spread and stays flag-only.
    intercept = coefficients[coefficients["term"] == mod.LOG_PHI_INTERCEPT]
    assert not intercept.empty
    assert intercept["sd"].isna().all()
    modifier = coefficients[coefficients["term"] == ec.LOG_PHI_IS_SWR]
    assert not modifier.empty
    assert (modifier["sd"] > 0).all()
    assert "separated_terms" in coefficients.columns


def test_the_concentration_figure_refuses_a_rung_it_has_no_fits_for(one_rung, tmp_path):
    with pytest.raises(ValueError, match="no identified Dirichlet-multinomial fits"):
        mod.plot_log_phi_by_session(
            one_rung.coefficients,
            one_rung.bootstrap,
            tmp_path / "log_phi_by_session.png",
            rung="not_a_fitted_rung",
        )


def test_the_concentration_figure_is_optional_but_the_provenance_is_not(
    one_rung, cohort, tmp_path, monkeypatch
):
    """A rung selection without the log-phi rung must still finish the run.

    The figure is pinned to one rung; raising there aborted AFTER the tables
    were written and BEFORE the provenance sidecar, leaving artifacts no reader
    could date or trace back to their inputs.
    """
    monkeypatch.setattr(mod, "assemble_arm", lambda *args: (cohort.copy(), {}))
    monkeypatch.setattr(
        mod, "select_outcome_stratum", lambda ev, stratum: (ev, {"stratum": "all"})
    )
    monkeypatch.setattr(mod, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(mod, "run_comparison", lambda *args, **kwargs: one_rung)
    figures_dir = tmp_path / "figures"

    # M3 pulls in its parent M2; neither is the concentration figure's rung.
    mod.main(
        [
            "--rungs",
            "M3",
            "--output-dir",
            str(tmp_path / "out"),
            "--figures-dir",
            str(figures_dir),
            "--n-bootstrap",
            "10",
        ]
    )

    assert not (figures_dir / "log_phi_by_session.png").exists()
    provenance = tmp_path / "out" / f"{mod.COEFFICIENTS_BASENAME}.provenance.json"
    assert provenance.exists()
    assert json.loads(provenance.read_text())["rungs"] == ["M2", "M3"]


@pytest.mark.parametrize(
    ("names", "expected"),
    [(["C2_base"], mod.DEFAULT_LOG_PHI_RUNG), (["M3"], None)],
)
def test_the_figure_rung_is_reported_only_when_it_was_selected(names, expected):
    assert mod.log_phi_figure_rung(mod.resolve_rungs(names)) == expected
