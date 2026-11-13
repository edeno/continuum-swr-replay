"""The three-estimator comparison's statistics, on tables built by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.content.dirichlet_multinomial import (
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.replay_models.content.estimator_comparison import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    attach_delta_aic,
    estimator_rows,
    identified_rows,
    n_unpaired_triples,
    unidentified_rows,
)
from continuum_swr_replay.analysis.replay_models.fitting import (
    CONDITIONAL_POISSON_COUNT,
    CONDITIONAL_POISSON_FRACTION,
)


def test_bootstrap_defaults_are_two_thousand_draws_at_seed_zero():
    assert N_BOOTSTRAP == 2000
    assert BOOTSTRAP_SEED == 0


def _row(term, coef, sd, separated_terms="", estimator=CONDITIONAL_POISSON_COUNT):
    """One synthetic coefficient row, enough for the identifiability rule."""
    return {
        "animal": "alpha",
        "session_date": "20200101",
        "epoch": "02_r1",
        "model": "B_spatial",
        "term": term,
        "estimator": estimator,
        "coef": coef,
        "sd": sd,
        "separated_terms": separated_terms,
        "separated_term": term in set(separated_terms.split("|")) - {""},
    }


def test_a_steep_coefficient_on_a_small_sd_predictor_stays_identified():
    """|coef| = 12 at SD 0.05 is an effect of 0.6, not a runaway fit."""
    rows = pd.DataFrame([_row("is_current_arm", 12.0, 0.05)])
    assert not unidentified_rows(rows).iloc[0]


def test_the_same_coefficient_on_a_unit_sd_predictor_is_dropped():
    rows = pd.DataFrame([_row("is_current_arm", 12.0, 1.0)])
    assert unidentified_rows(rows).iloc[0]


def test_a_flagged_term_is_dropped_however_small_its_effect():
    rows = pd.DataFrame(
        [_row("is_current_arm", 0.3, 0.5, separated_terms="is_current_arm")]
    )
    assert unidentified_rows(rows).iloc[0]


def test_a_large_concentration_intercept_is_kept_because_it_has_no_design_scale():
    """``log_phi_intercept`` is unbounded; only the fitter's flag may drop it."""
    rows = pd.DataFrame(
        [
            _row("log_phi_intercept", 25.0, np.nan, estimator=DIRICHLET_MULTINOMIAL),
            _row(
                "log_phi_is_swr",
                25.0,
                np.nan,
                separated_terms="log_phi_is_swr",
                estimator=DIRICHLET_MULTINOMIAL,
            ),
        ]
    )
    assert list(unidentified_rows(rows)) == [False, True]


def test_a_term_the_fit_could_not_estimate_drops_its_estimator_triple():
    """A NaN coefficient means the session does not inform the term.

    The conditional-Poisson fitter reports such a term (``unidentified_terms``)
    with a NaN coefficient; the paired rule drops the term's triple, so the
    estimators are still compared on the same sessions, and keeps the others.
    """
    rows = pd.DataFrame(
        [
            _row(
                "is_candidate_arm_0", np.nan, 0.3, estimator=CONDITIONAL_POISSON_COUNT
            ),
            _row(
                "is_candidate_arm_0",
                np.nan,
                0.3,
                estimator=CONDITIONAL_POISSON_FRACTION,
            ),
            _row("is_candidate_arm_0", -20.0, 0.3, estimator=DIRICHLET_MULTINOMIAL),
            _row("is_current_arm", 0.7, 0.3, estimator=CONDITIONAL_POISSON_COUNT),
            _row("is_current_arm", 0.7, 0.3, estimator=CONDITIONAL_POISSON_FRACTION),
            _row("is_current_arm", 0.8, 0.3, estimator=DIRICHLET_MULTINOMIAL),
        ]
    )
    assert list(unidentified_rows(rows)) == [True, True, False, False, False, False]
    kept = identified_rows(rows)
    assert set(kept["term"]) == {"is_current_arm"}
    assert len(kept) == 3


def test_an_unidentified_row_drops_the_whole_estimator_triple():
    """Otherwise one summary row would average different sessions per estimator."""
    rows = pd.DataFrame(
        [
            _row("is_current_arm", 12.0, 1.0, estimator=CONDITIONAL_POISSON_COUNT),
            _row("is_current_arm", 0.7, 1.0, estimator=CONDITIONAL_POISSON_FRACTION),
            _row("is_current_arm", 0.8, 1.0, estimator=DIRICHLET_MULTINOMIAL),
            _row("is_prev_arm", -0.4, 1.0, estimator=CONDITIONAL_POISSON_COUNT),
            _row("is_prev_arm", -0.4, 1.0, estimator=CONDITIONAL_POISSON_FRACTION),
            _row("is_prev_arm", -0.4, 1.0, estimator=DIRICHLET_MULTINOMIAL),
        ]
    )
    kept = identified_rows(rows)
    assert set(kept["term"]) == {"is_prev_arm"}
    assert len(kept) == 3
    assert n_unpaired_triples(rows) == 1


def test_the_fraction_fits_non_finite_sandwich_falls_back_to_the_model_se():
    """A NaN clustered SE keeps the row on its model SE, labelled as such.

    This is the shipped aggregation's own rule (``session_se``): a predictor
    that is constant within one session has no sandwich, and dropping the
    session would be selection on the estimator's numerical trouble.
    """
    fit = {
        "coef_is_current_arm": 0.8,
        "bse_is_current_arm": 0.1,
        "bse_cluster_is_current_arm": np.nan,
        "llf": -100.0,
        "llf_includes_constants": False,
        "n_params": 2,
    }
    rows = estimator_rows(
        ("alpha", "20200101", "02_r1"),
        model="B_spatial",
        parent=None,
        leaf_only=False,
        fits={CONDITIONAL_POISSON_FRACTION: fit},
        info={"n_events_in": 10},
        log_constant=0.0,
        n_events=10,
    )
    (row,) = rows
    assert row["se"] == 0.1
    assert row["se_source"] == "model_fallback"


def test_the_fraction_fit_without_a_sandwich_column_is_refused():
    fit = {
        "coef_is_current_arm": 0.8,
        "bse_is_current_arm": 0.1,
        "llf": -100.0,
        "llf_includes_constants": False,
        "n_params": 2,
    }
    with pytest.raises(ValueError, match="bse_cluster_is_current_arm"):
        estimator_rows(
            ("alpha", "20200101", "02_r1"),
            model="B_spatial",
            parent=None,
            leaf_only=False,
            fits={CONDITIONAL_POISSON_FRACTION: fit},
            info={"n_events_in": 10},
            log_constant=0.0,
            n_events=10,
        )


def _fit_rows(session, model, parent, aic, n_events, terms=("a", "b"), **extra):
    """One fit's rows: one per term, the fit's AIC and event count on each."""
    return [
        {
            "animal": "alpha",
            "session_date": session,
            "epoch": "02_r1",
            "model": model,
            "parent": parent,
            "term": term,
            "estimator": CONDITIONAL_POISSON_COUNT,
            "aic_equalized": aic,
            "n_events": n_events,
            **extra,
        }
        for term in terms
    ]


def test_the_aic_difference_is_child_minus_parent_where_the_fits_compare():
    """s1: same events, the child's two terms both get 90 - 100 = -10. s2: the
    child lost two events, so the fits do not compare. s3: the parent was not
    fit, so there is nothing to compare and nothing is flagged. The parent
    rung itself has no parent."""
    coefficients = pd.DataFrame(
        [
            *_fit_rows("s1", "P", None, 100.0, 20),
            *_fit_rows("s1", "C", "P", 90.0, 20),
            *_fit_rows("s2", "P", None, 100.0, 20),
            *_fit_rows("s2", "C", "P", 80.0, 18),
            *_fit_rows("s3", "C", "P", 70.0, 20),
        ]
    )

    out = attach_delta_aic(coefficients)

    assert len(out) == len(coefficients)
    child = out[out["model"] == "C"].set_index(["session_date", "term"])
    assert child.loc[("s1", "a"), "delta_aic_vs_parent"] == -10.0
    assert child.loc[("s1", "b"), "delta_aic_vs_parent"] == -10.0
    assert child["delta_aic_vs_parent"].isna().sum() == 4
    assert child["delta_aic_row_mismatch"].to_dict() == {
        ("s1", "a"): False,
        ("s1", "b"): False,
        ("s2", "a"): True,
        ("s2", "b"): True,
        ("s3", "a"): False,
        ("s3", "b"): False,
    }
    parent = out[out["model"] == "P"]
    assert parent["delta_aic_vs_parent"].isna().all()
    assert not parent["delta_aic_row_mismatch"].any()
    assert list(out.columns) == [
        *coefficients.columns,
        "delta_aic_vs_parent",
        "delta_aic_row_mismatch",
    ]


def test_fits_that_disagree_on_a_carried_comparability_column_do_not_compare():
    """The one rule decides, on whatever comparability columns the rows carry:
    equal event counts are not enough when one likelihood keeps the constants
    and the other does not."""
    coefficients = pd.DataFrame(
        [
            *_fit_rows("s1", "P", None, 100.0, 20, llf_includes_constants=False),
            *_fit_rows("s1", "C", "P", 90.0, 20, llf_includes_constants=True),
            *_fit_rows("s2", "P", None, 100.0, 20, llf_includes_constants=False),
            *_fit_rows("s2", "C", "P", 90.0, 20, llf_includes_constants=False),
        ]
    )

    child = attach_delta_aic(coefficients).query("model == 'C'")

    assert child.groupby("session_date")["delta_aic_vs_parent"].first().to_dict() == {
        "s1": pytest.approx(np.nan, nan_ok=True),
        "s2": -10.0,
    }
    assert child.groupby("session_date")["delta_aic_row_mismatch"].all().to_dict() == {
        "s1": True,
        "s2": False,
    }


def test_an_empty_comparison_gets_the_two_columns():
    out = attach_delta_aic(
        pd.DataFrame(columns=["model", "parent", "aic_equalized", "n_events"])
    )
    assert out.empty
    assert {"delta_aic_vs_parent", "delta_aic_row_mismatch"} <= set(out.columns)
