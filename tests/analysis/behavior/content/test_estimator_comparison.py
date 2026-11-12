"""The three-estimator comparison's statistics, on tables built by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial import (
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.behavior.content.estimator_comparison import (
    BOOTSTRAP_PERCENTILES,
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    estimator_rows,
    identified_rows,
    n_unpaired_triples,
    unidentified_rows,
)
from continuum_swr_replay.analysis.behavior.fitting import (
    CONDITIONAL_POISSON_COUNT,
    CONDITIONAL_POISSON_FRACTION,
)


def test_bootstrap_defaults_are_two_thousand_draws_at_seed_zero():
    assert N_BOOTSTRAP == 2000
    assert BOOTSTRAP_SEED == 0
    assert BOOTSTRAP_PERCENTILES == (2.5, 50.0, 97.5)


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
