"""One content rung fit on every session: which sessions, and what each row says."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial import (
    concentration_cols_for_cohort,
)
from continuum_swr_replay.analysis.behavior.content.session_fits import (
    fit_model_per_session,
)
from continuum_swr_replay.analysis.behavior.tables.table_contracts import (
    CV_RESULT_COLUMNS,
)
from tests.support._content_tables import dm_session, simulated_session


@pytest.mark.parametrize(
    "response_col,estimator",
    [
        ("arm_fraction", "conditional_poisson_fraction"),
        ("arm_count_hard", "conditional_poisson_count"),
    ],
)
def test_fitted_rows_record_the_estimator_and_the_response(
    response_col, estimator
) -> None:
    result = fit_model_per_session(
        simulated_session(response_col),
        "X",
        ["is_current_arm"],
        response_col=response_col,
    )
    assert len(result) == 1
    assert result["estimator"].tolist() == [estimator]
    assert result["response_col"].tolist() == [response_col]


def test_fitted_rows_carry_the_optimizer_diagnostics() -> None:
    result = fit_model_per_session(
        simulated_session("arm_fraction"),
        "X",
        ["is_current_arm"],
        response_col="arm_fraction",
    )
    row = result.iloc[0]
    assert row["llf"] < 0
    assert row["newton_decrement"] < 1e-8
    assert row["max_abs_grad"] < 1e-3
    assert row["n_newton_steps"] >= 0
    assert row["unidentified_terms"] == ""


def test_a_session_of_one_event_is_not_fit() -> None:
    """Two events is the floor: 18 rows on nine arms, 6 on three patches."""
    full = simulated_session("arm_fraction")
    one_event = full[full["event_id"].eq(full["event_id"].iloc[0])].assign(
        epoch="04_r2"
    )
    cohort = pd.concat([full, one_event], ignore_index=True)

    result = fit_model_per_session(
        cohort, "X", ["is_current_arm"], response_col="arm_fraction"
    )

    assert result["epoch"].tolist() == ["02_r1"]


def test_no_fitted_session_gives_an_empty_frame() -> None:
    full = simulated_session("arm_fraction")
    one_event = full[full["event_id"].eq(full["event_id"].iloc[0])]

    result = fit_model_per_session(
        one_event, "X", ["is_current_arm"], response_col="arm_fraction"
    )

    assert result.empty


def test_a_failed_cross_validation_keeps_the_fit_with_nan_scores(monkeypatch) -> None:
    """Cross-validation is a diagnostic: it never costs a session its fit."""
    from continuum_swr_replay.analysis.behavior.content import session_fits

    monkeypatch.setattr(
        session_fits, "cross_validate_conditional_poisson", lambda *a, **k: None
    )

    result = fit_model_per_session(
        simulated_session("arm_fraction"),
        "X",
        ["is_current_arm"],
        run_cv=True,
        response_col="arm_fraction",
    )

    assert len(result) == 1
    assert result[list(CV_RESULT_COLUMNS)].isna().all(axis=None)
    assert np.isfinite(result["coef_is_current_arm"]).all()


def test_the_event_grain_fits_one_observation_per_event() -> None:
    session = simulated_session("arm_count_hard")

    result = fit_model_per_session(
        session, "X", ["is_current_arm"], response_col="arm_count_hard", grain="event"
    )

    assert result["n_events"].tolist() == [session["event_id"].nunique()]
    assert result["estimator"].tolist() == ["conditional_logit_dominant"]


def test_short_dm_session_borrows_its_animals_concentration() -> None:
    """A 20-event session cannot pin down its own concentration; a 60-event one can."""
    cohort = pd.concat(
        [
            dm_session(60, seed=0, epoch="02_r1"),
            dm_session(20, seed=1, epoch="04_r2"),
        ],
        ignore_index=True,
    )

    result = fit_model_per_session(
        cohort,
        "X",
        ["is_current_arm"],
        response_col="arm_count_hard",
        estimator="dirichlet_multinomial",
        n_jobs=2,
        min_events_session_phi=40,
    )

    assert result["estimator"].tolist() == ["dirichlet_multinomial"] * 2
    assert result["response_col"].tolist() == ["arm_count_hard"] * 2
    rows = result.set_index("epoch")
    assert rows.loc["02_r1", "n_events"] == 60
    assert rows.loc["04_r2", "n_events"] == 20
    assert rows.loc["02_r1", "phi_source"] == "session"
    assert rows.loc["04_r2", "phi_source"] == "animal_pooled"
    assert rows.loc["04_r2", "coef_log_phi_intercept"] == pytest.approx(
        rows.loc["02_r1", "coef_log_phi_intercept"]
    )
    # The concentration modifier is estimated because the cohort carries both
    # kinds of event.
    assert "coef_log_phi_is_swr" in result.columns
    assert result["coef_log_phi_is_swr"].notna().all()
    assert bool(result["llf_includes_constants"].all())


def test_dm_concentration_modifier_is_dropped_within_an_swr_subset() -> None:
    """Inside a subset ``is_swr`` is constant, so there is no contrast to fit."""
    cohort = dm_session(60, seed=0)
    assert concentration_cols_for_cohort(cohort) == ("is_swr",)
    subset = cohort[cohort["is_swr"] == 1.0]
    assert concentration_cols_for_cohort(subset) == ()
    assert concentration_cols_for_cohort(cohort.drop(columns="is_swr")) == ()


def test_dm_session_with_a_refused_design_is_an_audited_skip() -> None:
    """A predictor constant within every event raises; the ladder books it instead."""
    good = dm_session(60, seed=0, epoch="02_r1")
    bad = dm_session(60, seed=2, epoch="04_r2").assign(is_prev_arm=0.0)
    cohort = pd.concat([good, bad], ignore_index=True)

    result = fit_model_per_session(
        cohort,
        "X",
        ["is_current_arm", "is_prev_arm"],
        response_col="arm_count_hard",
        estimator="dirichlet_multinomial",
        n_jobs=2,
        min_events_session_phi=40,
    )

    assert result["epoch"].tolist() == ["02_r1"]
    skips = result.attrs["estimator_skips"]
    assert [skip["session"] for skip in skips] == ["chimi/20200212/04_r2"]
    assert skips[0]["reason"].startswith("dm_design:")
