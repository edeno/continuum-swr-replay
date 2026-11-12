"""Content rungs fit on every session: which sessions, and what each row says."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior import contrast_aggregation
from continuum_swr_replay.analysis.behavior.content import session_fits
from continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial import (
    concentration_cols_for_cohort,
)
from continuum_swr_replay.analysis.behavior.content.session_fits import (
    LOGIT_SKIP_COLUMNS,
    MIN_PATCH_EVENTS,
    fit_conditional_logit_robustness,
    fit_content_ladder,
    fit_contrast_ladder,
    fit_contrast_session,
    fit_model_per_session,
    fit_n_categories,
    fit_patch_switch,
    patch_models_for_source,
    patch_switch_models_for_source,
    resolve_concentration_cols,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    ModelSpec,
    leaf_only_cohort,
)
from continuum_swr_replay.analysis.behavior.tables.table_contracts import (
    CV_RESULT_COLUMNS,
)
from tests.support._content_tables import (
    dm_contrast_session,
    dm_session,
    fail_logit_for,
    logit_session,
    simulated_session,
    with_event_length,
)


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


# ── The conditional-logit robustness fits ──

_LOGIT_LADDER = {
    "N": ModelSpec("N", (), "uniform null", None),
    "X": ModelSpec("X", ("is_current_arm",), "current", "N"),
    "Y": ModelSpec("Y", ("is_current_arm", "is_prev_arm"), "current + prev", "X"),
}


def _logit_cohort() -> pd.DataFrame:
    return pd.concat(
        [logit_session("02_r1", seed=0), logit_session("04_r2", seed=1)],
        ignore_index=True,
    )


def _robustness(cohort: pd.DataFrame, primary_epochs=("02_r1", "04_r2")):
    primary = cohort[["animal", "session_date", "epoch"]].drop_duplicates()
    return fit_conditional_logit_robustness(
        cohort,
        primary[primary["epoch"].isin(primary_epochs)],
        _LOGIT_LADDER,
        ["N", "X", "Y"],
        leaf_only=False,
        response_col="arm_fraction",
        event_col="event_id",
        response_renormalized=True,
    )


def test_the_logit_fits_cover_the_primary_sessions_of_every_rung_with_predictors():
    """The null rung has nothing to fit; a session outside the primary ladder's
    is not fit at all; each row records what it was fit on."""
    logit, skips = _robustness(_logit_cohort(), primary_epochs=("02_r1",))

    assert set(logit) == {"X", "Y"}
    for model, frame in logit.items():
        assert frame["epoch"].tolist() == ["02_r1"]
        assert frame["model"].tolist() == [model]
        assert frame["response_col"].tolist() == ["arm_fraction"]
        assert frame["response_renormalized"].tolist() == [True]
    assert skips.empty
    assert tuple(skips.columns) == LOGIT_SKIP_COLUMNS


def test_a_session_the_logit_fit_refuses_is_skipped_and_recorded(monkeypatch):
    fail_logit_for(
        monkeypatch,
        "04_r2",
        2,
        ValueError("need covariance of parameters for computing"),
    )

    logit, skips = _robustness(_logit_cohort())

    assert skips.to_dict("records") == [
        {
            "model": "Y",
            "animal": "chimi",
            "session_date": "20200212",
            "epoch": "04_r2",
            "reason": "ValueError: need covariance of parameters for computing",
        }
    ]
    # X fitted 04_r2, but the session leaves every logit rung so the rungs
    # stay on one common set -- a subset of the primary ladder's sessions.
    assert set(logit) == {"X", "Y"}
    for frame in logit.values():
        assert frame["epoch"].tolist() == ["02_r1"]


def test_an_unexpected_logit_error_still_propagates(monkeypatch):
    fail_logit_for(monkeypatch, "04_r2", 2, KeyError("a coding bug"))

    with pytest.raises(KeyError, match="a coding bug"):
        _robustness(_logit_cohort())


# ── The is_swr-contrast ladder ──

_BELIEF = "reward_belief_mean_candidate_arm_preoutcome_z"


def _audit(n_modeled: int, reason: str = "") -> dict:
    """A session audit in which every event entered was modeled (or none was)."""
    return {
        "n_in": 2,
        "n_drop_qvals_gap": 0,
        "n_drop_own_predictor_missing": 0,
        "n_drop_nonfinite": 0,
        "n_drop_zero_response": 0,
        "n_excluded_session": 2 - n_modeled,
        "n_modeled": n_modeled,
        "reason": reason,
        "dropped": pd.DataFrame(),
    }


def _nine_arm_cohort() -> pd.DataFrame:
    """Two events x nine arms; each spends a third of its mass on a trunk arm."""
    rows = []
    for index, event in enumerate(("a", "b")):
        counts = np.zeros(9)
        counts[index] = 6.0
        counts[index + 1] = 2.0
        counts[6] = 4.0
        for arm in range(9):
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": event,
                    "candidate_arm": arm,
                    "arm_fraction": counts[arm] / counts.sum(),
                    "is_swr": index == 0,
                    _BELIEF: float(arm),
                }
            )
    return pd.DataFrame(rows)


def _fit_row() -> dict:
    return {
        f"coef_{_BELIEF}": 0.1,
        f"bse_{_BELIEF}": 0.05,
        f"sd_{_BELIEF}": 1.0,
        f"pval_{_BELIEF}": 0.5,
        "separated_terms": "",
        "n_events": 2,
        "n_params": 1,
        "estimator": "conditional_poisson_fraction",
        "converged": True,
        "separated": False,
        "n_convergence_warnings": 0,
    }


def _stub_session_fit(monkeypatch, fit) -> list:
    """Replace the per-session fit by ``fit(g, n_categories)``; return what it saw."""
    seen = []

    def _fit(g, predictors, *, response_col, n_categories):
        seen.append((n_categories, g.copy()))
        return fit(g, n_categories)

    monkeypatch.setattr(session_fits, "fit_contrast_session", _fit)
    return seen


def test_contrast_fit_rows_carry_the_optimizer_diagnostics(monkeypatch):
    """The refit bundle records whether each fit reached its optimum, and which
    terms its session could not inform, so its gates can check them."""
    diagnostics = {
        "llf": -12.5,
        "newton_decrement": 3e-12,
        "max_abs_grad": 1e-7,
        "n_newton_steps": 1,
        "unidentified_terms": "is_candidate_arm_0:separated_to_minus_inf",
    }
    _stub_session_fit(
        monkeypatch, lambda g, n: ({**_fit_row(), **diagnostics}, _audit(2))
    )

    res, _audits = fit_contrast_ladder(
        _nine_arm_cohort(),
        {"C1_nine_arm": ([_BELIEF], "pooled")},
        response_col="arm_fraction",
        response="fraction",
        n_categories=9,
    )

    row = res.iloc[0]
    for column, value in diagnostics.items():
        assert row[column] == value
    assert row[f"coef_{_BELIEF}"] == pytest.approx(0.1)
    assert bool(row["converged"]) is True


@pytest.mark.parametrize(
    ("converged", "n_flagged"), [(True, 0), (False, 1), (None, 1), ("absent", 1)]
)
def test_a_contrast_fit_without_a_true_convergence_flag_counts_as_not_converged(
    monkeypatch, converged, n_flagged
):
    """A fit is converged only when its flag is True: a missing flag has not shown it."""

    def _fit(g, n):
        fit = _fit_row()
        if converged == "absent":
            del fit["converged"]
        else:
            fit["converged"] = converged
        return fit, _audit(2)

    _stub_session_fit(monkeypatch, _fit)

    _res, audits = fit_contrast_ladder(
        _nine_arm_cohort(),
        {"C1_nine_arm": ([_BELIEF], "pooled")},
        response_col="arm_fraction",
        response="fraction",
        n_categories=9,
    )

    assert [a["n_sessions_flagged_nonconverged"] for a in audits] == [n_flagged]


@pytest.mark.parametrize(
    ("separated", "n_flagged"), [(True, 1), (False, 0), (None, 0), ("absent", 0)]
)
def test_a_contrast_fit_is_separated_only_when_its_flag_is_true(
    monkeypatch, separated, n_flagged
):
    """A missing separation flag is not a separated fit (``fit_flag_counts``)."""

    def _fit(g, n):
        fit = _fit_row()
        if separated == "absent":
            del fit["separated"]
        else:
            fit["separated"] = separated
        return fit, _audit(2)

    _stub_session_fit(monkeypatch, _fit)

    _res, audits = fit_contrast_ladder(
        _nine_arm_cohort(),
        {"C1_nine_arm": ([_BELIEF], "pooled")},
        response_col="arm_fraction",
        response="fraction",
        n_categories=9,
    )

    assert [a["n_sessions_flagged_separated"] for a in audits] == [n_flagged]


def test_a_split_contrast_rung_is_fit_on_each_population(monkeypatch):
    """A split rung is fit on every event, the SWR events and the others; a
    pooled rung on every event only. Event "a" is the SWR event."""
    seen = _stub_session_fit(monkeypatch, lambda g, n: (_fit_row(), _audit(2)))

    res, audits = fit_contrast_ladder(
        _nine_arm_cohort(),
        {"C0_pooled": ([_BELIEF], "pooled"), "C1_split": ([_BELIEF], "split")},
        response_col="arm_fraction",
        response="fraction",
        n_categories=9,
    )

    events = [sorted(frame["event_id"].unique()) for _n, frame in seen]
    assert events == [["a", "b"], ["a", "b"], ["a"], ["b"]]
    assert list(zip(res["model"], res["population"], strict=True)) == [
        ("C0_pooled", "all"),
        ("C1_split", "all"),
        ("C1_split", "SWR"),
        ("C1_split", "nonSWR"),
    ]
    assert [(a["model"], a["population"]) for a in audits] == list(
        zip(res["model"], res["population"], strict=True)
    )


def test_leaf_only_contrast_rungs_are_fit_on_leaf_renormalized_fractions(monkeypatch):
    """The trunk share is not balanced across the contrast, so it cannot set the
    weight an event carries: leaf-only rungs get unit mass over the six leaves,
    while the nine-arm rungs in the same run keep the table's own fractions."""
    seen = _stub_session_fit(monkeypatch, lambda g, n: (_fit_row(), _audit(2)))
    ladder = {
        "C4_armvalue": ([_BELIEF], "pooled"),
        "C1_nine_arm": ([_BELIEF], "pooled"),
    }

    res, _audits = fit_contrast_ladder(
        _nine_arm_cohort(),
        ladder,
        response_col="arm_fraction",
        response="fraction",
        n_categories=9,
    )

    fitted = dict(seen)
    assert set(fitted) == {6, 9}
    leaf = fitted[6]
    assert sorted(leaf["candidate_arm"].unique()) == [0, 1, 2, 3, 4, 5]
    assert np.allclose(
        leaf.groupby("event_id")["arm_fraction"].sum().to_numpy(), 1.0, atol=1e-12
    )
    # Event "a" held 6/12 and 2/12 on leaves and 4/12 on a trunk.
    a_leaf = leaf[leaf["event_id"].eq("a")].set_index("candidate_arm")["arm_fraction"]
    assert a_leaf[0] == pytest.approx(0.75)
    assert a_leaf[1] == pytest.approx(0.25)

    nine = fitted[9]
    assert sorted(nine["candidate_arm"].unique()) == list(range(9))
    a_nine = nine[nine["event_id"].eq("a")].set_index("candidate_arm")["arm_fraction"]
    assert a_nine[0] == pytest.approx(0.5)
    assert a_nine[6] == pytest.approx(1 / 3)

    renormalized = res.set_index("model")["response_renormalized"]
    assert bool(renormalized["C4_armvalue"]) is True
    assert bool(renormalized["C1_nine_arm"]) is False


def test_leaf_only_contrast_count_rungs_keep_their_timebin_counts(monkeypatch):
    """A count response is already weighted by decoded timebins; leave it alone."""
    seen = _stub_session_fit(monkeypatch, lambda g, n: (None, _audit(0, "too_few")))
    cohort = _nine_arm_cohort()
    cohort["arm_count_hard"] = cohort["arm_fraction"] * 12.0

    res, audits = fit_contrast_ladder(
        cohort,
        {"C4_armvalue": ([_BELIEF], "pooled")},
        response_col="arm_count_hard",
        response="hard_count",
        n_categories=9,
    )

    (_n, leaf) = seen[0]
    expected = cohort[cohort["candidate_arm"] < 6]["arm_count_hard"].to_numpy()
    assert np.array_equal(leaf["arm_count_hard"].to_numpy(), expected)
    # The unfitted session is a session-level exclusion of both its events.
    assert res.empty
    assert audits[0]["n_excluded_session"] == 2
    assert audits[0]["n_modeled"] == 0


def test_a_contrast_session_drops_nan_events_not_rows(monkeypatch):
    """Missing arm covariates must drop an event, not shrink its arm set."""
    df = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 2, 2, 2],
            "arm_count_hard": [1, 0, 0, 0, 1, 0],
            "reward_fraction_candidate_arm_last10_visits": [
                np.nan,
                0.25,
                0.50,
                0.0,
                0.5,
                1.0,
            ],
        }
    )
    captured = {}

    def fake_fit(long_df, **kwargs):
        captured["long_df"] = long_df.copy()
        captured["kwargs"] = kwargs
        return {"n_events": long_df["event_id"].nunique()}

    monkeypatch.setattr(contrast_aggregation, "MIN_EVENTS", 1)
    monkeypatch.setattr(session_fits, "fit_conditional_poisson", fake_fit)

    out, audit = fit_contrast_session(
        df,
        ["reward_fraction_candidate_arm_last10_visits"],
        response_col="arm_count_hard",
        n_categories=3,
    )

    assert out == {"n_events": 1}
    assert captured["kwargs"]["n_categories"] == 3
    assert captured["long_df"]["event_id"].unique().tolist() == [2]
    assert len(captured["long_df"]) == 3
    # Event 1's NaN covariate is an own-predictor drop, not a qvals gap.
    assert audit["n_in"] == 2
    assert audit["n_drop_own_predictor_missing"] == 1
    assert audit["n_drop_qvals_gap"] == 0
    assert audit["n_modeled"] == 1
    drops = (
        audit["n_drop_qvals_gap"]
        + audit["n_drop_own_predictor_missing"]
        + audit["n_drop_zero_response"]
        + audit["n_excluded_session"]
    )
    assert audit["n_in"] - drops == audit["n_modeled"]


def test_a_contrast_session_missing_a_predictor_is_a_clean_skip(monkeypatch):
    """The fitter is never handed a design with absent columns: the session is
    an ``n_excluded_session`` skip that names the missing column."""
    df = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 2, 2, 2],
            "arm_count_hard": [1, 0, 0, 0, 1, 0],
        }
    )
    called = {"fit": False}

    def fake_fit(*args, **kwargs):
        called["fit"] = True
        return {"n_events": 0}

    monkeypatch.setattr(session_fits, "fit_conditional_poisson", fake_fit)

    out, audit = fit_contrast_session(
        df,
        ["reward_belief_mean_candidate_arm_preoutcome"],
        response_col="arm_count_hard",
        n_categories=9,
    )

    assert out is None
    assert called["fit"] is False
    assert audit["reason"].startswith("missing:")
    assert "reward_belief_mean_candidate_arm_preoutcome" in audit["reason"]
    assert audit["n_in"] == 2
    assert audit["n_excluded_session"] == 2
    assert audit["n_modeled"] == 0


def test_a_contrast_qvals_gap_drop_is_bucketed_separately(monkeypatch):
    """An event whose qvals-sourced covariate is NaN on a stop with no qvals row
    is a qvals-gap drop, not own-predictor missingness."""
    df = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 2, 2, 2],
            "has_qvals_row": [False, False, False, True, True, True],
            "arm_count_hard": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "reward_belief_mean_candidate_arm_preoutcome": [
                np.nan,
                0.2,
                0.3,
                0.1,
                0.4,
                0.5,
            ],
        }
    )
    monkeypatch.setattr(contrast_aggregation, "MIN_EVENTS", 1)
    monkeypatch.setattr(
        session_fits, "fit_conditional_poisson", lambda *a, **k: {"n_events": 1}
    )

    out, audit = fit_contrast_session(
        df,
        ["reward_belief_mean_candidate_arm_preoutcome"],
        response_col="arm_count_hard",
        n_categories=9,
    )

    assert out is not None
    assert audit["n_drop_qvals_gap"] == 1
    assert audit["n_drop_own_predictor_missing"] == 0
    assert audit["n_modeled"] == 1


def _leaf_session_with_a_trunk_only_event(n_events: int = 20) -> pd.DataFrame:
    """One session; event ``e0`` spends its whole allocation on the trunks."""
    rows = []
    for index in range(n_events):
        event = f"e{index}"
        current = index % 6
        counts = np.zeros(9)
        if index == 0:
            counts[6] = 8.0
        else:
            counts[current] = 6.0
            counts[(current + 1) % 6] = 2.0
        for arm in range(9):
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": event,
                    "candidate_arm": arm,
                    "arm_fraction": counts[arm] / counts.sum(),
                    "is_current_arm": float(arm == current),
                }
            )
    return pd.DataFrame(rows)


def test_a_contrast_event_left_without_leaf_mass_is_a_response_drop():
    """Its response is undefined, not its predictors; the audit must say so."""
    leaves, _renormalized = leaf_only_cohort(
        _leaf_session_with_a_trunk_only_event(),
        response_col="arm_fraction",
        response="fraction",
    )

    result, audit = fit_contrast_session(
        leaves, ["is_current_arm"], response_col="arm_fraction", n_categories=6
    )

    assert audit["n_drop_zero_response"] == 1
    assert audit["n_drop_own_predictor_missing"] == 0
    assert audit["n_drop_qvals_gap"] == 0
    assert audit["n_modeled"] == 19
    assert result is not None
    drops = (
        audit["n_drop_qvals_gap"]
        + audit["n_drop_own_predictor_missing"]
        + audit["n_drop_zero_response"]
        + audit["n_excluded_session"]
    )
    assert audit["n_in"] - drops == audit["n_modeled"]


def test_a_dm_contrast_run_reports_a_concentration_contrast_on_every_rung():
    """Every rung also answers "does an SWR event spread its mass more widely"."""
    cohort = pd.concat(
        [
            dm_contrast_session(epoch="02_r1", seed=0),
            dm_contrast_session(epoch="04_r2", seed=1),
        ],
        ignore_index=True,
    )
    ladder = {
        "C0_base": (["is_current_arm"], "pooled"),
        "C1_split": (["is_current_arm"], "split"),
    }

    res, audits = fit_contrast_ladder(
        cohort,
        ladder,
        response_col="arm_count_hard",
        response="hard_count",
        n_categories=9,
        estimator="dirichlet_multinomial",
        n_jobs=2,
        min_events_session_phi=40,
    )

    assert set(res["model"]) == {"C0_base", "C1_split"}
    assert res["estimator"].unique().tolist() == ["dirichlet_multinomial"]
    assert bool(res["llf_includes_constants"].all())
    assert res["phi_source"].unique().tolist() == ["session"]

    # The events behind the modifier travel with it: a contrast resting on one
    # or two SWR events is finite and unflagged, and nothing else would show it.
    pooled = res[res["population"] == "all"]
    assert (pooled["n_events_swr"] > 0).all()
    assert (
        pooled["n_events_swr"] + pooled["n_events_nonswr"] == pooled["n_events"]
    ).all()
    assert res.loc[res["population"] == "SWR", "n_events_swr"].isna().all()

    assert pooled["coef_log_phi_is_swr"].notna().all()
    assert pooled["bse_log_phi_is_swr"].notna().all()
    # Generated at log_phi_is_swr = -0.8: SWR events are the more concentrated.
    assert pooled["coef_log_phi_is_swr"].median() < 0.0

    # Within an SWR split the flag is constant, so there is no contrast to fit.
    split = res[res["population"] == "SWR"]
    assert len(split) == 2
    assert split["coef_log_phi_is_swr"].isna().all()

    assert {record["model"] for record in audits} == {"C0_base", "C1_split"}


class TestLengthAdjustedConcentration:
    """The length-adjusted concentration design is the twin of log_phi_is_swr."""

    def test_the_length_term_joins_the_design_only_where_is_swr_is_identified(self):
        cohort = with_event_length(dm_contrast_session(epoch="02_r1", seed=0), seed=0)

        assert resolve_concentration_cols(cohort, with_event_length=False) == (
            "is_swr",
        )
        assert resolve_concentration_cols(cohort, with_event_length=True) == (
            "is_swr",
            "log_n_nonlocal_bins_z",
        )
        # Inside an SWR split there is no concentration contrast to decompose.
        swr_only = cohort[cohort["is_swr"]]
        assert resolve_concentration_cols(swr_only, with_event_length=True) == ()

    def test_a_cohort_without_the_length_column_is_refused(self):
        """A silently dropped term would publish ``_dm_len`` fits that are not."""
        cohort = dm_contrast_session(epoch="02_r1", seed=0)
        with pytest.raises(ValueError, match="log_n_nonlocal_bins_z"):
            resolve_concentration_cols(cohort, with_event_length=True)

    def test_the_run_reports_both_concentration_terms(self):
        cohort = pd.concat(
            [
                with_event_length(dm_contrast_session(epoch="02_r1", seed=0), seed=0),
                with_event_length(dm_contrast_session(epoch="04_r2", seed=1), seed=1),
            ],
            ignore_index=True,
        )
        ladder = {"C0_base": (["is_current_arm"], "pooled")}
        kwargs = {
            "response_col": "arm_count_hard",
            "response": "hard_count",
            "n_categories": 9,
            "estimator": "dirichlet_multinomial",
            "n_jobs": 2,
            "min_events_session_phi": 40,
        }

        total, _ = fit_contrast_ladder(
            cohort, ladder, concentration_length=False, **kwargs
        )
        adjusted, _ = fit_contrast_ladder(
            cohort, ladder, concentration_length=True, **kwargs
        )

        assert "coef_log_phi_log_n_nonlocal_bins_z" not in total.columns
        assert adjusted["coef_log_phi_log_n_nonlocal_bins_z"].notna().all()
        assert adjusted["bse_log_phi_log_n_nonlocal_bins_z"].notna().all()
        # The effect-scale unit travels with the coefficient (the column is
        # already z-scored, so its SD is ~1).
        assert (adjusted["sd_log_phi_log_n_nonlocal_bins_z"] > 0).all()
        # One more free parameter than the unadjusted twin.
        assert (
            adjusted["n_params"].to_numpy() == total["n_params"].to_numpy() + 1
        ).all()


def test_a_refused_dm_contrast_session_is_named_in_the_audit():
    """A session the estimator refuses is named in the record, not just counted.

    Its events land in ``n_excluded_session`` like any unfitted session, so the
    audit's ``drop_notes`` is the only place a reader learns which session went
    missing from the rung and why.
    """
    good = dm_contrast_session(epoch="02_r1", seed=0)
    bad = dm_contrast_session(epoch="04_r2", seed=1).assign(is_prev_arm=0.0)
    cohort = pd.concat([good, bad], ignore_index=True)

    res, audits = fit_contrast_ladder(
        cohort,
        {"C0_base": (["is_current_arm", "is_prev_arm"], "pooled")},
        response_col="arm_count_hard",
        response="hard_count",
        n_categories=9,
        estimator="dirichlet_multinomial",
        n_jobs=2,
        min_events_session_phi=40,
    )

    assert res["epoch"].tolist() == ["02_r1"]
    (record,) = audits
    notes = record["drop_notes"]
    assert "no fit: dm_design" in notes
    assert "estimator skipped chimi/20200212/04_r2 (dm_design:" in notes
    # Counted as a session-level exclusion, not as dropped events.
    assert record["n_drop_own_predictor_missing"] == 0
    assert record["n_excluded_session"] == 60
    assert record["n_modeled"] == 60


def test_source_restricted_patch_models_drop_swr_interactions():
    assert tuple(patch_models_for_source("SWR")) == ("P0", "P1", "P2")
    assert tuple(patch_models_for_source("MUA")) == ("P0", "P1", "P2")
    # P3 is the swr_x_is_current_patch rung; under an event-source restriction
    # is_swr is constant, so it collapses onto the main effect.
    assert "P3" not in patch_models_for_source("SWR")
    assert "P3" in patch_models_for_source("non_local")


def test_source_restricted_switch_models_drop_swr_interactions():
    for source in ("SWR", "MUA"):
        assert tuple(patch_switch_models_for_source(source)) == ("PS0", "PS1", "PS2")
    assert "PS3" not in patch_switch_models_for_source("SWR")
    assert "PS3" in patch_switch_models_for_source("non_local")


def test_the_switch_block_sees_only_the_switch_events():
    """Events 1 and 3 follow a patch switch; 2 does not, 4 has no flag."""
    models = patch_switch_models_for_source("SWR")
    predictors = sorted({p for spec in models.values() for p in spec.predictors})
    patch = pd.DataFrame(
        [
            {
                "animal": "a",
                "session_date": "d",
                "epoch": "e",
                "event_id": event_id,
                "candidate_patch": patch,
                "patch_count_hard": float(patch == 1),
                **dict.fromkeys(predictors, 0.0),
            }
            for event_id in (1, 2, 3, 4)
            for patch in (1, 2, 3)
        ]
    )
    event_arm = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 3, 4],
            "did_last_completed_trip_switch_patch": [1, 0, 0, 1, np.nan],
        }
    )

    fits, provenance = fit_patch_switch(patch, event_arm, "SWR")

    (record,) = provenance.to_dict("records")
    # The first row of an event carries its flag.
    assert record["n_units_before"] == 2
    assert record["selection_filter"] == "did_last_completed_trip_switch_patch == 1"
    # Two events are fewer than a patch fit needs.
    assert MIN_PATCH_EVENTS > 2
    assert record["status"] == "too_few_units"
    assert all(frame.empty for frame in fits.values())


def test_leaf_only_fits_six_candidates_not_nine() -> None:
    assert fit_n_categories("arm", leaf_only=True) == 6
    assert fit_n_categories("arm", leaf_only=False) == 9
    assert fit_n_categories("patch", leaf_only=False) == 3


def test_leaf_only_ladder_fits_six_categories_on_the_rescaled_response() -> None:
    """On simulated data: six-category null, and the rows say so."""
    sim = simulated_session("arm_fraction").rename(columns={"arm_id": "candidate_arm"})
    long_df, renormalized = leaf_only_cohort(
        sim, response_col="arm_fraction", response="fraction"
    )
    complete = long_df.groupby("event_id")["arm_fraction"].transform(
        lambda values: values.notna().all()
    )
    long_df = long_df[complete].reset_index(drop=True)

    results, _skips = fit_content_ladder(
        long_df,
        {"X": ModelSpec("X", ("is_current_arm",), "current-arm bias", None)},
        ["X"],
        hierarchy_kind="arm",
        leaf_only=True,
        response_col="arm_fraction",
        event_col="event_id",
        response_renormalized=renormalized,
    )

    fitted = results["X"]
    assert fitted["response_renormalized"].tolist() == [True]
    n_events = int(fitted["n_events"].iloc[0])
    # Unit mass per event against a six-category uniform null.
    assert fitted["llf_null"].iloc[0] == pytest.approx(-n_events * np.log(6))
