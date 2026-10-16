"""Tests for analysis/replay/fitting.py.

Pins the behavior of the zero-total-count event filter: both
``fit_conditional_poisson`` and ``fit_conditional_logit`` must drop
events where every arm has zero count, because the conditional
likelihood (and the logit's dominant-arm logic) is degenerate there.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.base.model import LikelihoodModel
from statsmodels.discrete.conditional_models import ConditionalPoisson

from continuum_swr_replay.analysis.replay import fitting as fitting_mod
from continuum_swr_replay.analysis.replay.fitting import (
    event_clustered_covariance,
    fit_conditional_logit,
    fit_conditional_poisson,
    fit_null_model_conditional,
    fit_null_model_conditional_logit,
    within_group_full_rank,
)
from continuum_swr_replay.analysis.replay.simulation import (
    simulate_autocorrelated_counts,
    simulate_with_event_covariates,
)


def _make_long_df(event_arm_counts, predictor_values=None):
    """Build a long-format frame: one row per (event_id, arm).

    Parameters
    ----------
    event_arm_counts : list[list[int]]
        Outer list is events, inner is per-arm counts (fixed 3 arms for
        compactness). A row of all zeros is the zero-count event that
        both fitters must filter out.
    predictor_values : list[list[float]] | None
        Per-(event, arm) values for the single ``is_prev`` predictor.
        Defaults to 1 on arm 0, 0 elsewhere.
    """
    rows = []
    for ev, counts in enumerate(event_arm_counts):
        for arm, count in enumerate(counts):
            row = {
                "event_id": ev + 1,
                "candidate_arm": arm,
                "arm_count_hard": count,
                "is_prev_arm": 1.0 if arm == 0 else 0.0,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if predictor_values is not None:
        df["is_prev_arm"] = np.asarray(predictor_values).ravel()
    return df


def test_fit_conditional_logit_filters_zero_count_events():
    """fit_conditional_poisson drops events with total count == 0 at
    fitting.py:83-84. fit_conditional_logit must do the same, or
    zero-count events get an arbitrary 'dominant' arm via idxmax and
    enter the fit as spurious observations."""
    long_df = _make_long_df(
        event_arm_counts=[
            [0, 0, 0],  # event 1: zero total — must be filtered out
            [3, 1, 0],  # event 2
            [0, 2, 1],  # event 3
            [2, 0, 1],  # event 4
            [1, 1, 2],  # event 5
            [4, 1, 0],  # event 6
        ]
    )
    result = fit_conditional_logit(long_df, predictors=["is_prev_arm"])
    assert result is not None, "logit fit returned None on valid input"
    assert result["n_events"] == 5, (
        f"zero-count event not filtered: n_events={result['n_events']} "
        "(expected 5; event 1 had all-zero arm_count_hard)"
    )


def test_fit_conditional_poisson_filters_zero_count_events():
    """Same invariant for the Poisson fitter (sanity pin on existing
    behavior — the logit fix brings it in line with this function)."""
    long_df = _make_long_df(
        event_arm_counts=[
            [0, 0, 0],
            [3, 1, 0],
            [0, 2, 1],
            [2, 0, 1],
            [1, 1, 2],
            [4, 1, 0],
        ]
    )
    result = fit_conditional_poisson(long_df, predictors=["is_prev_arm"])
    assert result is not None
    assert result["n_events"] == 5


def test_null_model_filters_zero_count_events_like_non_null_model():
    """M0 must use the same non-degenerate event set as M1+.

    Otherwise M0/M1 comparisons see mismatched ``n_events`` and
    correctly invalidate every metric, even though the caller supplied
    the same input table to both fits.
    """
    long_df = _make_long_df(
        event_arm_counts=[
            [0, 0, 0],
            [3, 1, 0],
            [0, 2, 1],
            [2, 0, 1],
            [1, 1, 2],
            [4, 1, 0],
        ]
    )
    null = fit_conditional_poisson(long_df, predictors=[], n_categories=3)
    child = fit_conditional_poisson(long_df, predictors=["is_prev_arm"], n_categories=3)
    assert null["n_events"] == child["n_events"] == 5
    assert null["n_rows"] == 15


def test_fit_conditional_logit_ties_break_by_lowest_arm_id(monkeypatch):
    """Dominant-arm robustness fit should not depend on row order."""
    long_df = _make_long_df(
        event_arm_counts=[
            [2, 2, 0],
            [3, 1, 0],
            [0, 2, 1],
            [2, 0, 1],
            [1, 1, 2],
            [4, 1, 0],
        ],
        predictor_values=[
            [0.0, 1.0, 0.0],  # event 1 tie: arm 0 should be dominant
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
    )
    # Put tied arm 1 before tied arm 0; first-row idxmax would choose arm 1.
    event1 = long_df[long_df["event_id"] == 1].iloc[[1, 0, 2]]
    rest = long_df[long_df["event_id"] != 1]
    long_df = pd.concat([event1, rest], ignore_index=True)

    captured: dict[str, np.ndarray] = {}

    class FakeConditionalLogit:
        def __init__(self, y, X, groups):
            captured["y"] = np.asarray(y)
            captured["groups"] = np.asarray(groups)

        def fit(self, disp=0, **kwargs):
            # **kwargs so the stub tracks the real fit() signature. The fitters
            # pass maxiter (statsmodels' default of 100 leaves the larger rungs
            # short of the optimum); a stub that accepts only what it was written
            # against fails on the call rather than on the behaviour under test.
            captured["fit_kwargs"] = kwargs

            class Result:
                params = np.array([0.0])
                bse = np.array([1.0])
                pvalues = np.array([1.0])
                llf = -1.0

            return Result()

    monkeypatch.setattr(fitting_mod, "ConditionalLogit", FakeConditionalLogit)

    result = fit_conditional_logit(long_df, predictors=["is_prev_arm"])
    assert result is not None
    event1_y = captured["y"][captured["groups"] == 1]
    # Event 1 rows are ordered arm 1, arm 0, arm 2. Lowest-arm tie-break
    # should therefore mark the second row, not the first.
    np.testing.assert_array_equal(event1_y, np.array([0, 1, 0]))


def test_rank_deficient_design_is_rejected():
    """A perfectly collinear (duplicated) predictor must be rejected, not
    fit with finite-but-meaningless coefficients. Enough events so the
    n_events >= 2*n_predictors gate passes and the rank check is what
    triggers the None."""
    # 6 events, 3 arms; arm 0 carries the count so the dominant-arm logit
    # is well defined; all events non-zero count.
    long_df = _make_long_df([[3, 1, 1]] * 6)
    long_df["is_prev_dup"] = long_df["is_prev_arm"]  # exact duplicate -> rank 1 of 2

    preds = ["is_prev_arm", "is_prev_dup"]
    assert fit_conditional_poisson(long_df, predictors=preds) is None
    assert fit_conditional_logit(long_df, predictors=preds) is None

    # Sanity: the single non-duplicated predictor is full-rank and fits.
    assert fit_conditional_poisson(long_df, predictors=["is_prev_arm"]) is not None


def test_within_group_constant_predictor_is_rejected():
    """A predictor that varies across events but is CONSTANT within every
    event carries no within-group information for a conditional model;
    statsmodels would fit it with an exploding SE. The within-group rank
    check must reject it (a raw matrix_rank(X) would not, since the column
    is globally full-rank)."""
    long_df = _make_long_df([[3, 1, 1]] * 6)
    # "ev_level" is constant within each event (all 3 arms share the event
    # id) but varies across events -> full global rank, zero within-group.
    long_df["ev_level"] = long_df["event_id"].astype(float)

    assert fit_conditional_poisson(long_df, predictors=["ev_level"]) is None
    assert fit_conditional_logit(long_df, predictors=["ev_level"]) is None


# ── Collinearity diagnostics ──


def test_condition_number_is_scale_invariant_and_on_the_design_not_the_gram():
    """The reported number must be comparable with the conventional bands.

    Two variants are wrong here and both were live at some point: ``cond(X)``
    unscaled reports the columns' arbitrary relative units, and
    ``cond(X.T @ X)`` squares whatever that is. Only the column-scaled design
    condition number is comparable to the 10/30 bands -- and to the same
    column emitted by ``within_arm_horserace``.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))

    plain = fitting_mod._compute_conditional_diagnostics(X, ["a", "b", "c"])
    # Rescaling one column by 1000 changes its units, not its collinearity.
    rescaled = X.copy()
    rescaled[:, 1] *= 1000.0
    scaled = fitting_mod._compute_conditional_diagnostics(rescaled, ["a", "b", "c"])

    assert plain["cond_number"] == pytest.approx(scaled["cond_number"], rel=1e-6)
    # Near-orthogonal columns sit far below the warning band; the Gram-matrix
    # variant would report roughly this squared.
    assert plain["cond_number"] < 5.0


def test_collinear_design_warns_without_refusing_to_fit(caplog):
    """A structurally collinear rung is reported, not withheld."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(500, 1))
    # Second column is the first plus a whisker of noise: near-dependent.
    X = np.column_stack([base, base + rng.normal(scale=1e-3, size=(500, 1))])
    diagnostics = fitting_mod._compute_conditional_diagnostics(
        X, ["recency", "novelty"]
    )

    assert diagnostics["cond_number"] > fitting_mod.COND_NUMBER_WARN
    with caplog.at_level(logging.WARNING, logger=fitting_mod.__name__):
        fitting_mod._warn_on_collinear_design(diagnostics, ["recency", "novelty"])
    assert any("Collinear design" in r.message for r in caplog.records)
    assert any("condition number" in r.message for r in caplog.records)


def test_well_conditioned_design_is_silent(caplog):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    diagnostics = fitting_mod._compute_conditional_diagnostics(X, ["a", "b", "c"])
    with caplog.at_level(logging.WARNING, logger=fitting_mod.__name__):
        fitting_mod._warn_on_collinear_design(diagnostics, ["a", "b", "c"])
    assert not [r for r in caplog.records if "Collinear design" in r.message]


def test_zero_variance_column_does_not_divide_by_zero():
    """An arm indicator absent from a session is structurally zero."""
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=200), np.zeros(200)])
    diagnostics = fitting_mod._compute_conditional_diagnostics(X, ["a", "absent_arm"])
    assert np.isfinite(diagnostics["cond_number"]) or np.isinf(
        diagnostics["cond_number"]
    )


# ── Event-clustered covariance and the estimator result schema ──

CONTENT_BETA = {"is_current_arm": 1.2, "is_prev_arm": 0.6}
CONTENT_PREDICTORS = list(CONTENT_BETA)


def test_clustered_se_equals_model_se_when_each_event_is_a_single_draw():
    """With one timebin per event the two SEs estimate the same quantity.

    ``L_e = 1`` leaves nothing for the cluster correction to do -- each event
    contributes exactly one multinomial draw, which is what the model
    information already assumes -- so the sandwich must reproduce the model SE
    (agreement here is what licenses reading a *disagreement* on real,
    many-bin events as within-event mixing rather than as a coding error).
    """
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=500, L_e=1, seed=7)
    result = fit_conditional_poisson(long_df, CONTENT_PREDICTORS, n_categories=9)

    for term in CONTENT_PREDICTORS:
        model_se = result[f"bse_{term}"]
        cluster_se = result[f"bse_cluster_{term}"]
        assert abs(cluster_se - model_se) / model_se < 0.05


def test_clustered_covariance_matches_a_direct_per_event_computation():
    """Pins the sandwich against the definition, computed event by event."""
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=40, L_e=6, seed=11)
    X = long_df[CONTENT_PREDICTORS].to_numpy(dtype=float)
    y = long_df["arm_count_hard"].to_numpy(dtype=float)
    groups = long_df["event_id"].to_numpy()
    params = np.array([1.2, 0.6])

    bread = np.zeros((2, 2))
    meat = np.zeros((2, 2))
    for event in np.unique(groups):
        rows = groups == event
        x_e, y_e = X[rows], y[rows]
        prob = np.exp(x_e @ params)
        prob = prob / prob.sum()
        total = y_e.sum()
        xbar = prob @ x_e
        score = (y_e - total * prob) @ x_e
        bread += total * ((x_e.T * prob) @ x_e - np.outer(xbar, xbar))
        meat += np.outer(score, score)
    expected = np.linalg.inv(bread) @ meat @ np.linalg.inv(bread)

    covariance = event_clustered_covariance(X, y, groups, params)
    assert covariance == pytest.approx(expected, rel=1e-10)


def _with_event_fraction(long_df):
    """Convert ``arm_count_hard`` to the per-event allocation fraction."""
    out = long_df.copy()
    totals = out.groupby("event_id")["arm_count_hard"].transform("sum")
    out["arm_fraction"] = out["arm_count_hard"] / totals
    return out


@pytest.mark.slow
def test_clustered_se_tracks_the_sampling_sd_of_sticky_events():
    """Sticky within-event bins: the clustered SE, not the count SE, is honest.

    Each event's 20 timebins are a ρ=0.95 Markov chain over arms, so they carry
    far less information than 20 independent draws. Fitting the counts treats
    them as independent and reports an SE several times narrower than the
    coefficient's actual sampling SD across replicates; the event-clustered SE
    of the fraction fit recovers that sampling SD.
    """
    n_replicates = 200
    coefficients = []
    cluster_se = []
    count_se = []
    for replicate in range(n_replicates):
        long_df = _with_event_fraction(
            simulate_autocorrelated_counts(
                CONTENT_BETA,
                n_events=200,
                L_e=20,
                rho=0.95,
                seed=1000 + replicate,
            )
        )
        fraction_fit = fit_conditional_poisson(
            long_df, CONTENT_PREDICTORS, response_col="arm_fraction", n_categories=9
        )
        count_fit = fit_conditional_poisson(
            long_df, CONTENT_PREDICTORS, response_col="arm_count_hard", n_categories=9
        )
        coefficients.append([fraction_fit[f"coef_{t}"] for t in CONTENT_PREDICTORS])
        cluster_se.append(
            [fraction_fit[f"bse_cluster_{t}"] for t in CONTENT_PREDICTORS]
        )
        count_se.append([count_fit[f"bse_{t}"] for t in CONTENT_PREDICTORS])

    sampling_sd = np.std(np.array(coefficients), axis=0, ddof=1)
    typical_cluster_se = np.median(np.array(cluster_se), axis=0)
    typical_count_se = np.median(np.array(count_se), axis=0)

    assert np.all(np.abs(typical_cluster_se - sampling_sd) / sampling_sd < 0.20)
    assert np.all(sampling_sd / typical_count_se >= 2.0)


def test_fraction_response_null_is_one_draw_per_event():
    """Every event is a single draw, so ℓ₀ = -n_events · log(n_categories)."""
    long_df = _with_event_fraction(
        simulate_with_event_covariates(CONTENT_BETA, n_events=120, L_e=8, seed=5)
    )

    for n_categories in (9, 3):
        result = fit_conditional_poisson(
            long_df,
            CONTENT_PREDICTORS,
            response_col="arm_fraction",
            n_categories=n_categories,
        )
        assert result["n_events"] == 120
        assert result["llf_null"] == pytest.approx(-120 * np.log(n_categories))

        null = fit_null_model_conditional(
            long_df, response_col="arm_fraction", n_arms=n_categories
        )
        assert null["llf"] == pytest.approx(-120 * np.log(n_categories))


def test_estimator_names_the_response_scale_and_the_fitter():
    """``estimator`` distinguishes the three content fitters; none of their
    log-likelihoods carries the multinomial constant."""
    long_df = _with_event_fraction(
        simulate_with_event_covariates(CONTENT_BETA, n_events=150, L_e=8, seed=3)
    )

    count_fit = fit_conditional_poisson(
        long_df, CONTENT_PREDICTORS, response_col="arm_count_hard"
    )
    fraction_fit = fit_conditional_poisson(
        long_df, CONTENT_PREDICTORS, response_col="arm_fraction"
    )
    logit_fit = fit_conditional_logit(
        long_df, CONTENT_PREDICTORS, response_col="arm_count_hard"
    )

    assert count_fit["estimator"] == "conditional_poisson_count"
    assert fraction_fit["estimator"] == "conditional_poisson_fraction"
    assert logit_fit["estimator"] == "conditional_logit_dominant"
    for fit in (count_fit, fraction_fit, logit_fit):
        assert fit["llf_includes_constants"] is False


def test_null_models_carry_the_estimator_of_the_ladder_they_null():
    """A null rung is compared against its ladder, so it must be labeled like it."""
    long_df = _with_event_fraction(
        simulate_with_event_covariates(CONTENT_BETA, n_events=60, L_e=4, seed=9)
    )

    assert (
        fit_null_model_conditional(long_df)["estimator"] == "conditional_poisson_count"
    )
    assert (
        fit_null_model_conditional(
            long_df,
            response_col="arm_fraction",
            estimator="conditional_poisson_fraction",
        )["estimator"]
        == "conditional_poisson_fraction"
    )
    assert (
        fit_null_model_conditional_logit(long_df)["estimator"]
        == "conditional_logit_dominant"
    )
    # An empty predictor list routes through the null, which must still report
    # the response scale the caller asked for.
    empty_fit = fit_conditional_poisson(long_df, [], response_col="arm_fraction")
    assert empty_fit["estimator"] == "conditional_poisson_fraction"
    assert empty_fit["llf_includes_constants"] is False


# ── Reaching the optimum: within-event scaling, a Newton finish, analytic SEs ──

TINY_PREDICTORS = ["a", "r", "tiny"]


def _session_with_a_tiny_within_event_column(n_events=300, seed=2):
    """One session whose ``tiny`` column has a within-event SD of about 1e-3.

    ``tiny`` tracks ``r`` (correlation 0.9) and carries an ordinary effect
    (coefficient 500, so 0.5 log units per unit of ``1e-3 * r``), like the last
    recency-spline basis column in the content rungs. Its gradient is about
    1e-3 of the other columns', so BFGS on the raw columns leaves its
    coefficient near the start value. The response is the per-event fraction,
    as in the shipped content fits.
    """
    rng = np.random.default_rng(seed)
    n_arms = 9
    a = rng.normal(size=(n_events, n_arms))
    r = rng.normal(size=(n_events, n_arms))
    tiny = 1e-3 * (0.9 * r + np.sqrt(1 - 0.9**2) * rng.normal(size=(n_events, n_arms)))
    design = np.stack([a, r, tiny], axis=-1)
    eta = design @ np.array([0.8, -0.5, 500.0])
    prob = np.exp(eta - eta.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    counts = np.array([rng.multinomial(10, p) for p in prob])
    frame = pd.DataFrame(design.reshape(-1, 3), columns=TINY_PREDICTORS)
    frame["event_id"] = np.repeat(np.arange(n_events), n_arms)
    frame["arm_fraction"] = (counts / counts.sum(axis=1, keepdims=True)).reshape(-1)
    return frame


def _arrays(long_df, predictors, response_col):
    return (
        long_df[predictors].to_numpy(dtype=float),
        long_df[response_col].to_numpy(dtype=float),
        long_df["event_id"].to_numpy(),
    )


def _loglike_by_event(X, y, groups, beta):
    """Log-likelihood, gradient and information, one event at a time.

    Written independently of the module under test, from the definition
    ``l_e = sum_a y_ea eta_ea - n_e log sum_a exp(eta_ea)``.
    """
    llf = 0.0
    grad = np.zeros(beta.size)
    info = np.zeros((beta.size, beta.size))
    for event in np.unique(groups):
        rows = groups == event
        x_e, y_e = X[rows], y[rows]
        eta = x_e @ beta
        top = eta.max()
        weight = np.exp(eta - top)
        prob = weight / weight.sum()
        total = y_e.sum()
        llf += y_e @ eta - total * (top + np.log(weight.sum()))
        grad += (y_e - total * prob) @ x_e
        xbar = prob @ x_e
        info += total * ((x_e.T * prob) @ x_e - np.outer(xbar, xbar))
    return llf, grad, info


def _reference_optimum(X, y, groups, start):
    """The conditional-Poisson optimum by scipy's trust-region Newton."""
    result = minimize(
        lambda beta: tuple(-v for v in _loglike_by_event(X, y, groups, beta)[:2]),
        start,
        jac=True,
        hess=lambda beta: _loglike_by_event(X, y, groups, beta)[2],
        method="trust-exact",
        options={"gtol": 1e-10},
    )
    return -result.fun, result.x


def test_a_tiny_within_event_column_reaches_the_optimum():
    """F3: BFGS on raw columns stopped up to 4.8 log-likelihood units short."""
    long_df = _session_with_a_tiny_within_event_column()
    X, y, groups = _arrays(long_df, TINY_PREDICTORS, "arm_fraction")
    ref_llf, ref_beta = _reference_optimum(X, y, groups, np.array([0.8, -0.5, 500.0]))

    fit = fit_conditional_poisson(long_df, TINY_PREDICTORS, response_col="arm_fraction")

    assert fit["llf"] == pytest.approx(ref_llf, abs=1e-8)
    coef = np.array([fit[f"coef_{t}"] for t in TINY_PREDICTORS])
    assert coef == pytest.approx(ref_beta, rel=1e-5)
    # The case is a hard one: statsmodels' BFGS on the raw columns, which is
    # what this fitter ran before, stops far short of the same optimum.
    old = ConditionalPoisson(y, X, groups=groups).fit(
        disp=0, maxiter=fitting_mod.CONDITIONAL_FIT_MAXITER
    )
    assert old.llf < ref_llf - 1.0


def test_rescaling_a_column_is_a_pure_reparametrisation():
    """x1000 on one column divides its coefficient and both SEs by 1000 exactly.

    The fitter divides each column by its within-event SD, so the optimizer sees
    the same scaled problem either way and takes the same path. Nothing else
    may change: not the other terms, their SEs, the p-values or the llf.
    """
    long_df = _session_with_a_tiny_within_event_column()
    base = fit_conditional_poisson(
        long_df, TINY_PREDICTORS, response_col="arm_fraction"
    )
    rescaled = fit_conditional_poisson(
        long_df.assign(tiny=long_df["tiny"] * 1000.0),
        TINY_PREDICTORS,
        response_col="arm_fraction",
    )

    for kind in ("coef", "bse", "bse_cluster"):
        assert 1000.0 * rescaled[f"{kind}_tiny"] == pytest.approx(
            base[f"{kind}_tiny"], rel=1e-8
        )
        for term in ("a", "r"):
            assert rescaled[f"{kind}_{term}"] == pytest.approx(
                base[f"{kind}_{term}"], rel=1e-8
            )
    for term in TINY_PREDICTORS:
        assert rescaled[f"pval_{term}"] == pytest.approx(base[f"pval_{term}"], rel=1e-8)
    assert rescaled["llf"] == pytest.approx(base["llf"], abs=1e-8)


def test_model_se_is_the_inverse_of_the_analytic_information():
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=300, L_e=10, seed=4)
    X, y, groups = _arrays(long_df, CONTENT_PREDICTORS, "arm_count_hard")

    fit = fit_conditional_poisson(long_df, CONTENT_PREDICTORS)

    beta = np.array([fit[f"coef_{t}"] for t in CONTENT_PREDICTORS])
    bse = np.array([fit[f"bse_{t}"] for t in CONTENT_PREDICTORS])
    _, _, info = _loglike_by_event(X, y, groups, beta)
    assert bse == pytest.approx(np.sqrt(np.diag(np.linalg.inv(info))), rel=1e-9)
    # statsmodels' numerical Hessian at the same coefficients, as shipped before.
    numeric_hessian = ConditionalPoisson(y, X, groups=groups).hessian(beta)
    numeric_bse = np.sqrt(np.diag(np.linalg.inv(-numeric_hessian)))
    assert bse == pytest.approx(numeric_bse, rel=1e-4)
    pval = np.array([fit[f"pval_{t}"] for t in CONTENT_PREDICTORS])
    assert pval == pytest.approx(2 * norm.sf(np.abs(beta / bse)), rel=1e-12)


def test_a_fit_stopped_before_the_optimum_is_flagged_not_converged(monkeypatch, caplog):
    long_df = _session_with_a_tiny_within_event_column()
    full = fit_conditional_poisson(
        long_df, TINY_PREDICTORS, response_col="arm_fraction"
    )
    assert full["converged"] is True
    assert full["unidentified_terms"] == ""
    assert full["newton_decrement"] < fitting_mod.NEWTON_DECREMENT_TOL
    assert full["n_newton_steps"] >= 1
    assert full["max_abs_grad"] < 1e-3

    monkeypatch.setattr(fitting_mod, "NEWTON_MAX_STEPS", 0)
    with caplog.at_level(logging.WARNING, logger=fitting_mod.__name__):
        stopped = fit_conditional_poisson(
            long_df, TINY_PREDICTORS, response_col="arm_fraction"
        )

    assert stopped["n_newton_steps"] == 0
    assert stopped["newton_decrement"] >= fitting_mod.NEWTON_DECREMENT_TOL
    assert stopped["converged"] is False
    assert stopped["n_convergence_warnings"] == 1
    assert any("not converged" in r.getMessage() for r in caplog.records)
    # It keeps its numbers: a fit short of the optimum, not a refused one.
    assert stopped["llf"] < full["llf"]


def test_a_column_constant_within_every_event_is_reported_unidentified(caplog):
    """Such a column cancels exactly in the conditional likelihood.

    So the session cannot inform it, and dropping it changes nothing: the fit is
    kept, the column is reported as not estimated, and every other number is
    bit-identical to a fit without it.
    """
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=150, L_e=10, seed=2)
    long_df["ev_level"] = long_df["event_id"].astype(float)

    with caplog.at_level(logging.WARNING, logger=fitting_mod.__name__):
        fit = fit_conditional_poisson(long_df, [*CONTENT_PREDICTORS, "ev_level"])

    assert fit["unidentified_terms"] == "ev_level:no_within_event_variation"
    assert np.isnan(fit["coef_ev_level"]) and np.isnan(fit["pval_ev_level"])
    assert fit["bse_ev_level"] == np.inf and fit["bse_cluster_ev_level"] == np.inf
    without = fit_conditional_poisson(long_df, CONTENT_PREDICTORS)
    for term in CONTENT_PREDICTORS:
        for kind in ("coef", "bse", "bse_cluster", "pval"):
            assert fit[f"{kind}_{term}"] == without[f"{kind}_{term}"]
    assert fit["llf"] == without["llf"]
    named = [r.getMessage() for r in caplog.records if "ev_level" in r.getMessage()]
    assert named
    # Only the column is named, not the whole design.
    assert not any(term in named[0] for term in CONTENT_PREDICTORS)


def _with_arm_never_reached(arm, seed=0):
    """A session in which no event puts any mass on ``arm``."""
    long_df = simulate_with_event_covariates(
        CONTENT_BETA, n_events=200, L_e=10, seed=seed
    )
    long_df.loc[long_df["arm_id"] == arm, "arm_count_hard"] = 0
    return long_df


def test_an_arm_no_event_reaches_is_kept_and_reported_unidentified(caplog):
    """The session cannot inform the arm's dummy, but it informs every other term.

    The dummy's coefficient runs to -inf (F3: 62 real fits, each on a nuisance
    term like this), and in that limit the arm has probability 0. So the other
    terms are those of the fit without the arm's rows, and the dummy is reported
    as not estimated: NaN coefficient and p-value, infinite SEs.
    """
    long_df = _with_arm_never_reached(3)
    predictors = [*CONTENT_PREDICTORS, "is_candidate_arm_3"]
    X, _, groups = _arrays(long_df, predictors, "arm_count_hard")
    assert within_group_full_rank(X, groups)

    with caplog.at_level(logging.WARNING, logger=fitting_mod.__name__):
        fit = fit_conditional_poisson(long_df, predictors)

    assert fit is not None
    assert fit["unidentified_terms"] == "is_candidate_arm_3:separated_to_minus_inf"
    assert np.isnan(fit["coef_is_candidate_arm_3"])
    assert np.isnan(fit["pval_is_candidate_arm_3"])
    assert fit["bse_is_candidate_arm_3"] == np.inf
    assert fit["bse_cluster_is_candidate_arm_3"] == np.inf
    assert "is_candidate_arm_3" not in fit["separated_terms"].split("|")
    assert fit["converged"] is True
    assert any("is_candidate_arm_3" in r.getMessage() for r in caplog.records)

    without_arm = fit_conditional_poisson(
        long_df[long_df["arm_id"] != 3], CONTENT_PREDICTORS
    )
    for term in CONTENT_PREDICTORS:
        for kind in ("coef", "bse", "bse_cluster", "pval"):
            assert fit[f"{kind}_{term}"] == pytest.approx(
                without_arm[f"{kind}_{term}"], rel=1e-8
            )
    assert fit["llf"] == pytest.approx(without_arm["llf"], abs=1e-8)
    # Dropping only the column is a different model: it gives arm 3 a
    # probability that its zero mass contradicts, and moves the other terms.
    column_only = fit_conditional_poisson(long_df, CONTENT_PREDICTORS)
    assert abs(column_only["coef_is_current_arm"] - fit["coef_is_current_arm"]) > 1e-3


def test_a_term_left_without_variation_by_the_removed_rows_is_unidentified_too():
    """No event reaches the previous arm, so ``is_prev_arm`` runs to -inf.

    Its product with an event-level covariate is then zero on every row left,
    so the session cannot inform that term either.
    """
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=200, L_e=10, seed=1)
    long_df.loc[long_df["is_prev_arm"] == 1, "arm_count_hard"] = 0
    event_z = np.random.default_rng(1).normal(size=long_df["event_id"].nunique())
    long_df["z_x_is_prev_arm"] = (
        event_z[long_df["event_id"].to_numpy()] * long_df["is_prev_arm"]
    )
    predictors = ["is_current_arm", "is_prev_arm", "z_x_is_prev_arm"]

    fit = fit_conditional_poisson(long_df, predictors)

    # In the order the fit took them; the product is constant on the rows left.
    assert fit["unidentified_terms"] == (
        "is_prev_arm:separated_to_minus_inf|z_x_is_prev_arm:no_within_event_variation"
    )
    without_prev = fit_conditional_poisson(
        long_df[long_df["is_prev_arm"] == 0], ["is_current_arm"]
    )
    assert fit["coef_is_current_arm"] == pytest.approx(
        without_prev["coef_is_current_arm"], rel=1e-8
    )
    assert fit["llf"] == pytest.approx(without_prev["llf"], abs=1e-8)


def test_a_direction_mixing_several_columns_is_refused_and_named(caplog):
    """No event reaches arm 8, the reference of the arm dummies.

    The unbounded direction then moves all eight dummies together, so no single
    column carries it, and the design is refused.
    """
    long_df = _with_arm_never_reached(8)
    dummies = [f"is_candidate_arm_{arm}" for arm in range(8)]

    with caplog.at_level(logging.WARNING, logger=fitting_mod.__name__):
        assert fit_conditional_poisson(long_df, [*CONTENT_PREDICTORS, *dummies]) is None

    refusals = [
        r.getMessage() for r in caplog.records if "is_candidate_arm_0" in r.getMessage()
    ]
    assert refusals
    assert all(dummy in refusals[0] for dummy in dummies)


def test_a_well_scaled_fit_is_unchanged():
    """Where the old fit was already at the optimum, the numbers do not move.

    The old fitter's BFGS stopped on a per-row gradient of 1e-5, so on this
    design its coefficients sit about 5e-6 from the optimum (llf 1e-9 to 2e-8
    short): 1e-6 against it is not reachable. Allowed to finish, the same
    optimizer agrees to 1e-6.
    """
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=300, L_e=10, seed=4)
    X, y, groups = _arrays(long_df, CONTENT_PREDICTORS, "arm_count_hard")

    fit = fit_conditional_poisson(long_df, CONTENT_PREDICTORS)
    coef = np.array([fit[f"coef_{t}"] for t in CONTENT_PREDICTORS])

    model = ConditionalPoisson(y, X, groups=groups)
    old = model.fit(disp=0, maxiter=fitting_mod.CONDITIONAL_FIT_MAXITER)
    assert coef == pytest.approx(np.asarray(old.params), abs=2e-5)
    assert 0.0 <= fit["llf"] - old.llf < 1e-7
    finished = LikelihoodModel.fit(
        model, method="bfgs", gtol=1e-8, maxiter=1000, disp=0, skip_hessian=True
    )
    assert coef == pytest.approx(np.asarray(finished.params), abs=1e-6)


# ── Prediction from a fit with unidentified terms ──


def _fit_with_every_kind_of_unidentified_term():
    """No event reaches arm 3; ``ev_level`` is constant within every event; and
    ``holds_mass`` marks exactly the rows with mass, so it runs to +inf."""
    long_df = _with_arm_never_reached(3, seed=4)
    long_df["ev_level"] = long_df["event_id"].astype(float)
    long_df["holds_mass"] = (long_df["arm_count_hard"] > 0).astype(float)
    predictors = [*CONTENT_PREDICTORS, "is_candidate_arm_3", "ev_level"]
    return long_df, predictors


def test_the_linear_predictor_takes_unidentified_terms_to_their_limits():
    long_df, predictors = _fit_with_every_kind_of_unidentified_term()
    fit = fit_conditional_poisson(long_df, predictors)
    assert fit["unidentified_terms"] == (
        "ev_level:no_within_event_variation|is_candidate_arm_3:separated_to_minus_inf"
    )
    X = long_df[predictors].to_numpy(dtype=float)
    groups = long_df["event_id"].to_numpy()

    eta = fitting_mod.conditional_linear_predictor(X, groups, fit, predictors)

    arm_3 = long_df["arm_id"].to_numpy() == 3
    assert np.all(eta[arm_3] == -np.inf)
    beta = np.array([fit[f"coef_{t}"] for t in CONTENT_PREDICTORS])
    expected = long_df[CONTENT_PREDICTORS].to_numpy(dtype=float) @ beta
    np.testing.assert_array_equal(eta[~arm_3], expected[~arm_3])


def test_the_linear_predictor_rules_out_the_rows_a_plus_inf_term_moves_away_from():
    long_df, _ = _fit_with_every_kind_of_unidentified_term()
    predictors = [*CONTENT_PREDICTORS, "holds_mass"]
    fit = fit_conditional_poisson(long_df, predictors)
    assert fit["unidentified_terms"] == "holds_mass:separated_to_plus_inf"
    X = long_df[predictors].to_numpy(dtype=float)

    eta = fitting_mod.conditional_linear_predictor(
        X, long_df["event_id"].to_numpy(), fit, predictors
    )

    no_mass = long_df["holds_mass"].to_numpy() == 0
    assert np.all(eta[no_mass] == -np.inf)
    assert np.all(np.isfinite(eta[~no_mass]))


def test_the_linear_predictor_refuses_a_missing_coefficient_it_cannot_explain():
    long_df, _ = _fit_with_every_kind_of_unidentified_term()
    fit = fit_conditional_poisson(long_df, CONTENT_PREDICTORS)
    fit["coef_is_prev_arm"] = np.nan
    X = long_df[CONTENT_PREDICTORS].to_numpy(dtype=float)

    with pytest.raises(ValueError, match="is_prev_arm"):
        fitting_mod.conditional_linear_predictor(
            X, long_df["event_id"].to_numpy(), fit, CONTENT_PREDICTORS
        )


def test_pearson_residuals_take_a_fits_unidentified_terms_to_their_limits():
    from continuum_swr_replay.analysis.replay.model_checking import (
        compute_pearson_residuals,
    )

    long_df, predictors = _fit_with_every_kind_of_unidentified_term()
    fit = fit_conditional_poisson(long_df, predictors)

    out = compute_pearson_residuals(long_df, fit, predictors)

    arm_3 = out["arm_id"] == 3
    assert (out.loc[arm_3, "predicted_frac"] == 0).all()
    assert (out.loc[arm_3, "pearson_resid"] == 0).all()
    assert np.isfinite(out["pearson_resid"]).all()
    np.testing.assert_allclose(
        out.groupby("event_id")["predicted_frac"].sum().to_numpy(), 1.0, atol=1e-12
    )


# ── Review round: identifiability on the likelihood's rows, and untested parts ──


def test_a_column_that_varies_only_in_events_the_likelihood_leaves_out_is_inert():
    """An event whose response is the same on every row is left out of the
    likelihood. A column that varies only inside such an event is constant on
    every row the likelihood uses, so it is ``no_within_event_variation``, not a
    separated term, and prediction rules out no row (the review's probe 3F: the
    old code ruled out a row that has mass)."""
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=150, L_e=10, seed=3)
    first = long_df["event_id"].iloc[0]
    long_df.loc[long_df["event_id"] == first, "arm_count_hard"] = 1
    long_df["marker"] = 0.0
    long_df.loc[(long_df["event_id"] == first) & (long_df["arm_id"] == 2), "marker"] = (
        1.0
    )
    predictors = [*CONTENT_PREDICTORS, "marker"]

    fit = fit_conditional_poisson(long_df, predictors)

    assert fit["unidentified_terms"] == "marker:no_within_event_variation"
    X, _, groups = _arrays(long_df, predictors, "arm_count_hard")
    eta = fitting_mod.conditional_linear_predictor(X, groups, fit, predictors)
    assert np.isfinite(eta).all()


def test_the_information_test_reads_each_column_per_within_event_sd():
    """A column with a large spread between events and a small one within them
    is identified: per within-event SD its information is ordinary. Per global
    SD it would read 1e4 times smaller, as if the data could not bound it."""
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=300, L_e=10, seed=6)
    rng = np.random.default_rng(6)
    between = rng.normal(scale=1000.0, size=long_df["event_id"].nunique())
    long_df["spread"] = between[long_df["event_id"].to_numpy()] + rng.normal(
        scale=0.1, size=len(long_df)
    )

    fit = fit_conditional_poisson(long_df, [*CONTENT_PREDICTORS, "spread"])

    assert fit is not None
    assert fit["unidentified_terms"] == ""
    assert np.isfinite(fit["coef_spread"])
    assert fit["converged"] is True


def test_the_newton_finish_from_a_far_start_never_lowers_the_log_likelihood():
    """From (6, 3) the full Newton step overshoots: only the step-halving keeps
    the log-likelihood from falling, and the finish still reaches the optimum."""
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=200, L_e=10, seed=1)
    X, y, groups = _arrays(long_df, CONTENT_PREDICTORS, "arm_count_hard")
    X_sorted, y_sorted, starts = fitting_mod._sorted_by_event(X, y, groups)
    ref_llf, _ = _reference_optimum(X, y, groups, np.array([1.2, 0.6]))
    start = np.array([6.0, 3.0])
    start_llf = fitting_mod._event_terms(X_sorted, y_sorted, starts, start)[0]

    finish = fitting_mod._newton_finish(X_sorted, y_sorted, starts, start)

    assert finish.llf > start_llf
    assert finish.llf == pytest.approx(ref_llf, abs=1e-8)
    assert finish.decrement < fitting_mod.NEWTON_DECREMENT_TOL


def test_the_newton_finish_from_a_saturated_start_goes_on_to_the_optimum():
    """At (25, -30) the probabilities saturate and the information is tiny along
    both columns. That is the start, not the optimum: the finish must take steps
    rather than read it as a term the data cannot bound (the review's probe 2D)."""
    long_df = simulate_with_event_covariates(CONTENT_BETA, n_events=200, L_e=10, seed=1)
    X, y, groups = _arrays(long_df, CONTENT_PREDICTORS, "arm_count_hard")
    X_sorted, y_sorted, starts = fitting_mod._sorted_by_event(X, y, groups)
    ref_llf, _ = _reference_optimum(X, y, groups, np.array([1.2, 0.6]))

    finish = fitting_mod._newton_finish(
        X_sorted, y_sorted, starts, np.array([25.0, -30.0])
    )

    assert finish.n_steps > 0
    assert finish.llf == pytest.approx(ref_llf, abs=1e-8)
    assert finish.eigenvalues[0] >= fitting_mod.MIN_SCALED_INFORMATION


def test_max_abs_grad_is_the_gradient_on_the_original_scale():
    """Read where the gradient is not yet ~0: the fit stopped after BFGS."""
    long_df = _session_with_a_tiny_within_event_column()
    X, y, groups = _arrays(long_df, TINY_PREDICTORS, "arm_fraction")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fitting_mod, "NEWTON_MAX_STEPS", 0)
        fit = fit_conditional_poisson(
            long_df, TINY_PREDICTORS, response_col="arm_fraction"
        )
    beta = np.array([fit[f"coef_{t}"] for t in TINY_PREDICTORS])

    _, gradient, _ = _loglike_by_event(X, y, groups, beta)

    assert fit["max_abs_grad"] == pytest.approx(np.max(np.abs(gradient)), rel=1e-6)


def test_the_linear_predictor_applies_separated_limits_in_sequence():
    """Each limit acts on the rows the earlier ones left.

    In each event, ``a`` rules out its row first; ``b`` then rules out the rows
    above its minimum over the rows left (b = 2), keeping b = 1. Read over all
    rows, ``b``'s minimum would be the ruled-out row's 0, and it would rule out
    every row.
    """
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]] * 2)
    groups = np.repeat([1, 2], 3)
    fit = {
        "coef_a": np.nan,
        "coef_b": np.nan,
        "unidentified_terms": "a:separated_to_minus_inf|b:separated_to_minus_inf",
    }

    eta = fitting_mod.conditional_linear_predictor(X, groups, fit, ["a", "b"])

    np.testing.assert_array_equal(eta, [-np.inf, 0.0, -np.inf] * 2)
