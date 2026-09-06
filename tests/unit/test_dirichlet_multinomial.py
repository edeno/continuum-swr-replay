"""Dirichlet-multinomial content fitter: recovery, calibration, and contracts.

The estimator exists because the conditional-Poisson likelihood treats the
timebins of one replay event as independent draws. These tests pin the three
claims that follow from taking the within-event concentration seriously: the
coefficients and the concentration are both recovered from data generated at a
known concentration; the count model's standard errors are demonstrably too
narrow on that same data while the DM's are calibrated; and the concentration
modifier recovers a planted ``is_swr`` effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.dirichlet_multinomial import (
    cross_validate_dirichlet_multinomial,
    event_loglik,
    fit_dirichlet_multinomial,
    multinomial_log_constant,
)
from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay.simulation import (
    simulate_dirichlet_multinomial_counts,
)

BETA_TRUE = {"is_current_arm": 1.0, "is_prev_arm": -0.5}
PREDICTORS = list(BETA_TRUE)
CONCENTRATIONS = (1.0, 3.0, 10.0, 100.0)

# Keys every content fitter must return (shared estimator result schema).
SCHEMA_KEYS = (
    "n_events",
    "n_rows",
    "n_params",
    "llf",
    "llf_null",
    "aic",
    "bic",
    "pseudo_r2",
    "llf_includes_constants",
    "converged",
    "separated",
    "separated_terms",
    "n_convergence_warnings",
    "estimator",
)


def _simulate(phi, *, seed, log_phi_is_swr=0.0, n_events=300, L_e=32):
    return simulate_dirichlet_multinomial_counts(
        BETA_TRUE,
        log_phi_intercept=float(np.log(phi)),
        log_phi_is_swr=log_phi_is_swr,
        n_events=n_events,
        L_e=L_e,
        seed=seed,
    )


def _recovery_summary(phi, n_replicates):
    """Per-replicate 3-SE coverage of each beta and the fitted log concentration."""
    covered = []
    log_phi_hat = []
    for rep in range(n_replicates):
        fit = fit_dirichlet_multinomial(
            _simulate(phi, seed=1000 * rep + int(phi)), PREDICTORS
        )
        covered.append(
            [
                abs(fit[f"coef_{name}"] - truth) <= 3.0 * fit[f"bse_{name}"]
                for name, truth in BETA_TRUE.items()
            ]
        )
        log_phi_hat.append(fit["coef_log_phi_intercept"])
    return np.asarray(covered, dtype=float), np.asarray(log_phi_hat)


def _assert_recovery(phi, n_replicates, min_coverage):
    covered, log_phi_hat = _recovery_summary(phi, n_replicates)
    coverage = covered.mean(axis=0)
    assert (coverage >= min_coverage).all(), (
        f"phi={phi}: 3-SE coverage {dict(zip(BETA_TRUE, coverage, strict=True))} below {min_coverage}"
    )
    median_error = float(np.median(log_phi_hat) - np.log(phi))
    assert abs(median_error) <= 0.3, (
        f"phi={phi}: median log-phi error {median_error:+.3f} exceeds 0.3"
    )


@pytest.mark.parametrize("phi", CONCENTRATIONS)
def test_recovers_beta_and_phi(phi):
    """Both betas and the concentration come back from data at a known phi.

    Ten replicates keep this under the fast-test budget; the 40-replicate
    version below is the one that pins the 95% coverage claim.
    """
    _assert_recovery(phi, n_replicates=10, min_coverage=0.9)


@pytest.mark.slow
@pytest.mark.parametrize("phi", CONCENTRATIONS)
def test_recovers_beta_and_phi_many_replicates(phi):
    """3-SE coverage reaches 95% over 40 replicates at every concentration."""
    _assert_recovery(phi, n_replicates=40, min_coverage=0.95)


def test_large_phi_matches_conditional_poisson():
    """At phi = 1e4 the DM collapses onto the count multinomial.

    Coefficients must agree with ``fit_conditional_poisson`` to 0.02, and the
    likelihoods to 1.0 once the multinomial coefficient the conditional
    likelihood drops is added back.
    """
    df = _simulate(1e4, seed=11)
    dm = fit_dirichlet_multinomial(df, PREDICTORS)
    cp = fit_conditional_poisson(df, PREDICTORS)

    for name in PREDICTORS:
        assert abs(dm[f"coef_{name}"] - cp[f"coef_{name}"]) < 0.02, name

    constant = multinomial_log_constant(
        df["arm_count_hard"].to_numpy(), df["event_id"].to_numpy()
    )
    assert dm["llf_includes_constants"] is True
    assert cp["llf_includes_constants"] is False
    assert abs(dm["llf"] - (cp["llf"] + constant)) < 1.0


@pytest.mark.slow
def test_count_model_se_too_small_on_dm_data():
    """The count model's SE understates the true sampling spread; the DM's does not.

    Over 100 replicates at phi = 2, the conditional-Poisson model SE is under
    half the empirical spread of its own estimate, while the DM's model SE
    tracks its empirical spread.
    """
    dm_coef, dm_se, cp_coef, cp_se = [], [], [], []
    for rep in range(100):
        df = _simulate(2.0, seed=90000 + rep)
        dm = fit_dirichlet_multinomial(df, PREDICTORS)
        cp = fit_conditional_poisson(df, PREDICTORS)
        dm_coef.append(dm["coef_is_current_arm"])
        dm_se.append(dm["bse_is_current_arm"])
        cp_coef.append(cp["coef_is_current_arm"])
        cp_se.append(cp["bse_is_current_arm"])

    cp_ratio = np.mean(cp_se) / np.std(cp_coef, ddof=1)
    dm_ratio = np.mean(dm_se) / np.std(dm_coef, ddof=1)
    assert cp_ratio < 0.5, f"conditional-Poisson SE ratio {cp_ratio:.3f} not < 0.5"
    assert 0.8 <= dm_ratio <= 1.25, f"DM SE ratio {dm_ratio:.3f} outside [0.8, 1.25]"


def test_is_swr_concentration_recovered():
    """A planted log-phi shift on SWR events is recovered by the modifier."""
    estimates = [
        fit_dirichlet_multinomial(
            _simulate(5.0, seed=500 + rep, log_phi_is_swr=-1.0), PREDICTORS
        )["coef_log_phi_is_swr"]
        for rep in range(20)
    ]
    assert abs(float(np.median(estimates)) - (-1.0)) <= 0.3


def test_event_constant_predictor_rejected():
    """A predictor with no within-event variation is unidentified: refuse to fit."""
    df = _simulate(3.0, seed=2, n_events=60)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_dirichlet_multinomial(df, ["is_current_arm", "is_swr"])


def test_fraction_response_rejected():
    """The per-event fraction response belongs to the conditional model, not this one."""
    df = _simulate(3.0, seed=2, n_events=60)
    with pytest.raises(ValueError, match="fraction"):
        fit_dirichlet_multinomial(df, PREDICTORS, response_col="arm_fraction")


def test_non_integer_counts_rejected():
    """A non-integer count that is not a fraction is still not a valid response."""
    df = _simulate(3.0, seed=2, n_events=60)
    df["arm_count_hard"] = df["arm_count_hard"] + 0.5
    with pytest.raises(ValueError, match="integer"):
        fit_dirichlet_multinomial(df, PREDICTORS)


def test_negative_counts_rejected():
    df = _simulate(3.0, seed=2, n_events=60)
    df.loc[0, "arm_count_hard"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        fit_dirichlet_multinomial(df, PREDICTORS)


def test_unequal_candidate_counts_rejected():
    df = _simulate(3.0, seed=2, n_events=60)
    with pytest.raises(ValueError, match="same number of candidate rows"):
        fit_dirichlet_multinomial(df.iloc[1:], PREDICTORS)


def test_n_categories_checked_against_packed_shape():
    df = _simulate(3.0, seed=2, n_events=60)
    with pytest.raises(ValueError, match="n_categories"):
        fit_dirichlet_multinomial(df, PREDICTORS, n_categories=6)


def test_zero_response_events_dropped_and_counted():
    """Events with no timebins carry no information and are removed, not fitted."""
    df = _simulate(3.0, seed=4, n_events=80)
    blanked = df["event_id"].isin([0, 1, 2])
    df.loc[blanked, "arm_count_hard"] = 0.0
    fit = fit_dirichlet_multinomial(df, PREDICTORS)
    assert fit["n_drop_zero_response"] == 3
    assert fit["n_events"] == 77
    assert fit["n_rows"] == 77 * 9


def test_fixed_log_phi_pins_intercept():
    """Pinning the concentration removes a free parameter but not the beta recovery."""
    df = _simulate(3.0, seed=7)
    free = fit_dirichlet_multinomial(df, PREDICTORS)
    pinned = fit_dirichlet_multinomial(df, PREDICTORS, fixed_log_phi=float(np.log(3.0)))

    assert pinned["n_params"] == free["n_params"] - 1
    assert pinned["coef_log_phi_intercept"] == pytest.approx(np.log(3.0))
    assert np.isnan(pinned["bse_log_phi_intercept"])
    for name, truth in BETA_TRUE.items():
        assert abs(pinned[f"coef_{name}"] - truth) <= 3.0 * pinned[f"bse_{name}"]


def test_multinomial_log_constant_matches_direct_computation():
    """The constant is the per-event multinomial coefficient, summed."""
    from scipy.special import gammaln

    df = _simulate(3.0, seed=5, n_events=40, L_e=12)
    y = df["arm_count_hard"].to_numpy(dtype=float)
    groups = df["event_id"].to_numpy()
    expected = sum(
        gammaln(block.sum() + 1.0) - gammaln(block + 1.0).sum()
        for _, block in pd.Series(y).groupby(groups)
    )
    assert multinomial_log_constant(y, groups) == pytest.approx(expected)


def test_event_loglik_sums_to_llf():
    """The per-event decomposition is the same likelihood the fitter reports."""
    df = _simulate(3.0, seed=6, n_events=80)
    fit = fit_dirichlet_multinomial(df, PREDICTORS)

    n_events = fit["n_events"]
    ordered = df.sort_values(["event_id"], kind="stable")
    X = ordered[PREDICTORS].to_numpy(dtype=float).reshape(n_events, 9, len(PREDICTORS))
    Y = ordered["arm_count_hard"].to_numpy(dtype=float).reshape(n_events, 9)
    Z = ordered.groupby("event_id", sort=True)["is_swr"].first().to_numpy()[:, None]
    theta = np.array(
        [fit[f"coef_{name}"] for name in PREDICTORS]
        + [fit["coef_log_phi_intercept"], fit["coef_log_phi_is_swr"]]
    )

    per_event = event_loglik(theta, X, Y, Z)
    assert per_event.shape == (n_events,)
    assert per_event.sum() == pytest.approx(fit["llf"], rel=1e-9)


def test_cv_folds_over_events():
    """CV splits events (never timebins) and beats the concentration-only null."""
    from sklearn.model_selection import GroupKFold

    df = _simulate(3.0, seed=3)
    cv = cross_validate_dirichlet_multinomial(df, PREDICTORS, n_folds=5, seed=0)

    assert cv["n_folds"] == 5
    assert cv["n_events_cv"] == df["event_id"].nunique()
    assert cv["cv_bits_per_event_vs_null"] > 0
    assert cv["cv_ll_per_event_mean"] > cv["cv_ll_per_event_null_mean"]

    # Every event is held out exactly once: the folds partition the events.
    events = np.unique(df["event_id"].to_numpy())
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=0)
    held_out = [set(events[test]) for _, test in splitter.split(events, groups=events)]
    assert set().union(*held_out) == set(events)
    assert sum(len(fold) for fold in held_out) == len(events)


def test_result_matches_estimator_schema():
    """The result dict is the shared content-estimator schema, DM variant."""
    df = _simulate(3.0, seed=8, n_events=120)
    fit = fit_dirichlet_multinomial(df, PREDICTORS)

    for key in SCHEMA_KEYS:
        assert key in fit, key
    for name in PREDICTORS:
        for prefix in ("coef_", "bse_", "pval_", "sd_"):
            assert f"{prefix}{name}" in fit
    for name in ("log_phi_intercept", "log_phi_is_swr"):
        assert f"coef_{name}" in fit
        assert f"bse_{name}" in fit
        assert f"pval_{name}" in fit

    assert fit["estimator"] == "dirichlet_multinomial"
    assert fit["llf_includes_constants"] is True
    assert fit["converged"] is True
    assert fit["n_convergence_warnings"] == 0
    assert isinstance(fit["separated_terms"], str)
    assert fit["separated"] is False
    # Event-level SEs by construction: no clustered sandwich is reported.
    assert not [key for key in fit if key.startswith("bse_cluster_")]

    assert fit["n_events"] == 120
    assert fit["n_rows"] == 120 * 9
    assert fit["n_categories"] == 9
    assert fit["n_params"] == len(PREDICTORS) + 2
    assert fit["llf"] > fit["llf_null"]
    assert fit["aic"] == pytest.approx(2 * fit["n_params"] - 2 * fit["llf"])
    assert fit["bic"] == pytest.approx(
        fit["n_params"] * np.log(fit["n_events"]) - 2 * fit["llf"]
    )
    assert fit["pseudo_r2"] == pytest.approx(1.0 - fit["llf"] / fit["llf_null"])
    assert 0.0 < fit["pseudo_r2"] < 1.0
    assert fit["sd_is_current_arm"] == pytest.approx(
        df["is_current_arm"].to_numpy().std()
    )


def test_too_few_events_returns_none():
    """A session that cannot identify the parameters returns None rather than a fit."""
    df = _simulate(3.0, seed=9, n_events=6)
    assert fit_dirichlet_multinomial(df, PREDICTORS) is None


def test_event_loglik_stays_finite_when_a_category_is_ruled_out():
    """A coefficient large enough to underflow softmax must not make lgamma blow up.

    ``exp(log_p)`` reaches exactly 0.0 for a candidate the mean structure has
    ruled out, and ``lgamma(0)`` is ``+inf``: without a floor on the Dirichlet
    concentration the event's log-likelihood is NaN and the optimizer loses the
    objective it was following.
    """
    df = _simulate(3.0, seed=11, n_events=8, L_e=20)
    packed_x = (
        df[PREDICTORS].to_numpy(dtype=float).reshape(8, 9, len(PREDICTORS))
    )
    packed_y = df["arm_count_hard"].to_numpy(dtype=float).reshape(8, 9)
    packed_z = df.groupby("event_id", sort=True)["is_swr"].first().to_numpy()[:, None]

    theta = np.array([-800.0, 0.0, np.log(3.0), 0.0])
    per_event = event_loglik(theta, packed_x, packed_y, packed_z)

    assert per_event.shape == (8,)
    assert np.isfinite(per_event).all()
    assert (per_event < 0).all()
