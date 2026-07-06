"""Fit-diagnostics helpers flag convergence/separation without dropping fits."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalPoisson
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from continuum_swr_replay.analysis.replay._fit_diagnostics import (
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_status,
)


def _conditional_poisson_design(seed: int = 0):
    """A well-posed conditional-Poisson design: one chosen arm per event group."""
    rng = np.random.default_rng(seed)
    n_events, n_arms = 60, 4
    groups = np.repeat(np.arange(n_events), n_arms)
    x = rng.normal(size=n_events * n_arms)
    eta = 0.8 * x
    y = np.zeros_like(eta)
    for e in range(n_events):
        idx = np.where(groups == e)[0]
        p = np.exp(eta[idx])
        p /= p.sum()
        y[idx[rng.choice(len(idx), p=p)]] = 1.0
    return y, x[:, None], groups


@pytest.fixture
def benign_poisson():
    """A well-posed Poisson GLM design (converges cleanly, no separation)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    X = sm.add_constant(x)
    y = rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float)
    return y, X


@pytest.fixture
def separated_binary():
    """A perfectly separated 2-class design: y == (x > 0), no overlap."""
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y = (x > 0).astype(float)
    return y, sm.add_constant(x)


def test_glm_converged(benign_poisson):
    y, X = benign_poisson
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    assert fit_status(res, "glm") == {"converged": True, "separated": False}


def test_separation_flagged(separated_binary):
    """Perfect separation must be flagged (large coef / non-finite SE), not returned clean."""
    y, X = separated_binary
    with capture_fit_warnings():  # separation emits warnings we don't want on the console
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    status = fit_status(res, "glm")
    assert status["separated"] is True
    # Guard against a tautology: the benign design in the same helper is NOT separated.
    assert np.max(np.abs(res.params)) >= 10.0


def test_benign_not_separated(benign_poisson):
    """The separation flag is discriminating, not always-on."""
    y, X = benign_poisson
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    assert fit_status(res, "glm")["separated"] is False


def test_nonconverged_flagged(benign_poisson):
    """A GLM stopped at maxiter=1 has not converged."""
    y, X = benign_poisson
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=1)
    assert fit_status(res, "glm")["converged"] is False


def test_mle_kind_reads_mle_retvals(separated_binary):
    """Discrete-MLE results read convergence from mle_retvals, not .converged."""
    y, X = separated_binary
    with capture_fit_warnings():
        res = sm.Logit(y, X).fit(disp=0)
    status = fit_status(res, "mle")
    # Perfectly separated logit does not converge and is flagged separated.
    assert status["converged"] is False
    assert status["separated"] is True


def test_conditional_converged_not_falsely_flagged():
    """Regression: ConditionalResultsWrapper exposes neither .converged nor
    mle_retvals, so a naive kind='mle' read reports a *successful* conditional
    fit as nonconverged. kind='conditional' must recover the true status from the
    captured warnings and report converged=True for a well-posed fit."""
    y, X, groups = _conditional_poisson_design()
    with capture_fit_warnings() as caught:
        res = ConditionalPoisson(y, X, groups=groups).fit(disp=0)
    status = fit_status(res, "conditional", caught=caught)
    assert status == {"converged": True, "separated": False}
    # The naive path this replaced would have (wrongly) said False:
    assert not hasattr(res, "converged") or res.converged is False
    assert not hasattr(res, "mle_retvals")


def test_conditional_nonconverged_flagged():
    """A conditional fit stopped at maxiter=1 emits a ConvergenceWarning, which
    kind='conditional' reads as converged=False."""
    y, X, groups = _conditional_poisson_design()
    with capture_fit_warnings() as caught:
        res = ConditionalPoisson(y, X, groups=groups).fit(disp=0, maxiter=1)
    assert fit_status(res, "conditional", caught=caught)["converged"] is False


def test_conditional_unknown_without_caught():
    """Without a captured-warnings list convergence is unknowable for a conditional
    wrapper -> reported as None, not a false False."""
    y, X, groups = _conditional_poisson_design()
    with capture_fit_warnings():
        res = ConditionalPoisson(y, X, groups=groups).fit(disp=0)
    assert fit_status(res, "conditional")["converged"] is None


def test_invalid_kind_raises(benign_poisson):
    y, X = benign_poisson
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    with pytest.raises(ValueError, match="kind"):
        fit_status(res, "not_a_kind")


def test_warnings_captured():
    """capture_fit_warnings records a ConvergenceWarning rather than swallowing it."""
    with capture_fit_warnings() as caught:
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=1)
    assert count_fit_quality_warnings(caught) == 1


@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning")
def test_warnings_captured_even_after_prior_emit():
    """simplefilter('always') ensures a repeat warning is still recorded (not de-duped)."""
    # Emit once outside so Python's default 'once/default' filter would suppress a repeat.
    warnings.warn("did not converge", ConvergenceWarning, stacklevel=1)
    with capture_fit_warnings() as caught:
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=1)
    assert count_fit_quality_warnings(caught) == 1
