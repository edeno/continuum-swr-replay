"""Fit-diagnostics helpers flag convergence/separation without dropping fits."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from continuum_swr_replay.analysis.replay._fit_diagnostics import (
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_status,
)


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
