"""The log penalty at which a smooth term has a target median edf over sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression import calibration
from continuum_swr_replay.analysis.regression.calibration import log_penalty_for_edf
from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.terms import (
    Offset,
    Slope,
    Smooth,
    build_design,
)

TERM = "s(time_since_s)"


def _design(pooled: pd.DataFrame, penalty: float | str = 2.0):
    """The rate design: a smooth of the clock, a slope of ``x``, the offset."""
    return build_design(
        [
            Smooth("time_since_s", 0.0, 20.0, 12, penalty),
            Slope("x", "none"),
            Offset("exposure"),
        ],
        pooled,
        intercept=True,
    )


def _calibrate(design, sessions, target: float):
    return log_penalty_for_edf(
        design,
        sessions,
        Poisson(),
        term=TERM,
        target_edf=target,
        response="count",
        unit="well_visit_id",
    )


def _edf_at(sessions, pooled, log_penalty: float) -> np.ndarray:
    """Each session's edf of the smooth, the design built at ``log_penalty``."""
    records = fit_sessions(
        _design(pooled, log_penalty),
        sessions,
        Poisson(),
        model="test",
        kind="SWR",
        response="count",
        unit="well_visit_id",
        table_sha256={},
    )
    return np.array([r.edf[TERM] for r in records])


def test_the_penalty_gives_the_target_median_edf(rate_sessions) -> None:
    """At the returned log penalty, the median edf of the smooth over the
    sessions is the target to 0.01, and fitting the design built at that
    penalty gives the same edf in each session."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    found = _calibrate(_design(pooled), rate_sessions, 4.0)
    assert found.median_edf == pytest.approx(4.0, abs=0.01)
    assert np.median(found.edf.to_numpy()) == found.median_edf
    assert list(found.edf.index) == list(rate_sessions)
    np.testing.assert_allclose(
        _edf_at(rate_sessions, pooled, found.log_penalty),
        found.edf.to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_edf_falls_as_the_penalty_rises(rate_sessions) -> None:
    """A smaller target needs a larger log penalty, and every session's edf is
    lower at the larger penalty."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    design = _design(pooled)
    found = [_calibrate(design, rate_sessions, target) for target in (5.0, 3.5, 2.5)]
    penalties = [f.log_penalty for f in found]
    assert penalties[0] < penalties[1] < penalties[2]
    for looser, stiffer in zip(found[:-1], found[1:], strict=True):
        assert (stiffer.edf.to_numpy() < looser.edf.to_numpy()).all()


def test_it_is_deterministic(rate_sessions) -> None:
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    design = _design(pooled)
    first, second = (_calibrate(design, rate_sessions, 5.0) for _ in range(2))
    assert first.log_penalty == second.log_penalty
    pd.testing.assert_series_equal(first.edf, second.edf)


def test_the_other_penalties_are_held(rate_sessions) -> None:
    """With two smooths, the other keeps its declared log penalty: its value
    changes the answer."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)

    def design(other: float):
        return build_design(
            [
                Smooth("time_since_s", 0.0, 20.0, 12, 2.0),
                Smooth("time_until_s", 0.0, 20.0, 12, other),
                Offset("exposure"),
            ],
            pooled,
            intercept=True,
        )

    stiff, loose = (_calibrate(design(v), rate_sessions, 4.0) for v in (15.0, -5.0))
    assert stiff.log_penalty != loose.log_penalty


@pytest.mark.parametrize("target", [0.5, 50.0])
def test_a_target_out_of_reach_inside_the_bounds_raises(rate_sessions, target) -> None:
    """The constrained smooth of 11 columns has an edf between its line (1)
    and 11: a target outside what the bounds reach is refused, naming it."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    with pytest.raises(ValueError, match="cannot be reached"):
        _calibrate(_design(pooled), rate_sessions, target)


def test_a_term_without_a_fixed_penalty_is_refused(rate_sessions) -> None:
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    with pytest.raises(ValueError, match="no fixed penalty"):
        _calibrate(_design(pooled, "none"), rate_sessions, 4.0)
    with pytest.raises(ValueError, match="no fixed penalty"):
        log_penalty_for_edf(
            _design(pooled),
            rate_sessions,
            Poisson(),
            term="x",
            target_edf=1.0,
            response="count",
            unit="well_visit_id",
        )
    with pytest.raises(ValueError, match="not a term"):
        log_penalty_for_edf(
            _design(pooled),
            rate_sessions,
            Poisson(),
            term="s(nothing)",
            target_edf=4.0,
            response="count",
            unit="well_visit_id",
        )


def test_every_session_is_counted_in_the_median(rate_sessions) -> None:
    """With every session fitted, the median is over all of them."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    found = _calibrate(_design(pooled), rate_sessions, 4.0)
    assert found.n_sessions_used == len(rate_sessions)


def test_a_session_without_an_edf_is_counted_out(rate_sessions, monkeypatch) -> None:
    """A session whose edf is NaN at every penalty (no kept fit) is left out
    of the median, and ``n_sessions_used`` says so."""
    real = calibration._term_edf

    def one_missing(*args, **kwargs):
        edf = real(*args, **kwargs).copy()
        edf.iloc[0] = np.nan
        return edf

    monkeypatch.setattr(calibration, "_term_edf", one_missing)
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    found = _calibrate(_design(pooled), rate_sessions, 4.0)
    assert found.n_sessions_used == len(rate_sessions) - 1
    assert np.isnan(found.edf.iloc[0])
    assert found.median_edf == pytest.approx(4.0, abs=0.01)


def test_a_median_that_jumps_past_the_target_raises(rate_sessions, monkeypatch) -> None:
    """An edf that is a step of the penalty (8 below a log penalty of 3, 2
    above it), as when sessions drop in and out of the median: the bisection
    ends at the step, the median there is far from the target 4, and the
    calibration refuses it rather than return it."""

    def step(design, sessions, lik, term, response, unit):
        rho = design.log_penalties[design.by_name[term].block]
        return pd.Series(8.0 if rho < 3.0 else 2.0, index=pd.Index(list(sessions)))

    monkeypatch.setattr(calibration, "_term_edf", step)
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    with pytest.raises(ValueError, match="differs from the target"):
        _calibrate(_design(pooled), rate_sessions, 4.0)
