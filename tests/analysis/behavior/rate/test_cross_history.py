"""The cross-history fit runs a session only when its two cohorts compare."""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.rate import cross_history as mod


def _session(n_stops: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One synthetic session: stops and non-local onsets, half with an SWR."""
    rng = np.random.default_rng(seed)
    stop_rows, onset_rows = [], []
    for stop_id in range(n_stops):
        immobility = float(rng.uniform(1.5, 2.5))
        stop_rows.append(
            {
                "animal": "a",
                "session_date": 1,
                "epoch": "02_r1",
                "stop_id": stop_id,
                "immobility_duration_s": immobility,
                "reward_history_rcos_0": float(stop_id % 2),
                "reward_history_rcos_1": float(rng.uniform(0, 1)),
                "reward_history_rcos_2": float(rng.uniform(0, 1)),
                "reward_history_incomplete": 0.0,
            }
        )
        for _ in range(rng.poisson(10 * immobility)):
            onset_rows.append(
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "stop_id": stop_id,
                    "event_type": "non_local",
                    "t_since_stop_onset": float(rng.uniform(0, immobility)),
                    "is_swr": bool(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(stop_rows), pd.DataFrame(onset_rows)


# The synthetic stops carry neither the arrival clocks nor the outcome lags,
# so the fit's designs opt out of both by name.
_design = functools.partial(
    mod.aggregate_to_timebins, arrival_clocks=False, outcome_shift=False
)


def _design_missing_cross_history(*args, **kwargs):
    """The fit's design, with the cross design missing one bin's cross history."""
    design = _design(*args, **kwargs)
    if "cross_onset_times" in kwargs:
        design.loc[design.index[0], "cross_hist_rcos_0"] = np.nan
    return design


@pytest.fixture
def session():
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield _session()


def test_a_session_whose_cohorts_match_is_fit(session, monkeypatch):
    stops, onsets = session
    monkeypatch.setattr(mod, "aggregate_to_timebins", _design)

    models = mod.fit_cross_history(stops, onsets)

    assert set(models["direction"]) == set(mod.DIRECTIONS)
    assert np.isfinite(models["delta_aic"]).all()


def test_a_session_whose_cohorts_differ_is_not_fit(session, monkeypatch, caplog):
    """One bin the cross design lacks: the AIC difference would not compare."""
    stops, onsets = session
    monkeypatch.setattr(mod, "aggregate_to_timebins", _design_missing_cross_history)

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        models = mod.fit_cross_history(stops, onsets)

    assert models.empty
    refusals = [r for r in caplog.records if "cohorts differ" in r.getMessage()]
    assert len(refusals) == len(mod.DIRECTIONS)
    assert "'n_bins'" in refusals[0].getMessage()
