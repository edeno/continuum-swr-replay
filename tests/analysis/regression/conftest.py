"""Synthetic sessions for the convention's tools (terms, support, the loop).

Two kinds of rows, each built from a fixed seed:

- rate rows: 15 ms bins of the visits of one session, with a Poisson count of
  event starts, the bin's exposure, two clocks, the last outcome (0/1), ten
  lagged outcomes and one continuous covariate;
- content rows: one row per candidate arm of each event (four candidates: the
  current arm, the previous arm, another leaf arm and a trunk arm), with the
  event's share on each, a clock and a reward fraction (missing on the trunk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_LAGS = 10
ROLES = ("current", "previous", "other", "trunk")


def rate_rows(seed: int, n_visits: int = 60) -> pd.DataFrame:
    """One session's visit bins (columns described in the module docstring).

    Visit durations are lognormal (median 2 s); the log rate rises over the
    first 2 s of a visit and is 0.4 higher after a rewarded last outcome.

    Returns
    -------
    pd.DataFrame, shape (n_bins, 9 + N_LAGS)
        ``animal``, ``session_date``, ``epoch``, ``well_visit_id``,
        ``exposure`` (s), ``count``, ``time_since_s``, ``time_until_s``,
        ``last_rewarded`` (0/1), ``x`` (N(0, 1) per visit) and
        ``lag_1`` .. ``lag_10`` (0/1).
    """
    rng = np.random.default_rng(seed)
    duration = 2.0 * np.exp(0.6 * rng.normal(size=n_visits))
    n_bins = np.maximum(np.ceil(duration / 0.015).astype(int), 1)
    visit = np.repeat(np.arange(n_visits), n_bins)
    k = np.arange(n_bins.sum()) - np.repeat(np.cumsum(n_bins) - n_bins, n_bins)
    start = 0.015 * k
    end = np.minimum(start + 0.015, duration[visit])
    since = 0.5 * (start + end)
    outcomes = (rng.uniform(size=(n_visits, N_LAGS)) < 0.6).astype(float)
    x = rng.normal(size=n_visits)
    log_rate = (
        np.log(0.5)
        + np.log(3.0) * (1.0 - np.exp(-since / 0.7))
        + 0.4 * outcomes[visit, 0]
        + 0.2 * x[visit]
    )
    exposure = end - start
    frame = pd.DataFrame(
        {
            "animal": "sim",
            "session_date": f"2026{seed:04d}",
            "epoch": "02_r1",
            "well_visit_id": visit,
            "exposure": exposure,
            "count": rng.poisson(exposure * np.exp(log_rate)).astype(float),
            "time_since_s": since,
            "time_until_s": duration[visit] - since,
            "last_rewarded": outcomes[visit, 0],
            "x": x[visit],
        }
    )
    for lag in range(N_LAGS):
        frame[f"lag_{lag + 1}"] = outcomes[visit, lag]
    return frame


def content_rows(seed: int, n_events: int = 150) -> pd.DataFrame:
    """One session's candidate rows (columns in the module docstring).

    Each event's shares are a Dirichlet draw around a softmax that favors the
    current arm and rises with the reward fraction.

    Returns
    -------
    pd.DataFrame, shape (4 * n_events, 8)
        ``animal``, ``session_date``, ``epoch``, ``event_id``, ``role``,
        ``share``, ``clock_s`` (the event's time since arrival) and
        ``reward_fraction`` (NaN on the trunk row).
    """
    rng = np.random.default_rng(10_000 + seed)
    event = np.repeat(np.arange(n_events), len(ROLES))
    role = np.tile(np.array(ROLES, dtype=object), n_events)
    fraction = rng.uniform(0.0, 1.0, size=event.size)
    fraction[role == "trunk"] = np.nan
    clock = np.repeat(rng.uniform(0.0, 20.0, size=n_events), len(ROLES))
    eta = np.select(
        [role == "current", role == "previous", role == "other"],
        [1.0, 0.3, 0.0],
        default=-0.5,
    ) + 0.8 * np.nan_to_num(fraction)
    weight = np.exp(eta).reshape(n_events, len(ROLES))
    p = weight / weight.sum(axis=1, keepdims=True)
    shares = np.concatenate([rng.dirichlet(20.0 * row) for row in p])
    return pd.DataFrame(
        {
            "animal": "sim",
            "session_date": f"2026{seed:04d}",
            "epoch": "02_r1",
            "event_id": event,
            "role": role,
            "share": shares,
            "clock_s": clock,
            "reward_fraction": fraction,
        }
    )


def by_session(frames: list[pd.DataFrame]) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Session key ``(animal, session_date, epoch)`` -> its rows."""
    return {
        tuple(frame.loc[0, ["animal", "session_date", "epoch"]]): frame
        for frame in frames
    }


@pytest.fixture
def rate_sessions() -> dict[tuple[str, str, str], pd.DataFrame]:
    """Six rate sessions (seeds 0 to 5)."""
    return by_session([rate_rows(seed) for seed in range(6)])


@pytest.fixture
def content_sessions() -> dict[tuple[str, str, str], pd.DataFrame]:
    """Six content sessions (seeds 0 to 5)."""
    return by_session([content_rows(seed) for seed in range(6)])
