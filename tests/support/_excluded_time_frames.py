"""Synthetic decode frames with excluded time, for the per-bin summaries' tests.

``two_sessions`` holds two sessions on a 500 Hz grid, as ``load_data_over_time``
concatenates them, each with the ``in_excluded_time`` column the read adds
(``analysed_time.with_swr_source``). Session ``a`` has three excluded stretches,
filled with values that change every statistic if they are counted:

- bins 200-249: moving and local, with a 200 cm error;
- bins 400-499: the middle of a collapse run (``Local_prob`` 0.1 over bins
  380-519), so the run's analysed parts are 380-399 and 500-519;
- bins 800-869: immobile, local (800-849) and then non-local (850-869).

In all three, hpd is 500, theta phase 0.1, ``SWR_zscore`` 9, and ``MUA_zscore``
and ``No-Spike_prob`` take the top ranks of both. Session ``b`` holds the same
values with nothing excluded, so its results must not change. Bins 0-699 are
moving (10 cm/s), 700-999 immobile (1 cm/s). One non-local event, bins 900-919,
lies outside excluded time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FS = 500.0
N = 1000
MOVING = np.arange(N) < 700
COLLAPSE = (np.arange(N) >= 380) & (np.arange(N) < 520)
EXCLUDED = np.zeros(N, dtype=bool)
EXCLUDED[200:250] = True
EXCLUDED[400:500] = True
EXCLUDED[800:870] = True
NL_EVENT = (np.arange(N) >= 900) & (np.arange(N) < 920)


def session(animal: str, *, excluded: bool) -> pd.DataFrame:
    """One session's frame; `excluded` marks the three stretches."""
    rng = np.random.default_rng(7)
    local = np.where(COLLAPSE, 0.1, rng.uniform(0.55, 1.0, N))
    local[200:250] = 0.95
    local[800:850] = 0.95
    local[850:870] = 0.05
    local[NL_EVENT] = 0.05
    error = np.where(COLLAPSE, 30.0, rng.uniform(0.0, 5.0, N))
    error[EXCLUDED] = 200.0
    hpd = np.where(COLLAPSE, 50.0, 5.0)
    hpd[EXCLUDED] = 500.0
    mua = rng.normal(0.0, 1.0, N)
    nospike = np.clip(0.02 - 0.01 * mua + rng.normal(0.0, 0.002, N), 0.0, 0.05)
    n_excluded = int(EXCLUDED.sum())
    mua[EXCLUDED] = 20.0 + np.arange(n_excluded)
    nospike[EXCLUDED] = 0.05 + 0.001 * np.arange(n_excluded)
    theta_phase = rng.uniform(0.0, 2 * np.pi, N)
    theta_phase[EXCLUDED] = 0.1
    swr_z = rng.normal(0.0, 1.0, N)
    swr_z[EXCLUDED] = 9.0
    nonlocal_prob = 1.0 - local
    return pd.DataFrame(
        {
            "animal": animal,
            "day": 20200101,
            "epoch": "02_r1",
            "head_speed": np.where(MOVING, 10.0, 1.0),
            "Local_prob": local,
            "No-Spike_prob": nospike,
            "Non-Local_Continuous_prob": 0.7 * nonlocal_prob,
            "Non-Local_Fragmented_prob": 0.3 * nonlocal_prob,
            "Non-Local_prob": nonlocal_prob,
            "ahead_behind_distance": error,
            "hpd": hpd,
            "track_segment_id": np.repeat(np.arange(5), N // 5),
            "linear_position": np.linspace(0.0, 100.0, N),
            "non_local_event": NL_EVENT.astype(int),
            "SWR_event": 0,
            "MUA_event": 0,
            "theta_phase": theta_phase,
            "theta_power_zscore": rng.normal(0.0, 1.0, N),
            "SWR_zscore": swr_z,
            "MUA_zscore": mua,
            "in_excluded_time": EXCLUDED.copy() if excluded else np.zeros(N, bool),
        },
        index=pd.Index(np.arange(N) / FS, name="time"),
    )


def two_sessions() -> pd.DataFrame:
    """Session ``a`` (with excluded time) and session ``b`` (without)."""
    return pd.concat([session("a", excluded=True), session("b", excluded=False)])


def rows_removed(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame without its excluded rows (nothing left to exclude)."""
    return frame.loc[~frame["in_excluded_time"].to_numpy()]


def as_gap(frame: pd.DataFrame) -> pd.DataFrame:
    """The excluded rows kept as rows that no run or statistic can use:
    immobile, with no hpd, and no longer marked."""
    out = frame.copy()
    gap = out["in_excluded_time"].to_numpy()
    out.loc[gap, "head_speed"] = 0.0
    out.loc[gap, "hpd"] = np.nan
    out["in_excluded_time"] = False
    return out


def unfiltered(frame: pd.DataFrame) -> pd.DataFrame:
    """What every summary saw before task 26b: the excluded rows counted."""
    return frame.assign(in_excluded_time=False)
