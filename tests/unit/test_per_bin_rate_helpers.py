"""Tests for the per-bin event-rate aggregation helpers.

Covers the two shared primitives behind the per-bin rate models:

* ``attach_is_swr_to_onsets`` -- the within-stop time-rank match that tags each
  non-local onset with its canonical ``is_swr`` (checked on a real session slice
  against the stop-table SWR-split counts).
* ``aggregate_to_timebins`` -- the (stop x bin) point-process design (exposure,
  truncation, and within-stop self-history lags, checked on a synthetic fixture).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import (
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"


# ── Synthetic fixture for aggregate_to_timebins ──


@pytest.fixture
def two_stop_design() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two stops covering an empty bin, a stop boundary, and tmax truncation.

    With ``bin_s=0.1``, ``tmax=0.5``:

    * Stop 1 (immobility 0.25 s) -> 3 bins, exposures [0.1, 0.1, 0.05]; onsets
      in bins 0 and 1, bin 2 empty.
    * Stop 2 (immobility 0.85 s) -> truncated to 5 bins (not 9), all exposure
      0.1; a single onset in bin 2.
    """
    stops = pd.DataFrame(
        {
            "animal": ["x", "x"],
            "day": [1, 1],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [1, 2],
            "immobility_time": [0.25, 0.85],
            "reward_lag1": [1.0, 0.0],
        }
    )
    onsets = pd.DataFrame(
        {
            "stop_id": [1, 1, 2],
            "t_since_stop_onset": [0.05, 0.15, 0.25],
        }
    )
    return stops, onsets


def test_aggregate_exposure_and_lags(two_stop_design):
    stops, onsets = two_stop_design
    df = aggregate_to_timebins(
        stops, onsets, bin_s=0.1, tmax=0.5, reward_lags=1, self_history_lags=1
    )

    s1 = df[df.stop_id == 1].sort_values("binj")
    s2 = df[df.stop_id == 2].sort_values("binj")

    # Bins capped at tmax: stop 2 is 5 bins, not ceil(0.85 / 0.1) = 9.
    assert s1.binj.tolist() == [0, 1, 2]
    assert s2.binj.tolist() == [0, 1, 2, 3, 4]

    # Exposure: per-bin remainder; the untruncated stop sums to its immobility.
    assert s1.exposure.tolist() == pytest.approx([0.1, 0.1, 0.05])
    assert s1.exposure.sum() == pytest.approx(0.25)
    assert s2.exposure.tolist() == pytest.approx([0.1] * 5)

    # Bin centers.
    assert s1.tc.tolist() == pytest.approx([0.05, 0.15, 0.25])

    # Counts: onsets binned by floor(t / bin_s); bin 2 of stop 1 is empty.
    assert s1["count"].tolist() == [1.0, 1.0, 0.0]
    assert s2["count"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]

    # Self-history: count shifted one bin within stop, reset to 0 at each
    # stop's first bin (never crosses the boundary).
    assert s1.onset_lag1.tolist() == [0.0, 1.0, 1.0]
    assert s2.onset_lag1.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]

    # Per-stop covariate carried by stop_id.
    assert (s1.reward_lag1 == 1.0).all()
    assert (s2.reward_lag1 == 0.0).all()


def test_aggregate_drops_zero_immobility():
    """Stops with no immobility contribute no bins."""
    stops = pd.DataFrame(
        {
            "animal": ["x", "x"],
            "day": [1, 1],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [1, 2],
            "immobility_time": [0.0, 0.2],
        }
    )
    onsets = pd.DataFrame({"stop_id": [2], "t_since_stop_onset": [0.05]})
    df = aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5)
    assert set(df.stop_id) == {2}


def test_reward_groupby_drops_nan_reward():
    """Stops with NaN reward_lag1 vanish from a reward groupby.

    The reward GLMs rely on pandas' default ``dropna=True`` to reproduce the
    old explicit ``reward.notna()`` filter when collapsing the (stop x bin)
    design into (reward x bin) cells.
    """
    stops = pd.DataFrame(
        {
            "animal": ["x", "x"],
            "day": [1, 1],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [1, 2],
            "immobility_time": [0.3, 0.3],
            "reward_lag1": [1.0, np.nan],
        }
    )
    onsets = pd.DataFrame(
        {"stop_id": [1, 2], "t_since_stop_onset": [0.05, 0.05]}
    )
    df = aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5, reward_lags=1)
    assert {1, 2} <= set(df.stop_id)  # both stops produce bins
    cells = df.groupby(["reward_lag1", "binj"], as_index=False)["count"].sum()
    assert cells["reward_lag1"].notna().all()  # NaN-reward stop excluded
    assert set(cells["reward_lag1"]) == {1.0}


# ── Real-session slice for attach_is_swr_to_onsets ──


@pytest.fixture(scope="module")
def session_slice():
    """One real session's onset / event / stop rows (skip if data absent)."""
    paths = {
        name: RC / f"{name}.pkl"
        for name in ("event_onset_table", "event_table", "stop_table")
    }
    for p in paths.values():
        if not p.exists():
            pytest.skip(f"missing {p}")
    onset = pd.read_pickle(paths["event_onset_table"])
    event = pd.read_pickle(paths["event_table"])
    stop = pd.read_pickle(paths["stop_table"])

    # Pick the session with the most non-local onsets for a meaningful slice.
    sc = ["animal", "day", "epoch"]
    key = (
        onset[onset.event_type == "non_local"]
        .groupby(sc)
        .size()
        .idxmax()
    )
    sel = pd.Series(True, index=onset.index)
    for col, val in zip(sc, key, strict=True):
        sel &= onset[col] == val
    onset_s = onset[sel]
    stop_s = stop[(stop[sc] == key).all(axis=1)]
    event_s = event[event.stop_id.isin(stop_s.stop_id)]
    return onset_s, event_s, stop_s


def test_attach_is_swr_match_rate(session_slice):
    onset_s, event_s, stop_s = session_slice
    tagged = attach_is_swr_to_onsets(onset_s, event_s)

    nl = tagged[tagged.event_type == "non_local"]
    assert nl["is_swr"].notna().mean() >= 0.999

    # Non-local SWR / no-SWR onset counts match the stop-table split totals.
    n_swr = int((nl.is_swr == True).sum())  # noqa: E712
    n_no_swr = int((nl.is_swr == False).sum())  # noqa: E712
    assert n_swr == int(stop_s["n_non_local_SWR"].sum())
    assert n_no_swr == int(stop_s["n_non_local_no_SWR"].sum())

    # SWR/MUA onset rows are left untagged.
    assert tagged.loc[tagged.event_type != "non_local", "is_swr"].isna().all()


def test_attach_is_swr_does_not_mutate(session_slice):
    onset_s, event_s, _ = session_slice
    before = onset_s.copy()
    attach_is_swr_to_onsets(onset_s, event_s)
    pd.testing.assert_frame_equal(onset_s, before)
