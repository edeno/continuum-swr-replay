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
    raised_cosine_history_basis,
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


def test_aggregate_nondivisible_tmax_keeps_final_window():
    """tmax not a multiple of bin_s: the final partial window still exists and
    counts onsets, with its exposure clipped to tmax.

    Regression: ``int(tmax / bin_s)`` used to drop bins in
    ``[floor(tmax/bin_s)*bin_s, tmax)``, silently losing onsets there.
    """
    stops = pd.DataFrame(
        {
            "animal": ["x"],
            "day": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_time": [1.0],  # longer than tmax
        }
    )
    # onset in the final partial window [0.4, 0.45) -- dropped by the old path.
    onsets = pd.DataFrame({"stop_id": [1], "t_since_stop_onset": [0.42]})
    df = aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.45).sort_values("binj")
    # ceil(0.45 / 0.1) = 5 bins (0..4), not int()'s 4.
    assert list(df.binj) == [0, 1, 2, 3, 4]
    # the onset is counted in the final bin rather than dropped.
    assert df.loc[df.binj == 4, "count"].iloc[0] == 1
    assert df["count"].sum() == 1
    # final bin exposure is clipped to tmax (0.45 - 0.4 = 0.05), not a full bin.
    assert df.loc[df.binj == 4, "exposure"].iloc[0] == pytest.approx(0.05)
    assert df.loc[df.binj == 0, "exposure"].iloc[0] == pytest.approx(0.1)


def test_raised_cosine_basis_shape_and_nonneg():
    B = raised_cosine_history_basis(10, 4)
    assert B.shape == (4, 10)
    assert (B >= 0).all()
    # Each bump peaks (==1) at its own log-spaced center; centers are distinct.
    peak_lags = B.argmax(axis=1)
    assert len(set(peak_lags.tolist())) == 4
    assert peak_lags[0] < peak_lags[-1]  # first bump early, last bump late


def test_aggregate_history_basis_is_projection():
    """hist_rcos_k equals the onset_lag counts projected onto the basis."""
    stops = pd.DataFrame({
        "animal": ["x"], "day": [1], "epoch": ["02_r1"], "stop_id": [1],
        "immobility_time": [0.5],
    })
    onsets = pd.DataFrame({"stop_id": [1, 1], "t_since_stop_onset": [0.05, 0.20]})
    df = aggregate_to_timebins(
        stops, onsets, bin_s=0.05, tmax=0.5,
        self_history_lags=4, self_history_basis=3,
    )
    assert {"hist_rcos_0", "hist_rcos_1", "hist_rcos_2"} <= set(df.columns)
    B = raised_cosine_history_basis(4, 3)
    lag_cols = [f"onset_lag{j}" for j in range(1, 5)]
    expected = df[lag_cols].to_numpy() @ B.T
    np.testing.assert_allclose(
        df[["hist_rcos_0", "hist_rcos_1", "hist_rcos_2"]].to_numpy(), expected
    )


def test_history_basis_requires_lags():
    stops = pd.DataFrame({
        "animal": ["x"], "day": [1], "epoch": ["02_r1"], "stop_id": [1],
        "immobility_time": [0.5],
    })
    onsets = pd.DataFrame({"stop_id": [1], "t_since_stop_onset": [0.05]})
    with pytest.raises(ValueError, match="self_history_lags"):
        aggregate_to_timebins(
            stops, onsets, bin_s=0.05, tmax=0.5, self_history_basis=3
        )


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


def test_aggregate_cross_history_lags():
    """cross_lag{j} = the OTHER stream's onset count j bins back, within stop.

    Bivariate conditional intensity: the non-SWR response's design carries lagged
    counts of the with-SWR (cross) onset stream, reset at the stop boundary.
    """
    stops = pd.DataFrame(
        {
            "animal": ["x"], "day": [1], "epoch": ["02_r1"],
            "stop_id": [1], "immobility_time": [1.0],
        }
    )
    # response (non-SWR) onsets in bins 3, 4
    resp = pd.DataFrame({"stop_id": [1, 1], "t_since_stop_onset": [0.35, 0.45]})
    # cross (with-SWR) onsets in bins 2, 5
    cross = pd.DataFrame({"stop_id": [1, 1], "t_since_stop_onset": [0.25, 0.55]})
    df = aggregate_to_timebins(
        stops, resp, bin_s=0.1, tmax=1.0,
        self_history_lags=2, cross_onset_times=cross, cross_history_lags=2,
    ).sort_values("binj").set_index("binj")
    # response counts unaffected by the cross stream
    assert df.loc[3, "count"] == 1 and df.loc[4, "count"] == 1
    # cross_lag1 fires one bin after each cross onset (bins 2->3, 5->6)
    assert df.loc[3, "cross_lag1"] == 1
    assert df.loc[6, "cross_lag1"] == 1
    # cross_lag2 fires two bins after (bins 2->4, 5->7)
    assert df.loc[4, "cross_lag2"] == 1
    assert df.loc[7, "cross_lag2"] == 1
    # boundary reset: bin 0 has no prior bins in the stop
    assert df.loc[0, "cross_lag1"] == 0
    # the cross stream does not leak into the self-history (own count) lags
    assert df.loc[3, "onset_lag1"] == 0  # no non-SWR onset in bin 2


def test_aggregate_cross_history_requires_stream():
    """cross_history_lags without cross_onset_times is a loud error."""
    stops = pd.DataFrame(
        {"animal": ["x"], "day": [1], "epoch": ["02_r1"],
         "stop_id": [1], "immobility_time": [0.5]}
    )
    onsets = pd.DataFrame({"stop_id": [1], "t_since_stop_onset": [0.05]})
    with pytest.raises(ValueError, match="cross_onset_times"):
        aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5, cross_history_lags=2)
