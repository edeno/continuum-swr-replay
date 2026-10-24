"""Tests for the per-bin event-rate aggregation helpers.

Covers the two shared primitives behind the per-bin rate models:

* ``attach_is_swr_to_onsets`` -- the within-stop time-rank match that tags each
  non-local onset with its canonical ``is_swr`` (checked on a real session slice
  against the stop-table SWR-split counts).
* ``aggregate_to_timebins`` -- the (stop x bin) point-process design (exposure,
  truncation, and within-stop self-history lags, checked on a synthetic fixture).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import (
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    raised_cosine_history_basis,
)
from continuum_swr_replay.analysis.replay.covariates.history.reward_history import (
    reward_basis_from_lags,
)
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, table_exists

RC = REPLAY_CONTENT_DIR


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
            "session_date": [1, 1],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [1, 2],
            "immobility_duration_s": [0.25, 0.85],
            "reward_history_rcos_0": [1.0, 0.0],
            "reward_history_rcos_1": [0.8, 0.2],
            "reward_history_rcos_2": [0.6, 0.4],
            "reward_history_incomplete": [0.0, 0.0],
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
        stops, onsets, bin_s=0.1, tmax=0.5, reward_basis=True, self_history_lags=1
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

    # Per-stop covariates carried by stop_id: every basis feature plus the
    # structural-missingness indicator.
    assert (s1.reward_history_rcos_0 == 1.0).all()
    assert (s1.reward_history_rcos_1 == 0.8).all()
    assert (s1.reward_history_rcos_2 == 0.6).all()
    assert (s2.reward_history_rcos_0 == 0.0).all()
    assert (s2.reward_history_rcos_1 == 0.2).all()
    assert (s2.reward_history_rcos_2 == 0.4).all()
    assert (df.reward_history_incomplete == 0.0).all()


def test_aggregate_drops_zero_immobility():
    """Stops with no immobility contribute no bins."""
    stops = pd.DataFrame(
        {
            "animal": ["x", "x"],
            "session_date": [1, 1],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [1, 2],
            "immobility_duration_s": [0.0, 0.2],
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
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [1.0],  # longer than tmax
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


def test_onset_on_bin_edge():
    """An onset on a bin edge belongs to the bin that starts there.

    ``(t / bin_s).astype(int)`` truncates the quotient, so an edge time whose
    binary-float quotient lands just below the integer (0.3 / 0.1 evaluates to
    2.9999999999999996) fell one bin early -- a systematic backward shift of
    the estimated time course. Bin 2 spans [0.2, 0.3), so 0.3 is bin 3.
    """
    stops = pd.DataFrame(
        {
            "animal": ["x"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [0.5],
        }
    )
    onsets = pd.DataFrame({"stop_id": [1, 1], "t_since_stop_onset": [0.2, 0.3]})
    df = aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5).sort_values("binj")
    assert df["count"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]


def test_onset_at_the_end_of_immobility_raises():
    """An onset at ``t == immobility_duration_s`` is outside the stop's window.

    It is binned (the stop's last bin covers it) but not expected, so the
    (stop x bin) design would carry a count the exposure never earned. The
    count reconciliation raises rather than fitting the inconsistency.
    """
    stops = pd.DataFrame(
        {
            "animal": ["x"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [0.25],
        }
    )
    onsets = pd.DataFrame({"stop_id": [1, 1], "t_since_stop_onset": [0.05, 0.25]})
    with pytest.raises(ValueError, match=r"binned 2 onset\(s\) but 1"):
        aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5)


def test_onset_in_a_dropped_sliver_bin_is_counted_not_raised():
    """A stop ending a sliver past a bin edge loses that bin, and any onset in it.

    ``immobility_duration_s = 0.30000005`` gives bin 3 an exposure of 5e-8, which
    the ``exposure > 1e-4`` filter removes. An onset at 0.300000001 is inside the
    stop but inside that dropped bin, so it is counted nowhere -- a real (tiny)
    loss rather than a counts-vs-exposure inconsistency, so it is recorded rather
    than raised on.
    """
    stops = pd.DataFrame(
        {
            "animal": ["x"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [0.30000005],
        }
    )
    onsets = pd.DataFrame({"stop_id": [1], "t_since_stop_onset": [0.300000001]})
    df = aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5)
    assert df.binj.tolist() == [0, 1, 2]  # the 5e-8-exposure bin 3 is gone
    assert df["count"].sum() == 0
    assert df.attrs["n_onsets_in_dropped_bins"] == 1


def test_no_onsets_are_lost_in_an_ordinary_design(two_stop_design):
    stops, onsets = two_stop_design
    df = aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5)
    assert df["count"].sum() == 3
    assert df.attrs["n_onsets_in_dropped_bins"] == 0


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
    stops = pd.DataFrame(
        {
            "animal": ["x"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [0.5],
        }
    )
    onsets = pd.DataFrame({"stop_id": [1, 1], "t_since_stop_onset": [0.05, 0.20]})
    df = aggregate_to_timebins(
        stops,
        onsets,
        bin_s=0.05,
        tmax=0.5,
        self_history_lags=4,
        self_history_basis=3,
    )
    assert {"hist_rcos_0", "hist_rcos_1", "hist_rcos_2"} <= set(df.columns)
    B = raised_cosine_history_basis(4, 3)
    lag_cols = [f"onset_lag{j}" for j in range(1, 5)]
    expected = df[lag_cols].to_numpy() @ B.T
    np.testing.assert_allclose(
        df[["hist_rcos_0", "hist_rcos_1", "hist_rcos_2"]].to_numpy(), expected
    )


def test_history_basis_requires_lags():
    stops = pd.DataFrame(
        {
            "animal": ["x"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [0.5],
        }
    )
    onsets = pd.DataFrame({"stop_id": [1], "t_since_stop_onset": [0.05]})
    with pytest.raises(ValueError, match="self_history_lags"):
        aggregate_to_timebins(stops, onsets, bin_s=0.05, tmax=0.5, self_history_basis=3)


def test_reward_groupby_drops_nan_reward():
    """Stops with a NaN reward-history feature vanish from a reward groupby.

    The reward GLMs rely on pandas' default ``dropna=True`` to reproduce the
    old explicit ``reward.notna()`` filter when collapsing the (stop x bin)
    design into (reward x bin) cells.
    """
    stops = pd.DataFrame(
        {
            "animal": ["x", "x"],
            "session_date": [1, 1],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [1, 2],
            "immobility_duration_s": [0.3, 0.3],
            "reward_history_rcos_0": [1.0, np.nan],
            "reward_history_rcos_1": [1.0, np.nan],
            "reward_history_rcos_2": [1.0, np.nan],
            "reward_history_incomplete": [0.0, 0.0],
        }
    )
    onsets = pd.DataFrame({"stop_id": [1, 2], "t_since_stop_onset": [0.05, 0.05]})
    df = aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5, reward_basis=True)
    assert {1, 2} <= set(df.stop_id)  # both stops produce bins
    cells = df.groupby(["reward_history_rcos_0", "binj"], as_index=False)["count"].sum()
    assert cells["reward_history_rcos_0"].notna().all()  # NaN-reward stop excluded
    assert set(cells["reward_history_rcos_0"]) == {1.0}


# ── Real-session slice for attach_is_swr_to_onsets ──


@pytest.fixture(scope="module")
def session_slice():
    """One real session's onset / event / stop rows (fails if data absent).

    Only ``real_data`` tests may request it.
    """
    paths = {
        name: RC / f"{name}.pkl"
        for name in ("event_onset_table", "event_table", "stop_table")
    }
    for p in paths.values():
        if not table_exists(p):
            pytest.fail(f"missing {p}")
    onset = read_table(paths["event_onset_table"])
    event = read_table(paths["event_table"])
    stop = read_table(paths["stop_table"])
    for frame, table in (
        (onset, "event_onset"),
        (event, "event"),
        (stop, "stop"),
    ):
        try:
            require_covariate_schema(frame, table=table)
        except ValueError as exc:
            pytest.fail(f"canonical artifacts require regeneration: {exc}")

    # Pick the session with the most non-local onsets for a meaningful slice.
    sc = ["animal", "session_date", "epoch"]
    key = onset[onset.event_type == "non_local"].groupby(sc).size().idxmax()
    sel = pd.Series(True, index=onset.index)
    for col, val in zip(sc, key, strict=True):
        sel &= onset[col] == val
    onset_s = onset[sel]
    stop_s = stop[(stop[sc] == key).all(axis=1)]
    event_s = event[event.stop_id.isin(stop_s.stop_id)]
    return onset_s, event_s, stop_s


@pytest.mark.real_data
def test_attach_is_swr_match_rate(session_slice):
    onset_s, event_s, stop_s = session_slice
    tagged = attach_is_swr_to_onsets(onset_s, event_s)

    nl = tagged[tagged.event_type == "non_local"]
    assert nl["is_swr"].notna().all()

    # Non-local SWR / no-SWR onset counts match the stop-table split totals.
    n_swr = int((nl.is_swr == True).sum())  # noqa: E712
    n_no_swr = int((nl.is_swr == False).sum())  # noqa: E712
    assert n_swr == int(stop_s["n_nonlocal_swr_events"].sum())
    assert n_no_swr == int(stop_s["n_nonlocal_non_swr_events"].sum())

    # SWR/MUA onset rows are left untagged.
    assert tagged.loc[tagged.event_type != "non_local", "is_swr"].isna().all()


def test_attach_is_swr_requires_exact_nonlocal_partition():
    onsets = pd.DataFrame(
        {
            "stop_id": [1, 1],
            "event_type": ["non_local", "non_local"],
            "onset_time": [1.0, 2.0],
        }
    )
    events = pd.DataFrame({"stop_id": [1], "event_start_s": [1.0], "is_swr": [True]})

    with pytest.raises(ValueError, match="exact.*partition|unmatched"):
        attach_is_swr_to_onsets(onsets, events)


def test_attach_is_swr_freezes_shared_sessions_then_is_strict_within_each():
    """An onset-only session is audited/excluded, not allowed to block a valid one."""
    onsets = pd.DataFrame(
        {
            "animal": ["a", "a", "b"],
            "session_date": [1, 1, 2],
            "epoch": ["e", "e", "e"],
            "stop_id": [1, 1, 2],
            "event_type": ["non_local", "SWR", "non_local"],
            "onset_time": [1.0, 1.1, 2.0],
        }
    )
    events = pd.DataFrame(
        {
            "animal": ["a"],
            "session_date": [1],
            "epoch": ["e"],
            "stop_id": [1],
            "event_start_s": [1.0],
            "is_swr": [True],
        }
    )

    with pytest.warns(RuntimeWarning, match="onset_only"):
        tagged = attach_is_swr_to_onsets(onsets, events)

    assert tagged[["animal", "session_date", "epoch"]].drop_duplicates().to_records(
        index=False
    ).tolist() == [("a", 1, "e")]
    nonlocal_rows = tagged[tagged["event_type"] == "non_local"]
    assert nonlocal_rows["is_swr"].tolist() == [True]
    audit = tagged.attrs["event_session_alignment"]
    assert audit["n_shared_sessions"] == 1
    assert audit["onset_only_sessions"] == (("b", 2, "e"),)
    assert audit["n_onset_rows_input"] == 3
    assert audit["n_onset_rows_retained"] == 2


def test_attach_is_swr_rejects_mismatch_inside_a_shared_session():
    onsets = pd.DataFrame(
        {
            "animal": ["a", "a"],
            "session_date": [1, 1],
            "epoch": ["e", "e"],
            "stop_id": [1, 1],
            "event_type": ["non_local", "non_local"],
            "onset_time": [1.0, 2.0],
        }
    )
    events = pd.DataFrame(
        {
            "animal": ["a"],
            "session_date": [1],
            "epoch": ["e"],
            "stop_id": [1],
            "event_start_s": [1.0],
            "is_swr": [True],
        }
    )

    with pytest.raises(ValueError, match="within every shared session"):
        attach_is_swr_to_onsets(onsets, events)


@pytest.mark.real_data
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
            "animal": ["x"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [1.0],
        }
    )
    # response (non-SWR) onsets in bins 3, 4
    resp = pd.DataFrame({"stop_id": [1, 1], "t_since_stop_onset": [0.35, 0.45]})
    # cross (with-SWR) onsets in bins 2, 5
    cross = pd.DataFrame({"stop_id": [1, 1], "t_since_stop_onset": [0.25, 0.55]})
    df = (
        aggregate_to_timebins(
            stops,
            resp,
            bin_s=0.1,
            tmax=1.0,
            self_history_lags=2,
            cross_onset_times=cross,
            cross_history_lags=2,
        )
        .sort_values("binj")
        .set_index("binj")
    )
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
        {
            "animal": ["x"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "stop_id": [1],
            "immobility_duration_s": [0.5],
        }
    )
    onsets = pd.DataFrame({"stop_id": [1], "t_since_stop_onset": [0.05]})
    with pytest.raises(ValueError, match="cross_onset_times"):
        aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5, cross_history_lags=2)


# ── Arrival / outcome-availability clocks and the boundary-shifted history ──


LAG_COLUMNS = [f"known_reward_outcome_lag_{j}" for j in range(1, 11)]


def _bout_start_features(lags: list[float]) -> tuple[np.ndarray, float]:
    """Stop-level reward-history features for one lag vector.

    Calls the same primitives the stop table uses, so a test expectation can
    never drift from the production kernel.
    """
    raw = reward_basis_from_lags(
        np.asarray([lags], dtype=float), raised_cosine_history_basis(10, 3)
    )
    return np.nan_to_num(raw[0], nan=0.0), float(np.isnan(raw[0]).any())


#: Fully known history for the stop that crosses its outcome boundary.
FULL_LAGS = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
#: One known outcome only -- the slowest bump is undefined at bout start.
ONE_LAG = [1.0] + [np.nan] * 9


@pytest.fixture
def arrival_clock_stops() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seven stops covering every branch of the outcome-boundary rule.

    With ``bin_s=0.1``, ``tmax=0.5`` (left edges are ``stop_start + binj * 0.1``):

    * stop 1 -- ``at_well``, boundary 0.15 s into a 0.35 s bout: bins 0-1 keep the
      bout-start history, bins 2-3 are shifted.
    * stop 2 -- ``at_well`` with a single known outcome, boundary 0.05 s in: the
      shift supplies a second outcome, so ``reward_history_incomplete`` flips 1 -> 0.
    * stop 3 -- ``mixed`` but the next stop's arrival is nowhere near it, so the
      bout spans no arrival: shifted like any other crossing bout.
    * stop 4 -- the outcome was already known at bout start: never shifted.
    * stop 5 -- no completed arrival: both clocks NaN, never shifted.
    * stop 6 -- genuinely arrival-spanning: stop 7's arrival (600.2) falls inside
      ``(600.0, 600.35]``. Never shifted, and counted.
    * stop 7 -- the successor whose arrival stop 6 contains; already past its own
      outcome at bout start, so it is never shifted itself.
    """
    starts = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]
    arrivals = [99.5, 199.4, np.nan, 399.0, np.nan, 599.5, 600.2]
    outcomes = [100.15, 200.05, 300.15, 399.5, np.nan, 600.15, 699.5]
    lags = [FULL_LAGS, ONE_LAG, FULL_LAGS, FULL_LAGS, ONE_LAG, FULL_LAGS, FULL_LAGS]
    stops = pd.DataFrame(
        {
            "animal": ["x"] * 7,
            "session_date": [1] * 7,
            "epoch": ["02_r1"] * 7,
            "stop_id": [1, 2, 3, 4, 5, 6, 7],
            "stop_start": starts,
            "stop_end": [s + 0.35 for s in starts],
            "immobility_duration_s": [0.35] * 7,
            "occupancy_state": [
                "at_well",
                "at_well",
                "mixed",
                "at_well",
                "in_transit",
                "mixed",
                "at_well",
            ],
            "completed_trial_reward": [1.0, 0.0, 1.0, 1.0, np.nan, 1.0, 1.0],
            "last_well_arrival_time_s": arrivals,
            "outcome_known_time_s": outcomes,
        }
    )
    for j, col in enumerate(LAG_COLUMNS):
        stops[col] = [row[j] for row in lags]
    features = [_bout_start_features(row) for row in lags]
    for k in range(3):
        stops[f"reward_history_rcos_{k}"] = [f[0][k] for f in features]
    stops["reward_history_incomplete"] = [f[1] for f in features]
    onsets = pd.DataFrame(
        {"stop_id": [1, 2], "t_since_stop_onset": [0.05, 0.25]}, dtype=float
    )
    onsets["stop_id"] = onsets["stop_id"].astype(int)
    return stops, onsets


def _aggregate_with_clocks(stops, onsets):
    return aggregate_to_timebins(
        stops, onsets, bin_s=0.1, tmax=0.5, reward_basis=True
    ).sort_values(["stop_id", "binj"])


def test_time_since_arrival_and_outcome(arrival_clock_stops):
    """Both per-bin clocks are the bin centre minus the absolute stop clock."""
    stops, onsets = arrival_clock_stops
    df = _aggregate_with_clocks(stops, onsets)

    clocks = stops.set_index("stop_id")
    center_abs = (
        df["stop_id"].map(clocks["stop_start"]).to_numpy() + (df["binj"] + 0.5) * 0.1
    )
    assert df["time_since_last_well_arrival_s"].to_numpy() == pytest.approx(
        center_abs - df["stop_id"].map(clocks["last_well_arrival_time_s"]).to_numpy(),
        nan_ok=True,
    )
    assert df["time_since_outcome_known_s"].to_numpy() == pytest.approx(
        center_abs - df["stop_id"].map(clocks["outcome_known_time_s"]).to_numpy(),
        nan_ok=True,
    )

    # A NaN clock yields a NaN elapsed time -- never a silent zero.
    stop5 = df[df.stop_id == 5]
    assert stop5["time_since_last_well_arrival_s"].isna().all()
    assert stop5["time_since_outcome_known_s"].isna().all()

    # The arrival clock is non-negative for an at-well bout (the animal arrived
    # before the bout began); the outcome clock is negative before the boundary.
    stop1 = df[df.stop_id == 1]
    assert (stop1["time_since_last_well_arrival_s"] >= 0).all()
    assert stop1["time_since_outcome_known_s"].tolist() == pytest.approx(
        [-0.1, 0.0, 0.1, 0.2]
    )

    # The absolute clocks themselves are not carried into the design.
    assert "last_well_arrival_time_s" not in df.columns
    assert "outcome_known_time_s" not in df.columns
    assert "stop_start" not in df.columns
    assert not [c for c in LAG_COLUMNS if c in df.columns]


def test_reward_features_shift_at_outcome_boundary(arrival_clock_stops):
    """Bins at/after the boundary carry the history including the new outcome."""
    stops, onsets = arrival_clock_stops
    df = _aggregate_with_clocks(stops, onsets)
    stop1 = df[df.stop_id == 1]

    before, before_incomplete = _bout_start_features(FULL_LAGS)
    after, after_incomplete = _bout_start_features([1.0, *FULL_LAGS[:-1]])
    assert before != pytest.approx(after)

    # Left edges 100.0 / 100.1 precede the 100.15 boundary; 100.2 / 100.3 do not.
    pre = stop1[stop1.binj < 2]
    post = stop1[stop1.binj >= 2]
    for k in range(3):
        col = f"reward_history_rcos_{k}"
        assert pre[col].tolist() == pytest.approx([before[k]] * len(pre))
        assert post[col].tolist() == pytest.approx([after[k]] * len(post))
    assert pre["reward_history_incomplete"].tolist() == [before_incomplete] * len(pre)
    assert post["reward_history_incomplete"].tolist() == [after_incomplete] * len(post)


def test_reward_history_incomplete_flips_at_the_second_known_outcome(
    arrival_clock_stops,
):
    """A stop that learns its second outcome mid-bout stops being zero-filled."""
    stops, onsets = arrival_clock_stops
    df = _aggregate_with_clocks(stops, onsets)
    stop2 = df[df.stop_id == 2]

    # Boundary 200.05: only bin 0 (left edge 200.0) precedes it.
    assert stop2.loc[stop2.binj == 0, "reward_history_incomplete"].tolist() == [1.0]
    assert stop2.loc[stop2.binj > 0, "reward_history_incomplete"].tolist() == [0.0] * 3
    # The slowest component was zero-filled at bout start and is defined after.
    assert stop2.loc[stop2.binj == 0, "reward_history_rcos_2"].tolist() == [0.0]
    assert (stop2.loc[stop2.binj > 0, "reward_history_rcos_2"] > 0).all()


def test_arrival_spanning_and_out_of_stop_boundaries_are_not_shifted(
    arrival_clock_stops,
):
    """A stop is shifted unless it contains a later arrival or has no boundary."""
    stops, onsets = arrival_clock_stops
    df = _aggregate_with_clocks(stops, onsets)

    full, full_incomplete = _bout_start_features(FULL_LAGS)
    one, one_incomplete = _bout_start_features(ONE_LAG)
    # stop 6 contains stop 7's arrival; stop 4 and stop 7 already knew their
    # outcome at bout start; stop 5 has no completed arrival at all. None of
    # them moves off bout start.
    for stop_id, (expected, expected_incomplete) in {
        4: (full, full_incomplete),
        5: (one, one_incomplete),
        6: (full, full_incomplete),
        7: (full, full_incomplete),
    }.items():
        rows = df[df.stop_id == stop_id]
        for k in range(3):
            assert rows[f"reward_history_rcos_{k}"].tolist() == pytest.approx(
                [expected[k]] * len(rows)
            )
        assert rows["reward_history_incomplete"].tolist() == [
            expected_incomplete
        ] * len(rows)

    # stop 3 is ``mixed`` but contains no arrival, so it IS shifted at its
    # boundary (left edges 300.0 / 300.1 precede 300.15; 300.2 / 300.3 do not).
    after, after_incomplete = _bout_start_features([1.0, *FULL_LAGS[:-1]])
    stop3 = df[df.stop_id == 3]
    for k in range(3):
        col = f"reward_history_rcos_{k}"
        assert stop3.loc[stop3.binj < 2, col].tolist() == pytest.approx([full[k]] * 2)
        assert stop3.loc[stop3.binj >= 2, col].tolist() == pytest.approx([after[k]] * 2)
    assert (
        stop3["reward_history_incomplete"].tolist()
        == [full_incomplete] * 2 + [after_incomplete] * 2
    )

    # Counted once per stop, not once per bin, and only for the arrival-spanning
    # bout whose boundary actually falls inside it.
    assert df.attrs["n_stops_arrival_spanning"] == 1


def test_arrival_columns_absent_leaves_the_design_unchanged(two_stop_design):
    """A stop frame without the clocks produces exactly the pre-existing design."""
    stops, onsets = two_stop_design
    df = aggregate_to_timebins(
        stops, onsets, bin_s=0.1, tmax=0.5, reward_basis=True, self_history_lags=1
    )
    assert "time_since_last_well_arrival_s" not in df.columns
    assert "time_since_outcome_known_s" not in df.columns
    assert "n_stops_arrival_spanning" not in df.attrs

    # Frozen reference: value hash of the full design plus pinned cells, so a
    # change to any carried column or row order fails here rather than silently
    # moving every fitted per-bin coefficient.
    frozen = aggregate_to_timebins(
        FROZEN_STOPS,
        FROZEN_ONSETS,
        bin_s=0.1,
        tmax=0.5,
        reward_basis=True,
        self_history_lags=2,
        self_history_basis=2,
        cross_onset_times=FROZEN_CROSS_ONSETS,
        cross_history_lags=2,
        cross_history_basis=2,
        stop_covariates=("is_swr_stop",),
    )
    assert list(frozen.columns) == FROZEN_COLUMNS
    assert frozen.shape == (13, 21)
    digest = hashlib.sha256(
        frozen.sort_values(["stop_id", "binj"]).round(12).to_csv(index=False).encode()
    ).hexdigest()
    assert digest == FROZEN_DIGEST
    last = frozen[frozen.stop_id == 3].sort_values("binj").iloc[-1]
    assert last.binj == 4
    assert last.exposure == pytest.approx(0.02)
    assert last.tc == pytest.approx(0.45)
    assert last.onset_lag1 == pytest.approx(1.0)
    assert last.cross_hist_rcos_1 == pytest.approx(1.0)


# ── Frozen (stop x bin) design: the shape and values before the per-bin clocks ──
#
# Three stops exercising tmax truncation, a partial final bin, self- and
# cross-history lags with their raised-cosine projections, and an extra per-stop
# covariate. The digest is over the sorted, 12-decimal CSV of the whole frame.

FROZEN_STOPS = pd.DataFrame(
    {
        "animal": ["x", "x", "x"],
        "session_date": [1, 1, 1],
        "epoch": ["02_r1", "02_r1", "02_r1"],
        "stop_id": [1, 2, 3],
        "immobility_duration_s": [0.25, 0.85, 0.42],
        "reward_history_rcos_0": [1.0, 0.0, 0.5],
        "reward_history_rcos_1": [0.8, 0.2, 0.25],
        "reward_history_rcos_2": [0.6, 0.4, 0.125],
        "reward_history_incomplete": [0.0, 0.0, 1.0],
        "is_swr_stop": [True, False, True],
    }
)
FROZEN_ONSETS = pd.DataFrame(
    {"stop_id": [1, 1, 2, 3, 3], "t_since_stop_onset": [0.05, 0.15, 0.25, 0.02, 0.31]}
)
FROZEN_CROSS_ONSETS = pd.DataFrame(
    {"stop_id": [1, 2, 3], "t_since_stop_onset": [0.12, 0.05, 0.22]}
)
FROZEN_COLUMNS = [
    "animal",
    "session_date",
    "epoch",
    "stop_id",
    "reward_history_rcos_0",
    "reward_history_rcos_1",
    "reward_history_rcos_2",
    "reward_history_incomplete",
    "is_swr_stop",
    "binj",
    "tc",
    "exposure",
    "count",
    "onset_lag1",
    "onset_lag2",
    "hist_rcos_0",
    "hist_rcos_1",
    "cross_lag1",
    "cross_lag2",
    "cross_hist_rcos_0",
    "cross_hist_rcos_1",
]
FROZEN_DIGEST = "7a2311d8acfde43383573a07f6da8a08c7220eb6797a38a3bb962c3dc014fc3c"


# ── The optional branches say whether they ran ──

_DESIGN_LOGGER = "continuum_swr_replay.analysis.replay.per_bin_design"


def test_aggregate_logs_the_optional_branches_it_applied(arrival_clock_stops, caplog):
    """A stop table carrying the clock and shift inputs reports both as applied."""
    stops, onsets = arrival_clock_stops
    with caplog.at_level("DEBUG", logger=_DESIGN_LOGGER):
        _aggregate_with_clocks(stops, onsets)
    messages = [r.getMessage() for r in caplog.records if r.name == _DESIGN_LOGGER]
    assert messages == [
        "aggregate_to_timebins: per-bin arrival/outcome clocks applied; "
        "outcome-boundary reward-history shift applied"
    ]


def test_aggregate_logs_the_optional_branches_it_skipped(two_stop_design, caplog):
    """Without their inputs the branches are named as skipped, not left silent."""
    stops, onsets = two_stop_design
    with caplog.at_level("DEBUG", logger=_DESIGN_LOGGER):
        aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5, reward_basis=True)
        aggregate_to_timebins(stops, onsets, bin_s=0.1, tmax=0.5)
    messages = [r.getMessage() for r in caplog.records if r.name == _DESIGN_LOGGER]
    assert len(messages) == 2
    assert "arrival/outcome clocks skipped (stop table lacks" in messages[0]
    assert "reward-history shift skipped (stop table lacks" in messages[0]
    assert "reward-history shift not requested (reward_basis=False)" in messages[1]
