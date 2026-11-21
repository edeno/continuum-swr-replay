"""The well-visit time-bin design (``replay_models/visit_design.py``).

One hand-built session, in 0.1 s bins, with two visits (port experiences):
visit 0 at well 1 from 10.0 to 13.0 s, its omission known at 10.25 s, and
visit 1 at well 2 from 20.0 to 24.0 s, its reward known at 21.05 s (first
poke-in to final poke-out), and excluded time from 22.0 to 22.2 s. The animal
is immobile in five stops:

====  ============  ==================================  ====================
stop  span (s)      where                               analyzed in-visit s
====  ============  ==================================  ====================
0     9.5 - 10.5    starts before visit 0's poke-in     0.5 (10.0 - 10.5)
1     11.0 - 12.0   inside visit 0                      1.0
2     12.5 - 13.3   crosses visit 0's final poke-out    0.5 (12.5 - 13.0)
3     15.0 - 16.0   on the track, between the visits    0.0
4     20.0 - 23.5   inside visit 1, its last stop,      3.3 (less 22.0 - 22.2)
                    ends 0.5 s before the poke-out
====  ============  ==================================  ====================

Events (start - end, kind, label):

- 10.2 - 10.3 No SWR, in_visit (in stop 0's in-visit part)
- 11.25 - 11.45 SWR, in_visit
- 11.55 - 11.6 No SWR, in_visit (covers half of the bin 11.5 - 11.6)
- 12.9 - 13.1 SWR, spans_visit_end (crosses the final poke-out)
- 15.2 - 15.3 SWR, track_leaf_arm (outside the visits)
- 20.3 - 20.35 SWR, in_visit
- 20.5 - 20.7 SWR, in_visit (starts on a bin edge)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.constants import IN_WELL_VISIT, STARTS_IN_WELL_VISIT
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    excluded_intervals_by_session,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context,
)
from continuum_swr_replay.analysis.replay_models.visit_design import (
    EVENT_KINDS,
    VISIT_BIN_COLUMNS,
    visit_timebins,
)
from continuum_swr_replay.intervals import intersect, subtract, total_duration
from continuum_swr_replay.paths import table_path
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from scripts.replay_models.rate.describe_visit_clocks import session_port_experiences

BIN_S = 0.1
SESSION = {"animal": "rat", "session_date": "20200101", "epoch": "02_r1"}
VISITS = np.array([[10.0, 13.0], [20.0, 24.0]])
EXCLUDED = pd.DataFrame({"start_time": [22.0], "end_time": [22.2]}).assign(**SESSION)
LAGS = [f"known_reward_outcome_lag_{j}" for j in range(1, 11)]


def _stop_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": [100, 101, 102, 103, 104],
            "stop_start": [9.5, 11.0, 12.5, 15.0, 20.0],
            "stop_end": [10.5, 12.0, 13.3, 16.0, 23.5],
            "visit_context": [
                "spans_visit_start",
                "in_visit",
                "spans_visit_end",
                "track_leaf_arm",
                "in_visit",
            ],
            "overlapping_visit_id": pd.array([0, 0, 0, pd.NA, 1], dtype="Int64"),
            "visit_first_poke_in_s": [10.0, 10.0, 10.0, np.nan, 20.0],
            "visit_last_poke_out_s": [13.0, 13.0, 13.0, np.nan, 24.0],
            "in_visit_immobile_s": [0.5, 1.0, 0.5, 0.0, 3.3],
            "visit_stretch_index": pd.array([0, 1, 2, pd.NA, 0], dtype="Int64"),
        }
    ).assign(**SESSION)


def _port_experiences() -> pd.DataFrame:
    """The two visits: an omission at well 1, then a reward at well 2."""
    return pd.DataFrame(
        {
            "experience_id": [0, 1],
            "movement_trial_number": pd.array([pd.NA, 1], dtype="Int64"),
            "is_initial_experience": [True, False],
            "is_boundary_derived": [False, False],
            "outcome_observed": [True, True],
            "port": [1, 2],
            "patch": [1, 2],
            "arrival_time": VISITS[:, 0],
            "departure_time": VISITS[:, 1],
            "reward_on_time": [np.nan, 21.05],
            "outcome_known_time": [10.25, 21.05],
            "is_reward": [False, True],
        }
    )


def _events() -> pd.DataFrame:
    rows = [
        (10.2, 10.3, False, "in_visit", 0),
        (11.25, 11.45, True, "in_visit", 0),
        (11.55, 11.6, False, "in_visit", 0),
        (12.9, 13.1, True, "spans_visit_end", 0),
        (15.2, 15.3, True, "track_leaf_arm", None),
        (20.3, 20.35, True, "in_visit", 1),
        (20.5, 20.7, True, "in_visit", 1),
    ]
    events = pd.DataFrame(
        rows,
        columns=["event_start_s", "event_end_s", "is_swr", "visit_context", "visit"],
    )
    visit = events["visit"].to_numpy(dtype=float)
    inside = ~np.isnan(visit)
    index = np.where(inside, visit, 0).astype(int)
    start = events["event_start_s"].to_numpy()
    events["well_visit_id"] = pd.array(events["visit"], dtype="Int64")
    events["time_since_visit_start_s"] = np.where(
        inside, start - VISITS[index, 0], np.nan
    )
    events["time_until_visit_end_s"] = np.where(
        inside, VISITS[index, 1] - start, np.nan
    )
    return events.drop(columns="visit").assign(**SESSION)


def _bins(kind: str | None, events: pd.DataFrame | None = None) -> pd.DataFrame:
    return visit_timebins(
        _stop_table(),
        _events() if events is None else events,
        _port_experiences(),
        kind=kind,
        bin_s=BIN_S,
        excluded_intervals=EXCLUDED,
    )


def _bin_at(bins: pd.DataFrame, start: float) -> pd.Series:
    rows = bins[np.isclose(bins["bin_start_s"], start)]
    assert len(rows) == 1, start
    return rows.iloc[0]


def _in_visit_event_seconds(events: pd.DataFrame) -> float:
    """Seconds of the events inside the stops' analyzed in-visit time."""
    stops = _stop_table()
    analyzed = subtract(
        stops[["stop_start", "stop_end"]].to_numpy(),
        EXCLUDED[["start_time", "end_time"]].to_numpy(),
    )
    return total_duration(
        intersect(
            intersect(analyzed, VISITS),
            events[["event_start_s", "event_end_s"]].to_numpy(),
        )
    )


def test_clocks_and_labels_per_bin() -> None:
    bins = _bins("SWR")
    assert list(bins.columns) == list(VISIT_BIN_COLUMNS)
    inside = bins[bins["visit_context"] == IN_WELL_VISIT]
    visit = inside["well_visit_id"].to_numpy(dtype=int)
    s = inside["time_since_visit_start_s"].to_numpy()
    u = inside["time_until_visit_end_s"].to_numpy()
    m = inside["time_until_stretch_end_s"].to_numpy()
    # s + u is the visit's length in every bin
    np.testing.assert_allclose(s + u, VISITS[visit, 1] - VISITS[visit, 0])
    # m is the time to the stretch's end clipped at the poke-out, so m <= u
    center = (inside["bin_start_s"] + inside["bin_end_s"]).to_numpy() / 2
    stop_end = inside["stop_id"].map(_stop_table().set_index("stop_id")["stop_end"])
    np.testing.assert_allclose(
        m, np.minimum(stop_end.to_numpy(), VISITS[visit, 1]) - center
    )
    assert np.all(m <= u + 1e-12)
    # the stop that crosses the poke-out: m equals u; visit 1's last stop ends
    # 0.5 s before its poke-out: m is u less 0.5 s
    crossing = inside["stop_id"].eq(102).to_numpy()
    np.testing.assert_allclose(m[crossing], u[crossing])
    last = inside["stop_id"].eq(104).to_numpy()
    np.testing.assert_allclose(u[last] - m[last], 0.5)
    # the in-visit bins of each stop and their stretch index (the No SWR
    # design, whose events drop no bin at a stop's ends)
    no_swr = _bins("No SWR")
    extent = no_swr[no_swr["visit_context"] == IN_WELL_VISIT].groupby("stop_id")
    in_visit_s = extent["bin_end_s"].max() - extent["bin_start_s"].min()
    np.testing.assert_allclose(
        in_visit_s.loc[[100, 101, 102, 104]], [0.5, 1.0, 0.5, 3.5]
    )
    assert inside.groupby("stop_id")["visit_stretch_index"].first().to_dict() == {
        100: 0,
        101: 1,
        102: 2,
        104: 0,
    }
    # no bin crosses a visit boundary or touches excluded time
    for start, end in VISITS:
        assert not ((bins["bin_start_s"] < start) & (bins["bin_end_s"] > start)).any()
        assert not ((bins["bin_start_s"] < end) & (bins["bin_end_s"] > end)).any()
    assert not ((bins["bin_start_s"] < 22.2) & (bins["bin_end_s"] > 22.0)).any()
    # bins outside the visits: their stop's label, the time after a final
    # poke-out "after_visit_at_well"; no visit, no clocks
    outside = bins[bins["visit_context"] != IN_WELL_VISIT]
    assert outside.groupby("stop_id")["visit_context"].unique().map(list).to_dict() == {
        100: ["spans_visit_start"],
        102: ["after_visit_at_well"],
        103: ["track_leaf_arm"],
    }
    assert outside["well_visit_id"].isna().all()
    assert outside["visit_stretch_index"].isna().all()
    clocks = [
        "time_since_visit_start_s",
        "time_until_visit_end_s",
        "time_until_stretch_end_s",
    ]
    assert outside[clocks].isna().all().all()
    assert not inside[clocks].isna().any().any()
    # the selection keeps exactly the in-visit bins
    selected = select_visit_context(bins)
    pd.testing.assert_frame_equal(selected, inside.reset_index(drop=True))


def test_at_risk_excludes_ongoing_events() -> None:
    no_swr = _bins("No SWR")
    swr = _bins("SWR")
    # the No SWR event 11.55 - 11.6 covers half of its bin
    assert _bin_at(no_swr, 11.5)["exposure"] == pytest.approx(0.05)
    assert _bin_at(no_swr, 11.5)["count"] == 1
    # and is nothing to the SWR design
    assert _bin_at(swr, 11.5)["exposure"] == pytest.approx(BIN_S)
    assert _bin_at(swr, 11.5)["count"] == 0
    # the SWR event 11.25 - 11.45: half of the bin it starts in, the next bin
    # wholly inside it (dropped), half of the one it ends in
    assert _bin_at(swr, 11.2)["exposure"] == pytest.approx(0.05)
    assert _bin_at(swr, 11.2)["count"] == 1
    assert not np.isclose(swr["bin_start_s"], 11.3).any()
    assert _bin_at(swr, 11.4)["exposure"] == pytest.approx(0.05)
    assert _bin_at(no_swr, 11.3)["exposure"] == pytest.approx(BIN_S)
    # an event that starts on a bin edge (20.5) is counted in the bin that
    # ends there, which holds the at-risk time before it
    assert _bin_at(swr, 20.4)["count"] == 1
    assert _bin_at(swr, 20.4)["exposure"] == pytest.approx(BIN_S)
    assert not np.isclose(swr["bin_start_s"], 20.5).any()


def test_reward_lags_are_the_outcomes_known_at_each_bin() -> None:
    """A bin's lags are the outcomes known at its left edge: the visit's own
    outcome is lag 1 from the first bin that starts at or after it became
    known, in the stop that began before the poke-in too."""
    bins = _bins("SWR")
    start = bins["bin_start_s"].to_numpy()
    lag1, lag2, lag3 = (bins[LAGS[j]].to_numpy() for j in range(3))
    # nothing known before visit 0's omission at 10.25 s
    before = start < 10.25
    assert np.isnan(bins.loc[before, LAGS].to_numpy()).all()
    # stop 0 started at 9.5 s, before visit 0's poke-in at 10.0 s: its bins
    # from 10.3 s carry the omission it became known of (the bin 10.2 - 10.3
    # holds the boundary and keeps what was known at 10.2 s)
    stop0 = bins[bins["stop_id"] == 100].set_index("bin_start_s")[LAGS[0]]
    assert np.isnan(stop0[stop0.index < 10.25]).all()
    assert stop0[stop0.index > 10.25].eq(0.0).all()
    assert (stop0.index > 10.25).sum() == 2
    # visit 1's reward at 21.05 s: lag 1 rewarded and lag 2 the omission from
    # the first bin that starts after it; the bin that holds it keeps lag 1
    middle = (start >= 10.25) & (start < 21.05)
    assert (lag1[middle] == 0.0).all() and np.isnan(lag2[middle]).all()
    after = start >= 21.05
    assert after.any()
    assert (lag1[after] == 1.0).all() and (lag2[after] == 0.0).all()
    assert np.isnan(lag3[after]).all()
    assert lag1[np.isclose(start, 21.0)].item() == 0.0
    assert lag1[np.isclose(start, 21.1)].item() == 1.0
    # one stop contains a first poke-in after its start: stop 0
    assert bins.attrs["n_stops_arrival_spanning"] == 1


def test_visits_that_disagree_with_the_port_experiences_are_refused() -> None:
    stops = _stop_table()
    stops.loc[0, "visit_first_poke_in_s"] = 9.0
    with pytest.raises(ValueError, match="port experiences"):
        visit_timebins(
            stops,
            _events(),
            _port_experiences(),
            kind="SWR",
            bin_s=BIN_S,
            excluded_intervals=EXCLUDED,
        )


@pytest.mark.parametrize(
    ("kind", "exposure"), [("SWR", 5.3 - 0.55), ("No SWR", 5.3 - 0.15)]
)
def test_exposure_sums_to_in_visit_immobile_seconds(kind: str, exposure: float) -> None:
    """In-visit exposure plus the kind's in-event time inside the in-visit
    analyzed time is the stops' in_visit_immobile_s, 5.3 s."""
    bins = _bins(kind)
    events = _events()
    kind_events = events[events["is_swr"] == EVENT_KINDS[kind]]
    in_visit = bins.loc[bins["visit_context"] == IN_WELL_VISIT, "exposure"].sum()
    # by hand: SWR 0.2 + 0.1 (12.9 - 13.0) + 0.05 + 0.2; No SWR 0.1 + 0.05
    assert _in_visit_event_seconds(kind_events) == pytest.approx(5.3 - exposure)
    assert in_visit == pytest.approx(exposure)
    assert in_visit + _in_visit_event_seconds(kind_events) == pytest.approx(
        _stop_table()["in_visit_immobile_s"].sum()
    )


@pytest.mark.parametrize("kind", sorted(EVENT_KINDS))
def test_same_events_as_content(kind: str) -> None:
    """The in-visit bins count exactly the kind's events of the default set,
    each once, in the bin that holds its start."""
    bins = _bins(kind)
    events = _events()
    kind_events = events[events["is_swr"] == EVENT_KINDS[kind]]
    default = select_visit_context(kind_events)
    selected = select_visit_context(bins)
    assert selected["count"].sum() == len(default)
    # each default event's start lies in one selected bin with a count
    starts = default["event_start_s"].to_numpy()
    holds = (selected["bin_start_s"].to_numpy()[:, None] < starts[None, :]) & (
        starts[None, :] <= selected["bin_end_s"].to_numpy()[:, None]
    )
    assert (holds.sum(axis=0) == 1).all()
    assert (selected["count"].to_numpy() == holds.sum(axis=1)).all()
    # no other event is counted in them; every event of the kind is counted once
    assert (selected["count"] > 0).sum() == len(default)
    assert bins["count"].sum() == len(kind_events)
    assert bins.attrs["n_starts_not_at_risk"] == 0


def test_event_at_the_start_of_a_stop_has_no_at_risk_time() -> None:
    """An event that starts on a stop's first sample (a piece start) has no
    at-risk time before it in any bin: it is not counted, and ``attrs`` says
    so."""
    events = pd.concat(
        [
            _events(),
            pd.DataFrame(
                {
                    "event_start_s": [11.0],
                    "event_end_s": [11.2],
                    "is_swr": [True],
                    "visit_context": ["in_visit"],
                    "well_visit_id": pd.array([0], dtype="Int64"),
                    "time_since_visit_start_s": [1.0],
                    "time_until_visit_end_s": [2.0],
                }
            ).assign(**SESSION),
        ],
        ignore_index=True,
    )
    bins = _bins("SWR", events)
    assert bins.attrs["n_starts_not_at_risk"] == 1
    assert bins.attrs["starts_not_at_risk_s"] == [11.0]
    assert bins.attrs["n_starts_not_at_risk_by_reason"]["stop_start"] == 1
    assert not np.isclose(bins["bin_start_s"], 11.0).any()
    assert bins["count"].sum() == 5


def _with_swr_events(*rows: tuple[float, float, str, int]) -> pd.DataFrame:
    """The module's events plus SWR events ``(start, end, label, visit)``."""
    extra = pd.DataFrame(
        {
            "event_start_s": [row[0] for row in rows],
            "event_end_s": [row[1] for row in rows],
            "is_swr": True,
            "visit_context": [row[2] for row in rows],
            "well_visit_id": pd.array([row[3] for row in rows], dtype="Int64"),
            "time_since_visit_start_s": [row[0] - VISITS[row[3], 0] for row in rows],
            "time_until_visit_end_s": [VISITS[row[3], 1] - row[0] for row in rows],
        }
    ).assign(**SESSION)
    return pd.concat([_events(), extra], ignore_index=True)


def test_a_start_at_the_start_of_any_piece_has_no_at_risk_time() -> None:
    """A start is lost at the start of any piece of analyzed time: a stop's
    first sample, a poke-in, or the end of excluded time; ``attrs`` counts
    each reason."""
    events = _with_swr_events(
        (10.0, 10.11, "in_visit", 0),  # on visit 0's poke-in, inside stop 0
        (11.0, 11.2, "in_visit", 0),  # on stop 1's first sample
        (22.2, 22.35, "in_visit", 1),  # where excluded time ends, in stop 4
    )
    bins = _bins("SWR", events)
    assert bins.attrs["n_starts_not_at_risk"] == 3
    np.testing.assert_allclose(bins.attrs["starts_not_at_risk_s"], [10.0, 11.0, 22.2])
    assert bins.attrs["n_starts_not_at_risk_by_reason"] == {
        "stop_start": 1,
        "poke_in": 1,
        "poke_out": 0,
        "excluded_end": 1,
        "after_event": 0,
    }
    assert bins["count"].sum() == 5


def test_a_label_that_disagrees_with_the_visits_is_refused() -> None:
    events = _events()
    events.loc[events["event_start_s"] == 15.2, "visit_context"] = "in_visit"
    with pytest.raises(ValueError, match="disagrees"):
        _bins("SWR", events)


def test_one_kind_or_every_event() -> None:
    every = _bins(None)
    assert every["count"].sum() == len(_events())
    with pytest.raises(ValueError, match="kind"):
        _bins("MUA")


@pytest.mark.real_data
def test_one_real_epoch() -> None:
    """On one epoch of the canonical tables: the in-visit bins count the
    default-set events of each kind (less those with no at-risk time before
    them, which ``attrs`` lists), and exposure plus in-event time is the stop
    table's in_visit_immobile_s."""
    session = {"animal": "chimi", "session_date": "20200212", "epoch": "02_r1"}

    def _read(name: str) -> pd.DataFrame:
        table = read_swr_table(table_path(name))
        keep = np.logical_and.reduce(
            [table[column].astype(str) == value for column, value in session.items()]
        )
        return table.loc[keep].reset_index(drop=True)

    stops, events = _read("stop_table"), _read("event_table")
    assert len(stops) and len(events)
    excluded = excluded_intervals_by_session(stops)
    analyzed = subtract(
        stops[["stop_start", "stop_end"]].to_numpy(dtype=float),
        excluded[["start_time", "end_time"]].to_numpy(dtype=float),
    )
    placed = stops["overlapping_visit_id"].notna()
    visits = (
        stops.loc[placed, ["visit_first_poke_in_s", "visit_last_poke_out_s"]]
        .drop_duplicates()
        .to_numpy(dtype=float)
    )
    experiences = session_port_experiences(**session)
    for kind, is_swr in EVENT_KINDS.items():
        bins = visit_timebins(stops, events, experiences, kind=kind)
        kind_events = events[events["is_swr"] == is_swr]
        default = kind_events[kind_events["visit_context"].isin(STARTS_IN_WELL_VISIT)]
        lost = default["event_start_s"].isin(bins.attrs["starts_not_at_risk_s"]).sum()
        selected = select_visit_context(bins)
        assert selected["count"].sum() + lost == len(default)
        assert lost <= 0.01 * len(default)
        in_event = total_duration(
            intersect(
                intersect(analyzed, visits),
                kind_events[["event_start_s", "event_end_s"]].to_numpy(dtype=float),
            )
        )
        assert selected["exposure"].sum() + in_event == pytest.approx(
            stops["in_visit_immobile_s"].sum(),
            abs=1e-5 * bins.attrs["n_bins_not_at_risk"] + 1e-9,
        )
        assert set(
            selected[list(SESSION_COLUMNS)].astype(str).itertuples(index=False)
        ) == {tuple(session.values())}


def test_events_and_excluded_time_name_their_session() -> None:
    """The events and the excluded time are matched to the stop table's session
    by its columns, compared as text: an integer ``session_date`` is the same
    session as its string, a frame without the session columns is refused, and
    a non-empty frame with no row of the session is refused rather than read as
    a session with no events."""
    expected = _bins("SWR")
    as_int = _events().assign(session_date=int(SESSION["session_date"]))
    pd.testing.assert_frame_equal(_bins("SWR", as_int), expected)
    with pytest.raises(ValueError, match="session_date"):
        _bins("SWR", _events().drop(columns="session_date"))
    with pytest.raises(ValueError, match="no row of the session"):
        _bins("SWR", _events().assign(epoch="04_r2"))
    with pytest.raises(ValueError, match="no row of the session"):
        visit_timebins(
            _stop_table(),
            _events(),
            _port_experiences(),
            kind="SWR",
            bin_s=BIN_S,
            excluded_intervals=EXCLUDED.assign(**{**SESSION, "epoch": "04_r2"}),
        )
