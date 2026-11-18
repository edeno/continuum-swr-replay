"""One held-out evidence window per canonical event, and its matched window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence.state_evidence import (
    STATES,
    event_windows,
)

N_TIME = 40
TIME = 100.0 + np.arange(N_TIME) * 0.002
EVENT_BINS = {"A": (2, 4, True), "B": (8, 10, False), "C": (19, 21, False),
              "D": (31, 32, False)}  # fmt: skip
"""Each event's first and last bin (inclusive) and its ``is_swr``."""


def _inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Local everywhere but the four events (P(Non-Local Continuous) 0.9 there);
    every bin immobile but bin 20 (in no regime) and bins 30-34 (running)."""
    probabilities = np.tile([0.97, 0.01, 0.01, 0.01], (N_TIME, 1))
    for first, last, _ in EVENT_BINS.values():
        probabilities[first : last + 1] = [0.05, 0.05, 0.9, 0.0]
    regime = np.full(N_TIME, "immobile", dtype="<U8")
    regime[20] = ""
    regime[30:35] = "running"
    events = pd.DataFrame(
        {
            "event_id": list(EVENT_BINS),
            "event_start_s": [TIME[first] for first, _, _ in EVENT_BINS.values()],
            "event_end_s": [TIME[last] for _, last, _ in EVENT_BINS.values()],
            "is_swr": [swr for _, _, swr in EVENT_BINS.values()],
            "overlap_class": ["swr_partial", "non_swr", "non_swr", "non_swr"],
        }
    )
    return probabilities, regime, events


def test_each_kept_event_is_scored_on_every_bin_and_labelled_by_is_swr():
    probabilities, regime, events = _inputs()
    windows, dropped = event_windows(TIME, probabilities, regime, events, seed=1)

    bouts = windows[windows.kind == "bout"].set_index("event_id")
    assert list(bouts.index) == ["A", "B"]
    assert bouts.loc["A", ["s", "e", "n_bins", "bout_len"]].tolist() == [2, 4, 3, 3]
    assert bouts.loc["B", ["s", "e", "bout_s", "bout_e"]].tolist() == [8, 10, 8, 10]
    assert bouts.loc["A", "state"] == "Non-Local | SWR"
    assert bouts.loc["B", "state"] == "Non-Local | no SWR"
    # the label's events before any is dropped: A alone is SWR; B, C, D are not
    assert bouts.n_found.tolist() == [1, 3]
    assert bouts.loc["A", "overlap_class"] == "swr_partial"  # carried over
    assert (bouts.event_bin_share == 1.0).all()
    assert np.allclose(bouts["p_Non-Local Continuous"], 0.9)
    assert set(windows.regime) == {"immobile"}
    # one partner pool per kind, whatever the label
    assert set(windows.pool) == {
        "bout|Non-Local|immobile",
        "baseline|Non-Local|immobile",
    }

    assert dropped.set_index("event_id").reason.to_dict() == {
        "C": "a bin in no regime",
        "D": "a running bin",
    }
    assert dropped.is_swr.tolist() == [False, False]


def test_each_event_has_one_matched_window_of_its_length_wholly_immobile():
    probabilities, regime, events = _inputs()
    windows, _ = event_windows(TIME, probabilities, regime, events, seed=1)

    bouts = windows[windows.kind == "bout"].set_index("bout_s")
    matched = windows[windows.kind == "baseline"].set_index("matched_bout_s")
    assert sorted(matched.index) == sorted(bouts.index)
    for bout_s, row in matched.iterrows():
        assert row.n_bins == bouts.loc[bout_s, "n_bins"]
        assert (regime[row.s : row.e + 1] == "immobile").all()
        assert row.event_id == bouts.loc[bout_s, "event_id"]
        assert row.state == bouts.loc[bout_s, "state"]
        in_event = np.zeros(N_TIME, dtype=bool)
        for first, last, _ in EVENT_BINS.values():
            in_event[first : last + 1] = True
        assert row.event_bin_share == in_event[row.s : row.e + 1].mean()
    # mean posteriors are the window's own
    for _, row in windows.iterrows():
        for j, state in enumerate(STATES):
            assert row[f"p_{state}"] == pytest.approx(
                probabilities[row.s : row.e + 1, j].mean()
            )


def test_the_matched_draw_is_seeded():
    probabilities, regime, events = _inputs()
    first, _ = event_windows(TIME, probabilities, regime, events, seed=1)
    again, _ = event_windows(TIME, probabilities, regime, events, seed=1)
    pd.testing.assert_frame_equal(first, again)
    starts = {
        seed: tuple(
            event_windows(TIME, probabilities, regime, events, seed=seed)[0]
            .query("kind == 'baseline'")
            .s
        )
        for seed in range(6)
    }
    assert len(set(starts.values())) > 1


def test_events_are_taken_in_time_order_whatever_the_rows_order():
    probabilities, regime, events = _inputs()
    windows, _ = event_windows(TIME, probabilities, regime, events, seed=1)
    shuffled, _ = event_windows(TIME, probabilities, regime, events.iloc[::-1], seed=1)
    pd.testing.assert_frame_equal(windows, shuffled)


def test_an_event_off_the_decode_grid_raises():
    probabilities, regime, events = _inputs()
    events.loc[0, "event_start_s"] += 0.001  # half a bin
    with pytest.raises(ValueError, match="do not start and end on the decode"):
        event_windows(TIME, probabilities, regime, events, seed=1)


def test_overlapping_events_raise():
    probabilities, regime, events = _inputs()
    events.loc[1, "event_start_s"] = TIME[4]
    with pytest.raises(ValueError, match="disjoint"):
        event_windows(TIME, probabilities, regime, events, seed=1)


def test_an_event_bin_below_the_non_local_threshold_raises():
    probabilities, regime, events = _inputs()
    probabilities[9] = [0.97, 0.01, 0.01, 0.01]  # a bin of B is Local
    with pytest.raises(ValueError, match="not this decode's"):
        event_windows(TIME, probabilities, regime, events, seed=1)


def test_a_carried_column_must_not_be_one_it_writes():
    probabilities, regime, events = _inputs()
    with pytest.raises(ValueError, match="carry columns"):
        event_windows(TIME, probabilities, regime, events.assign(state="x"), seed=1)


def test_no_events_give_no_windows():
    probabilities, regime, events = _inputs()
    windows, dropped = event_windows(TIME, probabilities, regime, events[:0], seed=1)
    assert windows.empty and dropped.empty
