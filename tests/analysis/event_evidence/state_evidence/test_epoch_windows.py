"""``epoch_windows``: the events and Local-rest controls of panel h as decode bins."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    epoch_windows,
)

TIME = 10.0 + 0.002 * np.arange(30)
EVENTS = pd.DataFrame(
    {
        "event_id": ["x", "y", "other_epoch"],
        "event_start_s": TIME[[2, 10, 0]],
        "event_end_s": TIME[[5, 11, 1]],
    }
)
PARTICIPATION = pd.DataFrame(
    {
        "event_id": ["x", "y", "y"],
        "kind": ["event", "event", "control"],
        "group": ["no SWR", "SWR", "SWR"],
        "start_s": [TIME[2], TIME[10], TIME[20]],
        "n_bins": [4, 2, 2],
    }
)


def test_windows_are_the_events_then_their_controls():
    windows = epoch_windows(TIME, EVENTS, PARTICIPATION)
    assert windows.to_dict("list") == {
        "window_id": [0, 1, 2],
        "event_id": ["x", "y", "y"],
        "kind": ["event", "event", "control"],
        "group": ["no SWR", "SWR", "SWR"],
        "s": [2, 10, 20],
        "e": [5, 11, 21],
    }


def test_a_measured_event_missing_from_the_event_table_raises():
    with pytest.raises(ValueError, match="not in the event table"):
        epoch_windows(TIME, EVENTS[EVENTS["event_id"] != "y"], PARTICIPATION)
