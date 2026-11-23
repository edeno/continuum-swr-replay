"""The held-out tetrode evidence of events at low-rate places, and events
against their controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.low_rate_places import (
    event_minus_control,
    events_outside_low_rate,
    low_rate_split_statistics,
    place_groups,
)


def _place_rate() -> pd.DataFrame:
    """Events 0 to 5: ranks 0.05, 0.5, 0.0, 0.95, 0.1 (at the cut: other)
    and 0.2."""
    return pd.DataFrame(
        {"event_id": range(6), "place_rate_rank": [0.05, 0.5, 0.0, 0.95, 0.1, 0.2]}
    )


def test_the_place_groups_cut_below_the_low_rate_rank():
    groups = place_groups(pd.Series([0, 1, 2, 3, 4, 5, 9]), _place_rate())
    assert list(groups[:6]) == ["low_rate", "other", "low_rate", "other", "other", "other"]
    # an event without a rank is in neither
    assert pd.isna(groups.iloc[6])
    with pytest.raises(ValueError, match="two rows"):
        place_groups(pd.Series([0]), pd.concat([_place_rate(), _place_rate()]))


def test_the_events_outside_the_low_rate_group_keep_the_unranked():
    events = pd.DataFrame({"event_id": [0, 1, 2, 3, 9]})
    assert list(events_outside_low_rate(events, _place_rate())["event_id"]) == [1, 3, 9]


def _windows() -> pd.DataFrame:
    """Animals a (epoch a1) and b (b1); each event and its control. Low-rate
    events: a1 0 (no SWR, 1.0), a1 2 (no SWR, 3.0), b1 4 is at the cut. Other:
    a1 1 (no SWR, 5.0), b1 3 (SWR, 2.0), b1 4 (no SWR, 7.0), b1 5 (no SWR,
    -1.0). Each control's value is its event's minus 0.5, event 5 has none."""
    events = pd.DataFrame(
        {
            "epoch_id": ["a1", "a1", "a1", "b1", "b1", "b1"],
            "animal": ["a", "a", "a", "b", "b", "b"],
            "event_id": [0, 1, 2, 3, 4, 5],
            "group": ["no SWR", "no SWR", "no SWR", "SWR", "no SWR", "no SWR"],
            "kind": "event",
            "score": [1.0, 5.0, 3.0, 2.0, 7.0, -1.0],
        }
    )
    controls = events.iloc[:5].assign(kind="control", score=lambda t: t["score"] - 0.5)
    return pd.concat([events, controls], ignore_index=True)


def test_the_split_reads_the_events_of_each_place_and_group():
    stats = low_rate_split_statistics(_windows(), _place_rate(), "score", n_boot=50)
    # low-rate no SWR: a only, median of 1 and 3
    low = stats.loc[("low_rate", "no SWR")]
    np.testing.assert_allclose(low["animals"], [2.0])
    assert low["n_animals_positive"] == 1
    # other no SWR: a 5; b median of 7 and -1 = 3; mean 4
    other = stats.loc[("other", "no SWR")]
    assert other["estimate"] == pytest.approx(4.0)
    assert stats.loc[("other", "SWR"), "estimate"] == pytest.approx(2.0)
    # the controls are not read
    assert other["n_events"] == 3
    assert ("low_rate", "SWR") not in stats.index


def test_each_event_is_paired_with_its_control():
    paired = event_minus_control(_windows(), "score")
    assert list(paired["kind"].unique()) == ["event"]
    np.testing.assert_allclose(
        paired["score_minus_control"], [0.5, 0.5, 0.5, 0.5, 0.5, np.nan]
    )
    doubled = pd.concat([_windows(), _windows().iloc[[0]]])
    with pytest.raises(ValueError, match="two event windows"):
        event_minus_control(doubled, "score")
