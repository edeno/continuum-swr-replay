"""The non-local event bins split by SWR overlap, and their normalized position."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.posterior_analysis import (
    normalized_replay_position_by_swr,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_detection import (
    nonlocal_event_bins_by_swr,
)


def _frame() -> pd.DataFrame:
    # A frame as the read hands it over: the speed rule and the probability
    # contract are the read's (analysed_time.with_swr_source), so every label
    # here is an eligible event. Event 1 overlaps SWR 3 in its last two bins.
    return pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "e1",
            "head_speed": 1.0,
            "non_local_event": [1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2],
            "SWR_event": [0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0],
            "Non-Local_Continuous_prob": [0.9] * 4 + [0.1] * 3 + [0.9] * 4,
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=np.arange(11, dtype=float),
    )


def test_partial_overlap_assigns_whole_event():
    swr, non_swr = nonlocal_event_bins_by_swr(_frame())

    assert np.flatnonzero(swr.to_numpy()).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(non_swr.to_numpy()).tolist() == [7, 8, 9, 10]
    assert not (swr & non_swr).any()


def test_position_is_a_fraction_of_the_session_track_and_bins_follow_the_median():
    # Four sessions, tracks of 100, 200, 300 and 1000 cm: the median track is
    # 250 cm, 125 two-cm bins (the mean, 400 cm, would give 200).
    frames = []
    for epoch, track in (("e1", 100.0), ("e2", 200.0), ("e3", 300.0), ("e4", 1000.0)):
        frame = _frame().assign(epoch=epoch, max_linear_position=track)
        frame["map_estimate"] = np.arange(11, dtype=float) * track / 20.0
        frames.append(frame)
    data = pd.concat(frames)

    positions, n_bins = normalized_replay_position_by_swr(data)

    assert n_bins == 125
    # Each session's event-1 bins (0-3) are at 0, 1/20, 2/20 and 3/20 of its
    # track, and its event-2 bins (7-10) at 7/20 ... 10/20.
    np.testing.assert_allclose(positions["swr"], np.tile([0, 1, 2, 3], 4) / 20.0)
    np.testing.assert_allclose(positions["non_swr"], np.tile([7, 8, 9, 10], 4) / 20.0)
