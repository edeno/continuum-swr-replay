"""Regression tests for Figure 3's event-level SWR grouping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.visualization.figure_panels import (
    _nonlocal_swr_event_bin_masks,
)


def test_partial_overlap_assigns_whole_event_and_rejects_mobile_swr():
    data = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "e1",
            "head_speed": [1.0] * 6 + [5.0] + [1.0] * 4,
            "non_local_event": [1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2],
            "SWR_event": [0, 0, 3, 3, 3, 0, 4, 4, 4, 4, 4],
            "Non-Local_Continuous_prob": [0.9] * 4
            + [0.1] * 3
            + [0.9] * 4,
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=np.arange(11, dtype=float),
    )

    swr, non_swr = _nonlocal_swr_event_bin_masks(data)

    assert np.flatnonzero(swr.to_numpy()).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(non_swr.to_numpy()).tolist() == [7, 8, 9, 10]
    assert not (swr & non_swr).any()
