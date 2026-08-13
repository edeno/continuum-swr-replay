"""Regression tests for Figure 3's event-level SWR grouping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    EventDefinitionError,
)
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


def test_custom_nonlocal_threshold_accepts_labels_built_at_that_threshold():
    """Figure 3's masks must validate at the threshold that made the labels."""
    data = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "e1",
            "head_speed": 1.0,
            "non_local_event": [1, 1, 0, 0],
            "SWR_event": [2, 2, 0, 0],
            # Valid at 0.5, invalid at the 0.8 default.
            "Non-Local_Continuous_prob": [0.6, 0.6, 0.0, 0.0],
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=np.arange(4, dtype=float),
    )

    with pytest.raises(EventDefinitionError, match="probability-threshold contract"):
        _nonlocal_swr_event_bin_masks(data)

    swr, non_swr = _nonlocal_swr_event_bin_masks(data, nonlocal_threshold=0.5)
    assert np.flatnonzero(swr.to_numpy()).tolist() == [0, 1]
    assert not non_swr.any()
