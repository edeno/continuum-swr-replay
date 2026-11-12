"""Regression test for episode labeling preserving immobile gaps.

The pre-fix code filtered ``data`` to ``head_speed > threshold`` and
labeled ``Local_prob < threshold`` (or ``nonlocal_prob > threshold``)
on the FILTERED frame. ``scipy.ndimage.label`` then saw the moving
rows as contiguous, merging events separated by immobile gaps into
one big episode. With times [0, 1, 2] and speeds [5, 0, 5], two
1-bin events became a single 3-second event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.quality_metrics import (
    compute_local_state_collapse_details,
    compute_non_local_during_movement,
    compute_state_fraction_during_movement,
)


def _three_row_frame() -> pd.DataFrame:
    times = np.array([0.0, 1.0, 2.0])
    return pd.DataFrame(
        {
            "animal": ["mouse"] * 3,
            "day": [20200101] * 3,
            "epoch": ["02_r1"] * 3,
            "head_speed": [5.0, 0.0, 5.0],
            # collapse + nonlocal only at moving bins.
            "Local_prob": [0.0, 1.0, 0.0],
            "No-Spike_prob": [0.0, 0.0, 0.0],
            "Non-Local_Continuous_prob": [1.0, 0.0, 1.0],
            "Non-Local_Fragmented_prob": [0.0, 0.0, 0.0],
            "ahead_behind_distance": [0.0, 0.0, 0.0],
            "hpd": [0.1, 0.1, 0.1],
            "in_excluded_time": [False] * 3,
        },
        index=times,
    )


class TestImmobileGapBreaksEpisodes:
    def test_state_fraction_collapse_episodes_split_by_immobile_gap(self):
        out = compute_state_fraction_during_movement(
            _three_row_frame(),
            speed_threshold=4.0,
            local_prob_threshold=0.5,
            sampling_frequency=1.0,
        )
        row = out.iloc[0]
        assert int(row["n_collapse_episodes"]) == 2, (
            f"expected 2 episodes, got {int(row['n_collapse_episodes'])}; "
            "immobile row should break the run"
        )
        # Each episode is one bin at sampling_frequency=1Hz → 1.0s.
        assert row["max_collapse_duration_sec"] == 1.0, (
            f"expected 1.0s, got {row['max_collapse_duration_sec']}"
        )

    def test_non_local_during_movement_split_by_immobile_gap(self):
        out = compute_non_local_during_movement(
            _three_row_frame(),
            speed_threshold=4.0,
            non_local_threshold=0.5,
            sampling_frequency=1.0,
        )
        row = out.iloc[0]
        assert int(row["n_nonlocal_movement_events"]) == 2, (
            f"expected 2 events, got {int(row['n_nonlocal_movement_events'])}"
        )

    def test_collapse_details_split_by_immobile_gap(self):
        details = compute_local_state_collapse_details(
            _three_row_frame(),
            speed_threshold=4.0,
            local_prob_threshold=0.5,
            sampling_frequency=1.0,
        )
        # Two 1-bin episodes, not one 3-second episode.
        assert len(details) == 2
        # Each episode has a single bin so start_time == end_time.
        for _, row in details.iterrows():
            assert row["start_time"] == row["end_time"]
