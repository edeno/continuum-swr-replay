"""Ripple power and MUA by decoder state, on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties import state_by_signal as sbs
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    rest_signal_counts,
)


def _frame() -> pd.DataFrame:
    """Six immobile bins: two Local, two non-local, one No-Spike, one none of
    them; one moving bin and one in excluded time (not counted)."""
    local = [0.9, 0.85, 0.05, 0.0, 0.1, 0.4, 0.9, 0.9]
    continuous = [0.05, 0.1, 0.5, 0.9, 0.0, 0.3, 0.05, 0.05]
    fragmented = [0.05, 0.05, 0.45, 0.05, 0.0, 0.2, 0.05, 0.05]
    no_spike = [
        1 - a - b - c for a, b, c in zip(local, continuous, fragmented, strict=True)
    ]
    return pd.DataFrame(
        {
            "Local_prob": local,
            "Non-Local_Continuous_prob": continuous,
            "Non-Local_Fragmented_prob": fragmented,
            "No-Spike_prob": no_spike,
            "SWR_zscore": [0.1, -0.1, 2.5, 3.0, 0.0, 1.0, 5.0, 5.0],
            "MUA_zscore": [0.0, 0.0, 2.0, 1.0, -1.0, 0.0, 5.0, 5.0],
            "head_speed": [0, 0, 0, 0, 0, 0, 10.0, 0],
            "in_excluded_time": [False] * 7 + [True],
        }
    )


def test_counts_each_bin_by_signal_bin_and_state_class():
    """Each immobile, analyzed bin once per signal in its class; the non-local
    counts are those of the stored run above 0.8, bin by bin."""
    frame = _frame()
    in_visit = np.ones(len(frame), bool)
    counts = sbs.state_signal_counts(frame, in_visit)
    ripple = counts[counts["signal"] == "ripple_power"]
    by_state = ripple.groupby("state")["n_bins"].sum()
    assert by_state.to_dict() == {"local": 2, "no_spike": 1, "non_local": 2, "other": 1}
    stored = rest_signal_counts(frame, in_visit)
    stored = stored[stored["probability_low"] >= 0.8]
    for signal in ("ripple_power", "mua"):
        mine = counts[(counts["signal"] == signal) & (counts["state"] == "non_local")]
        theirs = (
            stored[stored["signal"] == signal]
            .groupby("signal_low", sort=True)["n_bins"]
            .sum()
        )
        np.testing.assert_array_equal(
            mine.set_index("signal_low")["n_bins"].sort_index().to_numpy(),
            theirs.to_numpy(),
        )
    with pytest.raises(ValueError, match="one bool per bin"):
        sbs.state_signal_counts(frame, in_visit[:-1])


def _stored(epochs: dict) -> pd.DataFrame:
    """Rows as stored for the ripple-power signal, from
    ``{(animal, epoch): {(signal_low, state): n}}`` over the bins -1, 0, 1, 2, 3
    (each 1 z wide)."""
    edges = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    rows = []
    for (animal, epoch), counts in epochs.items():
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            for state in sbs.STATE_CLASSES:
                rows.append(
                    {"animal": animal, "session_date": 1, "epoch": epoch,
                     "signal": "ripple_power", "signal_low": low, "signal_high": high,
                     "state": state, "n_bins": counts.get((low, state), 0)}
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_discrimination_by_hand():
    """Epoch a1: Local 2 at -1 and 2 at 0, non-local 1 at 0 and 3 at 2. The
    area: the non-local bin at 0 beats 2 Local bins and ties 2 (2 + 1 = 3); each
    at 2 beats all 4 (12); (3 + 12) / 16. At 2 z: non-local 3/4, Local 0; to catch
    90 %, the level is 0 (all four), Local 2/4 above it; purity at 2 z 1, at 0
    4 / (4 + 2). Epoch a2 has no Local time: left out. Animal b: Local and
    non-local both 1 at 1 (a tie, area 1/2; at 2 z nothing)."""
    counts = _stored(
        {
            ("a", "1"): {(-1.0, "local"): 2, (0.0, "local"): 2, (0.0, "non_local"): 1,
                         (2.0, "non_local"): 3, (3.0, "other"): 7},
            ("a", "2"): {(2.0, "non_local"): 5},
            ("b", "1"): {(1.0, "local"): 1, (1.0, "non_local"): 1},
        }
    )  # fmt: skip
    stats = sbs.discrimination_statistics(counts, "ripple_power", 2.0)
    a = {
        "auc": 15 / 16, "nonlocal_share_at_level": 3 / 4, "local_share_at_level": 0.0,
        "catch_level": 0.0, "local_share_at_catch_level": 1 / 2,
        "nonlocal_purity_at_level": 1.0, "nonlocal_purity_at_catch_level": 4 / 6,
    }  # fmt: skip
    b = {
        "auc": 0.5, "nonlocal_share_at_level": 0.0, "local_share_at_level": 0.0,
        "catch_level": 1.0, "local_share_at_catch_level": 1.0,
        "nonlocal_purity_at_level": np.nan, "nonlocal_purity_at_catch_level": 0.5,
    }  # fmt: skip
    for measure in sbs.DISCRIMINATION_MEASURES:
        values = [v for v in (a[measure], b[measure]) if not np.isnan(v)]
        row = stats.loc[measure]
        assert row["animal_mean"] == pytest.approx(np.mean(values)), measure
        assert row["animal_min"] == pytest.approx(min(values)), measure
        assert row["animal_max"] == pytest.approx(max(values)), measure
    assert set(stats["n_epochs"]) == {2} and set(stats["n_animals"]) == {2}
    with pytest.raises(ValueError, match="not an edge"):
        sbs.discrimination_statistics(counts, "ripple_power", 2.5)


def test_each_class_distribution_is_the_mean_over_an_animals_epochs():
    """Animal a's non-local time: epoch 1 a quarter at 0 and three quarters at 2,
    epoch 2 all at 2: its percent at 2 is (75 + 100) / 2. Local only where
    there is Local time."""
    counts = _stored(
        {
            ("a", "1"): {(0.0, "non_local"): 1, (2.0, "non_local"): 3,
                         (-1.0, "local"): 4},
            ("a", "2"): {(2.0, "non_local"): 5},
        }
    )  # fmt: skip
    distribution = sbs.state_distribution(counts, "ripple_power")
    nonlocal_ = distribution[distribution["state"] == "non_local"].set_index(
        "signal_low"
    )["percent"]
    assert nonlocal_[2.0] == pytest.approx(87.5)
    assert nonlocal_[0.0] == pytest.approx(12.5)
    assert nonlocal_.sum() == pytest.approx(100.0)
    local = distribution[distribution["state"] == "local"].set_index("signal_low")
    assert local.loc[-1.0, "percent"] == pytest.approx(100.0)
