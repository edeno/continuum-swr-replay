"""``start_and_rest_theta`` on a 20-bin synthetic decode frame."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties import (
    start_and_rest_theta as module,
)
from continuum_swr_replay.analysis.event_tables.trajectory import brain_state_means

RUNNING_THETA = 0.5
ENVIRONMENT = SimpleNamespace(
    place_bin_centers_nodes_df_=None, distance_between_nodes_=None
)


def frame() -> pd.DataFrame:
    """Event 1 in bins 2-5, its first MAP NaN; event 2's label split over bins
    9-10 and 13-14; one NaN theta bin inside each event; rest elsewhere (the
    longest rest run, bins 15-19, holds a window of either length)."""
    label = np.zeros(20, dtype=int)
    label[2:6] = 1
    label[[9, 10, 13, 14]] = 2
    theta = np.linspace(-1.0, 1.0, 20)
    theta[4] = np.nan
    theta[14] = np.nan
    map_estimate = np.arange(20, dtype=float) * 10.0 + 100.0
    map_estimate[2] = np.nan
    return pd.DataFrame(
        {
            "non_local_event": label,
            "map_estimate": map_estimate,
            "linear_position": np.full(20, 5.0),
            "theta_power_zscore": theta,
        },
        index=pd.Index(np.arange(20) * 0.002, name="time"),
    )


@pytest.fixture
def stubbed(monkeypatch) -> None:
    """The regimes, the brain-state reference and the track graph are tested
    elsewhere: here rest is every bin outside an event, the running median is
    fixed, and the track distance is the absolute difference of positions."""
    monkeypatch.setattr(
        module,
        "regime_masks",
        lambda dot: {"rest": dot["non_local_event"].to_numpy() == 0},
    )
    monkeypatch.setattr(
        module,
        "brain_state_reference",
        lambda dot: {"theta_running_median": RUNNING_THETA, "theta_rest_median": 0.0},
    )
    monkeypatch.setattr(
        module,
        "track_distance_cm",
        lambda animal, decoded, *_: np.abs(decoded - animal),
    )


def measure(events: pd.DataFrame, n_windows: int = 4, seed: int = 0) -> pd.DataFrame:
    return module.start_and_rest_theta(
        frame(), ENVIRONMENT, events, n_windows, np.random.default_rng(seed)
    )


EVENTS = pd.DataFrame({"event_id": ["e1", "e2"], "non_local_event": [1, 2]})


@pytest.fixture
def measured(stubbed) -> pd.DataFrame:
    return measure(EVENTS).set_index("event_id")


def test_start_distance_reads_the_first_bin_with_a_finite_map(measured):
    # event 1's first bin (2) has no MAP: its start is bin 3's MAP
    assert measured.loc["e1", "start_distance_cm"] == pytest.approx(130.0 - 5.0)
    assert measured.loc["e2", "start_distance_cm"] == pytest.approx(190.0 - 5.0)


def test_a_split_label_counts_every_bin_and_is_flagged(measured):
    assert measured.loc["e1", "n_bins"] == 4 and measured.loc["e1", "bins_contiguous"]
    assert measured.loc["e2", "n_bins"] == 4
    assert not measured.loc["e2", "bins_contiguous"]


def test_theta_mean_is_the_event_table_theta_z(measured):
    # the event table's theta_z: brain_state_means over the event's bins
    dot = frame()
    in_event = dot["non_local_event"].to_numpy() > 0
    ids, means = brain_state_means(
        event_id=dot["non_local_event"].to_numpy()[in_event],
        state_probs={},
        extra_means={"theta_z": dot["theta_power_zscore"].to_numpy()[in_event]},
    )
    theta_z = dict(zip(ids, means["theta_z"], strict=True))
    assert measured.loc["e1", "theta_mean"] == pytest.approx(theta_z[1])
    assert measured.loc["e2", "theta_mean"] == pytest.approx(theta_z[2])
    assert np.isfinite(measured["theta_mean"]).all()


def test_rest_windows_match_each_event_length(measured):
    assert (measured["rest_windows"] == 4).all()
    assert measured["theta_running_median"].eq(RUNNING_THETA).all()
    assert measured["rest_theta_like_share"].between(0, 1).all()


def test_the_share_counts_windows_reaching_the_running_median_among_finite_ones(
    stubbed,
):
    """A one-bin event, so each window is one rest bin. Of the nine rest bins,
    one has no theta; of the other eight, the one at the running median (a
    tie) and the one above it are theta-like: 2 of 8 (2 of 9 if the empty
    window counted, 1 of 8 if a tie did not)."""
    theta = [2.0, 0.5, 0.4, np.nan, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    dot = pd.DataFrame(
        {
            "non_local_event": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "map_estimate": np.full(10, 100.0),
            "linear_position": np.full(10, 5.0),
            "theta_power_zscore": theta,
        }
    )
    events = pd.DataFrame({"event_id": ["e1"], "non_local_event": [1]})

    row = module.start_and_rest_theta(
        dot, ENVIRONMENT, events, 20_000, np.random.default_rng(0)
    ).iloc[0]

    assert row.rest_theta_like_share == pytest.approx(2 / 8, abs=0.01)
    assert row.rest_windows == pytest.approx(20_000 * 8 / 9, rel=0.02)
    # the event's own bin is not rest
    assert row.theta_mean == 2.0


def test_an_event_whose_label_is_not_in_the_frame_is_refused(stubbed):
    events = pd.DataFrame({"event_id": ["e9"], "non_local_event": [9]})
    with pytest.raises(ValueError, match="has no bin in the frame"):
        measure(events)
