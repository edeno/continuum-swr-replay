"""The canonical event rule on a probability trace, on a 500 Hz frame small
enough to check by hand. The SWR source and the artifact run are synthetic run
folders under ``tmp_path`` (``tests/support/_swr_inventory_fixtures.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import heldout_events as he
from tests.support._swr_inventory_fixtures import (
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

EPOCH = "j1620210706_02_r1_HPC"
N = 1000
TIME = np.round(100.0 + np.arange(N) / 500.0, 6)
"""100.000 to 101.998 s."""
SPANS = {
    "a": (100, 109),  # 10 samples; an SWR overlaps it
    "short": (200, 205),  # 6 samples: too short
    "moving": (295, 315),  # touches the moving samples 300-309
    "d": (400, 409),  # an SWR starts at its last sample: no strict overlap
    "f": (495, 505),  # starts inside the visit, ends after it
    "g": (550, 559),  # starts outside the visit
    "excluded": (800, 809),  # inside excluded time
}
SWRS = [(TIME[105], TIME[120]), (TIME[409], TIME[420])]
VISIT = np.arange(N) < 500
"""Bins 0-499 are inside a well visit."""


def _raw() -> pd.DataFrame:
    speed = np.zeros(N)
    speed[300:310] = 10.0
    return pd.DataFrame(
        {"animal": "j16", "day": 20210706, "epoch": "02_r1", "head_speed": speed},
        index=pd.Index(TIME, name="time"),
    )


def _trace() -> tuple[np.ndarray, np.ndarray]:
    """0.9 over each span (0.6 continuous, 0.3 fragmented), 0.1 elsewhere."""
    prob = np.full(N, 0.1)
    for s, e in SPANS.values():
        prob[s : e + 1] = 0.9
    return 2 * prob / 3, prob / 3


@pytest.fixture
def source(tmp_path, monkeypatch):
    """The SWR source with the two SWRs, and an artifact whose excluded time
    (padded 0.1 s, widened 0.3 s) covers 101.4 s to the end."""
    inventory = write_swr_inventory(tmp_path, {EPOCH: SWRS})
    artifacts = write_artifact_run(tmp_path, {EPOCH: [("spk", 101.8, 101.8)]})
    use_swr_source(monkeypatch, inventory, artifacts)


def test_the_rule_keeps_long_immobile_events_outside_excluded_time(source):
    continuous, fragmented = _trace()
    frame = he.labeled_frame(_raw(), continuous, fragmented)
    rows = he.event_rows(frame, VISIT)
    # short, moving (dropped whole) and excluded are not events
    assert list(zip(rows["s"], rows["e"], strict=True)) == [
        SPANS[name] for name in ("a", "d", "f", "g")
    ]
    # strict overlap decides the group: d only touches its SWR at one sample
    assert rows["group"].tolist() == ["SWR", "no SWR", "no SWR", "no SWR"]
    # the first bin decides the well visit: f ends after it, g starts outside
    assert rows["starts_in_well_visit"].tolist() == [True, True, True, False]
    np.testing.assert_array_equal(rows["event_start_s"], TIME[[100, 400, 495, 550]])
    np.testing.assert_array_equal(rows["event_end_s"], TIME[[109, 409, 505, 559]])
    # the frame holds the trace's probabilities, and only the kept events
    np.testing.assert_allclose(frame[he.PROBABILITY_COLUMNS[0]], continuous)
    labels = frame[he.LABEL_COLUMN].to_numpy()
    assert (labels[800:810] == 0).all() and (labels[295:316] == 0).all()
    assert set(np.unique(labels[labels > 0])) == set(rows["label"])


def test_the_raw_frame_is_left_as_it_was(source):
    raw = _raw()
    he.labeled_frame(raw, *_trace())
    assert list(raw.columns) == ["animal", "day", "epoch", "head_speed"]


def test_a_visit_flag_of_another_length_is_refused(source):
    frame = he.labeled_frame(_raw(), *_trace())
    with pytest.raises(ValueError, match="one value per row"):
        he.event_rows(frame, VISIT[:-1])


def test_label_bins_gives_each_events_first_and_last_bin():
    labels = np.array([0, 3, 3, 0, 0, 1, 1, 1, 0, 7])
    rows = he.label_bins(labels)
    assert rows.to_dict("list") == {"label": [1, 3, 7], "s": [5, 1, 9], "e": [7, 2, 9]}
    assert he.label_bins(np.zeros(4, dtype=int)).empty
    with pytest.raises(ValueError, match="one run"):
        he.label_bins(np.array([1, 1, 0, 1]))


def test_shared_bin_events_both_ways():
    a = np.array([1, 1, 0, 2, 2, 0, 0, 3, 0, 4])
    b = np.array([0, 5, 5, 0, 0, 6, 6, 6, 0, 0])
    assert he.shared_bin_events(a, b).to_dict() == {
        1: True,
        2: False,
        3: True,
        4: False,
    }
    assert he.shared_bin_events(b, a).to_dict() == {5: True, 6: True}
    with pytest.raises(ValueError, match="same time axis"):
        he.shared_bin_events(a, b[:-1])


def test_found_again_statistics_pool_events_and_balance_animals():
    """Animal x: epoch x1 2 of 2 no-SWR events found, x2 0 of 1 SWR event and 1
    of 1 no-SWR event; animal y: 1 of 4 no-SWR events."""
    events = pd.DataFrame(
        {
            "epoch_id": ["x1", "x1", "x2", "x2", *["y1"] * 4],
            "animal": [*"xxxx", *"yyyy"],
            "group": ["no SWR", "no SWR", "SWR", "no SWR", *["no SWR"] * 4],
            "found": [True, True, False, True, True, False, False, False],
        }
    )
    table = he.found_again_statistics(events, "found")
    assert table.index.tolist() == ["all", "no SWR", "SWR"]
    assert table["n_events"].tolist() == [8, 7, 1]
    assert table["n_found"].tolist() == [4, 4, 0]
    assert table.loc["all", "share"] == pytest.approx(0.5)
    assert table.loc["no SWR", "share"] == pytest.approx(4 / 7)
    # all: x's epochs 1 and 0.5, mean 0.75; y 0.25
    assert table.loc["all", "share_animal_min"] == pytest.approx(0.25)
    assert table.loc["all", "share_animal_max"] == pytest.approx(0.75)
    # no SWR: x's epochs 1 and 1; y 0.25
    assert table.loc["no SWR", "share_animal_max"] == pytest.approx(1.0)
    assert table.loc["SWR", "n_animals"] == 1
    assert table.loc["SWR", "share"] == pytest.approx(0.0)
