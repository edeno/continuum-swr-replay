"""``measure_event_start_and_rest_theta.measure_epoch`` on a 20-bin synthetic decode."""

from __future__ import annotations

import hashlib
import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.trajectory import brain_state_means
from continuum_swr_replay.decoding.files import decode_paths

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/detection/measure_event_start_and_rest_theta.py"
)
_SPEC = importlib.util.spec_from_file_location("measure_start_rest", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

IDENTIFIER = "j1620210706_02_r1_HPC"
RUNNING_THETA = 0.5


def _frame() -> pd.DataFrame:
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
def measured(processed_data_root, monkeypatch) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _frame()
    paths = decode_paths(IDENTIFIER, processed_data_root)
    raw = pickle.dumps(frame)
    paths.data_over_time.write_bytes(raw)
    paths.environment.write_bytes(
        pickle.dumps(
            SimpleNamespace(
                place_bin_centers_nodes_df_=None, distance_between_nodes_=None
            )
        )
    )
    # The SWR source, the regimes and the track graph are tested elsewhere:
    # here the frame is used as is, rest is every bin outside an event, and the
    # track distance is the absolute difference of linear positions.
    monkeypatch.setattr(script.analysed_time, "with_swr_source", lambda dot: dot)
    monkeypatch.setattr(
        script,
        "regime_masks",
        lambda dot: {"rest": dot["non_local_event"].to_numpy() == 0},
    )
    monkeypatch.setattr(
        script,
        "brain_state_reference",
        lambda dot: {"theta_running_median": RUNNING_THETA, "theta_rest_median": 0.0},
    )
    monkeypatch.setattr(
        script,
        "track_distance_cm",
        lambda animal, decoded, *_: np.abs(decoded - animal),
    )
    events = pd.DataFrame({"event_id": ["e1", "e2"], "non_local_event": [1, 2]})
    table = script.measure_epoch(
        (IDENTIFIER, hashlib.sha256(raw).hexdigest(), events, 4)
    )
    return frame, table.set_index("event_id")


def test_start_distance_reads_the_first_bin_with_a_finite_map(measured):
    frame, table = measured
    # event 1's first bin (2) has no MAP: its start is bin 3's MAP
    assert table.loc["e1", "start_distance_cm"] == pytest.approx(130.0 - 5.0)
    assert table.loc["e2", "start_distance_cm"] == pytest.approx(190.0 - 5.0)


def test_a_split_label_counts_every_bin_and_is_flagged(measured):
    _, table = measured
    assert table.loc["e1", "n_bins"] == 4 and table.loc["e1", "bins_contiguous"]
    assert table.loc["e2", "n_bins"] == 4
    assert not table.loc["e2", "bins_contiguous"]


def test_theta_mean_is_the_event_table_theta_z(measured):
    frame, table = measured
    # the event table's theta_z: brain_state_means over the event's bins
    in_event = frame["non_local_event"].to_numpy() > 0
    ids, means = brain_state_means(
        event_id=frame["non_local_event"].to_numpy()[in_event],
        state_probs={},
        extra_means={"theta_z": frame["theta_power_zscore"].to_numpy()[in_event]},
    )
    theta_z = dict(zip(ids, means["theta_z"], strict=True))
    assert table.loc["e1", "theta_mean"] == pytest.approx(theta_z[1])
    assert table.loc["e2", "theta_mean"] == pytest.approx(theta_z[2])
    assert np.isfinite(table["theta_mean"]).all()


def test_rest_windows_match_each_event_length(measured):
    _, table = measured
    assert (table["rest_windows"] == 4).all()
    assert table["theta_running_median"].eq(RUNNING_THETA).all()
    assert table["rest_theta_like_share"].between(0, 1).all()


def test_a_decode_with_another_hash_is_refused(processed_data_root, measured):
    events = pd.DataFrame({"event_id": ["e1"], "non_local_event": [1]})
    with pytest.raises(ValueError, match="is not the run's"):
        script.measure_epoch((IDENTIFIER, "0" * 64, events, 4))
