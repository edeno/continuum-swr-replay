"""``measure_event_start_and_rest_theta.measure_epoch``: the frame it reads and the
columns it adds. The measures are tested in
``tests/analysis/event_properties/test_start_and_rest_theta.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties import start_and_rest_theta
from continuum_swr_replay.decoding.files import decode_paths
from tests.analysis.event_properties.test_start_and_rest_theta import EVENTS, frame

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/event_properties/measure_event_start_and_rest_theta.py"
)
_SPEC = importlib.util.spec_from_file_location("measure_start_rest", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

IDENTIFIER = "j1620210706_02_r1_HPC"


@pytest.fixture
def stored(processed_data_root, monkeypatch) -> str:
    """The synthetic decode frame and an environment, written where the script
    reads them; its sha256. The SWR source and the measures' regimes, reference
    and track graph are tested elsewhere and stubbed here."""
    paths = decode_paths(IDENTIFIER, processed_data_root)
    raw = pickle.dumps(frame())
    paths.data_over_time.write_bytes(raw)
    paths.environment.write_bytes(
        pickle.dumps(
            SimpleNamespace(
                place_bin_centers_nodes_df_=None, distance_between_nodes_=None
            )
        )
    )
    monkeypatch.setattr(script.analyzed_time, "with_swr_source", lambda dot: dot)
    # one well visit holding the whole frame (it spans 0 to 0.038 s)
    monkeypatch.setattr(
        script,
        "well_visit_times",
        lambda identifier: (np.array([-1.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        start_and_rest_theta,
        "regime_masks",
        lambda dot, in_visit: {
            "rest": (dot["non_local_event"].to_numpy() == 0) & in_visit
        },
    )
    monkeypatch.setattr(
        start_and_rest_theta,
        "brain_state_reference",
        lambda dot, in_visit: {"theta_running_median": 0.5, "theta_rest_median": 0.0},
    )
    monkeypatch.setattr(
        start_and_rest_theta,
        "track_distance_cm",
        lambda animal, decoded, *_: np.abs(decoded - animal),
    )
    return hashlib.sha256(raw).hexdigest()


def test_each_event_carries_its_epoch_and_the_decode_hash(stored):
    table = script.measure_epoch((IDENTIFIER, stored, EVENTS, 4))

    assert list(table.columns[:3]) == ["event_id", "epoch_id", "decode_sha256"]
    assert table.epoch_id.eq(IDENTIFIER).all()
    assert table.decode_sha256.eq(stored).all()
    assert table.event_id.tolist() == ["e1", "e2"]


def test_an_epochs_draws_are_seeded_by_its_identifier(stored):
    first = script.measure_epoch((IDENTIFIER, stored, EVENTS, 4))
    again = script.measure_epoch((IDENTIFIER, stored, EVENTS, 4))

    pd.testing.assert_frame_equal(first, again)


def test_a_decode_with_another_hash_is_refused(stored):
    events = pd.DataFrame({"event_id": ["e1"], "non_local_event": [1]})
    with pytest.raises(ValueError, match="is not the run's"):
        script.measure_epoch((IDENTIFIER, "0" * 64, events, 4))


def test_a_refusal_of_the_measures_names_the_epoch(stored):
    events = pd.DataFrame({"event_id": ["e9"], "non_local_event": [9]})
    with pytest.raises(ValueError, match=f"{IDENTIFIER}: an event's label has no bin"):
        script.measure_epoch((IDENTIFIER, stored, events, 4))


def test_the_rest_windows_are_drawn_inside_the_epochs_well_visits(stored, monkeypatch):
    """With the visits ending before the frame, no bin is rest: every event's
    rest windows are empty."""
    monkeypatch.setattr(
        script,
        "well_visit_times",
        lambda identifier: (np.array([-1.0]), np.array([-0.5])),
    )
    table = script.measure_epoch((IDENTIFIER, stored, EVENTS, 4))
    assert table.rest_windows.eq(0).all()
    assert table.rest_theta_like_share.isna().all()
