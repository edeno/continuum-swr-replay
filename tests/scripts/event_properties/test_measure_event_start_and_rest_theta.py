"""``measure_event_start_and_rest_theta``: the frame ``measure_epoch`` reads and the
columns it adds, and a whole run of ``main`` to its manifest. The measures are
tested in
``tests/analysis/event_properties/test_start_and_rest_theta.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.analysis.event_properties import start_and_rest_theta
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import (
    EVENT_START_REST_THETA_DIR,
    STATE_SPACE_FEATURES_RUN,
    event_features_root,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import files_sha256
from tests.analysis.event_properties.test_start_and_rest_theta import EVENTS, frame
from tests.support._session_cache_fixtures import write_session_cache_entries

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


def test_main_writes_every_epoch_and_the_manifest(
    stored, processed_data_root, monkeypatch
):
    """The whole run, without ``--epochs``: the event table holds the two events
    of the frame inside a well visit and a third, on the track, whose label is
    in no bin (measuring it would raise), so the run reads only the selected
    events; the manifest is written last, as on real data. The pool is a
    thread pool here: the measures are stubbed in this process."""
    table = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "animal": "j16",
            "session_date": 20210706,
            "epoch": "02_r1",
            "non_local_event": [1, 2, 3],
            "visit_context": ["in_visit", "spans_visit_end", "track_center"],
        }
    )
    event_table = table_path("event_table", replay_content_dir())
    event_table.parent.mkdir(parents=True)
    table.to_parquet(event_table)
    monkeypatch.setattr(script, "read_swr_table", lambda path, columns: table)
    run = event_features_root() / STATE_SPACE_FEATURES_RUN
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {"epochs": {IDENTIFIER: {"status": "ok", "decode": {"sha256": stored}}}}
        )
    )
    monkeypatch.setattr(script, "ProcessPoolExecutor", ThreadPoolExecutor)
    entries = write_session_cache_entries([IDENTIFIER])

    script.main(["--n-windows", "4"])

    folder = processed_data_root / EVENT_START_REST_THETA_DIR
    measured = pd.read_parquet(folder / f"{IDENTIFIER}.parquet")
    assert measured.event_id.tolist() == ["e1", "e2"]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["subset"] == ANALYZED_SUBSET
    assert manifest["n_epochs"] == 1
    assert manifest["n_windows"] == 4
    assert manifest["visit_context_counts"]["track_center"] == 1
    assert manifest["visit_context_counts"]["in_visit"] == 1
    assert manifest["event_table"]["path"] == str(event_table)
    assert manifest["session_cache"]["sha256"] == files_sha256(entries)
