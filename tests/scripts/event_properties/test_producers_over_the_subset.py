"""The producers of the events summary's inputs, each run to its manifest on a
tiny synthetic input: each reads the events that start inside a well visit (or
the time inside one), records the subset and, when its content depends on the
visits, the session-cache entries it read them from. The measures themselves
are tested in ``tests/analysis/event_properties``; here they are stubbed, and
the stubs record what each was given."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.paths import (
    BRAIN_STATE_REFERENCE_DIR,
    EVENT_PROPERTY_BOOTSTRAP_DIR,
    EVENT_SPEED_DIR,
    EVENT_START_REST_THETA_DIR,
    NONLOCAL_BY_SIGNAL_DIR,
    event_features_root,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import files_sha256, sha256_file
from scripts.event_properties import _epoch_counts as epoch_counts
from scripts.event_properties import bootstrap_event_properties as bootstrap
from scripts.event_properties import measure_brain_state_reference as brain_state
from scripts.event_properties import measure_event_gamma as gamma
from scripts.event_properties import measure_nonlocal_by_signal as power
from scripts.event_properties import measure_ripple_class_structure as structure
from tests.support._session_cache_fixtures import write_session_cache_entries

EPOCH = "j1620210706_02_r1_HPC"
TIME = np.arange(0.0, 1.0, 0.002)
VISITS = (np.array([0.1]), np.array([0.5]))
"""The epoch's one well visit: 0.1 s to 0.5 s."""


def _table() -> pd.DataFrame:
    """Two events that start inside the visit and a third, on the track, after it."""
    return pd.DataFrame(
        {
            "animal": "j16",
            "session_date": 20210706,
            "epoch": "02_r1",
            "event_id": pd.Series([1, 2, 3], dtype="uint64"),
            "event_start_s": [0.2, 0.4, 0.7],
            "event_end_s": [0.25, 0.55, 0.75],
            "event_duration_s": [0.05, 0.15, 0.05],
            "is_swr": [True, False, False],
            "visit_context": ["in_visit", "spans_visit_end", "track_center"],
        }
    )


def _write_event_table() -> str:
    """The event table where the producers read it; its sha256."""
    path = table_path("event_table", replay_content_dir())
    path.parent.mkdir(parents=True, exist_ok=True)
    _table().to_parquet(path)
    return sha256_file(path)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"animal": "j16", "day": 20210706, "epoch": "02_r1"},
        index=pd.Index(TIME, name="time"),
    )


def _features_run(name: str) -> None:
    run = event_features_root() / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"epochs": {EPOCH: {"status": "ok", "decode": {"sha256": "d"}}}})
    )


@pytest.fixture
def decoded(processed_data_root, monkeypatch):
    """One epoch's decode frame (read through a stub) and its well visit."""
    for module in (brain_state, epoch_counts):
        monkeypatch.setattr(
            module.analyzed_time,
            "read_verified_dot_with_swr_source",
            lambda path, sha256: _frame(),
        )
        monkeypatch.setattr(module, "well_visit_times", lambda identifier: VISITS)
        monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    return write_session_cache_entries([EPOCH])


def test_the_brain_state_reference_is_over_the_rest_inside_the_visits(
    decoded, monkeypatch
):
    _features_run("run1")
    seen = []

    def reference(dot, in_visit):
        seen.append(in_visit)
        return {"n_rest_bins": int(in_visit.sum())}

    monkeypatch.setattr(brain_state, "brain_state_reference", reference)
    brain_state.main(["--event-features-run", "run1"])
    np.testing.assert_array_equal(seen[0], (TIME >= 0.1) & (TIME < 0.5))
    folder = brain_state.processed_data_dir() / BRAIN_STATE_REFERENCE_DIR
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["subset"] == ANALYZED_SUBSET
    assert manifest["session_cache"]["sha256"] == files_sha256(decoded)
    assert pd.read_parquet(folder / "reference.parquet")["epoch_id"].tolist() == [EPOCH]


def test_the_counts_by_signal_are_over_the_immobile_time_inside_the_visits(
    decoded, monkeypatch
):
    _features_run("run1")
    seen = []

    def counts(dot, in_visit):
        seen.append(in_visit)
        return pd.DataFrame({"n_bins": [int(in_visit.sum())]})

    monkeypatch.setattr(power, "rest_signal_counts", counts)
    monkeypatch.setattr(power, "immobile_analyzed_bins", lambda dot: TIME < 0.3)
    power.main(["--event-features-run", "run1"])
    np.testing.assert_array_equal(seen[0], (TIME >= 0.1) & (TIME < 0.5))
    folder = epoch_counts.processed_data_dir() / NONLOCAL_BY_SIGNAL_DIR
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["subset"] == ANALYZED_SUBSET
    assert manifest["session_cache"]["sha256"] == files_sha256(decoded)
    # immobile below 0.3 s: 0.1 to 0.3 s inside the visit, 0 to 0.1 s outside
    assert manifest["epochs"][EPOCH] == {
        "n_immobile_bins_in_visit": int(((TIME >= 0.1) & (TIME < 0.3)).sum()),
        "n_immobile_bins_outside_visit": int((TIME < 0.1).sum()),
    }


@pytest.fixture
def gamma_inputs(processed_data_root, monkeypatch):
    """The event table and the epoch's cache entry; the LFP measure stubbed,
    recording the events it is given."""
    event_table_sha256 = _write_event_table()
    monkeypatch.setattr(gamma, "read_swr_table", lambda path, columns: _table())
    monkeypatch.setattr(
        gamma.swr_source, "swr_inventory_record", lambda identifier: {"ok": True}
    )
    monkeypatch.setattr(gamma, "ProcessPoolExecutor", ThreadPoolExecutor)
    calls = []

    def measured(identifier, record, events, n_windows):
        calls.append(events["event_id"].tolist())
        table = events[["event_id", "event_start_s"]].assign(slow_gamma_z=0.5)
        reference = {"epoch_id": identifier, "slow_gamma_rest_median": 0.0}
        return table, reference, pd.DataFrame({"time": TIME})

    monkeypatch.setattr(gamma, "epoch_gamma", measured)
    return calls, event_table_sha256, write_session_cache_entries([EPOCH])


def test_the_gamma_pass_measures_the_selected_events_and_reuses_by_table_subset_and_visits(
    gamma_inputs,
):
    calls, event_table_sha256, entries = gamma_inputs
    gamma.main([])
    assert calls == [[1, 2]]
    folder = gamma.out_dir()
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["subset"] == ANALYZED_SUBSET
    assert manifest["event_table"]["sha256"] == event_table_sha256
    assert manifest["session_cache"]["sha256"] == files_sha256(entries)
    assert manifest["visit_context_counts"]["track_center"] == 1
    reference = json.loads((folder / f"{EPOCH}_reference.json").read_text())
    assert reference["subset"] == ANALYZED_SUBSET
    assert reference["session_cache_sha256"] == sha256_file(entries[0])

    # the same table, subset and visits: the epoch is reused
    gamma.main([])
    assert len(calls) == 1
    # another subset, another event table, other visits: each is measured again
    reference["subset"] = "every event"
    (folder / f"{EPOCH}_reference.json").write_text(json.dumps(reference))
    gamma.main([])
    assert len(calls) == 2
    table_path("event_table", replay_content_dir()).write_bytes(b"another table")
    gamma.main([])
    assert len(calls) == 3
    entries[0].write_bytes(b"other visits")
    gamma.main([])
    assert len(calls) == 4
    gamma.main([])
    assert len(calls) == 4
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["session_cache"]["sha256"] == files_sha256(entries)


def test_the_bootstrap_resamples_the_selected_events(processed_data_root, monkeypatch):
    event_table_sha256 = _write_event_table()
    monkeypatch.setattr(bootstrap, "read_swr_table", lambda path: _table())
    starts = bootstrap.processed_data_dir() / EVENT_START_REST_THETA_DIR
    starts.mkdir()
    pd.DataFrame(
        {"event_id": pd.Series([1, 2, 3], dtype="uint64"), "start_distance_cm": 5.0}
    ).to_parquet(starts / f"{EPOCH}.parquet")
    speeds = bootstrap.processed_data_dir() / EVENT_SPEED_DIR
    speeds.mkdir()
    pd.DataFrame(
        {
            "event_id": pd.Series([1, 2, 3], dtype="uint64"),
            "map_line_fit_speed_cm_s": 100.0,
            "transport_slope_speed_cm_s": [150.0, np.nan, 150.0],
        }
    ).to_parquet(speeds / f"{EPOCH}.parquet")
    seen = {}

    def contrasts(values, is_swr, animals, sessions, durations, **kwargs):
        seen["durations"] = durations
        return pd.DataFrame({"duration": [1.0]})

    monkeypatch.setattr(bootstrap, "bootstrap_contrasts", contrasts)
    monkeypatch.setattr(
        bootstrap,
        "bootstrap_intervals",
        lambda replicates, n_animals: pd.DataFrame({"estimate": [1.0]}),
    )
    bootstrap.main(["--n-replicates", "2", "--properties", "duration"])
    assert seen["durations"].tolist() == [0.05, 0.15]
    folder = bootstrap.processed_data_dir() / EVENT_PROPERTY_BOOTSTRAP_DIR
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["subset"] == ANALYZED_SUBSET
    assert manifest["event_table"]["sha256"] == event_table_sha256
    assert manifest["n_events"] == 2
    assert manifest["event_speed"]["n_files"] == 1
    assert manifest["visit_context_counts"]["track_center"] == 1


def test_the_class_structure_analysis_reads_the_selected_events(
    processed_data_root, monkeypatch
):
    """The inputs of the class-structure tests (one event in a visit, one on the
    track), each event with its animal; the statistics and the simulations are
    stubbed, recording the events they are given."""
    from continuum_swr_replay.analysis.event_definition.analyzed_time import (
        stamp_swr_source,
    )
    from continuum_swr_replay.tables.io import write_table
    from tests.scripts.event_properties.test_measure_ripple_class_structure import (
        _lay_out_inputs,
    )

    _lay_out_inputs()
    path = table_path("event_table", replay_content_dir())
    table = pd.read_parquet(path).assign(
        animal="j16", session_date=20210706, epoch="02_r1"
    )
    write_table(stamp_swr_source(table), path)
    seen = []

    def distribution(events):
        seen.append(events["event_id"].tolist())
        return pd.DataFrame({"animal": ["pooled"], "dip_p": [1.0]})

    monkeypatch.setattr(structure.cc, "distribution_table", distribution)
    monkeypatch.setattr(
        structure.cc, "step_table", lambda events, windows, knots: _steps()
    )
    for name, value in {
        "plot_distributions": lambda *args: None,
        "plot_property": lambda *args: None,
        "plot_spot_checks": lambda *args: None,
        "distribution_simulation": lambda *args: _simulated(),
        "step_simulation": lambda *args: _step_simulated(),
        "spot_check_events": lambda events: pd.DataFrame({"event_id": [5]}),
    }.items():
        monkeypatch.setattr(structure, name, value)

    structure.main(["analyze", "--n-dip-reps", "1", "--n-step-reps", "1"])

    assert seen == [[5]]
    manifest = json.loads((structure.out_dir() / "manifest.json").read_text())
    assert manifest["subset"] == ANALYZED_SUBSET
    assert manifest["checks"]["n_events"] == 1
    assert manifest["checks"]["visit_context_counts"]["track_center"] == 1
    assert manifest["inputs"]["event_table"]["sha256"] == sha256_file(path)


def _steps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_knots": [8],
            "power": ["criterion"],
            "window_z": [1.5],
            "property": ["duration"],
            "covariate": ["none"],
            "has_step": [False],
            "step_share": [0.0],
            "heldout_gain": [0.0],
            "rows_fraction": [1.0],
        }
    )


def _simulated() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["one"],
            "separation_factor": [1.0],
            "power": ["peak"],
            "dip_p": [1.0],
        }
    )


def _step_simulated() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["one"],
            "planted_step": [0.0],
            "power": ["peak"],
            "covariate": ["none"],
            "has_step": [False],
            "covers_planted": [True],
            "step": [0.0],
            "step_share": [0.0],
        }
    )
