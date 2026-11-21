"""The events summary: values read from an output made from another event table are pending,
and nothing is dropped silently from the values behind a quoted number."""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
)
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.reported import events
from continuum_swr_replay.reported.events import (
    _PROPERTIES,
    class_step_values_with_staleness,
    default_inputs,
    event_table_staleness,
    gamma_db_values,
    property_interval_values,
    rest_theta_values,
    restriction_window_values,
    spike_evidence_staleness,
    window_run_note_staleness,
)
from continuum_swr_replay.reported.values import Value, pend
from continuum_swr_replay.runs.io import sha256_file
from tests.reported.test_reported_events_values import (
    _restriction_windows,
    step_fits,
)
from tests.support._session_cache_fixtures import write_session_cache_entries

CURRENT = "a" * 64


def _value(key: str, pending: str | None = None) -> Value:
    return Value(
        key=key,
        value=1.0,
        kind="quantity",
        style="dec2",
        unit="",
        description="d",
        questions=("2d.8",),
        figures=(),
        source="s",
        function="f",
        pending=pending,
    )


def test_every_input_is_read_from_the_data_root(processed_data_root):
    """A fresh checkout holds none of the inputs: all are under the data root."""
    inputs = default_inputs()

    outside = [
        field.name
        for field in fields(inputs)
        if not getattr(inputs, field.name).is_relative_to(processed_data_dir())
    ]

    assert processed_data_dir() == processed_data_root
    assert outside == []


@pytest.mark.parametrize(
    "manifest",
    [
        {"event_table": {"path": "x", "sha256": CURRENT}},
        {"inputs": {"event_table": {"path": "x", "sha256": CURRENT}}},
    ],
)
def test_a_manifest_recording_the_current_table_is_fresh(manifest):
    assert event_table_staleness(manifest, CURRENT, "out") is None


def test_a_manifest_recording_another_table_is_stale():
    note = event_table_staleness(
        {"event_table": {"sha256": "b" * 64}}, CURRENT, "Processed-Data/out"
    )
    assert "Processed-Data/out was made from another event table" in note
    assert "bbbbbbbbbbbb" in note and "aaaaaaaaaaaa" in note


def test_a_manifest_recording_no_table_is_stale():
    note = event_table_staleness({"epochs": {}}, CURRENT, "event_gamma")
    assert note == "event_gamma records no event-table SHA-256; rerun it before citing"


def test_an_output_made_from_the_current_spike_evidence_run_is_fresh():
    manifest = {"inputs": {"spike_evidence": {"n_files": 2, "sha256": CURRENT}}}
    assert spike_evidence_staleness(manifest, CURRENT, "out") is None


def test_an_output_made_from_another_spike_evidence_run_is_stale():
    # the event table it records is the current one: only the run differs
    manifest = {
        "inputs": {
            "event_table": {"sha256": CURRENT},
            "spike_evidence": {"sha256": "c" * 64},
        }
    }
    assert event_table_staleness(manifest, CURRENT, "steps") is None
    note = spike_evidence_staleness(manifest, CURRENT, "steps")
    assert note == (
        "steps was made from another spike-evidence run than the current one "
        "(sha256 cccccccccccc, current aaaaaaaaaaaa); rerun it before citing"
    )


def test_an_output_recording_no_spike_evidence_run_is_stale():
    note = spike_evidence_staleness({"inputs": {"event_table": {}}}, CURRENT, "steps")
    assert note == "steps records no spike-evidence SHA-256; rerun it before citing"


def test_the_pending_note_reaches_every_value():
    values = [_value("slow_gamma_median_swr"), _value("theta_median_swr")]
    assert [v.pending for v in pend(values, "stale")] == ["stale", "stale"]
    # an existing note is kept, and no note leaves the values unchanged
    assert pend([_value("x", "untracked")], "stale")[0].pending == "untracked; stale"
    assert pend(values, None) == values
    # several notes are added in order, a None among them skipped, as nested
    # calls would add them
    both = pend([_value("x", "untracked"), _value("y")], "stale", None, "other run")
    assert [v.pending for v in both] == [
        "untracked; stale; other run",
        "stale; other run",
    ]
    assert both == pend(
        pend(pend([_value("x", "untracked"), _value("y")], "stale"), None), "other run"
    )
    assert pend(values) == values


def _gamma_reference() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"animal": np.repeat(["a", "b", "c"], 4)})
    for word in ("slow_gamma", "fast_gamma"):
        for group in ("non_swr", "swr"):
            frame[f"{word}_db_{group}"] = rng.normal(0.5, 0.2, len(frame))
    return frame


def test_gamma_db_values_refuse_an_epoch_without_a_value():
    assert len(gamma_db_values(_gamma_reference(), "s")) == 8
    reference = _gamma_reference()
    reference.loc[3, "fast_gamma_db_swr"] = np.nan
    with pytest.raises(ValueError, match="1 epochs have no fast_gamma_db_swr"):
        gamma_db_values(reference, "s")


def _rest_events(rest_windows) -> pd.DataFrame:
    n = len(rest_windows)
    return pd.DataFrame(
        {
            "animal": ["a", "b"] * (n // 2),
            "session_date": "d1",
            "epoch": "e1",
            "is_swr": [False, False, True, True] * (n // 4),
            "theta_z": np.linspace(-1, 1, n),
            "theta_running_median": 0.0,
            "rest_windows": rest_windows,
            "rest_theta_like_share": 0.2,
        }
    )


def test_rest_theta_values_refuse_an_event_without_rest_windows():
    assert len(rest_theta_values(_rest_events([10] * 8), "s")) == 4
    with pytest.raises(ValueError, match="1 events have no duration-matched"):
        rest_theta_values(_rest_events([10] * 7 + [0]), "s")


def _intervals(nan_matched: int = 0) -> pd.DataFrame:
    columns = ["median_non_swr", "median_swr", "difference", "matched_difference"]
    rows = {}
    for word, *_ in _PROPERTIES:
        row = {}
        for c in columns:
            row |= {c: 1.0, f"{c}_low": 0.5, f"{c}_high": 1.5, f"{c}_nan_replicates": 0}
        row["matched_difference_nan_replicates"] = nan_matched
        row["n_replicates"] = 1000
        row["matched_replicates_skipping_strata"] = 3
        row["n_animals"] = 5
        rows[word] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def test_property_intervals_name_their_skipped_strata_and_refuse_many_nans():
    values = {v.key: v for v in property_interval_values(_intervals(), "s")}
    low = values["extent_matched_difference_low"].description
    assert "3 of 1000 replicates left out a stratum" in low
    assert "0 had no estimate" in low
    assert "small-sample t correction over the 5 animals" in low
    with pytest.raises(ValueError, match="too many bootstrap replicates"):
        property_interval_values(_intervals(nan_matched=50), "s")
    old = _intervals().drop(
        columns=[c for c in _intervals().columns if c.endswith("_nan_replicates")]
    )
    with pytest.raises(ValueError, match="does not count"):
        property_interval_values(old, "s")
    with pytest.raises(ValueError, match="percentile intervals"):
        property_interval_values(_intervals().drop(columns="n_animals"), "s")


_EVIDENCE_KEYS = {
    "step_evidence_criterion_animals",
    "step_evidence_criterion_animals_same_sign",
    "step_evidence_criterion_share_median",
}


def _step_pending(manifest: dict, event_table_stale: str | None) -> dict:
    """Each step value's pending note, from the hand-made step fits."""
    values = class_step_values_with_staleness(
        manifest, CURRENT, event_table_stale, step_fits(), "s", "steps"
    )
    pending = {value.key: value.pending for value in values}
    assert len(pending) == len(values)
    assert set(pending) == {
        *_EVIDENCE_KEYS,
        "step_other_properties_criterion_animals_max",
    }
    return pending


@pytest.mark.parametrize(
    ("spike_evidence", "note"),
    [
        ({"sha256": CURRENT}, None),
        (
            {"sha256": "c" * 64},
            "steps was made from another spike-evidence run than the current one "
            "(sha256 cccccccccccc, current aaaaaaaaaaaa); rerun it before citing",
        ),
        (None, "steps records no spike-evidence SHA-256; rerun it before citing"),
    ],
)
def test_only_spike_evidence_steps_are_pending_for_another_spike_evidence_run(
    spike_evidence, note
):
    inputs = {"event_table": {"sha256": CURRENT}}
    if spike_evidence is not None:
        inputs["spike_evidence"] = spike_evidence
    pending = _step_pending({"inputs": inputs}, None)
    assert pending == {
        "step_other_properties_criterion_animals_max": None,
        **dict.fromkeys(_EVIDENCE_KEYS, note),
    }


def test_every_step_value_is_pending_for_another_event_table():
    manifest = {"inputs": {"spike_evidence": {"sha256": CURRENT}}}
    assert set(_step_pending(manifest, "stale table").values()) == {"stale table"}
    # both notes reach spike evidence's values when both inputs have changed
    pending = _step_pending({"inputs": {}}, "stale table")
    assert pending["step_other_properties_criterion_animals_max"] == "stale table"
    assert {pending[key] for key in _EVIDENCE_KEYS} == {
        "stale table; steps records no spike-evidence SHA-256; rerun it before citing"
    }


def _window_manifest(note: dict | None) -> dict:
    """A window run's manifest, with the event-table note when given."""
    settings = {"contrasts": {}}
    if note is not None:
        settings["event_table_note"] = note
    return {"settings": settings, "inputs": {}, "epochs": {}}


def test_window_values_are_pending_when_the_window_run_carries_a_note():
    """A window run on another event table than the held-out tetrode run's:
    every restriction window value is pending, and the reason names both
    tables and the runs to rerun."""
    note = {
        "note": "the event table is not the one the held-out tetrode run read",
        "event_table_sha256": "c" * 64,
        "tetrode_holdout_event_table_sha256": "d" * 64,
    }
    reason = window_run_note_staleness(_window_manifest(note), "Processed-Data/w")
    assert reason.startswith("Processed-Data/w read another event table")
    assert "cccccccccccc" in reason and "dddddddddddd" in reason
    assert "held-out tetrode run" in reason and "rerun" in reason
    values = pend(
        sum(restriction_window_values(_restriction_windows(), "s"), []),
        reason,
    )
    assert values and all(v.pending == reason for v in values)


def test_window_values_are_not_pending_without_the_note():
    assert window_run_note_staleness(_window_manifest(None), "w") is None
    values = pend(
        sum(restriction_window_values(_restriction_windows(), "s"), []),
        window_run_note_staleness(_window_manifest(None), "w"),
    )
    assert all(v.pending is None for v in values)


def _heldout_runs(root, change=None):
    """A restriction run (the fold fit on whole trials), a per-bin facts run made
    from the current session cache, and a held-out event run made from both and
    from the event table ``CURRENT``, under ``root``; ``change`` replaces one
    record of one manifest: (run, key, value)."""
    write_session_cache_entries(["e1", "e2"])
    runs = {name: root / name for name in ("restriction", "facts", "heldout")}
    manifests = {
        "restriction": {"settings": {"fit": "folds", "fold_design": "trials"}},
        "facts": {
            "inputs": {"session_cache": session_cache_record(["e1", "e2"])},
            "epochs": {"e1": {"status": "ok"}, "e2": {"status": "ok"}},
        },
    }
    for name in ("restriction", "facts"):
        if change and change[0] == name:
            manifests[name][change[1]] = change[2]
        runs[name].mkdir()
        (runs[name] / "manifest.json").write_text(json.dumps(manifests[name]))
    heldout = {
        "inputs": {
            "event_table": {"sha256": CURRENT},
            "nonlocal_restriction": {
                "sha256": sha256_file(runs["restriction"] / "manifest.json")
            },
            "bin_facts": {"sha256": sha256_file(runs["facts"] / "manifest.json")},
        },
        "epochs": {"e1": {"bin_s": 0.002}, "e2": {"bin_s": 0.004}},
    }
    if change and change[0] == "heldout":
        heldout["inputs"][change[1]] = change[2]
    runs["heldout"].mkdir()
    (runs["heldout"] / "manifest.json").write_text(json.dumps(heldout))
    return runs


def _read_heldout(runs, monkeypatch):
    tables = {
        "heldout_events": pd.DataFrame(
            {"heldout_event_id": range(4), "starts_in_well_visit": [1, 0, 1, 0]}
        ),
        "heldout_time_sums": pd.DataFrame({"epoch_id": ["e1", "e2"]}),
    }
    read = []

    def read_complete_run(folder, name, epochs):
        read.append((folder, name, epochs))
        return tables[name]

    monkeypatch.setattr(events, "read_complete_run", read_complete_run)
    heldout = events.read_heldout_restriction(
        runs["heldout"], runs["restriction"], runs["facts"], iter(["e1", "e2"]), CURRENT
    )
    assert read == [
        (runs["heldout"], name, ["e1", "e2"])
        for name in ("heldout_events", "heldout_time_sums")
    ]
    return heldout


REASONS = ("restriction_run", "bin_facts_run", "session_cache", "event_table")


def test_the_heldout_restriction_reader_selects_the_events_in_a_visit(
    processed_data_root, monkeypatch
):
    """The held-out events that start inside a well visit, the time sums whole,
    each epoch's time step, and no reason to pend when every run is current."""
    heldout = _read_heldout(_heldout_runs(processed_data_root), monkeypatch)
    assert heldout.events["heldout_event_id"].tolist() == [0, 2]
    assert heldout.time_sums["epoch_id"].tolist() == ["e1", "e2"]
    assert heldout.bin_s.to_dict() == {"e1": 0.002, "e2": 0.004}
    assert all(getattr(heldout, reason) is None for reason in REASONS)


@pytest.mark.parametrize(
    ("change", "reason", "note"),
    [
        (
            ("heldout", "nonlocal_restriction", {"sha256": "0" * 64}),
            "restriction_run",
            "another nonlocal_restriction run than the current one",
        ),
        (
            ("heldout", "bin_facts", {"sha256": "0" * 64}),
            "bin_facts_run",
            "another bin_facts run than the current one",
        ),
        (
            ("facts", "inputs", {}),
            "session_cache",
            "facts records no session-cache SHA-256",
        ),
        (
            ("heldout", "event_table", {"sha256": "0" * 64}),
            "event_table",
            "another event table than the current one",
        ),
    ],
    ids=REASONS,
)
def test_the_heldout_restriction_reader_names_each_reason_to_pend(
    processed_data_root, monkeypatch, change, reason, note
):
    """Each input not current sets its own reason, and only that one."""
    heldout = _read_heldout(_heldout_runs(processed_data_root, change), monkeypatch)
    assert note in getattr(heldout, reason)
    assert [r for r in REASONS if getattr(heldout, r) is not None] == [reason]


def test_the_heldout_restriction_reader_refuses_a_run_not_the_fold_fit(
    processed_data_root, monkeypatch
):
    runs = _heldout_runs(
        processed_data_root, ("restriction", "settings", {"fit": "folds"})
    )
    with pytest.raises(ValueError, match="fold_design None, not 'trials'"):
        _read_heldout(runs, monkeypatch)
