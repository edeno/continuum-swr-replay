"""The events summary: values read from an output made from another event table are pending,
and nothing is dropped silently from the values behind a quoted number."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.reported.events import (
    _PROPERTIES,
    class_step_values_with_staleness,
    default_inputs,
    event_table_staleness,
    gamma_db_values,
    property_interval_values,
    rest_theta_values,
    spike_evidence_staleness,
)
from continuum_swr_replay.reported.values import Value, pend
from tests.reported.test_reported_events_values import step_fits

CURRENT = "a" * 64


def _value(key: str, pending: str | None = None) -> Value:
    return Value(
        key=key,
        value=1.0,
        kind="quantity",
        style="dec2",
        unit="",
        description="d",
        questions=("2c.1",),
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
        rows[word] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def test_property_intervals_name_their_skipped_strata_and_refuse_many_nans():
    values = {v.key: v for v in property_interval_values(_intervals(), "s")}
    low = values["extent_matched_difference_low"].description
    assert "3 of 1000 replicates left out a stratum" in low
    assert "0 had no estimate" in low
    with pytest.raises(ValueError, match="too many bootstrap replicates"):
        property_interval_values(_intervals(nan_matched=50), "s")
    old = _intervals().drop(
        columns=[c for c in _intervals().columns if c.endswith("_nan_replicates")]
    )
    with pytest.raises(ValueError, match="does not count"):
        property_interval_values(old, "s")


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
