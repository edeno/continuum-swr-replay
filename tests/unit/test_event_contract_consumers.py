"""Direct scientific consumers must fail closed on malformed event labels."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    EventDefinitionError,
)
from scripts.content import analyze_content_occupancy
from scripts.qa import analyze_event_spike_evidence


def _noncontiguous_dot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "01_r1",
            "non_local_event": [1, 0, 1],
            "SWR_event": 0,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": [0.9, 0.1, 0.9],
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=[0.0, 0.1, 0.2],
    )


def test_content_occupancy_propagates_event_contract_error(monkeypatch):
    monkeypatch.setattr(
        analyze_content_occupancy,
        "session_content_distances",
        lambda _epoch_id: (_ for _ in ()).throw(
            EventDefinitionError("stale event labels")
        ),
    )

    with pytest.raises(EventDefinitionError, match="stale event labels"):
        analyze_content_occupancy.analyze(["a_01_r1_HPC"])


def test_content_occupancy_validates_dot_before_environment_load(monkeypatch):
    # The frame as the SWR source hands it over (stubbed: no inventory here).
    monkeypatch.setattr(
        analyze_content_occupancy,
        "read_dot_with_swr_source",
        lambda _path: _noncontiguous_dot(),
    )
    monkeypatch.setattr(
        analyze_content_occupancy.joblib,
        "load",
        lambda _path: pytest.fail("environment loaded before event validation"),
    )

    with pytest.raises(EventDefinitionError, match="noncontiguous"):
        analyze_content_occupancy.session_content_distances("a_01_r1_HPC")


def test_spike_evidence_main_propagates_event_contract_error(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        analyze_event_spike_evidence,
        "analyze_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            EventDefinitionError("stale event labels")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(analyze_event_spike_evidence.__file__),
            "--epoch-identifier",
            "a_01_r1_HPC",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(EventDefinitionError, match="stale event labels"):
        analyze_event_spike_evidence.main()

    assert list(tmp_path.iterdir()) == []


def test_spike_evidence_validates_dot_before_encoding_load(monkeypatch, tmp_path):
    # The frame as the SWR source hands it over (stubbed: no inventory here).
    monkeypatch.setattr(
        analyze_event_spike_evidence,
        "read_dot_with_swr_source",
        lambda _path: _noncontiguous_dot(),
    )
    monkeypatch.setattr(
        analyze_event_spike_evidence,
        "load_encoding",
        lambda _epoch_id: pytest.fail("encoding loaded before event validation"),
    )

    with pytest.raises(EventDefinitionError, match="noncontiguous"):
        analyze_event_spike_evidence.analyze_session(
            "a_01_r1_HPC",
            tmp_path,
            n_sample=1,
        )
