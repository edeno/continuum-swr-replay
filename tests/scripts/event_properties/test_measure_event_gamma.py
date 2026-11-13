"""``measure_event_gamma.epochs_to_measure``: the epochs the gamma run measures
are the event table's, each with the record the SWR source serves it from."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.swr_source import SwrSourceError
from scripts.event_properties import measure_event_gamma as script
from tests.support._swr_inventory_fixtures import use_swr_source, write_swr_inventory

OK = "j1620210706_02_r1_HPC"
PARTIAL = "j1620210706_04_r2_HPC"


def _events(*epochs: str) -> pd.DataFrame:
    """Two events per epoch, with the session keys the event table carries."""
    rows = []
    for epoch in epochs:
        for start in (10.0, 20.0):
            rows.append(
                {
                    "animal": "j16",
                    "session_date": 20210706,
                    "epoch": epoch,
                    "event_id": f"{epoch}-{start}",
                    "event_start_s": start,
                    "event_end_s": start + 0.1,
                    "is_swr": start == 10.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def inventory(processed_data_root, monkeypatch):
    """An SWR run with one ``ok`` epoch and one ``partial`` one whose Kay
    inventory was written (the SWR source serves both)."""
    directory = write_swr_inventory(
        processed_data_root, {OK: [(10.0, 10.05)], PARTIAL: [(10.0, 10.05)]}
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["epochs"][PARTIAL]["status"] = "partial"
    manifest_path.write_text(json.dumps(manifest))
    use_swr_source(monkeypatch, directory)
    return manifest


def test_a_partial_epoch_the_source_serves_is_measured(inventory):
    measured = script.epochs_to_measure(_events("02_r1", "04_r2"))

    assert [identifier for identifier, _, _ in measured] == [OK, PARTIAL]
    for identifier, record, frame in measured:
        assert record == inventory["epochs"][identifier]
        assert frame["event_start_s"].tolist() == [10.0, 20.0]


def test_an_epoch_the_source_does_not_serve_raises(inventory):
    with pytest.raises(SwrSourceError, match="j1620210706_06_r3_HPC"):
        script.epochs_to_measure(_events("02_r1", "06_r3"))
