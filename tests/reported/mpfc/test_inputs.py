"""The mPFC summary's bundles: values read from bundles fit on other tables than
the current ones are pending, and the note says how many."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from continuum_swr_replay.reported.mpfc.inputs import staleness
from continuum_swr_replay.runs.staleness import table_hashes


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tables(tmp_path) -> tuple[Path, Path]:
    event, stop = tmp_path / "event_table.parquet", tmp_path / "stop_table.parquet"
    event.write_bytes(b"events")
    stop.write_bytes(b"stops")
    return event, stop


def test_bundle_staleness_names_the_count(tables):
    event, stop = tables
    fresh = {
        "canonical_event_table_sha256": _sha(event),
        "stop_table_sha256": _sha(stop),
    }
    sidecars = [dict(fresh), dict(fresh), dict(fresh)]
    assert staleness(sidecars, table_hashes(event, stop)) is None
    sidecars[1]["stop_table_sha256"] = "old"
    note = staleness(sidecars, table_hashes(event, stop))
    assert note.startswith("the mPFC bundles were made from other inputs")
    assert "stop_table_sha256 differs in 1 of 3" in note
    assert "canonical_event_table_sha256" not in note
