"""Reading which inputs an output was made from: a manifest's recorded digest,
and the count of records that name another digest than the current file's."""

from __future__ import annotations

import hashlib
import json

from continuum_swr_replay.runs.staleness import (
    read_output_manifest,
    recorded_sha256,
    stale_counts,
    table_hashes,
)


def test_recorded_sha256_reads_the_top_level_then_inputs():
    assert recorded_sha256({"event_table": {"sha256": "a"}}, "event_table") == "a"
    assert (
        recorded_sha256({"inputs": {"event_table": {"sha256": "b"}}}, "event_table")
        == "b"
    )
    # the top-level record wins over inputs
    both = {"event_table": {"sha256": "a"}, "inputs": {"event_table": {"sha256": "b"}}}
    assert recorded_sha256(both, "event_table") == "a"


def test_recorded_sha256_is_none_when_nothing_is_recorded():
    assert recorded_sha256({}, "event_table") is None
    assert recorded_sha256({"inputs": {}}, "event_table") is None
    assert recorded_sha256({"inputs": {"event_table": {}}}, "event_table") is None
    assert recorded_sha256({"event_table": {"sha256": "a"}}, "spike_evidence") is None


def test_stale_counts_counts_each_input_separately():
    current = {"event": "e1", "stop": "s1"}
    records = [
        {"event": "e1", "stop": "s1"},
        {"event": "e0", "stop": "s1"},
        {"event": "e0", "stop": "s0"},
        {"stop": "s1"},  # records no event digest: counts as another
    ]
    assert stale_counts(records, current) == {"event": 3, "stop": 1}
    assert list(stale_counts(records, current)) == ["event", "stop"]


def test_stale_counts_reads_a_generator_once_per_input():
    current = {"event": "e1", "stop": "s1"}
    records = ({"event": "e0", "stop": "s0"} for _ in range(2))
    assert stale_counts(records, current) == {"event": 2, "stop": 2}


def test_stale_counts_of_no_records_is_zero():
    assert stale_counts([], {"event": "e1"}) == {"event": 0}


def test_table_hashes_names_the_tables_as_the_provenance_does(tmp_path):
    event, stop = tmp_path / "event.parquet", tmp_path / "stop.parquet"
    event.write_bytes(b"events")
    stop.write_bytes(b"stops")
    assert table_hashes(event, stop) == {
        "canonical_event_table_sha256": hashlib.sha256(b"events").hexdigest(),
        "stop_table_sha256": hashlib.sha256(b"stops").hexdigest(),
    }


def test_read_output_manifest_needs_no_epochs_record(tmp_path):
    record = {"event_table": {"sha256": "a"}, "n_epochs": 3}
    (tmp_path / "manifest.json").write_text(json.dumps(record))
    assert read_output_manifest(tmp_path) == record
