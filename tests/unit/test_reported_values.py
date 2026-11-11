"""What a reported value and a summary refuse, and how input files are recorded."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from continuum_swr_replay.reported.values import (
    Summary,
    Value,
    file_record,
    files_record,
)

GOOD: dict[str, Any] = {
    "key": "non_swr_share",
    "value": 0.5,
    "kind": "fraction",
    "style": "pct",
    "unit": "",
    "description": "the share of events outside SWRs",
    "questions": ("2d.2",),
    "figures": (),
    "source": "the event table",
    "function": "event_cohort_values",
    "pending": None,
}


def test_a_well_formed_value_is_accepted():
    assert Value(**GOOD).value == 0.5


@pytest.mark.parametrize(
    "change, error",
    [
        # a key becomes a LaTeX macro name: lower-case words only, no digits
        ({"key": "share_2"}, "lower-case words"),
        ({"key": "Share"}, "lower-case words"),
        ({"kind": "ratio"}, "kind 'ratio'"),
        ({"style": "sig3"}, "style 'sig3'"),
        ({"value": float("nan")}, "not finite"),
        ({"value": float("inf")}, "not finite"),
        ({"value": 1.2}, r"must be in \[0, 1\]"),
        ({"value": -0.1}, r"must be in \[0, 1\]"),
        ({"kind": "count", "style": "count", "value": 2.5}, "must be whole"),
        ({"questions": ()}, "name the questions"),
        ({"function": "event cohort"}, "not a function's name"),
        # an empty note would be pending in one place and not in another
        ({"pending": ""}, "must say why"),
        ({"pending": "  "}, "must say why"),
        # a second line would fall outside the macro file's comment
        ({"description": "a share\nof events"}, "single lines"),
        ({"pending": "rerun it\nfirst"}, "single lines"),
    ],
)
def test_a_malformed_value_is_refused(change, error):
    with pytest.raises(ValueError, match=error):
        Value(**{**GOOD, **change})


def test_a_summary_refuses_a_second_value_with_the_same_key():
    summary = Summary(name="events", produced_by="by")
    summary.add(Value(**GOOD))
    with pytest.raises(ValueError, match="duplicate key non_swr_share"):
        summary.add(Value(**{**GOOD, "value": 0.6}))
    assert summary.values["non_swr_share"].value == 0.5


def test_a_file_record_is_the_path_under_the_root_and_the_content_s_digest(tmp_path):
    table = tmp_path / "data" / "table.csv"
    table.parent.mkdir()
    table.write_bytes(b"a\n1\n")

    assert file_record(table, tmp_path) == {
        "path": "data/table.csv",
        "sha256": hashlib.sha256(b"a\n1\n").hexdigest(),
    }


def test_a_file_set_s_digest_changes_with_any_file_s_content(tmp_path):
    for epoch, content in (("a", b"1"), ("b", b"2")):
        (tmp_path / epoch).mkdir()
        (tmp_path / epoch / "metrics.parquet").write_bytes(content)
    paths = sorted(tmp_path.glob("*/metrics.parquet"))

    record = files_record("metrics.parquet", paths, tmp_path)
    assert record["n_files"] == 2

    (tmp_path / "b" / "metrics.parquet").write_bytes(b"3")
    changed = files_record("metrics.parquet", paths, tmp_path)
    assert changed["sha256"] != record["sha256"]
    assert changed["path"] == record["path"]
