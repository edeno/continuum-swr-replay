"""Behavioral tests for the canonical-table storage layer.

Every canonical table now goes through :mod:`continuum_swr_replay.tables.io`, so
a silent fidelity loss here corrupts every downstream fit at once. These tests
pin the properties the analysis code actually relies on: values, dtypes and
``attrs`` survive a round-trip; a parquet table wins over a stale pickle beside
it; and a missing table raises rather than returning something empty.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.events.analysed_time import publish_swr_table
from continuum_swr_replay.tables.io import (
    parquet_path,
    publish_canonical,
    read_table,
    table_exists,
    umask_file_mode,
    write_table,
)


def _mixed_frame() -> pd.DataFrame:
    """A frame carrying every dtype family the canonical tables actually use."""
    return pd.DataFrame(
        {
            "float_with_nan": [1.5, np.nan, 3.25, -0.0],
            "int64": np.array([1, 2, 3, 4], dtype="int64"),
            "uint64": np.array([10, 20, 30, 40], dtype="uint64"),
            "bool": [True, False, True, False],
            "nullable_boolean": pd.array([True, None, False, True], dtype="boolean"),
            "nullable_int": pd.array([1, None, 3, 4], dtype="Int64"),
            "string": ["chimi", "j16", "peanut", "senor"],
            "object_bool": np.array([True, None, False, True], dtype=object),
        }
    )


def test_roundtrip_preserves_values_dtypes_and_attrs(tmp_path):
    """The whole point of the layer: what goes in comes back out unchanged."""
    df = _mixed_frame()
    df.attrs = {"entropy_cov": 0.98, "leaf_agree": 1.0, "source": "test"}

    loaded = read_table(write_table(df, tmp_path / "t"))

    assert loaded.equals(df), "values changed across the round-trip"
    assert loaded.dtypes.astype(str).to_dict() == df.dtypes.astype(str).to_dict()
    assert loaded.attrs == df.attrs, "attrs did not survive"


def test_object_bool_column_keeps_object_dtype(tmp_path):
    """Arrow would hand back a ``boolean`` column for object-dtype bools.

    ``stop_table`` carries such columns and callers do ``.eq(True)`` on them,
    whose result differs between object and nullable-boolean dtype, so the
    original dtype has to be restored explicitly.
    """
    df = pd.DataFrame({"at_well": np.array([True, None, False], dtype=object)})
    assert str(df["at_well"].dtype) == "object"  # the condition under test

    loaded = read_table(write_table(df, tmp_path / "t"))

    assert str(loaded["at_well"].dtype) == "object"
    assert loaded["at_well"].eq(True).tolist() == df["at_well"].eq(True).tolist()


def test_non_default_index_is_preserved(tmp_path):
    """Some callers index by ``stop_id``; a dropped index would misalign joins."""
    df = _mixed_frame().set_index("string")

    loaded = read_table(write_table(df, tmp_path / "t"))

    assert loaded.index.equals(df.index)
    assert loaded.index.name == "string"


def test_parquet_wins_over_a_stale_pickle_sibling(tmp_path):
    """A leftover pickle beside a parquet must never be the one that is read.

    This is the migration's central hazard: the directories held both formats
    for a while, and a reader that picked the pickle would silently serve the
    pre-migration table.
    """
    fresh = pd.DataFrame({"x": [1, 2, 3]})
    stale = pd.DataFrame({"x": [999, 999, 999]})
    write_table(fresh, tmp_path / "t")
    stale.to_pickle(tmp_path / "t.pkl")
    assert (tmp_path / "t.pkl").exists()  # the condition under test

    assert read_table(tmp_path / "t.pkl")["x"].tolist() == [1, 2, 3]


def test_falls_back_to_pickle_when_no_parquet(tmp_path):
    """Un-migrated directories (the historical backups) stay readable."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    df.to_pickle(tmp_path / "t.pkl")
    assert not parquet_path(tmp_path / "t").exists()  # the condition under test

    assert read_table(tmp_path / "t.pkl").equals(df)


def test_missing_table_raises_rather_than_returning_empty(tmp_path):
    """A missing table must fail loudly; an empty frame would fit silently."""
    with pytest.raises(FileNotFoundError, match="no table at"):
        read_table(tmp_path / "absent.pkl")


def test_column_subset_reads_only_the_requested_columns(tmp_path):
    df = _mixed_frame()
    path = write_table(df, tmp_path / "t")

    loaded = read_table(path, columns=["int64", "string"])

    assert list(loaded.columns) == ["int64", "string"]
    assert loaded["int64"].tolist() == df["int64"].tolist()


def test_column_subset_works_on_the_pickle_fallback(tmp_path):
    """Same contract in both formats, so callers need not know which they hit."""
    df = _mixed_frame()
    df.to_pickle(tmp_path / "t.pkl")

    loaded = read_table(tmp_path / "t.pkl", columns=["int64", "string"])

    assert list(loaded.columns) == ["int64", "string"]


@pytest.mark.parametrize("suffix", [".pkl", ".parquet", ""])
def test_table_exists_finds_either_format_from_any_suffix(tmp_path, suffix):
    """Callers guard with whichever path they historically held."""
    assert not table_exists(tmp_path / f"t{suffix}")
    write_table(pd.DataFrame({"x": [1]}), tmp_path / "t")
    assert table_exists(tmp_path / f"t{suffix}")


def test_empty_frame_with_schema_roundtrips_with_column_order(tmp_path):
    """The empty-fallback path writes a schema-only frame; order is a contract."""
    cols = ["event_id", "animal", "candidate_arm", "prob"]
    df = pd.DataFrame(columns=cols)

    loaded = read_table(write_table(df, tmp_path / "t"))

    assert list(loaded.columns) == cols
    assert len(loaded) == 0


def test_written_table_is_group_readable(tmp_path):
    """mkstemp forces 0600; the promoted table must keep the umask-normal mode.

    The shared ``Processed-Data`` tree is read by other accounts, so an
    owner-only canonical table breaks every non-owner reader with a
    PermissionError. The atomic writer goes through ``mkstemp`` (0600) plus
    ``os.replace`` (which preserves the mode), so without an explicit chmod the
    file lands 0600 instead of the 0644 a plain ``to_parquet`` would produce.
    """
    old = os.umask(0o022)  # a normal, deterministic umask for the assertion
    try:
        path = write_table(pd.DataFrame({"x": [1, 2, 3]}), tmp_path / "t")
    finally:
        os.umask(old)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o644, f"expected 0o644, got {oct(mode)}"


def test_umask_file_mode_respects_umask():
    """The helper returns 0666 masked by the current umask and restores it."""
    old = os.umask(0o077)
    try:
        assert umask_file_mode() == 0o600
        assert os.umask(0o022) == 0o077, "umask was not restored after the query"
    finally:
        os.umask(old)


def _valid_stop_frame() -> pd.DataFrame:
    """A stop table as its producer hands it over: stamped with the SWR source."""
    from continuum_swr_replay.analysis.events.analysed_time import stamp_swr_source

    return stamp_swr_source(
        pd.DataFrame(
            {
                "animal": ["chimi", "chimi"],
                "session_date": [20200212, 20200212],
                "epoch": ["02_r1", "02_r1"],
                "stop_id": [10, 11],
                "n_selected_events": [1, 0],
            }
        )
    )


@pytest.mark.parametrize("stamp", [None, "kay@earlier+artifacts@none"])
def test_publish_swr_table_refuses_an_swr_table_from_another_source(tmp_path, stamp):
    """A table built from SWR labels must name the configured SWR source."""
    from continuum_swr_replay.analysis.events.swr_source import SwrSourceError

    frame = _valid_stop_frame()
    frame.attrs.pop("swr_source")
    if stamp is not None:
        frame.attrs["swr_source"] = stamp

    with pytest.raises(SwrSourceError, match="the stop table to publish"):
        publish_swr_table(frame, tmp_path, table="stop", basename="stop_table")
    assert not (tmp_path / "stop_table.parquet").exists()


@pytest.mark.parametrize("stamp", ["missing", None])
def test_publish_canonical_refuses_an_swr_table_without_a_source_stamp(tmp_path, stamp):
    """The foundation door alone still refuses an SWR table that names no
    SWR source (no stamp, or a None one), pointing at publish_swr_table;
    nothing is written."""
    frame = _valid_stop_frame()
    frame.attrs.pop("swr_source")
    if stamp != "missing":
        frame.attrs["swr_source"] = stamp
    with pytest.raises(TypeError, match="publish_swr_table"):
        publish_canonical(frame, tmp_path, table="stop", basename="stop_table")
    assert not (tmp_path / "stop_table.parquet").exists()


def test_publish_canonical_keeps_the_swr_stamp_and_needs_none_for_qvals(tmp_path):
    from continuum_swr_replay.analysis.events.swr_source import swr_source_name

    stop = read_table(
        publish_swr_table(
            _valid_stop_frame(), tmp_path, table="stop", basename="stop_table"
        )
    )
    qvals = read_table(
        publish_canonical(
            pd.DataFrame({"stop_id": [10, 11]}), tmp_path, table="qvals", basename="q"
        )
    )

    assert stop.attrs["swr_source"] == swr_source_name()
    assert "swr_source" not in qvals.attrs


def test_publish_canonical_stamps_and_writes(tmp_path):
    """publish_canonical stamps the schema and writes; equals a manual stamp+write."""
    from continuum_swr_replay.tables.covariate_registry import (
        stamp_covariate_schema,
    )

    frame = _valid_stop_frame()
    path = publish_canonical(frame, tmp_path, table="stop", basename="stop_table")
    loaded = read_table(path)

    assert loaded.attrs.get("covariate_schema_table") == "stop"
    manual = stamp_covariate_schema(_valid_stop_frame(), table="stop")
    assert loaded.attrs.get("covariate_schema_version") == manual.attrs.get(
        "covariate_schema_version"
    )
    assert loaded.drop(columns=[]).equals(manual)


def test_publish_canonical_rejects_duplicate_rows(tmp_path):
    """A frame that violates the row-key contract must not be written."""
    bad = _valid_stop_frame()
    bad.loc[1, "stop_id"] = 10  # duplicate stop_id
    with pytest.raises(ValueError):
        publish_canonical(bad, tmp_path, table="stop", basename="stop_table")
    assert not (tmp_path / "stop_table.parquet").exists()


def test_publish_canonical_basename_override(tmp_path):
    path = publish_canonical(
        _valid_stop_frame(), tmp_path, table="stop", basename="renamed"
    )
    assert path == parquet_path(tmp_path / "renamed")
    assert path.exists()


CANONICAL_TABLE_NAMES = (
    "event_table",
    "event_arm_table",
    "event_patch_table",
    "event_sequence_table",
    "stop_table",
    "qvals_table",
    "event_onset_table",
)


def test_no_module_reads_a_canonical_table_with_read_pickle():
    """A canonical table must never be read with ``pd.read_pickle``.

    The tables live beside their retired pickles in the archive directories, and
    for a while lived beside them outright. A reader on ``pd.read_pickle`` either
    fails outright (the pickle has moved) or, worse, quietly serves a
    pre-migration copy -- which is how a whole value block once came back NaN
    without anything raising.

    This catches the common form, where the filename is a literal on the same
    line. It cannot see paths built through variables, so it is a backstop, not
    a proof.
    """
    repo_root = Path(__file__).resolve().parents[2]
    offenders = []
    for directory in ("src", "scripts", "tests"):
        for path in (repo_root / directory).rglob("*.py"):
            if "archive" in path.parts or path.name == "convert_tables_to_parquet.py":
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if "pd.read_pickle(" not in line:
                    continue
                if any(f"{name}.pkl" in line for name in CANONICAL_TABLE_NAMES):
                    offenders.append(f"{path.relative_to(repo_root)}:{number}")

    assert not offenders, (
        "canonical tables must be read via tables.io.read_table, not pd.read_pickle: "
        + ", ".join(offenders)
    )
