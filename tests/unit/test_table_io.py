"""Behavioral tests for the canonical-table storage layer.

Every canonical table now goes through :mod:`continuum_swr_replay.table_io`, so
a silent fidelity loss here corrupts every downstream fit at once. These tests
pin the properties the analysis code actually relies on: values, dtypes and
``attrs`` survive a round-trip; a parquet table wins over a stale pickle beside
it; and a missing table raises rather than returning something empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.table_io import (
    parquet_path,
    read_table,
    table_exists,
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
    cols = ["event_id", "animal", "arm", "prob"]
    df = pd.DataFrame(columns=cols)

    loaded = read_table(write_table(df, tmp_path / "t"))

    assert list(loaded.columns) == cols
    assert len(loaded) == 0


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
        "canonical tables must be read via table_io.read_table, not pd.read_pickle: "
        + ", ".join(offenders)
    )
