"""On-disk format for the canonical analysis tables.

The canonical tables (events, event-arms, event-patches, sequences, stops,
qvals) are plain dataframes of numbers and short strings, and they are read far
more often than they are written. Parquet stores them columnar and compressed:
roughly half the bytes of a pickle, and a fraction of the read time when a
caller wants a handful of columns out of a hundred.

The reason to move off pickle is not size, though -- it is that a pickle is
only readable by the library versions that wrote it. A numpy 2 pickle cannot
be opened under numpy 1.x, which turned a routine Spyglass upgrade into a
choice between the upgrade and every artifact on disk. Parquet is a versioned
format with a written spec, so an artifact outlives the environment.

One thing parquet does not carry on its own is ``DataFrame.attrs`` -- the
provenance and coverage diagnostics that several builders stash on the frame
(e.g. ``qvals_table``'s ``entropy_cov``) -- so it is round-tripped explicitly
here as JSON in the parquet file's key-value metadata. Column dtypes, including
the object-dtype boolean columns in ``stop_table``, are restored by arrow's own
pandas metadata and need no help.

Only dataframes belong here. Fitted models, detectors, environments and the
session cache hold live objects (statsmodels results, NetworkX graphs) that
have no columnar representation; those stay pickled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["read_table", "write_table", "parquet_path", "table_exists"]

_ATTRS_KEY = b"continuum_swr_replay.attrs"


def parquet_path(path: str | Path) -> Path:
    """Return `path` with a ``.parquet`` suffix, whatever suffix it arrived with."""
    return Path(path).with_suffix(".parquet")


def write_table(frame: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write `frame` to parquet at `path`, preserving ``attrs`` and dtypes.

    Parameters
    ----------
    frame : pd.DataFrame
        Table to write. A non-default index is preserved.
    path : str or Path
        Destination. Any suffix is replaced with ``.parquet``.
    **kwargs
        Passed to ``pyarrow.parquet.write_table`` (e.g. ``compression``).

    Returns
    -------
    Path
        The path written.
    """
    out = parquet_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(frame, preserve_index=True)
    metadata = dict(table.schema.metadata or {})
    metadata[_ATTRS_KEY] = json.dumps(frame.attrs, default=str).encode()
    table = table.replace_schema_metadata(metadata)

    kwargs.setdefault("compression", "zstd")
    pq.write_table(table, out, **kwargs)
    return out


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a canonical table, preferring parquet and falling back to pickle.

    Callers pass whichever path they have written historically -- typically
    ``...table.pkl``. If the parquet sibling exists it wins, so a directory can
    be migrated a table at a time without touching every reader at once.

    Parameters
    ----------
    path : str or Path
        Table path, with either suffix.
    columns : list of str, optional
        Read only these columns. Parquet reads them without touching the rest
        of the file; the pickle fallback loads everything and then subsets, so
        the speedup only applies once the table is migrated.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If neither the parquet nor the pickle form exists.
    """
    parquet = parquet_path(path)
    if not parquet.exists():
        pickle = Path(path).with_suffix(".pkl")
        if not pickle.exists():
            raise FileNotFoundError(
                f"no table at {parquet} or {pickle} -- was it built? "
                "The canonical tables are produced by "
                "scripts/decoding/prepare_replay_content_data.py"
            )
        frame = pd.read_pickle(pickle)
        return frame if columns is None else frame[columns]

    table = pq.read_table(parquet, columns=columns)
    metadata = table.schema.metadata or {}
    frame = table.to_pandas()

    frame.attrs = json.loads(metadata.get(_ATTRS_KEY, b"{}"))
    return frame


def table_exists(path: str | Path) -> bool:
    """Whether a table is on disk in either format.

    Callers that guard on a table's presence -- to raise a helpful error, or to
    skip a test that needs materialized data -- must not test the pickle path
    alone once a directory is migrated, or they silently report "missing" for a
    table that is sitting right there in parquet.
    """
    return parquet_path(path).exists() or Path(path).with_suffix(".pkl").exists()
