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

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "read_table",
    "read_attrs",
    "write_table",
    "publish_canonical",
    "parquet_path",
    "resolved_table_path",
    "table_exists",
    "umask_file_mode",
    "sha256_file",
]


def sha256_file(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks (no whole-file load)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ATTRS_KEY = b"continuum_swr_replay.attrs"


def umask_file_mode() -> int:
    """Return the file mode a normal umask-respecting create would produce.

    ``tempfile.mkstemp`` hard-codes mode ``0o600`` regardless of the process
    umask, and ``os.replace`` preserves that mode. Every atomic writer here goes
    through ``mkstemp``, so without this the promoted file lands owner-only
    (``-rw-------``) instead of the group-readable ``-rw-r--r--`` a plain
    ``open(..., "wb")`` would have produced -- which silently breaks any
    group/other reader of the shared ``Processed-Data`` tree. Reading the umask
    is a get-and-restore (there is no read-only query), so the window in which
    the umask is 0 is kept to these two lines.
    """
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def publish_canonical(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    table: str,
    basename: str | None = None,
) -> Path:
    """Stamp, validate, and atomically write a canonical table.

    The one supported way to persist a canonical table. ``table`` is the schema
    key (``"event"``, ``"event_arm"``, ``"stop"``, ``"qvals"``, ...). The frame
    is stamped with the covariate schema -- which also checks the registered row
    keys are unique and nonmissing and the vocabulary is canonical
    (``stamp_covariate_schema`` calls ``assert_canonical_vocabulary`` +
    ``assert_unique_scientific_rows``) -- and then written atomically. A writer
    that bypasses this and calls ``write_table`` directly on a canonical frame
    persists an unstamped table that ``require_covariate_schema`` rejects
    downstream at fit time; routing through here makes the stamp a property of
    the write, not a convention each caller must remember.

    ``basename`` writes to ``path / basename`` (for the source-tagged ContFrag
    reducers that build the filename from an event source and brain area).

    A frame already carrying the current stamp for this table (a caller that
    stamped it and did not mutate it since) skips the re-validation -- the two
    full-frame ``isna``/``duplicated`` passes in ``stamp_covariate_schema`` --
    and is written as-is; the stamp is the record that it was validated.

    A table built from SWR labels (``swr_source.SWR_SOURCE_TABLES``) must also
    carry the SWR source's current stamp, which only its producer can give it
    (``swr_source.stamp_swr_source``, after reading its decode frames through
    the source). One without it, or with another source's, is refused, so no
    canonical table lands without naming the SWRs it was built from.
    """
    # Function-local imports: covariate_registry's and swr_source's consumers
    # import table_io, so a module-level import here would close a cycle.
    from continuum_swr_replay.analysis.hippocampus.swr_source import (
        SWR_SOURCE_TABLES,
        require_swr_source,
    )
    from continuum_swr_replay.analysis.replay.covariate_registry import (
        COVARIATE_SCHEMA_VERSION,
        stamp_covariate_schema,
    )

    if table in SWR_SOURCE_TABLES:
        require_swr_source(frame, what=f"the {table} table to publish")

    already_stamped = (
        frame.attrs.get("covariate_schema_version") == COVARIATE_SCHEMA_VERSION
        and frame.attrs.get("covariate_schema_table") == table
    )
    stamped = frame if already_stamped else stamp_covariate_schema(frame, table=table)
    target = Path(path) if basename is None else Path(path) / basename
    return write_table(stamped, target)


def parquet_path(path: str | Path) -> Path:
    """Return `path` with a ``.parquet`` suffix, whatever suffix it arrived with."""
    return Path(path).with_suffix(".parquet")


def resolved_table_path(path: str | Path) -> Path:
    """Return the file :func:`read_table` would actually open for `path`.

    Parquet wins when it exists, otherwise the pickle sibling. Callers that
    hash the table they read must resolve it through this function rather than
    assume parquet: a directory still holding the pickle form would otherwise
    be hashed at a path that does not exist.
    """
    parquet = parquet_path(path)
    return parquet if parquet.exists() else Path(path).with_suffix(".pkl")


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
    # Atomic write: a crash or kill mid-write leaves the destination untouched
    # rather than a truncated parquet that a later read would silently accept or
    # loudly fail on. The temp file shares the destination directory so
    # ``os.replace`` is a same-filesystem atomic rename. This matters for the
    # multi-table regeneration reducers, where a half-written table would break
    # the atomic event/stop table set.
    handle, tmp_name = tempfile.mkstemp(
        dir=out.parent, prefix=f".{out.name}.", suffix=".tmp"
    )
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        pq.write_table(table, tmp, **kwargs)
        # mkstemp created tmp at 0600; restore the umask-respecting mode a plain
        # write would have produced before promoting it (see umask_file_mode).
        os.chmod(tmp, umask_file_mode())
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
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
    frame = table.to_pandas()

    frame.attrs = _attrs_from_metadata(table.schema.metadata)
    return frame


def _attrs_from_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, Any]:
    """Decode the ``attrs`` blob ``write_table`` stashed in parquet metadata."""
    return json.loads((metadata or {}).get(_ATTRS_KEY, b"{}"))


def read_attrs(path: str | Path) -> dict[str, Any]:
    """Return the ``attrs`` a :func:`read_table` of `path` would restore.

    Reads the parquet footer only -- no row groups -- so a caller that wants
    just the provenance stamp (``covariate_schema_version`` and friends) off a
    multi-hundred-megabyte table does not pay for the table.

    Parameters
    ----------
    path : str or Path
        Table path, with either suffix.

    Returns
    -------
    dict
        The frame's ``attrs``; empty if the file carries none.

    Raises
    ------
    FileNotFoundError
        If neither the parquet nor the pickle form exists.
    """
    parquet = parquet_path(path)
    if not parquet.exists():
        pickle = Path(path).with_suffix(".pkl")
        if not pickle.exists():
            raise FileNotFoundError(f"no table at {parquet} or {pickle}")
        return dict(pd.read_pickle(pickle).attrs)
    return _attrs_from_metadata(pq.read_schema(parquet).metadata)


def table_exists(path: str | Path) -> bool:
    """Whether a table is on disk in either format.

    Callers that guard on a table's presence -- to raise a helpful error, or to
    skip a test that needs materialized data -- must not test the pickle path
    alone once a directory is migrated, or they silently report "missing" for a
    table that is sitting right there in parquet.
    """
    return parquet_path(path).exists() or Path(path).with_suffix(".pkl").exists()
