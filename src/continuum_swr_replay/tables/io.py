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

The module is named ``io`` like the standard-library module on purpose: the
interpreter loads ``io`` at startup, so a tool run from inside this directory
still gets the standard one, and ``<package>.io`` is a common name for a
package's file handling (``tests/repository/test_package_dir_shadows_no_stdlib.py``
exempts the modules the interpreter preloads).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from continuum_swr_replay.paths import CANONICAL_TABLES
from continuum_swr_replay.tables.covariate_registry import (
    COVARIATE_SCHEMA_VERSION,
    stamp_covariate_schema,
)

__all__ = [
    "read_table",
    "read_attrs",
    "write_table",
    "publish_canonical",
    "SWR_SOURCE_ATTR",
    "SWR_SOURCE_TABLES",
    "parquet_path",
    "resolved_table_path",
    "table_exists",
    "umask_file_mode",
    "write_atomically",
]


_ATTRS_KEY = b"continuum_swr_replay.attrs"

SWR_SOURCE_ATTR = "swr_source"
"""The ``attrs`` key of a table's SWR-source stamp (``analysed_time.stamp_swr_source``)."""

SWR_SOURCE_TABLES = frozenset(CANONICAL_TABLES) - {"qvals"}
"""The canonical tables (covariate-schema keys) built from decode frames read
through the SWR source (``analysed_time``); ``publish_canonical`` refuses one
without a source stamp, and ``analysed_time.publish_swr_table`` one without the
source's current stamp. qvals, built from behavior alone, is the
one canonical table that is not among them. event_sequence holds no SWR column
but is published with its atomic set, so it carries the set's stamp."""


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

    A table built from SWR labels (:data:`SWR_SOURCE_TABLES`) is published
    through ``analysed_time.publish_swr_table``, which refuses it unless it
    carries the SWR source's current stamp (only its producer can give it,
    ``analysed_time.stamp_swr_source``) and then calls this function. Here, as
    foundation code that imports no analysis code, such a table is refused only
    when it carries no stamp at all, so none lands without naming the SWRs it
    was built from.

    Raises
    ------
    TypeError
        If `table` is one of :data:`SWR_SOURCE_TABLES` and `frame` carries no
        SWR-source stamp (:data:`SWR_SOURCE_ATTR`).
    """
    if table in SWR_SOURCE_TABLES and not isinstance(
        frame.attrs.get(SWR_SOURCE_ATTR), str
    ):
        raise TypeError(
            f"the {table} table is built from SWR labels but carries no SWR "
            "source stamp; publish it with analysed_time.publish_swr_table"
        )

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
    """Return the parquet file :func:`read_table` opens for `path`.

    A suffix-less path (a table basename such as ``root / "stop_table"``) or a
    ``.parquet`` path names the parquet file. A ``.pkl`` path is refused: the
    tables are parquet, and an archived pickle is read with
    ``pandas.read_pickle``, explicitly.

    Raises
    ------
    ValueError
        If `path` ends in ``.pkl``.
    """
    if Path(path).suffix == ".pkl":
        raise ValueError(
            f"{path} is a pickle path, but the canonical tables are parquet: pass "
            "the .parquet path (read an archived pickle with pandas.read_pickle)"
        )
    return parquet_path(path)


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
    # Atomic: a half-written table would break the regeneration reducers'
    # atomic event/stop table set.
    write_atomically(out, lambda temporary: pq.write_table(table, temporary, **kwargs))
    return out


def write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Write `path` through a temporary file beside it, so it is never half-written.

    A reader that trusts any file that exists (``results_exist``, a marks
    cache, a resumed manifest, a canonical table) must never meet a truncated
    file under the final name after a crash mid-write. Every atomic write of
    the project goes through here: :func:`write_table`, and ``runs.io``'s
    ``write_json`` and ``replace_outputs``.

    Parameters
    ----------
    path : Path
        The final file. Its folder must exist.
    write : Callable[[Path], object]
        Writes the whole file to the path it is given (for example
        ``detector.save_model``), overwriting the empty file there.

    Notes
    -----
    `write` gets ``.{name}.<random>.tmp`` in the same folder, made by
    ``tempfile.mkstemp``, so two writers of one path, in any process or
    thread, never share it. Only after `write` returns is the file given the
    mode a plain create would have (:func:`umask_file_mode`: mkstemp makes it
    owner-only) and moved onto `path` with ``os.replace``, which is atomic
    within a filesystem. If `write` raises, the temporary file is removed and
    `path` is left as it was. The temporary name does not end in the final
    suffix, so a ``*_environment.pkl``-style glob never picks it up, even
    after a hard kill.
    """
    path = Path(path)
    handle, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(name)
    try:
        write(temporary)
        os.chmod(temporary, umask_file_mode())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_table(
    path: str | Path,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    """Read a canonical table from parquet, restoring its ``attrs``.

    Parameters
    ----------
    path : str or Path
        Table path, ``.parquet`` or without a suffix (:func:`resolved_table_path`).
    columns : list of str, optional
        Read only these columns, without touching the rest of the file.
    filters : list of (column, operator, value), optional
        Read only the rows that pass, as ``pyarrow.parquet.read_table`` takes them.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist.
    ValueError
        If `path` ends in ``.pkl``.
    """
    parquet = resolved_table_path(path)
    if not parquet.exists():
        raise FileNotFoundError(
            f"no table at {parquet} -- was it built? "
            "The canonical tables are produced by "
            "scripts/tables/prepare_replay_content_data.py"
        )

    table = pq.read_table(parquet, columns=columns, filters=filters)
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
        Table path, ``.parquet`` or without a suffix (:func:`resolved_table_path`).

    Returns
    -------
    dict
        The frame's ``attrs``; empty if the file carries none.

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist.
    ValueError
        If `path` ends in ``.pkl``.
    """
    parquet = resolved_table_path(path)
    if not parquet.exists():
        raise FileNotFoundError(f"no table at {parquet}")
    return _attrs_from_metadata(pq.read_schema(parquet).metadata)


def table_exists(path: str | Path) -> bool:
    """Whether the parquet file of table `path` is on disk.

    Raises
    ------
    ValueError
        If `path` ends in ``.pkl``.
    """
    return resolved_table_path(path).exists()
