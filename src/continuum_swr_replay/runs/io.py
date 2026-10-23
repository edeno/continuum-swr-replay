"""How a run writes its files: atomically, as strict JSON, and with their hashes.

Every runner's outputs, manifests and caches are written through
``tables.io.write_atomically`` (the one atomic write, below this package so
``tables.io.write_table`` shares it), so a crash or a kill never leaves a
half-written file under its final name, and a reader that trusts any file that
exists (a resume, a cache) never reads one. :func:`replace_outputs` writes one
epoch's files as a set. Records go through :func:`json_ready` before
:func:`write_json`, which refuses NaN and infinity.

The module is named ``io`` like the standard-library module on purpose: the
interpreter loads ``io`` at startup, so a tool run from inside this directory
still gets the standard one, and ``<package>.io`` is a common name for a
package's file handling (``tests/unit/test_package_dir_shadows_no_stdlib.py``
exempts the modules the interpreter preloads).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.tables.io import write_atomically, write_table

__all__ = [
    "array_sha256",
    "json_ready",
    "replace_outputs",
    "sha256_file",
    "write_json",
]


def sha256_file(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks (no whole-file load)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: NDArray) -> str:
    """The sha256 of an array's float64 bytes, in C order."""
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).data
    ).hexdigest()


def json_ready(value: Any) -> Any:
    """``value`` in strict JSON types, for a manifest or a record.

    Mappings (``MappingProxyType`` included) become dicts with string keys;
    arrays, lists and tuples become lists; numpy scalars become Python ones;
    and a non-finite float becomes None. Anything else is returned as it is.
    """
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(
    path: Path,
    content: object,
    *,
    indent: int | None = None,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Write `content` as strict JSON (no NaN or infinity), atomically.

    Parameters
    ----------
    path : Path
        The file; its folder is created if missing.
    content : object
        JSON-ready (:func:`json_ready`).
    indent : int, optional
        ``json.dumps``' indent; None (the default) writes one line.
    default : callable, optional
        ``json.dumps``' ``default``, for an object JSON cannot hold; None
        (the default) refuses one. The state-evidence runner passes ``str``.

    Raises
    ------
    ValueError
        If `content` holds a non-finite float.
    TypeError
        If `content` holds an object JSON cannot hold and `default` is None.
    """
    text = json.dumps(content, indent=indent, allow_nan=False, default=default)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, lambda temporary: temporary.write_text(text))


def replace_outputs(
    paths: Mapping[str, Path], outputs: Mapping[str, pd.DataFrame | Mapping]
) -> dict[str, dict[str, str]]:
    """Write an epoch's outputs and remove its others; on a failure, remove all.

    The folder then matches the manifest: an output not made this time keeps
    no file from an earlier run, and a write that fails (or is interrupted)
    removes every one of `paths` before re-raising, so no part set is left.

    Parameters
    ----------
    paths : Mapping[str, Path]
        Every file the epoch can write, by output name.
    outputs : Mapping[str, pd.DataFrame | Mapping]
        By name in `paths`: a table, written as parquet with its ``attrs``
        (``tables.io.write_table``), or any other content, written as strict
        JSON (``write_json``).

    Returns
    -------
    dict[str, dict[str, str]]
        Each written file's ``path`` and ``sha256``, by name.

    Raises
    ------
    ValueError
        If `outputs` names an output not in `paths` (a misspelt name would
        otherwise drop that output without a word); nothing is written or
        removed.
    """
    unknown = sorted(set(outputs) - set(paths))
    if unknown:
        raise ValueError(
            f"replace_outputs: outputs {unknown} are not among the epoch's files "
            f"{sorted(paths)}; name each output as its output_paths entry does"
        )
    written = {}
    try:
        for name, path in paths.items():
            if name not in outputs:
                path.unlink(missing_ok=True)
                continue
            content = outputs[name]
            if isinstance(content, pd.DataFrame):
                write_table(content, path)
            else:
                write_json(path, content)
            written[name] = {"path": str(path), "sha256": sha256_file(path)}
    except BaseException:
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise
    return written
