"""How a run writes its files: atomically, as strict JSON, and with their hashes;
and how a reader finds a run's epochs.

Every runner's outputs, manifests and caches are written through
``tables.io.write_atomically`` (the one atomic write, below this package so
``tables.io.write_table`` shares it), so a crash or a kill never leaves a
half-written file under its final name, and a reader that trusts any file that
exists (a resume, a cache) never reads one. :func:`replace_outputs` writes one
epoch's files as a set. Records go through :func:`json_ready` before
:func:`write_json`, which refuses NaN and infinity.

A run folder holds ``manifest.json`` (:func:`read_manifest`) with a record of
each epoch; only the epochs it marks ``ok`` are read (:func:`ok_epochs`), and
each epoch's table is ``{identifier}_{name}.parquet`` (:func:`output_path`,
:func:`ok_table_paths`).

The module is named ``io`` like the standard-library module on purpose: the
interpreter loads ``io`` at startup, so a tool run from inside this directory
still gets the standard one, and ``<package>.io`` is a common name for a
package's file handling (``tests/repository/test_package_dir_shadows_no_stdlib.py``
exempts the modules the interpreter preloads).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.tables.io import write_atomically, write_table

__all__ = [
    "array_sha256",
    "files_sha256",
    "is_nonempty_file",
    "json_ready",
    "ok_epochs",
    "ok_table_paths",
    "output_path",
    "read_manifest",
    "recorded_output",
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


def files_sha256(paths: Iterable[str | Path]) -> str:
    """One hex sha256 over a set of files: each file's name, then its sha256.

    The files are taken in sorted path order, so the digest does not depend on
    the order given. It changes when a file's bytes or name change, or a file
    is added or removed; the folders the files sit in do not enter it.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(path) for path in paths):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
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
    sort_keys: bool = False,
    final_newline: bool = False,
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
    sort_keys : bool, optional
        ``json.dumps``' ``sort_keys``.
    final_newline : bool, optional
        End the file with a newline.

    Raises
    ------
    ValueError
        If `content` holds a non-finite float.
    TypeError
        If `content` holds an object JSON cannot hold and `default` is None.
    """
    text = json.dumps(
        content, indent=indent, allow_nan=False, default=default, sort_keys=sort_keys
    ) + ("\n" if final_newline else "")
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


def is_nonempty_file(path: Path) -> bool:
    """Return whether an artifact component was completely materialized.

    Parameters
    ----------
    path : Path
        Component to test.

    Returns
    -------
    bool
        True when the path is a regular file of non-zero size.
    """
    return path.is_file() and path.stat().st_size > 0


def output_path(identifier: str, output_dir: Path, name: str) -> Path:
    """Where an epoch's table `name` is in the run folder `output_dir`."""
    return output_dir / f"{identifier}_{name}.parquet"


def read_manifest(run_dir: Path, *, names_folder: bool = False) -> dict:
    """The run's ``manifest.json``, with its record of each epoch.

    Parameters
    ----------
    run_dir : Path
        The run folder.
    names_folder : bool, optional
        Also require the manifest's ``run_id`` to be the folder's name, so a
        folder copied or renamed from another run is not read as this one. A
        runner whose run id names its folder records it; the state-evidence
        run records none.

    Raises
    ------
    ValueError
        If the run has no manifest, it cannot be read as JSON, it holds no
        ``epochs`` record, or (with `names_folder`) it names another run.
    """
    path = run_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"no readable run manifest at {path}: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("epochs"), dict):
        raise ValueError(f"no readable run manifest at {path}: no epochs record")
    if names_folder and manifest.get("run_id") != run_dir.name:
        raise ValueError(
            f"{path} is run {manifest.get('run_id')!r}, not the folder's "
            f"{run_dir.name!r}"
        )
    return manifest


def recorded_output(path: Path, record: Mapping, name: str) -> str:
    """Check a run's output file against its epoch's manifest record; return its
    sha256.

    Parameters
    ----------
    path : Path
        The file.
    record : Mapping
        The epoch's manifest record, whose ``outputs[name]["sha256"]`` is the
        file's sha256 when the runner wrote it.
    name : str
        The output's name in the record.

    Raises
    ------
    ValueError
        If the file is missing, or its sha256 is not the recorded one (an
        output recorded without a sha256 is a mismatch).
    """
    if not path.exists():
        raise ValueError(f"{path.name} is missing from {path.parent}")
    output = (record.get("outputs") or {}).get(name)
    expected = output.get("sha256") if isinstance(output, Mapping) else None
    sha = sha256_file(path)
    if sha != expected:
        raise ValueError(
            f"{path.name} has sha256 {sha[:12]}, not the {str(expected)[:12]} the "
            "manifest recorded"
        )
    return sha


def ok_epochs(manifest: Mapping) -> list[str]:
    """The epochs the run's manifest marks ``ok``, sorted: the only epochs of a
    run that are read. A failed epoch's tables, or tables a manifest does not
    know, may still be in the run folder."""
    return sorted(i for i, r in manifest["epochs"].items() if r["status"] == "ok")


def ok_table_paths(run_dir: Path, name: str) -> list[Path]:
    """Table ``name`` of each epoch the run's manifest marks ``ok``, in the
    order of :func:`ok_epochs`; an empty list when it marks none.

    Raises
    ------
    ValueError
        If the run has no readable manifest (:func:`read_manifest`).
    """
    return [
        output_path(identifier, run_dir, name)
        for identifier in ok_epochs(read_manifest(run_dir))
    ]
