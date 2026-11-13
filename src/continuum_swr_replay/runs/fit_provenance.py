"""Sidecar records tying a fitted artifact to the tables it was fit on.

A fitted model pickle is a snapshot of tables that keep moving: the stop table
gains a column, the event tables are regenerated on a new cohort, and the
pickle on disk still looks fine. Nothing in the pickle says which bytes it came
from, so a summary or a figure built from a stale fit is indistinguishable from
a current one.

Every runner that writes a fitted artifact therefore writes
``<artifact-stem>.provenance.json`` beside it, recording the SHA-256, mtime and
covariate schema version of each input table as it was read, plus the commit
and command line that produced the fit. Readers hash the inputs again and
refuse to run against a fit whose inputs have changed since -- a fit with no
sidecar at all counts as stale, because the absence of the record is exactly
the situation the sidecar exists to end.

The hashes describe the bytes the fit read, not the bytes on disk when the fit
finished. A ladder over 218 sessions runs for long enough that a table can be
regenerated underneath it, so each runner calls ``record_inputs`` immediately
after loading its tables and hands the resulting records to the writer at the
end; hashing at write time would certify a table the fit never saw. Each record
carries its own ``hashed_at``, and the sidecar's ``written_at`` is the write
time, so a reader can see the window the two bracket.

The hashed file is the file that was read: a table input is its ``.parquet``
path (the readers refuse ``.pkl``); a ``.pkl`` input is a fitted bundle, hashed
as it is.

A runner that skips a bundle already fit reads its sidecar against the fit's
contract (:func:`matching_sidecar`), and removes a stale bundle whole before
refitting it (:func:`remove_bundle`).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from continuum_swr_replay.runs.io import is_nonempty_file, sha256_file, write_json
from continuum_swr_replay.runs.provenance import git_commit, git_dirty
from continuum_swr_replay.tables.io import read_attrs, resolved_table_path

__all__ = [
    "fit_provenance_path",
    "git_hash",
    "record_inputs",
    "write_fit_provenance",
    "check_fit_provenance",
    "require_fresh_inputs",
    "matching_sidecar",
    "remove_bundle",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]

# One input's provenance: sha256, mtime, covariate_schema_version, hashed_at.
InputRecord = dict[str, object]


def fit_provenance_path(artifact_path: str | Path) -> Path:
    """Return the sidecar path for `artifact_path`.

    Parameters
    ----------
    artifact_path : str or Path
        Path of the fitted artifact (e.g. ``.../all_models.pkl``).

    Returns
    -------
    Path
        ``<artifact-stem>.provenance.json`` in the artifact's directory.
    """
    return Path(artifact_path).with_suffix(".provenance.json")


def _read_git_hash(root: Path) -> str:
    """Return the commit checked out at `root`, suffixed ``-dirty`` if modified.

    ``git rev-parse HEAD``, plus ``-dirty`` when
    ``git status --porcelain --untracked-files=no`` is non-empty
    (``runs.provenance.git_commit`` and ``git_dirty``). Untracked files do not
    count as dirty: they cannot change how the code ran. git runs with
    ``--no-optional-locks``: by default ``status`` refreshes the index and
    writes it back under a lock, and the runners that record this commit run
    in a checkout that other work commits in.
    """
    head = git_commit(root)
    return f"{head}-dirty" if git_dirty(root) else head


# The checkout is read once, when this module is imported, because that is when
# the process loaded its code. A run that lasts hours in a checkout that gets a
# new commit or an edited file still ran what it loaded. A failure is kept and
# raised by ``git_hash``, so a reader that never writes a sidecar can still
# import this module outside a git checkout.
try:
    _LOADED_GIT_HASH: str | None = _read_git_hash(_REPO_ROOT)
    _LOADED_GIT_ERROR: Exception | None = None
except (OSError, subprocess.CalledProcessError) as error:
    _LOADED_GIT_HASH, _LOADED_GIT_ERROR = None, error


def git_hash() -> str:
    """Return the commit this process loaded, suffixed ``-dirty`` if it was modified.

    The checkout is read when this module is imported, not when this is
    called, so later commits and file edits do not change the result. A worker
    process (dask, loky, multiprocessing) may import this module later than its
    parent; a sidecar written there should be given the parent's value through
    ``write_fit_provenance(..., commit=...)``.

    Returns
    -------
    str
        ``git rev-parse HEAD`` at import, plus ``-dirty`` when
        ``git status --porcelain --untracked-files=no`` was non-empty then
        (see ``_read_git_hash``).

    Raises
    ------
    RuntimeError
        If the checkout could not be read at import.
    """
    if _LOADED_GIT_HASH is None:
        raise RuntimeError(
            f"could not read the git checkout at {_REPO_ROOT} when "
            "fit_provenance was imported"
        ) from _LOADED_GIT_ERROR
    return _LOADED_GIT_HASH


def _resolve_input(path: str | Path) -> Path:
    """Return the file a fit read for input `path`.

    A ``.pkl`` input is a fitted bundle an aggregate-only run read (fitted
    models stay pickled), hashed as it is; any other input is a table, hashed
    at its parquet file (``tables.io.resolved_table_path``).

    Raises
    ------
    FileNotFoundError
        If that file does not exist.
    """
    path = Path(path)
    resolved = path if path.suffix == ".pkl" else resolved_table_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"no input at {resolved}")
    return resolved


def _input_record(path: str | Path) -> InputRecord:
    """Hash, mtime, covariate schema version and hash time of one input."""
    resolved = _resolve_input(path)
    attrs = (
        pd.read_pickle(resolved).attrs
        if resolved.suffix == ".pkl"
        else read_attrs(resolved)
    )
    return {
        "sha256": sha256_file(resolved),
        "mtime": resolved.stat().st_mtime,
        "covariate_schema_version": attrs.get("covariate_schema_version"),
        "hashed_at": datetime.now(UTC).isoformat(),
    }


def record_inputs(inputs: dict[str, Path]) -> dict[str, InputRecord]:
    """Hash the input tables now, while the fit is reading them.

    Call this immediately after the tables are loaded and pass the result to
    :func:`write_fit_provenance` at the end of the run. Hashing at write time
    instead would record whatever is on disk hours later, certifying a
    regenerated table as the input to a fit that never saw it.

    Parameters
    ----------
    inputs : dict of str to Path
        Display name -> path of each table (``.parquet``) or fitted bundle
        (``.pkl``) the fit was built from.

    Returns
    -------
    dict of str to dict
        Name -> ``{sha256, mtime, covariate_schema_version, hashed_at}``.
    """
    return {name: _input_record(path) for name, path in inputs.items()}


def write_fit_provenance(
    artifact_path: str | Path,
    inputs: dict[str, Path] | dict[str, InputRecord],
    estimator: str,
    cli: list[str],
    extra: Mapping[str, object] | None = None,
    commit: str | None = None,
) -> Path:
    """Write the provenance sidecar for a fitted artifact.

    Parameters
    ----------
    artifact_path : str or Path
        The artifact just written (one sidecar per artifact).
    inputs : dict of str to Path, or dict of str to dict
        Either the records :func:`record_inputs` returned when the fit loaded
        its tables -- the form every long-running runner uses -- or bare paths,
        which are hashed here and so describe the tables as of *now*.
    estimator : str
        The estimator that produced the fits (e.g. ``"poisson_glm"``).
    cli : list of str
        The command line that produced the artifact: the program
        (``sys.argv[0]``) and the arguments the script's ``main`` parsed.
    extra : Mapping of str to object, optional
        Further JSON-serializable facts about how the fit was produced, merged
        into the record (e.g. whether the response was rescaled to the fitted
        candidate set). Keys the record already defines are refused.
    commit : str, optional
        The commit to record as ``git_hash``. Omitted, it is this process's
        :func:`git_hash`, read when the code was loaded. A sidecar written in a
        worker process takes its parent's value here, because the worker may
        have imported the code after the checkout moved on.

    Returns
    -------
    Path
        The sidecar path written.

    Raises
    ------
    ValueError
        If ``extra`` would overwrite a field the sidecar defines itself, or the
        record holds NaN or infinity: the sidecar is strict JSON, written
        atomically (``runs.io.write_json``), so a reader never sees half of it.
    """
    artifact = Path(artifact_path)
    record = {
        "artifact": artifact.name,
        "written_at": datetime.now(UTC).isoformat(),
        "git_hash": git_hash() if commit is None else commit,
        "estimator": estimator,
        "inputs": {
            name: value if isinstance(value, dict) else _input_record(value)
            for name, value in inputs.items()
        },
        "cli": list(cli),
    }
    if extra:
        clashes = sorted(set(extra) & set(record))
        if clashes:
            raise ValueError(
                f"provenance extras {clashes} would overwrite the sidecar's own "
                "fields; name them differently"
            )
        record.update(extra)
    out = fit_provenance_path(artifact)
    write_json(out, record, indent=2, sort_keys=True, final_newline=True)
    return out


def check_fit_provenance(
    artifact_path: str | Path, inputs: dict[str, Path]
) -> list[str]:
    """Return the names of `inputs` that have changed since the fit was written.

    Parameters
    ----------
    artifact_path : str or Path
        The fitted artifact whose sidecar is checked.
    inputs : dict of str to Path
        The same name -> path mapping the runner recorded.

    Returns
    -------
    list of str
        Sorted names whose current SHA-256 differs from the recorded one, or
        which the sidecar never recorded at all. Empty when the fit is current.

    Raises
    ------
    FileNotFoundError
        If the sidecar is missing. A fit with no provenance is stale by
        definition -- there is no record to compare against.
    """
    sidecar = fit_provenance_path(artifact_path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no provenance sidecar at {sidecar}: {Path(artifact_path).name} was "
            "written without a record of the tables it was fit on, so it is stale "
            "by definition. Refit it, or pass --allow-stale to read it anyway."
        )
    recorded = json.loads(sidecar.read_text()).get("inputs", {})
    stale = [
        name
        for name, path in inputs.items()
        if recorded.get(name, {}).get("sha256") != sha256_file(_resolve_input(path))
    ]
    return sorted(stale)


def require_fresh_inputs(
    artifact_path: str | Path, inputs: dict[str, Path], *, allow_stale: bool
) -> None:
    """Raise unless the fitted artifact is current for `inputs`.

    The reader-side gate: it is the only thing standing between a regenerated
    table and a summary or figure quietly built from the previous fit.

    Parameters
    ----------
    artifact_path : str or Path
        The fitted artifact about to be read.
    inputs : dict of str to Path
        The name -> path mapping the runner recorded.
    allow_stale : bool
        Skip the check entirely. Reading a fit that no longer matches its
        inputs is a deliberate act, so it takes an explicit flag.

    Raises
    ------
    RuntimeError
        If any input changed since the fit was written.
    FileNotFoundError
        If the sidecar is missing (see ``check_fit_provenance``).
    """
    if allow_stale:
        return
    stale = check_fit_provenance(artifact_path, inputs)
    if stale:
        raise RuntimeError(
            f"{Path(artifact_path).name} is stale: {', '.join(stale)} changed "
            "since it was fit. Refit it, or pass --allow-stale to read it anyway."
        )


def matching_sidecar(
    artifact: Path, fit_contract: Mapping[str, object]
) -> Mapping[str, object] | None:
    """A bundle's provenance sidecar, when it records exactly `fit_contract`.

    The sidecar is the bundle's commit marker: it is written last and removed
    before a refit. Whether the rest of the bundle is complete is each
    runner's own rule.

    Parameters
    ----------
    artifact : Path
        The bundle's primary ``.pkl`` statistics table.
    fit_contract : Mapping of str to object
        Every key must be recorded with the same value in the sidecar.

    Returns
    -------
    Mapping of str to object or None
        The sidecar's record; None when the sidecar is missing, empty, not a
        JSON mapping, or records another value for a contract key.
    """
    sidecar = fit_provenance_path(artifact)
    if not is_nonempty_file(sidecar):
        return None
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(record, Mapping):
        return None
    if not all(record.get(key) == value for key, value in fit_contract.items()):
        return None
    return record


def remove_bundle(artifact: Path, companions: Iterable[Path]) -> None:
    """Remove every component of a previous bundle before refitting.

    Parameters
    ----------
    artifact : Path
        The bundle's primary ``.pkl`` statistics table.
    companions : iterable of Path
        The bundle's other files.

    Returns
    -------
    None
        The sidecar, the table and the companions are unlinked, so an
        interrupted refit can never leave a mixture of old and new components
        that looks complete on the next run.
    """
    fit_provenance_path(artifact).unlink(missing_ok=True)
    artifact.unlink(missing_ok=True)
    for path in companions:
        path.unlink(missing_ok=True)
