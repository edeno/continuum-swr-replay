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

The hashed file is the file that was *read*: ``read_table`` prefers the
``.parquet`` sibling of a ``.pkl`` path, so the paths handed in here are
resolved the same way before hashing.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from continuum_swr_replay.table_io import parquet_path, read_attrs, sha256_file

__all__ = [
    "fit_provenance_path",
    "git_hash",
    "record_inputs",
    "write_fit_provenance",
    "check_fit_provenance",
    "require_fresh_inputs",
]

_REPO_ROOT = Path(__file__).resolve().parents[4]

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


def git_hash() -> str:
    """Return the current commit, suffixed ``-dirty`` if the tree is modified.

    Returns
    -------
    str
        ``git rev-parse HEAD``, plus ``-dirty`` when
        ``git status --porcelain --untracked-files=no`` is non-empty. Untracked
        files do not count as dirty: they cannot change how the code ran.
    """
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if porcelain else head


def _resolve_input(path: str | Path) -> Path:
    """Return the file ``read_table`` would have read for `path`.

    Raises
    ------
    FileNotFoundError
        If neither the parquet nor the pickle form exists.
    """
    parquet = parquet_path(path)
    if parquet.exists():
        return parquet
    pickle = Path(path).with_suffix(".pkl")
    if pickle.exists():
        return pickle
    raise FileNotFoundError(f"no input table at {parquet} or {pickle}")


def _input_record(path: str | Path) -> InputRecord:
    """Hash, mtime, covariate schema version and hash time of one input table."""
    resolved = _resolve_input(path)
    return {
        "sha256": sha256_file(resolved),
        "mtime": resolved.stat().st_mtime,
        "covariate_schema_version": read_attrs(resolved).get(
            "covariate_schema_version"
        ),
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
        Display name -> path of each table the fit was built from. A ``.pkl``
        path is resolved through ``table_io.parquet_path`` so the hashed file
        is the one ``read_table`` actually read.

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
        The command line that produced the artifact, typically ``sys.argv``.

    Returns
    -------
    Path
        The sidecar path written.
    """
    artifact = Path(artifact_path)
    record = {
        "artifact": artifact.name,
        "written_at": datetime.now(UTC).isoformat(),
        "git_hash": git_hash(),
        "estimator": estimator,
        "inputs": {
            name: value if isinstance(value, dict) else _input_record(value)
            for name, value in inputs.items()
        },
        "cli": list(cli),
    }
    out = fit_provenance_path(artifact)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
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
