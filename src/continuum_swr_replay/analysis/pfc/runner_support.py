"""Session-selection, seeding, and sidecar helpers shared by the PFC runners.

The marked-kernel and marked-model-family runners are two entry points into
the same corpus: they select sessions the same way, seed their random streams
the same way, certify a bundle on the same immobile at-well stop cohort,
read a bundle's sidecar against its fit contract and remove a stale bundle the
same way, and report the same peri-event window. Keeping one copy of those rules here is
what makes two bundles comparable -- a second copy of ``session_seed`` that
drifts by a single character silently re-folds every session.

Runner-specific contracts (the bundle contract version, the estimator name,
the model ladder) stay in each runner: those version independently and are
meant to differ.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.fit_provenance import fit_provenance_path

__all__ = [
    "MAX_NONCONVERGED_UNIT_FRACTION",
    "REPORT_HALF_WINDOW_S",
    "STOP_COHORT",
    "filter_epoch_info",
    "is_nonempty_file",
    "json_safe",
    "matching_sidecar",
    "read_session_list",
    "remove_bundle",
    "session_seed",
]

# The canonical well-anchored stop cohort every PFC marked analysis is fit on.
STOP_COHORT = "at_well"
# Curves are fit over a wider lag window than they are reported on, so the
# spline kernel returns to baseline inside the fit window rather than at the
# edge of the reported one.
REPORT_HALF_WINDOW_S = 0.2
# A session whose full-data fit fails to converge for more than this fraction
# of units is not certified: its held-out deltas and curves would be a mixture
# of converged and unconverged units with no way to tell them apart.
MAX_NONCONVERGED_UNIT_FRACTION = 0.1


def session_seed(stream: str, nwb_file_name: str, epoch_name: str) -> int:
    """Return a deterministic seed for one session and random stream.

    Parameters
    ----------
    stream : str
        Name of the random stream (e.g. ``"marked_folds"``), so two streams of
        the same session never share a seed.
    nwb_file_name : str
        Session NWB file name.
    epoch_name : str
        Session epoch (interval list) name.

    Returns
    -------
    int
        The first 64 bits of the SHA-256 digest of
        ``"<stream>|<nwb_file_name>|<epoch_name>"``, little-endian.
    """
    key = f"{stream}|{nwb_file_name}|{epoch_name}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


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


def json_safe(value: object) -> object:
    """Return `value` with every non-finite float replaced by None.

    ``json.dumps`` writes ``NaN``/``Infinity`` tokens by default, which are not
    JSON and which a strict reader refuses. A provenance sidecar is read back
    by the runner's own skip check, so a diagnostics NaN must not be able to
    make the sidecar unparseable.

    Parameters
    ----------
    value : object
        Any nested combination of mappings, sequences, and scalars destined for
        a provenance sidecar.

    Returns
    -------
    object
        The same structure with mappings as ``dict``, tuples and lists as
        ``list``, non-finite floats as None, and every other scalar unchanged.
    """
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def read_session_list(path: Path) -> list[tuple[str, str]]:
    """Read unique ``nwb_file_name epoch_name`` pairs in file order.

    Parameters
    ----------
    path : Path
        Text file with one whitespace- or comma-separated session per line;
        ``#`` starts a comment.

    Returns
    -------
    list of tuple of str
        ``(nwb_file_name, epoch_name)`` pairs, de-duplicated, in file order.

    Raises
    ------
    ValueError
        If a line does not hold exactly two fields, or the file names no
        sessions.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        if len(fields) != 2:
            raise ValueError(
                f"{path}:{number}: expected '<nwb_file_name> <epoch_name>', "
                f"got {raw.strip()!r}"
            )
        key = (fields[0], fields[1])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    if not pairs:
        raise ValueError(f"{path} names no sessions")
    return pairs


def filter_epoch_info(
    epoch_info: pd.DataFrame,
    *,
    sessions: Path | None = None,
    animal: str | None = None,
    date: int | None = None,
    epoch: str | None = None,
) -> pd.DataFrame:
    """Apply the runners' session filters without changing the source order.

    The filters are taken as explicit fields rather than as a parsed
    ``Namespace`` so that two runners with different remaining CLI surfaces
    still select sessions identically.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        Session table indexed by ``(nwb_file_name, interval_list_name)``.
    sessions : Path, optional
        ``--sessions`` file listing the sessions to keep, in the order they
        should run.
    animal : str, optional
        Keep only this animal.
    date : int, optional
        Keep only this session date.
    epoch : str, optional
        Keep only this interval list name.

    Returns
    -------
    pd.DataFrame
        The selected rows, in `sessions` order when that file is given and in
        source order otherwise.

    Raises
    ------
    ValueError
        If `sessions` names a session that is not available.
    """
    keep = np.ones(len(epoch_info), dtype=bool)
    keys = list(
        zip(
            epoch_info.index.get_level_values("nwb_file_name"),
            epoch_info.index.get_level_values("interval_list_name"),
            strict=True,
        )
    )
    requested_order = read_session_list(sessions) if sessions is not None else None
    if requested_order is not None:
        available = set(keys)
        missing = [key for key in requested_order if key not in available]
        if missing:
            raise ValueError(f"{sessions} names unavailable session(s): {missing[:10]}")
        requested = set(requested_order)
        keep &= np.array([key in requested for key in keys], dtype=bool)
    if animal is not None:
        keep &= epoch_info.animal.to_numpy() == animal
    if date is not None:
        keep &= epoch_info.date.to_numpy() == date
    if epoch is not None:
        keep &= (
            epoch_info.index.get_level_values("interval_list_name").to_numpy() == epoch
        )
    filtered = epoch_info.loc[keep]
    if requested_order is not None:
        order = {key: index for index, key in enumerate(requested_order)}
        filtered_keys = list(
            zip(
                filtered.index.get_level_values("nwb_file_name"),
                filtered.index.get_level_values("interval_list_name"),
                strict=True,
            )
        )
        filtered = filtered.iloc[np.argsort([order[key] for key in filtered_keys])]
    return filtered
