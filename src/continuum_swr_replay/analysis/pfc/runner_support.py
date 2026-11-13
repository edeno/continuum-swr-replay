"""Seeding and sidecar helpers shared by the PFC runners.

The marked-kernel and marked-model-family runners are two entry points into
the same corpus: they select sessions the same way (``runs.cli.filter_epoch_info``),
seed their random streams the same way (the canonical bundle runner,
``run_pfc_analysis_bandit.py``, seeds its folds and null draws with the same
function), certify a bundle on the same immobile at-well stop cohort,
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

from continuum_swr_replay.analysis.behavior.fit_provenance import fit_provenance_path

__all__ = [
    "MAX_NONCONVERGED_UNIT_FRACTION",
    "STOP_COHORT",
    "fails_convergence_gate",
    "is_nonempty_file",
    "json_safe",
    "matching_sidecar",
    "remove_bundle",
    "session_seed",
]

# The canonical well-anchored stop cohort every PFC marked analysis is fit on.
STOP_COHORT = "at_well"
# A session whose full-data fit fails to converge for more than this fraction
# of units is not certified: its held-out deltas and curves would be a mixture
# of converged and unconverged units with no way to tell them apart.
MAX_NONCONVERGED_UNIT_FRACTION = 0.1


def fails_convergence_gate(n_not_converged: int, n_units: int) -> bool:
    """Whether a fit has too many units whose full-data fit did not converge.

    Parameters
    ----------
    n_not_converged : int
        Units whose full-data fit did not converge.
    n_units : int
        Units fit.

    Returns
    -------
    bool
        True when more than :data:`MAX_NONCONVERGED_UNIT_FRACTION` of the units
        did not converge; the fit is then not certified.
    """
    return n_not_converged > MAX_NONCONVERGED_UNIT_FRACTION * n_units


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
