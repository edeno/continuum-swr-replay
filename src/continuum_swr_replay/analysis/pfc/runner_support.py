"""Seeding and the convergence gate shared by the PFC runners.

The marked-kernel and marked-model-family runners are two entry points into
the same corpus: they select sessions the same way (``runs.cli.filter_epoch_info``),
seed their random streams the same way (the canonical bundle runner,
``run_pfc_analysis_bandit.py``, seeds its folds and null draws with the same
function), certify a bundle on the same immobile at-well stop cohort,
read a bundle's sidecar against its fit contract and remove a stale bundle the
same way (``runs.fit_provenance.matching_sidecar``, ``remove_bundle``), and
report the same peri-event window. Keeping one copy of those rules here is
what makes two bundles comparable -- a second copy of ``session_seed`` that
drifts by a single character silently re-folds every session.

Runner-specific contracts (the bundle contract version, the estimator name,
the model ladder) stay in each runner: those version independently and are
meant to differ.
"""

from __future__ import annotations

import hashlib

__all__ = [
    "MAX_NONCONVERGED_UNIT_FRACTION",
    "STOP_COHORT",
    "fails_convergence_gate",
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
