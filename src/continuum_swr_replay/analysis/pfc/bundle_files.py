"""The file names of a session's PFC modulation bundle.

The PFC runner (``scripts/pfc/run_pfc_analysis_bandit.py``) writes one bundle per
session: ``{epoch_identifier}_modulation{suffix}.pkl`` (the unit statistics),
beside the curves ``{epoch_identifier}_modulation{suffix}.<curve>.npy`` and the
provenance sidecar ``{epoch_identifier}_modulation{suffix}.provenance.json``.
The epoch identifier is ``{animal}{date}_{epoch}_{brain_area}``
(``decoding.identifiers.format_epoch_identifier``); the suffix names a
sensitivity bundle (another stop cohort, another prior) and is empty for the
canonical at-well bundle at the canonical prior.

The runner builds its names here and the readers build or read them back here,
so a reader names what was written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODULATION = "_modulation"
"""What follows the epoch identifier in every file of a bundle."""

MPFC_STEM = f"_mPFC{MODULATION}"
"""What follows the session name in the canonical (at-well, sigma 3) mPFC
bundle's files; other cohorts and priors carry a suffix after it."""


def bundle_path(
    folder: Path, epoch_identifier: str, suffix: str = "", extension: str = ".pkl"
) -> Path:
    """One file of a session's bundle.

    Parameters
    ----------
    folder : Path
        The folder the bundles were written to.
    epoch_identifier : str
        Such as ``"wilbur20210329_02_r1_mPFC"``.
    suffix : str, optional
        The sensitivity suffix, such as ``"_allstops"``; ``""`` (the canonical
        bundle) by default.
    extension : str, optional
        ``".pkl"`` (the statistics, by default), ``".provenance.json"`` or
        ``".<curve>.npy"``.

    Returns
    -------
    Path
        ``{folder}/{epoch_identifier}_modulation{suffix}{extension}``.
    """
    return folder / f"{epoch_identifier}{MODULATION}{suffix}{extension}"


def session_bundle_path(
    folder: Path,
    session: str,
    suffix: str = "",
    extension: str = ".pkl",
    brain_area: str = "mPFC",
) -> Path:
    """:func:`bundle_path` of a session named ``{animal}{date}_{epoch}``.

    The session name is ``decoding.identifiers.session_name``; the bundle's
    identifier adds `brain_area`.
    """
    return bundle_path(folder, f"{session}_{brain_area}", suffix, extension)


def bundle_identifiers(
    folder: Path, suffix: str = "", extension: str = ".pkl"
) -> dict[str, Path]:
    """The bundles of any brain area in `folder`, by epoch identifier.

    Parameters
    ----------
    folder : Path
    suffix : str, optional
        Only the bundles of this sensitivity suffix (``""``: the canonical ones).
    extension : str, optional
        Which file of each bundle to list, ``".pkl"`` by default.

    Returns
    -------
    dict of str to Path
        ``{epoch_identifier: path}``, in the order of the sorted paths.
    """
    tail = f"{MODULATION}{suffix}{extension}"
    return {
        path.name.removesuffix(tail): path for path in sorted(folder.glob(f"*{tail}"))
    }


def bundle_sessions(
    folder: Path, suffix: str = "", extension: str = ".pkl", brain_area: str = "mPFC"
) -> dict[str, Path]:
    """The `brain_area` bundles in `folder`, by session name ``{animal}{date}_{epoch}``.

    Parameters
    ----------
    folder : Path
    suffix : str, optional
        Only the bundles of this sensitivity suffix (``""``: the canonical ones).
    extension : str, optional
        Which file of each bundle to list, ``".pkl"`` by default.
    brain_area : str, optional
        ``"mPFC"`` by default.

    Returns
    -------
    dict of str to Path
        ``{session: path}``, in the order of the sorted paths.
    """
    tail = f"_{brain_area}{MODULATION}{suffix}{extension}"
    return {
        path.name.removesuffix(tail): path for path in sorted(folder.glob(f"*{tail}"))
    }


def read_sidecars(folder: Path) -> list[dict[str, Any]]:
    """Every canonical mPFC bundle's provenance sidecar in `folder`, in session order.

    Parameters
    ----------
    folder : Path
        The folder the bundles were written to.

    Returns
    -------
    list of dict
        Each sidecar's contents, in the order of the sorted sidecar paths.

    Raises
    ------
    ValueError
        `folder` holds no bundle, or a bundle has no sidecar or a sidecar no
        bundle.
    """
    bundles = bundle_sessions(folder)
    sidecars = bundle_sessions(folder, extension=".provenance.json")
    if not bundles or bundles.keys() != sidecars.keys():
        raise ValueError(
            f"bundles and sidecars in {folder} differ: "
            f"{sorted(bundles.keys() ^ sidecars.keys())}"
        )
    return [json.loads(p.read_text()) for p in sidecars.values()]
