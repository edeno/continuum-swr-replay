"""The one naming rule for a decode's files.

A decode writes four files into its folder, each named
``{epoch_identifier}_*``: ``data_over_time`` (named for the non-local threshold
in percent, or for none in a ContFrag decode), ``results.nc``, ``detector.pkl``
and ``environment.pkl``. The decode runners write these names
(``scripts/decoding/_decode_runner.py``) and every reader builds them through
:func:`decode_paths`, so a reader names what was written. The identifier itself,
with its run suffixes (ripple detector, non-stationary, training type), is built
and read back by ``decoding.identifiers``, for the runners and ``load_results``
alike.

A leaf module: it imports only the standard library and ``parameters``, so an
offline table builder or figure that reads a decode folder does not pull in the
decode runners' infrastructure (``runs.batch``'s dask, its distributed
scheduler, CUDA).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD


def non_local_threshold_percent(non_local_threshold: float) -> int:
    """The percent a decode's ``data_over_time`` file is named for.

    ``int(non_local_threshold * 100)``, truncated as the decoders have always
    written it, so a threshold of 0.29 names ``_28_`` (0.29 * 100 is
    28.999999999999996).
    """
    return int(non_local_threshold * 100)


DEFAULT_NON_LOCAL_THRESHOLD_PERCENT = non_local_threshold_percent(
    DEFAULT_NON_LOCAL_THRESHOLD
)
"""The percent the canonical decode's ``data_over_time`` file is named for (80)."""


def data_over_time_suffix(
    threshold_percent: int | None = DEFAULT_NON_LOCAL_THRESHOLD_PERCENT,
) -> str:
    """What follows the epoch identifier in a decode's ``data_over_time`` name.

    ``_{threshold_percent}_data_over_time.pkl``, or ``_data_over_time.pkl``
    when `threshold_percent` is None (the ContFrag decode, whose identifier ends
    in ``_contfrag``, names no threshold).
    """
    if threshold_percent is None:
        return "_data_over_time.pkl"
    return f"_{threshold_percent}_data_over_time.pkl"


class DecodePaths(NamedTuple):
    """The four files one decode writes, all named ``{epoch_identifier}_*``."""

    data_over_time: Path
    results: Path
    detector: Path
    environment: Path


def decode_paths(
    epoch_identifier: str,
    root: Path,
    *,
    threshold_percent: int | None = DEFAULT_NON_LOCAL_THRESHOLD_PERCENT,
) -> DecodePaths:
    """The decode files of `epoch_identifier` in `root`: the one naming rule.

    The decode runners write these names (``save_results_and_model``) and
    every reader builds them here, so a reader names what was written.

    Parameters
    ----------
    epoch_identifier : str
        The decode's full identifier, suffixes included, such as
        ``"chimi20200212_02_r1_HPC"`` or ``"chimi20200212_02_r1_HPC_contfrag"``.
    root : Path
        The decode folder, such as ``PROCESSED_DATA_DIR``.
    threshold_percent : int or None, optional
        The non-local threshold in percent that names ``data_over_time``
        (:func:`non_local_threshold_percent`; 80 by default), or None for a
        decode that names none (:func:`data_over_time_suffix`).

    Returns
    -------
    DecodePaths
        ``data_over_time``, ``results`` (``.nc``), ``detector`` and
        ``environment`` (``.pkl``). Whether they exist is not checked.
    """
    return DecodePaths(
        data_over_time=root
        / f"{epoch_identifier}{data_over_time_suffix(threshold_percent)}",
        results=root / f"{epoch_identifier}_results.nc",
        detector=root / f"{epoch_identifier}_detector.pkl",
        environment=root / f"{epoch_identifier}_environment.pkl",
    )


def results_exist(
    save_folder: Path,
    epoch_identifier: str,
    overwrite: bool,
) -> bool:
    """Check if results, detector, and environment files exist."""
    if overwrite:
        return False

    paths = decode_paths(epoch_identifier, save_folder)
    required_files = [paths.results, paths.detector, paths.environment]
    return all(f.exists() for f in required_files)
