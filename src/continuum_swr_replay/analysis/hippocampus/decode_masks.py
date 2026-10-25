"""The HPC decode's masks, from the paper's SWR source.

Both ``run_decoding_bandit.py`` (NonLocal 4-state detector) and
``run_decoding_contfrag.py`` (ContFrag 2-state classifier) use them:
``is_missing`` (the artifact run's mask) and ``is_training`` (that mask and
every event of the Kay inventory left out), both from the paper's SWR source
(``analysis/events/swr_source.py``). An HPC decode refuses to start unless
the source names both runs (``require_hpc_decode_sources``). The decode's
configuration, file names and results reader are in
``continuum_swr_replay.decoding``; the run infrastructure the decoders share
(atomic writes, provenance, the command line and the batch loop) is in
``continuum_swr_replay.runs``.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.events import swr_source
from continuum_swr_replay.analysis.events.artifact_masks import PADDING_S
from continuum_swr_replay.decoding.identifiers import TrainingType
from continuum_swr_replay.intervals import interval_mask


def require_hpc_decode_sources() -> None:
    """Refuse an HPC decode unless the SWR source names both of its runs, and
    its inventory was masked with its artifact run.

    The decode's ``is_missing``, encoding exclusion, MUA detection and
    ``SWR_event`` read ``swr_source.SWR_INVENTORY_RUN_ID`` (the Kay inventory)
    and ``swr_source.ARTIFACT_INTERVALS_RUN_ID`` (the artifact run). Either
    left unset would decode without the masks, so it is refused (fail closed).
    The inventory's manifest names the artifact run the SWR runner masked its
    events with (``settings.artifacts.run_id``; absent from a run made before
    it masked artifacts, such as R3's). An inventory masked with no artifact
    run or another one is refused too: its events, which encoding leaves out,
    would not be the masked inventory's.

    Raises
    ------
    ValueError
        Naming each run id that is unset, or the inventory, the artifact run it
        was masked with and the configured one when those two differ; or when
        the inventory has no manifest, or another run's
        (``swr_source.inventory_manifest``).
    """
    unset = [
        f"swr_source.{name}"
        for name in ("SWR_INVENTORY_RUN_ID", "ARTIFACT_INTERVALS_RUN_ID")
        if getattr(swr_source, name) is None
    ]
    if unset:
        raise ValueError(
            f"An HPC decode needs {' and '.join(unset)} set, so that its artifact "
            "mask, encoding exclusion and SWR labels come from named runs; "
            "refusing to decode without them."
        )
    manifest = swr_source.inventory_manifest()
    masked_with = manifest.get("settings", {}).get("artifacts", {}).get("run_id")
    if masked_with != swr_source.ARTIFACT_INTERVALS_RUN_ID:
        recorded = (
            "no artifact run"
            if masked_with is None
            else f"artifact run {masked_with!r}"
        )
        raise ValueError(
            f"The SWR inventory {swr_source.SWR_INVENTORY_RUN_ID!r} was masked with "
            f"{recorded}, but the configured artifact run is "
            f"{swr_source.ARTIFACT_INTERVALS_RUN_ID!r}: rerun the SWR runner with the "
            "configured artifact run and set swr_source.SWR_INVENTORY_RUN_ID to it; "
            "refusing to decode."
        )


def require_saved_decode_masks(attrs: Mapping[str, Any], what: str) -> None:
    """Refuse to resume an HPC decode whose saved results were made under other
    masks.

    A resume reuses ``results.nc`` and rewrites ``data_over_time`` with the
    configured source's SWR labels, so the results must have been decoded with
    the configured Kay inventory and artifact run: their ``decode_masks``
    record (``hpc_decode_masks``, stamped by a fresh save) must name both.

    Parameters
    ----------
    attrs : Mapping
        The saved results' ``attrs``.
    what : str
        What they are, for the message (the file, say).

    Raises
    ------
    ValueError
        If the record is missing (a decode made before the masks) or names
        another inventory or artifact run.
    """
    stamp = attrs.get("decode_masks")
    if stamp is None:
        raise ValueError(
            f"{what} has no decode_masks: it was decoded without the artifact "
            "masks; refusing to resume. Decode it afresh (--overwrite)."
        )
    record = json.loads(stamp)
    configured = {
        "swr_inventory_run_id": swr_source.SWR_INVENTORY_RUN_ID,
        "artifact_run_id": swr_source.ARTIFACT_INTERVALS_RUN_ID,
    }
    saved = {name: record.get(name) for name in configured}
    if saved != configured:
        raise ValueError(
            f"{what} was decoded with {saved}, not the configured {configured}; "
            "refusing to resume. Decode it afresh (--overwrite)."
        )


def hpc_artifact_intervals(nwb_file_name: str, epoch_name: str) -> pd.DataFrame:
    """The epoch's artifact mask, from the configured run.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        Such as ``"chimi20200225_.nwb"`` and ``"02_r1"``.

    Returns
    -------
    pd.DataFrame
        ``start_time`` and ``end_time`` in seconds, closed, disjoint and in
        time order: the run's cores padded by ``PADDING_S``, the one mask
        every stage uses (``swr_source.artifact_intervals``).

    Raises
    ------
    ValueError
        If a run id is unset (``require_hpc_decode_sources``).
    SwrSourceError
        If the artifact run cannot serve the epoch.
    """
    require_hpc_decode_sources()
    return swr_source.artifact_intervals(
        swr_source.swr_epoch_id(nwb_file_name, epoch_name)
    )


def hpc_ripple_power_inputs(nwb_file_name: str, epoch_name: str) -> dict[str, Any]:
    """The SWR source's ripple-power inputs for ``load_data``, from the
    configured runs (``swr_source.ripple_power_inputs``).

    With them the decode's ``SWR_zscore`` is the Kay detector's consensus over
    the tetrodes rule R-T kept for the configured inventory, z-scored with the
    artifact mask left out and NaN inside it.

    Raises
    ------
    ValueError
        If the source's runs are not set or do not match
        (``require_hpc_decode_sources``).
    SwrSourceError
        If the inventory or the artifact run cannot serve the epoch.
    """
    require_hpc_decode_sources()
    return swr_source.ripple_power_inputs(
        swr_source.swr_epoch_id(nwb_file_name, epoch_name)
    )


@dataclass(frozen=True)
class HpcDecodeMasks:
    """An HPC decode's masks on its time grid (the position grid)."""

    is_missing: NDArray[np.bool_]
    """Shape (n_time,): bins in an artifact interval, whose
    log-likelihood the decoder sets to 0."""
    is_training: NDArray[np.bool_]
    """Shape (n_time,): bins the encoding model is fitted on."""
    record: dict[str, Any]
    """What they were built from and what they removed, in seconds (JSON)."""


def hpc_decode_masks(
    nwb_file_name: str,
    epoch_name: str,
    position_info: pd.DataFrame,
    artifact_mask: pd.DataFrame,
    training_type: str | TrainingType,
) -> HpcDecodeMasks:
    """Build an HPC decode's ``is_missing`` and ``is_training``.

    - ``is_missing``: the bins (closed) inside an artifact interval.
    - ``is_training``: for ``no_ripple``, the bins outside every event of the
      configured Kay inventory and outside ``is_missing``; for ``all``, the
      bins outside ``is_missing``. Every inventory event counts, at any speed,
      as the stored-ripple rule it replaces excluded every stored ripple: the
      inventory's events passed the detectors' own endpoint speed rule (speed
      at the first and last sample at most 4 cm/s), as the stored ripples did,
      and the SWR runner has already dropped those overlapping the artifact
      mask.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
    position_info : pd.DataFrame
        The decode's position, indexed by time in seconds: the decode grid.
    artifact_mask : pd.DataFrame
        ``hpc_artifact_intervals``' table for the epoch.
    training_type : str or TrainingType
        ``"no_ripple"`` or ``"all"``.

    Returns
    -------
    HpcDecodeMasks
        The masks, and a record of the source's name, both run ids, the
        bins and seconds set missing, the inventory's event count, and the
        bins and seconds removed from encoding (missing ones included).

    Raises
    ------
    ValueError
        If the source's runs are not set or do not match
        (``require_hpc_decode_sources``), or `training_type` is not one of the two.
    SwrSourceError
        If the inventory cannot serve the epoch.
    """
    require_hpc_decode_sources()
    training_type = TrainingType(training_type)
    epoch_id = swr_source.swr_epoch_id(nwb_file_name, epoch_name)
    time = position_info.index.to_numpy(dtype=np.float64)
    is_missing = interval_mask(time, artifact_mask)
    if training_type == TrainingType.NO_RIPPLE:
        events = swr_source.read_swr_inventory(epoch_id)
        is_swr = interval_mask(time, events)
        n_events: int | None = len(events)
    else:
        is_swr = np.zeros(time.shape, dtype=bool)
        n_events = None
    is_training = ~is_swr & ~is_missing
    bin_s = float(np.median(np.diff(time)))
    n_missing = int(np.count_nonzero(is_missing))
    n_removed = int(np.count_nonzero(~is_training))
    record = {
        "swr_source": swr_source.swr_source_name(),
        "swr_inventory_run_id": swr_source.SWR_INVENTORY_RUN_ID,
        "artifact_run_id": swr_source.ARTIFACT_INTERVALS_RUN_ID,
        "padding_s": dict(PADDING_S),
        "training_type": training_type.value,
        "n_bins": int(time.size),
        "bin_s": bin_s,
        "n_missing_bins": n_missing,
        "missing_s": n_missing * bin_s,
        "n_swr_events": n_events,
        "n_encoding_removed_bins": n_removed,
        "encoding_removed_s": n_removed * bin_s,
        "encoding_removed_swr_s": int(np.count_nonzero(is_swr)) * bin_s,
    }
    return HpcDecodeMasks(is_missing=is_missing, is_training=is_training, record=record)
