"""Strict JSON records, code versions and the decode's stamp, shared by the
state-evidence runners.

It imports nothing that holds JAX to a device. The CPU state-evidence runner
(``scripts/qa/run_state_evidence.py``) uses it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from continuum_swr_replay.analysis.events import swr_inventory
from continuum_swr_replay.decoding_common import (
    require_saved_decode_masks,
    write_atomically,
)


def json_ready(value: Any) -> Any:
    """``value`` with strict JSON types.

    Parameters
    ----------
    value : object
        Nested dicts, lists and tuples of scalars.

    Returns
    -------
    object
        The same structure, with dict keys as str, tuples as lists, numpy
        scalars as Python ones, and every non-finite float as None.
    """
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, content: object) -> None:
    """Write ``content`` as strict JSON through a temporary file beside ``path``.

    Parameters
    ----------
    path : Path
        The file; its folder is created if missing.
    content : object
        JSON-ready (:func:`json_ready`); any other object is written as its
        ``str``.

    Raises
    ------
    ValueError
        ``content`` holds a non-finite float.
    """
    text = json.dumps(content, indent=1, allow_nan=False, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, lambda temporary: temporary.write_text(text))


def code_version(provenance: dict) -> dict:
    """A provenance record's commit and dirty flag.

    Parameters
    ----------
    provenance : dict
        ``continuum_swr_replay.decoding_common.provenance_record``'s.

    Returns
    -------
    dict
        ``commit`` (str) and ``dirty`` (bool).
    """
    return {"commit": provenance["commit"], "dirty": bool(provenance["dirty"])}


def saved_decode_masks(results_path: Path) -> dict:
    """The canonical decode's ``decode_masks``, refusing a decode the runners
    cannot use.

    State evidence and the half decodes score a canonical HPC decode made with
    the configured SWR source's masks (R11 on): its ``results.nc`` carries the
    ``decode_masks`` the decoder stamps (``decoding_common.hpc_decode_masks``),
    naming the configured inventory and artifact runs
    (``decoding_common.require_saved_decode_masks`` checks them). Read from the
    file's attributes alone, before anything is loaded.

    Parameters
    ----------
    results_path : Path
        The decode's ``{epoch_id}_results.nc``.

    Returns
    -------
    dict
        The saved record.

    Raises
    ------
    ValueError
        The decode has no ``decode_masks`` (it was made before R11) or names
        other runs; the message says to rerun once R11's decodes are in the
        decode folder, which no ``--overwrite`` of these runners can replace.
    """
    import xarray as xr

    with xr.open_dataset(results_path, engine="h5netcdf") as results:
        attrs = dict(results.attrs)
    try:
        require_saved_decode_masks(attrs, str(results_path))
    except ValueError as error:
        configured = {
            "swr_inventory_run_id": swr_inventory.SWR_INVENTORY_RUN_ID,
            "artifact_run_id": swr_inventory.ARTIFACT_INTERVALS_RUN_ID,
        }
        stamp = attrs.get("decode_masks")
        if stamp is None:
            found = "has no decode_masks: it was decoded before R11"
        else:
            saved = json.loads(stamp)
            runs = {name: saved.get(name) for name in configured}
            found = f"was decoded with {runs}"
        raise ValueError(
            f"{Path(results_path).name} {found}, not with the configured SWR "
            f"source's masks {configured}. State evidence needs the canonical "
            "decode made with them (R11): rerun once R11's decodes are in the "
            "decode folder; --overwrite does not help."
        ) from error
    record: dict = json.loads(attrs["decode_masks"])
    return record
