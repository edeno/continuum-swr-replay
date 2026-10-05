"""Strict JSON records and code versions, shared by the state-evidence runners.

It imports nothing that holds JAX to a device, so the GPU half-decode runner
(``scripts/qa/run_half_tetrode_decodes.py``) and the CPU state-evidence runner
(``scripts/qa/run_state_evidence.py``) both use it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from continuum_swr_replay.decoding_common import write_atomically


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
