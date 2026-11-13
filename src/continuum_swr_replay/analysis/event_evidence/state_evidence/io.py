"""Code versions and the decode's stamp, shared by the state-evidence runners.

Their strict JSON records go through ``runs.io.json_ready`` and
``runs.io.write_json``.

It imports nothing that holds JAX to a device. The CPU state-evidence runner
(``scripts/detection/run_state_evidence.py``) uses it.

The module is named ``io`` like the standard-library module on purpose: the
interpreter loads ``io`` at startup, so a tool run from inside this directory
still gets the standard one, and ``<package>.io`` is a common name for a
package's file handling (``tests/repository/test_package_dir_shadows_no_stdlib.py``
exempts the modules the interpreter preloads).
"""

from __future__ import annotations

import json
from pathlib import Path

from continuum_swr_replay.analysis.detection.decode_masks import (
    require_saved_decode_masks,
)
from continuum_swr_replay.analysis.events import swr_source
from continuum_swr_replay.decoding.results import open_results


def code_version(provenance: dict) -> dict:
    """A provenance record's commit and dirty flag.

    Parameters
    ----------
    provenance : dict
        ``continuum_swr_replay.runs.provenance.provenance_record``'s.

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
    ``decode_masks`` the decoder stamps (``decode_masks.hpc_decode_masks``),
    naming the configured inventory and artifact runs
    (``decode_masks.require_saved_decode_masks`` checks them). Read from the
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
    with open_results(results_path) as results:
        attrs = dict(results.attrs)
    try:
        require_saved_decode_masks(attrs, str(results_path))
    except ValueError as error:
        configured = {
            "swr_inventory_run_id": swr_source.SWR_INVENTORY_RUN_ID,
            "artifact_run_id": swr_source.ARTIFACT_INTERVALS_RUN_ID,
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
