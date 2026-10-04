#!/usr/bin/env python
"""Audit that the refreshed pipeline's artifacts share their inputs.

Stage 8 of the full pipeline refresh
(``docs/design/2026-10-03-full-pipeline-refresh-design.md``): every canonical
artifact must rest on the same decodes, made by one commit in one
environment. ``audit`` checks that, reading only, and writes what it found to
one JSON report. ``record-tables`` writes the tables manifest that report
checks (Stage 5), and nothing else.

``audit`` runs these checks, each reported as ``{"ok", "failures",
"details"}`` under its name:

- ``one_commit``: every NonLocal and ContFrag decode names one commit, a clean
  tree and one environment (``uv.lock``'s sha256, the ``ripple_detection``
  commit, the ``non_local_detector`` and CUDA plugin versions). Each decode's
  record is read from its ``results.nc`` (``attrs["provenance"]``), or else
  from its log's ``PROVENANCE`` line, unless the log is marked ``RESUMED``.
  A decode listed under ``--exceptions`` with a reason is set aside. This
  commit is the decode commit the later checks compare against.
- ``decoder_settings_nonlocal`` and ``decoder_settings_contfrag``: every
  epoch's fitted detector carries the same discrete-transition settings, and
  the expected ones: by default those ``expected_decoder_settings()`` builds
  from the code, or else those an ``--expected-settings`` file names. Until
  the pin (decision R10) the decoders left these settings to the library's
  defaults, and a library upgrade changed them unnoticed between April and
  the refresh (32 % more non-local events).
- ``grids_match``: per epoch, NonLocal and ContFrag share time grid, position
  and speed, and the grid passes ``validate_time_grid``.
- ``minimum_sample_counts``: ``minimum_sample_count(time, 0.015)``, the
  non-local events' minimum length in samples, is the same for every epoch
  (decision R9).
- ``widloski_hashes``: the decode files the Widloski run read are the
  canonical ones.
- ``swr_spans``: the SWR runner's position span is each decode's.
- ``spike_loader_unchanged``: ``data_loaders/spikes.py`` is the same at every
  Widloski and SWR run commit as at the decode commit.
- ``tables_recorded``: the promoted tables are the ones their manifests
  record, built at a commit that descends from the decode commit, from its
  decodes.
- ``pfc_sidecars``: every mPFC bundle names the canonical event table.
- ``fit_provenance``: ``check_fit_provenance.py`` passes on both table roots,
  and on the fits at the ``replay_content`` root (Stage 5B and Stage 6).
- ``epoch_sets``: an epoch missing from any artifact (either decode, either
  run, any table) has a reason recorded for that artifact. A table names its
  epochs by its session columns or, keyed by ``stop_id`` alone as the qvals
  table is, through its folder's stop table.
- ``encoding_marks``, with ``--check-marks`` only, since it reads the
  database: the marks reloaded through the project's loader hold every
  encoding mark of both decodes.

``audit.json`` records under ``expected_settings`` the decoder settings the
detectors were held to, and their source: ``expected_decoder_settings()`` or
the ``--expected-settings`` file. It also lists under ``not_checked`` what the
audit did not check: the encoding marks without ``--check-marks``, a decoder
an ``--expected-settings`` file names no settings for, a table whose epochs
cannot be told, and a family of fits at the ``replay_content`` root with no
fit at all. A pass covers the checks, and nothing listed there. ``audit``
exits 1 if any check fails, and 0 otherwise.

Usage:
    uv run python scripts/qa/audit_refresh_consistency.py record-tables \\
        Processed-Data/replay_content/_staging_redecode_tables_20261005 \\
        --commit $(git rev-parse HEAD) \\
        --inputs '{"decode_commit": "<C0>", "behavior_cache": "<cache dir>"}'
    uv run python scripts/qa/audit_refresh_consistency.py audit \\
        --output Processed-Data/_refresh_20261008/audit.json \\
        --widloski-manifest <memoryless run folder>/manifest.json \\
        --swr-manifest <swr run folder>/manifest.json --check-marks
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5netcdf
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from non_local_detector import (
    ContFragClusterlessClassifier,
    NonLocalClusterlessDetector,
)
from numpy.typing import NDArray
from ripple_detection import minimum_sample_count

from continuum_swr_replay.analysis.constants import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    check_marks_match_encoding,
    load_clusterless_encoding_model,
    validate_time_grid,
)
from continuum_swr_replay.data_loaders.spikes import filter_spike_times, get_hpc_marks
from continuum_swr_replay.decoding_common import (
    NonLocalDecodingConfig,
    write_atomically,
)
from continuum_swr_replay.decoding_transitions import (
    contfrag_transition_settings,
    nonlocal_transition_settings,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import resolved_table_path, sha256_file

REPO = Path(__file__).resolve().parents[2]
"""The checkout this script belongs to, whose ``check_fit_provenance.py`` runs."""

PRODUCED_BY = "scripts/qa/audit_refresh_consistency.py"
CHECK_FIT_PROVENANCE = Path("scripts") / "qa" / "check_fit_provenance.py"
SPIKE_LOADER = "src/continuum_swr_replay/data_loaders/spikes.py"
"""The loader the Widloski runner reloads marks through, as ``load_data`` does."""

DECODERS = ("nonlocal", "contfrag")
DECODE_FILES = ("results", "detector", "environment", "data_over_time")
NON_LOCAL_THRESHOLD_PERCENT = round(100 * DEFAULT_NON_LOCAL_THRESHOLD)
"""The NonLocal ``data_over_time`` file is named for the decode's threshold."""

CANONICAL_EPOCH_ID = re.compile(r"[A-Za-z0-9]+_\d+_r\d+_HPC")
"""``{animal}{date}_{epoch}_HPC``: a canonical HPC decode, with no variant suffix."""

PROVENANCE_FIELDS = (
    "commit",
    "dirty",
    "uv_lock_sha256",
    "ripple_detection_commit",
    "non_local_detector_version",
    "cuda_plugin_version",
)
"""``decoding_common.provenance_record``'s fields."""

SINGLE_VALUED_FIELDS = tuple(field for field in PROVENANCE_FIELDS if field != "dirty")
PROVENANCE_MARKER = "PROVENANCE "
RESUMED_MARKER = "RESUMED from existing results in"

NON_LOCAL_MINIMUM_DURATION_S = 0.015
"""The minimum non-local event length ``get_confident_times`` segments with."""

DISCRETE_SETTINGS = (
    "discrete_transition_type",
    "discrete_transition_stickiness",
    "discrete_transition_concentration",
    "discrete_transition_prior_weight",
    "discrete_transition_regularization",
    "frozen_discrete_transition_rows",
    "local_position_std",
)
"""The detector settings ``check_decoder_settings`` compares.

``discrete_transition_type`` holds the initial transition matrix (its
``values``, or ``diagonal_values`` for a diagonal type). ``local_position_std``
is the NonLocal Local state's width; a ContFrag classifier stores None."""

TABLES_MANIFEST = "tables_manifest.json"
TABLE_SUFFIXES = (".parquet", ".json")
QVALS_TABLE_FILES = (
    "models_per_bin_rate/qvals_table.parquet",
    "models_per_bin_rate/qvals_standardization_reference.json",
)
"""The tables ``build_qvals_table.py`` writes below a table folder."""

FIT_SIDECAR_SUFFIX = ".provenance.json"
PFC_SIDECAR_PATTERN = "*_mPFC_modulation*.provenance.json"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _result(failures: list[str], details: Any) -> dict:
    return {"ok": not failures, "failures": failures, "details": details}


def _names(names: Iterable[object], limit: int = 8) -> str:
    """Sorted names, comma-separated, the ones past `limit` counted."""
    ordered = sorted(str(name) for name in names)
    shown = ", ".join(ordered[:limit])
    return (
        shown if len(ordered) <= limit else f"{shown} and {len(ordered) - limit} more"
    )


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a read-only git command in `repo`; the caller reads the exit code."""
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def _same_commit(repo: Path, first: str, second: str) -> bool:
    """Whether two names, full or abbreviated, are one commit of `repo`."""
    if first == second:
        return True
    resolved = [
        _git(repo, "rev-parse", "--verify", "--quiet", f"{name}^{{commit}}")
        for name in (first, second)
    ]
    return all(r.returncode == 0 for r in resolved) and (
        resolved[0].stdout.strip() == resolved[1].stdout.strip()
    )


def is_canonical_epoch_id(epoch_id: str) -> bool:
    """Whether `epoch_id` names a canonical HPC decode (``chimi20200225_02_r1_HPC``)."""
    return CANONICAL_EPOCH_ID.fullmatch(epoch_id) is not None


def decode_id(epoch_id: str, decoder: str) -> str:
    """The decode's own identifier, which names its files and its log."""
    return epoch_id if decoder == "nonlocal" else f"{epoch_id}_contfrag"


def decode_files(epoch_id: str, root: Path, decoder: str) -> dict[str, Path]:
    """The four files a decode of `epoch_id` saves under `root`, by kind."""
    identifier = decode_id(epoch_id, decoder)
    data_over_time = (
        f"{identifier}_{NON_LOCAL_THRESHOLD_PERCENT}_data_over_time.pkl"
        if decoder == "nonlocal"
        else f"{identifier}_data_over_time.pkl"
    )
    return {
        "results": root / f"{identifier}_results.nc",
        "detector": root / f"{identifier}_detector.pkl",
        "environment": root / f"{identifier}_environment.pkl",
        "data_over_time": root / data_over_time,
    }


def find_decodes(root: Path, decoder: str) -> tuple[set[str], dict[str, list[str]]]:
    """Canonical epochs whose `decoder` decode has all four files under `root`.

    Returns
    -------
    complete : set of str
        Epoch identifiers with every file.
    incomplete : dict of str to list of str
        Epoch identifiers with some files, each with the kinds it lacks.
    """
    found: dict[str, set[str]] = {}
    for kind, template in decode_files("", root, decoder).items():
        suffix = template.name
        for path in root.glob(f"*{suffix}"):
            epoch_id = path.name[: -len(suffix)]
            if is_canonical_epoch_id(epoch_id):
                found.setdefault(epoch_id, set()).add(kind)
    complete = {epoch_id for epoch_id, kinds in found.items() if len(kinds) == 4}
    incomplete = {
        epoch_id: sorted(set(DECODE_FILES) - kinds)
        for epoch_id, kinds in sorted(found.items())
        if len(kinds) < 4
    }
    return complete, incomplete


def _time_indexed(data_over_time: pd.DataFrame) -> pd.DataFrame:
    """A ``data_over_time`` frame indexed by time, as the Widloski runner reads it."""
    if "time" in data_over_time.columns:
        return data_over_time.set_index("time")
    return data_over_time


# --------------------------------------------------------------------------
# 1. Commit and environment
# --------------------------------------------------------------------------


def _results_provenance(path: Path) -> dict | None:
    """The record a fresh save stamped into ``results.nc``, or None."""
    if not path.exists():
        return None
    with h5netcdf.File(path, "r") as results:
        text = results.attrs.get("provenance")
    if text is None:
        return None
    record: dict = json.loads(text.decode() if isinstance(text, bytes) else text)
    return record


def _log_provenance(path: Path) -> tuple[dict | None, bool]:
    """The log's first ``PROVENANCE`` record, or None, and whether it was a resume."""
    if not path.exists():
        return None, False
    text = path.read_text(encoding="utf-8", errors="replace")
    resumed = RESUMED_MARKER in text
    start = text.find(PROVENANCE_MARKER)
    if start < 0:
        return None, resumed
    line = text[start + len(PROVENANCE_MARKER) :].split("\n", 1)[0]
    record: dict = json.loads(line)
    return record, resumed


def decode_provenance(log_dir: Path, root: Path, epoch_ids: list[str]) -> pd.DataFrame:
    """Each decode's provenance record, NonLocal and ContFrag, for `epoch_ids`.

    The record comes from ``results.nc``'s ``attrs["provenance"]``, which the
    fresh save wrote and a resume leaves alone. When ``results.nc`` exists
    without it, the record comes from the JSON after the first
    ``PROVENANCE `` in ``logs/{decode_id}.log``, unless the log says
    ``RESUMED from existing results in``: a resume rewrites the log, so its
    record describes the resume and not the decode, and the decode then has
    no record (``source`` None). A decode with no ``results.nc`` was never
    saved (``saved`` False): it has no record, whatever its log says.

    Parameters
    ----------
    log_dir : Path
        The decoders' ``logs/`` folder.
    root : Path
        Where the decodes are saved.
    epoch_ids : list of str
        Canonical epoch identifiers, such as ``chimi20200225_02_r1_HPC``.

    Returns
    -------
    pd.DataFrame
        One row per epoch and decoder: ``epoch_id``, ``decoder``
        (``"nonlocal"`` or ``"contfrag"``), ``decode_id`` (the name of its
        files and log), ``saved`` (whether ``results.nc`` exists), ``source``
        (``"results.nc"``, ``"log"`` or None), ``resumed``, the provenance
        fields, and ``log_commit`` and
        ``log_dirty``: what the log's own record says, which differs from the
        decode's when a later run (such as a resume, which rewrites
        ``data_over_time``) rewrote the log.
    """
    rows = []
    for epoch_id in epoch_ids:
        for decoder in DECODERS:
            identifier = decode_id(epoch_id, decoder)
            results = root / f"{identifier}_results.nc"
            saved = results.exists()
            from_results = _results_provenance(results)
            from_log, resumed = _log_provenance(log_dir / f"{identifier}.log")
            if from_results is not None:
                record, source = from_results, "results.nc"
            elif saved and from_log is not None and not resumed:
                record, source = from_log, "log"
            else:
                record, source = {}, None
            rows.append(
                {
                    "epoch_id": epoch_id,
                    "decoder": decoder,
                    "decode_id": identifier,
                    "saved": saved,
                    "source": source,
                    "resumed": resumed,
                    **{field: record.get(field) for field in PROVENANCE_FIELDS},
                    "log_commit": None if from_log is None else from_log.get("commit"),
                    "log_dirty": None if from_log is None else from_log.get("dirty"),
                }
            )
    columns = [
        "epoch_id",
        "decoder",
        "decode_id",
        "saved",
        "source",
        "resumed",
        *PROVENANCE_FIELDS,
        "log_commit",
        "log_dirty",
    ]
    return pd.DataFrame(rows, columns=columns).astype(object)


def check_one_commit(
    provenance: pd.DataFrame,
    allowed: Mapping[str, str],
    *,
    explained: Mapping[str, str] | None = None,
) -> dict:
    """One commit, a clean tree and one environment across the decodes.

    Passes when every saved decode not in `allowed` has a provenance record;
    every field of those records has exactly one value across them (one
    ``commit``, ``uv_lock_sha256``, ``ripple_detection_commit``,
    ``non_local_detector_version`` and ``cuda_plugin_version``, none
    missing); every ``dirty`` is False; every log's own record, where there
    is one, names the same commit and dirtiness as its decode's; every entry
    of `allowed` gives a reason; and every decode that was never saved is
    explained in `explained`.

    Parameters
    ----------
    provenance : pd.DataFrame
        ``decode_provenance``'s rows.
    allowed : Mapping of str to str
        ``decode_id`` (the epoch id, or the epoch id plus ``_contfrag``) to
        the reason it may differ. These rows are set aside and reported. An
        entry without a reason is refused, and its row is checked.
    explained : Mapping of str to str, optional
        ``decode_id`` to the reason ``check_epoch_sets`` records for that
        decode's absence. A decode never saved (no ``results.nc``) has no
        provenance to check: with a reason here it is reported, not failed;
        without one it fails, as it does in ``check_epoch_sets``.

    Returns
    -------
    dict
        ``details["commit"]`` is the one commit, or None when there is not
        exactly one.
    """
    failures: list[str] = []
    explained = explained or {}
    reasonless = sorted(name for name, reason in allowed.items() if not reason)
    if reasonless:
        failures.append(
            f"the exception for {_names(reasonless)} has no reason, so it is not "
            "applied"
        )
    excepted = (
        provenance["decode_id"]
        .isin([name for name, reason in allowed.items() if reason])
        .to_numpy(dtype=bool)
    )
    exceptions = {
        row.decode_id: {
            "reason": allowed[row.decode_id],
            "commit": row.commit,
            "dirty": row.dirty,
            "source": row.source,
        }
        for row in provenance.loc[excepted].itertuples()
    }
    rows = provenance.loc[~excepted]
    if rows.empty:
        failures.append("no decode provenance to check")

    unsaved = ~rows["saved"].astype(bool)
    absent_decodes = {
        str(name): explained[name]
        for name in rows.loc[unsaved, "decode_id"]
        if explained.get(name)
    }
    unexplained = sorted(
        str(name)
        for name in rows.loc[unsaved, "decode_id"]
        if name not in absent_decodes
    )
    if unexplained:
        failures.append(
            f"{_names(unexplained)}: no results.nc, so the decode was never saved "
            "and has no provenance; epoch_sets records no reason for its absence"
        )
    rows = rows.loc[~unsaved]

    no_source = rows["source"].map(_missing).astype(bool)
    unread = rows.loc[no_source]
    resumed = unread["resumed"].astype(bool)
    if resumed.any():
        failures.append(
            f"{_names(unread.loc[resumed, 'decode_id'])}: results.nc carries no "
            "provenance and the log is marked RESUMED, so the log's PROVENANCE "
            "line describes the resume, not the decode"
        )
    if (~resumed).any():
        failures.append(
            f"{_names(unread.loc[~resumed, 'decode_id'])}: no provenance in "
            "results.nc or in the log"
        )

    read = rows.loc[~no_source]
    for field in SINGLE_VALUED_FIELDS:
        absent = read[field].map(_missing).astype(bool)
        if absent.any():
            failures.append(
                f"{field} is not recorded for {_names(read.loc[absent, 'decode_id'])}"
            )
        by_value = read.loc[~absent].groupby(field)["decode_id"].apply(list)
        if len(by_value) > 1:
            failures.append(
                f"{len(by_value)} values of {field}: "
                + "; ".join(
                    f"{value} ({len(ids)} decodes: {_names(ids)})"
                    for value, ids in by_value.items()
                )
            )
    not_recorded = read["dirty"].map(_missing).astype(bool)
    if not_recorded.any():
        failures.append(
            f"dirty is not recorded for {_names(read.loc[not_recorded, 'decode_id'])}"
        )
    dirty = (
        read["dirty"]
        .map(lambda value: not _missing(value) and bool(value))
        .astype(bool)
    )
    if dirty.any():
        failures.append(
            f"made from a dirty tree: {_names(read.loc[dirty, 'decode_id'])}"
        )

    for row in read.itertuples():
        if _missing(row.log_commit):
            continue
        if row.log_commit != row.commit or bool(row.log_dirty) != bool(row.dirty):
            failures.append(
                f"{row.decode_id}: its log is from a later run at "
                f"{str(row.log_commit)[:12]}{' (dirty)' if row.log_dirty else ''}"
                f"{', resumed' if row.resumed else ''}, not from the decode's commit "
                f"{str(row.commit)[:12]}{' (dirty)' if row.dirty else ''}; a resume "
                "rewrites data_over_time"
            )

    commits = sorted({str(commit) for commit in read["commit"] if not _missing(commit)})
    details = {
        "commit": commits[0] if len(commits) == 1 else None,
        "n_decodes": int(len(rows)),
        "sources": dict(Counter(str(source) for source in rows["source"])),
        "n_resumed": int(rows["resumed"].astype(bool).sum()),
        "values": {
            field: sorted({str(value) for value in read[field] if not _missing(value)})
            for field in SINGLE_VALUED_FIELDS
        },
        "exceptions": exceptions,
        "unused_exceptions": sorted(set(allowed) - set(provenance["decode_id"])),
        "absent": {
            name: f"explained in epoch_sets: {why}"
            for name, why in absent_decodes.items()
        },
    }
    return _result(failures, details)


# --------------------------------------------------------------------------
# Decoder settings
# --------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """`value` as JSON types: arrays as lists, dataclasses as their fields."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        return {"class": type(value).__name__, **fields}
    if hasattr(value, "__dict__"):
        return {"class": type(value).__name__, **_plain(vars(value))}
    return repr(value)


def _close(value: Any, expected: Any) -> bool:
    """Whether two JSON-typed values agree, numbers to a relative 1e-9."""
    if isinstance(value, dict) and isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _close(value[key], expected[key]) for key in value
        )
    if isinstance(value, list) and isinstance(expected, list):
        return len(value) == len(expected) and all(
            _close(item, other) for item, other in zip(value, expected, strict=True)
        )
    numbers = (int, float)
    if (
        isinstance(value, numbers)
        and isinstance(expected, numbers)
        and not isinstance(value, bool)
        and not isinstance(expected, bool)
    ):
        return math.isclose(value, expected, rel_tol=1e-9, abs_tol=0.0)
    return bool(value == expected)


def decoder_settings(detector: object) -> dict[str, Any]:
    """A fitted detector's :data:`DISCRETE_SETTINGS`, as JSON types."""
    return {name: _plain(getattr(detector, name, None)) for name in DISCRETE_SETTINGS}


def _default_argument(detector_class: type, name: str) -> object:
    """The value `detector_class` gives its argument `name` when none is passed."""
    return inspect.signature(detector_class).parameters[name].default


def expected_decoder_settings() -> dict[str, dict[str, object]]:
    """The :data:`DISCRETE_SETTINGS` each decoder's detectors must carry, from the code.

    The audit's default expectation, so that the check rests on the code at
    the audit's commit rather than on a file outside git. For each decoder:

    - the five pinned transition settings, from
      ``decoding_transitions.nonlocal_transition_settings()`` and
      ``contfrag_transition_settings()``. Their keys are the detectors'
      attribute names, which :data:`DISCRETE_SETTINGS` uses too;
    - ``discrete_transition_regularization``: no decoder passes it, so a
      detector carries its class's default. It is read from the signature of
      the class each HPC decode builds, ``NonLocalClusterlessDetector`` or
      ``ContFragClusterlessClassifier``;
    - ``local_position_std``: for NonLocal, the value ``run_decoding_bandit.py``
      passes, ``NonLocalDecodingConfig().local_position_std`` (by default
      ``parameters.LOCAL_POSITION_STD``, 2.5 cm); for ContFrag, which has no
      Local state, None.

    Returns
    -------
    dict of str to dict of str to object
        ``{"nonlocal": {...}, "contfrag": {...}}``, each setting in JSON types
        (``_plain``), the form an ``--expected-settings`` file holds.
    """
    regularization = "discrete_transition_regularization"
    settings = {
        "nonlocal": {
            **nonlocal_transition_settings(),
            regularization: _default_argument(
                NonLocalClusterlessDetector, regularization
            ),
            "local_position_std": NonLocalDecodingConfig().local_position_std,
        },
        "contfrag": {
            **contfrag_transition_settings(),
            regularization: _default_argument(
                ContFragClusterlessClassifier, regularization
            ),
            "local_position_std": None,
        },
    }
    return {decoder: _plain(values) for decoder, values in settings.items()}


def check_decoder_settings(
    detector_paths: Mapping[str, Path], expected: Mapping[str, object] | None = None
) -> dict:
    """The same discrete-transition settings in every epoch's detector.

    Passes when every detector carries identical :data:`DISCRETE_SETTINGS`
    (exactly, since one code path sets them all), and, for each setting
    `expected` names, a value equal to it, numbers to a relative 1e-9. That
    tolerance only absorbs how a value was written down (0.0025 against the
    0.0025000000000000044 a detector computes): it is far below any drift that
    matters, such as April's 0.98 self-transition against today's 0.96, or
    stickiness 1e6 against 1e5. A setting `expected` names that no detector
    carries fails, so a misspelled name is not silently left unchecked.

    Parameters
    ----------
    detector_paths : Mapping of str to Path
        Epoch identifier to one decoder's ``_detector.pkl``. They are loaded
        one at a time, so only one is in memory at once.
    expected : Mapping of str to object, optional
        Setting name to its expected value, in JSON types; the transition
        type as ``{"class": ..., <its fields>}``.

    Returns
    -------
    dict
        ``details["settings"]`` holds the most common settings; when epochs
        disagree, ``details["groups"]`` lists every distinct set with its
        epochs.
    """
    groups: dict[str, list[str]] = {}
    for epoch_id, path in sorted(detector_paths.items()):
        settings = decoder_settings(joblib.load(path))
        groups.setdefault(json.dumps(settings, sort_keys=True), []).append(epoch_id)

    failures: list[str] = []
    if not groups:
        failures.append("no detectors to check")
        return _result(failures, {"n_epochs": 0, "settings": None, "groups": None})
    reference_key = max(groups, key=lambda key: len(groups[key]))
    reference = json.loads(reference_key)
    for key, epochs in groups.items():
        if key == reference_key:
            continue
        theirs = json.loads(key)
        differing = [
            name for name in DISCRETE_SETTINGS if theirs[name] != reference[name]
        ]
        failures.append(
            f"{len(epochs)} epoch(s) carry another {', '.join(differing)} than the "
            f"other {len(groups[reference_key])}: {_names(epochs, limit=20)} ("
            + "; ".join(
                f"{name} {theirs[name]} against {reference[name]}" for name in differing
            )
            + ")"
        )
    if expected:
        unknown = sorted(set(expected) - set(DISCRETE_SETTINGS))
        if unknown:
            failures.append(
                f"expected names settings no detector carries: {', '.join(unknown)}"
            )
        for key, epochs in groups.items():
            settings = json.loads(key)
            off = [
                name
                for name in expected
                if name in DISCRETE_SETTINGS
                and not _close(settings[name], _plain(expected[name]))
            ]
            if off:
                failures.append(
                    f"{_names(epochs, limit=20)}: "
                    + "; ".join(
                        f"{name} is {settings[name]}, not the expected "
                        f"{_plain(expected[name])}"
                        for name in off
                    )
                )
    details = {
        "n_epochs": sum(len(epochs) for epochs in groups.values()),
        "settings": reference,
        "groups": (
            [
                {"epochs": epochs, "settings": json.loads(key)}
                for key, epochs in groups.items()
            ]
            if len(groups) > 1
            else None
        ),
    }
    return _result(failures, details)


# --------------------------------------------------------------------------
# 2. NonLocal against ContFrag, and the minimum sample count
# --------------------------------------------------------------------------


def check_grids_match(nonlocal_dot: pd.DataFrame, contfrag_dot: pd.DataFrame) -> dict:
    """One epoch's NonLocal and ContFrag ``data_over_time`` share grid and position.

    Passes when the time index, ``linear_position`` and ``head_speed`` are
    identical (NaN where the other is NaN), as they are when both decoders
    read one ``load_data``, and the grid passes ``validate_time_grid``
    (uniform, at 500 Hz), which the Widloski detector requires.
    """
    nonlocal_dot = _time_indexed(nonlocal_dot)
    contfrag_dot = _time_indexed(contfrag_dot)
    failures: list[str] = []
    nonlocal_time = nonlocal_dot.index.to_numpy(dtype=np.float64)
    contfrag_time = contfrag_dot.index.to_numpy(dtype=np.float64)
    if nonlocal_time.shape != contfrag_time.shape:
        failures.append(
            f"the time grids have {nonlocal_time.size} (NonLocal) and "
            f"{contfrag_time.size} (ContFrag) samples"
        )
    elif not np.array_equal(nonlocal_time, contfrag_time):
        differ = np.flatnonzero(nonlocal_time != contfrag_time)
        first = int(differ[0])
        failures.append(
            f"the time grids differ at {differ.size} samples, first at sample "
            f"{first} (NonLocal {nonlocal_time[first]!r}, ContFrag "
            f"{contfrag_time[first]!r})"
        )
    for column in ("linear_position", "head_speed"):
        lacking = [
            name
            for name, frame in (("NonLocal", nonlocal_dot), ("ContFrag", contfrag_dot))
            if column not in frame.columns
        ]
        if lacking:
            failures.append(f"{column} is missing from {' and '.join(lacking)}")
            continue
        ours = nonlocal_dot[column].to_numpy(dtype=np.float64)
        theirs = contfrag_dot[column].to_numpy(dtype=np.float64)
        if ours.shape != theirs.shape or not np.array_equal(
            ours, theirs, equal_nan=True
        ):
            failures.append(f"{column} differs between NonLocal and ContFrag")
    try:
        validate_time_grid(nonlocal_time, MemorylessSequenceConfig())
    except ValueError as error:
        failures.append(f"the grid fails validate_time_grid: {error}")
    details = {
        "n_samples": int(nonlocal_time.size),
        "span_s": (
            [float(nonlocal_time[0]), float(nonlocal_time[-1])]
            if nonlocal_time.size
            else None
        ),
    }
    return _result(failures, details)


def check_minimum_sample_counts(times: Mapping[str, NDArray]) -> dict:
    """``minimum_sample_count(time, 0.015)`` is the same for every epoch (R9).

    Parameters
    ----------
    times : Mapping of str to ndarray, each shape (n_time,)
        Epoch identifier to its decode's time grid, in seconds.
    """
    counts = {
        epoch_id: int(
            minimum_sample_count(
                np.asarray(time, dtype=np.float64), NON_LOCAL_MINIMUM_DURATION_S
            )
        )
        for epoch_id, time in sorted(times.items())
    }
    by_count: dict[int, list[str]] = {}
    for epoch_id, count in counts.items():
        by_count.setdefault(count, []).append(epoch_id)
    failures: list[str] = []
    if not counts:
        failures.append("no time grids to check")
    elif len(by_count) > 1:
        failures.append(
            "the minimum non-local sample count differs between epochs: "
            + "; ".join(
                f"{count} ({len(epochs)} epochs: {_names(epochs)})"
                for count, epochs in sorted(by_count.items())
            )
        )
    details = {
        "minimum_duration_s": NON_LOCAL_MINIMUM_DURATION_S,
        "n_epochs_by_count": {
            str(count): len(epochs) for count, epochs in sorted(by_count.items())
        },
        "per_epoch": counts,
    }
    return _result(failures, details)


# --------------------------------------------------------------------------
# 3. Marks, 4. Widloski, 5. SWR inventories
# --------------------------------------------------------------------------


def check_encoding_marks(
    epoch_id: str,
    root: Path,
    *,
    load_marks: Callable[[str], tuple[list, list]] = get_hpc_marks,
    load_encoding_model: Callable[[Path], dict] = load_clusterless_encoding_model,
) -> dict:
    """The reloaded marks hold every encoding mark of both decodes of an epoch.

    The marks are read as the Widloski runner reads them, which is how
    ``load_data`` read them for the decodes: ``get_hpc_marks``, then
    ``filter_spike_times`` over the NonLocal decode's time span. Each
    decode's encoding model is then checked with ``check_marks_match_encoding``.
    ``get_hpc_marks`` reads the Spyglass database.

    Parameters
    ----------
    epoch_id : str
        A canonical epoch identifier, such as ``chimi20200225_02_r1_HPC``.
    root : Path
        Where the decodes are saved.
    load_marks : Callable, optional
        ``(nwb_file_name) -> (spike_times, spike_waveform_features)``.
    load_encoding_model : Callable, optional
        ``(detector_path) -> encoding model``.
    """
    stem = epoch_id.split("_", 1)[0]
    dot = _time_indexed(
        pd.read_pickle(decode_files(epoch_id, root, "nonlocal")["data_over_time"])
    )
    time = dot.index.to_numpy(dtype=np.float64)
    span = np.array([time[0], time[-1]])
    spike_times, spike_waveform_features = load_marks(f"{stem}_.nwb")
    in_span_times, in_span_features = filter_spike_times(
        {"HPC": spike_times}, {"HPC": spike_waveform_features}, span
    )
    times = in_span_times.get("HPC", [])
    features = in_span_features.get("HPC", [])
    failures = []
    for decoder in DECODERS:
        detector = decode_files(epoch_id, root, decoder)["detector"]
        try:
            check_marks_match_encoding(load_encoding_model(detector), times, features)
        except ValueError as error:
            failures.append(f"{decoder}: {error}")
    details = {
        "span_s": span.tolist(),
        "n_electrodes": len(times),
        "n_marks": int(sum(len(electrode) for electrode in times)),
    }
    return _result(failures, details)


def check_widloski_hashes(manifest: dict, root: Path) -> dict:
    """The decode files every ``ok`` Widloski epoch read are the current ones.

    Each recorded input (``detector`` and ``data_over_time``) is looked up
    under `root` by its file name, so the run is compared with the canonical
    decode files wherever it read them from; its sha256 must equal the one
    the run recorded.
    """
    records = manifest.get("epochs", {})
    ok = {
        epoch_id: record
        for epoch_id, record in records.items()
        if record.get("status") == "ok"
    }
    failures: list[str] = []
    if not ok:
        failures.append("the manifest records no epoch as ok")
    n_files = 0
    for epoch_id, record in sorted(ok.items()):
        inputs = record.get("inputs") or {}
        if not inputs:
            failures.append(f"{epoch_id}: no input hashes recorded")
        for name, recorded in sorted(inputs.items()):
            current = root / Path(recorded["path"]).name
            n_files += 1
            if not current.exists():
                failures.append(f"{epoch_id}: {name} {current} is missing")
            elif sha256_file(current) != recorded["sha256"]:
                failures.append(
                    f"{epoch_id}: {name} {current} is not the file the run read: "
                    "its sha256 differs from the recorded one"
                )
    details = {
        "run_id": manifest.get("run_id"),
        "git_commit": manifest.get("settings", {}).get("git_commit"),
        "n_epochs": len(ok),
        "n_files": n_files,
    }
    return _result(failures, details)


def check_swr_spans(
    swr_manifest: dict,
    decode_spans: Mapping[str, tuple[float, float]],
    tolerance_s: float = 0.002,
) -> dict:
    """The SWR runner's position span is each decode's, within `tolerance_s`.

    Every epoch of `decode_spans` with an ``ok`` or ``partial`` SWR record is
    compared; such a record without ``position_span_s`` fails. An epoch the
    SWR run skipped, failed or lacks is left to ``check_epoch_sets`` and
    listed under ``details["not_compared"]``. Nothing compared at all fails.

    The comparison is between the SWR runner's ``position_info`` span and the
    decode's ``data_over_time`` span, so a failure says the two differ, not
    which one moved. It can come from the SWR runner, or from a change to the
    decode grid, such as ``get_data_over_time`` trimming it.

    Parameters
    ----------
    swr_manifest : dict
        The SWR run's ``manifest.json``.
    decode_spans : Mapping of str to (float, float)
        Epoch identifier to its decode grid's first and last time, in seconds.
    tolerance_s : float
        The largest difference allowed at either end, in seconds.
    """
    records = swr_manifest.get("epochs", {})
    failures: list[str] = []
    not_compared: dict[str, str] = {}
    n_compared, largest = 0, 0.0
    for epoch_id, (start, end) in sorted(decode_spans.items()):
        record = records.get(epoch_id)
        status = "absent" if record is None else str(record.get("status"))
        if status not in ("ok", "partial"):
            not_compared[epoch_id] = status
            continue
        span = record.get("position_span_s")
        if span is None:
            failures.append(f"{epoch_id}: the SWR record has no position_span_s")
            continue
        difference = max(abs(span[0] - start), abs(span[1] - end))
        n_compared += 1
        largest = max(largest, difference)
        if not difference <= tolerance_s:  # NaN fails too
            failures.append(
                f"{epoch_id}: the SWR runner's position span [{span[0]:.6f}, "
                f"{span[1]:.6f}] differs from the decode's [{start:.6f}, {end:.6f}] "
                f"by up to {difference:.6f} s, more than {tolerance_s} s"
            )
    if n_compared == 0 and not failures:
        failures.append("no epoch of the decodes has an SWR span to compare")
    details = {
        "run_id": swr_manifest.get("run_id"),
        "n_compared": n_compared,
        "largest_difference_s": largest,
        "tolerance_s": tolerance_s,
        "speed_source": swr_manifest.get("settings", {}).get("speed"),
        "not_compared": not_compared,
    }
    return _result(failures, details)


def check_spike_loader_unchanged(
    repo: Path, decode_commit: str, run_commit: str
) -> dict:
    """``data_loaders/spikes.py`` is the same at `run_commit` as at `decode_commit`.

    Passes when ``git --no-optional-locks diff --quiet <decode_commit>
    <run_commit> -- src/continuum_swr_replay/data_loaders/spikes.py`` exits 0.
    The Widloski runner reloads marks through this loader, so it must be the
    loader the decodes read them with; ``check_marks_match_encoding`` sees
    only the training span's marks. A run commit ending in ``-dirty`` fails:
    its loader cannot be tied to a commit.
    """
    details = {
        "decode_commit": decode_commit,
        "run_commit": run_commit,
        "path": SPIKE_LOADER,
    }
    if run_commit.endswith("-dirty"):
        return _result(
            [
                f"the run was made from a dirty tree ({run_commit}), so the spike "
                "loader it ran cannot be tied to a commit"
            ],
            details,
        )
    diff = _git(repo, "diff", "--quiet", decode_commit, run_commit, "--", SPIKE_LOADER)
    if diff.returncode == 0:
        return _result([], details)
    if diff.returncode == 1:
        return _result(
            [
                f"{SPIKE_LOADER} differs between the decode commit "
                f"{decode_commit[:12]} and the run commit {run_commit[:12]}: the run "
                "reloaded marks through another loader than the decodes read them "
                "with"
            ],
            details,
        )
    return _result(
        [
            f"git could not compare {decode_commit} with {run_commit}: "
            f"{diff.stderr.strip()}"
        ],
        details,
    )


# --------------------------------------------------------------------------
# 6. Tables
# --------------------------------------------------------------------------


def table_files(table_dir: Path) -> list[str]:
    """The tables in `table_dir`, as paths relative to it.

    Every ``*.parquet`` and ``*.json`` file directly in `table_dir`, except the
    tables manifest and fit sidecars (``*.provenance.json``, which change
    whenever a model is refit), plus the qvals table and its standardization
    reference in ``models_per_bin_rate/`` when present. Nothing else below
    `table_dir` is a table: its subfolders hold fits, backups and shards.
    """
    names = [
        path.name
        for path in table_dir.iterdir()
        if path.is_file()
        and path.suffix in TABLE_SUFFIXES
        and not path.name.startswith(".")
        and path.name != TABLES_MANIFEST
        and not path.name.endswith(FIT_SIDECAR_SUFFIX)
    ]
    names += [
        relative for relative in QVALS_TABLE_FILES if (table_dir / relative).is_file()
    ]
    return sorted(names)


def record_tables(table_dir: Path, commit: str, inputs: Mapping[str, str]) -> Path:
    """Write ``table_dir/tables_manifest.json``: each table's sha256 and its origin.

    Parameters
    ----------
    table_dir : Path
        A folder of tables (``table_files``).
    commit : str
        The commit the tables were built at.
    inputs : Mapping of str to str
        What they were built from, such as ``decode_commit``, the qvals
        source and the behavioral cache. ``check_tables_recorded`` requires
        ``decode_commit``.

    Returns
    -------
    Path
        The manifest, the only file written (atomically).

    Raises
    ------
    ValueError
        If `table_dir` holds no table.
    """
    files = table_files(table_dir)
    if not files:
        raise ValueError(f"{table_dir} holds no table (*.parquet or *.json) to record.")
    manifest = {
        "produced_by": f"{PRODUCED_BY} record-tables",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "inputs": dict(inputs),
        "files": {relative: sha256_file(table_dir / relative) for relative in files},
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path = table_dir / TABLES_MANIFEST
    write_atomically(path, lambda temporary: temporary.write_text(text))
    return path


def check_tables_recorded(
    table_dirs: list[Path], decode_commit: str, *, repo: Path = REPO
) -> dict:
    """Each table folder holds exactly the tables its manifest records.

    Passes, per folder, when ``tables_manifest.json`` exists and records at
    least one table; the folder's tables (``table_files``) are exactly the
    recorded ones, each with its recorded sha256; the manifest's ``commit``
    descends from `decode_commit` (``git merge-base --is-ancestor``, which
    counts a commit as its own descendant); and its ``inputs`` name
    `decode_commit` as ``decode_commit``.

    Parameters
    ----------
    table_dirs : list of Path
        The promoted table folders.
    decode_commit : str
        The decodes' commit.
    repo : Path, optional
        The git repository the commits belong to.
    """
    failures: list[str] = []
    details: dict[str, dict] = {}
    if not table_dirs:
        failures.append("no table folders to check")
    for table_dir in table_dirs:
        path = table_dir / TABLES_MANIFEST
        if not path.is_file():
            failures.append(f"{table_dir}: no {TABLES_MANIFEST}")
            continue
        manifest = json.loads(path.read_text())
        recorded: dict[str, str] = manifest.get("files") or {}
        present = set(table_files(table_dir))
        if not recorded:
            failures.append(f"{table_dir}: the manifest records no table")
        missing = sorted(set(recorded) - present)
        if missing:
            failures.append(f"{table_dir}: recorded but missing: {', '.join(missing)}")
        unrecorded = sorted(present - set(recorded))
        if unrecorded:
            failures.append(
                f"{table_dir}: present but not recorded: {', '.join(unrecorded)}"
            )
        for relative in sorted(set(recorded) & present):
            if sha256_file(table_dir / relative) != recorded[relative]:
                failures.append(
                    f"{table_dir}: {relative} changed after it was recorded"
                )
        commit = manifest.get("commit")
        if not commit:
            failures.append(f"{table_dir}: the manifest records no commit")
        else:
            commit = str(commit)
            ancestry = _git(repo, "merge-base", "--is-ancestor", decode_commit, commit)
            if ancestry.returncode == 1:
                failures.append(
                    f"{table_dir}: built at {commit}, which does not descend from "
                    f"the decode commit {decode_commit}"
                )
            elif ancestry.returncode != 0:
                failures.append(
                    f"{table_dir}: git could not tell whether {commit} descends "
                    f"from {decode_commit}: {ancestry.stderr.strip()}"
                )
        inputs = manifest.get("inputs") or {}
        named = inputs.get("decode_commit")
        if named is None or not _same_commit(repo, str(named), decode_commit):
            failures.append(
                f"{table_dir}: its inputs name decode_commit {named!r}, not the "
                f"decode commit {decode_commit}"
            )
        details[str(table_dir)] = {
            "commit": commit,
            "inputs": inputs,
            "recorded_at": manifest.get("recorded_at"),
            "n_files": len(recorded),
        }
    return _result(failures, details)


# --------------------------------------------------------------------------
# 7. Models
# --------------------------------------------------------------------------


def check_pfc_sidecars(sidecars: list[dict], event_table_sha256: str) -> dict:
    """Every PFC bundle's provenance sidecar names the canonical event table.

    Passes when there is at least one sidecar and each one's
    ``canonical_event_table_sha256`` is `event_table_sha256`.
    """
    failures: list[str] = []
    if not sidecars:
        failures.append("no PFC provenance sidecars to check")
    others = [
        f"{sidecar.get('artifact') or f'sidecar {index}'} "
        f"({str(sidecar.get('canonical_event_table_sha256'))[:12]})"
        for index, sidecar in enumerate(sidecars)
        if sidecar.get("canonical_event_table_sha256") != event_table_sha256
    ]
    if others:
        failures.append(
            f"{len(others)} of {len(sidecars)} PFC sidecars name another event table "
            f"than the canonical one ({event_table_sha256[:12]}): "
            f"{_names(others, limit=20)}"
        )
    details = {
        "n_sidecars": len(sidecars),
        "event_table_sha256": event_table_sha256,
        "named": dict(
            Counter(str(s.get("canonical_event_table_sha256")) for s in sidecars)
        ),
    }
    return _result(failures, details)


def check_fit_provenance(
    repo: Path, roots: list[Path], *, artifacts: Sequence[Path] = ()
) -> dict:
    """``scripts/qa/check_fit_provenance.py`` passes on each root.

    The script runs once per root (``--root``, sweeping its ``models*/``
    trees), and once per folder of `artifacts`, which names fits outside any
    ``models*/`` tree, such as the Stage 5B and Stage 6 fits at the
    ``replay_content`` root. Passes when every run exits 0 and every root
    holds at least one fit.

    Parameters
    ----------
    repo : Path
        The checkout whose ``scripts/qa/check_fit_provenance.py`` runs.
    roots : list of Path
        Folders holding ``models*/`` trees.
    artifacts : Sequence of Path, optional
        Fitted artifacts, each checked against its own sidecar.
    """
    runs: list[tuple[str, list[str]]] = [
        (str(root), ["--root", str(root)]) for root in roots
    ]
    for parent in sorted({artifact.parent for artifact in artifacts}):
        named = sorted(str(a) for a in artifacts if a.parent == parent)
        runs.append((f"{parent} (named fits)", ["--root", str(parent), *named]))
    failures: list[str] = []
    details: dict[str, dict] = {}
    if not runs:
        failures.append("no roots to check")
    script = repo / CHECK_FIT_PROVENANCE
    for label, arguments in runs:
        process = subprocess.run(
            [sys.executable, str(script), "--json", *arguments],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        try:
            results = json.loads(process.stdout)
        except json.JSONDecodeError:
            stderr = process.stderr.strip().splitlines()
            failures.append(
                f"{label}: check_fit_provenance.py exited {process.returncode} with "
                f"no report: {stderr[-1] if stderr else 'no output'}"
            )
            details[label] = {"exit_code": process.returncode}
            continue
        not_fresh = [
            f"{result['artifact']} ({result['status']})"
            for result in results
            if result["status"] != "fresh"
        ]
        if process.returncode != 0:
            failures.append(
                f"{label}: check_fit_provenance.py exited {process.returncode}: "
                f"{len(not_fresh)} of {len(results)} fits not fresh: "
                f"{_names(not_fresh)}"
            )
        elif not results:
            failures.append(f"{label}: no fitted artifact to check")
        details[label] = {
            "exit_code": process.returncode,
            "n_artifacts": len(results),
            "statuses": dict(Counter(result["status"] for result in results)),
            "not_fresh": not_fresh,
        }
    return _result(failures, details)


# --------------------------------------------------------------------------
# 8. Epoch sets
# --------------------------------------------------------------------------


def run_manifest_epochs(manifest: dict) -> tuple[set[str], dict[str, str]]:
    """A run manifest's canonical epochs that are ``ok``, and why others were skipped.

    Only ``skipped`` records carry a reason that explains an absence; a
    ``failed`` or ``partial`` epoch is neither present nor explained.

    Returns
    -------
    present : set of str
    reasons : dict of str to str
        Epoch identifier to its ``skipped`` record's reason.
    """
    records = manifest.get("epochs", {})
    present = {
        epoch_id
        for epoch_id, record in records.items()
        if is_canonical_epoch_id(epoch_id) and record.get("status") == "ok"
    }
    reasons = {
        epoch_id: str(record["reason"])
        for epoch_id, record in records.items()
        if is_canonical_epoch_id(epoch_id)
        and record.get("status") == "skipped"
        and record.get("reason")
    }
    return present, reasons


def check_epoch_sets(
    sets: Mapping[str, set[str]],
    reasons: Mapping[str, Mapping[str, str]],
    *,
    known: Iterable[str] = (),
) -> dict:
    """Every artifact holds every epoch, or a recorded reason says why not.

    The epochs expected of every artifact are those any artifact holds, plus
    `known` (the epochs the SWR runner recorded, which cover
    ``get_epoch_info``), so an epoch no artifact holds is not overlooked. An
    epoch missing from an artifact passes only with a non-empty reason in
    ``reasons[artifact]``.

    Parameters
    ----------
    sets : Mapping of str to set of str
        Artifact name to the epoch identifiers it holds.
    reasons : Mapping of str to Mapping of str to str
        Artifact name to epoch identifier to why the artifact lacks it.
    known : Iterable of str, optional
        Further epochs every artifact should hold.
    """
    expected = set(known).union(*sets.values())
    failures: list[str] = []
    if not expected:
        failures.append("no epochs to check")
    details: dict[str, dict] = {}
    for artifact, present in sets.items():
        recorded = reasons.get(artifact, {})
        missing = sorted(expected - present)
        explained = {
            epoch_id: recorded[epoch_id]
            for epoch_id in missing
            if recorded.get(epoch_id)
        }
        unexplained = [epoch_id for epoch_id in missing if epoch_id not in explained]
        if unexplained:
            failures.append(
                f"{artifact} lacks {len(unexplained)} epoch(s) with no recorded "
                f"reason: {', '.join(unexplained)}"
            )
        details[artifact] = {
            "n_present": len(present),
            "n_expected": len(expected),
            "explained": explained,
            "unexplained": unexplained,
        }
    return _result(failures, details)


def _epoch_ids(sessions: pd.DataFrame) -> pd.Series:
    """``{animal}{session_date}_{epoch}_HPC`` for each row of the session columns."""
    return (
        sessions["animal"].astype(str)
        + sessions["session_date"].astype(str)
        + "_"
        + sessions["epoch"].astype(str)
        + "_HPC"
    )


def stop_epochs(stop_table: Path) -> pd.Series | None:
    """``stop_id`` to epoch identifier, from a stop table; None if it has neither."""
    if not stop_table.is_file():
        return None
    columns = ["stop_id", *SESSION_COLUMNS]
    if not set(columns) <= set(pq.read_schema(stop_table).names):
        return None
    stops = pq.read_table(stop_table, columns=columns).to_pandas()
    return pd.Series(_epoch_ids(stops).to_numpy(), index=stops["stop_id"].to_numpy())


def table_epochs(path: Path, stops: pd.Series | None = None) -> set[str] | None:
    """The canonical epochs a table holds.

    From its session columns (``animal``, ``session_date``, ``epoch``); else,
    for a table keyed by ``stop_id`` alone, as the qvals table is, through
    `stops` (``stop_epochs`` of the folder's stop table).

    Returns
    -------
    set of str, or None
        None when the table has neither session columns nor a ``stop_id``
        that `stops` can map.

    Raises
    ------
    ValueError
        If some of its ``stop_id`` values are not in `stops`.
    """
    names = set(pq.read_schema(path).names)
    if set(SESSION_COLUMNS) <= names:
        sessions = pq.read_table(path, columns=list(SESSION_COLUMNS)).to_pandas()
        return set(_epoch_ids(sessions.drop_duplicates()))
    if "stop_id" not in names or stops is None:
        return None
    stop_ids = pq.read_table(path, columns=["stop_id"]).to_pandas()["stop_id"]
    held = stop_ids.map(stops)
    unmapped = int(held.isna().sum())
    if unmapped:
        raise ValueError(
            f"{unmapped} of its {held.size} stop_ids are not in the folder's stop "
            "table, so their epochs cannot be told"
        )
    return set(held)


def table_epoch_sets(
    table_dirs: Sequence[Path],
) -> tuple[dict[str, set[str]], list[str], list[str]]:
    """Each recorded parquet table's epochs, named ``<folder>/<file>``.

    A table keyed by ``stop_id`` alone is read through its folder's
    ``stop_table.parquet``.

    Returns
    -------
    sets : dict of str to set of str
    failures : list of str
        Tables some of whose stops the folder's stop table lacks.
    not_checked : list of str
        Tables with neither session columns nor a mappable ``stop_id``, which
        the epoch-set check cannot cover.
    """
    sets: dict[str, set[str]] = {}
    failures: list[str] = []
    not_checked: list[str] = []
    for table_dir in table_dirs:
        if not table_dir.is_dir():
            continue
        stops = stop_epochs(table_dir / "stop_table.parquet")
        for relative in table_files(table_dir):
            if not relative.endswith(".parquet"):
                continue
            name = f"{table_dir.name}/{relative}"
            try:
                held = table_epochs(table_dir / relative, stops)
            except ValueError as error:
                failures.append(f"{name}: {error}")
                continue
            if held is None:
                not_checked.append(
                    f"epoch_sets: {name} has no session columns and no stop_id "
                    "its folder's stop_table.parquet maps, so its epochs are not "
                    "checked"
                )
            else:
                sets[name] = held
    return sets, failures, not_checked


# --------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------


def _per_epoch(results: Mapping[str, dict], nothing: str) -> dict:
    """One check made of per-epoch results, failing if there were none."""
    failures = [
        f"{epoch_id}: {failure}"
        for epoch_id, result in sorted(results.items())
        for failure in result["failures"]
    ]
    if not results:
        failures.append(nothing)
    details = {
        "n_epochs": len(results),
        "failing": {
            epoch_id: result["details"]
            for epoch_id, result in sorted(results.items())
            if not result["ok"]
        },
    }
    return _result(failures, details)


def _read_json(path: Path) -> dict:
    content: dict = json.loads(path.read_text())
    return content


def _run_commits(widloski: dict | None, swr: dict | None) -> dict[str, set[str]]:
    """Each run's commits: the Widloski run's settings and records, the SWR records'."""
    commits: dict[str, set[str]] = {}
    if widloski is not None:
        records = widloski.get("epochs", {}).values()
        commits["widloski"] = {
            str(commit)
            for commit in [
                widloski.get("settings", {}).get("git_commit"),
                *(
                    record.get("git_commit")
                    for record in records
                    if record.get("status") == "ok"
                ),
            ]
            if commit
        }
    if swr is not None:
        provenances = [
            record["provenance"]
            for record in swr.get("epochs", {}).values()
            if record.get("status") in ("ok", "partial") and record.get("provenance")
        ]
        commits["swr"] = {
            f"{record['commit']}{'-dirty' if record.get('dirty') else ''}"
            for record in provenances
        }
    return commits


ROOT_FIT_FAMILIES = {
    "Stage 5B": "stage5b_*.pkl",
    "Stage 6": "stage6_*.pkl",
    "stop_replay_summary": "stop_replay_summary*.pkl",
}
"""The fits at the ``replay_content`` root, outside every ``models*/`` tree."""


def root_fits(replay_content_dir: Path) -> tuple[list[Path], list[str]]:
    """The fits at the ``replay_content`` root, and each family with none.

    Every artifact a sidecar there describes, and every ``.pkl`` of
    :data:`ROOT_FIT_FAMILIES`, with a sidecar or without one, which
    ``check_fit_provenance.py`` then reports as ``missing_sidecar``.

    Returns
    -------
    artifacts : list of Path
    not_checked : list of str
        One entry per family with no fit at all.
    """
    artifacts = {
        sidecar.with_name(sidecar.name[: -len(FIT_SIDECAR_SUFFIX)] + ".pkl")
        for sidecar in replay_content_dir.glob(f"*{FIT_SIDECAR_SUFFIX}")
    }
    not_checked = []
    for family, pattern in ROOT_FIT_FAMILIES.items():
        artifacts |= set(replay_content_dir.glob(pattern))
        if not any(artifact.match(pattern) for artifact in artifacts):
            not_checked.append(
                f"fit_provenance: no {family} fit ({pattern}) at {replay_content_dir}"
            )
    return sorted(artifacts), not_checked


def run_audit(
    *,
    root: Path,
    log_dir: Path,
    widloski_manifest: Path | None,
    swr_manifest: Path | None,
    replay_content_dir: Path,
    contfrag_dir: Path,
    repo: Path,
    exceptions: Mapping[str, Any],
    expected_settings: Mapping[str, Mapping[str, object]],
    check_marks: bool,
) -> tuple[dict[str, dict], list[str]]:
    """Run every check, reading only.

    Returns
    -------
    checks : dict of str to dict
        Each check's result, by name.
    not_checked : list of str
        What the audit did not check, so that a pass is not read as covering
        more than it did.
    """
    checks: dict[str, dict] = {}
    not_checked: list[str] = []
    complete: dict[str, set[str]] = {}
    incomplete: dict[str, dict[str, list[str]]] = {}
    for decoder in DECODERS:
        complete[decoder], incomplete[decoder] = find_decodes(root, decoder)
    decoded = complete["nonlocal"] | complete["contfrag"]
    widloski = None if widloski_manifest is None else _read_json(widloski_manifest)
    swr = None if swr_manifest is None else _read_json(swr_manifest)
    table_dirs = [replay_content_dir, contfrag_dir]

    # The epochs each artifact holds, and why it may lack one: the epoch-set
    # check's inputs, built first so that one_commit can tell an absence
    # epoch_sets explains from one it does not.
    sets: dict[str, set[str]] = {
        "nonlocal_decode": complete["nonlocal"],
        "contfrag_decode": complete["contfrag"],
    }
    reasons: dict[str, dict[str, str]] = {}
    known: set[str] = set()
    no_input: dict[str, str] = {}
    for run, manifest in (("widloski", widloski), ("swr", swr)):
        present, skipped = (
            (set(), {}) if manifest is None else run_manifest_epochs(manifest)
        )
        sets[run], reasons[run] = present, skipped
    if swr is not None:
        known = {
            epoch_id
            for epoch_id in swr.get("epochs", {})
            if is_canonical_epoch_id(epoch_id)
        }
        # an epoch the SWR runner skipped for lack of inputs, and with no
        # decode, can have nothing built on a decode either
        no_input = {
            epoch_id: f"no inputs, so no decode: {reason}"
            for epoch_id, reason in reasons["swr"].items()
            if epoch_id not in decoded
        }
    table_sets, table_failures, table_not_checked = table_epoch_sets(table_dirs)
    sets.update(table_sets)
    not_checked += table_not_checked
    recorded = exceptions.get("epochs", {})
    for artifact in sets:
        reasons[artifact] = {
            **no_input,
            **reasons.get(artifact, {}),
            **recorded.get(artifact, {}),
        }

    provenance = decode_provenance(log_dir, root, sorted(decoded))
    explained = {
        decode_id(epoch_id, decoder): reasons[f"{decoder}_decode"][epoch_id]
        for decoder in DECODERS
        for epoch_id in decoded - complete[decoder]
        if reasons[f"{decoder}_decode"].get(epoch_id)
    }
    checks["one_commit"] = check_one_commit(
        provenance, exceptions.get("commit", {}), explained=explained
    )
    decode_commit: str | None = checks["one_commit"]["details"]["commit"]

    for decoder in DECODERS:
        checks[f"decoder_settings_{decoder}"] = check_decoder_settings(
            {
                epoch_id: decode_files(epoch_id, root, decoder)["detector"]
                for epoch_id in complete[decoder]
            },
            expected_settings.get(decoder),
        )
        if not expected_settings.get(decoder):
            not_checked.append(
                f"decoder_settings_{decoder}: compared across epochs only, since "
                f"--expected-settings names no {decoder} settings"
            )

    grids: dict[str, dict] = {}
    times: dict[str, NDArray] = {}
    spans: dict[str, tuple[float, float]] = {}
    for epoch_id in sorted(complete["nonlocal"]):
        nonlocal_dot = _time_indexed(
            pd.read_pickle(decode_files(epoch_id, root, "nonlocal")["data_over_time"])
        )
        time = nonlocal_dot.index.to_numpy(dtype=np.float64)
        times[epoch_id] = time
        spans[epoch_id] = (float(time[0]), float(time[-1]))
        if epoch_id in complete["contfrag"]:
            contfrag_dot = pd.read_pickle(
                decode_files(epoch_id, root, "contfrag")["data_over_time"]
            )
            grids[epoch_id] = check_grids_match(nonlocal_dot, contfrag_dot)
    checks["grids_match"] = _per_epoch(grids, "no epoch has both decodes")
    checks["minimum_sample_counts"] = check_minimum_sample_counts(times)

    checks["widloski_hashes"] = (
        _result(["no Widloski manifest given (--widloski-manifest)"], {})
        if widloski is None
        else check_widloski_hashes(widloski, root)
    )
    checks["swr_spans"] = (
        _result(["no SWR manifest given (--swr-manifest)"], {})
        if swr is None
        else check_swr_spans(swr, spans)
    )

    # The Widloski runner reloads marks through the spike loader. The SWR
    # runner reads LFP and position rather than marks, so for its commits the
    # comparison is a conservative extra, kept as the refresh plan rules.
    loader_failures: list[str] = []
    loader_details: dict[str, dict] = {}
    run_commits = _run_commits(widloski, swr)
    for run in ("widloski", "swr"):
        if not run_commits.get(run):
            loader_failures.append(
                f"{run}: no run commit to compare (no manifest, or no epoch run)"
            )
    if decode_commit is None:
        loader_failures.append(
            "no single decode commit to compare with (see one_commit)"
        )
    else:
        for run, commits in sorted(run_commits.items()):
            for commit in sorted(commits):
                result = check_spike_loader_unchanged(repo, decode_commit, commit)
                loader_failures += [
                    f"{run}: {failure}" for failure in result["failures"]
                ]
                loader_details[f"{run} {commit}"] = result["details"]
    checks["spike_loader_unchanged"] = _result(loader_failures, loader_details)

    checks["tables_recorded"] = (
        _result(["no single decode commit to compare with (see one_commit)"], {})
        if decode_commit is None
        else check_tables_recorded(table_dirs, decode_commit, repo=repo)
    )

    event_table = resolved_table_path(replay_content_dir / "event_table.parquet")
    sidecars = [_read_json(path) for path in sorted(root.glob(PFC_SIDECAR_PATTERN))]
    checks["pfc_sidecars"] = (
        check_pfc_sidecars(sidecars, sha256_file(event_table))
        if event_table.exists()
        else _result([f"no canonical event table at {event_table}"], {})
    )

    named_fits, missing_families = root_fits(replay_content_dir)
    not_checked += missing_families
    checks["fit_provenance"] = check_fit_provenance(
        REPO, table_dirs, artifacts=named_fits
    )

    epoch_sets = check_epoch_sets(sets, reasons, known=known)
    epoch_sets["failures"] = table_failures + epoch_sets["failures"]
    epoch_sets["ok"] = not epoch_sets["failures"]
    for decoder in DECODERS:
        # which files each partly saved decode lacks
        epoch_sets["details"][f"{decoder}_decode"]["incomplete"] = incomplete[decoder]
    checks["epoch_sets"] = epoch_sets

    if check_marks:
        checks["encoding_marks"] = _per_epoch(
            {
                epoch_id: check_encoding_marks(epoch_id, root)
                for epoch_id in sorted(complete["nonlocal"] & complete["contfrag"])
            },
            "no epoch has both decodes",
        )
    else:
        not_checked.append(
            "encoding_marks: not run; --check-marks reloads every epoch's marks "
            "from the database"
        )
    return checks, not_checked


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _json_object(text: str) -> dict:
    """An ``argparse`` type: a JSON object."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"not JSON: {error}") from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser(
        "record-tables",
        help="Write a table folder's tables_manifest.json, and nothing else.",
    )
    record.add_argument("table_dir", type=Path)
    record.add_argument(
        "--commit", required=True, help="The commit that built the tables."
    )
    record.add_argument(
        "--inputs",
        required=True,
        type=_json_object,
        help='What they were built from, as a JSON object; "decode_commit" is required '
        "by the audit.",
    )

    run = commands.add_parser("audit", help="Run every check, reading only.")
    run.add_argument(
        "--output", type=Path, required=True, help="Where audit.json goes."
    )
    run.add_argument(
        "--check-marks",
        action="store_true",
        help="Also reload every epoch's marks from the database (slow).",
    )
    run.add_argument(
        "--root", type=Path, default=PROCESSED_DATA_DIR, help="The decodes."
    )
    run.add_argument(
        "--log-dir", type=Path, default=ROOT_DIR / "logs", help="The decoders' logs."
    )
    run.add_argument(
        "--widloski-manifest", type=Path, help="The Widloski run's manifest.json."
    )
    run.add_argument("--swr-manifest", type=Path, help="The SWR run's manifest.json.")
    run.add_argument(
        "--replay-content-dir", type=Path, default=PROCESSED_DATA_DIR / "replay_content"
    )
    run.add_argument(
        "--contfrag-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "contfrag_replay_content",
    )
    run.add_argument(
        "--repo",
        type=Path,
        default=REPO,
        help="The git repository the recorded commits are looked up in.",
    )
    run.add_argument(
        "--exceptions",
        type=Path,
        help='A JSON file: {"commit": {decode_id: reason}, "epochs": {artifact: '
        "{epoch_id: reason}}}, the decodes the one-commit check sets aside and "
        "the absences the epoch-set check accepts, each with its reason. "
        "Artifacts are named as audit.json's epoch_sets details name them: "
        "nonlocal_decode, contfrag_decode, widloski, swr, and each table as "
        "<folder>/<file>, such as replay_content/event_table.parquet.",
    )
    run.add_argument(
        "--expected-settings",
        type=Path,
        help='A JSON file: {"nonlocal": {setting: value}, "contfrag": {...}}, the '
        "discrete-transition settings every detector must carry. It overrides the "
        "default, expected_decoder_settings(), which the audit builds from the code.",
    )
    args = parser.parse_args(argv)

    if args.command == "record-tables":
        path = record_tables(args.table_dir, args.commit, args.inputs)
        n_files = len(json.loads(path.read_text())["files"])
        print(f"recorded {n_files} tables -> {path}")
        return 0

    exceptions = {} if args.exceptions is None else _read_json(args.exceptions)
    if set(exceptions) - {"commit", "epochs"}:
        parser.error(
            f"--exceptions takes only 'commit' and 'epochs', not {sorted(exceptions)}"
        )
    if args.expected_settings is None:
        expected_settings = expected_decoder_settings()
        expected_source = "expected_decoder_settings()"
    else:
        expected_settings = _read_json(args.expected_settings)
        expected_source = str(args.expected_settings)
    if set(expected_settings) - set(DECODERS):
        parser.error(
            f"--expected-settings takes only {list(DECODERS)}, not "
            f"{sorted(expected_settings)}"
        )
    checks, not_checked = run_audit(
        root=args.root,
        log_dir=args.log_dir,
        widloski_manifest=args.widloski_manifest,
        swr_manifest=args.swr_manifest,
        replay_content_dir=args.replay_content_dir,
        contfrag_dir=args.contfrag_dir,
        repo=args.repo,
        exceptions=exceptions,
        expected_settings=expected_settings,
        check_marks=args.check_marks,
    )
    failed = [name for name, check in checks.items() if not check["ok"]]
    head = _git(args.repo, "rev-parse", "HEAD")
    report = {
        "produced_by": f"{PRODUCED_BY} audit",
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "repo_head": head.stdout.strip() if head.returncode == 0 else None,
        "ok": not failed,
        "failed_checks": failed,
        # a pass covers the checks above and nothing listed here
        "not_checked": not_checked,
        # what the detectors' settings were held to, and where it came from
        "expected_settings": {"source": expected_source, "values": expected_settings},
        "arguments": {
            name: (str(value) if isinstance(value, Path) else value)
            for name, value in vars(args).items()
        },
        "checks": checks,
    }
    text = json.dumps(report, indent=2, default=_json_default) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(args.output, lambda temporary: temporary.write_text(text))
    for name, check in checks.items():
        first = f": {check['failures'][0]}" if check["failures"] else ""
        more = (
            f" (+{len(check['failures']) - 1} more)"
            if len(check["failures"]) > 1
            else ""
        )
        print(f"{'ok  ' if check['ok'] else 'FAIL'}  {name}{first}{more}")
    for gap in not_checked:
        print(f"not checked  {gap}")
    print(f"audit -> {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
