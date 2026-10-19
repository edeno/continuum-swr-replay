#!/usr/bin/env python
"""Compare the replay-detection inventories, one epoch at a time.

Nine inventories of each analysed epoch are read through their adapters
(``inventory_comparison.adapters``) and compared on one substrate, with
per-detector denominators. Design:
``docs/design/replay_detection_comparison/03-inventory-comparison.md``.

The roster (:data:`ROSTER`):

- the SWR detectors ``kay``, ``karlsson``, ``yu`` and ``zugaro`` (the SWR
  source run, ``swr_source.SWR_INVENTORY_RUN_ID``) and ``carey``
  (``--carey-run-id``), the five SWR definitions;
- ``mua`` and ``state_space``, from the canonical decode frame;
- ``memoryless`` (cohort ``published``) and ``memoryless_criteria`` (cohort
  ``replay_criteria``), from the memoryless run (``--memoryless-run-id``).

Per epoch:

1. **Decode.** The canonical decode's ``data_over_time`` pickle is read once:
   its drops (``swr_source.artifact_event_drops``) from the frame as saved,
   and the frame through the SWR source (``swr_source.with_swr_source``). Its
   sha256 is recorded, and its first 12 characters name the decode in the
   ``mua`` and ``state_space`` tables.
2. **Inventories.** Each adapter returns its inventory, its observation table
   and the events the analysed-time rule set aside. A run with no usable
   record for the epoch reads unavailable, never empty.
3. **Speed.** One trace, the frame's ``head_speed`` on its grid, with one
   threshold (``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``) and one largest
   step (:data:`MAX_GAP_S`), gives the still intervals, where the speed is
   known (``exclusion_by_reason``'s ``speed_known``) and the samples that were
   not still (the recount's ``not_still_s``). The grid must step by
   :data:`SAMPLE_INTERVAL_S`. Every detector's epoch is the grid's first to
   last time, which must be the position span the SWR source run cut its band
   to (its record's ``position_span_s``); otherwise the epoch fails, with that
   reason, and writes nothing.
4. **Coverage.** ``coverage_from_observations``; each set-aside event's span
   leaves its own detector's coverage (``set_aside_events``); the
   speed-eligible events (``combine_inventories``) that bridge their own
   detector's unobserved time are set aside too (``remove_bridging_events``,
   reason ``bridging``).
5. **Comparison.** Every overlap pair, flagged; directional agreement and time
   coverage over the nine; the state-space fragment recount at every
   ``FRAGMENT_GAPS_S``, each followed by its pairs and directional agreement;
   the three cells (:data:`CELLS`); the overlap components, the exactly-one
   cell and no-detector time over the eight that leave ``memoryless_criteria``
   out (every published event is also a criteria event); the SWR label
   matrix of the three sequence cohorts under the five SWR definitions; and
   excluded time by reason for every pair, for ``all`` nine and for each matrix
   population (:data:`COMPARISONS`). Any second outside a comparison's shared exposure
   with no reason fails the epoch.

Each epoch writes, under
``Processed-Data/replay_detection_comparison/inventory/<run_id>/``, one
``{epoch_id}_<table>.parquet`` per table of :data:`TABLE_DTYPES`, as one set:
every table is made before any is written, and a failed epoch keeps none.
``manifest.json`` beside them holds the run's settings (the input runs, the
SWR source, the roster and its cohorts, the comparisons, the recount and the
speed rule, and why MUA and state-space native counts equal their eligible
counts) and one record per epoch: its status, the decode's sha256, the counts,
the unavailable detectors, each output's path and sha256, its wall clock and
the process's peak RSS so far (``process_peak_rss_gib``, cumulative over the
epochs before it). Only this process writes it, atomically, after
each epoch. A rerun under the same run id skips the epochs already ``ok``
unless ``--overwrite`` is given, and refuses settings other than the ones the
manifest was written with. The epochs are the epoch register's analysed ones,
narrowed by ``--animal``, ``--date`` and ``--epoch``, or the one
``--epoch-id``, run one at a time in this process.

Usage:
    python scripts/qa/run_inventory_comparison.py --carey-run-id <run id> \\
        --memoryless-run-id d0964cc76ed2-seed20260922 \\
        --epoch-id chimi20200225_02_r1_HPC
"""

import argparse
import json
import os
import resource
import sys
import time as clock
import traceback
from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# CPU only: the package imports JAX, which this comparison never needs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continuum_swr_replay.analysis.constants import (  # noqa: E402
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus import swr_source  # noqa: E402
from continuum_swr_replay.analysis.inventory_comparison.adapters import (  # noqa: E402
    CAREY,
    MUA,
    NATIVE_IS_ELIGIBLE,
    SWR_DETECTORS,
    InventoryTables,
    carey_inventory,
    memoryless_inventory,
    mua_inventory,
    state_space_inventory,
    swr_inventory,
)
from continuum_swr_replay.analysis.inventory_comparison.availability import (  # noqa: E402
    EXCLUSION_DTYPES,
    GAP_FACTOR,
    OBSERVATION_DTYPES,
    coverage_from_observations,
    exclusion_by_reason,
    not_still_times,
    sample_runs,
    still_intervals,
)
from continuum_swr_replay.analysis.inventory_comparison.cells import (  # noqa: E402
    CELL_DTYPES,
    NO_DETECTOR_DTYPES,
    membership_against,
    no_detector_time,
)
from continuum_swr_replay.analysis.inventory_comparison.components import (  # noqa: E402
    COMPONENT_DTYPES,
    overlap_components,
)
from continuum_swr_replay.analysis.inventory_comparison.intervals import (  # noqa: E402
    contains,
    total_duration,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (  # noqa: E402
    DIRECTIONAL_DTYPES,
    PAIR_DTYPES,
    SET_ASIDE_DTYPES,
    TIME_COVERAGE_DTYPES,
    directional_agreement,
    event_matches,
    flag_pairs,
    overlap_pairs,
    remove_bridging_events,
    set_aside_events,
    time_coverage,
)
from continuum_swr_replay.analysis.inventory_comparison.recount import (  # noqa: E402
    FRAGMENT_GAPS_S,
    merge_fragments,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (  # noqa: E402
    INVENTORY_DTYPES,
    MEMORYLESS,
    STATE_SPACE,
    combine_inventories,
)
from continuum_swr_replay.analysis.inventory_comparison.swr_matrix import (  # noqa: E402
    DISAGREEMENT_DTYPES,
    LABEL_DTYPES,
    MATRIX_DTYPES,
    PAIRED_LABEL_DTYPES,
    definition_disagreement,
    paired_labels,
    swr_label_matrix,
    swr_labels,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_epoch_ids,
)
from continuum_swr_replay.decoding_common import (  # noqa: E402
    parse_run_id,
    provenance_record,
    replace_outputs,
    write_atomically,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY  # noqa: E402
from continuum_swr_replay.table_io import sha256_file  # noqa: E402
from scripts.decoding.run_swr_detection import (  # noqa: E402
    json_ready,
    run_lock,
)

COMPARISON_ROOT = PROCESSED_DATA_DIR / "replay_detection_comparison"
OUTPUT_ROOT = COMPARISON_ROOT / "inventory"
CAREY_ROOT = COMPARISON_ROOT / "carey"
MEMORYLESS_ROOT = COMPARISON_ROOT / "memoryless"
DECODE_ROOT = PROCESSED_DATA_DIR
"""Where the canonical HPC decodes are."""
PRODUCED_BY = "scripts/qa/run_inventory_comparison.py"
DESIGN_SPEC = "docs/design/replay_detection_comparison/03-inventory-comparison.md"

MEMORYLESS_CRITERIA = "memoryless_criteria"
ROSTER: Mapping[str, str | None] = MappingProxyType(
    {
        **dict.fromkeys(SWR_DETECTORS),
        CAREY: None,
        MUA: None,
        STATE_SPACE: None,
        MEMORYLESS: "published",
        MEMORYLESS_CRITERIA: "replay_criteria",
    }
)
"""Every inventory compared, with its memoryless cohort (``None`` for the
others)."""
DETECTORS = tuple(ROSTER)
SWR_DEFINITIONS = (*SWR_DETECTORS, CAREY)
SEQUENCE_COHORTS = (STATE_SPACE, MEMORYLESS, MEMORYLESS_CRITERIA)
"""The cohorts the SWR label matrix labels, and the sequences cell's
sources."""
CELL_DETECTORS = tuple(d for d in DETECTORS if d != MEMORYLESS_CRITERIA)
"""The eight of the exactly-one cell, no-detector time and the overlap
components: every published memoryless event is also a criteria event, so with
both, no memoryless event could be found by exactly one detector, and every
component holding one would count one detector twice."""
DECODERS = (STATE_SPACE, MEMORYLESS)
TRIGGERED = (*SWR_DEFINITIONS, MUA)
"""The LFP- and rate-triggered inventories."""
CELLS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = MappingProxyType(
    {
        "exactly_one_detector": (CELL_DETECTORS, CELL_DETECTORS),
        "swr_matched_by_no_decoder": (SWR_DEFINITIONS, DECODERS),
        "sequence_matched_by_no_trigger": (SEQUENCE_COHORTS, TRIGGERED),
    }
)
"""Each cell's source detectors, and the detectors they are tested against."""
COMPARISONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {f"{a}+{b}": (a, b) for a, b in combinations(sorted(DETECTORS), 2)}
    | {"all": DETECTORS}
    | {f"swr_matrix_{c}": (c, *SWR_DEFINITIONS) for c in SEQUENCE_COHORTS}
)
"""Each comparison whose excluded time is split by reason: every pair, all
nine, and each SWR-matrix population."""

SAMPLE_INTERVAL_S = 0.002
"""The decode grid's step, on which the recount counts its gaps."""
MAX_GAP_S = GAP_FACTOR * SAMPLE_INTERVAL_S
"""A step longer than this breaks the speed trace's runs."""
BRIDGING = "bridging"
"""The set-aside reason of an event bridging its own detector's unobserved
time."""
SPEED_TRACE = "head_speed of the canonical decode frame, on its grid"
BOUNDS_TOLERANCE_S = SAMPLE_INTERVAL_S / 2
"""How far the decode grid's first or last time may lie from the SWR run's
position span: less than half a sample, so both name the same samples."""

COUNT_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector": object,
        "run_id": object,
        "n_native": np.int64,
        "n_set_aside": np.int64,
        "n_moving": np.int64,
        "n_eligible": np.int64,
        "exposure_s": np.float64,
        "n_in_exposure": np.int64,
        "rate_per_s": np.float64,
        "native_note": object,
    }
)
"""Per detector: native = eligible + moving + set aside (dropped by the rule or
bridging); the rate is the eligible events wholly inside the detector's own
exposure per exposure second, NaN with no exposure. ``native_note`` says why a
native count equals its eligible count (``NATIVE_IS_ELIGIBLE``)."""

TABLE_DTYPES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "inventory": {**INVENTORY_DTYPES, "component_id": "Int64"},
        "set_aside": SET_ASIDE_DTYPES,
        "observations": OBSERVATION_DTYPES,
        "exclusion_by_reason": EXCLUSION_DTYPES,
        "counts": COUNT_DTYPES,
        "pairs": {
            **PAIR_DTYPES,
            "a_within_shared_exposure": np.bool_,
            "b_within_shared_exposure": np.bool_,
        },
        "directional": DIRECTIONAL_DTYPES,
        "time_coverage": TIME_COVERAGE_DTYPES,
        "components": COMPONENT_DTYPES,
        "recount_directional": {**DIRECTIONAL_DTYPES, "gap_s": np.float64},
        "cells": {"cell": object, **CELL_DTYPES},
        "no_detector_time": NO_DETECTOR_DTYPES,
        "swr_labels": LABEL_DTYPES,
        "swr_matrix": MATRIX_DTYPES,
        "paired_labels": PAIRED_LABEL_DTYPES,
        "definition_disagreement": DISAGREEMENT_DTYPES,
    }
)
"""Every table an epoch writes, with its columns and dtypes. ``inventory``
holds the compared events and the moving ones; ``component_id`` is missing for
the moving ones and for ``memoryless_criteria``'s, which the components leave
out (:data:`CELL_DETECTORS`); ``set_aside`` the events the rule dropped and the
bridging ones."""


def decode_path(identifier: str) -> Path:
    """The canonical decode's ``data_over_time`` pickle."""
    threshold = round(100 * DEFAULT_NON_LOCAL_THRESHOLD)
    path: Path = DECODE_ROOT / f"{identifier}_{threshold}_data_over_time.pkl"
    return path


def _typed(
    frames: Iterable[pd.DataFrame], dtypes: Mapping[str, object]
) -> pd.DataFrame:
    """The frames in one table with the columns and dtypes given."""
    rows = [frame for frame in frames if len(frame)]
    table = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            {name: pd.Series(dtype=dtype) for name, dtype in dtypes.items()}
        )
    )
    return table[list(dtypes)].astype(dict(dtypes)).reset_index(drop=True)


def _speed_trace(dot: pd.DataFrame) -> tuple[NDArray, NDArray]:
    """The frame's grid and its ``head_speed``.

    Raises
    ------
    ValueError
        The grid does not step by ``SAMPLE_INTERVAL_S``.
    """
    time = dot["time"] if "time" in dot.columns else dot.index
    time = np.asarray(time, dtype=np.float64)
    step = float(np.median(np.diff(time)))
    if not np.isclose(step, SAMPLE_INTERVAL_S, rtol=0.0, atol=1e-5):
        raise ValueError(
            f"The decode grid steps by {step} s, not {SAMPLE_INTERVAL_S} s, the "
            "step the recount counts gaps on."
        )
    return time, dot["head_speed"].to_numpy(dtype=np.float64)


def epoch_inventories(
    dot: pd.DataFrame,
    drops: pd.DataFrame,
    *,
    decode_run_id: str,
    carey_dir: Path,
    memoryless_dir: Path,
) -> dict[str, InventoryTables]:
    """Every inventory of the roster for the frame's epoch, by detector."""
    swr_dir = swr_source.swr_inventory_dir()
    tables = {
        detector: swr_inventory(swr_dir, detector, dot=dot, drops=drops)
        for detector in SWR_DETECTORS
    }
    tables[CAREY] = carey_inventory(carey_dir, dot=dot, drops=drops)
    tables[MUA] = mua_inventory(dot, drops=drops, run_id=decode_run_id)
    tables[STATE_SPACE] = state_space_inventory(dot, drops=drops, run_id=decode_run_id)
    for detector in (MEMORYLESS, MEMORYLESS_CRITERIA):
        tables[detector] = memoryless_inventory(
            memoryless_dir,
            cohort=str(ROSTER[detector]),
            detector=detector,
            dot=dot,
            drops=drops,
        )
    return tables


def _bridging_set_aside(bridging: pd.DataFrame) -> pd.DataFrame:
    """The bridging events in the set-aside layout, their spans as their own."""
    return pd.DataFrame(
        {
            **{c: bridging[c] for c in (*SESSION_COLUMNS_DAY, "detector", "event_key")},
            "interval_start_s": bridging["event_start_s"],
            "interval_end_s": bridging["event_end_s"],
            "reason": BRIDGING,
        }
    ).astype(dict(SET_ASIDE_DTYPES))


def _counts(
    inventories: Mapping[str, InventoryTables],
    kept: pd.DataFrame,
    moving: pd.DataFrame,
    bridging: pd.DataFrame,
    coverage: Mapping[tuple, Any],
    key: tuple,
) -> pd.DataFrame:
    rows = []
    for detector, tables in inventories.items():
        events = kept[kept.detector == detector]
        exposure = coverage[(*key, detector)].exposure
        exposure_s = total_duration(exposure)
        n_inside = int(
            contains(
                exposure, events.event_start_s.to_numpy(), events.event_end_s.to_numpy()
            ).sum()
        )
        rows.append(
            dict(zip(SESSION_COLUMNS_DAY, key, strict=True))
            | {
                "detector": detector,
                "run_id": tables.observations.run_id.iloc[0],
                "n_native": len(tables.inventory) + len(tables.set_aside),
                "n_set_aside": len(tables.set_aside)
                + int((bridging.detector == detector).sum()),
                "n_moving": int((moving.detector == detector).sum()),
                "n_eligible": len(events),
                "exposure_s": exposure_s,
                "n_in_exposure": n_inside,
                "rate_per_s": n_inside / exposure_s if exposure_s > 0 else np.nan,
                "native_note": NATIVE_IS_ELIGIBLE.get(detector, ""),
            }
        )
    counts = pd.DataFrame(rows)
    if not (
        counts.n_native == counts.n_eligible + counts.n_moving + counts.n_set_aside
    ).all():
        raise RuntimeError("A native count is not eligible + moving + set aside.")
    return counts


def compare_epoch(
    dot: pd.DataFrame,
    drops: pd.DataFrame,
    *,
    decode_run_id: str,
    carey_dir: Path,
    memoryless_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Every table of one epoch, by name, with :data:`TABLE_DTYPES`.

    Parameters
    ----------
    dot : DataFrame
        The epoch's decode frame, read through the SWR source.
    drops : DataFrame
        ``swr_source.artifact_event_drops`` of the frame as saved.
    decode_run_id : str
        Names the decode in the ``mua`` and ``state_space`` tables.
    carey_dir, memoryless_dir : Path
        The Carey and memoryless run folders.

    Raises
    ------
    ValueError
        The grid is not the recount's, or ``exclusion_by_reason`` finds time
        with no reason: the coverage was not built from these inputs.
    """
    time, speed = _speed_trace(dot)
    inventories = epoch_inventories(
        dot,
        drops,
        decode_run_id=decode_run_id,
        carey_dir=carey_dir,
        memoryless_dir=memoryless_dir,
    )
    observations = _typed(
        (tables.observations for tables in inventories.values()), OBSERVATION_DTYPES
    )
    dropped = _typed(
        (tables.set_aside for tables in inventories.values()), SET_ASIDE_DTYPES
    )
    animal, day, epoch = observations[list(SESSION_COLUMNS_DAY)].iloc[0]
    key = (str(animal), int(day), str(epoch))
    threshold = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    still = {key: still_intervals(time, speed, threshold, MAX_GAP_S)}
    speed_known = {key: sample_runs(time, np.isfinite(speed), MAX_GAP_S)}
    not_still = {key: not_still_times(time, speed, threshold)}

    coverage = set_aside_events(
        coverage_from_observations(observations, still), dropped
    )
    eligible, moving = combine_inventories(
        tables.inventory for tables in inventories.values()
    )
    kept, coverage, bridging = remove_bridging_events(eligible, coverage)
    set_aside = _typed([dropped, _bridging_set_aside(bridging)], SET_ASIDE_DTYPES)

    pairs = overlap_pairs(kept)
    matches = event_matches(kept, pairs, coverage, detectors=DETECTORS)
    # the components leave the criteria twin out, its events and their pairs
    twin = kept.detector == MEMORYLESS_CRITERIA
    twin_pair = (pairs.detector_a == MEMORYLESS_CRITERIA) | (
        pairs.detector_b == MEMORYLESS_CRITERIA
    )
    labelled, components = overlap_components(kept[~twin], pairs[~twin_pair])
    recount = []
    for gap_s in FRAGMENT_GAPS_S:
        merged = merge_fragments(
            kept,
            coverage,
            detector=STATE_SPACE,
            gap_s=gap_s,
            sample_interval_s=SAMPLE_INTERVAL_S,
            not_still_s=not_still,
        )
        recount.append(
            directional_agreement(
                merged, overlap_pairs(merged), coverage, detectors=DETECTORS
            ).assign(gap_s=gap_s)
        )
    cells = [
        membership_against(kept, matches, sources=sources, against=against).assign(
            cell=name
        )
        for name, (sources, against) in CELLS.items()
    ]
    labels = swr_labels(
        kept,
        matches,
        coverage,
        cohorts=SEQUENCE_COHORTS,
        swr_definitions=SWR_DEFINITIONS,
    )
    tables = {
        "inventory": [
            labelled,
            kept[twin].assign(component_id=pd.NA),
            moving.assign(component_id=pd.NA),
        ],
        "set_aside": [set_aside],
        "observations": [observations],
        "exclusion_by_reason": [
            exclusion_by_reason(
                observations,
                coverage,
                still,
                set_aside,
                comparisons=COMPARISONS,
                speed_known=speed_known,
            )
        ],
        "counts": [_counts(inventories, kept, moving, bridging, coverage, key)],
        "pairs": [flag_pairs(pairs, matches)],
        "directional": [
            directional_agreement(kept, pairs, coverage, detectors=DETECTORS)
        ],
        "time_coverage": [time_coverage(kept, coverage, detectors=DETECTORS)],
        "components": [components],
        "recount_directional": recount,
        "cells": cells,
        "no_detector_time": [
            no_detector_time(kept, coverage, detectors=CELL_DETECTORS)
        ],
        "swr_labels": [labels],
        "swr_matrix": [
            swr_label_matrix(
                labels,
                coverage,
                cohorts=SEQUENCE_COHORTS,
                swr_definitions=SWR_DEFINITIONS,
            )
        ],
        "paired_labels": [paired_labels(pairs, labels, cohorts=SEQUENCE_COHORTS)],
        "definition_disagreement": [definition_disagreement(labels)],
    }
    out = {name: _typed(tables[name], dtypes) for name, dtypes in TABLE_DTYPES.items()}
    out["inventory"] = out["inventory"].sort_values(
        ["detector", "event_start_s"], kind="stable", ignore_index=True
    )
    return out


class _EpochRefused(Exception):
    """The epoch cannot be compared; the message says why."""


def bounds_mismatch(identifier: str, bounds: tuple[float, float]) -> str | None:
    """Why the decode grid's first and last time are not the epoch's, or None.

    They must be the position span the SWR source run cut its band to (its
    record's ``position_span_s``), within :data:`BOUNDS_TOLERANCE_S`: every
    detector is compared over that one span.
    """
    span = swr_source.ripple_power_inputs(identifier)["position_span_s"]
    if np.allclose(bounds, span, rtol=0.0, atol=BOUNDS_TOLERANCE_S):
        return None
    return (
        f"The decode grid spans {bounds[0]!r} to {bounds[1]!r} s, not the position "
        f"span {span[0]!r} to {span[1]!r} s SWR run "
        f"{swr_source.SWR_INVENTORY_RUN_ID} cut its band to; the detectors would "
        "not share one epoch."
    )


def output_path(identifier: str, output_dir: Path, name: str) -> Path:
    return output_dir / f"{identifier}_{name}.parquet"


def output_paths(identifier: str, output_dir: Path) -> dict[str, Path]:
    """Every table an epoch can write, by name."""
    return {name: output_path(identifier, output_dir, name) for name in TABLE_DTYPES}


def stamped_tables(
    tables: Mapping[str, pd.DataFrame], stamp: Mapping[str, str]
) -> Mapping[str, pd.DataFrame]:
    """Stamp each table (in place, and return them) with the SWR source and `stamp`.

    The SWR source's stamp (``swr_source.stamp_swr_source``) goes first, then
    `stamp`'s keys are merged into each table's ``attrs``.
    """
    for frame in tables.values():
        swr_source.stamp_swr_source(frame)
        frame.attrs |= stamp
    return tables


def run_epoch(
    identifier: str,
    *,
    output_dir: Path,
    carey_dir: Path,
    memoryless_dir: Path,
    provenance: Mapping[str, Any],
) -> dict:
    """Compare one epoch's inventories and write its tables.

    Returns
    -------
    dict
        The epoch's manifest record, in strict JSON types: ``status`` ``"ok"``,
        or ``"failed"`` with a ``reason`` (the decode grid does not span the
        position span the SWR run cut to) or the ``error`` and its
        ``traceback``, its outputs removed.
    """
    started = clock.perf_counter()
    stamp = {
        "produced_by": PRODUCED_BY,
        "run_id": output_dir.name,
        "epoch_id": identifier,
        "commit": str(provenance["commit"]),
    }
    record: dict[str, Any]
    try:
        path = decode_path(identifier)
        decode_sha256 = sha256_file(path)
        saved = pd.read_pickle(path)
        drops = swr_source.artifact_event_drops(saved)
        dot = swr_source.with_swr_source(saved)
        del saved
        time, _ = _speed_trace(dot)
        bounds = (float(time[0]), float(time[-1]))
        reason = bounds_mismatch(identifier, bounds)
        if reason is not None:
            replace_outputs(output_paths(identifier, output_dir), {})
            raise _EpochRefused(reason)
        tables = compare_epoch(
            dot,
            drops,
            decode_run_id=decode_sha256[:12],
            carey_dir=carey_dir,
            memoryless_dir=memoryless_dir,
        )
        observations = tables["observations"]
        observed = observations.groupby("detector").available.any()
        counts = tables["counts"].set_index("detector")
        record = {
            "status": "ok",
            "decode": {"path": str(path), "sha256": decode_sha256},
            "epoch_bounds_s": list(bounds),
            "counts": {
                detector: counts.loc[
                    detector, ["n_native", "n_eligible", "n_moving", "n_set_aside"]
                ].to_dict()
                for detector in DETECTORS
            },
            "unavailable": {
                detector: observations.loc[
                    observations.detector == detector, "reason"
                ].iloc[0]
                for detector in DETECTORS
                if not observed[detector]
            },
            "outputs": replace_outputs(
                output_paths(identifier, output_dir), stamped_tables(tables, stamp)
            ),
        }
    except _EpochRefused as refused:
        record = {
            "status": "failed",
            "reason": str(refused),
            "decode": {"path": str(path), "sha256": decode_sha256},
            "epoch_bounds_s": list(bounds),
        }
    except Exception as error:  # noqa: BLE001 - recorded; the run goes on
        replace_outputs(output_paths(identifier, output_dir), {})
        record = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    record["provenance"] = dict(provenance)
    record["wall_clock_s"] = clock.perf_counter() - started
    # the process's peak resident set over its life so far, every earlier
    # epoch included: not this epoch's own (KiB on Linux)
    record["process_peak_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    )
    ready: dict = json_ready(record)
    return ready


def summary(record: dict) -> str:
    """One epoch's outcome, as the console reports it."""
    if record["status"] != "ok":
        return f"{record['status']}: {record.get('reason') or record['error']}"
    eligible = ", ".join(
        f"{detector} {counts['n_eligible']}"
        for detector, counts in record["counts"].items()
    )
    unavailable = ", ".join(record["unavailable"]) or "none"
    return (
        f"ok: eligible {eligible}; unavailable: {unavailable}; "
        f"{record['wall_clock_s']:.0f} s, process peak RSS "
        f"{record['process_peak_rss_gib']:.1f} GiB"
    )


def run_settings(carey_run_id: str, memoryless_run_id: str) -> dict:
    """What every epoch of a run shares; a rerun under its run id must match."""
    settings: dict = json_ready(
        {
            "input_runs": {
                "swr": swr_source.SWR_INVENTORY_RUN_ID,
                "carey": carey_run_id,
                "memoryless": memoryless_run_id,
                "artifacts": swr_source.ARTIFACT_INTERVALS_RUN_ID,
            },
            "swr_source": swr_source.swr_source_name(),
            "roster": ROSTER,
            "swr_definitions": SWR_DEFINITIONS,
            "swr_cohorts": SEQUENCE_COHORTS,
            "cells": {
                name: {"sources": sources, "against": against}
                for name, (sources, against) in CELLS.items()
            },
            "no_detector_time": CELL_DETECTORS,
            "comparisons": COMPARISONS,
            "recount": {
                "detector": STATE_SPACE,
                "gaps_s": FRAGMENT_GAPS_S,
                "sample_interval_s": SAMPLE_INTERVAL_S,
            },
            "speed": {
                "trace": SPEED_TRACE,
                "threshold_cm_per_s": DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
                "max_gap_s": MAX_GAP_S,
            },
            "epoch_bounds": {
                "rule": "the decode grid's first and last time, which must be "
                "the SWR run's position_span_s",
                "tolerance_s": BOUNDS_TOLERANCE_S,
            },
            "peak_rss": "process_peak_rss_gib: the process's peak so far, "
            "cumulative over the epochs run before",
            "decode": decode_path("{epoch_id}").name + ", read through the SWR source",
            "native_is_eligible": NATIVE_IS_ELIGIBLE,
        }
    )
    return settings


def _epoch_parts(identifier: str) -> tuple[str, int, str]:
    """``(animal, date, epoch)`` of ``{animal}{date}_{epoch}_HPC``."""
    stem, rest = identifier.split("_", 1)
    return stem[:-8], int(stem[-8:]), rest.removesuffix("_HPC")


def select_epochs(args: argparse.Namespace) -> list[str]:
    """The analysed epochs the arguments name, as epoch ids."""
    analysed = analysed_epoch_ids()
    if args.epoch_id is not None:
        if args.epoch_id not in analysed:
            sys.exit(f"{args.epoch_id} is not an analysed epoch of the epoch register.")
        return [args.epoch_id]
    selected = []
    for identifier in analysed:
        animal, date, epoch = _epoch_parts(identifier)
        if (
            (args.animal is None or animal == args.animal)
            and (args.date is None or date == args.date)
            and (args.epoch is None or epoch == args.epoch)
        ):
            selected.append(identifier)
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--carey-run-id",
        required=True,
        type=parse_run_id,
        help="The Carey run folder's name.",
    )
    parser.add_argument(
        "--memoryless-run-id",
        required=True,
        type=parse_run_id,
        help="The memoryless run folder's name.",
    )
    parser.add_argument("--animal", help="Only this animal's epochs, such as chimi.")
    parser.add_argument(
        "--date", type=int, help="Only epochs recorded on this date, such as 20200225."
    )
    parser.add_argument("--epoch", help="Only this epoch, such as 02_r1.")
    parser.add_argument(
        "--epoch-id",
        help="Only this epoch, such as chimi20200225_02_r1_HPC; not with --animal, "
        "--date or --epoch.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo epochs already written."
    )
    parser.add_argument(
        "--run-id",
        type=parse_run_id,
        help="The run's folder: letters, digits, '.', '_' and '-'. Defaults to the "
        "first 12 characters of git rev-parse HEAD.",
    )
    args = parser.parse_args(argv)
    if args.epoch_id is not None and any(
        value is not None for value in (args.animal, args.date, args.epoch)
    ):
        parser.error("--epoch-id names one epoch; drop --animal, --date and --epoch.")

    provenance = provenance_record(ROOT_DIR)
    run_id = provenance["commit"][:12] if args.run_id is None else args.run_id
    if provenance["dirty"]:
        print(
            f"warning: uncommitted changes; run {run_id} cannot be tied to a commit",
            flush=True,
        )
    output_dir = OUTPUT_ROOT / run_id
    with run_lock(output_dir):
        run(args, provenance, output_dir)


def run(args: argparse.Namespace, provenance: dict, output_dir: Path) -> None:
    """``main``'s run, under the run folder's lock."""
    carey_dir = CAREY_ROOT / args.carey_run_id
    memoryless_dir = MEMORYLESS_ROOT / args.memoryless_run_id
    for option, run_dir in (
        ("--carey-run-id", carey_dir),
        ("--memoryless-run-id", memoryless_dir),
    ):
        # a misspelt run would read unavailable in every epoch
        if not (run_dir / "manifest.json").exists():
            sys.exit(f"{option}: no run manifest at {run_dir}.")
    settings = run_settings(args.carey_run_id, args.memoryless_run_id)
    epochs = select_epochs(args)
    if not epochs:
        sys.exit("No analysed epochs match --animal, --date and --epoch.")

    manifest_path = output_dir / "manifest.json"
    records: dict[str, dict] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["settings"] != settings:
            sys.exit(
                f"{manifest_path} was written with other settings; a run must "
                "not mix them. Start a new --run-id."
            )
        records = previous["epochs"]

    def write_manifest() -> None:
        text = json.dumps(
            {
                "produced_by": PRODUCED_BY,
                "run_id": output_dir.name,
                "design_spec": DESIGN_SPEC,
                "settings": settings,
                "epochs": records,
            },
            indent=2,
            allow_nan=False,
        )
        write_atomically(manifest_path, lambda path: path.write_text(text))

    to_run = []
    for identifier in epochs:
        if records.get(identifier, {}).get("status") == "ok" and not args.overwrite:
            print(f"{identifier}: already written", flush=True)
        else:
            # the record goes first, so a run stopped part way cannot leave new
            # files under an old record
            records.pop(identifier, None)
            to_run.append(identifier)
    write_manifest()
    print(f"{len(to_run)} epochs to run, one at a time", flush=True)

    failed = []
    for number, identifier in enumerate(to_run, start=1):
        record = run_epoch(
            identifier,
            output_dir=output_dir,
            carey_dir=carey_dir,
            memoryless_dir=memoryless_dir,
            provenance=provenance,
        )
        records[identifier] = record
        print(f"[{number}/{len(to_run)}] {identifier}: {summary(record)}", flush=True)
        if record["status"] != "ok":
            failed.append(identifier)
        write_manifest()
    print(f"manifest -> {manifest_path}", flush=True)
    if failed:
        sys.exit(
            f"{len(failed)} epochs failed: {', '.join(failed)}; see {manifest_path}"
        )


if __name__ == "__main__":
    main()
