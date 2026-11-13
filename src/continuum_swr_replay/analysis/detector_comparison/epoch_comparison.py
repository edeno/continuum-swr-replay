"""One epoch's inventories compared on one substrate.

:func:`compare_epoch` makes every table an epoch of the inventory comparison run
writes (``inventory_run.TABLE_DTYPES``), from the epoch's decode frame read
through the SWR source: each detector's inventory through its adapter
(:func:`epoch_inventories`), the still intervals and coverage from the frame's
speed, the bridging events set aside (:data:`BRIDGING`), and then the overlap
pairs, matches, components, the state-space fragment recount, the cells, the
SWR label matrix and the counts (:func:`detector_counts`).
``scripts/detection/run_inventory_comparison.py`` runs it per epoch and writes
the tables.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.detector_comparison.adapters import (
    CAREY,
    MUA,
    NATIVE_IS_ELIGIBLE,
    SWR_DETECTORS,
    InventoryTables,
    carey_inventory,
    memoryless_inventory,
    mua_inventory,
    state_space_inventory,
    swr_detector_inventory,
)
from continuum_swr_replay.analysis.detector_comparison.availability import (
    OBSERVATION_DTYPES,
    coverage_from_observations,
    exclusion_by_reason,
    not_still_times,
    sample_runs,
    still_intervals,
)
from continuum_swr_replay.analysis.detector_comparison.cells import (
    membership_against,
    no_detector_time,
)
from continuum_swr_replay.analysis.detector_comparison.components import (
    overlap_components,
)
from continuum_swr_replay.analysis.detector_comparison.inventory_run import (
    CELL_DETECTORS,
    CELLS,
    COMPARISONS,
    DETECTORS,
    MAX_GAP_S,
    ROSTER,
    SEQUENCE_COHORTS,
    SWR_DEFINITIONS,
    TABLE_DTYPES,
    speed_trace,
)
from continuum_swr_replay.analysis.detector_comparison.overlap import (
    SET_ASIDE_DTYPES,
    directional_agreement,
    event_matches,
    flag_pairs,
    overlap_pairs,
    remove_bridging_events,
    set_aside_events,
    time_coverage,
)
from continuum_swr_replay.analysis.detector_comparison.recount import (
    FRAGMENT_GAPS_S,
    merge_fragments,
)
from continuum_swr_replay.analysis.detector_comparison.schema import (
    MEMORYLESS,
    MEMORYLESS_CRITERIA,
    STATE_SPACE,
    combine_inventories,
    typed_table,
)
from continuum_swr_replay.analysis.detector_comparison.swr_matrix import (
    definition_disagreement,
    paired_labels,
    swr_label_matrix,
    swr_labels,
)
from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.intervals import (
    contains,
    total_duration,
)
from continuum_swr_replay.parameters import DECODE_SAMPLE_S
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "BRIDGING",
    "bridging_set_aside",
    "compare_epoch",
    "detector_counts",
    "epoch_inventories",
]

BRIDGING = "bridging"
"""The set-aside reason of an event bridging its own detector's unobserved
time."""


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
        detector: swr_detector_inventory(swr_dir, detector, dot=dot, drops=drops)
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


def bridging_set_aside(bridging: pd.DataFrame) -> pd.DataFrame:
    """The bridging events in the set-aside layout, their spans as their own."""
    return pd.DataFrame(
        {
            **{c: bridging[c] for c in (*SESSION_COLUMNS_DAY, "detector", "event_key")},
            "interval_start_s": bridging["event_start_s"],
            "interval_end_s": bridging["event_end_s"],
            "reason": BRIDGING,
        }
    ).astype(dict(SET_ASIDE_DTYPES))


def detector_counts(
    inventories: Mapping[str, InventoryTables],
    kept: pd.DataFrame,
    moving: pd.DataFrame,
    bridging: pd.DataFrame,
    coverage: Mapping[tuple, Any],
    key: tuple,
) -> pd.DataFrame:
    """Each detector's events in one epoch, from native to eligible, and its rate.

    Parameters
    ----------
    inventories : Mapping of str to InventoryTables
        Each detector's tables, as its adapter returned them.
    kept, moving, bridging : DataFrame
        The eligible events kept, those set aside as moving, and those set
        aside as bridging their own detector's unobserved time.
    coverage : Mapping of tuple to coverage
        ``(*key, detector)`` to an object whose ``exposure`` holds the
        detector's exposure intervals, shape ``(n_intervals, 2)``.
    key : tuple
        The epoch's ``(animal, day, epoch)``.

    Returns
    -------
    DataFrame
        One row per detector: ``n_native`` (inventory plus set aside by its
        adapter), ``n_set_aside`` (by its adapter or as bridging),
        ``n_moving``, ``n_eligible`` (kept), ``exposure_s``,
        ``n_in_exposure`` (kept events wholly inside the exposure) and
        ``rate_per_s`` (``n_in_exposure / exposure_s``, NaN without exposure).

    Raises
    ------
    RuntimeError
        A native count is not eligible plus moving plus set aside.
    """
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
        ``analysed_time.artifact_event_drops`` of the frame as saved.
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
    time, speed = speed_trace(dot)
    inventories = epoch_inventories(
        dot,
        drops,
        decode_run_id=decode_run_id,
        carey_dir=carey_dir,
        memoryless_dir=memoryless_dir,
    )
    observations = typed_table(
        (tables.observations for tables in inventories.values()), OBSERVATION_DTYPES
    )
    dropped = typed_table(
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
    set_aside = typed_table([dropped, bridging_set_aside(bridging)], SET_ASIDE_DTYPES)

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
            sample_interval_s=DECODE_SAMPLE_S,
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
        "counts": [detector_counts(inventories, kept, moving, bridging, coverage, key)],
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
    out = {
        name: typed_table(tables[name], dtypes) for name, dtypes in TABLE_DTYPES.items()
    }
    out["inventory"] = out["inventory"].sort_values(
        ["detector", "event_start_s"], kind="stable", ignore_index=True
    )
    return out
