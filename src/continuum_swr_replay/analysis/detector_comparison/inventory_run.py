"""The inventory comparison run's layout: its roster, its grid and its tables.

What ``scripts/detector_comparison/run_inventory_comparison.py`` compares and writes, and
what the scripts that read its runs (the event features runner, the two
reports and the example figures) read it by: the roster of nine inventories
and the comparisons over them (:data:`ROSTER`, :data:`CELLS`,
:data:`COMPARISONS`), the decode it reads and its grid (:func:`decode_path`,
:func:`speed_trace`, which steps by ``parameters.DECODE_SAMPLE_S``, the step
the recount counts its gaps on), and every table an epoch
writes (:data:`TABLE_DTYPES`, :func:`output_paths`, :func:`stamped_tables`).
Design: ``docs/design/replay_detection_comparison/03-inventory-comparison.md``.
"""

from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detector_comparison.adapters import (
    CAREY,
    MUA,
    SWR_DETECTORS,
)
from continuum_swr_replay.analysis.detector_comparison.availability import (
    EXCLUSION_DTYPES,
    OBSERVATION_DTYPES,
)
from continuum_swr_replay.analysis.detector_comparison.cells import (
    CELL_DTYPES,
    NO_DETECTOR_DTYPES,
)
from continuum_swr_replay.analysis.detector_comparison.components import (
    COMPONENT_DTYPES,
)
from continuum_swr_replay.analysis.detector_comparison.overlap import (
    DIRECTIONAL_DTYPES,
    PAIR_DTYPES,
    SET_ASIDE_DTYPES,
    TIME_COVERAGE_DTYPES,
)
from continuum_swr_replay.analysis.detector_comparison.schema import (
    INVENTORY_DTYPES,
    MEMORYLESS,
    MEMORYLESS_CRITERIA,
    STATE_SPACE,
)
from continuum_swr_replay.analysis.detector_comparison.swr_matrix import (
    DISAGREEMENT_DTYPES,
    LABEL_DTYPES,
    MATRIX_DTYPES,
    PAIRED_LABEL_DTYPES,
)
from continuum_swr_replay.analysis.event_definition import analyzed_time
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.intervals import max_step_s, sample_step
from continuum_swr_replay.parameters import DECODE_SAMPLE_S
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.runs.io import output_path

__all__ = [
    "CELLS",
    "CELL_DETECTORS",
    "COMPARISONS",
    "COUNT_DTYPES",
    "DECODERS",
    "DETECTORS",
    "MAX_GAP_S",
    "ROSTER",
    "SEQUENCE_COHORTS",
    "SWR_DEFINITIONS",
    "TABLE_DTYPES",
    "TRIGGERED",
    "decode_path",
    "output_paths",
    "speed_trace",
    "stamped_tables",
]

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

MAX_GAP_S = max_step_s(DECODE_SAMPLE_S)
"""A step longer than this breaks the speed trace's runs."""
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
    """The canonical decode's ``data_over_time`` pickle (``decode_paths``),
    under ``paths.processed_data_dir()``."""
    return decode_paths(identifier, processed_data_dir()).data_over_time


def speed_trace(dot: pd.DataFrame) -> tuple[NDArray, NDArray]:
    """The frame's grid and its ``head_speed``.

    Raises
    ------
    ValueError
        The grid does not step by ``DECODE_SAMPLE_S``.
    """
    time = dot.index.to_numpy(dtype=np.float64)
    step = sample_step(time)
    if not np.isclose(step, DECODE_SAMPLE_S, rtol=0.0, atol=1e-5):
        raise ValueError(
            f"The decode grid steps by {step} s, not {DECODE_SAMPLE_S} s, the "
            "step the recount counts gaps on."
        )
    return time, dot["head_speed"].to_numpy(dtype=np.float64)


def output_paths(identifier: str, output_dir: Path) -> dict[str, Path]:
    """Every table an epoch can write, by name."""
    return {name: output_path(identifier, output_dir, name) for name in TABLE_DTYPES}


def stamped_tables(
    tables: Mapping[str, pd.DataFrame], stamp: Mapping[str, str]
) -> Mapping[str, pd.DataFrame]:
    """Stamp each table (in place, and return them) with the SWR source and `stamp`.

    The SWR source's stamp (``analyzed_time.stamp_swr_source``) goes first, then
    `stamp`'s keys are merged into each table's ``attrs``.
    """
    for frame in tables.values():
        analyzed_time.stamp_swr_source(frame)
        frame.attrs |= stamp
    return tables
