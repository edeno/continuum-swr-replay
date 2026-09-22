"""Each sequence cohort labelled under each SWR definition.

Spec 03, "The SWR label matrix". A sequence event is ``SWR`` under a
definition when an event of that SWR inventory overlaps it with positive
duration, ``non-SWR`` when it lies wholly within both detectors' exposure
and nothing overlaps it, and ``unknown`` otherwise: missing LFP is never a
missing ripple.

The matrix restricts each cohort to its events lying wholly within its own
exposure intersected with every SWR definition's, so every column counts the
same events and a change of definition is not confounded with a change of
coverage. A non-SWR fraction belongs to its definition and is never reported
without it. Where definitions disagree about an event, or a matched pair's
two events carry different labels because their bounds differ, that is
reported rather than resolved.
"""

from collections.abc import Collection, Mapping
from functools import reduce
from types import MappingProxyType

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.inventory_comparison.intervals import (
    contains,
    intersect,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import Coverage
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "LABEL_DTYPES",
    "MATRIX_DTYPES",
    "PAIRED_LABEL_DTYPES",
    "definition_disagreement",
    "paired_labels",
    "swr_label_matrix",
    "swr_labels",
]

LABEL = {"matched": "SWR", "unmatched": "non-SWR", "unknown": "unknown"}

LABEL_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "cohort": object,
        "event_key": object,
        "swr_definition": object,
        "label": object,
        "in_matrix": np.bool_,
    }
)
"""One row per sequence event and SWR definition. ``in_matrix`` marks the
events every column of the matrix shares."""

MATRIX_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "cohort": object,
        "swr_definition": object,
        "n_events": np.int64,
        "n_eligible": np.int64,
        "n_excluded": np.int64,
        "n_swr": np.int64,
        "n_non_swr": np.int64,
        "non_swr_fraction": np.float64,
    }
)

PAIRED_LABEL_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "swr_definition": object,
        "detector_a": object,
        "event_key_a": object,
        "label_a": object,
        "detector_b": object,
        "event_key_b": object,
        "label_b": object,
        "agreement": object,
    }
)
"""One row per overlapping pair of sequence events and SWR definition;
``agreement`` is ``agree``, ``disagree``, or ``unknown`` where either label
is unknown."""


def swr_labels(
    inventory: pd.DataFrame,
    matches: pd.DataFrame,
    coverage: Mapping[tuple, Coverage],
    *,
    cohorts: Collection[str],
    swr_definitions: Collection[str],
) -> pd.DataFrame:
    """Label every event of each sequence cohort under each SWR definition.

    Parameters
    ----------
    inventory : DataFrame
        Events in the common schema, sequence and SWR inventories together.
    matches : DataFrame
        ``event_matches`` over the same inventory and coverage.
    coverage : mapping
        ``(animal, day, epoch, detector) -> Coverage``. A definition with no
        entry for an epoch observed none of it.
    cohorts, swr_definitions : collection of str
        Detector names.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``LABEL_DTYPES``.

    """
    empty = np.empty((0, 2))
    events = inventory[inventory.detector.isin(cohorts)]
    in_matrix = np.zeros(len(events), dtype=bool)
    for (*session, cohort), group in events.groupby([*SESSION_COLUMNS_DAY, "detector"]):
        shared = reduce(
            intersect,
            [
                coverage.get((*session, d), Coverage(empty, empty)).exposure
                for d in [cohort, *swr_definitions]
            ],
        )
        in_matrix[events.index.get_indexer(group.index)] = contains(
            shared, group.event_start_s.to_numpy(), group.event_end_s.to_numpy()
        )
    base = pd.DataFrame(
        {
            **{c: events[c].to_numpy() for c in SESSION_COLUMNS_DAY},
            "cohort": events.detector.to_numpy(),
            "event_key": events.event_key.to_numpy(),
            "in_matrix": in_matrix,
        }
    )
    labels = base.merge(
        pd.DataFrame({"swr_definition": list(swr_definitions)}), how="cross"
    )
    membership = matches.set_index(["event_key", "other_detector"]).membership
    found = pd.MultiIndex.from_arrays([labels.event_key, labels.swr_definition])
    labels["label"] = membership.reindex(found).map(LABEL).fillna("unknown").to_numpy()
    return labels[list(LABEL_DTYPES)].astype(dict(LABEL_DTYPES))


def swr_label_matrix(labels: pd.DataFrame) -> pd.DataFrame:
    """Count each cohort's labels under each definition, per epoch.

    Parameters
    ----------
    labels : DataFrame
        From ``swr_labels``.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``MATRIX_DTYPES``. The fraction is over the
        eligible events and NaN where there are none.

    """
    eligible = labels.in_matrix
    matrix = (
        labels.assign(
            is_swr=eligible & (labels.label == "SWR"),
            is_non_swr=eligible & (labels.label == "non-SWR"),
        )
        .groupby([*SESSION_COLUMNS_DAY, "cohort", "swr_definition"])
        .agg(
            n_events=("event_key", "size"),
            n_eligible=("in_matrix", "sum"),
            n_swr=("is_swr", "sum"),
            n_non_swr=("is_non_swr", "sum"),
        )
        .reset_index()
    )
    matrix["n_excluded"] = matrix.n_events - matrix.n_eligible
    matrix["non_swr_fraction"] = matrix.n_non_swr / matrix.n_eligible.where(
        matrix.n_eligible > 0
    )
    return matrix[list(MATRIX_DTYPES)].astype(dict(MATRIX_DTYPES))


def paired_labels(
    pairs: pd.DataFrame, labels: pd.DataFrame, *, cohorts: Collection[str]
) -> pd.DataFrame:
    """Each overlapping pair of sequence events, with both events' labels.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``PAIRED_LABEL_DTYPES``.

    """
    both = pairs[pairs.detector_a.isin(cohorts) & pairs.detector_b.isin(cohorts)]
    label = labels[["event_key", "swr_definition", "label"]]
    paired = both.merge(
        label.rename(columns={"event_key": "event_key_a", "label": "label_a"}),
        on="event_key_a",
    ).merge(
        label.rename(columns={"event_key": "event_key_b", "label": "label_b"}),
        on=["event_key_b", "swr_definition"],
    )
    paired["agreement"] = np.where(
        (paired.label_a == "unknown") | (paired.label_b == "unknown"),
        "unknown",
        np.where(paired.label_a == paired.label_b, "agree", "disagree"),
    )
    return paired[list(PAIRED_LABEL_DTYPES)].astype(dict(PAIRED_LABEL_DTYPES))


def definition_disagreement(labels: pd.DataFrame) -> pd.DataFrame:
    """For each sequence event, whether the SWR definitions disagree about it.

    An unknown label is neither agreement nor disagreement; it is counted.

    Returns
    -------
    DataFrame
        One row per event: session, ``cohort``, ``event_key``, ``n_swr``,
        ``n_non_swr``, ``n_unknown`` and ``definitions_disagree``.

    """
    counts = (
        labels.assign(
            n_swr=labels.label == "SWR",
            n_non_swr=labels.label == "non-SWR",
            n_unknown=labels.label == "unknown",
        )
        .groupby([*SESSION_COLUMNS_DAY, "cohort", "event_key"])[
            ["n_swr", "n_non_swr", "n_unknown"]
        ]
        .sum()
        .astype(np.int64)
        .reset_index()
    )
    counts["definitions_disagree"] = (counts.n_swr > 0) & (counts.n_non_swr > 0)
    return counts
