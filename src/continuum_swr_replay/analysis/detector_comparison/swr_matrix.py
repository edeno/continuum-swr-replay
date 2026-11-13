"""Each sequence cohort labelled under each SWR definition.

A sequence event is ``SWR`` under a definition when an event of that SWR
inventory overlaps it with positive duration and ``non-SWR`` when nothing
does, but only for an event lying wholly within both detectors' exposure;
otherwise it is ``unknown``: missing LFP is never a missing ripple, and an
overlap in the observed part cannot say what the rest held.

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
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detector_comparison.overlap import (
    MATCHED,
    UNKNOWN,
    UNMATCHED,
    Coverage,
    validate_detectors,
)
from continuum_swr_replay.intervals import (
    contains,
    intersect,
    total_duration,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "DISAGREEMENT_DTYPES",
    "LABEL_DTYPES",
    "MATRIX_DTYPES",
    "NON_SWR",
    "PAIRED_LABEL_DTYPES",
    "SWR",
    "definition_disagreement",
    "paired_labels",
    "swr_label_matrix",
    "swr_labels",
]

SWR, NON_SWR = "SWR", "non-SWR"
LABEL = {MATCHED: SWR, UNMATCHED: NON_SWR, UNKNOWN: UNKNOWN}

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
        "available": np.bool_,
        "n_events": np.int64,
        "n_eligible": np.int64,
        "n_excluded": np.int64,
        "n_swr": np.int64,
        "n_non_swr": np.int64,
        "non_swr_fraction": np.float64,
    }
)

DISAGREEMENT_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "cohort": object,
        "event_key": object,
        "n_swr": np.int64,
        "n_non_swr": np.int64,
        "n_unknown": np.int64,
        "definitions_disagree": np.bool_,
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


def _matrix_exposure(
    coverage: Mapping[tuple, Coverage],
    session: tuple,
    cohort: str,
    swr_definitions: Collection[str],
) -> NDArray:
    """The cohort's exposure within every definition's: the matrix population.

    A detector with no entry for the epoch observed none of it.
    """
    return reduce(
        intersect,
        [
            coverage[(*session, d)].exposure
            if (*session, d) in coverage
            else np.empty((0, 2))
            for d in [cohort, *swr_definitions]
        ],
    )


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
        Detector names, both on the roster and disjoint: a sequence detector
        passed as an SWR definition would report inter-detector agreement
        dressed as an SWR label.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``LABEL_DTYPES``.

    """
    cohorts = validate_detectors(cohorts, coverage, "cohorts")
    swr_definitions = validate_detectors(swr_definitions, coverage, "swr_definitions")
    if set(cohorts) & set(swr_definitions):
        raise ValueError(
            f"{sorted(set(cohorts) & set(swr_definitions))} cannot be both a "
            "cohort and an SWR definition."
        )
    events = inventory[inventory.detector.isin(cohorts)]
    in_matrix = np.zeros(len(events), dtype=bool)
    for (*session, cohort), group in events.groupby([*SESSION_COLUMNS_DAY, "detector"]):
        shared = _matrix_exposure(coverage, tuple(session), cohort, swr_definitions)
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
    # a label needs the whole event observed by both: an overlap in the part
    # the SWR detector saw cannot say what the rest held
    found = matches.set_index(["event_key", "other_detector"]).reindex(
        pd.MultiIndex.from_arrays([labels.event_key, labels.swr_definition])
    )
    labels["label"] = np.where(
        found.within_shared_exposure.eq(True).to_numpy(),
        found.membership.map(LABEL).to_numpy(),
        UNKNOWN,
    )
    return labels[list(LABEL_DTYPES)].astype(dict(LABEL_DTYPES))


def swr_label_matrix(
    labels: pd.DataFrame,
    coverage: Mapping[tuple, Coverage],
    *,
    cohorts: Collection[str],
    swr_definitions: Collection[str],
) -> pd.DataFrame:
    """Count each cohort's labels under each definition, per epoch.

    One row per epoch in ``coverage``, cohort and definition, so a cohort
    that looked and found nothing has a row of zeros, and a cohort or
    definition that did not look in an epoch has one marked unavailable.

    Parameters
    ----------
    labels : DataFrame
        From ``swr_labels``.
    coverage : mapping
        The coverage the labels were made under.
    cohorts, swr_definitions : collection of str
        As given to ``swr_labels``.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``MATRIX_DTYPES``. ``available`` says whether
        the matrix population exists in the epoch: the cohort's exposure
        within every definition's, the same for every column, so one
        definition not running leaves every column unavailable. The fraction
        is over the eligible events and NaN where there are none.

    """
    cohorts = validate_detectors(cohorts, coverage, "cohorts")
    swr_definitions = validate_detectors(swr_definitions, coverage, "swr_definitions")
    eligible = labels.in_matrix
    counted = (
        labels.assign(
            is_swr=eligible & (labels.label == SWR),
            is_non_swr=eligible & (labels.label == NON_SWR),
        )
        .groupby([*SESSION_COLUMNS_DAY, "cohort", "swr_definition"])
        .agg(
            n_events=("event_key", "size"),
            n_eligible=("in_matrix", "sum"),
            n_swr=("is_swr", "sum"),
            n_non_swr=("is_non_swr", "sum"),
        )
    )
    rows = []
    for session in sorted({tuple(key[:3]) for key in coverage}):
        for cohort in cohorts:
            shared = _matrix_exposure(coverage, session, cohort, swr_definitions)
            for definition in swr_definitions:
                index = (*session, cohort, definition)
                counts = (
                    counted.loc[index]
                    if index in counted.index
                    else pd.Series(
                        {"n_events": 0, "n_eligible": 0, "n_swr": 0, "n_non_swr": 0}
                    )
                )
                rows.append(
                    dict(zip(SESSION_COLUMNS_DAY, session, strict=True))
                    | {
                        "cohort": cohort,
                        "swr_definition": definition,
                        "available": total_duration(shared) > 0,
                        **{k: int(counts[k]) for k in counts.index},
                    }
                )
    matrix = pd.DataFrame(rows)
    if (matrix.n_swr + matrix.n_non_swr != matrix.n_eligible).any():
        raise RuntimeError(
            "Every eligible event must be labelled SWR or non-SWR; the labels were "
            "not made under this coverage."
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
    cohorts = list(cohorts)
    both = pairs[pairs.detector_a.isin(cohorts) & pairs.detector_b.isin(cohorts)]
    # a cohort with no events has no labels and no pairs; one in a pair must
    # have been labelled, or the merge below would drop its pairs unseen
    paired_cohorts = set(both.detector_a) | set(both.detector_b)
    unlabelled = sorted(paired_cohorts - set(labels.cohort.unique()))
    if unlabelled:
        raise ValueError(
            f"{unlabelled} have pairs but no labels; label every cohort before pairing."
        )
    label = labels[["event_key", "swr_definition", "label"]]
    paired = both.merge(
        label.rename(columns={"event_key": "event_key_a", "label": "label_a"}),
        on="event_key_a",
    ).merge(
        label.rename(columns={"event_key": "event_key_b", "label": "label_b"}),
        on=["event_key_b", "swr_definition"],
    )
    n_definitions = labels.swr_definition.nunique()
    if len(paired) != len(both) * n_definitions:
        raise ValueError(
            f"{len(both)} sequence pairs and {n_definitions} definitions should give "
            f"{len(both) * n_definitions} rows, not {len(paired)}; a pair's event "
            "has no label under some definition."
        )
    paired["agreement"] = np.where(
        (paired.label_a == UNKNOWN) | (paired.label_b == UNKNOWN),
        UNKNOWN,
        np.where(paired.label_a == paired.label_b, "agree", "disagree"),
    )
    return paired[list(PAIRED_LABEL_DTYPES)].astype(dict(PAIRED_LABEL_DTYPES))


def definition_disagreement(labels: pd.DataFrame) -> pd.DataFrame:
    """For each sequence event, whether the SWR definitions disagree about it.

    An unknown label is neither agreement nor disagreement; it is counted.

    Returns
    -------
    DataFrame
        One row per event; columns and dtypes as ``DISAGREEMENT_DTYPES``.

    """
    if labels.empty:
        return pd.DataFrame(
            {k: pd.Series(dtype=v) for k, v in DISAGREEMENT_DTYPES.items()}
        )
    counts = (
        labels.assign(
            n_swr=labels.label == SWR,
            n_non_swr=labels.label == NON_SWR,
            n_unknown=labels.label == UNKNOWN,
        )
        .groupby([*SESSION_COLUMNS_DAY, "cohort", "event_key"])[
            ["n_swr", "n_non_swr", "n_unknown"]
        ]
        .sum()
        .astype(np.int64)
        .reset_index()
    )
    counts["definitions_disagree"] = (counts.n_swr > 0) & (counts.n_non_swr > 0)
    return counts[list(DISAGREEMENT_DTYPES)].astype(dict(DISAGREEMENT_DTYPES))
