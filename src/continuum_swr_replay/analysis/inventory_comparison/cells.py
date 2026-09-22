"""The cells no pairwise table reaches.

Spec 03, "The cells that pairwise counting misses". Each event cell asks
whether an event was found by none of a set of detectors:

- events found by exactly one detector: every detector against all the others;
- SWRs matched by neither decoding detector: SWR inventories against the two
  sequence detectors;
- sequences matched by no LFP- or rate-triggered inventory, the
  neither-ripple-nor-burst cell: sequence detectors against the SWR and MUA
  inventories.

An event is ``unmatched`` by the set only when every detector in it looked:
each ran in that epoch and was exposed over the whole event. Otherwise it is
``unknown``, reported separately and never counted as unmatched. The time
cell, time found by no detector, is shared exposure minus the union of every
detection.
"""

from collections.abc import Collection, Mapping
from functools import reduce
from types import MappingProxyType

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.inventory_comparison.intervals import (
    intersect,
    total_duration,
    union,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import Coverage
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "CELL_DTYPES",
    "NO_DETECTOR_DTYPES",
    "membership_against",
    "no_detector_time",
]

CELL_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector": object,
        "event_key": object,
        "n_against": np.int64,
        "membership": object,
    }
)
"""One row per source event. ``n_against`` counts the detectors it was
tested against, itself excluded; ``membership`` is ``matched``,
``unmatched`` or ``unknown``."""

NO_DETECTOR_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "shared_exposure_s": np.float64,
        "available": np.bool_,
        "detected_s": np.float64,
        "undetected_s": np.float64,
        "undetected_fraction": np.float64,
    }
)


def membership_against(
    inventory: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    sources: Collection[str],
    against: Collection[str],
) -> pd.DataFrame:
    """Whether each source event was found by any detector of a set.

    Parameters
    ----------
    inventory : DataFrame
        Events in the common schema.
    matches : DataFrame
        ``event_matches`` over the same inventory.
    sources : collection of str
        Detectors whose events are classified.
    against : collection of str
        The set tested against. A source in it is not tested against itself.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``CELL_DTYPES``.

    """
    against = set(against)
    for source in sources:
        if not against - {source}:
            raise ValueError(
                f"{source} must be tested against a detector other than itself."
            )
    events = inventory.loc[
        inventory.detector.isin(sources),
        [*SESSION_COLUMNS_DAY, "detector", "event_key"],
    ]
    relevant = matches[matches.other_detector.isin(against)]
    tally = relevant.groupby("event_key").agg(
        any_matched=("matched", "any"),
        n_unmatched=("membership", lambda m: int((m == "unmatched").sum())),
    )
    out = events.join(tally, on="event_key")
    out["n_against"] = [len(against - {d}) for d in out.detector]
    out["membership"] = np.where(
        out.any_matched.fillna(False).astype(bool),
        "matched",
        np.where(out.n_unmatched.eq(out.n_against), "unmatched", "unknown"),
    )
    return out[list(CELL_DTYPES)].astype(dict(CELL_DTYPES)).reset_index(drop=True)


def no_detector_time(
    inventory: pd.DataFrame,
    coverage: Mapping[tuple, Coverage],
    *,
    detectors: Collection[str],
) -> pd.DataFrame:
    """Time within every detector's exposure that no detector's event covers.

    Parameters
    ----------
    inventory : DataFrame
        Events in the common schema; detectors outside ``detectors`` are
        ignored.
    coverage : mapping
        ``(animal, day, epoch, detector) -> Coverage``. A detector with no
        entry for an epoch observed none of it.
    detectors : collection of str

    Returns
    -------
    DataFrame
        One row per epoch in ``coverage``; columns and dtypes as
        ``NO_DETECTOR_DTYPES``. Unavailable, with no fraction, where the
        detectors share no exposure.

    """
    empty = np.empty((0, 2))
    rows = []
    for session in sorted({tuple(key[:3]) for key in coverage}):
        shared = reduce(
            intersect,
            [
                coverage[(*session, d)].exposure if (*session, d) in coverage else empty
                for d in detectors
            ],
        )
        mask = inventory.detector.isin(detectors) & np.logical_and.reduce(
            [
                inventory[c] == v
                for c, v in zip(SESSION_COLUMNS_DAY, session, strict=True)
            ]
        )
        found = union(inventory.loc[mask, ["event_start_s", "event_end_s"]].to_numpy())
        shared_s = total_duration(shared)
        detected_s = total_duration(intersect(found, shared))
        rows.append(
            dict(zip(SESSION_COLUMNS_DAY, session, strict=True))
            | {
                "shared_exposure_s": shared_s,
                "available": shared_s > 0,
                "detected_s": detected_s,
                "undetected_s": shared_s - detected_s,
                "undetected_fraction": (shared_s - detected_s) / shared_s
                if shared_s > 0
                else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(
            {k: pd.Series(dtype=v) for k, v in NO_DETECTOR_DTYPES.items()}
        )
    return pd.DataFrame(rows)[list(NO_DETECTOR_DTYPES)].astype(dict(NO_DETECTOR_DTYPES))
