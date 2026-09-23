"""The fragment-merge recount: how much a count owes to segmentation.

Our detector can split one replay into several events where the non-local
probability dips below threshold. Here one detector's events closer than a
stated gap are merged, so the same overlap accounting can be rerun on the
merged intervals beside the native ones.

A merge never crosses an epoch, and never leaves one continuous interval of
the detector's exposure: the fragments and the whole gap between them must lie
in time that was observed with the animal still, so no merge bridges movement
or missing samples. One detector's events are disjoint (the inventory schema
requires it), so at a gap of 0 nothing merges and the native events come
back. The merged intervals are accounting intervals for this diagnostic, not a
replacement cohort; the native count stays primary.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.inventory_comparison.intervals import contains
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    validate_detectors,
)
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = ["FRAGMENT_GAPS_S", "RECOUNT_DTYPES", "merge_fragments"]

FRAGMENT_GAPS_S = (0.0, 0.010, 0.020, 0.050)
"""The stated grid of merge gaps, in seconds."""

RECOUNT_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "detector": object,
        "event_key": object,
        "event_start_s": np.float64,
        "event_end_s": np.float64,
        "interval_span_s": np.float64,
        "n_fragments": np.int64,
    }
)
"""The merged accounting table: the columns the overlap accounting reads,
with ``event_key`` the fragments' keys joined by ``|``. Narrower than the
inventory schema, so it is not a second inventory."""

COLUMNS = list(RECOUNT_DTYPES)


def _samples(duration_s: NDArray, sample_interval_s: float) -> NDArray:
    """Durations as whole numbers of samples; refuses any off the grid."""
    count = duration_s / sample_interval_s
    whole = np.rint(count)
    if not np.allclose(count, whole, rtol=0.0, atol=0.01):
        raise ValueError(
            f"A gap is not a whole number of {sample_interval_s} s samples; "
            "the events are not on the stated sample grid."
        )
    return whole.astype(np.int64)


def merge_fragments(
    inventory: pd.DataFrame,
    coverage: Mapping[tuple, Coverage],
    *,
    detector: str,
    gap_s: float,
    sample_interval_s: float,
) -> pd.DataFrame:
    """Merge one detector's events separated by less than ``gap_s``.

    Parameters
    ----------
    inventory : DataFrame
        Events in the common schema, bridging events already removed. One
        detector's events must be disjoint, as the schema requires.
    coverage : mapping
        ``(animal, day, epoch, detector) -> Coverage``; the merged detector's
        exposure bounds every merge.
    detector : str
        The detector whose events are merged; the others pass through.
    gap_s : float
        Consecutive events merge when the later starts less than this after
        the earlier ends. A whole number of samples.
    sample_interval_s : float
        The grid the events' bounds sit on. Gaps are counted in samples, so a
        gap of exactly ``gap_s`` never merges however the clock rounds.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``RECOUNT_DTYPES``: one row per merged interval,
        the other detectors' events passed through with ``n_fragments`` 1.
        The overlap accounting reads it as an inventory; the schema validator
        does not, since it lacks the columns that do not survive a merge.

    Raises
    ------
    ValueError
        ``gap_s`` is negative or off the sample grid, the detector is not on
        the roster, its events overlap one another, or an epoch of its events
        has no coverage.

    """
    if gap_s < 0:
        raise ValueError(f"gap_s must not be negative, got {gap_s}.")
    validate_detectors([detector], coverage, "detector")
    limit = _samples(np.array([gap_s]), sample_interval_s)[0]
    kept = inventory[inventory.detector != detector].assign(n_fragments=1)
    merged = [kept[[c for c in COLUMNS if c != "interval_span_s"]]]
    for session, group in inventory[inventory.detector == detector].groupby(
        list(SESSION_COLUMNS_DAY)
    ):
        own = coverage.get((*session, detector))
        if own is None:
            raise ValueError(
                f"{detector} has events in {session} but no coverage there."
            )
        group = group.sort_values("event_start_s")
        start = group.event_start_s.to_numpy()
        end = group.event_end_s.to_numpy()
        if (start[1:] < end[:-1]).any():
            raise ValueError(
                f"Two events of {detector} in {session} overlap; the recount needs "
                "one detector's events to be disjoint."
            )
        joins = (_samples(start[1:] - end[:-1], sample_interval_s) < limit) & contains(
            own.exposure, start[:-1], end[1:]
        )
        run = np.r_[0, np.cumsum(~joins)]
        merged.append(
            pd.DataFrame(
                {
                    "event_key": group.event_key.to_numpy(),
                    "event_start_s": start,
                    "event_end_s": end,
                    "run": run,
                }
            )
            .groupby("run")
            .agg(
                event_key=("event_key", "|".join),
                event_start_s=("event_start_s", "min"),
                event_end_s=("event_end_s", "max"),
                n_fragments=("event_key", "size"),
            )
            .assign(
                **dict(zip(SESSION_COLUMNS_DAY, session, strict=True)),
                detector=detector,
            )
        )
    out = pd.concat(merged, ignore_index=True)
    if out.empty:
        return pd.DataFrame(
            {name: pd.Series(dtype=dtype) for name, dtype in RECOUNT_DTYPES.items()}
        )
    out["interval_span_s"] = out.event_end_s - out.event_start_s
    return out[COLUMNS].astype(dict(RECOUNT_DTYPES))
