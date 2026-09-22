"""The fragment-merge recount: how much a count owes to segmentation.

Our detector can split one replay into
several events where the non-local probability dips below threshold. Here one
detector's events closer than a stated gap are merged, so the same overlap
accounting can be rerun on the merged intervals beside the native ones.

A merge never crosses an epoch, and never leaves one continuous interval of
the detector's exposure: the fragments and the whole gap between them must lie
in time that was observed with the animal still, so no merge bridges movement
or missing samples. At a gap of 0 nothing merges and the native events come
back. The merged intervals are accounting intervals for this diagnostic, not a
replacement cohort; the native count stays primary.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.inventory_comparison.intervals import contains
from continuum_swr_replay.analysis.inventory_comparison.overlap import Coverage
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = ["FRAGMENT_GAPS_S", "merge_fragments"]

FRAGMENT_GAPS_S = (0.0, 0.010, 0.020, 0.050)
"""The stated grid of merge gaps, in seconds."""

COLUMNS = [
    *SESSION_COLUMNS_DAY,
    "detector",
    "event_key",
    "event_start_s",
    "event_end_s",
    "interval_span_s",
    "n_fragments",
]


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
        Events in the common schema, bridging events already removed.
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
        One row per merged interval: session, ``detector``, ``event_key``
        (the fragments' keys joined by ``|``), bounds, ``interval_span_s``
        and ``n_fragments``. Accepted wherever the overlap accounting
        accepts an inventory.

    """
    limit = _samples(np.array([gap_s]), sample_interval_s)[0]
    kept = inventory[inventory.detector != detector].assign(n_fragments=1)
    merged = [kept[[c for c in COLUMNS if c != "interval_span_s"]]]
    for session, group in inventory[inventory.detector == detector].groupby(
        list(SESSION_COLUMNS_DAY)
    ):
        group = group.sort_values("event_start_s")
        start = group.event_start_s.to_numpy()
        end = group.event_end_s.to_numpy()
        joins = (_samples(start[1:] - end[:-1], sample_interval_s) < limit) & contains(
            coverage[(*session, detector)].exposure, start[:-1], end[1:]
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
    out["interval_span_s"] = out.event_end_s - out.event_start_s
    return out[COLUMNS].astype({"n_fragments": np.int64})
