"""The fragment-merge recount: how much a count owes to segmentation.

Spec 03, "Fragment-merge recount". Our detector can split one replay into
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

from continuum_swr_replay.analysis.inventory_comparison.intervals import contains
from continuum_swr_replay.analysis.inventory_comparison.overlap import Coverage

__all__ = ["FRAGMENT_GAPS_S", "merge_fragments"]

FRAGMENT_GAPS_S = (0.0, 0.010, 0.020, 0.050)
"""The stated grid of merge gaps, in seconds."""

SESSION = ["animal", "day", "epoch"]
COLUMNS = [
    *SESSION,
    "detector",
    "event_key",
    "event_start_s",
    "event_end_s",
    "interval_span_s",
    "n_fragments",
]


def merge_fragments(
    inventory: pd.DataFrame,
    coverage: Mapping[tuple, Coverage],
    *,
    detector: str,
    gap_s: float,
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
        the earlier ends.

    Returns
    -------
    DataFrame
        One row per merged interval: session, ``detector``, ``event_key``
        (the fragments' keys joined by ``|``), bounds, ``interval_span_s``
        and ``n_fragments``. Accepted wherever the overlap accounting
        accepts an inventory.

    """
    kept = inventory[inventory.detector != detector].assign(n_fragments=1)
    merged = [kept[[c for c in COLUMNS if c != "interval_span_s"]]]
    for session, group in inventory[inventory.detector == detector].groupby(SESSION):
        group = group.sort_values("event_start_s")
        start = group.event_start_s.to_numpy()
        end = group.event_end_s.to_numpy()
        joins = (start[1:] - end[:-1] < gap_s) & contains(
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
            .assign(**dict(zip(SESSION, session, strict=True)), detector=detector)
        )
    out = pd.concat(merged, ignore_index=True)
    out["interval_span_s"] = out.event_end_s - out.event_start_s
    return out[COLUMNS].astype({"n_fragments": np.int64})
