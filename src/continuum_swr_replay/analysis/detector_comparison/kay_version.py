"""The Kay version check: a direct Kay inventory against the canonical one.

``kay_version_check`` marks each event of a direct Kay inventory (Kay at the
pinned ``ripple_detection`` commit) and of the canonical Spyglass one matched,
added or missed, by strict positive-duration overlap (``overlap_pairs``). The
SWR runner (``scripts/detection/run_swr_detection.py``) writes its rows per
epoch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detector_comparison.overlap import overlap_pairs
from continuum_swr_replay.analysis.detector_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)

__all__ = ["kay_version_check"]

_DIRECT_KAY = "kay"
"""``kay_version_check``'s name for the direct Kay inventory."""

_CANONICAL_KAY = "kay_spyglass"
"""``kay_version_check``'s name for the canonical Spyglass Kay inventory."""


def kay_version_check(
    direct: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    session: SessionIdentity,
    run_id: str,
    speed_time_s: NDArray,
    speed_cm_per_s: NDArray,
    direct_detector: str = _DIRECT_KAY,
) -> pd.DataFrame:
    """Mark each event of two Kay inventories matched, added or missed.

    ``direct`` is Kay at the pinned ``ripple_detection`` commit (detector
    ``direct_detector``, by default ``"kay"``) and ``canonical`` the
    canonical Spyglass events (``"kay_spyglass"``). Two events match when they share time of positive
    duration, ``overlap_pairs``' strict rule; touching at an instant is no
    match. A direct event that matches nothing is ``added``, a canonical one
    ``missed``. A matched event's partner is the event of the other set it
    overlaps longest, the earlier one on a tie, so a split event can leave
    several matched events sharing one partner. Matching uses every event of
    both sets, eligible or not; ``whole_event_speed_eligible`` lets the report
    split them.

    Parameters
    ----------
    direct, canonical : pd.DataFrame
        Event tables with ``start_time`` and ``end_time`` in seconds, indexed
        by the detector's own event ids (the package's ``event_number``).
    session : SessionIdentity
        The epoch, named in the event keys.
    run_id : str
        The run, recorded on the inventories the keys come from.
    speed_time_s, speed_cm_per_s : NDArray, shape (n_samples,)
        The speed trace eligibility is read from, with strictly increasing
        times.
    direct_detector : str, optional
        The direct inventory's name in the rows and event keys; default
        ``"kay"``. Kay run on another input takes another name, so that its
        keys differ from the direct Kay's when the two share a table.

    Returns
    -------
    check : pd.DataFrame
        One row per event, the direct events first, each set in start order:

        - ``detector``: ``direct_detector`` or ``"kay_spyglass"``;
        - ``event_key``: ``"<detector>:<animal><day>_<epoch>:<event id>"``,
          as ``inventory_from_intervals`` makes it;
        - ``status``: ``"matched"``, ``"added"`` or ``"missed"``;
        - ``partner_event_key``: the partner's key, always of object dtype;
          NaN unless matched;
        - ``start_shift_s``, ``end_shift_s``: the direct event's bound minus
          the canonical one's, over the event and its partner, so both rows
          of a mutual pair agree; NaN unless matched;
        - ``whole_event_speed_eligible``: every speed sample from the event's
          start to its end, of which there must be one, finite and below
          4 cm/s, the canonical rule (``event_speeds_are_immobile``).
    """
    events = pd.concat(
        [
            inventory_from_intervals(
                table["start_time"].to_numpy(),
                table["end_time"].to_numpy(),
                native_keys=table.index.to_numpy(),
                detector=detector,
                session=session,
                run_id=run_id,
                speed_time_s=speed_time_s,
                speed_cm_per_s=speed_cm_per_s,
            )
            for detector, table in (
                (direct_detector, direct),
                (_CANONICAL_KAY, canonical),
            )
        ],
        ignore_index=True,
    )
    pairs = overlap_pairs(events)
    # every overlap, seen from each of its two events
    links = pd.concat(
        [
            pd.DataFrame(
                {
                    "event_key": pairs[f"event_key_{own}"],
                    "partner_event_key": pairs[f"event_key_{other}"],
                    "intersection_s": pairs["intersection_s"],
                }
            )
            for own, other in (("a", "b"), ("b", "a"))
        ],
        ignore_index=True,
    )
    longest = links.loc[links.groupby("event_key")["intersection_s"].idxmax()]
    partners = longest.set_index("event_key")["partner_event_key"]

    check = events[["detector", "event_key", "whole_event_speed_eligible"]].copy()
    # object even when nothing matches, where map alone would give float64 NaN
    check["partner_event_key"] = check["event_key"].map(partners).astype(object)
    is_direct = (check["detector"] == direct_detector).to_numpy()
    check["status"] = np.where(
        check["partner_event_key"].notna(),
        "matched",
        np.where(is_direct, "added", "missed"),
    )
    bounds = events.set_index("event_key")[["event_start_s", "event_end_s"]]
    own_bounds = bounds.loc[check["event_key"]].to_numpy()
    partner_bounds = bounds.reindex(check["partner_event_key"]).to_numpy()
    shift = np.where(is_direct, 1.0, -1.0)[:, np.newaxis] * (
        own_bounds - partner_bounds
    )
    check["start_shift_s"] = shift[:, 0]
    check["end_shift_s"] = shift[:, 1]
    return check[
        [
            "detector",
            "event_key",
            "status",
            "partner_event_key",
            "start_shift_s",
            "end_shift_s",
            "whole_event_speed_eligible",
        ]
    ]
