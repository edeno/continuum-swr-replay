"""Stable row identities: the ``event_id`` and ``stop_id`` hashes.

Both are 63-bit ``pd.util.hash_pandas_object`` digests of a natural key, so a
row keeps the same id across runs, across row orderings, and across the
independent passes that build the event tables -- which is what lets those
passes be joined on the id rather than on position. Changing what goes into a
hash renumbers every downstream table, so the two functions live here, alone,
where that consequence is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "hash_event_id",
    "hash_stop_id",
    "validate_event_ids",
]


def hash_event_id(
    events: pd.DataFrame,
    event_col: str = "non_local_event",
) -> np.ndarray:
    """63-bit stable hash of ``(animal, <session date>, epoch, event_col)``.

    The session-date column is read by value, so the hash is identical whether
    the frame spells it ``session_date`` (canonical, repository-owned tables) or
    ``day`` (the intermediate classified event table, not covariate-stamped).
    ``pd.util.hash_pandas_object`` ignores column labels, so the two spellings
    hash to the same id and still match across the event/classified merge.
    """
    date_col = "session_date" if "session_date" in events.columns else "day"
    keys = events[["animal", date_col, "epoch", event_col]]
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()


def validate_event_ids(event_table: pd.DataFrame) -> None:
    """Raise if ``event_id`` has hash collisions after cross-session concat.

    Call this after ``pd.concat`` across sessions to guard against the
    (unlikely but non-zero) birthday-paradox collision in 63-bit hashes.
    """
    n_unique = event_table["event_id"].nunique()
    n_total = len(event_table)
    if n_unique != n_total:
        raise RuntimeError(
            f"event_id hash collision: {n_total} rows but {n_unique} unique IDs"
        )


def hash_stop_id(
    animal: str,
    day: int,
    epoch: str,
    stop_idx: np.ndarray,
    *,
    brain_area: str | None = None,
    event_source: str | None = None,
) -> np.ndarray:
    """63-bit stable hash of ``(animal, day, epoch, stop_idx)`` per bout.

    Optional ``brain_area`` / ``event_source`` keywords are mixed into
    the hash when supplied. ContFrag callers pass them so SWR/MUA and
    HPC/mPFC/OFC stop tables never collide on ``stop_id``. The
    NonLocal pipeline omits both (single-area, single-source by
    construction) and keeps its existing hashes intact.
    """
    keys = pd.DataFrame(
        {
            "animal": animal,
            "day": day,
            "epoch": epoch,
            "stop_idx": np.asarray(stop_idx, dtype=np.int64),
        }
    )
    if brain_area is not None:
        keys["__brain_area"] = brain_area
    if event_source is not None:
        keys["__event_source"] = event_source
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()
