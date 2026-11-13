"""Event ends for toy onset tables, as ``attach_event_ends`` would add them.

The per-bin rate runners fit the at-risk design, which needs each onset's
event end (``duration_aware_history.END_COLUMN``) and requires one detector's
events not to overlap within a stop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
    END_COLUMN,
)


def with_event_ends(onsets: pd.DataFrame, max_duration_s: float = 0.02) -> pd.DataFrame:
    """Give every onset an end that does not reach the next event of its detector.

    Each event lasts ``max_duration_s`` or half the gap to the next onset of its
    ``event_type`` in the same stop, whichever is shorter.

    Parameters
    ----------
    onsets : pd.DataFrame
        Toy onsets with ``stop_id``, ``event_type`` and ``t_since_stop_onset``.
    max_duration_s : float
        Longest event, in seconds.

    Returns
    -------
    pd.DataFrame
        ``onsets`` with :data:`END_COLUMN` added, rows in their original order.
    """
    out = onsets.copy()
    if out.empty:
        out[END_COLUMN] = pd.Series(dtype=float)
        return out
    order = out.sort_values(["stop_id", "event_type", "t_since_stop_onset"])
    t = order["t_since_stop_onset"]
    gap = t.groupby([order["stop_id"], order["event_type"]]).shift(-1) - t
    duration = np.minimum(max_duration_s, (gap / 2).fillna(max_duration_s))
    out[END_COLUMN] = (t + duration).reindex(out.index)
    return out
