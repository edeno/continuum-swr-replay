"""Within-stop event-onset table.

:func:`build_event_onset_table` is **sparse**: one row per event *onset* (SWR /
non-local / MUA) in an immobility stop, with absolute ``onset_time``,
``t_since_stop_onset``, the continuous ``t_since_pump`` (carries across the
movement gaps), and the within-stop inter-onset interval. For autocorrelation
and the time-course numerator, and the onset input of every Family-B per-bin
rate fit, which builds its own 15 ms (stop x bin) design from ``stop_table``
and this table (``per_bin_design.aggregate_to_timebins``).

It recomputes immobility bouts from ``dot`` with the canonical helpers, so
``stop_id`` matches ``build_stop_table`` row-for-row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema
from continuum_swr_replay.tables.event_ids import hash_stop_id
from continuum_swr_replay.tables.session_keys import one_session

from .bouts import assign_bout, immobility_bout_intervals

EVENT_TYPES = {"SWR": "SWR_event", "non_local": "non_local_event", "MUA": "MUA_event"}


def _event_onset_times(dot: pd.DataFrame, times: np.ndarray, col: str) -> np.ndarray:
    """Sorted onset time (first labeled bin) of each event in ``col``.

    All three sources arrive held to the same atomic whole-event speed rule,
    and Non-Local labels to the saved probability threshold, by the read
    (``analyzed_time.with_swr_source``). An absent event column (e.g.
    ``MUA_event`` on older decodes) yields no onsets.
    """
    if col not in dot.columns:
        return np.array([], dtype=float)
    labels = dot[col].to_numpy()
    keep = labels > 0
    if not keep.any():
        return np.array([], dtype=float)
    df = pd.DataFrame({"t": times[keep], "lab": labels[keep]})
    return np.sort(df.groupby("lab", sort=False)["t"].min().to_numpy())


def _time_since_pump(query: np.ndarray, pump_times: np.ndarray) -> np.ndarray:
    """Seconds since the most recent pump at or before each query time (NaN if none)."""
    if pump_times.size == 0:
        return np.full(query.size, np.nan)
    pp = np.searchsorted(pump_times, query, side="right") - 1
    out = np.where(pp >= 0, query - pump_times[np.clip(pp, 0, None)], np.nan)
    return out


def _reward_bout_mask(
    pump_times: np.ndarray, bout_starts: np.ndarray, bout_ends: np.ndarray
) -> np.ndarray:
    """Boolean per bout: did a pump fire within the bout?"""
    mask = np.zeros(bout_starts.size, dtype=bool)
    if pump_times.size:
        pb = assign_bout(pump_times, bout_starts, bout_ends)
        mask[pb[pb >= 0]] = True
    return mask


def _session_keys(dot: pd.DataFrame) -> tuple[str, int, str]:
    animal, day, epoch = one_session(dot, "a time-bin table")
    return str(animal), int(day), str(epoch)


def build_event_onset_table(
    dot: pd.DataFrame,
    pump_times: np.ndarray,
) -> pd.DataFrame:
    """Sparse per-onset table (one row per SWR / non-local / MUA onset in a stop).

    ``dot`` is a frame read through ``analyzed_time.with_swr_source``, which holds
    its event labels to the speed rule and the Non-Local probability contract.
    """
    animal, day, epoch = _session_keys(dot)
    times = dot.index.to_numpy()
    bout_starts, bout_ends = immobility_bout_intervals(dot)
    pump_times = np.sort(np.asarray(pump_times, dtype=float))
    reward_mask = _reward_bout_mask(pump_times, bout_starts, bout_ends)

    frames = []
    for etype, col in EVENT_TYPES.items():
        onset_t = _event_onset_times(dot, times, col)
        if onset_t.size == 0:
            continue
        bidx = assign_bout(onset_t, bout_starts, bout_ends)
        valid = bidx >= 0
        onset_t, bidx = onset_t[valid], bidx[valid]
        frames.append(
            pd.DataFrame(
                {
                    "animal": animal,
                    "session_date": day,
                    "epoch": epoch,
                    "stop_idx": bidx,
                    "event_type": etype,
                    "onset_time": onset_t,
                    "t_since_stop_onset": onset_t - bout_starts[bidx],
                    "t_since_pump": _time_since_pump(onset_t, pump_times),
                    "is_reward_stop": reward_mask[bidx],
                }
            )
        )
    cols = [
        "animal",
        "session_date",
        "epoch",
        "stop_id",
        "stop_idx",
        "event_type",
        "onset_time",
        "t_since_stop_onset",
        "t_since_pump",
        "is_reward_stop",
        "isi_within_stop",
    ]
    if not frames:
        return stamp_covariate_schema(pd.DataFrame(columns=cols), table="event_onset")
    out = pd.concat(frames, ignore_index=True)
    out["stop_id"] = hash_stop_id(animal, day, epoch, out["stop_idx"].to_numpy())
    out = out.sort_values(["event_type", "stop_id", "onset_time"]).reset_index(
        drop=True
    )
    out["isi_within_stop"] = out.groupby(["event_type", "stop_id"])["onset_time"].diff()
    return stamp_covariate_schema(out[cols], table="event_onset")
