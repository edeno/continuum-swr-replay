"""Within-stop time-binned tables (Part B).

Two complementary structures over immobility stops, for the time-resolved
event-rate analyses (consummatory time course, autocorrelation, time-since-
reward at both within- and across-stop scales):

- :func:`build_event_onset_table` — **sparse**: one row per event *onset*
  (SWR / non-local / MUA), with absolute ``onset_time``, ``t_since_stop_onset``,
  the continuous ``t_since_pump`` (carries across the movement gaps), and the
  within-stop inter-onset interval. For autocorrelation and the time-course
  numerator.

- :func:`build_stop_timebin_table` — **dense**: one row per (stop, ``bin_s``-wide
  bin), with onset counts, the same two clocks, ``is_reward_stop`` and mean
  ``head_speed``. The equal-exposure substrate for the per-bin onset-rate model.

Both recompute immobility bouts from ``dot`` with the canonical helpers, so
``stop_id`` matches ``build_stop_table`` row-for-row. Movement is absence of
rows; ``t_since_pump`` (from absolute time) is the bridge across stops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from ..hippocampus.event_immobility import (
    filter_event_labels_by_speed,
    validate_nonlocal_event_probability_contract,
)
from .bouts import assign_bout, immobility_bout_intervals
from .covariate_registry import stamp_covariate_schema
from .event_ids import hash_stop_id

EVENT_TYPES = {"SWR": "SWR_event", "non_local": "non_local_event", "MUA": "MUA_event"}


def _event_onset_times(dot: pd.DataFrame, times: np.ndarray, col: str) -> np.ndarray:
    """Sorted onset time (first labelled bin) of each event in ``col``.

    All three sources arrive after the same atomic whole-event speed filter.
    Non-Local labels additionally require every event bin to clear the saved
    probability threshold. An absent event column (e.g. ``MUA_event`` on older
    decodes) yields no onsets; the dense builder then zero-fills that count.
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
    return (
        str(dot["animal"].iloc[0]),
        int(dot["day"].iloc[0]),
        str(dot["epoch"].iloc[0]),
    )


def build_event_onset_table(
    dot: pd.DataFrame,
    pump_times: np.ndarray,
    *,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """Sparse per-onset table (one row per SWR / non-local / MUA onset in a stop).

    ``nonlocal_threshold`` must match the threshold used to create
    ``dot``'s Non-Local labels — it is validated, not re-applied. Family A
    (``replay/tables_event.py``) threads the same parameter; keep the two in
    step or the families will disagree on which sessions are admissible.
    """
    if "non_local_event" in dot:
        validate_nonlocal_event_probability_contract(
            dot,
            probability_threshold=nonlocal_threshold,
        )
    event_cols = [column for column in EVENT_TYPES.values() if column in dot]
    dot = filter_event_labels_by_speed(
        dot,
        event_cols,
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
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
        "animal", "session_date", "epoch", "stop_id", "stop_idx", "event_type",
        "onset_time", "t_since_stop_onset", "t_since_pump", "is_reward_stop",
        "isi_within_stop",
    ]
    if not frames:
        return stamp_covariate_schema(
            pd.DataFrame(columns=cols), table="event_onset"
        )
    out = pd.concat(frames, ignore_index=True)
    out["stop_id"] = hash_stop_id(animal, day, epoch, out["stop_idx"].to_numpy())
    out = out.sort_values(["event_type", "stop_id", "onset_time"]).reset_index(drop=True)
    out["isi_within_stop"] = out.groupby(["event_type", "stop_id"])["onset_time"].diff()
    return stamp_covariate_schema(out[cols], table="event_onset")


def build_stop_timebin_table(
    dot: pd.DataFrame,
    pump_times: np.ndarray,
    *,
    bin_s: float = 0.004,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """Dense per-(stop, bin) table at ``bin_s`` resolution (default 4 ms).

    ``nonlocal_threshold`` must match the threshold used to create ``dot``'s
    Non-Local labels — see :func:`build_event_onset_table`.
    """
    if "non_local_event" in dot:
        validate_nonlocal_event_probability_contract(
            dot,
            probability_threshold=nonlocal_threshold,
        )
    event_cols = [column for column in EVENT_TYPES.values() if column in dot]
    dot = filter_event_labels_by_speed(
        dot,
        event_cols,
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    animal, day, epoch = _session_keys(dot)
    times = dot.index.to_numpy()
    speed = dot["head_speed"].to_numpy()
    bout_starts, bout_ends = immobility_bout_intervals(dot)
    n_bouts = bout_starts.size
    cols = [
        "animal", "session_date", "epoch", "stop_id", "bin_in_stop",
        "t_since_stop_onset", "t_since_pump", "is_reward_stop",
        "n_swr_events", "n_nonlocal_events", "n_mua_events", "head_speed",
    ]
    if n_bouts == 0:
        return stamp_covariate_schema(
            pd.DataFrame(columns=cols), table="stop_timebin"
        )

    # Variable bins-per-bout; build the (bout, within-bin) index for every bin.
    n_bins = np.maximum(1, np.ceil((bout_ends - bout_starts) / bin_s).astype(int))
    bout_of_bin = np.repeat(np.arange(n_bouts), n_bins)
    within = np.concatenate([np.arange(n) for n in n_bins])
    bin_offset = np.concatenate([[0], np.cumsum(n_bins)])
    n_total = int(bin_offset[-1])
    bin_center = bout_starts[bout_of_bin] + (within + 0.5) * bin_s

    def _global_bin(query: np.ndarray, qbidx: np.ndarray) -> np.ndarray:
        wb = np.floor((query - bout_starts[qbidx]) / bin_s).astype(int)
        wb = np.clip(wb, 0, n_bins[qbidx] - 1)
        return bin_offset[qbidx] + wb

    # Onset counts per bin.
    counts = {}
    for etype, col in EVENT_TYPES.items():
        onset_t = _event_onset_times(dot, times, col)
        bidx = assign_bout(onset_t, bout_starts, bout_ends)
        v = bidx >= 0
        gbin = _global_bin(onset_t[v], bidx[v])
        counts[etype] = np.bincount(gbin, minlength=n_total).astype(np.int32)

    # Mean head_speed per bin (from the 2 ms dot rows that fall in each bin).
    row_b = assign_bout(times, bout_starts, bout_ends)
    rv = row_b >= 0
    rgbin = _global_bin(times[rv], row_b[rv])
    ssum = np.bincount(rgbin, weights=speed[rv], minlength=n_total)
    scnt = np.bincount(rgbin, minlength=n_total)
    speed_mean = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)

    pump_times = np.sort(np.asarray(pump_times, dtype=float))
    reward_mask = _reward_bout_mask(pump_times, bout_starts, bout_ends)
    out = pd.DataFrame(
        {
            "animal": animal,
            "session_date": day,
            "epoch": epoch,
            "stop_id": hash_stop_id(animal, day, epoch, bout_of_bin),
            "bin_in_stop": within.astype(np.int32),
            "t_since_stop_onset": (bin_center - bout_starts[bout_of_bin]).astype(np.float32),
            "t_since_pump": _time_since_pump(bin_center, pump_times).astype(np.float32),
            "is_reward_stop": reward_mask[bout_of_bin],
            "n_swr_events": counts["SWR"],
            "n_nonlocal_events": counts["non_local"],
            "n_mua_events": counts["MUA"],
            "head_speed": speed_mean.astype(np.float32),
        }
    )[cols]
    return stamp_covariate_schema(out, table="stop_timebin")
