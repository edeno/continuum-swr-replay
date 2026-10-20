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
rows; ``t_since_pump`` (from absolute time) is the bridge across stops. So is
excluded time: a dense bin that touches the epoch's excluded time (the artifact
mask widened by 0.3 s) has no row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.analysed_time import excluded_intervals_of
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema
from continuum_swr_replay.tables.event_ids import hash_stop_id

from ...intervals import overlaps_intervals
from .bouts import assign_bout, immobility_bout_intervals

EVENT_TYPES = {"SWR": "SWR_event", "non_local": "non_local_event", "MUA": "MUA_event"}
ONSETS_IN_EXCLUDED_BINS_ATTR = "onsets_in_excluded_bins_by_event_type"
"""The dense table's ``attrs`` key: per event type, the onsets whose bin it
dropped for touching excluded time. Their events lie wholly outside excluded
time, so the sparse table keeps them."""


def _event_onset_times(dot: pd.DataFrame, times: np.ndarray, col: str) -> np.ndarray:
    """Sorted onset time (first labelled bin) of each event in ``col``.

    All three sources arrive held to the same atomic whole-event speed rule,
    and Non-Local labels to the saved probability threshold, by the read
    (``analysed_time.with_swr_source``). An absent event column (e.g.
    ``MUA_event`` on older decodes) yields no onsets; the dense builder then
    zero-fills that count.
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
) -> pd.DataFrame:
    """Sparse per-onset table (one row per SWR / non-local / MUA onset in a stop).

    ``dot`` is a frame read through ``analysed_time.with_swr_source``, which holds
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


def build_stop_timebin_table(
    dot: pd.DataFrame,
    pump_times: np.ndarray,
    *,
    bin_s: float = 0.004,
) -> pd.DataFrame:
    """Dense per-(stop, bin) table at ``bin_s`` resolution (default 4 ms).

    ``dot`` is a read frame, as :func:`build_event_onset_table`. A bin
    ``[left, min(left + bin_s, stop end)]`` that touches the epoch's excluded
    time (``analysed_time.excluded_intervals``; closed) is not analysed and has
    no row; the other bins keep their ``bin_in_stop``. An event kept by the
    event rule (wholly outside excluded time) can still start in such a bin;
    its onset stays in the sparse table and is counted, per event type, in
    ``attrs[ONSETS_IN_EXCLUDED_BINS_ATTR]``.
    """
    animal, day, epoch = _session_keys(dot)
    times = dot.index.to_numpy()
    speed = dot["head_speed"].to_numpy()
    bout_starts, bout_ends = immobility_bout_intervals(dot)
    n_bouts = bout_starts.size
    cols = [
        "animal",
        "session_date",
        "epoch",
        "stop_id",
        "bin_in_stop",
        "t_since_stop_onset",
        "t_since_pump",
        "is_reward_stop",
        "n_swr_events",
        "n_nonlocal_events",
        "n_mua_events",
        "head_speed",
    ]
    if n_bouts == 0:
        return stamp_covariate_schema(pd.DataFrame(columns=cols), table="stop_timebin")

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
            "t_since_stop_onset": (bin_center - bout_starts[bout_of_bin]).astype(
                np.float32
            ),
            "t_since_pump": _time_since_pump(bin_center, pump_times).astype(np.float32),
            "is_reward_stop": reward_mask[bout_of_bin],
            "n_swr_events": counts["SWR"],
            "n_nonlocal_events": counts["non_local"],
            "n_mua_events": counts["MUA"],
            "head_speed": speed_mean.astype(np.float32),
        }
    )[cols]
    excluded = excluded_intervals_of(dot)
    in_excluded_bins = dict.fromkeys(counts, 0)
    if len(excluded):
        left = bout_starts[bout_of_bin] + within * bin_s
        right = np.minimum(left + bin_s, bout_ends[bout_of_bin])
        dropped = overlaps_intervals(left, right, excluded)
        in_excluded_bins = {
            etype: int(count[dropped].sum()) for etype, count in counts.items()
        }
        out = out.loc[~dropped].reset_index(drop=True)
    out = stamp_covariate_schema(out, table="stop_timebin")
    out.attrs[ONSETS_IN_EXCLUDED_BINS_ATTR] = in_excluded_bins
    return out
