"""Canonical per-non-local-event classification table.

This module owns the single, figure-and-model-agnostic definition of a
non-local event's timing and its overlap relationship with detected
sharp-wave ripples (SWRs). One row per non-local event, derived from the
full event-label window (no immobility / probability filtering applied).

The public entry point :func:`build_classified_nonlocal_event_table`
takes a time-indexed ``data_over_time`` slice and emits, per event:

- **Timing** — ``event_start``, ``event_end``, ``event_duration``,
  ``event_midpoint_time`` over the *full* set of timebins carrying the
  event label.
- **Per-event speed summaries** — ``max_speed``, ``mean_speed``,
  ``start_speed``, ``end_speed`` and the immobility flag
  ``endpoints_speed_below_thresh`` (start- and end-bin speed both below
  ``speed_threshold``).
- **Peak content** — ``peak_nonlocal_prob`` (max of the summed
  continuous + fragmented non-local posterior within the event).
- **SWR overlap** — ``overlap_class`` (3-way:
  ``swr_contained`` / ``swr_partial`` / ``non_swr``), ``overlap_subclass``
  (6-way interval geometry), ``n_swrs_overlapping``, and the derived
  boolean ``is_swr`` (``overlap_class != "non_swr"``, i.e. strict
  positive-duration overlap with any SWR).
- **Matched SWR** — ``matched_swr_event`` (nullable ``Int64`` SWR label),
  ``matched_swr_start``, ``matched_swr_end``: the single SWR paired with
  the event (the containing SWR for ``swr_contained``, the earliest
  overlapping SWR for ``swr_partial``, ``<NA>`` / ``NaN`` for ``non_swr``).
- **Identifiers** — ``animal``, ``day``, ``epoch``, ``non_local_event``,
  and the stable hash ``event_id``.

SWR intervals are *always* derived from the ``SWR_event`` timebin labels
in the input frame via :func:`intervals_from_labels`. This is the same
label source used by the Figure 2 bar / triggered-average / timing panels
and by the replay-content tables, so a single overlap convention governs
the whole figure. The threshold-crossing ``ripple_times`` produced by
``load_data`` is not used for classification anywhere — only for plotting.

The replay-content tables in ``analysis/replay/tables.py`` build on top of
this table: ``build_event_table`` composes its legacy filtered-window
aggregates with the classification columns produced here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.replay._helpers import (
    hash_event_id,
    validate_event_ids,
)

from .event_detection import classify_swr_overlap_geometry

_SESSION_COLS = ["animal", "day", "epoch"]

_REQUIRED_DOT_COLUMNS = (
    "animal",
    "day",
    "epoch",
    "non_local_event",
    "SWR_event",
    "head_speed",
    "Non-Local_Continuous_prob",
    "Non-Local_Fragmented_prob",
)


def intervals_from_labels(session_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Return ``[label_col, start_time, end_time]`` for positive labels.

    Converts per-timebin integer labels (0 = background, positive ints =
    distinct events) into one interval per label, taking the earliest and
    latest timebin time (seconds) carrying each label. The label id is
    retained in the output so callers can recover, e.g., the matched
    SWR's id.

    Parameters
    ----------
    session_df : pd.DataFrame
        Single-session frame carrying ``label_col``. Time is taken from a
        ``"time"`` column when present (test frames), otherwise from the
        (named or unnamed) index (production ``data_over_time`` frames).
    label_col : str
        Integer label column (e.g. ``"non_local_event"`` or
        ``"SWR_event"``).

    Returns
    -------
    pd.DataFrame
        Columns ``[label_col, "start_time", "end_time"]``, one row per
        positive label. Empty input (no positive labels) returns a
        schema-complete zero-row frame.
    """
    if "time" in session_df.columns:
        df = session_df
    else:
        df = session_df.reset_index(names="time")

    active = df.loc[df[label_col] > 0]
    if len(active) == 0:
        return pd.DataFrame(columns=[label_col, "start_time", "end_time"])

    return (
        active.groupby(label_col)["time"]
        .agg(start_time="min", end_time="max")
        .reset_index()
    )


def _classified_event_table_columns() -> list[str]:
    """Canonical column order for the classified non-local event table."""
    return [
        "animal",
        "day",
        "epoch",
        "non_local_event",
        "event_id",
        "event_start",
        "event_end",
        "event_duration",
        "event_midpoint_time",
        "max_speed",
        "mean_speed",
        "start_speed",
        "end_speed",
        "endpoints_speed_below_thresh",
        "peak_nonlocal_prob",
        "is_swr",
        "overlap_class",
        "overlap_subclass",
        "n_swrs_overlapping",
        "matched_swr_event",
        "matched_swr_start",
        "matched_swr_end",
    ]


def _match_swr(
    overlap_class: str,
    event_start: float,
    event_end: float,
    swr_labels: np.ndarray,
    swr_starts: np.ndarray,
    swr_ends: np.ndarray,
) -> tuple[int | None, float, float]:
    """Pick the single SWR paired with one event.

    Returns ``(label, start, end)``: the integer SWR label (``None`` when
    unmatched) and its start/end times in seconds (``nan`` when unmatched).
    The label is kept as a Python ``int``/``None`` — never round-tripped
    through float — so the caller's nullable ``Int64`` cast is exact.

    - ``swr_contained`` : the containing SWR (``start <= event_start`` and
      ``end >= event_end``); earliest by ``start`` if several.
    - ``swr_partial`` : the earliest strictly-overlapping SWR (positive
      duration) by ``start``.
    - ``non_swr`` : unmatched.
    """
    if overlap_class == "non_swr":
        return (None, np.nan, np.nan)

    # Strict positive-duration overlap — byte-for-byte the predicate
    # classify_swr_overlap_geometry uses, applied to the same SWR
    # intervals. swr_contained additionally requires a single containing
    # SWR (boundary equality allowed), matching the classifier's
    # containment test.
    overlap_mask = (swr_ends > event_start) & (swr_starts < event_end)
    if overlap_class == "swr_contained":
        candidate_mask = (
            overlap_mask & (swr_starts <= event_start) & (swr_ends >= event_end)
        )
    else:  # swr_partial
        candidate_mask = overlap_mask

    idx = np.flatnonzero(candidate_mask)
    if len(idx) == 0:
        # The classifier and this matcher share the same predicate and
        # the same SWR intervals, so a non-"non_swr" class with no
        # matching SWR is a classifier/matcher divergence (boundary
        # convention drift, NaN bounds, or mutated event times), not a
        # data condition. Fail loudly rather than emit a row that is
        # is_swr=True with NaN matched-SWR columns.
        raise AssertionError(
            f"event [{event_start}, {event_end}] classified {overlap_class!r} "
            f"but no SWR satisfies the matcher's overlap predicate — "
            f"classify_swr_overlap_geometry and _match_swr have diverged."
        )
    chosen = int(idx[np.argmin(swr_starts[idx])])  # earliest by start time
    return (
        int(swr_labels[chosen]),
        float(swr_starts[chosen]),
        float(swr_ends[chosen]),
    )


def build_classified_nonlocal_event_table(
    dot: pd.DataFrame,
    *,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.DataFrame:
    """One row per non-local event with timing, overlap class, and per-event summaries.

    Parameters
    ----------
    dot : pd.DataFrame
        Time-indexed ``data_over_time`` slice for one or more sessions. Must
        carry ``animal``, ``day``, ``epoch``, ``non_local_event``,
        ``SWR_event``, ``head_speed``, ``Non-Local_Continuous_prob``,
        ``Non-Local_Fragmented_prob``.
    speed_threshold : float
        Immobility threshold (cm/s) used to populate
        ``endpoints_speed_below_thresh``.

    Returns
    -------
    pd.DataFrame
        Columns: ``animal, day, epoch, non_local_event, event_id,
        event_start, event_end, event_duration, event_midpoint_time,
        max_speed, mean_speed, start_speed, end_speed,
        endpoints_speed_below_thresh, peak_nonlocal_prob, is_swr,
        overlap_class, overlap_subclass, n_swrs_overlapping,
        matched_swr_event, matched_swr_start, matched_swr_end``.
        Empty frame (correct columns, zero rows) if no events.

    Notes
    -----
    Events are built over the *full* set of timebins carrying each event
    label — no immobility or probability filter is applied. SWR intervals
    are derived from ``dot["SWR_event"]`` labels via
    :func:`intervals_from_labels` and are session-scoped, so cross-session
    leakage is impossible by construction.

    Units: all ``*_start`` / ``*_end`` / ``*_time`` columns and
    ``event_duration`` are in seconds; the four ``*_speed`` columns are in
    cm/s.
    """
    missing = [c for c in _REQUIRED_DOT_COLUMNS if c not in dot.columns]
    if missing:
        raise KeyError(
            f"build_classified_nonlocal_event_table: dot is missing required "
            f"columns {missing}. Pass a data_over_time frame produced by "
            f"get_data_over_time()."
        )

    column_order = _classified_event_table_columns()

    if (dot["non_local_event"] > 0).sum() == 0:
        return pd.DataFrame(columns=column_order)

    session_frames: list[pd.DataFrame] = []

    for session_key, session_dot in dot.groupby(_SESSION_COLS, sort=False):
        animal, day, epoch = session_key
        event_rows = session_dot.loc[session_dot["non_local_event"] > 0]
        if len(event_rows) == 0:
            continue

        records: list[dict] = []
        for ev_id, g in event_rows.groupby("non_local_event", sort=True):
            times = g.index.to_numpy()
            speed = g["head_speed"]
            start_speed = float(speed.iloc[0])
            end_speed = float(speed.iloc[-1])
            nonlocal_prob = (
                g["Non-Local_Continuous_prob"] + g["Non-Local_Fragmented_prob"]
            )
            records.append(
                {
                    "animal": animal,
                    "day": day,
                    "epoch": epoch,
                    "non_local_event": int(ev_id),
                    "event_start": float(times[0]),
                    "event_end": float(times[-1]),
                    "event_duration": float(times[-1] - times[0]),
                    "event_midpoint_time": float(times[len(times) // 2]),
                    "max_speed": float(speed.max()),
                    "mean_speed": float(speed.mean()),
                    "start_speed": start_speed,
                    "end_speed": end_speed,
                    "endpoints_speed_below_thresh": bool(
                        (start_speed < speed_threshold) & (end_speed < speed_threshold)
                    ),
                    "peak_nonlocal_prob": float(nonlocal_prob.max()),
                }
            )

        events_df = pd.DataFrame.from_records(records)

        # --- SWR overlap classification (single canonical rule) ---
        swr_intervals = intervals_from_labels(session_dot, "SWR_event")
        classifier_events = events_df.rename(
            columns={"event_start": "start_time", "event_end": "end_time"}
        )
        classified = classify_swr_overlap_geometry(classifier_events, swr_intervals)
        events_df["overlap_class"] = classified["overlap_class"].to_numpy()
        events_df["overlap_subclass"] = classified["overlap_subclass"].to_numpy()
        events_df["n_swrs_overlapping"] = classified["n_swrs_overlapping"].to_numpy()
        events_df["is_swr"] = events_df["overlap_class"].to_numpy() != "non_swr"

        # --- Matched SWR per event ---
        swr_labels = swr_intervals["SWR_event"].to_numpy()
        swr_starts = swr_intervals["start_time"].to_numpy()
        swr_ends = swr_intervals["end_time"].to_numpy()
        matched = [
            _match_swr(
                cls,
                ev_start,
                ev_end,
                swr_labels,
                swr_starts,
                swr_ends,
            )
            for cls, ev_start, ev_end in zip(
                events_df["overlap_class"].to_numpy(),
                events_df["event_start"].to_numpy(),
                events_df["event_end"].to_numpy(),
                strict=True,
            )
        ]
        events_df["matched_swr_event"] = pd.array(
            [label for label, _, _ in matched], dtype="Int64"
        )
        events_df["matched_swr_start"] = np.array(
            [start for _, start, _ in matched], dtype=float
        )
        events_df["matched_swr_end"] = np.array(
            [end for _, _, end in matched], dtype=float
        )

        session_frames.append(events_df)

    if not session_frames:
        return pd.DataFrame(columns=column_order)

    out = pd.concat(session_frames, ignore_index=True)
    out["event_id"] = hash_event_id(out)
    validate_event_ids(out)

    return out[column_order]
