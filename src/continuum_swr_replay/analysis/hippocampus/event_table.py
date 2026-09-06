"""Canonical per-non-local-event classification table.

This module owns the single, figure-and-model-agnostic definition of a
non-local event's timing and its overlap relationship with detected
sharp-wave ripples (SWRs). One row per non-local event over its complete
event-label window: the label's probability contract is *validated* rather
than re-filtered, so no non-local event is dropped or clipped here, while
the SWR comparison population is restricted by the canonical whole-event
speed rule (see below).

The public entry point :func:`build_classified_nonlocal_event_table`
takes a time-indexed ``data_over_time`` slice and emits, per event:

- **Timing** — ``event_start``, ``event_end``, ``event_duration``,
  ``event_midpoint_time`` over the *full* set of timebins carrying the
  event label.
- **Per-event speed summaries** — ``max_speed``, ``mean_speed``,
  ``start_speed``, ``end_speed``, the canonical
  ``whole_event_speed_eligible`` flag, and the legacy endpoint diagnostic
  ``endpoints_speed_below_thresh``.
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

SWR intervals are derived from ``SWR_event`` timebin labels after applying
the same whole-event speed criterion used for Non-Local and MUA populations.
This is the same label source used by the Figure 2 bar / triggered-average /
timing panels and by the replay-content tables, so a single population and
overlap convention governs the whole figure.

The replay-content tables in ``analysis/replay/tables_*.py`` build on top of
this table: ``build_event_table`` computes aggregates over this same complete,
atomically validated label window and merges the classification columns here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
    DEFAULT_SAMPLING_FREQUENCY_HZ,
)
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

from .event_detection import (
    classify_swr_overlap,
    classify_swr_overlap_geometry,
    positive_duration_interval_mask,
)
from .event_immobility import (
    event_speeds_are_immobile,
    filter_event_labels_by_speed,
    intervals_from_labels,
    validate_nonlocal_event_probability_contract,
)

# NOTE: ``hash_event_id`` / ``validate_event_ids`` live in
# ``analysis/replay/_helpers``. Importing them at module load would close a
# cycle — ``replay/__init__`` eagerly imports the replay table builders, which
# imports this module — so they are imported lazily inside
# :func:`build_classified_nonlocal_event_table` instead. The natural
# dependency direction (replay builds on hippocampus) stays at module
# scope; only this upward reference is deferred to call time.

_SESSION_COLS = list(SESSION_COLUMNS_DAY)

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


#: Canonical column order for the classified non-local event table.
_CLASSIFIED_EVENT_TABLE_COLUMNS: tuple[str, ...] = (
    "animal",
    "day",
    "epoch",
    "non_local_event",
    "event_id",
    "event_start",
    "event_end",
    "event_duration",
    "event_midpoint_time",
    "animal_max_speed_cm_s",
    "animal_mean_speed_cm_s",
    "animal_start_speed_cm_s",
    "animal_end_speed_cm_s",
    "whole_event_speed_eligible",
    "endpoints_speed_below_thresh",
    "peak_nonlocal_prob",
    "is_swr",
    "overlap_class",
    "overlap_subclass",
    "n_swrs_overlapping",
    "matched_swr_event",
    "matched_swr_start",
    "matched_swr_end",
)


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
    overlap_mask = (
        positive_duration_interval_mask(swr_starts, swr_ends)
        & (swr_ends > event_start)
        & (swr_starts < event_end)
    )
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
        # is_swr=True with NaN matched-SWR columns. RuntimeError (not
        # AssertionError) so the guard survives ``python -O``.
        raise RuntimeError(
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
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
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
        Immobility threshold (cm/s) used for the canonical whole-event flag
        and the legacy endpoint diagnostic.
    nonlocal_threshold : float
        Minimum combined Non-Local posterior required at every labeled bin.
        This must match the threshold used to create or validate the primary
        Non-Local event cohort.

    Returns
    -------
    pd.DataFrame
        Columns: ``animal, day, epoch, non_local_event, event_id,
        event_start, event_end, event_duration, event_midpoint_time,
        max_speed, mean_speed, start_speed, end_speed,
        whole_event_speed_eligible, endpoints_speed_below_thresh,
        peak_nonlocal_prob, is_swr,
        overlap_class, overlap_subclass, n_swrs_overlapping,
        matched_swr_event, matched_swr_start, matched_swr_end``.
        Empty frame (correct columns, zero rows) if no events.

    Notes
    -----
    Events are built over the *full* set of timebins carrying each Non-Local
    label so their speed eligibility remains directly inspectable. SWR
    intervals are restricted atomically by the canonical whole-event speed
    rule and then derived from ``dot["SWR_event"]`` labels via
    :func:`intervals_from_labels` and are session-scoped, so cross-session
    leakage is impossible by construction.

    Units: all ``*_start`` / ``*_end`` / ``*_time`` columns and
    ``event_duration`` are in seconds; the four ``*_speed`` columns are in
    cm/s.
    """
    from continuum_swr_replay.analysis.replay._helpers import (
        hash_event_id,
        validate_event_ids,
    )

    missing = [c for c in _REQUIRED_DOT_COLUMNS if c not in dot.columns]
    if missing:
        raise KeyError(
            f"build_classified_nonlocal_event_table: dot is missing required "
            f"columns {missing}. Pass a data_over_time frame produced by "
            f"get_data_over_time()."
        )

    validate_nonlocal_event_probability_contract(
        dot,
        probability_threshold=nonlocal_threshold,
    )

    column_order = list(_CLASSIFIED_EVENT_TABLE_COLUMNS)

    # Single canonical time source. Event timing below reads ``g.index``
    # while :func:`intervals_from_labels` prefers a ``"time"`` column when
    # one is present. If a frame carried both a ``"time"`` column and a
    # different index, the two reads would classify events against
    # different timebases. Promote a ``"time"`` column to the index so
    # both paths read the same axis (no-op for production data_over_time,
    # which is already time-indexed with no ``"time"`` column).
    if "time" in dot.columns:
        dot = dot.set_index("time")

    # SWR coincidence uses the same globally qualified SWR population as its
    # own event tables. Preserve Non-Local rows here so this diagnostic table
    # can report their eligibility flag; model builders validate that flag.
    overlap_dot = filter_event_labels_by_speed(
        dot,
        ["SWR_event"],
        speed_threshold=speed_threshold,
    )
    overlap_sessions = overlap_dot.groupby(_SESSION_COLS, sort=False)

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
                    "event_duration": float(
                        times[-1] - times[0] + 1.0 / DEFAULT_SAMPLING_FREQUENCY_HZ
                    ),
                    "event_midpoint_time": float(times[len(times) // 2]),
                    "animal_max_speed_cm_s": float(speed.max()),
                    "animal_mean_speed_cm_s": float(speed.mean()),
                    "animal_start_speed_cm_s": start_speed,
                    "animal_end_speed_cm_s": end_speed,
                    "whole_event_speed_eligible": event_speeds_are_immobile(
                        speed.to_numpy(dtype=float),
                        speed_threshold=speed_threshold,
                    ),
                    "endpoints_speed_below_thresh": bool(
                        (start_speed < speed_threshold) & (end_speed < speed_threshold)
                    ),
                    "peak_nonlocal_prob": float(nonlocal_prob.max()),
                }
            )

        events_df = pd.DataFrame.from_records(records)

        # --- SWR overlap classification (single canonical rule) ---
        # Select from a grouping built once rather than rescanning the pooled
        # frame for every session. Session keys, rather than the time index,
        # are required because concatenated sessions routinely reuse times.
        session_overlap_dot = overlap_sessions.get_group(session_key)
        swr_intervals = intervals_from_labels(session_overlap_dot, "SWR_event")
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


def assert_drawn_class_matches_ripples(
    chosen: dict, ripple_times: pd.DataFrame
) -> None:
    """Raise if a chosen example's label-derived class disagrees with the
    threshold-crossing ``ripple_times`` it would be drawn against.

    Figure 2 panel a labels each example column by its label-derived
    ``overlap_class`` (from the ``SWR_event`` timebin labels) but shades the
    SWR span from the threshold-crossing ``ripple_times``. These are two
    independent discretizations of the same ripples and can disagree by
    ~one sample at a boundary; if they do for a chosen example, the drawn
    span would contradict the column header.

    Lives here (a DB-free module) rather than in the figure script so it is
    importable and unit-testable without pulling in ``load_data`` /
    DataJoint.

    Parameters
    ----------
    chosen : dict
        Mapping of expected overlap class -> an object indexable by
        ``"start_time"`` / ``"end_time"`` (the per-class example rows).
    ripple_times : pd.DataFrame
        Threshold-crossing SWR intervals with ``start_time`` / ``end_time``.

    Raises
    ------
    AssertionError
        If any chosen example's drawn class differs from its expected class.
    """
    chosen_df = pd.DataFrame(
        [
            {
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "expected_class": cls,
            }
            for cls, ev in chosen.items()
        ]
    )
    drawn = classify_swr_overlap(chosen_df, ripple_times)
    mismatch = drawn[drawn["overlap_class"] != drawn["expected_class"]]
    if not mismatch.empty:
        rows = mismatch[["expected_class", "overlap_class", "start_time", "end_time"]]
        raise AssertionError(
            "Panel-a example(s) classify differently against the threshold-"
            "crossing ripple_times that panel a shades than the label-derived "
            "column header:\n"
            f"{rows.to_string(index=False)}\n"
            "The drawn SWR span would contradict the column header. Pick a "
            "different example for the affected class(es), or widen the "
            "selection's peak/duration criteria so a non-boundary event is "
            "chosen."
        )
