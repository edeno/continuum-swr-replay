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
- **Per-event speed summaries** — ``animal_max_speed_cm_s``,
  ``animal_mean_speed_cm_s``, ``animal_start_speed_cm_s``,
  ``animal_end_speed_cm_s``, the canonical
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
The labels are the paper's SWR source's: callers read a decode's frame through
:func:`~continuum_swr_replay.analysis.event_definition.analyzed_time.read_dot_with_swr_source`
(or :func:`~continuum_swr_replay.analysis.event_definition.analyzed_time.with_swr_source`),
which puts the direct Kay inventory's labels in ``SWR_event`` in place of the
decode's stored ``RippleTimesV1`` ones. This is the same label source used by
the Figure 2 bar / triggered-average / timing panels and by the replay-content
tables, so a single population and overlap convention governs the whole figure.

The replay-content tables in ``analysis/event_tables/tables_*.py`` build on top of
this table: ``build_event_table`` computes aggregates over this same complete,
atomically validated label window and merges the classification columns here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_probability,
)
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.event_ids import hash_event_id, validate_event_ids
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

from .immobility import (
    event_speeds_are_immobile,
    intervals_from_labels,
    validate_nonlocal_event_probability_contract,
)
from .nonlocal_event_rule import (
    classify_swr_overlap,
    classify_swr_overlap_geometry,
    strict_overlap_matrix,
)

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


def match_swr(
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

    # Strict positive-duration overlap — the predicate
    # classify_swr_overlap_geometry uses (strict_overlap_matrix), applied to
    # the same SWR intervals. swr_contained additionally requires a single
    # containing SWR (boundary equality allowed), matching the classifier's
    # containment test.
    overlap_mask = strict_overlap_matrix(
        np.array([event_start]), np.array([event_end]), swr_starts, swr_ends
    )[0]
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
            f"classify_swr_overlap_geometry and match_swr have diverged."
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
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One row per non-local event with timing, overlap class, and per-event summaries.

    Parameters
    ----------
    dot : pd.DataFrame
        ``data_over_time`` slice for one or more sessions, its time the index
        (increasing within a session, as the SWR source's read checks), read
        through the SWR source so that ``SWR_event`` is the paper's SWRs. Must
        carry ``animal``, ``day``, ``epoch``, ``non_local_event``,
        ``SWR_event``, ``head_speed``, ``Non-Local_Continuous_prob``,
        ``Non-Local_Fragmented_prob``.
    nonlocal_threshold : float
        Minimum combined Non-Local posterior required at every labeled bin.
        This must match the threshold used to create or validate the primary
        Non-Local event cohort.

    Returns
    -------
    pd.DataFrame
        Columns: ``animal, day, epoch, non_local_event, event_id,
        event_start, event_end, event_duration, event_midpoint_time,
        animal_max_speed_cm_s, animal_mean_speed_cm_s,
        animal_start_speed_cm_s, animal_end_speed_cm_s,
        whole_event_speed_eligible, endpoints_speed_below_thresh,
        peak_nonlocal_prob, is_swr,
        overlap_class, overlap_subclass, n_swrs_overlapping,
        matched_swr_event, matched_swr_start, matched_swr_end``.
        Empty frame (correct columns, zero rows) if no events.

    Notes
    -----
    Events are built over the *full* set of timebins carrying each Non-Local
    label so their speed eligibility remains directly inspectable. SWR
    intervals, which the read restricted atomically by the canonical
    whole-event speed rule, are derived from ``dot["SWR_event"]`` labels via
    :func:`intervals_from_labels` and are session-scoped, so cross-session
    leakage is impossible by construction. The two speed flags use the
    canonical ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC`` (4 cm/s), the
    threshold the read gated the SWRs at.

    Units: all ``*_start`` / ``*_end`` / ``*_time`` columns and
    ``event_duration`` are in seconds; the four ``animal_*_speed_cm_s``
    columns are in cm/s.
    """
    require_columns(
        dot,
        _REQUIRED_DOT_COLUMNS,
        "dot (a data_over_time frame read through the SWR source)",
    )

    validate_nonlocal_event_probability_contract(
        dot,
        probability_threshold=nonlocal_threshold,
    )

    column_order = list(_CLASSIFIED_EVENT_TABLE_COLUMNS)

    # SWR coincidence uses the read's SWR population, which the speed rule
    # already gated (``analyzed_time.with_swr_source`` drops a moving SWR), as
    # every event table does.
    overlap_sessions = dot.groupby(list(SESSION_COLUMNS_DAY), sort=False)

    if (dot["non_local_event"] > 0).sum() == 0:
        return pd.DataFrame(columns=column_order)

    session_frames: list[pd.DataFrame] = []

    for session_key, session_dot in dot.groupby(list(SESSION_COLUMNS_DAY), sort=False):
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
            nonlocal_prob = nonlocal_probability(g)
            records.append(
                {
                    "animal": animal,
                    "day": day,
                    "epoch": epoch,
                    "non_local_event": int(ev_id),
                    "event_start": float(times[0]),
                    "event_end": float(times[-1]),
                    "event_duration": float(
                        times[-1] - times[0] + 1.0 / SAMPLING_FREQUENCY
                    ),
                    "event_midpoint_time": float(times[len(times) // 2]),
                    "animal_max_speed_cm_s": float(speed.max()),
                    "animal_mean_speed_cm_s": float(speed.mean()),
                    "animal_start_speed_cm_s": start_speed,
                    "animal_end_speed_cm_s": end_speed,
                    "whole_event_speed_eligible": event_speeds_are_immobile(
                        speed.to_numpy(dtype=float),
                        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
                    ),
                    "endpoints_speed_below_thresh": bool(
                        (start_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC)
                        & (end_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC)
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
            match_swr(
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
    chosen: dict, swr_intervals: pd.DataFrame
) -> None:
    """Raise if a chosen example's class disagrees with the SWR spans drawn for it.

    Figure 2 panel a labels each example column by its ``overlap_class`` (and
    Figure 3 panel a shows a ``non_swr`` example) and shades the SWR spans of
    the same session. Both must come from one SWR
    population: the shading is the session's SWR source intervals, the sampled
    bounds of the source-read ``SWR_event`` labels
    (``analyzed_time.swr_intervals``), which are the intervals the class was
    computed against. Reclassifying each example against the drawn spans must
    therefore give its header class back; if it does not, the spans come from
    another population (the stored ``ripple_times``, say, or another session),
    and the drawn span would contradict the column header.

    Lives here (a DB-free module) rather than in the figure script so it is
    importable and unit-testable without pulling in ``load_data`` /
    DataJoint.

    Parameters
    ----------
    chosen : dict
        Mapping of expected overlap class -> an object indexable by
        ``"start_time"`` / ``"end_time"`` (the per-class example rows).
    swr_intervals : pd.DataFrame
        The SWR spans the panel draws, with ``start_time`` / ``end_time``.

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
    drawn = classify_swr_overlap(chosen_df, swr_intervals)
    mismatch = drawn[drawn["overlap_class"] != drawn["expected_class"]]
    if not mismatch.empty:
        rows = mismatch[["expected_class", "overlap_class", "start_time", "end_time"]]
        raise AssertionError(
            "Panel-a example(s) classify differently against the SWR spans "
            "panel a draws than their column header:\n"
            f"{rows.to_string(index=False)}\n"
            "The drawn SWR span would contradict the column header. The spans "
            "must be the session's SWR source intervals "
            "(analyzed_time.swr_intervals on the source-read data_over_time), the "
            "ones the classes were computed against."
        )
