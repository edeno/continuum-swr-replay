"""Null-distribution generators for replay content QC.

Three shuffles specified in plan §Stage 0:

1. :func:`circular_time_shuffle` — permute timebin order within each
   event, preserving the marginal arm distribution. Primary null for
   ``path_continuity`` and sequence metrics.
2. :func:`stop_label_shuffle` — randomly permute behavioral labels
   across stops within a session, preserving events-per-stop counts.
   Null for replay–behavior alignment analyses (Stages 5–6).
3. :func:`next_arm_circular_shift` — circularly shift ``next_arm``
   (and associated post-hoc columns) across stops within a session.
   Null for Stage 5B retrospective alignment, guarding against patch
   geometry or block-drift correlations.

All three return a NEW DataFrame (or array) — the input is never
modified in place. Each accepts a ``seed`` or ``rng`` for
reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── 1. Within-event circular time shuffle ──


def circular_time_shuffle(
    decoded_arms: np.ndarray,
    event_ids: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Circularly permute timebin order within each event.

    For each event, the ``decoded_arms`` sequence is rolled by a
    random offset (uniform in ``[1, T_e - 1]`` so the original order
    is never returned). The marginal distribution of arm IDs per
    event is preserved; only the sequential structure is destroyed.

    Parameters
    ----------
    decoded_arms : np.ndarray, shape (T_total,)
        Per-timebin MAP arm across all in-event timebins
        (``event_sequence_table.decoded_arm_map``).
    event_ids : np.ndarray, shape (T_total,)
        Event label per timebin (same length as ``decoded_arms``).
    rng : numpy Generator, optional
        For reproducibility. Default: fresh generator.

    Returns
    -------
    np.ndarray, shape (T_total,)
        Shuffled ``decoded_arms`` — same length, same per-event
        marginal arm distribution, different sequential order.
    """
    if rng is None:
        rng = np.random.default_rng()
    decoded_arms = np.asarray(decoded_arms).copy()
    unique_events = np.unique(event_ids)
    for ev in unique_events:
        mask = event_ids == ev
        n = int(mask.sum())
        if n < 2:
            continue
        shift = rng.integers(1, n)  # [1, n-1] so original is excluded
        decoded_arms[mask] = np.roll(decoded_arms[mask], shift)
    return decoded_arms


# ── 2. Within-session stop-label shuffle ──


def stop_label_shuffle(
    stop_table: pd.DataFrame,
    *,
    columns: tuple[str, ...] = (
        "is_after_reward",
        "is_after_omission",
        "is_after_switch",
        "is_after_stay",
        "reward_outcome",
        "current_arm",
        "prev_arm",
        "last_rewarded_arm",
        "current_patch",
        "prev_patch",
    ),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Randomly permute behavioral labels across stops within a session.

    The number of events per stop (``n_events``) and the event
    content itself are untouched — only the LABELS that map behavior
    to stops change. This breaks the alignment between "what the
    animal did" and "which stop the replay occurred in" while
    preserving the marginal distribution of both.

    Parameters
    ----------
    stop_table : pd.DataFrame
        One session's ``stop_table``.
    columns : tuple of str
        Which columns to shuffle. The default covers the behavioral
        context and arm labels the hurdle / rate models use.
    rng : numpy Generator, optional

    Returns
    -------
    pd.DataFrame
        Copy of ``stop_table`` with the specified columns permuted
        across rows (stops).
    """
    if rng is None:
        rng = np.random.default_rng()
    # Guard: stop_table must be a single session. Cross-session
    # shuffle would swap labels between sessions, which is never
    # the intent (reviewer flag #10).
    session_cols = ["animal", "day", "epoch"]
    available = [c for c in session_cols if c in stop_table.columns]
    if available:
        n_sessions = stop_table[available].drop_duplicates().shape[0]
        if n_sessions > 1:
            raise ValueError(
                f"stop_label_shuffle received {n_sessions} sessions; "
                "pass one session at a time to avoid cross-session "
                "label contamination."
            )
    out = stop_table.copy()
    perm = rng.permutation(len(out))
    for col in columns:
        if col in out.columns:
            out[col] = out[col].to_numpy()[perm]
    return out


# ── 3. Within-session next_arm circular shift ──


def next_arm_circular_shift(
    event_table: pd.DataFrame,
    *,
    columns: tuple[str, ...] = (
        "next_arm",
        "next_patch",
        "is_stay",
    ),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Circularly shift ``next_arm`` and associated post-hoc columns.

    Orders events by ``event_midpoint_time``, picks a random shift
    ``k ∈ [1, n_events - 1]``, and rolls the post-hoc columns by
    ``k`` positions. This preserves the marginal distribution of
    ``next_arm`` across the session AND the temporal ordering of
    "next" labels (slow drifts in patch preference are carried over),
    but breaks the event-specific alignment between replay content
    and what the animal actually did next.

    Designed specifically as the null for Stage 5B retrospective
    alignment, where the concern is that patch geometry or block-drift
    correlations inflate alignment metrics.

    Parameters
    ----------
    event_table : pd.DataFrame
        One session's ``event_table``, sorted by
        ``event_midpoint_time`` (or sortable).
    columns : tuple of str
        Which columns to shift. Default includes ``next_arm``,
        ``next_patch``, and ``is_stay`` — all the post-hoc variables.
    rng : numpy Generator, optional

    Returns
    -------
    pd.DataFrame
        Copy of ``event_table`` with the post-hoc columns circularly
        shifted.
    """
    if rng is None:
        rng = np.random.default_rng()
    session_cols = ["animal", "day", "epoch"]
    available = [c for c in session_cols if c in event_table.columns]
    if available:
        n_sessions = event_table[available].drop_duplicates().shape[0]
        if n_sessions > 1:
            raise ValueError(
                f"next_arm_circular_shift received {n_sessions} sessions; "
                "pass one session at a time to avoid cross-session label "
                "contamination."
            )
    out = event_table.sort_values("event_midpoint_time").copy()
    n = len(out)
    if n < 2:
        return out
    shift = int(rng.integers(1, n))
    for col in columns:
        if col in out.columns:
            out[col] = np.roll(out[col].to_numpy(), shift)
    return out
