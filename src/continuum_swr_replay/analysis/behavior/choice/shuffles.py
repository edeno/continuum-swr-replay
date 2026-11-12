"""Null-distribution generators for replay content QC.

Three shuffles:

1. :func:`circular_time_shuffle` — permute timebin order within each
   event, preserving the marginal arm distribution. Primary null for
   ``path_continuity`` and sequence metrics.
2. :func:`stop_label_shuffle` — randomly permute behavioral labels
   across stops within a session, preserving events-per-stop counts.
   Null for replay–behavior alignment analyses.
3. :func:`next_arm_circular_shift` — circularly shift ``next_arm``
   (and associated post-hoc columns) across visit contexts within a session.
   Null for the retrospective alignment metrics, guarding against patch
   geometry or block-drift correlations.

All three return a NEW DataFrame (or array) — the input is never
modified in place. Each accepts a ``seed`` or ``rng`` for
reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from .._helpers import stay_indicator_from_patches

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
        "is_rewarded_outcome",
        "did_last_completed_trip_switch_patch",
        "occupancy_well_arm",
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
    session_cols = list(SESSION_COLUMNS)
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
    ),
    unit_col: str | None = None,
    strata_cols: tuple[str, ...] = (),
    variation_cols: tuple[str, ...] | None = None,
    require_movement: bool = False,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Circularly shift future-choice labels at the stop/visit grain.

    Canonical retrospective analyses pass a derived contiguous-current-arm-run
    column as ``unit_col``. Every future-choice label is then read once per
    visit, shifted among visits, and broadcast back to all events in that visit.
    This prevents events in one visit receiving incompatible shuffled choices
    while allowing a long immobility bout that crosses a trial boundary—or
    returns to an earlier arm—to contain distinct visits. ``strata_cols``
    restricts shifts to scientifically exchangeable visits. Within each
    stratum, units are ordered by their earliest ``event_midpoint_time`` and a
    non-identity circular shift is drawn.

    When ``unit_col`` is omitted, ``stop_id`` is used if present; otherwise each
    row is treated as its own unit for backward-compatible use on generic event
    frames.  Strata with fewer than two units, or whose relevant labels cannot
    change, are left untouched.  Set ``require_movement=True`` to fail instead;
    publication analyses use that guard after excluding non-identifiable strata.

    Designed specifically as the null for the retrospective
    alignment, where the concern is that patch geometry or block-drift
    correlations inflate alignment metrics.

    Parameters
    ----------
    event_table : pd.DataFrame
        One session's ``event_table``, sorted by
        ``event_midpoint_time`` (or sortable).
    columns : tuple of str
        Which columns to shift. Default is the post-hoc "next-trip"
        labels (``next_arm``, ``next_patch``); ``current_*`` and
        ``prev_*`` are NOT rolled because they describe the source
        row's situation. ``is_stay`` is also not rolled — it's a
        derived ``current_patch == next_patch`` indicator and is
        recomputed below from the destination row's ``current_patch``
        and the shifted ``next_patch``. Rolling the source row's
        ``is_stay`` would silently keep the wrong current-patch
        reference.
    unit_col : str, optional
        Behavioral unit whose events must retain one common label. Defaults to
        ``stop_id`` when available, otherwise one unit per row.
    strata_cols : tuple of str
        Unit-level columns that must match before labels can be exchanged.
        Every value must be constant within a unit.
    variation_cols : tuple of str, optional
        Shifted labels that must each change somewhere in a stratum. Defaults to
        every present member of ``columns``.
    require_movement : bool, default False
        Raise when no stratum admits a non-identity label shift.
    rng : numpy Generator, optional

    Returns
    -------
    pd.DataFrame
        Copy of ``event_table`` with the post-hoc columns circularly
        shifted and ``is_stay`` (when present) recomputed against the
        destination row's ``current_patch``.
    """
    if rng is None:
        rng = np.random.default_rng()
    session_cols = list(SESSION_COLUMNS)
    available = [c for c in session_cols if c in event_table.columns]
    if available:
        n_sessions = event_table[available].drop_duplicates().shape[0]
        if n_sessions > 1:
            raise ValueError(
                f"next_arm_circular_shift received {n_sessions} sessions; "
                "pass one session at a time to avoid cross-session label "
                "contamination."
            )
    out = event_table.sort_values("event_midpoint_s", kind="stable").copy()
    if out.empty:
        return out

    label_cols = tuple(column for column in columns if column in out.columns)
    if not label_cols:
        return out
    relevant_cols = label_cols if variation_cols is None else variation_cols
    missing_variation = sorted(set(relevant_cols).difference(label_cols))
    if missing_variation:
        raise KeyError(
            "variation_cols must be present in the shifted columns; missing "
            f"{missing_variation}"
        )

    temporary_unit_col = "_next_label_shuffle_row_unit"
    if unit_col is None:
        unit_col = "stop_id" if "stop_id" in out.columns else temporary_unit_col
    if unit_col == temporary_unit_col:
        out[unit_col] = np.arange(len(out), dtype=np.int64)
    elif unit_col not in out.columns:
        raise KeyError(f"next-label shuffle requires unit column {unit_col!r}")
    if out[unit_col].isna().any():
        raise ValueError(f"next-label shuffle unit column {unit_col!r} contains NaN")

    required_unit_columns = tuple(dict.fromkeys((*label_cols, *strata_cols)))
    missing = sorted(set(required_unit_columns).difference(out.columns))
    if missing:
        raise KeyError(f"next-label shuffle is missing unit column(s): {missing}")
    grouped = out.groupby(unit_col, sort=False, dropna=False)
    for column in required_unit_columns:
        n_unique = grouped[column].nunique(dropna=False)
        inconsistent = n_unique[n_unique > 1]
        if not inconsistent.empty:
            example = inconsistent.index[0]
            raise ValueError(
                f"next-label shuffle requires one {column!r} value per "
                f"{unit_col}; unit {example!r} has multiple values"
            )

    unit_rows = out.drop_duplicates(unit_col, keep="first")[
        [unit_col, *required_unit_columns]
    ].copy()
    unit_time = grouped["event_midpoint_s"].min()
    unit_rows["_unit_time"] = unit_rows[unit_col].map(unit_time)
    unit_rows = unit_rows.sort_values("_unit_time", kind="stable").reset_index(
        drop=True
    )
    shuffled_labels = unit_rows[[unit_col, *label_cols]].copy()

    if strata_cols:
        strata = unit_rows.groupby(list(strata_cols), sort=False, dropna=False)
        stratum_positions = [np.asarray(indices) for indices in strata.indices.values()]
    else:
        stratum_positions = [np.arange(len(unit_rows), dtype=np.int64)]

    n_moved_strata = 0
    for positions in stratum_positions:
        if len(positions) < 2:
            continue
        ordered = unit_rows.iloc[positions]
        candidates: list[int] = []
        for shift in range(1, len(ordered)):
            if all(
                not ordered[column]
                .reset_index(drop=True)
                .equals(pd.Series(np.roll(ordered[column].to_numpy(), shift)))
                for column in relevant_cols
            ):
                candidates.append(shift)
        if not candidates:
            continue
        candidate_index = int(rng.integers(1, len(candidates) + 1)) - 1
        shift = candidates[candidate_index]
        for column in label_cols:
            shuffled_labels.loc[positions, column] = np.roll(
                ordered[column].to_numpy(), shift
            )
        n_moved_strata += 1

    if require_movement and n_moved_strata != len(stratum_positions):
        raise ValueError(
            "next-label shuffle could not move every requested stratum; "
            f"moved {n_moved_strata}/{len(stratum_positions)} strata"
        )

    label_lookup = shuffled_labels.set_index(unit_col)
    for column in label_cols:
        out[column] = out[unit_col].map(label_lookup[column])

    if temporary_unit_col in out.columns:
        out = out.drop(columns=temporary_unit_col)

    # Recompute ``is_stay`` from the destination row's
    # ``current_patch`` and the just-shifted ``next_patch`` so the
    # derived indicator stays consistent with the labels actually on
    # the row. NaN in either side propagates.
    if (
        "is_stay" in out.columns
        and "current_patch" in out.columns
        and "next_patch" in out.columns
    ):
        out["is_stay"] = stay_indicator_from_patches(
            out["next_patch"].to_numpy(dtype=float),
            out["current_patch"].to_numpy(dtype=float),
        )
    return out
