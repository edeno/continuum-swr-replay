"""Scientific table contracts beyond the generic covariate schema stamp.

The naming registry certifies vocabulary and row-key uniqueness.  It cannot
certify that an event has the complete task choice set, that response counts
are conserved, or that a partially missing leaf family has not been silently
averaged.  Those are scientific properties of the table, and are validated
here at the producer/reducer boundary.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import ALL_ARMS
from continuum_swr_replay.session_keys import SESSION_COLUMNS

from .summaries import PROBABILITY_ROW_SUM_ATOL

_PATCHES = (1, 2, 3)
_LEAF_SUPPORT_COLUMNS = (
    "visit_recency_trials",
    "is_candidate_arm_never_visited",
    "reward_count_candidate_arm_last10_known_trials",
    "reward_fraction_candidate_arm_last10_visits",
    "task_reward_probability_candidate_arm",
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_belief_variance_candidate_arm_preoutcome",
)


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, table_name: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns {missing}")


def _require_stable_event_context(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    present = [column for column in columns if column in frame.columns]
    if not present or frame.empty:
        return
    varying = (
        frame.groupby("event_id", sort=False, dropna=False)[present]
        .nunique(dropna=False)
        .gt(1)
    )
    if varying.any(axis=None):
        bad = varying.index[varying.any(axis=1)].tolist()[:5]
        raise ValueError(
            f"event_arm has candidate-row-varying event context for event_id(s) {bad}"
        )


def _require_exact_group_members(
    frame: pd.DataFrame,
    *,
    group_column: str,
    member_column: str,
    expected: tuple[int, ...],
    label: str,
) -> None:
    if frame.empty:
        return
    values = pd.to_numeric(frame[member_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & (values == np.floor(values))
    if not valid.all():
        raise ValueError(
            f"{label} requires a finite integer {member_column}; found invalid values"
        )

    expected_set = frozenset(expected)
    observed = frame.assign(_member=values.astype(np.int64)).groupby(
        group_column, sort=False, dropna=False
    )["_member"]
    bad: list[object] = []
    for group_id, members in observed:
        member_list = members.tolist()
        if len(member_list) != len(expected) or frozenset(member_list) != expected_set:
            bad.append(group_id)
            if len(bad) == 5:
                break
    if bad:
        raise ValueError(
            f"{label} requires the exact {member_column} set {tuple(expected)} once "
            f"per {group_column}; invalid {group_column}(s) {bad}"
        )


def _require_event_set_matches(
    derived: pd.DataFrame, event_table: pd.DataFrame, *, table_name: str
) -> None:
    _require_columns(event_table, ("event_id",), table_name="event_table")
    if (
        event_table["event_id"].isna().any()
        or event_table["event_id"].duplicated().any()
    ):
        raise ValueError("event_table requires one nonmissing row per event_id")
    expected = set(event_table["event_id"].tolist())
    observed = set(derived["event_id"].tolist())
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        unexpected = sorted(observed - expected)[:5]
        raise ValueError(
            f"{table_name} event-ID set does not match event_table; "
            f"missing={missing}, unexpected={unexpected}"
        )

    session_columns = [c for c in SESSION_COLUMNS if c in derived and c in event_table]
    if not session_columns or derived.empty:
        return
    reference = event_table[["event_id", *session_columns]]
    observed_context = derived[["event_id", *session_columns]].drop_duplicates()
    joined = observed_context.merge(
        reference,
        on="event_id",
        how="outer",
        suffixes=("_derived", "_event"),
        validate="one_to_one",
        indicator=True,
    )
    mismatch = joined["_merge"].ne("both").to_numpy()
    for column in session_columns:
        mismatch |= joined[f"{column}_derived"].ne(joined[f"{column}_event"]).to_numpy()
    if mismatch.any():
        bad = joined.loc[mismatch, "event_id"].tolist()[:5]
        raise ValueError(
            f"{table_name} session context disagrees with event_table for "
            f"event_id(s) {bad}"
        )


def _require_response_conservation(frame: pd.DataFrame) -> None:
    if "arm_count_hard" not in frame:
        return
    counts = pd.to_numeric(frame["arm_count_hard"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or not np.equal(counts, np.floor(counts)).all()
    ):
        raise ValueError("arm_count_hard must be finite, nonnegative integer counts")

    total_column = None
    for candidate in ("n_nonlocal_bins", "n_timebins"):
        if candidate in frame:
            total_column = candidate
            break
    if total_column is None or "n_undecoded_bins" not in frame:
        return

    context = frame.groupby("event_id", sort=False, dropna=False)[
        [total_column, "n_undecoded_bins"]
    ].first()
    expected = pd.to_numeric(context[total_column], errors="coerce") - pd.to_numeric(
        context["n_undecoded_bins"], errors="coerce"
    )
    observed = frame.groupby("event_id", sort=False, dropna=False)[
        "arm_count_hard"
    ].sum()
    invalid_expected = (
        ~np.isfinite(expected.to_numpy(dtype=float))
        | (expected.to_numpy(dtype=float) < 0)
        | (expected.to_numpy(dtype=float) != np.floor(expected.to_numpy(dtype=float)))
    )
    mismatch = invalid_expected | observed.ne(expected).to_numpy()
    if mismatch.any():
        bad = observed.index[mismatch].tolist()[:5]
        raise ValueError(
            "event_arm response counts do not conserve decoded bins for "
            f"event_id(s) {bad}"
        )


def _require_uniform_leaf_support(frame: pd.DataFrame) -> None:
    leaf = frame[frame["candidate_arm"].isin(range(6))]
    for column in _LEAF_SUPPORT_COLUMNS:
        if column not in leaf:
            continue
        finite = np.isfinite(pd.to_numeric(leaf[column], errors="coerce"))
        support = finite.groupby(leaf["event_id"], sort=False).sum()
        partial = support[(support != 0) & (support != 6)]
        if not partial.empty:
            raise ValueError(
                f"event_arm has partial leaf support for {column!r} in "
                f"event_id(s) {partial.index.tolist()[:5]}; canonical covariate "
                "families must be all-known or all-missing across six leaves"
            )


def _require_fraction_normalization(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    n_candidates: int,
    table_name: str,
) -> None:
    """Require each observed fraction family to cover the candidate set and sum one.

    Parameters
    ----------
    frame : pd.DataFrame
        Candidate-grain table, one row per (event, candidate).
    columns : tuple of str
        Fraction column names to check; absent ones are skipped.
    n_candidates : int
        Size of the candidate set an event must cover (9 arms, 3 patches).
    table_name : str
        Table label used in the raised messages.
    """
    for column in columns:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        finite = np.isfinite(values)
        if ((values[finite] < 0.0) | (values[finite] > 1.0)).any():
            raise ValueError(f"{column} must lie in [0, 1] where finite")
        support = (
            pd.Series(finite, index=frame.index)
            .groupby(frame["event_id"], sort=False)
            .sum()
        )
        partial = support[(support != 0) & (support != n_candidates)]
        if not partial.empty:
            raise ValueError(
                f"{table_name} has partial candidate support for {column!r} in "
                f"event_id(s) {partial.index.tolist()[:5]}"
            )
        totals = numeric.groupby(frame["event_id"], sort=False).sum(min_count=1)
        total_values = totals.to_numpy(dtype=float)
        # arm_fraction_posterior traces to the float32-backed decoder posterior,
        # which accumulates ~1e-6 of row-sum error over hundreds of bins; use the
        # same established tolerance summaries.py applies to that upstream quantity
        # (1e-10 would reject correctly-built tables from real GPU output).
        bad_total = np.isfinite(total_values) & ~np.isclose(
            total_values,
            1.0,
            rtol=0.0,
            atol=PROBABILITY_ROW_SUM_ATOL,
        )
        if bad_total.any():
            raise ValueError(
                f"{column} must sum to one within every observed event; "
                f"invalid event_id(s) {totals.index[bad_total].tolist()[:5]}"
            )


def validate_event_arm_table(
    frame: pd.DataFrame, *, event_table: pd.DataFrame | None = None
) -> None:
    """Validate the complete nine-choice scientific row contract."""
    _require_columns(
        frame,
        ("event_id", "candidate_arm"),
        table_name="event_arm",
    )
    _require_exact_group_members(
        frame,
        group_column="event_id",
        member_column="candidate_arm",
        expected=tuple(int(arm) for arm in ALL_ARMS),
        label="event_arm candidate-arm set",
    )
    _require_stable_event_context(
        frame,
        (
            *SESSION_COLUMNS,
            "stop_id",
            "occupancy_state",
            "at_well",
            "occupancy_well_id",
            "occupancy_well_arm",
            "current_patch",
            "event_start_s",
            "event_end_s",
        ),
    )
    if event_table is not None:
        _require_event_set_matches(frame, event_table, table_name="event_arm")
    _require_response_conservation(frame)
    _require_uniform_leaf_support(frame)
    _require_fraction_normalization(
        frame,
        columns=("arm_fraction", "arm_fraction_posterior"),
        n_candidates=len(ALL_ARMS),
        table_name="event_arm",
    )


def validate_event_patch_table(
    frame: pd.DataFrame, *, event_table: pd.DataFrame | None = None
) -> None:
    """Validate the complete three-patch scientific row contract."""
    _require_columns(frame, ("event_id", "patch"), table_name="event_patch")
    _require_exact_group_members(
        frame,
        group_column="event_id",
        member_column="patch",
        expected=_PATCHES,
        label="event_patch patch set",
    )
    _require_stable_event_context(frame, SESSION_COLUMNS)
    if event_table is not None:
        _require_event_set_matches(frame, event_table, table_name="event_patch")
    if "patch_count_hard" in frame:
        counts = pd.to_numeric(frame["patch_count_hard"], errors="coerce").to_numpy(
            dtype=float
        )
        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
            or not np.equal(counts, np.floor(counts)).all()
        ):
            raise ValueError(
                "patch_count_hard must be finite, nonnegative integer counts"
            )
    # patch_fraction is patch_count_hard over the event's real-arm bin total, so
    # it obeys the same allocation contract as arm_fraction: all three patches
    # observed or none (an event with no real-arm bins), and a unit sum where
    # observed.
    _require_fraction_normalization(
        frame,
        columns=("patch_fraction",),
        n_candidates=len(_PATCHES),
        table_name="event_patch",
    )
