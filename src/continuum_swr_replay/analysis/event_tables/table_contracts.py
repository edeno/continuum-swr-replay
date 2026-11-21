"""Scientific table contracts beyond the generic covariate schema stamp.

The naming registry certifies vocabulary and row-key uniqueness.  It cannot
certify that an event has the complete task choice set, that response counts
are conserved, or that a partially missing leaf family has not been silently
averaged.  Those are scientific properties of the table, and are validated
here at the producer/reducer boundary.

The same reasoning covers the consumer boundary: what a fitting run must
establish about the table it reads and the fits it produced --
:func:`validate_requested_standardization`,
:func:`validate_canonical_table_contract` and
:func:`validate_and_match_fit_results` -- is a contract, not a fitting step,
so it lives here rather than inside a runner.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    LEAF_ARMS,
    N_LEAF_ARMS,
    TRACK_SEGMENT_TO_PATCH,
)
from continuum_swr_replay.paths import STANDARDIZATION_REFERENCE
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from continuum_swr_replay.tables.standardization import (
    assert_standardized_matches_reference,
    load_reference,
)

from ..constants import IN_WELL_VISIT, VISIT_CONTEXTS
from ._common import EVENT_ARM_STANDARDIZED_COLS
from .content_response import validate_fraction_response
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

# Consumer-side contract constants: the location states a canonical content
# table may carry, its per-hierarchy candidate set, and the cross-validation
# columns a fitted ladder must report.
_LOCATION_STATES = frozenset({"at_well", "in_transit", "mixed", "unknown"})
_CANDIDATE_COLUMNS = {
    "arm": ("candidate_arm", ALL_ARMS),
    "patch": ("patch", tuple(sorted(set(TRACK_SEGMENT_TO_PATCH.values())))),
}

CV_RESULT_COLUMNS = (
    "cv_ll_per_event_mean",
    "cv_ll_per_event_std",
    "cv_ll_per_timebin_mean",
    "cv_ll_per_timebin_std",
    "cv_n_folds",
    "cv_n_nonconverged_folds",
    "cv_n_separated_folds",
)
_CV_COUNT_COLUMNS = (
    "cv_n_folds",
    "cv_n_nonconverged_folds",
    "cv_n_separated_folds",
)


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


def require_visit_context_labels(frame: pd.DataFrame, table_name: str) -> None:
    """Check ``visit_context`` against its vocabulary and ``occupancy_state``.

    The non-local event tables carry the label (their build columns name it,
    ``_common.py``); the ContFrag tables, which share these validators, do not,
    and a table without ``visit_context`` is not checked. Where it is present
    every value is one of ``analysis.constants.VISIT_CONTEXTS``, and a row is
    ``in_visit`` exactly when its ``occupancy_state`` is ``at_well``.

    Raises
    ------
    ValueError
        If a label is missing or unknown, or disagrees with ``occupancy_state``.
    """
    if "visit_context" not in frame.columns or frame.empty:
        return
    labels = frame["visit_context"]
    unknown = sorted(set(labels.dropna().unique()) - set(VISIT_CONTEXTS))
    if labels.isna().any() or unknown:
        raise ValueError(
            f"{table_name} has missing or unknown visit_context values {unknown}"
        )
    if (
        "occupancy_state" in frame.columns
        and not (
            labels.eq(IN_WELL_VISIT).to_numpy()
            == frame["occupancy_state"].astype(object).eq("at_well").to_numpy()
        ).all()
    ):
        raise ValueError(
            f"{table_name} has visit_context in_visit rows that are not at_well, "
            "or at_well rows that are not in_visit"
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
    require_columns(event_table, ["event_id"], "event_table")
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


EVENT_BIN_TOTAL_COLUMNS = ("n_nonlocal_bins", "n_timebins")


def event_bin_total_column(frame: pd.DataFrame) -> str | None:
    """Name the column holding an event's total bin count, or ``None``.

    Non-local event tables carry ``n_nonlocal_bins``; ContFrag event tables
    carry ``n_timebins``. ``n_nonlocal_bins`` wins when both are present.
    Subtracting ``n_undecoded_bins`` from this column gives the event's
    decoded-bin count, which is the denominator of that table's
    ``arm_fraction``.
    """
    for candidate in EVENT_BIN_TOTAL_COLUMNS:
        if candidate in frame:
            return candidate
    return None


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

    total_column = event_bin_total_column(frame)
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
    leaf = frame[frame["candidate_arm"].isin(LEAF_ARMS)]
    for column in _LEAF_SUPPORT_COLUMNS:
        if column not in leaf:
            continue
        finite = np.isfinite(pd.to_numeric(leaf[column], errors="coerce"))
        support = finite.groupby(leaf["event_id"], sort=False).sum()
        partial = support[(support != 0) & (support != N_LEAF_ARMS)]
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
    require_columns(frame, ["event_id", "candidate_arm"], "event_arm")
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
            "visit_context",
        ),
    )
    require_visit_context_labels(frame, "event_arm")
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
    require_columns(frame, ["event_id", "patch"], "event_patch")
    _require_exact_group_members(
        frame,
        group_column="event_id",
        member_column="patch",
        expected=_PATCHES,
        label="event_patch patch set",
    )
    _require_stable_event_context(frame, (*SESSION_COLUMNS, "visit_context"))
    require_visit_context_labels(frame, "event_patch")
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


def validate_requested_standardization(
    long_df: pd.DataFrame,
    *,
    union_predictors: Sequence[str],
    input_dir: Path,
    table_label: object,
) -> None:
    """Fail before fitting if requested persisted ``_z`` units are stale.

    The canonical event-arm builder estimates these moments on the six leaf arms.
    Older materialized tables used all nine arms, which inflated uncertainty
    coefficients while preserving plausible column names.  Only inspect sources
    whose standardized columns are actually requested by this invocation.
    """
    sources = tuple(
        source
        for source in EVENT_ARM_STANDARDIZED_COLS
        if f"{source}_z" in union_predictors
    )
    if not sources:
        return

    reference_path = input_dir / STANDARDIZATION_REFERENCE
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Requested standardized predictor(s) {[f'{c}_z' for c in sources]} "
            f"from {table_label}, but {reference_path} is missing. Regenerate the "
            "canonical event-arm table and its leaf-support reference."
        )
    try:
        assert_standardized_matches_reference(
            long_df,
            sources,
            load_reference(reference_path),
            support_label="leaf_arms",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Canonical standardization for {table_label} is stale or invalid: "
            f"{exc}. Regenerate the event-arm table and "
            "standardization_reference.json before fitting."
        ) from exc


def validate_canonical_table_contract(
    long_df: pd.DataFrame,
    *,
    hierarchy_kind: str,
    table_label: object,
    response_col: str | None = None,
) -> None:
    """Validate location and novelty semantics that identify current tables.

    ``response_col`` is the column this run will actually fit. The hard counts
    are validated unconditionally because they are part of the canonical schema;
    a fraction response is additionally held to its own contract (defined for
    every candidate of an event or for none of it, and summing to 1.0).
    """
    if long_df.empty:
        raise ValueError(f"Canonical content table {table_label} is empty")
    if long_df[[*SESSION_COLUMNS, "event_id"]].isna().any(axis=None):
        raise ValueError(
            f"Canonical content table {table_label} has missing session/event keys"
        )

    candidate_col, expected_candidates = _CANDIDATE_COLUMNS[hierarchy_kind]
    n_candidates = len(expected_candidates)
    unit_cols = [*SESSION_COLUMNS, "event_id"]
    if (
        long_df.duplicated([*unit_cols, candidate_col]).any()
        or not long_df[candidate_col].isin(expected_candidates).all()
    ):
        raise ValueError(
            f"Canonical {hierarchy_kind} content table {table_label} must have "
            f"exactly the expected {candidate_col} candidates "
            f"{list(expected_candidates)} once per session/event"
        )

    by_event = long_df.groupby(unit_cols, sort=False, observed=True)
    event_sizes = by_event.size()
    if not event_sizes.eq(n_candidates).all():
        raise ValueError(
            f"Canonical {hierarchy_kind} content table {table_label} must have "
            f"exactly the expected {candidate_col} candidates "
            f"{list(expected_candidates)} once per session/event"
        )

    # ``event_id`` is canonicalized as globally unique. Enforcing the mapping
    # here also prevents later event-only cohort/audit groupbys from fusing two
    # sessions if a stale table reused a session-local event counter.
    event_session_counts = long_df.groupby("event_id", sort=False, observed=True)[
        list(SESSION_COLUMNS)
    ].nunique(dropna=False)
    if event_session_counts.gt(1).any(axis=None):
        raise ValueError(
            f"Canonical content table {table_label} reuses event_id across sessions"
        )

    context_cols = [
        *SESSION_COLUMNS,
        "occupancy_state",
        "at_well",
        "occupancy_well_id",
        "occupancy_well_arm",
    ]
    if (by_event[context_cols].nunique(dropna=False) != 1).any(axis=None):
        raise ValueError(
            f"Canonical content table {table_label} has event context that "
            "changes across candidate rows"
        )

    state = long_df["occupancy_state"]
    invalid_states = sorted(set(state.dropna().unique()) - _LOCATION_STATES)
    if state.isna().any() or invalid_states:
        raise ValueError(
            f"Canonical content table {table_label} has invalid location_state "
            f"values {invalid_states}; expected {sorted(_LOCATION_STATES)}"
        )
    at_well = state.eq("at_well")
    # ``at_well`` is a nullable boolean and is NA exactly for the ``unknown``
    # state (events before the first completed trip): the window's location was
    # never observed, so "is the animal at a well" has no answer. Transit and
    # mixed windows are explicitly False. Requiring non-null here would reject a
    # canonical table for carrying a state this validator itself accepts.
    unknown = state.eq("unknown")
    observed_at_well = long_df["at_well"].notna()
    if not observed_at_well.eq(~unknown).all() or not (
        long_df["at_well"].fillna(False).eq(at_well).all()
    ):
        raise ValueError(
            f"Canonical content table {table_label} has at_well values "
            "inconsistent with location_state"
        )

    # A current candidate is scientifically defined only for whole-window
    # at-well events. Numeric zeros off-well would retain those events in the
    # common cohort and silently change the estimand.
    current_col = "is_current_arm" if hierarchy_kind == "arm" else "is_current_patch"
    current = long_df[current_col]
    if not current.dropna().isin([0.0, 1.0]).all():
        raise ValueError(
            f"Canonical content table {table_label} has non-binary {current_col}"
        )
    if current.loc[~at_well].notna().any() or current.loc[at_well].isna().any():
        raise ValueError(
            f"Canonical content table {table_label} must encode {current_col} "
            "as one-hot at wells and NaN off-well"
        )
    current_sums = (
        long_df.loc[at_well]
        .groupby(unit_cols, sort=False, observed=True)[current_col]
        .sum()
    )
    if not current_sums.eq(1.0).all():
        raise ValueError(
            f"Canonical content table {table_label} must encode exactly one "
            f"{current_col} candidate for every at-well event"
        )

    for location_col in ("occupancy_well_id", "occupancy_well_arm"):
        if (
            long_df.loc[at_well, location_col].isna().any()
            or long_df.loc[~at_well, location_col].notna().any()
        ):
            raise ValueError(
                f"Canonical content table {table_label} must define "
                f"{location_col} only for at-well events"
            )
    current_reference = (
        long_df["occupancy_well_arm"]
        if hierarchy_kind == "arm"
        else long_df["occupancy_well_arm"].map(TRACK_SEGMENT_TO_PATCH)
    )
    expected_current = long_df[candidate_col].eq(current_reference).astype(float)
    if not current.loc[at_well].eq(expected_current.loc[at_well]).all():
        relation = (
            "occupancy_well_arm"
            if hierarchy_kind == "arm"
            else "the mapped occupancy_well_arm patch"
        )
        raise ValueError(
            f"Canonical content table {table_label}: {current_col} must match "
            f"{relation} exactly"
        )

    count_col = "arm_count_hard" if hierarchy_kind == "arm" else "patch_count_hard"
    response = long_df[count_col]
    numeric_response = pd.to_numeric(response, errors="coerce").to_numpy(dtype=float)
    invalid_response = (
        not pd.api.types.is_numeric_dtype(response.dtype)
        or pd.api.types.is_bool_dtype(response.dtype)
        or not np.isfinite(numeric_response).all()
        or (numeric_response < 0).any()
        or not np.equal(numeric_response, np.floor(numeric_response)).all()
    )
    if invalid_response:
        raise ValueError(
            f"Canonical content table {table_label} requires {count_col} to "
            "contain finite nonnegative integer counts"
        )
    if response_col is not None and response_col != count_col:
        validate_fraction_response(
            long_df,
            response_col,
            event_col="event_id",
            table_label=table_label,
        )

    novelty_col = (
        "is_candidate_arm_never_visited"
        if hierarchy_kind == "arm"
        else "is_patch_never_visited"
    )
    recency_col = (
        "visit_recency_trials_event_leaf_centered"
        if hierarchy_kind == "arm"
        else "visit_recency_trials_patch"
    )
    novelty = long_df[novelty_col]
    if not novelty.dropna().isin([0.0, 1.0]).all():
        raise ValueError(
            f"Canonical content table {table_label} has non-binary {novelty_col}"
        )
    novelty_counts = by_event[novelty_col].count()
    recency_counts = by_event[recency_col].count()
    valid_counts = {0, n_candidates}
    if (
        not novelty_counts.isin(valid_counts).all()
        or not recency_counts.isin(valid_counts).all()
        or not novelty_counts.eq(recency_counts).all()
    ):
        raise ValueError(
            f"Canonical content table {table_label} must carry {novelty_col} "
            f"and {recency_col} together for all candidates in an event, or "
            "leave both unknown before history exists"
        )


def validate_and_match_fit_results(
    fit_results: Mapping[str, pd.DataFrame],
    *,
    requested_model_names: Sequence[str],
    cohort: pd.DataFrame,
    event_col: str,
    require_cv: bool = False,
) -> dict[str, pd.DataFrame]:
    """Require a complete ladder on one exact shared session/event cohort."""
    requested = tuple(requested_model_names)
    missing_results = [
        model_name
        for model_name in requested
        if model_name not in fit_results or fit_results[model_name].empty
    ]
    if missing_results:
        raise RuntimeError(
            f"Requested content model(s) {missing_results} produced no session fits; "
            "the complete ladder will not be published"
        )

    expected_event_counts = {
        tuple(keys): int(group[event_col].nunique())
        for keys, group in cohort.groupby(
            list(SESSION_COLUMNS), sort=False, observed=True, dropna=False
        )
    }
    session_sets: dict[str, set[tuple]] = {}

    for model_name in requested:
        result = fit_results[model_name]
        # ``estimator`` and ``response_col`` are integrity columns: without them
        # a published bundle does not say at which grain its llf/AIC/SEs are
        # stated, and a fraction bundle is indistinguishable from a count one.
        required = [*SESSION_COLUMNS, "model", "n_events", "estimator", "response_col"]
        missing_columns = [column for column in required if column not in result]
        if missing_columns:
            raise RuntimeError(
                f"Fit result for {model_name} is missing integrity column(s) "
                f"{missing_columns}; no output will be published"
            )
        if result[list(SESSION_COLUMNS)].isna().any(axis=None):
            raise RuntimeError(
                f"Fit result for {model_name} has missing session keys; "
                "no output will be published"
            )
        if result.duplicated(list(SESSION_COLUMNS)).any():
            raise RuntimeError(
                f"Fit result for {model_name} has duplicate session rows; "
                "no output will be published"
            )
        if require_cv:
            missing_cv = [
                column for column in CV_RESULT_COLUMNS if column not in result
            ]
            if missing_cv:
                raise RuntimeError(
                    f"Requested CV result for {model_name} is missing column(s) "
                    f"{missing_cv}; no output will be published"
                )
            # Sessions whose CV did not run carry NaN in every CV column. Those
            # are tolerated -- a failed fold must not cost a session its primary
            # coefficient -- but a rung where NO session scored means CV is
            # broken rather than marginal, and that still refuses to publish.
            scored = result["cv_n_folds"].notna()
            if not scored.any():
                raise RuntimeError(
                    f"Requested CV produced no scored session for {model_name}; "
                    "no output will be published"
                )
            if not scored.all():
                logging.warning(
                    "%s: CV scored %d of %d sessions (%.0f%%); coefficients keep "
                    "the full cohort, held-out comparisons use the scored subset",
                    model_name,
                    int(scored.sum()),
                    len(result),
                    100 * scored.mean(),
                )
            cv_counts = result.loc[scored, list(_CV_COUNT_COLUMNS)].apply(
                pd.to_numeric, errors="coerce"
            )
            invalid_cv_counts = (
                ~np.isfinite(cv_counts.to_numpy(dtype=float)).all(axis=1)
                | cv_counts["cv_n_folds"].lt(2)
                | cv_counts.ne(np.floor(cv_counts)).any(axis=1)
                | cv_counts["cv_n_nonconverged_folds"].lt(0)
                | cv_counts["cv_n_separated_folds"].lt(0)
                | cv_counts["cv_n_nonconverged_folds"].gt(cv_counts["cv_n_folds"])
                | cv_counts["cv_n_separated_folds"].gt(cv_counts["cv_n_folds"])
            )
            if invalid_cv_counts.any():
                raise RuntimeError(
                    f"Requested CV result for {model_name} has invalid or missing "
                    "fold metadata; no output will be published"
                )
        if (
            set(result["model"].dropna().unique()) != {model_name}
            or result["model"].isna().any()
        ):
            raise RuntimeError(
                f"Fit result labeled as {model_name} contains inconsistent model "
                "labels; no output will be published"
            )

        sessions: set[tuple] = set()
        for row in result.itertuples(index=False):
            session = tuple(getattr(row, column) for column in SESSION_COLUMNS)
            if session not in expected_event_counts:
                raise RuntimeError(
                    f"Fit result for {model_name} contains session {session} outside "
                    "the frozen cohort; no output will be published"
                )
            n_events = row.n_events
            try:
                numeric_n_events = float(n_events)
            except (TypeError, ValueError):
                numeric_n_events = np.nan
            if (
                not np.isfinite(numeric_n_events)
                or not numeric_n_events.is_integer()
                or int(numeric_n_events) != expected_event_counts[session]
            ):
                raise RuntimeError(
                    f"Fit result for {model_name}/{session} reports n_events="
                    f"{n_events!r}, expected {expected_event_counts[session]} from "
                    "the frozen cohort; no output will be published"
                )
            sessions.add(session)
        session_sets[model_name] = sessions

    shared_sessions = set.intersection(
        *(session_sets[model_name] for model_name in requested)
    )
    if not shared_sessions:
        raise RuntimeError(
            "Requested content models have no common successfully fitted session; "
            "the complete ladder will not be published"
        )

    matched: dict[str, pd.DataFrame] = {}
    for model_name in requested:
        result = fit_results[model_name]
        session_index = pd.MultiIndex.from_frame(result[list(SESSION_COLUMNS)])
        keep = session_index.isin(shared_sessions)
        matched_result = (
            result.loc[keep]
            .sort_values(list(SESSION_COLUMNS), kind="stable")
            .reset_index(drop=True)
        )
        matched_sessions = set(
            matched_result[list(SESSION_COLUMNS)].itertuples(index=False, name=None)
        )
        if matched_sessions != shared_sessions:
            raise RuntimeError(
                f"Internal shared-session alignment failed for {model_name}; "
                "no output will be published"
            )
        matched[model_name] = matched_result

    return matched
