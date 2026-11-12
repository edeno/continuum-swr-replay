"""The events a content ladder is fit on, and where the others went.

A content fit reads one canonical candidate table: one row per event and
candidate arm (or patch). Three steps turn it into the fitting cohort, and each
is one function here, so the content runner and any check of its cohort select
the same events:

- :func:`prepare_canonical_fit_request` refuses a table or a request that
  cannot give the whole requested ladder, before anything is fit;
- :func:`select_content_cohort` restricts the table, in a fixed order, to the
  events every rung of the ladder is fit on;
- :func:`content_cohort_record` books, for one fitted rung, where every event
  that entered the common complete-case freeze went.

The is_swr-contrast ladder selects per rung and population instead:
:func:`contrast_fit_cohort` gives the rows, predictors and candidate count one
(rung, population) is fit on.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import pandas as pd

from continuum_swr_replay.analysis.events.cohort_selection import (
    freeze_common_cohort,
    select_event_cohort,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from .._helpers import add_arm_fixed_effects
from ..cohort_audit import (
    attach_has_qvals_row,
    cohort_record,
    drop_profile,
    estimator_skip_note,
    event_drop_buckets,
    is_qvals_sourced_column,
)
from ..content_response import resolve_response_column
from ..covariate_timing import assert_predictor_timing
from ..model_specs import (
    LEAF_ONLY_CONTRAST_RUNGS,
    LEAF_ONLY_N_CATEGORIES,
    leaf_only_cohort,
    leaf_only_predictors,
)
from ..tables.table_contracts import validate_canonical_table_contract
from .dirichlet_multinomial import ESTIMATOR_SKIP_CLASSES

logger = logging.getLogger(__name__)

# A canonical content table carries more than whichever columns one requested
# model happens to use. These fields distinguish the current location-aware,
# future-invariant history schema from older pickles that can otherwise satisfy
# M0-M3 and silently produce a truncated ladder.
_CANONICAL_SCHEMA_COLUMNS = {
    "arm": (
        *SESSION_COLUMNS,
        "event_id",
        "candidate_arm",
        "arm_count_hard",
        "occupancy_state",
        "at_well",
        "occupancy_well_id",
        "occupancy_well_arm",
        "is_current_arm",
        "visit_recency_trials_event_leaf_centered",
        "is_candidate_arm_never_visited",
    ),
    "patch": (
        *SESSION_COLUMNS,
        "event_id",
        "patch",
        "patch_count_hard",
        "occupancy_state",
        "at_well",
        "occupancy_well_id",
        "occupancy_well_arm",
        "is_current_patch",
        "visit_recency_trials_patch",
        "is_patch_never_visited",
    ),
}

# Per hierarchy, the row-gated column whose NaN marks an event on a stop with no
# qvals row (exactly those rows on the live tables: 8,676 event-arm and 2,892
# event-patch rows, checked 2026-10-24).
_QVALS_ROW_GATED_SOURCE = {
    "arm": "task_reward_probability_candidate_arm",
    "patch": "task_reward_probability_candidate_arm_patch",
}


def prepare_canonical_fit_request(
    long_df: pd.DataFrame,
    *,
    hierarchy: Mapping,
    model_names: Sequence[str],
    hierarchy_kind: str,
    table_label: object,
    response: str = "fraction",
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], str, str]:
    """Validate one complete canonical fit request before fitting or writing.

    This is the fail-closed boundary of a content fit. It first requires the
    current location/novelty table schema, then derives arm fixed effects, and
    finally verifies *every* requested model and predictor as one atomic
    request. A stale table therefore cannot fit the subset of an otherwise
    nested ladder that it happens to satisfy.

    The resolved response column is validated on the whole loaded table, so a
    table that cannot express the requested response fails before the first
    fit rather than halfway through the ladder.

    Parameters
    ----------
    long_df : pd.DataFrame
        The canonical event-arm or event-patch table, one row per event and
        candidate.
    hierarchy : Mapping
        Rung name to :class:`~continuum_swr_replay.analysis.behavior.model_specs.ModelSpec`.
    model_names : sequence of str
        The requested rungs, each a key of ``hierarchy``, none repeated.
    hierarchy_kind : {"arm", "patch"}
        Which canonical schema the table must carry.
    table_label : object
        Names the table in error messages (its path, usually).
    response : {"fraction", "hard_count"}, optional
        The allocation the ladder is fit on: the shipped per-event fraction or
        the timebin counts.

    Returns
    -------
    prepared : pd.DataFrame
        A copy of ``long_df``, with the arm fixed effects added for the arm
        hierarchy.
    model_names : tuple of str
        The requested model names, unchanged.
    predictors : tuple of str
        The sorted union of the requested rungs' predictors.
    response_col : str
        The resolved response column.
    event_col : str
        The event key, ``"event_id"``.

    Raises
    ------
    ValueError
        If ``hierarchy_kind`` is unknown, the table lacks a schema column,
        breaks the canonical table contract, or cannot give a requested
        predictor, or the request names no rung, repeats one, or names one the
        hierarchy does not hold.
    """
    try:
        required_schema = _CANONICAL_SCHEMA_COLUMNS[hierarchy_kind]
    except KeyError as exc:
        raise ValueError(
            f"hierarchy_kind must be 'candidate_arm' or 'patch', got {hierarchy_kind!r}"
        ) from exc

    missing_schema = [column for column in required_schema if column not in long_df]
    if missing_schema:
        raise ValueError(
            f"Canonical {hierarchy_kind} content table {table_label} is missing "
            "current-location/novelty schema column(s) "
            f"{missing_schema}. Regenerate it with the current prepare script."
        )
    response_col = resolve_response_column(hierarchy_kind, response)
    validate_canonical_table_contract(
        long_df,
        hierarchy_kind=hierarchy_kind,
        table_label=table_label,
        response_col=response_col,
    )

    requested = tuple(model_names)
    if not requested:
        raise ValueError("At least one content model must be requested")
    if len(set(requested)) != len(requested):
        raise ValueError(
            f"Requested content model names must be unique, got {list(requested)}"
        )
    unknown = [model_name for model_name in requested if model_name not in hierarchy]
    if unknown:
        raise ValueError(
            f"Unknown requested content model(s) {unknown}; valid names are "
            f"{list(hierarchy)}. Refusing to fit a partial ladder."
        )

    prepared = long_df.copy()
    if hierarchy_kind == "arm":
        # Fixed effects are intentionally derived at fit time, so their absence
        # from the persisted canonical table is not a schema error.
        add_arm_fixed_effects(prepared)

    missing_by_model = {
        model_name: [
            predictor
            for predictor in hierarchy[model_name].predictors
            if predictor not in prepared
        ]
        for model_name in requested
    }
    missing_by_model = {
        model_name: missing
        for model_name, missing in missing_by_model.items()
        if missing
    }
    if missing_by_model:
        details = ", ".join(
            f"{model_name}={missing}"
            for model_name, missing in missing_by_model.items()
        )
        raise ValueError(
            f"Canonical content table {table_label} cannot satisfy the complete "
            f"requested ladder: {details}. Regenerate the table; no requested "
            "model will be skipped."
        )

    # Apply the causal-hygiene gate to the whole request before the first fit
    # or output, rather than discovering a disallowed predictor midway through
    # a ladder after earlier rungs have already been persisted.
    for model_name in requested:
        assert_predictor_timing(hierarchy[model_name].predictors)

    predictors = tuple(
        sorted(
            {
                predictor
                for model_name in requested
                for predictor in hierarchy[model_name].predictors
            }
        )
    )
    return prepared, requested, predictors, response_col, "event_id"


class CohortSelection(NamedTuple):
    """The rows a content ladder is fit on, and the frame they were frozen from.

    Attributes
    ----------
    long_df : pd.DataFrame
        The common complete-case cohort every rung is fit on.
    long_df_prefilter : pd.DataFrame
        The rows that entered the complete-case freeze, for the cohort audit.
    n_events_before_complete_case : int
        Events in ``long_df_prefilter``: the audit's ``n_in``.
    response_renormalized : bool
        Whether the response was rescaled to unit mass over the leaf arms.
    """

    long_df: pd.DataFrame
    long_df_prefilter: pd.DataFrame
    n_events_before_complete_case: int
    response_renormalized: bool


def select_content_cohort(
    long_df: pd.DataFrame,
    *,
    hierarchy_kind: str,
    union_predictors: Sequence[str],
    response_col: str,
    response: str,
    table_label: object,
    event_col: str = "event_id",
    swr_subset: str = "all",
    leaf_only: bool = False,
) -> CohortSelection:
    """Restrict a prepared content table to the events a ladder is fit on.

    The steps run in this order, each logged:

    1. the SWR subset (``swr_subset`` ``"T"`` keeps the events with
       ``is_swr``, ``"F"`` those without, ``"all"`` keeps both);
    2. the at-well event cohort
       (:func:`~continuum_swr_replay.analysis.events.cohort_selection.select_event_cohort`),
       applied unconditionally: a table without ``occupancy_state`` raises;
    3. with ``leaf_only``, the six leaf arms, the fraction response rescaled
       to unit mass over them
       (:func:`~continuum_swr_replay.analysis.behavior.model_specs.leaf_only_cohort`);
    4. when a requested predictor comes from the qvals table, the
       ``has_qvals_row`` flag, read off the hierarchy's row-gated column;
    5. the common complete-case freeze: an event leaves every rung when any of
       its rows misses, or holds a non-finite value of, the response or any
       predictor of the union, or when its response sums to zero.

    Parameters
    ----------
    long_df : pd.DataFrame
        The table :func:`prepare_canonical_fit_request` returned.
    hierarchy_kind : {"arm", "patch"}
        Which row-gated column marks the qvals gap.
    union_predictors : sequence of str
        Every requested rung's predictors.
    response_col : str
        The resolved response column.
    response : {"fraction", "hard_count"}
        The response the column came from; only a fraction is rescaled.
    table_label : object
        Names the table in error messages.
    event_col : str, optional
        The event key.
    swr_subset : {"all", "T", "F"}, optional
        Which events by SWR overlap.
    leaf_only : bool, optional
        Whether to keep only the six leaf-arm candidates.

    Returns
    -------
    CohortSelection

    Raises
    ------
    ValueError
        If a subset is asked of a table without ``is_swr``.
    KeyError
        If the table has no ``occupancy_state``.
    RuntimeError
        If the complete-case freeze removes every event, or every survivor has
        a zero response.
    """
    # Apply the SWR subset filter (--swr-subset T/F). is_swr is
    # event-level on the event_arm_table and patch table. Drop whole
    # events / patch-rows where is_swr doesn't match.
    if swr_subset != "all":
        if "is_swr" not in long_df.columns:
            raise ValueError(
                "--swr-subset requires canonical column 'is_swr'; not found in "
                f"{table_label}. Regenerate the table."
            )
        keep = long_df["is_swr"] if swr_subset == "T" else ~long_df["is_swr"]
        n_before_rows = len(long_df)
        long_df = long_df.loc[keep].reset_index(drop=True)
        logger.info(
            "SWR subset filter (%s): %d -> %d rows",
            swr_subset,
            n_before_rows,
            len(long_df),
        )

    # Cohort: at-well events only, stated rather than inferred.
    #
    # Every location-derived predictor (`is_current` and everything built from
    # it) is NaN when the event window is not wholly at one well, so the
    # complete-case filter below would remove these events anyway -- the two
    # sets are identical, verified at 59,374 events with zero disagreements. But
    # arriving there via `dropna` makes a scope decision look like missing data:
    # it appears in the audit as "predictor NaN" rather than as a cohort, it
    # would silently not apply to a rung that happens to omit `is_current`, and
    # it cannot be varied to ask what in-transit events would show under a
    # different predictor set. Applying it here states the restriction, and the
    # complete-case filter remains as a safety net that should now find nothing
    # location-related to drop.
    #
    # The restriction is not balanced across the SWR contrast (3.3% of SWR
    # events are off-well against 13.0% of non-SWR) -- expected, since SWR
    # events concentrate at wells -- so the cohort belongs in Methods.
    #
    # The rule itself lives in `select_event_cohort`, the event-grain half of the
    # rule the rate runners apply to their stop table, so "at-well" means one
    # thing across both families rather than two filters that happen to agree
    # today. It is called unconditionally: a table without `occupancy_state`
    # cannot state a cohort, and raising there is the point -- the schema gate in
    # `prepare_canonical_fit_request` already refuses such a table, and a
    # `column in frame` guard here would silently fit the whole corpus if it ever
    # stopped doing so.
    long_df, cohort_report = select_event_cohort(
        long_df, cohort="at_well", event_col=event_col
    )
    logger.info(
        "cohort = at-well events (%s): %d -> %d events (%.1f%% retained)",
        cohort_report["rule"],
        cohort_report["n_events_before"],
        cohort_report["n_events_after"],
        100
        * cohort_report["n_events_after"]
        / max(cohort_report["n_events_before"], 1),
    )

    # Candidate set: the six leaf arms, when asked for. Applied before the
    # complete-case freeze so the cohort is the set of events the leaf-only
    # likelihood can actually use -- an event with no leaf mass left has a zero
    # response and is dropped there, and named as such in the audit.
    response_renormalized = False
    if leaf_only:
        n_before_rows = len(long_df)
        long_df, response_renormalized = leaf_only_cohort(
            long_df,
            response_col=response_col,
            response=response,
            event_col=event_col,
        )
        logger.info(
            "candidate set = %d leaf arms: %d -> %d rows (%s rescaled to unit "
            "mass per event: %s)",
            LEAF_ONLY_N_CATEGORIES,
            n_before_rows,
            len(long_df),
            response_col,
            response_renormalized,
        )

    # The cohort audit books a value rung's drop to the qvals gap only when the
    # event's stop has no qvals row. The candidate table carries no qvals join,
    # so the flag is read off a row-gated column (NaN exactly on those stops).
    if any(is_qvals_sourced_column(c) for c in union_predictors):
        long_df = attach_has_qvals_row(
            long_df,
            from_column=_QVALS_ROW_GATED_SOURCE[hierarchy_kind],
            event_col=event_col,
        )

    # Pre-filter to common complete-case event set: drop entire events
    # (all 9 arm rows) if ANY predictor in the union is NaN for ANY row.
    # Keep the pre-filter frame for the cohort audit (per-model drop attribution).
    long_df_prefilter = long_df
    required_cols = [response_col, event_col, *union_predictors]
    long_df, cohort_selection = freeze_common_cohort(
        long_df,
        unit_col=event_col,
        required_cols=required_cols,
        response_col=response_col,
        drop_zero_response=True,
    )
    n_before = cohort_selection["n_units_before"]
    n_complete_case = (
        n_before
        - cohort_selection["n_missing_units"]
        - cohort_selection["n_nonfinite_units"]
    )
    if n_before != n_complete_case:
        logger.info(
            "Filtered to common complete-case events: %d → %d events "
            "(%d missing, %d nonfinite in %s)",
            n_before,
            n_complete_case,
            cohort_selection["n_missing_units"],
            cohort_selection["n_nonfinite_units"],
            list(union_predictors),
        )
    if cohort_selection["n_zero_response_units"]:
        logger.info(
            "Removed %d complete-case events with zero total response",
            cohort_selection["n_zero_response_units"],
        )
    if n_complete_case == 0:
        # A predictor that is structurally NaN on some arm of every
        # event (e.g. a stale ``recent_reward_rate`` column predating
        # the trunk-arm 0.0 fix) wipes out the whole event set under
        # event-wise complete-case filtering. Fail loudly rather than
        # fitting M0 on zero events.
        raise RuntimeError(
            f"Complete-case filtering removed all {n_before} events because a "
            f"predictor in {list(union_predictors)} is missing or nonfinite "
            "on some candidate "
            "row of every event. Regenerate the canonical table."
        )
    if cohort_selection["n_units_after"] == 0:
        raise RuntimeError(
            f"Common-cohort filtering removed all {n_before} events because "
            "every complete-case event had zero total response."
        )

    logger.info(
        "Fitting on %d rows, %d sessions, %d events",
        len(long_df),
        long_df.groupby(list(SESSION_COLUMNS)).ngroups,
        long_df[event_col].nunique(),
    )
    return CohortSelection(
        long_df=long_df,
        long_df_prefilter=long_df_prefilter,
        n_events_before_complete_case=n_before,
        response_renormalized=response_renormalized,
    )


def zero_response_event_ids(
    long_df: pd.DataFrame, *, event_col: str, response_col: str
) -> set:
    """Events whose response is undefined on every row, or sums to nothing.

    ``min_count=1`` keeps an all-missing event's total missing rather than
    letting pandas report it as 0.0, so "the response could not be computed for
    this event" and "the response is zero everywhere" both land here and neither
    is mistaken for a fitted event.

    Parameters
    ----------
    long_df : pd.DataFrame
        One row per event and candidate.
    event_col : str
        The event key.
    response_col : str
        The response column.

    Returns
    -------
    set
        The ids of the events whose total response is missing or not positive.
    """
    totals = long_df.groupby(event_col, observed=True, sort=False)[response_col].sum(
        min_count=1
    )
    return set(totals.index[~(totals > 0)])


def content_cohort_record(
    model_name: str,
    fits: pd.DataFrame,
    *,
    long_df_prefilter: pd.DataFrame,
    predictors: Sequence[str],
    union_predictors: Sequence[str],
    response_col: str,
    event_col: str,
    n_events_in: int,
    population: str,
    flag_counts: Mapping[str, int],
    estimator_skips: Sequence[Mapping[str, str]] = (),
) -> dict:
    """One rung's cohort-audit record on the common complete-case cohort.

    Every event that entered the freeze is booked once. A dropped event goes to
    the first bucket of
    :func:`~continuum_swr_replay.analysis.behavior.cohort_audit.event_drop_buckets`
    it falls in (the qvals gap, this rung's own predictor missing, a non-finite
    value, another rung's predictor missing, a zero response); the events of a
    session the fitter did not fit are ``n_excluded_session``; the rest are
    ``n_modeled``. A session the estimator refused is not a dropped event: it is
    named in ``drop_notes``.

    Parameters
    ----------
    model_name : str
        The rung.
    fits : pd.DataFrame
        The rung's per-session fits, one row per session with ``n_events`` and
        ``animal``. Empty when no session was fit.
    long_df_prefilter : pd.DataFrame
        The rows that entered the complete-case freeze
        (:attr:`CohortSelection.long_df_prefilter`).
    predictors : sequence of str
        This rung's own predictors.
    union_predictors : sequence of str
        Every rung's predictors: the columns the freeze required.
    response_col : str
        The response column.
    event_col : str
        The event key.
    n_events_in : int
        Events that entered the freeze
        (:attr:`CohortSelection.n_events_before_complete_case`).
    population : str
        The population label the record carries (the SWR subset).
    flag_counts : Mapping[str, int]
        The fits' convergence and separation counts, carried through.
    estimator_skips : sequence of Mapping, optional
        The sessions the estimator refused, each with ``session`` and
        ``reason``.

    Returns
    -------
    dict
        The record :func:`~continuum_swr_replay.analysis.behavior.cohort_audit.cohort_record`
        returns, with ``cohort_mode="common"`` and the event grain.

    Raises
    ------
    RuntimeError
        If the buckets do not account for every event whose response is
        undefined or sums to zero.
    AssertionError
        If the record does not reconcile
        (:func:`~continuum_swr_replay.analysis.behavior.cohort_audit.cohort_record`).
    """
    sessions_in = long_df_prefilter.groupby(list(SESSION_COLUMNS)).ngroups
    # Events whose response is undefined or sums to nothing: on the fraction
    # response these are the all-junction events, which carry NaN on every
    # candidate. The cohort freeze removes them; the audit must also SAY so,
    # under n_drop_zero_response, or a reader cannot tell an event the response
    # could not describe from one no model wanted.
    zero_response_events = zero_response_event_ids(
        long_df_prefilter, event_col=event_col, response_col=response_col
    )
    # Attribute the union-filter drops to this rung's own vs the other rungs'
    # predictors.
    buckets = event_drop_buckets(
        long_df_prefilter,
        event_col=event_col,
        response_col=response_col,
        own_predictors=predictors,
        union_predictors=union_predictors,
    )
    n_modeled = int(fits["n_events"].sum()) if len(fits) else 0
    n_drops = sum(
        buckets[k]
        for k in (
            "n_drop_qvals_gap",
            "n_drop_own_predictor_missing",
            "n_drop_nonfinite",
            "n_drop_common_cohort_other",
            "n_drop_zero_response",
        )
    )
    unaccounted = zero_response_events - buckets["dropped_event_ids"]
    if unaccounted or buckets["n_drop_zero_response"] > len(zero_response_events):
        raise RuntimeError(
            f"Cohort audit for {model_name} does not account for the "
            f"{len(zero_response_events)} event(s) whose {response_col} is "
            f"undefined or sums to zero ({len(unaccounted)} unaccounted, "
            f"n_drop_zero_response={buckets['n_drop_zero_response']}); no "
            "output will be published"
        )
    dropped = long_df_prefilter[
        long_df_prefilter[event_col].isin(buckets["dropped_event_ids"])
    ].drop_duplicates(event_col)
    # A session the estimator itself refused (a rank-deficient within-event
    # design, or too few events for its parameters) is not a dropped event
    # and does not belong in a drop bucket -- its events are already
    # counted under n_excluded_session. It IS a fitted-cohort fact, so it is
    # named here rather than only in the run log.
    notes = drop_profile(dropped)
    # The routine classes are summarized; only an estimator REFUSAL names
    # the session it happened to. Same helper (and so the same policy) as
    # the is_swr-contrast runner's cohort audit.
    skip_note = estimator_skip_note(
        [(skip["session"], skip["reason"]) for skip in estimator_skips],
        ESTIMATOR_SKIP_CLASSES,
    )
    if skip_note:
        notes = "; ".join(filter(None, [notes, skip_note]))
    return cohort_record(
        model_name,
        population,
        "event",
        cohort_mode="common",
        n_in=n_events_in,
        n_modeled=n_modeled,
        sessions_in=sessions_in,
        sessions_modeled=len(fits),
        animals_modeled=int(fits["animal"].nunique()) if len(fits) else 0,
        drop_notes=notes,
        n_drop_qvals_gap=buckets["n_drop_qvals_gap"],
        n_drop_own_predictor_missing=buckets["n_drop_own_predictor_missing"],
        n_drop_nonfinite=buckets["n_drop_nonfinite"],
        n_drop_common_cohort_other=buckets["n_drop_common_cohort_other"],
        n_drop_zero_response=buckets["n_drop_zero_response"],
        n_excluded_session=n_events_in - n_drops - n_modeled,
        **flag_counts,
    )


#: The is_swr-contrast populations: every event, the SWR events, the others.
CONTRAST_POPULATIONS = ("all", "SWR", "nonSWR")


class ContrastFitCohort(NamedTuple):
    """What one (rung, population) of the is_swr-contrast ladder is fit on.

    Attributes
    ----------
    frame : pd.DataFrame
        The candidate rows of the population, leaf arms only on a leaf-only rung.
    predictors : list of str
        The rung's predictors, less the arm fixed effects a leaf-only fit cannot
        carry.
    n_categories : int
        Candidates per event under the rung's likelihood.
    response_renormalized : bool
        Whether the fraction response was rescaled to unit mass over the leaf arms.
    """

    frame: pd.DataFrame
    predictors: list[str]
    n_categories: int
    response_renormalized: bool


def contrast_fit_cohort(
    cohort: pd.DataFrame,
    rung: str,
    predictors: Sequence[str],
    population: str,
    *,
    response_col: str,
    response: str,
    n_categories: int,
) -> ContrastFitCohort:
    """The rows, predictors and candidate count one contrast rung is fit on.

    The population is every event (``"all"``), the SWR events (``"SWR"``) or the
    others (``"nonSWR"``), by ``is_swr``. A rung of
    :data:`~continuum_swr_replay.analysis.behavior.model_specs.LEAF_ONLY_CONTRAST_RUNGS`
    is fit over the six leaf arms
    (:func:`~continuum_swr_replay.analysis.behavior.model_specs.leaf_only_cohort`):
    the trunks carry a structural zero for every leaf-derived belief covariate,
    so they add nothing about those constructs, and a fraction response is
    rescaled to unit mass over the leaves.

    Parameters
    ----------
    cohort : pd.DataFrame
        The assembled arm or patch cohort, one row per event x candidate, with
        ``is_swr`` (and ``candidate_arm`` for a leaf-only rung).
    rung : str
        The rung's name.
    predictors : sequence of str
        The rung's predictors.
    population : {"all", "SWR", "nonSWR"}
        Which events.
    response_col : str
        The resolved response column.
    response : {"fraction", "hard_count"}
        The response the column came from; only a fraction is rescaled.
    n_categories : int
        Candidates per event in ``cohort``: 9 arms or 3 patches.

    Returns
    -------
    ContrastFitCohort
    """
    if population == "all":
        frame = cohort
    elif population == "SWR":
        frame = cohort[cohort.is_swr]
    elif population == "nonSWR":
        frame = cohort[~cohort.is_swr]
    else:
        raise ValueError(
            f"population must be one of {CONTRAST_POPULATIONS}, got {population!r}"
        )
    if rung not in LEAF_ONLY_CONTRAST_RUNGS:
        return ContrastFitCohort(frame, list(predictors), n_categories, False)
    leaves, renormalized = leaf_only_cohort(
        frame, response_col=response_col, response=response
    )
    return ContrastFitCohort(
        leaves, leaf_only_predictors(predictors), LEAF_ONLY_N_CATEGORIES, renormalized
    )
