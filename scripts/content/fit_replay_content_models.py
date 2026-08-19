#!/usr/bin/env python
"""Fit the replay content model hierarchy (M0-M7) on pre-computed data.

Primary model: Conditional Poisson regression (multinomial-count softmax).
Robustness check: Conditional Logit on dominant arm.

Loads the master long-format dataset from analysis/replay/prepare_data.py,
fits per-session models for each model in the hierarchy, compares nested
models via AIC (primary) and LRT, and saves results.

Usage:
    uv run python scripts/content/fit_replay_content_models.py                # base ladder
    uv run python scripts/content/fit_replay_content_models.py --models M0,M1,M2
    uv run python scripts/content/fit_replay_content_models.py --models M5,Mv,Mu,Mvu
    uv run python scripts/content/fit_replay_content_models.py --cv           # + CV
    uv run python scripts/content/fit_replay_content_models.py --logit        # + logit robustness
"""

import argparse
import logging
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.replay._helpers import add_arm_fixed_effects
from continuum_swr_replay.analysis.replay.cohort_audit import (
    cohort_record,
    drop_profile,
    event_drop_buckets,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.cohort_selection import (
    freeze_common_cohort,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.fitting import (
    fit_conditional_logit,
    fit_conditional_poisson,
    fit_null_model_conditional_logit,
)
from continuum_swr_replay.analysis.replay.model_checking import (
    cross_validate_conditional_poisson,
)
from continuum_swr_replay.analysis.replay.model_comparison import (
    run_full_hierarchy_comparison,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_HIERARCHY,
    MODEL_HIERARCHY_MUA,
    MODEL_HIERARCHY_SWR,
    MODEL_PATCH,
    MODEL_PATCH_MUA,
    MODEL_PATCH_SWR,
    default_content_model_names,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    TRACK_SEGMENT_TO_PATCH,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import read_table, table_exists

# Map event source to model hierarchy and input directory
_EVENT_SOURCE_CONFIG = {
    "non_local": {
        "hierarchy": MODEL_HIERARCHY,
        "input_dir_name": "replay_content",
        "table_name": "event_arm_table",
    },
    "SWR": {
        "hierarchy": MODEL_HIERARCHY_SWR,
        "input_dir_name": "contfrag_replay_content",
        "table_name": "event_arm_table_swr",
    },
    "MUA": {
        "hierarchy": MODEL_HIERARCHY_MUA,
        "input_dir_name": "contfrag_replay_content",
        "table_name": "event_arm_table_mua",
    },
}

# When --hierarchy patch is selected, the table and response column are
# different but the input directory is shared with the corresponding
# event-source. event_id remains the grouping key (globally unique).
# ``table_name`` here is informational only — the actual table name is
# resolved by ``resolve_contfrag_paths`` further down, which appends the
# brain-area suffix for the ContFrag sources. The SWR/MUA patch tables
# are built by ``prepare_contfrag_replay_content_data.py``.
_PATCH_OVERRIDES = {
    "non_local": {
        "hierarchy": MODEL_PATCH,
        "table_name": "event_patch_table",
    },
    "SWR": {"hierarchy": MODEL_PATCH_SWR, "table_name": "event_patch_table_swr"},
    "MUA": {"hierarchy": MODEL_PATCH_MUA, "table_name": "event_patch_table_mua"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

SESSION_COLS = ["animal", "day", "epoch"]

# A canonical content table carries more than whichever columns one requested
# model happens to use. These fields distinguish the current location-aware,
# future-invariant history schema from older pickles that can otherwise satisfy
# M0-M3 and silently produce a truncated ladder.
_CANONICAL_SCHEMA_COLUMNS = {
    "arm": (
        *SESSION_COLS,
        "event_id",
        "arm",
        "arm_count_hard",
        "location_state",
        "at_well",
        "location_well",
        "location_arm",
        "current_arm",
        "is_current",
        "visit_recency_demeaned",
        "arm_never_visited",
    ),
    "patch": (
        *SESSION_COLS,
        "event_id",
        "patch",
        "patch_count_hard",
        "location_state",
        "at_well",
        "location_well",
        "location_arm",
        "is_current_patch",
        "visit_recency_patch",
        "arm_never_visited_patch",
    ),
}

_LOCATION_STATES = frozenset({"at_well", "in_transit", "mixed", "unknown"})
_CANDIDATE_COLUMNS = {
    "arm": ("arm", ALL_ARMS),
    "patch": ("patch", tuple(sorted(set(TRACK_SEGMENT_TO_PATCH.values())))),
}

_CV_RESULT_COLUMNS = (
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


def _validate_canonical_table_contract(
    long_df: pd.DataFrame,
    *,
    hierarchy_kind: str,
    table_label: object,
) -> None:
    """Validate location and novelty semantics that identify current tables."""
    if long_df.empty:
        raise ValueError(f"Canonical content table {table_label} is empty")
    if long_df[[*SESSION_COLS, "event_id"]].isna().any(axis=None):
        raise ValueError(
            f"Canonical content table {table_label} has missing session/event keys"
        )

    candidate_col, expected_candidates = _CANDIDATE_COLUMNS[hierarchy_kind]
    n_candidates = len(expected_candidates)
    unit_cols = [*SESSION_COLS, "event_id"]
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
        SESSION_COLS
    ].nunique(dropna=False)
    if event_session_counts.gt(1).any(axis=None):
        raise ValueError(
            f"Canonical content table {table_label} reuses event_id across sessions"
        )

    context_cols = [
        *SESSION_COLS,
        "location_state",
        "at_well",
        "location_well",
        "location_arm",
    ]
    if hierarchy_kind == "arm":
        context_cols.append("current_arm")
    if (by_event[context_cols].nunique(dropna=False) != 1).any(axis=None):
        raise ValueError(
            f"Canonical content table {table_label} has event context that "
            "changes across candidate rows"
        )

    state = long_df["location_state"]
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
    current_col = "is_current" if hierarchy_kind == "arm" else "is_current_patch"
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

    for location_col in ("location_well", "location_arm"):
        if (
            long_df.loc[at_well, location_col].isna().any()
            or long_df.loc[~at_well, location_col].notna().any()
        ):
            raise ValueError(
                f"Canonical content table {table_label} must define "
                f"{location_col} only for at-well events"
            )
    if hierarchy_kind == "arm":
        same_current_reference = long_df["current_arm"].eq(long_df["location_arm"]) | (
            long_df["current_arm"].isna() & long_df["location_arm"].isna()
        )
        if not same_current_reference.all():
            raise ValueError(
                f"Canonical content table {table_label} requires current_arm to "
                "equal location_arm, including off-well missingness"
            )

    current_reference = (
        long_df["location_arm"]
        if hierarchy_kind == "arm"
        else long_df["location_arm"].map(TRACK_SEGMENT_TO_PATCH)
    )
    expected_current = long_df[candidate_col].eq(current_reference).astype(float)
    if not current.loc[at_well].eq(expected_current.loc[at_well]).all():
        relation = (
            "location_arm"
            if hierarchy_kind == "arm"
            else "the mapped location_arm patch"
        )
        raise ValueError(
            f"Canonical content table {table_label}: {current_col} must match "
            f"{relation} exactly"
        )

    response_col = "arm_count_hard" if hierarchy_kind == "arm" else "patch_count_hard"
    response = long_df[response_col]
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
            f"Canonical content table {table_label} requires {response_col} to "
            "contain finite nonnegative integer counts"
        )

    novelty_col = (
        "arm_never_visited" if hierarchy_kind == "arm" else "arm_never_visited_patch"
    )
    recency_col = (
        "visit_recency_demeaned" if hierarchy_kind == "arm" else "visit_recency_patch"
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


def resolve_contfrag_paths(
    event_source: str,
    brain_area: str,
    hierarchy_kind: str,
) -> tuple[str, str]:
    """Return ``(table_name, output_subdir)`` for a fit invocation.

    Pure helper extracted so the prep ↔ fitter filename contract has a
    direct unit-test target. ``event_source`` is one of
    ``"non_local"`` / ``"SWR"`` / ``"MUA"``; ``hierarchy_kind`` is
    ``"arm"`` / ``"patch"``.

    Non-local fits stay unsuffixed (their prep is single-area /
    single-source). ContFrag fits append ``_{brain_area}`` to the
    table name and ``_{source_lower}_{brain_area}`` to the output
    subdir so SWR/HPC, SWR/mPFC, MUA/HPC … never overwrite each
    other's ``M0.pkl`` / ``all_models.pkl``.
    """
    if event_source == "non_local":
        if hierarchy_kind == "patch":
            return "event_patch_table", "models_patch"
        return "event_arm_table", "models"

    base_table = (
        f"event_patch_table_{event_source.lower()}"
        if hierarchy_kind == "patch"
        else f"event_arm_table_{event_source.lower()}"
    )
    table_name = f"{base_table}_{brain_area}"

    base_subdir = "models_patch" if hierarchy_kind == "patch" else "models"
    output_subdir = f"{base_subdir}_{event_source.lower()}_{brain_area}"
    return table_name, output_subdir


def prepare_canonical_fit_request(
    long_df: pd.DataFrame,
    *,
    hierarchy: Mapping,
    model_names: Sequence[str],
    hierarchy_kind: str,
    table_label: object,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], str, str]:
    """Validate one complete canonical fit request before fitting or writing.

    This is the runner's fail-closed boundary. It first requires the current
    location/novelty table schema, then derives arm fixed effects, and finally
    verifies *every* requested model and predictor as one atomic request. A
    stale table therefore cannot fit the subset of an otherwise nested ladder
    that it happens to satisfy.

    Returns a prepared copy of the table, the unchanged requested model names,
    their predictor union, and the canonical response/event column names.
    """
    try:
        required_schema = _CANONICAL_SCHEMA_COLUMNS[hierarchy_kind]
    except KeyError as exc:
        raise ValueError(
            f"hierarchy_kind must be 'arm' or 'patch', got {hierarchy_kind!r}"
        ) from exc

    missing_schema = [column for column in required_schema if column not in long_df]
    if missing_schema:
        raise ValueError(
            f"Canonical {hierarchy_kind} content table {table_label} is missing "
            "current-location/novelty schema column(s) "
            f"{missing_schema}. Regenerate it with the current prepare script."
        )
    _validate_canonical_table_contract(
        long_df,
        hierarchy_kind=hierarchy_kind,
        table_label=table_label,
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
    response_col = "patch_count_hard" if hierarchy_kind == "patch" else "arm_count_hard"
    return prepared, requested, predictors, response_col, "event_id"


def fit_model_per_session(
    long_df: pd.DataFrame,
    model_name: str,
    predictors: list[str],
    run_cv: bool = False,
    response_col: str = "arm_count_hard",
    event_col: str = "event_id",
    n_categories: int = 9,
    grain: str = "timebin",
) -> pd.DataFrame:
    """Fit conditional Poisson model across all sessions.

    Parameters
    ----------
    long_df : pd.DataFrame
        Canonical event-arm or event-patch table.
    model_name : str
        Model identifier (e.g., "M0").
    predictors : list[str]
        Predictor columns for this model.
    run_cv : bool
        If True, also run event-based cross-validation per session.
    response_col : str
        Column name for the count response.
    event_col : str
        Column name for the event grouping key.

    Returns
    -------
    pd.DataFrame
        One row per session with coefficients and fit statistics.
    """
    results = []

    # Minimum 2 events per session, scaled by candidate-set size:
    # n_categories=9 (arm) → 18 rows, n_categories=3 (patch) → 6 rows.
    # The previous hardcoded 18 silently skipped valid 2–5 event patch
    # sessions before fit_conditional_poisson could apply its own
    # n_events < 2 * len(predictors) gate.
    min_rows = n_categories * 2
    for keys, group in long_df.groupby(SESSION_COLS):
        if len(group) < min_rows:
            continue

        if grain == "event" and not predictors:
            r = fit_null_model_conditional_logit(
                group,
                event_col=event_col,
                response_col=response_col,
                n_arms=n_categories,
            )
        elif grain == "event":
            # One observation per event -- which arm won -- instead of one per
            # timebin. The timebin grain treats a median 32 bins per event as
            # independent multinomial draws when they are heavily
            # autocorrelated, so its standard errors are optimistic by roughly
            # 5x; collapsing makes n honest at the event level, at the cost of
            # discarding the within-event distribution (so "SWR spreads mass
            # over more arms" is not askable here).
            r = fit_conditional_logit(
                group,
                predictors=list(predictors),
                response_col=response_col,
                event_col=event_col,
            )
        else:
            r = fit_conditional_poisson(
                group,
                predictors=list(predictors),
                response_col=response_col,
                event_col=event_col,
                n_categories=n_categories,
            )
        if r is None:
            continue

        if run_cv and grain != "event":
            cv = cross_validate_conditional_poisson(
                group,
                predictors=list(predictors),
                event_col=event_col,
                response_col=response_col,
                n_categories=n_categories,
            )
            if cv is None:
                # ``--cv`` is a requested analysis, not a best-effort
                # annotation. Exclude this session here so the later shared-
                # session intersection cannot publish a primary fit without
                # its corresponding held-out result.
                continue
            r.update(cv)

        r["model"] = model_name
        for col, val in zip(SESSION_COLS, keys, strict=True):
            r[col] = val

        results.append(r)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def _validate_and_match_fit_results(
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
            SESSION_COLS, sort=False, observed=True, dropna=False
        )
    }
    session_sets: dict[str, set[tuple]] = {}

    for model_name in requested:
        result = fit_results[model_name]
        required = [*SESSION_COLS, "model", "n_events"]
        missing_columns = [column for column in required if column not in result]
        if missing_columns:
            raise RuntimeError(
                f"Fit result for {model_name} is missing integrity column(s) "
                f"{missing_columns}; no output will be published"
            )
        if result[SESSION_COLS].isna().any(axis=None):
            raise RuntimeError(
                f"Fit result for {model_name} has missing session keys; "
                "no output will be published"
            )
        if result.duplicated(SESSION_COLS).any():
            raise RuntimeError(
                f"Fit result for {model_name} has duplicate session rows; "
                "no output will be published"
            )
        if require_cv:
            missing_cv = [
                column for column in _CV_RESULT_COLUMNS if column not in result
            ]
            if missing_cv:
                raise RuntimeError(
                    f"Requested CV result for {model_name} is missing column(s) "
                    f"{missing_cv}; no output will be published"
                )
            cv_counts = result[list(_CV_COUNT_COLUMNS)].apply(
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
            session = tuple(getattr(row, column) for column in SESSION_COLS)
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
        session_index = pd.MultiIndex.from_frame(result[SESSION_COLS])
        keep = session_index.isin(shared_sessions)
        matched_result = (
            result.loc[keep]
            .sort_values(SESSION_COLS, kind="stable")
            .reset_index(drop=True)
        )
        matched_sessions = set(
            matched_result[SESSION_COLS].itertuples(index=False, name=None)
        )
        if matched_sessions != shared_sessions:
            raise RuntimeError(
                f"Internal shared-session alignment failed for {model_name}; "
                "no output will be published"
            )
        matched[model_name] = matched_result

    return matched


def _publish_staged_directory(staged_dir: Path, output_dir: Path) -> None:
    """Swap a fully written staged bundle into place, restoring on failure.

    Directories cannot be swapped in a single rename, so there is a window
    between moving the old bundle aside and moving the new one in. Exceptions
    unwind it; a kill signal cannot. If the process dies inside that window the
    output directory is absent and the previous bundle is intact next to it, in
    a directory named ``.<output-name>.previous-*`` -- rename it back.
    """
    backup_root: Path | None = None
    backup_dir: Path | None = None
    if output_dir.exists():
        backup_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.previous-", dir=output_dir.parent
            )
        )
        backup_dir = backup_root / "output"
        output_dir.replace(backup_dir)

    try:
        staged_dir.replace(output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists():
            backup_dir.replace(output_dir)
        if backup_root is not None and backup_root.exists():
            shutil.rmtree(backup_root)
        raise
    else:
        if backup_root is not None:
            try:
                shutil.rmtree(backup_root)
            except OSError:
                logging.warning(
                    "Published %s but could not remove previous-output backup %s",
                    output_dir,
                    backup_root,
                    exc_info=True,
                )


def _write_complete_output_bundle(
    *,
    output_dir: Path,
    model_results: Mapping[str, pd.DataFrame],
    combined: pd.DataFrame,
    cohort_audits: list[dict],
    comparison: pd.DataFrame,
    logit_results: Mapping[str, pd.DataFrame],
) -> None:
    """Stage every artifact, then replace the output directory as one bundle."""
    staged_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        for model_name, result in model_results.items():
            result.to_pickle(staged_dir / f"{model_name}.pkl")
        combined.to_pickle(staged_dir / "all_models.pkl")
        write_cohort_audit(cohort_audits, staged_dir / "all_models_cohort_audit.csv")
        if not comparison.empty:
            comparison.to_pickle(staged_dir / "model_comparison.pkl")
        if logit_results:
            logit_dir = staged_dir / "logit"
            logit_dir.mkdir()
            for model_name, result in logit_results.items():
                result.to_pickle(logit_dir / f"{model_name}_logit.pkl")
        _publish_staged_directory(staged_dir, output_dir)
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)


def print_model_results(
    model_name: str,
    mdf: pd.DataFrame,
    predictors: list[str],
    hierarchy: dict | None = None,
) -> None:
    """Print summary for one model."""
    if hierarchy is None:
        hierarchy = MODEL_HIERARCHY
    print(f"\n{'=' * 60}")
    spec = hierarchy[model_name]
    print(f"{model_name}: {spec.description}")
    print(f"{'=' * 60}")

    print(f"\nFit quality (n={len(mdf)} sessions):")
    for metric in ["aic", "bic", "llf"]:
        if metric in mdf.columns:
            vals = mdf[metric].dropna()
            if len(vals) > 0:
                print(f"  {metric}: median={vals.median():.2f}")

    if not predictors:
        print("  (null model — uniform allocation)")
        return

    # Coefficients with FDR correction
    raw_pvals = []
    coef_info = {}
    for pred in predictors:
        col = f"coef_{pred}"
        if col not in mdf.columns:
            continue
        vals = mdf[col].dropna()
        coef_info[pred] = vals
        if len(vals) >= 5:
            try:
                _, p = wilcoxon(vals)
                raw_pvals.append((pred, p))
            except ValueError:
                pass

    corrected_map = {}
    if raw_pvals:
        _, corrected, _, _ = multipletests([x[1] for x in raw_pvals], method="fdr_bh")
        corrected_map = {x[0]: c for x, c in zip(raw_pvals, corrected, strict=True)}

    # PRIMARY evidence is at the animal level, matching the SWR-contrast ladder
    # (see contrast_aggregation's module docstring). Sessions within an animal are
    # not independent and the corpus is unbalanced -- two animals supply over half
    # the sessions and one supplies 6% -- so a session-level median is largely a
    # statement about those two, and a session-level p treats correlated
    # observations as independent. The per-animal median is balanced across
    # animals, and with n=5 the sign consistency IS the claim; the Wilcoxon stays
    # for cross-session consistency but is descriptive, not the population test.
    n_animals = mdf["animal"].nunique() if "animal" in mdf.columns else 0
    print(
        f"\nCoefficients (animal-level primary, n={n_animals} animals; "
        "session median/p descriptive):"
    )
    header = f"  {'Predictor':<25} {'AnimalMed':>10} {'agree':>7} {'SessMed':>9} {'SE':>8} {'sess p':>10}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for pred in predictors:
        if pred not in coef_info:
            continue
        vals = coef_info[pred]
        se_col = f"bse_{pred}"
        se = mdf[se_col].median() if se_col in mdf.columns else float("nan")
        p = corrected_map.get(pred)
        p_str = f"{p:.2e}" if p is not None else "—"
        if n_animals:
            per_animal = (
                mdf.dropna(subset=[f"coef_{pred}"])
                .groupby("animal")[f"coef_{pred}"]
                .median()
            )
            animal_med = per_animal.median()
            n_pos = int((per_animal > 0).sum())
            # Sign consistency is reported toward the effect's own direction, so
            # "5/5" always means unanimous rather than "unanimously positive".
            agree = f"{max(n_pos, len(per_animal) - n_pos)}/{len(per_animal)}"
        else:
            animal_med, agree = float("nan"), "—"
        print(
            f"  {pred:<25} {animal_med:10.3f} {agree:>7} {vals.median():9.3f} "
            f"{se:8.3f} {p_str:>10}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all M0-M7)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run event-based cross-validation per session (slower)",
    )
    parser.add_argument(
        "--grain",
        type=str,
        default="timebin",
        choices=["timebin", "event"],
        help=(
            "Observation grain. 'timebin' (default) fits the conditional "
            "Poisson over per-arm timebin counts. 'event' collapses each event "
            "to a single observation -- which arm won -- and fits the "
            "conditional logit, so n is the number of events rather than the "
            "number of timebins. The timebin grain treats ~32 autocorrelated "
            "bins per event as independent draws and its SEs are optimistic by "
            "roughly 5x; the event grain is honest about n but cannot see the "
            "within-event distribution. Written to a '_event_grain' output "
            "directory so the two never overwrite each other."
        ),
    )
    parser.add_argument(
        "--logit",
        action="store_true",
        help="Also fit conditional logit on dominant arm (robustness)",
    )
    parser.add_argument(
        "--event-source",
        type=str,
        default="non_local",
        choices=["non_local", "SWR", "MUA"],
        help="Event source: non_local (model-detected), SWR, or MUA (default: non_local)",
    )
    parser.add_argument(
        "--hierarchy",
        type=str,
        default="arm",
        choices=["arm", "patch"],
        help=(
            "Resolution: 'arm' (9 arms, M0-M7) or 'patch' (3 patches, "
            "P0-P4). Patch fits use event_patch_table.pkl and "
            "patch_count_hard."
        ),
    )
    parser.add_argument(
        "--swr-subset",
        type=str,
        default="all",
        choices=["all", "T", "F"],
        help=(
            "Restrict to the SWR-overlap subset of events. 'T' keeps "
            "events with is_swr=True; 'F' keeps is_swr=False. With a "
            "subset the hierarchy is reduced to MODEL_HIERARCHY_SWR / "
            "MODEL_PATCH_SWR (i.e. M0-M5 / P0-P3) since swr_x_* "
            "interactions are constant within a subset. Output is "
            "written to a suffixed subdirectory (e.g. models_swr_T)."
        ),
    )
    parser.add_argument(
        "--brain-area",
        type=str,
        default="HPC",
        choices=["HPC", "mPFC", "OFC"],
        help=(
            "Brain area suffix used when resolving ContFrag prep "
            "outputs (event_arm_table_{source}_{brain_area}.pkl). "
            "Ignored for the non_local event source, which is built "
            "by ``prepare_replay_content_data.py`` and is not "
            "partitioned per area."
        ),
    )
    args = parser.parse_args()

    # Select hierarchy and input paths based on event source. The
    # --hierarchy flag overrides the model dict and table name when
    # 'patch' is selected; the input directory follows the event source.
    source_config = _EVENT_SOURCE_CONFIG[args.event_source]
    if args.hierarchy == "patch":
        override = _PATCH_OVERRIDES[args.event_source]
        hierarchy = override["hierarchy"]
        table_name = override["table_name"]
        if hierarchy is None:
            raise NotImplementedError(
                f"Patch hierarchy not defined for event source {args.event_source!r}. "
                "Add an entry to _PATCH_OVERRIDES pointing at a MODEL_PATCH "
                "variant in model_specs.py."
            )
    else:
        hierarchy = source_config["hierarchy"]
        table_name = source_config["table_name"]
        if hierarchy is None:
            raise NotImplementedError(
                f"Event source {args.event_source!r} has no MODEL_HIERARCHY defined "
                "in analysis/replay/model_specs.py. Define it before running."
            )
    # The (table_name, output_subdir) contract with the prep scripts
    # lives in ``resolve_contfrag_paths`` so the path-resolution logic
    # has a direct unit-test target rather than only being exercised
    # at the end of the long ``main()``. The hierarchy / source
    # validity gates above are unchanged; this only resolves names.
    table_name, output_subdir = resolve_contfrag_paths(
        args.event_source, args.brain_area, args.hierarchy
    )

    input_dir = PROCESSED_DATA_DIR / source_config["input_dir_name"]
    output_dir = input_dir / output_subdir

    # SWR subset: restrict to is_swr=True or =False events. Within a
    # subset is_swr is constant, so the swr_x_* interactions (M6/M7,
    # P4) collapse onto their main effects and are unidentifiable —
    # use the *_SWR variants (M0-M5 / P0-P3) which already drop them.
    # Output goes to a suffixed subdirectory so the unrestricted fits
    # are not overwritten.
    if args.swr_subset != "all":
        if args.hierarchy == "patch":
            hierarchy = MODEL_PATCH_SWR
        else:
            hierarchy = MODEL_HIERARCHY_SWR
        output_dir = input_dir / f"{output_subdir}_swr_{args.swr_subset}"
        logging.info(
            "SWR subset = %s: hierarchy narrowed to %s, output -> %s",
            args.swr_subset,
            list(hierarchy.keys()),
            output_dir,
        )

    # Applied last so it composes with the SWR-subset suffix rather than being
    # overwritten by it: an event-grain SWR-subset run needs its own directory
    # too, or it would silently publish over the timebin-grain subset fits.
    if args.grain == "event":
        output_dir = output_dir.with_name(f"{output_dir.name}_event_grain")
        logging.info("grain = event (one observation per event) -> %s", output_dir)

    # Determine which models to fit
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        # Belief-dependent rungs require contingency/qvals columns, which are NaN
        # for unmapped stops. Including Mv/Mu/Mvu/Pv in a default run would shrink
        # the shared complete-case set for the whole base ladder. Request them
        # explicitly on their own matched cohort (e.g. --models M5,Mv,Mu,Mvu or
        # --models P3,Pv).
        model_names = list(default_content_model_names(hierarchy))

    # Canonical inputs are mandatory. The former implicit ``long_count.pkl``
    # fallback could fit M0-M3 from a stale schema and write those partial
    # results into the canonical ``models/`` directory.
    canonical_path = input_dir / f"{table_name}.pkl"
    legacy_path = input_dir / "long_count.pkl"
    if not table_exists(canonical_path):
        legacy_note = (
            f" Found legacy artifact {legacy_path}, but it cannot be used "
            "for canonical fits."
            if legacy_path.exists()
            else ""
        )
        raise FileNotFoundError(
            f"Canonical content table {canonical_path} was not found.{legacy_note} "
            "Legacy fallback is disabled; regenerate the canonical table with "
            "the appropriate prepare script."
        )

    logging.info("Loading %s...", canonical_path)
    long_df = read_table(canonical_path)
    (
        long_df,
        valid_model_names,
        union_predictors,
        response_col,
        event_col,
    ) = prepare_canonical_fit_request(
        long_df,
        hierarchy=hierarchy,
        model_names=model_names,
        hierarchy_kind=args.hierarchy,
        table_label=canonical_path,
    )
    logging.info(
        "Loaded %d rows, %d sessions",
        len(long_df),
        long_df.groupby(SESSION_COLS).ngroups,
    )

    # Apply the SWR subset filter (--swr-subset T/F). is_swr is
    # event-level on the event_arm_table and patch table. Drop whole
    # events / patch-rows where is_swr doesn't match.
    if args.swr_subset != "all":
        if "is_swr" not in long_df.columns:
            raise ValueError(
                "--swr-subset requires canonical column 'is_swr'; not found in "
                f"{canonical_path}. Regenerate the table."
            )
        keep = long_df["is_swr"] if args.swr_subset == "T" else ~long_df["is_swr"]
        n_before_rows = len(long_df)
        long_df = long_df.loc[keep].reset_index(drop=True)
        logging.info(
            "SWR subset filter (%s): %d -> %d rows",
            args.swr_subset,
            n_before_rows,
            len(long_df),
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
    n_complete_case = n_before - cohort_selection["n_missing_units"]
    if n_before != n_complete_case:
        logging.info(
            "Filtered to common complete-case events: %d → %d events "
            "(%d dropped due to NaN in %s)",
            n_before,
            n_complete_case,
            cohort_selection["n_missing_units"],
            list(union_predictors),
        )
    if cohort_selection["n_zero_response_units"]:
        logging.info(
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
            f"predictor in {list(union_predictors)} is NaN on some candidate "
            "row of every event. Regenerate the canonical table."
        )
    if cohort_selection["n_units_after"] == 0:
        raise RuntimeError(
            f"Common-cohort filtering removed all {n_before} events because "
            "every complete-case event had zero total response."
        )

    logging.info(
        "Fitting on %d rows, %d sessions, %d events",
        len(long_df),
        long_df.groupby(SESSION_COLS).ngroups,
        long_df[event_col].nunique(),
    )

    # Fit every requested rung into memory. Nothing under ``output_dir`` is
    # created or modified until the whole ladder and its shared cohort pass
    # integrity checks, so a late empty/failed rung cannot leave mixed results.
    fit_results: dict[str, pd.DataFrame] = {}
    for model_name in valid_model_names:
        spec = hierarchy[model_name]
        predictors = list(spec.predictors)
        cv_label = " + CV" if args.cv else ""
        logging.info(
            "Fitting %s (%d predictors%s)...", model_name, len(predictors), cv_label
        )
        t0 = time.time()

        # Patch hierarchy collapses 9 arms to 3 patches; the null /
        # pseudo-R² / CV baselines must use log(3) instead of log(9)
        # for P0–P3 to be on the right chance footing.
        n_categories = 3 if args.hierarchy == "patch" else 9

        mdf = fit_model_per_session(
            long_df,
            model_name,
            predictors,
            run_cv=args.cv,
            response_col=response_col,
            event_col=event_col,
            n_categories=n_categories,
            grain=args.grain,
        )

        elapsed = time.time() - t0
        logging.info("  %s: %d session fits in %.1fs", model_name, len(mdf), elapsed)
        fit_results[model_name] = mdf

    model_results = _validate_and_match_fit_results(
        fit_results,
        requested_model_names=valid_model_names,
        cohort=long_df,
        event_col=event_col,
        require_cv=args.cv,
    )
    shared_session_count = len(next(iter(model_results.values())))
    if any(len(result) != shared_session_count for result in model_results.values()):
        raise RuntimeError(
            "Internal content-ladder session matching failed; no output will be "
            "published"
        )
    logging.info(
        "Validated complete %d-rung ladder on %d shared sessions",
        len(model_results),
        shared_session_count,
    )

    # Build every audit record only after the model frames have been restricted
    # to the exact same successfully fitted session cohort.
    cohort_audits: list[dict] = []
    population = getattr(args, "swr_subset", None) or "all"
    sessions_in = long_df_prefilter.groupby(SESSION_COLS).ngroups
    for model_name, mdf in model_results.items():
        predictors = list(hierarchy[model_name].predictors)

        # Cohort-audit record (grain=event, common complete-case cohort). Attribute
        # the union-filter drops to this rung's own vs the other rungs' predictors.
        buckets = event_drop_buckets(
            long_df_prefilter,
            event_col=event_col,
            response_col=response_col,
            own_predictors=predictors,
            union_predictors=union_predictors,
        )
        n_modeled = int(mdf["n_events"].sum()) if len(mdf) else 0
        n_drops = sum(
            buckets[k]
            for k in (
                "n_drop_qvals_gap",
                "n_drop_own_predictor_missing",
                "n_drop_common_cohort_other",
                "n_drop_zero_response",
            )
        )
        dropped = long_df_prefilter[
            long_df_prefilter[event_col].isin(buckets["dropped_event_ids"])
        ].drop_duplicates(event_col)
        cohort_audits.append(
            cohort_record(
                model_name,
                population,
                "event",
                cohort_mode="common",
                n_in=n_before,
                n_modeled=n_modeled,
                sessions_in=sessions_in,
                sessions_modeled=len(mdf),
                animals_modeled=int(mdf["animal"].nunique()) if len(mdf) else 0,
                drop_notes=drop_profile(dropped),
                n_drop_qvals_gap=buckets["n_drop_qvals_gap"],
                n_drop_own_predictor_missing=buckets["n_drop_own_predictor_missing"],
                n_drop_common_cohort_other=buckets["n_drop_common_cohort_other"],
                n_drop_zero_response=buckets["n_drop_zero_response"],
                n_excluded_session=n_before - n_drops - n_modeled,
                n_sessions_flagged_nonconverged=(
                    int((mdf["converged"] == False).sum())  # noqa: E712
                    if "converged" in mdf.columns
                    else 0
                ),
                n_sessions_flagged_separated=(
                    int((mdf["separated"] == True).sum())  # noqa: E712
                    if "separated" in mdf.columns
                    else 0
                ),
            )
        )

        print_model_results(model_name, mdf, predictors, hierarchy)

    combined = pd.concat(model_results.values(), ignore_index=True)
    combined_models = tuple(combined["model"].drop_duplicates())
    if combined_models != tuple(valid_model_names):
        raise RuntimeError(
            f"Combined fit contains models {combined_models}, expected "
            f"{tuple(valid_model_names)}; no output will be published"
        )

    # Compute comparisons before publication as part of the complete bundle.
    comparison = pd.DataFrame()
    if len(model_results) >= 2:
        logging.info("Running nested model comparisons...")
        comparison = run_full_hierarchy_comparison(combined, hierarchy=hierarchy)
        if len(comparison) > 0:
            print(f"\n{'=' * 60}")
            print("MODEL COMPARISON (positive ΔAIC = child better)")
            print(f"{'=' * 60}")
            print(
                f"\n  {'Step':<12} {'ΔAIC animal':>12} {'animals':>8} "
                f"{'ΔAIC sess':>11} {'Child wins':>12} {'LRT sig':>10} {'sess p':>11}"
            )
            print(f"  {'-' * 78}")
            for _, row in comparison.iterrows():
                step = f"{row['parent']}→{row['child']}"
                wp = f"{row['wilcoxon_p']:.2e}" if pd.notna(row["wilcoxon_p"]) else "—"
                am = row.get("delta_aic_animal_median", float("nan"))
                am_str = f"{am:.1f}" if pd.notna(am) else "—"
                agree = (
                    f"{int(row.get('n_animals_child_better', 0))}/"
                    f"{int(row.get('n_animals', 0))}"
                    if row.get("n_animals", 0)
                    else "—"
                )
                print(
                    f"  {step:<12} {am_str:>12} {agree:>8} "
                    f"{row['delta_aic_median']:>11.1f} "
                    f"{row['child_better_aic']:>8}/{row['n_sessions']:<3} "
                    f"{row['lrt_significant']:>6}/{row['n_sessions']:<3} "
                    f"{wp:>11}"
                )

    # Conditional-logit robustness results are also collected before the output
    # bundle is staged. Restrict them to the primary ladder's shared sessions.
    shared_sessions = set(
        next(iter(model_results.values()))[SESSION_COLS].itertuples(
            index=False, name=None
        )
    )
    long_session_index = pd.MultiIndex.from_frame(long_df[SESSION_COLS])
    matched_long_df = long_df.loc[long_session_index.isin(shared_sessions)]
    logit_outputs: dict[str, pd.DataFrame] = {}
    if args.logit:
        logging.info("Fitting conditional logit robustness checks...")
        expected_logit_models = tuple(
            model_name
            for model_name in valid_model_names
            if hierarchy[model_name].predictors
        )
        for model_name in expected_logit_models:
            spec = hierarchy[model_name]
            predictors = list(spec.predictors)

            model_logit_results = []
            for keys, group in matched_long_df.groupby(SESSION_COLS):
                # Forward the resolved response column. Patch fits use
                # ``patch_count_hard``; without this, the logit call
                # silently fell back to ``arm_count_hard`` (the
                # ``fit_conditional_logit`` default), found no such
                # column on patch tables, and returned ``None`` for
                # every session — producing an empty ``logit/`` dir
                # with no robustness outputs.
                r = fit_conditional_logit(
                    group,
                    predictors=predictors,
                    event_col=event_col,
                    response_col=response_col,
                )
                if r is None:
                    raise RuntimeError(
                        "Requested conditional-logit robustness fit failed for "
                        f"{model_name}/{tuple(keys)}; no output will be published"
                    )
                r["model"] = model_name
                for col, val in zip(SESSION_COLS, keys, strict=True):
                    r[col] = val
                model_logit_results.append(r)

            logit_outputs[model_name] = pd.DataFrame(model_logit_results)

        if expected_logit_models:
            logit_outputs = _validate_and_match_fit_results(
                logit_outputs,
                requested_model_names=expected_logit_models,
                cohort=matched_long_df,
                event_col=event_col,
            )
            for model_name, logit_df in logit_outputs.items():
                logit_sessions = set(
                    logit_df[SESSION_COLS].itertuples(index=False, name=None)
                )
                if logit_sessions != shared_sessions:
                    raise RuntimeError(
                        "Requested conditional-logit robustness fits do not match "
                        f"the primary cohort for {model_name}; no output will be "
                        "published"
                    )
                logging.info(
                    "  %s logit: %d matched sessions staged",
                    model_name,
                    len(logit_df),
                )

    # Only now, after all requested fits and derived artifacts are valid, write
    # to a sibling staging directory and swap the complete directory into place.
    _write_complete_output_bundle(
        output_dir=output_dir,
        model_results=model_results,
        combined=combined,
        cohort_audits=cohort_audits,
        comparison=comparison,
        logit_results=logit_outputs,
    )
    logging.info(
        "Published %d model files and complete audit bundle to %s",
        len(model_results),
        output_dir,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
