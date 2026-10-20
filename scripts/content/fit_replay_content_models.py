#!/usr/bin/env python
"""Fit the replay content model hierarchy (M0-M6) on pre-computed data.

Primary model: Conditional Poisson regression (multinomial-count softmax).
Robustness check: Conditional Logit on dominant arm.

Loads the master long-format dataset from analysis/replay/prepare_data.py
(built by scripts/decoding/prepare_replay_content_data.py),
fits per-session models for each model in the hierarchy, compares nested
models via AIC (primary) and LRT, and saves results.

Inference note — read the standard errors honestly. On the default
``--grain timebin`` fit, a single session's coefficient standard errors are
OPTIMISTIC: the ~32 autocorrelated per-event timebins are treated as
independent draws, so one session's SEs are too small by roughly 5x. A
per-session SE is therefore not a calibrated uncertainty. Population inference
comes from the across-session distribution of the per-session coefficients
(the sign-rank / FDR aggregation over sessions), never from a single fit's SE;
``--grain event`` collapses each event to one observation for an honest
per-session n, at the cost of the within-event distribution.

The shipped response is the per-event allocation FRACTION (``arm_fraction`` /
``patch_fraction``): each event contributes one observation of unit mass, so the
likelihood, AIC and standard errors are stated at the event grain rather than
weighting an event by its number of decoded timebins. ``--response hard_count``
fits the counts instead, for the retired-count comparison, and publishes to its
own ``_hard_count`` directory.

Any run that is not the full default ladder is fit on its own complete-case
cohort, so it publishes into a directory keyed by the rungs it requested
(``models_M5-Mv-Mu-Mvu/``) rather than into the canonical ``models/`` bundle.
``--leaf-only`` conditions on the six leaf arms rather than all nine candidates
-- the Beta-belief covariates are structural zeros on the trunks -- which is a
six-category likelihood and so gets its own ``_leaf`` bundle; the belief rungs'
parents must be requested in the same run to keep the AIC comparisons nested.
Every pickle in a published bundle carries a ``.provenance.json`` sidecar naming
the table it was fit on and that table's SHA-256 as of the read, which the
figure and summary scripts check before reading the fit.

``--estimator dirichlet_multinomial`` replaces the conditional Poisson with a
softmax whose within-event concentration is estimated rather than assumed, so an
event's ~32 autocorrelated timebins stop counting as independent draws and
``coef_log_phi_is_swr`` says directly whether SWR events spread their mass over
more arms. It is defined on the integer counts, so it forces ``--response
hard_count``, and it publishes to its own ``_dm`` bundle: its likelihood keeps
the multinomial coefficient the conditional one drops, so its AIC is on a
different scale and must never be differenced against the conditional bundle's.

Usage:
    uv run python scripts/content/fit_replay_content_models.py                # base ladder
    uv run python scripts/content/fit_replay_content_models.py --response hard_count
    uv run python scripts/content/fit_replay_content_models.py --estimator dirichlet_multinomial
    uv run python scripts/content/fit_replay_content_models.py --models M0,M1,M2   # -> models_M0-M1-M2/
    uv run python scripts/content/fit_replay_content_models.py --models M5,Mv,Mu,Mvu
    uv run python scripts/content/fit_replay_content_models.py --leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu
    uv run python scripts/content/fit_replay_content_models.py --cv           # + CV
    uv run python scripts/content/fit_replay_content_models.py --logit        # + logit robustness
"""

import argparse
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.hippocampus.swr_source import require_swr_source
from continuum_swr_replay.analysis.replay._helpers import add_arm_fixed_effects
from continuum_swr_replay.analysis.replay.cohort_audit import (
    cohort_record,
    drop_profile,
    estimator_skip_note,
    event_drop_buckets,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.cohort_selection import (
    freeze_common_cohort,
    select_event_cohort,
)
from continuum_swr_replay.analysis.replay.content_response import (
    ESTIMATOR_CHOICES,
    RESPONSE_CHOICES,
    default_response_for_estimator,
    renormalize_fraction_over_candidates,
    resolve_response_choice,
    resolve_response_column,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    unidentified_for_term,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.dirichlet_multinomial import (
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    DEFAULT_N_JOBS,
    ESTIMATOR_SKIP_CLASSES,
    concentration_cols_for_cohort,
    fit_sessions_dirichlet_multinomial,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    record_inputs,
    write_fit_provenance,
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
    LEAF_ONLY_N_CATEGORIES,
    MODEL_HIERARCHY,
    MODEL_HIERARCHY_MUA,
    MODEL_HIERARCHY_SWR,
    MODEL_PATCH,
    MODEL_PATCH_MUA,
    MODEL_PATCH_SWR,
    ModelSpec,
    default_content_model_names,
    leaf_only_predictors,
)
from continuum_swr_replay.analysis.replay.table_contracts import (
    CV_RESULT_COLUMNS,
    validate_and_match_fit_results,
    validate_canonical_table_contract,
    validate_requested_standardization,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, table_exists
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

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

# The CLI's accepted values, as constants: the canonical output-directory names
# are the product of these, so a new choice cannot be added to the parser without
# also entering the set of bundles a partial run is forbidden to overwrite.
EVENT_SOURCE_CHOICES = ("non_local", "SWR", "MUA")
BRAIN_AREA_CHOICES = ("HPC", "mPFC", "OFC")
HIERARCHY_CHOICES = ("arm", "patch")
SWR_SUBSET_CHOICES = ("all", "T", "F")
GRAIN_CHOICES = ("timebin", "event")


def resolve_contfrag_paths(
    event_source: str,
    brain_area: str,
    hierarchy_kind: str,
) -> tuple[str, str]:
    """Return ``(table_name, output_subdir)`` for a fit invocation.

    Pure helper extracted so the prep ↔ fitter filename contract has a
    direct unit-test target. ``event_source`` is one of
    ``"non_local"`` / ``"SWR"`` / ``"MUA"``; ``hierarchy_kind`` is
    ``"candidate_arm"`` / ``"patch"``.

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


def canonical_output_subdir(
    event_source: str,
    brain_area: str,
    hierarchy_kind: str,
    swr_subset: str,
    grain: str,
    response: str = "fraction",
    leaf_only: bool = False,
    estimator: str = "conditional_poisson",
) -> str:
    """Directory name a FULL-ladder run of one configuration publishes into.

    The base name comes from :func:`resolve_contfrag_paths`; the SWR subset, the
    event grain, the response scale and the candidate set each append a suffix,
    in that order, so an event-grain subset run has its own bundle rather than
    publishing over the timebin one.

    Parameters
    ----------
    event_source : str
        One of :data:`EVENT_SOURCE_CHOICES`.
    brain_area : str
        One of :data:`BRAIN_AREA_CHOICES` (ignored for ``non_local``).
    hierarchy_kind : str
        One of :data:`HIERARCHY_CHOICES`.
    swr_subset : str
        One of :data:`SWR_SUBSET_CHOICES`; ``"all"`` adds no suffix.
    grain : str
        One of :data:`GRAIN_CHOICES`; ``"timebin"`` adds no suffix.
    response : str, optional
        One of :data:`RESPONSE_CHOICES`; the shipped ``"fraction"`` adds no
        suffix. A count fit states its likelihood, AIC and SEs at the timebin
        grain rather than the event grain, so its rungs are not comparable with
        a fraction bundle's and must not replace them.
    leaf_only : bool, optional
        Whether the fit conditions on the six leaf arms instead of all nine
        candidates. A six-category conditional likelihood is on a different
        scale from a nine-category one, so a leaf-only ladder gets its own
        ``_leaf`` bundle.
    estimator : str, optional
        One of :data:`~continuum_swr_replay.analysis.replay.content_response.ESTIMATOR_CHOICES`.
        The Dirichlet-multinomial appends ``_dm`` LAST, after every other
        suffix, so a reader can strip it to find the conditional-Poisson bundle
        the fit is the overdispersed counterpart of. Its likelihood carries the
        multinomial coefficient the conditional one drops, so its AIC is not on
        the conditional bundle's scale and must never replace it. The response
        suffix is omitted where the response is the estimator's own default,
        which is what the ``_hard_count`` suffix would otherwise say twice.

    Returns
    -------
    str
        The bundle directory name, relative to the run's input directory.

    Raises
    ------
    ValueError
        If ``leaf_only`` is requested for the patch hierarchy, which has no
        leaf subset.
    """
    if leaf_only and hierarchy_kind == "patch":
        raise ValueError(
            "leaf_only is an arm-hierarchy candidate set; the three patches are "
            "not arms and have no leaf subset."
        )
    _table_name, subdir = resolve_contfrag_paths(
        event_source, brain_area, hierarchy_kind
    )
    if swr_subset != "all":
        subdir = f"{subdir}_swr_{swr_subset}"
    if grain == "event":
        subdir = f"{subdir}_event_grain"
    if response != default_response_for_estimator(estimator):
        subdir = f"{subdir}_{response}"
    if leaf_only:
        subdir = f"{subdir}_leaf"
    if estimator != "conditional_poisson":
        subdir = f"{subdir}_dm"
    return subdir


def canonical_output_subdirs() -> frozenset[str]:
    """Every canonical bundle name, over all CLI configurations.

    A partial ladder is fit on its own complete-case cohort, so it must not be
    written into ANY of these -- not just the one the current invocation would
    have used. Built from :func:`canonical_output_subdir` over the product of
    the CLI choice tuples, so it cannot drift from the names full runs produce.
    """
    return frozenset(
        canonical_output_subdir(
            source, area, hierarchy_kind, subset, grain, response, leaf_only, estimator
        )
        for source in EVENT_SOURCE_CHOICES
        for area in BRAIN_AREA_CHOICES
        for hierarchy_kind in HIERARCHY_CHOICES
        for subset in SWR_SUBSET_CHOICES
        for grain in GRAIN_CHOICES
        for estimator in ESTIMATOR_CHOICES
        # Only the conditional Poisson has a choice of response; the
        # Dirichlet-multinomial is defined on counts and nothing else.
        for response in (
            RESPONSE_CHOICES
            if estimator == "conditional_poisson"
            else (default_response_for_estimator(estimator),)
        )
        # The patch hierarchy has no leaf-only variant to guard.
        for leaf_only in ((False,) if hierarchy_kind == "patch" else (False, True))
    )


# The leaf-only belief chain, as documented in the module usage block above and
# in docs/models_table.md: M0-M4 refit on the six leaves alongside Mv/Mu/Mvu so
# every rung shares one six-category conditional likelihood.
BELIEF_CHAIN_MODEL_NAMES: tuple[str, ...] = (
    "M0",
    "M0a",
    "M1",
    "M2",
    "M3",
    "M4",
    "Mv",
    "Mu",
    "Mvu",
)


def partial_ladder_subdir(canonical_subdir: str, model_names: Sequence[str]) -> str:
    """Bundle name for a run that fits ``model_names`` rather than a full ladder.

    A partial ladder is fit on its own complete-case cohort, so it is keyed by
    the rungs it fits and never written next to (or over) a canonical bundle.

    Parameters
    ----------
    canonical_subdir : str
        The name :func:`canonical_output_subdir` gives the same configuration.
    model_names : Sequence[str]
        The requested rungs, in ladder order.

    Returns
    -------
    str
        The bundle directory name, relative to the run's input directory.
    """
    return f"{canonical_subdir}_{'-'.join(model_names)}"


def belief_chain_output_subdir() -> str:
    """Bundle directory the leaf-only belief chain (``Mv``/``Mu``/``Mvu``) writes.

    Derived from the same helpers the runner uses, so a reader of ``Mu.pkl``
    cannot drift from where ``fit_replay_content_models.py --leaf-only --models
    M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu`` publishes it.

    Returns
    -------
    str
        The bundle directory name, relative to ``PROCESSED_DATA_DIR /
        "replay_content"``.
    """
    return partial_ladder_subdir(
        canonical_output_subdir(
            event_source="non_local",
            brain_area="HPC",  # ignored for non_local (see resolve_contfrag_paths)
            hierarchy_kind="arm",
            swr_subset="all",
            grain="timebin",
            response="fraction",
            leaf_only=True,
        ),
        BELIEF_CHAIN_MODEL_NAMES,
    )


def fit_n_categories(hierarchy_kind: str, leaf_only: bool) -> int:
    """Number of candidate categories the conditional likelihood conditions on.

    The null likelihood, pseudo-R² and cross-validation baselines are all stated
    against uniform allocation over this many slots, so it must match the rows
    actually fitted: nine arms, three patches, or the six leaf arms.

    Parameters
    ----------
    hierarchy_kind : str
        One of :data:`HIERARCHY_CHOICES`.
    leaf_only : bool
        Whether the trunk-arm rows were dropped from the cohort.

    Returns
    -------
    int
        ``3`` for the patch hierarchy, ``6`` for a leaf-only arm fit, ``9``
        otherwise.
    """
    if hierarchy_kind == "patch":
        return 3
    return LEAF_ONLY_N_CATEGORIES if leaf_only else 9


def rung_predictors(spec: ModelSpec, leaf_only: bool) -> list[str]:
    """Predictors one rung is fitted with, under the run's candidate set.

    On a leaf-only fit the arm dummies for arms 5-7 are identically zero (arm 5
    becomes the reference and the two trunk dummies have no rows left), so the
    design is rank-deficient unless they are dropped.

    Parameters
    ----------
    spec : ModelSpec
        The rung's specification.
    leaf_only : bool
        Whether the fit conditions on the six leaf arms.

    Returns
    -------
    list[str]
        Column names for this rung's design matrix.
    """
    predictors = list(spec.predictors)
    return leaf_only_predictors(predictors) if leaf_only else predictors


def restrict_to_leaf_arms(long_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the six leaf-arm candidate rows of every event.

    The trunk rows are dropped, not folded into the leaves: each retained row
    keeps the allocation it had over the nine candidates, so an event whose
    replay ran along a trunk contributes correspondingly less mass to the
    conditional likelihood.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format event × candidate-arm frame carrying ``candidate_arm``.

    Returns
    -------
    pd.DataFrame
        The leaf-arm rows, with a fresh index.

    Raises
    ------
    ValueError
        If ``candidate_arm`` is absent.
    """
    if "candidate_arm" not in long_df.columns:
        raise ValueError(
            "A leaf-only fit needs the candidate_arm column to identify the leaf "
            f"rows; columns are {list(long_df.columns)}."
        )
    leaves = long_df["candidate_arm"] < LEAF_ONLY_N_CATEGORIES
    return long_df.loc[leaves].reset_index(drop=True)


def leaf_only_cohort(
    long_df: pd.DataFrame,
    *,
    response_col: str,
    response: str,
    event_col: str = "event_id",
) -> tuple[pd.DataFrame, bool]:
    """Restrict a cohort to the leaf arms and put its response on that support.

    A fraction is a share of the event's whole nine-arm mass, and the
    conditional likelihood weights an event by the total it carries, so keeping
    the persisted fractions after dropping the trunk rows would weight an event
    by its leaf share. That share is not balanced across the comparisons these
    ladders exist for -- SWR events put a substantially larger part of their
    bins on the trunks than non-SWR events do -- so it is rescaled to unit mass
    over the six leaves. A count response is left alone: weighting an event by
    its number of decoded timebins is what the count response means.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format event x candidate-arm cohort, all nine arms.
    response_col : str
        The resolved response column.
    response : str
        The ``--response`` value the column came from, one of
        :data:`RESPONSE_CHOICES`.
    event_col : str, optional
        Event grouping key.

    Returns
    -------
    pd.DataFrame
        The leaf-arm rows, with the response rescaled when it is a fraction.
    bool
        Whether the response was rescaled, for the fit rows and the provenance.
    """
    leaves = restrict_to_leaf_arms(long_df)
    if response != "fraction":
        return leaves, False
    return (
        renormalize_fraction_over_candidates(
            leaves, response_col=response_col, event_col=event_col
        ),
        True,
    )


def content_fit_inputs(input_dir: Path, table_name: str) -> dict[str, Path]:
    """Name -> path of every table one content-ladder fit is built from.

    The ladder reads a single canonical candidate table, resolved from the event
    source and hierarchy. Recorded in the provenance sidecar of every artifact
    the run publishes, and re-derived by the readers of those artifacts, so a
    regenerated table cannot be read through a fit that predates it.

    Parameters
    ----------
    input_dir : Path
        The run's candidate-table root.
    table_name : str
        Table stem from :func:`resolve_contfrag_paths`.

    Returns
    -------
    dict of str to Path
        ``{table_name: <input_dir>/<table_name>.pkl}``.
    """
    return {table_name: Path(input_dir) / f"{table_name}.pkl"}


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

    This is the runner's fail-closed boundary. It first requires the current
    location/novelty table schema, then derives arm fixed effects, and finally
    verifies *every* requested model and predictor as one atomic request. A
    stale table therefore cannot fit the subset of an otherwise nested ladder
    that it happens to satisfy.

    ``response`` selects the allocation the ladder is fit on: the shipped
    per-event ``"fraction"`` or the timebin ``"hard_count"``. The resolved
    column is validated here, on the whole loaded table, so a table that cannot
    express the requested response fails before the first fit rather than
    halfway through the ladder.

    Returns a prepared copy of the table, the unchanged requested model names,
    their predictor union, and the resolved response/event column names.
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


def _fit_dirichlet_multinomial_per_session(
    long_df: pd.DataFrame,
    model_name: str,
    predictors: list[str],
    *,
    response_col: str,
    event_col: str,
    n_categories: int,
    min_rows: int,
    n_jobs: int,
    min_events_session_phi: int,
) -> pd.DataFrame:
    """Fit the Dirichlet-multinomial rung for every session of ``long_df``.

    The per-session fits run in a process pool with a pooled fallback for the
    short sessions; see
    :func:`~continuum_swr_replay.analysis.replay.dirichlet_multinomial.fit_sessions_dirichlet_multinomial`.
    Sessions the fitter refuses -- a rank-deficient within-event design raises,
    a session with too few events for its parameters returns nothing -- are
    recorded on the returned frame's ``attrs["estimator_skips"]`` so the cohort
    audit can name them, rather than crashing the ladder.
    """
    animal_position = SESSION_COLUMNS.index("animal")
    sessions = [
        (tuple(keys), keys[animal_position], group)
        for keys, group in long_df.groupby(list(SESSION_COLUMNS))
        if len(group) >= min_rows
    ]
    fits, skips = fit_sessions_dirichlet_multinomial(
        sessions,
        list(predictors),
        min_events=min_events_session_phi,
        n_jobs=n_jobs,
        response_col=response_col,
        event_col=event_col,
        concentration_cols=concentration_cols_for_cohort(long_df),
        n_categories=n_categories,
    )
    results = []
    for keys, result in fits:
        result["model"] = model_name
        result["response_col"] = response_col
        for col, val in zip(SESSION_COLUMNS, keys, strict=True):
            result[col] = val
        results.append(result)
    for keys, reason in skips:
        logging.warning(
            "%s: no Dirichlet-multinomial fit for session %s (%s)",
            model_name,
            "/".join(map(str, keys)),
            reason,
        )
    frame = pd.DataFrame(results)
    frame.attrs["estimator_skips"] = [
        {"session": "/".join(map(str, keys)), "reason": reason}
        for keys, reason in skips
    ]
    return frame


def fit_model_per_session(
    long_df: pd.DataFrame,
    model_name: str,
    predictors: list[str],
    run_cv: bool = False,
    response_col: str = "arm_count_hard",
    event_col: str = "event_id",
    n_categories: int = 9,
    grain: str = "timebin",
    estimator: str = "conditional_poisson",
    n_jobs: int = 1,
    min_events_session_phi: int = DEFAULT_MIN_EVENTS_SESSION_PHI,
) -> pd.DataFrame:
    """Fit one content rung across all sessions.

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
        Column name for the response: the per-event allocation fraction
        (``arm_fraction`` / ``patch_fraction``) or the timebin counts
        (``arm_count_hard`` / ``patch_count_hard``).
    event_col : str
        Column name for the event grouping key.
    estimator : str
        ``"conditional_poisson"`` (the shipped softmax fit, whose grain is set
        by ``grain``) or ``"dirichlet_multinomial"``, which adds an explicit
        within-event concentration and is fit on the counts only.
    n_jobs : int
        Worker processes for the Dirichlet-multinomial per-session fits.
    min_events_session_phi : int
        Sessions with fewer modelled events than this have their concentration
        intercept pinned to the animal-pooled value.

    Returns
    -------
    pd.DataFrame
        One row per session with coefficients and fit statistics, plus the
        ``estimator`` the fitter used and the ``response_col`` it was fit on --
        the two facts that say at which grain the row's ``llf`` / ``aic`` /
        ``bse_`` are stated, so a fraction and a count fit cannot be pooled by
        a later reader that sees only coefficients.
    """
    results = []

    # Minimum 2 events per session, scaled by candidate-set size:
    # n_categories=9 (arm) → 18 rows, n_categories=3 (patch) → 6 rows.
    # The previous hardcoded 18 silently skipped valid 2–5 event patch
    # sessions before fit_conditional_poisson could apply its own
    # n_events < 2 * len(predictors) gate.
    min_rows = n_categories * 2

    if estimator == "dirichlet_multinomial":
        return _fit_dirichlet_multinomial_per_session(
            long_df,
            model_name,
            predictors,
            response_col=response_col,
            event_col=event_col,
            n_categories=n_categories,
            min_rows=min_rows,
            n_jobs=n_jobs,
            min_events_session_phi=min_events_session_phi,
        )

    for keys, group in long_df.groupby(list(SESSION_COLUMNS)):
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
            # A failed CV must not cost the session its primary coefficient. CV
            # is a diagnostic on the fit, not a precondition for it, and letting
            # it gate the cohort means the reported effects depend on whether
            # cross-validation was requested -- measured at 213 sessions without
            # --cv against 208 with it. The CV columns are left NaN instead, so
            # held-out comparisons run on the sessions that have scores (and can
            # report that count) while the coefficients keep the full cohort.
            if cv is not None:
                r.update(cv)
            else:
                # Keep the CV schema stable whether or not a session scored, so
                # downstream checks can distinguish "this session has no score"
                # from "CV never ran at all" instead of tripping over an absent
                # column.
                r.update(dict.fromkeys(CV_RESULT_COLUMNS, np.nan))

        r["model"] = model_name
        r["response_col"] = response_col
        for col, val in zip(SESSION_COLUMNS, keys, strict=True):
            r[col] = val

        results.append(r)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


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
    fit_inputs: Mapping[str, object],
    estimator: str,
    cli: Sequence[str],
    response_renormalized: bool = False,
) -> None:
    """Stage every artifact, then replace the output directory as one bundle.

    Every pickle gets a provenance sidecar naming the table the fit read, its
    SHA-256 as of the read, the estimator that produced it, and whether the
    response was rescaled to the fitted candidate set, so a reader can refuse a
    fit whose input has moved since and can tell a leaf-only fraction from the
    table's own nine-arm one. The sidecars are written into the staging
    directory, so they are swapped in atomically with the pickles they
    describe -- a bundle can never be published without them.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    cli = list(cli)

    def _write(frame: pd.DataFrame, path: Path) -> None:
        frame.to_pickle(path)
        write_fit_provenance(
            path,
            dict(fit_inputs),
            estimator,
            cli,
            extra={"response_renormalized": bool(response_renormalized)},
        )

    try:
        for model_name, result in model_results.items():
            _write(result, staged_dir / f"{model_name}.pkl")
        _write(combined, staged_dir / "all_models.pkl")
        write_cohort_audit(cohort_audits, staged_dir / "all_models_cohort_audit.csv")
        if not comparison.empty:
            _write(comparison, staged_dir / "model_comparison.pkl")
        if logit_results:
            logit_dir = staged_dir / "logit"
            logit_dir.mkdir()
            for model_name, result in logit_results.items():
                _write(result, logit_dir / f"{model_name}_logit.pkl")
        _publish_staged_directory(staged_dir, output_dir)
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)


# How a fit's likelihood-based numbers should be read, by estimator. A count
# fit spends one observation per decoded timebin, of which an event supplies a
# median of ~32 that are near-copies of each other, so its llf/AIC/BIC are on a
# scale the data do not support. A fraction or dominant-choice fit spends one
# observation per event; its likelihood is a quasi-likelihood (the response is
# not a count), so AIC and LRT differences on it are event-scale quasi
# quantities rather than exact information criteria.
_SCALE_LABELS = {
    "conditional_poisson_fraction": "event-scale (quasi)",
    "conditional_logit_dominant": "event-scale (quasi)",
    "conditional_poisson_count": "timebin-scale",
    # The DM likelihood is a proper probability of the event's whole count
    # vector -- multinomial coefficient included -- so its AIC and LRT are exact
    # information criteria at the event grain rather than quasi ones. That same
    # constant is why they must not be differenced against a conditional fit's.
    "dirichlet_multinomial": "event-scale (full likelihood)",
}
_DEFAULT_SCALE_LABEL = "unlabelled scale"


def scale_label(estimator: object) -> str:
    """Return how likelihood-based numbers from ``estimator`` must be read.

    Parameters
    ----------
    estimator : object
        An estimator name from a fit result, or anything else (an absent or
        mixed column) for which no scale can be claimed.

    Returns
    -------
    str
        ``"event-scale (quasi)"``, ``"timebin-scale"``, or
        ``"unlabelled scale"`` when the estimator is unknown.
    """
    return _SCALE_LABELS.get(estimator, _DEFAULT_SCALE_LABEL)


def _frame_estimator(frame: pd.DataFrame) -> object:
    """The single estimator a frame's rows were fit with, else ``None``."""
    if "estimator" not in frame.columns:
        return None
    values = set(frame["estimator"].dropna().unique())
    return values.pop() if len(values) == 1 else None


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

    estimator = _frame_estimator(mdf)
    print(f"\nFit quality (n={len(mdf)} sessions; {scale_label(estimator)}):")
    for metric in ["aic", "bic", "llf"]:
        if metric in mdf.columns:
            vals = mdf[metric].dropna()
            if len(vals) > 0:
                print(f"  {metric}: median={vals.median():.2f}")
    if "pseudo_r2" in mdf.columns:
        vals = mdf["pseudo_r2"].dropna()
        if len(vals) > 0:
            # McFadden's ratio is not comparable across estimators: the DM's
            # llf and llf_null both carry the multinomial constant that a
            # conditional likelihood omits, and its null refits the
            # concentration, so the ratio sits on a different scale.
            note = (
                f" ({scale_label(estimator)}; llf and llf_null both include the "
                "multinomial constant, so NOT on a conditional fit's scale)"
                if estimator == "dirichlet_multinomial"
                else ""
            )
            print(f"  pseudo_r2: median={vals.median():.3f}{note}")

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
    # ``unident``: sessions that cannot inform the predictor at all (NaN
    # coefficient, named in ``unidentified_terms``); they are in no median here.
    header = (
        f"  {'Predictor':<25} {'AnimalMed':>10} {'agree':>7} {'SessMed':>9} "
        f"{'SE':>8} {'sess p':>10} {'unident':>7}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for pred in predictors:
        if pred not in coef_info:
            continue
        vals = coef_info[pred]
        se_col = f"bse_{pred}"
        # Over the sessions with a coefficient: a session that cannot inform the
        # term reports it with a NaN coefficient and an infinite SE.
        se = (
            mdf.loc[vals.index, se_col].median()
            if se_col in mdf.columns
            else float("nan")
        )
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
        n_unidentified = int(unidentified_for_term(mdf, pred).sum())
        print(
            f"  {pred:<25} {animal_med:10.3f} {agree:>7} {vals.median():9.3f} "
            f"{se:8.3f} {p_str:>10} {n_unidentified:>7d}"
        )


def format_comparison_table(comparison: pd.DataFrame) -> list[str]:
    """Render the nested-comparison headline block, one string per line.

    What the headline may show depends on the scale the rungs were fit at. On a
    count response every event contributes ~32 autocorrelated timebins as if
    they were independent draws, so the per-session counts of AIC wins and of
    significant likelihood-ratio tests are anticonservative by roughly that
    factor: they are kept in ``model_comparison.pkl`` for auditing but are not
    printed as a headline result. On the fraction (and on the dominant-choice
    logit) an event is one observation, so the same counts are printed, marked
    as the event-scale quasi-likelihood quantities they are.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of ``run_full_hierarchy_comparison``; one row per step.

    Returns
    -------
    list of str
        Lines to print. Empty when there is nothing to compare.
    """
    if comparison.empty:
        return []
    estimator = _frame_estimator(comparison)
    label = scale_label(estimator)
    event_scale = label.startswith("event-scale")

    lines = [
        "=" * 60,
        f"MODEL COMPARISON (positive ΔAIC = child better; {label})",
        "=" * 60,
        "",
    ]
    header = f"  {'Step':<12} {'ΔAIC animal':>12} {'animals':>8} {'ΔAIC sess':>11}"
    if event_scale:
        header += f" {'Child wins':>12} {'LRT sig':>10}"
    header += f" {'sess p':>11}"
    lines.append(header)
    lines.append(f"  {'-' * (len(header) - 2)}")

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
        line = f"  {step:<12} {am_str:>12} {agree:>8} {row['delta_aic_median']:>11.1f}"
        if event_scale:
            line += (
                f" {row['child_better_aic']:>8}/{row['n_sessions']:<3}"
                f" {row['lrt_significant']:>6}/{row['n_sessions']:<3}"
            )
        line += f" {wp:>11}"
        lines.append(line)

    if event_scale:
        # Read off the label rather than hardcoded: the conditional fits' counts
        # are quasi-likelihood ones, the Dirichlet-multinomial's are not.
        lines.append(f"  Child wins / LRT sig are {label} counts.")
    else:
        lines.append(
            "  Per-session child_better_aic / lrt_significant are timebin-scale "
            "(events' bins counted as independent draws); they stay in "
            "model_comparison.pkl and are not reported here."
        )
    return lines


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated model names (default: the arm ladder "
            "M0,M0a,M1-M6 or the patch ladder P0-P3; the value/uncertainty "
            "rungs Mv,Mu,Mvu,Pv are opt-in and must be named here)"
        ),
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
        choices=list(GRAIN_CHOICES),
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
        choices=list(EVENT_SOURCE_CHOICES),
        help="Event source: non_local (model-detected), SWR, or MUA (default: non_local)",
    )
    parser.add_argument(
        "--hierarchy",
        type=str,
        default="arm",
        choices=list(HIERARCHY_CHOICES),
        help=(
            "Resolution: 'arm' (9 candidate arms, M0,M0a,M1-M6 plus opt-in "
            "Mv/Mu/Mvu) or 'patch' (3 patches, P0-P3 plus opt-in Pv). Patch "
            "fits use event_patch_table.parquet and patch_count_hard."
        ),
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="conditional_poisson",
        choices=list(ESTIMATOR_CHOICES),
        help=(
            "Per-session estimator. 'conditional_poisson' (default, shipped) "
            "fits the softmax allocation at the grain --response selects. "
            "'dirichlet_multinomial' fits the same softmax mean structure with "
            "an explicit within-event concentration, so an event's "
            "autocorrelated timebins are not counted as independent draws and "
            "coef_log_phi_is_swr measures directly whether SWR events spread "
            "their mass more widely. It is defined on the integer counts, so it "
            "forces --response hard_count, and it publishes to a '_dm' "
            "directory: its likelihood carries the multinomial coefficient the "
            "conditional one drops, so its AIC is not on the conditional "
            "bundle's scale."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Worker processes for the Dirichlet-multinomial per-session fits "
            f"(default: {DEFAULT_N_JOBS}). Ignored by the conditional Poisson."
        ),
    )
    parser.add_argument(
        "--min-events-session-phi",
        type=int,
        default=DEFAULT_MIN_EVENTS_SESSION_PHI,
        help=(
            "Modelled events a session needs to estimate its own "
            "Dirichlet-multinomial concentration intercept (default: "
            f"{DEFAULT_MIN_EVENTS_SESSION_PHI}). Shorter sessions are pinned to "
            "the median intercept of that animal's longer sessions, and record "
            "which pooled value they used in phi_source. This changes the "
            "fitted COHORT, not just a nuisance parameter: a session just under "
            "the threshold is refit with its intercept pinned, and if that "
            "refit fails it is dropped (reason 'pinned_fit_none') even though "
            "its free fit had converged."
        ),
    )
    parser.add_argument(
        "--response",
        type=str,
        default=None,
        choices=list(RESPONSE_CHOICES),
        help=(
            "Per-event response. Defaults to the estimator's own: 'fraction' "
            "for the conditional Poisson, 'hard_count' for the "
            "Dirichlet-multinomial (which has no fraction form, so asking for "
            "one is an error). 'fraction' (the shipped response) fits "
            "arm_fraction / patch_fraction, so every event contributes one "
            "observation of unit mass and the likelihood, AIC and SEs are at "
            "the event grain. 'hard_count' fits arm_count_hard / "
            "patch_count_hard, which weights an event by its number of decoded "
            "timebins and treats those autocorrelated bins as independent "
            "draws; it is kept for the retired-count comparison and writes to "
            "a '_hard_count' output directory so the two never overwrite each "
            "other."
        ),
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help=(
            "Condition on the six leaf arms instead of all nine candidates. "
            "The Beta-belief covariates (Mv/Mu/Mvu) are structural zeros on the "
            "three trunks, which inflates their within-event collinearity when "
            "the trunks are carried; fitting them leaf-only requires their "
            "parent rungs to be leaf-only too, so request the whole chain in "
            "one run (--leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu). The "
            "arm dummies for arms 5-7 are dropped (arm 5 is the reference) and "
            "the null baseline becomes log(6). Arm hierarchy only; publishes to "
            "a '_leaf' output directory."
        ),
    )
    parser.add_argument(
        "--swr-subset",
        type=str,
        default="all",
        choices=list(SWR_SUBSET_CHOICES),
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
        choices=list(BRAIN_AREA_CHOICES),
        help=(
            "Brain area suffix used when resolving ContFrag prep "
            "outputs (event_arm_table_{source}_{brain_area}.pkl). "
            "Ignored for the non_local event source, which is built "
            "by ``prepare_replay_content_data.py`` and is not "
            "partitioned per area."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Explicit candidate-table root. Defaults to the canonical root for "
            "--event-source. Use this for staging validation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact model-output directory. Defaults to the source/hierarchy "
            "subdirectory under --input-dir. Existing outputs are replaced only "
            "after the complete requested bundle validates."
        ),
    )
    return parser


class _RunConfig(NamedTuple):
    """Resolved paths, model hierarchy, and requested rung names for one run."""

    hierarchy: dict[str, ModelSpec]
    input_dir: Path
    output_dir: Path
    table_name: str
    model_names: Sequence[str]
    # ``--response`` after the estimator default is applied, so no downstream
    # step has to re-derive it (or forget to).
    response: str = "fraction"


class _ContentCohort(NamedTuple):
    """The loaded, subset, and complete-case-frozen fitting cohort."""

    long_df: pd.DataFrame
    long_df_prefilter: pd.DataFrame
    valid_model_names: Sequence[str]
    union_predictors: Sequence[str]
    response_col: str
    event_col: str
    n_events_before_complete_case: int
    fit_inputs: dict[str, object]
    response_renormalized: bool


def _resolve_run_config(args: argparse.Namespace) -> _RunConfig:
    """Resolve the hierarchy, input/output paths, and requested model names."""
    # Select hierarchy and input paths based on event source. The
    # --hierarchy flag overrides the model dict and table name when
    # 'patch' is selected; the input directory follows the event source.
    source_config = _EVENT_SOURCE_CONFIG[args.event_source]
    estimator = args.estimator
    # One resolution point for the response: ``--response`` defaults to the
    # estimator's own, and a combination the estimator cannot fit is refused
    # here rather than silently overridden.
    response = resolve_response_choice(estimator, args.response)
    if estimator == "dirichlet_multinomial":
        if args.grain != "timebin":
            raise ValueError(
                "--grain event collapses each event to the arm that won, which "
                "discards the within-event distribution the "
                "Dirichlet-multinomial concentration is estimated from. Drop "
                "--grain event or use --estimator conditional_poisson."
            )
        if args.cv:
            raise ValueError(
                "--cv runs the conditional-Poisson cross-validation, whose "
                "held-out score is not on the Dirichlet-multinomial likelihood "
                "scale. Drop --cv or use --estimator conditional_poisson."
            )
    if args.leaf_only and args.hierarchy == "patch":
        raise ValueError(
            "--leaf-only restricts the nine candidate arms to the six leaves; "
            "the patch hierarchy's three candidates are patches, not arms, and "
            "have no leaf subset. Drop --leaf-only or use --hierarchy arm."
        )
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
    table_name, _output_subdir = resolve_contfrag_paths(
        args.event_source, args.brain_area, args.hierarchy
    )

    input_dir = (
        args.input_dir.resolve()
        if args.input_dir is not None
        else PROCESSED_DATA_DIR / source_config["input_dir_name"]
    )
    # SWR subset: restrict to is_swr=True or =False events. Within a
    # subset is_swr is constant, so the swr_x_* interactions (M6/M7,
    # P4) collapse onto their main effects and are unidentifiable —
    # use the *_SWR variants (M0-M5 / P0-P3) which already drop them.
    # The subset and the event grain each get their own bundle
    # directory (see ``canonical_output_subdir``) so neither publishes
    # over the unrestricted timebin fits.
    if args.swr_subset != "all":
        hierarchy = (
            MODEL_PATCH_SWR if args.hierarchy == "patch" else MODEL_HIERARCHY_SWR
        )
        logging.info(
            "SWR subset = %s: hierarchy narrowed to %s",
            args.swr_subset,
            list(hierarchy.keys()),
        )
    output_dir = input_dir / canonical_output_subdir(
        args.event_source,
        args.brain_area,
        args.hierarchy,
        args.swr_subset,
        args.grain,
        response,
        args.leaf_only,
        estimator,
    )
    if args.leaf_only:
        logging.info(
            "candidate set = %d leaf arms -> %s", LEAF_ONLY_N_CATEGORIES, output_dir
        )
    if response != "fraction":
        logging.info("response = %s -> %s", response, output_dir)
    if estimator != "conditional_poisson":
        logging.info("estimator = %s -> %s", estimator, output_dir)
    if args.grain == "event":
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

    # A run that fits anything other than the full default ladder is fit on its
    # own complete-case cohort, so its rungs are not comparable with the
    # canonical bundle's and must not be written next to them (or over them).
    # It gets its own directory, keyed by the rungs it fits.
    canonical_dir = output_dir
    is_partial_ladder = set(model_names) != set(default_content_model_names(hierarchy))
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        # Any canonical bundle, not just this invocation's: an --output-dir can
        # name the bundle of another hierarchy, SWR subset, grain, event source
        # or brain area just as easily as its own.
        if is_partial_ladder and output_dir.name in canonical_output_subdirs():
            raise ValueError(
                f"--models {','.join(model_names)} is not the full "
                f"{canonical_dir.name}/ ladder "
                f"({','.join(default_content_model_names(hierarchy))}); writing it "
                f"to {output_dir}, the canonical {output_dir.name}/ bundle, would "
                "replace part of a cohort-matched bundle with rungs fit on a "
                "different event set. Drop --output-dir to get a directory keyed "
                "by the requested rungs, or name a directory of your own."
            )
        logging.info("explicit model output -> %s", output_dir)
    elif is_partial_ladder:
        output_dir = canonical_dir.with_name(
            partial_ladder_subdir(canonical_dir.name, model_names)
        )
        logging.info("partial ladder (%s) -> %s", ", ".join(model_names), output_dir)

    return _RunConfig(
        hierarchy=hierarchy,
        input_dir=input_dir,
        output_dir=output_dir,
        table_name=table_name,
        model_names=model_names,
        response=response,
    )


def _load_and_filter_cohort(
    args: argparse.Namespace, config: _RunConfig
) -> _ContentCohort:
    """Load the canonical table, apply the SWR subset and at-well cohort, and
    freeze the common complete-case event set."""
    hierarchy = config.hierarchy
    input_dir = config.input_dir
    table_name = config.table_name
    model_names = config.model_names

    # Canonical inputs are mandatory. The former implicit ``long_count.pkl``
    # fallback could fit M0-M3 from a stale schema and write those partial
    # results into the canonical ``models/`` directory.
    fit_input_paths = content_fit_inputs(input_dir, table_name)
    canonical_path = fit_input_paths[table_name]
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
    # Hash the table now, while the fit is reading it: the ladder runs long
    # enough that the table could be regenerated before the bundle is written,
    # and a sidecar hashed at write time would certify bytes the fit never saw.
    fit_inputs = record_inputs(fit_input_paths)
    require_covariate_schema(
        long_df, table="event_arm" if args.hierarchy == "arm" else "event_patch"
    )
    # Its events and is_swr must come from the configured SWR source, artifact
    # run included.
    require_swr_source(long_df, what=str(canonical_path))
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
        response=config.response,
    )
    logging.info("response = %s (%s)", response_col, config.response)
    validate_requested_standardization(
        long_df,
        union_predictors=union_predictors,
        input_dir=input_dir,
        table_label=canonical_path,
    )
    logging.info(
        "Loaded %d rows, %d sessions",
        len(long_df),
        long_df.groupby(list(SESSION_COLUMNS)).ngroups,
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
    logging.info(
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
    if args.leaf_only:
        n_before_rows = len(long_df)
        long_df, response_renormalized = leaf_only_cohort(
            long_df,
            response_col=response_col,
            response=config.response,
            event_col=event_col,
        )
        logging.info(
            "candidate set = %d leaf arms: %d -> %d rows (%s rescaled to unit "
            "mass per event: %s)",
            LEAF_ONLY_N_CATEGORIES,
            n_before_rows,
            len(long_df),
            response_col,
            response_renormalized,
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
        logging.info(
            "Filtered to common complete-case events: %d → %d events "
            "(%d missing, %d nonfinite in %s)",
            n_before,
            n_complete_case,
            cohort_selection["n_missing_units"],
            cohort_selection["n_nonfinite_units"],
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
            f"predictor in {list(union_predictors)} is missing or nonfinite "
            "on some candidate "
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
        long_df.groupby(list(SESSION_COLUMNS)).ngroups,
        long_df[event_col].nunique(),
    )

    return _ContentCohort(
        long_df=long_df,
        long_df_prefilter=long_df_prefilter,
        valid_model_names=valid_model_names,
        union_predictors=union_predictors,
        response_col=response_col,
        event_col=event_col,
        n_events_before_complete_case=n_before,
        fit_inputs=fit_inputs,
        response_renormalized=response_renormalized,
    )


def _fit_content_ladder(
    args: argparse.Namespace, config: _RunConfig, cohort: _ContentCohort
) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]]]:
    """Fit every requested rung per session and validate the matched ladder.

    Returns the matched per-rung results and, per rung, the sessions the
    estimator produced no fit for, each with its reason -- an auditable skip
    rather than a crash of the whole ladder.
    """
    hierarchy = config.hierarchy
    long_df = cohort.long_df
    valid_model_names = cohort.valid_model_names
    response_col = cohort.response_col
    event_col = cohort.event_col

    # Fit every requested rung into memory. Nothing under ``output_dir`` is
    # created or modified until the whole ladder and its shared cohort pass
    # integrity checks, so a late empty/failed rung cannot leave mixed results.
    fit_results: dict[str, pd.DataFrame] = {}
    estimator_skips: dict[str, list[dict]] = {}
    for model_name in valid_model_names:
        spec = hierarchy[model_name]
        predictors = rung_predictors(spec, args.leaf_only)
        cv_label = " + CV" if args.cv else ""
        logging.info(
            "Fitting %s (%d predictors%s)...", model_name, len(predictors), cv_label
        )
        t0 = time.time()

        # Patch hierarchy collapses 9 arms to 3 patches and --leaf-only drops
        # the 9 arms to 6 leaves; the null / pseudo-R² / CV baselines must use
        # log(3) or log(6) instead of log(9) for those fits to be on the right
        # chance footing.
        n_categories = fit_n_categories(args.hierarchy, args.leaf_only)

        mdf = fit_model_per_session(
            long_df,
            model_name,
            predictors,
            run_cv=args.cv,
            response_col=response_col,
            event_col=event_col,
            n_categories=n_categories,
            grain=args.grain,
            estimator=args.estimator,
            n_jobs=args.n_jobs,
            min_events_session_phi=args.min_events_session_phi,
        )
        # Read off the frame the fitter returned, before matching reindexes it:
        # attrs are not a contract pandas carries through every operation.
        estimator_skips[model_name] = list(mdf.attrs.get("estimator_skips", []))

        elapsed = time.time() - t0
        logging.info("  %s: %d session fits in %.1fs", model_name, len(mdf), elapsed)
        # Whether the response these numbers were fit on was rescaled to the
        # fitted candidate set travels with the numbers: a leaf-only fraction
        # fit is not on the same response as the nine-arm bundle's.
        mdf["response_renormalized"] = cohort.response_renormalized
        fit_results[model_name] = mdf

    model_results = validate_and_match_fit_results(
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

    return model_results, estimator_skips


def _fit_conditional_logit_robustness(
    args: argparse.Namespace,
    config: _RunConfig,
    cohort: _ContentCohort,
    model_results: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Optional conditional-logit robustness fits on the dominant arm.

    Returns an empty mapping unless ``--logit`` is set; otherwise fits each
    rung per session on the primary ladder's shared-session cohort and
    validates that the logit sessions match it exactly.
    """
    hierarchy = config.hierarchy
    long_df = cohort.long_df
    valid_model_names = cohort.valid_model_names
    response_col = cohort.response_col
    event_col = cohort.event_col

    # Conditional-logit robustness results are also collected before the output
    # bundle is staged. Restrict them to the primary ladder's shared sessions.
    shared_sessions = set(
        next(iter(model_results.values()))[list(SESSION_COLUMNS)].itertuples(
            index=False, name=None
        )
    )
    long_session_index = pd.MultiIndex.from_frame(long_df[list(SESSION_COLUMNS)])
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
            predictors = rung_predictors(spec, args.leaf_only)

            model_logit_results = []
            for keys, group in matched_long_df.groupby(list(SESSION_COLUMNS)):
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
                r["response_col"] = response_col
                r["response_renormalized"] = cohort.response_renormalized
                for col, val in zip(SESSION_COLUMNS, keys, strict=True):
                    r[col] = val
                model_logit_results.append(r)

            logit_outputs[model_name] = pd.DataFrame(model_logit_results)

        if expected_logit_models:
            logit_outputs = validate_and_match_fit_results(
                logit_outputs,
                requested_model_names=expected_logit_models,
                cohort=matched_long_df,
                event_col=event_col,
            )
            for model_name, logit_df in logit_outputs.items():
                logit_sessions = set(
                    logit_df[list(SESSION_COLUMNS)].itertuples(index=False, name=None)
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

    return logit_outputs


def _zero_response_event_ids(
    long_df: pd.DataFrame, *, event_col: str, response_col: str
) -> set:
    """Events whose response is undefined on every row, or sums to nothing.

    ``min_count=1`` keeps an all-missing event's total missing rather than
    letting pandas report it as 0.0, so "the response could not be computed for
    this event" and "the response is zero everywhere" both land here and neither
    is mistaken for a fitted event.
    """
    totals = long_df.groupby(event_col, observed=True, sort=False)[response_col].sum(
        min_count=1
    )
    return set(totals.index[~(totals > 0)])


def _audit_compare_and_publish(
    args: argparse.Namespace,
    config: _RunConfig,
    cohort: _ContentCohort,
    model_results: dict[str, pd.DataFrame],
    estimator_skips: dict[str, list[dict]] | None = None,
) -> None:
    """Build the cohort audit, run nested comparisons and optional conditional-
    logit robustness fits, then stage and publish the complete output bundle."""
    hierarchy = config.hierarchy
    output_dir = config.output_dir
    long_df_prefilter = cohort.long_df_prefilter
    valid_model_names = cohort.valid_model_names
    union_predictors = cohort.union_predictors
    response_col = cohort.response_col
    event_col = cohort.event_col
    n_before = cohort.n_events_before_complete_case
    estimator_skips = estimator_skips or {}

    # Build every audit record only after the model frames have been restricted
    # to the exact same successfully fitted session cohort.
    cohort_audits: list[dict] = []
    population = getattr(args, "swr_subset", None) or "all"
    sessions_in = long_df_prefilter.groupby(list(SESSION_COLUMNS)).ngroups
    # Events whose response is undefined or sums to nothing: on the fraction
    # response these are the all-junction events, which carry NaN on every
    # candidate. The cohort freeze removes them; the audit must also SAY so,
    # under n_drop_zero_response, or a reader cannot tell an event the response
    # could not describe from one no model wanted.
    zero_response_events = _zero_response_event_ids(
        long_df_prefilter, event_col=event_col, response_col=response_col
    )
    for model_name, mdf in model_results.items():
        predictors = rung_predictors(hierarchy[model_name], args.leaf_only)

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
        skipped = estimator_skips.get(model_name, [])
        notes = drop_profile(dropped)
        # The routine classes are summarized; only an estimator REFUSAL names
        # the session it happened to. Same helper (and so the same policy) as
        # the is_swr-contrast runner's cohort audit.
        skip_note = estimator_skip_note(
            [(skip["session"], skip["reason"]) for skip in skipped],
            ESTIMATOR_SKIP_CLASSES,
        )
        if skip_note:
            notes = "; ".join(filter(None, [notes, skip_note]))
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
                drop_notes=notes,
                n_drop_qvals_gap=buckets["n_drop_qvals_gap"],
                n_drop_own_predictor_missing=buckets["n_drop_own_predictor_missing"],
                n_drop_nonfinite=buckets["n_drop_nonfinite"],
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
        headline = format_comparison_table(comparison)
        if headline:
            print("\n" + "\n".join(headline))

    logit_outputs = _fit_conditional_logit_robustness(
        args, config, cohort, model_results
    )

    estimator = _frame_estimator(combined)
    if estimator is None:
        raise RuntimeError(
            "Combined fit does not carry a single estimator label; the bundle's "
            "provenance could not say how its numbers were produced, so no "
            "output will be published"
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
        fit_inputs=cohort.fit_inputs,
        estimator=str(estimator),
        cli=sys.argv,
        response_renormalized=cohort.response_renormalized,
    )
    logging.info(
        "Published %d model files and complete audit bundle to %s",
        len(model_results),
        output_dir,
    )

    logging.info("Done!")


def run_content_fit(args: argparse.Namespace) -> None:
    """Load the content tables, fit the ladder, compare, and publish the bundle."""
    config = _resolve_run_config(args)
    cohort = _load_and_filter_cohort(args, config)
    model_results, estimator_skips = _fit_content_ladder(args, config, cohort)
    _audit_compare_and_publish(args, config, cohort, model_results, estimator_skips)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_content_fit(args)


if __name__ == "__main__":
    main()
