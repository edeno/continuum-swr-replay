"""Fit a block of models per session on one frozen complete-case cohort.

A *block* is a set of models fit on the same table (the patch hierarchy, the
per-stop rate hierarchy, the choice models). Within each session every model of
the block is fit on one cohort, frozen once over the union of the block's
predictors by :func:`~continuum_swr_replay.analysis.event_definition.cohort_selection.freeze_common_cohort`,
so the models of a block are compared on the same units. A session whose cohort
is too small, or where any model fails, contributes no fit to any model of the
block, and its per-session provenance says why
(:func:`fit_per_session_common_cohort`). The provenance is the input of
:func:`~continuum_swr_replay.analysis.replay_models.cohort_audit.records_from_common_cohort_provenance`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping

import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    freeze_common_cohort,
    sorted_unit_ids,
)
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from .covariate_timing import assert_predictor_timing
from .model_specs import DESCRIPTIVE_CONTENT_RUNGS

SESS = list(SESSION_COLUMNS)


def predictor_union(models: Mapping) -> tuple[str, ...]:
    """Sorted union of predictors used by a model block."""
    return tuple(
        sorted({predictor for spec in models.values() for predictor in spec.predictors})
    )


def block_required_columns(
    models: Mapping, *, unit_col: str, required_base
) -> list[str]:
    """Columns a model block reads: its unit ID, base columns and predictors."""
    return list(dict.fromkeys([unit_col, *required_base, *predictor_union(models)]))


def block_table_name(analysis: str, event_source: str | None) -> str:
    """How a refusal names a block's table: the block, and its event source."""
    source = "" if event_source is None else f" of event source {event_source!r}"
    return f"the {analysis} table{source}"


def hash_unit_ids(ids: tuple) -> str:
    """Hash sorted included IDs so cohorts can be compared cheaply."""
    digest = hashlib.sha256()
    for unit_id in ids:
        digest.update(str(unit_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def freeze_session_cohort(
    group: pd.DataFrame,
    *,
    keys,
    analysis: str,
    models: Mapping,
    unit_col: str,
    required_base: list[str],
    min_units: int,
    response_col: str | None = None,
    drop_zero_response: bool = False,
    offset_col: str | None = None,
    choice_response_col: str | None = None,
    selection_filter: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return one complete-case cohort shared by every model in a block.

    Parameters
    ----------
    group : pd.DataFrame
        One session's rows.
    keys : tuple
        The session's values of the session key columns.
    analysis : str
        Block name, recorded in the provenance.
    models : mapping of str to ModelSpec
        The block's models; the cohort is complete over all their predictors.
    unit_col : str
        The unit a failed row removes whole (``event_id`` or ``stop_id``).
    required_base : list of str
        Columns every model reads besides its predictors.
    min_units : int
        Units the cohort needs; below it the status is ``too_few_units``.
    response_col, drop_zero_response, offset_col, choice_response_col
        Passed to
        :func:`~continuum_swr_replay.analysis.event_definition.cohort_selection.freeze_common_cohort`.
    selection_filter : str, optional
        The selection the table was cut by before the block, recorded as text.

    Returns
    -------
    (cohort, provenance)
        The cohort's rows, and one provenance record: the session keys, the
        block, the unit counts before and after and at each drop, the sorted
        included IDs and their hash, and ``status`` (``ok`` or
        ``too_few_units``).
    """
    session = dict(zip(SESS, keys, strict=True))
    predictors = predictor_union(models)
    required_cols = block_required_columns(
        models, unit_col=unit_col, required_base=required_base
    )
    ids_before = sorted_unit_ids(group[unit_col])

    provenance = {
        **session,
        "analysis": analysis,
        "models": tuple(models),
        "unit_col": unit_col,
        "predictors": predictors,
        "required_columns": tuple(required_cols),
        "selection_filter": selection_filter,
        "n_rows_before": len(group),
        "n_units_before": len(ids_before),
        "n_rows_after": 0,
        "n_units_after": 0,
        "n_missing_units": 0,
        "n_nonfinite_units": 0,
        "n_nonpositive_offset_units": 0,
        "n_zero_response_units": 0,
        "n_invalid_choice_units": 0,
        "included_ids": (),
        "included_ids_hash": hash_unit_ids(()),
        "id_hash_algorithm": "sha256_sorted_unit_ids_v1",
        "status": "ok",
        "failed_models": (),
    }

    cohort, selection = freeze_common_cohort(
        group,
        unit_col=unit_col,
        required_cols=required_cols,
        response_col=response_col,
        drop_zero_response=drop_zero_response,
        offset_col=offset_col,
        choice_response_col=choice_response_col,
    )
    for field in (
        "n_rows_after",
        "n_units_after",
        "n_missing_units",
        "n_nonfinite_units",
        "n_nonpositive_offset_units",
        "n_zero_response_units",
        "n_invalid_choice_units",
        "included_ids",
    ):
        provenance[field] = selection[field]
    provenance["included_ids_hash"] = hash_unit_ids(selection["included_ids"])
    if selection["n_units_after"] < min_units:
        provenance["status"] = "too_few_units"

    return cohort, provenance


def fit_per_session_common_cohort(
    table: pd.DataFrame,
    models: Mapping,
    fit_fn: Callable[[pd.DataFrame, object], dict | None],
    *,
    analysis: str,
    unit_col: str,
    required_base: list[str],
    min_units: int,
    response_col: str | None = None,
    drop_zero_response: bool = False,
    offset_col: str | None = None,
    choice_response_col: str | None = None,
    selection_filter: str | None = None,
    event_source: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fit all models on one frozen per-session cohort and record provenance.

    Each session's cohort is :func:`freeze_session_cohort`. ``fit_fn(cohort,
    spec)`` fits one model and returns its result dict, or ``None`` when it
    produced no fit. A session where any model returns ``None`` is dropped from
    every model of the block (status ``fit_failed``, naming the models); a fitted
    session (status ``fit``) carries ``cohort_n_units``, ``cohort_n_rows`` and
    ``cohort_included_ids_hash`` on each model's row.

    Returns
    -------
    (fits, provenance)
        One frame per model name, one row per fitted session; and one
        provenance row per session.

    Raises
    ------
    ValueError
        If a non-empty ``table`` lacks ``unit_col``, a ``required_base``
        column or a predictor of any model. The check is on the whole table,
        before any session is fit, so a broken table contract stops the run
        instead of yielding a block with no fits.
    """
    out = {name: [] for name in models}
    provenance_rows = []
    if len(table) == 0:
        return {name: pd.DataFrame() for name in models}, pd.DataFrame()
    require_columns(
        table,
        block_required_columns(models, unit_col=unit_col, required_base=required_base),
        block_table_name(analysis, event_source),
    )

    for keys, group in table.groupby(SESS):
        cohort, provenance = freeze_session_cohort(
            group,
            keys=keys,
            analysis=analysis,
            models=models,
            unit_col=unit_col,
            required_base=required_base,
            min_units=min_units,
            response_col=response_col,
            drop_zero_response=drop_zero_response,
            offset_col=offset_col,
            choice_response_col=choice_response_col,
            selection_filter=selection_filter,
        )
        if provenance["status"] != "ok":
            provenance_rows.append(provenance)
            continue

        session_results = {}
        failed_models = []
        for name, spec in models.items():
            fit = fit_fn(cohort, spec)
            if fit is None:
                failed_models.append(name)
            else:
                session_results[name] = fit

        if failed_models:
            provenance["status"] = "fit_failed"
            provenance["failed_models"] = tuple(failed_models)
            provenance_rows.append(provenance)
            continue

        provenance["status"] = "fit"
        provenance_rows.append(provenance)
        session = dict(zip(SESS, keys, strict=True))
        cohort_fields = {
            "cohort_n_units": provenance["n_units_after"],
            "cohort_n_rows": provenance["n_rows_after"],
            "cohort_included_ids_hash": provenance["included_ids_hash"],
        }
        for name, fit in session_results.items():
            out[name].append({**session, **cohort_fields, **fit})

    return (
        {name: pd.DataFrame(rows) for name, rows in out.items()},
        pd.DataFrame(provenance_rows),
    )


def assert_ladder_timing(models: Mapping, *, ladder: str) -> None:
    """Gate every model in a ladder through the causal-hygiene registry.

    A leaky covariate added to a ladder fails here unless its rung is declared
    descriptive by name (:data:`~continuum_swr_replay.analysis.replay_models.model_specs.DESCRIPTIVE_CONTENT_RUNGS`),
    exactly as the content and rate runners declare theirs.
    """
    for name, spec in models.items():
        assert_predictor_timing(
            spec.predictors,
            descriptive=name in DESCRIPTIVE_CONTENT_RUNGS,
        )
    logging.debug("timing gate passed for %d %s model(s)", len(models), ladder)
