"""Poisson rate model for replay event occurrence.

Tests whether the NUMBER of replay events per stop depends on
behavioral context, independently of replay CONTENT. The response is
``n_selected_events`` (count of the table's selected event source per stop); the offset is
``log(immobility_duration_s)`` so that coefficients are interpretable as
multiplicative changes in event rate per second of immobility.

Model hierarchy (``MODEL_PER_STOP_RATE`` in ``model_specs.py``):

- **R0**: intercept only — baseline event rate.
- **R1**: + the reward-history raised-cosine basis
  (``reward_history_rcos_0..2``) and its structural-missingness indicator
  ``reward_history_incomplete`` — does the event rate track recent reward
  history? (The immediate-switch flag was removed: it is often constant within
  a session's complete-case cohort, making the design rank-deficient.) The
  features are weighted fractions of the outcomes already known at bout start;
  each known outcome is closed at the **outcome** clock: pump onset when
  rewarded, otherwise arrival plus ``OMISSION_KNOWN_LATENCY_SEC`` capped at the
  collapsed task visit's final departure. Before an arrived trip reaches that
  boundary, the newest known outcome remains the preceding one; afterward it
  names the arrived trip. The basis is therefore past-only but not a fixed
  current- or previous-trial index. ``reward_history_incomplete`` is a nuisance
  main effect (fit, never reported) for stops with fewer than two known
  outcomes, whose features are zero-filled.

A future R2 (+ ``trials_since_block_change``, ``block_number``) is
deferred until the contingency-block loader is wired. Fitting R0-R1
separately on SWR-only / MUA-only / neither stops is an analysis
variant, not a separate model.

``exp(γ)`` = multiplicative change in events per second of
immobility. ``exp(γ) = 1.5`` means 50% more events per second when
the predictor is active.

:func:`fit_per_stop_rate_hierarchy` fits the ladder over sessions, each on one
cohort frozen over every rung's predictors, and returns its cohort audit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from continuum_swr_replay.analysis.events.cohort_selection import (
    freeze_common_cohort,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from .._fit_diagnostics import (
    EXPECTED_FIT_ERRORS,
    NUISANCE_TERMS,
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_flag_counts,
    fit_status,
)
from ..cohort_audit import cohort_record
from ..model_specs import MODEL_PER_STOP_RATE

logger = logging.getLogger(__name__)

# Canonical per-stop count outcomes used by the production hierarchy runner.
# Keeping this beside the fitter lets notebooks and scripts share one mapping
# instead of independently spelling the stop-table column names.
PER_STOP_RATE_OUTCOMES: dict[str, str] = {
    "swr": "n_swr_events",
    "non_local": "n_selected_events",
    "non_local_swr": "n_nonlocal_swr_events",
    "non_local_no_swr": "n_nonlocal_non_swr_events",
}


def fit_per_stop_rate_model(
    stop_table: pd.DataFrame,
    predictors: list[str],
    *,
    response_col: str = "n_selected_events",
    offset_col: str = "immobility_duration_s",
    min_stops: int = 20,
) -> dict | None:
    """Fit a Poisson GLM on event counts per stop.

    Parameters
    ----------
    stop_table : pd.DataFrame
        One session's ``stop_table`` (includes zero-event stops).
    predictors : list[str]
        Columns to use as fixed effects. Empty list fits the
        intercept-only null (R0).
    response_col : str
        Count column. Default ``n_selected_events``.
    offset_col : str
        Exposure column — ``log(offset_col)`` enters as an offset. Every stop
        must carry ``offset_col > 0``; zero-duration bouts belong to the
        cohort freeze, not to this fitter.
    min_stops : int
        Minimum number of valid stops to attempt fitting.

    Returns
    -------
    dict | None
        Coefficients (``coef_*``), standard errors (``bse_*``),
        p-values (``pval_*``), rate ratios (``rr_*`` = exp(coef)),
        plus fit statistics. None if fitting fails.

    Raises
    ------
    ValueError
        If any required column carries a missing value, or any stop has
        ``offset_col <= 0``. The fitter never subsets: rung-specific dropping
        is what splits a ladder into several cohorts, so the caller freezes one
        cohort per session with
        :func:`cohort_selection.freeze_common_cohort` before fitting.
    """
    required = [response_col, offset_col] + predictors
    missing = [c for c in required if c not in stop_table.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        return None

    df = stop_table[required].copy()
    nan_cols = {c: int(df[c].isna().sum()) for c in required if df[c].isna().any()}
    if nan_cols:
        raise ValueError(
            f"per-stop rate design has missing values in {nan_cols}; freeze the "
            "ladder's common cohort with freeze_common_cohort() before fitting "
            "(the fitter must not subset, or the rungs stop sharing a cohort)."
        )
    n_bad_offset = int((df[offset_col] <= 0).sum())
    if n_bad_offset:
        raise ValueError(
            f"per-stop rate design has {n_bad_offset} stop(s) with "
            f"{offset_col} <= 0 (log offset undefined); freeze the ladder's "
            "common cohort with freeze_common_cohort(offset_col=...) before "
            "fitting."
        )
    df = df.reset_index(drop=True)

    if len(df) < min_stops:
        return None

    y = df[response_col].values.astype(float)
    offset = np.log(df[offset_col].values.astype(float))

    if predictors:
        # Force ``add_constant`` to always prepend an intercept column,
        # even if one of the predictors is itself constant within
        # ``df``. With the default ``has_constant="skip"`` the call is
        # a no-op for a constant predictor, the returned ``X`` has
        # ``len(predictors)`` columns instead of
        # ``len(predictors) + 1``, ``col_names`` claims an extra
        # ``"Intercept"`` slot, and the post-fit
        # ``zip(col_names, result.params, strict=True)`` raises
        # ValueError after the GLM has already run.
        raw_X = df[predictors].to_numpy(dtype=float)
        X = sm.add_constant(raw_X, has_constant="add")
        col_names = ["Intercept", *predictors]

        # Full-rank check rejects constant predictors, perfect
        # collinearity, duplicated columns, and any other zero
        # linear combination of design columns — statsmodels would
        # otherwise pseudo-invert and emit finite coefficients /
        # p-values for an unidentified model.
        rank = int(np.linalg.matrix_rank(X))
        if rank < X.shape[1]:
            logger.warning(
                "Poisson GLM not fit: design matrix [Intercept, %s] is "
                "rank-deficient (rank %d < %d) within the %d-stop subset.",
                ", ".join(predictors),
                rank,
                X.shape[1],
                len(df),
            )
            return None
    else:
        X = np.ones((len(df), 1))
        col_names = ["Intercept"]

    try:
        with capture_fit_warnings() as caught:
            model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
            result = model.fit(disp=0)
    except EXPECTED_FIT_ERRORS:
        logger.warning(
            "Poisson GLM failed for predictors %s", predictors, exc_info=True
        )
        return None
    predictor_sd = np.asarray(X, dtype=float).std(axis=0)
    # The constant column carries the formula-style name "Intercept", so the
    # shared nuisance set and every saved column use one spelling in both
    # families.
    status = fit_status(
        result,
        "glm",
        predictor_sd=predictor_sd,
        term_names=col_names,
        nuisance_terms=NUISANCE_TERMS,
    )

    coefs = dict(zip(col_names, result.params, strict=True))
    bse = dict(zip(col_names, result.bse, strict=True))
    pvals = dict(zip(col_names, result.pvalues, strict=True))
    n_params = len(col_names)

    output = {
        **{f"coef_{k}": float(v) for k, v in coefs.items()},
        **{f"bse_{k}": float(v) for k, v in bse.items()},
        **{f"pval_{k}": float(v) for k, v in pvals.items()},
        **{f"rr_{k}": float(np.exp(v)) for k, v in coefs.items()},
        "n_stops": len(df),
        "n_events_total": int(y.sum()),
        "n_params": n_params,
        "llf": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic_llf),
        "deviance": float(result.deviance),
        "pearson_chi2": float(result.pearson_chi2),
        "mean_rate": float(y.sum() / np.exp(offset).sum()),
        **{
            f"sd_{name}": float(s)
            for name, s in zip(col_names, predictor_sd, strict=True)
        },
        "converged": status["converged"],
        "separated": status["separated"],
        # "|"-joined so the tuple survives a round trip through pickle and CSV.
        "separated_terms": "|".join(status["separated_terms"]),
        "n_convergence_warnings": count_fit_quality_warnings(caught),
    }

    return output


def cross_validate_per_stop_rate_model(
    stop_table: pd.DataFrame,
    predictors: list[str],
    response_col: str = "n_selected_events",
    offset_col: str = "immobility_duration_s",
    n_folds: int = 5,
    min_stops: int = 20,
    seed: int = 0,
) -> dict | None:
    """K-fold held-out log-likelihood for the per-stop rate Poisson.

    Stop-based K-fold CV mirroring :func:`fit_per_stop_rate_model`'s design (intercept +
    predictors, ``log(offset_col)`` offset). For each fold the model is fit on
    the training stops and the mean held-out per-stop Poisson log-likelihood is
    evaluated on the test stops; the returned ``cv_ll_per_stop_mean`` is the
    average across folds (higher = better predictive fit). This is the rate
    analogue of :func:`model_checking.cross_validate_conditional_poisson`, which
    only applies to the conditional (arm/patch) models.

    Returns None if columns are missing, too few stops, or any fold's training
    design is rank-deficient (matching ``fit_per_stop_rate_model``'s identifiability
    guard).
    """
    from scipy.stats import poisson
    from sklearn.model_selection import KFold

    required = [response_col, offset_col] + predictors
    if any(c not in stop_table.columns for c in required):
        return None
    df = stop_table[required].dropna()
    df = df[df[offset_col] > 0].reset_index(drop=True)
    if len(df) < max(min_stops, n_folds * 2):
        return None

    y = df[response_col].to_numpy(dtype=float)
    offset = np.log(df[offset_col].to_numpy(dtype=float))
    if predictors:
        X = sm.add_constant(df[predictors].to_numpy(dtype=float), has_constant="add")
    else:
        X = np.ones((len(df), 1))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ll = []
    n_nonconverged = 0
    n_separated = 0
    for train, test in kf.split(X):
        if np.linalg.matrix_rank(X[train]) < X.shape[1]:
            return None
        try:
            res = sm.GLM(
                y[train], X[train], family=sm.families.Poisson(), offset=offset[train]
            ).fit(disp=0)
        except EXPECTED_FIT_ERRORS:
            logger.debug(
                "CV fold fit failed for predictors %s", predictors, exc_info=True
            )
            return None
        status = fit_status(
            res,
            "glm",
            predictor_sd=np.asarray(res.model.exog, dtype=float).std(axis=0),
            term_names=["Intercept", *predictors],
            nuisance_terms=NUISANCE_TERMS,
        )
        n_nonconverged += not status["converged"]
        n_separated += status["separated"]
        mu = np.exp(np.clip(X[test] @ res.params + offset[test], -30, 30))
        ll = poisson.logpmf(y[test], mu)
        if not np.all(np.isfinite(ll)):
            return None
        fold_ll.append(float(np.mean(ll)))

    # Deterministic rule: any nonconverged/separated fold invalidates the whole CV
    # metric (nan) rather than dropping the fold -- dropping a fold changes the
    # held-out set and makes CV metrics non-comparable across rungs. The fold counts
    # are reported alongside so the invalidation is visible.
    invalid = (n_nonconverged > 0) or (n_separated > 0)
    return {
        "cv_ll_per_stop_mean": np.nan if invalid else float(np.mean(fold_ll)),
        "cv_ll_per_stop_std": np.nan if invalid else float(np.std(fold_ll)),
        "cv_n_folds": n_folds,
        "cv_n_nonconverged_folds": int(n_nonconverged),
        "cv_n_separated_folds": int(n_separated),
    }


OFFSET_COL = "immobility_duration_s"
# Union of every rung's predictors: the ladder is frozen to one cohort over
# these, so R0 and R1 are fit to the same stops and their AIC is comparable.
ALL_PREDICTORS = list(
    dict.fromkeys(c for spec in MODEL_PER_STOP_RATE.values() for c in spec.predictors)
)


def incomplete_stops(group: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Rows a complete-case freeze would remove for ``columns`` (+ the offset).

    Mirrors :func:`freeze_common_cohort`'s first three filters at row grain:
    missing value, non-finite numeric value, nonpositive offset. The stop table
    is one row per stop, so row grain and unit grain coincide.
    """
    present = [c for c in columns if c in group.columns]
    bad = (
        group[present].isna().any(axis=1)
        if present
        else pd.Series(False, index=group.index)
    )
    numeric = [c for c in present if pd.api.types.is_numeric_dtype(group[c])]
    if numeric:
        values = group[numeric].to_numpy(dtype=float)
        bad = bad | ~np.isfinite(values).all(axis=1)
    if OFFSET_COL in group.columns:
        bad = bad | (group[OFFSET_COL] <= 0)
    return bad


def fit_per_stop_rate_hierarchy(
    stop_table: pd.DataFrame, run_cv: bool = True
) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (outcome, model, session): R0-R1 fit stats, coefs, CV-LL.

    Every rung of a session is fit to ONE cohort, frozen once per session over
    the union of the ladder's predictors, the outcome columns present, and a
    positive ``immobility_duration_s`` — so R0 and R1 report the same
    ``n_stops`` and their AIC / BIC / LRT are comparable.

    Also returns one stop-grain cohort-audit record per (model, outcome): a
    stop missing THIS rung's own predictor (or with ``immobility<=0``) goes to
    ``n_drop_own_predictor_missing``; a stop the freeze removes only because
    ANOTHER rung's predictor is missing there goes to
    ``n_drop_common_cohort_other``; a skipped session's remaining stops
    (min_stops / rank gate) go to ``n_excluded_session``; the fitted
    ``n_stops`` sum is ``n_modeled``. Reconciles by construction.
    """
    missing_predictors = [c for c in ALL_PREDICTORS if c not in stop_table.columns]
    if missing_predictors:
        raise KeyError(
            "stop table is missing per-stop rate predictor(s) "
            f"{missing_predictors}; the ladder cannot be frozen to one cohort."
        )
    responses = [c for c in PER_STOP_RATE_OUTCOMES.values() if c in stop_table.columns]

    rows = []
    acc: dict = {}  # (model, outcome) -> counts
    for keys, group in stop_table.groupby(list(SESSION_COLUMNS), sort=False):
        # ONE cohort for the whole ladder and every outcome of this session.
        cohort, _selection = freeze_common_cohort(
            group,
            unit_col="stop_id",
            required_cols=[*ALL_PREDICTORS, *responses],
            offset_col=OFFSET_COL,
        )
        dropped = ~group["stop_id"].isin(cohort["stop_id"])
        for outcome, response_col in PER_STOP_RATE_OUTCOMES.items():
            if response_col not in group.columns:
                continue
            for name, spec in MODEL_PER_STOP_RATE.items():
                preds = list(spec.predictors)
                a = acc.setdefault(
                    (name, outcome),
                    {
                        "n_in": 0,
                        "n_own": 0,
                        "n_common": 0,
                        "n_modeled": 0,
                        "sessions_in": 0,
                        "sessions_modeled": 0,
                        "animals": set(),
                        "flags": [],
                    },
                )
                a["sessions_in"] += 1
                a["n_in"] += len(group)
                own_bad = incomplete_stops(group, [response_col, *preds])
                a["n_own"] += int(own_bad.sum())
                a["n_common"] += int((dropped & ~own_bad).sum())

                res = fit_per_stop_rate_model(cohort, preds, response_col=response_col)
                if res is None:
                    continue
                a["n_modeled"] += int(res["n_stops"])
                a["sessions_modeled"] += 1
                a["animals"].add(keys[0])
                a["flags"].append(
                    {
                        "converged": res.get("converged"),
                        "separated": res.get("separated"),
                    }
                )
                row = {
                    "outcome": outcome,
                    "model": name,
                    **dict(zip(SESSION_COLUMNS, keys, strict=True)),
                    **res,
                }
                if run_cv:
                    cv = cross_validate_per_stop_rate_model(
                        cohort, preds, response_col=response_col
                    )
                    if cv is not None:
                        row.update(cv)
                rows.append(row)

    records = [
        cohort_record(
            name,
            outcome,
            "stop",
            cohort_mode="common",
            n_in=a["n_in"],
            n_modeled=a["n_modeled"],
            sessions_in=a["sessions_in"],
            sessions_modeled=a["sessions_modeled"],
            animals_modeled=len(a["animals"]),
            drop_notes=(
                "own NaN predictor / immobility<=0 -> own; NaN only in another "
                "rung's predictor (frozen cohort) -> common; session gates -> "
                "excluded"
            ),
            n_drop_own_predictor_missing=a["n_own"],
            n_drop_common_cohort_other=a["n_common"],
            n_excluded_session=(
                a["n_in"] - a["n_own"] - a["n_common"] - a["n_modeled"]
            ),
            **fit_flag_counts(
                pd.DataFrame(a["flags"], columns=["converged", "separated"])
            ),
        )
        for (name, outcome), a in acc.items()
    ]
    return pd.DataFrame(rows), records
