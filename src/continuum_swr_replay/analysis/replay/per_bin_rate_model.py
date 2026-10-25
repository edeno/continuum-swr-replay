"""Per-bin conditional-intensity Poisson model for replay event onsets (Family B).

A point-process GLM conditional-intensity ladder (Truccolo-Eden-Brown 2005) on
the canonical (stop x bin) design from :func:`per_bin_design.aggregate_to_timebins`.
Each row is one ``BIN_S``-wide time bin within an immobility stop; the response
is the per-bin onset ``count`` and the offset is ``log(exposure)`` (seconds of
immobility in the bin), so ``exp(intercept)`` is the per-bin onset rate.

How this differs from ``per_stop_rate_model.py`` (Family A): that model fits one
row per stop with a ``log(immobility_duration_s)`` dwell offset, averaging the
whole stop into a single rate; here the within-stop time course is modelled
explicitly by a ``bs(tc)`` spline and the equal-exposure offset, and the same-
event self-history (refractoriness / bursting) is carried as autoregressive
``onset_lag{j}`` terms. The two families answer complementary questions: Family
A is the simple, literature-comparable dwell-offset rate; Family B is the
confound-free conditional intensity that can be checked as a point process
(time-rescaling KS, rescaled-ISI independence) via :func:`per_bin_rate_gof`.

The ladder (``MODEL_PER_BIN_RATE`` in ``model_specs.py``) toggles three blocks
(time spline, self-history, reward-history raised-cosine basis) plus the
reward x time interaction, all at one bin width (``BIN_S``) so the models are
nested and AIC / CV / deviance / GOF are directly comparable.

Nested comparability also requires one cohort: every rung of a ladder must be
fit to the SAME (stop, bin) rows, because AIC / deviance / held-out likelihood
all scale with the number of rows. :func:`freeze_per_bin_cohort` selects that
cohort once, over the union of every rung's design columns, before any rung is
fit; :func:`fit_per_bin_rate_model` and
:func:`cross_validate_per_bin_rate_model` then *raise* on an incomplete design
instead of quietly dropping rows per rung (which is what split the ladder into
several cohorts).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import PatsyError
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from ._fit_diagnostics import (
    NUISANCE_TERMS,
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_status,
)
from .model_specs import (
    ARRIVAL_CLOCK_COLUMN,
    REWARD_HISTORY_COLUMNS,
    TMAX_ARRIVAL,
    PerBinRateModelSpec,
)

logger = logging.getLogger(__name__)

# Numerical failures we expect from a per-session GLM fit/predict and treat as a
# skipped cell (-> None): rank-deficient/ill-conditioned designs, spline knots
# the held-out fold extrapolates past (patsy), perfect separation, optimizer
# failures. Anything else (e.g. a NameError in a spec formula, an offset shape
# mismatch) is a coding bug and is left to propagate rather than be swallowed as
# "fit failed" for every session.
_EXPECTED_FIT_ERRORS = (
    np.linalg.LinAlgError,
    ValueError,
    RuntimeError,
    PerfectSeparationError,
    PatsyError,
)


def _design_columns(spec: PerBinRateModelSpec) -> list[str]:
    """Columns the spec's formula references (besides the offset's ``exposure``).

    :meth:`PerBinRateModelSpec.design_columns`, read from the same term list
    as the spec's formula.
    """
    return spec.design_columns()


def freeze_per_bin_cohort(
    design: pd.DataFrame,
    specs: Iterable[PerBinRateModelSpec],
    *,
    extra_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict]:
    """Freeze ONE (stop x bin) cohort for a whole per-bin ladder.

    Keeps the bins that are complete for the **union** of every spec's design
    columns (plus ``count``, ``exposure`` and any ``extra_columns``) and that
    carry a positive exposure. Because every rung is then fit to exactly these
    rows, ``n_bins`` / ``n_stops`` are equal across the ladder and the nested
    AIC / CV / deviance comparison is valid. Callers must freeze BEFORE the
    first fit: :func:`fit_per_bin_rate_model` and
    :func:`cross_validate_per_bin_rate_model` raise rather than subset.

    Parameters
    ----------
    design : pd.DataFrame
        One (session, population) :func:`per_bin_design.aggregate_to_timebins` output,
        shape ``(n_bins, n_columns)``. Must carry ``stop_id``, ``count``,
        ``exposure`` and every required design column.
    specs : iterable of PerBinRateModelSpec
        The ladder whose rungs share the cohort. May be empty when the required
        columns are given directly through ``extra_columns`` (the contrast
        ladder names its columns per rung rather than through a spec object).
    extra_columns : sequence of str
        Additional required columns, e.g. the ``is_swr`` tag of the stacked
        contrast design or a rung's covariate list.

    Returns
    -------
    frozen : pd.DataFrame
        A copy of the retained rows with a fresh integer index and
        ``attrs["per_bin_cohort_frozen"] = True``. ``design`` is never mutated.
    report : dict
        ``n_bins_before`` / ``n_bins_after``, ``n_stops_before`` /
        ``n_stops_after``, ``n_event_bins_before`` / ``n_event_bins_after``
        (bins with ``count > 0``), the ``required_columns`` tuple the selection
        used, and — when an arrival-clock rung put ``ARRIVAL_CLOCK_COLUMN`` in
        that union — ``n_bins_dropped_arrival_clock`` /
        ``n_stops_dropped_arrival_clock``: the bins and stops that every other
        required column would have kept and that only the arrival clock removed
        (a stop that began before the session's first well arrival has no
        arrival clock). Because the freeze is common to the ladder, those bins
        leave EVERY rung, including the stop-clock ones, so the cost of adding
        the arrival rungs is reported rather than absorbed.

    Raises
    ------
    KeyError
        If any required column is absent from ``design`` — a design that cannot
        carry the ladder is a configuration error, not a data drop.
    """
    required = ["count", "exposure"]
    for spec in specs:
        required += _design_columns(spec)
    required += list(extra_columns)
    required = list(dict.fromkeys(required))

    missing = [c for c in (*required, "stop_id") if c not in design.columns]
    if missing:
        raise KeyError(f"freeze_per_bin_cohort missing required column(s): {missing}")

    keep = design[required].notna().all(axis=1) & (design["exposure"] > 0)
    frozen = design.loc[keep].reset_index(drop=True)
    frozen.attrs["per_bin_cohort_frozen"] = True
    report = {
        "n_bins_before": int(len(design)),
        "n_bins_after": int(len(frozen)),
        "n_stops_before": int(design["stop_id"].nunique()),
        "n_stops_after": int(frozen["stop_id"].nunique()),
        "n_event_bins_before": int((design["count"] > 0).sum()),
        "n_event_bins_after": int((frozen["count"] > 0).sum()),
        "required_columns": tuple(required),
    }
    if ARRIVAL_CLOCK_COLUMN in required:
        others = [c for c in required if c != ARRIVAL_CLOCK_COLUMN]
        keep_others = design[others].notna().all(axis=1) & (design["exposure"] > 0)
        report["n_bins_dropped_arrival_clock"] = int((keep_others & ~keep).sum())
        report["n_stops_dropped_arrival_clock"] = int(
            design.loc[keep_others, "stop_id"].nunique() - report["n_stops_after"]
        )
    return frozen, report


def apply_arrival_window(
    design: pd.DataFrame, *, tmax_arrival: float = TMAX_ARRIVAL
) -> tuple[pd.DataFrame, dict]:
    """Restrict a frozen cohort to ``0 <= arrival clock <= tmax_arrival``.

    Applied AFTER :func:`freeze_per_bin_cohort` and only to the rungs whose
    ``time_clock`` is ``"arrival"``. The order matters both ways: freezing
    first keeps the ladder on one cohort (the arrival column's own missingness
    removes bins from every rung, and is reported by the freeze), while
    windowing second keeps the arrival rungs inside the range their spline's
    boundary knots span, where the fitted curve is interpolation rather than
    extrapolation.

    The windowed rows are a strict subset of the frozen ones, so an arrival
    rung's information criteria are comparable with another arrival rung's and
    NOT with a stop-clock rung's — which is why the arrival sub-ladder carries
    its own constant-rate null.

    Parameters
    ----------
    design : pd.DataFrame
        A frozen per-bin cohort, shape ``(n_bins, n_columns)``, carrying
        ``stop_id``, ``count`` and ``ARRIVAL_CLOCK_COLUMN``.
    tmax_arrival : float
        Right edge of the window in seconds (the spline's upper boundary knot).

    Returns
    -------
    windowed : pd.DataFrame
        The retained rows with a fresh integer index. ``design`` is never
        mutated.
    report : dict
        ``n_bins_arrival_window`` / ``n_stops_arrival_window`` /
        ``n_event_bins_arrival_window`` and the shares of the frozen cohort
        retained: ``retained_share_arrival_bins`` and
        ``retained_share_arrival_events`` (event bins, i.e. ``count > 0`` — the
        population-specific share, since the population enters through
        ``count``).

    Raises
    ------
    KeyError
        If the arrival clock column is absent.
    ValueError
        If it carries a missing value — the cohort was not frozen over a union
        that included it, so the arrival rungs would not share the ladder's
        cohort.
    """
    if ARRIVAL_CLOCK_COLUMN not in design.columns:
        raise KeyError(
            f"apply_arrival_window needs {ARRIVAL_CLOCK_COLUMN}; the per-bin "
            "design was built without the stop's arrival clock."
        )
    clock = design[ARRIVAL_CLOCK_COLUMN].to_numpy(dtype=float)
    if np.isnan(clock).any():
        raise ValueError(
            f"apply_arrival_window: {ARRIVAL_CLOCK_COLUMN} has "
            f"{int(np.isnan(clock).sum())} missing value(s); freeze the "
            "ladder's common cohort over a union that includes it before "
            "windowing (the arrival rungs must share the ladder's cohort)."
        )
    keep = (clock >= 0.0) & (clock <= tmax_arrival)
    windowed = design.loc[keep].reset_index(drop=True)
    windowed.attrs["per_bin_cohort_frozen"] = True
    n_bins = int(len(design))
    n_event_bins = int((design["count"] > 0).sum())
    n_event_bins_kept = int((windowed["count"] > 0).sum())
    report = {
        "n_bins_arrival_window": int(len(windowed)),
        "n_stops_arrival_window": int(windowed["stop_id"].nunique()),
        "n_event_bins_arrival_window": n_event_bins_kept,
        "retained_share_arrival_bins": (
            float(len(windowed)) / n_bins if n_bins else np.nan
        ),
        "retained_share_arrival_events": (
            n_event_bins_kept / n_event_bins if n_event_bins else np.nan
        ),
    }
    return windowed, report


def require_frozen_per_bin_design(
    design: pd.DataFrame, needed: Sequence[str], *, name: str
) -> None:
    """Raise unless every ``needed`` column is complete and ``exposure > 0``.

    The per-bin fitters never subset their input: silently dropping rows per
    rung is what splits a ladder into several cohorts. Callers freeze the
    cohort once with :func:`freeze_per_bin_cohort` and pass the frozen design
    to every rung.

    Raises
    ------
    ValueError
        If any ``needed`` column has a missing value, or any row has
        ``exposure <= 0``.
    """
    nan_cols = [c for c in needed if design[c].isna().any()]
    if nan_cols:
        counts = {c: int(design[c].isna().sum()) for c in nan_cols}
        raise ValueError(
            f"{name}: per-bin design has missing values in {counts}; freeze the "
            "ladder's common cohort with freeze_per_bin_cohort() before fitting "
            "(the fitters must not subset, or the rungs stop sharing a cohort)."
        )
    n_bad = int((design["exposure"] <= 0).sum())
    if n_bad:
        raise ValueError(
            f"{name}: per-bin design has {n_bad} row(s) with exposure <= 0; "
            "freeze the ladder's common cohort with freeze_per_bin_cohort() "
            "before fitting."
        )


def fit_per_bin_rate_model(
    design: pd.DataFrame,
    spec: PerBinRateModelSpec,
    *,
    min_bins: int = 200,
    min_events: int = 20,
) -> dict | None:
    """Fit one session's per-bin Poisson GLM for ``spec``.

    Parameters
    ----------
    design : pd.DataFrame
        One session's :func:`per_bin_design.aggregate_to_timebins` output, already
        frozen by :func:`freeze_per_bin_cohort`, shape ``(n_bins, n_columns)``.
        Must carry ``count``, ``exposure``, ``stop_id``, ``binj`` plus the
        columns the spec's formula references, all complete.
    spec : PerBinRateModelSpec
        The model to fit. ``spec.formula()`` is fit with a ``log(exposure)``
        offset.
    min_bins : int
        Minimum number of bins required to attempt a fit.
    min_events : int
        Minimum number of event bins (``count > 0``) required.

    Returns
    -------
    dict | None
        ``coef_{term}`` / ``bse_{term}`` / ``pval_{term}`` / ``rr_{term}``
        (``rr = exp(coef)``) for every fitted term, plus ``n_bins``,
        ``n_events``, ``n_params``, ``llf``, ``aic``, ``deviance``,
        ``dispersion`` (= ``deviance / df_resid``), and ``predicted`` (the
        per-bin fitted mean count as a ``pd.Series`` indexed by the design
        rows, for :func:`per_bin_rate_gof`). ``None`` if columns are
        missing, too few bins/events, fewer than two distinct
        ``reward_history_rcos_0`` values when reward terms are present, or
        statsmodels raises a genuine numerical failure
        (rank deficiency / convergence error -> caught -> ``None``, never a
        crash).

    Raises
    ------
    ValueError
        If the design is not frozen: any needed column carries a missing value,
        or any row has ``exposure <= 0``. The fitter never subsets, so a rung
        cannot silently end up on a smaller cohort than its ladder.

    Notes
    -----
    Poisson perfect separation (e.g. a short self-history lag that perfectly
    predicts ``count == 0``) only emits a ``PerfectSeparationWarning`` rather
    than raising, so such a model returns a valid dict with an extreme,
    large-magnitude coefficient -- not ``None``. That is expected at the short
    self-history lags where the model is poorly identified.
    """
    needed = [*_design_columns(spec), "exposure"]
    missing = [c for c in needed if c not in design.columns]
    if missing:
        logger.warning("Missing columns for %s: %s", spec.name, missing)
        return None

    require_frozen_per_bin_design(design, needed, name=spec.name)
    df = design[[*needed, "stop_id", "binj"]]
    n_events = int((df["count"] > 0).sum())
    if len(df) < min_bins or n_events < min_events:
        logger.debug(
            "%s: too few data (%d bins, %d events; need %d/%d)",
            spec.name,
            len(df),
            n_events,
            min_bins,
            min_events,
        )
        return None
    # Reward coefficients are unidentified without contrast in the first
    # reward-history basis component (the recent-reward fraction).
    immediate = REWARD_HISTORY_COLUMNS[0]
    if (spec.reward_basis or spec.reward_time_interaction) and (
        df[immediate].nunique() < 2
    ):
        logger.debug(
            "%s: <2 %s levels, reward terms unidentified", spec.name, immediate
        )
        return None

    offset = np.log(df["exposure"].to_numpy(dtype=float))
    try:
        with capture_fit_warnings() as caught:
            result = smf.glm(
                spec.formula(),
                data=df,
                family=sm.families.Poisson(),
                offset=offset,
            ).fit()
    except _EXPECTED_FIT_ERRORS as exc:
        logger.warning("Per-bin GLM failed for %s: %r", spec.name, exc)
        return None
    # Effect-scale separation: the per-bin design carries spline bases whose columns
    # are far from O(1), so the legacy |coef| rule mis-flags them.
    predictor_sd = np.asarray(result.model.exog, dtype=float).std(axis=0)
    term_names = [str(t) for t in result.params.index]
    status = fit_status(
        result,
        "glm",
        predictor_sd=predictor_sd,
        term_names=term_names,
        nuisance_terms=NUISANCE_TERMS,
    )

    params = result.params
    bse = result.bse
    pvals = result.pvalues
    df_resid = float(result.df_resid)

    output: dict = {
        **{f"coef_{k}": float(v) for k, v in params.items()},
        **{f"bse_{k}": float(v) for k, v in bse.items()},
        **{f"pval_{k}": float(v) for k, v in pvals.items()},
        **{f"rr_{k}": float(np.exp(v)) for k, v in params.items()},
        "n_bins": int(len(df)),
        "n_events": n_events,
        "n_params": int(len(params)),
        "llf": float(result.llf),
        "aic": float(result.aic),
        "deviance": float(result.deviance),
        "dispersion": float(result.deviance / df_resid) if df_resid > 0 else np.nan,
        # distinct stops that contributed >=1 bin to the fit — the modeled-cohort
        # count for the stop-grain audit. The fitter drops no rows: a design with
        # a missing predictor or exposure<=0 raised above
        # (require_frozen_per_bin_design), and immobility<=0 stops never entered
        # the design.
        "n_stops": int(df["stop_id"].nunique()),
        **{
            f"sd_{name}": float(s)
            for name, s in zip(term_names, predictor_sd, strict=True)
        },
        "converged": status["converged"],
        "separated": status["separated"],
        # "|"-joined so the tuple survives a round trip through pickle and CSV.
        "separated_terms": "|".join(status["separated_terms"]),
        "n_convergence_warnings": count_fit_quality_warnings(caught),
        # fitted mean count per bin, indexed by retained rows (GOF input).
        "predicted": result.fittedvalues,
    }
    return output


def cross_validate_per_bin_rate_model(
    design: pd.DataFrame,
    spec: PerBinRateModelSpec,
    *,
    n_folds: int = 10,
    min_bins: int = 200,
    seed: int = 0,
) -> dict | None:
    """10-fold-over-stops held-out predictive score for ``spec``.

    Folds split on unique ``stop_id`` (mirroring
    :func:`per_stop_rate_model.cross_validate_per_stop_rate_model`) so a stop's
    bins never straddle train and test. Each fold scores the held-out per-bin
    Poisson log-likelihood, and ``M0_const`` (intercept + offset) is refit on
    the same train fold as the bits/event reference.

    Returns
    -------
    dict | None
        ``cv_ll_per_bin_mean`` / ``cv_ll_per_bin_std`` (per-fold mean held-out
        per-bin LL, averaged across folds), ``cv_bits_per_event`` (= summed
        held-out ``ll_model - ll_M0_const`` over all held-out bins, divided by
        ``n_events * ln 2``), and ``cv_n_folds``. ``None`` if too few
        bins/stops or any fold fails.

    Raises
    ------
    ValueError
        If the design is not frozen (see :func:`fit_per_bin_rate_model`); the
        held-out score of one rung is only comparable to another's when both
        score the same rows.
    """
    from scipy.stats import poisson
    from sklearn.model_selection import KFold

    needed = [*_design_columns(spec), "exposure", "stop_id", "binj"]
    missing = [c for c in needed if c not in design.columns]
    if missing:
        logger.debug("CV %s: missing columns %s", spec.name, missing)
        return None
    # binj is the integer bin index (never NaN from aggregate_to_timebins); it
    # is kept only to preserve temporal order, so it is excluded from the NaN
    # check while every modelled column (including stop_id) must be present.
    require_frozen_per_bin_design(
        design, [c for c in needed if c != "binj"], name=f"CV {spec.name}"
    )
    df = design[needed].reset_index(drop=True)
    if len(df) < min_bins:
        logger.debug("CV %s: too few bins (%d < %d)", spec.name, len(df), min_bins)
        return None

    stops = df["stop_id"].unique()
    if len(stops) < n_folds * 2:
        logger.debug(
            "CV %s: too few stops (%d < %d)", spec.name, len(stops), n_folds * 2
        )
        return None

    formula = spec.formula()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_means: list[float] = []
    sum_diff = 0.0
    n_events_total = 0
    n_nonconverged = 0
    n_separated = 0
    for train_idx, _test_idx in kf.split(stops):
        train_stops = set(stops[train_idx])
        is_train = df["stop_id"].isin(train_stops).to_numpy()
        tr, te = df[is_train], df[~is_train]
        if len(te) == 0 or int((tr["count"] > 0).sum()) < 1:
            logger.debug(
                "CV %s: degenerate fold (empty test or zero train events)", spec.name
            )
            return None
        off_tr = np.log(tr["exposure"].to_numpy(dtype=float))
        off_te = np.log(te["exposure"].to_numpy(dtype=float))
        # predict is inside the try: a spline model whose held-out fold has tc
        # outside the train fold's knots raises a patsy error -> CV None, not a
        # crash (mirrors the fit guard).
        try:
            res = smf.glm(
                formula, data=tr, family=sm.families.Poisson(), offset=off_tr
            ).fit()
            res0 = smf.glm(
                "count ~ 1", data=tr, family=sm.families.Poisson(), offset=off_tr
            ).fit()
            mu = np.asarray(res.predict(te, offset=off_te), dtype=float)
            mu0 = np.asarray(res0.predict(te, offset=off_te), dtype=float)
        except _EXPECTED_FIT_ERRORS as exc:
            logger.debug("CV %s: fold fit/predict failed: %r", spec.name, exc)
            return None
        # Fold counts as nonconverged/separated if EITHER the model or its baseline
        # fails the check.
        # Effect-scale separation, matching the main fit path. Without the
        # predictor scales this falls back to |coef| >= 10, which every
        # self-history rung trips on its raised-cosine basis (small-SD columns,
        # ordinary effects) -- that flagged all 10 folds separated and voided the
        # CV metric for M3/M4/M6/M7, leaving 2 of 215 sessions scored.
        sd_res = np.asarray(res.model.exog, dtype=float).std(axis=0)
        sd_res0 = np.asarray(res0.model.exog, dtype=float).std(axis=0)
        s_res = fit_status(
            res,
            "glm",
            predictor_sd=sd_res,
            term_names=[str(t) for t in res.params.index],
            nuisance_terms=NUISANCE_TERMS,
        )
        s_res0 = fit_status(
            res0,
            "glm",
            predictor_sd=sd_res0,
            term_names=[str(t) for t in res0.params.index],
            nuisance_terms=NUISANCE_TERMS,
        )
        n_nonconverged += (not s_res["converged"]) or (not s_res0["converged"])
        n_separated += s_res["separated"] or s_res0["separated"]
        y = te["count"].to_numpy(dtype=float)
        ll = poisson.logpmf(y, mu)
        ll0 = poisson.logpmf(y, mu0)
        if not (np.all(np.isfinite(ll)) and np.all(np.isfinite(ll0))):
            logger.debug("CV %s: non-finite held-out log-likelihood", spec.name)
            return None
        fold_means.append(float(np.mean(ll)))
        sum_diff += float(np.sum(ll - ll0))
        n_events_total += int((y > 0).sum())

    # Any nonconverged/separated fold invalidates the whole CV metric (nan) rather
    # than dropping the fold -- dropping a fold changes the held-out set and makes CV
    # scores non-comparable across rungs. Counts are reported alongside.
    invalid = (n_nonconverged > 0) or (n_separated > 0)
    bits = sum_diff / (n_events_total * np.log(2)) if n_events_total else np.nan
    return {
        "cv_ll_per_bin_mean": np.nan if invalid else float(np.mean(fold_means)),
        "cv_ll_per_bin_std": np.nan if invalid else float(np.std(fold_means)),
        "cv_bits_per_event": np.nan if invalid else float(bits),
        "cv_n_folds": n_folds,
        "cv_n_nonconverged_folds": int(n_nonconverged),
        "cv_n_separated_folds": int(n_separated),
    }


def per_bin_rate_gof(
    design: pd.DataFrame,
    fitted: dict,
    *,
    adjust_for_short_trials: bool = True,
) -> dict:
    """Point-process goodness-of-fit via the time-rescaling theorem.

    Treats the event train as a spike train (``is_spike = count > 0`` per bin,
    ``trial_id = stop_id``) and rescales the fitted conditional intensity with
    the time-rescaling theorem (Brown et al. 2002). Reuses
    ``non_local_detector.model_checking.sorted_spikes`` -- the time-rescaling
    math is not re-implemented here.

    The integrand passed to ``TimeRescaling`` is ``fitted['predicted']``, the
    per-bin **expected count** ``mu = exp(X@beta) * exposure`` (statsmodels folds
    the ``log(exposure)`` offset into ``fittedvalues``) -- *not* the rate
    ``exp(X@beta)``. This is deliberate and required: ``TimeRescaling`` integrates
    with ``cumulative_trapezoid(ci, dx=1)`` (unit spacing), so the integrand of
    ``Lambda(t) = integral lambda du`` over a bin of width ``Delta_i`` must be
    ``lambda_i * Delta_i`` -- the expected count -- which also correctly
    down-weights each stop's short final bin (``exposure < bin_s``). Dividing by
    ``exposure`` to "convert to a rate" would over-weight those partial bins and
    break the KS test (empirically KS ~0.10 with the expected count vs ~0.99 with
    the rate on a known-true homogeneous Poisson process).

    Parameters
    ----------
    design : pd.DataFrame
        The same design passed to :func:`fit_per_bin_rate_model` (rows are
        aligned to ``fitted['predicted']`` by index, then ordered by
        ``(stop_id, binj)`` so the rescaling integral runs in within-stop
        temporal order).
    fitted : dict
        The :func:`fit_per_bin_rate_model` return dict.
    adjust_for_short_trials : bool
        Wiener (2003) censoring correction for short / variable-length stops.

    Returns
    -------
    dict
        ``ks_statistic``, ``ks_within_band`` (``ks < 1.36 / sqrt(n_events)``),
        ``max_isi_autocorr``, ``dispersion``, ``n_events``, and ``mean_predicted``
        / ``max_predicted`` (the per-bin mean-count discretization check).
    """
    from non_local_detector.model_checking.sorted_spikes import TimeRescaling

    predicted = fitted["predicted"]
    sub = design.loc[predicted.index].sort_values(["stop_id", "binj"])
    # per-bin expected count (lambda * exposure) -- the correct integrand for the
    # unit-spacing integration in TimeRescaling; see the docstring. Do NOT divide
    # by exposure here.
    ci = predicted.reindex(sub.index).to_numpy(dtype=float)
    is_spike = sub["count"].to_numpy() > 0
    trial_id = sub["stop_id"].to_numpy()
    n_events = int(is_spike.sum())

    out: dict = {
        "n_events": n_events,
        "dispersion": float(fitted.get("dispersion", np.nan)),
        "mean_predicted": float(ci.mean()) if ci.size else np.nan,
        "max_predicted": float(ci.max()) if ci.size else np.nan,
        "ks_statistic": np.nan,
        "ks_within_band": False,
        "max_isi_autocorr": np.nan,
    }
    if n_events < 2:
        return out

    tr = TimeRescaling(
        conditional_intensity=ci,
        is_spike=is_spike,
        trial_id=trial_id,
        adjust_for_short_trials=adjust_for_short_trials,
    )
    ks = float(tr.ks_statistic())
    out["ks_statistic"] = ks
    out["ks_within_band"] = bool(ks < 1.36 / np.sqrt(n_events))

    isi_ac = tr.rescaled_ISI_autocorrelation()
    if isi_ac.size > 1:
        center = isi_ac.size // 2  # zero-lag (== 1 by construction)
        off_center = np.delete(np.abs(isi_ac), center)
        out["max_isi_autocorr"] = float(off_center.max())
    return out
