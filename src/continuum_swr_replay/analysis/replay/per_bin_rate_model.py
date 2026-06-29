"""Per-bin conditional-intensity Poisson model for replay event onsets (Family B).

A point-process GLM conditional-intensity ladder (Truccolo-Eden-Brown 2005) on
the canonical (stop x bin) design from :func:`_helpers.aggregate_to_timebins`.
Each row is one ``BIN_S``-wide time bin within an immobility stop; the response
is the per-bin onset ``count`` and the offset is ``log(exposure)`` (seconds of
immobility in the bin), so ``exp(intercept)`` is the per-bin onset rate.

How this differs from ``per_stop_rate_model.py`` (Family A): that model fits one
row per stop with a ``log(immobility_time)`` dwell offset, averaging the whole
stop into a single rate; here the within-stop time course is modelled
explicitly by a ``bs(tc)`` spline and the equal-exposure offset, and the same-
event self-history (refractoriness / bursting) is carried as autoregressive
``onset_lag{j}`` terms. The two families answer complementary questions: Family
A is the simple, literature-comparable dwell-offset rate; Family B is the
confound-free conditional intensity that can be checked as a point process
(time-rescaling KS, rescaled-ISI independence) via :func:`per_bin_rate_gof`.

The ladder (``MODEL_PER_BIN_RATE`` in ``model_specs.py``) toggles four blocks
(time spline, self-history, reward [lag-1 / kernel / continuous rate], switch)
plus the reward x time interaction, all at one bin width (``BIN_S``) so the
models are nested and AIC / CV / deviance / GOF are directly comparable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import PatsyError
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .model_specs import PerBinRateModelSpec

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
    """Columns the spec's formula references (besides the offset's ``exposure``)."""
    cols = ["count"]
    if spec.time_spline or spec.reward_time_interaction:
        cols.append("tc")
    cols += [f"reward_lag{j}" for j in range(1, spec.reward_lags + 1)]
    if spec.reward_time_interaction and spec.reward_lags == 0:
        cols.append("reward_lag1")
    if spec.reward_rate:
        cols.append("reward_rate_trial_k10")
    if spec.self_history_basis:
        cols += [f"hist_rcos_{k}" for k in range(spec.self_history_basis)]
    else:
        cols += [f"onset_lag{j}" for j in range(1, spec.self_history_lags + 1)]
    if spec.cross_history_basis:
        cols += [f"cross_hist_rcos_{k}" for k in range(spec.cross_history_basis)]
    else:
        cols += [f"cross_lag{j}" for j in range(1, spec.cross_history_lags + 1)]
    if spec.switch:
        cols.append("is_after_switch")
    cols += list(spec.extra_covariates)
    return list(dict.fromkeys(cols))  # de-duplicate, preserve order


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
        One session's :func:`_helpers.aggregate_to_timebins` output. Must carry
        ``count``, ``exposure``, ``stop_id``, ``binj`` plus the columns the
        spec's formula references.
    spec : PerBinRateModelSpec
        The model to fit. ``spec.formula()`` is fit with a ``log(exposure)``
        offset.
    min_bins : int
        Minimum number of (NaN-dropped) bins required to attempt a fit.
    min_events : int
        Minimum number of event bins (``count > 0``) required.

    Returns
    -------
    dict | None
        ``coef_{term}`` / ``bse_{term}`` / ``pval_{term}`` / ``rr_{term}``
        (``rr = exp(coef)``) for every fitted term, plus ``n_bins``,
        ``n_events``, ``n_params``, ``llf``, ``aic``, ``deviance``,
        ``dispersion`` (= ``deviance / df_resid``), and ``predicted`` (the
        per-bin fitted mean count as a ``pd.Series`` indexed by the retained
        design rows, for :func:`per_bin_rate_gof`). ``None`` if columns are
        missing, too few bins/events, fewer than two reward levels when reward
        terms are present, or statsmodels raises a genuine numerical failure
        (rank deficiency / convergence error -> caught -> ``None``, never a
        crash).

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

    df = design[[*needed, "stop_id", "binj"]].dropna(subset=needed)
    df = df[df["exposure"] > 0]
    n_events = int((df["count"] > 0).sum())
    if len(df) < min_bins or n_events < min_events:
        logger.debug(
            "%s: too few data (%d bins, %d events; need %d/%d)",
            spec.name, len(df), n_events, min_bins, min_events,
        )
        return None
    # Reward coefficients are unidentified without contrast in reward_lag1.
    if (spec.reward_lags >= 1 or spec.reward_time_interaction) and (
        df["reward_lag1"].nunique() < 2
    ):
        logger.debug("%s: <2 reward_lag1 levels, reward terms unidentified", spec.name)
        return None

    offset = np.log(df["exposure"].to_numpy(dtype=float))
    try:
        result = smf.glm(
            spec.formula(),
            data=df,
            family=sm.families.Poisson(),
            offset=offset,
        ).fit()
    except _EXPECTED_FIT_ERRORS as exc:
        logger.warning("Per-bin GLM failed for %s: %r", spec.name, exc)
        return None

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
    df = design[needed].dropna(subset=[c for c in needed if c != "binj"])
    df = df[df["exposure"] > 0].reset_index(drop=True)
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
    for train_idx, _test_idx in kf.split(stops):
        train_stops = set(stops[train_idx])
        is_train = df["stop_id"].isin(train_stops).to_numpy()
        tr, te = df[is_train], df[~is_train]
        if len(te) == 0 or int((tr["count"] > 0).sum()) < 1:
            logger.debug("CV %s: degenerate fold (empty test or zero train events)",
                         spec.name)
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
        y = te["count"].to_numpy(dtype=float)
        ll = poisson.logpmf(y, mu)
        ll0 = poisson.logpmf(y, mu0)
        if not (np.all(np.isfinite(ll)) and np.all(np.isfinite(ll0))):
            logger.debug("CV %s: non-finite held-out log-likelihood", spec.name)
            return None
        fold_means.append(float(np.mean(ll)))
        sum_diff += float(np.sum(ll - ll0))
        n_events_total += int((y > 0).sum())

    bits = sum_diff / (n_events_total * np.log(2)) if n_events_total else np.nan
    return {
        "cv_ll_per_bin_mean": float(np.mean(fold_means)),
        "cv_ll_per_bin_std": float(np.std(fold_means)),
        "cv_bits_per_event": float(bits),
        "cv_n_folds": n_folds,
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
