"""Held-out scoring of one session's unit models.

:func:`score_session` refits every model of a session on ``K - 1`` blocked
folds and scores it on the fold left out, reads each event class's held-out
improvement against redraws of the matched non-event control
(:mod:`~continuum_swr_replay.analysis.pfc.heldout`), and picks each unit's best
model per decomposition. :func:`maximum_likelihood_comparison` fits the
curve-bearing models unpenalised, so the prior's effect on each peak can be
read against the same curve fit without it. The batch runner
(``scripts/pfc/run_pfc_analysis_bandit.py``) calls them after ``run_session``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.batched_glm import (
    fit_poisson_glms,
    prepare_heldout_response,
    score_poisson_designs_heldout,
)
from continuum_swr_replay.analysis.pfc.event_sets import (
    EVENT_CLASSES,
    EXTENDED_EVENT_CLASSES,
    SWR_NO_NON_LOCAL,
)
from continuum_swr_replay.analysis.pfc.fit import model_penalties
from continuum_swr_replay.analysis.pfc.heldout import (
    BEST_MODEL_CANDIDATES,
    DEFAULT_NULL_DESIGN_BATCH_SIZE,
    ENGAGEMENT_PAIR,
    assign_blocked_folds,
    control_draw_null,
    empirical_pvalue,
    null_draw_seeds,
    required_usable_null_draws,
)
from continuum_swr_replay.analysis.pfc.modulation import (
    CURVE_SOURCES,
    EXTENDED_CURVE_SOURCES,
    EXTENDED_NESTED_MODEL_PAIRS,
    NESTED_MODEL_PAIRS,
    THREE_CLASS_MODEL,
    curve_coefficients,
    curve_peaks,
    is_separated,
    peak_log_modulation,
    peak_modulation_se,
    prior_binds,
)


def pair_name(full: str, reduced: str) -> str:
    """Column stem for one nested comparison, as the statistics table spells it."""
    return f"{full} vs. {reduced}"


def separated_column_counts(design: NDArray, counts: NDArray) -> NDArray:
    """Basis columns with no spikes anywhere under their support, per unit.

    A column whose support holds no spikes is the configuration that drives an
    unpenalised coefficient to minus infinity; under the prior it is finite but
    uninformed, so the count is reported beside the fit rather than acted on.
    The intercept is not counted -- its support is every bin.

    The bin set is every bin the models were fit on -- the whole fit mask, not
    one fold of it -- so the count describes the shipped full-data fit.

    Parameters
    ----------
    design : NDArray, shape (n_fit_bins, n_features)
        Design restricted to the fit mask; column 0 is the intercept.
    counts : NDArray, shape (n_fit_bins, n_units)
        Spike counts on those same bins.

    Returns
    -------
    NDArray of int, shape (n_units,)
    """
    support = np.asarray(design[:, 1:], dtype=float) > 0
    spikes_under_support = support.T.astype(float) @ np.asarray(counts, dtype=float)
    return (spikes_under_support == 0).sum(axis=0).astype(int)


def score_session(
    context: dict[str, object],
    *,
    modulation_curves: dict[str, NDArray],
    penalty_sigma: float,
    n_folds: int,
    fold_seed: int,
    n_control_draws: int,
    control_draw_seed: int,
    logger: logging.Logger,
    null_classes: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score every model out of fold, and read the improvements against a null.

    Each model is refit on ``K - 1`` folds of support-separated immobility
    blocks and
    scored by the Poisson deviance of the fold left out, under the same prior
    the shipped fit used. A nested pair's ``heldout_delta_dev`` is the reduced
    model's held-out deviance minus the full model's: positive means the added
    regressors predict spikes the fit never saw. The improvement a real event
    class buys is read against ``R`` draws of the matched non-event control --
    the same rules, no event content -- which is the only null available for a
    statistic that is neither a likelihood ratio nor independent across bins.
    If an event has no legal within-bout match, the empirical test uses one
    fixed matchable subset on both its observed and null sides; the all-event
    held-out difference remains separately reported.

    The model sets it reads -- :data:`ENGAGEMENT_PAIR`,
    :data:`BEST_MODEL_CANDIDATES` and :data:`CURVE_SOURCES` -- are defined in
    ``continuum_swr_replay.analysis.pfc.heldout`` and ``.modulation``, beside
    the folds, the null and the p-value it is built from.

    Parameters
    ----------
    context : dict
        The session's fit material, as ``run_session`` hands it back through
        ``scoring_out``.
    modulation_curves : dict[str, NDArray]
        The shipped penalised curves, keyed as :data:`CURVE_SOURCES` is, so the
        prior's effect on each can be read against the same curve fit without
        it.
    penalty_sigma : float
        The prior the shipped fit used, applied to every training fit.
    n_folds : int
        Folds ``K``.
    fold_seed : int
        Seed of the run-to-fold assignment, recorded in the sidecar.
    n_control_draws : int
        Draws ``R`` of the null. ``0`` skips the null entirely, and the
        empirical p-value columns are then not written at all rather than
        written empty.
    control_draw_seed : int
        Seed the ``R`` draw seeds are derived from, recorded in the sidecar.
    logger : logging.Logger
    null_classes : sequence of str, optional
        Event classes to draw the null for; every class of the session's event
        set when None. The engagement null is drawn either way.

    Returns
    -------
    statistics : pandas.DataFrame
        One row per unit, in unit order, carrying the held-out deviances, the
        nested differences (total and per fold), the empirical p-values and the
        best model per decomposition.
    diagnostics : dict
        ``fold_sizes``, ``events_per_fold`` for each event class, and the
        models found non-estimable -- what the choice of ``K`` is read from.
    """
    fit_mask = context["fit_mask"]
    designs: dict[str, NDArray] = context["designs"]
    fit_rows = np.asarray(fit_mask, dtype=bool)
    # The extended event set carries SWRs without replay, their models, their
    # nested pairs, a three-class best-model set and two more curves.
    extended = SWR_NO_NON_LOCAL in context["event_times"]
    nested_pairs = NESTED_MODEL_PAIRS + (
        EXTENDED_NESTED_MODEL_PAIRS if extended else ()
    )
    best_model_candidates = dict(BEST_MODEL_CANDIDATES)
    if extended:
        best_model_candidates["min_heldout_model_three_class"] = (
            "constant",
            "non_local",
            "non_local_SWR + non_local_no_SWR",
            THREE_CLASS_MODEL,
        )
    curve_sources = {**CURVE_SOURCES, **(EXTENDED_CURVE_SOURCES if extended else {})}
    event_classes = EXTENDED_EVENT_CLASSES if extended else EVENT_CLASSES
    tested_classes = [
        name
        for name in event_classes
        if null_classes is None or name in set(null_classes)
    ]
    counts = np.asarray(context["counts"], dtype=float)
    fit_counts = counts[fit_rows]
    n_units = counts.shape[1]

    support_radius_bins = int(np.max(np.abs(np.asarray(context["lags"]))))
    folds = assign_blocked_folds(
        fit_mask,
        n_folds,
        fold_seed,
        support_radius_bins=support_radius_bins,
    )
    penalties = model_penalties(designs, penalty_sigma)
    scoring_response = prepare_heldout_response(fit_counts, folds)

    # A design whose event class has no events leaves an all-zero basis block,
    # which is rank-deficient against the intercept. Such a model is not an
    # estimated zero effect, it is no estimate at all, so every statistic that
    # reads it is NaN -- the same rule the modulation curves are held to.
    non_estimable = {
        name
        for name, design in designs.items()
        if np.linalg.matrix_rank(design[fit_rows]) < design.shape[1]
    }
    if non_estimable:
        logger.warning(
            "Non-estimable designs (rank-deficient on the fit bins): %s",
            sorted(non_estimable),
        )

    deviance: dict[str, NDArray] = {}
    per_fold: dict[str, NDArray] = {}
    for name in non_estimable:
        deviance[name] = np.full(n_units, np.nan)
        per_fold[name] = np.full((n_folds, n_units), np.nan)
    widths = sorted(
        {
            design.shape[1]
            for name, design in designs.items()
            if name not in non_estimable
        }
    )
    for width in widths:
        names = [
            name
            for name, design in designs.items()
            if name not in non_estimable and design.shape[1] == width
        ]
        design_batch = np.stack([designs[name][fit_rows] for name in names])
        penalty_batch = np.stack(
            [penalties.get(name, np.zeros(width)) for name in names]
        )
        scored = score_poisson_designs_heldout(
            design_batch,
            scoring_response,
            penalty_batch,
            design_batch_size=DEFAULT_NULL_DESIGN_BATCH_SIZE,
        )
        for index, name in enumerate(names):
            deviance[name] = scored.total[index]
            per_fold[name] = scored.per_fold[index]
    logger.info("Scored %d models on %d folds", len(designs), n_folds)

    statistics: dict[str, object] = {}
    for name in designs:
        statistics[f"{name}: heldout_deviance"] = deviance[name]
        statistics[f"{name}: separated_columns"] = separated_column_counts(
            designs[name][fit_rows], fit_counts
        )

    for full, reduced in nested_pairs:
        stem = pair_name(full, reduced)
        statistics[f"{stem}: heldout_delta_dev"] = deviance[reduced] - deviance[full]
        difference = per_fold[reduced] - per_fold[full]
        statistics[f"{stem}: heldout_delta_dev_folds"] = list(difference.T)

    for column, candidates in best_model_candidates.items():
        usable = [name for name in candidates if name not in non_estimable]
        if usable:
            stacked = np.column_stack([deviance[name] for name in usable])
            best = np.argmin(np.where(np.isnan(stacked), np.inf, stacked), axis=1)
            statistics[column] = [
                None if not np.isfinite(row).any() else usable[index]
                for row, index in zip(stacked, best, strict=True)
            ]
        else:
            statistics[column] = [None] * n_units

    diagnostics: dict[str, object] = {
        "fold_sizes": np.bincount(folds, minlength=n_folds).tolist(),
        "events_per_fold": events_per_fold(context, folds),
        "non_estimable_models": sorted(non_estimable),
        "fold_support_radius_bins": support_radius_bins,
    }

    statistics.update(
        maximum_likelihood_comparison(
            designs,
            fit_rows,
            fit_counts,
            context,
            modulation_curves,
            logger,
            curve_sources=curve_sources,
        )
    )

    if n_control_draws > 0:
        required_draws = required_usable_null_draws(n_control_draws)
        seeds = null_draw_seeds(control_draw_seed, n_control_draws)
        logger.info(
            "Drawing the null: %d matched-control draws per class on %d folds...",
            n_control_draws,
            n_folds,
        )
        null = control_draw_null(
            context["event_times"],
            fit_mask,
            None,
            None,
            rng_seeds=seeds,
            subset_seed=control_draw_seed,
            spline_basis=context["spline_basis"],
            lags=context["lags"],
            penalty_sigma=penalty_sigma,
            classes=tested_classes,
            baseline_control=context["event_times"]["matched_control"],
            model_deviance=deviance,
            response=scoring_response,
            avoid=context["artifact_dropped"],
        )
        # Each class is read against a null drawn to its own fixed matchable
        # onset subset and durations: a regressor built from 30 windows buys
        # less than one built from 300 whatever either contains, so one shared
        # null would hold classes of different size to the same reference.
        usable_draws: dict[str, NDArray] = {}
        for event_class in tested_classes:
            stem = pair_name(event_class, "constant")
            statistics[f"{stem}: matched_null_observed_delta_dev"] = (
                null.observed_improvement[event_class]
            )
            usable = np.isfinite(null.improvement[event_class]).sum(axis=0)
            usable_draws[event_class] = usable
            statistics[f"{event_class}: n_usable_null_draws"] = usable
            statistics[f"{event_class}: p_heldout"] = empirical_pvalue(
                null.observed_improvement[event_class],
                null.improvement[event_class],
                min_usable_draws=required_draws,
            )
        engagement_stem = pair_name(*ENGAGEMENT_PAIR)
        statistics[f"{engagement_stem}: matched_null_observed_delta_dev"] = (
            null.observed_engagement
        )
        usable_draws["engagement"] = np.isfinite(null.engagement).sum(axis=0)
        statistics[f"{engagement_stem}: n_usable_null_draws"] = usable_draws[
            "engagement"
        ]
        statistics[f"{engagement_stem}: p_heldout"] = empirical_pvalue(
            null.observed_engagement,
            null.engagement,
            min_usable_draws=required_draws,
        )
        diagnostics["null_usable_draws"] = {
            name: values.astype(int).tolist() for name, values in usable_draws.items()
        }
        diagnostics["null_windows"] = {
            name: {
                "n_events_total": int(null.n_events[name]),
                "n_events_tested": int(null.n_tested[name]),
                "n_events_excluded_unmatchable": int(
                    null.n_events[name] - null.n_tested[name]
                ),
                "tested_source_indices": (
                    null.tested_source_indices[name].astype(int).tolist()
                ),
                "n_placed_median": float(np.median(null.n_placed[name])),
                "n_placed_min": int(np.min(null.n_placed[name])),
                "n_exact_draws": int(
                    np.sum(null.n_placed[name] == null.n_tested[name])
                ),
                "n_required_draws": required_draws,
            }
            for name in null.n_placed
        }
        # The engagement null's draws are independent of the shipped control
        # they are entered beside, so they may land on it -- which two real
        # event classes never do. The cost of drawing the null by the control's
        # own rules is measured rather than assumed away.
        overlap = np.asarray(null.engagement_overlap, dtype=float)
        diagnostics["engagement_null_overlap"] = (
            {
                "median": float(np.median(overlap)),
                "max": float(np.max(overlap)),
            }
            if overlap.size
            else None
        )

    return pd.DataFrame(statistics), diagnostics


def maximum_likelihood_comparison(
    designs: dict[str, NDArray],
    fit_rows: NDArray,
    fit_counts: NDArray,
    context: dict[str, object],
    modulation_curves: dict[str, NDArray],
    logger: logging.Logger,
    curve_sources: dict[str, tuple[str, int]] = CURVE_SOURCES,
) -> dict[str, NDArray]:
    """What the same curve looks like without the prior, and whether it binds.

    The shipped curve is the penalised one; the question a prior has to answer
    is where it did the work. So each curve-bearing model is fit unpenalised as
    well, and the two peaks are differenced against the estimation error the
    *unpenalised* fit itself reports -- the standard error of the kernel at its
    own peak lag, off-diagonal terms included, which a diagonal ``bse`` cannot
    give. A unit whose unpenalised fit ran away has no peak to difference
    against: it is flagged instead, and left out of the binding count.

    Parameters
    ----------
    designs : dict[str, NDArray]
        Full-length designs by model name.
    fit_rows : NDArray of bool, shape (n_time,)
        The bins the models were fit on.
    fit_counts : NDArray, shape (n_fit_bins, n_units)
        Spike counts on those bins.
    context : dict
        The session's fit material from ``run_session``; its ``spline_basis``
        and ``lag_time`` say what a coefficient block means.
    modulation_curves : dict[str, NDArray]
        The shipped penalised curves, ``(n_report_lags, n_units)`` each, keyed
        as :data:`CURVE_SOURCES` is.
    logger : logging.Logger
    curve_sources : dict
        The curves to compare and where each is read from;
        :data:`CURVE_SOURCES` plus, for the extended event set,
        :data:`EXTENDED_CURVE_SOURCES`.

    Returns
    -------
    dict[str, NDArray]
        Per curve: the penalised peak, the unpenalised peak and its standard
        error, whether the unpenalised fit converged or separated, and whether
        the prior binds.
    """
    spline_basis = context["spline_basis"]
    lag_time = context["lag_time"]
    n_basis = spline_basis.shape[1]
    wanted = sorted({model for model, _ in curve_sources.values()})
    logger.info("Fitting %d models unpenalised, to read the prior against", len(wanted))
    unpenalised = {
        model: fit_poisson_glms(designs[model][fit_rows], fit_counts, penalty=None)
        for model in wanted
    }

    columns: dict[str, NDArray] = {}
    for curve in curve_sources:
        model, block = curve_coefficients(curve, n_basis, curve_sources)
        result = unpenalised[model]
        raw_params = np.asarray(result.params)
        model_separated = is_separated(raw_params[:, 1:])
        converged = np.asarray(result.converged, dtype=bool)
        usable = converged & ~model_separated
        params = raw_params[:, block].copy()
        covariance = np.asarray(result.covariance)[:, block, block].copy()
        params[~usable] = np.nan
        covariance[~usable] = np.nan
        mle = peak_log_modulation(params, spline_basis, lag_time)
        mle_se = peak_modulation_se(covariance, spline_basis, mle.index)
        penalised_peak = curve_peaks(modulation_curves[curve])
        columns[f"{curve}: peak_log_modulation"] = penalised_peak
        columns[f"{curve}: peak_log_modulation_mle"] = mle.value
        columns[f"{curve}: peak_log_modulation_mle_se"] = mle_se
        columns[f"{curve}: mle_converged"] = converged
        columns[f"{curve}: mle_separated"] = model_separated
        columns[f"{curve}: prior_binds"] = prior_binds(
            penalised_peak, mle.value, mle_se
        )
    return columns


def events_per_fold(context: dict[str, object], folds: NDArray) -> dict[str, list[int]]:
    """Event onsets falling in each fold, per event class.

    The rarest class's count is what a choice of ``K`` has to live with: a fold
    holding a handful of events of the class under test cannot say much about
    a model built on it.
    """
    fit_mask = context["fit_mask"]
    time = np.asarray(fit_mask.index, dtype=float)
    fit_rows = np.asarray(fit_mask, dtype=bool)
    fold_of_bin = np.full(len(time), -1, dtype=np.int64)
    fold_of_bin[fit_rows] = folds
    n_folds = int(folds.max()) + 1
    per_class: dict[str, list[int]] = {}
    for name, times in context["event_times"].items():
        if not len(times):
            per_class[name] = [0] * n_folds
            continue
        index = np.searchsorted(time, times["start_time"].to_numpy(dtype=float))
        index = np.clip(index, 0, len(time) - 1)
        assigned = fold_of_bin[index]
        per_class[name] = [int((assigned == fold).sum()) for fold in range(n_folds)]
    return per_class
