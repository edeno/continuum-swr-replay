"""Generalized Linear Model analysis for PFC neural activity.

Fits Poisson GLMs relating single-unit spiking activity to hippocampal
replay events (SWR-coincident and non-SWR) during immobility, and tests
theta phase modulation during running: one neuron (``process_neuron``)
and one session (``run_session``).
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import dask
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.pfc.design import (
    _make_design_matrices,
    build_theta_design,
    symmetric_lags,
)
from continuum_swr_replay.analysis.pfc.event_sets import (
    DEFAULT_BASIS_HALF_WINDOW_S,
    EVENT_SETS,
    SWR_NO_NON_LOCAL,
    build_session_event_sets,
    build_session_fit_mask,
    count_events_covering_fit_bins,
    swr_without_non_local,
)
from continuum_swr_replay.analysis.pfc.fit import (
    _check_fitter,
    _fit_diagnostics,
    _fit_immobility_models,
    _stack_fit_diagnostics,
    fit_models_batched,
    fitter_provenance,
    model_penalties,
    test_theta_modulation,
)
from continuum_swr_replay.analysis.pfc.matched_control import (
    MatchedControlDraw,
    artifact_dropped_events,
    make_matched_control_times,
    matched_control_seed,
)
from continuum_swr_replay.analysis.pfc.modulation import (
    CURVE_SOURCES,
    NESTED_MODEL_PAIRS,
    curve_coefficients,
    get_frac_explained_deviance,
    get_frac_null_deviance,
    in_report_window,
)
from continuum_swr_replay.analysis.pfc.scoring import pair_name
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.intervals import (
    get_spikecount_per_time_bin,
    interval_mask,
    sample_step,
)
from continuum_swr_replay.paths import table_path

# ``load_data`` and the hippocampus helpers below open a Spyglass /
# DataJoint connection at import time. We defer those imports into
# ``run_session`` (the only function that uses them) so that
# ``process_neuron`` here, and the design-matrix builders it imports,
# stay offline-importable for unit tests on machines without VPN.

logger = logging.getLogger(__name__)


def process_neuron(
    neuron_spike_times: NDArray,
    is_immobile: pd.Series,
    theta_phase: pd.Series,
    time: NDArray,
    immobility_design_matrices: dict[str, NDArray],
    spline_basis: NDArray,
    lag_time: NDArray,
    logger_name: str | None = None,
    fit_mask: pd.Series | None = None,
    fitter: str = "statsmodels",
    prefitted_immobility: Mapping[str, Any] | None = None,
    theta_design: tuple[NDArray, NDArray] | None = None,
    design_ranks: Mapping[str, int] | None = None,
    fit_diagnostics_out: dict[str, Any] | None = None,
    penalty_sigma: float | None = None,
    prebinned_spike_count: NDArray | None = None,
    in_excluded_time: NDArray | None = None,
) -> tuple[dict[str, Any], NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Fit GLM models for a single neuron and compute modulation statistics.

    Fits 9 Poisson GLM models during immobility to test how the neuron's
    firing rate relates to SWR and non-local (replay) events. Models are
    organized into two decomposition strategies:

    **SWR-controlled decomposition** — separates the generic ripple response
    from replay-specific modulation:

    - ``constant``: intercept only (baseline)
    - ``SWR``: all whole-event-speed-eligible detected ripples
    - ``non_local``: all replay events regardless of SWR overlap
    - ``SWR + non_local_no_SWR``: ripples + non-SWR replay (key model;
      isolates replay-specific effects after controlling for SWR)

    **Replay-split decomposition** — tests whether SWR-coincident and
    non-SWR replay drive different responses:

    - ``non_local_SWR``: replay events that co-occur with a ripple
    - ``non_local_no_SWR``: replay events without a ripple
    - ``non_local_SWR + non_local_no_SWR``: both replay types with
      separate regressors

    **Engagement decomposition** -- separates replay from the immobility
    it happens in:

    - ``matched_control``: duration-matched non-event windows drawn from
      the same fitted immobility bouts (:func:`make_matched_control_times`)
    - ``matched_control + non_local_no_SWR``: the baseline plus non-SWR
      replay, the model the engagement test is run on

    The analysis addresses a sequence of scientific questions:

    **Q1. Do PFC neurons respond to hippocampal events during immobility?**
    ``SWR vs constant``, ``non_local vs constant``

    **Q2. Are non-local events and SWRs redundant or independent?**
    ``SWR + non_local_no_SWR vs SWR`` (replay adds beyond ripples),
    ``SWR + non_local_no_SWR vs non_local_no_SWR`` (ripples add beyond
    replay).

    **Q3. Do PFC neurons respond to non-local events lacking a detected SWR?**
    ``non_local_no_SWR vs constant``. Modulation: ``no_swr_modulation``.

    **Q4. Is the non-SWR replay response robust to sub-threshold ripple
    confounds?**
    Compare ``no_swr_modulation`` (controlling for all SWRs) vs
    ``non_local_no_swr_modulation`` (controlling for replay-coincident
    SWRs only). ``non_local_SWR + non_local_no_SWR vs non_local_SWR``.

    **Q5. Do SWR-coincident and non-SWR replay drive different responses?**
    ``non_local_SWR + non_local_no_SWR vs non_local`` (does splitting
    improve fit?). Compare
    ``non_local_swr_modulation`` vs ``non_local_no_swr_modulation`` shapes.

    **Q6. Does replay engage the neuron beyond the immobility it occurs
    in?**
    ``matched_control + non_local_no_SWR vs. matched_control`` (the
    engagement test: non-SWR replay adds beyond a duration-matched
    non-event baseline drawn from the same bouts), and
    ``matched_control vs. constant`` (does that baseline itself modulate
    firing -- a positive answer means the peri-event window is not
    exchangeable with the rest of the bout, and the engagement test is the
    comparison to read). The control is matched one per *non-local* event,
    SWR-coincident ones included, so it is more numerous than the
    ``non_local_no_SWR`` class it is compared against; a source event's
    duration fixes its control's duration and constrains where the window
    fits, and nothing else about the event is matched.

    **Q7. What is the relationship to theta modulation during running?**
    ``theta_mod``, ``theta_preferred_phase``, firing rates.

    Each nested pair carries an effect size -- ``frac_null_dev`` and
    ``frac_explained_dev``, ratios of the deviances the added regressors buy.
    No likelihood-ratio statistic, p-value or AIC is produced here: the fits
    are penalised, so a likelihood ratio has no chi-square reference
    distribution at any stated degrees of freedom and an AIC has no stated
    number of parameters. What the models are compared and selected by is the
    held-out deviance the runner scores them on, read against a null drawn by
    redrawing the matched non-event control. Multiple-comparison correction
    across neurons and the population layer are downstream.

    Parameters
    ----------
    neuron_spike_times : NDArray, shape (n_spikes,)
        Spike times in seconds.
    is_immobile : pd.Series, shape (n_time,)
        Boolean series indicating immobility at each time bin.
    theta_phase : pd.Series, shape (n_lfp_samples,)
        Theta phase timeseries (radians).
    time : NDArray, shape (n_time,)
        Time bin centers in seconds (from ``position_info.index``).
    immobility_design_matrices : dict[str, NDArray]
        Pre-computed design matrices keyed by model name.
        Each value has shape ``(n_time, n_features)``.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Spline basis functions for reconstructing peri-event modulation.
    lag_time : NDArray, shape (n_lags,)
        Lag times in seconds corresponding to spline_basis rows.
    logger_name : str | None, optional
        Logger name for per-neuron messages. If None, uses module logger.
    fit_mask : pd.Series | None, shape (n_time,), optional
        Bins the immobility GLMs are fit on. Defaults to ``is_immobile``, i.e.
        every immobile bin in the epoch; a stop-cohort run passes
        ``is_immobile & in_cohort_stop`` so the models describe the same
        population the hippocampal ladders do. ``is_immobile`` itself still
        defines the *running* comparison (theta modulation and
        ``run_avg_firing_rate``): a bin outside the cohort is not running, so it
        must not be counted as such.
    fitter : {"statsmodels", "jax"}, optional
        Backend for the IRLS fits. Ignored for any model whose results are
        supplied prefitted. The default is the unpenalised control; the
        canonical bundles are fit through :func:`run_session` with
        ``fitter="jax"`` under the prior ``model_penalties(designs, sigma)``.
    prefitted_immobility : Mapping[str, Any] | None, optional
        This neuron's immobility-model results, keyed by model name, already
        fit on ``fit_mask``. :func:`run_session` fits every neuron of a session
        at once and passes each neuron's slice in here.
    theta_design : tuple | None, optional
        ``(joint_mask, design_matrix)`` from :func:`build_theta_design`, built
        once for the session rather than once per neuron.
    design_ranks : Mapping[str, int] | None, optional
        ``numpy.linalg.matrix_rank`` of each design on ``fit_mask``. The rank
        is a property of the design, not of the neuron, so a session computes
        it once; when omitted it is computed here, as it always was.
    fit_diagnostics_out : dict | None, optional
        When given, filled with this neuron's per-model fit summary (params,
        deviance, llf, aic, rank, convergence). For the fitter equivalence
        audit; never part of the shipped bundle.
    penalty_sigma : float | None, optional
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale, when this neuron's immobility models are fit here. ``None`` fits
        them unpenalised. Ignored when ``prefitted_immobility`` is given -- the
        prior is then whatever the session was prefit with.
    prebinned_spike_count : NDArray, shape (n_time,), optional
        Counts already binned on ``time``. The session runner bins every unit
        once for the batched and held-out fits and passes the corresponding
        column here; direct callers may continue to pass spike times only.
    in_excluded_time : NDArray, shape (n_time,), optional
        True at the bins in the session's excluded time (the artifact mask
        widened by 0.3 s, ``analysed_time.excluded_intervals``). They are left out
        of ``run_avg_firing_rate`` and
        ``immobility_avg_firing_rate_all_immobile``; ``fit_mask`` already
        leaves them out of the fit. None leaves no bin out.

    Returns
    -------
    neuron_stats : dict[str, Any]
        Per-neuron statistics including:

        - ``run_avg_firing_rate``: Hz over every moving bin outside excluded
          time.
        - ``immobility_avg_firing_rate``: Hz over the bins the models were fit
          on -- with a stop cohort, the cohort's immobile bins, not every
          immobile bin. This is the rate the downstream ``min_firing_rate``
          inclusion gate screens on, so it describes the sample the estimates
          come from; its value therefore depends on the cohort the bundle was
          fit on (named in the bundle's provenance sidecar).
        - ``immobility_avg_firing_rate_all_immobile``: Hz over every immobile
          bin outside excluded time, cohort or not. Equal to
          ``immobility_avg_firing_rate`` exactly when no stop restriction was
          applied and the session has no excluded time (the fit mask also
          leaves out the bins whose basis window reaches excluded time).
        - ``theta_mod``, ``theta_preferred_phase``: theta modulation during running
        - ``{model} vs. {baseline}: frac_null_dev, frac_explained_dev``:
          fractional deviance measures
        - ``SWR_max_mod``, ``non_local_no_SWR_max_mod``: peak modulation
          from SWR-controlled model
        - ``non_local_SWR_max_mod``, ``non_local_no_SWR_split_max_mod``:
          peak modulation from replay-split model
        - ``matched_control_max_mod``: peak modulation of the matched
          non-event baseline from the engagement model
        - ``*_max_mod_time``: corresponding peak lag times (s)

        A model with no usable fit -- NaN parameters, or an iteration that
        never settled -- is non-estimable for this neuron: every statistic
        that reads it is NaN, and the neuron's other models are reported as
        usual. Empty dict with only firing rates if the fitting call itself
        raised.

    swr_modulation : NDArray
        SWR peri-event modulation from SWR-controlled model
        (``SWR + non_local_no_SWR``). Shape (n_lag_times,).
    no_swr_modulation : NDArray
        Non-SWR replay modulation from SWR-controlled model, controlling
        for all whole-event-speed-eligible ripples. Shape (n_lag_times,).
    non_local_swr_modulation : NDArray
        SWR-coincident replay modulation from replay-split model
        (``non_local_SWR + non_local_no_SWR``). Shape (n_lag_times,).
    non_local_no_swr_modulation : NDArray
        Non-SWR replay modulation from replay-split model, controlling
        for replay-coincident ripples only. Shape (n_lag_times,).
        Comparing this with ``no_swr_modulation`` tests whether the
        non-SWR replay response is robust to the choice of SWR control.
    matched_control_modulation : NDArray
        Matched non-event baseline modulation from the engagement model
        (``matched_control + non_local_no_SWR``), i.e. the response around
        duration-matched control windows after accounting for real non-SWR
        replay. Shape (n_lag_times,). A curve that is flat while
        ``no_swr_modulation`` is not is the evidence that the replay
        response is to replay rather than to the immobility around it.

    """
    _logger = logging.getLogger(logger_name) if logger_name else logger

    neuron_spike_count = (
        get_spikecount_per_time_bin(neuron_spike_times, time)
        if prebinned_spike_count is None
        else np.asarray(prebinned_spike_count, dtype=float)
    )
    if neuron_spike_count.shape != np.asarray(time).shape:
        raise ValueError(
            "prebinned_spike_count must have one value per time bin; got "
            f"{neuron_spike_count.shape} for time {np.asarray(time).shape}"
        )
    sampling_frequency = 1 / sample_step(time)
    if fit_mask is None:
        fit_mask = is_immobile
    # The descriptive rates leave excluded time out, as the fit mask does.
    immobile_bins = np.asarray(is_immobile, dtype=bool)
    running_bins = ~immobile_bins
    if in_excluded_time is not None:
        analysed = ~np.asarray(in_excluded_time, dtype=bool)
        immobile_bins = immobile_bins & analysed
        running_bins = running_bins & analysed

    try:
        neuron_stats: dict[str, Any] = {}

        # Get run theta statistics
        (
            neuron_stats["theta_mod"],
            neuron_stats["theta_preferred_phase"],
        ) = test_theta_modulation(
            theta_phase,
            is_immobile,
            neuron_spike_count,
            sampling_frequency=sampling_frequency,
            design=theta_design,
        )
        neuron_stats.update(
            _firing_rates(
                neuron_spike_count,
                running_bins,
                fit_mask,
                immobile_bins,
                sampling_frequency,
            )
        )

        results, _ = _fit_immobility_models(
            fit_mask,
            immobility_design_matrices,
            neuron_spike_count,
            fitter=fitter,
            prefitted=prefitted_immobility,
            penalties=model_penalties(immobility_design_matrices, penalty_sigma),
        )

        # A model whose fit produced nothing usable -- NaN parameters, or an
        # iteration that never settled -- is marked non-estimable for THIS
        # neuron and routed to NaN below, exactly as an absent event class is.
        # The neuron's other models are unaffected: they are separate fits of
        # separate designs, and a neuron whose ``matched_control`` fit failed
        # still has a ``SWR`` fit that says something. Failing the whole neuron
        # would discard that, and would do it silently -- the neuron would come
        # back with firing rates and no statistics at all.
        failed_models = _failed_models(results)
        if failed_models:
            _logger.warning(
                "No usable fit for models %s (NaN parameters, or no "
                "convergence); their statistics are NaN for this neuron. "
                "Immobility firing rate: %.3f Hz.",
                sorted(failed_models),
                neuron_stats["immobility_avg_firing_rate"],
            )

        if fit_diagnostics_out is not None:
            fit_diagnostics_out.update(_fit_diagnostics(results))

        non_estimable_models = _non_estimable_models(
            failed_models, immobility_design_matrices, fit_mask, design_ranks
        )

        neuron_stats.update(_deviance_fractions(results, non_estimable_models))

        (
            modulation_stats,
            swr_modulation,
            no_swr_modulation,
            non_local_swr_modulation,
            non_local_no_swr_modulation,
            matched_control_modulation,
        ) = _modulation_curves(results, non_estimable_models, spline_basis, lag_time)
        neuron_stats.update(modulation_stats)

    except (ValueError, np.linalg.LinAlgError):
        _logger.warning(
            "GLM fitting failed for neuron; falling back to firing rates only",
            exc_info=True,
        )
        neuron_stats = _firing_rates(
            neuron_spike_count,
            running_bins,
            fit_mask,
            immobile_bins,
            sampling_frequency,
        )
        nan_modulation = np.full(spline_basis.shape[0], np.nan)
        swr_modulation = nan_modulation
        no_swr_modulation = nan_modulation
        non_local_swr_modulation = nan_modulation
        non_local_no_swr_modulation = nan_modulation
        matched_control_modulation = nan_modulation

    return (
        neuron_stats,
        swr_modulation,
        no_swr_modulation,
        non_local_swr_modulation,
        non_local_no_swr_modulation,
        matched_control_modulation,
    )


def _firing_rates(
    neuron_spike_count: NDArray,
    running_bins: NDArray,
    fit_mask: pd.Series,
    immobile_bins: NDArray,
    sampling_frequency: float,
) -> dict[str, float]:
    """One neuron's descriptive firing rates, in Hz.

    Parameters
    ----------
    neuron_spike_count : NDArray, shape (n_time,)
        Spike counts on the session's time bins.
    running_bins : NDArray of bool, shape (n_time,)
        Moving bins outside excluded time.
    fit_mask : pd.Series of bool, shape (n_time,)
        Bins the immobility models are fit on.
    immobile_bins : NDArray of bool, shape (n_time,)
        Immobile bins outside excluded time, cohort or not.
    sampling_frequency : float
        Bins per second.

    Returns
    -------
    dict
        ``run_avg_firing_rate``, ``immobility_avg_firing_rate`` and
        ``immobility_avg_firing_rate_all_immobile``, in that order.
    """
    neuron_stats: dict[str, float] = {}
    neuron_stats["run_avg_firing_rate"] = (
        neuron_spike_count[running_bins].mean() * sampling_frequency
    )

    # Get immobility SWR/non-local statistics. This is the mean rate over
    # the bins the models are actually fit on, so the downstream
    # minimum-firing-rate inclusion gate screens the sample the estimates
    # come from rather than a wider one. The column name is the same across
    # cohorts, so the rate over every immobile bin outside excluded time is
    # carried alongside it: the pair makes a bundle's cohort readable from
    # the table itself (the two are equal exactly when no stop restriction
    # was applied).
    neuron_stats["immobility_avg_firing_rate"] = (
        neuron_spike_count[fit_mask].mean() * sampling_frequency
    )
    neuron_stats["immobility_avg_firing_rate_all_immobile"] = (
        neuron_spike_count[immobile_bins].mean() * sampling_frequency
    )
    return neuron_stats


def _failed_models(results: Mapping[str, Any]) -> set[str]:
    """Models whose fit has NaN parameters or never converged.

    A result with no ``converged`` flag counts as not converged.
    """
    return {
        name
        for name, res in results.items()
        if np.any(np.isnan(res.params)) or not bool(getattr(res, "converged", False))
    }


def _non_estimable_models(
    failed_models: set[str],
    immobility_design_matrices: Mapping[str, NDArray],
    fit_mask: pd.Series,
    design_ranks: Mapping[str, int] | None,
) -> set[str]:
    """The failed models plus every model whose design is rank-deficient.

    Parameters
    ----------
    failed_models : set of str
        Models with no usable fit (:func:`_failed_models`).
    immobility_design_matrices : Mapping[str, NDArray]
        Design matrices keyed by model name, each ``(n_time, n_features)``.
    fit_mask : pd.Series of bool, shape (n_time,)
        Bins the models are fit on.
    design_ranks : Mapping[str, int] or None
        Each design's rank on ``fit_mask``, or None to compute it here.

    Returns
    -------
    set of str
    """
    # Detect non-estimable models. ``_make_conv_spline_basis``
    # builds spline regressors for every event class regardless
    # of whether that class has any events; an empty event class
    # leaves its spline columns all-zero, so the design matrix is
    # rank-deficient against the intercept. The fitter still
    # returns finite (zero) parameters for those columns and
    # finite degenerate LR / fractional-deviance values, so the
    # failed-fit check (``_failed_models``) doesn't catch them. Mark every
    # rank-deficient design as non-estimable up front so the caller routes
    # the affected deviance fractions and modulation
    # extractions to NaN — so downstream consumers can
    # tell "absent event class" from "estimated zero effect".
    non_estimable_models = set(failed_models)
    for name, design in immobility_design_matrices.items():
        X_im = design[fit_mask]
        if X_im.shape[0] == 0:
            non_estimable_models.add(name)
            continue
        # The rank is a property of the design, which every neuron of a
        # session shares, so a session-level caller supplies it rather than
        # paying for the same decomposition once per neuron.
        rank = (
            np.linalg.matrix_rank(X_im) if design_ranks is None else design_ranks[name]
        )
        if rank < X_im.shape[1]:
            non_estimable_models.add(name)
    return non_estimable_models


def _deviance_fractions(
    results: Mapping[str, Any], non_estimable_models: set[str]
) -> dict[str, float]:
    """``frac_null_dev`` and ``frac_explained_dev`` of every nested model pair.

    Parameters
    ----------
    results : Mapping[str, Any]
        Fitted results keyed by model name.
    non_estimable_models : set of str
        Models whose pairs are reported as NaN.

    Returns
    -------
    dict
        ``"{full} vs. {reduced}: frac_null_dev"`` and ``...: frac_explained_dev``
        for each pair of :data:`NESTED_MODEL_PAIRS`, in its order.
    """
    neuron_stats: dict[str, float] = {}
    ## Effect sizes: the deviance each nested pair's added regressors buy
    # frac_null_dev: (reduced - full) / constant deviance
    #   = fraction of total deviance uniquely explained by added regressors
    # frac_explained_dev: (reduced - full) / (constant - full)
    #   = fraction of full model's explained deviance due to added
    #     regressors
    #
    # These are ratios of deviances, not tests. No likelihood ratio and no
    # AIC is computed here: the fits carry a prior on the event-basis
    # coefficients, so a likelihood ratio has no chi-square reference
    # distribution at any stated degrees of freedom and an AIC has no
    # stated number of parameters. What the models are compared by is the
    # held-out deviance the runner scores them on.
    for model1, model2 in NESTED_MODEL_PAIRS:
        test_name = pair_name(model1, model2)
        if (
            model1 in non_estimable_models
            or model2 in non_estimable_models
            or "constant" in non_estimable_models
        ):
            neuron_stats[f"{test_name}: frac_null_dev"] = np.nan
            neuron_stats[f"{test_name}: frac_explained_dev"] = np.nan
        else:
            neuron_stats[f"{test_name}: frac_null_dev"] = get_frac_null_deviance(
                results[model1], results[model2], results["constant"]
            )
            neuron_stats[f"{test_name}: frac_explained_dev"] = (
                get_frac_explained_deviance(
                    results[model1], results[model2], results["constant"]
                )
            )
    return neuron_stats


#: The ``<name>_max_mod`` / ``<name>_max_mod_time`` statistic each curve's
#: peak is reported under.
_MAX_MOD_STAT = {
    "SWR": "SWR",
    "no_SWR": "non_local_no_SWR",
    "non_local_SWR": "non_local_SWR",
    "non_local_no_SWR": "non_local_no_SWR_split",
    "matched_control": "matched_control",
}


def _modulation_curves(
    results: Mapping[str, Any],
    non_estimable_models: set[str],
    spline_basis: NDArray,
    lag_time: NDArray,
) -> tuple[dict[str, float], NDArray, NDArray, NDArray, NDArray, NDArray]:
    """The peri-event modulation curves and their peaks, from the fitted models.

    Parameters
    ----------
    results : Mapping[str, Any]
        Fitted results keyed by model name.
    non_estimable_models : set of str
        Models whose curves and peaks are NaN.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Spline basis the event coefficients multiply.
    lag_time : NDArray, shape (n_lags,)
        Lag of each basis row, in seconds.

    Returns
    -------
    neuron_stats : dict
        The ``*_max_mod`` and ``*_max_mod_time`` statistics.
    swr_modulation, no_swr_modulation, non_local_swr_modulation, \
non_local_no_swr_modulation, matched_control_modulation : NDArray, shape (n_lags,)
        As :func:`process_neuron` returns them.
    """
    # Share fits with the deviance-fraction path so neuron_stats and
    # the modulation arrays can never disagree due to divergent
    # IRLS convergence. When the source model is non-estimable
    # (rank-deficient design — typically an absent event class),
    # the spline-by-coefficient product would silently emit a
    # zero curve from statsmodels' degenerate fit; route those
    # to NaN so consumers can tell "absent event class" from
    # "estimated zero modulation".
    #
    # Each curve's block is read by ``curve_coefficients``, in the design's
    # own order. For the replay-split model that is ``[constant,
    # non_local_no_SWR, non_local_SWR]`` -- NOT the order the model's name
    # lists them in. Reading them by the name's order is the bug that, between
    # 2026-04-24 and 2026-09-15, gave each curve the other class's kernel.
    n_basis_functions = spline_basis.shape[1]
    nan_modulation = np.full(spline_basis.shape[0], np.nan)
    neuron_stats: dict[str, float] = {}
    curves: dict[str, NDArray] = {}
    for curve in CURVE_SOURCES:
        model, block = curve_coefficients(curve, n_basis_functions)
        stat = _MAX_MOD_STAT[curve]
        if model in non_estimable_models:
            curves[curve] = nan_modulation
            neuron_stats[f"{stat}_max_mod"] = np.nan
            neuron_stats[f"{stat}_max_mod_time"] = np.nan
            continue
        curves[curve] = spline_basis @ results[model].params[block]
        peak = np.argmax(np.abs(curves[curve]))
        neuron_stats[f"{stat}_max_mod"] = curves[curve][peak]
        neuron_stats[f"{stat}_max_mod_time"] = lag_time[peak]
    return (
        neuron_stats,
        curves["SWR"],
        curves["no_SWR"],
        curves["non_local_SWR"],
        curves["non_local_no_SWR"],
        curves["matched_control"],
    )


def shipped_fit_counts(
    prefitted_immobility: list[dict[str, Any] | None],
) -> dict[str, int]:
    """How many of a session's shipped immobility fits never settled.

    A fit that runs to ``maxiter`` is not an estimate: ``process_neuron`` marks
    it non-estimable for that neuron and every statistic that reads it is NaN
    (the neuron's other models are reported as usual). The NaNs say a number is
    missing but not how many, or over how many neurons, so unless the count is
    taken here -- once, over the results that will actually be read -- there is
    no record of what a session dropped. A result with no ``converged`` flag
    counts as not converged.

    A neuron with no fit at all (no spikes on the fitted bins) enters neither
    total, which is why ``n_model_fits`` is reported beside the failures rather
    than assumed to be neurons x models.

    Parameters
    ----------
    prefitted_immobility : list of (dict or None)
        Per neuron, the results keyed by model name, as
        :func:`_prefit_session` assembles them.

    Returns
    -------
    dict[str, int]
        ``n_model_fits``, ``n_nonconverged_model_fits``,
        ``n_neurons_with_nonconverged_model``.
    """
    n_model_fits = 0
    n_nonconverged = 0
    n_neurons_nonconverged = 0
    for fits in prefitted_immobility:
        if fits is None:
            continue
        failures = sum(
            not bool(getattr(fit, "converged", False)) for fit in fits.values()
        )
        n_model_fits += len(fits)
        n_nonconverged += failures
        n_neurons_nonconverged += failures > 0
    return {
        "n_model_fits": int(n_model_fits),
        "n_nonconverged_model_fits": int(n_nonconverged),
        "n_neurons_with_nonconverged_model": int(n_neurons_nonconverged),
    }


def _prefit_session(
    spike_times: list[NDArray],
    time: NDArray,
    is_immobile: pd.Series,
    fit_mask: pd.Series,
    immobility_design_matrices: Mapping[str, NDArray],
    theta_phase: pd.Series,
    logger: logging.Logger,
    penalty_sigma: float | None = None,
    counts: NDArray | None = None,
) -> tuple[
    list[dict[str, Any] | None],
    dict[str, int],
    tuple[NDArray, NDArray],
    dict[str, int],
]:
    """Fit every neuron of a session against every design, in batches.

    Parameters
    ----------
    spike_times : list of NDArray
        Spike times per neuron, in seconds.
    time : NDArray, shape (n_time,)
        Time bin starts, the axis the designs and masks are on.
    is_immobile, fit_mask : pd.Series of bool, shape (n_time,)
        The running comparison's mask and the immobility models' mask.
    immobility_design_matrices : Mapping[str, NDArray]
        Full-length designs keyed by model name.
    theta_phase : pd.Series
        Theta phase timeseries, on its own clock.
    logger : logging.Logger
    penalty_sigma : float or None
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale, for the immobility models. ``None`` fits them unpenalised.
    counts : NDArray, shape (n_time, n_units), optional
        Spike counts already binned on ``time``. Supplying the session-wide
        matrix avoids binning the same spikes again for scoring and for each
        per-neuron summary.

    Returns
    -------
    prefitted_immobility : list of dict
        Per neuron, the immobility results keyed by model name.
    design_ranks : dict[str, int]
        Rank of each design on the fit mask.
    theta_design : tuple
        ``(joint_mask, design_matrix)``, built once for the session.
    fit_counts : dict[str, int]
        What the shipped fits say about themselves: ``n_model_fits`` (neuron x
        model cells that produced a fit at all), ``n_nonconverged_model_fits``
        and ``n_neurons_with_nonconverged_model``. A cell that never settled is
        reported per model rather than discarding the neuron, so the only place
        that fact survives is here.
    """
    if counts is None:
        counts = np.column_stack(
            [get_spikecount_per_time_bin(neuron, time) for neuron in spike_times]
        ).astype(float)
    else:
        counts = np.asarray(counts, dtype=float)
        if counts.shape != (len(time), len(spike_times)):
            raise ValueError(
                "counts must have shape (n_time, n_units); got "
                f"{counts.shape}, expected {(len(time), len(spike_times))}"
            )

    fit_rows = np.asarray(fit_mask, dtype=bool)
    logger.info(
        "Batch-fitting %d neurons x %d models on %d bins...",
        counts.shape[1],
        len(immobility_design_matrices),
        int(fit_rows.sum()),
    )
    batched = fit_models_batched(
        {
            model_name: design[fit_rows]
            for model_name, design in immobility_design_matrices.items()
        },
        counts[fit_rows],
        model_penalties(immobility_design_matrices, penalty_sigma),
    )
    design_ranks = {name: result.rank for name, result in batched.items()}

    # ``process_neuron`` derives the sampling frequency from the time axis and
    # passes it to the theta design's reindex tolerance. Derive it the same way
    # here, so the bins the batch is fit on are exactly the bins each neuron's
    # own call will read the regularized fit off.
    sampling_frequency = 1 / sample_step(time)
    joint_mask, theta_design = build_theta_design(
        theta_phase, is_immobile, sampling_frequency=sampling_frequency
    )

    prefitted_immobility: list[dict[str, Any]] = [
        {name: result.for_unit(i) for name, result in batched.items()}
        for i in range(counts.shape[1])
    ]

    return (
        prefitted_immobility,
        design_ranks,
        (joint_mask, theta_design),
        shipped_fit_counts(prefitted_immobility),
    )


def run_session(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: float = 500,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    run_parallel: bool = True,
    stop_cohort: str = "at_well",
    cohort_stops: pd.DataFrame | None = None,
    fitter: str = "jax",
    fit_diagnostics_out: dict[str, Any] | None = None,
    penalty_sigma: float | None = None,
    scoring_out: dict[str, Any] | None = None,
    event_set: str = "canonical",
) -> tuple[pd.DataFrame, NDArray, NDArray, NDArray, NDArray, NDArray, dict[str, Any]]:
    """Run PFC GLM analysis for all neurons in a single session.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name (e.g., "wilbur20210329_.nwb").
    epoch_name : str
        Epoch name (e.g., "02_r1").
    brain_area : str
        Brain area to analyze ("mPFC" or "OFC").
    sampling_frequency : float
        Sampling frequency in Hz.
    non_local_threshold : float
        Probability threshold the events were cut at. It is not applied here:
        non-local events are read from the canonical event table, which is cut
        at :data:`~continuum_swr_replay.analysis.constants.DEFAULT_NON_LOCAL_THRESHOLD`,
        and a different value is refused rather than silently ignored. It does
        select the cached decode frame the SWR labels are read from.
    run_parallel : bool
        Whether to use Dask for parallel neuron processing.
    stop_cohort : str
        Population to fit. ``"at_well"`` (canonical) keeps only immobility bins
        and events inside the well-anchored stops the hippocampal rate and
        content ladders are read on; ``"all_stops"`` applies no stop restriction
        at all, fitting every immobile bin and every speed-eligible event in the
        epoch. Under ``"all_stops"`` the fit mask and the event sets reproduce
        the pre-cohort pipeline exactly, and so does every fitted number when
        ``fitter="statsmodels"`` -- the batched fitter reaches the same point
        to the equivalence audit's tolerances rather than bit-for-bit, because
        it is a different arithmetic order on a different device. What differs
        in either case is the shape of the output -- the stats table gains the
        provenance columns ``stop_cohort`` and
        ``immobility_avg_firing_rate_all_immobile``, and this function returns a
        7-tuple rather than a 5-tuple.
    cohort_stops : pd.DataFrame, optional
        The session's kept stops from
        :func:`~continuum_swr_replay.analysis.pfc.cohort.load_session_stop_cohort`.
        Required unless ``stop_cohort`` is ``"all_stops"``.
    fitter : {"jax", "statsmodels"}
        How the GLMs are fit. ``"jax"`` (default) fits every neuron of the
        session against each design in one batched IRLS -- every neuron shares
        the design, so the per-neuron loop was refitting the same weighted
        cross-products once per neuron. ``"statsmodels"`` keeps the original
        per-neuron path, unchanged, for the equivalence audit. The two
        converge to the same point by construction (same starting values, same
        step, same convergence rule); the audit measures how close on real
        sessions.
    fit_diagnostics_out : dict, optional
        When given, filled with ``{model_name: {quantity: array over neurons}}``
        for the fitter equivalence audit. Never part of the shipped bundle.
    penalty_sigma : float or None
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale, for the batched fitter. ``None`` fits the models unpenalised.
        The statsmodels backend has no penalised path and refuses a value.
    scoring_out : dict, optional
        When given, filled with everything an out-of-fold score and its null
        need from this session: ``fit_mask`` (bool Series on the session's time
        axis), ``counts`` (``(n_time, n_units)`` spike counts in unit order),
        ``designs`` (the full-length design matrices by model name),
        ``spline_basis``, ``lag_time``, ``lags``, ``event_times`` (the
        matched control included, as drawn here) and ``artifact_dropped`` (the
        events the analysed-time rule dropped, which the null's controls avoid
        as the shipped one does). It is the same material the
        fits were built from, handed over rather than rebuilt: a caller that
        reassembled it from the same functions would pay a second load of the
        session and could drift from what was actually fit. Left empty for a
        session with no units in ``brain_area``.
    event_set : {"canonical", "with_swr_without_replay"}
        ``"with_swr_without_replay"`` adds the class ``SWR_no_non_local`` (the
        SWRs that overlap no non-local event) and its two models,
        ``SWR_no_non_local`` and
        :data:`~continuum_swr_replay.analysis.pfc.modulation.THREE_CLASS_MODEL`.
        The returned curves are the canonical five either way; the runner reads
        the extra curves off the fit material in ``scoring_out``.

    Returns
    -------
    neuron_stats : pd.DataFrame
        Per-neuron statistics. Shape (n_neurons, n_stats). Includes GLM
        comparison results, theta modulation and firing rates.
    swr_modulation : NDArray
        SWR peri-event modulation from SWR-controlled model.
        Shape (n_lag_times, n_neurons).
    no_swr_modulation : NDArray
        Non-SWR replay modulation from SWR-controlled model, controlling
        for all whole-event-speed-eligible ripples. Shape (n_lag_times, n_neurons).
    non_local_swr_modulation : NDArray
        SWR-coincident replay modulation from replay-split model.
        Shape (n_lag_times, n_neurons).
    non_local_no_swr_modulation : NDArray
        Non-SWR replay modulation from replay-split model, controlling
        for replay-coincident ripples only.
        Shape (n_lag_times, n_neurons).
    matched_control_modulation : NDArray
        Matched non-event baseline modulation from the engagement model.
        Shape (n_lag_times, n_neurons).
    cohort_info : dict
        How the population was cut and which event definition it was cut on:
        ``stop_cohort``, the bin counts (``n_bins_total``, ``n_bins_immobile``,
        ``n_bins_fit``), the event counts before and after the restriction, and
        the canonical-event provenance (``is_swr_definition``,
        ``immobility_inequality``, ``canonical_event_table_path`` /
        ``_sha256``, ``swr_interval_source``, ``swr_events_path`` /
        ``_sha256``, ``data_over_time_path`` / ``_sha256``), plus the matched
        non-event control's draw
        (``matched_control_seed``, ``n_matched_control``,
        ``n_matched_control_unplaced`` and the two causes that total,
        ``n_matched_control_onset_outside_fitted_bout`` and
        ``n_matched_control_bout_saturated``). Written to the run's provenance
        sidecar so an artifact states both the population and the events it
        was fit on. Also records how the models were fit: ``fitter``,
        ``fitter_tol`` and, for the batched fitter, ``jax_backend`` and
        ``jax_version``.

    Raises
    ------
    ValueError
        If ``non_local_threshold`` differs from the threshold the canonical
        events were cut at; it cannot be re-cut at run time.
    UnconfiguredDeviceError
        With ``fitter="jax"``, if no JAX device was chosen first. Call
        :func:`~continuum_swr_replay.analysis.pfc.gpu.configure_jax_gpu` (the
        runner does) or pin ``JAX_PLATFORMS=cpu``; otherwise JAX would take
        every card on a shared host.

    Notes
    -----
    Non-local events and their SWR classification are the canonical ones: the
    rows of ``Processed-Data/replay_content/event_table.parquet`` for this
    session, and that table's ``is_swr`` column (strict positive-duration
    overlap with the SWR source's speed-gated ``SWR_event`` labels). Figure 4 therefore
    describes the same ``event_id``s as the content and rate families.

    Immobility is ``head_speed`` strictly below
    :data:`~continuum_swr_replay.analysis.constants.DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC`,
    on the fit mask and on the SWR speed gate alike: the one threshold the
    canonical events and the SWRs behind ``is_swr`` were speed-gated at.

    The matched non-event control is drawn here rather than by the caller so
    every fit on this path carries one, with the seed
    :func:`matched_control_seed` derives from the session key -- a refit of one
    session redraws the batch's control population exactly.
    """
    _check_session_arguments(fitter, penalty_sigma, non_local_threshold)

    logger = logging.getLogger(
        format_epoch_identifier(nwb_file_name, epoch_name, brain_area)
    )

    results, data, position_info, alignment_counts = _load_aligned_session(
        nwb_file_name, epoch_name, logger
    )

    position_time = position_info.index
    time = position_time.to_numpy(dtype=float)

    # Fit with wider lag window (±250ms) so spline boundary artifacts
    # fall outside the reported window (±200ms). The returned modulation
    # arrays are trimmed to the reporting window after fitting.
    fit_lags = symmetric_lags(DEFAULT_BASIS_HALF_WINDOW_S, sampling_frequency)

    data, excluded, is_immobile, fit_mask, bin_counts = _session_fit_population(
        data,
        nwb_file_name,
        epoch_name,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        logger=logger,
    )

    (
        event_times,
        event_counts,
        event_provenance,
        artifact_dropped,
        control_seed,
        control_draw,
    ) = _session_events(
        data,
        results,
        nwb_file_name,
        epoch_name,
        time=time,
        fit_mask=fit_mask,
        fit_lags=fit_lags,
        sampling_frequency=sampling_frequency,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        non_local_threshold=non_local_threshold,
        logger=logger,
        event_set=event_set,
    )
    n_matched_control = int(len(control_draw.times))
    cohort_info = {
        **bin_counts,
        **event_counts,
        **event_provenance,
        **fitter_provenance(fitter),
        # Filled in after the fits: how many of the shipped fits ran to
        # ``maxiter`` without settling. ``None`` when no neuron was fit at all,
        # or on the statsmodels path, which never reaches the batched fitter.
        "n_model_fits": None,
        "n_nonconverged_model_fits": None,
        "n_neurons_with_nonconverged_model": None,
        "matched_control_seed": control_seed,
        "n_matched_control": n_matched_control,
        "n_matched_control_unplaced": control_draw.n_unplaced,
        "n_matched_control_onset_outside_fitted_bout": (
            control_draw.n_onset_outside_fitted_bout
        ),
        "n_matched_control_bout_saturated": control_draw.n_bout_saturated,
        "alignment": alignment_counts,
    }
    _log_session_events(event_counts, cohort_info, control_seed, logger)

    logger.info("Building design matrices...")
    immobility_design_matrices, spline_basis, lag_time = _make_design_matrices(
        event_times,
        data["position_info"],
        sampling_frequency=sampling_frequency,
        lags=fit_lags,
    )

    if brain_area not in data["spike_times"] or not data["spike_times"][brain_area]:
        logger.warning("No %s neurons found for this session", brain_area)
        empty = np.empty((0, 0))
        return pd.DataFrame(), empty, empty, empty, empty, empty, cohort_info

    spike_times = data["spike_times"][brain_area]
    # The JAX fit, held-out scorer and serial per-neuron summaries all consume
    # the same binned response. Bin it once instead of repeating the histogram
    # pass up to three times over every unit.
    session_counts = None
    if fitter == "jax" or scoring_out is not None:
        session_counts = np.column_stack(
            [get_spikecount_per_time_bin(neuron, time) for neuron in spike_times]
        ).astype(float)

    if scoring_out is not None:
        # Handed over, not rebuilt: an out-of-fold score has to be taken on the
        # bins, events and designs the shipped fit used, and reassembling them
        # from the same functions would cost a second load of the session and
        # could drift from what was fit.
        scoring_out.update(
            {
                "fit_mask": fit_mask,
                "counts": session_counts,
                "designs": immobility_design_matrices,
                "spline_basis": spline_basis,
                "lag_time": lag_time,
                "lags": fit_lags,
                "event_times": event_times,
                "artifact_dropped": artifact_dropped,
            }
        )

    result, fit_counts, n_neurons = _fit_session_neurons(
        spike_times,
        data,
        time=time,
        is_immobile=is_immobile,
        fit_mask=fit_mask,
        excluded=excluded,
        immobility_design_matrices=immobility_design_matrices,
        spline_basis=spline_basis,
        lag_time=lag_time,
        brain_area=brain_area,
        fitter=fitter,
        penalty_sigma=penalty_sigma,
        run_parallel=run_parallel,
        session_counts=session_counts,
        fit_diagnostics_out=fit_diagnostics_out,
        logger=logger,
    )
    cohort_info.update(fit_counts)
    if fit_counts["n_nonconverged_model_fits"]:
        logger.info(
            "%d of %d shipped model fits ran to maxiter without converging, "
            "over %d of %d neurons. Each is non-estimable for its neuron: "
            "every statistic that reads it is NaN, and this count is the only "
            "record of what was dropped.",
            fit_counts["n_nonconverged_model_fits"],
            fit_counts["n_model_fits"],
            fit_counts["n_neurons_with_nonconverged_model"],
            n_neurons,
        )

    (
        neuron_stats,
        swr_modulation,
        no_swr_modulation,
        non_local_swr_modulation,
        non_local_no_swr_modulation,
        matched_control_modulation,
    ) = _stack_session_outputs(result, lag_time)

    neuron_stats["nwb_file_name"] = nwb_file_name
    neuron_stats["epoch_name"] = epoch_name
    neuron_stats["brain_area"] = brain_area
    neuron_stats["stop_cohort"] = stop_cohort

    logger.info("Session complete: %d neurons analyzed", len(neuron_stats))

    return (
        neuron_stats,
        swr_modulation,
        no_swr_modulation,
        non_local_swr_modulation,
        non_local_no_swr_modulation,
        matched_control_modulation,
        cohort_info,
    )


def _check_session_arguments(
    fitter: str,
    penalty_sigma: float | None,
    non_local_threshold: float,
) -> None:
    """Refuse the :func:`run_session` arguments the canonical events cannot honour.

    Raises
    ------
    ValueError
        If ``penalty_sigma`` is given for a backend with no penalised path, or
        ``non_local_threshold`` differs from the one the canonical events were
        cut at.
    """
    _check_fitter(fitter)
    if penalty_sigma is not None and fitter != "jax":
        raise ValueError(
            f"penalty_sigma={penalty_sigma!r} needs fitter='jax'; the "
            f"{fitter!r} backend fits unpenalised models only"
        )
    if non_local_threshold != DEFAULT_NON_LOCAL_THRESHOLD:
        raise ValueError(
            f"non_local_threshold={non_local_threshold!r} is not the threshold "
            f"the canonical event table is cut at "
            f"({DEFAULT_NON_LOCAL_THRESHOLD}). Non-local events are read from "
            "that table, not re-detected, so a different threshold cannot be "
            "honoured here."
        )


def _load_aligned_session(
    nwb_file_name: str, epoch_name: str, logger: logging.Logger
) -> tuple[Any, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """The session's HPC decode and data, trimmed to their shared timestamps.

    Returns
    -------
    results : xarray.Dataset
        HPC decoding results, on the shared axis.
    data : dict
        :func:`~continuum_swr_replay.data_loaders.load_data`'s output, its
        ``position_info`` replaced by the aligned one.
    position_info : pd.DataFrame
        The aligned position frame (``data["position_info"]``).
    alignment_counts : dict
        What the alignment dropped, from ``align_results_to_position``.
    """
    # Deferred: opens Spyglass / DataJoint at import time (see the note at
    # the top of this module).
    from continuum_swr_replay.analysis.hippocampus import load_results
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )
    from continuum_swr_replay.data_loaders import load_data

    logger.info("Loading HPC decoding results...")
    results, _, _ = load_results(
        nwb_file_name,
        epoch_name,
        decoding_brain_area="HPC",
        load_detector=False,
    )

    logger.info("Loading session data...")
    data = load_data(nwb_file_name, epoch_name)

    # Decodes saved before get_position_info was trimmed carry a few more time
    # bins than load_data now returns. Intersect the two axes once, here, so
    # every downstream consumer (fit mask, event sets, design matrices, spike
    # binning) sees the same shared run.
    results, position_info, alignment_counts = align_results_to_position(
        results, data["position_info"]
    )
    data = {**data, "position_info": position_info}
    if alignment_counts["dropped_at"] != "none":
        logger.warning(
            "Trimmed decode/position axes to their shared timestamps: dropped "
            "%d results bins (%.3fs) and %d position rows (%.3fs) at the %s",
            alignment_counts["n_results_bins_dropped"],
            alignment_counts["dropped_seconds_results"],
            alignment_counts["n_position_rows_dropped"],
            alignment_counts["dropped_seconds_position"],
            alignment_counts["dropped_at"],
        )

    return results, data, position_info, alignment_counts


def _session_fit_population(
    data: dict[str, Any],
    nwb_file_name: str,
    epoch_name: str,
    cohort_stops: pd.DataFrame | None,
    stop_cohort: str,
    logger: logging.Logger,
) -> tuple[dict[str, Any], NDArray, pd.Series, pd.Series, dict[str, int]]:
    """The bins a session's models are fit on, and its excluded time.

    Returns
    -------
    data : dict
        ``data`` with theta phase set to NaN inside excluded time.
    excluded : NDArray, shape (n_intervals, 2)
        The session's excluded intervals, in seconds.
    is_immobile, fit_mask : pd.Series of bool, shape (n_time,)
        From :func:`build_session_fit_mask`.
    bin_counts : dict
        Its bin counts.
    """
    # Deferred: opens Spyglass / DataJoint at import time (see the note at
    # the top of this module).
    from continuum_swr_replay.analysis.events import analysed_time, swr_source

    # Excluded time (the artifact mask widened by 0.3 s) is not analysed. No
    # fitted bin's peri-event basis window reaches it, and no running bin's
    # theta phase is read from it.
    excluded = analysed_time.excluded_intervals(
        swr_source.swr_epoch_id(nwb_file_name, epoch_name)
    )
    is_immobile, fit_mask, bin_counts = build_session_fit_mask(
        data["position_info"],
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        immobility_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        excluded_intervals=excluded,
        basis_half_window_s=DEFAULT_BASIS_HALF_WINDOW_S,
    )
    if len(excluded):
        theta_phase = data["theta_phase"].copy()
        theta_phase.loc[
            interval_mask(theta_phase.index.to_numpy(dtype=float), excluded)
        ] = np.nan
        data = {**data, "theta_phase": theta_phase}
        logger.info(
            "Excluded time: %d intervals; %d time bins that would be fit lie "
            "within %.3f s of it and are not fit",
            len(excluded),
            bin_counts["n_bins_excluded"],
            DEFAULT_BASIS_HALF_WINDOW_S,
        )
    logger.info(
        "Fit bins: %d immobile, %d in cohort (%.3f of %d time bins)",
        bin_counts["n_bins_immobile"],
        bin_counts["n_bins_fit"],
        bin_counts["n_bins_fit"] / bin_counts["n_bins_total"],
        bin_counts["n_bins_total"],
    )

    return data, excluded, is_immobile, fit_mask, bin_counts


def _session_events(
    data: dict[str, Any],
    results: Any,
    nwb_file_name: str,
    epoch_name: str,
    time: NDArray,
    fit_mask: pd.Series,
    fit_lags: NDArray,
    sampling_frequency: float,
    cohort_stops: pd.DataFrame | None,
    stop_cohort: str,
    non_local_threshold: float,
    logger: logging.Logger,
    event_set: str = "canonical",
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    int,
    MatchedControlDraw,
]:
    """The session's event sets, artifact-dropped events and matched control.

    The matched non-event control is drawn against the event sets and the
    events the analysed-time rule dropped. With ``event_set`` set to
    ``"with_swr_without_replay"`` the SWRs that overlap no non-local event
    (:func:`swr_without_non_local`) are added as ``"SWR_no_non_local"``; they
    are a subset of ``"SWR"``, so the control is drawn exactly as without them.

    Returns
    -------
    event_times : dict of pd.DataFrame
        From :func:`build_session_event_sets`, with ``"matched_control"`` (and,
        for the extended set, ``"SWR_no_non_local"``) added.
    event_counts : dict
        Its counts, plus the artifact-dropped event counts.
    event_provenance : dict
        Paths and hashes of the canonical event inputs.
    artifact_dropped : pd.DataFrame
        From :func:`artifact_dropped_events`.
    control_seed : int
        From :func:`matched_control_seed`.
    control_draw : MatchedControlDraw
        From :func:`make_matched_control_times`.
    """
    # Deferred: opens Spyglass / DataJoint at import time (see the note at
    # the top of this module).
    from continuum_swr_replay.analysis.events import analysed_time
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        canonical_event_provenance,
        load_session_canonical_events,
        load_session_swr_intervals,
        session_data_over_time_path,
        session_swr_events_file,
    )

    logger.info("Loading canonical non-local events (stop cohort: %s)...", stop_cohort)
    canonical_events, event_table_sha256 = load_session_canonical_events(
        nwb_file_name, epoch_name
    )
    swr_intervals, dot_sha256 = load_session_swr_intervals(
        nwb_file_name,
        epoch_name,
        non_local_threshold=non_local_threshold,
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    swr_events_file = session_swr_events_file(nwb_file_name, epoch_name)
    event_provenance = canonical_event_provenance(
        event_table_path=table_path("event_table"),
        event_table_sha256=event_table_sha256,
        data_over_time_path=session_data_over_time_path(
            nwb_file_name, epoch_name, non_local_threshold=non_local_threshold
        ),
        data_over_time_sha256=dot_sha256,
        swr_events_path=swr_events_file["path"],
        swr_events_sha256=swr_events_file["sha256"],
    )

    event_times, event_counts = build_session_event_sets(
        data,
        results,
        canonical_events=canonical_events,
        swr_intervals=swr_intervals,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        fit_mask=fit_mask,
        basis_half_window_s=float(np.abs(fit_lags).max()) / sampling_frequency,
    )
    # The events the one analysed-time rule dropped (``analysed_time``): fitted
    # bins near one are fit with its regressor missing, which is counted, and
    # no matched control may land on one.
    artifact_dropped = artifact_dropped_events(
        analysed_time.read_artifact_event_drops(
            session_data_over_time_path(
                nwb_file_name, epoch_name, non_local_threshold=non_local_threshold
            ),
            non_local_threshold=non_local_threshold,
        )
    )
    event_counts["n_artifact_dropped_events"] = int(len(artifact_dropped))
    event_counts["n_artifact_dropped_overlapping_fit"] = count_events_covering_fit_bins(
        artifact_dropped["start_time"].to_numpy(dtype=float),
        time,
        np.asarray(fit_mask, dtype=bool),
        basis_half_window_s=DEFAULT_BASIS_HALF_WINDOW_S,
    )
    if event_set == "with_swr_without_replay":
        event_times[SWR_NO_NON_LOCAL], swr_counts = swr_without_non_local(
            event_times["SWR"], canonical_events, artifact_dropped
        )
        event_counts.update(swr_counts)
        logger.info(
            "SWRs without replay: %d of %d kept SWRs (%d with replay, %d unknown: "
            "overlapping a dropped window)",
            swr_counts["n_swr_no_non_local"],
            len(event_times["SWR"]),
            swr_counts["n_swr_with_non_local"],
            swr_counts["n_swr_no_non_local_unknown"],
        )
    elif event_set != "canonical":
        raise ValueError(f"unknown event_set {event_set!r}; one of {EVENT_SETS}")
    # The matched non-event control is drawn after the cohort restriction and
    # from the fit mask, so its windows sample exactly the bins the models are
    # fit on and avoid every event, the ones the restriction or the
    # analysed-time rule dropped included.
    control_seed = matched_control_seed(nwb_file_name, epoch_name)
    control_draw = make_matched_control_times(
        event_times,
        fit_mask,
        np.random.default_rng(control_seed),
        avoid=artifact_dropped,
    )
    event_times["matched_control"] = control_draw.times

    return (
        event_times,
        event_counts,
        event_provenance,
        artifact_dropped,
        control_seed,
        control_draw,
    )


def _log_session_events(
    event_counts: Mapping[str, Any],
    cohort_info: Mapping[str, Any],
    control_seed: int,
    logger: logging.Logger,
) -> None:
    """Log the session's event counts and its matched-control placement."""
    logger.info(
        "Events in cohort: %d/%d non-local (%d with SWR, %d without SWR), %d/%d SWRs",
        event_counts["n_nonlocal_after"],
        event_counts["n_nonlocal_before"],
        event_counts["n_nonlocal_swr"],
        event_counts["n_nonlocal_no_swr"],
        event_counts["n_ripple_after"],
        event_counts["n_ripple_before"],
    )
    logger.info(
        "Dropped events whose basis would have covered fitted bins: "
        "%s non-local, %s SWR",
        event_counts["n_nonlocal_dropped_overlapping_fit"],
        event_counts["n_ripple_dropped_overlapping_fit"],
    )
    logger.info(
        "Matched non-event controls: %d placed, %d unplaced "
        "(%d onset outside a fitted bout, %d bout saturated; seed %d)",
        cohort_info["n_matched_control"],
        cohort_info["n_matched_control_unplaced"],
        cohort_info["n_matched_control_onset_outside_fitted_bout"],
        cohort_info["n_matched_control_bout_saturated"],
        control_seed,
    )
    logger.info(
        "Canonical events outside the aligned position frame: %d (%d SWR intervals)",
        event_counts["n_events_outside_frame"],
        event_counts["n_swr_intervals_outside_frame"],
    )


def _fit_session_neurons(
    spike_times: list[NDArray],
    data: dict[str, Any],
    time: NDArray,
    is_immobile: pd.Series,
    fit_mask: pd.Series,
    excluded: NDArray,
    immobility_design_matrices: Mapping[str, NDArray],
    spline_basis: NDArray,
    lag_time: NDArray,
    brain_area: str,
    fitter: str,
    penalty_sigma: float | None,
    run_parallel: bool,
    session_counts: NDArray | None,
    fit_diagnostics_out: dict[str, Any] | None,
    logger: logging.Logger,
) -> tuple[Sequence[tuple], dict[str, int | None], int]:
    """Fit every neuron of the session: batched prefit, then ``process_neuron``.

    Returns
    -------
    result : list of tuple
        :func:`process_neuron`'s output for each neuron, in unit order.
    fit_counts : dict
        From :func:`shipped_fit_counts`; ``None`` values on the statsmodels
        path.
    n_neurons : int
    """
    n_neurons = len(spike_times)
    logger.info(
        "Fitting GLMs for %d %s neurons (fitter=%s, parallel=%s)...",
        n_neurons,
        brain_area,
        fitter,
        run_parallel,
    )

    prefitted_immobility: list[dict[str, Any]] | None = None
    design_ranks: dict[str, int] | None = None
    theta_design: tuple[NDArray, NDArray] | None = None
    # The statsmodels path fits inside ``process_neuron`` and never reaches the
    # batched fitter, so it has nothing to say here; the sidecar carries
    # ``None`` rather than a zero that would read as "nothing failed".
    fit_counts: dict[str, int | None] = {
        "n_model_fits": None,
        "n_nonconverged_model_fits": None,
        "n_neurons_with_nonconverged_model": None,
    }
    if fitter == "jax":
        # Every neuron shares the design; only the response column differs. Fit
        # the whole session against each design once, then hand each neuron its
        # own slice. The rank of each design is a property of the design, so it
        # is taken from the same batched fit rather than recomputed per neuron.
        (
            prefitted_immobility,
            design_ranks,
            theta_design,
            fit_counts,
        ) = _prefit_session(
            spike_times,
            time,
            is_immobile,
            fit_mask,
            immobility_design_matrices,
            data["theta_phase"],
            logger=logger,
            penalty_sigma=penalty_sigma,
            counts=session_counts,
        )

    # The descriptive rates leave excluded time out too (the fit mask already
    # does, with the basis window's reach).
    in_excluded_time = interval_mask(time, excluded)

    def _neuron_args(index: int) -> dict[str, Any]:
        prefit = None if prefitted_immobility is None else prefitted_immobility[index]
        return {
            "logger_name": logger.name,
            "fit_mask": fit_mask,
            "in_excluded_time": in_excluded_time,
            "fitter": fitter,
            "design_ranks": design_ranks,
            "theta_design": theta_design,
            "prefitted_immobility": prefit,
            "fit_diagnostics_out": (None if fit_diagnostics_out is None else {}),
            # Passing a full-length count vector to each Dask task is more
            # expensive than re-binning its sparse spikes. The production
            # runner is serial here; that path reuses the session matrix.
            "prebinned_spike_count": (
                None
                if run_parallel or session_counts is None
                else session_counts[:, index]
            ),
        }

    neuron_diagnostics: list[dict[str, Any]] = []
    if run_parallel:
        if fit_diagnostics_out is not None:
            # Dask returns copies, so an out-parameter cannot be filled through
            # a delayed call; the diagnostics are a serial-path option.
            raise ValueError("fit_diagnostics_out requires run_parallel=False")
        tasks = [
            dask.delayed(process_neuron)(
                neuron_spike_times,
                is_immobile,
                data["theta_phase"],
                time,
                immobility_design_matrices,
                spline_basis,
                lag_time,
                **_neuron_args(i),
            )
            for i, neuron_spike_times in enumerate(spike_times)
        ]

        result = dask.compute(*tasks)
    else:
        result = []
        for i, neuron_spike_times in enumerate(spike_times):
            logger.info("  Neuron %d/%d...", i + 1, n_neurons)
            kwargs = _neuron_args(i)
            neuron_diagnostics.append(kwargs["fit_diagnostics_out"])
            result.append(
                process_neuron(
                    neuron_spike_times,
                    is_immobile,
                    data["theta_phase"],
                    time,
                    immobility_design_matrices,
                    spline_basis,
                    lag_time,
                    **kwargs,
                )
            )
    if fit_diagnostics_out is not None:
        fit_diagnostics_out.update(
            _stack_fit_diagnostics(neuron_diagnostics, immobility_design_matrices)
        )

    return result, fit_counts, n_neurons


def _stack_session_outputs(
    result: Sequence[tuple],
    lag_time: NDArray,
) -> tuple[pd.DataFrame, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Stack the per-neuron outputs and trim the curves to the report window.

    Parameters
    ----------
    result : sequence of tuple
        Per neuron, ``process_neuron``'s statistics and its five curves, each
        curve ``(n_lags,)`` over the fit window.
    lag_time : NDArray, shape (n_lags,)
        Lag of each fit-window row, in seconds.

    Returns
    -------
    neuron_stats : pd.DataFrame
        One row per neuron.
    swr_modulation, no_swr_modulation, non_local_swr_modulation, \
non_local_no_swr_modulation, matched_control_modulation : NDArray
        Each ``(n_report_lags, n_neurons)``.
    """
    (
        neuron_stats_list,
        swr_modulation_list,
        no_swr_modulation_list,
        non_local_swr_modulation_list,
        non_local_no_swr_modulation_list,
        matched_control_modulation_list,
    ) = zip(*result, strict=True)

    neuron_stats = pd.DataFrame(list(neuron_stats_list))
    swr_modulation = np.stack(swr_modulation_list, axis=1)
    no_swr_modulation = np.stack(no_swr_modulation_list, axis=1)
    non_local_swr_modulation = np.stack(non_local_swr_modulation_list, axis=1)
    non_local_no_swr_modulation = np.stack(non_local_no_swr_modulation_list, axis=1)
    matched_control_modulation = np.stack(matched_control_modulation_list, axis=1)

    # Trim modulation arrays from fit window (±250ms) to report window (±200ms)
    # to remove spline boundary artifacts at the edges
    report_mask = in_report_window(lag_time)
    swr_modulation = swr_modulation[report_mask]
    no_swr_modulation = no_swr_modulation[report_mask]
    non_local_swr_modulation = non_local_swr_modulation[report_mask]
    non_local_no_swr_modulation = non_local_no_swr_modulation[report_mask]
    matched_control_modulation = matched_control_modulation[report_mask]

    return (
        neuron_stats,
        swr_modulation,
        no_swr_modulation,
        non_local_swr_modulation,
        non_local_no_swr_modulation,
        matched_control_modulation,
    )
