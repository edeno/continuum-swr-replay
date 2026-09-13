"""Generalized Linear Model analysis for PFC neural activity.

Fits Poisson GLMs relating single-unit spiking activity to hippocampal
replay events (SWR-coincident and non-SWR) during immobility, and tests
theta phase modulation during running.
"""

import logging
import warnings
from collections.abc import Mapping
from typing import Any

import dask
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from numpy.typing import NDArray
from patsy import dmatrix

from continuum_swr_replay.analysis.replay.cohort_selection import (
    events_in_stops,
    stop_interval_mask,
)

# ``load_data`` and the hippocampus helpers below open a Spyglass /
# DataJoint connection at import time. We defer those imports into
# ``run_session`` (the only function that uses them) so that the pure
# helpers in this module — ``get_spikecount_per_time_bin``,
# ``process_neuron``, the design-matrix builders — stay offline-
# importable for unit tests on machines without VPN.

logger = logging.getLogger(__name__)


def get_spikecount_per_time_bin(spike_times: NDArray, time: NDArray) -> NDArray:
    """Bin spike times into time bins.

    Parameters
    ----------
    spike_times : NDArray, shape (n_spikes,)
        Spike times in seconds.
    time : NDArray, shape (n_time_bins,)
        Time bin starts (left edges) in seconds — i.e. each ``time[i]``
        is the start of bin ``i``, not its center. Matches the
        sample-bin-start convention used by the data loader's
        multiunit pipeline. Spike ``s`` is assigned to bin ``i`` where
        ``time[i] <= s < time[i+1]``; spikes at exactly ``time[-1]``
        fall in the last bin.

    Returns
    -------
    NDArray, shape (n_time_bins,)
        Spike counts per time bin.

    """
    spike_times = spike_times[
        np.logical_and(spike_times >= time[0], spike_times <= time[-1])
    ]
    # ``np.digitize(s, time[1:])`` produces indices ``0..N-1`` against
    # a ``minlength=N`` bincount, so all output bins are reachable.
    # The previous ``time[1:-1]`` form maxed out at index ``N-2``,
    # leaving the final bin unreachable: e.g. with
    # ``time=[0,1,2,3]``, spikes at 2.5 and 3.0 both landed in bin 2
    # and bin 3 stayed at zero.
    return np.bincount(
        np.digitize(spike_times, time[1:]),
        minlength=time.shape[0],
    )


def reindex_by(
    df: pd.Series | pd.DataFrame,
    index: pd.Index,
    tolerance: float | None = None,
) -> pd.Series | pd.DataFrame:
    """Reindex pandas object to new index using nearest neighbor.

    Uses pd.merge_asof for efficient O(n log n) nearest-neighbor matching.
    The original pd.reindex(method='nearest') can use O(n*m) memory when
    source and target indices have different sampling rates.

    Parameters
    ----------
    df : pd.Series | pd.DataFrame
        Data to reindex. Index must be sorted.
    index : pd.Index
        Target index to align to. Must be sorted.
    tolerance : float | None, optional
        Maximum distance for nearest neighbor match (in index units, typically
        seconds for time indices). If None, uses unlimited matching which may
        match to distant points. Recommended to set this to the sampling period
        (e.g., 0.002 seconds for 500 Hz data) to prevent spurious matches.

    Returns
    -------
    pd.Series | pd.DataFrame
        Reindexed data with values from nearest time point within tolerance.
        Values outside tolerance will be NaN.

    """
    if tolerance is None:
        warnings.warn(
            "reindex_by called with tolerance=None; nearest neighbor matching will "
            "match to arbitrarily distant points. Consider setting tolerance to the "
            "sampling period (e.g., 0.002 for 500 Hz data) to prevent spurious matches.",
            UserWarning,
            stacklevel=2,
        )

    target_df = pd.DataFrame(index=index)
    source_df = df.copy() if isinstance(df, pd.DataFrame) else df.to_frame()

    # Ensure index dtypes match for merge_asof (empty Series may have int index)
    if len(source_df) == 0:
        source_df.index = source_df.index.astype(target_df.index.dtype)

    merged = pd.merge_asof(
        target_df,
        source_df,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=tolerance,
    )

    if isinstance(df, pd.Series):
        return merged.iloc[:, 0].rename(df.name)
    return merged


def likelihood_ratio_test(
    deviance_full: float,
    deviance_restricted: float,
    degrees_of_freedom_full: float,
    degrees_of_freedom_restricted: float,
) -> tuple[float, float]:
    """Compare goodness of fit of nested models.

    Parameters
    ----------
    deviance_full : float
        Deviance of the more complicated model
    deviance_restricted : float
        Deviance of the simpler model
    degrees_of_freedom_full : float
        Degrees of freedom of the more complicated model
    degrees_of_freedom_restricted : float
        Degrees of freedom of the simpler model

    Returns
    -------
    likelihood_ratio : float
    p_value : float

    """
    degrees_of_freedom = degrees_of_freedom_restricted - degrees_of_freedom_full
    likelihood_ratio = deviance_restricted - deviance_full
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df=degrees_of_freedom)

    return likelihood_ratio, p_value


def _lr_test(full_model: Any, restricted_model: Any) -> tuple[float, float]:
    return likelihood_ratio_test(
        full_model.deviance,
        restricted_model.deviance,
        full_model.df_resid,
        restricted_model.df_resid,
    )


def test_theta_modulation(
    theta_phase: pd.Series,
    is_immobile: pd.Series,
    neuron_spike_count: NDArray,
    sampling_frequency: float = 500.0,
) -> tuple[float, float, float, float]:
    """Test theta phase modulation of neuron firing during running.

    Uses an unregularized GLM for the likelihood ratio test (correct for
    hypothesis testing), then a ridge-regularized fit (alpha=1e-20, L1_wt=0)
    for extracting modulation amplitude and preferred phase to avoid
    overfitting.

    Parameters
    ----------
    theta_phase : pd.Series, shape (n_lfp_samples,)
        Theta phase timeseries (radians). May have different sampling
        rate from ``is_immobile``; reindexed internally.
    is_immobile : pd.Series, shape (n_time,)
        Boolean series indicating immobility at each time bin.
    neuron_spike_count : NDArray, shape (n_time,)
        Binned spike counts aligned to ``is_immobile.index``.
    sampling_frequency : float
        Sampling frequency in Hz, used to set reindex tolerance.

    Returns
    -------
    likelihood_ratio : float
        Likelihood ratio statistic (theta vs constant).
    p_value : float
        P-value from chi-square test.
    modulation : float
        Theta modulation strength (sqrt of sum of squared cos/sin coefficients).
    preferred_phase : float
        Preferred theta phase in radians (-pi to pi).

    """
    reindexed_theta = reindex_by(
        theta_phase,
        is_immobile.index,
        tolerance=1.0 / sampling_frequency,
    )
    # Joint mask: running AND theta phase available
    valid = reindexed_theta.notna().to_numpy().ravel()
    joint_mask = (~is_immobile).to_numpy() & valid

    formula = "1 + np.cos(theta_phase) + np.sin(theta_phase)"
    design_matrix = dmatrix(
        formula,
        {"theta_phase": reindexed_theta[joint_mask]},
        return_type="dataframe",
    )

    constant_glm_results = sm.GLM(
        neuron_spike_count[joint_mask],
        np.ones_like(neuron_spike_count[joint_mask]),
        family=sm.families.Poisson(),
    ).fit()

    theta_glm_model = sm.GLM(
        neuron_spike_count[joint_mask],
        design_matrix.values,
        family=sm.families.Poisson(),
    )
    theta_glm_results = theta_glm_model.fit()

    likelihood_ratio, p_value = _lr_test(theta_glm_results, constant_glm_results)

    # Fit regularized model to avoid overfitting for modulation and preferred phase
    theta_glm_results = theta_glm_model.fit_regularized(alpha=1e-20, L1_wt=0.0)

    modulation = np.sqrt(
        theta_glm_results.params[1] ** 2 + theta_glm_results.params[2] ** 2
    )
    preferred_phase = np.arctan2(
        theta_glm_results.params[2], theta_glm_results.params[1]
    )

    return likelihood_ratio, p_value, modulation, preferred_phase


def create_lagged_signal(signal: NDArray, lags: NDArray | None = None) -> NDArray:
    """Create matrix of time-shifted copies of a signal for convolution GLM.

    Parameters
    ----------
    signal : NDArray, shape (n_time,) or (n_time, 1)
        Input signal (e.g., event indicator).
    lags : NDArray, shape (n_lags,), optional
        Lag indices in samples. Default: np.arange(-100, 101).

    Returns
    -------
    NDArray, shape (n_time, n_lags)
        Each column is the signal shifted by the corresponding lag.
        Out-of-bounds values are zero-filled.

    """
    if lags is None:
        lags = np.arange(-100, 101, 1)
    signal_df = pd.DataFrame(signal)
    return np.concatenate(
        [signal_df.shift(lag, fill_value=0).to_numpy() for lag in lags], axis=1
    )


def create_conv_basis(
    count: NDArray, lags: NDArray, knots: NDArray, sampling_frequency: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Create convolution spline basis from event indicator counts.

    Parameters
    ----------
    count : NDArray, shape (n_time,)
        Event indicator (0/1) time series.
    lags : NDArray, shape (n_lags,)
        Lag indices in samples.
    knots : NDArray, shape (n_knots,)
        Internal knot positions for B-spline basis (in seconds).
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    conv_spline_basis : NDArray, shape (n_time, n_basis)
        Lagged signal projected onto spline basis.
    spline_basis : NDArray, shape (n_lags, n_basis)
        B-spline basis functions evaluated at lag times.
    lag_time : NDArray, shape (n_lags,)
        Lag times in seconds.

    """
    lag_time = lags / sampling_frequency
    spline_basis = np.asarray(
        dmatrix(
            "bs(lag_time, knots=knots) - 1",
            {"lag_time": lag_time, "knots": knots},
        )
    )
    conv_spline_basis = create_lagged_signal(count, lags) @ spline_basis

    return conv_spline_basis, spline_basis, lag_time


def _make_conv_spline_basis(
    times: dict[str, pd.DataFrame],
    position_info: pd.DataFrame,
    sampling_frequency: float = 500,
    lags: NDArray | None = None,
) -> tuple[dict[str, NDArray], NDArray, NDArray]:
    if lags is None:
        lags = np.arange(-100, 101, 1)
    # 50ms knot spacing resolves sharp peri-SWR responses (~50ms peaks)
    knots = np.arange(
        lags[0] / sampling_frequency, lags[-1] / sampling_frequency, 0.050
    )[1:]
    if not times:
        raise ValueError("times dict must be non-empty")

    # Compute spline basis and lag_time once (they depend only on lags/knots,
    # not on per-event indicator signals)
    lag_time = lags / sampling_frequency
    spline_basis = np.asarray(
        dmatrix(
            "bs(lag_time, knots=knots) - 1",
            {"lag_time": lag_time, "knots": knots},
        )
    )

    conv_spline_basis: dict[str, NDArray] = {}
    for times_name, times_df in times.items():
        is_time = get_spikecount_per_time_bin(
            times_df.start_time, position_info.index.to_numpy()
        )
        lagged = create_lagged_signal(is_time, lags)
        conv_spline_basis[times_name] = lagged @ spline_basis

    return conv_spline_basis, spline_basis, lag_time


def _make_design_matrices(
    times: dict[str, pd.DataFrame],
    position_info: pd.DataFrame,
    sampling_frequency: float = 500,
    lags: NDArray | None = None,
) -> tuple[dict[str, NDArray], NDArray, NDArray]:
    """Build GLM design matrices for all model variants.

    Parameters
    ----------
    times : dict[str, pd.DataFrame]
        Event times keyed by name. Each DataFrame must have ``start_time``.
    position_info : pd.DataFrame
        Position data (index defines time bins).
    sampling_frequency : float
        Sampling frequency in Hz.
    lags : NDArray, shape (n_lags,), optional
        Lag indices in samples.

    Returns
    -------
    design_matrices : dict[str, NDArray]
        Design matrices keyed by model name. Each shape
        (n_time, 1 + n_basis * n_event_types_in_model).
    spline_basis : NDArray, shape (n_lags, n_basis)
        B-spline basis functions.
    lag_time : NDArray, shape (n_lags,)
        Lag times in seconds.

    """
    constant_design_matrix = np.ones((len(position_info), 1))

    conv_spline_basis, spline_basis, lag_time = _make_conv_spline_basis(
        times, position_info, sampling_frequency=sampling_frequency, lags=lags
    )

    non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    non_local_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_SWR"],
        ],
        axis=1,
    )

    swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["SWR"],
        ],
        axis=1,
    )

    non_local_swr_and_non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_no_SWR"],
            conv_spline_basis["non_local_SWR"],
        ],
        axis=1,
    )

    swr_and_non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["SWR"],
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    non_local_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local"],
        ],
        axis=1,
    )

    design_matrices = {
        "constant": constant_design_matrix,
        "non_local_no_SWR": non_local_no_swr_design_matrix,
        "non_local_SWR": non_local_swr_design_matrix,
        "SWR": swr_design_matrix,
        "non_local_SWR + non_local_no_SWR": non_local_swr_and_non_local_no_swr_design_matrix,
        "SWR + non_local_no_SWR": swr_and_non_local_no_swr_design_matrix,
        "non_local": non_local_design_matrix,
    }

    return design_matrices, spline_basis, lag_time


def _fit_immobility_models(
    fit_mask: pd.Series,
    design_matrices: dict[str, NDArray],
    neuron_spike_count: NDArray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit Poisson GLMs for each design matrix on the fitted time bins.

    Parameters
    ----------
    fit_mask : pd.Series, shape (n_time,)
        Boolean mask for the time bins the models are fit on: immobility, and
        (when a stop cohort is applied) immobility inside a cohort stop.
    design_matrices : dict[str, NDArray]
        Design matrices keyed by model name, each shape (n_time, n_features).
    neuron_spike_count : NDArray, shape (n_time,)
        Spike counts per time bin.

    Returns
    -------
    glm_results : dict[str, statsmodels GLMResults]
        Fitted model results keyed by model name.
    glm_models : dict[str, statsmodels GLM]
        Unfitted model objects keyed by model name.

    """
    glm_models = {
        model_name: sm.GLM(
            neuron_spike_count[fit_mask],
            design_matrix[fit_mask],
            family=sm.families.Poisson(),
        )
        for model_name, design_matrix in design_matrices.items()
    }
    glm_results = {model_name: model.fit() for model_name, model in glm_models.items()}

    return glm_results, glm_models


def get_frac_null_deviance(
    full_model: Any, reduced_model: Any, constant_model: Any
) -> float:
    """Compute fraction of total (null) deviance explained by additional terms.

    Parameters
    ----------
    full_model : statsmodels GLMResults
        More complex model (lower deviance).
    reduced_model : statsmodels GLMResults
        Simpler nested model (higher deviance).
    constant_model : statsmodels GLMResults
        Intercept-only model (highest deviance = total deviance).

    Returns
    -------
    float
        (reduced.deviance - full.deviance) / constant.deviance.
        Non-negative for properly nested models.

    """
    return float(
        (reduced_model.deviance - full_model.deviance) / constant_model.deviance
    )


def get_frac_explained_deviance(
    full_model: Any, reduced_model: Any, constant_model: Any
) -> float:
    """Compute fraction of model-explained deviance attributable to reduced terms.

    Parameters
    ----------
    full_model : statsmodels GLMResults
        Most complex model.
    reduced_model : statsmodels GLMResults
        Simpler nested model.
    constant_model : statsmodels GLMResults
        Intercept-only model.

    Returns
    -------
    float
        (reduced.deviance - full.deviance) / (constant.deviance - full.deviance).
        Returns 0.0 if the full model explains no deviance.

    """
    denominator = constant_model.deviance - full_model.deviance
    if abs(denominator) < 1e-10:
        return 0.0
    return float(
        (reduced_model.deviance - full_model.deviance) / denominator
    )


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
) -> tuple[dict[str, Any], NDArray, NDArray, NDArray, NDArray]:
    """Fit GLM models for a single neuron and compute modulation statistics.

    Fits 7 Poisson GLM models during immobility to test how the neuron's
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

    The analysis addresses a sequence of scientific questions:

    **Q1. Do PFC neurons respond to hippocampal events during immobility?**
    ``SWR vs constant``, ``non_local vs constant``

    **Q2. Are non-local events and SWRs redundant or independent?**
    ``SWR + non_local_no_SWR vs SWR`` (replay adds beyond ripples),
    ``SWR + non_local_no_SWR vs non_local_no_SWR`` (ripples add beyond
    replay). AIC: ``min_aic_model``.

    **Q3. Do PFC neurons respond to non-local events lacking a detected SWR?**
    ``non_local_no_SWR vs constant``. Modulation: ``no_swr_modulation``.

    **Q4. Is the non-SWR replay response robust to sub-threshold ripple
    confounds?**
    Compare ``no_swr_modulation`` (controlling for all SWRs) vs
    ``non_local_no_swr_modulation`` (controlling for replay-coincident
    SWRs only). ``non_local_SWR + non_local_no_SWR vs non_local_SWR``.

    **Q5. Do SWR-coincident and non-SWR replay drive different responses?**
    ``non_local_SWR + non_local_no_SWR vs non_local`` (does splitting
    improve fit?). AIC: ``min_aic_model_replay_split``. Compare
    ``non_local_swr_modulation`` vs ``non_local_no_swr_modulation`` shapes.

    **Q6. What is the relationship to theta modulation during running?**
    ``theta_mod``, ``theta_preferred_phase``, firing rates.

    Every likelihood ratio test has a paired effect size (``frac_null_dev``,
    ``frac_explained_dev``). AIC model selection is performed separately
    for each decomposition (only comparable models compete). Multiple
    comparison correction (e.g., FDR per test type across neurons) and
    population-level analyses are deferred to downstream figure code.

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

    Returns
    -------
    neuron_stats : dict[str, Any]
        Per-neuron statistics including:

        - ``run_avg_firing_rate``: Hz over every moving bin.
        - ``immobility_avg_firing_rate``: Hz over the bins the models were fit
          on -- with a stop cohort, the cohort's immobile bins, not every
          immobile bin. This is the rate the downstream ``min_firing_rate``
          inclusion gate screens on, so it describes the sample the estimates
          come from; its value therefore depends on the cohort the bundle was
          fit on (named in the bundle's provenance sidecar).
        - ``immobility_avg_firing_rate_all_immobile``: Hz over every immobile
          bin, cohort or not. Equal to ``immobility_avg_firing_rate`` exactly
          when no stop restriction was applied.
        - ``theta_mod``, ``theta_preferred_phase``: theta modulation during running
        - ``{model} vs. {baseline}: lr, pval``: likelihood ratio tests
        - ``{model} vs. {baseline}: frac_null_dev, frac_explained_dev``:
          fractional deviance measures
        - ``min_aic_model``: best SWR-controlled model by AIC
        - ``min_aic_model_replay_split``: best replay-split model by AIC
        - ``SWR_max_mod``, ``non_local_no_SWR_max_mod``: peak modulation
          from SWR-controlled model
        - ``non_local_SWR_max_mod``, ``non_local_no_SWR_split_max_mod``:
          peak modulation from replay-split model
        - ``*_max_mod_time``: corresponding peak lag times (s)

        Empty dict with only firing rates if GLM fitting fails.

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

    """
    _logger = logging.getLogger(logger_name) if logger_name else logger

    neuron_spike_count = get_spikecount_per_time_bin(neuron_spike_times, time)
    sampling_frequency = 1 / np.median(np.diff(time))
    if fit_mask is None:
        fit_mask = is_immobile

    try:
        neuron_stats: dict[str, Any] = {}

        # Get run theta statistics
        (
            neuron_stats["theta vs. constant: lr"],
            neuron_stats["theta vs. constant: pval"],
            neuron_stats["theta_mod"],
            neuron_stats["theta_preferred_phase"],
        ) = test_theta_modulation(
            theta_phase, is_immobile, neuron_spike_count,
            sampling_frequency=sampling_frequency,
        )
        neuron_stats["run_avg_firing_rate"] = (
            neuron_spike_count[~is_immobile].mean() * sampling_frequency
        )

        # Get immobility SWR/non-local statistics. This is the mean rate over
        # the bins the models are actually fit on, so the downstream
        # minimum-firing-rate inclusion gate screens the sample the estimates
        # come from rather than a wider one. The column name is the same across
        # cohorts, so the unrestricted rate is carried alongside it: the pair
        # makes a bundle's cohort readable from the table itself (the two are
        # equal exactly when no stop restriction was applied).
        neuron_stats["immobility_avg_firing_rate"] = (
            neuron_spike_count[fit_mask].mean() * sampling_frequency
        )
        neuron_stats["immobility_avg_firing_rate_all_immobile"] = (
            neuron_spike_count[is_immobile].mean() * sampling_frequency
        )

        results, _ = _fit_immobility_models(
            fit_mask, immobility_design_matrices, neuron_spike_count
        )

        # Check for silent non-convergence (NaN parameters from rank-deficient design)
        nan_models = [
            name for name, res in results.items()
            if np.any(np.isnan(res.params))
        ]
        if nan_models:
            raise ValueError(
                f"GLM IRLS produced NaN params for models: {nan_models}. "
                f"Immobility firing rate: "
                f"{neuron_stats['immobility_avg_firing_rate']:.3f} Hz."
            )

        # Detect non-estimable models. ``_make_conv_spline_basis``
        # builds spline regressors for every event class regardless
        # of whether that class has any events; an empty event class
        # leaves its spline columns all-zero, so the design matrix is
        # rank-deficient against the intercept. statsmodels still
        # returns finite (zero) parameters for those columns and
        # finite degenerate LR / fractional-deviance values, so the
        # NaN-params check above doesn't catch them. Mark every
        # rank-deficient design as non-estimable up front and route
        # the affected LR tests, AIC selections, and modulation
        # extractions to NaN below — so downstream consumers can
        # tell "absent event class" from "estimated zero effect".
        non_estimable_models = set()
        for name, design in immobility_design_matrices.items():
            X_im = design[fit_mask]
            if X_im.shape[0] == 0:
                non_estimable_models.add(name)
                continue
            if np.linalg.matrix_rank(X_im) < X_im.shape[1]:
                non_estimable_models.add(name)

        ## Likelihood ratio tests
        # Every (full, reduced) pair must be nested — the reduced
        # regressors must lie in the column span of the full design.
        # ``("SWR + non_local_no_SWR", "non_local")`` is intentionally
        # absent: it's only nested when there are no SWR-only events.
        test_cases = [
            # Individual event types vs baseline
            ("SWR", "constant"),
            ("non_local", "constant"),
            ("non_local_no_SWR", "constant"),
            ("non_local_SWR", "constant"),
            # SWR-controlled: does separating SWR and replay help?
            ("SWR + non_local_no_SWR", "constant"),
            ("SWR + non_local_no_SWR", "SWR"),
            ("SWR + non_local_no_SWR", "non_local_no_SWR"),
            # Replay-split: does splitting replay by SWR status help?
            ("non_local_SWR + non_local_no_SWR", "constant"),
            ("non_local_SWR + non_local_no_SWR", "non_local_SWR"),
            ("non_local_SWR + non_local_no_SWR", "non_local_no_SWR"),
            ("non_local_SWR + non_local_no_SWR", "non_local"),
        ]

        for model1, model2 in test_cases:
            test_name = f"{model1} vs. {model2}"
            if (
                model1 in non_estimable_models
                or model2 in non_estimable_models
            ):
                # Either side is rank-deficient (typically because an
                # event class has zero events in this session), so the
                # LR statistic / p-value are not meaningful. Mark the
                # comparison as unavailable rather than emitting a
                # finite degenerate value.
                neuron_stats[f"{test_name}: lr"] = np.nan
                neuron_stats[f"{test_name}: pval"] = np.nan
            else:
                neuron_stats[f"{test_name}: lr"], neuron_stats[
                    f"{test_name}: pval"
                ] = _lr_test(results[model1], results[model2])

        ## Effect sizes: fractional deviance for every LR test
        # frac_null_dev: (reduced - full) / constant deviance
        #   = fraction of total deviance uniquely explained by added regressors
        # frac_explained_dev: (reduced - full) / (constant - full)
        #   = fraction of full model's explained deviance due to added regressors
        for model1, model2 in test_cases:
            test_name = f"{model1} vs. {model2}"
            if (
                model1 in non_estimable_models
                or model2 in non_estimable_models
                or "constant" in non_estimable_models
            ):
                neuron_stats[f"{test_name}: frac_null_dev"] = np.nan
                neuron_stats[f"{test_name}: frac_explained_dev"] = np.nan
            else:
                neuron_stats[f"{test_name}: frac_null_dev"] = (
                    get_frac_null_deviance(
                        results[model1], results[model2], results["constant"]
                    )
                )
                neuron_stats[f"{test_name}: frac_explained_dev"] = (
                    get_frac_explained_deviance(
                        results[model1], results[model2], results["constant"]
                    )
                )

        ## AIC — SWR-controlled: what best explains immobility firing?
        # Skip rank-deficient models so a non-estimable design (e.g. a
        # session with zero ``non_local_no_SWR`` events) cannot be
        # selected as the "best" model just because statsmodels
        # returned a finite AIC for its degenerate fit.
        swr_controlled_candidates = [
            "constant",
            "SWR",
            "non_local",
            "SWR + non_local_no_SWR",
        ]
        swr_controlled_aics = {
            model_name: results[model_name].aic
            for model_name in swr_controlled_candidates
            if model_name not in non_estimable_models
        }
        neuron_stats["min_aic_model"] = (
            min(swr_controlled_aics, key=swr_controlled_aics.get)
            if swr_controlled_aics
            else None
        )

        ## AIC — Replay-split: is replay one process or two?
        replay_split_candidates = [
            "constant",
            "non_local",
            "non_local_SWR + non_local_no_SWR",
        ]
        replay_split_aics = {
            model_name: results[model_name].aic
            for model_name in replay_split_candidates
            if model_name not in non_estimable_models
        }
        neuron_stats["min_aic_model_replay_split"] = (
            min(replay_split_aics, key=replay_split_aics.get)
            if replay_split_aics
            else None
        )

        ## Modulation time courses
        # Share fits with the LR-test / AIC path so neuron_stats and
        # the modulation arrays can never disagree due to divergent
        # IRLS convergence. When the source model is non-estimable
        # (rank-deficient design — typically an absent event class),
        # the spline-by-coefficient product would silently emit a
        # zero curve from statsmodels' degenerate fit; route those
        # to NaN so consumers can tell "absent event class" from
        # "estimated zero modulation".
        n_basis_functions = spline_basis.shape[1]
        nan_modulation = np.full(spline_basis.shape[0], np.nan)

        # SWR-controlled model: SWR and non-SWR replay modulation
        # controlling for all whole-event-speed-eligible ripples
        if "SWR + non_local_no_SWR" in non_estimable_models:
            swr_modulation = nan_modulation
            no_swr_modulation = nan_modulation
            neuron_stats["SWR_max_mod"] = np.nan
            neuron_stats["SWR_max_mod_time"] = np.nan
            neuron_stats["non_local_no_SWR_max_mod"] = np.nan
            neuron_stats["non_local_no_SWR_max_mod_time"] = np.nan
        else:
            swr_ctrl_params = results["SWR + non_local_no_SWR"].params
            swr_modulation = (
                spline_basis @ swr_ctrl_params[1 : n_basis_functions + 1]
            )
            no_swr_modulation = (
                spline_basis @ swr_ctrl_params[n_basis_functions + 1 :]
            )

            swr_max_mod_ind = np.argmax(np.abs(swr_modulation))
            neuron_stats["SWR_max_mod"] = swr_modulation[swr_max_mod_ind]
            neuron_stats["SWR_max_mod_time"] = lag_time[swr_max_mod_ind]

            no_swr_max_mod_ind = np.argmax(np.abs(no_swr_modulation))
            neuron_stats["non_local_no_SWR_max_mod"] = (
                no_swr_modulation[no_swr_max_mod_ind]
            )
            neuron_stats["non_local_no_SWR_max_mod_time"] = (
                lag_time[no_swr_max_mod_ind]
            )

        # Replay-split model: SWR-replay and non-SWR replay modulation
        # controlling for replay-coincident ripples only
        if "non_local_SWR + non_local_no_SWR" in non_estimable_models:
            non_local_swr_modulation = nan_modulation
            non_local_no_swr_modulation = nan_modulation
            neuron_stats["non_local_SWR_max_mod"] = np.nan
            neuron_stats["non_local_SWR_max_mod_time"] = np.nan
            neuron_stats["non_local_no_SWR_split_max_mod"] = np.nan
            neuron_stats["non_local_no_SWR_split_max_mod_time"] = np.nan
        else:
            replay_split_params = results["non_local_SWR + non_local_no_SWR"].params
            non_local_swr_modulation = (
                spline_basis @ replay_split_params[1 : n_basis_functions + 1]
            )
            non_local_no_swr_modulation = (
                spline_basis @ replay_split_params[n_basis_functions + 1 :]
            )

            non_local_swr_max_ind = np.argmax(np.abs(non_local_swr_modulation))
            neuron_stats["non_local_SWR_max_mod"] = (
                non_local_swr_modulation[non_local_swr_max_ind]
            )
            neuron_stats["non_local_SWR_max_mod_time"] = lag_time[non_local_swr_max_ind]

            non_local_no_swr_max_ind = np.argmax(np.abs(non_local_no_swr_modulation))
            neuron_stats["non_local_no_SWR_split_max_mod"] = (
                non_local_no_swr_modulation[non_local_no_swr_max_ind]
            )
            neuron_stats["non_local_no_SWR_split_max_mod_time"] = (
                lag_time[non_local_no_swr_max_ind]
            )

    except (ValueError, np.linalg.LinAlgError):
        _logger.warning(
            "GLM fitting failed for neuron; falling back to firing rates only",
            exc_info=True,
        )
        neuron_stats = {}
        neuron_stats["run_avg_firing_rate"] = (
            neuron_spike_count[~is_immobile].mean() * sampling_frequency
        )
        neuron_stats["immobility_avg_firing_rate"] = (
            neuron_spike_count[fit_mask].mean() * sampling_frequency
        )
        neuron_stats["immobility_avg_firing_rate_all_immobile"] = (
            neuron_spike_count[is_immobile].mean() * sampling_frequency
        )
        nan_modulation = np.full(spline_basis.shape[0], np.nan)
        swr_modulation = nan_modulation
        no_swr_modulation = nan_modulation
        non_local_swr_modulation = nan_modulation
        non_local_no_swr_modulation = nan_modulation

    return (
        neuron_stats,
        swr_modulation,
        no_swr_modulation,
        non_local_swr_modulation,
        non_local_no_swr_modulation,
    )


def build_session_fit_mask(
    position_info: pd.DataFrame,
    *,
    cohort_stops: pd.DataFrame | None = None,
    stop_cohort: str = "at_well",
    immobility_threshold: float = 4.0,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Bins the immobility GLMs are fit on, for one session.

    Two restrictions compose. Immobility is the long-standing one. The stop
    cohort is the new one: it keeps only the time the animal spends stopped at a
    well, which is the population the hippocampal rate and content ladders are
    read on, so a PFC claim and a replay claim describe the same stops.

    Parameters
    ----------
    position_info : pandas.DataFrame
        Time-indexed position table carrying ``head_speed``. The index is
        absolute seconds, the same clock the stop table's bounds are on.
    cohort_stops : pandas.DataFrame, optional
        Stops kept by
        :func:`~continuum_swr_replay.analysis.pfc.cohort.load_session_stop_cohort`.
        Required unless ``stop_cohort`` is ``"all_stops"``.
    stop_cohort : str
        Cohort name, recorded in the returned counts. ``"all_stops"`` means *no
        stop restriction at all* -- every immobile bin in the epoch is fit,
        reproducing the pre-cohort mask exactly.
    immobility_threshold : float
        Speed threshold for immobility (cm/s).

    Returns
    -------
    is_immobile : pandas.Series of bool, shape (n_time,)
        Immobility alone; still the definition of the *running* comparison.
    fit_mask : pandas.Series of bool, shape (n_time,)
        ``is_immobile & in_cohort_stop``, or ``is_immobile`` for ``all_stops``.
    counts : dict
        ``stop_cohort``, ``n_bins_total``, ``n_bins_immobile``, ``n_bins_fit``.

    Raises
    ------
    ValueError
        If a stop cohort is requested without ``cohort_stops``.
    """
    is_immobile = position_info.head_speed < immobility_threshold
    if stop_cohort == "all_stops":
        fit_mask = is_immobile
    else:
        if cohort_stops is None:
            raise ValueError(
                f"stop_cohort={stop_cohort!r} needs cohort_stops; pass the stops "
                "load_session_stop_cohort returned, or use 'all_stops' for no "
                "stop restriction"
            )
        in_cohort = stop_interval_mask(
            position_info.index.to_numpy(dtype=float), cohort_stops
        )
        fit_mask = is_immobile & pd.Series(in_cohort, index=position_info.index)
    counts = {
        "stop_cohort": stop_cohort,
        "n_bins_total": int(len(is_immobile)),
        "n_bins_immobile": int(is_immobile.sum()),
        "n_bins_fit": int(fit_mask.sum()),
    }
    return is_immobile, fit_mask, counts


#: Half-width of the peri-event spline basis, in seconds. The fit window is
#: +/-125 samples at 500 Hz; a dropped event within this of a fitted bin is an
#: event whose basis would have covered bins the model does fit.
DEFAULT_BASIS_HALF_WINDOW_S = 0.25


def count_events_covering_fit_bins(
    onsets: NDArray,
    times: NDArray,
    fit_mask: NDArray,
    *,
    basis_half_window_s: float = DEFAULT_BASIS_HALF_WINDOW_S,
) -> int:
    """How many of ``onsets`` have a fitted bin within the basis half-window.

    The cohort restriction drops events by onset, but the regressor basis spans
    +/-250 ms around each event, so an event dropped just before a stop's
    leading edge would have covered bins that *are* fit. Those bins are then fit
    with that event's contribution set to zero. This counts the events for which
    that is true, so a run can state the size of the effect rather than assume
    it away.

    Parameters
    ----------
    onsets : numpy.ndarray, shape (n_events,)
        Onset times of the dropped events, in seconds.
    times : numpy.ndarray, shape (n_time,)
        Sample times, ascending, on the same clock.
    fit_mask : numpy.ndarray of bool, shape (n_time,)
        The bins the models are fit on.
    basis_half_window_s : float
        Half-width of the peri-event basis, in seconds.

    Returns
    -------
    int
        Number of onsets with at least one fitted bin in
        ``[onset - basis_half_window_s, onset + basis_half_window_s]``.
    """
    onsets = np.asarray(onsets, dtype=float)
    if onsets.size == 0:
        return 0
    # Prefix sums make the per-event window test a pair of searchsorted lookups
    # rather than a scan over the time axis.
    fitted_before = np.concatenate(
        [[0], np.cumsum(np.asarray(fit_mask, dtype=bool), dtype=np.int64)]
    )
    first = np.searchsorted(times, onsets - basis_half_window_s, side="left")
    last = np.searchsorted(times, onsets + basis_half_window_s, side="right")
    return int((fitted_before[last] - fitted_before[first] > 0).sum())


def _basis_coverage_counts(
    dropped_non_local_onsets: NDArray,
    dropped_ripple_onsets: NDArray,
    *,
    kept_non_local: int,
    kept_ripple: int,
    time: NDArray,
    fit_mask: pd.Series | None,
    basis_half_window_s: float,
) -> dict[str, Any]:
    """The leading-edge coverage diagnostic, as four report fields.

    Named per event class and expressed both as a count and as a fraction of the
    events that were kept, because the count alone is not interpretable without
    the cohort's size.
    """
    if fit_mask is None:
        return {
            "n_nonlocal_dropped_overlapping_fit": None,
            "n_ripple_dropped_overlapping_fit": None,
            "n_nonlocal_dropped_overlapping_fit_frac_of_kept": None,
            "n_ripple_dropped_overlapping_fit_frac_of_kept": None,
        }
    mask = np.asarray(fit_mask, dtype=bool)
    n_non_local = count_events_covering_fit_bins(
        dropped_non_local_onsets,
        time,
        mask,
        basis_half_window_s=basis_half_window_s,
    )
    n_ripple = count_events_covering_fit_bins(
        dropped_ripple_onsets, time, mask, basis_half_window_s=basis_half_window_s
    )
    return {
        "n_nonlocal_dropped_overlapping_fit": n_non_local,
        "n_ripple_dropped_overlapping_fit": n_ripple,
        "n_nonlocal_dropped_overlapping_fit_frac_of_kept": (
            n_non_local / kept_non_local if kept_non_local else None
        ),
        "n_ripple_dropped_overlapping_fit_frac_of_kept": (
            n_ripple / kept_ripple if kept_ripple else None
        ),
    }


def build_session_event_sets(
    data: Mapping[str, Any],
    results: Any,
    *,
    cohort_stops: pd.DataFrame | None = None,
    stop_cohort: str = "at_well",
    immobility_threshold: float = 4.0,
    non_local_threshold: float = 0.8,
    fit_mask: pd.Series | None = None,
    basis_half_window_s: float = DEFAULT_BASIS_HALF_WINDOW_S,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build the four event sets the PFC design matrices are keyed on.

    Non-local events are the confident, whole-event-immobile windows of the HPC
    decode; ripples are the whole-event-immobile detected SWRs. Both are then
    restricted to the stop cohort by event onset, BEFORE the SWR overlap split,
    so ``non_local_SWR`` and ``non_local_no_SWR`` partition exactly the events
    that are kept -- restricting after the split would let a dropped ripple move
    a kept replay event from the SWR class into the non-SWR one.

    Parameters
    ----------
    data : Mapping
        Session data from ``data_loaders.load_data``: ``position_info`` and
        ``ripple_times``.
    results : xarray.Dataset
        HPC decoding results carrying ``acausal_state_probabilities``.
    cohort_stops : pandas.DataFrame, optional
        Stops kept for this session; required unless ``stop_cohort`` is
        ``"all_stops"``.
    stop_cohort : str
        Cohort name; ``"all_stops"`` applies no restriction.
    immobility_threshold : float
        Speed threshold for whole-event immobility (cm/s).
    non_local_threshold : float
        Probability threshold for a confident non-local event.
    fit_mask : pandas.Series of bool, optional
        The bins the models will be fit on. Used only to measure how many
        dropped events would have had basis coverage over fitted bins; when
        omitted those counts are ``None``.
    basis_half_window_s : float
        Half-width of the peri-event basis used for that measurement.

    Returns
    -------
    event_times : dict of str to pandas.DataFrame
        ``non_local_no_SWR``, ``non_local_SWR``, ``non_local``, ``SWR``.
    counts : dict
        ``stop_cohort`` and the event counts before and after the restriction
        (``n_nonlocal_before``/``n_nonlocal_after``,
        ``n_ripple_before``/``n_ripple_after``), the split
        (``n_nonlocal_swr``, ``n_nonlocal_no_swr``), and the leading-edge
        coverage diagnostic ``n_nonlocal_dropped_overlapping_fit`` /
        ``n_ripple_dropped_overlapping_fit`` with their fractions of the kept
        counts (``*_frac_of_kept``).

    Raises
    ------
    ValueError
        If a stop cohort is requested without ``cohort_stops``.
    """
    # Deferred imports: these helpers open Spyglass / DataJoint at import time.
    from continuum_swr_replay.analysis.hippocampus import (
        get_confident_times,
        get_overlap_times,
    )
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )
    from continuum_swr_replay.analysis.hippocampus.event_immobility import (
        build_speed_eligible_event_cohort,
    )

    # Idempotent when the caller already aligned the pair (run_session does):
    # the shared run is then the whole of both axes and this is the identity.
    results, position_info, _ = align_results_to_position(
        results, data["position_info"]
    )
    position_time = position_info.index
    speed = position_info.head_speed.to_numpy()
    non_local_prob = results.acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    non_local_prob = non_local_prob.assign_coords(
        time=position_time.to_numpy(dtype=float)
    )

    detected_non_local_times = get_confident_times(
        non_local_prob,
        speed,
        position_time.to_numpy(dtype=float),
        non_local_threshold,
    )
    non_local_times, _ = build_speed_eligible_event_cohort(
        detected_non_local_times,
        speed=speed,
        time=position_time,
        event_col="non_local_event",
        speed_threshold=immobility_threshold,
    )
    ripple_times, _ = build_speed_eligible_event_cohort(
        data["ripple_times"],
        speed=speed,
        time=position_time,
        event_col="SWR_event",
        speed_threshold=immobility_threshold,
    )

    counts: dict[str, Any] = {
        "stop_cohort": stop_cohort,
        "n_nonlocal_before": int(len(non_local_times)),
        "n_ripple_before": int(len(ripple_times)),
    }
    dropped_non_local_onsets = np.empty(0, dtype=float)
    dropped_ripple_onsets = np.empty(0, dtype=float)
    if stop_cohort != "all_stops":
        if cohort_stops is None:
            raise ValueError(
                f"stop_cohort={stop_cohort!r} needs cohort_stops; pass the stops "
                "load_session_stop_cohort returned, or use 'all_stops' for no "
                "stop restriction"
            )
        keep_non_local = events_in_stops(
            non_local_times.start_time.to_numpy(dtype=float), cohort_stops
        )
        keep_ripple = events_in_stops(
            ripple_times.start_time.to_numpy(dtype=float), cohort_stops
        )
        dropped_non_local_onsets = non_local_times.start_time.to_numpy(dtype=float)[
            ~keep_non_local
        ]
        dropped_ripple_onsets = ripple_times.start_time.to_numpy(dtype=float)[
            ~keep_ripple
        ]
        non_local_times = non_local_times.loc[keep_non_local]
        ripple_times = ripple_times.loc[keep_ripple]
    counts["n_nonlocal_after"] = int(len(non_local_times))
    counts["n_ripple_after"] = int(len(ripple_times))
    counts.update(
        _basis_coverage_counts(
            dropped_non_local_onsets,
            dropped_ripple_onsets,
            kept_non_local=counts["n_nonlocal_after"],
            kept_ripple=counts["n_ripple_after"],
            time=position_time.to_numpy(dtype=float),
            fit_mask=fit_mask,
            basis_half_window_s=basis_half_window_s,
        )
    )

    non_local_swr_times, non_local_no_swr_times = get_overlap_times(
        non_local_times, ripple_times
    )
    counts["n_nonlocal_swr"] = int(len(non_local_swr_times))
    counts["n_nonlocal_no_swr"] = int(len(non_local_no_swr_times))

    # Both populations use the same whole-event immobility definition, so the
    # SWR-controlled decomposition cannot reintroduce a movement-spanning SWR
    # as a secondary event source.
    event_times = {
        "non_local_no_SWR": non_local_no_swr_times,
        "non_local_SWR": non_local_swr_times,
        "non_local": non_local_times,
        "SWR": ripple_times,
    }
    return event_times, counts


def run_session(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: float = 500,
    immobility_threshold: float = 4.0,
    non_local_threshold: float = 0.8,
    run_parallel: bool = True,
    stop_cohort: str = "at_well",
    cohort_stops: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, NDArray, NDArray, NDArray, NDArray, dict[str, Any]]:
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
    immobility_threshold : float
        Speed threshold for immobility classification (cm/s).
    non_local_threshold : float
        Probability threshold for classifying non-local events.
    run_parallel : bool
        Whether to use Dask for parallel neuron processing.
    stop_cohort : str
        Population to fit. ``"at_well"`` (canonical) keeps only immobility bins
        and events inside the well-anchored stops the hippocampal rate and
        content ladders are read on; ``"all_stops"`` applies no stop restriction
        at all, fitting every immobile bin and every speed-eligible event in the
        epoch. Under ``"all_stops"`` the fit mask, the event sets and every
        fitted number reproduce the pre-cohort pipeline bit-for-bit; what
        differs is the shape of the output -- the stats table gains the
        provenance columns ``stop_cohort`` and
        ``immobility_avg_firing_rate_all_immobile``, and this function returns a
        6-tuple rather than a 5-tuple.
    cohort_stops : pd.DataFrame, optional
        The session's kept stops from
        :func:`~continuum_swr_replay.analysis.pfc.cohort.load_session_stop_cohort`.
        Required unless ``stop_cohort`` is ``"all_stops"``.

    Returns
    -------
    neuron_stats : pd.DataFrame
        Per-neuron statistics. Shape (n_neurons, n_stats). Includes GLM
        comparison results, theta modulation, firing rates, and AIC model
        selection.
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
    cohort_info : dict
        How the population was cut: ``stop_cohort``, the bin counts
        (``n_bins_total``, ``n_bins_immobile``, ``n_bins_fit``) and the event
        counts before and after the restriction. Written to the run's provenance
        sidecar so an artifact states the population it was fit on.

    """
    # Deferred imports: these helpers open Spyglass / DataJoint at
    # import time. Keeping them function-local lets the rest of the
    # module (pure GLM helpers + design-matrix builders) stay
    # offline-importable for unit tests.
    from continuum_swr_replay.analysis.hippocampus import load_results
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )
    from continuum_swr_replay.data_loaders import load_data

    logger = logging.getLogger(
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
    )

    logger.info("Loading HPC decoding results...")
    results, _, _ = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC", load_detector=False,
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

    position_time = position_info.index
    time = position_time.to_numpy(dtype=float)

    # Fit with wider lag window (±250ms) so spline boundary artifacts
    # fall outside the reported window (±200ms). The returned modulation
    # arrays are trimmed to the reporting window after fitting.
    fit_lags = np.arange(-125, 126, 1)
    report_lags = np.arange(-100, 101, 1)

    is_immobile, fit_mask, bin_counts = build_session_fit_mask(
        data["position_info"],
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        immobility_threshold=immobility_threshold,
    )
    logger.info(
        "Fit bins: %d immobile, %d in cohort (%.3f of %d time bins)",
        bin_counts["n_bins_immobile"],
        bin_counts["n_bins_fit"],
        bin_counts["n_bins_fit"] / bin_counts["n_bins_total"],
        bin_counts["n_bins_total"],
    )

    logger.info("Identifying non-local events (stop cohort: %s)...", stop_cohort)
    event_times, event_counts = build_session_event_sets(
        data,
        results,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        immobility_threshold=immobility_threshold,
        non_local_threshold=non_local_threshold,
        fit_mask=fit_mask,
        basis_half_window_s=float(np.abs(fit_lags).max()) / sampling_frequency,
    )
    cohort_info = {**bin_counts, **event_counts, "alignment": alignment_counts}
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
        return pd.DataFrame(), empty, empty, empty, empty, cohort_info

    n_neurons = len(data["spike_times"][brain_area])
    logger.info(
        "Fitting GLMs for %d %s neurons (parallel=%s)...",
        n_neurons,
        brain_area,
        run_parallel,
    )

    if run_parallel:
        tasks = [
            dask.delayed(process_neuron)(
                neuron_spike_times,
                is_immobile,
                data["theta_phase"],
                time,
                immobility_design_matrices,
                spline_basis,
                lag_time,
                logger_name=logger.name,
                fit_mask=fit_mask,
            )
            for neuron_spike_times in data["spike_times"][brain_area]
        ]

        result = dask.compute(*tasks)
    else:
        result = []
        for i, neuron_spike_times in enumerate(data["spike_times"][brain_area]):
            logger.info("  Neuron %d/%d...", i + 1, n_neurons)
            result.append(
                process_neuron(
                    neuron_spike_times,
                    is_immobile,
                    data["theta_phase"],
                    time,
                    immobility_design_matrices,
                    spline_basis,
                    lag_time,
                    logger_name=logger.name,
                    fit_mask=fit_mask,
                )
            )

    (
        neuron_stats_list,
        swr_modulation_list,
        no_swr_modulation_list,
        non_local_swr_modulation_list,
        non_local_no_swr_modulation_list,
    ) = zip(*result, strict=True)

    neuron_stats = pd.DataFrame(list(neuron_stats_list))
    swr_modulation = np.stack(swr_modulation_list, axis=1)
    no_swr_modulation = np.stack(no_swr_modulation_list, axis=1)
    non_local_swr_modulation = np.stack(non_local_swr_modulation_list, axis=1)
    non_local_no_swr_modulation = np.stack(
        non_local_no_swr_modulation_list, axis=1
    )

    # Trim modulation arrays from fit window (±250ms) to report window (±200ms)
    # to remove spline boundary artifacts at the edges
    report_mask = np.isin(
        np.round(lag_time, 6), np.round(report_lags / sampling_frequency, 6)
    )
    swr_modulation = swr_modulation[report_mask]
    no_swr_modulation = no_swr_modulation[report_mask]
    non_local_swr_modulation = non_local_swr_modulation[report_mask]
    non_local_no_swr_modulation = non_local_no_swr_modulation[report_mask]

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
        cohort_info,
    )
