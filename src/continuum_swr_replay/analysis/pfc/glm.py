"""Generalized Linear Model analysis for PFC neural activity.

Fits Poisson GLMs relating single-unit spiking activity to hippocampal
replay events (SWR-coincident and non-SWR) during immobility, and tests
theta phase modulation during running.
"""

import logging
import warnings
from typing import Any

import dask
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from numpy.typing import NDArray
from patsy import dmatrix

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
        Time bin centers in seconds (e.g., from position_info.index).
        Spike ``s`` is assigned to bin ``i`` where
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
    is_immobile: pd.Series,
    design_matrices: dict[str, NDArray],
    neuron_spike_count: NDArray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit Poisson GLMs for each design matrix during immobility.

    Parameters
    ----------
    is_immobile : pd.Series, shape (n_time,)
        Boolean mask for immobile time bins.
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
            neuron_spike_count[is_immobile],
            design_matrix[is_immobile],
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
) -> tuple[dict[str, Any], NDArray, NDArray, NDArray, NDArray]:
    """Fit GLM models for a single neuron and compute modulation statistics.

    Fits 7 Poisson GLM models during immobility to test how the neuron's
    firing rate relates to SWR and non-local (replay) events. Models are
    organized into two decomposition strategies:

    **SWR-controlled decomposition** — separates the generic ripple response
    from replay-specific modulation:

    - ``constant``: intercept only (baseline)
    - ``SWR``: all detected ripples
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

    Returns
    -------
    neuron_stats : dict[str, Any]
        Per-neuron statistics including:

        - ``run_avg_firing_rate``, ``immobility_avg_firing_rate``: Hz
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
        for all detected ripples. Shape (n_lag_times,).
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

        # Get immobility SWR/non-local statistics
        neuron_stats["immobility_avg_firing_rate"] = (
            neuron_spike_count[is_immobile].mean() * sampling_frequency
        )

        results, _ = _fit_immobility_models(
            is_immobile, immobility_design_matrices, neuron_spike_count
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

        ## Likelihood ratio tests
        # SWR-controlled decomposition: does replay add to SWR response?
        # Replay-split decomposition: do SWR-replay and non-SWR-replay differ?
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
            ("SWR + non_local_no_SWR", "non_local"),
            # Replay-split: does splitting replay by SWR status help?
            ("non_local_SWR + non_local_no_SWR", "constant"),
            ("non_local_SWR + non_local_no_SWR", "non_local_SWR"),
            ("non_local_SWR + non_local_no_SWR", "non_local_no_SWR"),
            ("non_local_SWR + non_local_no_SWR", "non_local"),
        ]

        for model1, model2 in test_cases:
            test_name = f"{model1} vs. {model2}"
            neuron_stats[f"{test_name}: lr"], neuron_stats[f"{test_name}: pval"] = (
                _lr_test(results[model1], results[model2])
            )

        ## Effect sizes: fractional deviance for every LR test
        # frac_null_dev: (reduced - full) / constant deviance
        #   = fraction of total deviance uniquely explained by added regressors
        # frac_explained_dev: (reduced - full) / (constant - full)
        #   = fraction of full model's explained deviance due to added regressors
        for model1, model2 in test_cases:
            test_name = f"{model1} vs. {model2}"
            neuron_stats[f"{test_name}: frac_null_dev"] = get_frac_null_deviance(
                results[model1], results[model2], results["constant"]
            )
            neuron_stats[f"{test_name}: frac_explained_dev"] = (
                get_frac_explained_deviance(
                    results[model1], results[model2], results["constant"]
                )
            )

        ## AIC — SWR-controlled: what best explains immobility firing?
        swr_controlled_aics = {
            model_name: results[model_name].aic
            for model_name in [
                "constant",
                "SWR",
                "non_local",
                "SWR + non_local_no_SWR",
            ]
        }
        neuron_stats["min_aic_model"] = min(
            swr_controlled_aics, key=swr_controlled_aics.get
        )

        ## AIC — Replay-split: is replay one process or two?
        replay_split_aics = {
            model_name: results[model_name].aic
            for model_name in [
                "constant",
                "non_local",
                "non_local_SWR + non_local_no_SWR",
            ]
        }
        neuron_stats["min_aic_model_replay_split"] = min(
            replay_split_aics, key=replay_split_aics.get
        )

        ## Modulation time courses
        # Share fits with the LR-test / AIC path so neuron_stats and
        # the modulation arrays can never disagree due to divergent
        # IRLS convergence.
        n_basis_functions = spline_basis.shape[1]

        # SWR-controlled model: SWR and non-SWR replay modulation
        # controlling for all detected ripples
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


def run_session(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: float = 500,
    immobility_threshold: float = 4.0,
    non_local_threshold: float = 0.8,
    run_parallel: bool = True,
) -> tuple[pd.DataFrame, NDArray, NDArray, NDArray, NDArray]:
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
        for all detected ripples. Shape (n_lag_times, n_neurons).
    non_local_swr_modulation : NDArray
        SWR-coincident replay modulation from replay-split model.
        Shape (n_lag_times, n_neurons).
    non_local_no_swr_modulation : NDArray
        Non-SWR replay modulation from replay-split model, controlling
        for replay-coincident ripples only.
        Shape (n_lag_times, n_neurons).

    """
    # Deferred imports: these helpers open Spyglass / DataJoint at
    # import time. Keeping them function-local lets the rest of the
    # module (pure GLM helpers + design-matrix builders) stay
    # offline-importable for unit tests.
    from continuum_swr_replay.analysis.hippocampus import (
        get_confident_times,
        get_overlap_times,
        load_results,
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

    logger.info("Identifying non-local events...")
    non_local_prob = results.acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")

    non_local_times = get_confident_times(
        non_local_prob,
        data["position_info"].head_speed.to_numpy(),
        data["position_info"].index.to_numpy(),
        non_local_threshold,
    )
    non_local_times = non_local_times[
        non_local_times.max_speed < immobility_threshold
    ]

    non_local_swr_times, non_local_no_swr_times = get_overlap_times(
        non_local_times, data["ripple_times"], results
    )
    logger.info(
        "Found %d non-local events (%d with SWR, %d without SWR), %d total SWRs",
        len(non_local_times),
        len(non_local_swr_times),
        len(non_local_no_swr_times),
        len(data["ripple_times"]),
    )

    # Non-local events are filtered to immobility (line above) because
    # non-local classification during running is unreliable. SWR times are
    # intentionally NOT filtered: SWR detection already constrains to
    # low-movement periods, and the full ripple set is needed so the SWR
    # convolution kernel properly controls for ripple-related responses
    # in the SWR-controlled decomposition.
    event_times = {
        "non_local_no_SWR": non_local_no_swr_times,
        "non_local_SWR": non_local_swr_times,
        "non_local": non_local_times,
        "SWR": data["ripple_times"],
    }

    time = data["position_info"].index.to_numpy()
    is_immobile = data["position_info"].head_speed < immobility_threshold
    logger.info(
        "Immobility fraction: %.3f (%d/%d time bins)",
        is_immobile.mean(), is_immobile.sum(), len(is_immobile),
    )

    # Fit with wider lag window (±250ms) so spline boundary artifacts
    # fall outside the reported window (±200ms). The returned modulation
    # arrays are trimmed to the reporting window after fitting.
    fit_lags = np.arange(-125, 126, 1)
    report_lags = np.arange(-100, 101, 1)

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
        return pd.DataFrame(), empty, empty, empty, empty

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

    logger.info("Session complete: %d neurons analyzed", len(neuron_stats))

    return (
        neuron_stats,
        swr_modulation,
        no_swr_modulation,
        non_local_swr_modulation,
        non_local_no_swr_modulation,
    )
