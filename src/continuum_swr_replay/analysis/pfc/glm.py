import warnings
from typing import Any

import dask
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from numpy.typing import NDArray
from patsy import dmatrix

from continuum_swr_replay.analysis.hippocampus import (
    get_confident_times,
    get_overlap_times,
    load_results,
)
from continuum_swr_replay.data_loaders import load_data


def get_spikecount_per_time_bin(spike_times: NDArray, time: NDArray) -> NDArray:
    """Bin spike times into time bins.

    Parameters
    ----------
    spike_times : NDArray
        Spike times in seconds
    time : NDArray
        Time bin edges in seconds

    Returns
    -------
    NDArray
        Spike counts per time bin (length matches time array)
    """
    spike_times = spike_times[
        np.logical_and(spike_times >= time[0], spike_times <= time[-1])
    ]
    return np.bincount(
        np.digitize(spike_times, time[1:-1]),
        minlength=time.shape[0],
    )


def reindex_by(
    df: pd.Series | pd.DataFrame,
    index: pd.Index,
    tolerance: float | None = None,
) -> pd.Series | pd.DataFrame:
    """Reindex pandas object to new index using nearest neighbor.

    Parameters
    ----------
    df : pd.Series | pd.DataFrame
        Data to reindex.
    index : pd.Index
        Target index to align to.
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
    return df.reindex(index, method="nearest", tolerance=tolerance)


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


def test_theta_modulation_v1(
    position_info: pd.DataFrame, theta_phase: pd.Series, neuron_spike_times: NDArray
) -> tuple[float, float, float, float]:
    """Test theta phase modulation of neuron firing (original version).

    NOTE: This is an older implementation kept for backward compatibility.
    Consider using `test_theta_modulation()` instead, which uses pre-computed
    spike counts and regularization for more robust parameter estimates.

    Differences from newer version:
    - Takes raw spike times instead of pre-computed counts
    - Derives running state from position_info instead of is_immobile flag
    - Does not use regularization for modulation estimation

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with head_speed column
    theta_phase : pd.Series
        Theta phase timeseries
    neuron_spike_times : NDArray
        Spike times in seconds

    Returns
    -------
    tuple[float, float, float, float]
        likelihood_ratio : Likelihood ratio statistic
        p_val : P-value from chi-square test
        modulation : Theta modulation strength
        preferred_phase : Preferred theta phase in radians
    """
    is_running = position_info.head_speed > 4.0

    formula = "1 + np.cos(theta_phase) + np.sin(theta_phase)"
    design_matrix = dmatrix(
        formula,
        {"theta_phase": reindex_by(theta_phase, position_info.index)},
        return_type="dataframe",
    )
    response = get_spikecount_per_time_bin(
        neuron_spike_times, position_info.index.to_numpy()
    )

    constant_glm_model = sm.GLM(
        response[is_running],
        np.ones_like(response[is_running]),
        family=sm.families.Poisson(),
    ).fit()

    theta_glm_model = sm.GLM(
        response[is_running],
        design_matrix.values[is_running],
        family=sm.families.Poisson(),
    ).fit()

    likelihood_ratio, p_val = likelihood_ratio_test(
        theta_glm_model.deviance,
        constant_glm_model.deviance,
        theta_glm_model.df_resid,
        constant_glm_model.df_resid,
    )

    modulation = np.sqrt(
        theta_glm_model.params[1] ** 2 + theta_glm_model.params[2] ** 2
    )
    preferred_phase = np.arctan2(theta_glm_model.params[2], theta_glm_model.params[1])

    return likelihood_ratio, p_val, modulation, preferred_phase


def _lr_test(full_model: Any, restricted_model: Any) -> tuple[float, float]:
    return likelihood_ratio_test(
        full_model.deviance,
        restricted_model.deviance,
        full_model.df_resid,
        restricted_model.df_resid,
    )


def test_theta_modulation(
    theta_phase: pd.Series, is_immobile: pd.Series, neuron_spike_count: NDArray
) -> tuple[float, float, float, float]:
    is_running = ~is_immobile

    formula = "1 + np.cos(theta_phase) + np.sin(theta_phase)"
    design_matrix = dmatrix(
        formula,
        {"theta_phase": reindex_by(theta_phase, is_immobile.index)},
        return_type="dataframe",
    )

    constant_glm_results = sm.GLM(
        neuron_spike_count[is_running],
        np.ones_like(neuron_spike_count[is_running]),
        family=sm.families.Poisson(),
    ).fit()

    theta_glm_model = sm.GLM(
        neuron_spike_count[is_running],
        design_matrix.values[is_running],
        family=sm.families.Poisson(),
    )
    theta_glm_results = theta_glm_model.fit()

    likelihood_ratio, p_val = _lr_test(theta_glm_results, constant_glm_results)

    # Fit regularized model to avoid overfitting for modulation and preferred phase
    theta_glm_results = theta_glm_model.fit_regularized(alpha=1e-20, L1_wt=0.0)

    modulation = np.sqrt(
        theta_glm_results.params[1] ** 2 + theta_glm_results.params[2] ** 2
    )
    preferred_phase = np.arctan2(
        theta_glm_results.params[2], theta_glm_results.params[1]
    )

    return likelihood_ratio, p_val, modulation, preferred_phase


def create_lagged_signal(signal: NDArray, lags: NDArray | None = None) -> NDArray:
    if lags is None:
        lags = np.arange(-100, 100, 1)
    signal_df = pd.DataFrame(signal)
    return np.concatenate(
        [signal_df.shift(lag, fill_value=0).to_numpy() for lag in lags], axis=1
    )


def create_conv_basis(
    count: NDArray, lags: NDArray, knots: NDArray, sampling_frequency: float
) -> tuple[NDArray, NDArray, NDArray]:
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
    knots = np.arange(
        lags[0] / sampling_frequency, lags[-1] / sampling_frequency, 0.100
    )[1:]
    conv_spline_basis: dict[str, NDArray] = {}
    for times_name, times_df in times.items():
        is_time = get_spikecount_per_time_bin(
            times_df.start_time, position_info.index.to_numpy()
        )
        conv_spline_basis[times_name], spline_basis, lag_time = create_conv_basis(
            is_time, lags, knots, sampling_frequency
        )

    return conv_spline_basis, spline_basis, lag_time


def _make_design_matrices(
    times: dict[str, pd.DataFrame],
    position_info: pd.DataFrame,
    sampling_frequency: float = 500,
    lags: NDArray | None = None,
) -> tuple[dict[str, NDArray], NDArray, NDArray]:
    constant_design_matrix = np.ones((len(position_info), 1))

    conv_spline_basis, spline_basis, lag_time = _make_conv_spline_basis(
        times, position_info, sampling_frequency=sampling_frequency, lags=lags
    )

    non_local_no_SWR_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    non_local_SWR_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_SWR"],
        ],
        axis=1,
    )

    SWR_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["SWR"],
        ],
        axis=1,
    )

    non_local_SWR_and_non_local_no_SWR_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_no_SWR"],
            conv_spline_basis["non_local_SWR"],
        ],
        axis=1,
    )

    SWR_and_non_local_no_SWR_design_matrix = np.concatenate(
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
        "non_local_no_SWR": non_local_no_SWR_design_matrix,
        "non_local_SWR": non_local_SWR_design_matrix,
        "SWR": SWR_design_matrix,
        "non_local_SWR + non_local_no_SWR": non_local_SWR_and_non_local_no_SWR_design_matrix,
        "SWR + non_local_no_SWR": SWR_and_non_local_no_SWR_design_matrix,
        "non_local": non_local_design_matrix,
    }

    return design_matrices, spline_basis, lag_time


def _fit_immobility_models(
    is_immobile: pd.Series,
    design_matrices: dict[str, NDArray],
    neuron_spike_count: NDArray,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return float(
        (reduced_model.deviance - full_model.deviance) / constant_model.deviance
    )


def get_frac_explained_deviance(
    full_model: Any, reduced_model: Any, constant_model: Any
) -> float:
    return float(
        (reduced_model.deviance - full_model.deviance)
        / (constant_model.deviance - full_model.deviance)
    )


def process_neuron(
    neuron_spike_times: NDArray,
    is_immobile: pd.Series,
    theta_phase: pd.Series,
    time: NDArray,
    immobility_design_matrices: dict[str, NDArray],
    spline_basis: NDArray,
    lag_time: NDArray,
) -> tuple[dict[str, Any], NDArray, NDArray]:
    try:
        neuron_spike_count = get_spikecount_per_time_bin(neuron_spike_times, time)
        sampling_frequency = 1 / np.median(np.diff(time))
        neuron_stats: dict[str, Any] = {}

        # Get run theta statistics
        (
            neuron_stats["theta vs. constant: lr"],
            neuron_stats["theta vs. constant: pval"],
            neuron_stats["theta_mod"],
            neuron_stats["theta_preferred_phase"],
        ) = test_theta_modulation(theta_phase, is_immobile, neuron_spike_count)
        neuron_stats["run_avg_firing_rate"] = (
            neuron_spike_count[~is_immobile].mean() * sampling_frequency
        )

        # Get immobility SWR/non-local statistics
        neuron_stats["immobility_avg_firing_rate"] = (
            neuron_spike_count[is_immobile].mean() * sampling_frequency
        )

        results, models = _fit_immobility_models(
            is_immobile, immobility_design_matrices, neuron_spike_count
        )

        ## Likelihood ratio tests
        test_cases = [
            ("SWR", "constant"),
            ("non_local_no_SWR", "constant"),
            ("SWR + non_local_no_SWR", "SWR"),
            ("SWR + non_local_no_SWR", "constant"),
            ("non_local", "constant"),
        ]

        for model1, model2 in test_cases:
            test_name = f"{model1} vs. {model2}"
            neuron_stats[f"{test_name}: lr"], neuron_stats[f"{test_name}: pval"] = (
                _lr_test(results[model1], results[model2])
            )

        ## Fractional null deviance
        test_cases = [
            ("SWR + non_local_no_SWR", "SWR"),
        ]
        for model1, model2 in test_cases:
            test_name = f"{model1} vs. {model2}"
            neuron_stats[f"{test_name}: frac_null_dev"] = get_frac_null_deviance(
                results[model1], results[model2], results["constant"]
            )

        ## AIC
        aics = {
            model_name: result.aic
            for model_name, result in results.items()
            if model_name
            in [
                "constant",
                "SWR",
                "SWR + non_local_no_SWR",
                "non_local",
            ]
        }
        neuron_stats["min_aic_model"] = min(aics, key=lambda k: aics[k])

        ## Modulation
        # Non-overlapping events controlling for SWR
        # Fit regularized model to avoid overfitting
        reg_results = models["SWR + non_local_no_SWR"].fit_regularized(
            alpha=1e-20, L1_wt=0.0
        )
        n_basis_functions = spline_basis.shape[1]
        SWR_modulation = spline_basis @ reg_results.params[1 : n_basis_functions + 1]
        SWR_max_mod_ind = np.argmax(np.abs(SWR_modulation))
        neuron_stats["SWR_max_mod"] = SWR_modulation[SWR_max_mod_ind]
        neuron_stats["SWR_max_mod_time"] = lag_time[SWR_max_mod_ind]

        no_SWR_modulation = spline_basis @ reg_results.params[n_basis_functions + 1 :]
        no_SWR_max_mod_ind = np.argmax(np.abs(no_SWR_modulation))
        neuron_stats["non_local_no_SWR_max_mod"] = no_SWR_modulation[no_SWR_max_mod_ind]
        neuron_stats["non_local_no_SWR_max_mod_time"] = lag_time[no_SWR_max_mod_ind]
    except ValueError:
        neuron_stats = {}
        neuron_stats["run_avg_firing_rate"] = (
            neuron_spike_count[~is_immobile].mean() * sampling_frequency
        )
        neuron_stats["immobility_avg_firing_rate"] = (
            neuron_spike_count[is_immobile].mean() * sampling_frequency
        )
        SWR_modulation = np.full(spline_basis.shape[0], np.nan)
        no_SWR_modulation = np.full(spline_basis.shape[0], np.nan)

    return neuron_stats, SWR_modulation, no_SWR_modulation


def run_session(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: float = 500,
    immobility_threshold: float = 4.0,
    non_local_threshold: float = 0.8,
    run_parallel: bool = True,
) -> tuple[pd.DataFrame, NDArray, NDArray]:
    results, _, _ = load_results(nwb_file_name, epoch_name, decoding_brain_area="HPC")
    data = load_data(nwb_file_name, epoch_name)

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

    non_local_SWR_times, non_local_no_SWR_times = get_overlap_times(
        non_local_times, data["ripple_times"], results
    )

    event_times = {
        "non_local_no_SWR": non_local_no_SWR_times,
        "non_local_SWR": non_local_SWR_times,
        "non_local": non_local_times,
        "SWR": data["ripple_times"],
    }

    time = data["position_info"].index.to_numpy()
    is_immobile = data["position_info"].head_speed <= immobility_threshold

    immobility_design_matrices, spline_basis, lag_time = _make_design_matrices(
        event_times,
        data["position_info"],
        sampling_frequency=sampling_frequency,
        lags=np.arange(-100, 101, 1),
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
            )
            for neuron_spike_times in data["spike_times"][brain_area]
        ]

        result = dask.compute(*tasks)
    else:
        result = [
            process_neuron(
                neuron_spike_times,
                is_immobile,
                data["theta_phase"],
                time,
                immobility_design_matrices,
                spline_basis,
                lag_time,
            )
            for neuron_spike_times in data["spike_times"][brain_area]
        ]

    neuron_stats_list, SWR_modulation_list, no_SWR_modulation_list = zip(
        *result, strict=True
    )
    neuron_stats = pd.DataFrame(list(neuron_stats_list))
    SWR_modulation = np.stack(SWR_modulation_list, axis=1)
    no_SWR_modulation = np.stack(no_SWR_modulation_list, axis=1)

    neuron_stats["nwb_file_name"] = nwb_file_name
    neuron_stats["epoch_name"] = epoch_name
    neuron_stats["brain_area"] = brain_area

    return neuron_stats, SWR_modulation, no_SWR_modulation
