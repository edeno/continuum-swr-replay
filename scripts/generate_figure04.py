import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

from continuum_swr_replay.analysis.hippocampus import (
    get_confident_times,
    get_overlap_times,
    load_results,
)
from continuum_swr_replay.analysis.pfc import (
    _fit_immobility_models,
    _make_design_matrices,
    process_neuron,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR


def plot_raster_relative_to_event(
    spike_times, event_times, window, ax=None, color="black"
):
    if ax is None:
        ax = plt.gca()

    event_times = np.asarray(event_times)
    relative_spike_times = []
    for event_time in event_times:
        event_relative_spike_times = spike_times - event_time

        relative_spike_times.append(
            event_relative_spike_times[
                np.logical_and(
                    event_relative_spike_times > window[0],
                    event_relative_spike_times < window[1],
                )
            ]
        )
    ax.eventplot(relative_spike_times, color=color)
    ax.set_xlim(window)
    ax.axvline(0, color="black", linestyle="--")


def plot_hist_relative_to_event(
    spike_times,
    event_times,
    window,
    ax=None,
    color="grey",
    bin_size=0.050,
):
    if ax is None:
        ax = plt.gca()

    n_bins = int((window[1] - window[0]) // bin_size)
    bins = np.arange(window[0], window[1] + bin_size, bin_size)

    event_times = np.asarray(event_times)
    relative_spike_counts = np.zeros_like(bins)
    for event_time in event_times:
        event_relative_spike_times = spike_times - event_time

        relative_spike_counts += get_spikecount_per_time_bin(
            event_relative_spike_times[
                np.logical_and(
                    event_relative_spike_times > window[0],
                    event_relative_spike_times < window[1],
                )
            ],
            bins,
        )

    n_events = len(event_times)
    ax.bar(
        bins,
        relative_spike_counts / (n_events * bin_size),
        width=bin_size,
        color=color,
        align="edge",
    )
    ax.set_xlim(window)
    ax.axvline(0, color="black", linestyle="--")


def plot_glm_results(
    lag_time,
    spline_basis,
    neuron_spike_times,
    glm_results,
    event_times,
    model_list=None,
):
    if model_list is None:
        model_list = ["SWR", "non_local_SWR", "non_local_no_SWR", "non_local"]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=len(model_list),
        figsize=(2.5 * len(model_list), 5),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    if len(model_list) == 1:
        axes = np.expand_dims(axes, 1)

    modulation_bounds = np.log(0.5), np.log(2.0)
    lag_bounds = lag_time[0], lag_time[-1]

    models = [glm_results[event_name] for event_name in model_list]

    model_list_to_mode_names = {
        "SWR": "SWR",
        "non_local_SWR": "Non-Local,\nSWR",
        "non_local_no_SWR": "Non-Local,\nNo SWR",
        "non_local": "Non-Local",
    }
    model_names = [model_list_to_mode_names[model_name] for model_name in model_list]

    model_names_to_event = {
        "SWR": "SWR",
        "Non-Local, SWR": "non_local_SWR",
        "Non-Local, No SWR": "non_local_no_SWR",
        "Non-Local": "non_local",
    }
    # 'SWR', 'non_local_SWR', 'non_local_no_SWR', 'non_local',
    event_start_times = [
        event_times[model_name].start_time
        for model_name in model_names_to_event.values()
    ]

    for ax, model, model_name in zip(axes[0], models, model_names):
        ax.plot(lag_time, model.params[1:] @ spline_basis.T)
        ax.fill_between(
            lag_time,
            (model.params[1:] + model.bse[1:]) @ spline_basis.T,
            (model.params[1:] - model.bse[1:]) @ spline_basis.T,
            alpha=0.25,
        )
        ax.axhline(0, color="black", linestyle="--")
        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(model_name)
        ax.set_ylim(modulation_bounds)
        ax.set_xticks((lag_bounds[0], 0, lag_bounds[1]))
        ax.set_yticks((modulation_bounds[0], 0, modulation_bounds[1]))
        ax.set_ylabel("Mod.")
        ax.set_yticklabels(
            (np.exp(modulation_bounds[0]), 1, np.exp(modulation_bounds[1]))
        )

    for ax, event_time in zip(axes[1], event_start_times):
        plot_raster_relative_to_event(neuron_spike_times, event_time, lag_bounds, ax)
        ax.set_ylabel("Event")

    for ax, event_time in zip(axes[2], event_start_times):
        plot_hist_relative_to_event(neuron_spike_times, event_time, lag_bounds, ax)
        ax.set_ylabel("Rate\n[sp/s]")

    fig.supxlabel("Time from\nevent start (s)")


def plot_AICs(glm_results, model_list=None):
    if model_list is None:
        model_list = [
            "constant",
            "SWR",
            "non_local_SWR",
            "non_local_no_SWR",
            "non_local",
            "SWR + non_local_no_SWR",
        ]
    fig, ax = plt.subplots(figsize=(4, 3))

    model_list_to_mode_names = {
        "constant": "Constant",
        "SWR": "SWR",
        "non_local_SWR": "Non-Local,\nSWR",
        "non_local_no_SWR": "Non-Local,\nNo SWR",
        "non_local": "Non-Local",
        "SWR + non_local_no_SWR": "SWR + Non-Local",
    }
    model_names = [model_list_to_mode_names[model_name] for model_name in model_list]

    aics = [glm_results[model_name].aic for model_name in model_list]

    ax.scatter(
        aics,
        np.arange(len(aics)),
    )
    ax.set_yticks(np.arange(len(aics)))
    ax.set_yticklabels(model_names)
    ax.set_xticks([np.min(aics), np.max(aics)])
    ax.set_xlabel("AIC (lower is better)")


nwb_file_name, epoch_name = "wilbur20210329_.nwb", "02_r1"


non_local_threshold = 0.8
immobility_threshold = 4.0
sampling_frequency = 500

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
non_local_times = non_local_times[non_local_times.max_speed < 4.0]

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

# neuron_ind = 2
# neuron_ind = 16
# neuron_ind = 25
# neuron_ind = 27 # 27 is a good example
# neuron_ind = 52
# neuron_ind = 55
# neuron_ind = 61
# neuron_ind = 73
# neuron_ind = 81
# neuron_ind = 89
# neuron_ind = 90 # 90 is a good example
neuron_ind = 27
# neuron_ind = 90
neuron_spike_times = data["spike_times"]["mPFC"][neuron_ind]
neuron_spike_count = get_spikecount_per_time_bin(neuron_spike_times, time)

glm_results, glm_models = _fit_immobility_models(
    is_immobile, immobility_design_matrices, neuron_spike_count
)

model_list = ["non_local_no_SWR"]
plot_glm_results(
    lag_time,
    spline_basis,
    neuron_spike_times,
    glm_results,
    event_times,
    model_list=model_list,
)
model_list = ["non_local_no_SWR", "SWR"]
plot_glm_results(
    lag_time,
    spline_basis,
    neuron_spike_times,
    glm_results,
    event_times,
    model_list=model_list,
)

model_list = [
    "constant",
    "SWR",
    "SWR + non_local_no_SWR",
    "non_local",
]
plot_AICs(glm_results, model_list=model_list)


data_path = PROCESSED_DATA_DIR.glob("*_mPFC_modulation.pkl")
df = pd.concat([pd.read_pickle(p) for p in data_path])
df["animal"] = df["nwb_file_name"].str[:-13]
df
