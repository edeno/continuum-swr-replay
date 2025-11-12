import copy

import holoviews as hv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import seaborn as sns
import sortingview.views as vv
import xarray as xr
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.analysis.posterior import maximum_a_posteriori_estimate
from non_local_detector.visualization.figurl_1D import create_1D_decode_view

from continuum_swr_replay.analysis.hippocampus.analysis_hpc_bandit import (
    get_conditional_non_local_posterior,
    get_confident_times,
)
from continuum_swr_replay.parameters import COLORS


def create_interactive_1D_decoding_figurl(
    linear_position: pd.Series,
    speed: pd.Series,
    head_direction: pd.Series,
    results: xr.Dataset,
    MUA_times: pd.DataFrame,
    MUA_zscore: pd.DataFrame,
    SWR_times: pd.DataFrame,
    SWR_zscore: pd.DataFrame,
    ahead_behind_distance: np.ndarray,
    hpd: np.ndarray,
    theta_filtered_lfp: pd.Series,
    posterior=None,
    view_height: int = 800,
    label: str = "",
) -> str:
    """Create an interactive figure for 1D decoding

    Parameters
    ----------
    linear_position : pd.Series, shape (n_time,)
        Position of the animal in 1D
    speed : pd.Series, shape (n_time,)
        Speed of the animal
    head_direction : pd.Series, shape (n_time,)
        Head direction of the animal
    results : xr.Dataset
        From decoding
    MUA_times : pd.DataFrame, shape (n_events, 2)
        Event times of multiunit activity
    MUA_zscore : pd.DataFrame, shape (n_time,)
        The zscore of the multiunit activity
    SWR_times : pd.DataFrame, shape (n_events, 2)
        Event times of sharp wave ripples
    SWR_zscore : pd.DataFrame, shape (n_time,)
        The zscore of the sharp wave ripple trace
    ahead_behind_distance : np.ndarray, shape (n_time,)
        Distance of the most likely position from the animal
    hpd : np.ndarray, shape (n_time,)
        Highest Posterior Density
    theta_filtered_lfp : pd.Series, shape (n_time,)
        LFP filtered in theta band
    posterior : _type_, optional, shape (n_time, n_position)
        Conditional non-local posterior, by default None
    view_height : int, optional
        Height of the view, by default 800
    label : str, optional
       Figurl label, by default ""

    Returns
    -------
    url: str
       Link to the figurl
    """
    if posterior is None:
        posterior = get_conditional_non_local_posterior(results)

    decode_view = create_1D_decode_view(
        posterior=posterior,
        linear_position=linear_position,
    )

    speed_view = vv.TimeseriesGraph().add_line_series(
        name="Speed [cm/s]",
        t=np.asarray(speed.index),
        y=np.asarray(speed, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )
    head_direction_view = vv.TimeseriesGraph(
        y_range=[-np.pi, np.pi],
    ).add_line_series(
        name="Dir. [rad]",
        t=np.asarray(head_direction.index),
        y=np.asarray(head_direction, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )

    multiunit_firing_rate_view = vv.TimeseriesGraph()
    multiunit_firing_rate_view.add_interval_series(
        name="MUA Events",
        t_start=MUA_times.start_time.to_numpy(),
        t_end=MUA_times.end_time.to_numpy(),
        color=COLORS["MUA"],
    )
    multiunit_firing_rate_view.add_line_series(
        name="MUA [std]",
        t=np.asarray(MUA_zscore.index),
        y=np.asarray(MUA_zscore, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )

    ripple_zscore_view = vv.TimeseriesGraph()
    ripple_zscore_view.add_interval_series(
        name="SWR Events",
        t_start=SWR_times.start_time.to_numpy(),
        t_end=SWR_times.end_time.to_numpy(),
        color=COLORS["SWR"],
    )
    ripple_zscore_view.add_line_series(
        name="SWR [std]",
        t=np.asarray(SWR_zscore.index),
        y=np.asarray(SWR_zscore, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )

    non_local_times = get_confident_times(
        results.sel(states="Non-Local Continuous").acausal_state_probabilities
        + results.sel(states="Non-Local Fragmented").acausal_state_probabilities,
        minimum_duration=0.015,
    )

    non_local_probability_view = vv.TimeseriesGraph()
    non_local_probability_view.add_interval_series(
        name="Non-Local Events",
        t_start=non_local_times.start_time.to_numpy(),
        t_end=non_local_times.end_time.to_numpy(),
        color=COLORS["Non-Local"],
    )
    non_local_probability_view.add_line_series(
        name="Non-Local Continuous",
        t=np.asarray(results.time),
        y=np.asarray(
            results.sel(states="Non-Local Continuous").acausal_state_probabilities,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )
    non_local_probability_view.add_line_series(
        name="Non-Local Fragmented",
        t=np.asarray(results.time),
        y=np.asarray(
            results.sel(states="Non-Local Fragmented").acausal_state_probabilities,
            dtype=np.float32,
        ).squeeze(),
        color="#d62728",
        width=1,
    )

    local_times = get_confident_times(
        results.sel(states="Local").acausal_state_probabilities,
        minimum_duration=0.0,
    )

    local_probability_view = vv.TimeseriesGraph()
    local_probability_view.add_interval_series(
        name="Local Events",
        t_start=local_times.start_time.to_numpy(),
        t_end=local_times.end_time.to_numpy(),
        color="#d3d3d3",
    )
    local_probability_view.add_line_series(
        name="No-Spike",
        t=np.asarray(results.time),
        y=np.asarray(
            results.sel(states="No-Spike").acausal_state_probabilities,
            dtype=np.float32,
        ).squeeze(),
        color="#ff7f0e",
        width=1,
    )
    local_probability_view.add_line_series(
        name="Local",
        t=np.asarray(results.time),
        y=np.asarray(
            results.sel(states="Local").acausal_state_probabilities,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )

    theta_view = vv.TimeseriesGraph()
    theta_view.add_line_series(
        name="Theta Filtered LFP",
        t=np.asarray(theta_filtered_lfp.index),
        y=np.asarray(
            theta_filtered_lfp,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )

    ahead_behind_distance_view = vv.TimeseriesGraph().add_line_series(
        name="Ahead-Behind Distance",
        t=np.asarray(results.time),
        y=np.asarray(
            ahead_behind_distance,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )

    hpd_view = vv.TimeseriesGraph().add_line_series(
        name="HPD",
        t=np.asarray(results.time),
        y=np.asarray(
            hpd,
            dtype=np.float32,
        ).squeeze(),
        color="black",
        width=1,
    )

    vertical_panel_content = [
        vv.LayoutItem(theta_view, stretch=1, title="Theta", collapsible=True),
        vv.LayoutItem(ripple_zscore_view, stretch=1, title="SWR", collapsible=True),
        vv.LayoutItem(
            multiunit_firing_rate_view, stretch=1, title="MUA", collapsible=True
        ),
        vv.LayoutItem(
            non_local_probability_view, stretch=1, title="Non-Local", collapsible=True
        ),
        vv.LayoutItem(
            local_probability_view, stretch=1, title="Local", collapsible=True
        ),
        vv.LayoutItem(decode_view, stretch=3, title="Decode", collapsible=True),
        vv.LayoutItem(
            ahead_behind_distance_view, stretch=1, title="Dist.", collapsible=True
        ),
        vv.LayoutItem(hpd_view, stretch=1, title="HPD", collapsible=True),
        vv.LayoutItem(speed_view, stretch=1, title="Speed", collapsible=True),
        vv.LayoutItem(
            head_direction_view, stretch=1, title="Head Dir.", collapsible=True
        ),
    ]

    view = vv.Box(
        direction="horizontal",
        show_titles=True,
        height=view_height,
        items=[
            vv.LayoutItem(
                vv.Box(
                    direction="vertical",
                    show_titles=True,
                    items=vertical_panel_content,
                )
            ),
        ],
    )

    return view.url(label=label)


def plot_peri_event_probability(
    results,
    ripple_zscore,
    mua_zscore,
    start_times,
    end_times,
    relative_time_range=(-0.4, 0.4),
    sort_by="TIME",
    show_end_times=False,
    sampling_frequency=500,
):
    non_local_prob = results.acausal_state_probabilities.sel(
        states=["Non-Local Continuous", "Non-Local Fragmented"]
    ).sum("states")

    n_time_bins = int(np.ceil(np.diff(relative_time_range) * sampling_frequency))
    relative_time = relative_time_range[0] + np.arange(n_time_bins) / sampling_frequency

    event_probs = []
    event_swr_zscores = []
    event_mua_zscores = []

    mean_event_prob = []
    max_swr_zscores = []
    max_mua_zscores = []

    for start_time, end_time in zip(start_times, end_times):
        time_slice = slice(
            start_time + relative_time_range[0], start_time + relative_time_range[1]
        )

        prob_event_time_slice = slice(start_time, end_time)

        # Non-local probability
        event_prob = non_local_prob.sel(time=time_slice)
        event_prob["time"] = event_prob.time - start_time
        event_prob = event_prob.reindex(
            {"time": relative_time},
            tolerance=1 / (2 * sampling_frequency),
            method="nearest",
            fill_value=np.nan,
        )
        event_probs.append(event_prob.values)
        mean_event_prob.append(
            non_local_prob.sel(time=prob_event_time_slice).mean().values
        )

        # SWR z-score
        event_swr_zscore = ripple_zscore.loc[time_slice]
        event_swr_zscore = event_swr_zscore.set_index(
            event_swr_zscore.index - start_time
        ).reindex(
            relative_time, tolerance=1 / (2 * sampling_frequency), method="nearest"
        )
        event_swr_zscores.append(event_swr_zscore.to_numpy().squeeze())
        max_swr_zscores.append(np.nanmax(ripple_zscore.loc[prob_event_time_slice]))

        # MUA z-score
        event_mua_zscore = mua_zscore.loc[time_slice]
        event_mua_zscore = event_mua_zscore.set_index(
            event_mua_zscore.index - start_time
        ).reindex(
            relative_time, tolerance=1 / (2 * sampling_frequency), method="nearest"
        )
        event_mua_zscores.append(event_mua_zscore.to_numpy().squeeze())
        max_mua_zscores.append(np.nanmax(mua_zscore.loc[prob_event_time_slice]))

    event_probs = np.stack(event_probs, axis=1)
    event_swr_zscores = np.stack(event_swr_zscores, axis=1)
    event_mua_zscores = np.stack(event_mua_zscores, axis=1)

    if sort_by.upper() == "PROB":
        sort_ind = np.argsort(mean_event_prob)[::-1]
    elif sort_by.upper() == "MUA":
        sort_ind = np.argsort(max_mua_zscores)[::-1]
    elif sort_by.upper() == "SWR":
        sort_ind = np.argsort(max_swr_zscores)[::-1]
    elif sort_by.upper() == "TIME":
        sort_ind = np.argsort(start_times)

    event_probs = event_probs[:, sort_ind]
    event_swr_zscores = event_swr_zscores[:, sort_ind]
    event_mua_zscores = event_mua_zscores[:, sort_ind]

    duration = end_times - start_times
    duration = duration[sort_ind]

    fig, axes = plt.subplots(
        6,
        1,
        sharex=True,
        figsize=(10, 20),
        gridspec_kw={"height_ratios": [3, 3, 3, 1, 1, 1]},
        constrained_layout=True,
    )
    n_events = len(start_times)
    t, n = np.meshgrid(relative_time, np.arange(n_events))

    # Non-local probability
    h = axes[0].pcolormesh(t, n, event_probs.T, vmin=0.0, vmax=1.0, cmap="cividis")
    axes[0].axvline(0, color=COLORS["clusterless"], linestyle="--")
    axes[0].set_ylabel("Event number")
    axes[0].set_title("Non-Local Event Probability")
    plt.colorbar(h, ax=axes[0])

    # SWR z-score
    h = axes[1].pcolormesh(t, n, event_swr_zscores.T, vmin=-3, vmax=3, cmap="RdBu_r")
    axes[1].axvline(0, color=COLORS["clusterless"], linestyle="--")
    axes[1].set_ylabel("Event number")
    axes[1].set_title("SWR z-score")
    plt.colorbar(h, ax=axes[1])

    # MUA z-score
    h = axes[2].pcolormesh(t, n, event_mua_zscores.T, vmin=-3, vmax=3, cmap="RdBu_r")
    axes[2].axvline(0, color=COLORS["clusterless"], linestyle="--")
    axes[2].set_ylabel("Event number")
    axes[2].set_title("MUA z-score")
    plt.colorbar(h, ax=axes[2])

    if show_end_times:
        for ax in axes[:3]:
            ax.scatter(duration, np.arange(n_events), color=COLORS["clusterless"], s=1)

    # Non-Local IQR
    prob_iqr = np.nanpercentile(event_probs, [25, 50, 75], axis=1)
    axes[3].axvline(0, color=COLORS["clusterless"], linestyle="--")
    axes[3].plot(
        relative_time, prob_iqr[1], color=COLORS["clusterless"], label="Non-Local"
    )
    axes[3].fill_between(
        relative_time,
        prob_iqr[0],
        prob_iqr[2],
        alpha=0.5,
        color=COLORS["clusterless"],
    )
    axes[3].set_ylabel("Prob.")
    axes[3].legend()

    # SWR IQR
    swr_iqr = np.nanpercentile(event_swr_zscores, [25, 50, 75], axis=1)
    axes[4].axvline(0, color=COLORS["clusterless"], linestyle="--")
    axes[4].plot(relative_time, swr_iqr[1], color=COLORS["SWR"], label="SWR")
    axes[4].fill_between(
        relative_time,
        swr_iqr[0],
        swr_iqr[2],
        alpha=0.5,
        color=COLORS["SWR"],
    )
    axes[4].set_ylabel("z-score")
    axes[4].set_ylim(-3, 3)
    axes[4].axhline(0, color="k", linestyle="--")
    axes[4].legend()

    # MUA IQR
    mua_iqr = np.nanpercentile(event_mua_zscores, [25, 50, 75], axis=1)
    axes[5].axvline(0, color=COLORS["clusterless"], linestyle="--")
    axes[5].plot(relative_time, mua_iqr[1], color=COLORS["MUA"], label="MUA")
    axes[5].fill_between(
        relative_time,
        mua_iqr[0],
        mua_iqr[2],
        alpha=0.5,
        color=COLORS["MUA"],
    )
    axes[5].set_ylabel("z-score")
    axes[5].set_ylim(-3, 3)
    axes[5].axhline(0, color=COLORS["clusterless"], linestyle="--")
    axes[5].legend()

    axes[-1].set_xlabel("Time relative to event (s)")


def plot_theta_lfp(data, time_slice, ax):
    theta_lfp = data["theta_lfp"].loc[time_slice]
    ax.plot(theta_lfp.index, theta_lfp, color="black")
    ax.set_ylabel("Theta\nTrace")


def plot_theta_power_zscore(data, time_slice, ax, plot_thresholds=False):
    theta_power_zscore = data["theta_power_zscore"].loc[time_slice]

    max_consensus = max(int(np.ceil(theta_power_zscore.max())), 5)
    min_consensus = min(int(np.ceil(theta_power_zscore.min())), -1)

    ax.fill_between(
        theta_power_zscore.index,
        theta_power_zscore.values.squeeze(),
        color="black",
    )
    ax.set_ylim((min_consensus, max_consensus))
    ax.set_yticks((min_consensus, max_consensus))
    ax.set_ylabel("Theta\nStd")

    if plot_thresholds:
        ax.axhline(2, color="grey", linestyle="--", linewidth=1, zorder=10)


def plot_theta_phase(data, time_slice, ax):
    theta_phase = data["theta_phase"].loc[time_slice]

    ax.plot(
        theta_phase.index,
        theta_phase.values.squeeze(),
        color="black",
    )
    ax.set_ylim((-np.pi, np.pi))
    ax.set_yticks((-np.pi, 0, np.pi))
    ax.set_ylabel("Theta\nPhase")
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])


def plot_multiunit_firing(
    data, time_slice, ax, plot_multiunit_zscore=False, plot_thresholds=False
):
    if plot_multiunit_zscore:
        multiunit_firing = data["multiunit_rate_zscore"]
        ax.set_ylabel("MUA\n[Std].")
        if plot_thresholds:
            ax.axhline(2, color="grey", linestyle="--", linewidth=1, zorder=10)
    else:
        multiunit_firing = data["multiunit_firing_rate"]
        ax.set_ylabel("MUA\n[spikes/s]")
    max_multiunit = np.ceil(multiunit_firing.values.max()).astype(int)
    min_multiunit = (
        np.floor(multiunit_firing.values.min()).astype(int)
        if plot_multiunit_zscore
        else 0
    )

    ax.fill_between(
        multiunit_firing.loc[time_slice].index.to_numpy(),
        multiunit_firing.loc[time_slice].to_numpy().squeeze(),
        color="black",
    )
    ax.fill_between(
        multiunit_firing.loc[time_slice].index.to_numpy(),
        np.ones_like(multiunit_firing.loc[time_slice].index.to_numpy()) * max_multiunit,
        where=data["multiunit_high_synchrony_labels"].loc[time_slice].values.squeeze()
        > 0,
        color=COLORS["MUA"],
        zorder=-2,
        alpha=0.6,
        step="pre",
        rasterized=True,
    )
    ax.set_ylim((min_multiunit, max_multiunit))
    ax.set_yticks((min_multiunit, max_multiunit))

    if plot_thresholds:
        ax.axhline(2, color="grey", linestyle="--", linewidth=1, zorder=10)


def plot_ripple_consensus_trace_zscore(data, time_slice, ax, plot_thresholds=False):
    ripple_consensus_trace = data["ripple_consensus_trace_zscore"].loc[time_slice]

    max_consensus = max(int(np.ceil(ripple_consensus_trace.max())), 5)
    min_consensus = min(int(np.ceil(ripple_consensus_trace.min())), -1)

    ax.fill_between(
        ripple_consensus_trace.index,
        np.ones_like(ripple_consensus_trace.index) * max_consensus,
        where=data["ripple_labels"].loc[time_slice].values.squeeze() > 0,
        color=COLORS["SWR"],
        zorder=-2,
        alpha=0.6,
        step="pre",
        rasterized=True,
    )
    ax.fill_between(
        ripple_consensus_trace.index,
        ripple_consensus_trace.values.squeeze(),
        color="black",
        rasterized=True,
    )
    ax.set_ylim((min_consensus, max_consensus))
    ax.set_yticks((min_consensus, max_consensus))
    ax.set_ylabel("SWR\n[Std]")

    if plot_thresholds:
        ax.axhline(2, color="grey", linestyle="--", linewidth=1, zorder=10)


def plot_spike_times(
    data,
    time_slice,
    detector,
    env,
    results,
    ax,
    data_type="clusterless",
):
    spike_times = (
        data["clusterless_spike_times"]
        if data_type == "clusterless"
        else data["sorted_spike_times"]
    )
    n_neurons = len(spike_times)
    acausal_state_probabilities = results.sel(
        time=time_slice
    ).acausal_state_probabilities
    non_local_probabilities = acausal_state_probabilities.sel(
        time=time_slice, states="Non-Local Continuous"
    ) + acausal_state_probabilities.sel(time=time_slice, states="Non-Local Fragmented")
    is_non_local = (non_local_probabilities > 0.8).values.squeeze()
    try:
        place_fields = detector.encoding_model_[("", 0)]["place_fields"]
        neuron_sort_ind = np.argsort(
            env.place_bin_centers_[np.nanargmax(place_fields, axis=1)].squeeze()
        )
        cell_label = "Neuron"
    except KeyError:
        neuron_sort_ind = np.arange(len(spike_times))
        cell_label = "Elec."
    new_spike_times = [
        spike_times[neuron_id][
            np.logical_and(
                spike_times[neuron_id] >= time_slice.start,
                spike_times[neuron_id] <= time_slice.stop,
            )
        ]
        for neuron_id in neuron_sort_ind
    ]
    ax.eventplot(
        new_spike_times,
        color="black",
        rasterized=True,
        linelengths=0.9,
        linewidths=1,
    )
    ax.set_yticks((1, n_neurons))
    ax.fill_between(
        non_local_probabilities.time,
        (np.ones_like(non_local_probabilities.values) * n_neurons) + 0.75,
        where=np.asarray(is_non_local).squeeze(),
        color=COLORS["Non-Local"],
        zorder=-1,
        alpha=0.6,
        step="pre",
    )
    ax.set_ylabel(cell_label)


def plot_acausal_probabilities(
    results, time_slice, ax, plot_thresholds=False, plot_local=False
):
    time = results.sel(time=time_slice).time
    non_local_prob = results.acausal_state_probabilities.sel(
        time=time_slice, states=["Non-Local Continuous", "Non-Local Fragmented"]
    ).sum("states")

    if plot_local:
        no_spike_prob = results.acausal_state_probabilities.sel(
            time=time_slice, states="No-Spike"
        )
        local_prob = results.acausal_state_probabilities.sel(
            time=time_slice, states="Local"
        )

        ax.plot(
            time,
            no_spike_prob,
            color=COLORS["No-Spike"],
            label="No-Spike",
            linewidth=1,
            clip_on=False,
        )
        ax.plot(
            time,
            local_prob,
            color=COLORS["Local"],
            label="Local",
            linewidth=1,
            clip_on=False,
        )
    ax.plot(
        time,
        non_local_prob,
        color=COLORS["Non-Local"],
        label="Non-Local",
        linewidth=1,
        clip_on=False,
    )
    if plot_local:
        ax.set_ylabel("State\nProb.")
    else:
        ax.set_ylabel("Non-\nLocal\nProb.")
    ax.set_xlabel("")
    ax.set_yticks((0, 1))
    ax.set_yticklabels((0, 1))
    ax.set_ylim((0, 1))

    if plot_thresholds:
        ax.axhline(0.8, color="grey", linestyle="--", linewidth=1, zorder=10)


def plot_conditional_non_local_posterior(
    data,
    time_slice,
    results,
    env,
    detector,
    ax,
    posterior_max=0.25,
):
    cmap = copy.copy(plt.get_cmap("bone_r"))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    state_ind = detector.state_ind_
    acausal_posterior = results.sel(time=time_slice).acausal_posterior.values

    t, x = np.meshgrid(results.sel(time=time_slice).time.values, env.place_bin_centers_)
    non_local_inds = np.nonzero(
        ["Non-Local" in state for state in detector.state_names]
    )[0]
    conditional_non_local_acausal_posterior = np.zeros(
        (len(results.sel(time=time_slice).time.values), len(env.place_bin_centers_))
    )
    for non_local_ind in non_local_inds:
        conditional_non_local_acausal_posterior += acausal_posterior[
            :, state_ind == non_local_ind
        ]
    conditional_non_local_acausal_posterior /= np.nansum(
        conditional_non_local_acausal_posterior, axis=1
    )[:, np.newaxis]
    conditional_non_local_acausal_posterior[:, ~env.is_track_interior_] = np.nan

    ax.pcolormesh(
        t,
        x,
        conditional_non_local_acausal_posterior.T,
        vmin=0.0,
        vmax=posterior_max,
        cmap=cmap,
        rasterized=True,
    )
    position_info = data["position_info"].loc[time_slice]
    ax.scatter(
        position_info.index,
        position_info.linear_position,
        s=1,
        color="magenta",
        clip_on=False,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlabel("")
    max_position = np.ceil(data["position_info"].linear_position.max()).astype(int)
    ax.set_yticks((0.0, max_position))
    ax.set_ylim((0.0, max_position))
    ax.set_ylabel("Position [cm]")


def plot_speed(data, time_slice, ax):
    position_info = data["position_info"].loc[time_slice]
    speed = np.asarray(position_info.nose_vel).squeeze()
    max_speed = np.ceil(speed.max()).astype(int)
    max_speed = max(5, max_speed)
    ax.fill_between(position_info.index, speed, color="lightgrey")
    # axes[subplot_offset + 5].axhline(4, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Speed\n[cm/s]")
    ax.set_yticks((0.0, max_speed))
    ax.set_ylim((0.0, max_speed))


def plot_head_direction(data, time_slice, ax):
    position_info = data["position_info"].loc[time_slice]
    direction = np.asarray(position_info.body_dir).squeeze()
    min_dir, max_dir = -np.pi, np.pi
    ax.scatter(position_info.index, direction, color="black", s=1, clip_on=False)
    ax.set_ylim(min_dir, max_dir)
    ax.set_yticks([min_dir, 0, max_dir])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])


def plot_detector(
    data,
    detector,
    results,
    figsize=(11, 6.0),
    plot_multiunit_zscore=True,
    plot_thresholds=False,
    plot_direction=False,
    plot_theta=False,
    time_slice=None,
    data_type="clusterless",
    posterior_max=0.25,
):
    if time_slice is None:
        time_slice = slice(results.time.values[0], results.time.values[-1])

    env = detector.environments[0]

    n_subplots = 8 if plot_theta else 6
    height_ratios = np.ones((n_subplots))
    height_ratios[-2] = 3
    height_ratios[-4] = 3
    subplot_offset = 2 if plot_theta else 0

    _, axes = plt.subplots(
        n_subplots,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
        dpi=100,
    )

    if plot_theta:
        plot_theta_lfp(data, time_slice, axes[0])
        plot_theta_power_zscore(data, time_slice, axes[1], plot_thresholds)
        plot_theta_phase(data, time_slice, axes[2])

    else:
        plot_ripple_consensus_trace_zscore(data, time_slice, axes[0], plot_thresholds)

    plot_multiunit_firing(
        data,
        time_slice,
        axes[subplot_offset + 1],
        plot_multiunit_zscore,
        plot_thresholds,
    )

    plot_spike_times(
        data,
        time_slice,
        detector,
        env,
        results,
        axes[subplot_offset + 2],
        data_type,
    )

    plot_acausal_probabilities(
        results, time_slice, axes[subplot_offset + 3], plot_thresholds=False
    )

    plot_conditional_non_local_posterior(
        data,
        time_slice,
        results,
        env,
        detector,
        axes[subplot_offset + 4],
        posterior_max,
    )

    plot_speed(data, time_slice, axes[subplot_offset + 5])

    if plot_direction:
        plot_head_direction(data, time_slice, axes[subplot_offset + 5].twinx())

    axes[-1].set_xticks((time_slice.start, time_slice.stop))
    axes[-1].set_xticklabels((time_slice.start, time_slice.stop))
    axes[-1].set_xlabel("Time [s]")
    axes[-1].xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    sns.despine(offset=5)


def create_multiregion_figurl(
    decoding_results,
    position_info,
    multiunit_firing_rate,
    view_height=800,
    label="",
):
    # Multiunit firing rate view
    multiunit_firing_rate_view = vv.TimeseriesGraph()
    for brain_area, rate in multiunit_firing_rate.items():
        multiunit_firing_rate_view.add_line_series(
            name=f"{brain_area} Rate [spikes/s]",
            t=np.asarray(position_info.index),
            y=np.asarray(rate, dtype=np.float32).squeeze(),
            color="red",
            width=1,
        )

    # Non-local probability view
    non_local_probability_view = vv.TimeseriesGraph()
    for brain_area, results in decoding_results.items():
        non_local_prob = results.acausal_state_probabilities.sel(
            states=["Non-Local Continuous", "Non-Local Fragmented"]
        ).sum("states")
        non_local_probability_view.add_line_series(
            name=f"{brain_area} Non-Local",
            t=np.asarray(non_local_prob.time),
            y=np.asarray(
                non_local_prob,
                dtype=np.float32,
            ).squeeze(),
            color="green",
            width=1,
        )

    # Decode views
    decode_views = dict()
    for brain_area, results in decoding_results.items():
        decode_views[brain_area] = create_1D_decode_view(
            posterior=NonLocalSortedSpikesDetector.get_conditional_non_local_posterior(
                results
            ),
            linear_position=position_info.linear_position,
        )

    # Speed view
    speed_view = vv.TimeseriesGraph().add_line_series(
        name="Speed [cm/s]",
        t=np.asarray(position_info.index),
        y=np.asarray(position_info.head_speed, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )

    vertical_panel_content = [
        vv.LayoutItem(multiunit_firing_rate_view, stretch=1, title="MUA"),
        vv.LayoutItem(non_local_probability_view, stretch=1, title="Non-Local"),
    ]
    for brain_area, decode_view in decode_views.items():
        vertical_panel_content.append(
            vv.LayoutItem(decode_view, stretch=3, title=brain_area)
        )

    vertical_panel_content.append(
        vv.LayoutItem(speed_view, stretch=1, title="Speed"),
    )

    view = vv.Box(
        direction="horizontal",
        show_titles=True,
        height=view_height,
        items=[
            vv.LayoutItem(
                vv.Box(
                    direction="vertical",
                    show_titles=True,
                    items=vertical_panel_content,
                )
            ),
        ],
    )

    return view.url(label=label)


# holoviews based plotting functions


def create_2D_decoding_plot(
    position_info,
    posterior,
    map_estimate=None,
    sampling_frequency=250,
    width=800,
    height=600,
    head_dir_radius=4,
    position_names=("x", "y"),
    head_dir_name="head_direction",
    cmap="viridis",
    animal_color="magenta",
    decode_opacity=1.0,
):
    n_time = len(posterior.time)

    if map_estimate is None:
        map_estimate = maximum_a_posteriori_estimate(posterior)

    def get_animal_position_dot(time_ind):
        head_point = hv.Scatter(
            position_info.iloc[[time_ind]],
            kdims=position_names[0],
            vdims=position_names[1],
        ).opts(size=10, color=animal_color, width=width, height=height)

        # Calculate the end point of the head direction line
        x_start = position_info.iloc[time_ind][position_names[0]]
        y_start = position_info.iloc[time_ind][position_names[1]]

        direction = position_info.iloc[time_ind][head_dir_name]

        x_end = x_start + head_dir_radius * np.cos(direction)
        y_end = y_start + head_dir_radius * np.sin(direction)

        head_direction_line = hv.Segments(
            [{"x0": x_start, "y0": y_start, "x1": x_end, "y1": y_end}]
        ).opts(color=animal_color, line_width=3)

        return head_point * head_direction_line

    def get_posterior_quadmesh(time_ind):
        posterior_slice = posterior.isel(time=time_ind)
        return hv.QuadMesh(
            posterior_slice,
            kdims=["x_position", "y_position"],
            vdims=["probability"],
        ).opts(cmap=cmap, alpha=decode_opacity, width=width, height=height)

    def get_most_likely_decoded_position(time_ind):
        return hv.Scatter(
            map_estimate[[time_ind]],
            kdims="x_position",
            vdims="y_position",
        ).opts(color="green", size=10, width=width, height=height)

    # Create player controls
    player = create_player(
        n_time, sampling_frequency=sampling_frequency, width=int(width * 0.7)
    )
    playback_speed_dropdown = create_playback_speed_dropdown(
        player, sampling_frequency=sampling_frequency, width=int(width * 0.2)
    )

    animal_position_dot = hv.DynamicMap(pn.bind(get_animal_position_dot, player))
    # posterior_image = datashade(hv.DynamicMap(pn.bind(get_posterior_quadmesh, player)))
    posterior_image = hv.DynamicMap(pn.bind(get_posterior_quadmesh, player))
    most_likely_decoded_position_dot = hv.DynamicMap(
        pn.bind(get_most_likely_decoded_position, player)
    )

    # Overlay the animal position on the posterior
    decode_plot = pn.Column(
        posterior_image * most_likely_decoded_position_dot * animal_position_dot,
        pn.Row(player, playback_speed_dropdown),
    )

    return decode_plot, player


def create_scrolling_line_plot(
    data,
    ylabel,
    ylim,
    player,
    window_lim=(-0.4, 0.4),
    height_factor=3,
    sampling_frequency=250,
    width=800,
    height=600,
):
    n_time = len(data)
    n_samples = int((window_lim[1] - window_lim[0]) * sampling_frequency)
    half_samples = n_samples // 2

    def plot_func(n):
        start_idx = max(0, n - half_samples)
        end_idx = min(n_time, n + half_samples)
        window = np.asarray(data, dtype=float)[start_idx:end_idx].squeeze()
        line = hv.Curve(
            (np.arange(start_idx - n, end_idx - n) / sampling_frequency, window),
            kdims="time relative to current (s)",
            vdims=ylabel,
        ).opts(
            color="black",
            width=width,
            height=int(height / height_factor),
            ylim=ylim,
            xlim=window_lim,
        )
        return line

    return hv.DynamicMap(pn.bind(plot_func, player))


def create_player(n_time, sampling_frequency=250, width=800):
    base_interval = int(1000 / sampling_frequency)  # ms/frame
    player = pn.widgets.Player(
        start=0,
        end=n_time - 1,
        value=0,
        step=1,
        interval=base_interval,
        name="Time",
        width=width,
        show_loop_controls=False,
    )
    return player


def create_playback_speed_dropdown(player, sampling_frequency=250, width=800):
    # Dropdown to control the speed of the player (interval)
    speed_options = {
        "1/8x": 8,
        "1/4x": 4,
        "1/2x": 2,
        "1x": 1,
        "2x": 0.5,
        "4x": 0.25,
        "8x": 0.125,
    }
    speed_dropdown = pn.widgets.Select(
        name="Playback Speed", options=speed_options, value=1, width=width
    )
    base_interval = 1000 / sampling_frequency  # ms/frame

    # Update the player's interval property based on the dropdown's value
    def update_interval(event):
        player.interval = int(event.new * base_interval)

    speed_dropdown.param.watch(update_interval, "value")

    return speed_dropdown


# extend to a version that will have a dropdown menu for different event times
def create_visualization(
    position_info,
    posterior,
    multiunit_firing_rate,
    non_local_prob,
    map_estimate=None,
    sampling_frequency=250,
    width=800,
    height=600,
    window_lim=(-0.4, 0.4),
    position_names=("x", "y"),
    speed_name="speed",
    head_dir_name="head_direction",
):
    decode_plot, player = create_2D_decoding_plot(
        position_info,
        posterior,
        map_estimate=map_estimate,
        sampling_frequency=sampling_frequency,
        width=width // 2,
        height=height,
        position_names=position_names,
        head_dir_name=head_dir_name,
    )

    multiunit_plot = create_scrolling_line_plot(
        multiunit_firing_rate,
        "Rate (spikes/s)",
        (0, multiunit_firing_rate.to_numpy().max()),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width // 2,
        height=height,
    )
    speed_plot = create_scrolling_line_plot(
        position_info[speed_name],
        "Speed (cm/s)",
        (0, 100),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width // 2,
        height=height,
    )
    non_local_plot = create_scrolling_line_plot(
        non_local_prob,
        "Non-Local prob",
        (0.0, 1.01),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width // 2,
        height=height,
    )

    # Layout the components using Panel
    layout = pn.Row(
        decode_plot,
        pn.Column(multiunit_plot, speed_plot, non_local_plot),
    )

    return layout


import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm.auto import tqdm


def _setup_line(ax, color, xlim, ylim, ylabel, title=None):
    (line,) = ax.plot([], color=color, linewidth=2, animated=True, clip_on=False)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_facecolor("black")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")

    if title is not None:
        ax.set_title(title)

    return line


def _update_line(time_ind, line, window_time, df, window_factor):
    try:
        line.set_data(
            window_time,
            np.asarray(df.iloc[time_ind + window_factor]),
        )
    except IndexError:
        pass


def make_non_local_movie(
    time_slice,
    position_info,
    posterior,
    non_local_prob,
    MUA_firing_rate,
    SWR_zscore,
    movie_name="video_name.mp4",
    sampling_frequency=500,
    video_slowdown=8,
    position_name=["head_position_x", "head_position_y"],
    direction_name="head_orientation",
    vmax=0.07,
):
    posterior = posterior.isel(time=time_slice)

    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    fps = sampling_frequency // video_slowdown
    writer = Writer(fps=fps, bitrate=-1)

    # Set up data
    map_position_ind = posterior.argmax(["x_position", "y_position"])
    map_position = np.stack(
        (
            posterior.x_position[map_position_ind["x_position"]],
            posterior.y_position[map_position_ind["y_position"]],
        ),
        axis=1,
    )

    position = np.asarray(position_info.iloc[time_slice][position_name])
    direction = np.asarray(position_info.iloc[time_slice][direction_name])

    window_size = 501

    window_ind = np.arange(window_size) - window_size // 2
    window_slice = slice(
        time_slice.start + window_ind[0], time_slice.stop + window_ind[-1]
    )
    non_local_prob = non_local_prob.iloc[window_slice]
    MUA_firing_rate = MUA_firing_rate.iloc[window_slice]
    SWR_zscore = SWR_zscore.iloc[window_slice]

    with plt.style.context("dark_background"):
        # Set up plots
        fig = plt.figure(layout="constrained", figsize=(6, 4))
        mosaic_list = [
            ["decode", "Non-Local"],
            ["decode", "SWR"],
            ["decode", "MUA"],
        ]

        ax_dict = fig.subplot_mosaic(mosaic_list, width_ratios=[2, 1])

        ax_dict["decode"].tick_params(colors="white", which="both")
        ax_dict["decode"].spines["bottom"].set_color("white")
        ax_dict["decode"].spines["left"].set_color("white")

        position_dot = ax_dict["decode"].scatter(
            [],
            [],
            s=80,
            zorder=102,
            color="magenta",
            label="actual position",
            animated=True,
        )
        (position_line,) = ax_dict["decode"].plot(
            [], [], color="magenta", linewidth=5, animated=True
        )
        map_dot = ax_dict["decode"].scatter(
            [],
            [],
            s=80,
            zorder=102,
            color="green",
            label="decoded position",
            animated=True,
        )
        (map_line,) = ax_dict["decode"].plot(
            [], [], "green", linewidth=3, animated=True
        )

        mesh = posterior.isel(time=0).plot(
            x="x_position",
            y="y_position",
            vmin=0.0,
            vmax=vmax,
            ax=ax_dict["decode"],
            animated=True,
            add_colorbar=False,
        )
        ax_dict["decode"].set_xlabel("")
        ax_dict["decode"].set_ylabel("")
        ax_dict["decode"].spines["top"].set_color("black")
        ax_dict["decode"].spines["right"].set_color("black")
        xlim = (
            position_info[position_name[0]].min() - 10,
            position_info[position_name[0]].max() + 10,
        )
        ylim = (
            position_info[position_name[1]].min() - 10,
            position_info[position_name[1]].max() + 10,
        )

        ax_dict["decode"].set_xlim(xlim)
        ax_dict["decode"].set_ylim(ylim)

        title = ax_dict["decode"].set_title(
            f"time = {posterior.isel(time=0).time.values:0.2f}",
        )
        fontprops = fm.FontProperties(size=16)
        scalebar = AnchoredSizeBar(
            ax_dict["decode"].transData,
            20,
            "20 cm",
            "lower right",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
        )

        ax_dict["decode"].add_artist(scalebar)
        ax_dict["decode"].axis("off")

        # Non-Local
        line_xlim = (
            window_ind[0] / sampling_frequency,
            window_ind[-1] / sampling_frequency,
        )
        non_local_line = _setup_line(
            ax=ax_dict["Non-Local"],
            color=COLORS["Non-Local"],
            xlim=line_xlim,
            ylim=(0.0, 1.05),
            ylabel="Non-Local\nProb.",
            title="Non-Local",
        )

        # SWR
        ax_dict["SWR"].axhline(0, color="white", linestyle="--")
        swr_line = _setup_line(
            ax=ax_dict["SWR"],
            color=COLORS["SWR"],
            xlim=line_xlim,
            ylim=(-3.0, 10.0),
            ylabel="SWR [std]",
            title="SWR",
        )

        # MUA
        mua_line = _setup_line(
            ax=ax_dict["MUA"],
            color=COLORS["MUA"],
            xlim=line_xlim,
            ylim=(0.0, 3000.0),
            ylabel="MUA [sp/s]",
            title="MUA",
        )
        ax_dict["MUA"].set_xlabel("Time [s]")

        n_frames = posterior.shape[0]
        progress_bar = tqdm()
        progress_bar.reset(total=n_frames)
        window_time = window_ind / sampling_frequency
        window_factor = (window_size // 2) + window_ind

        def _update_plot(time_ind):
            start_ind = max(0, time_ind - 5)
            time_slice = slice(start_ind, time_ind)

            position_dot.set_offsets(position[time_ind])

            r = 4
            position_line.set_data(
                [
                    position[time_ind, 0],
                    position[time_ind, 0] + r * np.cos(direction[time_ind]),
                ],
                [
                    position[time_ind, 1],
                    position[time_ind, 1] + r * np.sin(direction[time_ind]),
                ],
            )

            map_dot.set_offsets(map_position[time_ind])
            map_line.set_data(map_position[time_slice, 0], map_position[time_slice, 1])

            mesh.set_array(posterior.isel(time=time_ind).values.ravel(order="F"))

            title.set_text(f"time = {posterior.isel(time=time_ind).time.values:0.2f}")

            _update_line(
                time_ind=time_ind,
                line=non_local_line,
                window_time=window_time,
                df=non_local_prob,
                window_factor=window_factor,
            )
            _update_line(
                time_ind=time_ind,
                line=swr_line,
                window_time=window_time,
                df=SWR_zscore,
                window_factor=window_factor,
            )
            _update_line(
                time_ind=time_ind,
                line=mua_line,
                window_time=window_time,
                df=MUA_firing_rate,
                window_factor=window_factor,
            )

            progress_bar.update()

            return (
                position_dot,
                position_line,
                map_dot,
                map_line,
                mesh,
                title,
                non_local_line,
                swr_line,
                mua_line,
            )

        movie = animation.FuncAnimation(
            fig, _update_plot, frames=n_frames, interval=1000 / fps, blit=True
        )
        if movie_name is not None:
            movie.save(movie_name, writer=writer, dpi=200)

        plt.close(fig)


def plot_graph_as_1D(
    track_graph,
    ax=None,
    edge_order=None,
    edge_spacing=0.0,
    reward_well_nodes=None,
    other_axis_start=0,
    edge_colors=None,
    reward_well_size=10,
    edege_linewidth=2,
):
    if ax is None:
        ax = plt.gca()
    # If no edge_order is given, then arange edges in the order passed to
    # construct the track graph
    if edge_order is None:
        edge_order = np.asarray(track_graph.edges)
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        edge_colors = np.array(matplotlib.colormaps.get_cmap("tab10").colors)

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

    start_node_linear_position = 0.0

    for edge_ind, edge in enumerate(edge_order):
        end_node_linear_position = (
            start_node_linear_position + track_graph.edges[edge]["distance"]
        )
        ax.plot(
            (other_axis_start, other_axis_start),
            (start_node_linear_position, end_node_linear_position),
            color=edge_colors[edge_ind],
            clip_on=False,
            zorder=7,
            linewidth=edege_linewidth,
        )
        if edge[0] in reward_well_nodes:
            ax.scatter(
                other_axis_start,
                start_node_linear_position,
                color=edge_colors[edge_ind],
                s=reward_well_size,
                zorder=10,
                clip_on=False,
            )
        if edge[1] in reward_well_nodes:
            ax.scatter(
                other_axis_start,
                end_node_linear_position,
                color=edge_colors[edge_ind],
                s=reward_well_size,
                zorder=10,
                clip_on=False,
            )

        try:
            start_node_linear_position += (
                track_graph.edges[edge]["distance"] + edge_spacing[edge_ind]
            )
        except IndexError:
            pass
