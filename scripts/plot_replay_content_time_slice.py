#!/usr/bin/env python
"""Time-slice visualization showing how replay content covariates align with
the posterior and what the conditional Poisson model sees for a single event.

Left column (sharing a time axis):
    1. Ripple consensus trace (z-scored)
    2. Non-local state probability
    3. Decoder posterior with 1D track overlay, animal position, and trial
       boundary annotations
    4. Per-arm posterior mass (stacked area)
    5. Per-arm predictor markers at the event midpoint

Right column:
    A. Per-arm response (hard MAP counts and soft posterior-mass counts)
    B. Feature matrix heatmap — the full design-matrix row the model sees
    C. dist_to_arm text table

Use ``--event-type`` to find different kinds of events:
    off_arm  : replay to an arm far from the animal's current position
    switch   : off-arm replay during immobility after a patch switch
    omission : off-arm replay during immobility after a reward omission
"""

from __future__ import annotations

import argparse
import copy
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus import load_results
from continuum_swr_replay.analysis.hippocampus.replay_content.predictors import (
    compute_cumulative_reward_for_event,
    compute_dist_to_arm,
    compute_visit_recency_for_event,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_HIERARCHY,
)
from continuum_swr_replay.analysis.replay.spatial_mapping import (
    arm_to_well,
    well_to_arm_set,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from continuum_swr_replay.parameters import COLORS
from continuum_swr_replay.visualization.palettes import build_arm_color_map
from continuum_swr_replay.visualization.plots import plot_graph_as_1D

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())
PATCH_PALETTE = {1: "#4e79a7", 2: "#e15759", 3: "#59a14f"}

# Full M7 predictor list (what the model actually sees)
M7_PREDICTORS = list(MODEL_HIERARCHY["M7"].predictors)


# ── Posterior helpers ────────────────────────────────────────────────────


def compute_non_local_posterior(results, env, detector, time_slice):
    """Non-local state posterior, renormalized over real track bins only."""
    acausal_posterior = results.sel(time=time_slice).acausal_posterior.values
    state_ind = detector.state_ind_
    non_local_inds = np.nonzero(
        ["Non-Local" in s for s in detector.state_names]
    )[0]

    n_time = acausal_posterior.shape[0]
    n_bins = len(env.place_bin_centers_)
    cond_post = np.zeros((n_time, n_bins))
    for nl_ind in non_local_inds:
        cond_post += acausal_posterior[:, state_ind == nl_ind]

    edge_ids = env.place_bin_centers_nodes_df_["edge_id"].values
    cond_post[:, edge_ids == -1] = 0.0
    cond_post /= np.nansum(cond_post, axis=1, keepdims=True) + 1e-12
    return cond_post


def per_bin_replay_arm(cond_post, env):
    """MAP arm per timebin (assumes junctions already zeroed)."""
    map_ind = np.argmax(cond_post, axis=1)
    edge_ids = env.place_bin_centers_nodes_df_["edge_id"].values
    return edge_ids[map_ind].astype(float)


def posterior_mass_per_arm(cond_post, env):
    """Sum posterior mass over bins per arm → (n_time, n_arms)."""
    edge_ids = env.place_bin_centers_nodes_df_["edge_id"].values
    n_time = cond_post.shape[0]
    arm_mass = np.zeros((n_time, len(ALL_ARMS)))
    for i, arm in enumerate(ALL_ARMS):
        arm_bins = edge_ids == arm
        if arm_bins.any():
            arm_mass[:, i] = cond_post[:, arm_bins].sum(axis=1)
    return arm_mass


# ── Event selection ─────────────────────────────────────────────────────


def _nl_immobile_runs(data, results):
    """Return (starts, ends, durations) of non-local immobile runs."""
    non_local_prob = (
        results.acausal_state_probabilities.sel(states="Non-Local Continuous")
        + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    ).values
    time = results.time.values
    pos_times = data["position_info"].index.values
    speed_at_time = np.interp(
        time, pos_times, data["position_info"].head_speed.values
    )
    is_nl = (non_local_prob > DEFAULT_NON_LOCAL_THRESHOLD) & (
        speed_at_time < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    )
    diffs = np.diff(is_nl.astype(int), prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    durations = time[ends - 1] - time[starts]
    return starts, ends, durations, time


def find_interesting_window(
    data, results, env, detector,
    event_type: str = "off_arm",
    window_pad: float = 4.0,
    rank: int = 0,
):
    """Score non-local immobile events by event_type and return the rank-th best."""
    starts, ends, durations, time = _nl_immobile_runs(data, results)
    if len(starts) == 0:
        raise ValueError("No non-local immobile events found")

    ripple_times = data["ripple_times"]
    cond_post = compute_non_local_posterior(
        results, env, detector, slice(time[0], time[-1])
    )
    arm_mass_all = posterior_mass_per_arm(cond_post, env)

    pos_info = data["position_info"]
    seg_at_time = np.interp(
        time, pos_info.index.values, pos_info["track_segment_id"].values,
    )

    trials = data["trials"]
    trial_start_times = trials["start_time"].values

    scored = []
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        if durations[i] < 0.1 or durations[i] > 1.5:
            continue
        t_start_i, t_end_i = time[s], time[e - 1]
        event_mid = (t_start_i + t_end_i) / 2

        swr_overlap = (
            (ripple_times.start_time <= t_end_i)
            & (ripple_times.end_time >= t_start_i)
        ).any()
        if not swr_overlap:
            continue

        ev_segs = seg_at_time[s:e]
        if len(ev_segs) == 0 or np.all(np.isnan(ev_segs)):
            continue
        current_arm = int(np.round(np.nanmedian(ev_segs)))
        if current_arm not in ALL_ARMS:
            continue

        ev_mass = arm_mass_all[s:e]
        total_mass = ev_mass.sum()
        if total_mass < 1e-6:
            continue
        on_arm_mass = ev_mass[:, ALL_ARMS.index(current_arm)].sum()
        off_arm_frac = 1 - on_arm_mass / total_mass

        # Context flags (in-progress trip logic)
        current_trip_idx = (
            int(np.searchsorted(trial_start_times, event_mid, side="right")) - 1
        )
        if current_trip_idx < 0:
            continue
        trip = trials.iloc[current_trip_idx]
        is_after_switch = bool(trip.is_patch_change)
        is_after_omission = not bool(trip.is_reward)

        score = off_arm_frac * np.sqrt(durations[i])
        if event_type == "switch":
            if not is_after_switch:
                continue
        elif event_type == "omission":
            if not is_after_omission:
                continue
        elif event_type == "off_arm":
            pass
        else:
            raise ValueError(f"Unknown event_type: {event_type}")

        scored.append(
            {
                "score": score,
                "idx": i,
                "off_arm_frac": off_arm_frac,
                "duration": durations[i],
                "is_after_switch": is_after_switch,
                "is_after_omission": is_after_omission,
                "current_arm": current_arm,
            }
        )

    if not scored:
        raise ValueError(f"No events match event_type={event_type}")
    scored.sort(key=lambda x: -x["score"])
    if rank >= len(scored):
        raise ValueError(
            f"Only {len(scored)} matching events, requested rank {rank}"
        )
    chosen = scored[rank]
    logging.info(
        "Event rank %d (%s): score=%.3f off-arm=%.2f dur=%.3fs "
        "switch=%s omission=%s current_arm=%d",
        rank, event_type, chosen["score"], chosen["off_arm_frac"],
        chosen["duration"], chosen["is_after_switch"],
        chosen["is_after_omission"], chosen["current_arm"],
    )

    best_idx = chosen["idx"]
    t_start = time[starts[best_idx]]
    t_end = time[ends[best_idx] - 1]
    return slice(t_start - window_pad, t_end + window_pad), slice(t_start, t_end)


# ── Event context (in-progress trip semantics) ──────────────────────────


def get_event_context(data, event_slice, track_graph):
    """Event context using the in-progress trip."""
    trials = data["trials"]
    if trials is None or len(trials) == 0:
        return None

    event_time = (event_slice.start + event_slice.stop) / 2
    pos = data["position_info"]
    pos_in_ev = pos.loc[event_slice]

    if "track_segment_id" in pos_in_ev.columns and len(pos_in_ev) > 0:
        seg_modes = pos_in_ev["track_segment_id"].dropna()
        current_arm = (
            float(seg_modes.mode().iloc[0]) if len(seg_modes) > 0 else np.nan
        )
    else:
        current_arm = np.nan
    current_well = arm_to_well(current_arm, track_graph)

    in_progress = trials[
        (trials["start_time"] <= event_time)
        & (trials["end_time"] >= event_time)
    ]
    if len(in_progress) > 0:
        current_trip = trials.loc[in_progress.index[0]]
        pos_idx = trials.index.get_loc(in_progress.index[0])
    else:
        completed = trials[trials["end_time"] < event_time]
        if len(completed) == 0:
            return None
        current_trip = trials.loc[completed.index[-1]]
        pos_idx = trials.index.get_loc(completed.index[-1])

    prev_from_well = current_trip["from_well"]
    next_to_well = (
        trials.iloc[pos_idx + 1]["to_well"]
        if pos_idx + 1 < len(trials)
        else np.nan
    )
    is_after_switch = bool(current_trip["is_patch_change"])
    is_after_omission = not bool(current_trip["is_reward"])

    rewarded = trials[trials["is_reward"] & (trials["end_time"] <= event_time)]
    last_rewarded_well = (
        rewarded.iloc[-1]["to_well"] if len(rewarded) > 0 else np.nan
    )

    return {
        "current_well": current_well,
        "current_arm": current_arm,
        "prev_from_well": prev_from_well,
        "next_to_well": next_to_well,
        "last_rewarded_well": last_rewarded_well,
        "is_after_switch": is_after_switch,
        "is_after_omission": is_after_omission,
    }


# ── Feature matrix (what the model sees) ────────────────────────────────


def build_feature_matrix(
    arms_in_linear_order,
    current_arm,
    prev_arms,
    next_arms,
    rewarded_arms,
    vr,
    cr,
    dists,
    time_since_stop,
    is_swr,
    is_after_switch,
    is_after_omission,
):
    """Build a (n_arms × n_predictors) matrix in the order arms_in_linear_order."""
    n_arms = len(arms_in_linear_order)
    features = {}

    # Main effects
    features["is_current"] = np.array(
        [1.0 if a == current_arm else 0.0 for a in arms_in_linear_order]
    )
    features["is_prev"] = np.array(
        [1.0 if a in prev_arms else 0.0 for a in arms_in_linear_order]
    )
    features["is_next"] = np.array(
        [1.0 if a in next_arms else 0.0 for a in arms_in_linear_order]
    )
    features["is_last_rewarded"] = np.array(
        [1.0 if a in rewarded_arms else 0.0 for a in arms_in_linear_order]
    )
    features["visit_recency"] = np.array(
        [vr.get(a, np.nan) for a in arms_in_linear_order]
    )
    features["cumulative_reward"] = np.array(
        [cr.get(a, np.nan) for a in arms_in_linear_order]
    )
    features["dist_to_arm"] = np.array(
        [dists.get(a, np.nan) for a in arms_in_linear_order]
    )

    # Temporal interactions: time_since_stop × {is_current, is_prev, is_next}
    features["time_x_current"] = time_since_stop * features["is_current"]
    features["time_x_prev"] = time_since_stop * features["is_prev"]
    features["time_x_next"] = time_since_stop * features["is_next"]

    # SWR interactions
    features["swr_x_current"] = is_swr * features["is_current"]
    features["swr_x_prev"] = is_swr * features["is_prev"]
    features["swr_x_next"] = is_swr * features["is_next"]
    features["swr_x_rewarded"] = is_swr * features["is_last_rewarded"]

    # Context interactions
    features["switch_x_current"] = is_after_switch * features["is_current"]
    features["switch_x_prev"] = is_after_switch * features["is_prev"]
    features["omission_x_rewarded"] = (
        is_after_omission * features["is_last_rewarded"]
    )

    # Order columns according to M7 predictor list for consistent display
    matrix = np.zeros((n_arms, len(M7_PREDICTORS)))
    for j, pred in enumerate(M7_PREDICTORS):
        matrix[:, j] = features[pred]
    return matrix, M7_PREDICTORS


# ── Main plot ────────────────────────────────────────────────────────────


def generate_plot(
    nwb_file_name: str,
    epoch_name: str,
    output_path: str,
    event_type: str = "off_arm",
    rank: int = 0,
) -> None:
    logging.info("Loading session data...")
    data = load_data(nwb_file_name, epoch_name)
    results, detector, env = load_results(nwb_file_name, epoch_name)

    logging.info("Finding interesting event (type=%s, rank=%d)...", event_type, rank)
    time_slice, event_only_slice = find_interesting_window(
        data, results, env, detector,
        event_type=event_type, window_pad=4.0, rank=rank,
    )

    results_window = results.sel(time=time_slice)
    time = results_window.time.values
    t0 = time_slice.start
    t1 = time_slice.stop

    cond_post = compute_non_local_posterior(results, env, detector, time_slice)
    place_bin_centers = env.place_bin_centers_

    non_local_prob = (
        results_window.acausal_state_probabilities.sel(states="Non-Local Continuous")
        + results_window.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    ).values

    pos_in_window = data["position_info"].loc[time_slice]
    ripple_z = (
        data["zscored_ripple_consensus_trace"].loc[time_slice]
        if "zscored_ripple_consensus_trace" in data
        else None
    )

    bin_arms = per_bin_replay_arm(cond_post, env)
    arm_mass = posterior_mass_per_arm(cond_post, env)

    track_graph = data["track_graph"]
    linear_edge_order = data["linear_edge_order"]
    linear_edge_spacing = data["linear_edge_spacing"]
    arm_color_map = build_arm_color_map(track_graph, linear_edge_order)
    arms_in_linear_order = [
        int(track_graph.edges[tuple(edge)]["edge_id"])
        for edge in linear_edge_order
        if "edge_id" in track_graph.edges[tuple(edge)]
    ]

    # Context + arm sets
    ctx = get_event_context(data, event_only_slice, track_graph)
    if ctx is None:
        raise RuntimeError("No trial context for this event")

    current_arms = (
        {int(ctx["current_arm"])} if not np.isnan(ctx["current_arm"]) else set()
    )
    prev_arms = well_to_arm_set(ctx["prev_from_well"], track_graph)
    next_arms = well_to_arm_set(ctx["next_to_well"], track_graph)
    rewarded_arms = well_to_arm_set(ctx["last_rewarded_well"], track_graph)
    dists = compute_dist_to_arm(ctx["current_well"], track_graph)

    # Continuous per-arm predictors
    event_mid = (event_only_slice.start + event_only_slice.stop) / 2
    vr = compute_visit_recency_for_event(event_mid, data["trials"], track_graph)
    cr = compute_cumulative_reward_for_event(event_mid, data["trials"], track_graph)

    # time_since_stop: seconds from the start of the immobility bout
    event_duration = event_only_slice.stop - event_only_slice.start
    # Use the non-local immobility onset: walk back from event start while speed is low
    pos = data["position_info"]
    before_event = pos.loc[: event_only_slice.start]
    moving = before_event["head_speed"] >= DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    if moving.any():
        last_move_time = before_event.index[moving][-1]
        time_since_stop = float(event_mid - last_move_time)
    else:
        time_since_stop = float(event_mid - before_event.index[0])

    # Count the SWR flag for the event
    ripple_times = data["ripple_times"]
    is_swr_flag = int(
        (
            (ripple_times.start_time <= event_only_slice.stop)
            & (ripple_times.end_time >= event_only_slice.start)
        ).any()
    )

    # Build feature matrix (9 x n_predictors)
    feature_matrix, feature_names = build_feature_matrix(
        arms_in_linear_order,
        current_arm=int(ctx["current_arm"]) if not np.isnan(ctx["current_arm"]) else -1,
        prev_arms=prev_arms,
        next_arms=next_arms,
        rewarded_arms=rewarded_arms,
        vr=vr,
        cr=cr,
        dists=dists,
        time_since_stop=time_since_stop,
        is_swr=is_swr_flag,
        is_after_switch=int(ctx["is_after_switch"]),
        is_after_omission=int(ctx["is_after_omission"]),
    )

    # Per-arm counts for the event window
    ev_mask = (time >= event_only_slice.start) & (time <= event_only_slice.stop)
    ev_bin_arms = bin_arms[ev_mask]
    arm_counts_hard = {arm: int(np.sum(ev_bin_arms == arm)) for arm in ALL_ARMS}
    total_hard = sum(arm_counts_hard.values())
    arm_mass_event = arm_mass[ev_mask].sum(axis=0)
    arm_mass_total = arm_mass_event.sum()

    # ── Build figure ──
    fig = plt.figure(figsize=(16, 12), dpi=120, constrained_layout=True)
    outer = GridSpec(1, 2, figure=fig, width_ratios=[2.6, 1.0], wspace=0.08)

    # Left column: 5 stacked time-series rows sharing x
    left_gs = GridSpecFromSubplotSpec(
        5, 1,
        subplot_spec=outer[0, 0],
        height_ratios=[0.6, 0.6, 2.4, 1.0, 1.6],
        hspace=0.08,
    )
    ax_ripple = fig.add_subplot(left_gs[0])
    ax_nlprob = fig.add_subplot(left_gs[1], sharex=ax_ripple)
    ax_post = fig.add_subplot(left_gs[2], sharex=ax_ripple)
    ax_armbars = fig.add_subplot(left_gs[3], sharex=ax_ripple)
    ax_preds_time = fig.add_subplot(left_gs[4], sharex=ax_ripple)

    # Right column: counts, feature matrix, dist_to_arm
    right_gs = GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer[0, 1],
        height_ratios=[1.5, 2.2, 1.3],
        hspace=0.35,
    )
    ax_counts = fig.add_subplot(right_gs[0])
    ax_features = fig.add_subplot(right_gs[1])
    ax_dist = fig.add_subplot(right_gs[2])

    # Shared explicit x-limits on all time-series panels (relative to t0)
    xlim = (0.0, t1 - t0)
    ax_ripple.set_xlim(*xlim)

    # ── Panel 1: Ripple consensus trace ──
    if ripple_z is not None:
        r_t = ripple_z.index.values - t0
        r_v = ripple_z.values.squeeze()
        ax_ripple.fill_between(r_t, r_v, color="black", linewidth=0)
    ax_ripple.set_ylabel("Ripple\n(z)", fontsize=9)
    ax_ripple.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax_ripple.axhline(2, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_ripple.tick_params(labelbottom=False)

    # Highlight SWRs across all time-series panels
    swr_in_window = ripple_times[
        (ripple_times.start_time <= t1) & (ripple_times.end_time >= t0)
    ]
    for _, r in swr_in_window.iterrows():
        for ax in [ax_ripple, ax_nlprob, ax_post, ax_armbars, ax_preds_time]:
            ax.axvspan(
                max(r.start_time - t0, 0),
                min(r.end_time - t0, t1 - t0),
                alpha=0.15, color="red", zorder=0,
            )

    # Highlight the non-local event
    for ax in [ax_nlprob, ax_post, ax_armbars, ax_preds_time]:
        ax.axvspan(
            event_only_slice.start - t0, event_only_slice.stop - t0,
            alpha=0.12, color="gold", zorder=0,
        )

    # Trial boundaries + labels on posterior
    trials_in_window = data["trials"][
        (data["trials"]["start_time"] <= t1)
        & (data["trials"]["end_time"] >= t0)
    ]
    for _, trial in trials_in_window.iterrows():
        ax_post.axvline(
            trial.start_time - t0,
            color="dimgray", linewidth=0.7, linestyle=":", alpha=0.6, zorder=1,
        )
        mid = (
            max(trial.start_time, t0) + min(trial.end_time, t1)
        ) / 2 - t0
        from_w = (
            int(trial.from_well) if not np.isnan(trial.from_well) else "?"
        )
        to_w = int(trial.to_well) if not np.isnan(trial.to_well) else "?"
        rew = "✓" if trial.is_reward else "·"
        sw = "★" if trial.is_patch_change else ""
        ax_post.text(
            mid, place_bin_centers.max() * 1.02,
            f"{from_w}→{to_w}{rew}{sw}",
            ha="center", va="bottom", fontsize=7,
            color="dimgray", clip_on=False,
        )

    # ── Panel 2: Non-local probability ──
    ax_nlprob.plot(
        time - t0, non_local_prob,
        color=COLORS.get("Non-Local", "#CC6600"), linewidth=1.3,
    )
    ax_nlprob.fill_between(
        time - t0, 0, non_local_prob,
        color=COLORS.get("Non-Local", "#CC6600"), alpha=0.2,
    )
    ax_nlprob.axhline(
        DEFAULT_NON_LOCAL_THRESHOLD, color="gray",
        linewidth=0.6, linestyle="--", alpha=0.7,
    )
    ax_nlprob.set_ylabel("Non-Local\nprob.", fontsize=9)
    ax_nlprob.set_ylim(0, 1.05)
    ax_nlprob.set_yticks([0, 0.5, 1])
    ax_nlprob.tick_params(labelbottom=False)

    # ── Panel 3: Posterior heatmap with animal position + 1D track overlay ──
    cmap = copy.copy(plt.get_cmap("bone_r"))
    cmap.set_bad("lightgrey")
    post_masked = np.ma.masked_where(cond_post.T < 1e-4, cond_post.T)
    t_mesh, x_mesh = np.meshgrid(time - t0, place_bin_centers)
    ax_post.pcolormesh(
        t_mesh, x_mesh, post_masked,
        vmin=0, vmax=0.05, cmap=cmap, shading="auto", rasterized=True,
    )

    # 1D linearized track stripe
    graph_overlay_x = -0.02 * (t1 - t0)
    plot_graph_as_1D(
        track_graph=track_graph, ax=ax_post,
        edge_order=linear_edge_order, edge_spacing=linear_edge_spacing,
        reward_well_nodes=list(range(6)),
        other_axis_start=graph_overlay_x,
        edge_linewidth=6, reward_well_size=35,
    )
    cum_pos = 0.0
    n_edges = len(linear_edge_order)
    edge_spacing_list = (
        [linear_edge_spacing] * (n_edges - 1)
        if isinstance(linear_edge_spacing, (int, float))
        else list(linear_edge_spacing)
    )
    for edge_ind, edge in enumerate(linear_edge_order):
        d = track_graph.edges[tuple(edge)]["distance"]
        eid = track_graph.edges[tuple(edge)].get("edge_id")
        if eid is not None:
            ax_post.text(
                graph_overlay_x - 0.01 * (t1 - t0),
                cum_pos + d / 2, f"a{eid}",
                fontsize=8, va="center", ha="right",
                color=arm_color_map[int(eid)], fontweight="bold",
            )
        cum_pos += d + (edge_spacing_list[edge_ind] if edge_ind < n_edges - 1 else 0)

    ax_post.scatter(
        pos_in_window.index.values - t0, pos_in_window.linear_position.values,
        s=6, color="magenta", edgecolors="none", zorder=5,
    )
    ax_post.set_ylabel("Linear position (cm)", fontsize=9)
    ax_post.tick_params(labelbottom=False)

    # ── Panel 4: Posterior mass per arm (stacked area) ──
    ax_armbars.stackplot(
        time - t0, arm_mass.T,
        colors=[arm_color_map[a] for a in ALL_ARMS],
        edgecolor="none", alpha=0.95,
    )
    ax_armbars.set_ylim(0, 1.01)
    ax_armbars.set_yticks([0, 0.5, 1])
    ax_armbars.set_ylabel("Posterior\nmass/arm", fontsize=9)
    ax_armbars.tick_params(labelbottom=False)

    # ── Panel 5: Predictor markers at event midpoint ──
    n_arms_lin = len(arms_in_linear_order)
    ax_preds_time.set_ylim(-0.5, n_arms_lin - 0.5)
    ax_preds_time.set_yticks(range(n_arms_lin))
    ax_preds_time.set_yticklabels(
        [f"arm {a}" for a in arms_in_linear_order], fontsize=8,
    )
    ax_preds_time.set_ylabel("Predictors\n× arm", fontsize=9)
    for i, arm in enumerate(arms_in_linear_order):
        ax_preds_time.axhspan(
            i - 0.45, i + 0.45, alpha=0.08,
            color=PATCH_PALETTE[TRACK_SEGMENT_TO_PATCH[arm]],
        )
    pred_info = {
        "is_current":       ("o", "#1b9e77", current_arms,   -0.45),
        "is_prev":          ("^", "#7570b3", prev_arms,      -0.18),
        "is_next":          ("v", "#d95f02", next_arms,       0.09),
        "is_last_rewarded": ("*", "#e7298a", rewarded_arms,   0.36),
    }
    ev_mid_rel = (event_only_slice.start + event_only_slice.stop) / 2 - t0
    for _label, (marker, color, arms, rel_off) in pred_info.items():
        for arm in arms:
            if arm not in arms_in_linear_order:
                continue
            i = arms_in_linear_order.index(arm)
            ax_preds_time.plot(
                ev_mid_rel + rel_off * event_duration, i,
                marker=marker, color=color, markersize=12,
                markeredgecolor="black", markeredgewidth=0.5, zorder=3,
            )
    ax_preds_time.set_xlabel(
        f"Time (s) — t=0 at {t0:.2f}s abs", fontsize=9,
    )

    # ── Right A: per-arm response bars ──
    ys = np.arange(n_arms_lin)
    hard_counts = np.array(
        [arm_counts_hard[a] for a in arms_in_linear_order]
    )
    soft_counts = np.array(
        [arm_mass_event[ALL_ARMS.index(a)] for a in arms_in_linear_order]
    )
    bar_colors = [arm_color_map[a] for a in arms_in_linear_order]
    bar_h = 0.4
    ax_counts.barh(
        ys - bar_h / 2, hard_counts, bar_h,
        color=bar_colors, edgecolor="black", linewidth=0.4,
        label="hard MAP",
    )
    ax_counts.barh(
        ys + bar_h / 2, soft_counts, bar_h,
        color=bar_colors, edgecolor="black", linewidth=0.4,
        alpha=0.5, hatch="///", label="soft mass",
    )
    ax_counts.set_yticks(ys)
    ax_counts.set_yticklabels(
        [f"arm {a}" for a in arms_in_linear_order], fontsize=7,
    )
    ax_counts.set_xlabel("count / mass", fontsize=8)
    ax_counts.set_title(
        f"Response:  hard L_e={total_hard}  soft L_e={arm_mass_total:.1f}",
        fontsize=9, fontweight="bold",
    )
    ax_counts.legend(fontsize=6, loc="lower right")

    # ── Right B: feature matrix heatmap (what the model sees) ──
    # Matrix: rows = arms (in linear order, bottom = first in linear_edge_order),
    # columns = predictors (in M7 order).
    # Normalize each column separately for visual clarity (continuous vs binary).
    fm_display = feature_matrix.copy()
    # Column-wise normalize to [0, 1] for display, preserving 0 as 0
    col_max = np.nanmax(np.abs(fm_display), axis=0)
    col_max[col_max == 0] = 1.0
    fm_norm = fm_display / col_max

    ax_features.imshow(
        fm_norm, aspect="auto", cmap="viridis", origin="lower",
        vmin=0, vmax=1, interpolation="none",
    )
    ax_features.set_yticks(range(n_arms_lin))
    ax_features.set_yticklabels(
        [f"a{a}" for a in arms_in_linear_order], fontsize=7,
    )
    ax_features.set_xticks(range(len(feature_names)))
    ax_features.set_xticklabels(
        feature_names, fontsize=6, rotation=90,
    )
    ax_features.set_title(
        "Feature matrix (M7) — what the model sees\n"
        "(each column normalized by its max)",
        fontsize=9, fontweight="bold",
    )

    # Annotate cells with the actual (unnormalized) values
    for i in range(n_arms_lin):
        for j in range(len(feature_names)):
            val = feature_matrix[i, j]
            if np.isnan(val):
                text = "—"
            elif feature_names[j] in (
                "visit_recency", "cumulative_reward", "dist_to_arm"
            ) or feature_names[j].startswith(("time_x_", "swr_x_")):
                text = f"{val:.0f}"
            else:
                text = f"{int(val)}" if val in (0, 1) else f"{val:.1f}"
            text_color = "white" if fm_norm[i, j] < 0.5 else "black"
            ax_features.text(
                j, i, text,
                ha="center", va="center",
                fontsize=5, color=text_color,
            )

    # ── Right C: dist_to_arm + event metadata ──
    ax_dist.axis("off")
    meta_lines = [
        f"session:  {nwb_file_name.replace('_.nwb','')}  {epoch_name}",
        f"event:    [{event_only_slice.start:.3f}, {event_only_slice.stop:.3f}] s  "
        f"({event_duration*1000:.0f} ms)",
        f"animal:   well {int(ctx['current_well']) if not np.isnan(ctx['current_well']) else '?'}  "
        f"(arm {int(ctx['current_arm']) if not np.isnan(ctx['current_arm']) else '?'})",
        f"prev from:     well {int(ctx['prev_from_well']) if not np.isnan(ctx['prev_from_well']) else '?'}",
        f"next to:       well {int(ctx['next_to_well']) if not np.isnan(ctx['next_to_well']) else '?'}",
        f"last rewarded: well {int(ctx['last_rewarded_well']) if not np.isnan(ctx['last_rewarded_well']) else '?'}",
        f"is_swr:        {is_swr_flag}",
        f"is_after_switch:    {int(ctx['is_after_switch'])}",
        f"is_after_omission:  {int(ctx['is_after_omission'])}",
        f"time_since_stop:    {time_since_stop:.2f} s",
    ]
    y_cursor = 0.98
    for line in meta_lines:
        ax_dist.text(
            0.01, y_cursor, line,
            fontsize=7, family="monospace",
            transform=ax_dist.transAxes, va="top",
        )
        y_cursor -= 0.095

    fig.suptitle(
        f"Replay content: {event_type.replace('_', '-')} event "
        f"(rank {rank})  —  {nwb_file_name.replace('_.nwb','')} {epoch_name}",
        fontsize=12, fontweight="bold",
    )

    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    logging.info("Saved %s", output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb-file-name", type=str, default="wilbur20210329_.nwb")
    parser.add_argument("--epoch-name", type=str, default="02_r1")
    parser.add_argument(
        "--output", type=str, default="replay_content_time_slice.png"
    )
    parser.add_argument(
        "--event-type", type=str, default="off_arm",
        choices=["off_arm", "switch", "omission"],
    )
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    generate_plot(
        args.nwb_file_name, args.epoch_name, args.output,
        event_type=args.event_type, rank=args.rank,
    )


if __name__ == "__main__":
    main()
