#!/usr/bin/env python
"""Generate Figure 3: Non-local event characteristics and physiological correlates.

This script generates panels showing:
- Event duration distributions
- Spatial properties (distance, HPD)
- Decoded position metrics
- Event-triggered averages of physiological signals (theta, SWR, MUA)

Usage:
    python scripts/generate_figure03.py
    python scripts/generate_figure03.py --output-dir paper/figures --format pdf
"""

# %%
# %config InlineBackend.figure_format = 'retina'
import argparse
import logging
import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import label
from scipy.stats import sem

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    BINS_DISTANCE_CM,
    BINS_DURATION_MS,
    BINS_HPD_CM,
    BINS_SPEED_CM_PER_SEC,
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus import (
    get_event_durations_ms,
    get_event_start_end_times,
    get_immobile_events,
    get_signal_windows,
)
from continuum_swr_replay.parameters import (
    COLORS,
    PROCESSED_DATA_DIR,
    SAMPLING_FREQUENCY,
)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output figures (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output file format (default: pdf)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for raster formats (default: 300)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override PROCESSED_DATA_DIR for loading data",
    )
    return parser.parse_args()


# Parse arguments (but don't require them for notebook use)
try:
    args = parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
except SystemExit:
    # Allow script to run in notebook mode without arguments
    args = argparse.Namespace(output_dir=None, format="pdf", dpi=300, data_dir=None)

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def load_data_over_time():
    # Load all relevant data files
    data_dir = args.data_dir if args.data_dir else PROCESSED_DATA_DIR
    data_path = data_dir.glob("*_HPC_80_data_over_time.pkl")

    data_over_time = pd.concat([pd.read_pickle(p) for p in data_path])
    data_over_time["Non-Local_prob"] = (
        data_over_time["Non-Local_Continuous_prob"]
        + data_over_time["Non-Local_Fragmented_prob"]
    )
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])

    return data_over_time


def get_event_types(data_over_time: dict):
    non_local_immobile = get_immobile_events(
        data_over_time,
        event_col="non_local_event",
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        method="endpoints",
    )

    swr_events = data_over_time.loc[(data_over_time["SWR_event"] > 0)]
    mua_events = data_over_time.loc[(data_over_time["MUA_event"] > 0)]

    non_local_immobile_overlap_swr = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] > 0).any())
    non_local_immobile_overlap_mua = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["MUA_event"] > 0).any())

    non_local_immobile_no_overlap = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] == 0).all() & (g["MUA_event"] == 0).all())
    non_local_immobile_no_overlap_SWR = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["SWR_event"] == 0).all())
    non_local_immobile_no_overlap_MUA = non_local_immobile.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).filter(lambda g: (g["MUA_event"] == 0).all())
    swr_events_no_non_local = swr_events.groupby(
        ["animal", "day", "epoch", "SWR_event"]
    ).filter(lambda g: (g["non_local_event"] == 0).all())
    mua_events_no_non_local = mua_events.groupby(
        ["animal", "day", "epoch", "MUA_event"]
    ).filter(lambda g: (g["non_local_event"] == 0).all())

    return {
        "non_local_immobile": non_local_immobile,
        "swr_events": swr_events,
        "mua_events": mua_events,
        "non_local_immobile_overlap_swr": non_local_immobile_overlap_swr,
        "non_local_immobile_overlap_mua": non_local_immobile_overlap_mua,
        "non_local_immobile_no_overlap": non_local_immobile_no_overlap,
        "non_local_immobile_no_overlap_SWR": non_local_immobile_no_overlap_SWR,
        "non_local_immobile_no_overlap_MUA": non_local_immobile_no_overlap_MUA,
        "swr_events_no_non_local": swr_events_no_non_local,
        "mua_events_no_non_local": mua_events_no_non_local,
    }


# %%
data_over_time = load_data_over_time()
event_types = get_event_types(data_over_time)


# %%
# Use get_event_durations_ms() from library instead of local function

fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    duration = get_event_durations_ms(
        event_types[event_type], event_col="non_local_event"
    )

    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        duration,
        bins=BINS_DURATION_MS,
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Event Duration (ms)")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["hpd"],
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("HPD")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        np.abs(event_types[event_type]["ahead_behind_distance"]),
        bins=BINS_DISTANCE_CM,
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Distance from Animal (cm)")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["linear_position"]
        / event_types[event_type]["max_linear_position"],
        bins=np.arange(0, 1, 0.01),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Normalized Animal Position")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["map_speed"] / 100.0,
        bins=BINS_SPEED_CM_PER_SEC,
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Replay Speed")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["map_estimate"]
        / event_types[event_type]["max_linear_position"],
        bins=np.arange(0, 1, 0.01),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Normalized Replay Position")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["Non-Local_Continuous_prob"],
        bins=np.arange(0, 1, 0.01),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Non-Local Cont. Prob.")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["Non-Local_Fragmented_prob"],
        bins=np.arange(0, 1, 0.01),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Non-Local Frag. Prob.")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["Local_prob"],
        bins=np.arange(0, 1, 0.01),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Local Prob.")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["No-Spike_prob"],
        bins=np.arange(0, 1, 0.01),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("No-Spike Prob.")
    ax.set_ylabel("Density")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["Local_prob"],
        bins=np.arange(0, 1, 0.01),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Local Prob.")
    ax.set_ylabel("Density")

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
for event_type, ax in zip(
    ("non_local_immobile", "non_local_immobile_no_overlap_SWR"), axes, strict=True
):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist2d(
        event_types[event_type]["linear_position"]
        / event_types[event_type]["max_linear_position"],
        event_types[event_type]["map_estimate"]
        / event_types[event_type]["max_linear_position"],
        bins=(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)),
        color=color,
        edgecolor="none",
        density=True,
        vmin=0.0,
        vmax=40.0,
    )
    ax.set_xlabel("Normalized Animal Position")
    ax.set_ylabel("Normalized Replay Position")
    ax.set_title(event_type)

# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        event_types[event_type]["hpd"] / event_types[event_type]["max_coverage"],
        bins=np.arange(0, 1, 0.01),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("HPD proportion of track covered")
    ax.set_ylabel("Density")

# %%


# %%
fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    mean_event = (
        event_types[event_type]
        .groupby(["animal", "day", "epoch", "non_local_event"])
        .mean()
    )

    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        mean_event["hpd"],
        bins=BINS_HPD_CM,
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("HPD")
    ax.set_ylabel("Density")


# %%
non_local_start_end_time = get_event_start_end_times(
    event_types["non_local_immobile"], event_col="non_local_event"
)

# %%

signal_types = [
    "Non-Local_prob",
    "theta_power_zscore",
    "MUA_zscore",
    "SWR_zscore",
    "head_speed",
]

signal_windows = get_signal_windows(
    data=data_over_time,
    event_times=non_local_start_end_time,
    signal_cols=signal_types,
    time_col="start_time",
    window_sec=2.0,
    sampling_frequency=SAMPLING_FREQUENCY,
    event_id_col="non_local_event",
)

# %%
non_local_start_end_time = get_event_start_end_times(
    event_types["non_local_immobile_no_overlap_SWR"], event_col="non_local_event"
)

signal_windows = get_signal_windows(
    data=data_over_time,
    event_times=non_local_start_end_time,
    signal_cols=signal_types,
    time_col="start_time",
    window_sec=2.0,
    sampling_frequency=SAMPLING_FREQUENCY,
    event_id_col="non_local_event",
)

# %%
non_local_start_end_time = get_event_start_end_times(
    event_types["non_local_immobile_overlap_swr"], event_col="non_local_event"
)

signal_windows_overlap = get_signal_windows(
    data=data_over_time,
    event_times=non_local_start_end_time,
    signal_cols=signal_types,
    time_col="start_time",
    window_sec=2.0,
    sampling_frequency=SAMPLING_FREQUENCY,
    event_id_col="non_local_event",
)

# %%

fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(6, 4),
    sharex=True,
    sharey=False,
    constrained_layout=True,
)

no_overlap_mean_all = signal_windows.mean(dim="event")
overlap_mean_all = signal_windows_overlap.mean(dim="event")
no_overlap_sem_all = sem(signal_windows, axis=0, nan_policy="omit")
overlap_sem_all = sem(signal_windows_overlap, axis=0, nan_policy="omit")

for ax, (no_overlap_mean, overlap_mean, no_overlap_sem, overlap_sem) in zip(
    axes.flatten(),
    zip(
        no_overlap_mean_all.T,
        overlap_mean_all.T,
        no_overlap_sem_all.T,
        overlap_sem_all.T,
        strict=True,
    ),
    strict=True,
):
    # Add shaded error bars
    ax.fill_between(
        no_overlap_mean["time"],
        no_overlap_mean - no_overlap_sem,
        no_overlap_mean + no_overlap_sem,
        color="C0",
        alpha=0.7,
    )
    ax.fill_between(
        overlap_mean["time"],
        overlap_mean - overlap_sem,
        overlap_mean + overlap_sem,
        color="C1",
        alpha=0.7,
    )

    ax.plot(no_overlap_mean["time"], no_overlap_mean, label="No Overlap")
    ax.plot(overlap_mean["time"], overlap_mean, label="Overlap")

    if str(no_overlap_mean["signal"].values).split("_")[-1] == "zscore":
        name = str(no_overlap_mean["signal"].values).split("_")[0]
        if name == "theta":
            name = "Theta"
        ax.set_ylabel(f"{name}\nZ-score")
        ax.set_ylim(-2.5, 4.0)
        ax.axhline(
            2.0,
            color="lightgrey",
            linestyle="--",
            label="Z-score Threshold",
            alpha=0.5,
            zorder=-1,
            linewidth=0.75,
        )
    if no_overlap_mean["signal"].values == "head_speed":
        ax.set_ylabel("Head Speed\n(cm/s)")
        ax.axhline(
            4.0,
            color="lightgrey",
            linestyle="--",
            label="Immobile Threshold",
            alpha=0.5,
            zorder=-1,
        )
    if no_overlap_mean["signal"].values == "Non-Local_prob":
        ax.set_ylabel("Non-Local\nProb.")
        ax.axhline(
            0.8,
            color="lightgrey",
            linestyle="--",
            label="Prob. Threshold",
            alpha=0.5,
            zorder=-1,
            linewidth=0.75,
        )
        ax.set_ylim(0.0, 1.0)
    if ax is axes[-1]:
        ax.set_xlabel("Time (s)")
    ax.axvline(
        0.0,
        color="lightgrey",
        linestyle="--",
        alpha=0.5,
        zorder=-1,
        linewidth=0.75,
    )
    ax.axhline(
        0.0,
        color="lightgrey",
        linestyle="--",
        alpha=0.5,
        zorder=-1,
        linewidth=0.75,
    )
    ax.legend()

plt.xlim((-1.0, 1.0))

# %%
non_local_start_end_time = get_event_start_end_times(
    event_types["non_local_immobile_no_overlap_SWR"], event_col="non_local_event"
)

signal_windows = get_signal_windows(
    data=data_over_time,
    event_times=non_local_start_end_time,
    signal_cols=signal_types,
    time_col="end_time",
    window_sec=2.0,
    sampling_frequency=SAMPLING_FREQUENCY,
    event_id_col="non_local_event",
)

# %%
non_local_start_end_time = get_event_start_end_times(
    event_types["non_local_immobile_overlap_swr"], event_col="non_local_event"
)

signal_windows_overlap = get_signal_windows(
    data=data_over_time,
    event_times=non_local_start_end_time,
    signal_cols=signal_types,
    time_col="end_time",
    window_sec=2.0,
    sampling_frequency=SAMPLING_FREQUENCY,
    event_id_col="non_local_event",
)

# %%

fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(6, 4),
    sharex=True,
    sharey=False,
    constrained_layout=True,
)

no_overlap_mean_all2 = signal_windows.mean(dim="event")
overlap_mean_all2 = signal_windows_overlap.mean(dim="event")
no_overlap_sem_all2 = sem(signal_windows, axis=0, nan_policy="omit")
overlap_sem_all2 = sem(signal_windows_overlap, axis=0, nan_policy="omit")

for ax, (no_overlap_mean, overlap_mean, no_overlap_sem, overlap_sem) in zip(
    axes.flatten(),
    zip(
        no_overlap_mean_all2.T,
        overlap_mean_all2.T,
        no_overlap_sem_all2.T,
        overlap_sem_all2.T,
        strict=True,
    ),
    strict=True,
):
    # Add shaded error bars
    ax.fill_between(
        no_overlap_mean["time"],
        no_overlap_mean - no_overlap_sem,
        no_overlap_mean + no_overlap_sem,
        color="C0",
        alpha=0.7,
    )
    ax.fill_between(
        overlap_mean["time"],
        overlap_mean - overlap_sem,
        overlap_mean + overlap_sem,
        color="C1",
        alpha=0.7,
    )

    ax.plot(no_overlap_mean["time"], no_overlap_mean, label="No Overlap")
    ax.plot(overlap_mean["time"], overlap_mean, label="Overlap")

    if str(no_overlap_mean["signal"].values).split("_")[-1] == "zscore":
        name = str(no_overlap_mean["signal"].values).split("_")[0]
        if name == "theta":
            name = "Theta"
        ax.set_ylabel(f"{name}\nZ-score")
        ax.set_ylim(-2.5, 4.0)
        ax.axhline(
            2.0,
            color="lightgrey",
            linestyle="--",
            label="Z-score Threshold",
            alpha=0.5,
            zorder=-1,
            linewidth=0.75,
        )
    if no_overlap_mean["signal"].values == "head_speed":
        ax.set_ylabel("Head Speed\n(cm/s)")
        ax.axhline(
            4.0,
            color="lightgrey",
            linestyle="--",
            label="Immobile Threshold",
            alpha=0.5,
            zorder=-1,
        )
    if no_overlap_mean["signal"].values == "Non-Local_prob":
        ax.set_ylabel("Non-Local\nProb.")
        ax.axhline(
            0.8,
            color="lightgrey",
            linestyle="--",
            label="Prob. Threshold",
            alpha=0.5,
            zorder=-1,
            linewidth=0.75,
        )
        ax.set_ylim(0.0, 1.0)
    if ax is axes[-1]:
        ax.set_xlabel("Time (s)")
    ax.axvline(
        0.0,
        color="lightgrey",
        linestyle="--",
        alpha=0.5,
        zorder=-1,
        linewidth=0.75,
    )
    ax.axhline(
        0.0,
        color="lightgrey",
        linestyle="--",
        alpha=0.5,
        zorder=-1,
        linewidth=0.75,
    )
    ax.legend()

plt.xlim((-1.0, 1.0))


bins = int(np.median(data_over_time["max_linear_position"].unique() / 2))
density, bins, _ = plt.hist(
    data_over_time.loc[
        np.logical_and(
            data_over_time["non_local_event"] > 0, data_over_time["SWR_event"] > 0
        ),
        "map_estimate",
    ]
    / data_over_time.loc[data_over_time["non_local_event"] > 0, "max_linear_position"],
    bins=bins,
    density=True,
    alpha=0.8,
    color=COLORS["SWR"],
    edgecolor=COLORS["SWR"],
    label="SWR",
)
plt.hist(
    data_over_time.loc[
        np.logical_and(
            data_over_time["non_local_event"] > 0, data_over_time["SWR_event"] == 0
        ),
        "map_estimate",
    ]
    / data_over_time.loc[data_over_time["non_local_event"] > 0, "max_linear_position"],
    bins=bins,
    density=True,
    alpha=0.8,
    color=COLORS["Non-Local"],
    edgecolor=COLORS["Non-Local"],
    label="No SWR",
)
plt.legend()


bin_labels, n_labels = label(density)

well_labels = [2, 3, 5, 6, 8, 9]
edge_colors = np.array(matplotlib.colormaps.get_cmap("tab10").colors)
track_line_y = -0.3

for i in range(1, n_labels + 1):
    bin_ind = np.nonzero(bin_labels == i)[0]
    plt.plot(
        (bins[bin_ind[0]], bins[bin_ind[-1] + 1]),
        (track_line_y, track_line_y),
        zorder=10,
        clip_on=False,
        color=edge_colors[i - 1],
        linewidth=3,
    )

    if i in well_labels:
        plt.scatter(
            bins[bin_ind[-1]],
            track_line_y,
            zorder=11,
            clip_on=False,
            color=edge_colors[i - 1],
            s=30,
        )

plt.xlim((0, 1))
plt.yticks([0, 13])
plt.ylim((0, 13))
plt.ylabel("Density")
# Hide x-axis and x line
sns.despine(offset=5)
plt.gca().get_xaxis().set_ticks([])
plt.gca().spines["bottom"].set_visible(False)
plt.xlabel("Linear position", labelpad=10)


plt.hist(
    data_over_time.loc[
        np.logical_and(
            data_over_time.head_speed < 4.0, data_over_time["Local_prob"] > 0.8
        ),
        "SWR_zscore",
    ],
    bins=np.arange(-5, 11),
    alpha=0.8,
    density=True,
    label="Local",
    color=COLORS["Local"],
)
plt.hist(
    data_over_time.loc[
        np.logical_and(
            data_over_time.head_speed < 4.0, data_over_time["Non-Local_prob"] > 0.8
        ),
        "SWR_zscore",
    ],
    bins=np.arange(-5, 11),
    alpha=0.8,
    density=True,
    label="Non-Local",
    color=COLORS["Non-Local"],
)
plt.legend()
plt.ylabel("Density")
plt.xlabel("SWR z-score")
sns.despine()


plt.hist2d(
    data_over_time.loc[data_over_time.head_speed < 4.0]["SWR_zscore"],
    data_over_time.loc[data_over_time.head_speed < 4.0]["Non-Local_prob"],
    bins=(np.arange(-5, 11), np.arange(0, 1.1, 0.1)),
    vmin=0.0,
    vmax=0.1,
    density=True,
)
plt.xlabel("SWR Z-score")
plt.ylabel("Non-Local Probability")
plt.colorbar(label="Density")


bins = np.arange(0, 3, 0.1)
plt.hist(
    data_over_time.loc[
        np.logical_and(
            data_over_time["non_local_event"] > 0, data_over_time["SWR_event"] > 0
        ),
        "map_speed",
    ]
    / 1000,
    bins=bins,
    density=True,
    alpha=0.8,
    color=COLORS["SWR"],
    label="SWR",
)
plt.hist(
    data_over_time.loc[
        np.logical_and(
            data_over_time["non_local_event"] > 0, data_over_time["SWR_event"] == 0
        ),
        "map_speed",
    ]
    / 1000,
    bins=bins,
    density=True,
    alpha=0.8,
    color=COLORS["Non-Local"],
    label="No SWR",
)
plt.legend()
plt.xlabel("Replay Speed (m/s)")
plt.ylabel("Density")
sns.despine(offset=5)

# %%
# what are the spike amplitudes during these events
# how many neurons are spiking (What percentage of cells are spiking)
# what is the spike rate (MUA?)
# how synchronous are the spikes (all at the same time?)
# angular velocity of head
# interspike interval?
# speed of replay
# position of replay
# hpd of replay
