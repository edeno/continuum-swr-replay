# %%
# %config InlineBackend.figure_format = 'retina'
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import label

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.parameters import COLORS, PROCESSED_DATA_DIR

# Output directory for baseline figures
OUTPUT_DIR = Path("tests/baselines/figure02")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")

# %%
data_path = PROCESSED_DATA_DIR.glob("*_HPC_80_data_over_time.pkl")

data_over_time = pd.concat([pd.read_pickle(p) for p in data_path])
data_over_time["Non-Local_prob"] = (
    data_over_time["Non-Local_Continuous_prob"]
    + data_over_time["Non-Local_Fragmented_prob"]
)
data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])

# %%
# Need non-local events during immobility, swr events, mua events
# Need to know if overlaps with SWR or MUA events
#
non_local_immobile = (
    data_over_time.loc[(data_over_time["non_local_event"] > 0)]
    .groupby(["animal", "day", "epoch", "non_local_event"])
    .filter(lambda g: (g.head_speed.iloc[0] < 4.0) & (g.head_speed.iloc[-1] < 4.0))
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

# %%
# animal                                    j16
# day                                  20210709
# epoch                                   04_r2
# MUA_event                                  1

(
    non_local_immobile_no_overlap_MUA.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).MUA_event.mean()
    > 0
).any()

# %%
len(non_local_immobile)

# %%
is_immobile = data_over_time.head_speed < 4.0

# %% [markdown]
# Non-local probability is bi-modal, justifying the use of a 0.8 threshold.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

# Plot histogram
data_over_time.loc[is_immobile, "Non-Local_prob"].plot.hist(
    bins=50,
    color=COLORS["Non-Local"],
    alpha=0.7,
    density=True,
    edgecolor="none",
    histtype="stepfilled",
    range=(0, 1),
    ax=ax,
)

# Remove chartjunk: no top/right spines, minimal grid, no box
sns.despine(ax=ax, left=True, top=True, right=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

# Clean, informative labels
ax.set_xlabel("Non-Local Probability")
ax.set_ylabel("Density")
ax.set_title("Distribution of Non-Local Probability (Immobile)", pad=10)

# Remove legend, ticks on top/right, and reduce tick length
legend = ax.get_legend()
if legend is not None:
    legend.remove()
ax.tick_params(axis="both", which="both", length=3, direction="out")
ax.set_xlim(0, 1)
ax.axvline(0.8, color="black", linewidth=0.5, linestyle="--", label="0.8 Threshold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_01.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %% [markdown]
# Also bi-modal for each animal

# %%
# Get unique animals
animals = data_over_time["animal"].unique()

# Set up the grid for small multiples
n_animals = len(animals)
fig, axes = plt.subplots(1, n_animals, figsize=(6 * n_animals, 5), sharey=True)

if n_animals == 1:
    axes = [axes]

for ax, animal in zip(axes, animals, strict=True):
    group = data_over_time.loc[(data_over_time["animal"] == animal) & is_immobile]
    ax.hist(
        group["Non-Local_prob"],
        bins=10,
        alpha=0.7,
        color=COLORS.get("Non-Local", "#1b9e77"),
        density=True,
        edgecolor="black",
        linewidth=0.7,
        histtype="stepfilled",
    )
    sns.despine(ax=ax)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_xlabel("Non-Local Probability")
    ax.set_title(f"Animal: {animal}")

axes[0].set_ylabel("Density")
fig.suptitle("Distribution of Non-Local Probability (Immobile Periods)")
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(OUTPUT_DIR / "figure02_panel_02.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %% [markdown]
# Interestingly the non-local probability is less bimodal during the run state.

# %%
# Get unique animals
animals = data_over_time["animal"].unique()

# Set up the grid for small multiples
n_animals = len(animals)
fig, axes = plt.subplots(1, n_animals, figsize=(6 * n_animals, 5), sharey=True)

if n_animals == 1:
    axes = [axes]

for ax, animal in zip(axes, animals, strict=True):
    group = data_over_time.loc[(data_over_time["animal"] == animal) & (~is_immobile)]
    ax.hist(
        group["Non-Local_prob"],
        bins=10,
        alpha=0.7,
        color=COLORS.get("Non-Local", "#1b9e77"),
        density=True,
        edgecolor="black",
        linewidth=0.7,
        histtype="stepfilled",
    )
    sns.despine(ax=ax)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_xlabel("Non-Local Probability")
    ax.set_title(f"Animal: {animal}")

axes[0].set_ylabel("Density")
fig.suptitle("Distribution of Non-Local Probability (Run Periods)")
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(OUTPUT_DIR / "figure02_panel_03.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %% [markdown]
# The local probability is less bimodal during the immobile state, but still bimodal.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

# Plot histogram
data_over_time.loc[is_immobile, "Local_prob"].plot.hist(
    bins=50,
    color=COLORS["Local"],
    alpha=0.7,
    density=True,
    edgecolor="none",
    histtype="stepfilled",
    range=(0, 1),
    ax=ax,
)

# Remove chartjunk: no top/right spines, minimal grid, no box
sns.despine(ax=ax, left=True, top=True, right=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

# Clean, informative labels
ax.set_xlabel("Local Probability")
ax.set_ylabel("Density")
ax.set_title("Distribution of Local Probability (Immobile)", pad=10)

# Remove legend, ticks on top/right, and reduce tick length
legend = ax.get_legend()
if legend is not None:
    legend.remove()
ax.tick_params(axis="both", which="both", length=3, direction="out")
ax.set_xlim(0, 1)
ax.axvline(0.8, color="black", linewidth=0.5, linestyle="--", label="0.8 Threshold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_04.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
fig, ax = plt.subplots(figsize=(7, 4))

# Plot histogram
data_over_time.loc[is_immobile, "No-Spike_prob"].plot.hist(
    bins=50,
    color=COLORS["No-Spike"],
    alpha=0.7,
    density=True,
    edgecolor="none",
    histtype="stepfilled",
    range=(0, 1),
    ax=ax,
)

# Remove chartjunk: no top/right spines, minimal grid, no box
sns.despine(ax=ax, left=True, top=True, right=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

# Clean, informative labels
ax.set_xlabel("No-Spike Probability")
ax.set_ylabel("Density")
ax.set_title("Distribution of No-Spike Probability (Immobile)", pad=10)

# Remove legend, ticks on top/right, and reduce tick length
legend = ax.get_legend()
if legend is not None:
    legend.remove()
ax.tick_params(axis="both", which="both", length=3, direction="out")
ax.set_xlim(0, 1)
ax.axvline(0.8, color="black", linewidth=0.5, linestyle="--", label="0.8 Threshold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_05.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
half_window = 0.250  # seconds
dt = data_over_time.index.to_series().diff().median()  # seconds

signals = ["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"]
titles = [
    "Theta Power",
    "MUA",
    "SWR",
    "Head Speed",
]

fig, axes = plt.subplots(4, 1, figsize=(3, 4), sharex=True, constrained_layout=True)

dt = float(dt)
half_window_samples = int(round(half_window / dt))
window_len = 2 * half_window_samples + 1
offsets = np.arange(-half_window_samples, half_window_samples + 1)

for i, (signal, title) in enumerate(zip(signals, titles, strict=True)):
    all_windows = []
    for (_animal, _day, _epoch), group in data_over_time.groupby(
        ["animal", "day", "epoch"]
    ):
        time_index = group.index.values
        y = group[signal].values

        events_in_group = (
            group[group["non_local_event"] > 0]
            .groupby("non_local_event")
            .filter(
                lambda g: (g.head_speed.iloc[0] < 4.0) & (g.head_speed.iloc[-1] < 4.0)
            )
            .groupby("non_local_event")
            .head(1)
        )
        event_times = events_in_group.index.values
        if len(event_times) == 0:
            continue

        center_idxs = np.searchsorted(time_index, event_times)
        all_window_idxs = center_idxs[:, None] + offsets[None, :]

        mask = (all_window_idxs >= 0) & (all_window_idxs < len(y))
        windows = np.full((len(event_times), window_len), np.nan)
        valid_rows, valid_cols = np.where(mask)
        windows[valid_rows, valid_cols] = y[all_window_idxs[mask]]
        all_windows.append(windows)

    ax = axes[i]
    if all_windows:
        all_windows_matrix = np.vstack(all_windows)
        mean_y = np.nanmean(all_windows_matrix, axis=0)
        sem_y = np.nanstd(all_windows_matrix, ddof=1, axis=0) / np.sqrt(
            np.sum(~np.isnan(all_windows_matrix), axis=0)
        )
        x = offsets * dt * 1000

        ax.plot(x, mean_y, color="black", lw=1)
        ax.fill_between(
            x, mean_y - sem_y, mean_y + sem_y, color="gray", alpha=0.3, linewidth=0
        )

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.grid(False)
        ax.set_axisbelow(True)

        ax.set_ylabel(title, labelpad=2)
        ax.set_xlim(x[0], x[-1])
        ax.axvline(0, color="k", linestyle="--", lw=0.5)
        if signal != "head_speed":
            ax.set_ylim((-2, 2))
            ax.axhline(0, color="k", linestyle="--", lw=0.5)
        else:
            ax.set_ylim((0, 5))
            ax.axhline(4, color="k", linestyle="--", lw=0.5)

        ax.tick_params(axis="both", which="both", length=2, width=0.5, labelsize=7)
        ax.set_title("", pad=2)
    else:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_ylabel(title)

axes[-1].set_xlabel("Time from non-local event start (ms)", labelpad=2)
plt.subplots_adjust(hspace=0.15)
plt.savefig(OUTPUT_DIR / "figure02_panel_06.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
half_window = 0.250  # seconds
dt = data_over_time.index.to_series().diff().median()  # seconds

signals = ["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"]
titles = [
    "Theta Power",
    "MUA",
    "SWR",
    "Head Speed",
]

fig, axes = plt.subplots(4, 1, figsize=(3, 4), sharex=True, constrained_layout=True)

dt = float(dt)
half_window_samples = int(round(half_window / dt))
window_len = 2 * half_window_samples + 1
offsets = np.arange(-half_window_samples, half_window_samples + 1)

for i, (signal, title) in enumerate(zip(signals, titles, strict=True)):
    all_windows = []
    for (_animal, _day, _epoch), group in data_over_time.groupby(
        ["animal", "day", "epoch"]
    ):
        time_index = group.index.values
        y = group[signal].values

        events_in_group = (
            group[group["non_local_event"] > 0]
            .groupby("non_local_event")
            .filter(
                lambda g: (g.head_speed.iloc[0] < 4.0) & (g.head_speed.iloc[-1] < 4.0)
            )
            .groupby("non_local_event")
            .tail(1)
        )
        event_times = events_in_group.index.values
        if len(event_times) == 0:
            continue

        center_idxs = np.searchsorted(time_index, event_times)
        all_window_idxs = center_idxs[:, None] + offsets[None, :]

        mask = (all_window_idxs >= 0) & (all_window_idxs < len(y))
        windows = np.full((len(event_times), window_len), np.nan)
        valid_rows, valid_cols = np.where(mask)
        windows[valid_rows, valid_cols] = y[all_window_idxs[mask]]
        all_windows.append(windows)

    ax = axes[i]
    if all_windows:
        all_windows_matrix = np.vstack(all_windows)
        mean_y = np.nanmean(all_windows_matrix, axis=0)
        sem_y = np.nanstd(all_windows_matrix, ddof=1, axis=0) / np.sqrt(
            np.sum(~np.isnan(all_windows_matrix), axis=0)
        )
        x = offsets * dt * 1000

        ax.plot(x, mean_y, color="black", lw=1)
        ax.fill_between(
            x, mean_y - sem_y, mean_y + sem_y, color="gray", alpha=0.3, linewidth=0
        )

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.grid(False)
        ax.set_axisbelow(True)

        ax.set_ylabel(title, labelpad=2)
        ax.set_xlim(x[0], x[-1])
        ax.axvline(0, color="k", linestyle="--", lw=0.5)
        if signal != "head_speed":
            ax.set_ylim((-2, 2))
            ax.axhline(0, color="k", linestyle="--", lw=0.5)
        else:
            ax.set_ylim((0, 5))
            ax.axhline(4, color="k", linestyle="--", lw=0.5)

        ax.tick_params(axis="both", which="both", length=2, width=0.5, labelsize=7)
        ax.set_title("", pad=2)
    else:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_ylabel(title)

axes[-1].set_xlabel("Time from non-local event end (ms)", labelpad=2)
plt.subplots_adjust(hspace=0.15)
plt.savefig(OUTPUT_DIR / "figure02_panel_07.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
half_window = 0.250  # seconds
dt = data_over_time.index.to_series().diff().median()  # seconds

signals = ["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"]
titles = [
    "Theta Power",
    "MUA",
    "SWR",
    "Head Speed",
]

fig, axes = plt.subplots(4, 1, figsize=(3, 4), sharex=True, constrained_layout=True)

dt = float(dt)
half_window_samples = int(round(half_window / dt))
window_len = 2 * half_window_samples + 1
offsets = np.arange(-half_window_samples, half_window_samples + 1)

for i, (signal, title) in enumerate(zip(signals, titles, strict=True)):
    all_windows = []
    for (_animal, _day, _epoch), group in data_over_time.groupby(
        ["animal", "day", "epoch"]
    ):
        time_index = group.index.values
        y = group[signal].values

        events_in_group = (
            group[group["non_local_event"] > 0]
            .groupby("non_local_event")
            .filter(
                lambda g: (g.head_speed.iloc[0] < 4.0) & (g.head_speed.iloc[-1] < 4.0)
            )
            .groupby("non_local_event")
            .head(1)
        )
        event_times = events_in_group.index.values
        if len(event_times) == 0:
            continue

        center_idxs = np.searchsorted(time_index, event_times)
        all_window_idxs = center_idxs[:, None] + offsets[None, :]

        mask = (all_window_idxs >= 0) & (all_window_idxs < len(y))
        windows = np.full((len(event_times), window_len), np.nan)
        valid_rows, valid_cols = np.where(mask)
        windows[valid_rows, valid_cols] = y[all_window_idxs[mask]]
        all_windows.append(windows)

    ax = axes[i]
    if all_windows:
        all_windows_matrix = np.vstack(all_windows)
        pct_y = np.nanpercentile(all_windows_matrix, axis=0, q=[5, 25, 50, 75, 95])
        x = offsets * dt * 1000
        ax.fill_between(
            x,
            pct_y[0],
            pct_y[4],
            color="lightgray",
            alpha=0.3,
            linewidth=0,
            label="5th-95th Percentile",
        )
        ax.fill_between(
            x,
            pct_y[1],
            pct_y[3],
            color="gray",
            alpha=0.3,
            linewidth=0,
            label="25th-75th Percentile",
        )
        ax.plot(x, pct_y[2], color="black", lw=1)

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.grid(False)
        ax.set_axisbelow(True)

        ax.set_ylabel(title, labelpad=2)
        ax.set_xlim(x[0], x[-1])
        ax.axvline(0, color="k", linestyle="--", lw=0.5)
        if signal != "head_speed":
            ax.set_ylim((-5, 10))
            ax.axhline(0, color="k", linestyle="--", lw=0.5)
        else:
            ax.set_ylim((0, 5))
            ax.axhline(4, color="k", linestyle="--", lw=0.5)

        ax.tick_params(axis="both", which="both", length=2, width=0.5, labelsize=7)
        ax.set_title("", pad=2)
    else:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_ylabel(title)

axes[-1].set_xlabel("Time from non-local event start (ms)", labelpad=2)
axes[0].legend(
    loc="upper right",
    bbox_to_anchor=(1.15, 1),
    frameon=False,
    title="Percentiles",
    handlelength=1.5,
    handletextpad=0.5,
)
plt.subplots_adjust(hspace=0.15)
plt.savefig(OUTPUT_DIR / "figure02_panel_08.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
half_window = 0.250  # seconds
dt = data_over_time.index.to_series().diff().median()  # seconds

signals = ["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"]
titles = [
    "Theta Power",
    "MUA",
    "SWR",
    "Head Speed",
]

fig, axes = plt.subplots(4, 1, figsize=(3, 4), sharex=True, constrained_layout=True)

dt = float(dt)
half_window_samples = int(round(half_window / dt))
window_len = 2 * half_window_samples + 1
offsets = np.arange(-half_window_samples, half_window_samples + 1)

for i, (signal, title) in enumerate(zip(signals, titles, strict=True)):
    all_windows = []
    for (_animal, _day, _epoch), group in data_over_time.groupby(
        ["animal", "day", "epoch"]
    ):
        time_index = group.index.values
        y = group[signal].values

        events_in_group = (
            group[group["non_local_event"] > 0]
            .groupby("non_local_event")
            .filter(
                lambda g: (g.head_speed.iloc[0] < 4.0) & (g.head_speed.iloc[-1] < 4.0)
            )
            .groupby("non_local_event")
            .tail(1)
        )
        event_times = events_in_group.index.values
        if len(event_times) == 0:
            continue

        center_idxs = np.searchsorted(time_index, event_times)
        all_window_idxs = center_idxs[:, None] + offsets[None, :]

        mask = (all_window_idxs >= 0) & (all_window_idxs < len(y))
        windows = np.full((len(event_times), window_len), np.nan)
        valid_rows, valid_cols = np.where(mask)
        windows[valid_rows, valid_cols] = y[all_window_idxs[mask]]
        all_windows.append(windows)

    ax = axes[i]
    if all_windows:
        all_windows_matrix = np.vstack(all_windows)
        pct_y = np.nanpercentile(all_windows_matrix, axis=0, q=[5, 25, 50, 75, 95])
        x = offsets * dt * 1000
        ax.fill_between(
            x,
            pct_y[0],
            pct_y[4],
            color="lightgray",
            alpha=0.3,
            linewidth=0,
            label="5th-95th Percentile",
        )
        ax.fill_between(
            x,
            pct_y[1],
            pct_y[3],
            color="gray",
            alpha=0.3,
            linewidth=0,
            label="25th-75th Percentile",
        )
        ax.plot(x, pct_y[2], color="black", lw=1)

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.grid(False)
        ax.set_axisbelow(True)

        ax.set_ylabel(title, labelpad=2)
        ax.set_xlim(x[0], x[-1])
        ax.axvline(0, color="k", linestyle="--", lw=0.5)
        if signal != "head_speed":
            ax.set_ylim((-5, 10))
            ax.axhline(0, color="k", linestyle="--", lw=0.5)
        else:
            ax.set_ylim((0, 5))
            ax.axhline(4, color="k", linestyle="--", lw=0.5)

        ax.tick_params(axis="both", which="both", length=2, width=0.5, labelsize=7)
        ax.set_title("", pad=2)
    else:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_ylabel(title)

axes[-1].set_xlabel("Time from non-local event end (ms)", labelpad=2)
axes[0].legend(
    loc="upper right",
    bbox_to_anchor=(1.15, 1),
    frameon=False,
    title="Percentiles",
    handlelength=1.5,
    handletextpad=0.5,
)
plt.subplots_adjust(hspace=0.15)
plt.savefig(OUTPUT_DIR / "figure02_panel_09.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
non_local_events = data_over_time.loc[(data_over_time["non_local_event"] > 0)].groupby(
    ["animal", "day", "epoch", "non_local_event"]
)

non_local_event_duration = non_local_events.size() / 500.0


fig, ax = plt.subplots(figsize=(6, 3.5))

# Plot histogram
ax.hist(
    non_local_event_duration,
    bins=np.arange(0, 0.250, 0.010).tolist(),
    color=COLORS["Non-Local"],
    alpha=0.7,
    edgecolor="none",
    density=True,
)

# Minimalist, clean style
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

ax.set_xlabel("Non-local Event Duration (s)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Non-local Event Durations", pad=10)

ax.tick_params(axis="both", which="both", length=3, direction="out")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_10.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
non_local_events = data_over_time.loc[(data_over_time["non_local_event"] > 0)].groupby(
    ["animal", "day", "epoch", "non_local_event"]
)

immobile_non_local_events = non_local_events.filter(
    lambda g: (g.head_speed.iloc[0] < 4.0) and (g.head_speed.iloc[-1] < 4.0)
)

# %%
non_local_event_durations = (
    immobile_non_local_events.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).size()
    / 500.0
) * 1000.0  # Convert to milliseconds

SWR_event_durations = (
    data_over_time.groupby(["animal", "day", "epoch", "SWR_event"]).size() / 500.0
) * 1000.0  # Convert to milliseconds

MUA_event_durations = (
    data_over_time.groupby(["animal", "day", "epoch", "MUA_event"]).size() / 500.0
) * 1000.0  # Convert to milliseconds

fig, axes = plt.subplots(
    3, 1, figsize=(5, 6), sharex=True, sharey=True, constrained_layout=True
)

# Non-local
axes[0].hist(
    non_local_event_durations,
    bins=np.arange(0, 250, 10).tolist(),
    color=COLORS["Non-Local"],
    alpha=0.7,
    edgecolor="none",
    density=True,
)
axes[0].set_title("Non-local (Immobile)", pad=10)

# SWR
axes[1].hist(
    SWR_event_durations,
    bins=np.arange(0, 250, 10).tolist(),
    color=COLORS["SWR"],
    alpha=0.7,
    edgecolor="none",
    density=True,
)
axes[1].set_title("SWR", pad=10)

# MUA
axes[2].hist(
    MUA_event_durations,
    bins=np.arange(0, 250, 10).tolist(),
    color=COLORS["MUA"],
    alpha=0.7,
    edgecolor="none",
    density=True,
)
axes[2].set_xlabel("Event Duration (ms)")
axes[2].set_title("MUA", pad=10)

for ax in axes:
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel("Density")
    ax.tick_params(axis="both", which="both", length=3, direction="out")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.set_xlim(0, 250)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_11.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Plot all three histograms on one axis
ax.hist(
    non_local_event_durations,
    bins=np.arange(0, 250, 10).tolist(),
    color=COLORS["Non-Local"],
    alpha=0.6,
    edgecolor="none",
    density=True,
    label="Non-local (Immobile)",
)
ax.hist(
    SWR_event_durations,
    bins=np.arange(0, 250, 10).tolist(),
    color=COLORS["SWR"],
    alpha=0.6,
    edgecolor="none",
    density=True,
    label="SWR",
)
ax.hist(
    MUA_event_durations,
    bins=np.arange(0, 250, 10).tolist(),
    color=COLORS["MUA"],
    alpha=0.6,
    edgecolor="none",
    density=True,
    label="MUA",
)

# Clean style
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

ax.set_xlabel("Event Duration (ms)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Event Durations", pad=10)
ax.legend(frameon=False)
ax.tick_params(axis="both", which="both", length=3, direction="out")
ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.set_xlim(0, 250)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_12.pdf", bbox_inches="tight", dpi=300)
plt.close()


# %%
def get_event_times(group: pd.DataFrame, event_col: str, mode: str = "start") -> Any:
    """
    group: pd.DataFrame for a single session (animal, day, epoch)
    event_col: str, column with event flag (e.g., 'SWR_event', 'MUA_event', etc.)
    mode: 'start' or 'end'
    Returns: np.array of indices (float64 time)
    """
    is_event = group[event_col].values > 0
    changes = np.diff(is_event.astype(int), prepend=0)
    if mode == "start":
        event_idxs = np.where(changes == 1)[0]  # Rising edge
    elif mode == "end":
        event_idxs = np.where(changes == -1)[0]  # Falling edge
    else:
        raise ValueError("mode must be 'start' or 'end'")
    return group.index.values[event_idxs]


def plot_event_triggered(
    df: pd.DataFrame,
    event_col: str,
    signals: list[str],
    titles: list[str] | None = None,
    mode: str = "start",
    window: float = 0.250,
    filter_func: Any = None,
    percentile_bands: list[tuple[int, int]] | None = None,
    ylim_dict: dict[str, tuple[float, float]] | None = None,
    x_label: str = "Time from event (ms)",
) -> None:
    """
    df: DataFrame (all data)
    event_col: column name for event to align to
    signals: list of signal columns to plot
    titles: list of labels for plots
    mode: 'start' or 'end' (event alignment)
    window: float, half-window in seconds
    filter_func: function to filter candidate events (receives event group DataFrame)
    percentile_bands: list of (low, high) percentiles for plotting bands/lines
    ylim_dict: dict with signal->ylim mapping (or None)
    x_label: x-axis label
    """
    dt = 1 / 500.0
    half_window_samples = int(round(window / dt))
    window_len = 2 * half_window_samples + 1
    offsets = np.arange(-half_window_samples, half_window_samples + 1)
    if titles is None:
        titles = signals
    if percentile_bands is None:
        percentile_bands = [(5, 95), (25, 75), (50, 50)]
    if ylim_dict is None:
        ylim_dict = {}

    _fig, axes = plt.subplots(
        len(signals), 1, figsize=(3, len(signals)), sharex=True, constrained_layout=True
    )

    for i, (signal, title) in enumerate(zip(signals, titles, strict=True)):
        all_windows = []
        for _key, group in df.groupby(["animal", "day", "epoch"]):
            time_index = group.index.values
            y = group[signal].values

            # Find event times (start or end)
            event_times = get_event_times(group, event_col, mode=mode)

            # Optionally filter events (e.g., for low speed, etc.)
            if filter_func is not None:
                mask = filter_func(group, event_col, event_times)
                event_times = event_times[mask]

            if len(event_times) == 0:
                continue

            center_idxs = np.searchsorted(time_index, event_times)
            all_window_idxs = center_idxs[:, None] + offsets[None, :]
            mask = (all_window_idxs >= 0) & (all_window_idxs < len(y))
            windows = np.full((len(event_times), window_len), np.nan)
            valid_rows, valid_cols = np.where(mask)
            windows[valid_rows, valid_cols] = y[all_window_idxs[mask]]
            all_windows.append(windows)

        ax = axes[i]
        if all_windows:
            all_windows_matrix = np.vstack(all_windows)
            for band, color, alpha, label in zip(
                percentile_bands,
                ["lightgray", "gray", "black"],
                [0.3, 0.3, 1.0],
                ["{}th-{}th".format(*band) for band in percentile_bands],
                strict=True,
            ):
                q = np.nanpercentile(all_windows_matrix, axis=0, q=band)
                if band[0] != band[1]:
                    if label == "50th-50th":
                        label = "Median"
                    ax.fill_between(
                        offsets * dt * 1000,
                        q[0],
                        q[1],
                        color=color,
                        alpha=alpha,
                        linewidth=0,
                        label=label,
                    )
                else:
                    ax.plot(offsets * dt * 1000, q[0], color=color, lw=1, label=label)
            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
            ax.grid(False)
            ax.set_axisbelow(True)
            ax.set_ylabel(title, labelpad=2)
            ax.set_xlim(offsets[0] * dt * 1000, offsets[-1] * dt * 1000)
            ax.axvline(0, color="k", linestyle="--", lw=0.5)
            if signal in ylim_dict:
                ax.set_ylim(ylim_dict[signal])
            ax.tick_params(axis="both", which="both", length=2, width=0.5, labelsize=7)
            ax.set_title("", pad=2)
        else:
            ax.text(0.5, 0.5, "No events", ha="center", va="center")
            ax.set_ylabel(title)

    axes[-1].set_xlabel(x_label, labelpad=2)
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        frameon=False,
        title="Percentiles",
        handlelength=1.5,
        handletextpad=0.5,
    )
    plt.subplots_adjust(hspace=0.15)
    plt.savefig(OUTPUT_DIR / "figure02_panel_13.pdf", bbox_inches="tight", dpi=300)


plt.close()


def low_speed_filter(
    group: pd.DataFrame,
    _event_col: str,
    event_times: np.ndarray,
    threshold: float = 4.0,
) -> np.ndarray:
    # Get windows around each event, check speed at start/end
    mask = []
    speed = group["head_speed"]
    idxs = np.searchsorted(group.index.values, event_times)
    for i in idxs:
        ok = (i >= 0) and (i < len(speed)) and (speed.iloc[i] < threshold)
        mask.append(ok)
    return np.array(mask, dtype=bool)


def label_events_by_group(
    df: pd.DataFrame,
    prob_col: str,
    out_col: str,
    threshold: float = 0.8,
    group_cols: list[str] | None = None,
) -> None:
    """
    Labels contiguous above-threshold events in prob_col for each group,
    storing results in out_col.
    """
    if group_cols is None:
        group_cols = ["animal", "day", "epoch"]
    df[out_col] = 0  # Default value
    for _key, group in df.groupby(group_cols):
        mask = group[prob_col].values > threshold
        labels, _ = label(mask)
        df.loc[group.index, out_col] = labels  # Store label for this group only


label_events_by_group(
    data_over_time, prob_col="No-Spike_prob", out_col="no_spike_event", threshold=0.8
)

# For Local events
label_events_by_group(
    data_over_time, prob_col="Local_prob", out_col="local_event", threshold=0.8
)

# %%
plot_event_triggered(
    data_over_time,
    event_col="SWR_event",
    signals=["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"],
    titles=["Theta Power", "MUA", "SWR", "Head Speed"],
    mode="start",
    window=0.25,
    filter_func=low_speed_filter,
    ylim_dict={
        "theta_power_zscore": (-5, 10),
        "MUA_zscore": (-5, 10),
        "SWR_zscore": (-5, 10),
        "head_speed": (0, 5),
    },
    x_label="Time from SWR start (ms)",
)

# %%
plot_event_triggered(
    data_over_time,
    event_col="MUA_event",
    signals=["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"],
    titles=["Theta Power", "MUA", "SWR", "Head Speed"],
    mode="start",
    window=0.25,
    filter_func=low_speed_filter,
    ylim_dict={
        "theta_power_zscore": (-5, 10),
        "MUA_zscore": (-5, 10),
        "SWR_zscore": (-5, 10),
        "head_speed": (0, 5),
    },
    x_label="Time from MUA event start (ms)",
)

# %%
plot_event_triggered(
    data_over_time,
    event_col="no_spike_event",
    signals=["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"],
    titles=["Theta Power", "MUA", "SWR", "Head Speed"],
    mode="start",
    window=0.25,
    filter_func=low_speed_filter,
    ylim_dict={
        "theta_power_zscore": (-2, 2),
        "MUA_zscore": (-2, 2),
        "SWR_zscore": (-2, 2),
        "head_speed": (0, 5),
    },
    x_label="Time from No-Spike Event Start (ms)",
)

# %%
plot_event_triggered(
    data_over_time,
    event_col="local_event",
    signals=["theta_power_zscore", "MUA_zscore", "SWR_zscore", "head_speed"],
    titles=["Theta Power", "MUA", "SWR", "Head Speed"],
    mode="start",
    window=0.25,
    filter_func=low_speed_filter,
    ylim_dict={
        "theta_power_zscore": (-2, 2),
        "MUA_zscore": (-2, 2),
        "SWR_zscore": (-2, 2),
        "head_speed": (0, 5),
    },
    x_label="Time from local event start (ms)",
)

# %%
# Of all events detected by your model (i.e., non-local events), what percentage also overlap in time with an SWR or MUA event?
# Of all SWR or MUA events, what percentage are also detected as non-local by your model?

# %%
immobile_non_local_events.groupby(["animal", "day", "epoch", "non_local_event"]).apply(
    lambda g: g["SWR_event"].any()
).mean() * 100.0

# %%
non_local_swr_overlap = (
    immobile_non_local_events.groupby(["animal", "day", "epoch", "non_local_event"])
    .apply(lambda g: g["MUA_event"].any())
    .mean()
    * 100.0
)

# %%
non_local_mua_overlap = (
    immobile_non_local_events.groupby(["animal", "day", "epoch", "non_local_event"])
    .apply(lambda g: g["MUA_event"].any() or g["SWR_event"].any())
    .mean()
    * 100.0
)

# %%
swr_non_local_overlap_pct = (
    data_over_time.loc[data_over_time["SWR_event"] > 0]
    .groupby(["animal", "day", "epoch", "SWR_event"])
    .apply(lambda g: (g["non_local_event"] > 0).any())
    .mean()
    * 100.0
)

# %%
mua_non_local_overlap_pct = (
    data_over_time.loc[data_over_time["MUA_event"] > 0]
    .groupby(["animal", "day", "epoch", "MUA_event"])
    .apply(lambda g: (g["non_local_event"] > 0).any())
    .mean()
    * 100.0
)

# %%
fig, ax = plt.subplots(figsize=(3.5, 2.5))

labels = [
    "SWR\nwith\nNon-local",
    "MUA\nwith\nNon-local",
]
values = [
    swr_non_local_overlap_pct,
    mua_non_local_overlap_pct,
]
colors = [COLORS["SWR"], COLORS["MUA"]]

bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor="none")

# Add value labels above bars
for bar, value in zip(bars, values, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Minimalist style
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

ax.set_ylabel("Overlap (%)")
ax.set_ylim(0, 100)
ax.set_xticklabels(labels)
ax.set_title("MUA/SWRs with Non-Local activity", pad=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_14.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
df = data_over_time.loc[is_immobile]

# Boolean masks
swr = df["SWR_event"] > 0
mua = df["MUA_event"] > 0
non_local = df["non_local_event"] > 0

# Venn regions (as timepoint counts)
only_swr = swr & ~mua & ~non_local
only_mua = mua & ~swr & ~non_local
only_non_local = non_local & ~swr & ~mua

swr_and_mua = swr & mua & ~non_local
swr_and_non_local = swr & non_local & ~mua
mua_and_non_local = mua & non_local & ~swr

all_three = swr & mua & non_local

total_immobile_time = len(df)

# Count timepoints in each category
counts = {
    "SWR only": 100 * only_swr.sum() / total_immobile_time,
    "MUA only": 100 * only_mua.sum() / total_immobile_time,
    "Non-local only": 100 * only_non_local.sum() / total_immobile_time,
    "SWR ∩ MUA": 100 * swr_and_mua.sum() / total_immobile_time,
    "SWR ∩ Non-local": 100 * swr_and_non_local.sum() / total_immobile_time,
    "MUA ∩ Non-local": 100 * mua_and_non_local.sum() / total_immobile_time,
    "All three": 100 * all_three.sum() / total_immobile_time,
}


# Step 3: Plot
fig, ax = plt.subplots(figsize=(8, 8))

# Draw the three circles
circle_swr = plt.Circle((0.4, 0.5), 0.25, color=COLORS["SWR"], alpha=0.3)
circle_mua = plt.Circle((0.6, 0.5), 0.25, color=COLORS["MUA"], alpha=0.3)
circle_nonlocal = plt.Circle((0.5, 0.7), 0.25, color=COLORS["Non-Local"], alpha=0.3)

ax.add_patch(circle_swr)
ax.add_patch(circle_mua)
ax.add_patch(circle_nonlocal)

# Step 4: Label the regions
ax.text(0.27, 0.5, f"{counts['SWR only']:.1f}%", ha="center")
ax.text(0.73, 0.5, f"{counts['MUA only']:.1f}%", ha="center")
ax.text(0.5, 0.8, f"{counts['Non-local only']:.1f}%", ha="center")

ax.text(0.5, 0.55, f"{counts['All three']:.1f}%", ha="center")  # center overlap

ax.text(0.5, 0.4, f"{counts['SWR ∩ MUA']:.1f}%", ha="center")
ax.text(0.35, 0.65, f"{counts['SWR ∩ Non-local']:.1f}%", ha="center")
ax.text(0.65, 0.65, f"{counts['MUA ∩ Non-local']:.1f}%", ha="center")

# Circle labels
ax.text(0.25, 0.35, "SWR", color=COLORS["SWR"], fontweight="bold")
ax.text(0.70, 0.35, "MUA", color=COLORS["MUA"], fontweight="bold")
ax.text(
    0.5,
    0.90,
    "Non-local",
    color=COLORS["Non-Local"],
    fontweight="bold",
    ha="center",
)

# Final plot setup
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Percent of Immobile Time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_15.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
df = data_over_time.loc[is_immobile]

# Boolean masks
swr = df["SWR_event"] > 0
mua = df["MUA_event"] > 0
non_local = df["non_local_event"] > 0

# Venn regions (as timepoint counts)
only_swr = swr & ~mua & ~non_local
only_mua = mua & ~swr & ~non_local
only_non_local = non_local & ~swr & ~mua

swr_and_mua = swr & mua & ~non_local
swr_and_non_local = swr & non_local & ~mua
mua_and_non_local = mua & non_local & ~swr

all_three = swr & mua & non_local


def count_events(mask: pd.Series, label_col: str) -> int:
    return len(
        df.loc[mask & (df[label_col] > 0)].groupby(
            ["animal", "day", "epoch", label_col]
        )
    )


# Count timepoints in each category
event_counts = {
    "SWR only": count_events(only_swr, "SWR_event"),
    "MUA only": count_events(only_mua, "MUA_event"),
    "Non-local only": count_events(only_non_local, "non_local_event"),
    "SWR ∩ MUA": count_events(swr_and_mua, "SWR_event"),  # you could also use MUA_event
    "SWR ∩ Non-local": count_events(swr_and_non_local, "SWR_event"),
    "MUA ∩ Non-local": count_events(mua_and_non_local, "MUA_event"),
    "All three": count_events(all_three, "SWR_event"),  # or any label — all overlapping
}

# Step 3: Plot
fig, ax = plt.subplots(figsize=(8, 8))

# Draw the three circles
circle_swr = plt.Circle((0.4, 0.5), 0.25, color=COLORS["SWR"], alpha=0.3)
circle_mua = plt.Circle((0.6, 0.5), 0.25, color=COLORS["MUA"], alpha=0.3)
circle_nonlocal = plt.Circle((0.5, 0.7), 0.25, color=COLORS["Non-Local"], alpha=0.3)

ax.add_patch(circle_swr)
ax.add_patch(circle_mua)
ax.add_patch(circle_nonlocal)

# Step 4: Label the regions
ax.text(0.27, 0.5, f"{event_counts['SWR only']}", ha="center")
ax.text(0.73, 0.5, f"{event_counts['MUA only']}", ha="center")
ax.text(0.5, 0.8, f"{event_counts['Non-local only']}", ha="center")

ax.text(
    0.5,
    0.55,
    f"{event_counts['All three']}",
    ha="center",
)  # center overlap

ax.text(0.5, 0.4, f"{event_counts['SWR ∩ MUA']}", ha="center")
ax.text(0.35, 0.65, f"{event_counts['SWR ∩ Non-local']}", ha="center")
ax.text(0.65, 0.65, f"{event_counts['MUA ∩ Non-local']}", ha="center")

# Circle labels
ax.text(0.25, 0.35, "SWR", color=COLORS["SWR"], fontweight="bold")
ax.text(0.70, 0.35, "MUA", color=COLORS["MUA"], fontweight="bold")
ax.text(
    0.5,
    0.90,
    "Non-local",
    color=COLORS["Non-Local"],
    fontweight="bold",
    ha="center",
)

# Final plot setup
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Number of Events")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_16.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
fig, ax = plt.subplots(figsize=(7, 4))

labels = [
    "Non-local\nwith SWR",
    "Non-local\nwith MUA",
    "SWR\nwith\nNon-local",
    "MUA\nwith\nNon-local",
]
values = [
    non_local_swr_overlap,
    non_local_mua_overlap,
    swr_non_local_overlap_pct,
    mua_non_local_overlap_pct,
]
colors = [COLORS["SWR"], COLORS["MUA"], COLORS["Non-Local"], COLORS["Non-Local"]]

bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor="none")

# Add value labels above bars
for bar, value in zip(bars, values, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Minimalist style
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

ax.set_ylabel("Overlap (%)")
ax.set_ylim(0, 100)
ax.set_xticklabels(labels)
ax.set_title("Event Overlap: Model vs. SWR/MUA", pad=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_17.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
swr_non_local_prob = (
    data_over_time.loc[data_over_time["SWR_event"] > 0]
    .groupby(["animal", "day", "epoch", "SWR_event"])["Non-Local_prob"]
    .mean()
    .mean()
)

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(7, 3), sharey=True, sharex=True, constrained_layout=True
)

data_over_time[data_over_time["SWR_event"] > 0]["Non-Local_prob"].hist(
    bins=np.arange(0, 1.05, 0.05),
    color=COLORS["Non-Local"],
    alpha=0.7,
    edgecolor="none",
    density=True,
    label="SWR Events",
    ax=axes[0],
)


axes[0].set_title("SWR Event\nNon-Local Probability")
axes[0].set_xlabel("Non-Local Probability")
axes[0].set_ylabel("Density")

data_over_time[data_over_time["MUA_event"] > 0]["Non-Local_prob"].hist(
    bins=np.arange(0, 1.05, 0.05),
    color=COLORS["Non-Local"],
    alpha=0.7,
    edgecolor="none",
    density=True,
    label="MUA Events",
    ax=axes[1],
)
axes[1].set_title("MUA Event\nNon-Local Probability")
axes[1].set_xlabel("Non-Local Probability")
sns.despine(ax=axes[0], top=True, right=True, left=True, bottom=False)
sns.despine(ax=axes[1], top=True, right=True, left=True, bottom=False)
axes[0].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
axes[1].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
axes[0].set_axisbelow(True)
axes[1].set_axisbelow(True)

# %%
mua_non_local_prob = (
    data_over_time.loc[data_over_time["MUA_event"] > 0]
    .groupby(["animal", "day", "epoch", "MUA_event"])["Non-Local_prob"]
    .mean()
    .mean()
)

# %%
data_over_time.loc[data_over_time["SWR_event"] > 0].groupby(
    ["animal", "day", "epoch", "SWR_event"]
).filter(lambda g: (g["non_local_event"] == 0).all()).groupby(
    ["animal", "day", "epoch", "SWR_event"]
)["Local_prob"].mean().mean()

# %%
data_over_time.loc[data_over_time["MUA_event"] > 0].groupby(
    ["animal", "day", "epoch", "MUA_event"]
).filter(lambda g: (g["non_local_event"] == 0).all()).groupby(
    ["animal", "day", "epoch", "MUA_event"]
)["Local_prob"].mean().mean()

# %%
(immobile_non_local_events.map_speed / 1000).hist(
    bins=np.arange(0, 30, 1),
    color=COLORS["Non-Local"],
    alpha=0.7,
    edgecolor="none",
    density=True,
)

# %%
(
    immobile_non_local_events.groupby(
        ["animal", "day", "epoch", "non_local_event"]
    ).map_speed.median()
    / 1000
).hist(
    bins=np.arange(0, 30, 1),
    color=COLORS["Non-Local"],
    alpha=0.7,
    edgecolor="none",
    density=True,
)

# %%
row_titles = [
    "Immobile Non-local Events",
    "SWR Events",
    "MUA Events",
]
col_titles = [
    "Continuous\nNon-local Probability",
    "Fragmented\nNon-local Probability",
]

fig, axes = plt.subplots(
    3, 2, figsize=(7, 8), sharex=True, sharey=True, constrained_layout=True
)

# Data for each subplot
plot_data = [
    (
        immobile_non_local_events["Non-Local_Continuous_prob"],
        COLORS["Non-Local"],
        axes[0, 0],
        row_titles[0],
        col_titles[0],
    ),
    (
        data_over_time.loc[
            data_over_time["SWR_event"] > 0, "Non-Local_Continuous_prob"
        ],
        COLORS["SWR"],
        axes[1, 0],
        row_titles[1],
        None,
    ),
    (
        data_over_time.loc[
            data_over_time["MUA_event"] > 0, "Non-Local_Continuous_prob"
        ],
        COLORS["MUA"],
        axes[2, 0],
        row_titles[2],
        None,
    ),
    (
        immobile_non_local_events["Non-Local_Fragmented_prob"],
        COLORS["Non-Local"],
        axes[0, 1],
        None,
        col_titles[1],
    ),
    (
        data_over_time.loc[
            data_over_time["SWR_event"] > 0, "Non-Local_Fragmented_prob"
        ],
        COLORS["SWR"],
        axes[1, 1],
        None,
        None,
    ),
    (
        data_over_time.loc[
            data_over_time["MUA_event"] > 0, "Non-Local_Fragmented_prob"
        ],
        COLORS["MUA"],
        axes[2, 1],
        None,
        None,
    ),
]

for data, color, ax, row_title, col_title in plot_data:
    ax.hist(
        data,
        density=True,
        bins=np.arange(0, 1.05, 0.05),
        color=color,
        alpha=0.7,
        edgecolor="none",
        histtype="stepfilled",
    )
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", which="both", length=3, direction="out", labelsize=12)
    if row_title:
        ax.set_ylabel(row_title)
    if col_title:
        ax.set_title(col_title, pad=10)

for i in range(3):
    axes[i, 0].set_xlabel("Probability")
    axes[i, 1].set_xlabel("Probability")

plt.suptitle("Distribution of Non-local Probabilities", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_18.pdf", bbox_inches="tight", dpi=300)
plt.close()


# %%
non_local_events_in_SWR = (
    data_over_time.loc[data_over_time["SWR_event"] > 0]
    .groupby(["animal", "day", "epoch", "SWR_event"])
    .apply(lambda g: pd.unique(g.loc[g["non_local_event"] > 0].non_local_event))
)
swr_start_time = (
    data_over_time.loc[data_over_time["SWR_event"] > 0]
    .groupby(["animal", "day", "epoch", "SWR_event"])
    .apply(lambda g: g.index[0])
)
swr_non_local_overlap = (
    data_over_time.loc[data_over_time["SWR_event"] > 0]
    .groupby(["animal", "day", "epoch", "SWR_event"])
    .apply(lambda g: (g["non_local_event"] > 0).any())
)
swr_start_time = (
    swr_start_time.loc[swr_non_local_overlap]
    .reset_index()
    .rename(columns={0: "swr_start_time"})
    .set_index(["animal", "day", "epoch", "SWR_event"])
)
non_local_events_in_SWR = (
    non_local_events_in_SWR.loc[swr_non_local_overlap]
    .reset_index()
    .rename(columns={0: "non_local_events"})
    .explode("non_local_events", ignore_index=True)
    .rename(columns={"non_local_events": "non_local_event"})
)
non_local_start_time = (
    data_over_time.loc[data_over_time["non_local_event"] > 0]
    .groupby(["animal", "day", "epoch", "non_local_event"])
    .apply(lambda g: g.index[0])
    .reset_index()
    .rename(columns={0: "non_local_start_time"})
    .set_index(["animal", "day", "epoch", "non_local_event"])
)

non_local_events_in_SWR = (
    non_local_events_in_SWR.set_index(["animal", "day", "epoch", "non_local_event"])
    .join(non_local_start_time)
    .reset_index()
    .set_index(["animal", "day", "epoch", "SWR_event"])
    .join(swr_start_time)
    .reset_index()
    .assign(time_diff=lambda df: (df["non_local_start_time"] - df["swr_start_time"]))
)

# %%
# Filter once to avoid repeating the same filtering operations
swr_data = data_over_time[data_over_time["SWR_event"] > 0]
non_local_data = data_over_time[data_over_time["non_local_event"] > 0]

# Group SWR events
group_keys = ["animal", "day", "epoch", "SWR_event"]
swr_groups = swr_data.groupby(group_keys)

# Identify SWRs with non-local overlap
swr_overlap_mask = swr_groups["non_local_event"].transform(lambda x: (x > 0).any())
swr_data = swr_data[swr_overlap_mask]

# Get SWR start times
swr_start_time = (
    swr_data.groupby(group_keys).apply(lambda g: g.index[0]).rename("swr_start_time")
)

# Get non-local events per SWR
non_local_events = (
    swr_data[swr_data["non_local_event"] > 0]
    .groupby(group_keys)["non_local_event"]
    .unique()
    .explode()
    .rename("non_local_event")
)

# Get non-local event start times
non_local_start_time = (
    non_local_data.groupby(["animal", "day", "epoch", "non_local_event"])
    .apply(lambda g: g.index[0])
    .rename("non_local_start_time")
)

# Join everything
non_local_events_df = non_local_events.reset_index().set_index(
    ["animal", "day", "epoch", "non_local_event"]
)
non_local_events_df = non_local_events_df.join(non_local_start_time)
non_local_events_df = non_local_events_df.reset_index().set_index(
    ["animal", "day", "epoch", "SWR_event"]
)
non_local_events_df = non_local_events_df.join(swr_start_time)
non_local_events_df = non_local_events_df.reset_index()

# Compute time difference
non_local_events_df["time_diff"] = (
    non_local_events_df["non_local_start_time"] - non_local_events_df["swr_start_time"]
)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Histogram
ax.hist(
    non_local_events_df["time_diff"] * 1000,  # convert to ms
    bins=np.arange(-100, 100.1, 5).tolist(),
    color=COLORS["SWR"],
    alpha=0.7,
    edgecolor="none",
    density=True,
    histtype="stepfilled",
)

# Add KDE for smoother distribution
sns.kdeplot(
    non_local_events_df["time_diff"] * 1000,
    ax=ax,
    color="black",
    linewidth=2,
    label="KDE",
    clip=(-100, 100),
)

# Add mean and median lines
mean_ms = non_local_events_df["time_diff"].mean() * 1000
median_ms = non_local_events_df["time_diff"].median() * 1000
ax.axvline(
    mean_ms, color="red", linestyle="--", lw=1.5, label=f"Mean: {mean_ms:.1f} ms"
)
ax.axvline(
    median_ms, color="blue", linestyle=":", lw=1.5, label=f"Median: {median_ms:.1f} ms"
)
ax.axvline(0, color="k", linestyle="--", lw=1, label="SWR Start")

# Style
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)
ax.set_xlabel("Time Difference (ms)\n(Non-local event start minus SWR start)")
ax.set_ylabel("Density")
ax.set_title("Timing of Non-local Events Relative to SWR Start", pad=10)
ax.set_xlim(-100, 100)
ax.tick_params(axis="both", which="both", length=3, direction="out")

# Annotate regions
ax.text(
    40,
    ax.get_ylim()[1] * 0.9,
    "Non-local event starts later",
    ha="right",
)
ax.text(
    -40,
    ax.get_ylim()[1] * 0.9,
    "Non-local event starts earlier",
    ha="left",
)

# Legend
ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_19.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
# Filter once to avoid repeating the same filtering operations
swr_data = data_over_time[data_over_time["SWR_event"] > 0]
non_local_data = data_over_time[data_over_time["non_local_event"] > 0]

# Group SWR events
group_keys = ["animal", "day", "epoch", "SWR_event"]
swr_groups = swr_data.groupby(group_keys)

# Identify SWRs with non-local overlap
swr_overlap_mask = swr_groups["non_local_event"].transform(lambda x: (x > 0).any())
swr_data = swr_data[swr_overlap_mask]

# Get SWR start times
swr_end_time = (
    swr_data.groupby(group_keys).apply(lambda g: g.index[-1]).rename("swr_end_time")
)

# Get non-local events per SWR
non_local_events = (
    swr_data[swr_data["non_local_event"] > 0]
    .groupby(group_keys)["non_local_event"]
    .unique()
    .explode()
    .rename("non_local_event")
)

# Get non-local event start times
non_local_end_time = (
    non_local_data.groupby(["animal", "day", "epoch", "non_local_event"])
    .apply(lambda g: g.index[-1])
    .rename("non_local_end_time")
)

# Join everything
non_local_events_df = non_local_events.reset_index().set_index(
    ["animal", "day", "epoch", "non_local_event"]
)
non_local_events_df = non_local_events_df.join(non_local_end_time)
non_local_events_df = non_local_events_df.reset_index().set_index(
    ["animal", "day", "epoch", "SWR_event"]
)
non_local_events_df = non_local_events_df.join(swr_end_time)
non_local_events_df = non_local_events_df.reset_index()

# Compute time difference
non_local_events_df["time_diff"] = (
    non_local_events_df["non_local_end_time"] - non_local_events_df["swr_end_time"]
)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Histogram
ax.hist(
    non_local_events_df["time_diff"] * 1000,  # convert to ms
    bins=np.arange(-100, 100.1, 5).tolist(),
    color=COLORS["SWR"],
    alpha=0.7,
    edgecolor="none",
    density=True,
    histtype="stepfilled",
)

# Add KDE for smoother distribution
sns.kdeplot(
    non_local_events_df["time_diff"] * 1000,
    ax=ax,
    color="black",
    linewidth=2,
    label="KDE",
    clip=(-100, 100),
)

# Add mean and median lines
mean_ms = non_local_events_df["time_diff"].mean() * 1000
median_ms = non_local_events_df["time_diff"].median() * 1000
ax.axvline(
    mean_ms, color="red", linestyle="--", lw=1.5, label=f"Mean: {mean_ms:.1f} ms"
)
ax.axvline(
    median_ms, color="blue", linestyle=":", lw=1.5, label=f"Median: {median_ms:.1f} ms"
)
ax.axvline(0, color="k", linestyle="--", lw=1, label="SWR End")

# Style
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)
ax.set_xlabel("Time Difference (ms)\n(Non-local event end minus SWR end)")
ax.set_ylabel("Density")
ax.set_title("Timing of Non-local Events Relative to SWR end", pad=10)
ax.set_xlim(-100, 100)
ax.tick_params(axis="both", which="both", length=3, direction="out")

# Annotate regions
ax.text(
    40,
    ax.get_ylim()[1] * 0.9,
    "Non-local event ends later",
    ha="right",
)
ax.text(
    -40,
    ax.get_ylim()[1] * 0.9,
    "Non-local event ends earlier",
    ha="left",
)

# Legend
ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_20.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
# Filter once to avoid repeating the same filtering operations
mua_data = data_over_time[data_over_time["MUA_event"] > 0]
non_local_data = data_over_time[data_over_time["non_local_event"] > 0]

# Group MUA events
group_keys = ["animal", "day", "epoch", "MUA_event"]
mua_groups = mua_data.groupby(group_keys)

# Identify MUAs with non-local overlap
mua_overlap_mask = mua_groups["non_local_event"].transform(lambda x: (x > 0).any())
mua_data = mua_data[mua_overlap_mask]

# Get MUA start times
mua_start_time = (
    mua_data.groupby(group_keys).apply(lambda g: g.index[0]).rename("mua_start_time")
)

# Get non-local events per MUA
non_local_events = (
    mua_data[mua_data["non_local_event"] > 0]
    .groupby(group_keys)["non_local_event"]
    .unique()
    .explode()
    .rename("non_local_event")
)

# Get non-local event start times
non_local_start_time = (
    non_local_data.groupby(["animal", "day", "epoch", "non_local_event"])
    .apply(lambda g: g.index[0])
    .rename("non_local_start_time")
)

# Join everything
non_local_events_df = non_local_events.reset_index().set_index(
    ["animal", "day", "epoch", "non_local_event"]
)
non_local_events_df = non_local_events_df.join(non_local_start_time)
non_local_events_df = non_local_events_df.reset_index().set_index(
    ["animal", "day", "epoch", "MUA_event"]
)
non_local_events_df = non_local_events_df.join(mua_start_time)
non_local_events_df = non_local_events_df.reset_index()

# Compute time difference
non_local_events_df["time_diff"] = (
    non_local_events_df["non_local_start_time"] - non_local_events_df["mua_start_time"]
)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Histogram
ax.hist(
    non_local_events_df["time_diff"] * 1000,  # convert to ms
    bins=np.arange(-100, 100.1, 5).tolist(),
    color=COLORS["MUA"],
    alpha=0.7,
    edgecolor="none",
    density=True,
    histtype="stepfilled",
)

# Add KDE for smoother distribution
sns.kdeplot(
    non_local_events_df["time_diff"] * 1000,
    ax=ax,
    color="black",
    linewidth=2,
    label="KDE",
    clip=(-100, 100),
)

# Add mean and median lines
mean_ms = non_local_events_df["time_diff"].mean() * 1000
median_ms = non_local_events_df["time_diff"].median() * 1000
ax.axvline(
    mean_ms, color="red", linestyle="--", lw=1.5, label=f"Mean: {mean_ms:.1f} ms"
)
ax.axvline(
    median_ms, color="blue", linestyle=":", lw=1.5, label=f"Median: {median_ms:.1f} ms"
)
ax.axvline(0, color="k", linestyle="--", lw=1, label="MUA Start")

# Style
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)
ax.set_xlabel(
    "Time Difference (ms)\n(Non-local event start minus MUA start)",
)
ax.set_ylabel("Density")
ax.set_title("Timing of Non-local Events Relative to MUA Start", pad=10)
ax.set_xlim(-100, 100)
ax.tick_params(axis="both", which="both", length=3, direction="out")

# Annotate regions
ax.text(
    40,
    ax.get_ylim()[1] * 0.9,
    "Non-local event starts later",
    ha="right",
)
ax.text(
    -40,
    ax.get_ylim()[1] * 0.9,
    "Non-local event starts earlier",
    ha="left",
)

# Legend
ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_21.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
# Filter once to avoid repeating the same filtering operations
mua_data = data_over_time[data_over_time["MUA_event"] > 0]
non_local_data = data_over_time[data_over_time["non_local_event"] > 0]

# Group MUA events
group_keys = ["animal", "day", "epoch", "MUA_event"]
mua_groups = mua_data.groupby(group_keys)

# Identify MUAs with non-local overlap
mua_overlap_mask = mua_groups["non_local_event"].transform(lambda x: (x > 0).any())
mua_data = mua_data[mua_overlap_mask]

# Get MUA start times
mua_end_time = (
    mua_data.groupby(group_keys).apply(lambda g: g.index[-1]).rename("mua_end_time")
)

# Get non-local events per MUA
non_local_events = (
    mua_data[mua_data["non_local_event"] > 0]
    .groupby(group_keys)["non_local_event"]
    .unique()
    .explode()
    .rename("non_local_event")
)

# Get non-local event start times
non_local_end_time = (
    non_local_data.groupby(["animal", "day", "epoch", "non_local_event"])
    .apply(lambda g: g.index[-1])
    .rename("non_local_end_time")
)

# Join everything
non_local_events_df = non_local_events.reset_index().set_index(
    ["animal", "day", "epoch", "non_local_event"]
)
non_local_events_df = non_local_events_df.join(non_local_end_time)
non_local_events_df = non_local_events_df.reset_index().set_index(
    ["animal", "day", "epoch", "MUA_event"]
)
non_local_events_df = non_local_events_df.join(mua_end_time)
non_local_events_df = non_local_events_df.reset_index()

# Compute time difference
non_local_events_df["time_diff"] = (
    non_local_events_df["non_local_end_time"] - non_local_events_df["mua_end_time"]
)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Histogram
ax.hist(
    non_local_events_df["time_diff"] * 1000,  # convert to ms
    bins=np.arange(-100, 100.1, 5).tolist(),
    color=COLORS["MUA"],
    alpha=0.7,
    edgecolor="none",
    density=True,
    histtype="stepfilled",
)

# Add KDE for smoother distribution
sns.kdeplot(
    non_local_events_df["time_diff"] * 1000,
    ax=ax,
    color="black",
    linewidth=2,
    label="KDE",
    clip=(-100, 100),
)

# Add mean and median lines
mean_ms = non_local_events_df["time_diff"].mean() * 1000
median_ms = non_local_events_df["time_diff"].median() * 1000
ax.axvline(
    mean_ms, color="red", linestyle="--", lw=1.5, label=f"Mean: {mean_ms:.1f} ms"
)
ax.axvline(
    median_ms, color="blue", linestyle=":", lw=1.5, label=f"Median: {median_ms:.1f} ms"
)
ax.axvline(0, color="k", linestyle="--", lw=1, label="MUA End")

# Style
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)
ax.set_xlabel(
    "Time Difference (ms)\n(Non-local event end minus MUA end)",
)
ax.set_ylabel("Density")
ax.set_title("Timing of Non-local Events Relative to MUA End", pad=10)
ax.set_xlim(-100, 100)
ax.tick_params(axis="both", which="both", length=3, direction="out")

# Annotate regions
ax.text(
    40,
    ax.get_ylim()[1] * 0.9,
    "Non-local event ends later",
    ha="right",
)
ax.text(
    -40,
    ax.get_ylim()[1] * 0.9,
    "Non-local event ends earlier",
    ha="left",
)

# Legend
ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure02_panel_22.pdf", bbox_inches="tight", dpi=300)
plt.close()

# %%
