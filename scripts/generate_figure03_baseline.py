# %%
# %config InlineBackend.figure_format = 'retina'
import logging
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.ndimage import label
from scipy.stats import sem

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.parameters import (
    COLORS,
    PROCESSED_DATA_DIR,
    SAMPLING_FREQUENCY,
)
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")

# Output directory for baseline figures
OUTPUT_DIR = Path("tests/baselines/figure03")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_over_time():
    # Load all relevant data files
    data_path = PROCESSED_DATA_DIR.glob("*_HPC_80_data_over_time.pkl")
    data_over_time = pd.concat([pd.read_pickle(p) for p in data_path])
    data_over_time["Non-Local_prob"] = (
        data_over_time["Non-Local_Continuous_prob"]
        + data_over_time["Non-Local_Fragmented_prob"]
    )
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])

    return data_over_time


def get_event_types(data_over_time: dict):
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
def get_duration(df: pd.DataFrame, event_name="non_local_event") -> pd.Series:
    duration = (
        df.groupby(["animal", "day", "epoch", event_name]).size() / SAMPLING_FREQUENCY
    )
    return duration * 1000.0  # Convert to milliseconds


fig, ax = plt.subplots(figsize=(4, 3))
for event_type in ("non_local_immobile", "non_local_immobile_no_overlap_SWR"):
    duration = get_duration(event_types[event_type], event_name="non_local_event")

    color = (
        COLORS["Non-Local"]
        if event_type == "non_local_immobile_no_overlap_SWR"
        else "lightgrey"
    )
    ax.hist(
        duration,
        bins=np.arange(0, 250, 10),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Event Duration (ms)")
    ax.set_ylabel("Density")

plt.savefig(OUTPUT_DIR / "figure03_panel_01.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_02.pdf", bbox_inches="tight", dpi=300)
plt.close()
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
        bins=np.arange(0, 230, 2),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Distance from Animal (cm)")
    ax.set_ylabel("Density")

plt.savefig(OUTPUT_DIR / "figure03_panel_03.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_04.pdf", bbox_inches="tight", dpi=300)
plt.close()
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
        bins=np.arange(0, 20, 0.5),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("Replay Speed")
    ax.set_ylabel("Density")

plt.savefig(OUTPUT_DIR / "figure03_panel_05.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_06.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_07.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_08.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_09.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_10.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_11.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_12.pdf", bbox_inches="tight", dpi=300)
plt.close()
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

plt.savefig(OUTPUT_DIR / "figure03_panel_13.pdf", bbox_inches="tight", dpi=300)
plt.close()
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
        bins=np.arange(0, 250, 10),
        color=color,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("HPD")
    ax.set_ylabel("Density")


plt.savefig(OUTPUT_DIR / "figure03_panel_14.pdf", bbox_inches="tight", dpi=300)
plt.close()
# %%
def get_event_start_end(df: pd.DataFrame, event_name="non_local_event") -> pd.DataFrame:
    event_times = (
        df.reset_index()
        .groupby(["animal", "day", "epoch", event_name])["index"]
        .agg(["min", "max"])
        .rename(columns={"min": "start_time", "max": "end_time"})
        .reset_index()
    )
    return event_times


non_local_start_end_time = get_event_start_end(
    event_types["non_local_immobile"], event_name="non_local_event"
)

# %%


def _get_signal_windows(
    group: pd.DataFrame,
    start_end_time: pd.DataFrame,
    animal_day_epoch: tuple[str, int, str],
    time_type: str,
    signal_types: list[str],
    total_window_len_seconds=2.0,
    event_column: str | None = None,
) -> xr.DataArray:
    """Extract signal windows around event times.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing the signal data for a specific animal, day, and epoch.
    start_end_time : pd.DataFrame
        DataFrame containing the start and end times of events.
    animal : str
        Animal identifier.
    day : int
        Day identifier.
    epoch : str
        Epoch identifier.
    time_type : str
        Column name in start_end_time to use for event times ('start_time' or 'end_time').
    signal_type : str
        Column name in group to extract signal from.
    total_window_len_seconds : float, optional
        Total length of the window in seconds (default is 2.0 seconds).

    Returns
    -------
    np.ndarray, shape (n_events, window_length)
        signal windows.

    """
    animal, day, epoch = animal_day_epoch
    cur_event_times = start_end_time.loc[
        (start_end_time.animal == animal)
        & (start_end_time.day == day)
        & (start_end_time.epoch == epoch)
    ]
    event_times = cur_event_times[time_type].to_numpy()
    n_signals = len(signal_types)

    if len(event_times) == 0:
        return np.empty(
            (
                0,
                int(round(total_window_len_seconds * SAMPLING_FREQUENCY)) + 1,
                n_signals,
            )
        )

    window = total_window_len_seconds / 2.0  # Half window on each side
    dt = 1 / SAMPLING_FREQUENCY
    half_window_samples = int(round(window / dt))
    window_len = 2 * half_window_samples + 1
    offsets = np.arange(-half_window_samples, half_window_samples + 1)
    windows = np.full((len(event_times), window_len, n_signals), np.nan)
    center_idxs = group.index.searchsorted(event_times)
    all_window_idxs = center_idxs[:, None] + offsets[None, :]
    mask = (all_window_idxs >= 0) & (all_window_idxs < len(group))
    windows = np.full((len(event_times), window_len, n_signals), np.nan)
    valid_rows, valid_cols = np.where(mask)
    valid_ind = all_window_idxs[mask]

    for i, signal_type in enumerate(signal_types):
        windows[valid_rows, valid_cols, i] = group[signal_type].to_numpy()[valid_ind]

    time_axis = np.arange(-half_window_samples, half_window_samples + 1) * dt
    if event_column is not None:
        event_ids = pd.MultiIndex.from_frame(
            cur_event_times.loc[:, ["animal", "day", "epoch", event_column]]
        )
    else:
        event_ids = np.arange(len(event_times))
    da = xr.DataArray(
        windows,
        dims=("event", "time", "signal"),
        coords={
            "event": event_ids,
            "time": time_axis,
            "signal": signal_types,
        },
        attrs={
            "event_type": event_column,
        },
    )
    return da


signal_types = [
    "Non-Local_prob",
    "theta_power_zscore",
    "MUA_zscore",
    "SWR_zscore",
    "head_speed",
]

# _get_signal_windows(
#     group=group,
#     start_end_time=non_local_start_end_time,
#     animal_day_epoch=(animal, day, epoch),
#     time_type="start_time",
#     signal_types=signal_types,
#     total_window_len_seconds=2.0,
#     event_column="non_local_event",
# )

signal_windows = xr.concat(
    [
        _get_signal_windows(
            group=group,
            start_end_time=non_local_start_end_time,
            animal_day_epoch=(animal, day, epoch),
            time_type="start_time",
            signal_types=signal_types,
            total_window_len_seconds=2.0,
            event_column="non_local_event",
        )
        for (animal, day, epoch), group in data_over_time.groupby(
            ["animal", "day", "epoch"]
        )
    ],
    dim="event",
)

# %%
non_local_start_end_time = get_event_start_end(
    event_types["non_local_immobile_no_overlap_SWR"], event_name="non_local_event"
)

signal_windows = xr.concat(
    [
        _get_signal_windows(
            group=group,
            start_end_time=non_local_start_end_time,
            animal_day_epoch=(animal, day, epoch),
            time_type="start_time",
            signal_types=signal_types,
            total_window_len_seconds=2.0,
            event_column="non_local_event",
        )
        for (animal, day, epoch), group in data_over_time.groupby(
            ["animal", "day", "epoch"]
        )
    ],
    dim="event",
)

# %%
non_local_start_end_time = get_event_start_end(
    event_types["non_local_immobile_overlap_swr"], event_name="non_local_event"
)

signal_windows_overlap = xr.concat(
    [
        _get_signal_windows(
            group=group,
            start_end_time=non_local_start_end_time,
            animal_day_epoch=(animal, day, epoch),
            time_type="start_time",
            signal_types=signal_types,
            total_window_len_seconds=2.0,
            event_column="non_local_event",
        )
        for (animal, day, epoch), group in data_over_time.groupby(
            ["animal", "day", "epoch"]
        )
    ],
    dim="event",
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

plt.savefig(OUTPUT_DIR / "figure03_panel_15.pdf", bbox_inches="tight", dpi=300)
plt.close()
# %%
non_local_start_end_time = get_event_start_end(
    event_types["non_local_immobile_no_overlap_SWR"], event_name="non_local_event"
)

signal_windows = xr.concat(
    [
        _get_signal_windows(
            group=group,
            start_end_time=non_local_start_end_time,
            animal_day_epoch=(animal, day, epoch),
            time_type="end_time",
            signal_types=signal_types,
            total_window_len_seconds=2.0,
            event_column="non_local_event",
        )
        for (animal, day, epoch), group in data_over_time.groupby(
            ["animal", "day", "epoch"]
        )
    ],
    dim="event",
)

# %%
non_local_start_end_time = get_event_start_end(
    event_types["non_local_immobile_overlap_swr"], event_name="non_local_event"
)

signal_windows_overlap = xr.concat(
    [
        _get_signal_windows(
            group=group,
            start_end_time=non_local_start_end_time,
            animal_day_epoch=(animal, day, epoch),
            time_type="end_time",
            signal_types=signal_types,
            total_window_len_seconds=2.0,
            event_column="non_local_event",
        )
        for (animal, day, epoch), group in data_over_time.groupby(
            ["animal", "day", "epoch"]
        )
    ],
    dim="event",
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

plt.savefig(OUTPUT_DIR / "figure03_panel_16.pdf", bbox_inches="tight", dpi=300)
plt.close()
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
