# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: continuum-swr-replay (3.11.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Single-epoch walkthrough: from raw data to the replay-content regressions
#
# This notebook walks the **base non-local analysis pipeline for one session** —
# `chimi`, day `20200213`, epoch `02_r1`, hippocampus (HPC) — from the
# canonical decoder and behavioral-belief artifacts through four regression-
# model families: arm content (M), patch content (P), per-stop event rate (R,
# Family A), and next-arm choice (C). The arm/patch sections include the
# Beta-Bernoulli value and uncertainty branches that are central to the story.
#
# Not covered here: the per-bin conditional-intensity rate ladder (Family B,
# `MODEL_PER_BIN_RATE`), the dedicated SWR-contrast ladders, and event-source
# comparisons. See `docs/models_table.md` for the full model catalog.
#
# **Why one epoch is the right fitting unit.** The production pipeline
# (`scripts/content/fit_replay_content_models.py`,
# `scripts/content/compare_event_source_models.py`, and
# `scripts/rate/fit_per_stop_rate_models.py`) fits models separately for each
# `(animal, day, epoch)`. The population result aggregates those session fits;
# this notebook reproduces one fitting unit, not population inference.
#
# **Decoding is loaded, not re-run.** The non-local state-space decoder is
# GPU-heavy and runs for hours per epoch. Its outputs are already on disk
# (`Processed-Data/*_results.nc`, `*_environment.pkl`,
# `*_data_over_time.pkl`). We *show* the decode call for reference but **load**
# its outputs. Building the posterior-weighted tables is still memory-intensive.
# The externally fitted behavioral model is likewise **loaded, not re-fit**:
# `qvals_table.parquet` carries its pre-outcome trial beliefs, and the persisted
# all-session reference supplies the correct global standardization.
#
# **Current canonical assembly.** Table construction mirrors
# `continuum_swr_replay.analysis.replay.prepare_data`: validate `dot`, build the
# zero-inclusive stop denominator first, construct event/arm/sequence tables,
# merge retrospective metrics, derive the SWR/non-SWR stop split, and check
# cross-table invariants. Model fits use the package fitters and freeze one
# comparison-valid complete-case cohort per block. The rate section separately
# reports the same per-rung cross-validation metric as its production runner.
#
# ## Pipeline
#
# ```
# load_data(trials only; position is always loaded)
#              │
# precomputed dot + environment + posterior
#              │
# validate_dot_event_contract
#              │
#      ┌──────────┴───────────┐
#      ▼                     ▼
# build_stop_table       build_event_table       qvals + frozen global z
#      │                     │                           │
#      │             build_event_arm_table ◄────────────┘
#      │                │             │
#      │                ▼             ▼
#      │       event_sequence +    event_patch_table
#      │       retrospective merge
#      └──────────┬──────────────────┘
#                 ▼
#       invariant + common-cohort gates
#        │             │             │          │          │
#        ▼             ▼             ▼          ▼          ▼
#      M0–M7    M5/Mv/Mu/Mvu      P + Pv    R0–R2/Rlag   C primary
# ```

# %%
import logging
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D

# --- Data loading + decode outputs ---
from continuum_swr_replay.analysis.events.cohort_selection import freeze_common_cohort
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.replay import add_arm_fixed_effects
from continuum_swr_replay.analysis.replay.choice_prediction import (
    build_choice_dataset,
    fit_choice_model,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.covariates import load_contingency

# --- Fitters (statsmodels conditional models / Poisson GLM) ---
from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay.model_checking import (
    cross_validate_conditional_poisson,
)

# --- Model specifications (the four families walked here) ---
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_CHOICE,  # C_*          : next-arm choice
    MODEL_HIERARCHY,  # M0–M7        : arm content
    MODEL_PATCH,  # P0–P4        : patch content
    MODEL_PER_STOP_RATE,  # R0-R2, Rlag  : event rate per stop (Family A)
    N_REWARD_LAGS,
    default_content_model_names,
)
from continuum_swr_replay.analysis.replay.per_stop_rate_model import (
    PER_STOP_RATE_OUTCOMES,
    cross_validate_per_stop_rate_model,
    fit_per_stop_rate_model,
)
from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
    load_conditional_posterior,
)
from continuum_swr_replay.analysis.replay.retrospective import (
    compute_retrospective_metrics,
)
from continuum_swr_replay.tables.standardization import (
    attach_standardized,
    load_reference,
)

# --- Canonical table assembly (same functions as prepare_data) ---
from continuum_swr_replay.analysis.replay._tables_common import (
    EVENT_ARM_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.replay.tables_arm_patch import (
    build_event_arm_table,
    build_event_patch_table,
)
from continuum_swr_replay.analysis.replay.tables_event import (
    build_event_sequence_table,
    build_event_table,
)
from continuum_swr_replay.analysis.replay.tables_stop import (
    add_nonlocal_swr_split,
    build_stop_table,
)
from continuum_swr_replay.analysis.replay.uncertainty_content import (
    build_trial_leaf_table,
    hierarchical_quartile_profile,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, table_path
from continuum_swr_replay.data_types import SessionDataKeys
from continuum_swr_replay.visualization.session_tables import (
    BEHAVIORAL_MODEL_PANELS,
    plot_session_tables,
)

logging.basicConfig(level=logging.WARNING)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

# Explicit comparison blocks keep qvals coverage from shrinking unrelated base
# ladders while still fitting the behavioral-belief branches on one fair cohort.
ARM_MODELS = {
    name: MODEL_HIERARCHY[name]
    for name in default_content_model_names(MODEL_HIERARCHY)
}
PATCH_MODELS = {
    name: MODEL_PATCH[name]
    for name in default_content_model_names(MODEL_PATCH)
}
BELIEF_ARM_MODELS = {
    name: MODEL_HIERARCHY[name] for name in ("M5", "Mv", "Mu", "Mvu")
}
BELIEF_PATCH_MODELS = {name: MODEL_PATCH[name] for name in ("P3", "Pv")}
RATE_MODELS = dict(MODEL_PER_STOP_RATE)
CHOICE_MODELS = {
    name: MODEL_CHOICE[name]
    for name in ("C_replay_only", "C_behavior", "C_full")
}
CHOICE_SPLIT_MODELS = {
    name: MODEL_CHOICE[name] for name in ("C_behavior", "C_full_swr_split")
}

for model_block in (
    ARM_MODELS,
    PATCH_MODELS,
    BELIEF_ARM_MODELS,
    BELIEF_PATCH_MODELS,
    RATE_MODELS,
    CHOICE_MODELS,
    CHOICE_SPLIT_MODELS,
):
    for spec in model_block.values():
        assert_predictor_timing(spec.predictors)

# %% [markdown]
# ## 0. Session selection
#
# `chimi20200213_02_r1` — confirmed to have precomputed decoder outputs and a
# rich event count. The identifiers below drive every subsequent cell; point
# them at another `(nwb_file_name, epoch_name)` to repeat the whole notebook
# for a different epoch of the same day (`04_r2`, `06_r3`) or any other
# session.

# %%
NWB_FILE_NAME = "chimi20200213_.nwb"
EPOCH_NAME = "02_r1"
BRAIN_AREA = "HPC"

SESSION_ID = f"{NWB_FILE_NAME.replace('_.nwb', '')}_{EPOCH_NAME}_{BRAIN_AREA}"
DOT_FILE = PROCESSED_DATA_DIR / f"{SESSION_ID}_80_data_over_time.pkl"
ENV_FILE = PROCESSED_DATA_DIR / f"{SESSION_ID}_environment.pkl"
RESULTS_FILE = PROCESSED_DATA_DIR / f"{SESSION_ID}_results.nc"
QVALS_FILE = table_path("qvals_table")
ARM_Z_REFERENCE_FILE = REPLAY_CONTENT_DIR / "standardization_reference.json"
HORSERACE_COEFS_FILE = (
    REPLAY_CONTENT_DIR
    / "uncertainty_content"
    / "horserace_session_coefficients.csv"
)

for p in (
    DOT_FILE,
    ENV_FILE,
    RESULTS_FILE,
    QVALS_FILE,
    ARM_Z_REFERENCE_FILE,
    HORSERACE_COEFS_FILE,
):
    print(f"{'OK ' if p.exists() else 'MISSING'} {p.name}")

# %% [markdown]
# ## 1. Load trials and track geometry from Spyglass
#
# The canonical prep driver asks Spyglass only for `trials` and the track graph;
# position is always loaded because it defines the session time bounds. Neural
# signals and decoder summaries are already represented in the precomputed
# `dot` artifact, so loading them again would add cost without changing any
# table below. Production can serve trials/graph from its `_session_cache` for
# covariate-only rebuilds; this walkthrough performs the live load.

# %%
data = load_data(
    NWB_FILE_NAME,
    EPOCH_NAME,
    include=frozenset({SessionDataKeys.TRIALS}),
)

position_info = data["position_info"]
trials = data["trials"]
track_graph = data["track_graph"]
linear_edge_order = data["linear_edge_order"]
linear_edge_spacing = data["linear_edge_spacing"]

print("loaded keys   :", sorted(data))
print("position_info :", position_info.shape, "| time-indexed in seconds")
print("trials        :", trials.shape)
print("\nposition_info columns:\n", list(position_info.columns))

# %%
trials.head()

# %% [markdown]
# ### 1a. Orientation plots
#
# Linearized position over the whole epoch (left) and the 2-D track-graph
# layout with the nine arms the decoder maps replay onto (right). The
# "three-legged spider": six leaf arms (well ↔ patch junction, arms 0–5) and
# three trunk arms (patch junction ↔ center, arms 6–8).

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), width_ratios=[2, 1])

# Linearized position over time, colored by head speed.
ax = axes[0]
sc = ax.scatter(
    position_info.index,
    position_info["linear_position"],
    c=position_info["head_speed"],
    s=1,
    cmap="magma",
    vmin=0,
    vmax=50,
    rasterized=True,
)
ax.set(xlabel="time (s)", ylabel="linear position (cm)",
       title=f"{SESSION_ID}: linearized trajectory")
fig.colorbar(sc, ax=ax, label="head speed (cm/s)", pad=0.01)

# 2-D track graph with arm (edge) ids.
ax = axes[1]
node_pos = {n: track_graph.nodes[n]["pos"] for n in track_graph.nodes}
nx.draw_networkx_edges(track_graph, node_pos, ax=ax, edge_color="0.7", width=2)
nx.draw_networkx_nodes(track_graph, node_pos, ax=ax, node_size=120,
                       node_color="0.3")
nx.draw_networkx_labels(track_graph, node_pos, ax=ax, font_color="white",
                        font_size=8)
for u, v, d in track_graph.edges(data=True):
    if "edge_id" in d:
        mx = 0.5 * (node_pos[u][0] + node_pos[v][0])
        my = 0.5 * (node_pos[u][1] + node_pos[v][1])
        ax.text(mx, my, f"arm {d['edge_id']}", color="C3", fontsize=8,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                          alpha=0.8))
ax.set(title="track graph (nodes = wells/junctions)")
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()

# %%
# check that trials make sense
from matplotlib.patches import Patch

from continuum_swr_replay.data_loaders import WELL_PATCH_MAPPING
from continuum_swr_replay.visualization import plot_graph_as_1D

t_start = float(position_info.index[0])
t_stop = t_start + 750.0

# Linear position of each degree-1 node (well), mirroring the cumulative
# arithmetic in plot_graph_as_1D so markers land on the same y axis.
spacing = (
    [float(linear_edge_spacing)] * (len(linear_edge_order) - 1)
    if np.isscalar(linear_edge_spacing)
    else list(linear_edge_spacing)
)
well_linear = {}
edge_start = 0.0
for i, (u, v) in enumerate(np.asarray(linear_edge_order)):
    edge_end = edge_start + track_graph.edges[u, v]["distance"]
    if track_graph.degree(u) == 1:
        well_linear[int(u)] = edge_start
    if track_graph.degree(v) == 1:
        well_linear[int(v)] = edge_end
    edge_start = edge_end + (spacing[i] if i < len(spacing) else 0.0)

PATCH_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}  # Okabe-Ito

fig, ax = plt.subplots(figsize=(15, 3.5))

# spreadsheet-style alternating grey bands, one per trial
for row, (_, trial) in enumerate(trials.iterrows()):
    ax.axvspan(
        trial["movement_start_time"], trial["end_time"],
        color="0.85" if row % 2 else "0.95", linewidth=0, zorder=0,
    )

ax.scatter(
    position_info.index, position_info["linear_position"],
    s=1, color="magenta", rasterized=True, zorder=2,
)

blend = ax.get_xaxis_transform()
for _, trial in trials.iterrows():
    from_well, to_well = int(trial["from_well"]), int(trial["to_well"])
    # departure: open square at from_well, colored by from_patch
    ax.plot(
        trial["movement_start_time"], well_linear[from_well],
        marker="s", ms=6, mfc="none",
        mec=PATCH_COLORS[WELL_PATCH_MAPPING[from_well]], mew=1.5, zorder=4,
    )
    # arrival: circle at destination_first_poke_in_time, filled iff rewarded
    dest_color = PATCH_COLORS[WELL_PATCH_MAPPING[to_well]]
    ax.plot(
        trial["destination_first_poke_in_time"], well_linear[to_well],
        marker="o", ms=7,
        mfc=dest_color if trial["is_reward"] else "none",
        mec=dest_color, mew=1.5, zorder=4,
    )
    if trial["is_patch_change"]:
        ax.plot(
            trial["movement_start_time"], 0.97, marker="v", ms=6, color="black",
            transform=blend, zorder=5,
        )

ax.set(
    xlabel="time (s)", ylabel="linear position (cm)",
    title=f"{SESSION_ID}: position and trials", xlim=(t_start, t_stop),
)
well_axis = ax.secondary_yaxis("right")
well_axis.set_yticks([well_linear[w] for w in sorted(well_linear)])
well_axis.set_yticklabels([f"well {w}" for w in sorted(well_linear)], fontsize=8)


def _marker(marker, filled, label, color="0.35"):
    """Neutral-grey proxy handle — real markers are colored by patch."""
    return Line2D(
        [], [], linestyle="none", marker=marker, ms=7,
        mfc=color if filled else "none", mec=color, mew=1.5, label=label,
    )


legend_handles = [
    Line2D([], [], linestyle="none", marker="o", ms=4, color="magenta",
           label="linear position"),
    Patch(facecolor="0.85", edgecolor="none", label="trial (alternating)"),
    _marker("s", False, "departure (from_well)"),
    _marker("o", True, "arrival, rewarded"),
    _marker("o", False, "arrival, unrewarded"),
    _marker("v", True, "patch change", color="black"),
] + [
    Patch(facecolor=PATCH_COLORS[p], edgecolor="none", label=f"patch {p}")
    for p in sorted(PATCH_COLORS)
]
ax.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
    ncol=5, fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.5,
)

reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
plot_graph_as_1D(
    track_graph, ax=ax,
    edge_order=linear_edge_order, edge_spacing=linear_edge_spacing,
    reward_well_nodes=reward_well_nodes,
    other_axis_start=t_start,
    reward_well_size=120, edge_linewidth=5,
)

# Redundant columns — assert rather than draw.
patch_from = trials["from_well"].map(WELL_PATCH_MAPPING)
patch_to = trials["to_well"].map(WELL_PATCH_MAPPING)
print("max |duration - (end - start)| :",
      float((trials["duration"] - (trials["end_time"] - trials["movement_start_time"])).abs().max()))
print("from_patch/to_patch match wells:",
      bool((patch_from == trials["from_patch"]).all() and (patch_to == trials["to_patch"]).all()))
print("is_patch_change matches patches:",
      bool(((patch_from != patch_to) == trials["is_patch_change"]).all()))
print("destination_first_poke_in_time inside trial :",
      bool(((trials["destination_first_poke_in_time"] >= trials["movement_start_time"])
            & (trials["destination_first_poke_in_time"] <= trials["end_time"])).all()))
print("wells chain (to[i] == from[i+1]):",
      bool((trials["to_well"].values[:-1] == trials["from_well"].values[1:]).all()))

trials


# %%
# check that trials make sense
from matplotlib.patches import Patch

from continuum_swr_replay.data_loaders import WELL_PATCH_MAPPING
from continuum_swr_replay.visualization import plot_graph_as_1D

t_start = float(position_info.index[0]) + 750.0
t_stop = t_start + 750.0

# Linear position of each degree-1 node (well), mirroring the cumulative
# arithmetic in plot_graph_as_1D so markers land on the same y axis.
spacing = (
    [float(linear_edge_spacing)] * (len(linear_edge_order) - 1)
    if np.isscalar(linear_edge_spacing)
    else list(linear_edge_spacing)
)
well_linear = {}
edge_start = 0.0
for i, (u, v) in enumerate(np.asarray(linear_edge_order)):
    edge_end = edge_start + track_graph.edges[u, v]["distance"]
    if track_graph.degree(u) == 1:
        well_linear[int(u)] = edge_start
    if track_graph.degree(v) == 1:
        well_linear[int(v)] = edge_end
    edge_start = edge_end + (spacing[i] if i < len(spacing) else 0.0)

PATCH_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}  # Okabe-Ito

fig, ax = plt.subplots(figsize=(15, 3.5))

# spreadsheet-style alternating grey bands, one per trial
for row, (_, trial) in enumerate(trials.iterrows()):
    ax.axvspan(
        trial["movement_start_time"], trial["end_time"],
        color="0.85" if row % 2 else "0.95", linewidth=0, zorder=0,
    )

ax.scatter(
    position_info.index, position_info["linear_position"],
    s=1, color="magenta", rasterized=True, zorder=2,
)

blend = ax.get_xaxis_transform()
for _, trial in trials.iterrows():
    from_well, to_well = int(trial["from_well"]), int(trial["to_well"])
    # departure: open square at from_well, colored by from_patch
    ax.plot(
        trial["movement_start_time"], well_linear[from_well],
        marker="s", ms=6, mfc="none",
        mec=PATCH_COLORS[WELL_PATCH_MAPPING[from_well]], mew=1.5, zorder=4,
    )
    # arrival: circle at destination_first_poke_in_time, filled iff rewarded
    dest_color = PATCH_COLORS[WELL_PATCH_MAPPING[to_well]]
    ax.plot(
        trial["destination_first_poke_in_time"], well_linear[to_well],
        marker="o", ms=7,
        mfc=dest_color if trial["is_reward"] else "none",
        mec=dest_color, mew=1.5, zorder=4,
    )
    if trial["is_patch_change"]:
        ax.plot(
            trial["movement_start_time"], 0.97, marker="v", ms=6, color="black",
            transform=blend, zorder=5,
        )

ax.set(
    xlabel="time (s)", ylabel="linear position (cm)",
    title=f"{SESSION_ID}: position and trials", xlim=(t_start, t_stop),
)
well_axis = ax.secondary_yaxis("right")
well_axis.set_yticks([well_linear[w] for w in sorted(well_linear)])
well_axis.set_yticklabels([f"well {w}" for w in sorted(well_linear)], fontsize=8)


def _marker(marker, filled, label, color="0.35"):
    """Neutral-grey proxy handle — real markers are colored by patch."""
    return Line2D(
        [], [], linestyle="none", marker=marker, ms=7,
        mfc=color if filled else "none", mec=color, mew=1.5, label=label,
    )


legend_handles = [
    Line2D([], [], linestyle="none", marker="o", ms=4, color="magenta",
           label="linear position"),
    Patch(facecolor="0.85", edgecolor="none", label="trial (alternating)"),
    _marker("s", False, "departure (from_well)"),
    _marker("o", True, "arrival, rewarded"),
    _marker("o", False, "arrival, unrewarded"),
    _marker("v", True, "patch change", color="black"),
] + [
    Patch(facecolor=PATCH_COLORS[p], edgecolor="none", label=f"patch {p}")
    for p in sorted(PATCH_COLORS)
]
ax.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
    ncol=5, fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.5,
)

reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
plot_graph_as_1D(
    track_graph, ax=ax,
    edge_order=linear_edge_order, edge_spacing=linear_edge_spacing,
    reward_well_nodes=reward_well_nodes,
    other_axis_start=t_start,
    reward_well_size=120, edge_linewidth=5,
)


# %% [markdown]
# ## 2. Decoding — load precomputed outputs
#
# The non-local clusterless detector (`NonLocalClusterlessDetector`) was fit on
# no-ripple training timepoints and applied to the whole epoch by
# `scripts/decoding/run_decoding_bandit.py`. For reference, that call is:
#
# ```python
# # NOT executed here — GPU, hours. Outputs already on disk.
# from scripts.decoding.run_decoding_bandit import run_decode
# run_decode(nwb_file_name=NWB_FILE_NAME, epoch_name=EPOCH_NAME,
#            brain_area="HPC")
# ```
#
# We load its three artifacts:
#
# * **`data_over_time` (`dot`)** — the per-timebin summary table: state
#   probabilities, MAP decode (`map_estimate`), event labels
#   (`non_local_event`, `SWR_event`), head speed, etc. This is the master input
#   to every downstream table builder.
# * **`environment`** — the place-bin / track-graph environment used to map a
#   decoded position to an arm.
# * **conditional posterior** (`results.nc`) — the full `(time, position)`
#   probability distribution conditioned on the non-local state, used both for
#   visualization here and for the posterior-weighted table columns.

# %%
dot = pd.read_pickle(DOT_FILE)
validate_dot_event_contract(dot)
with open(ENV_FILE, "rb") as f:
    env = pickle.load(f)
posterior = load_conditional_posterior(RESULTS_FILE)

posterior_times = np.asarray(posterior["time"].values, dtype=float)
dot_times = dot.index.to_numpy(dtype=float)
if posterior_times.shape != dot_times.shape:
    raise ValueError(
        f"posterior has {len(posterior_times)} time bins; dot has {len(dot_times)}"
    )
max_time_drift = float(np.max(np.abs(posterior_times - dot_times)))
if max_time_drift > 1e-6:
    raise ValueError(f"posterior/dot time axes drift by up to {max_time_drift:.3g} s")
if len(dot_times) < 2:
    raise ValueError("dot needs at least two time bins to establish decoder cadence")
decoder_bin_width_s = float(np.median(np.diff(dot_times)))
if not np.isfinite(decoder_bin_width_s) or decoder_bin_width_s <= 0:
    raise ValueError(f"invalid decoder bin width: {decoder_bin_width_s!r} s")

n_nonlocal_events = int(dot.loc[dot["non_local_event"] > 0, "non_local_event"].nunique())
print("dot           :", dot.shape)
print("posterior     :", dict(posterior.sizes))
print("max time drift:", f"{max_time_drift:.3g} s")
print("decoder cadence:", f"{decoder_bin_width_s:.6g} s/bin")
print("n place bins  :", env.place_bin_centers_.shape[0])
print("n_nonlocal_events (raw labels):", n_nonlocal_events)
print("\ndot state/event columns:\n",
      [c for c in dot.columns if "prob" in c or "event" in c or c == "map_estimate"])

# %% [markdown]
# ### 2a. One decoded non-local event
#
# Pick the highest-probability **raw detector label** and show the conditional
# posterior over linear position through its window, with the animal's actual
# position overlaid. This is an illustrative, deliberately extreme example;
# the canonical event table below additionally applies its speed/probability
# window and this plot alone is not evidence of spatial departure or replay.

# %%
nonlocal_summed = (
    dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
)
# peak non-local prob per event label -> choose the strongest event
peak_per_event = nonlocal_summed.groupby(dot["non_local_event"]).max()
peak_per_event = peak_per_event[peak_per_event.index > 0]
example_label = peak_per_event.idxmax()

ev_mask = dot["non_local_event"] == example_label
ev_times = dot.index[ev_mask]
pad = 0.10  # seconds of context on each side
t0, t1 = ev_times.min() - pad, ev_times.max() + pad

post_win = posterior.sel(time=slice(t0, t1)).load()
pos_win = dot.loc[t0:t1, "linear_position"]

fig, ax = plt.subplots(figsize=(9, 4))
mesh = ax.pcolormesh(
    post_win["time"].values,
    post_win["position"].values,
    post_win.values.T,
    cmap="viridis",
    shading="auto",
    rasterized=True,
)
ax.plot(pos_win.index, pos_win.values, ".", color="magenta", ms=4,
        label="animal position")
ax.axvspan(ev_times.min(), ev_times.max(), color="red", alpha=0.12,
           label="non-local event")
ax.set(xlabel="time (s)", ylabel="linear position (cm)",
       title=f"event {int(example_label)}: conditional non-local posterior")
ax.legend(loc="upper right", fontsize=8)
fig.colorbar(mesh, ax=ax, label="P(position | non-local)", pad=0.01)
fig.tight_layout()

# %% [markdown]
# ## 3. Stop table (the denominator is built first)
#
# `build_stop_table` produces **one row per stopping period** (immobility
# bout), including zero-event stops. The canonical driver builds this table
# before touching the event table so a zero-event session still contributes its
# full exposure denominator to the rate models.

# %%
stop_table = build_stop_table(dot, trials, track_graph)
print("stop_table:", stop_table.shape)
print(
    "stops with >=1 event:",
    int((stop_table["n_non_local"] > 0).sum()),
    "/",
    len(stop_table),
)
print("zero-duration stops:", int((stop_table["immobility_time"] <= 0).sum()))
stop_table.head()

# %%
# check that trials make sense
from matplotlib.patches import Patch
from continuum_swr_replay.visualization.palettes import COLORS

from continuum_swr_replay.data_loaders import WELL_PATCH_MAPPING
from continuum_swr_replay.visualization import plot_graph_as_1D

t_start = float(position_info.index[0]) + 750.0
t_stop = t_start + 100.0

# Linear position of each degree-1 node (well), mirroring the cumulative
# arithmetic in plot_graph_as_1D so markers land on the same y axis.
spacing = (
    [float(linear_edge_spacing)] * (len(linear_edge_order) - 1)
    if np.isscalar(linear_edge_spacing)
    else list(linear_edge_spacing)
)
well_linear = {}
edge_start = 0.0
for i, (u, v) in enumerate(np.asarray(linear_edge_order)):
    edge_end = edge_start + track_graph.edges[u, v]["distance"]
    if track_graph.degree(u) == 1:
        well_linear[int(u)] = edge_start
    if track_graph.degree(v) == 1:
        well_linear[int(v)] = edge_end
    edge_start = edge_end + (spacing[i] if i < len(spacing) else 0.0)

PATCH_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}  # Okabe-Ito

fig, axs = plt.subplots(4, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1]})

ax = axs[0]

# spreadsheet-style alternating grey bands, one per trial
for row, (_, trial) in enumerate(trials.iterrows()):
    ax.axvspan(
        trial["movement_start_time"], trial["end_time"],
        color="0.85" if row % 2 else "0.95", linewidth=0, zorder=0,
    )

ax.scatter(
    position_info.index, position_info["linear_position"],
    s=1, color="magenta", rasterized=True, zorder=2,
)

blend = ax.get_xaxis_transform()
for _, trial in trials.iterrows():
    from_well, to_well = int(trial["from_well"]), int(trial["to_well"])
    # departure: open square at from_well, colored by from_patch
    ax.plot(
        trial["movement_start_time"], well_linear[from_well],
        marker="s", ms=6, mfc="none",
        mec=PATCH_COLORS[WELL_PATCH_MAPPING[from_well]], mew=1.5, zorder=4,
    )
    # arrival: circle at destination_first_poke_in_time, filled iff rewarded
    dest_color = PATCH_COLORS[WELL_PATCH_MAPPING[to_well]]
    ax.plot(
        trial["destination_first_poke_in_time"], well_linear[to_well],
        marker="o", ms=7,
        mfc=dest_color if trial["is_reward"] else "none",
        mec=dest_color, mew=1.5, zorder=4,
    )
    if trial["is_patch_change"]:
        ax.plot(
            trial["movement_start_time"], 0.97, marker="v", ms=6, color="black",
            transform=blend, zorder=5,
        )

ax.set(
    xlabel="time (s)", ylabel="linear position (cm)",
    title=f"{SESSION_ID}: position and trials", xlim=(t_start, t_stop),
)
well_axis = ax.secondary_yaxis("right")
well_axis.set_yticks([well_linear[w] for w in sorted(well_linear)])
well_axis.set_yticklabels([f"well {w}" for w in sorted(well_linear)], fontsize=8)


def _marker(marker, filled, label, color="0.35"):
    """Neutral-grey proxy handle — real markers are colored by patch."""
    return Line2D(
        [], [], linestyle="none", marker=marker, ms=7,
        mfc=color if filled else "none", mec=color, mew=1.5, label=label,
    )


legend_handles = [
    Line2D([], [], linestyle="none", marker="o", ms=4, color="magenta",
           label="linear position"),
    Patch(facecolor="0.85", edgecolor="none", label="trial (alternating)"),
    _marker("s", False, "departure (from_well)"),
    _marker("o", True, "arrival, rewarded"),
    _marker("o", False, "arrival, unrewarded"),
    _marker("v", True, "patch change", color="black"),
] + [
    Patch(facecolor=PATCH_COLORS[p], edgecolor="none", label=f"patch {p}")
    for p in sorted(PATCH_COLORS)
]
ax.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
    ncol=5, fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.5,
)

reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
plot_graph_as_1D(
    track_graph, ax=ax,
    edge_order=linear_edge_order, edge_spacing=linear_edge_spacing,
    reward_well_nodes=reward_well_nodes,
    other_axis_start=t_start,
    reward_well_size=120, edge_linewidth=5,
)



axs[1].plot(
    position_info.index, position_info["head_speed"],
    color="black", rasterized=True, zorder=2,
)
axs[1].set_ylabel("Speed [cm/s]")
for _, stop in stop_table.loc[(stop_table.stop_start >= t_start) & (stop_table.stop_end <= t_stop)].iterrows():
    axs[1].axvspan(
        stop.stop_start, stop.stop_end,
        color="C3", alpha=0.2, zorder=1
    )
    axs[1].text(stop.stop_start + 0.5 * (stop.stop_end - stop.stop_start), 1.05, f"{int(stop.stop_idx)}",
                color="C3", fontsize=8, ha="center", va="top", transform=axs[1].get_xaxis_transform(), zorder=3)
    if stop.n_SWR > 0:
        axs[2].text(stop.stop_start + 0.5 * (stop.stop_end - stop.stop_start), 1.05, f"{int(stop.n_SWR)}",
                    color=COLORS["SWR"], fontsize=8, ha="center", va="top", transform=axs[2].get_xaxis_transform(), zorder=3)
    if stop.n_non_local > 0:
        axs[3].text(stop.stop_start + 0.5 * (stop.stop_end - stop.stop_start), 1.05, f"{int(stop.n_non_local)}",
                    color=COLORS["Non-Local"], fontsize=8, ha="center", va="top", transform=axs[3].get_xaxis_transform(), zorder=3)

dot.loc[t_start:t_stop, "SWR_zscore"].plot(
    ax=axs[2], linewidth=1, alpha=0.8, rasterized=True, color=COLORS["SWR"], zorder=2
)
axs[2].set_ylabel("SWR z-score")
axs[2].axhline(0, color="black", linewidth=0.5, alpha=0.5, zorder=3, linestyle="--")
axs[2].axhline(2.0, color="black", linewidth=0.5, alpha=0.5, zorder=3, linestyle="--")

dot.loc[t_start:t_stop, [ "Non-Local_Continuous_prob", "Non-Local_Fragmented_prob"]].sum(axis=1).plot(
    ax=axs[3], linewidth=1, alpha=0.8, rasterized=True, color=COLORS["Non-Local"], zorder=2
)
axs[3].axhline(0.8, color="black", linewidth=0.5, alpha=0.5, zorder=3, linestyle="--")
axs[3].set_ylabel("Non-local prob")

# %%
# check that trials make sense
from matplotlib.patches import Patch
from continuum_swr_replay.visualization.palettes import COLORS

from continuum_swr_replay.data_loaders import WELL_PATCH_MAPPING
from continuum_swr_replay.visualization import plot_graph_as_1D

t_start = float(position_info.index[0]) + 10.0
t_stop = t_start + 20.0

# Linear position of each degree-1 node (well), mirroring the cumulative
# arithmetic in plot_graph_as_1D so markers land on the same y axis.
spacing = (
    [float(linear_edge_spacing)] * (len(linear_edge_order) - 1)
    if np.isscalar(linear_edge_spacing)
    else list(linear_edge_spacing)
)
well_linear = {}
edge_start = 0.0
for i, (u, v) in enumerate(np.asarray(linear_edge_order)):
    edge_end = edge_start + track_graph.edges[u, v]["distance"]
    if track_graph.degree(u) == 1:
        well_linear[int(u)] = edge_start
    if track_graph.degree(v) == 1:
        well_linear[int(v)] = edge_end
    edge_start = edge_end + (spacing[i] if i < len(spacing) else 0.0)

PATCH_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}  # Okabe-Ito

fig, axs = plt.subplots(4, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1]})

ax = axs[0]

# spreadsheet-style alternating grey bands, one per trial
for row, (_, trial) in enumerate(trials.iterrows()):
    ax.axvspan(
        trial["movement_start_time"], trial["end_time"],
        color="0.85" if row % 2 else "0.95", linewidth=0, zorder=0,
    )

ax.scatter(
    position_info.index, position_info["linear_position"],
    s=1, color="magenta", rasterized=True, zorder=2,
)

blend = ax.get_xaxis_transform()
for _, trial in trials.iterrows():
    from_well, to_well = int(trial["from_well"]), int(trial["to_well"])
    # departure: open square at from_well, colored by from_patch
    ax.plot(
        trial["movement_start_time"], well_linear[from_well],
        marker="s", ms=6, mfc="none",
        mec=PATCH_COLORS[WELL_PATCH_MAPPING[from_well]], mew=1.5, zorder=4,
    )
    # arrival: circle at destination_first_poke_in_time, filled iff rewarded
    dest_color = PATCH_COLORS[WELL_PATCH_MAPPING[to_well]]
    ax.plot(
        trial["destination_first_poke_in_time"], well_linear[to_well],
        marker="o", ms=7,
        mfc=dest_color if trial["is_reward"] else "none",
        mec=dest_color, mew=1.5, zorder=4,
    )
    if trial["is_patch_change"]:
        ax.plot(
            trial["movement_start_time"], 0.97, marker="v", ms=6, color="black",
            transform=blend, zorder=5,
        )

ax.set(
    xlabel="time (s)", ylabel="linear position (cm)",
    title=f"{SESSION_ID}: position and trials", xlim=(t_start, t_stop),
)
well_axis = ax.secondary_yaxis("right")
well_axis.set_yticks([well_linear[w] for w in sorted(well_linear)])
well_axis.set_yticklabels([f"well {w}" for w in sorted(well_linear)], fontsize=8)


def _marker(marker, filled, label, color="0.35"):
    """Neutral-grey proxy handle — real markers are colored by patch."""
    return Line2D(
        [], [], linestyle="none", marker=marker, ms=7,
        mfc=color if filled else "none", mec=color, mew=1.5, label=label,
    )


legend_handles = [
    Line2D([], [], linestyle="none", marker="o", ms=4, color="magenta",
           label="linear position"),
    Patch(facecolor="0.85", edgecolor="none", label="trial (alternating)"),
    _marker("s", False, "departure (from_well)"),
    _marker("o", True, "arrival, rewarded"),
    _marker("o", False, "arrival, unrewarded"),
    _marker("v", True, "patch change", color="black"),
] + [
    Patch(facecolor=PATCH_COLORS[p], edgecolor="none", label=f"patch {p}")
    for p in sorted(PATCH_COLORS)
]
ax.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
    ncol=5, fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.5,
)

reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
plot_graph_as_1D(
    track_graph, ax=ax,
    edge_order=linear_edge_order, edge_spacing=linear_edge_spacing,
    reward_well_nodes=reward_well_nodes,
    other_axis_start=t_start,
    reward_well_size=120, edge_linewidth=5,
)



axs[1].plot(
    position_info.index, position_info["head_speed"],
    color="black", rasterized=True, zorder=2,
)
axs[1].axhline(4.0, color="black", linewidth=0.5, alpha=0.5, zorder=3, linestyle="--")
axs[1].set_ylabel("Speed [cm/s]")
for _, stop in stop_table.loc[(stop_table.stop_start >= t_start) & (stop_table.stop_end <= t_stop)].iterrows():
    axs[1].axvspan(
        stop.stop_start, stop.stop_end,
        color="C3", alpha=0.2, zorder=1
    )
    axs[1].text(stop.stop_start + 0.5 * (stop.stop_end - stop.stop_start), 1.05, f"{int(stop.stop_idx)}",
                color="C3", fontsize=8, ha="center", va="top", transform=axs[1].get_xaxis_transform(), zorder=3)
    if stop.n_SWR > 0:
        axs[2].text(stop.stop_start + 0.5 * (stop.stop_end - stop.stop_start), 1.05, f"{int(stop.n_SWR)}",
                    color=COLORS["SWR"], fontsize=8, ha="center", va="top", transform=axs[2].get_xaxis_transform(), zorder=3)
    if stop.n_non_local > 0:
        axs[3].text(stop.stop_start + 0.5 * (stop.stop_end - stop.stop_start), 1.05, f"{int(stop.n_non_local)}",
                    color=COLORS["Non-Local"], fontsize=8, ha="center", va="top", transform=axs[3].get_xaxis_transform(), zorder=3)

dot.loc[t_start:t_stop, "SWR_zscore"].plot(
    ax=axs[2], linewidth=1, alpha=0.8, rasterized=True, color=COLORS["SWR"], zorder=2
)
axs[2].fill_between(
    dot.loc[t_start:t_stop].index,
    dot.loc[t_start:t_stop, "SWR_event"] > 0,
    where=dot.loc[t_start:t_stop, "SWR_event"] > 0,
    color=COLORS["SWR"], alpha=0.2, zorder=1
)

axs[2].set_ylabel("SWR z-score")
axs[2].axhline(0, color="black", linewidth=0.5, alpha=0.5, zorder=3, linestyle="--")
axs[2].axhline(2.0, color="black", linewidth=0.5, alpha=0.5, zorder=3, linestyle="--")

dot.loc[t_start:t_stop, [ "Non-Local_Continuous_prob", "Non-Local_Fragmented_prob"]].sum(axis=1).plot(
    ax=axs[3], linewidth=1, alpha=0.8, rasterized=True, color=COLORS["Non-Local"], zorder=2
)
axs[3].fill_between(
    dot.loc[t_start:t_stop].index,
    dot.loc[t_start:t_stop, "non_local_event"] > 0,
    where=dot.loc[t_start:t_stop, "non_local_event"] > 0,
    color=COLORS["Non-Local"], alpha=0.2, zorder=1
)
axs[3].axhline(0.8, color="black", linewidth=0.5, alpha=0.5, zorder=3, linestyle="--")
axs[3].set_ylabel("Non-local prob")

# %%
dot.loc[t_start:t_stop, "SWR_event"].unique()

# %%
stop_table.keys()

# %%
dot.keys()

# %% [markdown]
# ## 4. Behavioral-model beliefs: value and uncertainty
#
# The walkthrough consumes the externally fitted `beta_stable_withleaf`
# Beta-Bernoulli model; it does not estimate that model here. For each reward
# well and trial, the model provides a pre-outcome belief
# $\operatorname{Beta}(\alpha,\beta)$:
#
# * nominal value `p_reward_port*` is the experimenter's scheduled $p(R)$;
# * learned value `q_port*` is the posterior mean $\alpha/(\alpha+\beta)$;
# * uncertainty `betadist_var*` is the posterior variance
#   $\alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]$.
#
# The fitted model maintains six independent leaf-specific Beta trackers.
# Reward adds evidence to the chosen well's $\alpha$; non-reward adds
# depletion-scaled evidence to its $\beta$. Between-trial forgetting pulls all
# beliefs toward $\operatorname{Beta}(1,1)$, and the fitted patch/leaf softmax
# policy maps the beliefs—together with depletion and choice-bias terms—to
# choices. Variance is therefore a **model-derived** uncertainty
# measure—not decoder uncertainty or direct confidence—and depends
# mathematically on both posterior mean and concentration. In this model,
# forgetting also couples it strongly to time since visiting a well.
#
# Row $t$ is the belief **before** trial $t$'s outcome; that outcome first updates
# row $t+1$. This makes value and variance safe pre-outcome content covariates.
# RPE and value-update columns in the same artifact are post-outcome quantities
# and are intentionally not fitted below. The artifact broadcasts a trial's
# beliefs to every stop assigned to that trial, so orientation plots first
# collapse repeated stop rows back to one row per behavioral trial.

# %%
qvals_all = load_contingency(QVALS_FILE)
if qvals_all is None:
    raise FileNotFoundError(
        "the production qvals contingency artifact is required for this walkthrough: "
        f"{QVALS_FILE}"
    )

p_reward_port_cols = [f"p_reward_port{arm}" for arm in range(6)]
q_port_cols = [f"q_port{arm}" for arm in range(6)]
beta_mu_cols = [f"betadist_mu{arm}" for arm in range(1, 7)]
beta_var_cols = [f"betadist_var{arm}" for arm in range(1, 7)]
stem_probability_cols = [f"stem_{stem}_p" for stem in range(1, 4)]
belief_trial_cols = [
    "trial_number_by_epoch",
    "arm",
    "trial_reward",
    "q_chosen",
    *stem_probability_cols,
    *p_reward_port_cols,
    *q_port_cols,
    *beta_mu_cols,
    *beta_var_cols,
]
required_qvals_cols = {"stop_id", "epoch", *belief_trial_cols}
missing_qvals_cols = sorted(required_qvals_cols - set(qvals_all.columns))
if missing_qvals_cols:
    raise ValueError(
        f"qvals_table is missing current behavioral-model columns: {missing_qvals_cols}"
    )
if qvals_all["stop_id"].duplicated().any():
    raise ValueError("qvals_table.stop_id must be globally unique")

qvals_session = qvals_all[
    qvals_all["stop_id"].isin(stop_table["stop_id"])
].copy()
del qvals_all

if qvals_session.empty:
    raise ValueError("no stops in this epoch map to qvals_table")
if not qvals_session["epoch"].astype(str).eq(EPOCH_NAME).all():
    raise ValueError("qvals epoch labels disagree with the selected epoch")
if qvals_session["stop_id"].duplicated().any():
    raise ValueError("selected qvals rows are not unique on stop_id")

trial_constancy = qvals_session.groupby("trial_number_by_epoch")[
    belief_trial_cols[1:]
].nunique(dropna=False)
if not trial_constancy.le(1).all().all():
    bad_cols = trial_constancy.columns[trial_constancy.gt(1).any()].tolist()
    raise ValueError(
        "behavioral beliefs vary across stops assigned to the same trial: "
        f"{bad_cols}"
    )

belief_trials = (
    qvals_session.sort_values("trial_number_by_epoch")
    .drop_duplicates("trial_number_by_epoch")[belief_trial_cols]
    .reset_index(drop=True)
)

chosen_arm = belief_trials["arm"].to_numpy(dtype=int)
if not np.isin(chosen_arm, np.arange(6)).all():
    raise ValueError("behavioral-model chosen arm must be a reward-well arm 0..5")

p_reward_matrix = belief_trials[p_reward_port_cols].to_numpy(dtype=float)
q_matrix = belief_trials[q_port_cols].to_numpy(dtype=float)
uncertainty_matrix = belief_trials[beta_var_cols].to_numpy(dtype=float)
beta_mu_matrix = belief_trials[beta_mu_cols].to_numpy(dtype=float)
row = np.arange(len(belief_trials))

assert np.allclose(
    belief_trials["q_chosen"].to_numpy(dtype=float),
    q_matrix[row, chosen_arm],
    rtol=0,
    atol=1e-12,
)
np.testing.assert_allclose(q_matrix, beta_mu_matrix, rtol=0, atol=1e-12)
assert np.isfinite(p_reward_matrix).all() and np.isfinite(q_matrix).all()
assert np.isfinite(uncertainty_matrix).all()
assert ((0 <= p_reward_matrix) & (p_reward_matrix <= 1)).all()
assert ((0 <= q_matrix) & (q_matrix <= 1)).all()
assert ((0 <= uncertainty_matrix) & (uncertainty_matrix <= 1 / 12 + 1e-12)).all()

stop_qvals_coverage = qvals_session["stop_id"].nunique() / len(stop_table)
print(
    "behavioral beliefs:",
    f"{len(belief_trials)} trials,",
    f"{len(qvals_session)}/{len(stop_table)} stops mapped "
    f"({stop_qvals_coverage:.1%})",
)

# %% [markdown]
# ### 4a. Belief trajectories and behavioral orientation
#
# The three panels separate the task schedule, the animal's learned expected
# reward, and uncertainty about that expectation. White squares mark the chosen
# well; orange circles mark rewarded choices. The chosen-stem diagnostic is an
# in-sample orientation check on an externally fitted model, not held-out
# validation of this epoch. The Brier/log-loss table (lower is better) checks
# immediate-reward calibration of nominal $p$ and learned $q$; it is not a
# measure of the fitted choice policy.

# %%
trial_axis = np.arange(len(belief_trials))
rewarded = belief_trials["trial_reward"].to_numpy(dtype=bool)

fig, axes = plt.subplots(
    3, 1, figsize=(12, 8), sharex=True, constrained_layout=True
)
belief_panels = (
    (p_reward_matrix.T, "nominal reward probability", "cividis", 0.0, 1.0),
    (
        q_matrix.T,
        "Beta posterior mean (learned reward probability; pre-outcome)",
        "cividis",
        0.0,
        1.0,
    ),
    (
        uncertainty_matrix.T,
        "Beta posterior variance (belief uncertainty; pre-outcome)",
        "magma",
        0.0,
        1 / 12,
    ),
)
belief_images = []
for ax, (matrix, title, cmap, vmin, vmax) in zip(
    axes, belief_panels, strict=True
):
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    belief_images.append(image)
    chosen_markers = ax.scatter(
        trial_axis,
        chosen_arm,
        marker="s",
        s=15,
        facecolors="none",
        edgecolors="white",
        linewidths=0.6,
    )
    chosen_markers.set_path_effects(
        [patheffects.Stroke(linewidth=1.8, foreground="black"), patheffects.Normal()]
    )
    ax.scatter(
        trial_axis[rewarded],
        chosen_arm[rewarded],
        marker="o",
        s=10,
        color="#E69F00",
        edgecolors="0.2",
        linewidths=0.3,
    )
    ax.set(title=title, ylabel="reward-well arm")
    ax.set_yticks(range(6))
fig.colorbar(
    belief_images[0],
    ax=axes[:2],
    label="reward probability",
    ticks=(0.0, 0.5, 1.0),
    pad=0.01,
)
variance_colorbar = fig.colorbar(
    belief_images[2],
    ax=axes[2],
    label="Beta variance [p²]",
    ticks=(0.0, 1 / 24, 1 / 12),
    pad=0.01,
)
variance_colorbar.set_ticks(
    (0.0, 1 / 24, 1 / 12), labels=("0", "1/24", "1/12")
)
belief_legend_handles = (
    Line2D(
        [],
        [],
        marker="s",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.8,
        label="chosen well",
    ),
    Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="#E69F00",
        markeredgecolor="0.2",
        markeredgewidth=0.5,
        label="rewarded choice",
    ),
)
axes[0].legend(
    handles=belief_legend_handles,
    loc="upper left",
    bbox_to_anchor=(1.08, 1),
    fontsize=8,
)
axes[-1].set_xlabel("behavioral trial")

stem_probabilities = belief_trials[stem_probability_cols].to_numpy(dtype=float)
valid_stem_probability = (
    np.isfinite(stem_probabilities).all(axis=1)
    & ((0 <= stem_probabilities) & (stem_probabilities <= 1)).all(axis=1)
    & np.isclose(stem_probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-10)
)
if not valid_stem_probability.any():
    raise ValueError("no trials carry a valid three-stem policy distribution")
chosen_stem = chosen_arm // 2
stem_top1 = np.mean(
    np.argmax(stem_probabilities[valid_stem_probability], axis=1)
    == chosen_stem[valid_stem_probability]
)
mean_chosen_stem_probability = np.mean(
    stem_probabilities[
        row[valid_stem_probability], chosen_stem[valid_stem_probability]
    ]
)

q_midpoint_ecdf = (
    belief_trials["q_chosen"].rank(method="average") - 0.5
) / len(belief_trials)
belief_trials["q_chosen_quartile"] = pd.cut(
    q_midpoint_ecdf,
    bins=np.linspace(0, 1, 5),
    labels=["Q1", "Q2", "Q3", "Q4"],
    include_lowest=True,
)
reward_calibration = belief_trials.groupby(
    "q_chosen_quartile", observed=True
).agg(
    mean_predicted_reward=("q_chosen", "mean"),
    observed_reward_rate=("trial_reward", "mean"),
    n_trials=("trial_reward", "size"),
)
outcome = belief_trials["trial_reward"].to_numpy(dtype=float)
p_chosen = p_reward_matrix[row, chosen_arm]
q_chosen = q_matrix[row, chosen_arm]
calibration_metrics = []
for source, predicted in (("nominal p", p_chosen), ("Beta mean q", q_chosen)):
    clipped = np.clip(predicted, 1e-12, 1 - 1e-12)
    calibration_metrics.append(
        {
            "source": source,
            "Brier_score": np.mean((predicted - outcome) ** 2),
            "log_loss": -np.mean(
                outcome * np.log(clipped) + (1 - outcome) * np.log(1 - clipped)
            ),
        }
    )
print(
    f"chosen-stem top-1={stem_top1:.1%}; "
    f"mean P(chosen stem)={mean_chosen_stem_probability:.3f}; "
    f"valid policy trials={valid_stem_probability.sum()}/{len(belief_trials)}"
)
display(reward_calibration)  # noqa: F821
display(pd.DataFrame(calibration_metrics).set_index("source").round(3))  # noqa: F821

# %%
# Beliefs on the wall clock, aligned to the behavior panel.
from matplotlib.patches import Patch

t_start = float(position_info.index[0])
t_stop = t_start + 750.0

from continuum_swr_replay.analysis.replay.covariates import (
    attach_arm_uncertainty,
    attach_contingency_value_block,
)
from continuum_swr_replay.visualization.palettes import patch_grouped_arm_palette

WELL_COLORS = {w: c for w, c in patch_grouped_arm_palette().items() if w < 6}
TRIAL_SCALARS = (
    ("switch_advantage_mu_preoutcome_sess_z", "switch advantage ΔV", "-", "#0072B2"),
    ("beta_reward_entropy_z", "belief entropy", "-", "#009E73"),
    ("rpe_z", "RPE (post-outcome)", "--", "#D55E00"),
    ("global_mu_update_abs_postoutcome_sess_z", "|value update| (post-outcome)",
     "--", "#CC79A7"),
)

# Every belief value below is produced by the SAME builders that populate the
# fitted event-arm design matrix (tables_arm_patch.build_event_arm_table), keyed on stop_id --
# never on trial number, whose origin differs by animal.
stops = (
    stop_table[["stop_id", "stop_start", "stop_end"]]
    .sort_values("stop_start").reset_index(drop=True)
)
n_stops = len(stops)
arm_long = np.tile(np.arange(6), n_stops)
stop_id_long = np.repeat(stops["stop_id"].to_numpy(), 6)
p_reward_arm, q_arm = attach_contingency_value_block(
    arm_long, stop_id_long, qvals_session
)
arm_uncertainty = attach_arm_uncertainty(arm_long, stop_id_long, qvals_session)
p_reward_stop = p_reward_arm.reshape(n_stops, 6)
q_stop = q_arm.reshape(n_stops, 6)
uncertainty_stop = arm_uncertainty.reshape(n_stops, 6)

# Same one-to-one stop join build_trial_leaf_table performs.
scalar_cols = [col for col, *_ in TRIAL_SCALARS]
stop_scalars = stops.merge(
    qvals_session[["stop_id", *scalar_cols]],
    on="stop_id", how="left", validate="one_to_one",
).sort_values("stop_start").reset_index(drop=True)
chosen_arm_stop = stops.merge(
    qvals_session[["stop_id", "arm", "trial_reward", "trial_number_by_epoch"]],
    on="stop_id", how="left", validate="one_to_one",
)

# Cell edges: each stop's belief is held from its start to the next stop's
# start, so pcolormesh draws exact steps with no interpolation.
step_x = np.append(
    stops["stop_start"].to_numpy(dtype=float), float(stops["stop_end"].iloc[-1])
)
if not np.all(np.diff(step_x) > 0):
    raise ValueError("stop_start must be strictly increasing to step-hold belief")
well_edges = np.arange(7)
stop_mid = 0.5 * (step_x[:-1] + step_x[1:])


def _heat(ax, values, cmap, vmin, vmax):
    """Belief as wells x time; row order matches the behavior panel's well axis."""
    return ax.pcolormesh(step_x, well_edges, values.T, cmap=cmap, vmin=vmin,
                         vmax=vmax, shading="flat", rasterized=True)


def _well_axis(ax):
    ax.set_yticks(np.arange(6) + 0.5, labels=[f"well {w}" for w in range(6)],
                  fontsize=7)
    for tick, well in zip(ax.get_yticklabels(), range(6), strict=True):
        tick.set_color(WELL_COLORS[well])


fig, axes = plt.subplots(
    5, 1, figsize=(15, 10), sharex=True,
    height_ratios=[3.0, 0.7, 1.5, 1.5, 1.6], constrained_layout=True,
)

# ---- panel 1: position + trials ------------------------------------------
ax = axes[0]
for row_i, (_, trial) in enumerate(trials.iterrows()):
    ax.axvspan(trial["movement_start_time"], trial["end_time"],
               color="0.85" if row_i % 2 else "0.95", linewidth=0, zorder=0)
ax.scatter(position_info.index, position_info["linear_position"], s=1,
           color="magenta", rasterized=True, zorder=2)
blend = ax.get_xaxis_transform()
for _, trial in trials.iterrows():
    from_well, to_well = int(trial["from_well"]), int(trial["to_well"])
    ax.plot(trial["movement_start_time"], well_linear[from_well], marker="s", ms=6,
            mfc="none", mec=WELL_COLORS[from_well], mew=1.5, zorder=4)
    ax.plot(trial["destination_first_poke_in_time"], well_linear[to_well], marker="o", ms=7,
            mfc=WELL_COLORS[to_well] if trial["is_reward"] else "none",
            mec=WELL_COLORS[to_well], mew=1.5, zorder=4)
    if trial["is_patch_change"]:
        ax.plot(trial["movement_start_time"], 0.97, marker="v", ms=6, color="black",
                transform=blend, zorder=5)
ax.set(ylabel="linear position (cm)",
       title=f"{SESSION_ID}: behavior and Beta-model beliefs")
well_axis = ax.secondary_yaxis("right")
well_axis.set_yticks([well_linear[w] for w in sorted(well_linear)])
well_axis.set_yticklabels([f"well {w}" for w in sorted(well_linear)], fontsize=8)
for tick, well in zip(well_axis.get_yticklabels(), sorted(well_linear), strict=True):
    tick.set_color(WELL_COLORS[well])
ax.legend(handles=[
    Line2D([], [], ls="none", marker="o", ms=4, color="magenta", label="linear position"),
    Patch(facecolor="0.85", edgecolor="none", label="trial (alternating)"),
    Line2D([], [], ls="none", marker="s", ms=7, mfc="none", mec="0.35", mew=1.5, label="departure"),
    Line2D([], [], ls="none", marker="o", ms=7, mfc="0.35", mec="0.35", mew=1.5, label="arrival, rewarded"),
    Line2D([], [], ls="none", marker="o", ms=7, mfc="none", mec="0.35", mew=1.5, label="arrival, unrewarded"),
    Line2D([], [], ls="none", marker="v", ms=6, color="black", label="patch change"),
], loc="upper left", bbox_to_anchor=(1.09, 1.0), fontsize=8, frameon=False)

# ---- panel 2: programmed schedule (thin reference strip) ------------------
mesh_p = _heat(axes[1], p_reward_stop, "cividis", 0.0, 1.0)
_well_axis(axes[1])
axes[1].set_ylabel("p_reward_arm\n(programmed)", fontsize=8)

# ---- panel 3: learned value ----------------------------------------------
ax = axes[2]
mesh_q = _heat(ax, q_stop, "cividis", 0.0, 1.0)
_well_axis(ax)
ax.set_ylabel("q_arm\n(Beta mean, pre-outcome)", fontsize=8)
# Chosen well, ONE marker per behavioral trial (a trial spans several stops),
# ringed so it reads against either end of the colormap.
choice_marks = (
    chosen_arm_stop.assign(stop_mid=stop_mid)
    .dropna(subset=["arm", "trial_number_by_epoch"])
    .drop_duplicates("trial_number_by_epoch")
)
chosen = choice_marks["arm"].to_numpy(dtype=float) + 0.5
marks = ax.scatter(choice_marks["stop_mid"], chosen, marker="s", s=14,
                   facecolors="none", edgecolors="white", linewidths=0.7, zorder=3)
marks.set_path_effects([patheffects.Stroke(linewidth=1.8, foreground="black"),
                        patheffects.Normal()])
rewarded = choice_marks["trial_reward"].to_numpy() == 1
ax.scatter(choice_marks["stop_mid"][rewarded], chosen[rewarded], marker="o", s=8,
           color="#E69F00", edgecolors="0.2", linewidths=0.3, zorder=4)

# ---- panel 4: belief uncertainty -----------------------------------------
mesh_u = _heat(axes[3], uncertainty_stop, "magma", 0.0, 1 / 12)
_well_axis(axes[3])
axes[3].set_ylabel("arm_uncertainty\n(Beta variance)", fontsize=8)

# ---- panel 5: trial-grain scalars ----------------------------------------
ax = axes[4]
for col, label, ls, color in TRIAL_SCALARS:
    values = stop_scalars[col].to_numpy(dtype=float)
    ax.step(step_x, np.append(values, values[-1]), where="post",
            color=color, ls=ls, lw=1.4, label=label)
ax.axhline(0.0, color="0.6", lw=0.6, zorder=0)
ax.set(ylabel="pooled-global SD", xlabel="time (s)", xlim=(t_start, t_stop))
ax.legend(loc="upper left", bbox_to_anchor=(1.09, 1.0), fontsize=8, frameon=False,
          title="solid = pre-outcome\ndashed = post-outcome", title_fontsize=8)

fig.colorbar(mesh_q, ax=[axes[1], axes[2]], label="reward probability",
             ticks=(0, 0.5, 1), pad=0.005, aspect=18)
cbar_u = fig.colorbar(mesh_u, ax=axes[3], label="variance [p²]",
                      ticks=(0, 1 / 24, 1 / 12), pad=0.005, aspect=10)
cbar_u.set_ticks((0, 1 / 24, 1 / 12), labels=("0", "1/24", "1/12"))
axes[2].legend(handles=[
    Line2D([], [], ls="none", marker="s", ms=6, mfc="none", mec="white", mew=1.0,
           label="chosen well"),
    Line2D([], [], ls="none", marker="o", ms=5, mfc="#E69F00", mec="0.2",
           label="rewarded"),
], loc="upper left", bbox_to_anchor=(1.09, 1.0), fontsize=8, frameon=False)

# ---- the production path must reproduce the section-4a belief table -------
mapped = stops["stop_id"].isin(qvals_session["stop_id"]).to_numpy()
stop_trial = stops.merge(
    qvals_session[["stop_id", "trial_number_by_epoch"]], on="stop_id", how="left"
)["trial_number_by_epoch"].to_numpy()
trial_row = pd.Index(belief_trials["trial_number_by_epoch"]).get_indexer(
    stop_trial[mapped]
)
np.testing.assert_allclose(q_stop[mapped], q_matrix[trial_row], rtol=0, atol=1e-12)
print(f"stops={n_stops}, with belief={int(mapped.sum())} ({mapped.mean():.1%}); "
      "production q_arm matches the section-4a q_port matrix")


# %%
# Beliefs on the wall clock, aligned to the behavior panel.
from matplotlib.patches import Patch

from continuum_swr_replay.analysis.replay.covariates import (
    attach_arm_uncertainty,
    attach_contingency_value_block,
)
from continuum_swr_replay.visualization.palettes import patch_grouped_arm_palette

t_start = float(position_info.index[0])
t_stop = float(position_info.index[-1])

WELL_COLORS = {w: c for w, c in patch_grouped_arm_palette().items() if w < 6}
TRIAL_SCALARS = (
    ("switch_advantage_mu_preoutcome_sess_z", "switch advantage ΔV", "-", "#0072B2"),
    ("beta_reward_entropy_z", "belief entropy", "-", "#009E73"),
    ("rpe_z", "RPE (post-outcome)", "--", "#D55E00"),
    ("global_mu_update_abs_postoutcome_sess_z", "|value update| (post-outcome)",
     "--", "#CC79A7"),
)

# Every belief value below is produced by the SAME builders that populate the
# fitted event-arm design matrix (tables_arm_patch.build_event_arm_table), keyed on stop_id --
# never on trial number, whose origin differs by animal.
stops = (
    stop_table[["stop_id", "stop_start", "stop_end"]]
    .sort_values("stop_start").reset_index(drop=True)
)
n_stops = len(stops)
arm_long = np.tile(np.arange(6), n_stops)
stop_id_long = np.repeat(stops["stop_id"].to_numpy(), 6)
p_reward_arm, q_arm = attach_contingency_value_block(
    arm_long, stop_id_long, qvals_session
)
arm_uncertainty = attach_arm_uncertainty(arm_long, stop_id_long, qvals_session)
p_reward_stop = p_reward_arm.reshape(n_stops, 6)
q_stop = q_arm.reshape(n_stops, 6)
uncertainty_stop = arm_uncertainty.reshape(n_stops, 6)

# Same one-to-one stop join build_trial_leaf_table performs.
scalar_cols = [col for col, *_ in TRIAL_SCALARS]
stop_scalars = stops.merge(
    qvals_session[["stop_id", *scalar_cols]],
    on="stop_id", how="left", validate="one_to_one",
).sort_values("stop_start").reset_index(drop=True)
chosen_arm_stop = stops.merge(
    qvals_session[["stop_id", "arm", "trial_reward", "trial_number_by_epoch"]],
    on="stop_id", how="left", validate="one_to_one",
)

# Cell edges: each stop's belief is held from its start to the next stop's
# start, so pcolormesh draws exact steps with no interpolation.
step_x = np.append(
    stops["stop_start"].to_numpy(dtype=float), float(stops["stop_end"].iloc[-1])
)
if not np.all(np.diff(step_x) > 0):
    raise ValueError("stop_start must be strictly increasing to step-hold belief")
well_edges = np.arange(7)
stop_mid = 0.5 * (step_x[:-1] + step_x[1:])


def _heat(ax, values, cmap, vmin, vmax):
    """Belief as wells x time; row order matches the behavior panel's well axis."""
    return ax.pcolormesh(step_x, well_edges, values.T, cmap=cmap, vmin=vmin,
                         vmax=vmax, shading="flat", rasterized=True)


def _well_axis(ax):
    ax.set_yticks(np.arange(6) + 0.5, labels=[f"well {w}" for w in range(6)],
                  fontsize=7)
    for tick, well in zip(ax.get_yticklabels(), range(6), strict=True):
        tick.set_color(WELL_COLORS[well])


fig, axes = plt.subplots(
    5, 1, figsize=(15, 10), sharex=True,
    height_ratios=[3.0, 0.7, 1.5, 1.5, 1.6], constrained_layout=True,
)

# ---- panel 1: position + trials ------------------------------------------
ax = axes[0]
for row_i, (_, trial) in enumerate(trials.iterrows()):
    ax.axvspan(trial["movement_start_time"], trial["end_time"],
               color="0.85" if row_i % 2 else "0.95", linewidth=0, zorder=0)
ax.scatter(position_info.index, position_info["linear_position"], s=1,
           color="magenta", rasterized=True, zorder=2)
blend = ax.get_xaxis_transform()
for _, trial in trials.iterrows():
    from_well, to_well = int(trial["from_well"]), int(trial["to_well"])
    ax.plot(trial["movement_start_time"], well_linear[from_well], marker="s", ms=6,
            mfc="none", mec=WELL_COLORS[from_well], mew=1.5, zorder=4)
    ax.plot(trial["destination_first_poke_in_time"], well_linear[to_well], marker="o", ms=7,
            mfc=WELL_COLORS[to_well] if trial["is_reward"] else "none",
            mec=WELL_COLORS[to_well], mew=1.5, zorder=4)
    if trial["is_patch_change"]:
        ax.plot(trial["movement_start_time"], 0.97, marker="v", ms=6, color="black",
                transform=blend, zorder=5)
ax.set(ylabel="linear position (cm)",
       title=f"{SESSION_ID}: behavior and Beta-model beliefs")
well_axis = ax.secondary_yaxis("right")
well_axis.set_yticks([well_linear[w] for w in sorted(well_linear)])
well_axis.set_yticklabels([f"well {w}" for w in sorted(well_linear)], fontsize=8)
for tick, well in zip(well_axis.get_yticklabels(), sorted(well_linear), strict=True):
    tick.set_color(WELL_COLORS[well])
ax.legend(handles=[
    Line2D([], [], ls="none", marker="o", ms=4, color="magenta", label="linear position"),
    Patch(facecolor="0.85", edgecolor="none", label="trial (alternating)"),
    Line2D([], [], ls="none", marker="s", ms=7, mfc="none", mec="0.35", mew=1.5, label="departure"),
    Line2D([], [], ls="none", marker="o", ms=7, mfc="0.35", mec="0.35", mew=1.5, label="arrival, rewarded"),
    Line2D([], [], ls="none", marker="o", ms=7, mfc="none", mec="0.35", mew=1.5, label="arrival, unrewarded"),
    Line2D([], [], ls="none", marker="v", ms=6, color="black", label="patch change"),
], loc="upper left", bbox_to_anchor=(1.09, 1.0), fontsize=8, frameon=False)

# ---- panel 2: programmed schedule (thin reference strip) ------------------
mesh_p = _heat(axes[1], p_reward_stop, "cividis", 0.0, 1.0)
_well_axis(axes[1])
axes[1].set_ylabel("p_reward_arm\n(programmed)", fontsize=8)

# ---- panel 3: learned value ----------------------------------------------
ax = axes[2]
mesh_q = _heat(ax, q_stop, "cividis", 0.0, 1.0)
_well_axis(ax)
ax.set_ylabel("q_arm\n(Beta mean, pre-outcome)", fontsize=8)
# Chosen well, ONE marker per behavioral trial (a trial spans several stops),
# ringed so it reads against either end of the colormap.
choice_marks = (
    chosen_arm_stop.assign(stop_mid=stop_mid)
    .dropna(subset=["arm", "trial_number_by_epoch"])
    .drop_duplicates("trial_number_by_epoch")
)
chosen = choice_marks["arm"].to_numpy(dtype=float) + 0.5
marks = ax.scatter(choice_marks["stop_mid"], chosen, marker="s", s=14,
                   facecolors="none", edgecolors="white", linewidths=0.7, zorder=3)
marks.set_path_effects([patheffects.Stroke(linewidth=1.8, foreground="black"),
                        patheffects.Normal()])
rewarded = choice_marks["trial_reward"].to_numpy() == 1
ax.scatter(choice_marks["stop_mid"][rewarded], chosen[rewarded], marker="o", s=8,
           color="#E69F00", edgecolors="0.2", linewidths=0.3, zorder=4)

# ---- panel 4: belief uncertainty -----------------------------------------
mesh_u = _heat(axes[3], uncertainty_stop, "magma", 0.0, 1 / 12)
_well_axis(axes[3])
axes[3].set_ylabel("arm_uncertainty\n(Beta variance)", fontsize=8)

# ---- panel 5: trial-grain scalars ----------------------------------------
ax = axes[4]
for col, label, ls, color in TRIAL_SCALARS:
    values = stop_scalars[col].to_numpy(dtype=float)
    ax.step(step_x, np.append(values, values[-1]), where="post",
            color=color, ls=ls, lw=1.4, label=label)
ax.axhline(0.0, color="0.6", lw=0.6, zorder=0)
ax.set(ylabel="pooled-global SD", xlabel="time (s)", xlim=(t_start, t_stop))
ax.legend(loc="upper left", bbox_to_anchor=(1.09, 1.0), fontsize=8, frameon=False,
          title="solid = pre-outcome\ndashed = post-outcome", title_fontsize=8)

fig.colorbar(mesh_q, ax=[axes[1], axes[2]], label="reward probability",
             ticks=(0, 0.5, 1), pad=0.005, aspect=18)
cbar_u = fig.colorbar(mesh_u, ax=axes[3], label="variance [p²]",
                      ticks=(0, 1 / 24, 1 / 12), pad=0.005, aspect=10)
cbar_u.set_ticks((0, 1 / 24, 1 / 12), labels=("0", "1/24", "1/12"))
axes[2].legend(handles=[
    Line2D([], [], ls="none", marker="s", ms=6, mfc="none", mec="white", mew=1.0,
           label="chosen well"),
    Line2D([], [], ls="none", marker="o", ms=5, mfc="#E69F00", mec="0.2",
           label="rewarded"),
], loc="upper left", bbox_to_anchor=(1.09, 1.0), fontsize=8, frameon=False)

# ---- the production path must reproduce the section-4a belief table -------
mapped = stops["stop_id"].isin(qvals_session["stop_id"]).to_numpy()
stop_trial = stops.merge(
    qvals_session[["stop_id", "trial_number_by_epoch"]], on="stop_id", how="left"
)["trial_number_by_epoch"].to_numpy()
trial_row = pd.Index(belief_trials["trial_number_by_epoch"]).get_indexer(
    stop_trial[mapped]
)
np.testing.assert_allclose(q_stop[mapped], q_matrix[trial_row], rtol=0, atol=1e-12)
print(f"stops={n_stops}, with belief={int(mapped.sum())} ({mapped.mean():.1%}); "
      "production q_arm matches the section-4a q_port matrix")


# %% [markdown]
# ## 5. Event classification table
#
# `build_event_table` produces **one row per canonical non-local event**. The
# legacy `event_*` timing columns describe the filtered survivor window used by
# the content tables; `label_event_*`, speed summaries, peak probability, and
# SWR overlap classification describe the full detector-label window. Passing
# the conditional posterior also populates the posterior-weighted robustness
# columns; the primary regressions below still use hard MAP counts.

# %%
event_table = build_event_table(
    dot, env, trials, track_graph, conditional_posterior=posterior
)
stop_table = add_nonlocal_swr_split(stop_table, event_table)

print("event_table:", event_table.shape)
print("\noverlap_class counts:\n", event_table["overlap_class"].value_counts())
event_table[
    [
        "event_id",
        "event_duration",
        "label_event_duration",
        "mean_speed",
        "peak_nonlocal_prob",
        "overlap_class",
        "is_swr",
    ]
].head()

# %% [markdown]
# ### 5a. Event summary for this epoch

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

axes[0].hist(event_table["event_duration"] * 1e3, bins=40, color="C0")
axes[0].set(xlabel="event duration (ms)", ylabel="# events",
            title="filtered event durations")

axes[1].hist(event_table["peak_nonlocal_prob"], bins=40, color="C1")
axes[1].set(xlabel="peak non-local prob", title="peak non-local probability")

vc = event_table["overlap_class"].value_counts()
axes[2].bar(vc.index.astype(str), vc.values, color="C2")
axes[2].set(ylabel="# events", title="SWR overlap class")
axes[2].tick_params(axis="x", rotation=20)
fig.tight_layout()

# %% [markdown]
# ## 6. Long-format design matrices (arm and patch)
#
# `build_event_arm_table` expands the event table to **one row per
# (event, arm)** — nine arms per event — and computes the full per-event
# predictor set used by M0–M7:
#
# * `arm_count_hard` — response: count of filtered MAP decoder bins assigned to
#   this arm (the cadence is printed above; junction bins are excluded)
# * `is_current`, `is_prev`, `is_last_rewarded_elsewhere` — spatial context
# * `reward_rate_arm_k10`, `visit_recency_demeaned` — value / memory
# * `p_reward_arm`, `q_arm`, `arm_uncertainty` — nominal value, learned
#   Beta value, and Beta belief variance from `qvals_table`
# * `swr_x_*` — SWR-overlap interactions (M6/M7)
#
# The sequence table and retrospective metrics are also materialized by the
# current prep driver, even though the four base families below do not consume
# those retrospective columns directly. `build_event_patch_table` then
# collapses arms→patches (3 categories). Finally, `add_arm_fixed_effects` writes
# `arm_fe_0…arm_fe_7` at fit time (arm 8 is the reference), as the production
# arm-content runner does.

# %%
event_arm_table = build_event_arm_table(
    event_table,
    dot,
    env,
    trials,
    track_graph,
    conditional_posterior=posterior,
    contingency=qvals_session,
)

# Production computes these z-scores once after concatenating every session.
# Apply that frozen reference here; fitting a new mean/SD on this epoch would
# change the coefficient unit and make it incomparable with the population run.
arm_z_reference = load_reference(ARM_Z_REFERENCE_FILE)
event_arm_table, used_arm_z_reference = attach_standardized(
    event_arm_table,
    EVENT_ARM_STANDARDIZED_COLS,
    reference=arm_z_reference,
)

event_sequence_table = build_event_sequence_table(dot, env, posterior)
retrospective = compute_retrospective_metrics(
    event_table, event_arm_table, event_sequence_table
)
if not retrospective.empty:
    session_keys = [
        col for col in ("animal", "day", "epoch") if col in event_table.columns
    ]
    event_table = event_table.merge(
        retrospective,
        on=[*session_keys, "event_id"],
        how="left",
        validate="one_to_one",
    )

event_patch_table = build_event_patch_table(event_table, event_arm_table)
add_arm_fixed_effects(event_arm_table)  # in-place: adds arm_fe_0 … arm_fe_7

print("event_arm_table  :", event_arm_table.shape,
      f"({event_arm_table['event_id'].nunique()} events × 9 arms)")
print("event_patch_table:", event_patch_table.shape,
      f"({event_patch_table['event_id'].nunique()} events × 3 patches)")
print("event_sequence   :", event_sequence_table.shape)
print("\none event (9 arms):")
cols = [
    "event_id",
    "arm",
    "arm_count_hard",
    "is_current",
    "is_prev",
    "reward_rate_arm_k10",
    "visit_recency_demeaned",
    "p_reward_arm",
    "q_arm",
    "arm_uncertainty_z",
    "is_last_rewarded_elsewhere",
    "swr_x_current",
]
event_arm_table.loc[
    event_arm_table["event_id"] == event_arm_table["event_id"].iloc[0], cols
]

# %% [markdown]
# ### 6a. Cross-table, belief-join, and response invariants
#
# These are the high-value fail-loud checks from the canonical workflow. Counts
# and identities must match exactly; normalized fractions/posteriors allow only
# floating-point roundoff.

# %%
n_events = len(event_table)
assert len(event_arm_table) == 9 * n_events
assert len(event_patch_table) == 3 * n_events
assert (event_arm_table.groupby("event_id").size() == 9).all()
assert (event_arm_table.groupby("event_id")["arm"].nunique() == 9).all()
assert (event_patch_table.groupby("event_id").size() == 3).all()
assert (event_patch_table.groupby("event_id")["patch"].nunique() == 3).all()
assert int(stop_table["n_events"].sum()) == n_events
assert int(stop_table["n_non_local"].sum()) == n_events
assert (
    stop_table["n_non_local_SWR"] + stop_table["n_non_local_no_SWR"]
).equals(stop_table["n_non_local"])
assert int(stop_table["n_non_local_SWR"].sum()) == int(
    event_table["is_swr"].fillna(False).sum()
)
for response_col in PER_STOP_RATE_OUTCOMES.values():
    response = stop_table[response_col].to_numpy(dtype=float)
    assert np.isfinite(response).all()
    assert (response >= 0).all()
    assert np.equal(response, np.floor(response)).all()

assert event_table["stop_id"].notna().all()
event_stop_ids = set(event_table["stop_id"])
assert event_stop_ids <= set(stop_table["stop_id"])
assert set(event_sequence_table["event_id"]) == set(event_table["event_id"])
assert len(event_sequence_table) == int(event_table["n_nonlocal_bins"].sum())

hard_counts = event_arm_table["arm_count_hard"].to_numpy(dtype=float)
assert np.isfinite(hard_counts).all()
assert (hard_counts >= 0).all()
assert np.equal(hard_counts, np.floor(hard_counts)).all()

arm_totals = event_arm_table.groupby("event_id")["arm_count_hard"].sum().sort_index()
patch_totals = (
    event_patch_table.groupby("event_id")["patch_count_hard"].sum().sort_index()
)
pd.testing.assert_series_equal(arm_totals, patch_totals, check_names=False)
expected_real_arm_bins = (
    event_table.set_index("event_id")["n_nonlocal_bins"]
    - event_table.set_index("event_id")["n_junction_bins"]
).sort_index()
pd.testing.assert_series_equal(
    arm_totals.astype(float),
    expected_real_arm_bins.astype(float),
    check_names=False,
)

arm_fraction_sums = event_arm_table.groupby("event_id")["arm_fraction"].sum(
    min_count=1
)
has_real_arm_bins = expected_real_arm_bins > 0
assert np.allclose(
    arm_fraction_sums.loc[has_real_arm_bins], 1.0, rtol=0, atol=1e-10
)
assert arm_fraction_sums.loc[~has_real_arm_bins].isna().all()
max_arm_fraction_error = float(
    np.max(
        np.abs(arm_fraction_sums.loc[has_real_arm_bins] - 1).to_numpy(),
        initial=0.0,
    )
)

posterior_cols = [f"arm_posterior_{arm}" for arm in range(9)]
sequence_mass = event_sequence_table[posterior_cols].sum(axis=1).to_numpy()
assert np.isfinite(sequence_mass).all()
assert np.allclose(sequence_mass, 1.0, rtol=0, atol=1e-6)

belief_cols = ("p_reward_arm", "q_arm", "arm_uncertainty")
belief_present = event_arm_table.loc[:, belief_cols].notna()
assert belief_present.eq(belief_present.iloc[:, 0], axis=0).all().all()

belief_rows_per_event = event_arm_table.groupby("event_id")["q_arm"].count()
assert belief_rows_per_event.isin([0, 9]).all()
mapped_belief_events = belief_rows_per_event[belief_rows_per_event == 9].index
expected_mapped_events = event_table.set_index("event_id")["stop_id"].isin(
    qvals_session["stop_id"]
)
pd.testing.assert_series_equal(
    belief_rows_per_event.eq(9).sort_index(),
    expected_mapped_events.sort_index(),
    check_names=False,
)

mapped_belief_rows = event_arm_table["event_id"].isin(mapped_belief_events)
leaf_belief_rows = mapped_belief_rows & event_arm_table["arm"].lt(6)
trunk_belief_rows = mapped_belief_rows & event_arm_table["arm"].ge(6)
for col in belief_cols:
    assert np.allclose(event_arm_table.loc[trunk_belief_rows, col], 0.0)
assert event_arm_table.loc[leaf_belief_rows, "p_reward_arm"].between(0, 1).all()
assert event_arm_table.loc[leaf_belief_rows, "q_arm"].between(0, 1).all()
assert event_arm_table.loc[leaf_belief_rows, "arm_uncertainty"].between(
    0, 1 / 12 + 1e-12
).all()

qvals_by_stop = qvals_session.set_index("stop_id")
for arm in range(6):
    arm_rows = leaf_belief_rows & event_arm_table["arm"].eq(arm)
    arm_stop_ids = event_arm_table.loc[arm_rows, "stop_id"]
    for actual_col, source_col in (
        ("p_reward_arm", f"p_reward_port{arm}"),
        ("q_arm", f"q_port{arm}"),
        ("arm_uncertainty", f"betadist_var{arm + 1}"),
    ):
        expected = arm_stop_ids.map(qvals_by_stop[source_col]).to_numpy(dtype=float)
        actual = event_arm_table.loc[arm_rows, actual_col].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

assert set(used_arm_z_reference) == set(EVENT_ARM_STANDARDIZED_COLS)
for source_col, reference in used_arm_z_reference.items():
    expected_z = (
        event_arm_table[source_col].to_numpy(dtype=float) - reference["mean"]
    ) / reference["std"]
    np.testing.assert_allclose(
        event_arm_table[f"{source_col}_z"].to_numpy(dtype=float),
        expected_z,
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )

print(
    "all invariants passed:",
    f"{n_events} events, {len(stop_table)} stops,",
    f"{len(mapped_belief_events)} belief-mapped events,",
    f"max arm-fraction error={max_arm_fraction_error:.2e}",
)

# %% [markdown]
# ### 6b. Time-aligned behavioral and fitted-covariate audit
#
# The trial-level trajectories in Section 4 show the complete behavioral model.
# Here we inspect the subset of those pre-outcome beliefs that enters the replay
# design matrix, together with the spatial and reward-history covariates used by
# the content/rate models. Arm-grain colors occupy replay-event spans: blank
# time there means that no replay event is represented, not that the animal
# lacked a belief. The two bottom rate rows instead occupy complete immobility-
# stop intervals because those predictors are defined once per stop.
# The figure-level key distinguishes observed zero, missing input, and
# structural trunk N/A. The main comparison is ordered as the causal/readout
# chain: programmed reward probability → learned Beta mean → Beta variance,
# followed by empirical reward history and recency.
#
# A narrow window is deliberate. These are row-level audit panels rather than a
# session summary, and a full-epoch rendering would obscure individual events.
# Decoder/HMM panels are omitted because they are not needed for this check.

# %%
belief_event_times = event_table.loc[
    event_table["event_id"].isin(mapped_belief_events), "event_midpoint_time"
].sort_values()
if belief_event_times.empty:
    raise ValueError("no belief-mapped replay event is available for the audit figure")

audit_center = float(belief_event_times.iloc[len(belief_event_times) // 2])
audit_half_width_s = 10.0
audit_window = (
    max(float(posterior.time.values[0]), audit_center - audit_half_width_s),
    min(float(posterior.time.values[-1]), audit_center + audit_half_width_s),
)
audit_panels = (
    "trial_rail",
    "is_current",
    "is_prev",
    *BEHAVIORAL_MODEL_PANELS,
    "reward_rate_arm_k10",
    "visit_recency_demeaned",
    "reward_lag_kernel",
    "reward_rate_trial_k10",
)

fig, mismatch_table = plot_session_tables(
    data={"trials": trials, "track_graph": track_graph, "environment": env},
    detector=None,
    results=posterior,
    event_table=event_table,
    event_arm_table=event_arm_table,
    event_sequence_table=event_sequence_table,
    stop_table=stop_table,
    event_patch_table=event_patch_table,
    time_window=audit_window,
    panels=audit_panels,
    figsize=(13, 15),
    time_reference=audit_center,
    xlabel="Time from selected event [s]",
    title=(
        f"{SESSION_ID}: production behavioral covariates around the "
        "median belief-mapped event"
    ),
    panel_group_labels={
        "trial_rail": "Behavioral context (production trial timing)",
        "p_reward_arm": "Event × arm: programmed value → learned value → uncertainty",
        "reward_rate_arm_k10": "Event × arm: experienced reward history",
        "reward_lag_kernel": "Stop-level rate-model predictors",
    },
)
assert mismatch_table["match"].all(), mismatch_table.loc[~mismatch_table["match"]]
print(
    "trial-context audit:",
    f"{int((~mismatch_table['match']).sum())}/{len(mismatch_table)} mismatches",
)

# %% [markdown]
# ### 6c. Where does decoded content land? (unadjusted)
#
# Empirical share of MAP timebins per arm across all events in this epoch, with
# the current/previous-arm masses highlighted. This is a bin-weighted,
# unadjusted description — not the M1 coefficient. M1 conditions within event
# and controls static arm occupancy with arm fixed effects.

# %%
arm_mass = event_arm_table.groupby("arm")["arm_count_hard"].sum()
arm_mass = arm_mass / arm_mass.sum()
frac_current = (
    (event_arm_table["arm_count_hard"] * event_arm_table["is_current"]).sum()
    / event_arm_table["arm_count_hard"].sum()
)
frac_prev = (
    (event_arm_table["arm_count_hard"] * event_arm_table["is_prev"]).sum()
    / event_arm_table["arm_count_hard"].sum()
)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(arm_mass.index, arm_mass.values, color="C0")
ax.axhline(1 / 9, color="0.4", ls="--", lw=1, label="uniform (1/9)")
ax.set(xlabel="arm id", ylabel="fraction of replay timebins",
       title="replay content by arm (this epoch)")
ax.set_xticks(range(9))
ax.legend(fontsize=8)
ax.text(0.02, 0.95,
        f"on current arm: {frac_current:.0%}\non previous arm: {frac_prev:.0%}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
fig.tight_layout()

# %% [markdown]
# ### 6d. Model-light uncertainty/content view
#
# `build_trial_leaf_table` aggregates repeated stops to one row per behavioral
# trial × reward well and deliberately retains zero-replay trials. Uncertainty
# quartiles are assigned **within session and well**, so a persistently more
# uncertain physical well cannot create the trend by itself. The production
# profile helper first estimates each physical well and then weights wells
# equally; its absolute-rate estimand is a ratio of summed posterior mass to
# summed immobility exposure. We focus descriptively on unchosen destination
# wells: this conditioning uses the realized trial destination, so it is not a
# predictor-safe causal control. The conditional panel asks how observed replay
# content is allocated, while the absolute panel also retains no-replay trials.

# %%
trial_leaf_table, belief_replay_audit = build_trial_leaf_table(
    event_arm_table, stop_table, qvals_session
)
trial_group = ["session_id", "trial_number_by_epoch"]
assert len(trial_leaf_table) == 6 * trial_leaf_table[trial_group].drop_duplicates().shape[0]
assert (
    trial_leaf_table.groupby(trial_group, observed=True)["arm"].nunique() == 6
).all()

print(
    "belief/replay coverage:",
    f"stops={belief_replay_audit['stop_qvals_coverage']:.1%},",
    f"events={belief_replay_audit['event_qvals_coverage']:.1%},",
    f"posterior mass={belief_replay_audit['posterior_mass_qvals_coverage']:.1%}",
)

conditional_profiles = hierarchical_quartile_profile(
    trial_leaf_table,
    numerator_col="posterior_event_share",
    scope="unchosen",
)
absolute_profiles = hierarchical_quartile_profile(
    trial_leaf_table,
    numerator_col="posterior_mass",
    denominator_col="immobility_time",
    scope="unchosen",
)
conditional_overall = conditional_profiles["overall"]
absolute_overall = absolute_profiles["overall"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for _arm, group in conditional_profiles["arm"].groupby("arm", observed=True):
    x = group["uncertainty_quartile"].to_numpy(dtype=float)
    axes[0].plot(
        x, 100 * group["value"], color="0.75", lw=0.8, alpha=0.8
    )
for _arm, group in absolute_profiles["arm"].groupby("arm", observed=True):
    x = group["uncertainty_quartile"].to_numpy(dtype=float)
    axes[1].plot(x, group["value"], color="0.75", lw=0.8, alpha=0.8)

x = conditional_overall["uncertainty_quartile"].to_numpy(dtype=float)
axes[0].plot(
    x,
    100 * conditional_overall["value"],
    "o-",
    color="C4",
    lw=2,
    label="equal-well mean",
)
x = absolute_overall["uncertainty_quartile"].to_numpy(dtype=float)
axes[1].plot(
    x,
    absolute_overall["value"],
    "o-",
    color="C0",
    lw=2,
    label="equal-well mean",
)
axes[0].set(
    xlabel="same-well uncertainty quartile",
    ylabel="allocation to well (% of all replay content)",
    title="event-conditioned allocation\n(replay-bearing trials)",
)
axes[1].set(
    xlabel="same-well uncertainty quartile",
    ylabel="posterior-weighted decoder bins s⁻¹",
    title="absolute content rate\n(zero-replay trials retained)",
)
for ax in axes:
    ax.set_xticks([1, 2, 3, 4], labels=["Q1", "Q2", "Q3", "Q4"])
    ax.legend(fontsize=8)
fig.tight_layout()
display(  # noqa: F821
    pd.concat(
        {
            "conditional_event_share": conditional_overall,
            "absolute_rate": absolute_overall,
        },
        names=["estimand"],
    )
)

# %% [markdown]
# ### 6e. Within-arm exploit/explore sensitivity
#
# Beta forgetting makes uncertainty rise when a well has not been visited, and
# Beta variance also depends nonlinearly on mean value. An unadjusted uncertainty
# plot therefore cannot distinguish these ingredients. The production
# sensitivity fit restricts to non-current reward wells, removes each arm's mean,
# and co-fits uncertainty, learned value, distance, and visit recency. The
# production helper standardizes raw value, distance, and recency over the
# pooled **non-current leaf** subset; uncertainty retains the persisted
# all-event/all-arm standardization (including structural trunk zeros). These
# are fixed cross-session scales, but not identical reference populations. This
# cell therefore reads the small session-coefficient artifact produced by the
# full-table workflow rather than re-standardizing this epoch
# (`scripts/content/analyze_uncertainty_content.py`). The result is a
# descriptive OLS association after adjustment—not an independent uncertainty
# effect, a conditional-Poisson rate ratio, or population inference.
# `SWR` here means SWR-overlapping non-local events, not all detected SWRs.
# The response is hard-MAP `arm_fraction`; the posterior-weighted profiles above
# are a separate robustness estimand.
# Schema and row-count checks below establish cohort compatibility with this
# build; regenerating the production artifact remains the coefficient-freshness
# guarantee.

# %%
all_horserace = pd.read_csv(HORSERACE_COEFS_FILE)
horse_coef_columns = [
    "coef_arm_uncertainty_z",
    "coef_q_arm_z",
    "coef_dist_to_arm_z",
    "coef_visit_recency_z",
]
horse_required_columns = {
    "animal",
    "session_id",
    "population",
    "n_rows",
    "cond_number",
    *horse_coef_columns,
}
missing_horse_columns = sorted(horse_required_columns - set(all_horserace.columns))
if missing_horse_columns:
    raise ValueError(
        f"horse-race artifact is missing columns: {missing_horse_columns}"
    )
selected_horserace_id = "/".join(
    str(event_arm_table[col].iloc[0]) for col in ("animal", "day", "epoch")
)
belief_horserace = all_horserace.loc[
    all_horserace["session_id"].eq(selected_horserace_id)
].copy()
del all_horserace
if belief_horserace.empty:
    raise ValueError(f"no horse-race result for {selected_horserace_id}")
expected_populations = {"all", "SWR", "non_SWR"}
if (
    set(belief_horserace["population"]) != expected_populations
    or belief_horserace["population"].duplicated().any()
):
    raise ValueError(
        "horse-race artifact must have exactly one selected-session row for "
        f"each population: {sorted(expected_populations)}"
    )
selected_animal = str(event_arm_table["animal"].iloc[0])
if not belief_horserace["animal"].astype(str).eq(selected_animal).all():
    raise ValueError("horse-race artifact animal disagrees with the selected epoch")
horse_numeric_columns = ["n_rows", "cond_number", *horse_coef_columns]
if not np.isfinite(
    belief_horserace[horse_numeric_columns].to_numpy(dtype=float)
).all():
    raise ValueError("selected horse-race diagnostics and coefficients must be finite")
horse_predictors = [
    "arm_fraction",
    "arm_uncertainty_z",
    "q_arm",
    "dist_to_arm",
    "visit_recency",
]
local_horse_rows = event_arm_table.loc[
    event_arm_table["arm"].le(5) & event_arm_table["is_current"].eq(0)
]
population_masks = {
    "all": pd.Series(True, index=local_horse_rows.index),
    "SWR": local_horse_rows["is_swr"].astype(bool),
    "non_SWR": ~local_horse_rows["is_swr"].astype(bool),
}
for population, mask in population_masks.items():
    expected_rows = len(local_horse_rows.loc[mask, horse_predictors].dropna())
    observed_rows = int(
        belief_horserace.loc[
            belief_horserace["population"].eq(population), "n_rows"
        ].item()
    )
    assert observed_rows == expected_rows
horserace_columns = [
    "population",
    "n_rows",
    "cond_number",
    *horse_coef_columns,
]
belief_horserace[horserace_columns].set_index("population").round(4)

# %% [markdown]
# ## 7. Regression models (this epoch = one production unit)
#
# All four families are fit on **this single session's** events/stops. Arm and
# patch content condition on event; choice conditions on stop; the rate model
# is instead an ordinary Poisson GLM with an intercept and a log-exposure
# offset. Arm, patch, and choice comparison blocks use the same union
# complete-case unit gate as their production comparison workflows. The
# per-stop rate block instead calls each production fitter on the original
# `stop_table`, preserving the rate runner's per-rung complete-case cohorts.
# Consequently, rate AIC gains are shown only when parent and child use the
# exact same stop IDs (not merely the same count); held-out log likelihood and
# the reported `n_units` remain the primary production-parity diagnostics.
#
# The thin adapter below derives each displayed model block's predictor union;
# the package's production selector performs all cohort filtering. The other
# helpers only present fields returned by the package fitters.

# %%
def freeze_complete_case_units(
    frame: pd.DataFrame,
    specs: dict,
    *,
    unit_col: str,
    required_cols: tuple[str, ...],
    response_col: str | None = None,
    offset_col: str | None = None,
    drop_zero_response: bool = False,
    choice_response_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Thin notebook adapter over the production common-cohort selector."""
    predictors = sorted(
        {predictor for spec in specs.values() for predictor in spec.predictors}
    )
    return freeze_common_cohort(
        frame,
        unit_col=unit_col,
        required_cols=(*required_cols, *predictors),
        response_col=response_col,
        drop_zero_response=drop_zero_response,
        offset_col=offset_col,
        choice_response_col=choice_response_col,
    )


def cohort_count_summary(report: dict[str, object]) -> dict[str, object]:
    """Compact presentation of the production selector's full audit report."""
    fields = (
        "n_units_before",
        "n_missing_units",
        "n_nonpositive_offset_units",
        "n_zero_response_units",
        "n_invalid_choice_units",
        "n_units_after",
        "n_rows_after",
    )
    return {field: report[field] for field in fields}


def fits_to_stat_frame(
    fits: dict[str, dict | None],
    spec_map: dict,
    *,
    cohort_ids: dict[str, tuple] | None = None,
) -> pd.DataFrame:
    """Tidy per-model fit statistics from fitter return dicts (presentation only).

    The four fitters return slightly different key sets — the conditional
    Poisson emits ``aic``/``pseudo_r2``; the rate GLM emits ``aic`` and
    ``n_stops``; the conditional logit emits only the partial ``llf`` (no
    AIC). AIC gains are reported only against each specification's declared
    parent, preserving declared reporting branches such as Rlag ← R0;
    positive means the child has lower AIC and is favored.
    """
    rows = []
    for name, res in fits.items():
        spec = spec_map[name]
        if res is None:
            rows.append(
                {
                    "model": name,
                    "status": "skipped (insufficient/rank-deficient)",
                    "n_predictors": len(spec.predictors),
                }
            )
            continue
        status = "fit"
        if res.get("converged") is False:
            status = "nonconverged"
        elif res.get("separated") is True:
            status = "separated"
        rows.append(
            {
                "model": name,
                "n_predictors": len(spec.predictors),
                "n_units": res.get("n_events", res.get("n_stops")),
                "n_rows": res.get("n_rows"),
                "llf": res.get("llf"),
                "aic": res.get("aic"),
                "bic": res.get("bic"),
                "pseudo_r2": res.get("pseudo_r2"),
                "converged": res.get("converged"),
                "separated": res.get("separated"),
                "n_fit_warnings": res.get("n_convergence_warnings"),
                "cv_ll_per_stop_mean": res.get("cv_ll_per_stop_mean"),
                "cv_ll_per_stop_std": res.get("cv_ll_per_stop_std"),
                "cv_ll_per_event_mean": res.get("cv_ll_per_event_mean"),
                "cv_ll_per_event_std": res.get("cv_ll_per_event_std"),
                "cv_ll_per_timebin_mean": res.get("cv_ll_per_timebin_mean"),
                "cv_ll_per_timebin_std": res.get("cv_ll_per_timebin_std"),
                "cv_n_folds": res.get("cv_n_folds"),
                "cv_n_nonconverged_folds": res.get(
                    "cv_n_nonconverged_folds"
                ),
                "cv_n_separated_folds": res.get("cv_n_separated_folds"),
                "status": status,
            }
        )
    out = pd.DataFrame(rows).set_index("model")
    # Drop columns a given family never populates (e.g. no AIC for the logit).
    out = out.dropna(axis=1, how="all")
    if "aic" in out:
        out["aic_gain_vs_parent"] = np.nan
        for name in out.index:
            parent = spec_map[name].parent
            if cohort_ids is None:
                # Arm/patch blocks are passed one already-frozen common frame.
                same_cohort = (
                    "n_units" in out
                    and parent in out.index
                    and pd.notna(out.loc[parent, "n_units"])
                    and out.loc[parent, "n_units"] == out.loc[name, "n_units"]
                )
            else:
                # Per-rung rate fits start from the original stop table. Exact
                # IDs, not equal counts, are required for an AIC comparison.
                same_cohort = (
                    parent in cohort_ids
                    and name in cohort_ids
                    and cohort_ids[parent] == cohort_ids[name]
                )
            if (
                parent in out.index
                and out.loc[parent, "status"] == "fit"
                and out.loc[name, "status"] == "fit"
                and pd.notna(out.loc[name, "aic"])
                and same_cohort
            ):
                out.loc[name, "aic_gain_vs_parent"] = (
                    out.loc[parent, "aic"] - out.loc[name, "aic"]
                )
    return out


def coefs_to_frame(res: dict, spec) -> pd.DataFrame:
    """Coefficient / SE / p-value table for one fitted model."""
    rows = []
    for p in spec.predictors:
        rows.append({
            "predictor": p,
            "coef": res.get(f"coef_{p}"),
            "se": res.get(f"bse_{p}"),
            "pval": res.get(f"pval_{p}"),
            "rate_ratio": res.get(f"rr_{p}"),
            "vif": res.get(f"vif_{p}"),
        })
    return pd.DataFrame(rows).set_index("predictor").dropna(axis=1, how="all")


def fit_is_reliable(res: dict | None) -> bool:
    """Whether a fit is safe to summarize beyond its diagnostic row."""
    return (
        res is not None
        and res.get("converged", True) is not False
        and not bool(res.get("separated", False))
    )


# %% [markdown]
# ### 7a. Arm content — M0–M7 (conditional Poisson)
#
# Response `arm_count_hard`, grouped by `event_id`, 9 arms per event. M0 is the
# uniform null; M0a adds arm geometry; M1–M7 add spatial context, value/memory,
# and SWR interactions in nested steps. Parent→child AIC comparisons require
# the exact same events, so the union complete-case cohort is frozen first.

# %%
arm_fit_table, arm_cohort = freeze_complete_case_units(
    event_arm_table,
    ARM_MODELS,
    unit_col="event_id",
    required_cols=("arm_count_hard",),
    response_col="arm_count_hard",
    drop_zero_response=True,
)
print("arm common cohort:", cohort_count_summary(arm_cohort))

arm_fits = {
    name: fit_conditional_poisson(
        arm_fit_table,
        predictors=list(spec.predictors),
        response_col="arm_count_hard",
        event_col="event_id",
        n_categories=9,
    )
    for name, spec in ARM_MODELS.items()
}
arm_stats = fits_to_stat_frame(arm_fits, ARM_MODELS)
assert arm_stats.loc[arm_stats["n_units"].notna(), "n_units"].nunique() <= 1
arm_stats

# %% [markdown]
# Coefficients of the full model (M7). Because M7 carries `swr_x_*`, each main
# spatial/value coefficient is the non-SWR reference effect; its SWR counterpart
# is main + interaction. Exponentiating a coefficient gives a conditional
# arm-allocation ratio, not events/second. `vif` flags near-collinearity within
# this session.

# %%
if fit_is_reliable(arm_fits["M7"]):
    m7_coefs = coefs_to_frame(arm_fits["M7"], ARM_MODELS["M7"])
    display(m7_coefs.round(3))  # noqa: F821 (IPython display)
else:
    print("M7 is unavailable or flagged; coefficients are suppressed (see diagnostics).")

# %%
# AIC trajectory + key non-geometry coefficients across the hierarchy.
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

fit_models = arm_stats[arm_stats["status"] == "fit"]
axes[0].plot(fit_models.index, fit_models["aic"], "o-", color="C0")
axes[0].set(xlabel="model", ylabel="AIC", title="model fit (lower = better)")
axes[0].tick_params(axis="x", rotation=45)

# Track a few interpretable coefficients where present.
track = ["is_current", "is_prev", "reward_rate_arm_k10",
         "is_last_rewarded_elsewhere", "swr_x_current", "swr_x_prev"]
for predictor in track:
    xs, ys, es = [], [], []
    for name, res in arm_fits.items():
        if fit_is_reliable(res) and f"coef_{predictor}" in res:
            xs.append(name)
            ys.append(res[f"coef_{predictor}"])
            es.append(res.get(f"bse_{predictor}", np.nan))
    if xs:
        axes[1].errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=predictor)
axes[1].axhline(0, color="0.6", lw=1)
axes[1].set(xlabel="model (first appearance →)",
            ylabel="conditional log arm-allocation ratio",
            title="key coefficients across hierarchy")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ### 7b. Behavioral value and uncertainty branches
#
# These comparisons isolate the behavioral-belief story on one frozen cohort:
#
# * **M5 → Mv** adds the two-degree-of-freedom nominal + learned-value block;
# * **M5 → Mu** adds model-derived Beta uncertainty after the M5 spatial,
#   empirical reward-history, recency, and arm-identity adjustments;
# * **Mu → Mvu** adds learned value conditional on uncertainty;
# * **P3 → Pv** adds nominal + learned value after collapsing each pair of
#   reward wells into a patch.
#
# Mv and Mvu are not nested: Mv includes nominal value whereas Mvu replaces it
# with uncertainty, so no parent-relative gain is reported between them. The
# arm-level raw `p_reward_arm`/`q_arm` coefficients are per 1.0 probability,
# while `arm_uncertainty_z` is per pooled-global SD. The coefficient table below
# therefore also puts all three on the persisted global-SD scale for
# presentation. VIFs remain visible because value, variance, recency, and arm
# identity are substantially coupled in a single epoch.
#
# All M/P models ask **where content is allocated given a detected event**; they
# do not estimate whether an event occurs. AIC is in-sample. Five-fold
# event-held-out conditional log likelihood is shown beside it because a large
# AIC gain need not imply a large held-out improvement. These values are one
# illustrative session, not population evidence.

# %%
belief_arm_fit_table, belief_arm_cohort = freeze_complete_case_units(
    event_arm_table,
    BELIEF_ARM_MODELS,
    unit_col="event_id",
    required_cols=("arm_count_hard",),
    response_col="arm_count_hard",
    drop_zero_response=True,
)
print("behavioral arm common cohort:", cohort_count_summary(belief_arm_cohort))

belief_arm_fits = {}
for name, spec in BELIEF_ARM_MODELS.items():
    fit = fit_conditional_poisson(
        belief_arm_fit_table,
        predictors=list(spec.predictors),
        response_col="arm_count_hard",
        event_col="event_id",
        n_categories=9,
    )
    cv = cross_validate_conditional_poisson(
        belief_arm_fit_table,
        predictors=list(spec.predictors),
        response_col="arm_count_hard",
        event_col="event_id",
        n_categories=9,
    )
    if fit is not None and cv is not None:
        fit.update(cv)
    belief_arm_fits[name] = fit

belief_arm_stats = fits_to_stat_frame(belief_arm_fits, BELIEF_ARM_MODELS)
assert belief_arm_stats.loc[
    belief_arm_stats["n_units"].notna(), "n_units"
].nunique() <= 1
display(  # noqa: F821
    belief_arm_stats.reindex(
        columns=[
            "n_units",
            "aic",
            "aic_gain_vs_parent",
            "cv_ll_per_event_mean",
            "cv_ll_per_event_std",
            "cv_n_nonconverged_folds",
            "cv_n_separated_folds",
            "status",
        ]
    ).round(3)
)

# %%
belief_arm_terms = {
    "Mv": ("p_reward_arm", "q_arm"),
    "Mu": ("arm_uncertainty_z",),
    "Mvu": ("q_arm", "arm_uncertainty_z"),
}
belief_effect_rows = []
for name, predictors in belief_arm_terms.items():
    fit = belief_arm_fits[name]
    if not fit_is_reliable(fit):
        continue
    for predictor in predictors:
        if predictor.endswith("_z"):
            scale = 1.0
            input_unit = "1 pooled-global SD"
        else:
            scale = used_arm_z_reference[predictor]["std"]
            input_unit = "1.0 reward probability"
        coefficient = fit.get(f"coef_{predictor}", np.nan)
        beta_sd = coefficient * scale
        belief_effect_rows.append(
            {
                "model": name,
                "predictor": predictor,
                "input_unit": input_unit,
                "coef_in_fit_units": coefficient,
                "se_in_fit_units": fit.get(f"bse_{predictor}", np.nan),
                "pval_within_session": fit.get(f"pval_{predictor}", np.nan),
                "vif": fit.get(f"vif_{predictor}", np.nan),
                "coef_per_global_sd": beta_sd,
                "allocation_ratio_per_global_sd": np.exp(beta_sd),
            }
        )
if belief_effect_rows:
    belief_arm_effects = pd.DataFrame(belief_effect_rows).set_index(
        ["model", "predictor"]
    )
    display(belief_arm_effects.round(3))  # noqa: F821
else:
    print("No behavioral arm fit is both available and unflagged.")

# %%
belief_patch_fit_table, belief_patch_cohort = freeze_complete_case_units(
    event_patch_table,
    BELIEF_PATCH_MODELS,
    unit_col="event_id",
    required_cols=("patch_count_hard",),
    response_col="patch_count_hard",
    drop_zero_response=True,
)
print("behavioral patch common cohort:", cohort_count_summary(belief_patch_cohort))

belief_patch_fits = {}
for name, spec in BELIEF_PATCH_MODELS.items():
    fit = fit_conditional_poisson(
        belief_patch_fit_table,
        predictors=list(spec.predictors),
        response_col="patch_count_hard",
        event_col="event_id",
        n_categories=3,
    )
    cv = cross_validate_conditional_poisson(
        belief_patch_fit_table,
        predictors=list(spec.predictors),
        response_col="patch_count_hard",
        event_col="event_id",
        n_categories=3,
    )
    if fit is not None and cv is not None:
        fit.update(cv)
    belief_patch_fits[name] = fit

belief_patch_stats = fits_to_stat_frame(belief_patch_fits, BELIEF_PATCH_MODELS)
assert belief_patch_stats.loc[
    belief_patch_stats["n_units"].notna(), "n_units"
].nunique() <= 1
display(  # noqa: F821
    belief_patch_stats.reindex(
        columns=[
            "n_units",
            "aic",
            "aic_gain_vs_parent",
            "cv_ll_per_event_mean",
            "cv_ll_per_event_std",
            "cv_n_nonconverged_folds",
            "cv_n_separated_folds",
            "status",
        ]
    ).round(3)
)
if fit_is_reliable(belief_patch_fits["Pv"]):
    pv_coefs = coefs_to_frame(belief_patch_fits["Pv"], BELIEF_PATCH_MODELS["Pv"])
    display(  # noqa: F821
        pv_coefs.loc[["p_reward_arm_patch", "q_arm_patch"]].round(3)
    )
else:
    print("Pv is unavailable or flagged; coefficients are suppressed.")

# %% [markdown]
# ### 7c. Patch content — P0–P4 (conditional Poisson, 3 categories)
#
# Same estimator, arms collapsed to the three patches. Requires ≥10 events per
# session (`fit_conditional_poisson` rejects below `2 × n_predictors` events;
# the event-source comparison runner also gates patch fits at 10). P0–P4 share
# the union complete-case event cohort; the qvals-dependent P3→Pv comparison is
# kept in the separate matched-cohort branch above. Patch value is the mean of
# the two leaf values, not a value-to-go estimate. Coefficients are conditional
# log patch-allocation ratios, not event-rate effects.

# %%
patch_fit_table, patch_cohort = freeze_complete_case_units(
    event_patch_table,
    PATCH_MODELS,
    unit_col="event_id",
    required_cols=("patch_count_hard",),
    response_col="patch_count_hard",
    drop_zero_response=True,
)
print("patch common cohort:", cohort_count_summary(patch_cohort))

patch_fits = {
    name: fit_conditional_poisson(
        patch_fit_table,
        predictors=list(spec.predictors),
        response_col="patch_count_hard",
        event_col="event_id",
        n_categories=3,
    )
    for name, spec in PATCH_MODELS.items()
}
patch_stats = fits_to_stat_frame(patch_fits, PATCH_MODELS)
assert patch_stats.loc[patch_stats["n_units"].notna(), "n_units"].nunique() <= 1
patch_stats

# %%
if fit_is_reliable(patch_fits["P4"]):
    display(coefs_to_frame(patch_fits["P4"], PATCH_MODELS["P4"]).round(3))  # noqa: F821
else:
    print("P4 is unavailable or flagged; coefficients are suppressed (see diagnostics).")

# %% [markdown]
# ### 7d. Event rate — R0–R2, Rlag (Poisson GLM, log-time offset)
#
# `fit_per_stop_rate_model` regresses a count per stop with a
# `log(immobility_time)` offset, so coefficients act on events/second rather
# than raw counts. The current Family-A runner repeats the same ladder for four
# canonical outcomes: detected SWRs (positive control), all non-local events,
# and the SWR-overlap/non-SWR split of non-local events. Unit = stop;
# production gate is ≥20 valid stops per session.
#
# `R0 → R1 → R2` is nested: R1 adds past-only `reward_lag1` plus
# switch context; R2 adds the trailing 10-trial reward rate. `Rlag` estimates
# reward lags 1–10 separately. The declared reporting parents are R0→R1→R2
# and R0→Rlag. As in production, each rung receives the original stop table and
# `fit_per_stop_rate_model` applies that rung's own missingness/exposure gate.
# Early R0–R2 stops therefore need not all appear in Rlag. The table reports
# `n_units` for every rung and suppresses parent-relative AIC gain unless the
# exact stop IDs match. Production's five-fold held-out log likelihood is
# computed by the same package function here; higher `cv_ll_per_stop_mean` is
# better. Treat CV values from different rung-specific cohorts as production-
# parity diagnostics, not paired improvements. Compare fits only within the
# same outcome and identical cohort, never across the four response
# distributions.
#
# This is **Family A**, the literature-comparable per-stop layer. The
# time-resolved per-bin conditional-intensity ladder (**Family B**,
# `MODEL_PER_BIN_RATE`) is not walked in this notebook — see
# `docs/rate_models.md` and `scripts/rate/fit_per_bin_rate_models.py`.

# %%
rate_fits = {}
rate_cohort_sizes = {}
rate_stat_frames = {}
reward_lag_cols = [f"reward_lag{lag}" for lag in range(1, N_REWARD_LAGS + 1)]
np.testing.assert_allclose(
    stop_table["reward_rate_trial_k10"].to_numpy(dtype=float),
    stop_table[reward_lag_cols].mean(axis=1).to_numpy(dtype=float),
    rtol=0,
    atol=1e-12,
    equal_nan=True,
)

for outcome, response_col in PER_STOP_RATE_OUTCOMES.items():
    fits = {}
    outcome_cohort_ids = {}
    for name, spec in RATE_MODELS.items():
        _, selection = freeze_common_cohort(
            stop_table,
            unit_col="stop_id",
            required_cols=(response_col, *spec.predictors),
            response_col=response_col,
            offset_col="immobility_time",
        )
        outcome_cohort_ids[name] = selection["included_ids"]
        fits[name] = fit_per_stop_rate_model(
            stop_table,
            list(spec.predictors),
            response_col=response_col,
            offset_col="immobility_time",
            min_stops=20,
        )
        if fits[name] is not None:
            assert fits[name]["n_stops"] == len(outcome_cohort_ids[name])
    for name, spec in RATE_MODELS.items():
        cv = cross_validate_per_stop_rate_model(
            stop_table,
            list(spec.predictors),
            response_col=response_col,
            offset_col="immobility_time",
            min_stops=20,
        )
        if fits[name] is not None and cv is not None:
            fits[name].update(cv)
    stats = fits_to_stat_frame(
        fits, RATE_MODELS, cohort_ids=outcome_cohort_ids
    )
    rate_fits[outcome] = fits
    rate_cohort_sizes[outcome] = {
        name: None if fit is None else int(fit["n_stops"])
        for name, fit in fits.items()
    }
    rate_stat_frames[outcome] = stats

print("production per-rung rate cohort sizes:", rate_cohort_sizes)
rate_stats = pd.concat(rate_stat_frames, names=["outcome", "model"])
rate_stats

# %%
r2_coefs = {
    outcome: coefs_to_frame(fits["R2"], RATE_MODELS["R2"])
    for outcome, fits in rate_fits.items()
    if fit_is_reliable(fits["R2"])
}
if r2_coefs:
    display(pd.concat(r2_coefs, names=["outcome", "predictor"]).round(3))  # noqa: F821
else:
    print("No R2 fit is both available and unflagged (see diagnostics).")

# %% [markdown]
# ### 7e. Next-arm choice — C_* (conditional logit)
#
# Is replay content associated with where the animal *goes next*, beyond a
# behavioural baseline? `build_choice_dataset` makes a stop × candidate-arm
# table: six well arms minus the current arm = five alternatives, with trunk
# arms excluded. The current production workflow fits a primary block
# (`C_replay_only`, `C_behavior`, `C_full`) and a separate SWR/non-SWR replay
# block (`C_behavior`, `C_full_swr_split`), each on its own frozen stop cohort.
# These are in-sample conditional-logit fits; there is no held-out choice CV.
# `replay_frac` is the unweighted mean of per-event arm fractions within a stop;
# the SWR-split terms are the corresponding event-type-restricted means. It is
# not a duration- or decoder-bin-pooled fraction.
# Coefficients are within-stop log relative-choice-odds (utility) slopes, not
# calibrated predictive performance.

# %%
choice_df = build_choice_dataset(event_table, event_arm_table, stop_table)
if choice_df.empty:
    raise ValueError("this session has no valid replay stops with a next-arm choice")

choice_sizes = choice_df.groupby("stop_id").size()
chosen_per_stop = choice_df.groupby("stop_id")["chosen"].sum()
assert (choice_sizes == 5).all()
assert (chosen_per_stop == 1).all()
print(
    "choice_df:",
    choice_df.shape,
    f"({choice_df['stop_id'].nunique()} stops × 5 candidate well arms)",
)

choice_fit_table, choice_cohort = freeze_complete_case_units(
    choice_df,
    CHOICE_MODELS,
    unit_col="stop_id",
    required_cols=("chosen",),
    choice_response_col="chosen",
)
choice_split_fit_table, choice_split_cohort = freeze_complete_case_units(
    choice_df,
    CHOICE_SPLIT_MODELS,
    unit_col="stop_id",
    required_cols=("chosen",),
    choice_response_col="chosen",
)
print("choice primary cohort:", cohort_count_summary(choice_cohort))
print("choice SWR-split cohort:", cohort_count_summary(choice_split_cohort))

choice_fits = {
    name: fit_choice_model(
        choice_fit_table, list(spec.predictors), stop_col="stop_id"
    )
    for name, spec in CHOICE_MODELS.items()
}
choice_split_fits = {
    name: fit_choice_model(
        choice_split_fit_table, list(spec.predictors), stop_col="stop_id"
    )
    for name, spec in CHOICE_SPLIT_MODELS.items()
}
choice_primary_stats = fits_to_stat_frame(choice_fits, CHOICE_MODELS)
choice_split_stats = fits_to_stat_frame(choice_split_fits, CHOICE_SPLIT_MODELS)
assert choice_primary_stats.loc[
    choice_primary_stats["n_units"].notna(), "n_units"
].nunique() <= 1
assert choice_split_stats.loc[
    choice_split_stats["n_units"].notna(), "n_units"
].nunique() <= 1
choice_stats = pd.concat(
    {"primary": choice_primary_stats, "swr_split": choice_split_stats},
    names=["block", "model"],
)
choice_stats

# %% [markdown]
# The partial-log-likelihood gains below quantify incremental **in-sample fit**.
# They are guaranteed to be nonnegative when an identifiable predictor is added
# and are not evidence of out-of-sample predictive improvement. The coefficient
# p-value is the fitter's within-session Wald test.

# %%
if fit_is_reliable(choice_fits["C_full"]) and fit_is_reliable(
    choice_fits["C_behavior"]
):
    d_llf = choice_fits["C_full"]["llf"] - choice_fits["C_behavior"]["llf"]
    print(f"C_behavior partial llf : {choice_fits['C_behavior']['llf']:.2f}")
    print(f"C_full     partial llf : {choice_fits['C_full']['llf']:.2f}")
    print(f"2Δllf from +replay_frac: {2 * d_llf:.2f}  (1 added parameter)")
    print(f"replay_frac coef       : {choice_fits['C_full']['coef_replay_frac']:.3f}"
          f"  (p = {choice_fits['C_full']['pval_replay_frac']:.3g})")
else:
    print("C_full / C_behavior are not both available and unflagged.")

if (
    fit_is_reliable(choice_split_fits["C_full_swr_split"])
    and fit_is_reliable(choice_split_fits["C_behavior"])
):
    split_d_llf = (
        choice_split_fits["C_full_swr_split"]["llf"]
        - choice_split_fits["C_behavior"]["llf"]
    )
    print(f"2Δllf from SWR + non-SWR replay: {2 * split_d_llf:.2f} (2 added parameters)")
else:
    print("C_full_swr_split / C_behavior are not both available and unflagged.")

# %%
if fit_is_reliable(choice_fits["C_full"]):
    display(coefs_to_frame(choice_fits["C_full"], CHOICE_MODELS["C_full"]).round(3))  # noqa: F821
else:
    print("C_full is unavailable or flagged; coefficients are suppressed.")

if fit_is_reliable(choice_split_fits["C_full_swr_split"]):
    display(  # noqa: F821
        coefs_to_frame(
            choice_split_fits["C_full_swr_split"],
            CHOICE_SPLIT_MODELS["C_full_swr_split"],
        ).round(3)
    )
else:
    print("C_full_swr_split is unavailable or flagged; coefficients are suppressed.")

# %% [markdown]
# ## 8. Summary
#
# | Family | Estimator | Unit | Asks |
# |---|---|---|---|
# | **M0–M7** | conditional Poisson (9 arms) | event | *Which arm* does replay represent? |
# | **M5/Mv/Mu/Mvu** | conditional Poisson (9 arms) | event | Does conditional content allocation track nominal value, learned value, or Beta uncertainty? |
# | **P0–P4** | conditional Poisson (3 patches) | event | *Which patch* does replay represent? |
# | **P3/Pv** | conditional Poisson (3 patches) | event | Does conditional patch allocation track nominal and learned value? |
# | **R0–R2, Rlag** × 4 outcomes | Poisson GLM + log-time offset | stop | *How often* do events occur? (Family A) |
# | **C_\*** | conditional logit | stop | Is replay content associated with the *next choice*? |
#
# The behavioral section now shows the model state itself, model-light
# conditional-allocation and absolute-rate estimands, the production-scaled
# within-arm sensitivity analysis, and the value/uncertainty content branches.
# The valid nested paths are M5→Mv, M5→Mu, Mu→Mvu, and P3→Pv; Mv↔Mvu is not a
# nested comparison. Content AIC and event-held-out likelihood, rate AIC and
# held-out per-stop likelihood, choice partial-likelihood gains, coefficients,
# VIFs, and explicit fit diagnostics expose both signal and fragility.
#
# `q_arm` is a learned immediate reward probability, `arm_uncertainty` is its
# model-derived Beta variance, and both are pre-outcome. Uncertainty remains
# coupled to mean value and forgetting/recency; the adjusted association should
# not be called causal or fully independent. Structural trunk zeros define the
# design matrix, but scientific interpretation of these belief covariates is
# restricted to reward-well arms. The current epoch is an illustrative fitting
# unit—population conclusions still require the multi-session aggregation.
#
# **Caveat — single-session estimates are noisy and correlational.** Some models
# can fail the sample-size, rank, convergence, or separation checks surfaced
# above. The population runners aggregate the retained per-session fits with
# session-level inference, per-animal sign consistency, cohort audits, and
# multiple-comparison control. Read this notebook as one audited fitting unit,
# not as the scientific result.
