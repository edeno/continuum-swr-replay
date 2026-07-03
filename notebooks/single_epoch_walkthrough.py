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
# ---

# %% [markdown]
# # Single-epoch walkthrough: from raw data to the replay-content regressions
#
# This notebook walks the **complete analysis pipeline for one session** —
# `chimi`, day `20200213`, epoch `02_r1`, hippocampus (HPC) — from raw data
# loading through every regression-model family.
#
# **Why one epoch is the right unit.** The production pipeline
# (`scripts/content/fit_replay_content_models.py`,
# `scripts/content/compare_event_source_models.py`) fits **one model per session**,
# where a *session* is a single `(animal, day, epoch)` tuple. Each conditional
# model conditions on *event* or *stop* within that one session; the
# population result is an aggregate of these per-session fits. So a
# single-epoch fit here is not a toy — it is exactly one unit of the real
# analysis. The population-level conclusions (and the caveats that several
# single-session-suggestive effects only survive after autocorrelation /
# multiple-comparison control) live one level up, across sessions.
#
# **Decoding is loaded, not re-run.** The non-local state-space decoder is
# GPU-heavy and runs for hours per epoch. Its outputs are already on disk
# (`Processed-Data/*_results.nc`, `*_environment.pkl`,
# `*_data_over_time.pkl`). We *show* the decode call for reference but **load**
# its outputs, so this notebook runs end-to-end in well under a minute.
#
# **Real functions only.** Every table and every model is built with the
# project's own package functions (`continuum_swr_replay.analysis.replay.*`).
# Nothing statistical is re-implemented inline — the notebook is a thin driver,
# mirroring `scripts/prepare_replay_content_data.py:_process_session`.
#
# ## Pipeline
#
# ```
# load_data ─► [decode: load precomputed] ─► data_over_time (dot)
#    │                                              │
#    │                          ┌───────────────────┼────────────────────┐
#    │                          ▼                   ▼                    ▼
#    │                   build_event_table   build_stop_table   (posterior .nc)
#    │                          │                   │
#    │                          ▼                   │
#    │                 build_event_arm_table        │
#    │                   /            \             │
#    │                  ▼              ▼            │
#    │       build_event_patch    add_arm_fixed     │
#    │           _table            _effects         │
#    └──────────────────┬──────────────┬────────────┘
#                       ▼              ▼
#         ┌─────────────┴───────┬──────┴────────┬─────────────────┐
#         ▼                     ▼               ▼                 ▼
#   M0–M7 arm content   P0–P4 patch    R0–R3 event rate   C_* next-arm choice
#   (conditional        (conditional   (Poisson GLM,      (conditional logit,
#    Poisson)            Poisson)       log-time offset)   stop × candidate arm)
# ```

# %%
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

# --- Data loading + decode outputs ---
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.analysis.replay.posterior_io import (
    load_conditional_posterior,
)

# --- Canonical table builders (same ones the prep script calls) ---
from continuum_swr_replay.analysis.replay.tables import (
    build_event_table,
    build_event_arm_table,
    build_event_patch_table,
    build_stop_table,
)
from continuum_swr_replay.analysis.replay._helpers import add_arm_fixed_effects

# --- Model specifications (the four families) ---
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_HIERARCHY,  # M0–M7  : arm content
    MODEL_PATCH,      # P0–P4  : patch content
    MODEL_RATE,       # R0–R3  : event rate per stop
    MODEL_CHOICE,     # C_*    : next-arm choice
)

# --- Fitters (statsmodels conditional models / Poisson GLM) ---
from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay.rate_model import fit_rate_model
from continuum_swr_replay.analysis.replay.choice_prediction import (
    build_choice_dataset,
    fit_choice_model,
)

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

for p in (DOT_FILE, ENV_FILE, RESULTS_FILE):
    print(f"{'OK ' if p.exists() else 'MISSING'} {p.name}")

# %% [markdown]
# ## 1. Load raw session data
#
# `load_data()` orchestrates the whole loader: DLC position linearized onto the
# track graph, spike times per brain area, ripple-band LFP and detected
# ripples, DIO task events, trials, and derived task variables. It queries
# Spyglass, so this is the slowest cell (~30–60 s).

# %%
data = load_data(NWB_FILE_NAME, EPOCH_NAME, ripple_detector_name="Kay")

position_info = data["position_info"]
trials = data["trials"]
track_graph = data["track_graph"]
ripple_times = data["ripple_times"]

print("spike_times brain areas:", list(data["spike_times"].keys()))
print("position_info :", position_info.shape, "| time-indexed in seconds")
print("trials        :", trials.shape)
print("ripple_times  :", ripple_times.shape)
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
import networkx as nx

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

# %% [markdown]
# ## 2. Decoding — load precomputed outputs
#
# The non-local clusterless detector (`NonLocalClusterlessDetector`) was fit on
# no-ripple training timepoints and applied to the whole epoch by
# `scripts/decoding/run_decoding_bandit.py`. For reference, that call is:
#
# ```python
# # NOT executed here — GPU, hours. Outputs already on disk.
# from scripts.run_decoding_bandit import run_decode
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
with open(ENV_FILE, "rb") as f:
    env = pickle.load(f)
posterior = load_conditional_posterior(RESULTS_FILE)

n_nonlocal_events = int(dot.loc[dot["non_local_event"] > 0, "non_local_event"].nunique())
print("dot           :", dot.shape)
print("posterior     :", dict(posterior.sizes))
print("n place bins  :", env.place_bin_centers_.shape[0])
print("n_nonlocal_events (raw labels):", n_nonlocal_events)
print("\ndot state/event columns:\n",
      [c for c in dot.columns if "prob" in c or "event" in c or c == "map_estimate"])

# %% [markdown]
# ### 2a. One decoded non-local event
#
# Pick the highest-probability non-local event and show the conditional
# posterior over linear position through its window, with the animal's actual
# (stationary) position overlaid. A non-local event is a stretch where the
# decoded position departs from where the animal physically is — the
# "replay-like" content this whole pipeline quantifies.

# %%
nonlocal_summed = (
    dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
)
labels = dot.loc[dot["non_local_event"] > 0, "non_local_event"]
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
ax.plot(pos_win.index, pos_win.values, ".", color="white", ms=4,
        label="animal position")
ax.axvspan(ev_times.min(), ev_times.max(), color="red", alpha=0.12,
           label="non-local event")
ax.set(xlabel="time (s)", ylabel="linear position (cm)",
       title=f"event {int(example_label)}: conditional non-local posterior")
ax.legend(loc="upper right", fontsize=8)
fig.colorbar(mesh, ax=ax, label="P(position | non-local)", pad=0.01)
fig.tight_layout()

# %% [markdown]
# ## 3. Event classification table
#
# `build_event_table` produces **one row per non-local event**. It wraps the
# canonical `build_classified_nonlocal_event_table` (full-window timing,
# per-event speed summary, peak non-local probability, the 3-way SWR
# `overlap_class` and matched SWR) and adds the replay-content filtered-window
# columns. The conditional `posterior` is passed so posterior-weighted columns
# are populated.

# %%
event_table = build_event_table(
    dot, env, trials, track_graph, conditional_posterior=posterior
)
print("event_table:", event_table.shape)
print("\noverlap_class counts:\n", event_table["overlap_class"].value_counts())
event_table[
    [
        "event_id", "event_start", "event_end", "event_duration",
        "mean_speed", "peak_nonlocal_prob", "overlap_class", "is_swr",
    ]
].head()

# %% [markdown]
# ### 3a. Event summary for this epoch

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

axes[0].hist(event_table["event_duration"] * 1e3, bins=40, color="C0")
axes[0].set(xlabel="event duration (ms)", ylabel="# events",
            title="event durations")

axes[1].hist(event_table["peak_nonlocal_prob"], bins=40, color="C1")
axes[1].set(xlabel="peak non-local prob", title="peak non-local probability")

vc = event_table["overlap_class"].value_counts()
axes[2].bar(vc.index.astype(str), vc.values, color="C2")
axes[2].set(ylabel="# events", title="SWR overlap class")
axes[2].tick_params(axis="x", rotation=20)
fig.tight_layout()

# %% [markdown]
# ## 4. Stop table
#
# `build_stop_table` produces **one row per stopping period** (immobility
# bout), *including zero-event stops*. This is the denominator the event-rate
# model needs and the unit the choice model is built on. It is a function of
# immobility bouts, not of non-local events, so it is built directly from
# `dot` + `trials` + `track_graph`.

# %%
stop_table = build_stop_table(dot, trials, track_graph)
print("stop_table:", stop_table.shape)
print("stops with >=1 event:", int((stop_table["n_events"] > 0).sum()),
      "/", len(stop_table))
stop_table.head()

# %% [markdown]
# ## 5. Long-format design matrices (arm and patch)
#
# `build_event_arm_table` expands the event table to **one row per
# (event, arm)** — nine arms per event — and computes the full per-event
# predictor set used by M0–M7:
#
# * `arm_count_hard` — response: # MAP timebins this event spent on this arm
# * `is_current`, `is_prev`, `is_last_rewarded_elsewhere` — spatial context
# * `recent_reward_rate`, `visit_recency_demeaned` — value / memory
# * `swr_x_*` — SWR-overlap interactions (M6/M7)
#
# The only thing added on top is `add_arm_fixed_effects`, which writes the
# `arm_fe_0…arm_fe_7` dummies (arm 8 is the reference) — the M0a geometry
# baseline. `build_event_patch_table` collapses arms→patches (3 categories) for
# the patch hierarchy.

# %%
event_arm_table = build_event_arm_table(
    event_table, dot, env, trials, track_graph, conditional_posterior=posterior
)
add_arm_fixed_effects(event_arm_table)  # in-place: adds arm_fe_0 … arm_fe_7

event_patch_table = build_event_patch_table(event_table, event_arm_table)

print("event_arm_table  :", event_arm_table.shape,
      f"({event_arm_table['event_id'].nunique()} events × 9 arms)")
print("event_patch_table:", event_patch_table.shape,
      f"({event_patch_table['event_id'].nunique()} events × 3 patches)")
print("\none event (9 arms):")
cols = ["event_id", "arm", "arm_count_hard", "is_current", "is_prev",
        "recent_reward_rate", "visit_recency_demeaned",
        "is_last_rewarded_elsewhere", "swr_x_current"]
event_arm_table.loc[
    event_arm_table["event_id"] == event_arm_table["event_id"].iloc[0], cols
]

# %% [markdown]
# ### 5a. Where does replay content land?
#
# Empirical share of MAP timebins per arm across all events in this epoch, with
# the current/previous-arm masses highlighted. The current-arm enrichment is
# the M1 effect the regression formalizes.

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
# ## 6. Regression models (this epoch = one production unit)
#
# All four families are fit on **this single session's** events/stops, exactly
# as the production per-session loop does. Each family conditions on a
# within-session unit (event or stop), so no event-level constant is
# identifiable — there is no intercept in the conditional models.
#
# We keep a small presentation helper that only *extracts* fields already
# returned by the fitters (AIC/BIC/log-likelihood/pseudo-R²/coefficients).
# Nothing statistical is computed here.

# %%
def fits_to_stat_frame(fits: dict[str, dict | None], spec_map: dict) -> pd.DataFrame:
    """Tidy per-model fit statistics from fitter return dicts (presentation only).

    The four fitters return slightly different key sets — the conditional
    Poisson emits ``aic``/``pseudo_r2``; the rate GLM emits ``aic`` and
    ``n_stops``; the conditional logit emits only the partial ``llf`` (no
    AIC). We surface a common unit count (``n_units`` = events or stops) and
    only compute ``delta_aic`` over the models that actually report an AIC.
    """
    rows = []
    for name, res in fits.items():
        spec = spec_map[name]
        if res is None:
            rows.append({"model": name, "status": "skipped (insufficient data)",
                         "n_predictors": len(spec.predictors)})
            continue
        rows.append({
            "model": name,
            "n_predictors": len(spec.predictors),
            "n_units": res.get("n_events", res.get("n_stops")),
            "llf": res.get("llf"),
            "aic": res.get("aic"),
            "bic": res.get("bic"),
            "pseudo_r2": res.get("pseudo_r2"),
            "status": "fit",
        })
    out = pd.DataFrame(rows).set_index("model")
    # Drop columns a given family never populates (e.g. no AIC for the logit).
    out = out.dropna(axis=1, how="all")
    if "aic" in out:
        out["delta_aic"] = out["aic"] - out["aic"].min()
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
            "vif": res.get(f"vif_{p}"),
        })
    return pd.DataFrame(rows).set_index("predictor")


# %% [markdown]
# ### 6a. Arm content — M0–M7 (conditional Poisson)
#
# Response `arm_count_hard`, grouped by `event_id`, 9 arms per event. M0 is the
# uniform null; M0a adds arm geometry; M1–M7 add spatial context, value/memory,
# and SWR interactions in nested steps. Lower AIC = better.

# %%
arm_fits = {
    name: fit_conditional_poisson(
        event_arm_table,
        predictors=list(spec.predictors),
        response_col="arm_count_hard",
        event_col="event_id",
        n_categories=9,
    )
    for name, spec in MODEL_HIERARCHY.items()
}
arm_stats = fits_to_stat_frame(arm_fits, MODEL_HIERARCHY)
arm_stats

# %% [markdown]
# Coefficients of the full model (M7). Positive `is_current` = replay biased
# toward the arm the animal is currently on; `swr_x_*` terms = how SWR-overlap
# events differ. `vif` flags any predictor that became near-collinear within
# this one session.

# %%
if arm_fits["M7"] is not None:
    m7_coefs = coefs_to_frame(arm_fits["M7"], MODEL_HIERARCHY["M7"])
    display(m7_coefs.round(3))  # noqa: F821 (IPython display)
else:
    print("M7 did not fit on this epoch (insufficient/again collinear data).")

# %%
# AIC trajectory + key non-geometry coefficients across the hierarchy.
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

fit_models = arm_stats[arm_stats["status"] == "fit"]
axes[0].plot(fit_models.index, fit_models["aic"], "o-", color="C0")
axes[0].set(xlabel="model", ylabel="AIC", title="model fit (lower = better)")
axes[0].tick_params(axis="x", rotation=45)

# Track a few interpretable coefficients where present.
track = ["is_current", "is_prev", "recent_reward_rate",
         "is_last_rewarded_elsewhere", "swr_x_current", "swr_x_prev"]
for predictor in track:
    xs, ys, es = [], [], []
    for name, res in arm_fits.items():
        if res is not None and f"coef_{predictor}" in res:
            xs.append(name)
            ys.append(res[f"coef_{predictor}"])
            es.append(res.get(f"bse_{predictor}", np.nan))
    if xs:
        axes[1].errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=predictor)
axes[1].axhline(0, color="0.6", lw=1)
axes[1].set(xlabel="model (first appearance →)", ylabel="coefficient (log-rate)",
            title="key coefficients across hierarchy")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ### 6b. Patch content — P0–P4 (conditional Poisson, 3 categories)
#
# Same estimator, arms collapsed to the three patches. Requires ≥10 events per
# session (`fit_conditional_poisson` rejects below `2 × n_predictors` events;
# patch fits also gate at 10 in production).

# %%
patch_fits = {
    name: fit_conditional_poisson(
        event_patch_table,
        predictors=list(spec.predictors),
        response_col="patch_count_hard",
        event_col="event_id",
        n_categories=3,
    )
    for name, spec in MODEL_PATCH.items()
}
patch_stats = fits_to_stat_frame(patch_fits, MODEL_PATCH)
patch_stats

# %%
if patch_fits["P4"] is not None:
    display(coefs_to_frame(patch_fits["P4"], MODEL_PATCH["P4"]).round(3))  # noqa: F821
else:
    print("P4 did not fit on this epoch.")

# %% [markdown]
# ### 6c. Event rate — R0–R3 (Poisson GLM, log-time offset)
#
# `fit_rate_model` regresses `n_events` per stop with a `log(immobility_time)`
# offset (so the coefficient is on *rate*, not raw count). Unit = stop;
# production gate is ≥20 stops per session.

# %%
rate_fits = {
    name: fit_rate_model(
        stop_table,
        predictors=list(spec.predictors),
        response_col="n_events",
        offset_col="immobility_time",
        min_stops=20,
    )
    for name, spec in MODEL_RATE.items()
}
rate_stats = fits_to_stat_frame(rate_fits, MODEL_RATE)
rate_stats

# %%
if rate_fits["R3"] is not None:
    display(coefs_to_frame(rate_fits["R3"], MODEL_RATE["R3"]).round(3))  # noqa: F821
else:
    print("R3 did not fit on this epoch (need >=20 stops).")

# %% [markdown]
# ### 6d. Next-arm choice — C_* (conditional logit)
#
# Does replay content predict where the animal *goes next*, beyond a
# behavioural baseline? `build_choice_dataset` makes a stop × candidate-arm
# table (well arms minus the current arm); `fit_choice_model` fits a
# conditional logit grouped by stop. `C_full` vs `C_behavior` is the primary
# test (does `replay_frac` add predictive power).

# %%
choice_df = build_choice_dataset(event_table, event_arm_table, stop_table)
print("choice_df:", choice_df.shape,
      f"({choice_df['stop_id'].nunique()} stops × candidate arms)")

choice_fits = {}
for name, spec in MODEL_CHOICE.items():
    preds = [p for p in spec.predictors if p in choice_df.columns]
    choice_fits[name] = fit_choice_model(choice_df, preds, stop_col="stop_id")
choice_stats = fits_to_stat_frame(choice_fits, MODEL_CHOICE)
choice_stats

# %% [markdown]
# Primary choice test: does adding replay content (`C_full`) improve on the
# behavioural baseline (`C_behavior`)? The conditional logit reports a partial
# log-likelihood, so we read the gain directly (1 added predictor,
# `replay_frac`). A small gain = replay adds little to next-arm prediction
# beyond behaviour — consistent with the population finding.

# %%
if choice_fits["C_full"] and choice_fits["C_behavior"]:
    d_llf = choice_fits["C_full"]["llf"] - choice_fits["C_behavior"]["llf"]
    print(f"C_behavior partial llf : {choice_fits['C_behavior']['llf']:.2f}")
    print(f"C_full     partial llf : {choice_fits['C_full']['llf']:.2f}")
    print(f"gain from +replay_frac : {d_llf:.2f}  (1 df)")
    print(f"replay_frac coef       : {choice_fits['C_full']['coef_replay_frac']:.3f}"
          f"  (p = {choice_fits['C_full']['pval_replay_frac']:.3g})")
else:
    print("C_full / C_behavior did not both fit on this epoch.")

# %%
if choice_fits["C_full"] is not None:
    display(coefs_to_frame(choice_fits["C_full"], MODEL_CHOICE["C_full"]).round(3))  # noqa: F821
else:
    print("C_full did not fit on this epoch (need enough valid-next-arm stops).")

# %% [markdown]
# ## 7. Summary
#
# | Family | Estimator | Unit | Asks |
# |---|---|---|---|
# | **M0–M7** | conditional Poisson (9 arms) | event | *Which arm* does replay represent? |
# | **P0–P4** | conditional Poisson (3 patches) | event | *Which patch* does replay represent? |
# | **R0–R3** | Poisson GLM + log-time offset | stop | *How often* do events occur? |
# | **C_\*** | conditional logit | stop | Does replay predict the *next choice*? |
#
# What this one epoch shows is summarized in the AIC tables above: each added
# block either lowers AIC (earns its parameters) or does not. The robust,
# replicated population effects (current-arm bias; SWR broadening off the
# current/previous arm; suppression by cumulative reward) are *aggregates* of
# many such per-session fits.
#
# **Caveat — single-session estimates are noisy.** With one epoch's events,
# coefficients have wide standard errors and some models may skip on
# insufficient data. The population pipeline
# (`scripts/content/fit_replay_content_models.py` →
# `model_comparison.run_full_hierarchy_comparison`) fits this exact per-session
# table across all ~200 sessions and tests effects with Wilcoxon + multiple-
# comparison control. Per the project's own record, several effects that look
# suggestive in a single session do **not** survive autocorrelation /
# multiple-comparison correction — so read this notebook as *one unit of* the
# analysis, not as the result.
