"""Exploratory analysis: reward-location bias in non-SWR replay.

Questions:
1. Does replay represent the current patch, previous patch, or future patch?
2. Does this differ between SWR and non-SWR replay?
3. How does replay content evolve around patch switches?
4. Does reward omission modulate replay content?

Design:
- For each non-local immobile timepoint, classify replay as representing:
  current patch, previous trial's origin patch, next trial's destination patch, or other.
- Compare SWR vs non-SWR throughout.
- Report sample sizes at every step.
- For patch switches (rare), use trial-relative indexing to pool across sessions.
"""
# %% Imports

import pickle
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.constants import (
    NWB_FILES,
    TRACK_SEGMENT_TO_PATCH,
)
from continuum_swr_replay.data_types import SessionDataKeys
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.palettes import COLORS

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")

# %% Helper: map decoded linear position to patch ID


def map_estimate_to_patch(map_estimate, env):
    """Map decoded linear position (cm) to patch ID (1-3).

    Uses the environment's place_bin_centers_nodes_df_ to find the nearest
    place bin for each MAP estimate, then maps its edge_id to a patch via
    TRACK_SEGMENT_TO_PATCH.
    """
    map_est = np.asarray(map_estimate, dtype=float)
    bin_df = env.place_bin_centers_nodes_df_
    linear_positions = np.asarray(bin_df["linear_position"])
    edge_ids = np.asarray(bin_df["edge_id"])

    result = np.full(len(map_est), np.nan)
    valid = ~np.isnan(map_est)

    if valid.any():
        nearest_bin_idx = np.abs(
            linear_positions[np.newaxis, :] - map_est[valid, np.newaxis]
        ).argmin(axis=1)
        nearest_edge = edge_ids[nearest_bin_idx]
        result[valid] = [
            TRACK_SEGMENT_TO_PATCH.get(int(e), np.nan) for e in nearest_edge
        ]

    return result


# %% Helper: enrich data_over_time with trial context


def add_trial_context(dot, trials):
    """Add per-timepoint trial context columns to data_over_time.

    Trial assignment: each timepoint is assigned to the most recent trial
    by end_time via searchsorted. During immobility between trials, the
    "current" trial is the one that just ended.

    Columns added:
    - prev_trial_from_patch: origin patch of the most recent trial (where
      the animal came from — distinct from current location)
    - next_trial_to_patch: destination of the upcoming trial (where the
      animal will go — a "future" reference)
    - last_rewarded_patch: destination of the most recent rewarded trial
    - is_reward_current: whether the most recent trial was rewarded
    - trials_since_switch: number of trials since the last patch change.
      Value 0 = the inter-trial period AFTER the switch trial completed
      (not the switch trajectory itself, which is during movement).
    """
    dot_times = dot.index.values

    # Assign each timepoint to the most recent trial (by end_time)
    trial_end_times = trials["end_time"].values
    trial_idx = np.searchsorted(trial_end_times, dot_times, side="right")
    # trial_idx[t] = index of the trial that just ended before time t
    # So current trial is trial_idx - 1, next trial is trial_idx

    n_trials = len(trials)
    to_patches = trials["to_patch"].values
    from_patches = trials["from_patch"].values
    is_reward = trials["is_reward"].values
    is_patch_change = trials["is_patch_change"].values

    # Previous trial's destination patch (= where animal just arrived)
    prev_idx = np.clip(trial_idx - 1, 0, n_trials - 1)
    dot["prev_trial_to_patch"] = np.where(trial_idx > 0, to_patches[prev_idx], np.nan)

    # Previous trial's origin patch (= where animal came from)
    dot["prev_trial_from_patch"] = np.where(
        trial_idx > 0, from_patches[prev_idx], np.nan
    )

    # Next trial's destination patch (= where animal will go)
    next_idx = np.clip(trial_idx, 0, n_trials - 1)
    dot["next_trial_to_patch"] = np.where(
        trial_idx < n_trials, to_patches[next_idx], np.nan
    )

    # Most recently rewarded patch
    rewarded_mask = is_reward
    rewarded_end_times = trial_end_times[rewarded_mask]
    rewarded_patches = to_patches[rewarded_mask]
    if len(rewarded_end_times) > 0:
        rew_idx = np.searchsorted(rewarded_end_times, dot_times, side="right") - 1
        dot["last_rewarded_patch"] = np.where(
            rew_idx >= 0, rewarded_patches[np.clip(rew_idx, 0, None)], np.nan
        )
    else:
        dot["last_rewarded_patch"] = np.nan

    # Whether the current/most recent trial was rewarded
    dot["is_reward_current"] = np.where(trial_idx > 0, is_reward[prev_idx], np.nan)

    # Trials since last patch change
    switch_trial_indices = np.where(is_patch_change)[0]
    if len(switch_trial_indices) > 0:
        # For each trial, how many trials since last switch
        trials_since = np.full(n_trials, n_trials)  # default: no switch seen
        for si in switch_trial_indices:
            for ti in range(si, n_trials):
                dist = ti - si
                if dist < trials_since[ti]:
                    trials_since[ti] = dist
        dot["trials_since_switch"] = np.where(
            trial_idx > 0, trials_since[prev_idx], np.nan
        )
    else:
        dot["trials_since_switch"] = np.nan

    # Trial index for grouping
    dot["trial_idx"] = trial_idx

    return dot


# %% Load all sessions

dot_files = sorted(PROCESSED_DATA_DIR.glob("*_HPC_80_data_over_time.pkl"))
print(f"Found {len(dot_files)} data_over_time files")

sessions = []
for f in dot_files:
    match = re.match(r"(\w+\d{8})_(\d+_r\d+)_HPC_80_data_over_time\.pkl", f.name)
    if match:
        animal_date, epoch = match.group(1), match.group(2)
        nwb_match = [nwb for nwb in NWB_FILES if animal_date in nwb]
        if nwb_match:
            session_id = f"{animal_date}_{epoch}_HPC"
            sessions.append(
                {
                    "nwb_file_name": nwb_match[0],
                    "epoch_name": epoch,
                    "session_id": session_id,
                    "dot_file": f,
                    "env_file": PROCESSED_DATA_DIR / f"{session_id}_environment.pkl",
                }
            )

print(f"Matched {len(sessions)} sessions")

# %% Helper: compute empirical chance from place bin counts per patch


def compute_chance_per_patch(env):
    """Compute empirical chance level per patch from place bin coverage.

    Returns dict {patch_id: fraction_of_track} rather than assuming 1/3.
    """
    bin_df = env.place_bin_centers_nodes_df_
    edge_ids = bin_df["edge_id"]
    patch_ids = edge_ids.map(TRACK_SEGMENT_TO_PATCH)
    # Exclude junction bins (edge_id == -1 -> NaN after map)
    patch_ids = patch_ids.dropna()
    counts = patch_ids.value_counts(normalize=True).sort_index()
    return counts.to_dict()


# %% Process all sessions

all_session_data = []
session_chance_levels = []  # per-session empirical chance
n_sessions_to_load = len(sessions)  # Change for quick testing

for i, session in enumerate(sessions[:n_sessions_to_load]):
    nwb = session["nwb_file_name"]
    epoch = session["epoch_name"]

    try:
        # Load pre-saved data_over_time and environment (fast)
        dot = pd.read_pickle(session["dot_file"])
        with open(session["env_file"], "rb") as f:
            env = pickle.load(f)

        # Selectively load trials + task_variables from Spyglass
        include = frozenset({SessionDataKeys.TRIALS, SessionDataKeys.TASK_VARIABLES})
        data = load_data(nwb, epoch, include=include)

    except Exception as e:
        print(f"  Skipping {nwb} {epoch}: {e}")
        continue

    trials = data.get("trials")
    if trials is None or len(trials) < 3:
        print(f"  Skipping {nwb} {epoch}: insufficient trials")
        continue

    # Compute empirical chance for this session's track
    chance = compute_chance_per_patch(env)
    session_chance_levels.append(
        {
            "nwb_file_name": nwb,
            "epoch_name": epoch,
            **chance,
        }
    )

    # Merge task_variables
    if "task_variables" in data and data["task_variables"] is not None:
        task_vars = data["task_variables"].reindex(
            dot.index, method="nearest", tolerance=0.004
        )
        for col in task_vars.columns:
            if col not in dot.columns:
                dot[col] = task_vars[col]

    # Map decoded replay position to patch
    dot["replay_patch"] = map_estimate_to_patch(dot["map_estimate"], env)
    dot["Non-Local_prob"] = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    )

    # Add trial context
    dot = add_trial_context(dot, trials)

    # Session metadata (before filtering)
    n_switches = int(trials["is_patch_change"].sum())
    dot["n_trials_session"] = len(trials)
    dot["n_switches_session"] = n_switches

    # FILTER EARLY: keep only non-local immobile timepoints to save memory
    is_immobile = dot["head_speed"] < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    is_nonlocal = dot["Non-Local_prob"] > DEFAULT_NON_LOCAL_THRESHOLD
    dot_filtered = dot[is_immobile & is_nonlocal].copy()

    if len(dot_filtered) < 50:
        continue

    dot_filtered["is_swr"] = dot_filtered["SWR_event"] > 0
    all_session_data.append(dot_filtered)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{n_sessions_to_load} sessions")

print(f"\nLoaded {len(all_session_data)} sessions successfully")

# Empirical chance: verify 1/3 assumption
chance_df = pd.DataFrame(session_chance_levels)
print(f"\nEmpirical chance per patch (fraction of place bins):")
for p in [1.0, 2.0, 3.0]:
    if p in chance_df.columns:
        vals = chance_df[p]
        print(
            f"  Patch {int(p)}: mean={vals.mean():.3f}, range=[{vals.min():.3f}, {vals.max():.3f}]"
        )
# Use grand mean as chance level (per-patch bins are ~equal across sessions)
CHANCE_LEVEL = 1 / 3  # will verify this is approximately correct

# %% Concatenate (already filtered to non-local immobile)

nl = pd.concat(all_session_data, ignore_index=False)
swr = nl[nl["is_swr"]]
noswr = nl[~nl["is_swr"]]

print(f"Non-local immobile: {len(nl):,}")
print(f"  SWR: {len(swr):,} ({len(swr) / len(nl) * 100:.1f}%)")
print(f"  Non-SWR: {len(noswr):,} ({len(noswr) / len(nl) * 100:.1f}%)")
print(f"  Sessions: {nl.groupby(['animal', 'day', 'epoch']).ngroups}")

# %% Analysis 1: What patch does replay represent?
#
# For each non-local timepoint, classify replay_patch relative to:
# - patch_id: where the animal is sitting (current patch)
# - prev_trial_from_patch: where the animal departed from on the last trial
#   (distinct from current patch — this is the origin, not destination)
# - next_trial_to_patch: where the animal will go on the next trial
# - last_rewarded_patch: destination of the most recent rewarded trial
#
# Note: during immobility at well B after arriving from well A,
# patch_id = patch of well B, prev_trial_from_patch = patch of well A.
# These differ only when the preceding trial crossed patches.

MIN_EVENTS_PER_SESSION = 50  # minimum replay timepoints for stable fractions
session_cols = ["animal", "day", "epoch"]


def compute_replay_fractions(subset, label):
    """Compute fraction of replay at each reference patch, per session.

    Sessions with fewer than MIN_EVENTS_PER_SESSION valid replay timepoints
    are excluded to ensure stable proportion estimates.
    """
    rows = []
    for keys, group in subset.groupby(session_cols):
        rp = group["replay_patch"]
        valid = rp.notna()
        if valid.sum() < MIN_EVENTS_PER_SESSION:
            continue

        row = dict(zip(session_cols, keys, strict=True))
        row["n_timepoints"] = int(valid.sum())
        row["event_type"] = label

        # Fraction at each reference
        for ref_col, ref_name in [
            ("patch_id", "Current Patch"),
            ("prev_trial_from_patch", "Previous Origin"),
            ("next_trial_to_patch", "Future Destination"),
            ("last_rewarded_patch", "Last Rewarded"),
        ]:
            if ref_col in group.columns:
                ref = group[ref_col]
                both_valid = valid & ref.notna()
                if both_valid.sum() >= MIN_EVENTS_PER_SESSION:
                    row[ref_name] = (rp[both_valid] == ref[both_valid]).mean()
                    row[f"{ref_name}_n"] = int(both_valid.sum())
                else:
                    row[ref_name] = np.nan
                    row[f"{ref_name}_n"] = 0
            else:
                row[ref_name] = np.nan
                row[f"{ref_name}_n"] = 0

        rows.append(row)
    return pd.DataFrame(rows)


swr_fracs = compute_replay_fractions(swr, "SWR")
noswr_fracs = compute_replay_fractions(noswr, "Non-SWR")
all_fracs = pd.concat([swr_fracs, noswr_fracs], ignore_index=True)

print(f"\nSessions with SWR data: {len(swr_fracs)}")
print(f"Sessions with Non-SWR data: {len(noswr_fracs)}")

# %% Plot Analysis 1: replay at each reference patch

ref_names = ["Current Patch", "Previous Origin", "Future Destination", "Last Rewarded"]
fig, axes = plt.subplots(1, 4, figsize=(7, 2.5), constrained_layout=True, sharey=True)

for ax, ref in zip(axes, ref_names, strict=True):
    plot_data = all_fracs[["event_type", ref]].dropna()

    if len(plot_data) == 0:
        ax.set_title(ref, fontsize=7)
        continue

    sns.boxplot(
        data=plot_data,
        x="event_type",
        y=ref,
        palette={"SWR": COLORS["SWR"], "Non-SWR": COLORS["Non-Local"]},
        ax=ax,
        width=0.5,
        fliersize=2,
    )
    sns.stripplot(
        data=plot_data,
        x="event_type",
        y=ref,
        palette={"SWR": COLORS["SWR"], "Non-SWR": COLORS["Non-Local"]},
        ax=ax,
        size=2,
        alpha=0.3,
        jitter=True,
    )
    ax.axhline(CHANCE_LEVEL, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(ref, fontsize=7)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)

    # Stats
    swr_vals = all_fracs.loc[all_fracs["event_type"] == "SWR", ref].dropna()
    noswr_vals = all_fracs.loc[all_fracs["event_type"] == "Non-SWR", ref].dropna()
    if len(swr_vals) >= 5 and len(noswr_vals) >= 5:
        _, p = mannwhitneyu(swr_vals, noswr_vals, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(
            0.5,
            0.95,
            f"p={p:.1e} {sig}",
            transform=ax.transAxes,
            ha="center",
            fontsize=5,
            va="top",
        )

axes[0].set_ylabel("Fraction of Replay")
sns.despine()
plt.savefig("analysis1_replay_reference_patches.png", dpi=300)
plt.show()

# Print stats table (p-values are uncorrected for multiple comparisons)
print("\n=== Analysis 1: Replay at reference patches (p-values uncorrected) ===")
print(
    f"{'Reference':<25} {'SWR median':>12} {'Non-SWR median':>15} {'p-value':>10} {'n_SWR':>6} {'n_noSWR':>8}"
)
print("-" * 80)
for ref in ref_names:
    swr_vals = all_fracs.loc[all_fracs["event_type"] == "SWR", ref].dropna()
    noswr_vals = all_fracs.loc[all_fracs["event_type"] == "Non-SWR", ref].dropna()
    if len(swr_vals) >= 5 and len(noswr_vals) >= 5:
        _, p = mannwhitneyu(swr_vals, noswr_vals, alternative="two-sided")
        print(
            f"{ref:<25} {swr_vals.median():>12.3f} {noswr_vals.median():>15.3f} {p:>10.2e} {len(swr_vals):>6} {len(noswr_vals):>8}"
        )
    else:
        print(f"{ref:<25} {'insufficient data':>40}")

# %% Analysis 2: Replay at rewarded patch — above chance?
#
# Test each event type against chance (1/3) with Wilcoxon signed-rank.

print("\n=== Analysis 2: Above chance? ===")
for ref in ref_names:
    for etype in ["SWR", "Non-SWR"]:
        vals = all_fracs.loc[all_fracs["event_type"] == etype, ref].dropna()
        if len(vals) >= 5:
            stat, p = wilcoxon(vals - CHANCE_LEVEL)
            direction = "above" if vals.median() > 1 / 3 else "below"
            print(
                f"  {etype:>7} {ref:<25}: median={vals.median():.3f}, {direction} chance, "
                f"Wilcoxon p={p:.2e}, n={len(vals)}"
            )

# %% Analysis 3: Replay content around patch switches
#
# For each patch switch event, look at replay content in trials before and
# after the switch. Use trial-relative indexing to pool across sessions/switches.
# Focus on: does replay shift from old patch to new patch?

switch_data = []
n_sessions_skipped_switch = 0

for keys, session_group in nl.groupby(session_cols):
    animal, day, epoch = keys

    # Identify patch changes from task variables if available
    if "is_patch_change0" not in session_group.columns:
        n_sessions_skipped_switch += 1
        continue

    is_switch = session_group["is_patch_change0"].astype(bool)
    if is_switch.sum() == 0:
        continue

    # For timepoints during switch trials, record trial-relative info
    for swr_label, swr_mask in [
        ("SWR", session_group["is_swr"]),
        ("Non-SWR", ~session_group["is_swr"]),
    ]:
        subset = session_group[swr_mask]
        if len(subset) < 10:
            continue

        # Classify replay relative to animal's current vs previous patch
        rp = subset["replay_patch"]
        current = subset["patch_id"]
        prev_origin = subset["prev_trial_from_patch"]

        valid = rp.notna() & current.notna()
        if valid.sum() < 5:
            continue

        # Bin by trials_since_switch
        tss = subset.loc[valid, "trials_since_switch"]
        for trial_offset in range(0, 6):
            mask = tss == trial_offset
            if mask.sum() < 5:
                continue

            rp_sub = rp[valid][mask]
            cur_sub = current[valid][mask]

            switch_data.append(
                {
                    "animal": animal,
                    "day": day,
                    "epoch": epoch,
                    "event_type": swr_label,
                    "trials_since_switch": trial_offset,
                    "frac_at_current": (rp_sub == cur_sub).mean(),
                    "n_timepoints": int(mask.sum()),
                }
            )

switch_df = pd.DataFrame(switch_data)
print(f"\n=== Analysis 3: Replay around patch switches ===")
print(f"Total rows: {len(switch_df)}")
if len(switch_df) > 0:
    print(f"Sessions with switch data: {switch_df.groupby(session_cols).ngroups}")
print(f"Sessions skipped (no task_variables): {n_sessions_skipped_switch}")
print(
    "Note: trials_since_switch=0 is the inter-trial period AFTER the switch trial completed."
)

# %% Plot Analysis 3: replay at current patch as function of trials since switch

if len(switch_df) > 0:
    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)

    for etype, color in [("SWR", COLORS["SWR"]), ("Non-SWR", COLORS["Non-Local"])]:
        sub = switch_df[switch_df["event_type"] == etype]
        if len(sub) == 0:
            continue

        # Aggregate: median and IQR across sessions per trial offset
        agg = sub.groupby("trials_since_switch")["frac_at_current"].agg(
            [
                "median",
                "count",
                lambda x: np.nanpercentile(x, 25),
                lambda x: np.nanpercentile(x, 75),
            ]
        )
        agg.columns = ["median", "count", "q25", "q75"]
        agg = agg.reset_index()

        ax.fill_between(
            agg["trials_since_switch"],
            agg["q25"],
            agg["q75"],
            color=color,
            alpha=0.2,
        )
        ax.plot(
            agg["trials_since_switch"],
            agg["median"],
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label=etype,
        )

        # Annotate sample sizes
        for _, row in agg.iterrows():
            ax.text(
                row["trials_since_switch"],
                row["q75"] + 0.02,
                f"n={int(row['count'])}",
                fontsize=4,
                ha="center",
                color=color,
            )

    ax.axhline(1 / 3, color="black", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Trials Since Patch Switch")
    ax.set_ylabel("Fraction at Current Patch")
    ax.set_title("Replay Shifts to New Patch After Switch", fontsize=7)
    ax.legend(fontsize=5, frameon=False)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(6))
    sns.despine(ax=ax)
    plt.savefig("analysis3_replay_around_switch.png", dpi=300)
    plt.show()

    # Print sample sizes
    print("\nSample sizes by trials_since_switch:")
    pivot = switch_df.pivot_table(
        index="trials_since_switch",
        columns="event_type",
        values="n_timepoints",
        aggfunc=["count", "sum"],
    )
    print(pivot)

# %% Analysis 4: Effect of reward omission on replay content
#
# After a reward omission, does non-SWR replay still represent the
# previously rewarded patch, or does it shift?

if "is_no_reward_1" in nl.columns:
    omission_data = []

    for etype, subset in [("SWR", swr), ("Non-SWR", noswr)]:
        for keys, group in subset.groupby(session_cols):
            rp = group["replay_patch"]
            last_rew = group["last_rewarded_patch"]
            is_omission = group.get("is_no_reward_1")

            if is_omission is None:
                continue

            is_omission = is_omission.astype(bool)
            valid = rp.notna() & last_rew.notna() & is_omission.notna()
            if valid.sum() < 10:
                continue

            for cond_label, cond_mask in [
                ("After Reward", ~is_omission[valid]),
                ("After Omission", is_omission[valid]),
            ]:
                if cond_mask.sum() < 5:
                    continue

                rp_sub = rp[valid][cond_mask]
                lr_sub = last_rew[valid][cond_mask]

                omission_data.append(
                    {
                        **dict(zip(session_cols, keys, strict=True)),
                        "event_type": etype,
                        "condition": cond_label,
                        "frac_at_rewarded": (rp_sub == lr_sub).mean(),
                        "n_timepoints": int(cond_mask.sum()),
                    }
                )

    omission_df = pd.DataFrame(omission_data)
    print(f"\n=== Analysis 4: Reward omission effect ===")
    print(f"Total rows: {len(omission_df)}")

    if len(omission_df) > 0:
        fig, axes = plt.subplots(
            1, 2, figsize=(5.5, 2.5), constrained_layout=True, sharey=True
        )

        for ax, etype in zip(axes, ["SWR", "Non-SWR"], strict=True):
            sub = omission_df[omission_df["event_type"] == etype]
            if len(sub) == 0:
                ax.set_title(etype, fontsize=7)
                continue

            palette = {"After Reward": COLORS["Non-Local"], "After Omission": "#CC79A7"}
            sns.boxplot(
                data=sub,
                x="condition",
                y="frac_at_rewarded",
                palette=palette,
                ax=ax,
                width=0.5,
                fliersize=2,
                order=["After Reward", "After Omission"],
            )
            sns.stripplot(
                data=sub,
                x="condition",
                y="frac_at_rewarded",
                palette=palette,
                ax=ax,
                size=2,
                alpha=0.3,
                jitter=True,
                order=["After Reward", "After Omission"],
            )
            ax.axhline(1 / 3, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(etype, fontsize=7)
            ax.set_xlabel("")
            ax.set_ylim(0, 1)

            # Sample sizes
            for cond in ["After Reward", "After Omission"]:
                n = len(sub[sub["condition"] == cond])
                ax.text(
                    0 if cond == "After Reward" else 1,
                    -0.05,
                    f"n={n}",
                    fontsize=5,
                    ha="center",
                    transform=ax.get_xaxis_transform(),
                )

            # Stats
            rew_vals = sub.loc[sub["condition"] == "After Reward", "frac_at_rewarded"]
            omit_vals = sub.loc[
                sub["condition"] == "After Omission", "frac_at_rewarded"
            ]
            if len(rew_vals) >= 5 and len(omit_vals) >= 5:
                _, p = mannwhitneyu(rew_vals, omit_vals, alternative="two-sided")
                ax.text(
                    0.5,
                    0.95,
                    f"p={p:.2e}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=5,
                    va="top",
                )

        axes[0].set_ylabel("Fraction at Rewarded Patch")
        sns.despine()
        plt.savefig("analysis4_omission_effect.png", dpi=300)
        plt.show()

        # Print stats
        print("\nReward omission effect:")
        for etype in ["SWR", "Non-SWR"]:
            for cond in ["After Reward", "After Omission"]:
                vals = omission_df.loc[
                    (omission_df["event_type"] == etype)
                    & (omission_df["condition"] == cond),
                    "frac_at_rewarded",
                ]
                if len(vals) > 0:
                    print(
                        f"  {etype:>7} {cond:<20}: median={vals.median():.3f}, n={len(vals)} sessions"
                    )
else:
    print("is_no_reward_1 not available")

# %% Analysis 5: Comprehensive sample size report

print("\n" + "=" * 80)
print("SAMPLE SIZE REPORT")
print("=" * 80)

# Per-animal summary
animal_summary = (
    nl.groupby("animal")
    .agg(
        n_sessions=("epoch", "nunique"),
        n_timepoints=("replay_patch", "count"),
        n_swr=("is_swr", "sum"),
    )
    .reset_index()
)
animal_summary["n_noswr"] = animal_summary["n_timepoints"] - animal_summary["n_swr"]
print("\nPer-animal non-local immobile timepoints:")
print(animal_summary.to_string(index=False))

# Switch events
if "n_switches_session" in nl.columns:
    switch_summary = nl.groupby(session_cols)["n_switches_session"].first()
    print(
        f"\nPatch switches per session: median={switch_summary.median():.0f}, "
        f"range=[{switch_summary.min():.0f}, {switch_summary.max():.0f}], "
        f"total={switch_summary.sum():.0f}"
    )

# Reward omission prevalence
if "is_no_reward_1" in nl.columns:
    omission_rate = nl["is_no_reward_1"].astype(bool).mean()
    print(f"Timepoints after omission: {omission_rate * 100:.1f}%")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
If non-SWR replay is functionally reward-focused:
1. Non-SWR should show HIGHER fraction at 'Last Rewarded' than SWR
2. Non-SWR should show LOWER fraction at 'Current Patch' (it's not just local leakage)
3. After patch switch, replay should shift to new patch faster for non-SWR
4. After reward omission, bias toward rewarded patch should decrease
   (especially for non-SWR if it tracks current value)

Alternative interpretations to consider:
- High 'Current Patch' could indicate local processing leakage, not replay
- 'Last Rewarded' may correlate with 'Current Patch' if animal stays at reward
- 'Future Destination' bias could reflect planning, not reward memory
""")
