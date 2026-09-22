#!/usr/bin/env python
"""Visualize the anatomy of a replay event with all indicators/covariates.

Shows a window of events with the long-format predictor structure,
illustrating how the conditional Poisson model sees the data.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

INPUT_DIR = PROCESSED_DATA_DIR / "replay_content"

# ── Load data ──
long = pd.read_pickle(INPUT_DIR / "long.pkl")
events = pd.read_pickle(INPUT_DIR / "events.pkl")

# Pick session with most data
a, d, e = "chimi", 20200213, "06_r3"
ev = events[(events["animal"] == a) & (events["day"] == d) & (events["epoch"] == e)]
lg = long[(long["animal"] == a) & (long["day"] == d) & (long["epoch"] == e)]

# Find a window around an SWR event with good context
good = ev[ev.prev_trial_from_well.notna() & ev.last_rewarded_well.notna()]
swr_events = good[good.is_swr_flag == 1.0].non_local_event.values
target = swr_events[10]
window_events = ev[
    (ev.non_local_event >= target - 2) & (ev.non_local_event <= target + 2)
]

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())
PATCH_COLORS = {1: "#4e79a7", 2: "#e15759", 3: "#59a14f"}
ARM_PATCH = TRACK_SEGMENT_TO_PATCH

fig, axes = plt.subplots(
    3, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 2, 2]}
)

# ═══════════════════════════════════════════════════════════
# Panel A: Event × Arm matrix with indicators
# ═══════════════════════════════════════════════════════════
ax = axes[0]
event_ids = window_events.non_local_event.values
n_events = len(event_ids)

for col_idx, eid in enumerate(event_ids):
    erow = window_events[window_events.non_local_event == eid].iloc[0]
    elong = lg[lg.non_local_event == eid]

    for row_idx, arm in enumerate(ALL_ARMS):
        arm_row = elong[elong.arm_id == arm]
        if len(arm_row) == 0:
            continue
        arm_row = arm_row.iloc[0]

        x = col_idx
        y = row_idx

        # Background: patch color
        patch_id = ARM_PATCH[arm]
        bg_color = PATCH_COLORS[patch_id]
        rect = mpatches.FancyBboxPatch(
            (x - 0.42, y - 0.42),
            0.84,
            0.84,
            boxstyle="round,pad=0.02",
            facecolor=bg_color,
            alpha=0.12,
            edgecolor="none",
        )
        ax.add_patch(rect)

        # Replay target (binary response)
        if arm_row.replay_here == 1:
            ax.plot(x, y, "s", color="black", markersize=22, zorder=3)
            ax.plot(x, y, "s", color="gold", markersize=18, zorder=4)
            ax.text(
                x,
                y,
                "R",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                zorder=5,
            )
        else:
            # is_current
            if arm_row.is_current == 1:
                ax.plot(
                    x,
                    y,
                    "o",
                    color="#1b9e77",
                    markersize=16,
                    zorder=3,
                    markeredgewidth=2.5,
                    markerfacecolor="none",
                )
            # is_prev
            if arm_row.is_prev == 1:
                ax.plot(
                    x,
                    y,
                    "^",
                    color="#7570b3",
                    markersize=10,
                    zorder=3,
                    markeredgewidth=1.5,
                    markerfacecolor="none",
                )
            # is_next
            if arm_row.is_next == 1:
                ax.plot(
                    x,
                    y,
                    "v",
                    color="#d95f02",
                    markersize=10,
                    zorder=3,
                    markeredgewidth=1.5,
                    markerfacecolor="none",
                )
            # is_last_rewarded
            if arm_row.is_last_rewarded == 1:
                ax.plot(x, y, "*", color="#e7298a", markersize=14, zorder=3)

    # Event label
    swr_str = "SWR" if erow.is_swr_flag else "non-SWR"
    tss = erow.time_since_stop
    tss_str = f"{tss:.1f}s" if pd.notna(tss) else "?"
    ax.text(
        x, -1.0, f"E{eid}\n{swr_str}\nt={tss_str}", ha="center", va="top", fontsize=8
    )

ax.set_xlim(-0.6, n_events - 0.4)
ax.set_ylim(-1.8, len(ALL_ARMS) - 0.4)
ax.set_yticks(range(len(ALL_ARMS)))
arm_labels = [f"arm {a} (P{ARM_PATCH[a]})" for a in ALL_ARMS]
ax.set_yticklabels(arm_labels, fontsize=9)
ax.set_xticks([])
ax.set_title(
    f"Panel A: Event × Arm indicator matrix — {a} {d} {e}",
    fontsize=12,
    fontweight="bold",
)
ax.invert_yaxis()

# Legend
legend_elements = [
    mpatches.Patch(
        facecolor="gold", edgecolor="black", label="replay_here = 1 (modal arm)"
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markeredgecolor="#1b9e77",
        markeredgewidth=2.5,
        markersize=10,
        markerfacecolor="none",
        label="is_current_arm",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markeredgecolor="#7570b3",
        markeredgewidth=1.5,
        markersize=8,
        markerfacecolor="none",
        label="is_prev_arm",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="v",
        color="w",
        markeredgecolor="#d95f02",
        markeredgewidth=1.5,
        markersize=8,
        markerfacecolor="none",
        label="is_next_arm",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="*",
        color="#e7298a",
        markersize=10,
        linestyle="none",
        label="is_last_rewarded_arm",
    ),
]
for pid, col in PATCH_COLORS.items():
    legend_elements.append(
        mpatches.Patch(facecolor=col, alpha=0.25, label=f"Patch {pid}")
    )
ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)

# ═══════════════════════════════════════════════════════════
# Panel B: Per-arm continuous covariates for one event
# ═══════════════════════════════════════════════════════════
ax = axes[1]
focus_eid = target
focus_long = lg[lg.non_local_event == focus_eid].sort_values("arm_id")
focus_ev = window_events[window_events.non_local_event == focus_eid].iloc[0]

bar_width = 0.35
x_pos = np.arange(len(ALL_ARMS))

# visit_recency (cap at 50 for display)
vr = focus_long["visit_recency_trials"].clip(upper=50).values
cr = focus_long["reward_count_candidate_arm_last10_known_trials"].values

bars1 = ax.bar(
    x_pos - bar_width / 2,
    vr,
    bar_width,
    color="#4e79a7",
    alpha=0.7,
    label="visit_recency (trials ago, capped at 50)",
)
bars2 = ax.bar(
    x_pos + bar_width / 2,
    cr,
    bar_width,
    color="#e15759",
    alpha=0.7,
    label="cumulative_reward (last 10 trials)",
)

# Mark current arm
current_arm = (
    int(focus_ev.track_segment_id) if pd.notna(focus_ev.track_segment_id) else -1
)
if current_arm in ALL_ARMS:
    idx = ALL_ARMS.index(current_arm)
    ax.axvline(
        idx,
        color="#1b9e77",
        linewidth=2,
        linestyle="--",
        alpha=0.6,
        label="current arm",
    )

# Mark replay arm
replay_arm = int(focus_ev.replay_arm) if pd.notna(focus_ev.replay_arm) else -1
if replay_arm in ALL_ARMS:
    idx = ALL_ARMS.index(replay_arm)
    ax.axvline(
        idx, color="gold", linewidth=2, linestyle="-", alpha=0.8, label="replay arm"
    )

ax.set_xticks(x_pos)
ax.set_xticklabels([f"arm {a}" for a in ALL_ARMS], fontsize=9)
ax.set_ylabel("Value", fontsize=10)
swr_label = "SWR" if focus_ev.is_swr_flag else "non-SWR"
ax.set_title(
    f"Panel B: Per-arm covariates for event {focus_eid} "
    f"({swr_label}, t_stop={focus_ev.time_since_stop:.1f}s)",
    fontsize=12,
    fontweight="bold",
)
ax.legend(fontsize=8, loc="upper right")

# ═══════════════════════════════════════════════════════════
# Panel C: Time course of event-level variables
# ═══════════════════════════════════════════════════════════
ax = axes[2]

# Get a longer window of events
long_window = ev[
    (ev.non_local_event >= target - 15) & (ev.non_local_event <= target + 15)
].copy()
long_window = long_window[long_window.time_since_stop.notna()]

x = long_window.non_local_event.values
tss = long_window.time_since_stop.values
is_swr = long_window.is_swr_flag.values.astype(bool)

# Plot time_since_stop
ax.plot(
    x[~is_swr],
    tss[~is_swr],
    "o",
    color="#4e79a7",
    markersize=5,
    alpha=0.7,
    label="non-SWR events",
)
ax.plot(
    x[is_swr],
    tss[is_swr],
    "D",
    color="#e15759",
    markersize=5,
    alpha=0.7,
    label="SWR events",
)

# Highlight our focus window
ax.axvspan(target - 2.5, target + 2.5, alpha=0.1, color="gold", label="Panel A window")
ax.axvline(target, color="gold", linewidth=1.5, linestyle="--", alpha=0.6)

# Mark switch and omission events
if "did_last_completed_trip_switch_patch" in long_window.columns:
    switches = long_window[long_window.is_after_switch == 1.0]
    if len(switches) > 0:
        ax.plot(
            switches.non_local_event.values,
            switches.time_since_stop.values,
            "s",
            color="purple",
            markersize=8,
            markerfacecolor="none",
            markeredgewidth=2,
            label="after patch switch",
            zorder=5,
        )

ax.set_xlabel("Event number", fontsize=10)
ax.set_ylabel("time_since_stop (s)", fontsize=10)
ax.set_title(
    "Panel C: Event-level covariates across time", fontsize=12, fontweight="bold"
)
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(
    "/cumulus/edeno/continuum-swr-replay/event_anatomy.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved event_anatomy.png")
