#!/usr/bin/env python
"""Diagnostic figures for trial-detection and predictor semantics.

Two figures:

1. **trial_detection_sanity.{png,pdf}** — for one session, plot a window
   of the animal's track-segment trace overlaid with the detected trip
   start/end markers and from→to labels. If detection is correct, you
   should see clean point-to-point sweeps. If it's over-segmenting brief
   excursions or mis-detecting boundaries, you'll see it visually.

2. **predictor_collapse.{png,pdf}** — across all events in the dataset,
   for each event compute which arm each binary predictor marks
   (``is_current``, ``is_prev``, ``is_next``, ``is_last_rewarded``) and
   stack them as a heatmap. If predictors collapse (all mark the same
   arm), the heatmap rows look identical. The bar chart on the side
   quantifies the % of events where each predictor matches is_current.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_types import SessionDataKeys
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import nwb_file_name
from continuum_swr_replay.paths import processed_data_dir, replay_content_dir

ARM_COLORS = {
    0: "#1f77b4",
    1: "#5ba3d8",
    2: "#2ca02c",
    3: "#7fc97f",
    4: "#d62728",
    5: "#e87a7a",
    6: "#5a5a5a",
    7: "#7a7a7a",
    8: "#9a9a9a",
}


def figure_trial_sanity(
    *,
    animal: str,
    day: int,
    epoch: str,
    window_start: float | None,
    window_seconds: float,
    output_dir: Path,
) -> None:
    """Plot a window of position + speed + trip boundaries for one session."""
    base = swr_epoch_id_of_session(animal, day, epoch)
    dot = pd.read_pickle(decode_paths(base, processed_data_dir()).data_over_time)
    nwb = nwb_file_name(animal, day)
    data = load_data(nwb, epoch, include=frozenset({SessionDataKeys.TRIALS}))
    trials = data["trials"].sort_values("movement_start_time").reset_index(drop=True)

    if window_start is None:
        # Pick a window with a few trips — start near the median of the trips
        # to dodge any startup junk.
        window_start = float(trials["movement_start_time"].iloc[10])

    win = dot.loc[window_start : window_start + window_seconds]
    win_t = win.index.values - window_start
    trips_in_win = trials[
        (trials["movement_start_time"] < window_start + window_seconds)
        & (trials["end_time"] > window_start)
    ].copy()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.5, 0.7, 0.6]},
    )

    # (1) track_segment_id over time
    ax = axes[0]
    arms = win["track_segment_id"].values
    valid = ~np.isnan(arms)
    colors = [ARM_COLORS.get(int(a), "lightgrey") for a in arms[valid]]
    ax.scatter(win_t[valid], arms[valid], c=colors, s=8, marker="s")
    ax.set_yticks(range(9))
    ax.set_yticklabels([f"arm {a}" for a in range(9)])
    ax.set_ylabel("Animal position (track segment)")
    ax.invert_yaxis()
    # Overlay trip start/end markers
    for _, t in trips_in_win.iterrows():
        s = t["movement_start_time"] - window_start
        e = t["end_time"] - window_start
        from_arm = int(t["from_well"])  # leaf arm = well id
        to_arm = int(t["to_well"])
        ax.axvline(s, color="black", lw=0.6, alpha=0.6, ls="-")
        ax.axvline(e, color="red", lw=0.6, alpha=0.4, ls="--")
        # Label trip in the gap
        midt = (s + e) / 2
        ax.annotate(
            f"{from_arm}→{to_arm}",
            xy=(midt, -0.6),
            ha="center",
            va="top",
            fontsize=8,
            color="#222",
            annotation_clip=False,
        )
    ax.set_title(
        f"Trial-detection sanity: {animal} {day} {epoch}  "
        f"({window_seconds:.0f}-second window starting at trip ~10)\n"
        "black solid = trip start, red dashed = trip end, label = from→to_well"
    )

    # (2) head_speed
    ax = axes[1]
    ax.plot(win_t, win["head_speed"].values, color="#444", lw=1)
    ax.axhline(
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, color="#a55a3b", ls="--", lw=0.8
    )
    ax.set_ylabel("speed (cm/s)")
    ax.set_ylim(0, max(40, np.nanpercentile(win["head_speed"].values, 95)))

    # (3) Trip-membership label per timebin (which trip is in-progress?)
    ax = axes[2]
    # For each trip in window, draw a horizontal bar colored by the trip's to_well arm
    for _i, (_, t) in enumerate(trips_in_win.iterrows()):
        s = max(0, t["movement_start_time"] - window_start)
        e = min(window_seconds, t["end_time"] - window_start)
        to_arm = int(t["to_well"])
        ax.barh(
            0,
            e - s,
            left=s,
            height=0.5,
            color=ARM_COLORS[to_arm],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.text(
            (s + e) / 2,
            0,
            f"trip→{to_arm}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if to_arm < 6 else "black",
        )
    ax.set_yticks([])
    ax.set_ylabel("In-progress\ntrip")
    ax.set_xlabel("time relative to window start (s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"trial_detection_sanity.{ext}",
            dpi=200 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def figure_predictor_collapse(*, output_dir: Path) -> None:
    """Across all events, quantify how often each predictor collapses with is_current."""
    ev = pd.read_pickle(replay_content_dir() / "events.pkl")
    ev = ev.copy()
    ev["cur"] = ev["track_segment_id"]
    ev["prev"] = ev["prev_trial_from_well"]
    ev["prev_to"] = ev["prev_trial_to_well"]
    ev["next"] = ev["next_trial_to_well"]
    ev["rewarded"] = ev["last_rewarded_well"]
    n = len(ev)

    pcts = {
        "is_prev\n(prev_trial_from_well)": (ev["prev"] == ev["cur"]).mean() * 100,
        "(prev_trial_to_well)\n[not in model]": (ev["prev_to"] == ev["cur"]).mean()
        * 100,
        "is_next\n(next_trial_to_well)": (ev["next"] == ev["cur"]).mean() * 100,
        "is_last_rewarded\n(last_rewarded_well)": (ev["rewarded"] == ev["cur"]).mean()
        * 100,
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # (A) Bar chart: % of events where each context column == current arm
    ax = axes[0]
    names = list(pcts.keys())
    vals = list(pcts.values())
    bars = ax.barh(
        np.arange(len(names))[::-1],
        vals,
        color=["#d62728" if v > 50 else "#3b6ea5" for v in vals],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(np.arange(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("% of events where this predictor marks the same arm as is_current")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="grey", lw=0.5, ls="--")
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9
        )
    ax.set_title(
        f"(A) Predictor collapse with is_current  (n = {n:,} events)\n"
        "red = >50% redundant; columns named in the model on the left"
    )

    # (B) Joint distribution: how often does each combination occur?
    # Categorize each event by the count of context columns equal to is_current.
    ev["n_collapse"] = (
        (ev["prev"] == ev["cur"]).astype(int)
        + (ev["next"] == ev["cur"]).astype(int)
        + (ev["rewarded"] == ev["cur"]).astype(int)
    )
    ax = axes[1]
    dist = ev["n_collapse"].value_counts().sort_index()
    bars = ax.bar(dist.index, dist.values, color="#3b6ea5", edgecolor="black")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_xlabel(
        "Number of context predictors collapsing with is_current\n"
        "(out of: is_prev, is_next, is_last_rewarded)"
    )
    ax.set_ylabel("Number of events")
    for b, v in zip(bars, dist.values, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:,}\n({v / n * 100:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(
        f"(B) Per-event count of predictors marking the current arm\n"
        f"events with all 3 predictors collapsing: "
        f"{(ev['n_collapse'] == 3).sum():,} ({(ev['n_collapse'] == 3).mean() * 100:.0f}%)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"predictor_collapse.{ext}",
            dpi=200 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--animal", default="wilbur")
    p.add_argument("--day", type=int, default=20210402)
    p.add_argument("--epoch", default="02_r1")
    p.add_argument("--window-seconds", type=float, default=60.0)
    p.add_argument("--window-start", type=float, default=None)
    p.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/qc_diagnosis")
    )
    args = p.parse_args(argv)

    figure_trial_sanity(
        animal=args.animal,
        day=args.day,
        epoch=args.epoch,
        window_start=args.window_start,
        window_seconds=args.window_seconds,
        output_dir=args.output_dir,
    )
    figure_predictor_collapse(output_dir=args.output_dir)
    print(
        f"Saved diagnostic figures to {args.output_dir}/{{trial_detection_sanity,predictor_collapse}}.{{png,pdf}}"
    )


if __name__ == "__main__":
    main()
