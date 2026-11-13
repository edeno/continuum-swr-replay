#!/usr/bin/env python
"""Visualize the replay-content GLM pipeline as a data-flow figure.

Follows ONE real event through every transformation step, so the reader
can see the actual data shape changing at each stage:

    raw decoder timeseries
        ↓ filter to non-local immobile  +  event detection
    per-timebin replay_arm sequence  (this event only)
        ↓ collapse to event  +  per-arm counts
    9-element count vector  (the multinomial sufficient statistic)
        ↓ long-format expansion  +  add indicators
    9-row × predictor design matrix  (this event)
        ↓ pool across all sessions  +  fit ConditionalPoisson
    population coefficient distribution

Output:
- ``paper/figures/qc_diagnosis/pipeline_data_flow.{png,pdf}``
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.coefficient_statistics import (
    gated_coefficient_statistics,
)
from continuum_swr_replay.analysis.behavior.contrast_aggregation import (
    gate_term_sessions,
)
from continuum_swr_replay.analysis.behavior.fit_provenance import (
    require_fresh_inputs,
)
from continuum_swr_replay.analysis.behavior.spatial_mapping import (
    map_estimate_to_arm,
)
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    read_dot_with_swr_source,
    require_swr_source,
)
from continuum_swr_replay.analysis.events.swr_source import swr_epoch_id_of_session
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.results import nonlocal_probability  # noqa: E402
from continuum_swr_replay.paths import (
    figure_dir,
    model_dir,
    processed_data_dir,
    table_path,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table


def m3_path() -> Path:
    """The fitted M3 arm model the panel draws."""
    return model_dir("content_arm") / "M3.pkl"


def all_models_input_paths() -> dict[str, Path]:
    """The table the content ladder recorded when it fit M3, which gates the panel."""
    return {"event_arm_table": table_path("event_arm_table")}


# The event's long-format rows, as panel E shows them.
INDICATOR_COLUMNS = [
    "candidate_arm",
    "arm_count_hard",
    "is_current_arm",
    "is_prev_arm",
    "is_next_arm",
    "is_last_rewarded_arm",
]
# Panel E's header lines are wrapped at underscores to this many characters, so
# each name fits its column (six columns across half of the 13-inch figure).
HEADER_WIDTH = 12


def wrap_name(name: str, width: int = HEADER_WIDTH) -> str:
    """`name` on lines of at most `width` characters, each break after an
    underscore, which stays at the end of its line: the lines joined give the
    name back."""
    lines: list[str] = []
    line = ""
    for part in name.split("_"):
        candidate = f"{line}_{part}" if line else part
        if line and len(candidate) > width:
            lines.append(f"{line}_")
            line = part
        else:
            line = candidate
    lines.append(line)
    return "\n".join(lines)


# Colour code for each arm (0..8). Patches share families:
# arms 0-1 = patch 1 (blues), 2-3 = patch 2 (greens), 4-5 = patch 3 (reds),
# 6-8 = trunks (greys).
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


def load_session(animal: str, day: int, epoch: str):
    paths = decode_paths(
        swr_epoch_id_of_session(animal, day, epoch), processed_data_dir()
    )
    dot_path = paths.data_over_time
    env_path = paths.environment
    # Through the SWR source: its event labels, and the excluded-time column.
    dot = read_dot_with_swr_source(dot_path)
    with open(env_path, "rb") as f:
        env = pickle.load(f)
    dot["replay_arm"] = map_estimate_to_arm(dot["map_estimate"].values, env)
    dot["Non-Local_prob"] = nonlocal_probability(dot)
    return dot, env


def indicator_table(lc_event: pd.DataFrame, event_id: int) -> pd.DataFrame:
    """The event's long-format rows as integers, for panel E.

    Parameters
    ----------
    lc_event : pd.DataFrame
        The event's ``event_arm_table`` rows, one per candidate arm.
    event_id : int
        The session-local ``non_local_event`` label, for the error message.

    Returns
    -------
    pd.DataFrame
        :data:`INDICATOR_COLUMNS`, as int.

    Raises
    ------
    ValueError
        If an indicator is NaN. Under the covariate contract an off-well event
        has no current arm, so its ``is_current_arm`` is NaN; pick an event at a
        well.
    """
    table = lc_event[INDICATOR_COLUMNS]
    missing = [column for column in INDICATOR_COLUMNS if table[column].isna().any()]
    if missing:
        raise ValueError(
            f"event {event_id} has NaN in {missing}, so it cannot be drawn as "
            "integer indicators; choose an event at a well with every indicator "
            "(--event-id)"
        )
    return table.astype(int)


def figure_axes() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """The figure and its six panels, A-F (3 rows x 2 columns)."""
    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    axes = {
        "A": ax_a,
        "B": fig.add_subplot(gs[0, 1], sharex=ax_a),
        "C": fig.add_subplot(gs[1, 0]),
        "D": fig.add_subplot(gs[1, 1]),
        "E": fig.add_subplot(gs[2, 0]),
        "F": fig.add_subplot(gs[2, 1]),
    }
    return fig, axes


def draw_indicator_table(ax: plt.Axes, long_table: pd.DataFrame):
    """Panel E's table, with each header wrapped at underscores to fit its cell.

    Parameters
    ----------
    ax : plt.Axes
    long_table : pd.DataFrame
        :func:`indicator_table`'s rows.

    Returns
    -------
    matplotlib.table.Table
    """
    ax.axis("off")
    table_data = long_table.values
    col_labels = [wrap_name(name) for name in long_table.columns]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=["#e8eef5"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    # The header row is as tall as its longest wrapped name.
    n_lines = max(label.count("\n") + 1 for label in col_labels)
    for column in range(len(col_labels)):
        header = table[(0, column)]
        header.set_height(header.get_height() * n_lines)
    # Color candidate-arm cells by arm color for visual link
    for row in range(len(table_data)):
        arm = int(table_data[row, 0])
        cell = table[(row + 1, 0)]
        cell.set_facecolor(ARM_COLORS[arm])
        cell.set_text_props(color="white" if arm < 6 else "black", fontweight="bold")
    return table


#: Panel F's M3 terms.
COEFFICIENT_TERMS = (
    "is_current_arm",
    "is_prev_arm",
    "reward_fraction_candidate_arm_last10_visits",
)


def draw_coefficients(ax: plt.Axes, m3_fits: pd.DataFrame) -> None:
    """Panel F: each M3 term's per-session coefficients and their median.

    A term's sessions are those its per-term gate keeps
    (:func:`~continuum_swr_replay.analysis.behavior.contrast_aggregation.gate_term_sessions`),
    and the printed median is
    :func:`~continuum_swr_replay.analysis.behavior.coefficient_statistics.gated_coefficient_statistics`'s,
    printed as the summary prints a coefficient (``sig2s``).

    Parameters
    ----------
    ax : plt.Axes
    m3_fits : pd.DataFrame, shape (n_sessions, n_columns)
        The M3 rung's per-session fits.
    """
    kept = {
        term: m3_fits.loc[gate_term_sessions(m3_fits, term).kept, f"coef_{term}"]
        for term in COEFFICIENT_TERMS
    }
    statistics = {
        term: gated_coefficient_statistics(m3_fits, term) for term in COEFFICIENT_TERMS
    }
    positions = np.arange(len(COEFFICIENT_TERMS))
    # Clip view to a sensible range: a few kept sessions have large
    # coefficients that distort the y-axis without conveying signal.
    y_lo, y_hi = -2.5, 2.5
    n_clipped = sum(
        int(((kept[term] > y_hi) | (kept[term] < y_lo)).sum())
        for term in COEFFICIENT_TERMS
    )
    bp = ax.boxplot(
        [kept[term].clip(lower=y_lo, upper=y_hi) for term in COEFFICIENT_TERMS],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"markersize": 3, "alpha": 0.5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#3b6ea5")
        patch.set_alpha(0.6)
    for i, term in enumerate(COEFFICIENT_TERMS):
        med = float(statistics[term]["median"])
        ax.text(
            i,
            min(max(med, y_lo), y_hi) + 0.12,
            format_number(med, "sig2s"),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#1a2433",
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{term}\n({statistics[term]['n_sessions']} sessions)"
            for term in COEFFICIENT_TERMS
        ],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("coefficient (clipped to ±2.5)")
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(
        f"(F) Fitted coefficients (M3) of {len(m3_fits)} sessions, each term over "
        "the sessions its gate keeps\n"
        f"per-session ConditionalPoisson fits; {n_clipped} extreme outliers clipped"
    )


def make_figure(
    *,
    dot: pd.DataFrame,
    event_id: int,
    long_count: pd.DataFrame,
    events: pd.DataFrame,
    m3_fits: pd.DataFrame,
    output_dir: Path,
) -> None:
    event_row = events.loc[events["non_local_event"] == event_id]
    if len(event_row) != 1:
        raise ValueError(
            f"expected one session-local non_local_event={event_id}, "
            f"found {len(event_row)}"
        )
    local_event = int(event_row["non_local_event"].iloc[0])
    canonical_event_id = event_row["event_id"].iloc[0]
    # Pull the event we're going to feature, and a window around it.
    ev_rows = dot[dot["non_local_event"] == local_event]
    t_start = ev_rows.index.min()
    t_end = ev_rows.index.max()
    pad = max(1.0, 1.5 * (t_end - t_start))
    win = dot.loc[t_start - pad : t_end + pad].copy()
    win_t = win.index.values - t_start  # seconds relative to event onset

    # Get the per-arm count vector for this event from the saved long_count.
    lc_event = long_count[long_count["event_id"] == canonical_event_id].sort_values(
        "candidate_arm"
    )
    counts = lc_event.set_index("candidate_arm")["arm_count_hard"].reindex(
        range(9), fill_value=0
    )
    track_seg = int(event_row["modal_observed_track_segment_id"].iloc[0])

    # Indicator columns from the long_count for the long-format mini-table.
    long_table = indicator_table(lc_event, event_id)

    # ── Layout: 3 rows × 2 cols ──
    fig, axes = figure_axes()
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = (axes[k] for k in "ABCDEF")

    # ── (A) Filters: speed + non-local prob over a 6-sec window ──
    ax = ax_a
    ax.plot(win_t, win["head_speed"].values, color="#444", lw=1, label="head_speed")
    ax.axhline(
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        color="#444",
        ls="--",
        lw=0.8,
        alpha=0.6,
    )
    ax.set_ylabel("speed (cm/s)", color="#444")
    ax.tick_params(axis="y", labelcolor="#444")
    ax.set_xlabel("time relative to event onset (s)")
    ax.set_title(
        "(A) Raw decoder per-timebin\nfilter: head_speed < 4  AND  Non-Local_prob > 0.8"
    )
    ax2 = ax.twinx()
    ax2.plot(
        win_t,
        win["Non-Local_prob"].values,
        color="#a55a3b",
        lw=1,
        label="Non-Local_prob",
    )
    ax2.axhline(
        DEFAULT_NON_LOCAL_THRESHOLD, color="#a55a3b", ls="--", lw=0.8, alpha=0.6
    )
    ax2.set_ylabel("Non-Local prob", color="#a55a3b")
    ax2.tick_params(axis="y", labelcolor="#a55a3b")
    ax2.set_ylim(0, 1.05)
    # Shade non-local events in the window
    nl_in_window = win[win["non_local_event"] > 0]
    for ev_in_win, g in nl_in_window.groupby("non_local_event"):
        s = g.index.min() - t_start
        e = g.index.max() - t_start
        is_featured = ev_in_win == local_event
        ax.axvspan(
            s,
            e,
            color="#3b6ea5" if is_featured else "#cccccc",
            alpha=0.4 if is_featured else 0.25,
            zorder=0,
        )
    ax.text(
        0.02,
        0.95,
        f"event {event_id}",
        transform=ax.transAxes,
        fontsize=9,
        color="#3b6ea5",
        fontweight="bold",
        va="top",
    )

    # ── (B) MAP estimate scatter colored by replay_arm ──
    ax = ax_b
    valid = win["replay_arm"].notna() & (win["non_local_event"] > 0)
    colors = [ARM_COLORS[int(a)] for a in win.loc[valid, "replay_arm"]]
    ax.scatter(
        win_t[valid.values],
        win.loc[valid, "map_estimate"].values,
        c=colors,
        s=4,
        alpha=0.7,
    )
    # Faint scatter for non-event timebins
    other = win["replay_arm"].notna() & (win["non_local_event"] == 0)
    ax.scatter(
        win_t[other.values],
        win.loc[other, "map_estimate"].values,
        c="lightgrey",
        s=2,
        alpha=0.3,
    )
    ax.axvspan(0, t_end - t_start, color="#3b6ea5", alpha=0.15, zorder=0)
    ax.set_ylabel("MAP linear position (cm)")
    ax.set_xlabel("time relative to event onset (s)")
    ax.set_title(
        "(B) MAP → arm  (map_estimate_to_arm)\n"
        f"each non-local timebin assigned to one of 9 arms; "
        f"animal physically on arm {track_seg}"
    )

    # ── (C) Per-timebin replay_arm sequence for the featured event ──
    ax = ax_c
    ev_t = ev_rows.index.values - t_start
    ev_arms = ev_rows["replay_arm"].fillna(-1).astype(int).values
    for arm in range(9):
        mask = ev_arms == arm
        ax.scatter(
            ev_t[mask],
            np.full(mask.sum(), arm),
            color=ARM_COLORS[arm],
            s=18,
            marker="s",
        )
    ax.set_yticks(range(9))
    ax.set_yticklabels([f"arm {a}" for a in range(9)])
    ax.invert_yaxis()
    ax.set_xlabel("time relative to event onset (s)")
    ax.set_title(
        f"(C) Per-timebin replay_arm — event {event_id} only\n"
        f"{len(ev_rows)} timebins; this is what arm_count_hard sums"
    )
    ax.axhline(track_seg, color="#444", ls=":", lw=0.8, alpha=0.5)
    ax.text(
        ev_t[-1],
        track_seg + 0.4,
        f"  ← animal physically here (arm {track_seg})",
        fontsize=8,
        color="#444",
        va="center",
    )

    # ── (D) Per-arm count vector for the featured event ──
    ax = ax_d
    bars = ax.bar(
        range(9),
        counts.values,
        color=[ARM_COLORS[a] for a in range(9)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"arm {a}" for a in range(9)], rotation=0, fontsize=8)
    ax.set_ylabel("arm_count")
    ax.set_title(
        f"(D) Per-event count vector — event {event_id}\n"
        f"L_e = {int(counts.sum())} timebins; the multinomial sufficient statistic"
    )
    for b, v in zip(bars, counts.values, strict=True):
        if v > 0:
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.5,
                f"{int(v)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # ── (E) Long-format mini-table for this event ──
    ax = ax_e
    draw_indicator_table(ax, long_table)
    ax.set_title(
        f"(E) Long-format expansion — event {event_id} becomes 9 rows\n"
        "this is the input row block to fit_conditional_poisson",
        pad=12,
    )

    # ── (F) Population coefficient distribution across sessions ──
    draw_coefficients(ax_f, m3_fits)

    fig.suptitle(
        "Replay-content GLM pipeline — one event traced through every transformation",
        fontsize=14,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"pipeline_data_flow.{ext}",
            dpi=200 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal", default="chimi")
    parser.add_argument("--day", type=int, default=20200213)
    parser.add_argument("--epoch", default="02_r1")
    # The featured event: the session-local non_local_event label. 275 is at a
    # well, with every arm indicator (none NaN), content on at least 3 arms, some
    # on the current arm, and 80-160 ms long (124 ms, 62 bins on 4 arms, at
    # well arm 5). The old default, 14, is off-well under the current covariate
    # contract, so its is_current_arm is NaN.
    parser.add_argument("--event-id", type=int, default=275)
    parser.add_argument("--output-dir", type=Path, default=figure_dir("qc_diagnosis"))
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Draw the fitted model even if its input table has changed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    dot, env = load_session(args.animal, args.day, args.epoch)
    long_count = read_table(table_path("event_arm_table"))
    events = read_table(table_path("event_table"))
    require_covariate_schema(long_count, table="event_arm")
    require_covariate_schema(events, table="event")
    require_swr_source(long_count, what=str(table_path("event_arm_table")))
    require_swr_source(events, what=str(table_path("event_table")))
    require_fresh_inputs(
        m3_path(), all_models_input_paths(), allow_stale=args.allow_stale
    )
    m3_fits = pd.read_pickle(m3_path())

    sess_mask = (
        (long_count["animal"] == args.animal)
        & (long_count["session_date"] == args.day)
        & (long_count["epoch"] == args.epoch)
    )
    long_count = long_count[sess_mask]
    events = events[
        (events["animal"] == args.animal)
        & (events["session_date"] == args.day)
        & (events["epoch"] == args.epoch)
    ]

    make_figure(
        dot=dot,
        event_id=args.event_id,
        long_count=long_count,
        events=events,
        m3_fits=m3_fits,
        output_dir=args.output_dir,
    )
    print(f"Saved data-flow figure to {args.output_dir}/pipeline_data_flow.{{png,pdf}}")


if __name__ == "__main__":
    main()
