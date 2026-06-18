#!/usr/bin/env python
"""Per-event decoding-plot gallery for validating regression event categories.

Goal
----
Sanity-check that the SWR-overlap categories used in the replay-content
regressions make sense against the real decoded data. For a stratified
sample of events, render the same 4-row decoding stack used in Figure 2a
(SWR z-score trace / spike raster / conditional non-local posterior /
acausal state probabilities) and annotate each plot with *how that event is
scored in the regression* (its ``is_swr`` covariate, regression-population
membership, arm/patch content predictors, MUA flag, matched SWR).

Four categories (each its own folder under ``event_gallery/``):

    swr_contained     non-local event fully inside an SWR   (is_swr=True)
    swr_partial       non-local event partially overlaps SWR (is_swr=True)
    non_swr           non-local event, no SWR overlap        (is_swr=False)
    swr_not_nonlocal  SWR event with NO non-local event      (not in the
                      non-local regression; stop-level models only)

The first three come from ``build_classified_nonlocal_event_table``'s
``overlap_class``; the fourth is the SWR-side complement (SWR intervals
overlapping no non-local event).

Why a sample? Each epoch holds ~1.7k immobile non-local events (~370k across
218 epochs) — too many to render or eyeball. We instead take ~50 events per
category per animal, *stratified by ambiguity* (boundary cases: low-overlap
partials, near-miss non-SWR events, sub-threshold SWR-only windows, very
short events), which is what actually stresses the category definitions.

Two phases (the per-epoch Spyglass load dominates cost, so it is the
parallel unit):

    1. catalog  — stream every ``*_HPC_80_data_over_time.pkl`` (cheap, no
                  Spyglass load), classify + sample, write a small catalog
                  parquet of selected events.
    2. render   — for one epoch in the catalog: load_data + load_results
                  once, build_event_table for predictors, render each
                  selected event to PNG. Run in parallel across epochs via
                  ``run_event_gallery.sh`` (tmux).

Usage
-----
    # Phase 1: build the selection catalog (~10-20 min, single process)
    python scripts/generate_event_gallery.py catalog

    # Phase 2: render one epoch's selected events (the tmux unit)
    python scripts/generate_event_gallery.py render --epoch-id chimi20200213_02_r1

    # List the epoch-ids that have selected events (for the launcher)
    python scripts/generate_event_gallery.py list-epochs
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus import load_results
from continuum_swr_replay.analysis.hippocampus.event_table import (
    build_classified_nonlocal_event_table,
    intervals_from_labels,
)
from continuum_swr_replay.analysis.replay.posterior_io import (
    load_conditional_posterior,
)
from continuum_swr_replay.analysis.replay.tables import (
    build_event_arm_table,
    build_event_table,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.plots import (
    _track_edge_colors,
    plot_acausal_probabilities,
    plot_conditional_non_local_posterior,
    plot_linear_track_graph_strip,
    plot_multiunit_firing,
    plot_ripple_consensus_trace_zscore,
    plot_speed,
    plot_spike_times,
    plot_track_graph_2d,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")

CATEGORIES = ("swr_contained", "swr_partial", "non_swr", "swr_not_nonlocal")
DEFAULT_GALLERY_DIR = Path("event_gallery")
DEFAULT_N_PER_CATEGORY = 50
WINDOW_SEC = 0.4
SAMPLE_SEED = 20260617


# ---------------------------------------------------------------------------
# Epoch-id helpers
# ---------------------------------------------------------------------------


def epoch_id_from_pkl(pkl_path: Path) -> str:
    """``chimi20200213_02_r1_HPC_80_data_over_time.pkl`` -> ``chimi20200213_02_r1``."""
    return pkl_path.name.replace("_HPC_80_data_over_time.pkl", "")


def parse_epoch_id(epoch_id: str) -> tuple[str, str]:
    """``chimi20200213_02_r1`` -> (``chimi20200213_.nwb``, ``02_r1``)."""
    tokens = epoch_id.split("_")
    epoch_name = "_".join(tokens[-2:])
    nwb_stem = "_".join(tokens[:-2])
    return f"{nwb_stem}_.nwb", epoch_name


def list_epoch_pkls(data_dir: Path) -> list[Path]:
    pkls = sorted(data_dir.glob("*_HPC_80_data_over_time.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No *_HPC_80_data_over_time.pkl files found in {data_dir}"
        )
    return pkls


# ---------------------------------------------------------------------------
# Interval geometry
# ---------------------------------------------------------------------------


def _intervals_overlap_any(
    starts: np.ndarray,
    ends: np.ndarray,
    other_starts: np.ndarray,
    other_ends: np.ndarray,
) -> np.ndarray:
    """Boolean mask: each [starts, ends] interval overlaps any [other_*] interval.

    Half-open overlap test ``s < oe and os < e`` over all pairs. Sizes are
    small per session (hundreds), so the dense broadcast is fine.
    """
    if len(other_starts) == 0:
        return np.zeros(len(starts), dtype=bool)
    s = starts[:, None]
    e = ends[:, None]
    os = other_starts[None, :]
    oe = other_ends[None, :]
    return ((s < oe) & (os < e)).any(axis=1)


def _min_gap_to_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    other_starts: np.ndarray,
    other_ends: np.ndarray,
) -> np.ndarray:
    """Minimum temporal gap from each [starts, ends] interval to any other interval.

    0 where they overlap. Used to find non-SWR events that *barely* miss an
    SWR (small gap = boundary case near swr_partial).
    """
    if len(other_starts) == 0:
        return np.full(len(starts), np.inf)
    s = starts[:, None]
    e = ends[:, None]
    os = other_starts[None, :]
    oe = other_ends[None, :]
    # gap = max(0, os - e, s - oe) elementwise; 0 if overlapping
    gap = np.maximum.reduce([np.zeros_like(s + os), os - e, s - oe])
    return gap.min(axis=1)


# ---------------------------------------------------------------------------
# Phase 1: catalog
# ---------------------------------------------------------------------------


def _catalog_one_epoch(pkl_path: Path) -> pd.DataFrame:
    """Classify + feature one epoch into a per-event catalog frame (cheap).

    Returns one row per immobile non-local event (categories swr_contained /
    swr_partial / non_swr) plus one row per SWR-only event (swr_not_nonlocal),
    each carrying the ambiguity feature used for stratified sampling.
    """
    epoch_id = epoch_id_from_pkl(pkl_path)
    nwb_file_name, epoch_name = parse_epoch_id(epoch_id)

    dot = pd.read_pickle(pkl_path)
    dot["Non-Local_prob"] = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    )

    ct = build_classified_nonlocal_event_table(dot)
    rows: list[dict] = []

    if len(ct):
        animal = ct["animal"].iloc[0]
        day = int(ct["day"].iloc[0])
        # Immobile only (matches the regression analysis population).
        ct = ct[ct["endpoints_speed_below_thresh"]].copy()
        ev_dur = (ct["event_end"] - ct["event_start"]).to_numpy()

        swr_intervals = intervals_from_labels(dot, "SWR_event")
        swr_starts = swr_intervals["start_time"].to_numpy()
        swr_ends = swr_intervals["end_time"].to_numpy()

        # Ambiguity feature per category (see module docstring).
        #  - swr_partial: fraction of the EVENT inside its matched SWR
        #    (low -> barely overlapping -> near non_swr; high -> near contained)
        #  - swr_contained: event/SWR duration ratio (->1 means event nearly
        #    fills the SWR, i.e. near partial; small means snugly contained)
        #  - non_swr: temporal gap to nearest SWR (small -> near partial)
        for i, (_, ev) in enumerate(ct.iterrows()):
            cls = ev["overlap_class"]
            common = {
                "epoch_id": epoch_id,
                "animal": animal,
                "day": day,
                "epoch_name": epoch_name,
                "category": cls,
                "event_label": int(ev["non_local_event"]),
                "event_id": int(ev["event_id"]),
                "plot_start": float(ev["event_start"]),
                "plot_end": float(ev["event_end"]),
                "duration_ms": float(1000 * ev_dur[i]),
                "peak_nonlocal_prob": float(ev["peak_nonlocal_prob"]),
            }
            if cls == "swr_partial":
                ov_start = max(ev["event_start"], ev["matched_swr_start"])
                ov_end = min(ev["event_end"], ev["matched_swr_end"])
                ov = max(0.0, ov_end - ov_start)
                feat = ov / ev_dur[i] if ev_dur[i] > 0 else 0.0
            elif cls == "swr_contained":
                swr_dur = ev["matched_swr_end"] - ev["matched_swr_start"]
                feat = ev_dur[i] / swr_dur if swr_dur > 0 else 1.0
            else:  # non_swr
                feat = float(
                    _min_gap_to_intervals(
                        np.array([ev["event_start"]]),
                        np.array([ev["event_end"]]),
                        swr_starts,
                        swr_ends,
                    )[0]
                )
            common["ambiguity"] = float(feat)
            rows.append(common)

        # swr_not_nonlocal: SWR intervals overlapping no non-local event.
        nl_intervals = intervals_from_labels(dot, "non_local_event")
        nl_starts = nl_intervals["start_time"].to_numpy()
        nl_ends = nl_intervals["end_time"].to_numpy()
        if len(swr_intervals):
            overlaps_nl = _intervals_overlap_any(
                swr_starts, swr_ends, nl_starts, nl_ends
            )
            only = swr_intervals.loc[~overlaps_nl]
            for _, sw in only.iterrows():
                s, e = float(sw["start_time"]), float(sw["end_time"])
                win = dot.loc[s:e, "Non-Local_prob"]
                # ambiguity = peak non-local prob inside the SWR window
                # (high but no labeled event = borderline "almost an event")
                peak = float(win.max()) if len(win) else 0.0
                rows.append(
                    {
                        "epoch_id": epoch_id,
                        "animal": animal,
                        "day": day,
                        "epoch_name": epoch_name,
                        "category": "swr_not_nonlocal",
                        "event_label": int(sw["SWR_event"]),
                        "event_id": -1,
                        "plot_start": s,
                        "plot_end": e,
                        "duration_ms": float(1000 * (e - s)),
                        "peak_nonlocal_prob": peak,
                        "ambiguity": peak,
                    }
                )

    return pd.DataFrame(rows)


def _stratified_sample(
    df: pd.DataFrame, n: int, n_bins: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Sample ~n rows spread across quantile bins of ``ambiguity``.

    Quantile-binning then sampling evenly per bin guarantees the sample spans
    the full ambiguity range — both clear and boundary cases — rather than
    concentrating on the typical event.
    """
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    ranks = df["ambiguity"].rank(method="first")
    bins = pd.qcut(ranks, q=min(n_bins, len(df)), labels=False, duplicates="drop")
    df = df.assign(_bin=bins)
    n_bin = df["_bin"].nunique()
    per_bin = max(1, n // n_bin)
    picks = []
    for _, g in df.groupby("_bin"):
        take = min(len(g), per_bin)
        picks.append(g.iloc[rng.permutation(len(g))[:take]])
    res = pd.concat(picks)
    if len(res) < n:  # top up from the remainder
        extra = df.loc[~df.index.isin(res.index)]
        k = min(len(extra), n - len(res))
        if k:
            res = pd.concat([res, extra.iloc[rng.permutation(len(extra))[:k]]])
    return res.drop(columns="_bin")


def build_catalog(
    data_dir: Path, gallery_dir: Path, n_per_category: int
) -> pd.DataFrame:
    pkls = list_epoch_pkls(data_dir)
    logging.info("Cataloging %d epochs...", len(pkls))
    frames = []
    for i, pkl in enumerate(pkls, 1):
        frames.append(_catalog_one_epoch(pkl))
        if i % 20 == 0:
            logging.info("  ...%d/%d epochs", i, len(pkls))
    catalog = pd.concat(frames, ignore_index=True)
    logging.info(
        "Full catalog: %d events. By category:\n%s",
        len(catalog),
        catalog["category"].value_counts().to_string(),
    )

    # Stratified sample ~n per category per animal, deterministic per group.
    selected = []
    for (animal, cls), g in catalog.groupby(["animal", "category"]):
        seed = SAMPLE_SEED + (hash((animal, cls)) & 0xFFFF)
        selected.append(_stratified_sample(g, n_per_category, seed=seed))
    selected_df = pd.concat(selected, ignore_index=True)
    logging.info(
        "Selected %d events for the gallery. By category:\n%s",
        len(selected_df),
        selected_df["category"].value_counts().to_string(),
    )

    gallery_dir.mkdir(parents=True, exist_ok=True)
    out = gallery_dir / "_catalog.parquet"
    selected_df.to_parquet(out)
    logging.info("Wrote selection catalog -> %s", out)
    return selected_df


# ---------------------------------------------------------------------------
# Phase 2: render
# ---------------------------------------------------------------------------


def _arm_of_edge_map() -> dict:
    """frozenset({node, node}) -> track-segment (arm) id, from node identities.

    Leaf arm ``w`` (0-5) joins well ``w`` to junction ``6 + w // 2``; trunk
    arms 6/7/8 join junction 6/7/8 to the center node 9 (the track-graph
    convention documented in CLAUDE.md). Used to label arms with the same id
    the regression's ``current/prev/next_arm`` predictors carry.
    """
    arm_of_edge = {}
    for w in range(6):
        arm_of_edge[frozenset((w, 6 + w // 2))] = w
    for j in (6, 7, 8):
        arm_of_edge[frozenset((j, 9))] = j
    return arm_of_edge


def _arm_colors(edge_order: list, arm_of_edge: dict) -> dict:
    """arm id -> rgba, matching the 2D track and 1D strip arm colors.

    The track/strip color each edge by its position in ``edge_order`` (tab10);
    map that back to the arm id so the content bar uses the same color per arm.
    """
    edge_colors = _track_edge_colors(len(edge_order))
    colors = {}
    for i, (n1, n2) in enumerate(edge_order):
        arm = arm_of_edge.get(frozenset((n1, n2)))
        if arm is not None:
            colors[arm] = edge_colors[i]
    return colors


def _current_well(track_graph, x: float, y: float) -> int | None:
    """Nearest reward-well node (degree-1) to a 2D point, or None."""
    wells = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
    if not wells:
        return None
    pos = {n: track_graph.nodes[n]["pos"] for n in wells}
    return min(wells, key=lambda n: (pos[n][0] - x) ** 2 + (pos[n][1] - y) ** 2)


def _run_into_well(
    trials: pd.DataFrame, well: int | None, t: float
) -> pd.Series | None:
    """The run that delivered the animal to ``well`` (its ``to_well``) by ``t``.

    An immobile event happens while the animal sits at a well; the run to show
    is the most recent trial whose ``to_well`` is that well and that began by
    ``t`` (so the displayed path actually ends where the animal now is — not a
    run it is only part-way through). Falls back to the most recent trial that
    ended before ``t`` if no ``to_well`` match exists.
    """
    if well is not None:
        match = trials[(trials["to_well"] == well) & (trials["start_time"] <= t)]
        if len(match):
            return match.sort_values("start_time").iloc[-1]
    ended = trials[trials["end_time"] <= t].sort_values("end_time")
    return ended.iloc[-1] if len(ended) else None


def _plot_2d_context(
    ax,
    data: dict,
    center: float,
    window_sec: float,
    arm_of_edge: dict,
    et_row: pd.Series | None = None,
) -> pd.Series | None:
    """2D track with labeled wells/arms, the animal's position, and last run.

    Draws the arm-colored track + faint occupancy (shared color map with the
    1D strip), labels each well node and arm with its id, marks the animal's
    head position during the event window, and highlights the last run's
    ``from_well`` -> ``to_well``. Returns the trial row used (or None).
    """
    track_graph = data["track_graph"]
    plot_track_graph_2d(
        track_graph,
        data["position_info"],
        ax=ax,
        edge_order=data["linear_edge_order"],
    )
    node_pos = {n: track_graph.nodes[n]["pos"] for n in track_graph.nodes}

    # Well labels (degree-1 nodes 0-5). W0/W5 are bottom wells whose label
    # would overlap the animal/run markers, so place those two below; the
    # rest read fine above.
    for n in track_graph.nodes:
        if track_graph.degree(n) == 1:
            x, y = node_pos[n]
            dy = -12 if n in (0, 5) else 8
            va = "top" if n in (0, 5) else "bottom"
            ax.annotate(
                f"W{n}",
                (x, y),
                fontsize=6,
                fontweight="bold",
                ha="center",
                va=va,
                color="black",
                xytext=(0, dy),
                textcoords="offset points",
            )
    # Patch labels (6->P1, 7->P2, 8->P3), placed out in the fork of each
    # patch's "Y": from the junction, pushed in the direction away from the
    # center node (9) so the label sits in the open V between the two leaf
    # arms. Colored to match the per-patch content bar.
    patch_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    center_node = 9
    if center_node in node_pos:
        cx, cy = node_pos[center_node]
        # Typical leaf-arm length sets how far out to push the label.
        arm_len = float(
            np.median(
                [
                    np.hypot(
                        node_pos[n1][0] - node_pos[n2][0],
                        node_pos[n1][1] - node_pos[n2][1],
                    )
                    for n1, n2 in track_graph.edges
                ]
            )
        )
        for j in (6, 7, 8):
            if j not in node_pos:
                continue
            jx, jy = node_pos[j]
            dx, dy = jx - cx, jy - cy
            norm = float(np.hypot(dx, dy)) or 1.0
            lx = jx + (dx / norm) * arm_len * 0.55
            ly = jy + (dy / norm) * arm_len * 0.55
            ax.annotate(
                f"P{j - 5}",
                (lx, ly),
                fontsize=7,
                fontweight="bold",
                color=patch_colors[j - 5],
                ha="center",
                va="center",
            )

    # Arm labels, offset perpendicular to the edge so they sit beside (not on)
    # the track line.
    for n1, n2 in track_graph.edges:
        arm = arm_of_edge.get(frozenset((n1, n2)))
        if arm is None:
            continue
        (x1, y1), (x2, y2) = node_pos[n1], node_pos[n2]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        norm = float(np.hypot(dx, dy)) or 1.0
        px, py = -dy / norm, dx / norm  # unit perpendicular
        ax.annotate(
            f"a{arm}",
            (mx, my),
            fontsize=5.5,
            ha="center",
            va="center",
            color="dimgray",
            xytext=(7 * px, 7 * py),
            textcoords="offset points",
        )

    pos = data["position_info"]
    win = pos.loc[center - window_sec : center + window_sec]

    # The animal's current well comes from the pipeline-derived ``current_arm``
    # (modal decoded arm of the animal's physical position; leaf arm w == well
    # w). Fall back to nearest-well from position only when the pipeline value
    # is unavailable (swr_not_nonlocal events) or lands on a trunk arm (6-8).
    cur_well = None
    if et_row is not None:
        ca = et_row.get("current_arm")
        if ca is not None and not (isinstance(ca, float) and np.isnan(ca)):
            ca = int(ca)
            if ca in range(6):
                cur_well = ca
    if cur_well is None and len(win):
        cur_well = _current_well(
            track_graph,
            float(win["head_position_x"].median()),
            float(win["head_position_y"].median()),
        )

    # Actual run trajectory: the animal's real head-position path during the
    # run that delivered it to its current well (from_well -> cur_well). Use
    # the run portion [start_time, goal_arrival_time] when available, else the
    # whole trial interval.
    trial = _run_into_well(data["trials"], cur_well, center)
    if trial is not None:
        run_end = trial.get("goal_arrival_time")
        if run_end is None or (isinstance(run_end, float) and np.isnan(run_end)):
            run_end = trial["end_time"]
        run = pos.loc[trial["start_time"] : run_end]
        if len(run):
            # High-contrast run path: thick navy line with a white halo so it
            # reads clearly over the tab10 arm colors and faint occupancy.
            ax.plot(
                run["head_position_x"],
                run["head_position_y"],
                color="navy",
                lw=2.6,
                alpha=1.0,
                zorder=5,
                label="run",
                solid_capstyle="round",
                path_effects=[
                    patheffects.Stroke(linewidth=4.6, foreground="white"),
                    patheffects.Normal(),
                ],
            )
        fw, tw = int(trial["from_well"]), int(trial["to_well"])
        # Reward status of the run into the current well (pipeline-derived
        # trials.is_reward): green = rewarded, red = omission.
        rewarded = bool(trial.get("is_reward", False))
        rew_color = "limegreen" if rewarded else "red"
        if fw in node_pos:
            ax.scatter(
                *node_pos[fw],
                s=70,
                facecolor="none",
                edgecolor="navy",
                linewidth=1.4,
                zorder=6,
            )
        if tw in node_pos:
            ax.scatter(
                *node_pos[tw],
                s=95,
                facecolor="navy",
                edgecolor=rew_color,
                linewidth=1.8,
                zorder=6,
            )

    # Animal's actual position during the event, in magenta to match the
    # posterior's actual-position line (immobile -> ~a point).
    win = pos.loc[center - window_sec : center + window_sec]
    if len(win):
        ax.scatter(
            win["head_position_x"].iloc[len(win) // 2],
            win["head_position_y"].iloc[len(win) // 2],
            s=55,
            facecolor="magenta",
            edgecolor="white",
            linewidth=0.6,
            zorder=7,
            label="animal @ event",
        )

    if trial is not None:
        rew = "rewarded" if bool(trial.get("is_reward", False)) else "omission"
        ax.set_title(
            f"run: W{int(trial['from_well'])}→W{int(trial['to_well'])} ({rew})",
            fontsize=7,
        )
    else:
        ax.set_title("run: W?→W?", fontsize=7)
    return trial


def _plot_arm_content_bar(
    ax,
    arm_rows: pd.DataFrame,
    et_row: pd.Series | None,
    arm_colors: dict,
) -> None:
    """Per-arm posterior mass for the event — the content the M0-M7 models use.

    Bars are ``arm_posterior_mass`` (conditional posterior marginalized to each
    arm, summed over the event), colored to match the track. The regression's
    arm predictors (current/prev/next/last-rewarded) are flagged under the bars
    so content can be read against the design matrix.
    """
    arms = list(range(9))
    if arm_rows is None or not len(arm_rows):
        ax.set_title("per-arm content (M0–M7)", fontsize=7)
        ax.text(
            0.5,
            0.5,
            "n/a — not in\nnon-local regression",
            fontsize=6,
            ha="center",
            va="center",
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return
    mass = arm_rows.set_index("arm")["arm_posterior_mass"].reindex(arms).fillna(0.0)
    bar_colors = [arm_colors.get(a, "lightgray") for a in arms]
    ax.bar(arms, mass.to_numpy(), color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(arms)
    ax.set_xticklabels([f"a{a}" for a in arms], fontsize=5)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_ylabel("post. mass", fontsize=6)
    ax.set_title("per-arm content (M0–M7)", fontsize=7)

    ymax = float(mass.max()) or 1.0
    ax.set_ylim(0, ymax * 1.35)
    if et_row is not None:
        flags = {
            "cur": et_row.get("current_arm"),
            "prev": et_row.get("prev_arm"),
            "next": et_row.get("next_arm"),
            "lastR": et_row.get("last_rewarded_arm"),
        }
        labels: dict[int, list] = {}
        for name, val in flags.items():
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                labels.setdefault(int(val), []).append(name)
        # Predictor roles printed just above each flagged bar (inside the axis)
        # so they never collide with the footer below the panel.
        for arm, names in labels.items():
            ax.annotate(
                "\n".join(names),
                (arm, mass.get(arm, 0.0)),
                fontsize=4.2,
                color="black",
                ha="center",
                va="bottom",
                xytext=(0, 2),
                textcoords="offset points",
            )


def _plot_trial_history(ax, data: dict, center: float, n_trials: int = 20) -> None:
    """Trial-by-trial reward history (the trial-level behavioral input).

    Grid of wells (rows W0-W5) × recent trials (columns, time → right). Each
    trial's ``to_well`` is marked with the TRIAL-level reward (``is_reward`` =
    any poke in that trial's stay was rewarded): filled green if rewarded,
    open if an omission — the same trial-level label the regression uses
    (``reward_outcome``), not the per-poke reward. A faint line links
    consecutive trials to show the route sequence. This is exactly what
    ``visit_recency`` (horizontal gaps in a row), ``recent_reward_rate``
    (density of filled markers), and ``last_rewarded_elsewhere`` (most recent
    filled marker off the current row) are derived from. The current trial
    (the one at the event) is ringed in magenta.
    """
    ax.set_title("trial reward history (regression-level, → event)", fontsize=7)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"W{w}" for w in range(6)], fontsize=5)
    ax.set_ylim(-0.6, 5.6)
    ax.tick_params(axis="x", labelsize=5)
    ax.set_xlabel("trial (recent history → event)", fontsize=6)
    ax.grid(axis="y", color="0.9", lw=0.5)

    trials = data.get("trials")
    if trials is None or not len(trials):
        ax.text(
            0.5,
            0.5,
            "no trials",
            fontsize=6,
            ha="center",
            va="center",
            color="gray",
            transform=ax.transAxes,
        )
        return
    trials = trials.sort_values("start_time").reset_index(drop=True)
    started = trials.index[trials["start_time"] <= center]
    cur_idx = int(started[-1]) if len(started) else 0
    lo = max(0, cur_idx - n_trials + 1)
    window = trials.loc[lo:cur_idx]

    xs = list(range(len(window)))
    wells = [int(w) for w in window["to_well"].to_numpy()]
    ax.plot(xs, wells, color="0.8", lw=0.8, zorder=1)
    for x, (_, tr) in zip(xs, window.iterrows(), strict=True):
        well = int(tr["to_well"])
        if bool(tr["is_reward"]):
            ax.scatter(
                x,
                well,
                s=55,
                marker="o",
                facecolor="green",
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        else:
            ax.scatter(
                x,
                well,
                s=55,
                marker="o",
                facecolor="none",
                edgecolor="dimgray",
                linewidth=1.0,
                zorder=3,
            )
    # Current trial (the one containing the event) highlighted in magenta.
    ax.scatter(
        xs[-1],
        wells[-1],
        s=140,
        marker="o",
        facecolor="none",
        edgecolor="magenta",
        linewidth=1.6,
        zorder=4,
    )
    ax.set_xlim(-0.5, max(len(window) - 0.5, 0.5))
    ax.set_xticks(xs[:: max(1, len(xs) // 10)])

    # Compact legend.
    ax.scatter(
        [], [], s=40, marker="o", facecolor="green", edgecolor="black", label="rewarded"
    )
    ax.scatter(
        [],
        [],
        s=40,
        marker="o",
        facecolor="none",
        edgecolor="dimgray",
        label="omission",
    )
    ax.legend(
        loc="upper left",
        fontsize=4.6,
        ncol=2,
        framealpha=0.6,
        handletextpad=0.2,
        columnspacing=0.8,
        borderpad=0.2,
    )


def _plot_arm_predictor_heatmap(ax, arm_rows: pd.DataFrame) -> None:
    """Per-arm continuous/binary regressors the arm models use (M3-M5).

    A 3×9 strip (rows = ``visit_recency_demeaned``, ``recent_reward_rate``,
    ``is_last_rewarded_elsewhere``; columns = arms 0-8). Each row is colored
    on its own scale (these regressors live on different units) with the
    actual value printed in each cell, so the arm design matrix is visible
    alongside the content bar above.
    """
    preds = [
        ("visit_recency_demeaned", "visit_rec"),
        ("recent_reward_rate", "rew_rate"),
        ("is_last_rewarded_elsewhere", "lastR_elsw"),
    ]
    ax.set_title("per-arm regressors (M3–M5)", fontsize=7)
    arms = list(range(9))
    if arm_rows is None or not len(arm_rows):
        ax.text(
            0.5,
            0.5,
            "n/a",
            fontsize=6,
            ha="center",
            va="center",
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return
    indexed = arm_rows.set_index("arm")
    mat = np.vstack(
        [indexed[col].reindex(arms).to_numpy(dtype=float) for col, _ in preds]
    )
    # Per-row min-max for color only (values themselves are annotated).
    norm = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        row = mat[i]
        lo, hi = np.nanmin(row), np.nanmax(row)
        norm[i] = (row - lo) / (hi - lo) if hi > lo else 0.0
    ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(arms)
    ax.set_xticklabels([f"a{a}" for a in arms], fontsize=4.5)
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels([short for _, short in preds], fontsize=4.8)
    for i in range(mat.shape[0]):
        for j in arms:
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=3.6,
                color="white" if norm[i, j] < 0.5 else "black",
            )


def _plot_patch_content_bar(
    ax,
    arm_rows: pd.DataFrame,
    et_row: pd.Series | None,
) -> None:
    """Per-patch posterior mass — content for the P0-P4 patch models.

    Patch mass is the arm posterior mass summed over the arms in each patch
    (``TRACK_SEGMENT_TO_PATCH``), the same collapse the patch models use. The
    current/prev/next patch predictors are flagged above the bars.
    """
    patches = [1, 2, 3]
    ax.set_title("per-patch content (P0–P4)", fontsize=7)
    if arm_rows is None or not len(arm_rows):
        ax.text(
            0.5,
            0.5,
            "n/a",
            fontsize=6,
            ha="center",
            va="center",
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return
    by_patch = (
        arm_rows.assign(patch=arm_rows["arm"].map(TRACK_SEGMENT_TO_PATCH))
        .groupby("patch")["arm_posterior_mass"]
        .sum()
        .reindex(patches)
        .fillna(0.0)
    )
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax.bar(patches, by_patch.to_numpy(), color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(patches)
    # Label which wells each patch contains (P1=W0/1, P2=W2/3, P3=W4/5).
    patch_wells = {1: "W0/1", 2: "W2/3", 3: "W4/5"}
    ax.set_xticklabels([f"P{p}\n{patch_wells[p]}" for p in patches], fontsize=5)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_ylabel("post. mass", fontsize=6)
    ymax = float(by_patch.max()) or 1.0
    ax.set_ylim(0, ymax * 1.35)
    if et_row is not None:
        flags = {
            "cur": et_row.get("current_patch"),
            "prev": et_row.get("prev_patch"),
            "next": et_row.get("next_patch"),
        }
        labels: dict[int, list] = {}
        for name, val in flags.items():
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                labels.setdefault(int(val), []).append(name)
        for patch, names in labels.items():
            if patch in patches:
                ax.annotate(
                    "\n".join(names),
                    (patch, by_patch.get(patch, 0.0)),
                    fontsize=4.2,
                    color="black",
                    ha="center",
                    va="bottom",
                    xytext=(0, 2),
                    textcoords="offset points",
                )


def _current_trial(trials: pd.DataFrame, t: float) -> pd.Series | None:
    """The trial the regression assigns to time ``t``: latest trial started.

    Matches ``_add_trial_context`` (``searchsorted`` on trial start times), so
    its ``is_reward`` is exactly the event's ``reward_outcome`` covariate.
    """
    if trials is None or not len(trials):
        return None
    started = trials.sort_values("start_time")
    started = started[started["start_time"] <= t]
    return started.iloc[-1] if len(started) else None


def _plot_reward_timeline(ax, data: dict, center: float) -> None:
    """Trial-level reward — exactly what the regression uses.

    The regression scores reward at the *trial* level: ``reward_outcome`` is
    the current trial's ``is_reward`` (True if ANY poke in that trial's stay
    delivered reward), and ``time_since_reward`` counts from the most recent
    pump. A trial = one run + the whole stay at the destination well, so a
    rewarded trial can contain many unrewarded pokes — this panel reflects the
    trial label the model sees, not any single poke. Shows the trial reward
    status and the most recent pump-DIO (green ▼) reaching back to the last
    reward, relative to the event center.
    """
    ax.set_title("reward: trial-level (regression)", fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("Time from event center (s)", fontsize=6)
    ax.tick_params(axis="x", labelsize=5)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvspan(-WINDOW_SEC, WINDOW_SEC, color="magenta", alpha=0.18, zorder=2)
    ax.axvline(0, color="magenta", lw=0.8, zorder=3)

    trial = _current_trial(data.get("trials"), center)
    if trial is None:
        ax.text(
            0.5,
            0.5,
            "no trial",
            fontsize=6,
            ha="center",
            va="center",
            color="gray",
            transform=ax.transAxes,
        )
        return

    rewarded = bool(trial.get("is_reward", False))
    fw, tw = int(trial["from_well"]), int(trial["to_well"])

    # Most recent pump-DIO before the event = the reward the regression's
    # time_since_reward counts from.
    pumps = data["pump_events"]
    prior = pumps[pumps["start_time"] <= center]
    last_pump_rel = float(prior["start_time"].max()) - center if len(prior) else None

    lo = min(last_pump_rel if last_pump_rel is not None else -WINDOW_SEC, -WINDOW_SEC)
    ax.set_xlim(lo * 1.1 - 0.3, WINDOW_SEC + 0.3)

    status = "REWARDED" if rewarded else "OMISSION"
    status_color = "green" if rewarded else "red"
    ax.annotate(
        f"trial W{fw}→W{tw}: {status}  (reward_outcome={int(rewarded)})",
        (0.5, 0.92),
        fontsize=5.3,
        color=status_color,
        ha="center",
        va="top",
        xycoords="axes fraction",
    )
    if last_pump_rel is not None:
        ax.scatter(
            [last_pump_rel],
            [0],
            marker="v",
            s=45,
            color="limegreen",
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
        )
        ax.annotate(
            f"last reward\n{-last_pump_rel:.0f}s ago",
            (last_pump_rel, 0.12),
            fontsize=4.6,
            color="green",
            ha="center",
            va="bottom",
            xycoords=("data", "axes fraction"),
        )
    else:
        ax.annotate(
            "no prior reward",
            (0.5, 0.5),
            fontsize=6,
            color="red",
            ha="center",
            va="center",
            xycoords="axes fraction",
        )


def _plot_decoding_stack(
    axes: list,
    data: dict,
    results,
    detector,
    env,
    time_slice: slice,
    center: float,
    column_title: str,
) -> None:
    """Six time-aligned rows: SWR, MUA, raster, posterior, state probs, speed.

    Extends the Figure-2a four-row stack with a multiunit (MUA) row — since
    ``is_mua`` is a regression covariate — and a speed row with the immobility
    threshold, since the endpoints-immobility rule is a core inclusion filter.
    """
    ax_swr, ax_mua, ax_raster, ax_post, ax_probs, ax_speed = axes

    plot_ripple_consensus_trace_zscore(data, time_slice, ax_swr)
    plot_multiunit_firing(
        data, time_slice, ax_mua, plot_multiunit_zscore=True, plot_thresholds=True
    )
    plot_spike_times(data, time_slice, detector, env, results, ax_raster)
    plot_conditional_non_local_posterior(
        data, time_slice, results, env, detector, ax_post
    )
    plot_acausal_probabilities(results, time_slice, ax_probs, plot_local=True)
    plot_speed(data, time_slice, ax_speed)
    ax_speed.axhline(
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        color="red",
        ls="--",
        lw=0.8,
    )

    for ax in axes:
        ax.set_xlim(time_slice.start, time_slice.stop)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Bottom row (speed) gets event-centered tick labels.
    half = (time_slice.stop - time_slice.start) / 2.0
    offsets = np.arange(-0.4, 0.4 + 1e-9, 0.2)
    offsets = offsets[np.abs(offsets) <= half + 1e-9]
    ax_speed.set_xticks(center + offsets)
    ax_speed.set_xticklabels([f"{o:.1f}" for o in offsets])
    ax_speed.set_xlabel("Time from event center (s)")
    ax_swr.set_title(column_title, fontsize=8, pad=6)


def _arm_set(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "?"
    return str(int(value)) if not isinstance(value, str) else value


def _regression_footer(row: pd.Series, et_row: pd.Series | None) -> str:
    """Multi-line text describing how this event is scored in the regression."""
    cls = row["category"]
    if cls == "swr_not_nonlocal":
        return (
            f"SWR event #{row['event_label']} · {row['duration_ms']:.0f} ms\n"
            "NOT in non-local regression (SWR with no non-local event)\n"
            "enters stop-level models only (is_swr_stop / replay_frac_swr)\n"
            f"peak non-local prob in window: {row['peak_nonlocal_prob']:.2f}"
        )
    is_swr = cls in ("swr_contained", "swr_partial")
    head = (
        f"event_id {row['event_id']} · {row['duration_ms']:.0f} ms · "
        f"peak prob {row['peak_nonlocal_prob']:.2f}\n"
        f"overlap: {cls} -> is_swr={is_swr}"
    )
    if is_swr:
        head += " (pooled)"
    if et_row is None:
        return head + "\nNOT in M0-M7 population (failed in-event filter)"
    arms = (
        f"arms:  current={_arm_set(et_row.get('current_arm'))}"
        f"  prev={_arm_set(et_row.get('prev_arm'))}"
        f"  next={_arm_set(et_row.get('next_arm'))}"
        f"  last_rew={_arm_set(et_row.get('last_rewarded_arm'))}"
    )
    patch = (
        f"patch: current={_arm_set(et_row.get('current_patch'))}"
        f"  prev={_arm_set(et_row.get('prev_patch'))}"
        f"  next={_arm_set(et_row.get('next_patch'))}"
        f"  is_mua={bool(et_row.get('is_mua'))}"
    )
    ctx = (
        f"ctx: after_switch={bool(et_row.get('is_after_switch'))}"
        f" after_reward={bool(et_row.get('is_after_reward'))}"
        f" after_omission={bool(et_row.get('is_after_omission'))}"
    )
    tsr = et_row.get("time_since_reward")
    tsr_str = (
        f"{float(tsr):.1f}s"
        if tsr is not None and not (isinstance(tsr, float) and np.isnan(tsr))
        else "n/a"
    )
    outcome = et_row.get("reward_outcome")
    outcome_str = "rewarded" if outcome == 1 else "omission" if outcome == 0 else "n/a"
    rew = f"reward: outcome={outcome_str}  time_since_reward={tsr_str}"
    return f"{head}\nIN M0-M7 population\n{arms}\n{patch}\n{ctx}\n{rew}"


def _render_event(row: pd.Series, ctx: dict, out_dir: Path, overwrite: bool) -> bool:
    center = (row["plot_start"] + row["plot_end"]) / 2.0
    cls = row["category"]
    fname = f"{row['epoch_id']}_{cls}_lbl{row['event_label']}_{center:.2f}s.png"
    out_path = out_dir / cls / fname
    if out_path.exists() and not overwrite:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)

    et_row = None
    if cls != "swr_not_nonlocal" and ctx["event_table"] is not None:
        match = ctx["event_table"][ctx["event_table"]["event_id"] == row["event_id"]]
        if len(match):
            et_row = match.iloc[0]

    time_slice = slice(center - WINDOW_SEC, center + WINDOW_SEC)
    arm_rows = None
    if (
        cls != "swr_not_nonlocal"
        and ctx["arm_table"] is not None
        and et_row is not None
    ):
        arm_rows = ctx["arm_table"][ctx["arm_table"]["event_id"] == row["event_id"]]

    fig = plt.figure(figsize=(9.0, 13.5), dpi=200)
    # Two outer rows: the panels (two columns) on top, a full-width
    # well-visit/reward history strip at the bottom.
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[7.0, 1.0],
        hspace=0.12,
        top=0.96,
        bottom=0.04,
        left=0.06,
        right=0.97,
    )
    gs_top = outer[0].subgridspec(1, 2, width_ratios=[1.3, 1.35], wspace=0.28)

    # Left column packed to fill the full height: a large 2D context, the
    # regression-scoring text, the per-arm content bar and regressor heatmap,
    # the per-patch content bar, and the reward timeline.
    gs_left = gs_top[0].subgridspec(
        6, 1, height_ratios=[3.0, 0.95, 1.2, 0.85, 1.2, 1.05], hspace=0.32
    )
    ax_2d = fig.add_subplot(gs_left[0])
    _plot_2d_context(
        ax_2d, ctx["data"], center, WINDOW_SEC, ctx["arm_of_edge"], et_row=et_row
    )
    ax_text = fig.add_subplot(gs_left[1])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        _regression_footer(row, et_row),
        fontsize=6.5,
        family="monospace",
        va="top",
        ha="left",
        transform=ax_text.transAxes,
    )
    _plot_arm_content_bar(
        fig.add_subplot(gs_left[2]), arm_rows, et_row, ctx["arm_colors"]
    )
    _plot_arm_predictor_heatmap(fig.add_subplot(gs_left[3]), arm_rows)
    _plot_patch_content_bar(fig.add_subplot(gs_left[4]), arm_rows, et_row)
    _plot_reward_timeline(fig.add_subplot(gs_left[5]), ctx["data"], center)

    # Right column: 6-row decoding stack (SWR / MUA / raster / posterior /
    # state probs / speed).
    gs_stack = gs_top[1].subgridspec(
        6, 1, height_ratios=[0.5, 0.5, 0.85, 1.15, 0.7, 0.5], hspace=0.10
    )
    axes = [fig.add_subplot(gs_stack[r]) for r in range(6)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    title = f"{cls} · {row['duration_ms']:.0f} ms"
    _plot_decoding_stack(
        axes,
        ctx["data"],
        ctx["results"],
        ctx["detector"],
        ctx["env"],
        time_slice,
        center,
        title,
    )

    # 1D linearized track-graph strip flush to the posterior (its y-key).
    ax_post = axes[3]
    ax_strip = ax_post.inset_axes([-0.18, 0.0, 0.16, 1.0])
    max_position = float(np.ceil(ctx["data"]["position_info"].linear_position.max()))
    plot_linear_track_graph_strip(
        ctx["data"]["track_graph"],
        ctx["data"]["linear_edge_order"],
        ctx["data"]["linear_edge_spacing"],
        ax=ax_strip,
        max_position=max_position,
    )
    ax_post.set_ylabel("")
    ax_post.tick_params(labelleft=False)

    # Bottom: full-width well-visit / reward history grid.
    _plot_trial_history(fig.add_subplot(outer[1]), ctx["data"], center)

    fig.savefig(out_path)
    plt.close(fig)
    return True


def render_epoch(
    epoch_id: str, data_dir: Path, gallery_dir: Path, overwrite: bool
) -> None:
    catalog = pd.read_parquet(gallery_dir / "_catalog.parquet")
    sel = catalog[catalog["epoch_id"] == epoch_id]
    if not len(sel):
        logging.info("No selected events for %s — nothing to render.", epoch_id)
        return

    nwb_file_name, epoch_name = parse_epoch_id(epoch_id)
    logging.info("Loading %s %s (%d events)...", nwb_file_name, epoch_name, len(sel))
    data = load_data(nwb_file_name, epoch_name)
    results, detector, env = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC"
    )

    # Predictors + per-arm content for the footer and content bar (NL
    # categories only).
    event_table = None
    arm_table = None
    if (sel["category"] != "swr_not_nonlocal").any():
        dot = pd.read_pickle(data_dir / f"{epoch_id}_HPC_80_data_over_time.pkl")
        posterior = load_conditional_posterior(
            PROCESSED_DATA_DIR / f"{epoch_id}_HPC_results.nc"
        )
        event_table = build_event_table(
            dot,
            env,
            data["trials"],
            data["track_graph"],
            conditional_posterior=posterior,
        )
        arm_table = build_event_arm_table(
            event_table,
            dot,
            env,
            data["trials"],
            data["track_graph"],
            conditional_posterior=posterior,
        )

    arm_of_edge = _arm_of_edge_map()
    ctx = {
        "data": data,
        "results": results,
        "detector": detector,
        "env": env,
        "event_table": event_table,
        "arm_table": arm_table,
        "arm_of_edge": arm_of_edge,
        "arm_colors": _arm_colors(data["linear_edge_order"], arm_of_edge),
    }
    n_written = 0
    for _, row in sel.iterrows():
        n_written += int(_render_event(row, ctx, gallery_dir, overwrite))
    logging.info("Rendered %d/%d events for %s.", n_written, len(sel), epoch_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--gallery-dir", type=Path, default=DEFAULT_GALLERY_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cat = sub.add_parser("catalog", help="Phase 1: build the selection catalog.")
    p_cat.add_argument(
        "--n-per-category",
        type=int,
        default=DEFAULT_N_PER_CATEGORY,
        help="Events per category per animal (default 50).",
    )

    p_ren = sub.add_parser("render", help="Phase 2: render one epoch's events.")
    p_ren.add_argument("--epoch-id", type=str, required=True)
    p_ren.add_argument("--overwrite", action="store_true")

    sub.add_parser("list-epochs", help="Print epoch-ids present in the catalog.")

    args = parser.parse_args()

    if args.command == "catalog":
        build_catalog(args.data_dir, args.gallery_dir, args.n_per_category)
    elif args.command == "render":
        render_epoch(args.epoch_id, args.data_dir, args.gallery_dir, args.overwrite)
    elif args.command == "list-epochs":
        catalog = pd.read_parquet(args.gallery_dir / "_catalog.parquet")
        for eid in sorted(catalog["epoch_id"].unique()):
            print(eid)


if __name__ == "__main__":
    main()
