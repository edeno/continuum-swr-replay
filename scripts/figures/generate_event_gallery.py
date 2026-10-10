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
    python scripts/figures/generate_event_gallery.py catalog

    # Phase 2: render one epoch's selected events (the tmux unit)
    python scripts/figures/generate_event_gallery.py render --epoch-id chimi20200213_02_r1

    # List the epoch-ids that have selected events (for the launcher)
    python scripts/figures/generate_event_gallery.py list-epochs
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
from continuum_swr_replay.analysis.hippocampus.event_detection import (
    positive_duration_interval_mask,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    filter_event_labels_by_speed,
    intervals_from_labels,
)
from continuum_swr_replay.analysis.hippocampus.event_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (
    read_dot_with_swr_source,
    ripple_power_inputs,
    swr_epoch_id,
    swr_intervals,
)
from continuum_swr_replay.analysis.replay.bouts import (
    immobility_bout_intervals,
)
from continuum_swr_replay.analysis.replay.covariates import (
    trial_arrival_times,
    trial_outcome_times,
)
from continuum_swr_replay.analysis.replay.posterior_io import (
    load_conditional_posterior,
)
from continuum_swr_replay.analysis.replay.tables_arm_patch import build_event_arm_table
from continuum_swr_replay.analysis.replay.tables_event import build_event_table
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from continuum_swr_replay.parameters import COLORS, PROCESSED_DATA_DIR
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


def load_data(*args, **kwargs):
    """Import the Spyglass-backed loader only for the render phase."""
    from continuum_swr_replay.data_loaders import load_data as _load_data

    return _load_data(*args, **kwargs)


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
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    other_starts = np.asarray(other_starts, dtype=float)
    other_ends = np.asarray(other_ends, dtype=float)
    valid = positive_duration_interval_mask(starts, ends)
    valid_other = positive_duration_interval_mask(other_starts, other_ends)
    if not valid_other.any():
        return np.zeros(len(starts), dtype=bool)
    s = starts[:, None]
    e = ends[:, None]
    os = other_starts[valid_other][None, :]
    oe = other_ends[valid_other][None, :]
    return valid & ((s < oe) & (os < e)).any(axis=1)


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

    dot = read_dot_with_swr_source(pkl_path)
    dot["Non-Local_prob"] = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    )
    eligible_dot = filter_event_labels_by_speed(
        dot,
        [
            column
            for column in ("non_local_event", "SWR_event", "MUA_event")
            if column in dot
        ],
    )

    ct = build_classified_nonlocal_event_table(dot)
    rows: list[dict] = []

    if len(ct):
        animal = ct["animal"].iloc[0]
        day = int(ct["day"].iloc[0])
        # Immobile only (matches the regression analysis population).
        ct = ct[ct["whole_event_speed_eligible"]].copy()
        ev_dur = (ct["event_end"] - ct["event_start"]).to_numpy()

        swr_intervals = intervals_from_labels(eligible_dot, "SWR_event")
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
        nl_intervals = intervals_from_labels(eligible_dot, "non_local_event")
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
    convention). Used to label arms with the same id
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


def _arm_strip_labels(track_graph, edge_order: list, env) -> list:
    """(arm, linear_center_cm, color) for each arm, matching the 1D strip.

    Derived from a single source so labels can't drift from the strip's
    segments: the arm comes from node identities (``_arm_of_edge_map``), the
    linear center from the edge's own ``edge_id`` graph attribute looked up in
    ``env`` (junction bins, edge_id = -1, excluded), and the color from the
    edge's position in ``edge_order`` — exactly the tab10 index the strip and
    the content bars use. (The ``TRACK_SEGMENT_TO_EDGE_ID`` constant is *not*
    used; it does not match this env's edge_id numbering.)
    """
    edge_colors = _track_edge_colors(len(edge_order))
    arm_of_edge = _arm_of_edge_map()
    df = env.place_bin_centers_nodes_df_
    lp_by_eid = (
        df[df["edge_id"] >= 0].groupby("edge_id")["linear_position"].mean().to_dict()
    )
    out = []
    for i, (n1, n2) in enumerate(edge_order):
        arm = arm_of_edge.get(frozenset((n1, n2)))
        eid = track_graph.edges[n1, n2].get("edge_id")
        if arm is not None and eid in lp_by_eid:
            out.append((arm, lp_by_eid[eid], edge_colors[i]))
    return out


def _plot_2d_context(
    ax,
    data: dict,
    center: float,
    event_start: float,
    window_sec: float,
    arm_of_edge: dict,
    et_row: pd.Series | None = None,
) -> pd.Series | None:
    """2D track, event position, and the last run completed before the event.

    Draws the arm-colored track + faint occupancy (shared color map with the
    1D strip), labels each well node and arm with its id, marks the animal's
    head position during the event window, and highlights the last run whose
    outcome was available by ``event_start``. Returns that trial row (or None).
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

    # Regression-aligned historical path: never substitute the nearest well
    # for ``occupancy_well_arm`` and never reveal an in-progress
    # destination/outcome.
    # The magenta event-position marker below independently shows where the
    # animal physically was.
    completed = _completed_trials_at(data["trials"], event_start)
    trial = completed.iloc[-1] if len(completed) else None
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
        outcome_time = trial_outcome_times(pd.DataFrame([trial]))[0]
        outcome_known = bool(outcome_time <= event_start)
        rewarded = bool(trial.get("is_reward", False))
        rew_color = ("limegreen" if rewarded else "red") if outcome_known else "0.5"
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
        outcome_time = trial_outcome_times(pd.DataFrame([trial]))[0]
        timing = "known" if outcome_time <= event_start else "eventual"
        ax.set_title(
            "last completed run: "
            f"W{int(trial['from_well'])}→W{int(trial['to_well'])} "
            f"({timing} {rew})",
            fontsize=7,
        )
    else:
        ax.set_title("last completed run: none", fontsize=7)
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
    mass = (
        arm_rows.set_index("candidate_arm")["arm_posterior_mass"]
        .reindex(arms)
        .fillna(0.0)
    )
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
            "cur": et_row.get("occupancy_well_arm"),
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


def _completed_trials_at(trials: pd.DataFrame, event_start: float) -> pd.DataFrame:
    """Return trips *completed* (goal arrival) at or before ``event_start``.

    Completion is arrival, not outcome delivery -- a trip completed just
    before the event may have had its reward or omission resolved after
    it. This is a visualization helper; do not reuse it to gate a fit.
    """
    if trials is None or not len(trials):
        return pd.DataFrame() if trials is None else trials.copy()
    completed = trials.copy()
    completed["_completion_time"] = trial_arrival_times(completed)
    return (
        completed[completed["_completion_time"] <= event_start]
        .sort_values("_completion_time", kind="stable")
        .reset_index(drop=True)
    )


def _outcome_known_trials_at(trials: pd.DataFrame, event_start: float) -> pd.DataFrame:
    """Return trials whose reward/omission was observable by ``event_start``."""
    if trials is None or not len(trials):
        return pd.DataFrame() if trials is None else trials.copy()
    known = trials.copy()
    known["_outcome_known_time"] = trial_outcome_times(known)
    return (
        known[known["_outcome_known_time"] <= event_start]
        .sort_values("_outcome_known_time", kind="stable")
        .reset_index(drop=True)
    )


def _plot_trial_history(
    ax, data: dict, event_start: float, arm_rows: pd.DataFrame, n_trials: int = 20
) -> None:
    """Trial-by-trial reward history, marked up with the value covariates.

    Grid of wells (rows W0-W5) × recent trials (columns, time → right). Each
    trial's ``to_well`` is marked with the TRIAL-level reward (``is_reward`` =
    any poke in that trial's stay was rewarded): filled green if rewarded and
    open if an omission. Only outcomes known by the event start are shown.
    Destination/visit history may already include a more recent arrival whose
    outcome is still pending; that trip is deliberately absent here.

    Marked up so the value covariates can be read directly:
      - gold band over the **last 10 known-outcome trials** = the
        ``reward_count_candidate_arm_last10_known_trials`` window.
      - orange rings on each row's **last 10 visits** = the
        ``reward_fraction_candidate_arm_last10_visits`` window.
      - right margin prints the per-arm pipeline values for this event:
        ``vr`` (``visit_recency_trials``), ``rr`` (prior-visit reward
        fraction), and ``cum`` (known-outcome reward count over 10 trials).
    """
    ax.set_title("known-outcome reward history (→ event start)", fontsize=7)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"W{w}" for w in range(6)], fontsize=5)
    ax.set_ylim(-0.6, 6.2)
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
    completed = _outcome_known_trials_at(trials, event_start)
    if not len(completed):
        ax.text(
            0.5,
            0.5,
            "no known outcomes at event start",
            fontsize=6,
            ha="center",
            va="center",
            color="gray",
            transform=ax.transAxes,
        )
        return
    window = completed.tail(n_trials).reset_index(drop=True)

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
    n = len(window)

    # reward_count_candidate_arm_last10_known_trials window: the last 10 trials (gold band).
    band_lo = max(-0.5, n - 10 - 0.5)
    ax.axvspan(band_lo, n - 0.5, color="gold", alpha=0.12, zorder=0)

    # reward_fraction_candidate_arm_last10_visits window: ring each row's last 10 visits (orange).
    for w in range(6):
        cols_w = [p for p in xs if wells[p] == w]
        if cols_w:
            last_k = cols_w[-10:]
            ax.scatter(
                last_k,
                [w] * len(last_k),
                s=130,
                marker="o",
                facecolor="none",
                edgecolor="darkorange",
                linewidth=1.0,
                zorder=2,
            )

    # Right-margin per-arm covariate values (the exact pipeline values for this
    # event). Header column names spelled out; a key below explains them.
    if arm_rows is not None and len(arm_rows):
        by_arm = arm_rows.set_index("candidate_arm")
        xtext = n - 0.2
        ax.set_xlim(-0.5, n - 0.5 + 0.42 * max(n, 1))
        ax.annotate(
            "vr / rr / cum",
            (xtext, 5.9),
            fontsize=6.5,
            color="black",
            ha="left",
            va="bottom",
            fontweight="bold",
        )
        for w in range(6):
            if w not in by_arm.index:
                continue
            r = by_arm.loc[w]
            vr = r.get("visit_recency_trials")
            rr = r.get("reward_fraction_candidate_arm_last10_visits")
            cum = r.get("reward_count_candidate_arm_last10_known_trials")
            vr_s = "·" if pd.isna(vr) else f"{vr:.0f}"
            rr_s = "·" if pd.isna(rr) else f"{rr:.2f}"
            cum_s = "·" if pd.isna(cum) else f"{cum:.0f}"
            ax.annotate(
                f"{vr_s} / {rr_s} / {cum_s}",
                (xtext, w),
                fontsize=6.5,
                color="black",
                ha="left",
                va="center",
                family="monospace",
            )
    else:
        ax.set_xlim(-0.5, max(n - 0.5, 0.5))
    ax.set_xticks(list(range(n))[:: max(1, n // 10)])

    # Plain-language key for the markup, below the x-axis.
    ax.annotate(
        "right column = vr / rr / cum:   "
        "vr = visit_recency [trials since last visit]   ·   "
        "rr = reward_fraction_candidate_arm_last10_visits [last 10 visits = orange rings]   ·   "
        "cum = reward_count_candidate_arm_last10_known_trials [last 10 trials = gold band]",
        (0.5, -0.42),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=5.6,
        color="0.25",
    )

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

    Rows show ``visit_recency_trials_event_leaf_centered``,
    ``is_candidate_arm_never_visited``,
    ``reward_fraction_candidate_arm_last10_visits``, and
    ``is_last_rewarded_noncurrent_arm``; columns are
    arms 0-8. Each row is colored on its own scale (these regressors live on
    different units) with the actual value printed in each cell.
    """
    preds = [
        ("visit_recency_trials_event_leaf_centered", "visit_rec"),
        ("is_candidate_arm_never_visited", "never_vis"),
        ("reward_fraction_candidate_arm_last10_visits", "rew_rate"),
        ("is_last_rewarded_noncurrent_arm", "lastR_elsw"),
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
    indexed = arm_rows.set_index("candidate_arm")
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
        arm_rows.assign(patch=arm_rows["candidate_arm"].map(TRACK_SEGMENT_TO_PATCH))
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


def _active_trip_at_time(trials: pd.DataFrame, t: float) -> pd.Series | None:
    """Return the physical trip interval containing ``t``, if any.

    This is a post-hoc visualization of the animal's run/dwell interval. It is
    deliberately separate from regression context. Destination/visit context
    becomes available at arrival, while reward-bearing context is withheld
    until the separate reward-on or omission latency/departure boundary.
    """
    if trials is None or not len(trials):
        return None
    ordered = trials.sort_values("start_time", kind="stable")
    active = ordered[(ordered["start_time"] <= t) & (ordered["end_time"] >= t)]
    return active.iloc[-1] if len(active) else None


def _plot_behavioral_context(
    ax, data: dict, center: float, row: pd.Series, imm_bouts: tuple
) -> None:
    """Behavioral timeline around the event: trial run/dwell, stop, the event.

    Three lanes vs time-from-event-center:
      - TRIAL: run segment [start→goal_arrival] (solid) + post-arrival dwell
        [goal_arrival→end] (hatched), shown post hoc as physical context.
      - STOP: the immobility bout containing the event (dark grey) + the
        previous bout (light grey) — the bounded stop unit (time_since_stop).
      - EVENT: the event's true extent, in the non-local (or SWR) color.

    Showing the trial dwell and the immobility bout together makes the
    degenerate-trial mismatch visible: in a disengagement tail the dwell bar
    spans minutes while the stop bout stays short. The most recent reward pump
    (green ▼) and the active trip's eventual outcome are annotated. The
    annotation explicitly marks whether that outcome was already known.
    """
    cls = row["category"]
    ev_color = COLORS["SWR"] if cls == "swr_not_nonlocal" else COLORS["Non-Local"]
    lane_event, lane_stop, lane_trial = 0, 1, 2
    ax.set_title("behavioral context (time from event center)", fontsize=7)
    ax.set_xlabel("Time from event center (s)", fontsize=6)
    ax.tick_params(axis="x", labelsize=5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["event", "stop", "trial"], fontsize=5.5)
    ax.set_ylim(-0.6, 3.2)
    ax.axvline(0, color="black", lw=0.6, zorder=1)
    xs = [0.0]

    trial = _active_trip_at_time(data.get("trials"), center)
    rewarded = None
    outcome_known = False
    if trial is not None:
        ga = trial.get("goal_arrival_time")
        if ga is None or (isinstance(ga, float) and np.isnan(ga)):
            ga = trial["end_time"]
        rs, gr, te = (
            trial["start_time"] - center,
            float(ga) - center,
            trial["end_time"] - center,
        )
        rewarded = bool(trial.get("is_reward", False))
        outcome_time = trial_outcome_times(pd.DataFrame([trial]))[0]
        outcome_known = bool(outcome_time <= center)
        ax.barh(lane_trial, gr - rs, left=rs, height=0.5, color="steelblue", zorder=3)
        ax.barh(
            lane_trial,
            te - gr,
            left=gr,
            height=0.5,
            facecolor="none",
            edgecolor="steelblue",
            hatch="////",
            lw=0.4,
            zorder=3,
        )
        ax.annotate(
            f"run W{int(trial['from_well'])}→W{int(trial['to_well'])} | dwell",
            (rs, lane_trial + 0.32),
            fontsize=4.3,
            color="steelblue",
            va="bottom",
        )
        # goal arrival (well reached / reward site) — distinct from the
        # speed-defined immobility onset (start of the grey stop bar).
        ax.axvline(gr, color="darkorange", ls=":", lw=0.9, zorder=2)
        ax.annotate(
            "arrival",
            (gr, 2.55),
            fontsize=4.2,
            color="darkorange",
            ha="center",
            va="bottom",
        )
        outcome_rel = float(outcome_time) - center
        ax.axvline(outcome_rel, color="purple", ls="--", lw=0.8, zorder=2)
        ax.annotate(
            "outcome available",
            (outcome_rel, 2.75),
            fontsize=4.0,
            color="purple",
            ha="center",
            va="bottom",
        )
        xs += [rs, te, outcome_rel]

    starts, ends = np.asarray(imm_bouts[0]), np.asarray(imm_bouts[1])
    if len(starts):
        cont = np.where((starts <= center) & (ends >= center))[0]
        if len(cont):
            i = cont[-1]
            ax.barh(
                lane_stop,
                ends[i] - starts[i],
                left=starts[i] - center,
                height=0.5,
                color="0.55",
                zorder=3,
            )
            xs += [starts[i] - center, ends[i] - center]
        prev = np.where(ends <= center)[0]
        if len(prev):
            i = prev[-1]
            ax.barh(
                lane_stop,
                ends[i] - starts[i],
                left=starts[i] - center,
                height=0.5,
                color="0.82",
                zorder=2,
            )
            xs += [starts[i] - center]

    es, ee = row["plot_start"] - center, row["plot_end"] - center
    ax.barh(
        lane_event, max(ee - es, 0.004), left=es, height=0.5, color=ev_color, zorder=4
    )
    xs += [es, ee]

    pumps = data["pump_events"]
    prior = pumps[pumps["start_time"] <= center]
    if len(prior):
        lp = float(prior["start_time"].max()) - center
        ax.scatter(
            [lp],
            [2.85],
            marker="v",
            s=40,
            color="limegreen",
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
        )
        ax.annotate(
            "reward", (lp, 2.95), fontsize=4.3, color="green", ha="center", va="bottom"
        )
        xs += [lp]

    lo, hi = min(xs), max(xs)
    pad = 0.05 * (hi - lo) + 0.3
    ax.set_xlim(lo - pad, hi + pad)
    win_lo, win_hi = center + lo - pad, center + hi + pad

    # Poke (beam-break) marks — reward is poke-triggered, so these show when
    # the animal actually poked the well. Use merged poke visits (is_run False).
    visits = data.get("well_visits")
    if visits is not None and len(visits):
        pk = visits[
            (~visits["is_run"].astype(bool))
            & (visits["start_time"] >= win_lo)
            & (visits["start_time"] <= win_hi)
        ]
        if len(pk):
            ax.plot(
                pk["start_time"].to_numpy() - center,
                np.full(len(pk), -0.45),
                marker="|",
                ls="",
                ms=5,
                color="black",
                zorder=5,
            )
            ax.annotate(
                "pokes",
                (lo - pad, -0.45),
                fontsize=4.0,
                color="black",
                ha="left",
                va="center",
            )

    # Speed trace over the window (secondary axis): shows the run, the
    # deceleration, and the immobility-threshold crossing that defines the stop.
    pos = data.get("position_info")
    if pos is not None and "head_speed" in pos.columns:
        seg = pos.loc[win_lo:win_hi]
        if len(seg):
            step = max(1, len(seg) // 2000)
            ax2 = ax.twinx()
            ax2.plot(
                seg.index.to_numpy()[::step] - center,
                seg["head_speed"].to_numpy()[::step],
                color="0.45",
                lw=0.5,
                alpha=0.7,
                zorder=0,
            )
            ax2.axhline(
                DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
                color="red",
                ls="--",
                lw=0.5,
                alpha=0.7,
            )
            ax2.set_ylim(0, max(20, float(seg["head_speed"].max())))
            ax2.set_ylabel("speed (cm/s)", fontsize=5, color="0.4")
            ax2.tick_params(axis="y", labelsize=4.5, colors="0.4")

    if rewarded is not None:
        timing = "known" if outcome_known else "eventual"
        ax.annotate(
            f"{timing} " + ("REWARDED" if rewarded else "OMISSION"),
            (0.99, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=5.5,
            color="green" if rewarded else "red",
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
    threshold, since whole-event immobility is a core inclusion filter.
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
    if value is None or pd.isna(value):
        return "?"
    return str(int(value)) if not isinstance(value, str) else value


def _truth_label(value) -> str:
    """Format nullable booleans without treating unknown as true."""
    if value is None or pd.isna(value):
        return "n/a"
    return str(bool(value)).lower()


def _regression_footer(row: pd.Series, et_row: pd.Series | None) -> str:
    """Multi-line text describing how this event is scored in the regression."""
    cls = row["category"]
    if cls == "swr_not_nonlocal":
        return (
            f"SWR event #{row['event_label']} · {row['duration_ms']:.0f} ms\n"
            "NOT in non-local regression (SWR with no non-local event)\n"
            "enters stop-level models only (is_swr_stop / replay_fraction_swr)\n"
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
        return head + "\nNOT in canonical event table (failed in-event filter)"
    arms = (
        f"arms:  location={_arm_set(et_row.get('occupancy_well_arm'))}"
        f"  prev={_arm_set(et_row.get('prev_arm'))}"
        f"  next(post-hoc)={_arm_set(et_row.get('next_arm'))}"
        f"  last_rew={_arm_set(et_row.get('last_rewarded_arm'))}"
    )
    patch = (
        f"patch: location={_arm_set(et_row.get('current_patch'))}"
        f"  prev={_arm_set(et_row.get('prev_patch'))}"
        f"  next(post-hoc)={_arm_set(et_row.get('next_patch'))}"
        f"  is_mua={_truth_label(et_row.get('is_mua'))}"
    )
    ctx = (
        f"ctx: after_switch={_truth_label(et_row.get('did_last_completed_trip_switch_patch'))}"
        f" rewarded_outcome={_truth_label(et_row.get('is_rewarded_outcome'))}"
    )
    tsr = et_row.get("time_since_last_reward_s")
    tsr_str = f"{float(tsr):.1f}s" if tsr is not None and not pd.isna(tsr) else "n/a"
    outcome = et_row.get("is_rewarded_outcome")
    outcome_str = (
        "n/a"
        if outcome is None or pd.isna(outcome)
        else "rewarded"
        if outcome == 1
        else "omission"
        if outcome == 0
        else "n/a"
    )
    rew = f"reward: outcome={outcome_str}  time_since_reward={tsr_str}"
    # Presence in event_table is not equivalent to membership in a fitted
    # model cohort: complete-case selection can still exclude an event (for
    # example, when current-arm context is undefined off-well).  Keep this QC
    # footer factual and leave model-cohort membership to the persisted fit
    # audit rather than inferring it from table presence.
    return f"{head}\nIN canonical event table\n{arms}\n{patch}\n{ctx}\n{rew}"


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

    fig = plt.figure(figsize=(9.0, 14.0), dpi=200)
    # Two outer rows: the panels (two columns) on top, a full-width
    # trial reward-history strip at the bottom (taller, to fit the covariate
    # key and per-arm values).
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[6.2, 1.7],
        hspace=0.14,
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
    regression_event_start = (
        float(et_row["event_start_s"])
        if et_row is not None and pd.notna(et_row.get("event_start_s"))
        else float(row["plot_start"])
    )
    _plot_2d_context(
        ax_2d,
        ctx["data"],
        center,
        regression_event_start,
        WINDOW_SEC,
        ctx["arm_of_edge"],
        et_row=et_row,
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
    _plot_behavioral_context(
        fig.add_subplot(gs_left[5]), ctx["data"], center, row, ctx["imm_bouts"]
    )

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
    # Label each arm (a0-a8) at its linear-position center on the strip, so the
    # posterior's y-axis can be read as track arms (matching the content bars).
    for arm, lp, color in _arm_strip_labels(
        ctx["data"]["track_graph"], ctx["data"]["linear_edge_order"], ctx["env"]
    ):
        ax_strip.annotate(
            f"a{arm}",
            (-0.8, lp),
            fontsize=4.2,
            color=color,
            ha="center",
            va="center",
            fontweight="bold",
        )
    ax_post.set_ylabel("")
    ax_post.tick_params(labelleft=False)

    # Bottom: full-width trial reward-history grid, marked up with the
    # value covariates (visit_recency_trials / reward_fraction_candidate_arm_last10_visits /
    # reward_count_candidate_arm_last10_known_trials).
    _plot_trial_history(
        fig.add_subplot(outer[1]), ctx["data"], regression_event_start, arm_rows
    )

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
    # the ripple trace is the source's (the Kay consensus over R-T's tetrodes)
    data = load_data(
        nwb_file_name,
        epoch_name,
        ripple_power=ripple_power_inputs(swr_epoch_id(nwb_file_name, epoch_name)),
    )
    results, detector, env = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC"
    )

    # ``dot`` is always loaded: it is cheap relative to load_data and the
    # immobility bouts (stop unit) it provides are used by every event's
    # behavioral-context panel. Per-arm predictors/content additionally need
    # the conditional posterior (NL categories only).
    dot = read_dot_with_swr_source(data_dir / f"{epoch_id}_HPC_80_data_over_time.pkl")
    imm_bouts = immobility_bout_intervals(dot)
    # The SWR trace's shading: the epoch's SWR source intervals, the ones its
    # events were classified against (not load_data's stored ripples).
    data["ripple_times"] = swr_intervals(dot)

    event_table = None
    arm_table = None
    if (sel["category"] != "swr_not_nonlocal").any():
        posterior = load_conditional_posterior(
            PROCESSED_DATA_DIR / f"{epoch_id}_HPC_results.nc"
        )
        event_table = build_event_table(
            dot,
            env,
            data["trials"],
            data["port_experiences"],
            data["track_graph"],
            conditional_posterior=posterior,
        )
        arm_table = build_event_arm_table(
            event_table,
            dot,
            env,
            data["trials"],
            data["port_experiences"],
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
        "imm_bouts": imm_bouts,
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
