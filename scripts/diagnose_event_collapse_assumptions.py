#!/usr/bin/env python
"""Diagnose the two strongest assumptions baked into ``collapse_to_events``.

For each non-local event in a sample of real sessions, compute:

A. **Trial-boundary straddling rate.** Within an event's per-timebin
   rows, how many *distinct* values of ``prev_trial_from_well`` appear?
   If always 1, the modal-context choice is unambiguous. If frequently
   >1, the modal choice silently misassigns context for events that
   span a trial boundary.

B. **Trajectory fragmentation rate.** Within an event, what fraction
   of consecutive timebin pairs (t, t+1) have *different* decoded arms?
   ~0 means smooth-trajectory events (multinomial-independence
   assumption is badly violated; Tier-B SE under-coverage is large).
   ~0.9 means salt-and-pepper events (multinomial assumption is fine;
   per-arm counts genuinely summarise the event).

Outputs:
- ``paper/figures/qc_diagnosis/event_collapse_assumptions.{png,pdf}``
- ``paper/figures/qc_diagnosis/event_collapse_assumptions.csv`` (per-session
  summary table for the supplement)

Usage:
    uv run python scripts/diagnose_event_collapse_assumptions.py \
        --max-sessions 20

Notes:
- Loading trials goes through Spyglass and is slow (~30s/session). Cap
  ``--max-sessions`` for a quick check.
- The script reuses the production ``prepare_session`` plumbing only up
  to the ``nl`` per-timebin frame (we need pre-collapse data).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.data_preparation import (
    add_trial_context,
)
from continuum_swr_replay.analysis.replay.spatial_mapping import (
    map_estimate_to_arm,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.types import SessionDataKeys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_sessions(max_sessions: int | None = None) -> list[dict]:
    """Return a list of session metadata dicts with paths to dot/env pickles."""
    dot_files = sorted(PROCESSED_DATA_DIR.glob("*_HPC_80_data_over_time.pkl"))
    sessions = []
    for dot_path in dot_files:
        # Filename pattern: {animal}{date}_{epoch}_HPC_80_data_over_time.pkl
        stem = dot_path.stem.replace("_80_data_over_time", "")
        # Strip "_HPC" suffix
        base = stem.removesuffix("_HPC")
        env_path = PROCESSED_DATA_DIR / f"{base}_HPC_environment.pkl"
        if not env_path.exists():
            continue
        # Reconstruct nwb name and epoch
        # base like "chimi20200213_02_r1" → animal up to first digit
        # Find first digit
        idx = next((i for i, c in enumerate(base) if c.isdigit()), None)
        if idx is None:
            continue
        animal = base[:idx]
        date_epoch = base[idx:]  # "20200213_02_r1"
        date, epoch = date_epoch.split("_", 1)
        nwb = f"{animal}{date}_.nwb"
        sessions.append(
            {
                "animal": animal,
                "date": date,
                "epoch": epoch,
                "nwb": nwb,
                "dot_file": dot_path,
                "env_file": env_path,
            }
        )
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    return sessions


def diagnose_session(dot, env, trials) -> pd.DataFrame:
    """Return a per-event diagnostic frame for one session.

    Columns:
        non_local_event, n_bins, n_unique_prev_well, straddles_boundary,
        n_arm_transitions, n_consecutive_pairs, fragmentation_rate
    """
    dot = dot.copy()
    dot["replay_arm"] = map_estimate_to_arm(dot["map_estimate"].values, env)
    dot["Non-Local_prob"] = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    )
    dot = add_trial_context(dot, trials)
    nl = dot[
        (dot["head_speed"] < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC)
        & (dot["Non-Local_prob"] > DEFAULT_NON_LOCAL_THRESHOLD)
        & (dot["non_local_event"] > 0)
    ].copy()

    if len(nl) == 0:
        return pd.DataFrame()

    rows = []
    for ev_id, g in nl.groupby("non_local_event"):
        # (A) Trial-boundary straddling
        unique_prev = g["prev_trial_from_well"].dropna().unique()
        n_unique = int(len(unique_prev))
        # (B) Trajectory fragmentation
        arms = g["replay_arm"].values  # may include NaN
        valid_pairs = ~(np.isnan(arms[:-1]) | np.isnan(arms[1:]))
        n_pairs = int(valid_pairs.sum())
        n_transitions = int(
            (arms[:-1][valid_pairs] != arms[1:][valid_pairs]).sum()
        )
        rows.append(
            {
                "non_local_event": int(ev_id),
                "n_bins": int(len(g)),
                "n_unique_prev_well": n_unique,
                "straddles_boundary": int(n_unique > 1),
                "n_arm_transitions": n_transitions,
                "n_consecutive_pairs": n_pairs,
                "fragmentation_rate": (
                    n_transitions / n_pairs if n_pairs > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(events_all: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # (A) Histogram of n_unique_prev_well per event
    ax = axes[0]
    counts = events_all["n_unique_prev_well"].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(int), counts.values, color="#3b6ea5")
    ax.set_xticks(counts.index.astype(int))
    ax.set_xlabel("Distinct prev_trial_from_well per event")
    ax.set_ylabel("Number of events")
    ax.set_title(
        "(A) Trial-boundary straddling\n"
        f"{events_all['straddles_boundary'].mean() * 100:.1f}% of events span >1 trial"
    )
    for b, v in zip(bars, counts.values, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # (B) Histogram of fragmentation_rate
    ax = axes[1]
    valid = events_all["fragmentation_rate"].dropna()
    ax.hist(valid, bins=30, range=(0, 1), color="#a55a3b", edgecolor="white")
    ax.axvline(
        valid.mean(),
        color="black",
        linestyle="--",
        label=f"mean = {valid.mean():.2f}",
    )
    ax.set_xlabel("Fragmentation rate (fraction of arm transitions per event)")
    ax.set_ylabel("Number of events")
    ax.set_title(
        "(B) Trajectory smoothness\n"
        f"low values = smooth trajectories (multinomial independence violated)"
    )
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"event_collapse_assumptions.{ext}",
            dpi=200 if ext == "png" else None,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-sessions", type=int, default=10, help="Cap number of sessions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/qc_diagnosis"),
    )
    args = parser.parse_args()

    sessions = find_sessions(max_sessions=args.max_sessions)
    logger.info("Diagnosing %d sessions", len(sessions))

    all_events = []
    per_session_rows = []
    t0 = time.time()
    for i, s in enumerate(sessions):
        try:
            dot = pd.read_pickle(s["dot_file"])
            with open(s["env_file"], "rb") as f:
                env = pickle.load(f)
            data = load_data(
                s["nwb"], s["epoch"], include=frozenset({SessionDataKeys.TRIALS})
            )
            trials = data.get("trials")
        except Exception as e:
            logger.warning("Skip %s %s: %s", s["nwb"], s["epoch"], e)
            continue
        if trials is None or len(trials) < 3:
            continue

        ev = diagnose_session(dot, env, trials)
        if len(ev) == 0:
            continue

        ev["animal"] = s["animal"]
        ev["date"] = s["date"]
        ev["epoch"] = s["epoch"]
        all_events.append(ev)

        per_session_rows.append(
            {
                "animal": s["animal"],
                "date": s["date"],
                "epoch": s["epoch"],
                "n_events": len(ev),
                "pct_straddling": float(ev["straddles_boundary"].mean() * 100),
                "median_fragmentation": float(
                    ev["fragmentation_rate"].dropna().median()
                ),
                "mean_fragmentation": float(
                    ev["fragmentation_rate"].dropna().mean()
                ),
            }
        )
        elapsed = time.time() - t0
        logger.info(
            "[%d/%d] %s %s %s — %d events, %.1f%% straddle, "
            "median frag %.2f (elapsed %.1fs)",
            i + 1,
            len(sessions),
            s["animal"],
            s["date"],
            s["epoch"],
            len(ev),
            per_session_rows[-1]["pct_straddling"],
            per_session_rows[-1]["median_fragmentation"],
            elapsed,
        )

    if not all_events:
        logger.error("No events diagnosed; exiting.")
        return

    events_all = pd.concat(all_events, ignore_index=True)
    summary = pd.DataFrame(per_session_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "event_collapse_assumptions.csv", index=False)
    events_all.to_pickle(args.output_dir / "event_collapse_assumptions_events.pkl")
    make_figure(events_all, args.output_dir)

    print()
    print("=" * 70)
    print(f"Total events: {len(events_all):,}")
    print(
        f"% straddling trial boundary: "
        f"{events_all['straddles_boundary'].mean() * 100:.2f}%"
    )
    valid_frag = events_all["fragmentation_rate"].dropna()
    print(
        f"Fragmentation rate: median {valid_frag.median():.3f}, "
        f"mean {valid_frag.mean():.3f}"
    )
    print(f"Per-session summary saved to {args.output_dir}/event_collapse_assumptions.csv")
    print(f"Figure saved to {args.output_dir}/event_collapse_assumptions.{{png,pdf}}")


if __name__ == "__main__":
    main()
