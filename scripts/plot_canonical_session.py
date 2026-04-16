#!/usr/bin/env python
"""CLI driver for :func:`plot_session_tables`.

Builds the four canonical tables for one session and renders the
stacked inspection figure to a file. Supports either live Spyglass
loading (the normal path) or the cached-trials workaround documented
in ``docs/cached_trials_for_testing.md`` while Spyglass is down.

Usage::

    uv run python scripts/plot_canonical_session.py \\
        --session j1620210710_02_r1 \\
        --window 1625935780 1625935810 \\
        --out scratch/session_tables.png

    # with a specific subset of panels
    uv run python scripts/plot_canonical_session.py \\
        --session j1620210710_02_r1 \\
        --panels trial_rail arm_posterior_heatmap arm_markers mismatch_pointers \\
        --out scratch/session_tables_core.png

If ``--window`` is omitted the figure spans a 30-second slice around
the first event (a reasonable default for inspection).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — must be set before pyplot is imported

import pandas as pd
import xarray as xr

from continuum_swr_replay.analysis.replay.posterior_io import (
    load_conditional_posterior,
)
from continuum_swr_replay.analysis.replay.tables import (
    build_event_arm_table,
    build_event_sequence_table,
    build_event_table,
    build_stop_table,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.session_tables import (
    DEFAULT_PANELS,
    plot_session_tables,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

CACHE_ROOT = Path("/tmp/continuum_test_cache")


def _parse_session_id(session_id: str) -> tuple[str, str]:
    """Split ``chimi20200213_02_r1`` into ``("chimi20200213_.nwb", "02_r1")``.

    Animal prefix may contain letters and digits (e.g., ``j16``) so
    the first group is ``\\w+`` rather than strict letters-then-8-digits.
    """
    m = re.match(r"(\w+?\d{8})_(\d+_r\d+)$", session_id)
    if not m:
        raise ValueError(
            f"session_id {session_id!r} does not match the expected "
            "``<animal><date>_<epoch>`` format (e.g., chimi20200213_02_r1, "
            "j1620210710_02_r1)"
        )
    return f"{m.group(1)}_.nwb", m.group(2)


def _load_trials_and_graph(
    session_id: str,
) -> tuple[pd.DataFrame, object]:
    """Pull trials + track_graph, preferring Spyglass and falling back to cache.

    Live Spyglass is the normal path; the cached-trials workaround
    exists so development can continue while Spyglass is unavailable.
    See ``docs/cached_trials_for_testing.md``.
    """
    nwb_file_name, epoch_name = _parse_session_id(session_id)
    try:
        from continuum_swr_replay.data_loaders import load_data
        from continuum_swr_replay.types import SessionDataKeys

        data = load_data(
            nwb_file_name, epoch_name,
            include=frozenset({SessionDataKeys.TRIALS}),
        )
        trials = data.get("trials")
        track_graph = data.get("track_graph")
        if trials is None or track_graph is None:
            raise RuntimeError("load_data returned without trials or track_graph")
        logging.info("loaded trials + track_graph from Spyglass")
        return trials, track_graph
    except Exception as exc:  # noqa: BLE001 - fall back to cache with a warning
        logging.warning("Spyglass load failed (%s); trying cache", exc)

    trials_path = CACHE_ROOT / f"{session_id}_trials.pkl"
    graph_path = CACHE_ROOT / f"{session_id}_track_graph.pkl"
    if not (trials_path.exists() and graph_path.exists()):
        raise FileNotFoundError(
            f"cached trials / track_graph not found at {CACHE_ROOT}. "
            "Run the conversion snippet in docs/cached_trials_for_testing.md "
            "to populate the cache."
        )
    with open(trials_path, "rb") as f:
        payload = pickle.load(f)
    trials = pd.DataFrame({c: payload[c] for c in payload["columns"]})
    with open(graph_path, "rb") as f:
        track_graph = pickle.load(f)
    logging.info("loaded trials + track_graph from cache (%s)", CACHE_ROOT)
    return trials, track_graph


def _load_session_artifacts(session_id: str):
    """Load the decoder-output artifacts for one session.

    Returns ``(dot, env, posterior)``. ``posterior`` is ``None`` if
    ``results.nc`` is missing (degrades gracefully — posterior-column
    panels will show NaN).
    """
    ep_id = f"{session_id}_HPC"
    dot_path = PROCESSED_DATA_DIR / f"{ep_id}_80_data_over_time.pkl"
    env_path = PROCESSED_DATA_DIR / f"{ep_id}_environment.pkl"
    results_path = PROCESSED_DATA_DIR / f"{ep_id}_results.nc"

    if not dot_path.exists():
        raise FileNotFoundError(f"missing decoder pickle: {dot_path}")
    if not env_path.exists():
        raise FileNotFoundError(f"missing environment pickle: {env_path}")

    dot = pd.read_pickle(dot_path)
    with open(env_path, "rb") as f:
        env = pickle.load(f)
    posterior = None
    if results_path.exists():
        try:
            posterior = load_conditional_posterior(results_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("failed to load results.nc (%s); posterior panels degrade to NaN", exc)
    else:
        logging.warning("no results.nc at %s; posterior panels degrade to NaN", results_path)
    return dot, env, posterior


def _default_window(event_table: pd.DataFrame) -> tuple[float, float] | None:
    """30 s window centered on the first event, or ``None`` if empty."""
    if len(event_table) == 0:
        return None
    t = float(event_table["event_midpoint_time"].iloc[0])
    return (t - 15.0, t + 15.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session", required=True,
        help="session id in the form <animal><date>_<epoch> (e.g., j1620210710_02_r1)",
    )
    parser.add_argument(
        "--window", nargs=2, type=float, default=None, metavar=("START", "END"),
        help="time window in seconds (session epoch); default: 30 s around the first event",
    )
    parser.add_argument(
        "--panels", nargs="+", default=None,
        help=f"panel names, in render order. default: {list(DEFAULT_PANELS)}",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("scratch") / "session_tables.png",
        help="output figure path; default: scratch/session_tables.png",
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=(14.0, 12.0),
        metavar=("W", "H"), help="figure size in inches",
    )
    args = parser.parse_args(argv)

    trials, track_graph = _load_trials_and_graph(args.session)
    dot, env, posterior = _load_session_artifacts(args.session)

    logging.info("building canonical tables...")
    event_table = build_event_table(
        dot, env, trials, track_graph, conditional_posterior=posterior
    )
    event_arm_table = build_event_arm_table(
        event_table, dot, env, trials, track_graph, conditional_posterior=posterior
    )
    stop_table = build_stop_table(dot, trials, track_graph)
    event_sequence_table = (
        build_event_sequence_table(dot, env, posterior)
        if posterior is not None
        else pd.DataFrame()
    )
    logging.info(
        "tables: event %s, event_arm %s, stop %s, event_sequence %s",
        event_table.shape, event_arm_table.shape, stop_table.shape,
        event_sequence_table.shape,
    )

    window = tuple(args.window) if args.window else _default_window(event_table)
    if window is None:
        logging.error("session has no events — nothing to plot")
        return 1
    panels = tuple(args.panels) if args.panels else DEFAULT_PANELS

    stub_detector = types.SimpleNamespace(environments=[env])
    stub_results = xr.Dataset(
        {"_nop": ("time", [0, 0])},
        coords={"time": [dot.index.min(), dot.index.max()]},
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, mismatch_table = plot_session_tables(
        data={"trials": trials, "track_graph": track_graph},
        detector=stub_detector,
        results=stub_results,
        event_table=event_table,
        event_arm_table=event_arm_table,
        event_sequence_table=event_sequence_table,
        stop_table=stop_table,
        trials=trials,
        track_graph=track_graph,
        time_window=window,
        panels=panels,
        figsize=tuple(args.figsize),
        save_path=args.out,
    )
    logging.info("figure saved to %s", args.out)
    n_mismatch = int((~mismatch_table["match"]).sum())
    if n_mismatch > 0:
        logging.warning(
            "%d / %d events have canonical next_arm ≠ actual next_arm — "
            "see mismatch table",
            n_mismatch, len(mismatch_table),
        )
        bad = mismatch_table.loc[~mismatch_table["match"]].head(10)
        logging.warning("first mismatches:\n%s", bad.to_string(index=False))
    else:
        logging.info(
            "future-covariate check: all %d events match", len(mismatch_table)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
