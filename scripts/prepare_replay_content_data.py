#!/usr/bin/env python
"""Build the four canonical replay-content tables across all sessions.

Thin driver over :mod:`continuum_swr_replay.analysis.replay.tables`. For
each session it loads the pre-saved ``data_over_time.pkl``,
``environment.pkl``, and ``results.nc``, pulls ``trials`` +
``track_graph`` from Spyglass, and calls the four table builders.

Outputs (under ``PROCESSED_DATA_DIR/replay_content/``):

- ``event_table.pkl`` — one row per non-local event
- ``event_arm_table.pkl`` — one row per (event, arm)
- ``event_patch_table.pkl`` — one row per (event, patch); collapsed from
  ``event_arm_table`` via :func:`build_event_patch_table` so the
  ``--hierarchy patch`` path of ``fit_replay_content_models.py`` has its
  required input alongside the per-arm tables.
- ``event_sequence_table.pkl`` — one row per in-event timebin
- ``stop_table.pkl`` — one row per stopping period (incl. zero-event)

Usage::

    uv run python scripts/prepare_replay_content_data.py           # full run
    uv run python scripts/prepare_replay_content_data.py --test    # 2 per animal
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import time

import pandas as pd

from continuum_swr_replay.analysis.replay.posterior_io import (
    load_conditional_posterior,
)
from continuum_swr_replay.analysis.replay.tables import (
    build_event_arm_table,
    build_event_patch_table,
    build_event_sequence_table,
    build_event_table,
    build_stop_table,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.constants import NWB_FILES
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.types import SessionDataKeys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = PROCESSED_DATA_DIR / "replay_content"


def find_sessions(max_per_animal: int | None = None):
    """Yield every session that has ``data_over_time.pkl`` and ``environment.pkl`` on disk.

    ``results.nc`` absence is not a hard skip here — that's inspected
    per session in :func:`main` so sessions without the conditional
    posterior still contribute the event / event-arm / stop tables
    (with ``posterior_*`` columns as NaN).
    """
    dot_files = sorted(PROCESSED_DATA_DIR.glob("*_HPC_80_data_over_time.pkl"))
    env_files = {
        f.stem.replace("_environment", ""): f
        for f in PROCESSED_DATA_DIR.glob("*_HPC_environment.pkl")
    }

    by_animal: dict[str, list] = {}
    for f in dot_files:
        match = re.match(r"(\w+\d{8})_(\d+_r\d+)_HPC_80_data_over_time\.pkl", f.name)
        if not match:
            continue
        animal_date, epoch = match.group(1), match.group(2)
        # The outer regex enforces exactly 8 trailing digits (the date),
        # so the animal id is everything before them. The previous
        # ``re.match(r"([a-z]+)", animal_date)`` collapsed digit-bearing
        # ids — e.g. ``j162021...`` matched only ``j``, merging j16
        # sessions under ``j``.
        animal = animal_date[:-8]
        session_id = f"{animal_date}_{epoch}_HPC"
        if session_id not in env_files:
            continue

        nwb_match = [nwb for nwb in NWB_FILES if animal_date in nwb]
        if not nwb_match:
            continue

        results_path = PROCESSED_DATA_DIR / f"{session_id}_results.nc"

        by_animal.setdefault(animal, []).append(
            {
                "nwb_file_name": nwb_match[0],
                "epoch_name": epoch,
                "dot_file": f,
                "env_file": env_files[session_id],
                "results_file": results_path if results_path.exists() else None,
            }
        )

    selected = []
    for animal in sorted(by_animal):
        sessions = by_animal[animal]
        if max_per_animal is not None:
            sessions = sessions[:max_per_animal]
        selected.extend(sessions)

    return selected


def _process_session(session: dict) -> dict[str, pd.DataFrame] | None:
    """Run all four builders on one session; return the four frames or ``None``.

    Returns ``None`` when the session cannot be loaded or the decoder
    produced no non-local events. Callers concatenate the non-empty
    returns into the global canonical tables.
    """
    nwb = session["nwb_file_name"]
    epoch = session["epoch_name"]

    dot = pd.read_pickle(session["dot_file"])
    with open(session["env_file"], "rb") as f:
        env = pickle.load(f)

    # Trials + track_graph come from Spyglass (slow; ~30 s).
    data = load_data(nwb, epoch, include=frozenset({SessionDataKeys.TRIALS}))
    trials = data.get("trials")
    track_graph = data.get("track_graph")
    if trials is None or track_graph is None:
        return None

    # Conditional posterior is optional — sessions without a saved
    # ``results.nc`` still emit the three tables whose posterior
    # columns degrade to NaN.
    posterior = None
    if session["results_file"] is not None:
        try:
            posterior = load_conditional_posterior(session["results_file"])
        except Exception as e:  # noqa: BLE001 - surface the reason and keep going
            logging.warning("posterior load failed for %s %s: %s", nwb, epoch, e)

    # Build the stop table first — it's a function of immobility
    # bouts, not non-local events, and the documented
    # ``stop_table.pkl`` contract (this script's own docstring; the
    # builder's "**including zero-event bouts**" promise) keeps
    # zero-event sessions in the dataset. Returning early on an
    # empty event_table dropped those sessions silently and broke
    # the downstream rate model, which models n_events per stop
    # over a denominator that *must* include zero-event stops.
    stop_table = build_stop_table(dot, trials, track_graph)

    event_table = build_event_table(
        dot, env, trials, track_graph, conditional_posterior=posterior
    )
    event_arm_table = build_event_arm_table(
        event_table,
        dot,
        env,
        trials,
        track_graph,
        conditional_posterior=posterior,
    )
    event_sequence_table = (
        build_event_sequence_table(dot, env, posterior)
        if posterior is not None and len(event_table) > 0
        else None
    )

    return {
        "event_table": event_table,
        "event_arm_table": event_arm_table,
        "stop_table": stop_table,
        "event_sequence_table": event_sequence_table,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="2 sessions per animal")
    parser.add_argument("--max-per-animal", type=int, default=None)
    args = parser.parse_args()

    max_per_animal = 2 if args.test else args.max_per_animal
    sessions = find_sessions(max_per_animal=max_per_animal)
    logging.info("Found %d sessions", len(sessions))

    # Warn if the output directory still holds artifacts from the
    # legacy pipeline — silent coexistence lets downstream scripts
    # accidentally read stale pickles.
    for legacy_name in ("events.pkl", "long_count.pkl", "long_binary.pkl"):
        legacy_path = OUTPUT_DIR / legacy_name
        if legacy_path.exists():
            logging.warning(
                "legacy output %s still present — delete it to avoid "
                "stale reads downstream",
                legacy_path,
            )

    t0 = time.time()
    all_event_tables: list[pd.DataFrame] = []
    all_event_arm_tables: list[pd.DataFrame] = []
    all_stop_tables: list[pd.DataFrame] = []
    all_event_sequence_tables: list[pd.DataFrame] = []

    for i, session in enumerate(sessions):
        nwb = session["nwb_file_name"]
        epoch = session["epoch_name"]

        try:
            result = _process_session(session)
        except (FileNotFoundError, OSError, ValueError) as e:
            # Skip-and-continue is only appropriate for data-gap
            # errors (missing file, corrupt input, unexpected shape).
            # ``KeyError`` / ``AttributeError`` / ``AssertionError``
            # are propagated — they indicate a code bug and silent
            # skips would hide the regression across the run.
            logging.warning("skip %s %s: %s", nwb, epoch, e)
            continue

        if result is None:
            continue

        all_event_tables.append(result["event_table"])
        all_event_arm_tables.append(result["event_arm_table"])
        all_stop_tables.append(result["stop_table"])
        if result["event_sequence_table"] is not None:
            all_event_sequence_tables.append(result["event_sequence_table"])

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(sessions) - i - 1)
            logging.info(
                "[%d/%d] %ds elapsed, ~%ds remaining",
                i + 1,
                len(sessions),
                int(elapsed),
                int(remaining),
            )

    if not all_event_tables:
        logging.error("no sessions produced data")
        return

    event_table_all = pd.concat(all_event_tables, ignore_index=True)
    event_arm_table_all = pd.concat(all_event_arm_tables, ignore_index=True)
    stop_table_all = pd.concat(all_stop_tables, ignore_index=True)
    event_sequence_table_all = (
        pd.concat(all_event_sequence_tables, ignore_index=True)
        if all_event_sequence_tables
        else None
    )
    # Patch table is a deterministic projection of the per-arm table
    # (sums arm_count_hard within each patch and aggregates predictors
    # to patch level). Save it alongside the others so the documented
    # ``fit_replay_content_models.py --hierarchy patch`` path has its
    # required input.
    event_patch_table_all = build_event_patch_table(
        event_table_all, event_arm_table_all
    )

    session_cols = ["animal", "day", "epoch"]
    logging.info(
        "events: %d, event_arm rows: %d, stops: %d, sessions: %d",
        len(event_table_all),
        len(event_arm_table_all),
        len(stop_table_all),
        event_table_all.groupby(session_cols).ngroups,
    )

    # Cross-table invariant gate — run BEFORE saving, not after. A
    # failed check means the concatenated frames disagree across
    # tables; persisting them would publish an inconsistent dataset
    # to downstream consumers. The full invariant sweep runs in
    # tests/unit/test_replay_tables.py; this is a cheap end-to-end
    # guard for the concat path.
    sum_n_events = int(stop_table_all["n_events"].sum())
    if sum_n_events != len(event_table_all):
        logging.error(
            "cross-table drift: Σ stop_table.n_events = %d, "
            "len(event_table) = %d — refusing to write output",
            sum_n_events,
            len(event_table_all),
        )
        return
    logging.info(
        "cross-table invariant OK: Σ n_events == len(event_table) == %d",
        len(event_table_all),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("event_table", event_table_all),
        ("event_arm_table", event_arm_table_all),
        ("event_patch_table", event_patch_table_all),
        ("stop_table", stop_table_all),
        ("event_sequence_table", event_sequence_table_all),
    ):
        if frame is None:
            logging.info("  %s: skipped (no data)", name)
            continue
        path = OUTPUT_DIR / f"{name}.pkl"
        frame.to_pickle(path)
        logging.info("saved %s → %s (%d rows)", name, path, len(frame))

    logging.info("total time: %ds", int(time.time() - t0))


if __name__ == "__main__":
    main()
