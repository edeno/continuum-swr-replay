#!/usr/bin/env python
"""Model-free event-start rates on the three visit clocks, per kind and animal.

For every session of the canonical stop and event tables, builds the visit
time-bin design of each kind (``replay_models.visit_design.visit_timebins``:
SWR and No SWR non-local events) from the tables and the session's port
experiences (read as the table builders read them: the behavioral-context
cache, ``event_tables.prepare_data.load_behavioral_context``), keeps the bins inside the well visits
(``select_visit_context``, the default subset: the events that start inside a
visit), and sums the event starts and the at-risk seconds in bins of each visit
clock (``replay_models.rate.visit_clock_curves``):

- ``time_since_visit_start_s``, time since the first poke-in, 0.25 s bins to 20 s;
- ``time_until_visit_end_s``, time until the final poke-out, 0.05 s bins to 3 s;
- ``time_until_stretch_end_s``, time until the end of the current immobile
  stretch (clipped at the poke-out), 0.05 s bins to 3 s.

The two that count down reference the future (when the animal will move or
leave), so this is a description, not a prediction. The visit rate models are
compared with these curves.

Outputs, under the replay-content root
(``$CSR_PROCESSED_DATA_DIR/replay_content``):
``models_convention/descriptive/visit_clock_curves.parquet`` (one row per kind,
animal, clock and clock bin: ``n_starts``, ``at_risk_s``, ``rate``) and its
``visit_clock_curves.provenance.json`` sidecar: the input tables' SHA-256, the
commit and command line, the subset in words, the label counts of each kind's
events (``count_visit_context``), the event starts left out for having no
at-risk time before them, and the stops that start inside a visit, per animal,
by how they end (``stop_end_kind``). With ``--max-per-animal`` the file is
``visit_clock_curves_smoke.parquet``.

Usage:
    uv run python scripts/replay_models/rate/describe_visit_clocks.py
    uv run python scripts/replay_models/rate/describe_visit_clocks.py --max-per-animal 1
    uv run python scripts/replay_models/rate/describe_visit_clocks.py \
        --behavior-cache-dir <dir>
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    excluded_intervals_by_session,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    count_visit_context,
    select_visit_context,
)
from continuum_swr_replay.analysis.event_tables import prepare_data
from continuum_swr_replay.analysis.replay_models.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay_models.rate.visit_clock_curves import (
    VISIT_CLOCKS,
    stop_end_kind_counts,
    visit_clock_rate_curves,
)
from continuum_swr_replay.analysis.replay_models.visit_design import (
    EVENT_KINDS,
    visit_timebins,
)
from continuum_swr_replay.decoding.identifiers import (
    format_epoch_identifier,
    nwb_file_name,
)
from continuum_swr_replay.paths import replay_content_dir, table_path
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.runs.io import json_ready
from continuum_swr_replay.tables.io import write_atomically
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    subset_max_per_animal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

#: Clock bin edges (s) per clock.
CLOCK_EDGES: dict[str, np.ndarray] = {
    "time_since_visit_start_s": np.linspace(0.0, 20.0, 81),
    "time_until_visit_end_s": np.linspace(0.0, 3.0, 61),
    "time_until_stretch_end_s": np.linspace(0.0, 3.0, 61),
}

SUBSET = (
    "the time bins inside a well visit (first poke-in to final poke-out) and an "
    "immobility stop, less excluded time; the events that start inside a well "
    "visit (visit_context in_visit or spans_visit_end), in the analyzed epochs "
    "of the canonical tables"
)


def input_paths() -> dict[str, Path]:
    """The tables the curves are made from, under ``paths.replay_content_dir()``."""
    return {
        "stop_table": table_path("stop_table"),
        "event_table": table_path("event_table"),
    }


def output_path(smoke: bool) -> Path:
    """The curves file, under ``paths.replay_content_dir()``."""
    name = "visit_clock_curves_smoke.parquet" if smoke else "visit_clock_curves.parquet"
    return replay_content_dir() / "models_convention" / "descriptive" / name


def session_port_experiences(
    animal: str, session_date: object, epoch: str
) -> pd.DataFrame:
    """One session's port experiences, read as the table builders read them.

    From the behavioral-context cache (``prepare_data.session_cache_dir()``),
    keyed by the session's HPC epoch identifier, and from Spyglass when the
    cache has no current entry (``load_behavioral_context``).

    Raises
    ------
    RuntimeError
        If the session's behavioral context cannot be loaded.
    """
    nwb = nwb_file_name(animal, session_date)
    key = format_epoch_identifier(nwb, epoch, "HPC")
    _, experiences, _, _ = prepare_data.load_behavioral_context(
        {"nwb_file_name": nwb, "epoch_name": epoch, "session_id": key},
        from_cache=True,
        cache_key=key,
    )
    if experiences is None:
        raise RuntimeError(f"no port experiences for {key}")
    return experiences


def session_bins(
    stops: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """The in-visit design bins of every session and kind, and what each left out.

    Parameters
    ----------
    stops, events : pd.DataFrame
        The stop and event tables (any number of sessions). Each session's
        port experiences are read by :func:`session_port_experiences`.

    Returns
    -------
    bins : pd.DataFrame
        The selected bins, with ``kind``, ``animal``, the clocks, ``exposure``
        and ``count``.
    records : list of dict
        Per session and kind: the session, ``kind`` and
        ``n_starts_not_at_risk``.
    """
    keys = list(SESSION_COLUMNS)
    excluded = excluded_intervals_by_session(stops)
    events_by_session = dict(tuple(events.groupby(keys, sort=False, observed=True)))
    excluded_by_session = dict(tuple(excluded.groupby(keys, sort=False, observed=True)))
    parts, records = [], []
    for key, session_stops in stops.groupby(keys, sort=True, observed=True):
        session_events = events_by_session.get(key, events.iloc[:0])
        session_excluded = excluded_by_session.get(key, excluded.iloc[:0])
        experiences = session_port_experiences(*key)
        for kind in EVENT_KINDS:
            bins = visit_timebins(
                session_stops,
                session_events,
                experiences,
                kind=kind,
                excluded_intervals=session_excluded,
            )
            parts.append(
                select_visit_context(bins)[
                    ["animal", *VISIT_CLOCKS, "exposure", "count"]
                ].assign(kind=kind)
            )
            records.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "kind": kind,
                    "n_starts_not_at_risk": bins.attrs["n_starts_not_at_risk"],
                }
            )
    return pd.concat(parts, ignore_index=True), records


def main(argv: Sequence[str] | None = None) -> None:
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Smoke mode: keep only the first N sessions per animal.",
    )
    parser.add_argument(
        "--behavior-cache-dir",
        type=Path,
        default=None,
        help=(
            "The behavioral-context cache root the port experiences are read "
            "from; default the table builders' (replay_content/_session_cache)."
        ),
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    if args.behavior_cache_dir is not None:
        prepare_data.SESSION_CACHE_DIR = args.behavior_cache_dir.resolve()

    # The clocks that count down reference the future: a description only.
    assert_predictor_timing(list(VISIT_CLOCKS), descriptive=True)

    inputs = input_paths()
    stops = read_swr_table(inputs["stop_table"])
    events = read_swr_table(inputs["event_table"])
    table_records = record_inputs(inputs)
    if args.max_per_animal is not None:
        stops = subset_max_per_animal(stops, args.max_per_animal)
    sessions = stops[list(SESSION_COLUMNS)].drop_duplicates()
    events = events.merge(sessions, on=list(SESSION_COLUMNS), how="inner")
    logger.info("Building the visit bins of %d sessions...", len(sessions))

    bins, left_out = session_bins(stops, events)
    curves = pd.concat(
        [
            visit_clock_rate_curves(bins, clock=clock, edges=CLOCK_EDGES[clock])
            for clock in VISIT_CLOCKS
        ],
        ignore_index=True,
    )
    label_counts = {
        kind: count_visit_context(events[events["is_swr"] == is_swr])
        .reset_index()
        .to_dict(orient="records")
        for kind, is_swr in EVENT_KINDS.items()
    }
    stop_ends = stop_end_kind_counts(stops).to_dict(orient="records")

    out = output_path(smoke=args.max_per_animal is not None)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(out, lambda path: curves.to_parquet(path, index=False))
    write_fit_provenance(
        out,
        table_records,
        "model_free_rate",
        [sys.argv[0], *argv],
        extra=json_ready(
            {
                "analyzed_subset": SUBSET,
                "n_sessions": len(sessions),
                "clock_edges_s": {
                    clock: edges.tolist() for clock, edges in CLOCK_EDGES.items()
                },
                "event_visit_context_counts": label_counts,
                "starts_not_at_risk": left_out,
                "stop_end_kind_counts": stop_ends,
            }
        ),
    )
    logger.info(
        "Saved %d rows (%d sessions; %d starts in %.1f at-risk s per kind: %s) to %s",
        len(curves),
        len(sessions),
        int(bins["count"].sum()),
        bins["exposure"].sum() / len(EVENT_KINDS),
        bins.groupby("kind")["count"].sum().astype(int).to_dict(),
        out,
    )


if __name__ == "__main__":
    main()
