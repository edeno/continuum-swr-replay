#!/usr/bin/env python
"""Regenerate ``stop_table.pkl`` with the new count columns.

Re-runs the canonical :func:`build_stop_table` per session (which emits
``n_swr_events`` / ``n_mua_events`` / ``n_selected_events``), then adds the
canonical ``n_nonlocal_swr_events`` / ``n_nonlocal_non_swr_events`` split by joining
the existing ``event_table.pkl``'s ``is_swr`` on ``stop_id``. This rebuilds only
the stop table — it does NOT rebuild the event/arm/patch tables — so it is much
cheaper than a full ``prepare_replay_content_data.py`` run.

The stop-bout definition is unchanged, so ``stop_id`` matches the existing
``event_table`` row-for-row and the merge is exact.

Usage:
    uv run python scripts/decoding/regenerate_stop_table.py --max-per-animal 1   # smoke
    uv run python scripts/decoding/regenerate_stop_table.py                      # full
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.swr_source import (  # noqa: E402
    read_dot_with_swr_source,
    require_swr_source,
    stamp_swr_source,
)
from continuum_swr_replay.analysis.replay import build_stop_table  # noqa: E402
from continuum_swr_replay.analysis.replay.covariate_registry import (  # noqa: E402
    require_covariate_schema,
    stamp_covariate_schema,
)
from continuum_swr_replay.analysis.replay.prepare_data import find_sessions
from continuum_swr_replay.analysis.replay.tables_stop import (  # noqa: E402
    add_nonlocal_swr_split,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402
from continuum_swr_replay.session_keys import session_key_set
from continuum_swr_replay.table_io import (
    publish_canonical,
    read_table,
    table_exists,
    write_table,
)
from continuum_swr_replay.types import SessionDataKeys  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
RC = PROCESSED_DATA_DIR / "replay_content"


def build_one(session: dict) -> pd.DataFrame | None:
    """Load dot + trials for one session and build its stop table."""
    # imported here: loading data needs Spyglass, and the module's guards
    # must import without it
    from continuum_swr_replay.data_loaders import load_data

    # SWR labels from the one SWR source, not the decode's stored ripples
    dot = read_dot_with_swr_source(session["dot_file"])
    data = load_data(
        session["nwb_file_name"],
        session["epoch_name"],
        include=frozenset({SessionDataKeys.TRIALS, SessionDataKeys.PORT_EXPERIENCES}),
    )
    trials = data.get("trials")
    port_experiences = data.get("port_experiences")
    track_graph = data.get("track_graph")
    if trials is None or port_experiences is None or track_graph is None:
        return None
    return build_stop_table(dot, trials, port_experiences, track_graph)


def _assert_matches_event_table(stop: pd.DataFrame, event_table: pd.DataFrame) -> None:
    """Refuse to write a stop table that does not cover the event table's scope.

    ``stop_id`` is the join key for every stop-level covariate, and the whole
    premise of this cheap rebuild is that the stop bouts are unchanged, so the
    rebuilt table must span exactly the sessions the canonical ``event_table``
    already contains. A missing session (e.g. one that failed to load) would
    otherwise be discovered only when a downstream fit silently mis-joins, or
    not at all when that session had zero non-local events.
    """
    stop_sessions = session_key_set(stop)
    event_sessions = session_key_set(event_table)
    missing = event_sessions - stop_sessions
    extra = stop_sessions - event_sessions
    if missing or extra:
        raise ValueError(
            "rebuilt stop table scope does not match event_table: "
            f"{len(missing)} session(s) missing (e.g. {sorted(missing)[:3]}), "
            f"{len(extra)} extra (e.g. {sorted(extra)[:3]})"
        )
    if stop["stop_id"].isna().any() or stop["stop_id"].duplicated().any():
        raise ValueError("rebuilt stop table requires unique nonmissing stop_id")
    event_stop_ids = set(event_table["stop_id"].dropna().unique())
    orphan_events = event_stop_ids - set(stop["stop_id"])
    if orphan_events:
        raise ValueError(
            f"{len(orphan_events)} event_table stop_id(s) absent from the rebuilt "
            "stop table; the atomic stop/event set would be inconsistent"
        )


def validate(stop: pd.DataFrame) -> None:
    """Hard checks on the regenerated table before it is trusted.

    Uses ``raise`` rather than ``assert`` so the checks survive ``python -O``:
    a stripped assertion would let an invalid stop table be stamped and
    published silently, which is exactly what this gate exists to prevent.
    """
    split_ok = (
        stop["n_nonlocal_swr_events"] + stop["n_nonlocal_non_swr_events"]
        == stop["n_selected_events"]
    )
    if not split_ok.all():
        raise ValueError(f"{(~split_ok).sum()} stops where SWR split != selected total")
    if not (stop["is_swr_stop"] == (stop["n_swr_events"] > 0)).all():
        raise ValueError("is_swr_stop != n_swr_events>0")
    if not (stop["is_mua_stop"] == (stop["n_mua_events"] > 0)).all():
        raise ValueError("is_mua_stop != n_mua_events>0")
    count_cols = ["n_swr_events", "n_mua_events", "n_nonlocal_swr_events"]
    if not (stop[count_cols] >= 0).all().all():
        raise ValueError("negative count")
    # Lagged reward columns present and binary; lag 1 should agree with the
    # independently constructed known outcome at the bout start.
    lag_cols = [c for c in stop.columns if c.startswith("known_reward_outcome_lag_")]
    if not lag_cols:
        raise ValueError("no known_reward_outcome_lag_ columns produced")
    vals = stop[lag_cols].to_numpy()
    if not np.isin(vals[~np.isnan(vals)], [0.0, 1.0]).all():
        raise ValueError("known_reward_outcome_lag_ not 0/1/NaN")
    both = (
        stop["known_reward_outcome_lag_1"].notna() & stop["is_rewarded_outcome"].notna()
    )
    agree = float(
        (
            stop.loc[both, "known_reward_outcome_lag_1"]
            == stop.loc[both, "is_rewarded_outcome"]
        ).mean()
    )
    logger.info(
        "validation OK | n_swr_events>0 in %d/%d stops, total SWRs=%d, total non-local=%d "
        "(SWR-overlap=%d, no-SWR=%d) | %d known-outcome lag cols, "
        "lag1 vs known outcome agree=%.3f",
        int((stop["n_swr_events"] > 0).sum()),
        len(stop),
        int(stop["n_swr_events"].sum()),
        int(stop["n_selected_events"].sum()),
        int(stop["n_nonlocal_swr_events"].sum()),
        int(stop["n_nonlocal_non_swr_events"].sum()),
        len(lag_cols),
        agree,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-animal", type=int, default=None, help="smoke subset")
    ap.add_argument("--no-save", action="store_true", help="build + validate only")
    args = ap.parse_args()

    sessions = find_sessions(max_per_animal=args.max_per_animal)
    logger.info("building stop tables for %d sessions...", len(sessions))
    tables, failures, t0 = [], [], time.time()
    for i, s in enumerate(sessions):
        try:
            st = build_one(s)
        except Exception as e:  # noqa: BLE001
            failures.append((s["nwb_file_name"], s["epoch_name"], str(e)))
            continue
        # A session that loads but yields no stop bouts is a data error here:
        # the rebuilt table must match the existing event_table row-for-row on
        # stop_id, so a silently dropped session desynchronizes the two.
        if st is None or not len(st):
            failures.append((s["nwb_file_name"], s["epoch_name"], "no stop bouts"))
            continue
        tables.append(st)
        if (i + 1) % 20 == 0:
            logger.info("  %d/%d (%.0fs)", i + 1, len(sessions), time.time() - t0)

    # Fail closed. Silently skipping a session builds a stop table with fewer
    # sessions than the event_table it is merged against and then overwrites the
    # canonical table with it (the atomic-set violation warned about in
    # CLAUDE.md); every downstream fit would mis-join on stop_id with no error.
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(sessions)} session(s) failed to build a "
            f"stop table; sample={failures[:10]}. No stop table was written."
        )
    stop = pd.concat(tables, ignore_index=True)

    event_table = read_table(RC / "event_table.pkl")
    require_covariate_schema(event_table, table="event")
    # The split joins the shipped event table's is_swr onto stops whose SWR
    # columns were just built from the live source: both must be one source.
    require_swr_source(event_table, what=f"{RC / 'event_table'}")
    _assert_matches_event_table(stop, event_table)
    stop = add_nonlocal_swr_split(stop, event_table)
    stop = stamp_swr_source(stamp_covariate_schema(stop, table="stop"))
    validate(stop)

    if args.no_save:
        logger.info("--no-save: built %d stops, not written", len(stop))
        return
    out = RC / "stop_table"
    if table_exists(out):
        backup = write_table(read_table(out), RC / "stop_table_previous")
        logger.info("backed up existing -> %s", backup.name)
    out = publish_canonical(stop, RC, table="stop", basename="stop_table")
    logger.info("saved %d stops -> %s", len(stop), out)


if __name__ == "__main__":
    main()
