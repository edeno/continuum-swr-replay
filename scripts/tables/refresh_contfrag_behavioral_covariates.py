#!/usr/bin/env python
"""Rebuild ContFrag tables while reusing immutable posterior aggregates.

Behavioral clocks, locations, histories, event boundaries, hard counts, and
q-value attachments are recomputed from their canonical inputs.  Only values
that require the expensive conditional posterior are carried from a preserved
source bundle, after exact event/candidate-key validation.  This makes the
dependency boundary explicit and avoids re-reading every session's large
posterior file when the decoder results themselves have not changed.

Each session is rebuilt by
:func:`continuum_swr_replay.analysis.contfrag.prepare_data.rebuild_contfrag_session_tables`;
:func:`refresh_contfrag_behavioral_covariates` is the run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates import load_contingency
from continuum_swr_replay.analysis.contfrag.prepare_data import (
    contfrag_event_patch_table,
    contfrag_table_basenames,
    rebuild_contfrag_session_tables,
)
from continuum_swr_replay.analysis.event_tables import (
    prepare_data as replay_prepare_data,
)
from continuum_swr_replay.analysis.event_tables.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)
from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.paths import table_path
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tables._table_runner import (  # noqa: E402
    find_contfrag_sessions,
    publish_contfrag_tables,
)

LOGGER = logging.getLogger(__name__)


def _session_key(frame: pd.DataFrame) -> tuple[object, object, object]:
    keys = frame.loc[:, SESSION_COLUMNS].drop_duplicates()
    if len(keys) != 1:
        raise ValueError(f"expected one session, found {len(keys)}")
    return tuple(keys.iloc[0])


def _source_session(
    frame: pd.DataFrame, session: tuple[object, object, object]
) -> pd.DataFrame:
    mask = np.logical_and.reduce(
        [
            frame[column] == value
            for column, value in zip(SESSION_COLUMNS, session, strict=True)
        ]
    )
    return frame.loc[mask].reset_index(drop=True)


def refresh_contfrag_behavioral_covariates(
    *,
    source_dir: Path,
    output_dir: Path,
    replay_content_dir: Path,
    event_source: str,
    brain_area: str,
    expected_sessions: int,
) -> dict[str, pd.DataFrame]:
    """Rebuild a ContFrag bundle, carrying its posterior aggregates over.

    Behavioral clocks, locations, histories, event boundaries, hard counts and
    the qvals attachments are rebuilt from their inputs; the values that need
    the conditional posterior (``prepare_data.EVENT_POSTERIOR_COLUMNS`` and each
    arm's
    ``arm_posterior_mass``) are copied from the preserved bundle in
    `source_dir`, matched on the event label (and candidate arm), and
    ``arm_fraction_posterior`` is recomputed from the copied mass. This skips
    reading every session's posterior when the decodes have not changed.

    Parameters
    ----------
    source_dir : Path
        The preserved bundle the posterior aggregates are copied from.
    output_dir : Path
        Where the rebuilt tables are published.
    replay_content_dir : Path
        The replay-content root: its ``qvals_table`` and its behavioral cache
        (``_session_cache``), read with ``from_cache=True``.
    event_source : str
        ``"SWR"`` or ``"MUA"``.
    brain_area : str
        ``"HPC"``, ``"mPFC"`` or ``"OFC"``.
    expected_sessions : int
        The number of sessions the preserved bundle, and the rebuild, must hold.

    Returns
    -------
    dict of str to pd.DataFrame
        The published tables (``_table_runner.publish_contfrag_tables``).

    Raises
    ------
    ValueError
        If the source bundle or the matched decodes hold other than
        `expected_sessions` sessions, an event or candidate key does not match
        between the rebuild and the source exactly, or a rebuilt session breaks
        stop/event conservation or ``stop_id`` uniqueness.
    """
    basenames = contfrag_table_basenames(event_source, brain_area)
    old_event_all = read_table(source_dir / basenames["event_table"])
    old_arm_all = read_table(source_dir / basenames["event_arm_table"])
    # The posterior aggregates are carried on event ids that hash the SWR
    # labels, so the source bundle must come from the SWR source the rebuild
    # reads: another source's events are other events.
    for name, frame in (
        ("event_table", old_event_all),
        ("event_arm_table", old_arm_all),
    ):
        require_swr_source(frame, what=f"{source_dir / basenames[name]}")
    validate_event_arm_table(old_arm_all, event_table=old_event_all)
    contingency = load_contingency(table_path("qvals_table", replay_content_dir))
    replay_prepare_data.SESSION_CACHE_DIR = replay_content_dir / "_session_cache"
    source_sessions = set(
        old_event_all.loc[:, SESSION_COLUMNS].itertuples(index=False, name=None)
    )
    if len(source_sessions) != expected_sessions:
        raise ValueError(
            f"posterior source has {len(source_sessions)} sessions; "
            f"expected {expected_sessions}"
        )
    descriptors = []
    for descriptor in find_contfrag_sessions(brain_area=brain_area):
        session = (
            *parse_nwb_filename(descriptor["nwb_file_name"]),
            descriptor["epoch_name"],
        )
        if session in source_sessions:
            descriptors.append(descriptor)
    if len(descriptors) != expected_sessions:
        raise ValueError(
            f"found {len(descriptors)} matching raw sessions; "
            f"expected {expected_sessions}"
        )

    events: list[pd.DataFrame] = []
    arms: list[pd.DataFrame] = []
    stops: list[pd.DataFrame] = []
    for index, descriptor in enumerate(descriptors, start=1):
        session = (
            *parse_nwb_filename(descriptor["nwb_file_name"]),
            descriptor["epoch_name"],
        )
        if session not in source_sessions:
            raise ValueError(f"posterior source is missing session {session}")
        old_event = _source_session(old_event_all, session)
        old_arm = _source_session(old_arm_all, session)
        event, arm, stop = rebuild_contfrag_session_tables(
            descriptor,
            event_source=event_source,
            contingency=contingency,
            old_event=old_event,
            old_arm=old_arm,
            from_cache=True,
        )
        if _session_key(event) != session or _session_key(stop) != session:
            raise ValueError(f"rebuilt session identity drifted for {session}")
        validate_event_arm_table(arm, event_table=event)
        if int(stop["n_selected_events"].sum()) != len(event):
            raise ValueError(f"session {session} violates stop/event conservation")
        events.append(event)
        arms.append(arm)
        stops.append(stop)
        if index % 10 == 0:
            LOGGER.info("rebuilt %d/%d sessions", index, len(descriptors))

    event_table = pd.concat(events, ignore_index=True)
    event_arm_table = pd.concat(arms, ignore_index=True)
    stop_table = pd.concat(stops, ignore_index=True)
    event_patch_table = contfrag_event_patch_table(
        event_table, event_arm_table, event_source=event_source, brain_area=brain_area
    )
    validate_event_arm_table(event_arm_table, event_table=event_table)
    validate_event_patch_table(event_patch_table, event_table=event_table)
    # stop_id is the join key for every stop-level covariate; a null or a
    # cross-session collision silently mis-joins. validate_event_arm_table
    # covers event_id, but nothing checks stop_id here (the sibling reducers do).
    if stop_table["stop_id"].isna().any():
        raise ValueError("rebuilt stop_table contains null stop_id")
    if stop_table["stop_id"].duplicated().any():
        raise ValueError("rebuilt stop_table contains duplicate stop_id")
    if int(stop_table["n_selected_events"].sum()) != len(event_table):
        raise ValueError("rebuilt stop/event count conservation failed")

    # Rebuilt from decode frames read through the SWR source, so stamped with it.
    return publish_contfrag_tables(
        event_table,
        event_arm_table,
        event_patch_table,
        stop_table,
        output_dir=output_dir,
        basenames=basenames,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replay-content-dir", required=True, type=Path)
    parser.add_argument("--event-source", required=True, choices=("SWR", "MUA"))
    parser.add_argument("--brain-area", required=True, choices=("HPC", "mPFC", "OFC"))
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=len(analysed_epoch_ids()),
        help="Epochs (sessions) the tables must hold; default: the epoch register's "
        "analysed epochs (HPC cohort).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    refresh_contfrag_behavioral_covariates(
        source_dir=args.source_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        replay_content_dir=args.replay_content_dir.resolve(),
        event_source=args.event_source,
        brain_area=args.brain_area,
        expected_sessions=args.expected_sessions,
    )


if __name__ == "__main__":
    main()
