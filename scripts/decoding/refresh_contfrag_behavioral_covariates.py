#!/usr/bin/env python
"""Rebuild ContFrag tables while reusing immutable posterior aggregates.

Behavioral clocks, locations, histories, event boundaries, hard counts, and
q-value attachments are recomputed from their canonical inputs.  Only values
that require the expensive conditional posterior are carried from a preserved
source bundle, after exact event/candidate-key validation.  This makes the
dependency boundary explicit and avoids re-reading 218 large posterior files
when the decoder results themselves have not changed.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_contfrag_replay_content_data import (
    contfrag_table_basenames,
    find_sessions,
    publish_contfrag_tables,
)

from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_arm_table,
    build_contfrag_event_table,
    build_contfrag_stop_table,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    read_dot_with_swr_source,
    require_swr_source,
)
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.replay import build_event_patch_table
from continuum_swr_replay.analysis.replay import prepare_data as replay_prepare_data
from continuum_swr_replay.analysis.replay.covariates import load_contingency
from continuum_swr_replay.analysis.replay.summaries import (
    arm_fraction_from_event_total,
)
from continuum_swr_replay.analysis.replay.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

LOGGER = logging.getLogger(__name__)
EVENT_POSTERIOR_COLUMNS = (
    "posterior_peak_mean",
    "posterior_entropy_mean",
    "posterior_hpd_arm_count_mean_coverage_50pct",
    "posterior_hpd_arm_count_mean_coverage_80pct",
    "posterior_hpd_arm_count_mean_coverage_95pct",
    "multimodal_fraction",
)


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


def _copy_by_key(
    target: pd.DataFrame,
    source: pd.DataFrame,
    *,
    keys: list[str],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if target.duplicated(keys).any() or source.duplicated(keys).any():
        raise ValueError(f"posterior transfer requires unique keys {keys}")
    missing = sorted(set(columns) - set(source))
    if missing:
        raise ValueError(f"posterior source is missing columns {missing}")
    left_keys = pd.MultiIndex.from_frame(target.loc[:, keys])
    right = source.copy()
    right.index = pd.MultiIndex.from_frame(source.loc[:, keys])
    missing_keys = left_keys.difference(right.index)
    extra_keys = right.index.difference(left_keys)
    if len(missing_keys) or len(extra_keys):
        raise ValueError(
            f"posterior source key mismatch: missing={len(missing_keys)}, "
            f"extra={len(extra_keys)}"
        )
    output = target.copy()
    aligned = right.loc[left_keys, list(columns)].reset_index(drop=True)
    for column in columns:
        output[column] = aligned[column].to_numpy()
    return output


def _rebuild_session(
    descriptor: dict,
    *,
    event_source: str,
    contingency: pd.DataFrame,
    old_event: pd.DataFrame,
    old_arm: pd.DataFrame,
    from_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dot = read_dot_with_swr_source(descriptor["dot_file"])
    validate_dot_event_contract(dot)
    with Path(descriptor["env_file"]).open("rb") as stream:
        environment = pickle.load(stream)
    nwb = descriptor["nwb_file_name"]
    epoch = descriptor["epoch_name"]
    brain_area = descriptor["brain_area"]
    cache_key = f"{nwb.replace('_.nwb', '')}_{epoch}_{brain_area}"
    trials, experiences, graph, _ = replay_prepare_data.load_behavioral_context(
        {
            "nwb_file_name": nwb,
            "epoch_name": epoch,
            "session_id": cache_key,
        },
        from_cache=from_cache,
        cache_key=cache_key,
    )
    if trials is None or experiences is None or graph is None:
        raise ValueError(f"missing behavioral context for {cache_key}")

    event = build_contfrag_event_table(
        dot,
        environment,
        trials,
        experiences,
        graph,
        event_source=event_source,
        brain_area=brain_area,
        conditional_posterior=None,
    )
    event_label = f"{event_source}_event"
    event = _copy_by_key(
        event,
        old_event,
        keys=[event_label],
        columns=EVENT_POSTERIOR_COLUMNS,
    )
    arm = build_contfrag_event_arm_table(
        event,
        dot,
        environment,
        trials,
        experiences,
        graph,
        event_source=event_source,
        brain_area=brain_area,
        conditional_posterior=None,
        contingency=contingency,
    )
    arm = _copy_by_key(
        arm,
        old_arm,
        keys=[event_label, "candidate_arm"],
        columns=("arm_posterior_mass",),
    )
    arm["arm_fraction_posterior"] = arm_fraction_from_event_total(
        arm["arm_posterior_mass"].to_numpy(),
        arm["n_timebins"].to_numpy(dtype=float),
    )
    stop = build_contfrag_stop_table(
        dot,
        trials,
        experiences,
        graph,
        event_source=event_source,
        brain_area=brain_area,
    )
    for table in (event, arm, stop):
        table["event_source"] = event_source
        table["brain_area"] = brain_area
    return event, arm, stop


def refresh_contfrag_behavioral_covariates(
    *,
    source_dir: Path,
    output_dir: Path,
    replay_content_dir: Path,
    event_source: str,
    brain_area: str,
    expected_sessions: int,
) -> dict[str, pd.DataFrame]:
    """Rebuild non-posterior quantities and publish a validated bundle."""
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
    contingency = load_contingency(
        replay_content_dir / "models_per_bin_rate" / "qvals_table.pkl"
    )
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
    for descriptor in find_sessions(brain_area=brain_area):
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
        event, arm, stop = _rebuild_session(
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
    event_patch_table = build_event_patch_table(event_table, event_arm_table)
    event_patch_table["event_source"] = event_source
    event_patch_table["brain_area"] = brain_area
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


def main() -> None:
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
    args = parser.parse_args()
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
