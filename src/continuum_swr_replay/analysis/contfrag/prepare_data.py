"""Build the ContFrag replay-content tables across sessions.

Driver over the per-session builders in
:mod:`continuum_swr_replay.analysis.contfrag.tables`: for each session it loads
the saved ContFrag decode (``*_contfrag_data_over_time.pkl``,
``*_contfrag_environment.pkl`` and the combined posterior of
``*_contfrag_results.nc``), the session's trials and track graph from the
behavioral cache that the replay build shares
(``analysis.behavior.tables.prepare_data.load_behavioral_context``), builds the
event, event-arm and stop tables, and publishes them with the event-patch table
under per-(event source, brain area) basenames.
:func:`build_contfrag_tables` is the whole run; its command-line front end is
``scripts/tables/prepare_contfrag_replay_content_data.py``, where the outputs
and flags are documented.

Events are defined externally (SWR or MUA times), not by model-detected
probability thresholding.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior.tables import (
    prepare_data as replay_prepare_data,
)
from continuum_swr_replay.analysis.behavior.tables.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)
from continuum_swr_replay.analysis.behavior.tables.tables_arm_patch import (
    build_event_patch_table,
)
from continuum_swr_replay.analysis.contfrag.posterior_analysis import (
    get_contfrag_posterior,
)
from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_arm_table,
    build_contfrag_event_table,
    build_contfrag_stop_table,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    publish_swr_table,
    read_dot_with_swr_source,
    stamp_swr_source,
)
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epochs
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import (
    CONTFRAG_SUFFIX,
    format_epoch_identifier,
    parse_nwb_filename,
)
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.paths import (
    BASENAME_TO_TABLE,
    processed_data_dir,
)
from continuum_swr_replay.tables.covariate_registry import (
    stamp_covariate_schema,
)


def contfrag_table_basenames(event_source: str, brain_area: str) -> dict[str, str]:
    """Return the canonical ContFrag table basenames for one (source, area).

    The fitter resolves its inputs through the matching helper
    ``resolve_contfrag_paths`` in ``analysis/behavior/content/fit_outputs.py``;
    keeping the format string in one place lets a unit test pin the
    prep ↔ fitter contract.
    """
    src = event_source.lower()
    return {
        "event_table": f"event_table_{src}_{brain_area}",
        "event_arm_table": f"event_arm_table_{src}_{brain_area}",
        "event_patch_table": f"event_patch_table_{src}_{brain_area}",
        "stop_table": f"stop_table_{src}_{brain_area}",
    }


def publish_contfrag_tables(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_patch_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    *,
    output_dir: Path,
    basenames: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Stamp the four ContFrag tables and publish them to ``output_dir``.

    Each table is stamped with its covariate schema and the SWR source, then
    written through ``publish_canonical`` under its name in ``basenames``
    (from :func:`contfrag_table_basenames`). Every ContFrag build ends here.

    Returns
    -------
    dict of str to pd.DataFrame
        The stamped tables, keyed ``event_table``, ``event_arm_table``,
        ``event_patch_table`` and ``stop_table``.
    """
    tables = {
        "event_table": stamp_covariate_schema(event_table, table="event"),
        "event_arm_table": stamp_covariate_schema(event_arm_table, table="event_arm"),
        "event_patch_table": stamp_covariate_schema(
            event_patch_table, table="event_patch"
        ),
        "stop_table": stamp_covariate_schema(stop_table, table="stop"),
    }
    for frame in tables.values():
        stamp_swr_source(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        # The one write door: the source check, then publish_canonical (re-stamp
        # idempotent, atomic).
        path = publish_swr_table(
            frame,
            output_dir,
            table=BASENAME_TO_TABLE[name],
            basename=basenames[name],
        )
        logging.info("saved %s -> %s (%d rows)", name, path, len(frame))
    return tables


def find_sessions(
    max_per_animal: int | None = None,
    brain_area: str = "HPC",
) -> list[dict]:
    """The epoch register's analysed epochs, each with its ContFrag decode files.

    One brain area at a time: ``run_decoding_contfrag.py`` writes per-area files,
    and the saved tables would collide on event_id / stop_id across areas,
    because the canonical hashes do not include brain_area (see
    ``tables.event_ids.hash_event_id`` / ``hash_stop_id``).

    The sessions are ``analysed_epochs()``, in ``session_id`` order within each
    animal, the animals in order. For HPC, recorded in every analysed epoch, an
    analysed epoch without its decode stops the build. mPFC and OFC are recorded
    in some sessions only, so an analysed epoch with no decode for the area is
    outside that area's cohort and is left out.

    Parameters
    ----------
    max_per_animal : int, optional
        Only each animal's first epochs, for a smoke run.
    brain_area : str, optional
        ``HPC`` (default), ``mPFC`` or ``OFC``.

    Returns
    -------
    list of dict
        ``nwb_file_name``, ``epoch_name``, ``brain_area``, ``dot_file``,
        ``env_file`` and ``results_file`` per session.

    Raises
    ------
    FileNotFoundError
        If, for HPC, a selected epoch has no ContFrag ``data_over_time.pkl`` or
        ``environment.pkl`` under ``paths.processed_data_dir()``.
    """
    by_animal: dict[str, list] = {}
    decode_dir = processed_data_dir()
    for nwb_file_name, epoch in analysed_epochs().index:
        animal = parse_nwb_filename(nwb_file_name)[0]
        session_id = format_epoch_identifier(
            nwb_file_name, epoch, brain_area, CONTFRAG_SUFFIX
        )
        paths = decode_paths(session_id, decode_dir, threshold_percent=None)
        dot_file = paths.data_over_time
        env_file = paths.environment
        if brain_area != "HPC" and not (dot_file.exists() and env_file.exists()):
            continue
        results_path = paths.results
        by_animal.setdefault(animal, []).append(
            {
                "nwb_file_name": nwb_file_name,
                "epoch_name": epoch,
                "brain_area": brain_area,
                "dot_file": dot_file,
                "env_file": env_file,
                "results_file": results_path if results_path.exists() else None,
            }
        )

    selected = []
    for animal in sorted(by_animal):
        sessions = sorted(by_animal[animal], key=lambda session: session["dot_file"])
        if max_per_animal is not None:
            sessions = sessions[:max_per_animal]
        selected.extend(sessions)

    missing = [
        path.name
        for session in selected
        for path in (session["dot_file"], session["env_file"])
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} ContFrag decode file(s) of the epoch register's "
            f"analysed epochs are not under {decode_dir}: "
            f"{', '.join(missing[:10])}" + (", ..." if len(missing) > 10 else "")
        )
    return selected


def _load_contfrag_posterior(results_file):
    """Load the combined ContFrag posterior from a results NetCDF."""
    return get_contfrag_posterior(open_results(results_file), state="combined")


def _process_session(
    session: dict,
    event_source: str,
    contingency: pd.DataFrame | None = None,
    *,
    from_cache: bool = False,
) -> dict[str, pd.DataFrame] | None:
    """Run ContFrag table builders on one session."""
    nwb = session["nwb_file_name"]
    epoch = session["epoch_name"]
    brain_area = session["brain_area"]

    # Gate before any loading, matching the replay driver: the dot pickle and
    # the Spyglass trials query are not worth paying for a session we refuse.
    if session["results_file"] is None:
        raise ValueError(
            f"no results.nc for {nwb} {epoch}: refusing to build a ContFrag "
            "session without a conditional posterior (posterior_*, "
            "arm_posterior_mass and arm_fraction_posterior would all be NaN)."
        )

    # SWR labels (the ContFrag SWR events themselves) from the one SWR source
    dot = read_dot_with_swr_source(session["dot_file"])
    # Fail closed on input-contract violations before they silently corrupt the
    # tables; a violation here is caught by the per-session skip+tally of
    # build_contfrag_tables.
    validate_dot_event_contract(dot)
    with open(session["env_file"], "rb") as f:
        env = pickle.load(f)

    # Reuse the NonLocal per-session trials/track_graph cache. Those are
    # session facts, not decoder outputs, so the ContFrag build has no reason
    # to re-fetch them -- and on a host where NWB reads fail (hdmf namespace
    # mismatch) the cache is the only way this pipeline runs at all. The cache
    # is keyed by the NonLocal session_id, which is this one without the
    # ``_contfrag`` suffix.
    cache_key = format_epoch_identifier(nwb, epoch, brain_area)
    trials, port_experiences, track_graph, _ = (
        replay_prepare_data.load_behavioral_context(
            {
                "nwb_file_name": nwb,
                "epoch_name": epoch,
                "session_id": cache_key,
            },
            from_cache=from_cache,
            cache_key=cache_key,
        )
    )
    if trials is None or port_experiences is None or track_graph is None:
        return None

    try:
        posterior = _load_contfrag_posterior(session["results_file"])
    except Exception as e:  # noqa: BLE001
        # Exists but will not load -> same degraded state the gate refuses.
        raise ValueError(
            f"results.nc for {nwb} {epoch} exists but could not be loaded "
            f"({e}); refusing to build the ContFrag session without a "
            "conditional posterior."
        ) from e

    # Build the stop table first so zero-event sessions still
    # contribute their immobility bouts. Returning early on an empty
    # event_table dropped those sessions, contradicting the
    # ``stop_table_{source}_{brain_area}.pkl`` contract that
    # documents zero-event bouts as part of the canonical output.
    stop_table = build_contfrag_stop_table(
        dot,
        trials,
        port_experiences,
        track_graph,
        event_source=event_source,
        brain_area=brain_area,
    )

    event_table = build_contfrag_event_table(
        dot,
        env,
        trials,
        port_experiences,
        track_graph,
        event_source=event_source,
        brain_area=brain_area,
        conditional_posterior=posterior,
    )

    event_arm_table = build_contfrag_event_arm_table(
        event_table,
        dot,
        env,
        trials,
        port_experiences,
        track_graph,
        event_source=event_source,
        brain_area=brain_area,
        conditional_posterior=posterior,
        contingency=contingency,
    )

    # Stamp provenance onto every output row. The canonical event_id /
    # stop_id hashes already incorporate brain_area + event_source, but
    # the columns keep saved tables self-describing.
    for tbl in (event_table, event_arm_table, stop_table):
        tbl["event_source"] = event_source
        tbl["brain_area"] = brain_area

    return {
        "event_table": event_table,
        "event_arm_table": event_arm_table,
        "stop_table": stop_table,
    }


def build_contfrag_tables(
    sessions: list[dict],
    *,
    event_source: str,
    brain_area: str,
    contingency: pd.DataFrame | None,
    from_cache: bool,
    output_dir: Path,
) -> dict[str, pd.DataFrame] | None:
    """Build every session's ContFrag tables, assemble them, and publish them.

    Each session goes through :func:`_process_session`; the per-session tables
    are concatenated in session order, the event-patch table is projected from
    the whole event-arm table, and the four tables are published under
    :func:`contfrag_table_basenames`.

    Parameters
    ----------
    sessions : list of dict
        From :func:`find_sessions`, all of one brain area.
    event_source : str
        ``"SWR"`` or ``"MUA"``: the external detector whose events are used.
    brain_area : str
        The sessions' brain area, ``"HPC"``, ``"mPFC"`` or ``"OFC"``.
    contingency : pd.DataFrame or None
        The ``qvals_table`` (``covariates.load_contingency``) the event-arm
        value block is joined from.
    from_cache : bool
        Read trials and the track graph from the behavioral cache, falling back
        to Spyglass per session (``prepare_data.load_behavioral_context``).
    output_dir : Path
        Where the four tables are published.

    Returns
    -------
    dict of str to pd.DataFrame or None
        The published tables (:func:`publish_contfrag_tables`), or None when no
        session produced tables or stop/event count conservation fails, in
        which case nothing is written.

    Raises
    ------
    RuntimeError
        If any session failed to build; nothing is written.
    """
    t0 = time.time()
    all_event_tables: list[pd.DataFrame] = []
    all_event_arm_tables: list[pd.DataFrame] = []
    all_stop_tables: list[pd.DataFrame] = []

    skipped: list[tuple[str, str, str]] = []
    for i, session in enumerate(sessions):
        nwb = session["nwb_file_name"]
        epoch = session["epoch_name"]

        try:
            result = _process_session(
                session,
                event_source,
                contingency=contingency,
                from_cache=from_cache,
            )
        except (FileNotFoundError, OSError, ValueError) as e:
            logging.warning("skip %s %s: %s", nwb, epoch, e)
            skipped.append((nwb, epoch, str(e)))
            continue

        if result is None:
            # Behavioral context / track graph unavailable. This is a failure,
            # not a benign skip: count it toward the fail-closed gate below so a
            # dropped session cannot leave a complete-looking table set (mirrors
            # the Replay driver, analysis/behavior/tables/prepare_data.run_prepare_data).
            reason = "behavioral context or track graph unavailable"
            logging.warning("skip %s %s: %s", nwb, epoch, reason)
            skipped.append((nwb, epoch, reason))
            continue

        all_event_tables.append(result["event_table"])
        all_event_arm_tables.append(result["event_arm_table"])
        all_stop_tables.append(result["stop_table"])

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

    if skipped:
        logging.error(
            "SKIPPED %d/%d session(s) — value block / tables are MISSING these:",
            len(skipped),
            len(sessions),
        )
        for nwb, epoch, reason in skipped:
            logging.error("  skipped %s %s: %s", nwb, epoch, reason)
        raise RuntimeError(
            "ContFrag build is incomplete: "
            f"{len(skipped)} of {len(sessions)} session(s) failed. "
            "No scientific tables were written."
        )

    if not all_event_tables:
        logging.error("no sessions produced data")
        return None

    event_table_all = pd.concat(all_event_tables, ignore_index=True)
    event_arm_table_all = pd.concat(all_event_arm_tables, ignore_index=True)
    stop_table_all = pd.concat(all_stop_tables, ignore_index=True)

    session_cols = ["animal", "session_date", "epoch"]
    logging.info(
        "events: %d, event_arm rows: %d, stops: %d, sessions: %d",
        len(event_table_all),
        len(event_arm_table_all),
        len(stop_table_all),
        event_table_all.groupby(session_cols).ngroups,
    )

    # Cross-table invariant: Σ stop_table.n_selected_events == len(event_table)
    sum_n_events = int(stop_table_all["n_selected_events"].sum())
    if sum_n_events != len(event_table_all):
        logging.error(
            "cross-table drift: Σ stop_table.n_selected_events = %d, "
            "len(event_table) = %d — refusing to write output",
            sum_n_events,
            len(event_table_all),
        )
        return None
    logging.info(
        "cross-table invariant OK: Σ n_selected_events == len(event_table) == %d",
        len(event_table_all),
    )

    # Collapse to the event-patch table for the patch-level hierarchy.
    # Built from the concatenated frames so event_id is globally unique
    # (the ContFrag hash already folds in brain_area + event_source),
    # which build_event_patch_table requires. Provenance is re-stamped
    # because the helper carries the canonical session keys.
    event_patch_table_all = build_event_patch_table(
        event_table_all, event_arm_table_all
    )
    event_patch_table_all["event_source"] = event_source
    event_patch_table_all["brain_area"] = brain_area
    logging.info(
        "event_patch rows: %d (= 3 × %d events)",
        len(event_patch_table_all),
        event_table_all.groupby(session_cols)["event_id"].nunique().sum(),
    )

    # Per-brain-area filenames so HPC and mPFC prep runs don't overwrite each
    # other. Built here from decode frames read through the SWR source.
    tables = publish_contfrag_tables(
        event_table_all,
        event_arm_table_all,
        event_patch_table_all,
        stop_table_all,
        output_dir=output_dir,
        basenames=contfrag_table_basenames(event_source, brain_area),
    )
    logging.info("total time: %ds", int(time.time() - t0))
    return tables


def assemble_contfrag_shards(
    shard_dirs: list[Path],
    *,
    output_dir: Path,
    event_source: str,
    brain_area: str,
    expected_sessions: int,
) -> dict[str, pd.DataFrame]:
    """Join per-animal shards of the ContFrag tables into one published bundle.

    The shards' event, event-arm and stop tables are concatenated in shard
    order and the event-patch table is projected from the whole event-arm
    table, as :func:`build_contfrag_tables` does over every session at once.

    Parameters
    ----------
    shard_dirs : list of Path
        The shard folders, each written by :func:`build_contfrag_tables` for
        one animal.
    output_dir : Path
        Where the four tables are published.
    event_source : str
        ``"SWR"`` or ``"MUA"``.
    brain_area : str
        ``"HPC"``, ``"mPFC"`` or ``"OFC"``.
    expected_sessions : int
        The number of sessions the shards must hold together.

    Returns
    -------
    dict of str to pd.DataFrame
        The published tables (:func:`publish_contfrag_tables`).

    Raises
    ------
    ValueError
        If the shards overlap or miss sessions, a shard or the bundle breaks
        stop/event count conservation, or an ``event_id`` or ``stop_id`` is
        missing or repeats.
    """
    basenames = contfrag_table_basenames(event_source, brain_area)
    shards = replay_prepare_data.read_disjoint_shards(
        shard_dirs,
        {
            name: basenames[name]
            for name in ("event_table", "event_arm_table", "stop_table")
        },
        what="ContFrag",
        expected_sessions=expected_sessions,
    )
    for shard_dir, frames in zip(shard_dirs, shards, strict=True):
        validate_event_arm_table(
            frames["event_arm_table"], event_table=frames["event_table"]
        )
        if int(frames["stop_table"]["n_selected_events"].sum()) != len(
            frames["event_table"]
        ):
            raise ValueError(f"{shard_dir} violates stop/event count conservation")

    event_table = pd.concat(
        [frames["event_table"] for frames in shards], ignore_index=True
    )
    event_arm_table = pd.concat(
        [frames["event_arm_table"] for frames in shards], ignore_index=True
    )
    stop_table = pd.concat(
        [frames["stop_table"] for frames in shards], ignore_index=True
    )
    if (
        event_table["event_id"].isna().any()
        or event_table["event_id"].duplicated().any()
    ):
        raise ValueError("assembled event_table requires unique nonmissing event_id")
    if stop_table["stop_id"].isna().any() or stop_table["stop_id"].duplicated().any():
        raise ValueError("assembled stop_table requires unique nonmissing stop_id")
    validate_event_arm_table(event_arm_table, event_table=event_table)
    if int(stop_table["n_selected_events"].sum()) != len(event_table):
        raise ValueError("assembled stop/event count conservation failed")

    event_patch_table = build_event_patch_table(event_table, event_arm_table)
    event_patch_table["event_source"] = event_source
    event_patch_table["brain_area"] = brain_area
    validate_event_patch_table(event_patch_table, event_table=event_table)

    return publish_contfrag_tables(
        event_table,
        event_arm_table,
        event_patch_table,
        stop_table,
        output_dir=output_dir,
        basenames=basenames,
    )
