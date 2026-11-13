"""The runner parts the table drivers share.

Not a script. The drivers in ``scripts/tables`` import it as
``scripts.tables._table_runner``, after putting the repo root on ``sys.path``.
It selects the sessions a build covers (:func:`find_sessions` for the replay
tables, :func:`find_contfrag_sessions` for the ContFrag ones), reads the
per-animal shards a split build writes (:func:`read_disjoint_shards`) and
publishes a finished table set (:func:`publish_replay_tables`,
:func:`publish_contfrag_tables`). The tables themselves are built by
``continuum_swr_replay.analysis.event_tables.prepare_data`` and
``continuum_swr_replay.analysis.contfrag.prepare_data``.

Importing it sets the drivers' log format.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    publish_swr_table,
    require_swr_source,
    stamp_swr_source,
)
from continuum_swr_replay.analysis.event_definition.swr_source import swr_epoch_id
from continuum_swr_replay.data_loaders.epoch_register import analysed_epochs
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import (
    CONTFRAG_SUFFIX,
    format_epoch_identifier,
    parse_nwb_filename,
)
from continuum_swr_replay.paths import BASENAME_TO_TABLE, processed_data_dir
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import session_key_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)


def find_sessions(max_per_animal: int | None = None) -> list[dict]:
    """The epoch register's analysed epochs, each with its decode files.

    The sessions are ``analysed_epochs()``, in ``session_id`` order within each
    animal, the animals in order; a decode on disk whose epoch the register does
    not analyse is not one. ``results.nc`` absence is recorded here as
    ``results_file=None`` and acted on in ``prepare_data.build_session_tables``,
    which refuses the session: without a posterior there is no
    ``event_sequence_table`` and no retrospective metrics.

    Parameters
    ----------
    max_per_animal : int, optional
        Only each animal's first epochs, for a smoke run.

    Returns
    -------
    list of dict
        ``nwb_file_name``, ``epoch_name``, ``session_id``, ``dot_file``,
        ``env_file`` and ``results_file`` per session.

    Raises
    ------
    FileNotFoundError
        If a selected epoch has no ``data_over_time.pkl`` or ``environment.pkl``
        under ``paths.processed_data_dir()``: the tables would otherwise cover fewer
        epochs than the register analyses.
    """
    by_animal: dict[str, list] = {}
    decode_dir = processed_data_dir()
    for nwb_file_name, epoch in analysed_epochs().index:
        animal = parse_nwb_filename(nwb_file_name)[0]
        session_id = swr_epoch_id(nwb_file_name, epoch)
        paths = decode_paths(session_id, decode_dir)
        by_animal.setdefault(animal, []).append(
            {
                "nwb_file_name": nwb_file_name,
                "epoch_name": epoch,
                "session_id": session_id,
                "dot_file": paths.data_over_time,
                "env_file": paths.environment,
                "results_file": paths.results if paths.results.exists() else None,
            }
        )

    selected = []
    for animal in sorted(by_animal):
        sessions = sorted(by_animal[animal], key=lambda session: session["session_id"])
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
            f"{len(missing)} decode file(s) of the epoch register's analysed epochs "
            f"are not under {decode_dir}: {', '.join(missing[:10])}"
            + (", ..." if len(missing) > 10 else "")
        )
    return selected


def find_contfrag_sessions(
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


def read_disjoint_shards(
    shard_dirs: Sequence[Path],
    basenames: Mapping[str, str],
    *,
    what: str,
    expected_sessions: int,
) -> list[dict[str, pd.DataFrame]]:
    """Read per-animal table shards that together make one bundle.

    A full build can run as disjoint per-animal builds (``--animal``), each into
    its own folder; the shards are then joined into one bundle. A shard's
    sessions are those of its ``stop_table``.

    Parameters
    ----------
    shard_dirs : sequence of Path
        The shard folders.
    basenames : mapping of str to str
        Each table to read, by its name in the returned dicts, to its basename
        in a shard folder. Must include ``"stop_table"``.
    what : str
        What the shards are, for the messages (``"Replay"``, ``"ContFrag"``).
    expected_sessions : int
        The number of sessions the shards must hold together.

    Returns
    -------
    list of dict of str to pd.DataFrame
        Each shard's tables, keyed as in `basenames`, in the order of
        `shard_dirs`.

    Raises
    ------
    ValueError
        If there is no shard, if two shards hold a session in common, or if the
        shards together hold other than `expected_sessions` sessions.
    SwrSourceError
        If a table was not built from the configured SWR source.
    """
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    shards: list[dict[str, pd.DataFrame]] = []
    observed_sessions: set[tuple[object, ...]] = set()
    for shard_dir in shard_dirs:
        frames = {
            name: read_table(shard_dir / basename)
            for name, basename in basenames.items()
        }
        # Every shard must come from the SWR source the bundle is stamped with.
        for name, frame in frames.items():
            require_swr_source(frame, what=f"{shard_dir / basenames[name]}")
        shard_sessions = session_key_set(frames["stop_table"])
        overlap = observed_sessions & shard_sessions
        if overlap:
            raise ValueError(
                f"{what} shards overlap on {len(overlap)} session(s), e.g. "
                f"{next(iter(overlap))}"
            )
        observed_sessions |= shard_sessions
        shards.append(frames)
    if len(observed_sessions) != expected_sessions:
        raise ValueError(
            f"assembled {what} scope has {len(observed_sessions)} sessions; "
            f"expected {expected_sessions}"
        )
    return shards


def publish_replay_tables(
    frames: Iterable[tuple[str, str, pd.DataFrame | None]], output_dir: Path
) -> None:
    """Atomically publish each (basename, table_key, frame) to `output_dir`.

    ``publish_canonical`` stamps the covariate schema and writes atomically, so
    the tables cannot land unstamped; the per-table key selects the row-key
    contract. Every frame was built from decode frames read through the SWR
    source, so each is stamped with it.
    """
    for basename, table_key, frame in frames:
        if frame is None:
            logging.info("  %s: skipped (no data)", basename)
            continue
        stamp_swr_source(frame)
        path = publish_swr_table(
            frame,
            output_dir,
            table=table_key,
            basename=basename,
        )
        logging.info("saved %s → %s (%d rows)", basename, path, len(frame))


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
    (from ``contfrag.prepare_data.contfrag_table_basenames``). Every ContFrag
    build ends here.

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
