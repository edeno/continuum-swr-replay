#!/usr/bin/env python
"""Build canonical replay-content tables from ContFrag decoding results.

For each session it loads the pre-saved ``*_contfrag_data_over_time.pkl``
and ``*_contfrag_environment.pkl``, pulls ``trials`` + ``track_graph``
from the behavioral cache or Spyglass, and calls the ContFrag table builders
(:func:`continuum_swr_replay.analysis.contfrag.prepare_data.build_contfrag_session_tables`).

Events are defined externally (SWR or MUA times), not by model-detected
probability thresholding.

Outputs (under ``paths.contfrag_replay_content_dir()``), with
``{source}`` lowercase (``swr`` / ``mua``) and ``{brain_area}`` mirroring
the case used at decode time (``HPC`` / ``mPFC`` / ``OFC``):

- ``event_table_{source}_{brain_area}.pkl`` — one row per SWR (or MUA) event
- ``event_arm_table_{source}_{brain_area}.pkl`` — one row per (event, arm)
- ``event_patch_table_{source}_{brain_area}.pkl`` — one row per (event, patch),
  collapsed from the event-arm table for the patch-level hierarchy
- ``stop_table_{source}_{brain_area}.pkl`` — one row per immobility bout
  (incl. zero-event)

Each row also carries a ``brain_area`` column so cross-area concatenation
remains traceable. ``fit_replay_content_models.py`` accepts a matching
``--brain-area`` flag and resolves these per-area filenames.

Usage::

    uv run python scripts/tables/prepare_contfrag_replay_content_data.py                    # SWR / HPC
    uv run python scripts/tables/prepare_contfrag_replay_content_data.py --event-source MUA # MUA / HPC
    uv run python scripts/tables/prepare_contfrag_replay_content_data.py --brain-area mPFC  # SWR / mPFC
    uv run python scripts/tables/prepare_contfrag_replay_content_data.py --test             # 2 per animal
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates import load_contingency
from continuum_swr_replay.analysis.behavior.tables import (
    prepare_data as replay_prepare_data,
)
from continuum_swr_replay.analysis.contfrag.prepare_data import (
    build_contfrag_session_tables,
    contfrag_event_patch_table,
    contfrag_table_basenames,
)
from continuum_swr_replay.paths import (
    contfrag_replay_content_dir,
    replay_content_dir,
    table_path,
)

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tables._table_runner import (  # noqa: E402
    find_contfrag_sessions,
    publish_contfrag_tables,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)


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

    Each session goes through ``prepare_data.build_contfrag_session_tables``;
    the per-session tables are concatenated in session order, the event-patch
    table is projected from the whole event-arm table, and the four tables are
    published under ``prepare_data.contfrag_table_basenames``.

    Parameters
    ----------
    sessions : list of dict
        From :func:`find_contfrag_sessions`, all of one brain area.
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
            result = build_contfrag_session_tables(
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
            # the Replay driver, prepare_replay_content_data.run_prepare_data).
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
    event_patch_table_all = contfrag_event_patch_table(
        event_table_all,
        event_arm_table_all,
        event_source=event_source,
        brain_area=brain_area,
    )
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


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="2 sessions per animal")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Read trials/track_graph from the NonLocal per-session cache "
            "instead of Spyglass (falls back to Spyglass per session when an "
            "entry is missing). Required on hosts where NWB reads fail."
        ),
    )
    parser.add_argument("--max-per-animal", type=int, default=None)
    parser.add_argument(
        "--animal",
        default=None,
        help=(
            "Restrict the build to one animal. Use with an isolated "
            "--output-dir and assemble the disjoint shards with "
            "assemble_contfrag_shards.py."
        ),
    )
    parser.add_argument(
        "--event-source",
        type=str,
        default="SWR",
        choices=["SWR", "MUA"],
        help="External event detector for event boundaries",
    )
    parser.add_argument(
        "--brain-area",
        type=str,
        default="HPC",
        choices=["HPC", "mPFC", "OFC"],
        help=(
            "Restrict prep to one brain area. Without this filter, the "
            "script would silently merge per-area outputs even though "
            "event_id / stop_id hashes do not encode brain_area, which "
            "would corrupt downstream joins."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=contfrag_replay_content_dir(),
        help=(
            "Candidate ContFrag root. Pass an explicit staging directory so "
            "existing scientific artifacts are not overwritten."
        ),
    )
    parser.add_argument(
        "--replay-content-dir",
        type=Path,
        default=replay_content_dir(),
        help=(
            "Replay-content root providing qvals and, unless "
            "--behavior-cache-dir is given, the version-matched behavioral "
            "cache."
        ),
    )
    parser.add_argument(
        "--behavior-cache-dir",
        type=Path,
        default=None,
        help=(
            "Behavioral-cache root. Defaults to <--replay-content-dir>/"
            "_session_cache; set it to keep reading the canonical cache "
            "while --replay-content-dir points at a staged qvals table."
        ),
    )
    return parser


def configure_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the output and replay-content roots, and set the behavioral cache.

    Sets ``replay_prepare_data.SESSION_CACHE_DIR``, the cache that
    ``build_contfrag_session_tables`` reads trials and the track graph from, to the
    resolved ``--behavior-cache-dir`` when given, else to
    ``<replay_content_dir>/_session_cache``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from :func:`build_parser`.

    Returns
    -------
    output_dir : Path
        The resolved ``--output-dir``.
    replay_content_dir : Path
        The resolved ``--replay-content-dir``, which supplies the qvals table.
    """
    output_dir = args.output_dir.resolve()
    replay_content_dir = args.replay_content_dir.resolve()
    replay_prepare_data.SESSION_CACHE_DIR = (
        args.behavior_cache_dir.resolve()
        if args.behavior_cache_dir is not None
        else replay_content_dir / "_session_cache"
    )
    return output_dir, replay_content_dir


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    output_dir, replay_content_dir = configure_paths(args)

    max_per_animal = 2 if args.test else args.max_per_animal
    sessions = find_contfrag_sessions(
        max_per_animal=max_per_animal, brain_area=args.brain_area
    )
    if args.animal is not None:
        sessions = [
            session
            for session in sessions
            if session["nwb_file_name"].startswith(args.animal)
        ]
        if not sessions:
            raise SystemExit(
                f"no {args.brain_area} ContFrag sessions found for animal "
                f"{args.animal!r}"
            )
    logging.info(
        "Found %d %s sessions, event_source=%s",
        len(sessions),
        args.brain_area,
        args.event_source,
    )

    # Shared per-stop value artifact (qvals_table, exact poke_out alignment).
    # ContFrag's own stop_id mixes in brain_area/event_source, so the builder
    # recomputes the plain (animal,day,epoch,stop_idx) hash for the join.
    contingency = load_contingency(table_path("qvals_table", replay_content_dir))
    build_contfrag_tables(
        sessions,
        event_source=args.event_source,
        brain_area=args.brain_area,
        contingency=contingency,
        from_cache=args.from_cache,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
