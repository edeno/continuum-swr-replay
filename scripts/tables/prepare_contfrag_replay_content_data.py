#!/usr/bin/env python
"""Build canonical replay-content tables from ContFrag decoding results.

Command-line front end of
:func:`continuum_swr_replay.analysis.contfrag.prepare_data.build_contfrag_tables`.
For each session it loads the pre-saved ``*_contfrag_data_over_time.pkl``
and ``*_contfrag_environment.pkl``, pulls ``trials`` + ``track_graph``
from the behavioral cache or Spyglass, and calls the ContFrag table builders.

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
from collections.abc import Sequence
from pathlib import Path

from continuum_swr_replay.analysis.behavior.covariates import load_contingency
from continuum_swr_replay.analysis.behavior.tables import (
    prepare_data as replay_prepare_data,
)
from continuum_swr_replay.analysis.contfrag.prepare_data import (
    build_contfrag_tables,
    find_sessions,
)
from continuum_swr_replay.paths import (
    contfrag_replay_content_dir,
    replay_content_dir,
    table_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)


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
    ``_process_session`` reads trials and the track graph from, to the
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
    sessions = find_sessions(max_per_animal=max_per_animal, brain_area=args.brain_area)
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
