#!/usr/bin/env python
"""Build the four canonical replay-content tables across all sessions.

Thin driver over the per-grain table builders in
``continuum_swr_replay.analysis.replay.tables_event`` / ``tables_stop`` /
``tables_arm_patch``. For
each session it loads the pre-saved ``data_over_time.pkl``,
``environment.pkl``, and ``results.nc``, pulls ``trials`` +
``track_graph`` from Spyglass, and calls the four table builders.

Outputs (under ``paths.replay_content_dir()``):

- ``event_table`` — one row per non-local event
- ``event_arm_table`` — one row per (event, arm)
- ``event_patch_table`` — one row per (event, patch); collapsed from
  ``event_arm_table`` via :func:`build_event_patch_table` so the
  ``--hierarchy patch`` path of ``fit_replay_content_models.py`` has its
  required input alongside the per-arm tables.
- ``event_sequence_table`` — one row per in-event timebin
- ``stop_table`` — one row per stopping period (incl. zero-event)

Usage::

    uv run python scripts/tables/prepare_replay_content_data.py              # full run
    uv run python scripts/tables/prepare_replay_content_data.py --test       # 2 per animal
    uv run python scripts/tables/prepare_replay_content_data.py --from-cache # covariate-only rebuild

The run itself is :func:`continuum_swr_replay.analysis.replay.prepare_data.run_prepare_data`;
this file is its command line.

**Skipping Spyglass for covariate-only changes** (``--from-cache``): the
only per-session Spyglass fetch is ``trials`` + ``track_graph`` (the ~3 h
cost of a full run). Every run, ``--from-cache`` included, still reads the
database once: the sessions are the epoch register's analysed epochs, which
``find_sessions`` checks against ``get_epoch_info``. Every normal run pickles those to
``replay_content/_session_cache/<cache-version>/<session_id>.pkl``. With ``--from-cache``
the driver reads them back instead of querying Spyglass and re-runs the
*same* table builders — so a change that only touches covariate code
(e.g. a new/re-encoded predictor, a re-standardization, a new
interaction) rebuilds the canonical tables in minutes with byte-for-byte
parity on everything else. Sessions missing from the cache transparently
fall back to Spyglass and are cached for next time, so no session is ever
dropped. Cache paths are versioned by the behavioral derivation contract, so a
definition change never overwrites or reinterprets the prior cache. Caveat:
``--from-cache`` freezes ``trials``/``track_graph`` within that version; to pick
up genuine upstream Spyglass changes (re-curation, new intervals) run a normal
pass, which refreshes only the current version.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

# Imported as a module, not by name: ``run_prepare_data`` rebinds
# ``prepare_data.OUTPUT_DIR`` from ``--output-dir``, so the flag's default is
# read (``prepare_data.output_dir()``) when the parser is built.
from continuum_swr_replay.analysis.replay import prepare_data
from continuum_swr_replay.analysis.replay.prepare_data import run_prepare_data
from continuum_swr_replay.data_loaders.epoch_register import analysed_animals


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="2 sessions per animal")
    parser.add_argument("--max-per-animal", type=int, default=None)
    parser.add_argument(
        "--animal",
        choices=analysed_animals(),
        default=None,
        help="Restrict this invocation to one animal's sessions.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Populate and validate versioned behavioral caches, then exit "
            "without assembling or writing scientific tables. Disjoint animals "
            "may be run concurrently; follow with one serial --from-cache build."
        ),
    )
    parser.add_argument(
        "--stop-only",
        action="store_true",
        help=(
            "Build only the canonical stop table needed to generate qvals, "
            "then exit before posterior/event-arm construction."
        ),
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Covariate-only rebuild: read trials/track_graph from the "
            "per-session cache instead of Spyglass (falls back to Spyglass "
            "for uncached sessions). Rebuilds all tables via the same "
            "builders in minutes. Freezes upstream at the last full run — "
            "run without this flag to refresh the cache."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=prepare_data.output_dir(),
        help=(
            "Candidate replay-content root. Defaults to the production root; "
            "pass an explicit staging directory for regeneration so existing "
            "scientific artifacts are not overwritten."
        ),
    )
    parser.add_argument(
        "--behavior-cache-dir",
        type=Path,
        default=None,
        help=(
            "Override the behavioral-cache root. This permits disjoint table "
            "shards to share one validated cache without copying it."
        ),
    )
    parser.add_argument(
        "--contingency-dir",
        type=Path,
        default=None,
        help=(
            "Replay-content root providing qvals. Defaults to --output-dir; "
            "set this when building per-animal shards."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_prepare_data(args)


if __name__ == "__main__":
    main()
