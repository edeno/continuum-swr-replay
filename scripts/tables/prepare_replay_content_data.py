#!/usr/bin/env python
"""Build the four canonical replay-content tables across all sessions.

Thin driver over the per-grain table builders in
``continuum_swr_replay.analysis.event_tables.event_table`` / ``stop_table`` /
``arm_patch_tables``. For
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

Each session's tables are built by
:func:`continuum_swr_replay.analysis.event_tables.prepare_data.build_session_tables`;
this file runs the session loop, assembles, standardizes and publishes.

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
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pandas.errors import MergeError

try:  # datajoint does not connect on import; guard so offline runs still work
    from datajoint.errors import DataJointError as _DataJointError
except ImportError:  # pragma: no cover - datajoint always present in practice

    class _DataJointError(Exception):
        """Placeholder so the skip clause is well-formed without datajoint."""


from continuum_swr_replay.analysis.behavior.covariates import load_contingency
from continuum_swr_replay.analysis.decode_quality.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.event_tables import prepare_data
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    assemble_canonical_tables,
    build_session_tables,
    load_behavioral_context,
    standardize_event_arm,
)
from continuum_swr_replay.analysis.event_tables.stop_table import build_stop_table
from continuum_swr_replay.analysis.events.analysed_time import (
    publish_swr_table,
    read_dot_with_swr_source,
    stamp_swr_source,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_animals
from continuum_swr_replay.paths import (
    STANDARDIZATION_REFERENCE,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tables._table_runner import (  # noqa: E402
    find_sessions,
    publish_replay_tables,
)


def populate_behavioral_context_cache(
    sessions: list[dict], *, from_cache: bool
) -> None:
    """Populate and validate versioned behavioral caches without building tables.

    This is the safe parallelization boundary for a full regeneration: callers
    may assign disjoint animal/session sets to independent processes because
    every cache file has a unique session key. The scientific tables are still
    assembled once, serially, after all expected caches exist.
    """
    failures: list[dict[str, str]] = []
    for index, session in enumerate(sessions, start=1):
        try:
            trials, experiences, graph, hit_cache = load_behavioral_context(
                session, from_cache=from_cache
            )
        except (FileNotFoundError, OSError, ValueError, _DataJointError) as exc:
            failures.append(
                {
                    "nwb_file_name": session["nwb_file_name"],
                    "epoch": session["epoch_name"],
                    "reason": str(exc),
                }
            )
            continue
        if trials is None or experiences is None or graph is None:
            failures.append(
                {
                    "nwb_file_name": session["nwb_file_name"],
                    "epoch": session["epoch_name"],
                    "reason": "behavioral context or track graph unavailable",
                }
            )
            continue
        logging.info(
            "cache [%d/%d] %s %s: %s",
            index,
            len(sessions),
            session["nwb_file_name"],
            session["epoch_name"],
            "validated existing entry" if hit_cache else "refreshed from Spyglass",
        )

    if failures:
        raise RuntimeError(
            "behavioral cache build is incomplete: "
            f"{len(failures)} of {len(sessions)} session(s) failed; "
            f"sample={failures[:10]}"
        )


_STOP_SPLIT_COLUMNS = frozenset({"n_nonlocal_swr_events", "n_nonlocal_non_swr_events"})


def _refuse_to_clobber_complete_stop_table(output_dir: Path) -> None:
    """Raise if ``output_dir`` already holds a *complete* stop table.

    A complete stop table carries the SWR split columns that only the full
    build produces. The ``--stop-only`` bootstrap produces a split-less table,
    so overwriting a complete one would silently downgrade it.
    """
    target = table_path("stop_table", output_dir)
    if not target.exists():
        return
    if _STOP_SPLIT_COLUMNS.issubset(pq.read_schema(target).names):
        raise FileExistsError(
            f"{target} is a complete stop table (has the SWR split); the "
            "--stop-only bootstrap would overwrite it with a split-less table. "
            "Point --output-dir at a staging directory instead."
        )


def build_stop_table_scope(
    sessions: list[dict], *, from_cache: bool, output_dir: Path
) -> pd.DataFrame:
    """Build only the qvals-authoritative stop table for one scope.

    Qvals requires current stop IDs/times, while the complete event-arm table
    requires qvals before behavioral-model columns can be standardized. This
    explicit bootstrap avoids publishing an all-NaN value block.
    """
    # Fail fast, before any session work: this bootstrap deliberately omits the
    # SWR split (n_nonlocal_swr_events / n_nonlocal_non_swr_events), which the
    # full build adds via add_nonlocal_swr_split once the event table exists.
    # Writing it over a *complete* canonical stop table would downgrade that
    # table and desynchronize it from the event/arm tables (the atomic-set
    # invariant in CLAUDE.md), and every downstream rate fit would then KeyError
    # on the missing split columns. Refuse rather than clobber.
    _refuse_to_clobber_complete_stop_table(output_dir)

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for index, session in enumerate(sessions, start=1):
        try:
            dot = read_dot_with_swr_source(session["dot_file"])
            validate_dot_event_contract(dot)
            trials, experiences, graph, hit_cache = load_behavioral_context(
                session, from_cache=from_cache
            )
            if trials is None or experiences is None or graph is None:
                raise ValueError("behavioral context or track graph unavailable")
            frames.append(build_stop_table(dot, trials, experiences, graph))
            logging.info(
                "stop [%d/%d] %s %s: behavioral context from %s",
                index,
                len(sessions),
                session["nwb_file_name"],
                session["epoch_name"],
                "cache" if hit_cache else "Spyglass",
            )
        except (FileNotFoundError, OSError, ValueError, _DataJointError) as exc:
            failures.append(
                {
                    "nwb_file_name": session["nwb_file_name"],
                    "epoch": session["epoch_name"],
                    "reason": str(exc),
                }
            )

    if failures:
        raise RuntimeError(
            "stop-only build is incomplete: "
            f"{len(failures)} of {len(sessions)} session(s) failed; "
            f"sample={failures[:10]}. No stop table was written."
        )
    stop_table = pd.concat(frames, ignore_index=True)
    if stop_table["stop_id"].isna().any() or stop_table["stop_id"].duplicated().any():
        raise ValueError("stop-only build requires globally unique nonmissing stop_id")
    observed_sessions = stop_table[list(SESSION_COLUMNS)].drop_duplicates()
    if len(observed_sessions) != len(sessions):
        raise ValueError(
            f"stop-only build retained {len(observed_sessions)} of "
            f"{len(sessions)} expected sessions"
        )
    _refuse_to_clobber_complete_stop_table(output_dir)  # re-check after the slow build
    output_dir.mkdir(parents=True, exist_ok=True)
    # Built above from decode frames read through the SWR source.
    stamp_swr_source(stop_table)
    path = publish_swr_table(
        stop_table,
        output_dir,
        table="stop",
        basename="stop_table",
    )
    logging.info("saved stop_table -> %s (%d rows)", path, len(stop_table))
    return stop_table


def _build_all_session_tables(
    sessions: list[dict],
    contingency: pd.DataFrame | None,
    *,
    from_cache: bool,
    t0: float,
) -> tuple[list, list, list, list]:
    """Run every session through ``build_session_tables`` and collect its tables.

    Data-gap failures (missing/corrupt input, unexpected shape) are skipped and
    recorded; a code bug (``KeyError`` / ``AttributeError`` / ``AssertionError``)
    propagates so a regression is not silently skipped across the run. A
    missing required column raises ``ValueError``
    (:func:`~continuum_swr_replay.tables.columns.require_columns`), so it is
    recorded as a session failure rather than propagated. If any session
    failed, raise before returning so a partial corpus is never assembled or
    written. Returns the per-session event / event-arm / stop /
    event-sequence table lists.
    """
    all_event_tables: list[pd.DataFrame] = []
    all_event_arm_tables: list[pd.DataFrame] = []
    all_stop_tables: list[pd.DataFrame] = []
    all_event_sequence_tables: list[pd.DataFrame] = []

    session_failures: list[dict[str, str]] = []
    for i, session in enumerate(sessions):
        nwb = session["nwb_file_name"]
        epoch = session["epoch_name"]

        try:
            result = build_session_tables(
                session, contingency=contingency, from_cache=from_cache
            )
        except MergeError:
            # A one-to-one merge violation (duplicate event_id from the
            # retrospective metrics) is an integrity bug, not a data gap.
            # MergeError subclasses ValueError, so it must be re-raised here
            # *before* the data-gap clause or it would be silently skipped.
            raise
        except (FileNotFoundError, OSError, ValueError, _DataJointError) as e:
            # Skip-and-continue is only appropriate for data-gap
            # errors (missing file, corrupt input, unexpected shape).
            # ``KeyError`` / ``AttributeError`` / ``AssertionError``
            # are propagated — they indicate a code bug and silent
            # skips would hide the regression across the run. A missing
            # required column is a ValueError: it lands here and fails the
            # run after the loop.
            logging.warning("skip %s %s: %s", nwb, epoch, e)
            session_failures.append(
                {"nwb_file_name": nwb, "epoch": epoch, "reason": str(e)}
            )
            continue

        if result is None:
            session_failures.append(
                {
                    "nwb_file_name": nwb,
                    "epoch": epoch,
                    "reason": "behavioral context or track graph unavailable",
                }
            )
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

    if session_failures:
        sample = session_failures[:10]
        raise RuntimeError(
            "canonical replay build is incomplete: "
            f"{len(session_failures)} of {len(sessions)} session(s) failed; "
            f"sample={sample}. No scientific tables were written."
        )
    return (
        all_event_tables,
        all_event_arm_tables,
        all_stop_tables,
        all_event_sequence_tables,
    )


def run_prepare_data(args: argparse.Namespace) -> None:
    """Resolve the scope, build every session's tables, then assemble, verify,
    standardize, and publish the canonical replay tables.

    Reads as its pipeline: resolve scope -> (early cache/stop-only exits) ->
    load contingency -> build per session -> assemble -> cross-table gate ->
    standardize -> publish.
    """
    output_dir = args.output_dir.resolve()
    # The cache build_session_tables reads trials and the track graph from.
    prepare_data.SESSION_CACHE_DIR = (
        args.behavior_cache_dir.resolve()
        if args.behavior_cache_dir is not None
        else output_dir / "_session_cache"
    )
    contingency_dir = (
        args.contingency_dir.resolve()
        if args.contingency_dir is not None
        else output_dir
    )

    max_per_animal = 2 if args.test else args.max_per_animal
    sessions = find_sessions(max_per_animal=max_per_animal)
    if args.animal is not None:
        sessions = [
            session
            for session in sessions
            if session["session_id"].startswith(args.animal)
        ]
    if not sessions:
        raise RuntimeError("no sessions matched the requested regeneration scope")
    logging.info(
        "Found %d sessions (trials/track_graph source: %s)",
        len(sessions),
        "cache, Spyglass fallback"
        if args.from_cache
        else "Spyglass (refreshing cache)",
    )

    if args.cache_only:
        if args.stop_only:
            raise ValueError("--cache-only and --stop-only are mutually exclusive")
        populate_behavioral_context_cache(sessions, from_cache=args.from_cache)
        logging.info(
            "behavioral cache population complete for %d sessions", len(sessions)
        )
        return

    if args.stop_only:
        build_stop_table_scope(
            sessions, from_cache=args.from_cache, output_dir=output_dir
        )
        return

    # Per-stop value artifact (per-arm nominal p(R) + per-arm model value),
    # keyed by stop_id, from qvals_table (exact stop-start alignment).
    # The event-arm builder maps nominal task probability / reward-belief mean
    # per port to the corresponding candidate-arm fields. The belief mean is
    # the fitted Beta reward belief, not the separate local leaky-Q history.
    # Absent qvals rows leave those fields NaN.
    contingency_path = table_path("qvals_table", contingency_dir)
    contingency = load_contingency(contingency_path)

    # Warn if the output directory still holds artifacts from the
    # legacy pipeline — silent coexistence lets downstream scripts
    # accidentally read stale pickles.
    for legacy_name in ("events.pkl", "long_count.pkl", "long_binary.pkl"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            logging.warning(
                "legacy output %s still present — delete it to avoid "
                "stale reads downstream",
                legacy_path,
            )

    t0 = time.time()
    collected = _build_all_session_tables(
        sessions, contingency, from_cache=args.from_cache, t0=t0
    )
    if not collected[0]:
        logging.error("no sessions produced data")
        return

    (
        event_table_all,
        event_arm_table_all,
        event_patch_table_all,
        stop_table_all,
        event_sequence_table_all,
    ) = assemble_canonical_tables(*collected)

    session_cols = list(SESSION_COLUMNS)
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
    # tests/analysis/event_tables/test_replay_tables.py; this is a cheap end-to-end
    # guard for the concat path.
    sum_n_events = int(stop_table_all["n_selected_events"].sum())
    if sum_n_events != len(event_table_all):
        logging.error(
            "cross-table drift: Σ stop_table.n_selected_events = %d, "
            "len(event_table) = %d — refusing to write output",
            sum_n_events,
            len(event_table_all),
        )
        return
    logging.info(
        "cross-table invariant OK: Σ n_selected_events == len(event_table) == %d",
        len(event_table_all),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    event_arm_table_all = standardize_event_arm(
        event_arm_table_all, contingency, output_dir / STANDARDIZATION_REFERENCE
    )
    publish_replay_tables(
        (
            ("event_table", "event", event_table_all),
            ("event_arm_table", "event_arm", event_arm_table_all),
            ("event_patch_table", "event_patch", event_patch_table_all),
            ("stop_table", "stop", stop_table_all),
            ("event_sequence_table", "event_sequence", event_sequence_table_all),
        ),
        output_dir,
    )

    logging.info("total time: %ds", int(time.time() - t0))


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
        default=replay_content_dir(),
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
