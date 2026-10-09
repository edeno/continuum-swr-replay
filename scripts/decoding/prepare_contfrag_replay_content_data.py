#!/usr/bin/env python
"""Build canonical replay-content tables from ContFrag decoding results.

Thin driver over :mod:`continuum_swr_replay.analysis.contfrag.tables`.
For each session it loads the pre-saved ``*_contfrag_data_over_time.pkl``
and ``*_contfrag_environment.pkl``, pulls ``trials`` + ``track_graph``
from Spyglass, and calls the ContFrag table builders.

Events are defined externally (SWR or MUA times), not by model-detected
probability thresholding.

Outputs (under ``PROCESSED_DATA_DIR/contfrag_replay_content/``), with
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

    uv run python scripts/decoding/prepare_contfrag_replay_content_data.py                    # SWR / HPC
    uv run python scripts/decoding/prepare_contfrag_replay_content_data.py --event-source MUA # MUA / HPC
    uv run python scripts/decoding/prepare_contfrag_replay_content_data.py --brain-area mPFC  # SWR / mPFC
    uv run python scripts/decoding/prepare_contfrag_replay_content_data.py --test             # 2 per animal
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.contfrag.posterior_analysis import (
    get_contfrag_posterior,
)
from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_arm_table,
    build_contfrag_event_table,
    build_contfrag_stop_table,
)
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (
    read_dot_with_swr_source,
    stamp_swr_source,
)
from continuum_swr_replay.analysis.replay import build_event_patch_table
from continuum_swr_replay.analysis.replay import prepare_data as replay_prepare_data
from continuum_swr_replay.analysis.replay.covariates import load_contingency
from continuum_swr_replay.data_loaders.epoch_register import analysed_epochs
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import publish_canonical

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)


def contfrag_table_basenames(event_source: str, brain_area: str) -> dict[str, str]:
    """Return the canonical ContFrag table basenames for one (source, area).

    The fitter resolves its inputs through the matching helper
    ``resolve_contfrag_paths`` in ``fit_replay_content_models.py``;
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


OUTPUT_DIR = PROCESSED_DATA_DIR / "contfrag_replay_content"


def find_sessions(
    max_per_animal: int | None = None,
    brain_area: str = "HPC",
) -> list[dict]:
    """The epoch register's analysed epochs, each with its ContFrag decode files.

    One brain area at a time: ``run_decoding_contfrag.py`` writes per-area files,
    and the saved tables would collide on event_id / stop_id across areas,
    because the canonical hashes do not include brain_area (see
    ``analysis.replay.event_ids.hash_event_id`` / ``hash_stop_id``).

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
        ``environment.pkl`` under ``PROCESSED_DATA_DIR``.
    """
    by_animal: dict[str, list] = {}
    for nwb_file_name, epoch in analysed_epochs().index:
        animal_date = nwb_file_name.removesuffix("_.nwb")
        # The NWB stem ends in an 8-digit date, so the animal id is everything
        # before it (``j16`` keeps its digits).
        animal = animal_date[:-8]
        session_id = f"{animal_date}_{epoch}_{brain_area}_contfrag"
        dot_file = PROCESSED_DATA_DIR / f"{session_id}_data_over_time.pkl"
        env_file = PROCESSED_DATA_DIR / f"{session_id}_environment.pkl"
        if brain_area != "HPC" and not (dot_file.exists() and env_file.exists()):
            continue
        results_path = PROCESSED_DATA_DIR / f"{session_id}_results.nc"
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
            f"analysed epochs are not under {PROCESSED_DATA_DIR}: "
            f"{', '.join(missing[:10])}" + (", ..." if len(missing) > 10 else "")
        )
    return selected


def _load_contfrag_posterior(results_file):
    """Load the combined ContFrag posterior from a results NetCDF."""
    import xarray as xr

    results = xr.open_dataset(results_file, engine="h5netcdf")
    state_bin_coord_names = list(results["state_bins"].coords)
    results = results.set_index(state_bins=state_bin_coord_names)
    return get_contfrag_posterior(results, state="combined")


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
    # tables; a violation here is caught by the per-session skip+tally in main().
    validate_dot_event_contract(dot)
    with open(session["env_file"], "rb") as f:
        env = pickle.load(f)

    # Reuse the NonLocal per-session trials/track_graph cache. Those are
    # session facts, not decoder outputs, so the ContFrag build has no reason
    # to re-fetch them -- and on a host where NWB reads fail (hdmf namespace
    # mismatch) the cache is the only way this pipeline runs at all. The cache
    # is keyed by the NonLocal session_id, which is this one without the
    # ``_contfrag`` suffix.
    cache_key = f"{nwb.replace('_.nwb', '')}_{epoch}_{brain_area}"
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
        default=OUTPUT_DIR,
        help=(
            "Candidate ContFrag root. Pass an explicit staging directory so "
            "existing scientific artifacts are not overwritten."
        ),
    )
    parser.add_argument(
        "--replay-content-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "replay_content",
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


def main():
    args = build_parser().parse_args()
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
    event_source = args.event_source
    logging.info(
        "Found %d %s sessions, event_source=%s",
        len(sessions),
        args.brain_area,
        event_source,
    )

    t0 = time.time()
    all_event_tables: list[pd.DataFrame] = []
    all_event_arm_tables: list[pd.DataFrame] = []
    all_stop_tables: list[pd.DataFrame] = []

    # Shared per-stop value artifact (qvals_table, exact poke_out alignment).
    # ContFrag's own stop_id mixes in brain_area/event_source, so the builder
    # recomputes the plain (animal,day,epoch,stop_idx) hash for the join.
    contingency_path = replay_content_dir / "models_per_bin_rate" / "qvals_table.pkl"
    contingency = load_contingency(contingency_path)

    skipped: list[tuple[str, str, str]] = []
    for i, session in enumerate(sessions):
        nwb = session["nwb_file_name"]
        epoch = session["epoch_name"]

        try:
            result = _process_session(
                session,
                event_source,
                contingency=contingency,
                from_cache=args.from_cache,
            )
        except (FileNotFoundError, OSError, ValueError) as e:
            logging.warning("skip %s %s: %s", nwb, epoch, e)
            skipped.append((nwb, epoch, str(e)))
            continue

        if result is None:
            # Behavioral context / track graph unavailable. This is a failure,
            # not a benign skip: count it toward the fail-closed gate below so a
            # dropped session cannot leave a complete-looking table set (mirrors
            # the Replay driver, analysis/replay/prepare_data.run_prepare_data).
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
        return

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
        return
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
    event_patch_table_all["brain_area"] = args.brain_area
    logging.info(
        "event_patch rows: %d (= 3 × %d events)",
        len(event_patch_table_all),
        event_table_all.groupby(session_cols)["event_id"].nunique().sum(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Per-brain-area filenames so HPC and mPFC prep runs don't
    # overwrite each other. The basename helper is exposed so the
    # fitter contract can be unit-tested against this exact format.
    basenames = contfrag_table_basenames(event_source, args.brain_area)
    for name, table_key, frame in (
        (basenames["event_table"], "event", event_table_all),
        (basenames["event_arm_table"], "event_arm", event_arm_table_all),
        (basenames["event_patch_table"], "event_patch", event_patch_table_all),
        (basenames["stop_table"], "stop", stop_table_all),
    ):
        # Built here from decode frames read through the SWR source.
        stamp_swr_source(frame)
        path = publish_canonical(frame, output_dir, table=table_key, basename=name)
        logging.info("saved %s -> %s (%d rows)", name, path, len(frame))

    logging.info("total time: %ds", int(time.time() - t0))


if __name__ == "__main__":
    main()
