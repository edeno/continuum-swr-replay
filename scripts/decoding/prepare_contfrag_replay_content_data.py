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
import re
import time

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
from continuum_swr_replay.analysis.replay import build_event_patch_table
from continuum_swr_replay.analysis.replay.covariates import load_contingency
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.constants import NWB_FILES
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.types import SessionDataKeys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)


def contfrag_table_basenames(
    event_source: str, brain_area: str
) -> dict[str, str]:
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
    brain_area: str | None = None,
):
    """Yield sessions that have ContFrag ``data_over_time.pkl`` and ``environment.pkl``.

    When ``brain_area`` is provided, only sessions for that area are
    returned. Without filtering, the prep script would silently merge
    HPC + mPFC + OFC sessions into a single combined output even
    though ``run_decoding_contfrag.py`` writes per-area files; the
    saved tables would also collide on event_id / stop_id because
    the canonical hashes do not include brain_area
    (see ``analysis.replay._helpers.hash_event_id`` /
    ``hash_stop_id``).
    """
    dot_files = sorted(PROCESSED_DATA_DIR.glob("*_contfrag_data_over_time.pkl"))
    env_files = {
        f.stem.replace("_environment", ""): f
        for f in PROCESSED_DATA_DIR.glob("*_contfrag_environment.pkl")
    }

    by_animal: dict[str, list] = {}
    for f in dot_files:
        match = re.match(
            r"(\w+\d{8})_(\d+_r\d+)_(\w+)_contfrag_data_over_time\.pkl", f.name
        )
        if not match:
            continue
        animal_date, epoch, file_brain_area = (
            match.group(1),
            match.group(2),
            match.group(3),
        )
        if brain_area is not None and file_brain_area != brain_area:
            continue
        # animal_date is captured as ``\w+\d{8}`` so the animal id is
        # everything before the trailing 8-digit date. ``re.match(r"[a-z]+", ...)``
        # would collapse digit-bearing ids (``j162021...`` → ``j``).
        animal = animal_date[:-8]
        session_id = f"{animal_date}_{epoch}_{file_brain_area}_contfrag"
        if session_id not in env_files:
            continue

        nwb_match = [nwb for nwb in NWB_FILES if animal_date in nwb]
        if not nwb_match:
            continue

        results_path = PROCESSED_DATA_DIR / f"{session_id}_results.nc"

        by_animal.setdefault(animal, []).append(
            {
                "nwb_file_name": nwb_match[0],
                "epoch_name": epoch,
                "brain_area": file_brain_area,
                "dot_file": f,
                "env_file": env_files[session_id],
                "results_file": results_path if results_path.exists() else None,
            }
        )

    selected = []
    for animal in sorted(by_animal):
        sessions = by_animal[animal]
        if max_per_animal is not None:
            sessions = sessions[:max_per_animal]
        selected.extend(sessions)

    return selected


def _load_contfrag_posterior(results_file):
    """Load the combined ContFrag posterior from a results NetCDF."""
    import xarray as xr

    results = xr.open_dataset(results_file, engine="h5netcdf")
    state_bin_coord_names = list(results["state_bins"].coords)
    results = results.set_index(state_bins=state_bin_coord_names)
    return get_contfrag_posterior(results, state="combined")


def _process_session(
    session: dict, event_source: str, contingency: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame] | None:
    """Run ContFrag table builders on one session."""
    nwb = session["nwb_file_name"]
    epoch = session["epoch_name"]
    brain_area = session["brain_area"]

    dot = pd.read_pickle(session["dot_file"])
    # Fail closed on input-contract violations before they silently corrupt the
    # tables; a violation here is caught by the per-session skip+tally in main().
    validate_dot_event_contract(dot)
    with open(session["env_file"], "rb") as f:
        env = pickle.load(f)

    data = load_data(nwb, epoch, include=frozenset({SessionDataKeys.TRIALS}))
    trials = data.get("trials")
    track_graph = data.get("track_graph")
    if trials is None or track_graph is None:
        return None

    posterior = None
    if session["results_file"] is not None:
        try:
            posterior = _load_contfrag_posterior(session["results_file"])
        except Exception as e:  # noqa: BLE001
            logging.warning("posterior load failed for %s %s: %s", nwb, epoch, e)
    else:
        # Same gate as the replay driver. ContFrag builds no sequence table, so
        # it does not lose the retrospective metrics, but the session would
        # still ship with every posterior_* column, arm_posterior_mass and
        # arm_fraction_posterior NaN while every other session has them -- a
        # column-inhomogeneous corpus that no downstream check would notice. A
        # missing results.nc usually means the decode failed or never ran.
        # 0 of 218 sessions hit this today.
        raise ValueError(
            f"no results.nc for {nwb} {epoch}: refusing to build a ContFrag "
            "session without a conditional posterior (posterior_*, "
            "arm_posterior_mass and arm_fraction_posterior would all be NaN)."
        )

    # Build the stop table first so zero-event sessions still
    # contribute their immobility bouts. Returning early on an empty
    # event_table dropped those sessions, contradicting the
    # ``stop_table_{source}_{brain_area}.pkl`` contract that
    # documents zero-event bouts as part of the canonical output.
    stop_table = build_contfrag_stop_table(
        dot,
        trials,
        track_graph,
        event_source=event_source,
        brain_area=brain_area,
    )

    event_table = build_contfrag_event_table(
        dot,
        env,
        trials,
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="2 sessions per animal")
    parser.add_argument("--max-per-animal", type=int, default=None)
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
    args = parser.parse_args()

    max_per_animal = 2 if args.test else args.max_per_animal
    sessions = find_sessions(
        max_per_animal=max_per_animal, brain_area=args.brain_area
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
    contingency_path = (
        PROCESSED_DATA_DIR
        / "replay_content"
        / "models_per_bin_rate"
        / "qvals_table.pkl"
    )
    contingency = load_contingency(contingency_path)

    skipped: list[tuple[str, str, str]] = []
    for i, session in enumerate(sessions):
        nwb = session["nwb_file_name"]
        epoch = session["epoch_name"]

        try:
            result = _process_session(session, event_source, contingency=contingency)
        except (FileNotFoundError, OSError, ValueError) as e:
            logging.warning("skip %s %s: %s", nwb, epoch, e)
            skipped.append((nwb, epoch, str(e)))
            continue

        if result is None:
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

    if not all_event_tables:
        logging.error("no sessions produced data")
        return

    event_table_all = pd.concat(all_event_tables, ignore_index=True)
    event_arm_table_all = pd.concat(all_event_arm_tables, ignore_index=True)
    stop_table_all = pd.concat(all_stop_tables, ignore_index=True)

    session_cols = ["animal", "day", "epoch"]
    logging.info(
        "events: %d, event_arm rows: %d, stops: %d, sessions: %d",
        len(event_table_all),
        len(event_arm_table_all),
        len(stop_table_all),
        event_table_all.groupby(session_cols).ngroups,
    )

    # Cross-table invariant: Σ stop_table.n_events == len(event_table)
    sum_n_events = int(stop_table_all["n_events"].sum())
    if sum_n_events != len(event_table_all):
        logging.error(
            "cross-table drift: Σ stop_table.n_events = %d, "
            "len(event_table) = %d — refusing to write output",
            sum_n_events,
            len(event_table_all),
        )
        return
    logging.info(
        "cross-table invariant OK: Σ n_events == len(event_table) == %d",
        len(event_table_all),
    )

    # Collapse to the event-patch table for the patch-level hierarchy.
    # Built from the concatenated frames so event_id is globally unique
    # (the ContFrag hash already folds in brain_area + event_source),
    # which build_event_patch_table requires. Provenance is re-stamped
    # because the helper only carries the animal/day/epoch session keys.
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Per-brain-area filenames so HPC and mPFC prep runs don't
    # overwrite each other. The basename helper is exposed so the
    # fitter contract can be unit-tested against this exact format.
    basenames = contfrag_table_basenames(event_source, args.brain_area)
    for name, frame in (
        (basenames["event_table"], event_table_all),
        (basenames["event_arm_table"], event_arm_table_all),
        (basenames["event_patch_table"], event_patch_table_all),
        (basenames["stop_table"], stop_table_all),
    ):
        path = OUTPUT_DIR / f"{name}.pkl"
        frame.to_pickle(path)
        logging.info("saved %s -> %s (%d rows)", name, path, len(frame))

    logging.info("total time: %ds", int(time.time() - t0))


if __name__ == "__main__":
    main()
