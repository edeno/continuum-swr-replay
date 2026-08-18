#!/usr/bin/env python
"""Build the four canonical replay-content tables across all sessions.

Thin driver over :mod:`continuum_swr_replay.analysis.replay.tables`. For
each session it loads the pre-saved ``data_over_time.pkl``,
``environment.pkl``, and ``results.nc``, pulls ``trials`` +
``track_graph`` from Spyglass, and calls the four table builders.

Outputs (under ``PROCESSED_DATA_DIR/replay_content/``):

- ``event_table.pkl`` — one row per non-local event
- ``event_arm_table.pkl`` — one row per (event, arm)
- ``event_patch_table.pkl`` — one row per (event, patch); collapsed from
  ``event_arm_table`` via :func:`build_event_patch_table` so the
  ``--hierarchy patch`` path of ``fit_replay_content_models.py`` has its
  required input alongside the per-arm tables.
- ``event_sequence_table.pkl`` — one row per in-event timebin
- ``stop_table.pkl`` — one row per stopping period (incl. zero-event)

Usage (this driver moved from ``scripts/`` into the package; run it as a
module)::

    uv run python -m continuum_swr_replay.analysis.replay.prepare_data         # full run
    uv run python -m continuum_swr_replay.analysis.replay.prepare_data --test  # 2 per animal
    uv run python -m continuum_swr_replay.analysis.replay.prepare_data --from-cache  # covariate-only rebuild

**Skipping Spyglass for covariate-only changes** (``--from-cache``): the
only per-session Spyglass fetch is ``trials`` + ``track_graph`` (the ~3 h
cost of a full run). Every normal run pickles those to
``replay_content/_session_cache/<session_id>.pkl``. With ``--from-cache``
the driver reads them back instead of querying Spyglass and re-runs the
*same* table builders — so a change that only touches covariate code
(e.g. a new/re-encoded predictor, a re-standardization, a new
interaction) rebuilds the canonical tables in minutes with byte-for-byte
parity on everything else. Sessions missing from the cache transparently
fall back to Spyglass and are cached for next time, so no session is ever
dropped. Caveat: ``--from-cache`` freezes ``trials``/``track_graph`` at
the last full run; to pick up genuine upstream Spyglass changes
(re-curation, new intervals) run a normal pass, which refreshes the cache.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import time

import numpy as np
import pandas as pd
from pandas.errors import MergeError

try:  # datajoint does not connect on import; guard so offline runs still work
    from datajoint.errors import DataJointError as _DataJointError
except ImportError:  # pragma: no cover - datajoint always present in practice

    class _DataJointError(Exception):
        """Placeholder so the skip clause is well-formed without datajoint."""


from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.replay.covariates import load_contingency
from continuum_swr_replay.analysis.replay.posterior_io import (
    load_conditional_posterior,
)
from continuum_swr_replay.analysis.replay.retrospective import (
    compute_retrospective_metrics,
)
from continuum_swr_replay.analysis.replay.standardization import (
    assert_standardized_present,
    attach_standardized,
    save_reference,
)
from continuum_swr_replay.analysis.replay.tables import (
    EVENT_ARM_STANDARDIZED_COLS,
    add_nonlocal_swr_split,
    build_event_arm_table,
    build_event_patch_table,
    build_event_sequence_table,
    build_event_table,
    build_stop_table,
)
from continuum_swr_replay.data_loaders.constants import NWB_FILES
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import write_table
from continuum_swr_replay.types import SessionDataKeys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = PROCESSED_DATA_DIR / "replay_content"
# Per-session cache of the only Spyglass-sourced inputs (trials + track_graph).
# Populated on every full run; consumed by --from-cache to skip load_data.
SESSION_CACHE_DIR = OUTPUT_DIR / "_session_cache"


def find_sessions(max_per_animal: int | None = None):
    """Yield every session that has ``data_over_time.pkl`` and ``environment.pkl`` on disk.

    ``results.nc`` absence is recorded here as ``results_file=None`` and
    acted on in :func:`_process_session`, which refuses the session — without
    a posterior there is no ``event_sequence_table`` and no retrospective
    metrics.
    """
    dot_files = sorted(PROCESSED_DATA_DIR.glob("*_HPC_80_data_over_time.pkl"))
    env_files = {
        f.stem.replace("_environment", ""): f
        for f in PROCESSED_DATA_DIR.glob("*_HPC_environment.pkl")
    }

    by_animal: dict[str, list] = {}
    for f in dot_files:
        match = re.match(r"(\w+\d{8})_(\d+_r\d+)_HPC_80_data_over_time\.pkl", f.name)
        if not match:
            continue
        animal_date, epoch = match.group(1), match.group(2)
        # The outer regex enforces exactly 8 trailing digits (the date),
        # so the animal id is everything before them. The previous
        # ``re.match(r"([a-z]+)", animal_date)`` collapsed digit-bearing
        # ids — e.g. ``j162021...`` matched only ``j``, merging j16
        # sessions under ``j``.
        animal = animal_date[:-8]
        session_id = f"{animal_date}_{epoch}_HPC"
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
                "session_id": session_id,
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


# Written by ``load_data`` since the outcome clock landed. Their absence dates a
# cache entry to before it, and dates its reward-history covariates with it.
OUTCOME_CLOCK_TRIAL_COLUMNS = ("reward_on_time", "outcome_known_time")


def load_trials_track_graph(
    session: dict, *, from_cache: bool, cache_key: str | None = None
) -> tuple[pd.DataFrame | None, object | None, bool]:
    """Return ``(trials, track_graph, hit_cache)``, caching the Spyglass fetch.

    ``trials`` + ``track_graph`` are the only per-session inputs pulled from
    Spyglass (the ~3 h cost of a full run). They are pickled to
    :data:`SESSION_CACHE_DIR` keyed by ``session_id`` so a covariate-only
    rebuild (``from_cache=True``) can skip ``load_data`` entirely.

    ``from_cache=True`` reads the cache when present and otherwise falls back
    to Spyglass (and refreshes the cache) so a session is never dropped for a
    missing entry. ``from_cache=False`` always queries Spyglass and rewrites
    the cache, keeping it fresh after every real regen. ``hit_cache`` reports
    whether the values came from the cache (vs Spyglass). ``trials`` /
    ``track_graph`` are ``None`` when the session cannot be loaded.

    **Staleness is presence-only, with one exception.** The cache is otherwise
    trusted whenever the pickle exists — there is no schema/version tag. So
    ``--from-cache`` freezes not just Spyglass-side data (re-curation, new
    intervals) but also ``load_data``'s own trial-derivation *code*: if that
    logic changes, cached trials keep the old derivation silently. Run a full
    (non-cache) pass to refresh after any such change; ``--from-cache`` is for
    covariate-only work.

    The exception is the outcome clock. An entry written before
    ``reward_on_time`` / ``outcome_known_time`` existed makes every reward-history
    covariate silently resolve at poke-out instead of pump onset, and that is
    not visible in any output. Such an entry is treated as a cache miss, so the
    session is re-fetched from Spyglass and fails loudly there if it cannot be
    read -- which is the honest outcome, since a session whose NWB file will not
    open cannot be rebuilt correctly from a stale cache either.
    """
    # ``cache_key`` lets a sibling pipeline share this cache. trials and
    # track_graph are session facts, independent of which decoder produced the
    # posteriors, so the ContFrag prep reuses these entries under the NonLocal
    # session_id rather than paying for (or duplicating) its own Spyglass fetch.
    cache_path = SESSION_CACHE_DIR / f"{cache_key or session['session_id']}.pkl"
    if from_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        stale = [
            column
            for column in OUTCOME_CLOCK_TRIAL_COLUMNS
            if column not in cached["trials"].columns
        ]
        if stale:
            logging.warning(
                "session cache entry %s predates the outcome clock (missing %s); "
                "ignoring it and re-fetching from Spyglass",
                cache_path.name,
                ", ".join(stale),
            )
        else:
            return cached["trials"], cached["track_graph"], True

    # Imported lazily: `data_loaders` imports `spyglass.common` at module top
    # level, which opens a DataJoint connection. Keeping this out of module
    # scope lets --from-cache (with a warm cache), --help, and tests run offline.
    from continuum_swr_replay.data_loaders import load_data

    data = load_data(
        session["nwb_file_name"],
        session["epoch_name"],
        include=frozenset({SessionDataKeys.TRIALS}),
    )
    trials = data.get("trials")
    track_graph = data.get("track_graph")
    if trials is not None and track_graph is not None:
        SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"trials": trials, "track_graph": track_graph}, f)
    return trials, track_graph, False


def _process_session(
    session: dict,
    contingency: pd.DataFrame | None = None,
    *,
    from_cache: bool = False,
) -> dict[str, pd.DataFrame] | None:
    """Run all four builders on one session; return the four frames or ``None``.

    Returns ``None`` only when the session cannot be loaded (missing
    trials or track graph). Sessions whose decoder produced no
    non-local events still return a dict — with empty
    ``event_table`` / ``event_arm_table`` and a ``None``
    ``event_sequence_table``, but a fully populated ``stop_table``
    that records the session's immobility bouts. ``stop_table.pkl``
    is documented to include zero-event bouts, and the rate model
    fits ``n_events`` per stop over a denominator that must include
    those rows. Callers concatenate every non-``None`` return into
    the global canonical tables; ``pd.concat`` accepts the empty
    per-event frames without complaint.
    """
    nwb = session["nwb_file_name"]
    epoch = session["epoch_name"]

    # Gate before any loading: a ~1M-row dot pickle is not worth reading for a
    # session we are about to refuse.
    if session["results_file"] is None:
        raise ValueError(
            f"no results.nc for {nwb} {epoch}: refusing to build a session "
            "without a conditional posterior (no event_sequence_table, no "
            "retrospective metrics, all posterior_* columns NaN)."
        )

    dot = pd.read_pickle(session["dot_file"])
    # Fail closed on input-contract violations (multi-session frame, unsorted
    # time, non-contiguous event labels) before they silently corrupt the tables.
    validate_dot_event_contract(dot)
    with open(session["env_file"], "rb") as f:
        env = pickle.load(f)

    # Trials + track_graph come from Spyglass (slow; the ~3 h/full-run cost),
    # unless served from the per-session cache (--from-cache).
    trials, track_graph, hit_cache = load_trials_track_graph(
        session, from_cache=from_cache
    )
    if trials is None or track_graph is None:
        return None
    logging.info(
        "  %s %s: trials/track_graph from %s",
        nwb,
        epoch,
        "cache" if hit_cache else "Spyglass",
    )

    # Conditional posterior is optional — sessions without a saved
    # ``results.nc`` still emit the three tables whose posterior
    # columns degrade to NaN.
    # results_file is guaranteed non-None by the gate at the top of this
    # function, so there is no missing-posterior branch left here.
    try:
        posterior = load_conditional_posterior(session["results_file"])
    except Exception as e:  # noqa: BLE001 - surface the reason
        # A results.nc that exists but will not load leaves us in exactly the
        # state the gate refuses. A truncated or corrupt file is likelier than
        # a missing one, since find_sessions only records None when the path
        # is absent.
        raise ValueError(
            f"results.nc for {nwb} {epoch} exists but could not be loaded "
            f"({e}); refusing to build the session without a conditional "
            "posterior."
        ) from e

    # Build the stop table first — it's a function of immobility
    # bouts, not non-local events, and the documented
    # ``stop_table.pkl`` contract (this script's own docstring; the
    # builder's "**including zero-event bouts**" promise) keeps
    # zero-event sessions in the dataset. Returning early on an
    # empty event_table dropped those sessions silently and broke
    # the downstream rate model, which models n_events per stop
    # over a denominator that *must* include zero-event stops.
    stop_table = build_stop_table(dot, trials, track_graph)

    event_table = build_event_table(
        dot, env, trials, track_graph, conditional_posterior=posterior
    )
    event_arm_table = build_event_arm_table(
        event_table,
        dot,
        env,
        trials,
        track_graph,
        conditional_posterior=posterior,
        contingency=contingency,
    )
    event_sequence_table = (
        build_event_sequence_table(dot, env, posterior)
        if posterior is not None and len(event_table) > 0
        else None
    )

    # Merge the prospective + retrospective alignment metrics into the
    # canonical event table on event_id (left-join keeps every event; those
    # without a valid next_arm/prev_arm get NaN). These need the arm and
    # sequence tables, so they are joined here at the assembly level rather
    # than inside build_event_table. This replaces the previously
    # standalone post-hoc pass for the per-event columns.
    if event_sequence_table is not None and len(event_table) > 0:
        sess_keys = [
            c for c in ("animal", "day", "epoch") if c in event_table.columns
        ]
        retro = compute_retrospective_metrics(
            event_table, event_arm_table, event_sequence_table
        )
        if not retro.empty:
            event_table = event_table.merge(
                retro, on=[*sess_keys, "event_id"], how="left", validate="one_to_one"
            )

    return {
        "event_table": event_table,
        "event_arm_table": event_arm_table,
        "stop_table": stop_table,
        "event_sequence_table": event_sequence_table,
    }


def _assert_value_block_matches_source(
    event_arm_table: pd.DataFrame,
    contingency: pd.DataFrame | None,
) -> None:
    """Fail if the built value block disagrees with the qvals table it came from.

    ``q_arm`` / ``p_reward_arm`` / ``arm_uncertainty`` are baked into the
    event-arm table from ``qvals_table.pkl`` at build time, so the dependency
    runs qvals -> tables. Rebuilding the tables *before* refreshing qvals
    silently leaves the value block one fit behind, and the SWR-contrast runner
    then joins a fresh ``rpe`` onto a stale ``q_arm`` for the same stop. That
    happened once (281 rows across 36 stops, all five animals) and nothing
    caught it -- the per-table standardization checks compare each table only
    against its own reference.
    """
    if contingency is None or event_arm_table.empty:
        return
    port_cols = [f"q_port{i}" for i in range(6)]
    if not set(port_cols).issubset(contingency.columns):
        return
    leaf = event_arm_table.loc[
        event_arm_table["arm"] <= 5, ["stop_id", "arm", "q_arm"]
    ]
    source = contingency.drop_duplicates("stop_id").set_index("stop_id")[port_cols]
    joined = leaf.join(source, on="stop_id")
    port_values = joined[port_cols].to_numpy(dtype=float)
    comparable = np.isfinite(port_values).all(axis=1) & np.isfinite(
        joined["q_arm"].to_numpy(dtype=float)
    )
    if not comparable.any():
        return
    expected = port_values[
        np.arange(len(joined)), joined["arm"].to_numpy(dtype=int)
    ]
    mismatched = comparable & ~np.isclose(
        expected, joined["q_arm"].to_numpy(dtype=float), rtol=0, atol=1e-9
    )
    if mismatched.any():
        bad_stops = joined.loc[mismatched, "stop_id"].nunique()
        raise ValueError(
            f"event_arm q_arm disagrees with qvals_table on "
            f"{int(mismatched.sum())} leaf row(s) across {bad_stops} stop(s). "
            "The value block is stale: rebuild qvals_table.pkl first, then "
            "rerun this script."
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="2 sessions per animal")
    parser.add_argument("--max-per-animal", type=int, default=None)
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
    args = parser.parse_args()

    max_per_animal = 2 if args.test else args.max_per_animal
    sessions = find_sessions(max_per_animal=max_per_animal)
    logging.info(
        "Found %d sessions (trials/track_graph source: %s)",
        len(sessions),
        "cache, Spyglass fallback" if args.from_cache else "Spyglass (refreshing cache)",
    )

    # Per-stop value artifact (per-arm nominal p(R) + per-arm model value),
    # keyed by stop_id, from qvals_table (exact poke_out alignment -- no difflib).
    # The event-arm builder maps p_reward_port/q_port -> the per-arm
    # p_reward_arm / q_arm; q_port is now the beta-model belief per port (was
    # leaky-Q). Absent -> those columns stay NaN.
    contingency_path = OUTPUT_DIR / "models_per_bin_rate" / "qvals_table.pkl"
    contingency = load_contingency(contingency_path)

    # Warn if the output directory still holds artifacts from the
    # legacy pipeline — silent coexistence lets downstream scripts
    # accidentally read stale pickles.
    for legacy_name in ("events.pkl", "long_count.pkl", "long_binary.pkl"):
        legacy_path = OUTPUT_DIR / legacy_name
        if legacy_path.exists():
            logging.warning(
                "legacy output %s still present — delete it to avoid "
                "stale reads downstream",
                legacy_path,
            )

    t0 = time.time()
    all_event_tables: list[pd.DataFrame] = []
    all_event_arm_tables: list[pd.DataFrame] = []
    all_stop_tables: list[pd.DataFrame] = []
    all_event_sequence_tables: list[pd.DataFrame] = []

    for i, session in enumerate(sessions):
        nwb = session["nwb_file_name"]
        epoch = session["epoch_name"]

        try:
            result = _process_session(
                session, contingency=contingency, from_cache=args.from_cache
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
            # skips would hide the regression across the run.
            logging.warning("skip %s %s: %s", nwb, epoch, e)
            continue

        if result is None:
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

    if not all_event_tables:
        logging.error("no sessions produced data")
        return

    event_table_all = pd.concat(all_event_tables, ignore_index=True)
    event_arm_table_all = pd.concat(all_event_arm_tables, ignore_index=True)
    stop_table_all = pd.concat(all_stop_tables, ignore_index=True)
    # Add the canonical non-local SWR split (n_non_local_SWR / n_non_local_no_SWR)
    # from event_table.is_swr — an assembly-level join, like the retrospective
    # merge below, kept out of build_stop_table (which is event-table-independent).
    stop_table_all = add_nonlocal_swr_split(stop_table_all, event_table_all)
    # Always materialize ``event_sequence_table.pkl`` — even when no
    # session contributed any sequence rows. The retrospective pass (the
    # retrospective-alignment step) does
    # ``read_table(INPUT_DIR / "event_sequence_table.pkl")``
    # unconditionally, so a missing file would crash a posterior-
    # unavailable run with FileNotFoundError. The retrospective
    # helpers already treat an empty sequence frame as a legitimate
    # input, so writing an empty schema-complete frame is the right
    # contract here.
    if all_event_sequence_tables:
        event_sequence_table_all = pd.concat(
            all_event_sequence_tables, ignore_index=True
        )
    else:
        from continuum_swr_replay.analysis.replay.tables import (
            _EVENT_SEQUENCE_TABLE_COLUMNS,
        )

        event_sequence_table_all = pd.DataFrame(
            columns=list(_EVENT_SEQUENCE_TABLE_COLUMNS)
        )
    # Patch table is a deterministic projection of the per-arm table
    # (sums arm_count_hard within each patch and aggregates predictors
    # to patch level). Save it alongside the others so the documented
    # ``fit_replay_content_models.py --hierarchy patch`` path has its
    # required input.
    event_patch_table_all = build_event_patch_table(
        event_table_all, event_arm_table_all
    )

    session_cols = ["animal", "day", "epoch"]
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
    # tests/unit/test_replay_tables.py; this is a cheap end-to-end
    # guard for the concat path.
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pooled-GLOBAL standardization of the continuous per-arm covariates (<col>_z).
    # This is the persisted-table _z (folded in from the retired _add_standardized_cols
    # one-off); it MUST run here over the full concatenated table, not per session
    # (a per-session z would use the wrong mean/SD). The reference (means/SDs) is
    # persisted so the transform is reproducible and auditable. NB: this is NOT the
    # fit-time _z the is_swr-contrast runners compute over their event/stop cohort —
    # those are a different scope and are intentionally left untouched.
    event_arm_table_all, z_reference = attach_standardized(
        event_arm_table_all, EVENT_ARM_STANDARDIZED_COLS
    )
    save_reference(z_reference, OUTPUT_DIR / "standardization_reference.json")
    # Rebuild-completeness guard: a future refactor must not silently drop the _z
    # columns (the exact regression this phase fixes).
    assert_standardized_present(event_arm_table_all, EVENT_ARM_STANDARDIZED_COLS)
    _assert_value_block_matches_source(event_arm_table_all, contingency)
    logging.info(
        "standardized event_arm _z columns: %s",
        [f"{c}_z" for c in EVENT_ARM_STANDARDIZED_COLS],
    )

    for name, frame in (
        ("event_table", event_table_all),
        ("event_arm_table", event_arm_table_all),
        ("event_patch_table", event_patch_table_all),
        ("stop_table", stop_table_all),
        ("event_sequence_table", event_sequence_table_all),
    ):
        if frame is None:
            logging.info("  %s: skipped (no data)", name)
            continue
        path = write_table(frame, OUTPUT_DIR / name)
        logging.info("saved %s → %s (%d rows)", name, path, len(frame))

    logging.info("total time: %ds", int(time.time() - t0))


if __name__ == "__main__":
    main()
