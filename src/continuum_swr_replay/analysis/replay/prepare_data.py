"""Build the four canonical replay-content tables across all sessions.

Thin driver over the per-grain table builders in
``continuum_swr_replay.analysis.replay.tables_event`` / ``tables_stop`` /
``tables_arm_patch``. :func:`run_prepare_data` is the whole run; the
command-line front end that resolves its arguments is
``scripts/decoding/prepare_replay_content_data.py``, which is also where the
outputs, the flags, and the ``--from-cache`` contract are documented.

Also the home of the session-scope and behavioral-cache helpers
(:func:`find_sessions`, :func:`load_behavioral_context`,
:func:`populate_behavioral_context_cache`, :data:`SESSION_CACHE_DIR`) that the
other ``scripts/decoding`` drivers share.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.errors import MergeError

try:  # datajoint does not connect on import; guard so offline runs still work
    from datajoint.errors import DataJointError as _DataJointError
except ImportError:  # pragma: no cover - datajoint always present in practice

    class _DataJointError(Exception):
        """Placeholder so the skip clause is well-formed without datajoint."""


from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (
    read_dot_with_swr_source,
    stamp_swr_source,
    swr_epoch_id,
)
from continuum_swr_replay.analysis.replay._tables_common import (
    EVENT_ARM_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.replay.covariates import (
    assert_qvals_aligned,
    load_contingency,
)
from continuum_swr_replay.analysis.replay.posterior_io import (
    load_conditional_posterior,
)
from continuum_swr_replay.analysis.replay.retrospective import (
    _session_keys,
    compute_retrospective_metrics,
)
from continuum_swr_replay.analysis.replay.standardization import (
    assert_standardized_present,
    attach_standardized,
    save_reference,
)
from continuum_swr_replay.analysis.replay.tables_arm_patch import (
    build_event_arm_table,
    build_event_patch_table,
)
from continuum_swr_replay.analysis.replay.tables_event import (
    build_event_sequence_table,
    build_event_table,
)
from continuum_swr_replay.analysis.replay.tables_stop import (
    add_nonlocal_swr_split,
    build_stop_table,
)
from continuum_swr_replay.data_loaders.constants import WELL_ARMS
from continuum_swr_replay.data_loaders.epoch_register import analysed_epochs
from continuum_swr_replay.data_loaders.trials import (
    validate_port_experience_pump_audit,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.decode_files import decode_paths
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import (
    parquet_path,
    publish_canonical,
    umask_file_mode,
)
from continuum_swr_replay.types import SessionDataKeys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = PROCESSED_DATA_DIR / "replay_content"
# Per-session cache of the Spyglass-sourced behavioral context and track graph.
# Populated on every full run; consumed by --from-cache to skip load_data.
SESSION_CACHE_DIR = OUTPUT_DIR / "_session_cache"
from continuum_swr_replay.analysis.replay.versions import (  # noqa: E402
    BEHAVIORAL_CONTEXT_CACHE_VERSION,  # value centralized in versions.py
    assert_versions_compatible,
)

PORT_EXPERIENCE_CACHE_COLUMNS = (
    "is_boundary_derived",
    "outcome_observed",
    "first_poke_in_time",
    "first_poke_out_time",
    "final_poke_out_time",
    "n_physical_poke_bouts",
)


def find_sessions(max_per_animal: int | None = None) -> list[dict]:
    """The epoch register's analysed epochs, each with its decode files.

    The sessions are ``analysed_epochs()``, in ``session_id`` order within each
    animal, the animals in order; a decode on disk whose epoch the register does
    not analyse is not one. ``results.nc`` absence is recorded here as
    ``results_file=None`` and acted on in :func:`_process_session`, which
    refuses the session: without a posterior there is no
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
        under ``PROCESSED_DATA_DIR``: the tables would otherwise cover fewer
        epochs than the register analyses.
    """
    by_animal: dict[str, list] = {}
    for nwb_file_name, epoch in analysed_epochs().index:
        animal = parse_nwb_filename(nwb_file_name)[0]
        session_id = swr_epoch_id(nwb_file_name, epoch)
        paths = decode_paths(session_id, PROCESSED_DATA_DIR)
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
            f"are not under {PROCESSED_DATA_DIR}: {', '.join(missing[:10])}"
            + (", ..." if len(missing) > 10 else "")
        )
    return selected


# Written by ``load_data`` under the Comrie-aligned final-exit trial contract.
# Their absence dates a cache entry to an earlier behavioral definition. The
# versioned directory ensures those older inputs remain available for parity
# comparisons rather than being overwritten or reinterpreted.
OUTCOME_CLOCK_TRIAL_COLUMNS = (
    "reward_on_time",
    "outcome_known_time",
    "movement_start_time",
    "destination_first_poke_in_time",
    "destination_first_poke_out_time",
)


def behavioral_context_cache_path(
    session: dict, *, cache_key: str | None = None
) -> Path:
    """Return the versioned cache path without mutating an older cache bundle."""
    return (
        SESSION_CACHE_DIR
        / BEHAVIORAL_CONTEXT_CACHE_VERSION
        / f"{cache_key or session['session_id']}.pkl"
    )


def _behavioral_cache_version_error(cache_version: object) -> str | None:
    """Why a cache entry's stamp is unusable, or ``None`` when it is current.

    A mismatch is a cache *miss*, not an error: the entry is ignored and the
    session re-fetched. The comparison itself goes through the centralized
    version check so the cache and the covariate contract cannot drift apart.
    """
    if cache_version is None:
        return "entry carries no behavioral-context version stamp"
    try:
        assert_versions_compatible(behavioral_context_cache_version=str(cache_version))
    except ValueError as exc:
        return str(exc)
    return None


def load_behavioral_context(
    session: dict, *, from_cache: bool, cache_key: str | None = None
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, object | None, bool]:
    """Return trials, port experiences, graph, and cache status.

    These behavioral inputs are pulled from Spyglass (the ~3 h cost of a full
    run) and pickled to
    :data:`SESSION_CACHE_DIR` keyed by ``session_id`` so a covariate-only
    rebuild (``from_cache=True``) can skip ``load_data`` entirely.

    ``from_cache=True`` reads the cache when present and otherwise falls back
    to Spyglass (and refreshes the cache) so a session is never dropped for a
    missing entry. ``from_cache=False`` always queries Spyglass and rewrites
    the cache, keeping it fresh after every real regen. ``hit_cache`` reports
    whether the values came from the cache (vs Spyglass). The returned inputs
    are ``None`` when the session cannot be loaded.

    Cache paths and payloads are versioned by the behavioral derivation contract.
    A new contract writes to a new subdirectory, preserving older caches for
    explicit old/new comparison. Missing/mismatched versions or required columns
    are treated as cache misses and re-fetched from Spyglass; they are never
    silently interpreted under the current covariate definition.
    """
    # ``cache_key`` lets a sibling pipeline share this cache. trials and
    # track_graph are session facts, independent of which decoder produced the
    # posteriors, so the ContFrag prep reuses these entries under the NonLocal
    # session_id rather than paying for (or duplicating) its own Spyglass fetch.
    cache_path = behavioral_context_cache_path(session, cache_key=cache_key)
    if from_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        stale_trials = [
            column
            for column in OUTCOME_CLOCK_TRIAL_COLUMNS
            if column not in cached["trials"].columns
        ]
        cache_version = cached.get("cache_schema_version")
        cache_version_error = _behavioral_cache_version_error(cache_version)
        experiences = cached.get("port_experiences")
        stale_experiences = (
            list(PORT_EXPERIENCE_CACHE_COLUMNS)
            if experiences is None
            else [
                column
                for column in PORT_EXPERIENCE_CACHE_COLUMNS
                if column not in experiences.columns
            ]
        )
        pump_audit_error = None
        if experiences is not None:
            try:
                validate_port_experience_pump_audit(experiences)
            except (KeyError, ValueError) as exc:
                pump_audit_error = str(exc)
        if cache_version_error is not None:
            logging.warning(
                "session cache entry %s is off the current behavioral-context "
                "contract (%s); ignoring it and re-fetching from Spyglass",
                cache_path.name,
                cache_version_error,
            )
        elif stale_trials:
            logging.warning(
                "session cache entry %s predates the outcome clock (missing %s); "
                "ignoring it and re-fetching from Spyglass",
                cache_path.name,
                ", ".join(stale_trials),
            )
        elif stale_experiences:
            logging.warning(
                "session cache entry %s predates the physical-poke experience "
                "contract (missing %s); ignoring it and re-fetching from Spyglass",
                cache_path.name,
                ", ".join(stale_experiences),
            )
        elif pump_audit_error is not None:
            logging.warning(
                "session cache entry %s has invalid post-recovery pump audit "
                "metadata (%s); ignoring it and re-fetching from Spyglass",
                cache_path.name,
                pump_audit_error,
            )
        else:
            return (
                cached["trials"],
                cached["port_experiences"],
                cached["track_graph"],
                True,
            )

    # Imported lazily: `data_loaders` imports `spyglass.common` at module top
    # level, which opens a DataJoint connection. Keeping this out of module
    # scope lets --help and tests run offline, and --from-cache (with a warm
    # cache) fetch nothing per session. Every run still reads the database
    # once, when find_sessions checks the epoch register against
    # get_epoch_info.
    from continuum_swr_replay.data_loaders import load_data

    data = load_data(
        session["nwb_file_name"],
        session["epoch_name"],
        include=frozenset({SessionDataKeys.TRIALS, SessionDataKeys.PORT_EXPERIENCES}),
    )
    trials = data.get("trials")
    port_experiences = data.get("port_experiences")
    track_graph = data.get("track_graph")
    if trials is not None and port_experiences is not None and track_graph is not None:
        validate_port_experience_pump_audit(port_experiences)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: two sibling shards can target this shared cache dir, and
        # a crash or concurrent write to the same path would otherwise leave a
        # truncated pickle that the next read fails on. Write to a
        # same-directory temp file and os.replace into place.
        handle, tmp_name = tempfile.mkstemp(
            dir=cache_path.parent, prefix=f".{cache_path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(handle, "wb") as f:
                pickle.dump(
                    {
                        "cache_schema_version": BEHAVIORAL_CONTEXT_CACHE_VERSION,
                        "trials": trials,
                        "port_experiences": port_experiences,
                        "track_graph": track_graph,
                    },
                    f,
                )
            # mkstemp created tmp_path at 0600; restore the umask-respecting
            # mode a plain write would have produced so other readers of the
            # shared cache dir keep access (see table_io.umask_file_mode).
            os.chmod(tmp_path, umask_file_mode())
            os.replace(tmp_path, cache_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return trials, port_experiences, track_graph, False


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
    fits ``n_selected_events`` per stop over a denominator that must include
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

    # SWR labels come from the one SWR source, not the decode's stored ripples.
    dot = read_dot_with_swr_source(session["dot_file"])
    # Fail closed on input-contract violations (multi-session frame, unsorted
    # time, non-contiguous event labels) before they silently corrupt the tables.
    validate_dot_event_contract(dot)
    with open(session["env_file"], "rb") as f:
        env = pickle.load(f)

    # Behavioral context + track_graph come from Spyglass (slow; the ~3 h/full-run cost),
    # unless served from the per-session cache (--from-cache).
    trials, port_experiences, track_graph, hit_cache = load_behavioral_context(
        session, from_cache=from_cache
    )
    if trials is None or port_experiences is None or track_graph is None:
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
    # the downstream rate model, which models n_selected_events per stop
    # over a denominator that *must* include zero-event stops.
    stop_table = build_stop_table(dot, trials, port_experiences, track_graph)
    # The value block is joined on stop_id, a hash that carries no clock.
    # Check the session's stops against the qvals rows claiming them before
    # anything downstream joins on that hash.
    assert_qvals_aligned(stop_table, contingency)

    event_table = build_event_table(
        dot,
        env,
        trials,
        port_experiences,
        track_graph,
        conditional_posterior=posterior,
    )
    event_arm_table = build_event_arm_table(
        event_table,
        dot,
        env,
        trials,
        port_experiences,
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
        sess_keys = _session_keys(event_table)
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

    Candidate-arm reward-belief mean, task probability, and belief variance
    are baked into the event-arm table from ``qvals_table`` at build time, so
    the dependency runs qvals -> tables. Rebuilding the tables *before*
    refreshing qvals silently leaves the value block one fit behind, and the
    SWR-contrast runner then joins a fresh RPE onto a stale candidate-arm
    belief mean for the same stop. That
    happened once (281 rows across 36 stops, all five animals) and nothing
    caught it -- the per-table standardization checks compare each table only
    against its own reference.
    """
    if contingency is None or event_arm_table.empty:
        return
    port_cols = [f"reward_belief_mean_port_{i}" for i in range(6)]
    if not set(port_cols).issubset(contingency.columns):
        return
    if contingency["stop_id"].isna().any() or contingency["stop_id"].duplicated().any():
        raise ValueError(
            "qvals source must contain exactly one nonmissing row per stop_id "
            "before event-arm value validation"
        )
    leaf = event_arm_table.loc[
        event_arm_table["candidate_arm"] <= 5,
        ["stop_id", "candidate_arm", "reward_belief_mean_candidate_arm_preoutcome"],
    ]
    source = contingency.set_index("stop_id")[port_cols]
    joined = leaf.join(source, on="stop_id")
    port_values = joined[port_cols].to_numpy(dtype=float)
    comparable = np.isfinite(port_values).all(axis=1) & np.isfinite(
        joined["reward_belief_mean_candidate_arm_preoutcome"].to_numpy(dtype=float)
    )
    if not comparable.any():
        return
    expected = port_values[
        np.arange(len(joined)), joined["candidate_arm"].to_numpy(dtype=int)
    ]
    mismatched = comparable & ~np.isclose(
        expected,
        joined["reward_belief_mean_candidate_arm_preoutcome"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-9,
    )
    if mismatched.any():
        bad_stops = joined.loc[mismatched, "stop_id"].nunique()
        raise ValueError(
            f"event_arm q_arm disagrees with qvals_table on "
            f"{int(mismatched.sum())} leaf row(s) across {bad_stops} stop(s). "
            "The value block is stale: rebuild qvals_table.pkl first, then "
            "rerun this script."
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
    target = parquet_path(output_dir / "stop_table")
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
    path = publish_canonical(
        stop_table, output_dir, table="stop", basename="stop_table"
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
    """Run every session through ``_process_session`` and collect its tables.

    Data-gap failures (missing/corrupt input, unexpected shape) are skipped and
    recorded; a code bug (``KeyError`` / ``AttributeError`` / ``AssertionError``)
    propagates so a regression is not silently skipped across the run. If any
    session failed, raise before returning so a partial corpus is never
    assembled or written. Returns the per-session event / event-arm / stop /
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
            result = _process_session(
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
            # skips would hide the regression across the run.
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


def _assemble_canonical_tables(
    all_event_tables: list,
    all_event_arm_tables: list,
    all_stop_tables: list,
    all_event_sequence_tables: list,
) -> tuple:
    """Concatenate the per-session tables and derive the assembly-level columns.

    Adds the non-local SWR split to the stop table, always materializes a
    schema-complete event-sequence frame (empty if no session produced rows, so
    the retrospective pass never hits a missing file), and projects the
    deterministic event-patch table. Returns the five concatenated frames.
    """
    event_table_all = pd.concat(all_event_tables, ignore_index=True)
    event_arm_table_all = pd.concat(all_event_arm_tables, ignore_index=True)
    stop_table_all = pd.concat(all_stop_tables, ignore_index=True)
    # Add the canonical non-local SWR split
    # (n_nonlocal_swr_events / n_nonlocal_non_swr_events)
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
        from continuum_swr_replay.analysis.replay._tables_common import (
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
    return (
        event_table_all,
        event_arm_table_all,
        event_patch_table_all,
        stop_table_all,
        event_sequence_table_all,
    )


def _standardize_event_arm(
    event_arm_table_all: pd.DataFrame, contingency: pd.DataFrame | None
) -> pd.DataFrame:
    """Attach the pooled-global ``_z`` columns and persist the reference.

    The pooled standardization runs over the full concatenated table (a
    per-session z would use the wrong mean/SD) and only on the six reward-site
    leaves — structural trunk rows must not shrink the reference moments. The
    reference is saved for reproducibility, a completeness guard asserts the
    ``_z`` columns are present, and the value block is checked against the
    contingency source. Returns the standardized event-arm table.
    """
    # Pooled-GLOBAL standardization of the continuous per-arm covariates (<col>_z).
    # This is the persisted-table _z (folded in from the retired _add_standardized_cols
    # one-off); it MUST run here over the full concatenated table, not per session
    # (a per-session z would use the wrong mean/SD). The reference (means/SDs) is
    # persisted so the transform is reproducible and auditable. NB: this is NOT the
    # fit-time _z the is_swr-contrast runners compute over their event/stop cohort —
    # those are a different scope and are intentionally left untouched.
    # All four quantities are learned/defined on the six reward-site arms.
    # Structural trunk rows remain in the model table, but they must not shrink
    # the scientific unit by contributing zeros to the reference moments.
    leaf_support = event_arm_table_all["candidate_arm"].isin(WELL_ARMS)
    event_arm_table_all, z_reference = attach_standardized(
        event_arm_table_all,
        EVENT_ARM_STANDARDIZED_COLS,
        support_mask=leaf_support,
        support_label="leaf_arms",
    )
    save_reference(z_reference, OUTPUT_DIR / "standardization_reference.json")
    # Rebuild-completeness guard: a future refactor must not silently drop the _z
    # columns (the exact regression this phase fixes).
    assert_standardized_present(
        event_arm_table_all,
        EVENT_ARM_STANDARDIZED_COLS,
        support_mask=leaf_support,
    )
    _assert_value_block_matches_source(event_arm_table_all, contingency)
    logging.info(
        "standardized event_arm _z columns: %s",
        [f"{c}_z" for c in EVENT_ARM_STANDARDIZED_COLS],
    )
    return event_arm_table_all


def _publish_canonical_tables(frames: tuple) -> None:
    """Atomically publish each (basename, table_key, frame) via publish_canonical.

    ``publish_canonical`` stamps the covariate schema and writes atomically, so
    the tables cannot land unstamped; the per-table key selects the row-key
    contract. Every frame was built here from decode frames read through the
    SWR source, so each is stamped with it.
    """
    for basename, table_key, frame in frames:
        if frame is None:
            logging.info("  %s: skipped (no data)", basename)
            continue
        stamp_swr_source(frame)
        path = publish_canonical(frame, OUTPUT_DIR, table=table_key, basename=basename)
        logging.info("saved %s → %s (%d rows)", basename, path, len(frame))


def run_prepare_data(args: argparse.Namespace) -> None:
    """Resolve the scope, build every session's tables, then assemble, verify,
    standardize, and publish the canonical replay tables.

    Reads as its pipeline: resolve scope -> (early cache/stop-only exits) ->
    load contingency -> build per session -> assemble -> cross-table gate ->
    standardize -> publish.
    """
    global OUTPUT_DIR, SESSION_CACHE_DIR

    OUTPUT_DIR = args.output_dir.resolve()
    SESSION_CACHE_DIR = (
        args.behavior_cache_dir.resolve()
        if args.behavior_cache_dir is not None
        else OUTPUT_DIR / "_session_cache"
    )
    contingency_dir = (
        args.contingency_dir.resolve()
        if args.contingency_dir is not None
        else OUTPUT_DIR
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
            sessions, from_cache=args.from_cache, output_dir=OUTPUT_DIR
        )
        return

    # Per-stop value artifact (per-arm nominal p(R) + per-arm model value),
    # keyed by stop_id, from qvals_table (exact stop-start alignment).
    # The event-arm builder maps nominal task probability / reward-belief mean
    # per port to the corresponding candidate-arm fields. The belief mean is
    # the fitted Beta reward belief, not the separate local leaky-Q history.
    # Absent qvals rows leave those fields NaN.
    contingency_path = contingency_dir / "models_per_bin_rate" / "qvals_table.pkl"
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
    ) = _assemble_canonical_tables(*collected)

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
    # tests/unit/test_replay_tables.py; this is a cheap end-to-end
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_arm_table_all = _standardize_event_arm(event_arm_table_all, contingency)
    _publish_canonical_tables(
        (
            ("event_table", "event", event_table_all),
            ("event_arm_table", "event_arm", event_arm_table_all),
            ("event_patch_table", "event_patch", event_patch_table_all),
            ("stop_table", "stop", stop_table_all),
            ("event_sequence_table", "event_sequence", event_sequence_table_all),
        )
    )

    logging.info("total time: %ds", int(time.time() - t0))
