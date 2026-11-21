"""The per-session replay-table build and the corpus-level steps around it.

:func:`build_session_tables` builds one session's event, event-arm, stop and
event-sequence tables from its saved decode, its conditional posterior and its
behavioral context, through the per-grain builders in
``continuum_swr_replay.analysis.event_tables.event_table`` / ``stop_table`` /
``arm_patch_tables``. :func:`assemble_canonical_tables` joins the sessions'
tables and derives the assembly-level columns, and
:func:`standardize_event_arm` fits the pooled leaf-arm standardization over the
joined event-arm table. The table drivers in ``scripts/tables`` (the session
loops, the shard assembly and the publishing) call these.

Also the home of the behavioral-context cache (:func:`load_behavioral_context`,
:func:`session_cache_dir`), the per-session pickle of the Spyglass trials,
port experiences and track graph that the replay and ContFrag builds share,
and of :func:`well_visit_times`, an epoch's well visits read from it, with
the record a time-masked output keeps of them (:func:`session_cache_record`,
checked by :func:`session_cache_staleness` and :func:`time_mask_staleness`).
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates import assert_qvals_aligned
from continuum_swr_replay.analysis.behavior.covariates.history._common import (
    _prepare_port_experiences,
)
from continuum_swr_replay.analysis.decode_quality.posterior_analysis import (
    load_conditional_posterior,
)
from continuum_swr_replay.analysis.detection.nonlocal_events import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_dot_with_swr_source,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    subset_staleness,
)
from continuum_swr_replay.analysis.event_tables._common import (
    EVENT_ARM_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.event_tables.arm_patch_tables import (
    build_event_arm_table,
    build_event_patch_table,
)
from continuum_swr_replay.analysis.event_tables.event_table import (
    build_event_sequence_table,
    build_event_table,
)
from continuum_swr_replay.analysis.event_tables.retrospective_metrics import (
    compute_retrospective_metrics,
)
from continuum_swr_replay.analysis.event_tables.stop_table import (
    add_nonlocal_swr_split,
    build_stop_table,
)
from continuum_swr_replay.data_loaders.constants import WELL_ARMS, port_columns
from continuum_swr_replay.data_loaders.trials import (
    validate_port_experience_pump_audit,
)
from continuum_swr_replay.data_types import SessionDataKeys
from continuum_swr_replay.paths import replay_content_dir
from continuum_swr_replay.runs.io import files_sha256
from continuum_swr_replay.runs.staleness import recorded_sha256
from continuum_swr_replay.tables.io import write_atomically
from continuum_swr_replay.tables.session_keys import present_session_columns
from continuum_swr_replay.tables.standardization import (
    assert_standardized_present,
    attach_standardized,
    save_reference,
)
from continuum_swr_replay.tables.versions import (
    BEHAVIORAL_CONTEXT_CACHE_VERSION,  # value centralized in versions.py
    assert_versions_compatible,
)

SESSION_CACHE_DIR: Path | None = None
"""The per-session cache of the Spyglass-sourced behavioral context and track
graph, populated on every full run and read by ``--from-cache`` to skip
``load_data``. Set by the table drivers (``--behavior-cache-dir``, or the
output folder's ``_session_cache``); None means
``paths.replay_content_dir() / "_session_cache"``, resolved when read
(:func:`session_cache_dir`)."""


def session_cache_dir() -> Path:
    """:data:`SESSION_CACHE_DIR`, or ``<replay_content_dir()>/_session_cache`` while unset."""
    if SESSION_CACHE_DIR is None:
        return replay_content_dir() / "_session_cache"
    return SESSION_CACHE_DIR


PORT_EXPERIENCE_CACHE_COLUMNS = (
    "is_boundary_derived",
    "outcome_observed",
    "first_poke_in_time",
    "first_poke_out_time",
    "final_poke_out_time",
    "n_physical_poke_bouts",
)


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
        session_cache_dir()
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


def _read_cache_entry(cache_path: Path) -> dict:
    """The pickled behavioral context at ``cache_path``."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _cache_entry_problem(cached: dict) -> str | None:
    """Why a cache entry is off the current behavioral-context contract, or None.

    The entry's version stamp, its trials' outcome-clock columns, its port
    experiences' columns and their pump audit are checked, in that order;
    the first that fails is named.
    """
    cache_version_error = _behavioral_cache_version_error(
        cached.get("cache_schema_version")
    )
    if cache_version_error is not None:
        return f"is off the current behavioral-context contract ({cache_version_error})"
    stale_trials = [
        column
        for column in OUTCOME_CLOCK_TRIAL_COLUMNS
        if column not in cached["trials"].columns
    ]
    if stale_trials:
        return f"predates the outcome clock (missing {', '.join(stale_trials)})"
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
    if stale_experiences:
        return (
            "predates the physical-poke experience contract "
            f"(missing {', '.join(stale_experiences)})"
        )
    try:
        validate_port_experience_pump_audit(experiences)
    except (KeyError, ValueError) as exc:
        return f"has invalid post-recovery pump audit metadata ({exc})"
    return None


def load_behavioral_context(
    session: dict, *, from_cache: bool, cache_key: str | None = None
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, object | None, bool]:
    """Return trials, port experiences, graph, and cache status.

    These behavioral inputs are pulled from Spyglass (the ~3 h cost of a full
    run) and pickled to
    :func:`session_cache_dir` keyed by ``session_id`` so a covariate-only
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
        cached = _read_cache_entry(cache_path)
        problem = _cache_entry_problem(cached)
        if problem is None:
            return (
                cached["trials"],
                cached["port_experiences"],
                cached["track_graph"],
                True,
            )
        logging.warning(
            "session cache entry %s %s; ignoring it and re-fetching from Spyglass",
            cache_path.name,
            problem,
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
        # truncated pickle that the next read fails on.
        context = {
            "cache_schema_version": BEHAVIORAL_CONTEXT_CACHE_VERSION,
            "trials": trials,
            "port_experiences": port_experiences,
            "track_graph": track_graph,
        }

        def write_cache(path: Path) -> None:
            with path.open("wb") as f:
                pickle.dump(context, f)

        write_atomically(cache_path, write_cache)
    return trials, port_experiences, track_graph, False


def cached_port_experiences(epoch_id: str) -> pd.DataFrame:
    """An epoch's port experiences, read from the session cache only.

    Read from the behavioral-context cache entry the table builders wrote
    (:func:`behavioral_context_cache_path`) and checked as
    :func:`load_behavioral_context` checks an entry. Only the cache is read:
    a missing or off-contract entry is refused, never re-fetched from
    Spyglass, and nothing is written.

    Parameters
    ----------
    epoch_id : str
        The epoch identifier, such as ``"chimi20200212_02_r1_HPC"``: the
        cache's key.

    Returns
    -------
    pd.DataFrame
        The entry's port experiences, one row per well visit, as cached.

    Raises
    ------
    FileNotFoundError
        If the epoch has no cache entry.
    ValueError
        If its entry is off the current behavioral-context contract.
    """
    cache_path = behavioral_context_cache_path({"session_id": epoch_id})
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{epoch_id} has no session cache entry ({cache_path}); the table "
            "builders write it"
        )
    cached = _read_cache_entry(cache_path)
    problem = _cache_entry_problem(cached)
    if problem is not None:
        raise ValueError(f"session cache entry {cache_path} {problem}")
    return cached["port_experiences"]


def well_visit_times(epoch_id: str) -> tuple[np.ndarray, np.ndarray]:
    """An epoch's well visits: each one's first poke-in and final poke-out.

    A well visit is one port experience, from ``arrival_time`` (the first
    nose-poke in at a well) to ``departure_time`` (the final poke-out there),
    read from the session cache only (:func:`cached_port_experiences`).
    ``event_definition.well_visits.in_well_visit`` takes the two arrays.

    Parameters
    ----------
    epoch_id : str
        The epoch identifier, such as ``"chimi20200212_02_r1_HPC"``: the
        cache's key.

    Returns
    -------
    arrival_times, departure_times : np.ndarray, each shape (n_visits,)
        Seconds, in visit order (``arrival_times`` ascending).

    Raises
    ------
    FileNotFoundError
        If the epoch has no cache entry.
    ValueError
        If its entry is off the current behavioral-context contract.
    """
    experiences = _prepare_port_experiences(cached_port_experiences(epoch_id))
    return (
        experiences["arrival_time"].to_numpy(dtype=float),
        experiences["departure_time"].to_numpy(dtype=float),
    )


def session_cache_record(epoch_ids: Iterable[str]) -> dict[str, object]:
    """The record a time-masked output keeps of the well visits it read.

    An output whose content depends on the visits (quiet rest, immobile time
    or bouts inside a visit) records one SHA-256 over the session-cache
    entries of the epochs it read (``runs.io.files_sha256``), under
    ``session_cache`` in its manifest, so that a refreshed cache cannot pass
    as the one it was made from (:func:`session_cache_staleness`).

    Parameters
    ----------
    epoch_ids : iterable of str
        The epochs the output read the visits of.

    Returns
    -------
    dict
        ``path`` (the cache folder), ``n_files`` and ``sha256``.

    Raises
    ------
    FileNotFoundError
        If an epoch has no cache entry.
    """
    paths = [
        behavioral_context_cache_path({"session_id": identifier})
        for identifier in sorted(set(epoch_ids))
    ]
    return {
        "path": str(session_cache_dir() / BEHAVIORAL_CONTEXT_CACHE_VERSION),
        "n_files": len(paths),
        "sha256": files_sha256(paths),
    }


def session_cache_staleness(
    manifest: Mapping[str, Any], epoch_ids: Iterable[str], what: str
) -> str | None:
    """Why a time-masked output was not made from the current well visits, or None.

    Parameters
    ----------
    manifest : Mapping
        The output's manifest, with its ``session_cache`` record
        (:func:`session_cache_record`).
    epoch_ids : iterable of str
        The epochs the output holds; the current digest is taken over their
        cache entries.
    what : str
        The output, for the note.

    Raises
    ------
    FileNotFoundError
        If an epoch has no cache entry.
    """
    recorded = recorded_sha256(manifest, "session_cache")
    if recorded is None:
        return f"{what} records no session-cache SHA-256; rerun it before citing"
    current = session_cache_record(epoch_ids)["sha256"]
    if recorded != current:
        return (
            f"{what} was made from other well visits than the current session "
            f"cache's (sha256 {recorded[:12]}, current {current[:12]}); rerun it "
            "before citing"
        )
    return None


def time_mask_staleness(
    manifest: Mapping[str, Any], epoch_ids: Iterable[str], what: str
) -> str | None:
    """Why a time-masked output is not over the analyses' subset and the current
    well visits, or None: ``cohort_selection.subset_staleness`` and
    :func:`session_cache_staleness`, joined."""
    notes = [
        subset_staleness(manifest, what),
        session_cache_staleness(manifest, epoch_ids, what),
    ]
    return "; ".join(note for note in notes if note) or None


def build_session_tables(
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
    that records the session's immobility bouts. ``stop_table``
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
    # ``stop_table`` contract (the replay driver's docstring; the
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
        sess_keys = present_session_columns(event_table)
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
    port_cols = port_columns("reward_belief_mean")
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
            f"event_arm reward_belief_mean_candidate_arm_preoutcome disagrees "
            f"with qvals_table on "
            f"{int(mismatched.sum())} leaf row(s) across {bad_stops} stop(s). "
            "The value block is stale: rebuild qvals_table first, then "
            "rerun this script."
        )


def assemble_canonical_tables(
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
    # Always materialize ``event_sequence_table`` — even when no
    # session contributed any sequence rows. Later readers (e.g.
    # ``scripts/tables/write_regeneration_manifest.py``) read it
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
        from continuum_swr_replay.analysis.event_tables._common import (
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


def standardize_event_arm(
    event_arm_table_all: pd.DataFrame,
    contingency: pd.DataFrame | None,
    reference_path: Path,
) -> pd.DataFrame:
    """Attach the pooled-global ``_z`` columns and save the reference to `reference_path`.

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
    save_reference(z_reference, reference_path)
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
