"""The per-session ContFrag table build.

:func:`build_contfrag_session_tables` builds one session's event, event-arm and
stop tables with the builders in
:mod:`continuum_swr_replay.analysis.contfrag.tables`, from the saved ContFrag
decode (``*_contfrag_data_over_time.pkl``, ``*_contfrag_environment.pkl`` and
the combined posterior of ``*_contfrag_results.nc``) and the session's trials
and track graph from the behavioral cache that the replay build shares
(``analysis.event_tables.prepare_data.load_behavioral_context``).
:func:`rebuild_contfrag_session_tables` builds the same tables without the
posterior, carrying its aggregates over from a preserved bundle, and
:func:`contfrag_event_patch_table` projects the event-patch table from the
joined event-arm table. The table drivers in ``scripts/tables`` (the session
loops, the shard assembly, the covariate refresh and the publishing) call these,
under the basenames of :func:`contfrag_table_basenames`.

Events are defined externally (SWR or MUA times), not by model-detected
probability thresholding.
"""

from __future__ import annotations

import pickle
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
from continuum_swr_replay.analysis.decode_quality.data_aggregation import (
    validate_dot_event_contract,
)
from continuum_swr_replay.analysis.event_tables import (
    prepare_data as replay_prepare_data,
)
from continuum_swr_replay.analysis.event_tables.arm_patch_tables import (
    build_event_patch_table,
)
from continuum_swr_replay.analysis.event_tables.summaries import (
    arm_fraction_from_event_total,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    read_dot_with_swr_source,
)
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.tables.columns import require_columns


def contfrag_table_basenames(event_source: str, brain_area: str) -> dict[str, str]:
    """Return the canonical ContFrag table basenames for one (source, area).

    The fitter resolves its inputs through the matching helper
    ``resolve_contfrag_paths`` in ``analysis/replay_models/content/fit_outputs.py``;
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


def _load_contfrag_posterior(results_file):
    """Load the combined ContFrag posterior from a results NetCDF."""
    return get_contfrag_posterior(open_results(results_file), state="combined")


def build_contfrag_session_tables(
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
    # tables; a violation here is caught by the per-session skip+tally of
    # build_contfrag_tables.
    validate_dot_event_contract(dot)
    with open(session["env_file"], "rb") as f:
        env = pickle.load(f)

    # Reuse the NonLocal per-session trials/track_graph cache. Those are
    # session facts, not decoder outputs, so the ContFrag build has no reason
    # to re-fetch them -- and on a host where NWB reads fail (hdmf namespace
    # mismatch) the cache is the only way this pipeline runs at all. The cache
    # is keyed by the NonLocal session_id, which is this one without the
    # ``_contfrag`` suffix.
    cache_key = format_epoch_identifier(nwb, epoch, brain_area)
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


def contfrag_event_patch_table(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    *,
    event_source: str,
    brain_area: str,
) -> pd.DataFrame:
    """The event-patch table of joined ContFrag event and event-arm tables.

    ``build_event_patch_table`` over the whole tables, so ``event_id`` is unique
    across sessions as that builder requires (the ContFrag hash folds in
    ``brain_area`` and ``event_source``), with the provenance columns stamped
    again because the projection carries only the canonical session keys.

    Parameters
    ----------
    event_table, event_arm_table : pd.DataFrame
        The ContFrag tables of every session, concatenated.
    event_source : str
        ``"SWR"`` or ``"MUA"``.
    brain_area : str
        ``"HPC"``, ``"mPFC"`` or ``"OFC"``.

    Returns
    -------
    pd.DataFrame
        One row per (event, patch).
    """
    event_patch_table = build_event_patch_table(event_table, event_arm_table)
    event_patch_table["event_source"] = event_source
    event_patch_table["brain_area"] = brain_area
    return event_patch_table


EVENT_POSTERIOR_COLUMNS = (
    "posterior_peak_mean",
    "posterior_entropy_mean",
    "posterior_hpd_arm_count_mean_coverage_50pct",
    "posterior_hpd_arm_count_mean_coverage_80pct",
    "posterior_hpd_arm_count_mean_coverage_95pct",
    "multimodal_fraction",
)


def _copy_by_key(
    target: pd.DataFrame,
    source: pd.DataFrame,
    *,
    keys: list[str],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """`target` with `columns` copied from `source`'s row of the same `keys`.

    Both frames must hold each key once, and the same set of keys.
    """
    if target.duplicated(keys).any() or source.duplicated(keys).any():
        raise ValueError(f"posterior transfer requires unique keys {keys}")
    require_columns(source, columns, "the posterior source")
    left_keys = pd.MultiIndex.from_frame(target.loc[:, keys])
    right = source.copy()
    right.index = pd.MultiIndex.from_frame(source.loc[:, keys])
    missing_keys = left_keys.difference(right.index)
    extra_keys = right.index.difference(left_keys)
    if len(missing_keys) or len(extra_keys):
        raise ValueError(
            f"posterior source key mismatch: missing={len(missing_keys)}, "
            f"extra={len(extra_keys)}"
        )
    output = target.copy()
    aligned = right.loc[left_keys, list(columns)].reset_index(drop=True)
    for column in columns:
        output[column] = aligned[column].to_numpy()
    return output


def rebuild_contfrag_session_tables(
    descriptor: dict,
    *,
    event_source: str,
    contingency: pd.DataFrame,
    old_event: pd.DataFrame,
    old_arm: pd.DataFrame,
    from_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One session's event, event-arm and stop tables, built without the posterior.

    The posterior aggregates come from `old_event` and `old_arm`, that session's
    rows of the preserved bundle (:func:`_copy_by_key`).
    """
    dot = read_dot_with_swr_source(descriptor["dot_file"])
    validate_dot_event_contract(dot)
    with Path(descriptor["env_file"]).open("rb") as stream:
        environment = pickle.load(stream)
    nwb = descriptor["nwb_file_name"]
    epoch = descriptor["epoch_name"]
    brain_area = descriptor["brain_area"]
    cache_key = format_epoch_identifier(nwb, epoch, brain_area)
    trials, experiences, graph, _ = replay_prepare_data.load_behavioral_context(
        {
            "nwb_file_name": nwb,
            "epoch_name": epoch,
            "session_id": cache_key,
        },
        from_cache=from_cache,
        cache_key=cache_key,
    )
    if trials is None or experiences is None or graph is None:
        raise ValueError(f"missing behavioral context for {cache_key}")

    event = build_contfrag_event_table(
        dot,
        environment,
        trials,
        experiences,
        graph,
        event_source=event_source,
        brain_area=brain_area,
        conditional_posterior=None,
    )
    event_label = f"{event_source}_event"
    event = _copy_by_key(
        event,
        old_event,
        keys=[event_label],
        columns=EVENT_POSTERIOR_COLUMNS,
    )
    arm = build_contfrag_event_arm_table(
        event,
        dot,
        environment,
        trials,
        experiences,
        graph,
        event_source=event_source,
        brain_area=brain_area,
        conditional_posterior=None,
        contingency=contingency,
    )
    arm = _copy_by_key(
        arm,
        old_arm,
        keys=[event_label, "candidate_arm"],
        columns=("arm_posterior_mass",),
    )
    arm["arm_fraction_posterior"] = arm_fraction_from_event_total(
        arm["arm_posterior_mass"].to_numpy(),
        arm["n_timebins"].to_numpy(dtype=float),
    )
    stop = build_contfrag_stop_table(
        dot,
        trials,
        experiences,
        graph,
        event_source=event_source,
        brain_area=brain_area,
    )
    for table in (event, arm, stop):
        table["event_source"] = event_source
        table["brain_area"] = brain_area
    return event, arm, stop
