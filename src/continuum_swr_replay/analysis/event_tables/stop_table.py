"""Canonical stop-grain builder: one row per stopping period."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    excluded_intervals_of,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_probability,
)
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema
from continuum_swr_replay.tables.event_ids import (
    hash_stop_id as _hash_stop_id,
)
from continuum_swr_replay.tables.session_keys import one_session

from ..behavior.covariates import (
    add_trial_context,
    lagged_trial_rewards,
    recent_trial_reward_rate,
    reward_history_basis_features,
)
from ..behavior.design_constants import (
    N_REWARD_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from ._common import (
    _STOP_TABLE_BUILD_COLUMNS,
    _in_event_mask,
)
from ._common import (
    first_row_floats as _first_row_floats,
)
from .bouts import (
    assign_bout as _assign_bout,
)
from .bouts import (
    bout_exposure_s as _bout_exposure_s,
)
from .bouts import (
    bout_start_context as _bout_start_context,
)
from .bouts import (
    immobility_bout_intervals as _immobility_bout_intervals,
)
from .bouts import (
    nearest_well_distance_cm as _nearest_well_distance_cm,
)


def build_stop_table(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One row per immobility bout, **including zero-event bouts**.

    The stop-level canonical view of the session. Every contiguous
    immobility bout (``head_speed`` below
    ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``, 4 cm/s) becomes a row,
    regardless of whether any non-local event fired during it. The
    per-stop rate models (Family A) need the zero-event rows.

    ``immobility_duration_s`` is the stop's exposure: ``stop_end -
    stop_start`` less the epoch's excluded time inside the stop
    (``analyzed_time.excluded_intervals``, the artifact mask widened by 0.3 s;
    nothing with no artifact run). A stop wholly inside it has zero exposure.
    The stop's location, segment and distance summaries (``occupancy_state``,
    ``at_well``, ``occupancy_well_id``, ``modal_observed_track_segment_id``,
    ``nearest_well_distance_cm_*`` and the columns derived from them) read only
    its rows outside excluded time (``bouts.summarize_bout_location``, through
    ``bouts.bout_start_context``): a stop wholly inside it is ``unknown``, with
    NaN well, segment and distances, so it is in no well-anchored cohort. The
    bout-start context columns read the stop's first row, as before.

    The ``stop_id`` column is a 63-bit hash of
    ``(animal, day, epoch, stop_idx)`` and matches
    :func:`build_event_table`'s ``stop_id`` row-for-row, so the two
    tables join cleanly.

    The table carries no replay-content summaries (e.g. Σ ``arm_count_hard``
    per arm); those live in ``event_arm_table``.

    Note
    ----
    Single-sample bouts — when the animal dips below the speed
    threshold for exactly one timebin — and stops wholly inside excluded
    time yield ``immobility_duration_s == 0`` rows. They are kept as-is
    here; ``stop_table`` is the raw canonical frame and downstream
    consumers decide whether to filter them. Both rate families leave
    them out: the per-stop fit's cohort freeze drops a non-positive
    ``log(immobility_duration_s)`` offset, and ``aggregate_to_timebins``
    gives such a stop no bins.


    Parameters
    ----------
    dot, trials, port_experiences, track_graph, nonlocal_threshold
        Same roles as in :func:`build_event_table`. ``env`` is not
        needed here — the stop table never touches the decoded arm of
        replay timebins.

    Returns
    -------
    pd.DataFrame
        Columns: see :data:`_STOP_TABLE_COLUMNS`.
    """
    require_columns(dot, ["animal"], "dot (set by get_data_over_time)")
    animal, day, epoch = one_session(dot, "build_stop_table")
    animal, day, epoch = str(animal), int(day), str(epoch)

    dot = dot.copy()
    dot["nonlocal_prob"] = nonlocal_probability(dot).to_numpy()
    nonlocal_event_mask = _in_event_mask(dot, nonlocal_threshold).to_numpy()
    dot = add_trial_context(dot, trials, port_experiences)
    dot["nearest_well_distance_cm"] = _nearest_well_distance_cm(dot, track_graph)

    bout_starts, bout_ends = _immobility_bout_intervals(
        dot,
        speed_col="head_speed",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    column_order = list(_STOP_TABLE_BUILD_COLUMNS)
    if bout_starts.size == 0:
        return pd.DataFrame(columns=column_order)

    n_bouts = bout_starts.size
    times = dot.index.to_numpy()
    row_bout_idx = _assign_bout(times, bout_starts, bout_ends)  # -1 for mobile rows

    # Per-bout aggregations — use pandas groupby on the in-bout rows.
    in_bout = row_bout_idx >= 0
    bout_rows = dot.loc[in_bout].assign(_bout_idx=row_bout_idx[in_bout])
    grouped = bout_rows.groupby("_bout_idx", sort=True)

    per_bout = pd.DataFrame({"stop_idx": np.arange(n_bouts, dtype=np.int64)})
    per_bout["stop_start"] = bout_starts
    per_bout["stop_end"] = bout_ends
    # The stop's exposure: its span less the excluded time inside it (the
    # artifact mask widened by 0.3 s is no rate's denominator). The bout itself
    # is unchanged. The location summaries below read the same intervals.
    excluded = excluded_intervals_of(dot)
    per_bout["immobility_duration_s"] = _bout_exposure_s(
        bout_starts, bout_ends, excluded
    )
    per_bout["reward_fraction_last10_known_trials"] = recent_trial_reward_rate(
        bout_starts, port_experiences
    )
    # Per-lag reward outcomes (known_reward_outcome_lag_1..N), ordered from the most recent
    # outcome already observable at bout start through earlier known outcomes.
    # Lag 1 is deliberately not called "the previous trial": after feedback it
    # can be the just-arrived trip; before feedback it remains the prior outcome.
    lag_rewards = lagged_trial_rewards(bout_starts, port_experiences, N_REWARD_LAGS)
    for j in range(N_REWARD_LAGS):
        per_bout[f"known_reward_outcome_lag_{j + 1}"] = lag_rewards[:, j]

    # Smooth reward kernel: the same known-outcome lag vector projected onto
    # log-spaced raised-cosine bumps, plus the structural-missingness flag for
    # stops whose slowest bump had no known outcome to average.
    reward_history, reward_history_incomplete = reward_history_basis_features(
        bout_starts, port_experiences
    )
    for j, column in enumerate(REWARD_HISTORY_COLUMNS):
        per_bout[column] = reward_history[:, j]
    per_bout[REWARD_HISTORY_INCOMPLETE] = reward_history_incomplete

    # Absolute bout-start clocks: when the animal arrived at the well it is
    # sitting at, and when that trip's outcome became observable. Both come
    # from the bout's first row as-is (``first(skipna=False)``), so a bout that
    # spans an arrival keeps the clock it started under rather than inheriting
    # the later trip's, and a bout with no completed trip yet stays NaN. The
    # outcome clock is a boundary, not an outcome-gated value: it is finite
    # from arrival onward, including bins that precede the boundary.
    clocks = _first_row_floats(grouped, ("_goal_arrival_time", "_outcome_known_time"))
    per_bout["last_well_arrival_time_s"] = clocks["_goal_arrival_time"]
    per_bout["outcome_known_time_s"] = clocks["_outcome_known_time"]

    per_bout["is_swr_stop"] = (
        grouped["SWR_event"]
        .apply(lambda s: (s > 0).any())
        .reindex(np.arange(n_bouts), fill_value=False)
        .to_numpy()
    )
    # n_swr_events: number of distinct whole-event-speed-eligible ripple labels in
    # the bout — the SWR analog of n_selected_events, for SWR-rate models.
    # (is_swr_stop == n_swr_events > 0.)
    per_bout["n_swr_events"] = (
        grouped["SWR_event"]
        .apply(lambda s: int(s[s > 0].nunique()))
        .reindex(np.arange(n_bouts), fill_value=0)
        .astype(np.int64)
        .to_numpy()
    )
    if "MUA_event" in dot.columns:
        per_bout["is_mua_stop"] = (
            grouped["MUA_event"]
            .apply(lambda s: (s > 0).any())
            .reindex(np.arange(n_bouts), fill_value=False)
            .to_numpy()
        )
        per_bout["n_mua_events"] = (
            grouped["MUA_event"]
            .apply(lambda s: int(s[s > 0].nunique()))
            .reindex(np.arange(n_bouts), fill_value=0)
            .astype(np.int64)
            .to_numpy()
        )
    else:
        per_bout["is_mua_stop"] = False
        per_bout["n_mua_events"] = 0

    # ``_in_event_mask`` validates complete labels before any bout slicing,
    # so a Non-Local event is counted atomically and can never be clipped into
    # multiple apparent events by a mobile interior sample. Every sample of such
    # an event is immobile, so all its rows lie in one bout: counting it in the
    # bout it has rows in is the ContFrag stop table's count by event start.
    event_rows = bout_rows.loc[nonlocal_event_mask[in_bout]]
    n_events_per_bout = (
        event_rows.groupby("_bout_idx")["non_local_event"]
        .nunique()
        .reindex(np.arange(n_bouts), fill_value=0)
        .astype(np.int64)
        .to_numpy()
    )
    per_bout["has_replay"] = n_events_per_bout > 0
    per_bout["n_selected_events"] = n_events_per_bout

    # The context at the stop's start (its first row, NaN included) and where
    # the stop was, as in the ContFrag stop table (one helper).
    # ``completed_trial_reward`` is read the same way; only this table has it.
    context = _bout_start_context(bout_rows, n_bouts, track_graph, excluded)
    for column in context.columns:
        per_bout[column] = context[column]
    per_bout["completed_trial_reward"] = (
        grouped["completed_trial_reward"]
        .first(skipna=False)
        .reindex(np.arange(n_bouts))
        .to_numpy()
    )

    per_bout.insert(0, "animal", animal)
    per_bout.insert(1, "session_date", day)
    per_bout.insert(2, "epoch", epoch)
    per_bout["stop_id"] = _hash_stop_id(
        animal, day, epoch, per_bout["stop_idx"].to_numpy()
    )

    return stamp_covariate_schema(per_bout[column_order], table="stop")


def add_nonlocal_swr_split(
    stop_table: pd.DataFrame, event_table: pd.DataFrame
) -> pd.DataFrame:
    """Add canonical per-stop SWR/non-SWR nonlocal-event counts.

    Splits each stop's non-local event count by the **canonical** event-level
    ``is_swr`` (strict positive-duration SWR overlap, from ``event_table``)
    aggregated on ``stop_id`` — so the split matches ``event_table`` exactly
    rather than introducing a second inclusive-overlap convention. Stops with
    no events get 0/0, and ``n_nonlocal_swr_events +
    n_nonlocal_non_swr_events`` equals ``n_selected_events`` for every stop.

    This is an assembly-level join (like the retrospective-metrics merge): it
    needs both canonical tables and so lives outside :func:`build_stop_table`,
    which is deliberately event-table-independent.
    """
    # Idempotent: drop any prior split so re-calling never produces _x/_y dups.
    out = stop_table.drop(
        columns=["n_nonlocal_swr_events", "n_nonlocal_non_swr_events"],
        errors="ignore",
    ).copy()
    require_columns(out, ["n_selected_events"], "add_nonlocal_swr_split's stop_table")
    total = out["n_selected_events"]
    if (
        len(event_table) == 0
        or "stop_id" not in event_table.columns
        or "is_swr" not in event_table.columns
    ):
        out["n_nonlocal_swr_events"] = 0
        out["n_nonlocal_non_swr_events"] = total.astype(np.int64)
        return out

    ev = event_table.dropna(subset=["stop_id"]).copy()
    # A non-local event whose stop_id is absent from stop_table would be silently
    # dropped by the left-merge below, leaving that stop a phantom 0/0 split. That
    # only happens if the two tables were built from different decodes/passes --
    # fail loud rather than corrupt the SWR split feeding the rate models.
    missing_stops = set(ev["stop_id"].unique()) - set(out["stop_id"])
    if missing_stops:
        raise ValueError(
            f"add_nonlocal_swr_split: {len(missing_stops)} stop_id(s) in "
            "event_table are absent from stop_table (e.g. "
            f"{next(iter(missing_stops))!r}); their non-local events would be "
            "silently dropped. The two tables must be built from the same decodes."
        )
    ev["_swr"] = ev["is_swr"].fillna(False).astype(bool)
    counts = (
        ev.groupby(["stop_id", "_swr"])
        .size()
        .unstack("_swr", fill_value=0)
        .rename(
            columns={
                True: "n_nonlocal_swr_events",
                False: "n_nonlocal_non_swr_events",
            }
        )
    )
    for col in ("n_nonlocal_swr_events", "n_nonlocal_non_swr_events"):
        if col not in counts.columns:
            counts[col] = 0
    out = out.merge(
        counts[["n_nonlocal_swr_events", "n_nonlocal_non_swr_events"]],
        left_on="stop_id",
        right_index=True,
        how="left",
    )
    split_cols = ["n_nonlocal_swr_events", "n_nonlocal_non_swr_events"]
    out[split_cols] = out[split_cols].fillna(0).astype(np.int64)
    return out
