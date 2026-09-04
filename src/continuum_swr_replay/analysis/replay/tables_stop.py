"""Canonical stop-grain builder: one row per stopping period."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)

from ._helpers import (
    arm_to_patch as _arm_to_patch,
)
from ._helpers import (
    assign_bout as _assign_bout,
)
from ._helpers import (
    hash_stop_id as _hash_stop_id,
)
from ._helpers import (
    immobility_bout_intervals as _immobility_bout_intervals,
)
from ._helpers import (
    nearest_well_distance_cm as _nearest_well_distance_cm,
)
from ._helpers import (
    safe_mode as _safe_mode,
)
from ._helpers import (
    single_arm_from_well as _single_arm_from_well,
)
from ._helpers import (
    strict_first as _strict_first,
)
from ._tables_common import (
    _STOP_TABLE_BUILD_COLUMNS,
    _filter_external_event_labels,
    _in_event_mask,
)
from .covariate_registry import stamp_covariate_schema
from .covariates import (
    add_trial_context,
    lagged_trial_rewards,
    recent_trial_reward_rate,
    summarize_behavioral_location,
)
from .model_specs import N_REWARD_LAGS


def build_stop_table(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One row per immobility bout, **including zero-event bouts**.

    The stop-level canonical view of the session. Every contiguous
    immobility bout (defined by ``head_speed < speed_threshold``) becomes a
    row, regardless of whether any non-local event fired during it. The
    hurdle occurrence model requires the zero-event rows.

    The ``stop_id`` column is a 63-bit hash of
    ``(animal, day, epoch, stop_idx)`` and matches
    :func:`build_event_table`'s ``stop_id`` row-for-row, so the two
    tables join cleanly.

    Deferred to later chunks: aggregated replay-summary columns
    (Σ ``arm_count_hard`` per arm, etc.) that require
    ``event_arm_table``.

    Note
    ----
    Single-sample bouts — when the animal dips below the speed
    threshold for exactly one timebin — yield ``immobility_time == 0``
    rows. They are kept as-is here; ``stop_table`` is the raw
    canonical frame and downstream consumers decide whether to filter
    them (e.g., the hurdle logistic, which uses
    ``log(immobility_time)``, will need to exclude or clip them).


    Parameters
    ----------
    dot, trials, port_experiences, track_graph, speed_threshold, nonlocal_threshold
        Same roles as in :func:`build_event_table`. ``env`` is not
        needed here — the stop table never touches the decoded arm of
        replay timebins.

    Returns
    -------
    pd.DataFrame
        Columns: see :data:`_STOP_TABLE_COLUMNS`.
    """
    if "animal" not in dot.columns:
        raise KeyError("dot must carry 'animal' column (set by get_data_over_time)")
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    dot = dot.copy()
    dot["nonlocal_prob"] = (
        dot["Non-Local_Continuous_prob"].to_numpy()
        + dot["Non-Local_Fragmented_prob"].to_numpy()
    )
    dot = _filter_external_event_labels(dot, speed_threshold)
    nonlocal_event_mask = _in_event_mask(
        dot, speed_threshold, nonlocal_threshold
    ).to_numpy()
    dot = add_trial_context(dot, trials, port_experiences)
    dot["nearest_well_distance_cm"] = _nearest_well_distance_cm(dot, track_graph)

    bout_starts, bout_ends = _immobility_bout_intervals(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
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
    per_bout["immobility_duration_s"] = bout_ends - bout_starts
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

    per_bout["is_swr_stop"] = (
        grouped["SWR_event"]
        .apply(lambda s: (s > 0).any())
        .reindex(np.arange(n_bouts), fill_value=False)
        .to_numpy()
    )
    # n_swr_events: number of distinct whole-event-speed-eligible ripple labels in
    # the bout — the SWR analogue of n_selected_events, for SWR-rate models.
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
    # multiple apparent events by a mobile interior sample.
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

    # Trial-context at bout start: first in-bout row's trial-derived
    # values. ``skipna=False`` is critical — the default ``first()``
    # silently skips NaNs and surfaces the first valid row anywhere
    # in the bout, which leaks future trial context into stops whose
    # actual bout start had no trial active yet (e.g. a bout
    # spanning a trial boundary inherits the post-boundary trial's outcome).
    bout_first_row = grouped.first(skipna=False)
    for col in (
        "did_last_completed_trip_switch_patch",
        "completed_trial_reward",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
    ):
        per_bout[col] = bout_first_row[col].reindex(np.arange(n_bouts)).to_numpy()

    # Persist one canonical ``is_rewarded_outcome``. The stay/omission
    # complements are derived downstream, not persisted here.
    per_bout["is_rewarded_outcome"] = (
        bout_first_row["is_rewarded_outcome"].reindex(np.arange(n_bouts)).to_numpy()
    )

    per_bout["completed_trial_id"] = pd.array(
        bout_first_row["completed_trial_id"].reindex(np.arange(n_bouts)),
        dtype="Int64",
    )

    # Behavioral-location and physical-segment fields use the whole bout.
    # ``occupancy_well_arm`` is defined only for strict at-well occupancy; a
    # trunk/in-transit/mixed bout is never rewritten as a
    # destination leaf arm. ``_strict_first`` below is reserved for contextual
    # previous/next/rewarded wells at the bout start.
    location_state_vals: list[str] = []
    at_well_vals: list[object] = []
    location_well_vals = np.full(n_bouts, np.nan)
    physical_arm_vals = np.full(n_bouts, np.nan)
    nearest_min_vals = np.full(n_bouts, np.nan)
    nearest_mean_vals = np.full(n_bouts, np.nan)
    nearest_max_vals = np.full(n_bouts, np.nan)
    current_arm_vals = np.full(n_bouts, np.nan)
    prev_arm_vals = np.full(n_bouts, np.nan)
    next_arm_vals = np.full(n_bouts, np.nan)
    last_rew_vals = np.full(n_bouts, np.nan)
    completed_trial_arm_vals = np.full(n_bouts, np.nan)
    for bi, (_, g) in enumerate(grouped):
        physical_arm = (
            _safe_mode(g["track_segment_id"]) if "track_segment_id" in g else np.nan
        )
        location = summarize_behavioral_location(
            g[["occupancy_state", "occupancy_well_id"]]
        )
        location_well = float(location["occupancy_well_id"])
        location_arm = _single_arm_from_well(location_well, track_graph)
        location_state_vals.append(str(location["occupancy_state"]))
        at_well_vals.append(location["at_well"])
        location_well_vals[bi] = location_well
        physical_arm_vals[bi] = physical_arm
        current_arm_vals[bi] = location_arm
        distance = g["nearest_well_distance_cm"].to_numpy(dtype=float)
        finite_distance = distance[np.isfinite(distance)]
        if finite_distance.size:
            nearest_min_vals[bi] = finite_distance.min()
            nearest_mean_vals[bi] = finite_distance.mean()
            nearest_max_vals[bi] = finite_distance.max()
        prev_arm_vals[bi] = _single_arm_from_well(
            _strict_first(g["prev_well"]), track_graph
        )
        next_arm_vals[bi] = _single_arm_from_well(
            _strict_first(g["next_well"]), track_graph
        )
        last_rew_vals[bi] = _single_arm_from_well(
            _strict_first(g["last_rewarded_well"]), track_graph
        )
        completed_trial_arm_vals[bi] = _single_arm_from_well(
            _strict_first(g["last_visited_well"]), track_graph
        )

    per_bout["occupancy_state"] = location_state_vals
    per_bout["at_well"] = pd.array(at_well_vals, dtype="boolean")
    per_bout["occupancy_well_id"] = location_well_vals
    per_bout["modal_observed_track_segment_id"] = physical_arm_vals
    per_bout["nearest_well_distance_cm_min"] = nearest_min_vals
    per_bout["nearest_well_distance_cm_mean"] = nearest_mean_vals
    per_bout["nearest_well_distance_cm_max"] = nearest_max_vals
    per_bout["occupancy_well_arm"] = current_arm_vals
    per_bout["prev_arm"] = prev_arm_vals
    per_bout["next_arm"] = next_arm_vals
    per_bout["last_rewarded_arm"] = last_rew_vals
    per_bout["completed_trial_arm"] = completed_trial_arm_vals
    cur_patch = np.array([_arm_to_patch(a) for a in current_arm_vals])
    nxt_patch = np.array([_arm_to_patch(a) for a in next_arm_vals])
    per_bout["current_patch"] = cur_patch
    per_bout["prev_patch"] = [_arm_to_patch(a) for a in prev_arm_vals]
    per_bout["next_patch"] = nxt_patch
    # is_stay: will the next trip be within the same patch? Post-hoc.
    both_valid = ~np.isnan(cur_patch.astype(float)) & ~np.isnan(nxt_patch.astype(float))
    per_bout["is_stay"] = np.where(
        both_valid, (cur_patch == nxt_patch).astype(float), np.nan
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
    total = out.get("n_selected_events")
    if total is None:
        raise KeyError("add_nonlocal_swr_split requires canonical 'n_selected_events'")
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


