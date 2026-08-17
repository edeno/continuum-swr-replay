"""Shared helpers for event-table builders.

These utilities are used by both the NonLocal (``tables.py``) and
ContFrag (``contfrag/tables.py``) table builders. They handle
trial-context annotation, immobility-bout detection, spatial
arm/well/patch mappings, and generic DataFrame helpers.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.data_loaders.constants import (
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)

from .model_specs import ARM_FE_PREDICTORS
from .spatial_mapping import well_to_arm_set

# ── Trial-derived context ──


_LOCATION_STATES = ("at_well", "in_transit", "mixed", "unknown")


def _validate_well_values(values: np.ndarray, column: str) -> None:
    """Require finite integral IDs from the canonical six-well task."""
    valid = np.asarray(tuple(WELL_PATCH_MAPPING), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"trial {column} values must be finite")
    if not np.equal(values, np.floor(values)).all() or not np.isin(values, valid).all():
        raise ValueError(
            f"trial {column} values must be canonical integral well IDs "
            f"{tuple(WELL_PATCH_MAPPING)}"
        )


def _prepare_trials_for_context(
    trials: pd.DataFrame,
    *,
    required_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate and order trials by their canonical destination-arrival time.

    A trip reaches its destination at ``goal_arrival_time``, with the
    established per-row ``end_time`` fallback. Sorting by that clock is
    essential for location and visit history. Reward history uses the separate
    outcome clock returned by :func:`_trial_outcome_times`.
    """
    if trials.empty:
        return trials.copy(), np.array([], dtype=float)

    base_columns = {"start_time", "end_time", "to_well", *required_columns}
    missing = sorted(base_columns.difference(trials.columns))
    if missing:
        raise KeyError(f"trial context requires columns {missing}")

    prepared = trials.copy()
    starts = prepared["start_time"].to_numpy(dtype=float)
    ends = prepared["end_time"].to_numpy(dtype=float)
    cutoffs = _trial_arrival_times(prepared)
    to_wells = prepared["to_well"].to_numpy(dtype=float)

    if not np.isfinite(starts).all():
        raise ValueError("trial start_time values must be finite")
    if not np.isfinite(ends).all():
        raise ValueError("trial end_time values must be finite")
    if not np.isfinite(cutoffs).all():
        raise ValueError("trial destination-arrival cutoffs must be finite")
    _validate_well_values(to_wells, "to_well")
    if np.any(cutoffs < starts):
        raise ValueError("a trial destination arrival occurs before its start")
    if np.any(cutoffs > ends):
        raise ValueError("a trial destination arrival occurs after its end_time")

    order = np.argsort(cutoffs, kind="stable")
    prepared = prepared.iloc[order].reset_index(drop=True)
    cutoffs = cutoffs[order]
    starts = starts[order]

    # Consecutive trips may meet at a boundary.  They may not overlap: an
    # animal cannot be in two transit intervals at once.  This check also
    # catches trial orders that disagree between departure and completion.
    if len(prepared) > 1 and np.any(starts[1:] < cutoffs[:-1]):
        raise ValueError("trial transit intervals overlap")

    return prepared, cutoffs


def behavioral_location_at_times(
    times: np.ndarray,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Classify timestamps from trip departure/arrival intervals.

    ``[start_time, completion_cutoff)`` is ``"in_transit"``.  From the
    completion cutoff through the next trip's start is ``"at_well"`` at the
    completed trial's destination.  Consequently, exact arrival is at-well
    and exact departure is in-transit.  Times before the first departure are
    ``"unknown"``.  The final dwell extends beyond the last recorded trial.

    The returned columns are ``location_state`` (categorical), ``at_well``
    (nullable boolean), and ``location_well`` (nullable ``Float64``).
    """
    sample_times = np.asarray(times, dtype=float)
    if sample_times.ndim != 1:
        raise ValueError("times must be a one-dimensional sequence")
    if not np.isfinite(sample_times).all():
        raise ValueError("times must contain only finite timestamps")

    index = pd.Index(sample_times, name="time")
    if trials.empty:
        return pd.DataFrame(
            {
                "location_state": pd.Categorical(
                    ["unknown"] * len(sample_times), categories=_LOCATION_STATES
                ),
                "at_well": pd.array([pd.NA] * len(sample_times), dtype="boolean"),
                "location_well": pd.array([pd.NA] * len(sample_times), dtype="Float64"),
            },
            index=index,
        )

    prepared, cutoffs = _prepare_trials_for_context(trials)
    starts = prepared["start_time"].to_numpy(dtype=float)
    to_wells = prepared["to_well"].to_numpy(dtype=float)

    started_idx = np.searchsorted(starts, sample_times, side="right") - 1
    has_started = started_idx >= 0
    clipped = np.clip(started_idx, 0, len(prepared) - 1)
    is_transit = has_started & (sample_times < cutoffs[clipped])
    is_at_well = has_started & ~is_transit

    state = np.full(len(sample_times), "unknown", dtype=object)
    state[is_transit] = "in_transit"
    state[is_at_well] = "at_well"
    at_well = pd.array([pd.NA] * len(sample_times), dtype="boolean")
    at_well[is_transit] = False
    at_well[is_at_well] = True
    location_well = pd.array([pd.NA] * len(sample_times), dtype="Float64")
    location_well[is_at_well] = to_wells[clipped[is_at_well]]

    return pd.DataFrame(
        {
            "location_state": pd.Categorical(state, categories=_LOCATION_STATES),
            "at_well": at_well,
            "location_well": location_well,
        },
        index=index,
    )


def summarize_behavioral_location(
    states: pd.DataFrame | pd.Series | np.ndarray | list[str],
    wells: pd.Series | np.ndarray | list[float] | None = None,
) -> dict[str, object]:
    """Roll time-resolved location labels up over one complete window.

    A window is ``at_well`` only when every sampled bin is at the same well.
    It is ``in_transit`` only when every bin is in transit, ``unknown`` when
    empty or wholly unknown, and ``mixed`` whenever states differ or an
    at-well window spans multiple wells.  ``at_well`` is nullable only for an
    unknown window; mixed windows are explicitly false.
    """
    if isinstance(states, pd.DataFrame):
        if wells is not None:
            raise ValueError("wells must be omitted when states is a DataFrame")
        required = {"location_state", "location_well"}
        missing = sorted(required.difference(states.columns))
        if missing:
            raise KeyError(f"location frame requires columns {missing}")
        wells = states["location_well"]
        states = states["location_state"]

    state_values = np.asarray(states, dtype=object)
    if state_values.ndim != 1:
        raise ValueError("location states must be one-dimensional")
    if wells is None:
        well_values = np.full(len(state_values), np.nan)
    else:
        well_values = pd.to_numeric(pd.Series(wells), errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        if well_values.ndim != 1 or len(well_values) != len(state_values):
            raise ValueError("location states and wells must have equal lengths")

    if len(state_values) == 0:
        return {
            "location_state": "unknown",
            "at_well": pd.NA,
            "location_well": np.nan,
        }

    allowed = set(_LOCATION_STATES) - {"mixed"}
    observed_states = set(state_values.tolist())
    invalid = observed_states.difference(allowed)
    if invalid:
        raise ValueError(f"invalid time-resolved location states: {sorted(invalid)}")

    if observed_states == {"unknown"}:
        return {
            "location_state": "unknown",
            "at_well": pd.NA,
            "location_well": np.nan,
        }
    if observed_states == {"in_transit"}:
        return {
            "location_state": "in_transit",
            "at_well": False,
            "location_well": np.nan,
        }
    if observed_states == {"at_well"}:
        unique_wells = np.unique(well_values[np.isfinite(well_values)])
        if len(unique_wells) == 1 and np.isfinite(well_values).all():
            return {
                "location_state": "at_well",
                "at_well": True,
                "location_well": float(unique_wells[0]),
            }

    return {
        "location_state": "mixed",
        "at_well": False,
        "location_well": np.nan,
    }


def nearest_well_distance_cm(
    dot: pd.DataFrame,
    track_graph: nx.Graph,
) -> np.ndarray:
    """Euclidean distance from each projected position to the nearest well.

    Wells are the degree-one nodes of ``track_graph``.  A malformed track
    graph raises: an all-NaN column is indistinguishable from "the animal
    was never near a well", and would silently empty any downstream
    near-a-well cohort for the whole session.  Missing position *data*
    (absent coordinate columns, individual non-finite rows) is a genuine
    data gap and still yields NaN.

    Parameters
    ----------
    dot : pd.DataFrame
        Timebin frame; uses ``projected_x_position`` / ``projected_y_position``.
    track_graph : nx.Graph
        Track graph whose degree-one nodes are wells, each with a ``pos``
        attribute holding a finite ``(x, y)`` pair.

    Returns
    -------
    np.ndarray, shape (len(dot),)
        Distance in cm, NaN where the position is unavailable.

    Raises
    ------
    ValueError
        If the graph has no degree-one nodes, or any well lacks a finite
        two-element ``pos``.
    """
    n_rows = len(dot)
    coordinate_columns = ("projected_x_position", "projected_y_position")
    if not set(coordinate_columns).issubset(dot.columns):
        return np.full(n_rows, np.nan)

    well_nodes = [node for node in track_graph.nodes if track_graph.degree(node) == 1]
    if not well_nodes:
        raise ValueError(
            "track_graph has no degree-one nodes, so no wells can be located"
        )
    node_positions = nx.get_node_attributes(track_graph, "pos")
    missing_pos = [node for node in well_nodes if node not in node_positions]
    if missing_pos:
        raise ValueError(
            f"track_graph well nodes lack a 'pos' attribute: {missing_pos}"
        )

    try:
        well_positions = np.asarray(
            [node_positions[node] for node in well_nodes], dtype=float
        )
    except (TypeError, ValueError) as err:
        raise ValueError(
            "track_graph well 'pos' attributes are not numeric (x, y) pairs"
        ) from err
    if (
        well_positions.shape != (len(well_nodes), 2)
        or not np.isfinite(well_positions).all()
    ):
        raise ValueError(
            "track_graph well 'pos' attributes must be finite two-element "
            f"(x, y) pairs; got shape {well_positions.shape}"
        )

    positions = dot.loc[:, coordinate_columns].to_numpy(dtype=float)
    output = np.full(n_rows, np.nan)
    valid = np.isfinite(positions).all(axis=1)
    if valid.any():
        displacement = positions[valid, None, :] - well_positions[None, :, :]
        output[valid] = np.linalg.norm(displacement, axis=2).min(axis=1)
    return output


def add_trial_context(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Attach per-timebin trial-derived columns to ``dot`` (in place, returned).

    Destination and switch context is indexed by the most recent trip that
    reached its goal at or before ``t``. Reward-bearing context uses a second
    clock and stays unknown until that same trip's outcome is observable. An
    in-progress trip cannot expose its destination, reward, or switch outcome
    through these columns. ``next_well`` is the
    sole forward-looking column: it names a destination the animal has not
    reached yet, so it is a response variable, never a causal predictor.
    (Elsewhere in this codebase "retrospective" means past-referenced -- see
    ``retrospective.py`` -- so it is deliberately not used for ``next_well``.)

    Columns added
    -------------
    - ``prev_well``, ``current_well``, ``last_visited_well`` (past-only)
    - ``next_well`` (forward-looking: the not-yet-reached destination)
    - ``last_rewarded_well`` — advances only when a reward is observable
    - ``completed_trial_id`` — zero-based completion-order ID regenerated
      within the session (not an index copied from the source trial table)
    - ``recent_switch_rate`` — canonical five-completed-trial rolling switch
      rate through the identified trial, computed from the full trial stream
    - ``leaky_q_value_diff`` — canonical alpha-0.3 pre-update Q margin for the
      identified trial, computed while updating on every completed trial
    - ``is_after_switch`` — last completed trip's ``is_patch_change``
    - ``is_after_omission`` — current arrived trip's omission, NaN until known
    - ``is_after_reward`` — current arrived trip's reward, NaN until known
    - ``reward_outcome`` — current arrived trip's outcome, NaN until known
    - ``completed_trial_reward`` — the same outcome, ungated (post-outcome;
      descriptive constructs only, never a predictor)
    - ``location_state``, ``at_well``, ``location_well``
    """
    dot_times = np.asarray(dot.index.values, dtype=float)
    if dot_times.ndim != 1 or not np.isfinite(dot_times).all():
        raise ValueError("dot index must contain only finite timestamps")

    location = behavioral_location_at_times(dot_times, trials)
    for column in location.columns:
        dot[column] = location[column].array

    if trials.empty:
        for col in (
            "prev_well",
            "current_well",
            "last_visited_well",
            "next_well",
            "last_rewarded_well",
            "reward_outcome",
            "completed_trial_reward",
            "is_after_reward",
            "is_after_omission",
            "is_after_switch",
            "_goal_arrival_time",
            "_last_reward_time",
            "recent_switch_rate",
            "leaky_q_value_diff",
        ):
            dot[col] = np.nan
        dot["completed_trial_id"] = pd.array([pd.NA] * len(dot), dtype="Int64")
        return dot

    trials, arrival_times = _prepare_trials_for_context(
        trials,
        required_columns=("from_well", "is_reward", "is_patch_change"),
    )
    outcome_times = _trial_outcome_times(trials)
    n_trials = len(trials)
    trial_start_times = trials["start_time"].to_numpy(dtype=float)
    to_wells = trials["to_well"].to_numpy(dtype=float)
    from_wells = trials["from_well"].to_numpy(dtype=float)
    reward_values = trials["is_reward"].to_numpy(dtype=float)
    patch_change_values = trials["is_patch_change"].to_numpy(dtype=float)
    _validate_well_values(from_wells, "from_well")
    for name, values in (
        ("is_reward", reward_values),
        ("is_patch_change", patch_change_values),
    ):
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"trial {name} values must be finite binary values")
    is_reward = reward_values.astype(bool)
    is_patch_change = patch_change_values.astype(bool)
    if np.any(np.diff(outcome_times) < 0):
        raise ValueError("trial outcome-known times must be nondecreasing")
    if n_trials > 1 and np.any(outcome_times[:-1] > arrival_times[1:]):
        raise ValueError(
            "a trial outcome remains unknown after the next destination arrival"
        )

    # Local import avoids the module's deliberate re-export cycle: covariates
    # imports low-level trial helpers from this module, while this call needs
    # the canonical scientific definition owned by covariates. The function is
    # invoked only after both modules have finished importing.
    from .covariates import canonical_trial_uncertainty_history

    trial_switch_rate, trial_q_margin = canonical_trial_uncertainty_history(
        to_wells,
        reward_values,
        patch_change_values,
    )

    completed_idx = np.searchsorted(arrival_times, dot_times, side="right") - 1
    has_completed = completed_idx >= 0
    completed_clipped = np.clip(completed_idx, 0, n_trials - 1)
    # This ID is deliberately regenerated after canonical completion-time
    # sorting. Source trial indices are not stable across loaders, whereas a
    # zero-based completion order is the visit identity needed by downstream
    # history covariates. Before the first observed completion it is unknown.
    completed_trial_id = pd.array(completed_idx, dtype="Int64")
    completed_trial_id[~has_completed] = pd.NA
    dot["completed_trial_id"] = completed_trial_id
    dot["recent_switch_rate"] = np.where(
        has_completed, trial_switch_rate[completed_clipped], np.nan
    )
    dot["leaky_q_value_diff"] = np.where(
        has_completed, trial_q_margin[completed_clipped], np.nan
    )

    dot["prev_well"] = np.where(has_completed, from_wells[completed_clipped], np.nan)
    last_visited = np.where(has_completed, to_wells[completed_clipped], np.nan)
    dot["last_visited_well"] = last_visited
    # Compatibility alias: unlike the previous implementation, current_well
    # is now strictly past-only and never means an in-progress destination.
    dot["current_well"] = last_visited

    started_idx = np.searchsorted(trial_start_times, dot_times, side="right") - 1
    next_idx = completed_idx + 1
    next_valid = (started_idx >= 0) & (next_idx < n_trials)
    next_clipped = np.clip(next_idx, 0, n_trials - 1)
    dot["next_well"] = np.where(next_valid, to_wells[next_clipped], np.nan)

    # Reward identity is not available merely because the animal arrived.
    # Advance this history at the physical outcome clock: exact pump onset for
    # newly generated rewarded trials, poke-out for omissions, or the explicit
    # conservative end-time fallback documented by _trial_outcome_times.
    reward_timestamps = outcome_times[is_reward]
    rewarded_wells = to_wells[is_reward]
    if len(reward_timestamps) > 0:
        rew_idx = np.searchsorted(reward_timestamps, dot_times, side="right") - 1
        rew_idx_clipped = np.clip(rew_idx, 0, None)
        dot["last_rewarded_well"] = np.where(
            rew_idx >= 0, rewarded_wells[rew_idx_clipped], np.nan
        )
    else:
        dot["last_rewarded_well"] = np.nan

    dot["_goal_arrival_time"] = np.where(
        has_completed, arrival_times[completed_clipped], np.nan
    )

    if len(reward_timestamps) > 0:
        dot["_last_reward_time"] = np.where(
            rew_idx >= 0, reward_timestamps[rew_idx_clipped], np.nan
        )
    else:
        dot["_last_reward_time"] = np.nan

    current_outcome_known = has_completed & (
        dot_times >= outcome_times[completed_clipped]
    )
    current_reward = np.where(
        current_outcome_known,
        is_reward[completed_clipped].astype(float),
        np.nan,
    )
    dot["reward_outcome"] = current_reward
    dot["is_after_reward"] = current_reward
    dot["is_after_omission"] = np.where(
        current_outcome_known, 1.0 - current_reward, np.nan
    )
    # The completed trip's realized outcome, NOT gated on when it became
    # observable. The gated columns above are the ones a predictor may use;
    # this one exists for constructs that are *about* the outcome the animal
    # just experienced and are already barred from predictive use for another
    # reason. ``wsls_deviation`` is the case: it pairs this outcome with
    # ``is_stay``, the eventual outbound choice, so it is future-referencing
    # whatever the outcome clock does, and gating it at the bout start only
    # made it undefined at every at-well stop.
    dot["completed_trial_reward"] = np.where(
        has_completed, is_reward[completed_clipped].astype(float), np.nan
    )
    dot["is_after_switch"] = np.where(
        has_completed, is_patch_change[completed_clipped].astype(float), np.nan
    )

    return dot


def trial_context_at_times(
    times: np.ndarray,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the canonical trial context at arbitrary timestamps.

    This is the read-only, timestamp-oriented entry point for consumers that
    need the exact same boundary, sorting, and missing-value semantics as
    :func:`add_trial_context` without constructing a decoder-sized ``dot``
    frame.  The returned frame is newly allocated; ``trials`` is not mutated.

    Parameters
    ----------
    times : array-like
        Timestamps at which to sample trial context.
    trials : pandas.DataFrame
        Canonical trial table accepted by :func:`add_trial_context`.

    Returns
    -------
    pandas.DataFrame
        One row per input timestamp, indexed by time, with the columns written
        by :func:`add_trial_context`.
    """
    sample_times = np.asarray(times, dtype=float)
    if sample_times.ndim != 1:
        raise ValueError("times must be a one-dimensional sequence")
    if not np.isfinite(sample_times).all():
        raise ValueError("times must contain only finite timestamps")
    sampled = pd.DataFrame(index=pd.Index(sample_times, name="time"))
    return add_trial_context(sampled, trials)


# ── Immobility-bout helpers ──


def immobility_bout_starts(
    dot: pd.DataFrame,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> np.ndarray:
    """Timestamps at which speed crossed from above to below the threshold."""
    imm = (dot[speed_col] < speed_threshold).to_numpy()
    imm_times = dot.index.to_numpy()
    transitions = np.where(~imm[:-1] & imm[1:])[0] + 1
    if len(imm) > 0 and imm[0]:
        transitions = np.concatenate([[0], transitions])
    return imm_times[transitions]


def immobility_bout_intervals(
    dot: pd.DataFrame,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Start and end timestamps for every contiguous immobility bout.

    Returns
    -------
    starts : np.ndarray, shape (n_bouts,)
    ends : np.ndarray, shape (n_bouts,)
    """
    imm = (dot[speed_col] < speed_threshold).to_numpy()
    times = dot.index.to_numpy()
    if imm.size == 0 or not imm.any():
        return np.array([]), np.array([])

    padded = np.concatenate(([False], imm, [False]))
    diff = np.diff(padded.astype(np.int8))
    start_idx = np.where(diff == 1)[0]
    end_idx = np.where(diff == -1)[0] - 1
    return times[start_idx], times[end_idx]


def time_since_stop(
    event_times: np.ndarray,
    bout_start_times: np.ndarray,
) -> np.ndarray:
    """Seconds from most recent immobility-bout start to each event time."""
    event_times = np.asarray(event_times, dtype=float)
    if bout_start_times.size == 0:
        return np.full_like(event_times, np.nan)
    idx = np.searchsorted(bout_start_times, event_times, side="right") - 1
    idx = np.clip(idx, 0, bout_start_times.size - 1)
    return np.clip(event_times - bout_start_times[idx], 0, None)


# ── Scalar extractions from per-arm sets ──


def single_arm_from_well(well: float, track_graph: nx.Graph) -> float:
    """Deterministic scalar arm for a well, ``NaN`` if the well is NaN or isolated."""
    if well is None or (isinstance(well, float) and np.isnan(well)):
        return np.nan
    arms = well_to_arm_set(well, track_graph)
    if len(arms) == 0:
        return np.nan
    return float(min(arms))


def arm_to_patch(arm: float) -> float:
    """Patch ID for an arm, ``NaN`` propagating."""
    if arm is None or (isinstance(arm, float) and np.isnan(arm)):
        return np.nan
    return float(TRACK_SEGMENT_TO_PATCH.get(int(arm), np.nan))


# ── Posterior alignment ──


def align_posterior_to_dot(
    conditional_posterior,
    dot: pd.DataFrame,
) -> np.ndarray:
    """Return a ``(T_dot, n_position_bins)`` posterior array aligned to ``dot.index``."""
    post_vals = np.asarray(conditional_posterior.values)
    if post_vals.shape[0] != len(dot):
        raise ValueError(
            f"conditional_posterior time axis has {post_vals.shape[0]} "
            f"entries but dot has {len(dot)}; the two must be aligned "
            "1-to-1 (both come from the same decoder run)."
        )
    if "time" in conditional_posterior.coords:
        post_times = np.asarray(conditional_posterior["time"].values, dtype=float)
        drift = float(np.abs(post_times - dot.index.to_numpy(dtype=float)).max())
        if drift > 1e-6:
            raise ValueError(
                f"conditional_posterior time axis drifts from dot.index "
                f"by up to {drift:.3g}s — the two must share the same "
                "time origin, not just the same length."
            )
    return post_vals


# ── Generic helpers ──


def safe_mode(series: pd.Series) -> float:
    """Mode of a series (first value on tie), ``NaN`` if empty or all NaN."""
    if len(series) == 0:
        return np.nan
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    mode = valid.mode()
    if len(mode) == 0:
        return np.nan
    return float(mode.iloc[0])


def safe_first(series: pd.Series) -> float:
    """First non-NaN value of a series, ``NaN`` if empty or all NaN."""
    if len(series) == 0:
        return np.nan
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.iloc[0])


def strict_first(series: pd.Series) -> float:
    """First row's value as-is — NaN propagates.

    Use this when extracting event-level / bout-level context columns
    that come from ``add_trial_context``: those columns are NaN at every
    timebin before the first trial start. ``safe_first`` would silently
    skip leading NaNs and inherit the *next* trial's context, leaking
    future state into events that started before any trial existed.
    ``strict_first`` returns the first row's value as-is (NaN stays
    NaN), so before-trial events surface as missing context instead of
    fabricated post-hoc values.
    """
    if len(series) == 0:
        return np.nan
    v = series.iloc[0]
    return float(v) if pd.notna(v) else np.nan


def strict_mean(series: pd.Series) -> float:
    """``Series.mean`` with ``skipna=False``.

    pandas' default ``"mean"`` aggregator silently drops NaN, so a
    groupby reduction over a column that contains a mix of finite and
    NaN values emits the mean of the surviving rows. For Stage-5/6
    stop-level / arm-level aggregations that's a per-event survivorship
    bias: a stop with one finite event and one NaN event reports the
    finite event's value while ``n_events`` still counts both. Use
    this aggregator instead so any NaN propagates to the output.
    """
    return float(series.mean(skipna=False))


def strict_sum(series: pd.Series) -> float:
    """``Series.sum`` with ``skipna=False``."""
    return float(series.sum(skipna=False))


def hash_event_id(
    events: pd.DataFrame,
    event_col: str = "non_local_event",
) -> np.ndarray:
    """63-bit stable hash of ``(animal, day, epoch, event_col)``."""
    keys = events[["animal", "day", "epoch", event_col]]
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()


def validate_event_ids(event_table: pd.DataFrame) -> None:
    """Raise if ``event_id`` has hash collisions after cross-session concat.

    Call this after ``pd.concat`` across sessions to guard against the
    (unlikely but non-zero) birthday-paradox collision in 63-bit hashes.
    """
    n_unique = event_table["event_id"].nunique()
    n_total = len(event_table)
    if n_unique != n_total:
        raise RuntimeError(
            f"event_id hash collision: {n_total} rows but {n_unique} unique IDs"
        )


def hash_stop_id(
    animal: str,
    day: int,
    epoch: str,
    stop_idx: np.ndarray,
    *,
    brain_area: str | None = None,
    event_source: str | None = None,
) -> np.ndarray:
    """63-bit stable hash of ``(animal, day, epoch, stop_idx)`` per bout.

    Optional ``brain_area`` / ``event_source`` keywords are mixed into
    the hash when supplied. ContFrag callers pass them so SWR/MUA and
    HPC/mPFC/OFC stop tables never collide on ``stop_id``. The
    NonLocal pipeline omits both (single-area, single-source by
    construction) and keeps its existing hashes intact.
    """
    keys = pd.DataFrame(
        {
            "animal": animal,
            "day": day,
            "epoch": epoch,
            "stop_idx": np.asarray(stop_idx, dtype=np.int64),
        }
    )
    if brain_area is not None:
        keys["__brain_area"] = brain_area
    if event_source is not None:
        keys["__event_source"] = event_source
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()


def assign_bout(
    event_times: np.ndarray,
    bout_starts: np.ndarray,
    bout_ends: np.ndarray,
) -> np.ndarray:
    """Return the bout index containing each timestamp, or ``-1`` if none."""
    event_times = np.asarray(event_times, dtype=float)
    if bout_starts.size == 0:
        return np.full(event_times.shape, -1, dtype=np.int64)
    idx = np.searchsorted(bout_starts, event_times, side="right") - 1
    clipped = np.clip(idx, 0, bout_starts.size - 1)
    in_bout = (idx >= 0) & (event_times <= bout_ends[clipped])
    return np.where(in_bout, idx, -1).astype(np.int64)


def eq_indicator(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Elementwise ``int(values == reference)`` with NaN-in-reference = 0.

    Use this only where an absent reference genuinely means "no arm
    qualifies" -- ``is_prev`` on the first trip of a session, or
    ``is_last_rewarded`` before the first reward.  Where an absent
    reference instead means "unobserved", use
    :func:`eq_indicator_or_nan` so the row is excluded by complete-case
    fitting rather than silently labelled a zero.
    """
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    out = (values == reference).astype(int)
    out[np.isnan(reference)] = 0
    return out


def eq_indicator_or_nan(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Elementwise ``values == reference`` as 1.0/0.0, NaN where reference is NaN.

    The missing-data counterpart of :func:`eq_indicator`, for indicators
    whose reference can be *unobserved* rather than merely absent.
    ``current_arm`` is NaN whenever an event window is not wholly at one
    well (``in_transit``, ``mixed``, ``unknown``); the animal still
    occupies some arm, we just cannot say which.  Encoding that as 0 on
    every arm would assert "not the current arm" for all nine, which is
    a fabricated observation.  Propagating NaN instead lets complete-case
    cohort selection drop the event through its logged path.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        Per-row candidate values (e.g. arm or patch IDs).
    reference : np.ndarray, shape (n_rows,)
        Per-row reference value, broadcast alongside ``values``.

    Returns
    -------
    np.ndarray, shape (n_rows,), dtype float
        1.0 where equal, 0.0 where unequal, NaN where ``reference`` is NaN.
    """
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    out = (values == reference).astype(float)
    out[np.isnan(reference)] = np.nan
    return out


def align_count_matrix(
    counts: np.ndarray,
    ids_sorted: np.ndarray,
    target_ids: np.ndarray,
    n_arms: int,
) -> np.ndarray:
    """Reindex a ``(n_events_src, n_arms)`` count matrix to ``target_ids``.

    An id in ``target_ids`` with no row in ``ids_sorted`` is an event that
    contributed no counted rows -- every one of its in-event timebins failed to
    attribute to a real arm. Its output row is filled, and **the fill value
    depends on the input dtype**: integer counts fill ``0`` (a real count of
    zero) and float counts fill ``NaN`` (unknown). That is deliberate --
    ``arm_count_hard`` genuinely observed zero real-arm bins, while
    ``arm_fraction``'s 0/0 must stay undefined rather than become 0 or +inf --
    but it is subtle enough to be worth stating, since the two paths differ
    only by ``counts.dtype``.

    On the shipped corpus no event triggers the fill (``n_junction_bins`` is 0
    for all 65,913), so this path is exercised only by tests.
    """
    mapping = {int(i): k for k, i in enumerate(ids_sorted)}
    fill = np.nan if np.issubdtype(counts.dtype, np.floating) else 0
    out = np.full((target_ids.size, n_arms), fill, dtype=counts.dtype)
    for k, tid in enumerate(target_ids):
        src_idx = mapping.get(int(tid))
        if src_idx is not None:
            out[k] = counts[src_idx]
    return out


def add_arm_fixed_effects(
    df: pd.DataFrame,
    arm_col: str = "arm",
) -> pd.DataFrame:
    """Add the arm fixed-effect dummy columns named by ``ARM_FE_PREDICTORS``.

    ``arm_fe_k`` is ``1`` where ``df[arm_col] == k`` and ``0`` otherwise.
    The dropped reference arm (the ninth) is absent from
    ``ARM_FE_PREDICTORS`` — including all nine dummies would be
    collinear with the conditional-Poisson per-event normalization.

    The columns are added in place and ``df`` is returned (matching
    the in-place contract of :func:`add_trial_context`). Existing
    ``arm_fe_*`` columns are overwritten.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format event-arm frame. Must carry ``arm_col``.
    arm_col : str
        Integer arm-id column (``"arm"`` in the canonical event-arm
        table, ``"arm_id"`` in simulator output).

    Returns
    -------
    pd.DataFrame
        ``df`` with the dummy columns added.
    """
    if arm_col not in df.columns:
        raise KeyError(f"add_arm_fixed_effects: missing arm column {arm_col!r}")
    arm = df[arm_col].to_numpy()
    for k, col in enumerate(ARM_FE_PREDICTORS):
        df[col] = (arm == k).astype(np.int8)
    return df


# ── Per-arm reward/visit history helpers ──


def build_well_to_arm_map(track_graph: nx.Graph) -> dict[int, int]:
    """Precompute well -> arm mapping from the track graph.

    On the real bandit graph each well (degree-1 node) maps to exactly
    one arm (edge).  Building this once avoids repeated graph traversals
    inside the per-event helpers.
    """
    mapping: dict[int, int] = {}
    for u, v, data in track_graph.edges(data=True):
        eid = data.get("edge_id")
        if eid is None:
            continue
        for node in (u, v):
            if track_graph.degree(node) == 1:
                mapping[node] = eid
    return mapping


# ── Vectorized batch helpers (operate on all events at once) ──


def _trial_arrival_times(trials: pd.DataFrame) -> np.ndarray:
    """Per-trial destination-arrival timestamps for visit/location history.

    Uses ``goal_arrival_time`` (the moment the animal *arrives at* the
    goal well -- poke-in, which precedes reward delivery) when available,
    falling back to ``end_time`` per row when ``goal_arrival_time`` is NaN,
    and per frame when the column is absent. This clock controls destination,
    visit-recency, and switch history only. Reward-bearing history must use
    :func:`_trial_outcome_times` instead.
    """
    if "goal_arrival_time" in trials.columns:
        cutoff = trials["goal_arrival_time"].to_numpy(dtype=float)
        end = trials["end_time"].to_numpy(dtype=float)
        missing = ~np.isfinite(cutoff)
        return np.where(missing, end, cutoff)
    return trials["end_time"].to_numpy(dtype=float)


def _trial_cutoff_times(trials: pd.DataFrame) -> np.ndarray:
    """Backward-compatible alias for the destination-arrival clock.

    New code should name the intended clock explicitly with
    :func:`_trial_arrival_times` or :func:`_trial_outcome_times`.
    """
    return _trial_arrival_times(trials)


def _trial_outcome_times(trials: pd.DataFrame) -> np.ndarray:
    """Per-trial timestamp at which reward versus omission is observable.

    Newly generated trial tables carry ``outcome_known_time``: first pump onset
    for a rewarded trial and destination poke-out for an omission. Older or
    manually constructed frames lack that field; for those frames ``end_time``
    is an explicit conservative fallback for both outcomes. This fallback can
    delay rewarded history, but it cannot expose reward before delivery.

    Every returned value must be finite and lie within
    ``[destination_arrival, end_time]``. Invalid explicit timestamps raise
    rather than silently falling back to another clock.
    """
    if len(trials) == 0:
        return np.array([], dtype=float)
    if "end_time" not in trials.columns:
        raise KeyError("trial outcome timing requires end_time")

    end = trials["end_time"].to_numpy(dtype=float)
    arrival = _trial_arrival_times(trials)
    reward = None
    if "is_reward" in trials.columns:
        reward = trials["is_reward"].to_numpy(dtype=float)
        if not np.isfinite(reward).all() or not np.isin(reward, (0.0, 1.0)).all():
            raise ValueError("trial is_reward values must be finite binary values")

    reward_column = next(
        (
            column
            for column in ("reward_on_time", "reward_on_ts")
            if column in trials.columns
        ),
        None,
    )
    expected_from_physical_clock = None
    if reward_column is not None:
        if reward is None:
            raise KeyError(
                f"{reward_column} requires is_reward to identify rewarded trials"
            )
        reward_on = trials[reward_column].to_numpy(dtype=float)
        rewarded = reward.astype(bool)
        if (~np.isfinite(reward_on[rewarded])).any():
            raise ValueError("rewarded trials must have finite reward-on timestamps")
        if np.isfinite(reward_on[~rewarded]).any():
            raise ValueError("omission trials cannot carry reward-on timestamps")
        expected_from_physical_clock = np.where(rewarded, reward_on, end)

    if "outcome_known_time" in trials.columns:
        outcome = trials["outcome_known_time"].to_numpy(dtype=float)
        if expected_from_physical_clock is not None and not np.allclose(
            outcome,
            expected_from_physical_clock,
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(
                "outcome_known_time conflicts with reward-on/end-time semantics"
            )
    elif expected_from_physical_clock is not None:
        outcome = expected_from_physical_clock
    else:
        outcome = end.copy()

    invalid = (
        ~np.isfinite(end)
        | ~np.isfinite(arrival)
        | ~np.isfinite(outcome)
        | (outcome < arrival)
        | (outcome > end)
    )
    if invalid.any():
        bad = tuple(np.flatnonzero(invalid)[:10])
        raise ValueError(
            "trial outcome-known times must be finite and lie between "
            f"destination arrival and end_time; invalid row(s): {bad}"
        )
    return outcome


def trial_cutoff_times(trials: pd.DataFrame) -> np.ndarray:
    """Return the canonical destination-arrival cutoff for each trial.

    This public compatibility wrapper is for visit/location history. Reward
    history uses :func:`trial_outcome_times`.
    """
    return _trial_arrival_times(trials)


def trial_outcome_times(trials: pd.DataFrame) -> np.ndarray:
    """Return the canonical reward/omission availability time per trial."""
    return _trial_outcome_times(trials)


_SESSION_COLS = ["animal", "day", "epoch"]


# ── Per-bin event-rate aggregation ──


def attach_is_swr_to_onsets(
    onset_df: pd.DataFrame, event_table: pd.DataFrame
) -> pd.DataFrame:
    """Add a canonical ``is_swr`` column to the non-local onset rows.

    Matches each non-local onset to its ``event_table`` event by within-stop
    time rank (both are ordered by time within a stop and are 1:1 -- the
    non-local onset count equals the event-table row count). SWR/MUA onset
    rows get ``is_swr = NaN``. Returns a copy; does not mutate the input.

    Parameters
    ----------
    onset_df : pd.DataFrame
        Event-onset rows. Must carry ``stop_id``, ``event_type``, and
        ``onset_time``.
    event_table : pd.DataFrame
        Per-event table. Must carry ``stop_id``, ``event_start``, ``is_swr``.

    Returns
    -------
    pd.DataFrame
        Copy of ``onset_df`` with an added ``is_swr`` column (the matched
        boolean for ``event_type == "non_local"`` rows, ``NaN`` elsewhere).
    """
    out = onset_df.copy()
    # object dtype holds the matched booleans alongside NaN for SWR/MUA rows
    # without the float coercion pandas now warns about.
    out["is_swr"] = pd.Series(np.nan, index=out.index, dtype="object")
    # Join on the onset time itself, not on within-stop ordinal rank. A
    # non-local onset IS its event's start, so the two are exactly equal by
    # construction. Ranking instead makes one missing onset shift every later
    # rank in that stop, relabelling events that were present and correct --
    # and because the shifted ranks still all find *a* row, the match-rate
    # guard below stays silent through it. Matching on time contains the
    # failure to the row that is actually missing.
    ev = event_table[["stop_id", "event_start", "is_swr"]].copy()
    nl = out[out["event_type"] == "non_local"].copy()
    merged = nl.merge(
        ev.rename(columns={"event_start": "onset_time"}),
        on=["stop_id", "onset_time"],
        how="left",
        suffixes=("", "_ev"),
        # Duplicate (stop_id, onset_time) keys would expand rows and misalign
        # the positional assignment below.
        validate="one_to_one",
    )
    # A gross mismatch means the onset table and event_table were built from
    # divergent non-local/speed thresholds or decodes -- fail loud rather than
    # silently leave is_swr unlabelled downstream. On the shipped artifacts the
    # time join matches exactly, so this tolerance is slack, not an expected
    # boundary loss.
    matched = merged["is_swr_ev"].notna()
    if len(merged) and matched.mean() < 0.99:
        raise ValueError(
            "attach_is_swr_to_onsets: only "
            f"{matched.mean():.1%} of non-local onsets matched an event_table "
            "row by (stop_id, within-stop rank). The onset table and "
            "event_table must be built from the same non-local/speed thresholds "
            "and decodes."
        )
    out.loc[nl.index, "is_swr"] = merged["is_swr_ev"].to_numpy()
    return out


def raised_cosine_history_basis(n_lags: int, n_basis: int) -> np.ndarray:
    """Log-spaced raised-cosine self-history basis (Pillow et al. 2008).

    Smooth, overlapping bumps over within-stop lags ``1..n_lags`` used to encode
    the self-history block. Projecting the ``onset_lag{j}`` counts onto these
    ``n_basis`` bumps replaces ``n_lags`` collinear lag dummies that otherwise
    perfectly separate on the refractory bins (coefficients -> -inf) and wreck
    held-out prediction.

    Parameters
    ----------
    n_lags : int
        Number of within-stop lags spanned (the history window in bins).
    n_basis : int
        Number of raised-cosine bumps.

    Returns
    -------
    np.ndarray, shape (n_basis, n_lags)
        ``B[k, j]`` is the weight of lag ``j+1`` in basis function ``k``.
    """
    x = np.log(np.arange(1, n_lags + 1))
    centers = np.linspace(x.min(), x.max(), n_basis)
    spacing = centers[1] - centers[0] if n_basis > 1 else 1.0
    B = np.zeros((n_basis, n_lags))
    for k, c in enumerate(centers):
        arg = np.clip((x - c) * np.pi / (2 * spacing), -np.pi, np.pi)
        B[k] = 0.5 * (1 + np.cos(arg))
    return B


def aggregate_to_timebins(
    stop_table: pd.DataFrame,
    onset_times: pd.DataFrame,
    *,
    bin_s: float,
    tmax: float,
    reward_lags: int = 0,
    self_history_lags: int = 0,
    self_history_basis: int = 0,
    cross_onset_times: pd.DataFrame | None = None,
    cross_history_lags: int = 0,
    cross_history_basis: int = 0,
    stop_covariates: tuple[str, ...] = (),
) -> pd.DataFrame:
    """One row per (stop, bin) up to ``tmax``, at ``bin_s`` resolution.

    Builds the canonical (stop x bin) point-process design used by the per-bin
    event-rate models. Bins are built from ``stop_table.immobility_time``
    (continuous), independent of any dense 4 ms grid, so non-4-ms widths (e.g.
    15 ms) are exact. The last bin's ``exposure`` is the partial remainder.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Per-stop table (unique ``stop_id``). Must carry ``animal``, ``day``,
        ``epoch``, ``stop_id``, ``immobility_time``, plus any ``reward_lag{j}``
        and ``stop_covariates`` columns requested.
    onset_times : pd.DataFrame
        Event-onset rows for ONE population (already filtered). Must carry
        ``stop_id`` and ``t_since_stop_onset``.
    bin_s : float
        Bin width in seconds.
    tmax : float
        Right-truncation; bins beyond ``tmax`` are dropped.
    reward_lags : int
        Number of ``reward_lag{j}`` columns to carry from ``stop_table``.
    self_history_lags : int
        Number of ``onset_lag{j}`` self-history columns to build (same-
        population onset counts ``j`` bins back, within stop, reset at the
        stop boundary).
    self_history_basis : int
        If > 0, also add ``hist_rcos_0..self_history_basis-1`` columns: the
        ``onset_lag`` counts projected onto a log-spaced raised-cosine basis
        (see :func:`raised_cosine_history_basis`). Requires ``self_history_lags
        > 0`` (the basis spans that window). Used by the ``*_histcos`` models,
        which generalize where the lag dummies overfit the refractory bins.
    stop_covariates : tuple[str, ...]
        Extra per-stop columns carried from ``stop_table`` by ``stop_id``.

    Returns
    -------
    pd.DataFrame
        Columns: ``animal``, ``day``, ``epoch``, ``stop_id``, ``binj``,
        ``tc`` (bin-center s = time since stop onset), ``exposure`` (s of
        immobility in the bin), ``reward_lag1..reward_lags``, each name in
        ``stop_covariates``, ``count`` (population onsets in the bin),
        ``onset_lag1..self_history_lags``, and (if ``self_history_basis``)
        ``hist_rcos_0..self_history_basis-1``.
    """
    if self_history_basis and not self_history_lags:
        raise ValueError("self_history_basis requires self_history_lags > 0")
    if cross_history_basis and not cross_history_lags:
        raise ValueError("cross_history_basis requires cross_history_lags > 0")
    if cross_history_lags and cross_onset_times is None:
        raise ValueError("cross_history_lags requires cross_onset_times")
    reward_cols = [f"reward_lag{j}" for j in range(1, reward_lags + 1)]
    carry = reward_cols + list(stop_covariates)

    s = stop_table[stop_table["immobility_time"] > 0]
    # ceil, not int: when tmax is not a multiple of bin_s, int() drops the final
    # partial window [floor(tmax/bin_s)*bin_s, tmax) so onsets there are lost.
    nmax = int(np.ceil(tmax / bin_s))
    nbin = np.minimum(np.ceil(s["immobility_time"].to_numpy() / bin_s), nmax).astype(
        int
    )
    idx = np.repeat(np.arange(len(s)), nbin)
    binj = (
        np.concatenate([np.arange(n) for n in nbin])
        if len(s)
        else np.empty(0, dtype=int)
    )
    cols = [*_SESSION_COLS, "stop_id", "immobility_time", *carry]
    out = s.iloc[idx][cols].reset_index(drop=True)
    out["binj"] = binj
    out["tc"] = (binj + 0.5) * bin_s
    # bin width, clipped by both the stop's immobility end and tmax (the final
    # bin of a stop longer than tmax is exposed only up to tmax, not bin_s).
    out["exposure"] = np.minimum(
        bin_s, np.minimum(out["immobility_time"], tmax) - binj * bin_s
    )
    out = out[out["exposure"] > 1e-4]

    src = onset_times[onset_times["t_since_stop_onset"] < tmax]
    # clip into the design's bin range so an onset in the final partial window
    # [(_nmax-1)*bin_s, tmax) lands in the last bin rather than a phantom bin
    # that the merge below would silently drop.
    src_binj = np.clip(
        (src["t_since_stop_onset"].to_numpy() / bin_s).astype(int), 0, nmax - 1
    )
    cnt = (
        pd.DataFrame({"stop_id": src["stop_id"].to_numpy(), "binj": src_binj})
        .groupby(["stop_id", "binj"])
        .size()
        .rename("count")
    )
    out = out.merge(cnt, on=["stop_id", "binj"], how="left")
    out["count"] = out["count"].fillna(0)

    if self_history_lags:
        out = out.sort_values(["stop_id", "binj"])
        grouped = out.groupby("stop_id")["count"]
        for j in range(1, self_history_lags + 1):
            out[f"onset_lag{j}"] = grouped.shift(j).fillna(0)

    if self_history_basis:
        B = raised_cosine_history_basis(self_history_lags, self_history_basis)
        lag_cols = [f"onset_lag{j}" for j in range(1, self_history_lags + 1)]
        proj = out[lag_cols].to_numpy() @ B.T
        for k in range(self_history_basis):
            out[f"hist_rcos_{k}"] = proj[:, k]

    # Cross-population history: lagged onset counts of a different event stream
    # (e.g. with-SWR non-local onsets as predictors of the non-SWR rate). Binned
    # on the SAME grid as the response, shifted within stop (lag >= 1 -> only the
    # cross stream's past predicts the current bin), reset at the stop boundary.
    if cross_history_lags:
        xsrc = cross_onset_times[cross_onset_times["t_since_stop_onset"] < tmax]
        xbinj = np.clip(
            (xsrc["t_since_stop_onset"].to_numpy() / bin_s).astype(int), 0, nmax - 1
        )
        xcnt = (
            pd.DataFrame({"stop_id": xsrc["stop_id"].to_numpy(), "binj": xbinj})
            .groupby(["stop_id", "binj"])
            .size()
            .rename("cross_count")
        )
        out = out.merge(xcnt, on=["stop_id", "binj"], how="left")
        out["cross_count"] = out["cross_count"].fillna(0)
        out = out.sort_values(["stop_id", "binj"])
        xgrouped = out.groupby("stop_id")["cross_count"]
        for j in range(1, cross_history_lags + 1):
            out[f"cross_lag{j}"] = xgrouped.shift(j).fillna(0)
        if cross_history_basis:
            Bx = raised_cosine_history_basis(cross_history_lags, cross_history_basis)
            xlag_cols = [f"cross_lag{j}" for j in range(1, cross_history_lags + 1)]
            xproj = out[xlag_cols].to_numpy() @ Bx.T
            for k in range(cross_history_basis):
                out[f"cross_hist_rcos_{k}"] = xproj[:, k]
        out = out.drop(columns=["cross_count"])

    out = out.drop(columns=["immobility_time"])
    return out.reset_index(drop=True)


# given onset_df already passed through attach_is_swr_to_onsets:
POPULATIONS = {
    "swr": lambda o: o[o.event_type == "SWR"],
    "non_local": lambda o: o[o.event_type == "non_local"],
    "non_local_swr": lambda o: o[
        (o.event_type == "non_local") & (o.is_swr == True)  # noqa: E712
    ],
    "non_local_no_swr": lambda o: o[
        (o.event_type == "non_local") & (o.is_swr == False)  # noqa: E712
    ],
}


# ── Scalar per-event wrappers (preserve existing API) ──


# Behavioral covariate definitions were extracted to covariates.py; they are
# re-exported here so existing ``from ..._helpers import <name>`` call sites keep
# resolving. covariates.py imports the explicit arrival/outcome clocks and
# build_well_to_arm_map from this module, so _helpers must finish loading first
# (it does: the package __init__ imports _helpers before covariates).
from .covariates import (  # noqa: E402,F401
    UNCERTAINTY_COVARIATES,
    _leaky_q_value_diff,
    _omission_streak,
    add_uncertainty_covariates,
    attach_arm_uncertainty,
    attach_contingency_value_block,
    cumulative_reward_all_events,
    cumulative_reward_per_arm,
    lagged_trial_rewards,
    recent_reward_rate_all_events,
    recent_reward_rate_per_arm,
    recent_trial_reward_rate,
    visit_recency_all_events,
    visit_recency_per_arm,
)
