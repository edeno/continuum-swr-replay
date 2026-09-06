"""Time-local behavioral location and completed-trial context.

This module owns the conversion from canonical trials to time-indexed context.
Arrival-derived and outcome-derived fields use the explicit clocks in
:mod:`.clocks`. Movement/switch context uses movement trials; visit, reward,
and local-value histories use the complete starting-port-plus-destinations
physical-poke experience stream. Epoch boundaries establish neither occupancy
nor outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING

from .clocks import trial_arrival_times, trial_outcome_times
from .history import (
    canonical_port_value_history,
    canonical_trial_switch_history,
)
from .history._common import _prepare_port_experiences

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
    outcome clock returned by :func:`trial_outcome_times`.
    """
    if trials.empty:
        return trials.copy(), np.array([], dtype=float)

    base_columns = {"start_time", "end_time", "to_well", *required_columns}
    missing = sorted(base_columns.difference(trials.columns))
    if missing:
        raise KeyError(f"trial context requires columns {missing}")

    prepared = trials.copy()
    departure_column = (
        "movement_start_time"
        if "movement_start_time" in prepared.columns
        else "start_time"
    )
    starts = prepared[departure_column].to_numpy(dtype=float)
    ends = prepared["end_time"].to_numpy(dtype=float)
    cutoffs = trial_arrival_times(prepared)
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


def _validate_trial_experience_alignment(
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Validate the one initial occupancy plus destination experience stream.

    This is the single cross-source boundary between movement trials and port
    experiences. Internal validity of either frame is insufficient: a stale or
    shifted experience stream can remain perfectly ordered while assigning a
    plausible reward history to the wrong port.
    """
    if trials.empty:
        experiences = _prepare_port_experiences(port_experiences)
        if not experiences.empty:
            raise ValueError(
                "port experiences must be empty when there are no movement trials"
            )
        prepared, arrivals = _prepare_trials_for_context(trials)
        return prepared, arrivals, experiences

    required = (
        "from_well",
        "is_reward",
        "is_patch_change",
    )
    raw_arrivals = trial_arrival_times(trials)
    order = np.argsort(raw_arrivals, kind="stable")
    trial_ids = pd.to_numeric(pd.Index(trials.index), errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(trial_ids).all()
        or not np.equal(trial_ids, np.floor(trial_ids)).all()
        or len(np.unique(trial_ids)) != len(trial_ids)
    ):
        raise ValueError("movement trial index must contain unique integer IDs")
    ordered_trial_ids = trial_ids[order]

    prepared, arrivals = _prepare_trials_for_context(
        trials,
        required_columns=required,
    )
    experiences = _prepare_port_experiences(port_experiences)
    n_trials = len(prepared)
    if len(experiences) != n_trials + 1:
        raise ValueError(
            "port-experience stream must contain the initial occupancy plus one "
            f"destination per movement trial; got {len(experiences)} for "
            f"{n_trials} trials"
        )

    initial = experiences.iloc[0]
    destinations = experiences.iloc[1:].reset_index(drop=True)
    from_wells = prepared["from_well"].to_numpy(dtype=float)
    to_wells = prepared["to_well"].to_numpy(dtype=float)
    _validate_well_values(from_wells, "from_well")
    if float(initial["port"]) != from_wells[0]:
        raise ValueError("initial port experience does not match first trial origin")
    if n_trials > 1 and not np.array_equal(to_wells[:-1], from_wells[1:]):
        raise ValueError("movement trials must form a continuous physical-poke chain")

    movement_ids = destinations["movement_trial_number"].to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.array_equal(movement_ids, ordered_trial_ids):
        raise ValueError(
            "destination experience movement trial IDs do not match chronological "
            "movement trials"
        )
    if not np.array_equal(destinations["port"].to_numpy(dtype=float), to_wells):
        raise ValueError("port experiences do not align with trial destinations")
    experience_ports = experiences["port"].to_numpy(dtype=float)
    if not np.array_equal(experience_ports[:-1], from_wells):
        raise ValueError(
            "each movement origin must equal its preceding port experience"
        )
    experience_patch_change = (
        experiences["patch"].to_numpy(dtype=float)[1:]
        != experiences["patch"].to_numpy(dtype=float)[:-1]
    )
    trial_patch_change = prepared["is_patch_change"].to_numpy(dtype=bool)
    if not np.array_equal(experience_patch_change, trial_patch_change):
        raise ValueError(
            "trial patch-switch labels disagree with the physical-poke sequence"
        )

    departure_column = (
        "movement_start_time"
        if "movement_start_time" in prepared.columns
        else "start_time"
    )
    departure_times = prepared[departure_column].to_numpy(dtype=float)
    if not np.isclose(
        float(initial["departure_time"]),
        departure_times[0],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "initial port departure does not match first movement departure"
        )
    if not np.allclose(
        destinations["departure_time"].to_numpy(dtype=float),
        prepared["end_time"].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("destination departures do not align with movement trials")
    if not np.allclose(
        destinations["arrival_time"].to_numpy(dtype=float),
        arrivals,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("port-experience arrivals do not align with movement trials")

    destination_observed = destinations["outcome_observed"].to_numpy(dtype=bool)
    reward_values = prepared["is_reward"].to_numpy(dtype=float)
    destination_rewards = destinations["is_reward"].to_numpy(
        dtype=float, na_value=np.nan
    )
    if (
        not np.array_equal(
            destination_rewards[destination_observed],
            reward_values[destination_observed],
        )
        or np.isfinite(destination_rewards[~destination_observed]).any()
    ):
        raise ValueError("port-experience rewards do not align with trial outcomes")
    expected_outcomes = trial_outcome_times(prepared)
    expected_reward_on = (
        prepared["reward_on_time"].to_numpy(dtype=float)
        if "reward_on_time" in prepared
        else np.where(reward_values.astype(bool), expected_outcomes, np.nan)
    )
    for column, expected in (
        ("reward_on_time", expected_reward_on),
        ("outcome_known_time", expected_outcomes),
    ):
        got = destinations[column].to_numpy(dtype=float)
        got_observed = got[destination_observed]
        expected_observed = expected[destination_observed]
        finite_expected = np.isfinite(expected_observed)
        if (
            np.isfinite(got[~destination_observed]).any()
            or not np.array_equal(np.isnan(got_observed), np.isnan(expected_observed))
            or not np.allclose(
                got_observed[finite_expected],
                expected_observed[finite_expected],
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise ValueError(
                f"port-experience {column} does not align with movement trials"
            )

    return prepared, arrivals, experiences


def behavioral_location_from_port_experiences(
    times: np.ndarray,
    port_experiences: pd.DataFrame,
) -> pd.DataFrame:
    """Classify timestamps from explicit port-dwell intervals.

    Each canonical task experience supplies ``[arrival_time, departure_time)``
    from its first poke-in through its final poke-out. Consecutive same-port
    re-pokes are one experience, matching the Comrie task parser, so a brief
    gap between those physical photogate bouts remains the same task-port
    context even though the beam is not literally broken. Use
    ``photogate_engagement_at_times`` for the narrower hardware state. Gaps
    between departure from one port and arrival at a different port are
    transit. Consequently exact arrival is at-well and exact departure is
    in-transit. Epoch-boundary coordinates are absent by contract and cannot
    establish occupancy.

    This builder-local helper returns ``location_state`` (categorical),
    ``at_well`` (nullable boolean), and ``location_well`` (nullable
    ``Float64``). Persisted tables rename the state to ``occupancy_state`` and
    the well-derived arm to ``occupancy_well_arm``.

    **These are task states, not positions, and the names undersell that.**
    Two different questions are answered by two similarly-named columns:

    * builder-local ``location_well`` / ``location_arm`` -- *which well is the
      animal visiting?* The persisted arm is ``occupancy_well_arm``. It is
      derived from trip intervals, not from tracked position, and is **NaN
      during a trip**: the destination is never named before the animal reaches
      it, which is what keeps it usable as a predictor.
    * ``physical_arm`` (built in ``tables_event.py``) -- *which track segment does the
      tracked position fall on?* Always defined, trunks included.

    They are allowed to disagree, and do on 46 of 59,374 at-well events (0.077%,
    measured 2026-08-19): the animal is still inside its well-visit interval
    while its tracked position is a mean 48.4 cm away (max 82.3 cm), usually on
    the adjoining trunk. Both labels are right about their own construct.
    Neither is a fallback for the other, so a cohort that needs occupancy
    off-well must say which one it means rather than defaulting.

    ``"in_transit"`` names the **trip interval, not the animal's motion**, and
    this reliably misleads. Every event in the modelled cohort is speed-gated
    immobility; off-well events have a median speed of 1.35 cm/s. An
    ``in_transit`` event is an animal *standing still partway through a trip*,
    not a running animal. The reason the content models exclude those events is
    narrow -- the well-visit construct is undefined there, so ``is_current`` and
    everything derived from it is NaN -- and not that the animal was moving.
    """
    sample_times = np.asarray(times, dtype=float)
    if sample_times.ndim != 1:
        raise ValueError("times must be a one-dimensional sequence")
    if not np.isfinite(sample_times).all():
        raise ValueError("times must contain only finite timestamps")

    index = pd.Index(sample_times, name="time")
    experiences = _prepare_port_experiences(port_experiences)
    if experiences.empty:
        return pd.DataFrame(
            {
                "occupancy_state": pd.Categorical(
                    ["unknown"] * len(sample_times), categories=_LOCATION_STATES
                ),
                "at_well": pd.array([pd.NA] * len(sample_times), dtype="boolean"),
                "occupancy_well_id": pd.array(
                    [pd.NA] * len(sample_times), dtype="Float64"
                ),
            },
            index=index,
        )

    arrivals = experiences["arrival_time"].to_numpy(dtype=float)
    departures = experiences["departure_time"].to_numpy(dtype=float)
    ports = experiences["port"].to_numpy(dtype=float)

    arrived_idx = np.searchsorted(arrivals, sample_times, side="right") - 1
    has_arrived = arrived_idx >= 0
    clipped = np.clip(arrived_idx, 0, len(experiences) - 1)
    is_at_well = has_arrived & (sample_times < departures[clipped])
    # Once a recorded dwell ends, the animal is off-well. Keeping the final
    # post-departure state as transit makes earlier labels invariant when a
    # later experience is appended.
    is_between_visits = has_arrived & ~is_at_well

    state = np.full(len(sample_times), "unknown", dtype=object)
    state[is_between_visits] = "in_transit"
    state[is_at_well] = "at_well"
    at_well = pd.array([pd.NA] * len(sample_times), dtype="boolean")
    at_well[is_between_visits] = False
    at_well[is_at_well] = True
    location_well = pd.array([pd.NA] * len(sample_times), dtype="Float64")
    location_well[is_at_well] = ports[clipped[is_at_well]]

    return pd.DataFrame(
        {
            "occupancy_state": pd.Categorical(state, categories=_LOCATION_STATES),
            "at_well": at_well,
            "occupancy_well_id": location_well,
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
        required = {"occupancy_state", "occupancy_well_id"}
        missing = sorted(required.difference(states.columns))
        if missing:
            raise KeyError(f"location frame requires columns {missing}")
        wells = states["occupancy_well_id"]
        states = states["occupancy_state"]

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
            "occupancy_state": "unknown",
            "at_well": pd.NA,
            "occupancy_well_id": np.nan,
        }

    allowed = set(_LOCATION_STATES) - {"mixed"}
    observed_states = set(state_values.tolist())
    invalid = observed_states.difference(allowed)
    if invalid:
        raise ValueError(f"invalid time-resolved location states: {sorted(invalid)}")

    if observed_states == {"unknown"}:
        return {
            "occupancy_state": "unknown",
            "at_well": pd.NA,
            "occupancy_well_id": np.nan,
        }
    if observed_states == {"in_transit"}:
        return {
            "occupancy_state": "in_transit",
            "at_well": False,
            "occupancy_well_id": np.nan,
        }
    if observed_states == {"at_well"}:
        unique_wells = np.unique(well_values[np.isfinite(well_values)])
        if len(unique_wells) == 1 and np.isfinite(well_values).all():
            return {
                "occupancy_state": "at_well",
                "at_well": True,
                "occupancy_well_id": float(unique_wells[0]),
            }

    return {
        "occupancy_state": "mixed",
        "at_well": False,
        "occupancy_well_id": np.nan,
    }


def add_trial_context(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
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
    - ``prev_well``, ``current_well``, ``last_visited_well`` (internal,
      past-only well identities; persisted occupancy uses canonical names)
    - ``next_well`` (forward-looking: the not-yet-reached destination)
    - ``last_rewarded_well`` — advances only when a reward is observable
    - ``completed_trial_id`` — zero-based completion-order ID regenerated
      within the session (not an index copied from the source trial table)
    - ``patch_switch_fraction_last5_completed_trips`` — canonical rolling
      switch fraction through the identified trial, computed from the full
      trial stream
    - ``leaky_q_value_diff`` — canonical alpha-0.3 pre-update Q margin for the
      identified destination, computed after every earlier observed physical
      port outcome
    - ``did_last_completed_trip_switch_patch`` — last completed trip's
      ``is_patch_change``
    - ``is_rewarded_outcome`` — current arrived trip's outcome, NaN until known
    - ``completed_trial_reward`` — the same observed outcome, ungated
      (post-outcome; descriptive constructs only, never a predictor)
    - internal location fields, finalized at the persisted-table boundary as
      ``occupancy_state``, ``at_well``, and ``occupancy_well_id``
    """
    dot_times = np.asarray(dot.index.values, dtype=float)
    if dot_times.ndim != 1 or not np.isfinite(dot_times).all():
        raise ValueError("dot index must contain only finite timestamps")

    trials, arrival_times, experiences = _validate_trial_experience_alignment(
        trials, port_experiences
    )
    location = behavioral_location_from_port_experiences(dot_times, experiences)
    for column in location.columns:
        dot[column] = location[column].array

    if trials.empty:
        for col in (
            "prev_well",
            "current_well",
            "last_visited_well",
            "next_well",
            "last_rewarded_well",
            "is_rewarded_outcome",
            "completed_trial_reward",
            "did_last_completed_trip_switch_patch",
            "_goal_arrival_time",
            "_outcome_known_time",
            "_last_reward_time",
            "patch_switch_fraction_last5_completed_trips",
            "leaky_q_value_diff",
        ):
            dot[col] = np.nan
        dot["completed_trial_id"] = pd.array([pd.NA] * len(dot), dtype="Int64")
        return dot

    n_trials = len(trials)
    departure_column = (
        "movement_start_time" if "movement_start_time" in trials else "start_time"
    )
    trial_start_times = trials[departure_column].to_numpy(dtype=float)
    to_wells = trials["to_well"].to_numpy(dtype=float)
    from_wells = trials["from_well"].to_numpy(dtype=float)
    destination_experiences = experiences.iloc[1:].reset_index(drop=True)
    outcome_observed = destination_experiences["outcome_observed"].to_numpy(dtype=bool)
    outcome_times = destination_experiences["outcome_known_time"].to_numpy(dtype=float)
    reward_values = destination_experiences["is_reward"].to_numpy(
        dtype=float, na_value=np.nan
    )
    patch_change_values = trials["is_patch_change"].to_numpy(dtype=float)
    _validate_well_values(from_wells, "from_well")
    if (
        not np.isfinite(reward_values[outcome_observed]).all()
        or not np.isin(reward_values[outcome_observed], (0.0, 1.0)).all()
        or np.isfinite(reward_values[~outcome_observed]).any()
    ):
        raise ValueError(
            "observed movement outcomes must be binary and unobserved outcomes "
            "must be missing"
        )
    if (
        not np.isfinite(patch_change_values).all()
        or not np.isin(patch_change_values, (0.0, 1.0)).all()
    ):
        raise ValueError("trial is_patch_change values must be finite binary values")
    is_patch_change = patch_change_values.astype(bool)
    if np.any(np.diff(outcome_times[outcome_observed]) < 0):
        raise ValueError("trial outcome-known times must be nondecreasing")
    if n_trials > 1 and np.any(
        outcome_observed[:-1] & (outcome_times[:-1] > arrival_times[1:])
    ):
        raise ValueError(
            "a trial outcome remains unknown after the next destination arrival"
        )

    trial_switch_rate = canonical_trial_switch_history(patch_change_values)
    trial_q_margin = canonical_port_value_history(experiences)[1:]

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
    dot["patch_switch_fraction_last5_completed_trips"] = np.where(
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
    # newly generated rewarded trials, the canonical latency/departure boundary
    # for omissions, or the explicit
    # conservative end-time fallback documented by trial_outcome_times.
    experience_reward = experiences["outcome_observed"].to_numpy(
        dtype=bool
    ) & experiences["is_reward"].fillna(False).to_numpy(dtype=bool)
    reward_timestamps = experiences.loc[
        experience_reward, "outcome_known_time"
    ].to_numpy(dtype=float)
    rewarded_wells = experiences.loc[experience_reward, "port"].to_numpy(dtype=float)
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
    # Companion clock to the arrival time: when the identified trip's outcome
    # became physically observable (pump onset when rewarded, the canonical
    # latency/poke-out boundary when omitted). This is the *boundary* itself,
    # not an outcome-gated value, so it is carried from arrival onward -- a
    # timebin before the boundary has to know where the boundary lies. NaN
    # before the first completed arrival. The ``outcome_observed`` conjunct is
    # defensive only: ``_prepare_port_experiences`` refuses an unobserved
    # experience outright, so the canonical derivation cannot currently reach
    # it.
    dot["_outcome_known_time"] = np.where(
        has_completed & outcome_observed[completed_clipped],
        outcome_times[completed_clipped],
        np.nan,
    )

    if len(reward_timestamps) > 0:
        dot["_last_reward_time"] = np.where(
            rew_idx >= 0, reward_timestamps[rew_idx_clipped], np.nan
        )
    else:
        dot["_last_reward_time"] = np.nan

    current_outcome_known = (
        has_completed
        & outcome_observed[completed_clipped]
        & (dot_times >= outcome_times[completed_clipped])
    )
    current_reward = np.where(
        current_outcome_known,
        reward_values[completed_clipped],
        np.nan,
    )
    # One canonical outcome column: 1.0 rewarded, 0.0 omitted, NaN until the
    # outcome is observable. The omission complement (is_omitted_outcome) is
    # derived downstream from this, so it is not stored separately.
    dot["is_rewarded_outcome"] = current_reward
    # The completed trip's realized outcome, NOT gated on when it became
    # observable. The gated columns above are the ones a predictor may use;
    # this one exists for constructs that are *about* the outcome the animal
    # just experienced and are already barred from predictive use for another
    # reason. ``wsls_deviation`` is the case: it pairs this outcome with
    # ``is_stay``, the eventual outbound choice, so it is future-referencing
    # whatever the outcome clock does, and gating it at the bout start only
    # made it undefined at every at-well stop.
    dot["completed_trial_reward"] = np.where(
        has_completed, reward_values[completed_clipped], np.nan
    )
    dot["did_last_completed_trip_switch_patch"] = np.where(
        has_completed, is_patch_change[completed_clipped].astype(float), np.nan
    )

    return dot


def trial_context_at_times(
    times: np.ndarray,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
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
    port_experiences : pandas.DataFrame
        Canonical starting-port-plus-destinations stream accepted by
        :func:`add_trial_context`.

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
    return add_trial_context(sampled, trials, port_experiences)
