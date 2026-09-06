"""Machine-readable scientific definitions for model-facing covariates.

The naming registry records how legacy columns became canonical columns.  This
catalog records what the canonical quantities *mean*.  It is intentionally
limited to scientific predictors/labels rather than every diagnostic column.
Model and timing tests use it so a fitted covariate cannot exist only as an
unexplained string in a formula.

``producer`` names the code that *emits the column onto the model table*. For
behavioral-model value covariates (the ``reward_belief_*`` /
``task_reward_probability_*`` family) that emitter is a *join*: the quantity is
fitted upstream by the Beta-Bernoulli behavioral model in
``analysis/behavior/qvals_table.py`` (persisted as ``qvals_table``) and merged
onto event-arm rows here. Their ``producer`` names both altitudes so a reader is
not sent to the joiner expecting the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import pandas as pd

from ..model_specs import (
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)


class Availability(StrEnum):
    """Earliest inferentially safe availability class."""

    PRE_OUTCOME = "pre_outcome"
    POST_CHOICE = "post_choice_pre_outcome"
    POST_OUTCOME = "post_outcome"
    FUTURE_REFERENCING = "future_referencing"


class QueryClock(StrEnum):
    """Physical or logical clock at which a value is evaluated."""

    EVENT_START = "event_start"
    STOP_START = "stop_start"
    GOAL_ARRIVAL = "goal_arrival"
    OUTCOME_AVAILABILITY = "outcome_availability"
    MODEL_PREOUTCOME_STATE = "behavioral_model_preoutcome_state"
    EVENT_RESPONSE = "event_response"
    FUTURE_CHOICE = "future_choice"


@dataclass(frozen=True, slots=True)
class CovariateDefinition:
    """Scientific contract for one canonical model-facing quantity."""

    name: str
    description: str
    grain: str
    units: str
    query_clock: QueryClock
    availability: Availability
    support: str
    window: str | None
    missingness: str
    structural_zero: str | None
    producer: str


def _definition(
    name: str,
    description: str,
    *,
    grain: str,
    units: str,
    clock: QueryClock,
    availability: Availability,
    support: str,
    producer: str,
    window: str | None = None,
    missingness: str = "unavailable at the declared query clock",
    structural_zero: str | None = None,
) -> CovariateDefinition:
    return CovariateDefinition(
        name=name,
        description=description,
        grain=grain,
        units=units,
        query_clock=clock,
        availability=availability,
        support=support,
        window=window,
        missingness=missingness,
        structural_zero=structural_zero,
        producer=producer,
    )


_EVENT_ARM = "event × candidate_arm"
_STOP = "immobility stop"
_TRIAL_STOP = "behavioral-model trial broadcast at stop start"
# Absolute clock readings, not durations: only differences between them are
# interpretable, so they must never be standardized or entered as a predictor.
# The units string is load-bearing -- ``covariate_timing`` derives its
# absolute-clock deny set from it, so tagging a column with it is what keeps the
# raw clock out of every model.
ABSOLUTE_SECONDS_UNITS = "seconds on the absolute session clock (differences only)"

# These quantities use a pre-outcome model state, but selecting one port needs
# the realized destination at goal arrival.  The enforcement registry currently
# places them in its more conservative POST_OUTCOME gate because fitted callers
# do not expose a separate post-choice gate; the scientific catalog preserves
# their true earliest availability.
POST_CHOICE_COVARIATES = frozenset(
    {
        "reward_belief_mean_chosen_arm_preoutcome",
        "task_reward_probability_chosen_arm",
    }
)

_DEFINITIONS = [
    _definition(
        "is_current_arm",
        "Candidate arm is the single well arm occupied throughout the event.",
        grain=_EVENT_ARM,
        units="binary indicator",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="all nine candidate arms",
        producer="covariates.candidate_arm.attach_candidate_arm_covariates",
        missingness="event is not wholly at one well",
    ),
    _definition(
        "is_prev_arm",
        "Candidate arm is the origin arm of the most recently completed trip.",
        grain=_EVENT_ARM,
        units="binary indicator",
        clock=QueryClock.GOAL_ARRIVAL,
        availability=Availability.PRE_OUTCOME,
        support="all nine candidate arms",
        producer="covariates.candidate_arm.attach_candidate_arm_covariates",
        missingness="no completed trip; encoded as all-zero candidate indicators",
    ),
    _definition(
        "is_last_rewarded_arm",
        "Candidate arm is the most recent arm with an outcome-known reward.",
        grain=_EVENT_ARM,
        units="binary indicator",
        clock=QueryClock.OUTCOME_AVAILABILITY,
        availability=Availability.PRE_OUTCOME,
        support="all nine candidate arms",
        producer="covariates.candidate_arm.attach_candidate_arm_covariates",
        missingness="no known reward; encoded as all-zero candidate indicators",
    ),
    _definition(
        "is_last_rewarded_noncurrent_arm",
        "Last-rewarded-arm indicator restricted to a noncurrent arm.",
        grain=_EVENT_ARM,
        units="binary indicator",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="all nine candidate arms",
        producer="covariates.candidate_arm.attach_candidate_arm_covariates",
        missingness="current occupancy is unresolved",
    ),
    _definition(
        "is_same_patch_as_current",
        "Candidate arm belongs to the patch occupied throughout the event.",
        grain=_EVENT_ARM,
        units="binary indicator",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="all nine candidate arms",
        producer="covariates.candidate_arm.attach_candidate_arm_covariates",
        missingness="event is not wholly at one well",
    ),
    _definition(
        "distance_to_candidate_arm_cm",
        "Shortest graph distance from the occupied well node to either endpoint of the candidate segment.",
        grain=_EVENT_ARM,
        units="cm",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="all nine candidate arms",
        producer="covariates.candidate_arm._distance_and_rank_by_event",
        missingness="event is not wholly at one well",
    ),
    _definition(
        "candidate_arm_distance_rank",
        "Rank of candidate-arm graph distance among all nine task arms.",
        grain=_EVENT_ARM,
        units="ordinal rank",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="all nine candidate arms",
        producer="covariates.candidate_arm._distance_and_rank_by_event",
        missingness="event is not wholly at one well",
    ),
    _definition(
        "visit_recency_trials",
        "Arrived port-experience count since the candidate reward-site arm was last visited, including the starting port.",
        grain=_EVENT_ARM,
        units="port experiences",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="six leaf arms; trunks are off-support",
        producer="covariates.history.visit_recency_and_never_visited_all_events",
        window="all arrivals through event start",
        missingness="query precedes the observed starting-port occupancy",
        structural_zero="trunk arms",
    ),
    _definition(
        "is_candidate_arm_never_visited",
        "Candidate reward-site arm has not been visited by event start (the "
        "starting-port experience counts as a visit).",
        grain=_EVENT_ARM,
        units="binary indicator",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="six leaf arms; trunks are off-support",
        producer="covariates.history.visit_recency_and_never_visited_all_events",
        window="all arrivals through event start",
        missingness="query precedes the observed starting-port occupancy",
        structural_zero="trunk arms",
    ),
    _definition(
        "reward_count_candidate_arm_last10_known_trials",
        "Rewards delivered on the candidate arm among the last ten outcome-known port experiences overall, including the starting port only when its outcome was observed.",
        grain=_EVENT_ARM,
        units="reward count",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="six leaf arms; trunks are off-support",
        producer="covariates.history.rolling_reward_count_all_events",
        window="last 10 outcome-known port experiences",
        missingness="no outcome-known port experience yet",
        structural_zero="trunk arms after history begins",
    ),
    _definition(
        "reward_fraction_candidate_arm_last10_visits",
        "Fraction rewarded among the last ten outcome-known visits to the "
        "candidate arm, including the starting port only when its outcome was observed.",
        grain=_EVENT_ARM,
        units="fraction",
        clock=QueryClock.EVENT_START,
        availability=Availability.PRE_OUTCOME,
        support="six leaf arms; trunks are off-support",
        producer="covariates.history.recent_reward_rate_all_events",
        window="last 10 visits to that arm with known outcomes",
        missingness="no outcome-known port experience yet",
        structural_zero="trunks and never-visited leaves after history begins",
    ),
    _definition(
        "task_reward_probability_candidate_arm",
        "Scheduled task reward probability for the candidate reward-site arm.",
        grain=_EVENT_ARM,
        units="probability",
        clock=QueryClock.STOP_START,
        availability=Availability.PRE_OUTCOME,
        support="six leaf arms; trunks are off-support",
        producer="analysis.behavior.qvals_table (Beta-Bernoulli fit) -> joined by covariates.history.attach_contingency_value_block",
        missingness="no qvals row for the stop",
        structural_zero="trunk arms when the qvals row is present",
    ),
    _definition(
        "reward_belief_mean_candidate_arm_preoutcome",
        "Pre-outcome Beta-model mean reward belief for the candidate arm.",
        grain=_EVENT_ARM,
        units="probability",
        clock=QueryClock.MODEL_PREOUTCOME_STATE,
        availability=Availability.PRE_OUTCOME,
        support="six leaf arms; trunks are off-support",
        producer="analysis.behavior.qvals_table (Beta-Bernoulli fit) -> joined by covariates.history.attach_contingency_value_block",
        missingness="no qvals row for the stop",
        structural_zero="trunk arms when the qvals row is present",
    ),
    _definition(
        "reward_belief_variance_candidate_arm_preoutcome",
        "Pre-outcome Beta-model reward-belief variance for the candidate arm.",
        grain=_EVENT_ARM,
        units="probability squared",
        clock=QueryClock.MODEL_PREOUTCOME_STATE,
        availability=Availability.PRE_OUTCOME,
        support="six leaf arms; trunks are off-support",
        producer="analysis.behavior.qvals_table (Beta-Bernoulli fit) -> joined by covariates.history.attach_arm_uncertainty",
        missingness="no qvals row for the stop",
        structural_zero="trunk arms when the qvals row is present",
    ),
    _definition(
        "reward_fraction_last10_known_trials",
        "Fraction rewarded among the ten most recent outcome-known port experiences, including the starting port only when its outcome was observed.",
        grain=_STOP,
        units="fraction",
        clock=QueryClock.STOP_START,
        availability=Availability.PRE_OUTCOME,
        support="all stops",
        producer="covariates.history.recent_trial_reward_rate",
        window="last 10 outcome-known port experiences",
        missingness="no outcome-known port experience yet",
    ),
    _definition(
        "patch_switch_fraction_last5_completed_trips",
        "Fraction of the five most recently arrived trips that changed patch, including the latest arrival.",
        grain=_STOP,
        units="fraction",
        clock=QueryClock.GOAL_ARRIVAL,
        availability=Availability.PRE_OUTCOME,
        support="stops after at least one completed arrival",
        producer="covariates.history.canonical_trial_switch_history",
        window="last 5 completed arrivals",
        missingness="no completed arrival yet",
    ),
    _definition(
        "leaky_q_value_diff",
        "Pre-update difference between the largest and second-largest fixed-alpha local action values.",
        grain=_STOP,
        units="reward probability difference",
        clock=QueryClock.GOAL_ARRIVAL,
        availability=Availability.PRE_OUTCOME,
        support="stops after at least one completed arrival",
        producer="covariates.history.canonical_port_value_history",
        window="initial-port outcome and earlier destination outcomes; alpha=0.3",
        missingness="no completed arrival yet",
    ),
    _definition(
        "last_well_arrival_time_s",
        "Absolute session time at which the animal arrived at the well occupied at the start of the stop.",
        grain=_STOP,
        units=ABSOLUTE_SECONDS_UNITS,
        clock=QueryClock.GOAL_ARRIVAL,
        availability=Availability.PRE_OUTCOME,
        support="stops after at least one completed arrival",
        producer="covariates.context.add_trial_context -> replay.tables_stop.build_stop_table",
        missingness="no completed arrival at the bout start",
    ),
    _definition(
        "outcome_known_time_s",
        "Absolute session time at which the outcome of that arrival became physically observable: pump onset when rewarded, the canonical latency/poke-out boundary when omitted.",
        grain=_STOP,
        units=ABSOLUTE_SECONDS_UNITS,
        clock=QueryClock.OUTCOME_AVAILABILITY,
        availability=Availability.POST_OUTCOME,
        support="stops after at least one completed arrival",
        producer="covariates.context.add_trial_context -> replay.tables_stop.build_stop_table",
        missingness="no completed arrival at the bout start",
    ),
    _definition(
        "omission_streak",
        "Number of consecutive omissions in the physically known reward-lag vector.",
        grain=_STOP,
        units="trial count",
        clock=QueryClock.STOP_START,
        availability=Availability.PRE_OUTCOME,
        support="all stops with known lag 1",
        producer="covariates.history._omission_streak",
        window="known_reward_outcome_lag_1 through lag_10",
        missingness="lag 1 is unavailable",
    ),
    _definition(
        "rpe_postoutcome",
        "Realized reward minus the chosen-arm pre-outcome Beta belief mean.",
        grain=_TRIAL_STOP,
        units="reward probability difference",
        clock=QueryClock.OUTCOME_AVAILABILITY,
        availability=Availability.POST_OUTCOME,
        support="qvals-mapped stops",
        producer="analysis.behavior.qvals_table.add_experienced_value_columns",
        missingness="behavioral model or current outcome unavailable",
    ),
    _definition(
        "reward_belief_mean_update_l1_all_ports_postoutcome",
        "L1 magnitude of the current trial's Beta-belief mean update over all six ports.",
        grain=_TRIAL_STOP,
        units="summed absolute probability change",
        clock=QueryClock.OUTCOME_AVAILABILITY,
        availability=Availability.POST_OUTCOME,
        support="qvals-mapped stops",
        producer="analysis.behavior.qvals_table canonical ingestion adapter",
        missingness="behavioral model or current outcome unavailable",
    ),
    _definition(
        "relative_value_switch_minus_stay_preoutcome",
        "Pre-outcome switch value minus stay value for the current trial.",
        grain=_TRIAL_STOP,
        units="reward-belief value difference",
        clock=QueryClock.MODEL_PREOUTCOME_STATE,
        availability=Availability.PRE_OUTCOME,
        support="qvals-mapped stops",
        producer="analysis.behavior._spatial_bandit_rl via qvals adapter",
        missingness="behavioral model unavailable",
    ),
    _definition(
        "reward_belief_differential_entropy_leaf_mean_preoutcome",
        "Mean differential entropy of the six pre-outcome Beta reward beliefs.",
        grain=_TRIAL_STOP,
        units="nats",
        clock=QueryClock.MODEL_PREOUTCOME_STATE,
        availability=Availability.PRE_OUTCOME,
        support="qvals-mapped stops",
        producer="analysis.behavior.qvals_table.add_beta_belief_entropy",
        missingness="behavioral model unavailable",
    ),
    _definition(
        "periswitch_index_trials",
        "Signed distance to the nearest patch switch, including future switches.",
        grain=_TRIAL_STOP,
        units="trials",
        clock=QueryClock.FUTURE_CHOICE,
        availability=Availability.FUTURE_REFERENCING,
        support="sessions with at least one patch switch",
        producer="analysis.behavior.qvals_table.peri_switch_index",
        missingness="session contains no patch switch",
    ),
    _definition(
        "is_periswitch_post",
        "Nearest patch switch lies at or before the indexed trial.",
        grain=_TRIAL_STOP,
        units="binary indicator",
        clock=QueryClock.FUTURE_CHOICE,
        availability=Availability.FUTURE_REFERENCING,
        support="sessions with at least one patch switch",
        producer="analysis.behavior.qvals_table.peri_switch_index",
        missingness="session contains no patch switch",
    ),
    _definition(
        "is_stay",
        "The eventual outbound choice remains within the currently occupied patch.",
        grain=_STOP,
        units="binary indicator",
        clock=QueryClock.FUTURE_CHOICE,
        availability=Availability.FUTURE_REFERENCING,
        support="stops with an observed next choice",
        producer="replay.tables.build_stop_table (stop grain); also computed per "
        "event window in build_event_table (event_table.is_stay), which "
        "has wider coverage than the joined stop value",
        missingness="next choice unavailable",
    ),
    _definition(
        "wsls_deviation",
        "One for win-shift or lose-stay and zero for win-stay or lose-shift.",
        grain=_STOP,
        units="binary indicator",
        clock=QueryClock.FUTURE_CHOICE,
        availability=Availability.FUTURE_REFERENCING,
        support="stops with a completed outcome and observed next choice",
        producer="covariates.history.add_uncertainty_covariates",
        window="most recently completed trip plus its future outbound choice",
        missingness="completed outcome or next choice unavailable",
    ),
]

_DEFINITIONS.extend(
    [
        _definition(
            "visit_recency_trials_event_leaf_centered",
            "Candidate-arm visit recency minus the event-local mean over visited leaf arms.",
            grain=_EVENT_ARM,
            units="trials",
            clock=QueryClock.EVENT_START,
            availability=Availability.PRE_OUTCOME,
            support="six leaf arms; trunks are structural zero",
            producer="covariates.history.center_visit_recency_within_event",
            window="all arrivals through event start",
            missingness="no visited leaf arm at the event",
            structural_zero="trunk arms",
        ),
        _definition(
            "visit_recency_trials_patch",
            "Minimum visit recency over the two reward-site arms belonging to a candidate patch (the more-recently-visited leaf sets the patch value).",
            grain="event × candidate_patch",
            units="trials",
            clock=QueryClock.EVENT_START,
            availability=Availability.PRE_OUTCOME,
            support="all three candidate patches",
            producer="replay.tables.build_event_patch_table",
            window="all arrivals through event start",
            missingness="either leaf-arm recency in the patch is unavailable",
        ),
        _definition(
            "is_patch_never_visited",
            "Both reward-site arms in the candidate patch have never been visited (leaf-minimum of the per-arm never-visited indicator).",
            grain="event × candidate_patch",
            units="binary indicator",
            clock=QueryClock.EVENT_START,
            availability=Availability.PRE_OUTCOME,
            support="all three candidate patches",
            producer="replay.tables.build_event_patch_table",
            missingness="leaf-arm novelty support is incomplete",
        ),
        _definition(
            "is_current_patch",
            "Candidate patch contains the single well occupied throughout the event.",
            grain="event × candidate_patch",
            units="binary indicator",
            clock=QueryClock.EVENT_START,
            availability=Availability.PRE_OUTCOME,
            support="all three candidate patches",
            producer="replay.tables.build_event_patch_table",
            missingness="event is not wholly at one well",
        ),
        _definition(
            "is_prev_patch",
            "Candidate patch contains the origin of the most recently completed trip.",
            grain="event × candidate_patch",
            units="binary indicator",
            clock=QueryClock.GOAL_ARRIVAL,
            availability=Availability.PRE_OUTCOME,
            support="all three candidate patches",
            producer="replay.tables.build_event_patch_table",
            missingness="no completed trip; encoded as all-zero indicators",
        ),
        _definition(
            "is_last_rewarded_noncurrent_patch",
            "Candidate patch is the most recently rewarded patch and is not current.",
            grain="event × candidate_patch",
            units="binary indicator",
            clock=QueryClock.EVENT_START,
            availability=Availability.PRE_OUTCOME,
            support="all three candidate patches",
            producer="replay.tables.build_event_patch_table",
            missingness="current occupancy is unresolved",
        ),
        _definition(
            "did_last_completed_trip_switch_patch",
            "The most recently completed trip changed task patch.",
            grain=_STOP,
            units="binary indicator",
            clock=QueryClock.GOAL_ARRIVAL,
            availability=Availability.PRE_OUTCOME,
            support="stops after a completed trip",
            producer="covariates.context.add_trial_context",
            missingness="no completed trip",
        ),
        _definition(
            "time_since_stop_start_s",
            "Elapsed physical time from immobility-stop start to the response bin or event.",
            grain="stop × response time",
            units="seconds",
            clock=QueryClock.EVENT_RESPONSE,
            availability=Availability.PRE_OUTCOME,
            support="all response bins/events",
            producer="replay._helpers.aggregate_to_timebins or replay.tables",
            missingness="response time or stop start unavailable",
        ),
        _definition(
            "n_nonlocal_bins",
            "Number of decoder time bins the non-local event spans.",
            grain="event",
            units="decoder bins",
            clock=QueryClock.EVENT_RESPONSE,
            availability=Availability.PRE_OUTCOME,
            support="all non-local events; at least one bin by construction",
            producer="replay.tables_event.build_event_table",
            missingness="never missing on a materialized event",
        ),
        _definition(
            "trials_since_block_change",
            "Number of completed trials since the current reward-probability block began.",
            grain=_TRIAL_STOP,
            units="trials",
            clock=QueryClock.STOP_START,
            availability=Availability.PRE_OUTCOME,
            support="qvals-mapped stops",
            producer="analysis.behavior.qvals_table",
            missingness="behavioral model block metadata unavailable",
        ),
        _definition(
            "task_reward_probability_patch_mean_gap_best_vs_second",
            "Difference between the largest and second-largest mean task reward probability across patches.",
            grain=_TRIAL_STOP,
            units="probability difference",
            clock=QueryClock.STOP_START,
            availability=Availability.PRE_OUTCOME,
            support="qvals-mapped stops",
            producer="analysis.behavior.qvals_table",
            missingness="task contingency unavailable",
        ),
        _definition(
            "reward_belief_mean_chosen_arm_preoutcome",
            "Pre-outcome Beta reward-belief mean selected at the realized destination arm.",
            grain=_TRIAL_STOP,
            units="probability",
            clock=QueryClock.GOAL_ARRIVAL,
            availability=Availability.POST_CHOICE,
            support="qvals-mapped stops after choice arrival",
            producer="analysis.behavior.qvals_table.add_experienced_value_columns",
            missingness="destination or behavioral-model row unavailable",
        ),
        _definition(
            "task_reward_probability_chosen_arm",
            "Scheduled reward probability selected at the realized destination arm.",
            grain=_TRIAL_STOP,
            units="probability",
            clock=QueryClock.GOAL_ARRIVAL,
            availability=Availability.POST_CHOICE,
            support="qvals-mapped stops after choice arrival",
            producer="analysis.behavior.qvals_table.add_experienced_value_columns",
            missingness="destination or task contingency unavailable",
        ),
        _definition(
            "reward_belief_mean_update_l1_alternative_patches_postoutcome",
            "L1 magnitude of the current-trial belief-mean update over ports in the two unchosen patches.",
            grain=_TRIAL_STOP,
            units="summed absolute probability change",
            clock=QueryClock.OUTCOME_AVAILABILITY,
            availability=Availability.POST_OUTCOME,
            support="qvals-mapped stops",
            producer="analysis.behavior.qvals_table canonical ingestion adapter",
            missingness="behavioral model or current outcome unavailable",
        ),
        _definition(
            "is_rewarded_outcome",
            "The current assigned trial delivered reward.",
            grain=_TRIAL_STOP,
            units="binary indicator",
            clock=QueryClock.OUTCOME_AVAILABILITY,
            availability=Availability.POST_OUTCOME,
            support="stops mapped to an outcome-known trial",
            producer="analysis.behavior.qvals_table",
            missingness="current outcome is not yet available",
        ),
        _definition(
            "is_omitted_outcome",
            "The current assigned trial delivered no reward.",
            grain=_TRIAL_STOP,
            units="binary indicator",
            clock=QueryClock.OUTCOME_AVAILABILITY,
            availability=Availability.POST_OUTCOME,
            support="stops mapped to an outcome-known trial",
            producer="analysis.behavior.qvals_table",
            missingness="current outcome is not yet available",
        ),
        _definition(
            "completed_trial_reward",
            "Eventual binary reward outcome of the most recently arrived trip; a "
            "descriptive/future-bearing column that stores the outcome at arrival, "
            "not gated on when feedback became observable.",
            grain=_STOP,
            units="binary reward",
            clock=QueryClock.OUTCOME_AVAILABILITY,
            availability=Availability.POST_OUTCOME,
            support="stops after a completed trip",
            producer="covariates.context.add_trial_context",
            missingness="no completed physical trip yet",
        ),
        _definition(
            "next_arm",
            "Arm reached by the future outbound trip after the current stop.",
            grain=_STOP,
            units="arm identifier",
            clock=QueryClock.FUTURE_CHOICE,
            availability=Availability.FUTURE_REFERENCING,
            support="stops with an observed subsequent trip",
            producer="replay.tables.build_stop_table",
            missingness="no subsequent trip",
        ),
        _definition(
            "is_next_arm",
            "Candidate arm equals the future outbound destination arm.",
            grain=_EVENT_ARM,
            units="binary indicator",
            clock=QueryClock.FUTURE_CHOICE,
            availability=Availability.FUTURE_REFERENCING,
            support="all candidate arms for events with a subsequent trip",
            producer="covariates.candidate_arm.attach_candidate_arm_covariates",
            missingness="no subsequent trip; encoded as all-zero indicators",
        ),
        _definition(
            "chosen_arm_fraction",
            "Decoded event content fraction assigned to the future chosen arm.",
            grain="event",
            units="fraction",
            clock=QueryClock.FUTURE_CHOICE,
            availability=Availability.FUTURE_REFERENCING,
            support="events with future choice and decoded arm fractions",
            producer="replay.retrospective.chosen_arm_fraction",
            missingness="future choice or decoded content unavailable",
        ),
        _definition(
            "ends_at_next",
            "Final decoded MAP arm equals the future chosen arm.",
            grain="event",
            units="binary indicator",
            clock=QueryClock.FUTURE_CHOICE,
            availability=Availability.FUTURE_REFERENCING,
            support="events with a future choice",
            producer="replay.retrospective.ends_at_next",
            missingness="future choice unavailable; missing endpoint is observed failure",
        ),
        _definition(
            "contains_path_to_next",
            "Decoded MAP-arm set contains every arm on the task route to the future chosen arm.",
            grain="event",
            units="binary indicator",
            clock=QueryClock.FUTURE_CHOICE,
            availability=Availability.FUTURE_REFERENCING,
            support="events with future choice and at least one decoded arm",
            producer="replay.retrospective.contains_path_to_next",
            missingness="future choice or decoded sequence unavailable",
        ),
        _definition(
            "replay_fraction",
            "Mean event-level candidate-arm decoded fraction within the stop.",
            grain="stop × candidate_arm",
            units="mean event fraction",
            clock=QueryClock.EVENT_RESPONSE,
            availability=Availability.PRE_OUTCOME,
            support="candidate arms in stops containing at least one replay event",
            producer="replay.choice_prediction.build_choice_dataset",
            window="all selected replay events in the stop, equally weighted by event",
            missingness="stop contains no selected replay event",
        ),
        _definition(
            "replay_fraction_swr",
            "Mean candidate-arm decoded fraction over SWR-overlapping replay events in the stop.",
            grain="stop × candidate_arm",
            units="mean event fraction",
            clock=QueryClock.EVENT_RESPONSE,
            availability=Availability.PRE_OUTCOME,
            support="candidate arms in stops containing an SWR-overlapping replay event",
            producer="replay.choice_prediction.build_choice_dataset",
            window="SWR-overlapping replay events in the stop, equally weighted by event",
            missingness="stop contains no SWR-overlapping replay event",
        ),
        _definition(
            "replay_fraction_non_swr",
            "Mean candidate-arm decoded fraction over non-SWR replay events in the stop.",
            grain="stop × candidate_arm",
            units="mean event fraction",
            clock=QueryClock.EVENT_RESPONSE,
            availability=Availability.PRE_OUTCOME,
            support="candidate arms in stops containing a non-SWR replay event",
            producer="replay.choice_prediction.build_choice_dataset",
            window="non-SWR replay events in the stop, equally weighted by event",
            missingness="stop contains no non-SWR replay event",
        ),
    ]
)

for basis in range(1, 4):
    _DEFINITIONS.append(
        _definition(
            f"visit_recency_trials_event_leaf_centered_spline_{basis}",
            f"Spline basis {basis} of event-leaf-centered candidate-arm visit recency.",
            grain=_EVENT_ARM,
            units="basis value",
            clock=QueryClock.EVENT_START,
            availability=Availability.PRE_OUTCOME,
            support="six leaf arms; trunks are structural zero",
            producer="covariates.history.spline_visit_recency_within_event",
            window="all arrivals through event start",
            missingness="centered recency unavailable",
            structural_zero="trunk arms",
        )
    )

for arm in range(8):
    _DEFINITIONS.append(
        _definition(
            f"is_candidate_arm_{arm}",
            f"Candidate-arm fixed-effect indicator for arm {arm}; arm 8 is the reference.",
            grain=_EVENT_ARM,
            units="binary indicator",
            clock=QueryClock.EVENT_RESPONSE,
            availability=Availability.PRE_OUTCOME,
            support="all nine candidate arms",
            producer="replay._helpers.add_arm_fixed_effects",
            missingness="candidate arm unavailable",
        )
    )

for lag in range(1, N_REWARD_LAGS + 1):
    _DEFINITIONS.append(
        _definition(
            f"known_reward_outcome_lag_{lag}",
            f"The {lag}-th most recent physically known binary reward outcome at stop start.",
            grain=_STOP,
            units="binary reward",
            clock=QueryClock.STOP_START,
            availability=Availability.PRE_OUTCOME,
            support="stops with sufficient known outcome history",
            producer="covariates.history.lagged_trial_rewards",
            window=f"known-outcome lag {lag}",
            missingness=f"fewer than {lag} outcomes are known",
        )
    )

for bump, name in enumerate(REWARD_HISTORY_COLUMNS):
    _DEFINITIONS.append(
        _definition(
            name,
            f"Raised-cosine reward-history component {bump} at stop start: the "
            "known-outcome lag vector averaged under bump "
            f"{bump} of a log-spaced basis, normalized over the lags actually "
            "known. Component 0 weights the most recent outcomes; later "
            "components weight progressively older ones.",
            grain=_STOP,
            units="weighted reward fraction",
            clock=QueryClock.STOP_START,
            availability=Availability.PRE_OUTCOME,
            support="all stops",
            producer="covariates.history.reward_history_basis_features",
            window=(
                "log-spaced raised-cosine bump over the last 10 known outcomes, "
                "normalized over the available lags"
            ),
            missingness=(
                "none; undefined bumps are zero-filled and flagged by "
                f"{REWARD_HISTORY_INCOMPLETE}"
            ),
        )
    )

_DEFINITIONS.append(
    _definition(
        REWARD_HISTORY_INCOMPLETE,
        "Structural-missingness nuisance term: 1 for stops with fewer than two "
        "known reward outcomes, whose slowest raised-cosine component had no "
        "outcome in its support and was zero-filled. It separates a zero-filled "
        "unknown history from a genuinely reward-free one; it is not a "
        "behavioral quantity and is never reported.",
        grain=_STOP,
        units="binary indicator",
        clock=QueryClock.STOP_START,
        availability=Availability.PRE_OUTCOME,
        support="all stops",
        producer="covariates.history.reward_history_basis_features",
        window="the last 10 known outcomes",
        missingness="none",
    )
)

for lag in range(1, N_HISTORY_LAGS + 1):
    for prefix, description in (
        ("onset_lag", "Own-population event occurrence"),
        ("cross_lag", "Other-population event occurrence"),
    ):
        _DEFINITIONS.append(
            _definition(
                f"{prefix}{lag}",
                f"{description} {lag} response bins before the current bin.",
                grain="stop × response bin",
                units="binary indicator",
                clock=QueryClock.EVENT_RESPONSE,
                availability=Availability.PRE_OUTCOME,
                support="per-bin rate design",
                producer="replay._helpers.aggregate_to_timebins",
                missingness="lag precedes stop start",
            )
        )

for basis in range(N_HISTORY_BASIS):
    for prefix, description in (
        ("hist_rcos_", "Own-population event-history"),
        ("cross_hist_rcos_", "Other-population event-history"),
    ):
        _DEFINITIONS.append(
            _definition(
                f"{prefix}{basis}",
                f"Raised-cosine basis {basis} for {description.lower()}.",
                grain="stop × response bin",
                units="basis-weighted event history",
                clock=QueryClock.EVENT_RESPONSE,
                availability=Availability.PRE_OUTCOME,
                support="per-bin rate design",
                producer="replay._helpers.aggregate_to_timebins",
                window="past 150 ms",
                missingness="none; pre-stop history is structural zero",
            )
        )

_DEFINITIONS.append(
    _definition(
        "patch_fraction",
        "Share of the event's real-arm decoded timebins allocated to this patch.",
        grain="event × candidate_patch",
        units="fraction",
        clock=QueryClock.EVENT_RESPONSE,
        availability=Availability.PRE_OUTCOME,
        support="all three task patches",
        producer="replay.tables_arm_patch.build_event_patch_table",
        window="all in-event timebins whose MAP estimate lands on a real arm",
        missingness="event has no real-arm bins",
    )
)

_DEFINITIONS.append(
    _definition(
        "tc",
        "Response-bin time relative to immobility-stop start.",
        grain="stop × response bin",
        units="seconds",
        clock=QueryClock.EVENT_RESPONSE,
        availability=Availability.PRE_OUTCOME,
        support="per-bin rate design",
        producer="replay._helpers.aggregate_to_timebins",
        missingness="none for constructed bins",
    )
)

_DEFINITIONS.append(
    _definition(
        "time_since_last_well_arrival_s",
        "Time elapsed since the arrival at the well occupied at the start of "
        "the stop. Two incarnations share the name and the clock: on the "
        "per-bin rate design it is the bin centre minus the arrival, zero at "
        "the moment of arrival and non-negative for every bin of an at-well "
        "stop (the bout can only begin once the animal has arrived); on the "
        "event table it is the event's first timebin minus the same arrival.",
        grain="event; stop × response bin",
        units="seconds",
        clock=QueryClock.GOAL_ARRIVAL,
        availability=Availability.PRE_OUTCOME,
        support="event table; per-bin rate design; requires a completed arrival",
        producer=(
            "replay.tables_event.build_event_table (event grain); "
            "replay._helpers.aggregate_to_timebins (per-bin grain)"
        ),
        missingness="no completed arrival before the event / at the bout start",
    )
)

_DEFINITIONS.append(
    _definition(
        "time_since_outcome_known_s",
        "Response-bin time relative to the moment the current trip's reward or "
        "omission became physically observable. NEGATIVE for bins that precede "
        "that boundary, zero at it, positive after it.",
        grain="stop × response bin",
        units="seconds",
        clock=QueryClock.OUTCOME_AVAILABILITY,
        availability=Availability.POST_OUTCOME,
        support="per-bin rate design; stops with a completed arrival",
        producer="replay._helpers.aggregate_to_timebins",
        missingness="no completed arrival at the bout start",
    )
)

COVARIATE_DEFINITIONS: dict[str, CovariateDefinition] = {
    definition.name: definition for definition in _DEFINITIONS
}
if len(COVARIATE_DEFINITIONS) != len(_DEFINITIONS):
    raise RuntimeError("duplicate canonical names in COVARIATE_DEFINITIONS")


def definition_for(name: str) -> CovariateDefinition:
    """Return one definition or fail loudly for an undocumented covariate."""
    try:
        return COVARIATE_DEFINITIONS[name]
    except KeyError as error:
        raise KeyError(
            f"no scientific covariate definition is registered for {name!r}"
        ) from error


def definitions_frame() -> pd.DataFrame:
    """Return the catalog as a stable, human-readable dataframe."""
    return pd.DataFrame(
        [asdict(definition) for definition in COVARIATE_DEFINITIONS.values()]
    ).sort_values(["grain", "name"], ignore_index=True)
