"""Per-covariate pre/post-outcome timing registry + model-build enforcement.

The single in-code home for "when is this covariate known relative to the response?".
The three availability frozensets are DERIVED from ``COVARIATE_DEFINITIONS``
(each covariate's ``Availability`` field) — the provenance audit at
``.claude/docs/reviews/06_covariate_provenance.csv`` is a cross-check reference,
not the source; editing that CSV does not change enforcement. Two distinct leaks
get two distinct remedies (see :func:`assert_predictor_timing`):

- **POST_OUTCOME** — carries a current-trip value update
  (``rpe_postoutcome``,
  ``reward_belief_mean_update_l1_all_ports_postoutcome``).
  Fixable by restricting the fit to **post-current-outcome** bins (``retrospective=True``).
- **FUTURE_REFERENCING** — the value itself embeds a not-yet-occurred event
  (``periswitch_index_trials`` signs toward the *next* switch;
  ``next_arm``/``is_next_arm``).
  **No temporal gate makes it predictive-safe** — only ``descriptive=True`` (peri-switch
  alignment) or a strictly past-only replacement (e.g. ``trials_since_last_switch``).

Derived/interaction columns (the canonical current-arm belief-update product,
``is_swr:rpe_postoutcome``, and
``swr_x_reward_belief_mean_candidate_arm_preoutcome``)
resolve to their **source atoms** through :data:`DERIVED_COVARIATE_SOURCES`; the
mapping is explicit because product spelling is not an information-time model, so
the completeness test asserts every ladder predictor resolves into
:data:`KNOWN_ATOMS`.
"""

from __future__ import annotations

from .covariates.definitions import (
    ABSOLUTE_SECONDS_UNITS,
    COVARIATE_DEFINITIONS,
    POST_CHOICE_COVARIATES,
    Availability,
)
from .model_specs import N_REWARD_LAGS

# Timing classification is derived from the scientific definition catalog so a
# rename or new fitted atom cannot acquire a second, contradictory hand-written
# timing definition here. Chosen-arm scalars are scientifically POST_CHOICE;
# current fitters conservatively use the existing post-outcome gate for them.
POST_OUTCOME = (
    frozenset(
        name
        for name, definition in COVARIATE_DEFINITIONS.items()
        if definition.availability is Availability.POST_OUTCOME
    )
    | POST_CHOICE_COVARIATES
)
FUTURE_REFERENCING = frozenset(
    name
    for name, definition in COVARIATE_DEFINITIONS.items()
    if definition.availability is Availability.FUTURE_REFERENCING
)
PRE_OUTCOME_SAFE = frozenset(
    name
    for name, definition in COVARIATE_DEFINITIONS.items()
    if definition.availability is Availability.PRE_OUTCOME
)

# Gating factors, not outcome-bearing -- dropped when resolving a patsy interaction.
SAFE_MODIFIERS = frozenset({"is_swr", "is_mua", "Intercept", "1"})

# Raw readings of the absolute session clock (e.g. ``last_well_arrival_time_s``,
# ``outcome_known_time_s``). Only DIFFERENCES between them are interpretable, so
# an epoch clock is never itself a predictor -- as a regressor it would fit
# session drift and the arbitrary epoch origin rather than any elapsed-time
# effect. Derived durations (``time_since_last_well_arrival_s``,
# ``time_since_outcome_known_s``) are fine and are gated only by their source
# atom's availability. Derived from the catalog's units string so a new clock
# column inherits the ban by being tagged, not by being listed here.
ABSOLUTE_SESSION_CLOCKS = frozenset(
    name
    for name, definition in COVARIATE_DEFINITIONS.items()
    if definition.units == ABSOLUTE_SECONDS_UNITS
)

DERIVED_COVARIATE_SOURCES = {  # derived/pre-materialized column -> source covariate atom(s)
    # Stop-local summaries. Omission streak is a deterministic reduction of the
    # causal reward-lag vector. WSLS uses the eventual outbound choice and is
    # therefore future-referencing even after the current outcome is known.
    "omission_streak": [
        f"known_reward_outcome_lag_{j}" for j in range(1, N_REWARD_LAGS + 1)
    ],
    # Reads completed_trial_reward since 1290257, not reward_outcome.
    "wsls_deviation": ["completed_trial_reward", "is_stay"],
    # Per-bin clocks built by ``aggregate_to_timebins`` as bin centre minus an
    # absolute stop clock; each inherits its source clock's availability, so the
    # outcome-relative one stays behind the post-outcome gate even though its
    # negative values sit before the boundary.
    "time_since_last_well_arrival_s": ["last_well_arrival_time_s"],
    "time_since_outcome_known_s": ["outcome_known_time_s"],
    # Patch aggregates of already-classified arm covariates. Registered
    # explicitly so the patch ladder resolves, rather than relying on a
    # name-split heuristic that would treat "<x>_patch" as its own atom.
    "reward_count_candidate_arm_last10_known_trials_patch": [
        "reward_count_candidate_arm_last10_known_trials"
    ],
    "task_reward_probability_candidate_arm_patch": [
        "task_reward_probability_candidate_arm"
    ],
    "reward_belief_mean_candidate_arm_preoutcome_patch": [
        "reward_belief_mean_candidate_arm_preoutcome"
    ],
    "reward_fraction_candidate_arm_last10_visits_patch": [
        "reward_fraction_candidate_arm_last10_visits"
    ],
    # Replay content during the stop, measured before the outbound choice.
    "replay_fraction": ["replay_fraction"],
    "replay_fraction_swr": ["replay_fraction_swr"],
    "replay_fraction_non_swr": ["replay_fraction_non_swr"],
    # Content products: each name exposes every canonical factor.
    "rpe_postoutcome_z_x_is_current_arm": ["rpe_postoutcome"],
    "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm": [
        "relative_value_switch_minus_stay_preoutcome"
    ],
    "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm": [
        "reward_belief_differential_entropy_leaf_mean_preoutcome"
    ],
    "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm": [
        "reward_belief_mean_update_l1_all_ports_postoutcome"
    ],
    "periswitch_index_trials_z_x_is_current_arm": ["periswitch_index_trials"],
    "is_periswitch_post_x_is_current_arm": ["is_periswitch_post"],
    "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm": [
        "periswitch_index_trials",
        "is_periswitch_post",
    ],
    "trials_since_block_change_z_x_is_current_arm": ["trials_since_block_change"],
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm": [
        "patch_switch_fraction_last5_completed_trips"
    ],
    "log1p_time_since_stop_start_s_z_x_is_current_arm": [
        "time_since_stop_start_s",
        "is_current_arm",
    ],
    # Event-length control: z(log n_nonlocal_bins) crossed with the arm/patch
    # indicator it modulates. Both factors are exposed so neither hides.
    "log_n_nonlocal_bins_z_x_is_current_arm": [
        "n_nonlocal_bins",
        "is_current_arm",
    ],
    "log_n_nonlocal_bins_z_x_is_prev_arm": [
        "n_nonlocal_bins",
        "is_prev_arm",
    ],
    # pre-materialized swr_x_* columns used directly in the CS-rungs (model_specs.py)
    "swr_x_is_current_arm": ["is_current_arm"],
    "swr_x_is_prev_arm": ["is_prev_arm"],
    "swr_x_is_next_arm": ["is_next_arm"],  # FUTURE_REFERENCING
    "swr_x_is_last_rewarded_arm": ["is_last_rewarded_arm"],
    "swr_x_is_last_rewarded_noncurrent_arm": ["is_last_rewarded_noncurrent_arm"],
    "swr_x_reward_fraction_candidate_arm_last10_visits": [
        "reward_fraction_candidate_arm_last10_visits"
    ],
    "swr_x_visit_recency_trials_event_leaf_centered": [
        "visit_recency_trials_event_leaf_centered"
    ],
    "swr_x_visit_recency_trials_event_leaf_centered_spline_1": [
        "visit_recency_trials_event_leaf_centered_spline_1"
    ],
    "swr_x_visit_recency_trials_event_leaf_centered_spline_2": [
        "visit_recency_trials_event_leaf_centered_spline_2"
    ],
    "swr_x_visit_recency_trials_event_leaf_centered_spline_3": [
        "visit_recency_trials_event_leaf_centered_spline_3"
    ],
    "swr_x_is_candidate_arm_never_visited": ["is_candidate_arm_never_visited"],
    "swr_x_reward_belief_mean_candidate_arm_preoutcome": [
        "reward_belief_mean_candidate_arm_preoutcome"
    ],
    # Beta belief variance is qvals-dependent and standardized on leaf support;
    # the raw variance has SD ~0.02, much smaller than the belief mean.
    "reward_belief_variance_candidate_arm_preoutcome_z": [
        "reward_belief_variance_candidate_arm_preoutcome"
    ],
    "swr_x_reward_belief_variance_candidate_arm_preoutcome": [
        "reward_belief_variance_candidate_arm_preoutcome"
    ],
    "swr_x_reward_belief_variance_candidate_arm_preoutcome_z": [
        "reward_belief_variance_candidate_arm_preoutcome"
    ],
    # patch-cohort pre-materialized interactions (content patch ladder)
    "swr_x_is_current_patch": ["is_current_patch"],
    "swr_x_is_last_rewarded_noncurrent_patch": ["is_last_rewarded_noncurrent_patch"],
    "swr_x_is_prev_patch": ["is_prev_patch"],
    "periswitch_index_trials_z_x_is_current_patch": ["periswitch_index_trials"],
    "log1p_time_since_stop_start_s_z_x_is_current_patch": [
        "time_since_stop_start_s",
        "is_current_patch",
    ],
    "log_n_nonlocal_bins_z_x_is_current_patch": [
        "n_nonlocal_bins",
        "is_current_patch",
    ],
    "log_n_nonlocal_bins_z_x_is_prev_patch": [
        "n_nonlocal_bins",
        "is_prev_patch",
    ],
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch": [
        "patch_switch_fraction_last5_completed_trips"
    ],
}

# Universe every source atom must fall in. Derived columns resolve to these
# atoms through ``DERIVED_COVARIATE_SOURCES`` before enforcement.
KNOWN_ATOMS = frozenset(COVARIATE_DEFINITIONS)

# Load-bearing consistency check (keeps PRE_OUTCOME_SAFE from silently rotting):
# every catalogued covariate carries exactly one Availability, so the three
# availability-derived classes must cover the whole catalog. POST_OUTCOME also
# folds in POST_CHOICE_COVARIATES (which are PRE_OUTCOME by availability but
# gated conservatively), so it can only add names already inside KNOWN_ATOMS.
assert (PRE_OUTCOME_SAFE | POST_OUTCOME | FUTURE_REFERENCING) == KNOWN_ATOMS, (
    "covariate timing classes do not partition the definition catalog: "
    f"unclassified={sorted(KNOWN_ATOMS - (PRE_OUTCOME_SAFE | POST_OUTCOME | FUTURE_REFERENCING))}"
)


def timing_of(predictor: str) -> list[str]:
    """Resolve a predictor (plain, patsy interaction, or pre-materialized ``swr_x_*``) to
    its source atoms.

    Splits ALL ``:`` operands (not just the first), drops safe gating modifiers, and
    expands each remaining operand through :data:`DERIVED_COVARIATE_SOURCES` so a
    post-outcome operand can't hide behind a safe left operand
    (``is_swr:rpe -> ['rpe_postoutcome']``, not ``['is_swr']``).
    """
    if predictor in DERIVED_COVARIATE_SOURCES:
        return list(DERIVED_COVARIATE_SOURCES[predictor])
    atoms: list[str] = []
    for op in predictor.split(":"):
        if op in SAFE_MODIFIERS:
            continue
        atoms.extend(DERIVED_COVARIATE_SOURCES.get(op, [op]))
    return atoms


def assert_predictor_timing(
    predictors, *, retrospective: bool = False, descriptive: bool = False
) -> None:
    """Enforce covariate timing at model-build; raise on an un-remedied leak.

    ``retrospective=True`` whitelists **only** POST_OUTCOME covariates (valid once the
    current outcome is known — the fit must be gated to post-current-outcome bins).
    FUTURE_REFERENCING covariates are **never** gate-fixable and require
    ``descriptive=True`` (or a strictly past-only replacement).

    A raw :data:`ABSOLUTE_SESSION_CLOCKS` reading is refused outright under every
    flag: no gate makes an epoch origin a meaningful regressor.
    """
    for p in predictors:
        raw_clocks = ABSOLUTE_SESSION_CLOCKS.intersection(p.split(":"))
        if raw_clocks:
            raise ValueError(
                f"{p} enters an absolute session clock ({sorted(raw_clocks)}) as a "
                "predictor; only differences between clocks are interpretable. Use "
                "the derived elapsed-time column (e.g. "
                "time_since_last_well_arrival_s) instead."
            )
        atoms = set(timing_of(p))
        unknown = atoms - KNOWN_ATOMS
        if unknown:
            raise ValueError(
                f"{p} resolves to unclassified timing atom(s) {sorted(unknown)}; "
                "add an explicit timing classification before fitting"
            )
        if atoms & FUTURE_REFERENCING and not descriptive:
            raise ValueError(
                f"{p} is future-referencing (e.g. trials_since_switch signs toward the "
                f"NEXT switch); no temporal gate makes it predictive-safe. Use "
                f"descriptive=True, or replace it with a strictly past-only covariate "
                f"(e.g. trials_since_last_switch)."
            )
        if atoms & POST_OUTCOME and not (retrospective or descriptive):
            raise ValueError(
                f"{p} is post-outcome; gate the fit to post-current-outcome bins "
                f"(retrospective=True) or set descriptive=True."
            )
