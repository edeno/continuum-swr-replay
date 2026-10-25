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

Derived and interaction columns resolve to their **source atoms**. A product
column (the canonical current-arm belief-update product,
``swr_x_reward_belief_mean_candidate_arm_preoutcome``) resolves to the source of
each factor declared for it in
:data:`~.covariates.interactions.INTERACTION_FACTORS`; a patsy interaction
(``is_swr:rpe_postoutcome``) to the sources of its operands; any other derived
column through :data:`DERIVED_COVARIATE_SOURCES`. Both mappings are explicit
because product spelling is not an information-time model, so the completeness
test asserts every ladder predictor resolves into :data:`KNOWN_ATOMS`.
"""

from __future__ import annotations

from .covariates.definitions import (
    ABSOLUTE_SECONDS_UNITS,
    COVARIATE_DEFINITIONS,
    POST_CHOICE_COVARIATES,
    Availability,
)
from .covariates.interactions import INTERACTION_FACTORS, factor_source
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

# Derived, pre-materialized column -> source covariate atom(s). Interaction
# products are not listed: their factors are declared in INTERACTION_FACTORS.
DERIVED_COVARIATE_SOURCES = {
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
    # Beta belief variance is qvals-dependent and standardized on leaf support;
    # the raw variance has SD ~0.02, much smaller than the belief mean.
    "reward_belief_variance_candidate_arm_preoutcome_z": [
        "reward_belief_variance_candidate_arm_preoutcome"
    ],
}

# Universe every source atom must fall in. Derived columns resolve to these
# atoms through ``DERIVED_COVARIATE_SOURCES`` (products through their declared
# factors) before enforcement.
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


def _column_sources(column: str) -> list[str]:
    """Source atoms of one design column: a product's factors', else its own."""
    if column in INTERACTION_FACTORS:
        return [
            factor_source(factor)
            for factor in INTERACTION_FACTORS[column]
            if factor not in SAFE_MODIFIERS
        ]
    return list(DERIVED_COVARIATE_SOURCES.get(column, [column]))


def timing_of(predictor: str) -> list[str]:
    """Resolve a predictor (plain, patsy interaction, or pre-materialized product) to
    its source atoms.

    Splits ALL ``:`` operands (not just the first), drops safe gating modifiers, and
    expands each remaining operand -- a declared product through the sources of
    its factors (:data:`~.covariates.interactions.INTERACTION_FACTORS`), any
    other column through :data:`DERIVED_COVARIATE_SOURCES` -- so a post-outcome
    operand can't hide behind a safe left operand
    (``is_swr:rpe -> ['rpe_postoutcome']``, not ``['is_swr']``).
    """
    atoms: list[str] = []
    for op in predictor.split(":"):
        if op in SAFE_MODIFIERS:
            continue
        atoms.extend(_column_sources(op))
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
