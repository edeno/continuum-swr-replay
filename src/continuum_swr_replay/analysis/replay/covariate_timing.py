"""Per-covariate pre/post-outcome timing registry + model-build enforcement.

The single in-code home for "when is this covariate known relative to the response?",
promoting the ``time_availability`` column of
``.claude/docs/reviews/06_covariate_provenance.csv`` into frozensets the fitters can
assert against. Two distinct leaks get two distinct remedies (see
:func:`assert_predictor_timing`):

- **POST_OUTCOME** — carries the current trip's reward (``hmm_rpe``/``rpe``,
  ``hmm_value_update``, ``is_after_reward``/``is_after_omission``/``reward_outcome``).
  Fixable by restricting the fit to **post-current-outcome** bins (``retrospective=True``).
- **FUTURE_REFERENCING** — the value itself embeds a not-yet-occurred event
  (``trials_since_switch`` signs toward the *next* switch; ``next_arm``/``is_next``).
  **No temporal gate makes it predictive-safe** — only ``descriptive=True`` (peri-switch
  alignment) or a strictly past-only replacement (e.g. ``trials_since_last_switch``).

Derived/interaction columns (``update_x_current``, ``is_swr:hmm_rpe``, ``swr_x_q_arm_hmm``)
resolve to their **source atoms** through :data:`DERIVED_COVARIATE_SOURCES` — a name-split
heuristic silently fails (``update_x_current`` does not contain ``hmm_value_update``), so
the mapping is explicit and a completeness test asserts every ladder predictor resolves
into :data:`KNOWN_ATOMS`.
"""

from __future__ import annotations

from .model_specs import N_HISTORY_BASIS, N_HISTORY_LAGS, N_REWARD_LAGS

POST_OUTCOME = frozenset({  # NOT available before the current trip's reward
    "reward_outcome", "is_after_reward", "is_after_omission",
    "hmm_rpe", "rpe", "hmm_value_update",
})
FUTURE_REFERENCING = frozenset({  # references a not-yet-occurred event
    "next_arm", "is_next", "ends_at_next", "contains_path_to_next",
    "chosen_arm_fraction", "trials_since_switch", "is_post_switch",
})
PRE_OUTCOME_SAFE = frozenset({  # trailing/backward, safe as predictors pre-reward
    "reward_lag1", "reward_lag2", "reward_rate_trial_k10", "reward_rate_arm_k10",
    "visit_recency", "visit_recency_demeaned", "visit_recency_patch", "cumulative_reward",
    "is_current", "is_prev", "is_last_rewarded", "is_last_rewarded_elsewhere",
    "is_after_switch",
    "hmm_relative_value", "q_arm", "q_arm_hmm", "p_reward_arm",
    "reward_entropy", "state_entropy",
    "arm_uncertainty",  # pre-outcome Beta belief variance (variance counterpart of q_arm)
})

# Gating factors, not outcome-bearing -- dropped when resolving a patsy interaction.
SAFE_MODIFIERS = frozenset({"is_swr", "is_mua", "Intercept", "1"})

DERIVED_COVARIATE_SOURCES = {  # derived/pre-materialized column -> source covariate atom(s)
    # content `_x_current` interactions (source is the z-scalar's base covariate)
    "rpe_x_current": ["hmm_rpe"], "rpe_x_current_z": ["hmm_rpe"],
    "relval_x_current": ["hmm_relative_value"],
    "entropy_x_current": ["reward_entropy"],
    "update_x_current": ["hmm_value_update"],            # POST_OUTCOME
    "tss_x_current": ["trials_since_switch"],             # FUTURE_REFERENCING
    "postsw_x_current": ["is_post_switch"],               # FUTURE_REFERENCING
    "tsspostsw_x_current": ["trials_since_switch", "is_post_switch"],
    "block_x_current": ["trials_since_block_change"],     # safe (backward)
    "switchrate_x_current": ["recent_switch_rate"],       # safe (backward)
    "tsc_x_current": [],                                   # within-stop time, safe & value-free
    # pre-materialized swr_x_* columns used directly in the CS-rungs (model_specs.py)
    "swr_x_current": ["is_current"], "swr_x_prev": ["is_prev"],
    "swr_x_next": ["is_next"],                            # FUTURE_REFERENCING
    "swr_x_last_rewarded": ["is_last_rewarded"],
    "swr_x_is_last_rewarded_elsewhere": ["is_last_rewarded_elsewhere"],
    "swr_x_reward_rate_arm_k10": ["reward_rate_arm_k10"],
    "swr_x_visit_recency_demeaned": ["visit_recency_demeaned"],
    "swr_x_q_arm": ["q_arm"], "swr_x_q_arm_hmm": ["q_arm_hmm"],  # qvals-dependent
    "swr_x_arm_uncertainty": ["arm_uncertainty"],  # qvals-dependent (beta belief variance)
    # patch-cohort pre-materialized interactions (content patch ladder)
    "swr_x_current_patch": ["is_current_patch"],
    "swr_x_last_rewarded_patch_elsewhere": ["is_last_rewarded_patch_elsewhere"],
    "swr_x_prev_patch": ["is_prev_patch"],
    "currentpatch_x_tss": ["trials_since_switch"],        # FUTURE_REFERENCING
    "currentpatch_x_tsc": [],                              # within-stop time, safe
    "currentpatch_x_switchrate": ["recent_switch_rate"],  # safe (backward)
    "lastrewpatch_x_tss": ["trials_since_switch"],        # FUTURE_REFERENCING
    "swr_x_lastrewpatch_tss": ["trials_since_switch"],    # FUTURE_REFERENCING
}

# Rate-design basis terms — safe backward/within-stop covariates. Generated from the same
# constants the specs use (do not hard-code the ranges).
_DESIGN_BASIS = (
    {"tc"}                                                             # bs(tc,...) spline argument
    | {f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)}          # reward_lag1..10
    | {f"onset_lag{j}" for j in range(1, N_HISTORY_LAGS + 1)}          # onset_lag1..10 self-history
    | {f"cross_lag{j}" for j in range(1, N_HISTORY_LAGS + 1)}          # cross-history dummies
    | {f"hist_rcos_{k}" for k in range(N_HISTORY_BASIS)}               # raised-cosine history basis
    | {f"cross_hist_rcos_{k}" for k in range(N_HISTORY_BASIS)}         # cross-history basis
)

# Universe every atom must fall in — the completeness test fails on anything unclassified.
KNOWN_ATOMS = POST_OUTCOME | FUTURE_REFERENCING | PRE_OUTCOME_SAFE | _DESIGN_BASIS | frozenset({
    "recent_switch_rate", "trials_since_block_change",   # backward-safe context
    "arm_dist_rank", "is_same_patch", "dist_to_arm",     # spatial, value-free
    "is_current_patch", "is_prev_patch",                 # patch spatial identity
    "is_last_rewarded_patch_elsewhere",                  # patch backward-safe
}) | {f"arm_fe_{i}" for i in range(8)}


def timing_of(predictor: str) -> list[str]:
    """Resolve a predictor (plain, patsy interaction, or pre-materialized ``swr_x_*``) to
    its source atoms.

    Splits ALL ``:`` operands (not just the first), drops safe gating modifiers, and
    expands each remaining operand through :data:`DERIVED_COVARIATE_SOURCES` so a
    post-outcome operand can't hide behind a safe left operand
    (``is_swr:hmm_rpe -> ['hmm_rpe']``, not ``['is_swr']``).
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
    """
    for p in predictors:
        atoms = set(timing_of(p))
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
