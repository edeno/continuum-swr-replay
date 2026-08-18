"""Per-covariate pre/post-outcome timing registry + model-build enforcement.

The single in-code home for "when is this covariate known relative to the response?",
promoting the ``time_availability`` column of
``.claude/docs/reviews/06_covariate_provenance.csv`` into frozensets the fitters can
assert against. Two distinct leaks get two distinct remedies (see
:func:`assert_predictor_timing`):

- **POST_OUTCOME** — carries a current-trip value update (``rpe``,
  ``global_mu_update_abs_postoutcome_sess``).
  Fixable by restricting the fit to **post-current-outcome** bins (``retrospective=True``).
- **FUTURE_REFERENCING** — the value itself embeds a not-yet-occurred event
  (``trials_since_switch`` signs toward the *next* switch; ``next_arm``/``is_next``).
  **No temporal gate makes it predictive-safe** — only ``descriptive=True`` (peri-switch
  alignment) or a strictly past-only replacement (e.g. ``trials_since_last_switch``).

Derived/interaction columns (``update_x_current``, ``is_swr:rpe``, ``swr_x_q_arm``)
resolve to their **source atoms** through :data:`DERIVED_COVARIATE_SOURCES` — a name-split
heuristic silently fails (``update_x_current`` does not contain ``global_mu_update_abs_postoutcome_sess``), so
the mapping is explicit and a completeness test asserts every ladder predictor resolves
into :data:`KNOWN_ATOMS`.
"""

from __future__ import annotations

from .model_specs import N_HISTORY_BASIS, N_HISTORY_LAGS, N_REWARD_LAGS

POST_OUTCOME = frozenset(
    {  # NOT available before the current trip's reward
        "rpe",  # beta reward prediction error (reward - q_chosen)
        "global_mu_update_abs_postoutcome_sess",  # beta value-update (post-outcome by name)
        "unchosen_patches_mu_update_abs_postoutcome_sess",  # alternative-patch update
        # Trip *completion* is goal arrival (poke-in), which is strictly earlier
        # than the outcome: reward onset for a rewarded trip, and poke-out for an
        # omission (there is no negative-feedback cue to timestamp -- see
        # qvals_table._outcome_known_time). Between arrival and that boundary
        # these three already carry the outcome, so "completed by event start"
        # does not make them pre-outcome safe. Gate on ``outcome_known_time`` to
        # use them; do not reclassify.
        "reward_outcome",
        "is_after_reward",
        "is_after_omission",
        # Not outcome-bearing -- these are *chosen-arm* quantities, and the
        # chosen arm is the assigned trial's, where a stop is assigned to the
        # trial whose poke-out comes next after stop_start. On 20.5% of stops
        # (measured 2026-08-17; 99.9% of them in transit, median 4.2 s lead)
        # the bout ends before that trial's poke-in, so these describe a choice
        # the animal has not yet made. They belong here rather than in
        # FUTURE_REFERENCING because the remedy is the same one this class
        # names: 0.00% of those stops survive the post-outcome gate. Under
        # ``retrospective=True`` they are safe; ungated they are not.
        "q_current",
        "p_reward_current",
        # The same outcome with no gate at all. It exists so a descriptive
        # construct can name the outcome the animal just experienced --
        # ``wsls_deviation`` pairs it with the eventual choice, so it is
        # future-referencing whatever the outcome clock does, and gating it at
        # the bout start only made it undefined at every at-well stop.
        "completed_trial_reward",
    }
)
FUTURE_REFERENCING = frozenset(
    {  # references a not-yet-occurred event
        "next_arm",
        "is_next",
        "ends_at_next",
        "contains_path_to_next",
        "chosen_arm_fraction",
        "trials_since_switch",
        "is_post_switch",
        # Stop-table ``is_stay`` is the eventual outbound choice from the current
        # well (current_patch == next_patch), not the completed inbound trip.
        "is_stay",
    }
)
PRE_OUTCOME_SAFE = frozenset(
    {  # trailing/backward, safe as predictors pre-reward
        "reward_lag1",
        "reward_lag2",
        "reward_rate_trial_k10",
        "reward_rate_arm_k10",
        "visit_recency",
        "visit_recency_demeaned",
        "visit_recency_spline1_demeaned",
        "visit_recency_spline2_demeaned",
        "visit_recency_spline3_demeaned",
        "visit_recency_patch",
        "arm_never_visited",
        "arm_never_visited_patch",
        "cumulative_reward",
        "is_current",
        "is_prev",
        "is_last_rewarded",
        "is_last_rewarded_elsewhere",
        "is_after_switch",
        "is_after_stay",
        "q_arm",
        "p_reward_arm",
        "value_diff_patch",
        "time_since_stop",
        "leaky_q_value_diff",
        "arm_uncertainty",  # pre-outcome Beta belief variance (variance counterpart of q_arm)
        # beta-bernoulli value/decision block (pre-outcome by name): relative value + belief entropy
        "switch_advantage_mu_preoutcome_sess",
        # Positively classified rather than passing by omission from the two
        # leak sets: "safe" and "never reviewed" should not look the same.
        "is_same_patch",
        "dist_to_arm",
        "arm_dist_rank",
        "recent_switch_rate",
        "trials_since_block_change",
        "beta_reward_entropy",
    }
)

# Gating factors, not outcome-bearing -- dropped when resolving a patsy interaction.
SAFE_MODIFIERS = frozenset({"is_swr", "is_mua", "Intercept", "1"})

DERIVED_COVARIATE_SOURCES = {  # derived/pre-materialized column -> source covariate atom(s)
    # fit_uncertainty_rate renames the beta-model RPE to distinguish it from
    # older nominal stand-ins; its information time is unchanged.
    "rpe_model": ["rpe"],
    # Stop-local summaries. Omission streak is a deterministic reduction of the
    # causal reward-lag vector. WSLS uses the eventual outbound choice and is
    # therefore future-referencing even after the current outcome is known.
    "omission_streak": [f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)],
    # Reads completed_trial_reward since 1290257, not reward_outcome.
    "wsls_deviation": ["completed_trial_reward", "is_stay"],
    # Patch aggregates of already-classified arm covariates. Registered
    # explicitly so the patch ladder resolves, rather than relying on a
    # name-split heuristic that would treat "<x>_patch" as its own atom.
    "cumulative_reward_patch": ["cumulative_reward"],
    "p_reward_arm_patch": ["p_reward_arm"],
    "q_arm_patch": ["q_arm"],
    "reward_rate_arm_k10_patch": ["reward_rate_arm_k10"],
    # Replay content during the stop, measured before the outbound choice.
    "replay_frac": [],
    "replay_frac_swr": [],
    "replay_frac_non_swr": [],
    # content `_x_current` interactions (source is the z-scalar's base covariate)
    "rpe_x_current": ["rpe"],
    "rpe_x_current_z": ["rpe"],
    "relval_x_current": ["switch_advantage_mu_preoutcome_sess"],
    "entropy_x_current": ["beta_reward_entropy"],
    "update_x_current": ["global_mu_update_abs_postoutcome_sess"],  # POST_OUTCOME
    "tss_x_current": ["trials_since_switch"],  # FUTURE_REFERENCING
    "postsw_x_current": ["is_post_switch"],  # FUTURE_REFERENCING
    "tsspostsw_x_current": ["trials_since_switch", "is_post_switch"],
    "block_x_current": ["trials_since_block_change"],  # safe (backward)
    "switchrate_x_current": ["recent_switch_rate"],  # safe (backward)
    "tsc_x_current": [],  # within-stop time, safe & value-free
    # pre-materialized swr_x_* columns used directly in the CS-rungs (model_specs.py)
    "swr_x_current": ["is_current"],
    "swr_x_prev": ["is_prev"],
    "swr_x_next": ["is_next"],  # FUTURE_REFERENCING
    "swr_x_last_rewarded": ["is_last_rewarded"],
    "swr_x_is_last_rewarded_elsewhere": ["is_last_rewarded_elsewhere"],
    "swr_x_reward_rate_arm_k10": ["reward_rate_arm_k10"],
    "swr_x_visit_recency_demeaned": ["visit_recency_demeaned"],
    "swr_x_visit_recency_spline1_demeaned": ["visit_recency_spline1_demeaned"],
    "swr_x_visit_recency_spline2_demeaned": ["visit_recency_spline2_demeaned"],
    "swr_x_visit_recency_spline3_demeaned": ["visit_recency_spline3_demeaned"],
    "swr_x_arm_never_visited": ["arm_never_visited"],
    "swr_x_q_arm": ["q_arm"],  # qvals-dependent
    # beta belief variance (qvals-dependent). Used STANDARDIZED (arm_uncertainty_z):
    # raw betadist_var has SD ~0.02 (~5-6x smaller than q_arm), which makes the
    # per-session covariance singular ("need covariance") — the content models use the
    # _z form. The runner builds swr_x_arm_uncertainty_z = is_swr * arm_uncertainty_z.
    "arm_uncertainty_z": ["arm_uncertainty"],
    "swr_x_arm_uncertainty": ["arm_uncertainty"],
    "swr_x_arm_uncertainty_z": ["arm_uncertainty"],
    # patch-cohort pre-materialized interactions (content patch ladder)
    "swr_x_current_patch": ["is_current_patch"],
    "swr_x_last_rewarded_patch_elsewhere": ["is_last_rewarded_patch_elsewhere"],
    "swr_x_prev_patch": ["is_prev_patch"],
    "currentpatch_x_tss": ["trials_since_switch"],  # FUTURE_REFERENCING
    "currentpatch_x_tsc": [],  # within-stop time, safe
    "currentpatch_x_switchrate": ["recent_switch_rate"],  # safe (backward)
    "lastrewpatch_x_tss": ["trials_since_switch"],  # FUTURE_REFERENCING
    "swr_x_lastrewpatch_tss": ["trials_since_switch"],  # FUTURE_REFERENCING
}

# Rate-design basis terms — safe backward/within-stop covariates. Generated from the same
# constants the specs use (do not hard-code the ranges).
_DESIGN_BASIS = (
    {"tc"}  # bs(tc,...) spline argument
    | {f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)}  # reward_lag1..10
    | {
        f"onset_lag{j}" for j in range(1, N_HISTORY_LAGS + 1)
    }  # onset_lag1..10 self-history
    | {f"cross_lag{j}" for j in range(1, N_HISTORY_LAGS + 1)}  # cross-history dummies
    | {f"hist_rcos_{k}" for k in range(N_HISTORY_BASIS)}  # raised-cosine history basis
    | {f"cross_hist_rcos_{k}" for k in range(N_HISTORY_BASIS)}  # cross-history basis
)

# Universe every atom must fall in — the completeness test fails on anything unclassified.
KNOWN_ATOMS = (
    POST_OUTCOME
    | FUTURE_REFERENCING
    | PRE_OUTCOME_SAFE
    | _DESIGN_BASIS
    | frozenset(
        {
            "recent_switch_rate",
            "trials_since_block_change",  # backward-safe context
            "arm_dist_rank",
            "is_same_patch",
            "dist_to_arm",  # spatial, value-free
            "is_current_patch",
            "is_prev_patch",  # patch spatial identity
            "is_last_rewarded_patch_elsewhere",  # patch backward-safe
        }
    )
    | {f"arm_fe_{i}" for i in range(8)}
)


def timing_of(predictor: str) -> list[str]:
    """Resolve a predictor (plain, patsy interaction, or pre-materialized ``swr_x_*``) to
    its source atoms.

    Splits ALL ``:`` operands (not just the first), drops safe gating modifiers, and
    expands each remaining operand through :data:`DERIVED_COVARIATE_SOURCES` so a
    post-outcome operand can't hide behind a safe left operand
    (``is_swr:rpe -> ['rpe']``, not ``['is_swr']``).
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
