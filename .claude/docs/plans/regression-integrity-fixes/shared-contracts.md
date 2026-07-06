# Shared contracts

[← back to PLAN.md](PLAN.md)

Cross-phase contracts. Phase 3 *produces* the cohort-audit table; phases 4 and 5 *consume* it.
Phase 4 *classifies* covariate timing; phase 5 relies on that classification for its sensitivity fits.

## Contract 1 — Cohort-audit table schema  {#cohort-audit-schema}

Every fit runner (content, rate, choice) emits one cohort-audit record per (model, population) fit,
written next to the fit outputs as `<run>_cohort_audit.csv`. One row per fit. The `grain` column names
the counting unit; **all `n_*` counts are in that unit** (a content record counts *events*, not
event-arm rows). Drops are **mutually exclusive** — each dropped unit is assigned to exactly one bucket
by the priority order below, so the reconciliation closes even when a unit fails several checks.

| column | meaning |
| --- | --- |
| `model` | model/rung id (e.g. `M3_armreward`, `S7_rpe`, `C_full`) |
| `population` | population/split id (e.g. `all`, `swr`, `non_swr`, `patch`) |
| `grain` | counting unit for every `n_*`: `event` / `stop` / `stop_bin` |
| `cohort_mode` | `per_rung` (default) or `common` (union complete-case across rungs) |
| `n_in` | candidate units entering the runner for this (model, population) |
| `n_drop_qvals_gap` | **[priority 1]** dropped because an HMM/value covariate this model uses is missing (mechanism B — the 2,056-stop tail). 0 for models with no value covariate. |
| `n_drop_own_predictor_missing` | **[priority 2]** dropped for NaN in *this model's own* required predictor (excludes the qvals-gap rows already counted). Name the dominant covariate in `drop_notes`. |
| `n_drop_zero_response` | **[priority 3]** dropped because all responses zero (content: `Σcount==0`) |
| `n_drop_common_cohort_other` | **[priority 4]** `common` mode only: dropped *solely* because another rung's predictor is NaN while this model's own predictors are all present. 0 in `per_rung` mode. |
| `n_excluded_session` | **[priority 5]** units in sessions failing the **data-level** gates `min_*`/rank/`n<2·n_pred`. Does **NOT** include the `|coef|>=10` quasi-separation filter — that is term-level and post-fit (see below). |
| `n_modeled` | final units fit |
| `n_sessions_flagged_nonconverged` | **informational, session-level, NOT a drop bucket** — number of *modeled session-fits* with `converged==False`. Counts fits, not units; excluded from reconciliation. |
| `n_sessions_flagged_separated` | **informational, session-level, NOT a drop bucket** — number of modeled session-fits with `separated==True`. Counts fits, not units; excluded from reconciliation. |
| `n_sessions_in` / `n_sessions_modeled` | session counts before/after |
| `n_animals_modeled` | distinct animals surviving |
| `drop_notes` | free text: dominant own-missing predictor + the drop profile summary below |

Two consistency rules the emitter enforces (F1/F2):
- **Grain:** every `n_drop_*`, `n_in`, `n_modeled` is in the row `grain` (event/stop/bin). The two
  `n_sessions_flagged_*` are explicitly **session/fit** counts (named `n_sessions_*` to signal that),
  informational, and excluded from reconciliation.
- **Quasi-separation is NOT a cohort exclusion.** The pre-existing `|coef|>=10` filter
  (`contrast_aggregation.py:83`) drops a *session's estimate for one term*, per (session, term) — the
  session's data still entered the fit. A model-level cohort row cannot represent a term-specific
  coefficient exclusion, so it is kept **out of cohort reconciliation** entirely and reported in a
  separate **term-level diagnostics table** (Contract 3). Phase 2's convergence/separation flags are
  likewise informational, never drop buckets, unless a separate explicit decision starts excluding on them.

## Contract 3 — Term-level diagnostics table  {#term-diagnostics}

Quasi-separation and convergence are properties of a *fitted coefficient*, not of the cohort, so they get
their own table `<run>_term_diagnostics.csv` (one row per model × population × term):

| column | meaning |
| --- | --- |
| `model` / `population` / `term` | the coefficient identified |
| `n_sessions_fit` | sessions that produced an estimate for this term |
| `n_sessions_quasiseparated` | sessions dropped for this term by the `|coef|>=10` filter (`contrast_aggregation.py:83`) |
| `n_sessions_used` | sessions entering the per-animal/Wilcoxon aggregate for this term (`= n_sessions_fit − n_sessions_quasiseparated`) |
| `n_animals_used` | distinct animals among `n_sessions_used` (the `>=5`/sign-consistency base) |

Producer: the aggregation step (`contrast_aggregation.py` already logs these — surface them as a table).
This is where term-specific exclusions live; the cohort-audit table (Contract 1) stays purely about which
rows entered the fit.

**Priority rule (mutual exclusivity):** classify each dropped unit by the first bucket it hits in the
order qvals_gap → own_predictor_missing → zero_response → common_cohort_other → session. This keeps the
`n_drop_qvals_gap` (mechanism B) and `n_drop_common_cohort_other` (F4 — M0–M2 dropped for a *later*
rung's predictor) distinguishable from a model's own missingness, which is the whole point of the audit.

**Reconciliation invariant (must hold):**
`n_in − n_drop_qvals_gap − n_drop_own_predictor_missing − n_drop_zero_response − n_drop_common_cohort_other − n_excluded_session == n_modeled`.
The emitter asserts this and fails loud if it does not close.

**Drop profile (for non-random-drop detection):** alongside counts, `drop_notes` records, for the
dropped rows, the median `time_since_stop`/trial-index quantile, the reward fraction, and (where value
covariates apply) the value quantile — so a reviewer can see *whether* the drop is biased toward
early-session (mechanism A) or late-session/low-value (mechanism B).

Producer: [phase-3](phase-3-cohort-audit-tables.md). Consumers: phases 4, 5 attach these to every
refit they run. **Do not** let a runner emit fit outputs without the matching cohort-audit record.

## Contract 2 — Covariate pre/post-outcome timing reference  {#covariate-timing}

The authoritative per-covariate timing classification lives in
`.claude/docs/reviews/06_covariate_provenance.csv` (column `time_availability`). Phase 4 promotes the
relevant flags into an in-code registry so fitters can assert timing at model-build:

```python
# analysis/replay/covariate_timing.py  (new, phase 4)
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
    "visit_recency", "visit_recency_demeaned", "cumulative_reward",
    "is_current", "is_prev", "is_last_rewarded", "is_after_switch",
    "hmm_relative_value", "q_arm", "q_arm_hmm", "reward_entropy", "state_entropy",
})
```

**Derived/interaction covariates must map back to their source (F2).** The content ladder builds
interaction columns whose names do **not** contain the source covariate, so a name-splitting heuristic
(`_base_name("update_x_current") -> "update"`) silently fails to flag the leaky ones. Phase 4 ships an
explicit map (verified against `fit_swr_contrast_content.py:133-150`), not a string heuristic:

```python
SAFE_MODIFIERS = frozenset({"is_swr", "is_mua", "Intercept", "1"})  # gating factors, not outcome-bearing

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
    # pre-materialized swr_x_* columns used directly in the CS-rungs (model_specs.py:894-901)
    "swr_x_current": ["is_current"], "swr_x_prev": ["is_prev"],
    "swr_x_next": ["is_next"],                            # FUTURE_REFERENCING
    "swr_x_last_rewarded": ["is_last_rewarded"],
    "swr_x_is_last_rewarded_elsewhere": ["is_last_rewarded_elsewhere"],
    "swr_x_reward_rate_arm_k10": ["reward_rate_arm_k10"],
    "swr_x_visit_recency_demeaned": ["visit_recency_demeaned"],
    "swr_x_q_arm": ["q_arm"], "swr_x_q_arm_hmm": ["q_arm_hmm"],   # value/qvals-dependent (drop → qvals_gap)
}

# Rate-design basis terms — safe backward/within-stop covariates. GENERATED from the same
# constants the specs use (import them; do not hard-code the ranges):
#   from .model_specs import N_REWARD_LAGS, N_HISTORY_LAGS, N_HISTORY_BASIS   # 10, 10, 4
_DESIGN_BASIS = (
    {"tc"}                                                             # bs(tc,...) spline argument
    | {f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)}          # reward_lag1..10
    | {f"onset_lag{j}" for j in range(1, N_HISTORY_LAGS + 1)}          # onset_lag1..10 self-history
    | {f"hist_rcos_{k}" for k in range(N_HISTORY_BASIS)}               # raised-cosine history basis
    | {f"cross_hist_rcos_{k}" for k in range(N_HISTORY_BASIS)}         # cross-history basis
)

# Universe every atom must fall in — the completeness test fails on anything unclassified.
KNOWN_ATOMS = POST_OUTCOME | FUTURE_REFERENCING | PRE_OUTCOME_SAFE | _DESIGN_BASIS | frozenset({
    "recent_switch_rate", "trials_since_block_change",  # backward-safe context
    "arm_dist_rank", "is_same_patch", "dist_to_arm",    # spatial, value-free
}) | {f"arm_fe_{i}" for i in range(8)}

def timing_of(predictor):
    """Resolve a predictor (plain, patsy interaction, or pre-materialized swr_x_*) to its source atoms.
    Splits ALL ':' operands (not just the first), drops safe gating modifiers, and expands each
    remaining operand through DERIVED_COVARIATE_SOURCES so post-outcome operands can't hide behind
    a safe left operand (e.g. is_swr:hmm_rpe -> ['hmm_rpe'], not ['is_swr'])."""
    if predictor in DERIVED_COVARIATE_SOURCES:
        return list(DERIVED_COVARIATE_SOURCES[predictor])
    atoms = []
    for op in predictor.split(":"):
        if op in SAFE_MODIFIERS:
            continue
        atoms.extend(DERIVED_COVARIATE_SOURCES.get(op, [op]))
    return atoms
```

**Rule the fitters enforce (phase 4) — two DIFFERENT remedies for two different leaks:**

```python
def assert_predictor_timing(predictors, *, retrospective=False, descriptive=False):
    """POST_OUTCOME leakage is fixable by restricting to post-current-outcome bins;
    FUTURE_REFERENCING is NOT fixable by any within-stop gate — the covariate value
    itself embeds a not-yet-occurred event regardless of which bin you evaluate it in."""
    for p in predictors:
        atoms = set(timing_of(p))
        if atoms & FUTURE_REFERENCING and not descriptive:
            raise ValueError(
                f"{p} is future-referencing (e.g. trials_since_switch signs toward the NEXT "
                f"switch); no temporal gate makes it predictive-safe. Use descriptive=True, or "
                f"replace it with a strictly past-only covariate (e.g. trials_since_last_switch).")
        if atoms & POST_OUTCOME and not (retrospective or descriptive):
            raise ValueError(
                f"{p} is post-outcome; gate the fit to post-current-outcome bins "
                f"(retrospective=True) or set descriptive=True.")
```

So: `retrospective=True` whitelists **only** `POST_OUTCOME` (current-reward covariates, valid once the
reward is delivered — `hmm_rpe`, `hmm_value_update`, `is_after_reward`). `FUTURE_REFERENCING`
(`trials_since_switch`, `is_post_switch`, `next_arm`, `swr_x_next`) is **never** predictive-safe — a
post-reward bin still carries a `trials_since_switch` that counts toward a switch that hasn't happened
(`qvals_table.py:169`, the `-until` branch). Such a rung must be `descriptive=True` **or** swap in a
strictly past-only switch covariate (`trials_since_last_switch` = the `since` component only). This also
catches the rate `is_swr * (... + hmm_rpe/hmm_value_update/trials_since_switch)` S-rungs
(`model_specs.py:851-861`), whose patsy expansion yields `is_swr:hmm_rpe`/`is_swr:trials_since_switch`.
`is_after_switch` is pre-outcome-safe (destination chosen at departure).

**Completeness rule (F1/F2/F4):** phase 4 tests that for **every** predictor of **every** rung in
`SWR_CONTRAST_CONTENT_ARM_LADDER`, the rate `_s_specs`/`_g_specs` ladders, and the per-stop/per-bin
ladders, `timing_of(pred) ⊆ KNOWN_ATOMS`. **Resolve rate designs through the specs' declared column
lists, not raw patsy strings** — the per-bin formula contains `bs(tc, knots=..., ...)` and
`reward_lag1:bs(tc, ...)` (`model_specs.py:565,663`) which do not tokenize on `:`; iterate the block
provenance (`_blocks_cols` / `ModelSpec.predictors` / the `_HistorySpec.formula` term list at
`model_specs.py:648-663`) so each design term maps to a `_DESIGN_BASIS` atom (`tc`, `reward_lag{j}`,
`onset_lag{j}`, `hist_rcos_{k}`, `cross_hist_rcos_{k}`). An unmapped interaction (or a new `swr_x_*`)
resolving to an unknown atom **fails the test** rather than silently passing as its own name. This same
atom resolution drives cohort drop-classification: a rung using `swr_x_q_arm_hmm` inherits `q_arm_hmm`'s
qvals-dependence, so its qvals-tail drops are attributed to `n_drop_qvals_gap`, not own-predictor.

Consumers: phase 4 (enforcement), phase 5 (labels which sensitivity fits are retrospective).
