# Phase 4 — Covariate causal hygiene (pre/post-outcome; mechanism-A encoding)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Contracts referenced:** [covariate pre/post-outcome timing](shared-contracts.md#covariate-timing) — this
phase creates the in-code registry and enforcement; [cohort-audit schema](shared-contracts.md#cohort-audit-schema) —
attach a cohort record to every refit here.

This phase changes fitted numbers. It (1) makes rate R1/R2 pre-outcome, (2) enforces that
post/future-outcome covariates are only predictors in explicitly-retrospective models, and (3)
implements mechanism A by **encoding** (decided — overview *Resolved decisions #1*). **Depends on
phase 3** (cohort tables) so every refit reports its cohort, and runs **after phase 6** (the builder
must own `_z`/`q_arm_hmm` before this adds `has_arm_history` — see overview Rollout).

> **Implementation deviations (as built, 2026-07-07).** Two decisions diverged from the
> draft below:
> 1. **Descriptive relabel, not gating.** All leaky rungs (POST_OUTCOME *and*
>    FUTURE_REFERENCING) were **relabeled descriptive** via `DESCRIPTIVE_CONTENT_RUNGS` /
>    `DESCRIPTIVE_RATE_RUNGS` allowlists — the outcome-gating path (`post_current_outcome`
>    mask, retained-omission plumbing) was **not built**. Consequence: the Validation-slice
>    rows for gating (`test_post_outcome_mask`, `test_retrospective_cohort_has_both_outcomes`,
>    the `retrospective=True` refits) are **N/A** and were not written.
> 2. **`has_arm_history` dropped — encode 0.0 only.** The unvisited-leaf indicator was
>    implemented, measured (~11% of events, ~13% novel-arm effect), then **removed** (commit
>    `649952b`); the `reward_rate_arm_k10` coefficient absorbs the modest novel-arm effect.
>    So `test_recent_reward_rate.py` asserts the 0.0 encode / no-history NaN only (no
>    `has_arm_history` column), and the phase-6-ordering rationale above ("before this adds
>    `has_arm_history`") is moot.
>
> What *was* built and verified: the timing registry + enforcement, the R1/R2→`reward_lag1`
> swap, the mechanism-A 0.0 encode (full 218-session regen; **+7.4k** modeled M3 events,
> 215→218 sessions converge, `reward_rate_arm_k10` sign-stable), refreshed
> standardization reference (`test_z_reference_refreshed.py`), and refits of every affected
> ladder (`per_bin_rate` / `swr_contrast_rate` / base-patch confirmed byte-identical no-ops
> and skipped). A `prepare_data --from-cache` fast path (skip the Spyglass fetch for
> covariate-only rebuilds) was added alongside (`test_prepare_data_cache.py`).

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/_helpers.py:80-95](../../../../src/continuum_swr_replay/analysis/replay/_helpers.py) and [:122-130](../../../../src/continuum_swr_replay/analysis/replay/_helpers.py) — `reward_outcome`/`is_after_reward`/`is_after_omission` from the trip-in-progress trial.
- [src/continuum_swr_replay/analysis/replay/covariates.py:179-233](../../../../src/continuum_swr_replay/analysis/replay/covariates.py) — `recent_reward_rate_all_events`: trunk→0.0, unvisited leaf→NaN (the `:196` branch this phase changes to `0.0` + indicator).
- [src/continuum_swr_replay/analysis/replay/tables.py:1296-1345](../../../../src/continuum_swr_replay/analysis/replay/tables.py) — where `reward_rate_arm_k10`, `visit_recency_demeaned`, and `swr_x_*` interactions are attached to `event_arm_table`.
- [src/continuum_swr_replay/analysis/replay/model_specs.py:485-532](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py) — `MODEL_PER_STOP_RATE` R1/R2 (`is_after_omission`) and the `Rlag` note (`:520` already documents `reward_lag1` as the past-only alternative).
- [src/continuum_swr_replay/analysis/replay/model_specs.py:840-901](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py) — rate rungs G5/S7 (`rpe`), G7/S9 (`update`), G1/S3 (`trials_since_switch`); content C6 (`rpe_x_current`), C9 (`update_x_current`), C10 (`tss_x_current`).
- `.claude/docs/reviews/06_covariate_provenance.csv` — the authoritative `time_availability` classification.

## Tasks

- **Create the timing registry** `analysis/replay/covariate_timing.py` with `POST_OUTCOME`,
  `FUTURE_REFERENCING`, `PRE_OUTCOME_SAFE` frozensets and the `DERIVED_COVARIATE_SOURCES` map +
  `timing_of()`, plus the two-remedy `assert_predictor_timing(predictors, *, retrospective, descriptive)`
  — **all exactly as in** [contract 2](shared-contracts.md#covariate-timing) (single home; do not re-derive
  here). Do **not** use a name-splitting heuristic — it silently fails on the derived columns
  (`update_x_current` does not contain `hmm_value_update`; verified against
  `fit_swr_contrast_content.py:133-150`). The contract's `assert_predictor_timing` distinguishes
  `POST_OUTCOME` (fixable by `retrospective=True` post-reward gating) from `FUTURE_REFERENCING` (fixable
  **only** by `descriptive=True` or a past-only replacement) — see the gating task below.
  Every interaction column in `DERIVED_COVARIATE_SOURCES` (incl. `update_x_current`, `tss_x_current`,
  `postsw_x_current`, `tsspostsw_x_current`, `swr_x_q_arm_hmm`) resolves to its source scalar, so the leaky
  rungs are actually caught. A completeness test (below) asserts every predictor appearing in any ladder rung
  is resolvable by `timing_of` — a new interaction with no map entry fails the test rather than passing
  silently.
- **Rate R1/R2 → pre-outcome.** In `MODEL_PER_STOP_RATE` (`model_specs.py:485-512`) replace the
  `is_after_omission`/`is_after_reward` main effects in R1/R2 with `reward_lag1` (past-only, goal-arrival
  cutoff — the exact substitution Family B already uses; see the `:520` note). Keep `is_after_switch`
  (pre-outcome-safe). **The swap REPARAMETERIZES, it does not just re-cohort (Medium, round 7):**
  `reward_lag1` is the past reward indicator ≈ `is_after_reward = 1 − is_after_omission`, so even on stops
  where the outcome is already available the coefficient **sign flips** (a positive omission effect becomes
  a negative reward effect) and the intercept absorbs the shift. Do **not** describe this as "differs only
  on the ~10% approach stops" — that is only where the *causal vs leaky* value genuinely disagrees; the
  reparameterization changes the coefficient everywhere. Document the sign flip in `docs/rate_models.md`
  so the new `reward_lag1` coefficient is not read against the old `is_after_omission` sign. *If the user
  prefers to keep R1/R2 as-is for literature comparability*, instead tag the per-stop runner
  `retrospective=True` with a docstring that R1/R2 condition on the current outcome (post-arrival stops
  only) — but the `reward_lag1` swap is recommended.
- **Gate the rungs — but with the RIGHT remedy per leak type (F1).** Call
  `assert_predictor_timing(..., retrospective=..., descriptive=...)` in the content and rate ladder
  builders. There are **two distinct leaks with two distinct fixes** — do not treat them the same:
  - **POST_OUTCOME** (`hmm_rpe`/`rpe`, `hmm_value_update`, `is_after_reward`): current-reward info, valid
    once the outcome is known → **fixable by restricting to post-current-outcome bins**
    (`retrospective=True`, using the corrected bout mask below). Rungs G5/G7, S7/S9, C6/C9.
  - **FUTURE_REFERENCING** (`trials_since_switch`, `is_post_switch`): signed toward the *next* switch
    (`qvals_table.py:169`) → **no temporal gate makes these predictive-safe.** A post-reward bin still
    carries a `trials_since_switch` counting toward a switch that hasn't happened. These rungs
    (G1/S3/C10) must be `descriptive=True` (relabeled *peri-switch alignment* in `model_specs.py` +
    `docs/models_table.md`), **or** swap the covariate for a strictly past-only `trials_since_last_switch`
    (the `since` component only, `qvals_table.py:165`) and add that as a new backward-safe atom. Choose
    per rung; the default is relabel-descriptive (preserves the existing peri-switch descriptive intent).
  Restricting a FUTURE_REFERENCING rung to post-reward bins is **not** an acceptable fix and
  `assert_predictor_timing` will still reject it unless `descriptive=True`.
  **The gate is `post_current_OUTCOME`, NOT `post_current_reward` (High, round 6).** `hmm_rpe = reward − Q`
  with `reward = 0` on omissions (`qvals_table.py:292`), so gating on "a pump occurred in this bout" keeps
  **only rewarded trials** and silently deletes every negative/omission RPE — truncating the RPE
  distribution and biasing the RPE/update coefficients. The outcome (reward *or* omission) is determined
  at the goal-arrival cutoff, so define the per-trial outcome-known time and gate on that:
  ```python
  # outcome is KNOWN at: the in-bout pump time on rewarded trials; the goal-arrival /
  # trial-cutoff time on omitted trials (no pump ever fires). Reuse _trial_cutoff_times /
  # goal_arrival_time (_helpers.py:455) — the same causal cutoff reward_lag1 already uses.
  outcome_time = np.where(is_rewarded_trial, bout_pump_time, goal_arrival_time)
  post_current_outcome = query_time >= outcome_time    # True for BOTH rewarded and omitted trials
  ```
  Also **do NOT gate on `t_since_pump >= 0`** — `_time_since_pump` (`timebin_tables.py:65`) measures
  seconds since the *most recent session-wide pump*, positive for approach bins after *any* earlier
  reward (F1); and it is undefined on omission trials.
  - **Rate (Family B):** compute `post_current_outcome` per (stop, bin) from the trial's outcome-known
    time and filter to it when `retrospective=True`.
  - **Content:** `event_arm_table` carries only `time_since_stop` (`fit_swr_contrast_content.py:146`), so
    merge each event's current-trial outcome-known time (bout pump time if rewarded, else goal-arrival)
    onto the event, then gate events whose onset is at/after it. Do **not** approximate with the stop-level
    `is_after_reward` (the leaky covariate being removed).
  Recommended: (a) gate for the rate rungs; (a) or (b) relabel-descriptive for content (see overview
  Open Question 1 — the content merge scope).
- **Mechanism A — ENCODE (decided).** In `covariates.py:recent_reward_rate_all_events`, change the
  unvisited-leaf branch (currently NaN at `:196`) to return `0.0` for a leaf arm with zero prior visits
  *in a session that has prior trials* (trunk arms already return `0.0`; the no-prior-trials rows stay
  NaN). Add a sibling helper `has_arm_history_all_events(...)` returning a `(n_events, n_arms)` 0/1
  indicator (1 where the arm has ≥1 prior visit). Add `KNOWN_ATOMS`/`PRE_OUTCOME_SAFE` entries for
  `has_arm_history` (backward, safe). Attach `has_arm_history` to `event_arm_table` next to
  `reward_rate_arm_k10` (`tables.py:~1300`) and to `_event_arm_table_columns`; add it as a **main effect
  only** to the content rungs from M3 onward in `model_specs.py`. **Do NOT add a
  `has_arm_history:reward_rate_arm_k10` interaction (F3):** with this encoding `reward_rate_arm_k10` is
  nonzero only where `has_arm_history==1`, so the interaction column equals the `reward_rate_arm_k10`
  main effect exactly — perfectly collinear. Regenerate `event_arm_table`/`event_patch_table`
  (after phase 6's builder consolidation — see intro). Update `docs/covariate_dictionary.md` with the new
  `has_arm_history` covariate and the changed `reward_rate_arm_k10` semantics (unvisited leaf = 0 with
  indicator, not NaN). This **retains the ~11.6% of events** the old NaN-drop discarded — the novel-arm /
  early-session regime is now a modeled category. **Report base rungs both encoded and (once, for
  comparison) on the old restricted cohort** so the estimand change is measured, not silent. *Restrict
  was considered and rejected (overview Resolved decisions #1).*
- **Refresh the standardization reference after encoding (Low, round 7).** Phase 6 runs *before* this and
  persists `standardization_reference.json` with `reward_rate_arm_k10`'s pooled mean/SD — but the encoding
  changes `reward_rate_arm_k10` (unvisited leaf → 0.0 instead of NaN), so that reference and the
  `reward_rate_arm_k10_z` column are **stale after this phase**. Re-run the event_arm standardization on the
  encoded table and overwrite the reference's `reward_rate_arm_k10` entry (the other ARM_CONT columns are
  unchanged by the encoding). Validate that the reference's `reward_rate_arm_k10` mean/SD actually changed
  and `reward_rate_arm_k10_z` is recomputed from the new pooled stats.
- **Annotate Stage 6.** `learning.py` `is_stay` regressions use `is_after_omission` (`:173`) but the
  modeled unit (the stay/leave decision) is *post-outcome by construction*, so this is valid. Add a
  one-line docstring at the Stage-6 model asserting the response is post-outcome; do **not** change it.

## Deliberately not in this phase

- The pseudoreplication/inference change and the `swr_x_value` construct/cohort sensitivity — [phase 5](phase-5-inference-validity.md).
- Folding `_z`/`q_arm_hmm` into the builder — [phase 6](phase-6-repro-plumbing.md), which runs **before** this phase (the builder must own `_z`/`q_arm_hmm` before this adds `has_arm_history`).
- Touching the response-only future columns (`next_arm`, `ends_at_next`, …) — confirmed correct; leave alone.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/unit/test_covariate_timing.py::test_rejects_derived_post_outcome` | `assert_predictor_timing(["update_x_current"], retrospective=False)` raises (via `hmm_value_update` source), and `tss_x_current`/`postsw_x_current` likewise — the F2 failures |
| `::test_rate_swr_interaction_caught` | `assert_predictor_timing(["is_swr:hmm_rpe"], retrospective=False)` raises — the patsy expansion of the S7/S9 `is_swr * (...+rpe/update)` rungs; `timing_of("is_swr:hmm_rpe") == ["hmm_rpe"]`, not `["is_swr"]` (F2) |
| `::test_post_outcome_fixable_by_retrospective` | `hmm_rpe` passes with `retrospective=True` but `trials_since_switch`/`tss_x_current` still **raise** with `retrospective=True` — future-referencing is not gate-fixable; only `descriptive=True` (or a past-only replacement) clears it (F1) |
| `::test_every_ladder_predictor_resolvable` | every predictor in `SWR_CONTRAST_CONTENT_ARM_LADDER` + the expanded rate `_s_specs`/`_g_specs` terms resolves via `timing_of` into `KNOWN_ATOMS` (no silent unmapped interaction — incl. `swr_x_q_arm_hmm`) |
| `::test_is_after_switch_is_safe` | `is_after_switch` not rejected (pre-outcome) |
| `tests/unit/test_post_outcome_mask.py::test_not_tsp_ge_zero` | an approach bin with `t_since_pump>0` (from a *previous* reward) but before the current trial's outcome time is **excluded** — the gate is not `t_since_pump>=0` (F1) |
| `::test_omissions_retained` | on a synthetic omitted trial (no pump), bins after `goal_arrival_time` are **retained** by `post_current_outcome` — the gate does not select rewarded-only (round-6 High) |
| `tests/integration/test_retrospective_cohort_has_both_outcomes.py` (slow) | a `retrospective=True` RPE/update fit's retained rows contain **both** rewarded and omitted trials (assert `reward_outcome` has both 0 and 1); guards against silent RPE-distribution truncation |
| `tests/unit/test_recent_reward_rate.py::test_unvisited_leaf` | unvisited leaf → 0.0 and `has_arm_history==0`; a visited leaf → its fraction and `has_arm_history==1`; trunk still 0.0; no-prior-trials rows still NaN |
| `tests/integration/test_rate_R1_pre_outcome.py::test_reparameterization` (slow) | fitting the OLD R1 with `is_after_reward = 1 − is_after_omission` **on the post-arrival subset** reproduces the new `reward_lag1` fit (coef matches, and equals `−1×` the old `is_after_omission` coef); confirms the swap is a sign-flip reparameterization, not a silent estimand change |
| `::test_disagreement_subset` | the causal (`reward_lag1`) and leaky (`is_after_reward` from trip-in-progress) versions differ **only** on the ~10% approach-phase stops where the current outcome isn't yet delivered |
| cohort audit (phase 3) on refits | content M3 `n_modeled` rises by ~7.6k events vs the old restricted cohort; the audit row shows the retained events |
| `tests/integration/test_z_reference_refreshed.py` (slow) | after the encoded rebuild, `standardization_reference.json`'s `reward_rate_arm_k10` mean/SD **differ** from the phase-6 pre-encoding values, and `reward_rate_arm_k10_z` equals `(reward_rate_arm_k10 − new_mean)/new_sd` (Low, round 7 — stale reference caught) |

## Fixtures

Synthesize a small trials frame in `conftest.py` with (a) an arm never visited before event t, (b) an
arm visited and rewarded, (c) trunk arms — to exercise the encode/NaN/zero branches deterministically.
Use one cached session for the R1 integration refit.

## Review

Dispatch `code-reviewer`. Confirm:
- The **encode** path is implemented (unvisited leaf → 0.0 + `has_arm_history`); no restrict-only fallback left behind.
- The `reward_lag1` swap is a sign-flip **reparameterization** (`reward_lag1 ≈ 1 − is_after_omission`): the coefficient changes everywhere (reviewer checks `test_reparameterization`), while the causal-vs-leaky *values* disagree only on the ~10% approach stops (`test_disagreement_subset`). The `docs/rate_models.md` sign-flip note is present.
- `assert_predictor_timing` is called in the ladder builders. POST_OUTCOME rungs are gated to post-current-outcome bins (outcome-known time = pump if rewarded, else goal-arrival — retains omissions); FUTURE_REFERENCING rungs (`trials_since_switch`/`is_post_switch`) are **relabeled descriptive or swapped for a past-only covariate** — a future-referencing rung is never "fixed" by outcome gating (F1).
- `event_arm_table` regenerated, `_event_arm_table_columns` updated, `docs/covariate_dictionary.md` updated, and the ×9-arm/×3-patch invariants still hold (re-run the phase-invariant check).
- Stage-6 annotation added, behavior unchanged.
