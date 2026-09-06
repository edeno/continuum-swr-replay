# Model design history

The design-decision narrative for the replay-content model ladders, moved here
from the `analysis/replay/model_specs.py` module docstring so that module reads
as the terse `ModelSpec` catalog it is. This is the scientific audit trail —
including reversed reasoning and retracted numbers — and is preserved verbatim.
The machine-readable catalog (which predictors each rung carries) lives in
`model_specs.py`; this file is why.

The notes below concern the conditional-Poisson **arm and patch** hierarchies.
Their defining constraint: only arm-varying predictors are identifiable —
event-level constants (time since stop start, `is_swr`, etc.) cancel in the
within-event normalization and appear only as interactions with arm-varying
predictors. That rule does NOT apply to `MODEL_PER_STOP_RATE` (ordinary Poisson,
carries an intercept) or `MODEL_CHOICE`; see their own sections in the catalog.

## Design decisions (exploratory analysis on 7 j16 sessions)

- **Arm fixed effects** (`is_candidate_arm_0` … `is_candidate_arm_7`, 8 dummies
  with arm 8 as reference) enter as a baseline model `M0a` and are carried by
  every richer model. They absorb each arm's static replay occupancy so
  `is_current_arm` / `is_prev_arm` / `is_last_rewarded_arm` measure bias *above*
  an arm's own baseline rather than being inflated by it — important in
  camp-heavy sessions where the animal sits at one well and `is_current_arm`
  then concentrates on a single arm. See `ARM_FE_PREDICTORS`.

- **Reward history uses `reward_fraction_candidate_arm_last10_visits`** (rewarded
  fraction of the last 10 visits to an arm) at M3/M4 rather than
  `cumulative_reward` (a raw reward *count*). The count conflates value with
  visit frequency — a frequently-visited arm scores high even at a mediocre
  reward fraction — so its M3 coefficient partly re-encodes spatial recency.
  Once any trial has completed, a never-visited leaf is encoded as 0.0 rather
  than NaN; M4 adds `is_candidate_arm_never_visited` so that right-censored
  novelty is distinguishable from an observed zero reward rate. Events before
  the first completed trial remain unknown and are removed by complete-case
  selection.

- **M4 uses an event-local recency spline** rather than a session-wide
  transformation. At each event, the mean recency over the *visited* reward-site
  arms is subtracted from every leaf-arm recency; never-visited leaves are
  excluded from that mean because their `n_completed + 1` value is a censoring
  rule, not a measured recency. `is_candidate_arm_never_visited` carries that
  state separately and trunk arms remain structural zero. Appending future
  events cannot change an earlier design row, and the coefficient represents
  relative recency within the event above the arm-FE baseline.

- **Reward memory has NO rung in the per-session arm ladder.** Neither form of
  the last-rewarded indicator is usable here: the raw `is_last_rewarded_arm`
  correlates 0.919 with `is_current_arm` (it marks the same arm on 93.2% of
  location-resolved events), and the `is_last_rewarded_noncurrent_arm` residual
  has support on only 6.1% of events and is perfectly separated in 99 of 208
  sessions. Both survive in the pooled SWR-contrast rungs (C5_lastrew,
  CS3_lastrew), where pooling across sessions supplies the variation a
  per-session fit lacks.

  History, because the reasoning reversed twice and the intermediate numbers are
  still quoted elsewhere. The "elsewhere" residual was introduced to break a
  ~98% overlap between `is_last_rewarded_arm` and `is_current_arm`. On 2026-08-17
  that overlap was measured at 33.7% and the residual at 58.2% of events, which
  read as the collinearity having dissolved. Both figures were artifacts: they
  came from tables where an unresolved location made `is_current_arm` 0 rather
  than NaN, so every in-transit event silently counted the last-rewarded arm as
  "elsewhere". Off-well events are 9.1% of the corpus and carry
  `is_last_rewarded_arm` on 93.4% of themselves. Once they are NaN, the overlap
  is 93.2% and the residual 6.1% — the original ~98% regime, essentially intact.
  **Do not cite the 33.7% or 58.2% figures.**

- **M3 is the only rung carrying `reward_fraction_candidate_arm_last10_visits`
  without `is_candidate_arm_never_visited`.** A never-visited leaf is encoded
  `0.0` there, identical to "visited and never rewarded", and M4 adds the
  disambiguator. So the M3→M4 step is not a pure novelty test: part of it is M4
  cleaning up an encoding M3 was already using ambiguously. The nesting is
  deliberate and the step is a valid likelihood comparison, but do not report it
  as "replay tracks novelty" on its own — cite `is_candidate_arm_never_visited`'s
  own coefficient for that.

- **`is_next_arm` excluded:** the alternation structure makes
  `prev_arm == next_arm` on **84.5%** of the events where both are defined
  (measured 2026-08-18; an earlier ">90%" here was never re-derived). They
  separate on 15.5% of events, but **98.4% of those are also patch switches**,
  so the separating population is the same event class `is_prev_arm` plus the
  completed-trip switch flag already indexes — which is why the exclusion still
  stands on the corrected number.

- **`current_x_last_rewarded` excluded:** VIF 97-100 with both `is_current_arm`
  and `is_last_rewarded_arm`. Measured under the ~98% overlap regime, which the
  93.2% figure above shows still holds, so the exclusion stands.

- **`distance_to_candidate_arm_cm` and `is_same_patch` excluded:** proven
  uninformative (null when estimated independently; VIF 28 and 12 with
  `is_current_arm`). The spatial effect is binary "here vs not here," not a
  distance gradient.

- **SWR modulation is split across two rungs:** M5 adds the spatial interactions
  (`swr_x_is_current_arm`, `swr_x_is_prev_arm`), and M6 adds the reward-rate and
  relative-recency interactions. Superseded note: an elsewhere-reward-memory
  interaction was once a third term here and was removed, because
  `is_last_rewarded_noncurrent_arm` itself has support on only ~2% of events, so
  convergence and uncertainty require real-data re-validation after canonical
  table regeneration.

- **MUA interactions are a separate analysis** (split, not interaction) to avoid
  the same collinearity.

## Retirements in the per-bin rate ladder (Family B)

- **The five `onset_lag1..10` dummy rungs are retired (2026-09-06):**
  `M3_hist`, `M4_hist_rew`, `M6_time_hist`, `M7_time_hist_rew` and
  `M9_rew_x_time`. Ten collinear lag dummies over a 150 ms window separate on
  the refractory bins — the first lags have essentially no events, the
  coefficients run to −inf — and a fit carrying a separated non-nuisance term
  has its whole cross-validation voided. Measured on the 2026-09-06 refit of
  218 sessions, `cv_bits_per_event` was **NaN in 99.08 % of the non-local fits**
  of each of the five (99.08 % non-local, 99.54 % non-local-no-SWR, 99.54 %
  non-local-with-SWR, 100.00 % detected SWR; 99.54 % pooled over the four
  populations). A rung that can never be scored out of sample cannot take part
  in the ladder's held-out comparison, which is one of the two things the
  ladder is read on.

  The raised-cosine variants `M3r_histcos` and `M7r_time_histcos_rew` span the
  same 150 ms window with four smooth, non-separating parameters and **are**
  scored (61.9 % / 63.8 % NaN on the non-local population — high, but not
  categorical). They were built for exactly this reason and now carry the
  self-history block alone. `M7r_time_histcos_rew` becomes the ladder's full
  additive rung and is what the summarizer, the figures and the refit audit
  read.

  The retirement also removed `M7r`'s natural null: with `M6_time_hist` gone it
  parented straight onto `M5_time`, so "the reward step" was the reward block
  and the self-history block together. **`M6r_time_histcos`** (2026-09-06)
  restores the missing rung — `M5_time` + the raised-cosine self-history block —
  so the reward-given-time-and-self-history step `M7r − M6r` exists again.

  The reward × time interaction survives, re-parented onto the raised-cosine
  full model as **`M9r_rew_x_time`** (`M7r` + `reward_history_rcos_0:bs(tc)`).
  Its specification is otherwise unchanged.

  Consequence for the per-lag self-history kernel: it is no longer read off ten
  `coef_onset_lag{j}` values but **reconstructed** from the four
  `coef_hist_rcos_k` per session
  (`_helpers.reconstruct_self_history_kernels`), which projects through the
  **un-normalized** raised-cosine basis — `hist_rcos_k` is a plain projection of
  the lag counts, unlike the reward block's bump-weighted fractions. The
  reported kernel is therefore smooth: it cannot represent a lag-specific spike.
  Measured against the retired dummies on the same sessions, that costs less
  than it sounds: **both** parameterizations read RR ≈ 0 at lags 1–2 in all
  four populations, and the smooth kernel is shallower only from lag 3 on,
  in 4/4 populations (non-local 0.31 vs 0.22, non-local with SWR 0.15 vs 0.00,
  non-local without SWR 0.02 vs 0.00, detected SWR 0.01 vs 0.00).
  Refractory/bursty shape and its ordering across
  populations are what the reconstruction preserves.

## Where the rest of the narrative lives

The per-rung inline comments (the reasoning attached to individual `ModelSpec`
definitions) remain in `model_specs.py` next to the code they explain — moving
them here would orphan the explanation from the definition. This file holds the
module-level design journal; the code-adjacent notes stay code-adjacent.
