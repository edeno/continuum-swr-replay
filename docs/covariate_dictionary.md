# Covariate dictionary — canonical definitions across replay model families

Single source of truth for how each behavioral covariate is defined and how it
enters each model family, so coefficients are **comparable across rate / content /
choice / coupling**. Agreed decisions D1–D5 (2026-06-29) are recorded at the end.
This is the target the model specs and covariate-computation layer implement
against; the harmonization refactor brings the code to this spec.

**Status: definitions implemented; canonical tables and model artifacts require
regeneration before paper results are updated.** The covariate
definitions live in `analysis/replay/covariates.py`; the per-arm value block
(`reward_rate_arm_k10` at k=10, `p_reward_arm`, `q_arm`) is emitted by the event
builders from the per-stop contingency artifact (`p_reward_port`/`q_port`,
per-animal fitted α in `contingency_csv_alpha.json`); the value rungs are `Mv`
(arm) / `Pv` (patch); `q_current` enters the MU/coupling runners; the choice
baseline is `arm_dist_rank`. Existing coefficients predate the location and
two-clock history contracts and must not be treated as results from these
definitions.

## A. Entry-rule contract (forced by estimator type)

A covariate that is constant across arms within an event **cancels** in a
conditional/multinomial model, so it cannot be a main effect there. Hence:

| estimator | family | scalar covariates (reward / value / switch) | arm- or patch-relative covariates |
|---|---|---|---|
| **absolute** | per-bin rate (B), per-stop rate (A), coupling/power | **main effects** | — |
| **conditional (content)** | arm hierarchy, patch hierarchy | only as **`× is_current`** (or `× role`) interactions | **main effects** |
| **conditional (choice)** | next-arm logit | (cancel — omit) | **main effects** over the choice set |

"Parallel" therefore means **same construct, same window and units**, entered in
the form each estimator allows — *not* literally the same regressor everywhere. This
makes coefficients **admissible and same-construct**, not numerically interchangeable:
an absolute-rate main effect (log rate ratio) and a conditional-content interaction
(arm-relative log-odds) are different estimands — **interpret within an estimator
class**, compare *direction/consistency* across classes, not raw magnitudes.

## B. Canonical constructs

### 1. Reward memory
Named by **scope + window** so the same word never hides a different definition.

| canonical name | definition | window (N=10) | scope | units | enters |
|---|---|---|---|---|---|
| `reward_lag{1..10}` | outcome of the j-th most recent trial whose outcome was observable by the query time | 10 known outcomes, per-lag | all-arm (per-trial) | binary | rate A (Rlag), rate B, MU, coupling — main effects |
| `reward_rate_trial_k10` | rewarded fraction of the last 10 trials with known outcomes | 10 known outcomes | all-arm | fraction | rate A (R2), standalone only — **never co-fit with the kernel** (it is the kernel's mean → VIF ∞) |
| `reward_rate_arm_k10` | rewarded fraction of the last 10 **visits to that arm**; an unvisited leaf (no prior visit, but the session has prior trials) is encoded `0.0`, while the explicit `arm_never_visited` term separates novelty from an observed zero. Trunk arms are structurally `0.0`; no-prior-trials rows are NaN. | 10 arm-visits | per-arm | fraction | content (main effect), choice (main effect); rate/coupling via `× is_current` |

Change from prior code (implemented): the per-arm rate was **5-visit**
(`recent_reward_rate`) and is now **10-visit** (`reward_rate_arm_k10`). The all-arm
10-trial rate (`recent_trial_reward_rate`) was renamed `reward_rate_trial_k10`. Trunk
arms keep the structural `0.0`; never-visited leaf arms are `0.0` plus
`arm_never_visited = 1`.

### 2. Value — both nominal and experienced, available to every family

| canonical name | definition | scope | source | enters |
|---|---|---|---|---|
| `p_reward_current` | nominal contingency p(reward) of the current arm | scalar | `qvals_table` (DataJoint) | rate/coupling main effect |
| `value_diff_patch` | nominal **patch**-level best-minus-second-best: the gap between the highest and second-highest patch mean p(reward). Always >= 0; a global scalar per trial, entered as a main effect only (never `× is_current_patch`) | scalar | `qvals_table` (nominal schedule) | rate/coupling main effect |
| `p_reward_arm` | nominal contingency p(reward) of **each** arm | per-arm | `qvals_table` (DataJoint) | content/choice main effect; rate via `× is_current` |
| `q_current` | beta-model **belief** (value) of the current/chosen arm (`q_chosen` = `Q{leaf}`) | scalar | behavior model (`qvals_table`, `beta_stable_withleaf`) | rate/coupling main effect |
| `q_arm` | beta-model **belief** of **each** arm (`q_port{i}` = `Q{i+1}`) | per-arm | behavior model (`beta_stable_withleaf`) | content value main effect — **the value construct used by C4/CS5/Mv** (and the is_swr rate ladder's value covariates); **not** used in the Stage-5A choice model. (The HMM `q_arm_hmm` construct was **retired 2026-07-13** when the paper standardized on the beta-bernoulli model.) |
| `leaky_q_value_diff` | pre-update leaky-Q best-minus-second **margin** for the most recently completed trial (α=0.3), computed over the full trial stream and broadcast to its stops | scalar | `canonical_trial_uncertainty_history` | **retained comparison** (MU only), like `is_after_switch` under D4 |

Changes (implemented): `p_reward_arm` **replaced the formerly-deferred
`nominal_reward_prob` column** and is now populated (`nominal_reward_prob`→`p_reward_arm`
in the `tables.py` schema + the viz registry, no orphan). `q_current`/`q_arm` are the
experienced-value covariate — now the **beta-model belief** (`beta_stable_withleaf`),
not a local leaky-Q learner: all value covariates were migrated to `qvals_table.parquet`
(DataJoint, exact `poke_out` alignment; retires the difflib contingency CSV). The
model's per-port `Q{i}` populate `q_port{i}` → `q_arm`, and `Q{leaf}` = `q_chosen` →
`q_current`. **`leaky_q_value_diff` is NOT this** — it is a *different*, still-local
construct (one pre-update best-minus-second margin per completed trial, α=0.3,
computed over every completed trial in `covariates.py`, persisted on stop rows,
and then broadcast to events by `stop_id`), kept as a reported MU-only comparison. `value_diff_patch`
(nominal) remains a scalar
rate/coupling covariate. **`rpe` is now the model RPE everywhere** (`reward − q_chosen`,
from `qvals_table.parquet`; see [behavior_model_covariates.md](behavior_model_covariates.md)) —
after the migration every consumer (MU-uncertainty as `rpe_model`, plus `fit_swr_power`
and `_covariate_ranking`) uses the model RPE, not a nominal one.

### 3. Exploration — canonical continuous + retained binary comparison

| canonical name | definition | window | units | enters |
|---|---|---|---|---|
| `recent_switch_rate` | rolling mean of `is_patch_change` over the last (up to) 5 completed trials, including the most recently completed trial | 5 completed trials | fraction | **canonical** cross-family: rate/coupling main effect; content via `× is_current` |
| `is_after_switch` | the most recently completed trip changed patch | 1 completed trial | binary | **retained comparison** (per D4): the immediate-binary complement, available alongside `recent_switch_rate` wherever the estimator allows |
| `omission_streak` | leading consecutive omissions: count of leading 0s in `reward_lag1, reward_lag2, …`, stopping at the first 1. A `NaN` lag truncates the count. **`NaN` when `reward_lag1` itself is unresolved** — with no known most-recent outcome there is nothing to count from, and returning 0 would be indistinguishable from "the last trial was rewarded" (1,359 / 107,400 stops, the earliest of each session) | up to 10 known outcomes | count | model-free uncertainty block; excluded from the MU rate fit for collinearity with `reward_lag1` |
| `wsls_deviation` | 0 on win-stay or lose-shift, 1 on win-shift or lose-stay. Reads `completed_trial_reward` (the trip's *realized* outcome, ungated) paired with `is_stay`. `NaN` where no trip has completed or the outbound choice is unresolved | 1 completed trial | binary | **descriptive only** — embeds the eventual choice, so future-referencing however the outcome is clocked |

(D4: both kept — `recent_switch_rate` is the canonical graded covariate; `is_after_switch`
is a valued immediate-contrast comparison, not dropped.)

### 4. Spatial / role — already consistent, no change

**Two occupancy columns, two constructs.** `location_arm` (aliased as
`current_arm`) is a *task state*: which well the animal is visiting, derived
from trip intervals, and **NaN during a trip** — the destination is never named
before arrival, which is what makes it safe as a predictor. `physical_arm` is a
*position*: the modal track segment the tracking falls on, always defined,
trunks included. They disagree on 46 of 59,374 at-well events (0.077%), where
the animal is still inside its well-visit interval but a mean 48.4 cm from the
well. Neither is a fallback for the other; a cohort needing occupancy off-well
must state which construct it means.

**`in_transit` does not mean "moving".** It labels the trip interval. The
modelled cohort is speed-gated immobility throughout, and off-well events have
a median speed of 1.35 cm/s — they are animals standing still partway through a
trip. The content models exclude them because the well-visit construct is
undefined there (so `is_current` and everything derived from it is NaN), not
because of motion. The at-well cohort is stated explicitly in the runner rather
than emerging from that NaN pattern.


`is_current`, `is_current_patch`, `is_prev`, `is_prev_patch`,
`is_last_rewarded_elsewhere` / `is_last_rewarded_patch_elsewhere` (disjoint
support), `arm_fe_{0..7}` (content baseline), `arm_dist_rank` (choice baseline,
replaces `dist_to_arm` for identifiability). Definitions unchanged.

Both `_elsewhere` columns are **pooled-only as of 2026-08-18**: they are built
and feed the pooled SWR-contrast rungs, but no per-session ladder rung uses
them. They fire on 6.1% (arm) and 2.9% (patch) of events and are perfectly
separated in roughly half the sessions, which made every rung in a
frozen-union cohort inherit the refusals. See `model_specs.py`.

They are also **omission-only by construction**, which the name does not convey:
a rewarded stop leaves the last-rewarded arm equal to the current arm, so the
residual is identically zero there. Measured: 0.0% firing on rewarded at-well
events, 67.4% on omission ones, and 100% of firing events are omissions. Read
any coefficient as "within omission stops", and note that the ~10% omission
rate — not the location-cohort rule — is the ceiling on their support.

### 5. Time / self-history — rate-only, no change

`bs(tc)` (fixed knots (1,2,3) s, [0,4]), the reward-lag kernel, `onset_lag{j}` /
`hist_rcos` self-history. These are per-bin within-stop terms and live only in the
rate families; conditional models access within-stop time only as `tsc_x_current`
(`× is_current`) when a within-stop content effect is tested.

## C. Per-family covariate matrix (target)

| construct | rate B (per-bin) | rate A (per-stop) | coupling | content (arm/patch) | choice |
|---|---|---|---|---|---|
| reward kernel `reward_lag{j}` | main | main (Rlag) | main | `× is_current` | — |
| `reward_rate_trial_k10` | (M2r standalone) | main (R2) | — | — | — |
| `reward_rate_arm_k10` | `× is_current` | — | `× is_current` | **main** | **main** |
| `p_reward_current` / `p_reward_arm` | main / `×is_current` | — | main | **`p_reward_arm` main** | **`p_reward_arm` main** |
| `q_current` / `q_arm` | main | — | main | **`q_arm` main** | **`q_arm` main** |
| `rpe`, `value_diff_patch` | main | — | main | (nominal via interaction if tested) | — |
| `recent_switch_rate` | main | main | main | `× is_current` | (main if predictive) |
| `is_after_switch` | main (M8) | main (R1+) | comparison | `× is_current` comparison | — |
| spatial role / `arm_fe` | — | — | — | main | choice-set + `arm_dist_rank` |
| `bs(tc)`, self-history | main | — | — | `tsc_x_current` only | — |

## D. Decisions of record (2026-06-29)

- **D1** — recent-reward-rate window standardized to **N = 10** in each construct's
  natural unit: `reward_rate_trial_k10` (10 trials, all-arm) and `reward_rate_arm_k10`
  (10 visits, per-arm). Per-arm window changes 5→10. **Caveat:** "10" is not the same
  *real-time* span (a single arm is visited far less often than trials elapse) — it is
  10 in each construct's own unit, not 10 in time.
- **D2** — add **per-arm nominal value** `p_reward_arm` to content/choice — by
  **replacing the formerly-deferred `nominal_reward_prob` column** (renamed and now
  populated), not a new parallel column.
- **D3** — **add** the experienced-value covariate `q_current` / `q_arm` (now the
  **beta-model belief** from `qvals_table`, not a local leaky-Q learner) as canonical,
  and **keep `leaky_q_value_diff`** (the completed-trial pre-update margin, a
  different local construct) as a *reported comparison*. (Revised from "drop":
  the two are different
  algorithms, and the margin is deliberately kept in the MU dissociation table.)
- **D4** — exploration canonical = **`recent_switch_rate`**; **keep `is_after_switch`**
  as the immediate-binary comparison (not dropped).
- **D5** — adopt the **entry-rule contract** (§A) as the standing rule.

## E. Notes / non-goals

- Coefficients remain comparable **only within an estimator class**; across classes
  (absolute vs conditional) the same construct is read as a rate ratio vs an
  arm-relative log-odds — interpret accordingly.
- The Q-learner α is fit per animal (not shared), so `q_*` is the animal's own
  learning timescale; report the per-animal α range.
- Everything stays **correlational**; this dictionary harmonizes *definitions*, not
  causal status.
- The full SpatialBanditTask **behavioral-model** latent variables (Beta-Bernoulli
  value, ΔV / Stay / Switch, value updates, Beta variance, beta belief entropy) live
  in a separate per-stop table; see
  [`behavior_model_covariates.md`](behavior_model_covariates.md).
- Every regression model that consumes these covariates, in R formula notation (per
  family, with the phase-2 changed/parity classification): [`models_table.md`](models_table.md).

## F. Temporal causality (pre- vs post-outcome) — leakage audit

Each stop-level covariate is broadcast to the non-local events within that stop.
A covariate is safe iff its value is determined by information available **before**
the event it predicts. Classification of the covariates fitted by the rate models:

| covariate | class | available before event? |
|---|---|---|
| `p_reward_current`, `p_reward_arm`, `value_diff_patch`, `trials_since_block_change` | nominal schedule | yes — fixed by the block, known at trial start |
| `q_current` / `q_arm`, `switch_advantage_*_preoutcome_sess`, Beta `*_mu`/`*_var` `preoutcome` | model, **pre-outcome** belief (row *t* = belief entering trial *t*) | yes |
| destination/visit recency, `is_after_switch`, `recent_switch_rate` | **arrival clock** | yes — destination and inbound patch change become known at goal arrival |
| reward lags/rates/counts, last-rewarded well | **outcome clock** | yes — advance only at first pump onset (reward) or destination poke-out (omission) |
| `beta_reward_entropy` | beta belief entropy (mean per-leaf Beta differential entropy) | yes (pre-outcome posterior) |
| `rpe` (= reward − q_chosen), `*_mu_update_abs_postoutcome_sess` | model, **post-outcome** (uses the current trial's realized reward) | **only if the event follows reward delivery** |
| `reward_outcome`, `is_after_reward`, `is_after_omission` | most recently arrived trip's outcome | finite **only after** that outcome is observable; otherwise NaN |
| `is_stay`, `wsls_deviation` | eventual outbound choice | no — future-referencing; descriptive use only |
| `completed_trial_reward` | completed trip's realized outcome, **ungated** | no — post-outcome; exists only so `wsls_deviation` can name the outcome the animal just experienced |

**Arrival is not the outcome boundary.** Canonical trials therefore carry two
clocks. `goal_arrival_time` controls destination, visit, and inbound-switch
history. `outcome_known_time` is the first pump onset for a rewarded trial and
the destination poke-out for an omission (there is no separate negative-feedback
cue). Between those boundaries, arrival-derived fields are populated while all
current reward-bearing fields remain NaN. Older/manual trial frames without a
physical outcome column use `end_time` for both outcomes: conservative for
rewards, but never anticipatory.

Outcome-bearing fields remain in `POST_OUTCOME` in `covariate_timing.py` because
their finite values semantically encode the realized current outcome. A model
may use them only with a physical `outcome_known_time` gate
(`retrospective=True`). The uncertainty-rate and SWR-power runners audit counts
before/after that gate separately for rewarded and omission observations, and
skip a session/population cell if either class disappears. Future-referencing
outbound-choice fields (`is_stay`, `wsls_deviation`, `next_*`) cannot be repaired
by this gate and require explicitly descriptive use.

**The previously recorded
"0.0 % of events occur before the reward pump" measurement predates the unified
speed gating and the event-location work and must be re-derived** before it is
cited again — the immobility/event-window definition it warned about has since
changed. Re-check this exposure whenever that definition changes.

### F.0 Whose knowledge? Scheduled is not the same as known

Timing-safety answers "was this value determined before the response?" It does
not answer "could the animal have been representing it?" Three classes, and the
distinction matters for what a coefficient licenses you to claim:

| class | covariates | a coefficient means |
| --- | --- | --- |
| **experienced** | `is_current`, `is_prev`, `dist_to_arm`, `time_since_stop`, `reward_lag{j}`, `omission_streak`, `cumulative_reward`, `reward_rate_arm_k10`, `last_rewarded_arm`, `is_after_switch`, `visit_recency`, `arm_never_visited` | content covaries with something the animal directly underwent |
| **inferred** (model belief) | `q_arm`, `q_current`, `arm_uncertainty`, `beta_reward_entropy`, `switch_advantage_mu_preoutcome_sess`, `leaky_q_value_diff`, `rpe`, `global_mu_update_*` | content covaries with a *hypothesised* internal state, conditional on the belief model being right |
| **scheduled** (experimenter-only) | `p_reward_arm`, `p_reward_current`, `value_diff_patch`, `trials_since_block_change` | content covaries with the **programmed contingency** |

The third row is the one that gets over-claimed. The animal never observes a
block boundary or a nominal p(reward); it only ever holds a noisy estimate. A
coefficient on `trials_since_block_change` is informative *insofar as the
animal has learned the schedule* — it is not evidence that the animal
represents trials-since-block-change. Report these as schedule effects, and if
the claim is about representation, use the inferred-belief covariates instead.

(Timing itself is fine in all three: `trials_since_block_change` is a
backward `cumcount` within a block, unlike `trials_since_switch`, which is
signed toward the nearest switch — including future ones — and is therefore
future-referencing.)

### F.1 Two clocks can coexist in one design row

The value block (`q_arm`, `p_reward_arm`, `arm_uncertainty`) is the belief
*entering* the trial, so it is pre-outcome by construction. The reward-history
block (`cumulative_reward`, `reward_rate_arm_k10`, `reward_lag{j}`) advances on
the outcome clock. On **83.9 % of events** (54,537 / 65,020 matched) the event starts after its
trial's `outcome_known_time`, and 54,451 of those trials were rewarded — so
within a single design row the two blocks summarise history through
**different endpoints, differing by the just-received reward**.

> **That 83.9 % must be re-derived.** It was measured against the *previous*
> omission boundary (destination poke-out). The boundary is now poke-in +
> `OMISSION_KNOWN_LATENCY_SEC`, which is earlier by a median 0.87 s, so more
> events clear it and the share will move — upward. The qualitative claim (the
> two blocks close at different endpoints, differing by the just-received
> reward) is unaffected; only the number is stale.

This is not leakage: the value block carries strictly *less* recent
information than the reward block, never more. It is verified pre-update —
`q_port[chosen]` moves +0.078 after a reward and −0.091 after an omission on
the following trial, and `rpe == trial_reward − q_chosen` exactly.

It is an interpretation hazard. A coefficient on `q_arm` read against one on
`reward_rate_arm_k10` is not "value vs. reward history at the same moment";
the value term is one outcome behind. Say so when the two appear in the same
rung (M v, C4/CS5, C13/CS6).

A related, smaller offset: the value block is frozen at `stop_start` (it is
joined by `stop_id` from the per-stop qvals artifact) while every other
event-arm history covariate is frozen at `event_start`. Median gap 2.39 s. The
practical impact is small — qvals' assigned arm equals the event's own
`current_arm` on 100.00 % of at-well events whose stop is `mixed` — but the
two are not the same instant.

### F.2 Scope condition: the at-well cohort is not SWR-neutral

`is_current` is undefined off-well, so every rung containing it fits the
at-well cohort (see
[`known_issues/current_arm_trunk_fallback.md`](known_issues/current_arm_trunk_fallback.md)).
That exclusion is **not balanced across the SWR contrast**: it removes

| | off-well share |
| --- | --- |
| SWR events | **3.28 %** (866 / 26,403) |
| non-SWR events | **12.98 %** (5,129 / 39,510) |

(χ² = 1800, p ≈ 0; measured 2026-08-17 on the regenerated tables.) The SWR
share of the cohort therefore rises from 40.06 % to 42.62 %, and 5,129 non-SWR
events — disproportionately locomotor-pause, locally-anchored ones — never
enter any SWR-vs-non-SWR contrast.

Two consequences to state wherever those contrasts are reported:

1. The estimand is "SWR vs non-SWR content **among at-well events**", not in
   general. The contrast within the cohort is valid; the generalisation is not.
2. The exclusion rate is animal-dependent (senor 4.92 % → chimi 11.64 %), so
   per-animal consistency checks — the "5/5 animals" statements — compare
   differentially restricted populations.

### F.3 `time_since_stop` and `tc` are different quantities

`event_table.time_since_stop` is the **median over the event's bins**, i.e.
the event *midpoint* minus `stop_start`. The Family-B design's `tc` is the bin
centre, and `event_onset_table.t_since_stop_onset` is the event *start* minus
`stop_start`. They agree on **0 of 65,913 events**, differing by exactly
`event_duration / 2` (median 31 ms, max 1.71 s). Negligible for
`time_x_current`-style interactions, but they are not one construct.

## G. Base covariate reference

Definitions for the base (non-interaction) covariates on the canonical tables.
Interactions (`swr_x_*`, `mua_x_*`, `time_x_*`, `switch_x_*`, `current_x_*`,
`omission_x_*`) are products of a base covariate with a modifier (`is_swr`,
`is_mua`, the time-course, `is_after_switch`, `is_current`, `is_after_omission`)
and are not catalogued separately. `_patch` variants aggregate the per-arm column
to the 3 patches. Value/RL covariates (`p_reward_*`, `q_*`, `rpe`, ΔV, entropy,
Beta `*_mu`/`*_var`, `stem_*_p`) are in
[`behavior_model_covariates.md`](behavior_model_covariates.md).

**Spatial / trajectory** (per event-arm):
- `location_state` / `at_well` / `location_well` — whole-window behavioral
  location. An event is `at_well` only if every bin is at the same well;
  `in_transit`, `mixed`, and `unknown` events remain in the canonical tables so
  analyses can filter them explicitly. `physical_arm` is the modal observed
  track segment, including trunks. `current_arm` is a compatibility alias of
  `location_arm` and is NaN off-well; there is no trunk-to-destination fallback.
  9.10 % of events are off-well on the regenerated tables — see
  [`known_issues/current_arm_trunk_fallback.md`](known_issues/current_arm_trunk_fallback.md).
- `is_current`, `is_prev`, `is_next` — arm equals the animal's at-well /
  previous-trial-origin / upcoming-trial-destination arm. `is_next` is **future-referenced**
  (the destination the animal will run to); used only where a spec includes it, and
  interpreted as forward representation, not a causal predictor.
  `is_current` is **NaN** (not 0) wherever `current_arm` is, so any spec
  containing it is fit on the at-well cohort by complete-case selection. That
  is the only cohort in which "the current arm" is defined — it is not an
  optional sensitivity subset. `is_prev` / `is_next` / `is_last_rewarded`
  instead encode an absent reference as 0, because "no previous trip" genuinely
  means no arm qualifies.
- `is_same_patch` — arm shares a patch with `current_arm`; NaN wherever
  `is_current` is.
- `dist_to_arm` — track-graph distance from the animal's current position to the arm
  (0 at the current arm); `arm_dist_rank` is its within-event min-rank.

**Reward / value history** (per event-arm, from the trial sequence):
- `cumulative_reward` — number of rewards received at that arm among the prior
  (up to) 10 outcome-known trials; this is a finite-window count, not a
  session-to-date cumulative total.
- `visit_recency` — **trials-ago** since the animal last visited that arm (a discrete
  trial count, not elapsed time), measured at the **event start** using only
  trials completed by then. A reachable arm not yet visited receives the
  time-local right-censored value `n_completed_trials + 1`;
  `arm_never_visited` separately marks that state. Appending future trials
  therefore cannot change an earlier event's recency. **Trunk arms carry a
  structural `0.0`** once any history exists — they are not reward sites, so
  "trials since last visit" is undefined for them and zero is a placeholder,
  not "just visited". Any aggregation over arms must mask trunks first
  (`visit_recency_patch` takes its minimum over leaf arms only for exactly
  this reason).
  `visit_recency_demeaned` subtracts the mean across the **visited** leaf arms
  within that same event (trunks remain structural zero). Never-visited arms
  are excluded from that mean: their value is set by the censoring rule rather
  than measured, and including it would displace every other arm in the event.
  An event with no visited leaf arm is NaN. This is a time-local relative
  recency contrast: appending later events cannot change earlier rows.
  **Caveat on the censored arms themselves.** Excluding them from the *mean*
  does not remove the censoring rule from their own values: a never-visited
  leaf still carries `n_completed + 1` minus the visited mean, which on the
  regenerated tables is 18,897 / 395,478 leaf rows (4.78 %) with mean +9.66 and
  max +163.5 against ~0 for visited arms. Every shipped rung pairs
  `visit_recency_demeaned` with `arm_never_visited`, which absorbs that state —
  a rung using the former without the latter would estimate roughly 5 % of its
  slope off the censoring rule. Pair them.
- **Trunk-arm convention (applies to the value block too).** Trunk arms 6/7/8
  are not reward sites. `reward_rate_arm_k10` and `visit_recency` document their
  structural `0.0`; `q_arm`, `p_reward_arm` and `arm_uncertainty` follow the
  identical convention (structural `0.0` for a mapped stop, `NaN` when the stop
  has no qvals row at all) even though it was previously stated only for the
  first two. Any aggregation over arms must mask trunks first — a structural
  zero entering a `min` or a `mean` is silently wrong, which is why
  `visit_recency_patch` takes its minimum over leaf arms only.
- `arm_never_visited` — 1.0 where the animal has never visited that leaf arm by
  the event start, else 0.0; trunk arms 0.0. Carries the right-censored state
  that `visit_recency`'s fill value can only approximate, and enters M4+,
  `BASEc`, and the choice ladder alongside `visit_recency_demeaned`. Because a
  never-visited arm always holds the maximum pre-centering recency, the two
  move together by construction — check conditioning before reading their
  coefficients separately.
- `visit_recency_patch` / `arm_never_visited_patch` — per-patch aggregates,
  both taken as the **minimum over the patch's two leaf arms** (trunks
  excluded). A patch is novel only if both its leaves are novel.
- `reward_rate_arm_k10` — rewarded fraction of the last 10 outcome-known visits to that arm
  (trunk arms 0.0; a leaf arm not yet visited is also encoded 0.0, *not* NaN —
  only rows with no prior trials at all stay NaN). See §2.

**Recent-trial context** (per stop, broadcast to the stop's events):
- `reward_outcome` / `is_after_reward` / `is_after_omission` — the most recently
  arrived trip's outcome, exact complements once known and all NaN while that
  outcome is pending.
- `is_after_stay` / `is_after_switch` — the completed inbound transition stayed
  in / switched patch (exact complements; available at destination arrival).
- `is_stay` — whether the later outbound trip stays within the patch. This is a
  future-referencing response, not the same construct as `is_after_stay`.

**Event / state indicators** (per event):
- `is_swr` — event overlaps a sharp-wave ripple (strict positive-duration overlap;
  see the classified-event-table contract in CLAUDE.md).
- `is_mua` — event overlaps a multi-unit-activity population event.
- `non_local_event` — the event is a non-local (replay) event.

**Timing / exposure** (per event unless noted):
- `immobility_time` — stop (immobility bout) duration; the **`log(immobility_time)`
  exposure offset** in the rate models (per stop).
- `time_since_stop` — time from the immobility-bout onset to the event.
- `time_since_last_port_entry` — time from the most recent well/port entry
  completed by the event start to the event start. `time_since_port_entry` is
  a compatibility alias. An arrival during an event never enters this value.
  Both are nonnegative when present; an in-progress trip's future arrival is
  never used.
- `time_since_reward` — time from the last reward available by event start to
  the event start.
- `event_duration`, `event_start`, `event_end`, `event_midpoint_time` — the event's
  span and boundary/mid timestamps.

**Decode content** (per event-arm; the content-model response family):
- `arm_fraction` — MAP-based: fraction of the event's real-arm timebins whose decoded
  MAP arm is this arm (junction bins excluded; sums to 1 over the 9 arms).
- `arm_fraction_posterior` — posterior-mass-based: this arm's summed posterior mass
  over the total real-arm mass (also sums to 1).
- `arm_posterior_mass` / `arm_count_soft` / `arm_count_hard` — unnormalized posterior
  mass / soft (posterior-weighted) and hard (MAP) timebin counts per arm.
- `replay_frac` — mean `arm_fraction` across the events in the stop (choice dataset);
  `replay_frac_swr` / `replay_frac_non_swr` restrict to the stop's SWR / non-SWR events.

**Self-history** (per-bin rate design; built at fit time, not stored):
- `tc` — time (s) since stop onset for each bin; entered via a `bs(tc)` spline (the
  time-course backbone of Family B).
- `onset_lag{j}` — count of the event's own onsets `j` bins back (past-only, resets
  per stop); `hist_rcos_k` projects those causal lags onto raised-cosine basis `k`.
- `cross_hist_rcos_k` — the same raised-cosine history for the *other* event stream.
