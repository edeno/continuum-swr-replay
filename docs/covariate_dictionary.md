# Covariate dictionary — canonical definitions across replay model families

Single source of truth for how each behavioral covariate is defined and how it
enters each model family, so coefficients are **comparable across rate / content /
choice / coupling**. Agreed decisions D1–D5 (2026-06-29) are recorded at the end.
This is the target the model specs and covariate-computation layer implement
against; the harmonization refactor brings the code to this spec.

**Status: definitions implemented; canonical tables and model artifacts require
regeneration before paper results are updated.** The covariate
definitions live in the `analysis/replay/covariates/` package: `clocks.py`
owns arrival/outcome timing, `context.py` owns time-local trial and location
labels, the `history/` subpackage owns reward, visit, switch, and local
leaky-Q history (`reward_history.py`, `visit_recency.py`, `value_block.py`,
`value_uncertainty.py`, `diagnostics.py`, over a shared `_common.py`),
and `candidate_arm.py` is the single shared Replay/ContFrag implementation of
the nine-arm predictor block. `definitions.py` provides the machine-readable
scientific catalog (units, grain, clock, support, missingness, and producer),
while `analysis/replay/table_contracts.py` (one level above the `covariates/`
package) validates complete candidate sets, response conservation, and
all-or-none leaf support at producer/reducer boundaries. The
package `__init__.py` is the stable public import surface. The candidate-arm
value block (`reward_fraction_candidate_arm_last10_visits`,
`task_reward_probability_candidate_arm`, and
`reward_belief_mean_candidate_arm_preoutcome`) is emitted by the event
builders from the canonical per-stop qvals artifact. The local leaky-Q
comparison uses the fixed α documented below; the value rungs are `Mv`
(arm) / `Pv` (patch); `reward_belief_mean_chosen_arm_preoutcome` enters the
MU/coupling runners; the choice
baseline includes the learned candidate-arm reward belief, recent reward
fraction, recency/novelty, and `candidate_arm_distance_rank`. Existing coefficients predate the location and
two-clock history contracts and must not be treated as results from these
definitions.

## A. Entry-rule contract (forced by estimator type)

A covariate that is constant across arms within an event **cancels** in a
conditional/multinomial model, so it cannot be a main effect there. Hence:

| estimator | family | scalar covariates (reward / value / switch) | arm- or patch-relative covariates |
|---|---|---|---|
| **absolute** | per-bin rate (B), per-stop rate (A), coupling/power | **main effects** | — |
| **conditional (content)** | arm hierarchy, patch hierarchy | only as **`× is_current_arm`** (or another explicit role) interactions | **main effects** |
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
| `known_reward_outcome_lag_{1..10}` | outcome of the j-th most recent port experience whose outcome was observable by the query time; experience 0 enters only when its outcome was physically observed | 10 known outcomes, per-lag | all-port | binary | **stop metadata, not a ladder predictor** — the rate ladders see this history only through the `reward_history_rcos_*` basis below, never as per-lag dummies (the one direct consumer left is the SWR-power OLS, which z-scores lag 1) |
| `reward_fraction_last10_known_trials` | rewarded fraction of the last 10 outcome-known port experiences; experience 0 enters only when its outcome was physically observed | 10 known outcomes | all-port | fraction | **stop metadata, not a ladder predictor** (it is the kernel's unweighted mean → VIF ∞ against the basis) |
| `reward_history_rcos_{0,1,2}` | the known-outcome lag vector projected onto three log-spaced raised-cosine bumps: `Σ_j B[k,j]·lag_j / Σ_j B[k,j]`, summed over the lags whose outcome is known at the query time, so each feature is a bump-weighted reward *fraction*. Bump 0 weights lags 1–7 (1.00, 0.79, 0.54, …) — the **immediate** component; bump 1 spans all ten lags and peaks at lag 3; bump 2 spans lags 2–10 and peaks at lag 10. A bump with no known lag in its support is `0.0` and flagged by `reward_history_incomplete`. Producer: `covariates.history.reward_history_basis_features`, evaluated at bout start. | 10 known outcomes, log-spaced smoothing | all-port | weighted reward fraction in [0, 1] | **the reward-history block of every rate ladder** — Family A `R1`, Family B `M1_rew`/`M7r_time_histcos_rew`/`M9r_rew_x_time`/`M7ra_time_arrival_histcos_rew`, `BASE` of the `is_swr`-contrast G/S ladder, and `MU_uncertainty`. Coefficients are **per unit of the fraction** (a 0→1 contrast), not per SD; read the per-lag effect off the kernel reconstruction, not off one coefficient. |
| `reward_history_incomplete` | 1 when fewer than two outcomes are known at the query time — the slowest bump then has no finite lag in its support and all three features are zero-filled — else 0. Flagged stops form an unbroken prefix of each epoch (1.296 % of stops, 1.576 % of events). | epoch prefix | all-port | binary indicator | **structural-missingness nuisance main effect**: fit in every rung that carries the basis, always outside any `is_swr` expansion, never aggregated or reported, and excluded from the separation judgment (`NUISANCE_TERMS`). |
| `reward_fraction_candidate_arm_last10_visits` | rewarded fraction of the last 10 **visits to that candidate arm**; an unvisited leaf is encoded `0.0`, while `is_candidate_arm_never_visited` separates novelty from an observed zero. Trunk arms are structurally `0.0`; rows before any known port outcome are NaN. | 10 arm-visits | candidate arm | fraction | content and choice main effect; rate/coupling via current-arm interaction |

The candidate-arm fraction uses a **10-visit** window. The all-port fraction
uses the last 10 outcome-known port experiences. The persisted column names
retain “trial” for artifact continuity, but their canonical source is an
explicit experience stream: starting-port occupancy first, then one destination
per complete physical poke-to-poke movement. Epoch boundaries do not establish
occupancy and enter neither visit nor reward history. No synthetic movement
trial, visit, or outcome is inserted. Trunk arms keep the structural `0.0`;
never-visited leaf arms are `0.0` plus
`is_candidate_arm_never_visited = 1`.

### 2. Value — both nominal and experienced, available to every family

| canonical name | definition | scope | source | enters |
|---|---|---|---|---|
| `task_reward_probability_chosen_arm` | nominal contingency p(reward) of the realized destination arm; defined once that choice is known | scalar | `qvals_table` (DataJoint) | post-choice rate/coupling main effect |
| `task_reward_probability_patch_mean_gap_best_vs_second` | nominal **patch**-level best-minus-second-best: the gap between the highest and second-highest patch mean p(reward). Always >= 0; a global scalar per trial, entered as a main effect only (never `× is_current_patch`) | scalar | `qvals_table` (nominal schedule) | rate/coupling main effect |
| `task_reward_probability_candidate_arm` | nominal contingency p(reward) of each candidate arm | candidate arm | `qvals_table` | content value rung; retained as a separate choice sensitivity rather than co-fit with learned belief |
| `reward_belief_mean_chosen_arm_preoutcome` | Beta-model reward-probability belief selected at the realized destination arm; the belief state itself precedes the outcome, but selecting this scalar requires the choice to be known | scalar | behavior model (`qvals_table`, `beta_stable_withleaf`) | post-choice rate/coupling main effect |
| `reward_belief_mean_candidate_arm_preoutcome` | Beta-model reward-probability belief mean for each candidate arm | candidate arm | behavior model (`beta_stable_withleaf`) | content value main effect and primary Stage-5A choice-value control |
| `leaky_q_value_diff` | pre-update leaky-Q best-minus-second **margin** at the most recently arrived physical movement destination (α=0.3); every earlier observed port outcome updates the state | scalar | `canonical_port_value_history` | **retained comparison** (MU only) |

Migration note: the two event-arm value columns were renamed 2026-08-21 to
`task_reward_probability_candidate_arm` and
`reward_belief_mean_candidate_arm_preoutcome` (old spellings and the full rename
table: [design/2026-08-21-covariate-naming-scheme.md](design/2026-08-21-covariate-naming-scheme.md));
the formerly deferred `nominal_reward_prob` field was removed. The model's native per-port values are
adapted to `reward_belief_mean_port_{0..5}`, and the chosen value is exposed as
`reward_belief_mean_chosen_arm_preoutcome`. **`leaky_q_value_diff` is NOT this** — it is a different local
construct (one pre-update best-minus-second margin per arrived movement destination, α=0.3,
computed from the complete starting-port-plus-destinations experience stream in `covariates/history/`, persisted on stop rows,
and then broadcast to events by `stop_id`), kept as a reported MU-only comparison. `task_reward_probability_patch_mean_gap_best_vs_second`
(nominal) remains a scalar
rate/coupling covariate. **`rpe_postoutcome` is the model RPE everywhere**
(`reward − reward_belief_mean_chosen_arm_preoutcome`,
from `qvals_table.parquet`; see [behavior_model_covariates.md](behavior_model_covariates.md)) —
after the migration every consumer (`fit_uncertainty_rate`, `fit_swr_power`, and
`_covariate_ranking`) uses this canonical model RPE, not a nominal stand-in.

### 3. Exploration — canonical continuous + retained binary comparison

| canonical name | definition | window | units | enters |
|---|---|---|---|---|
| `patch_switch_fraction_last5_completed_trips` | rolling mean of patch change over the last (up to) 5 completed trips, including the most recently completed trip | 5 completed trips | fraction | cross-family rate/coupling main effect; content via current-arm interaction |
| `did_last_completed_trip_switch_patch` | the most recently completed trip changed patch | 1 completed trip | binary | behavioral metadata and dedicated switch-cohort selector; **retired as a fitted predictor** because it is often constant within a session cohort |
| `omission_streak` | leading consecutive omissions in `known_reward_outcome_lag_1, known_reward_outcome_lag_2, …`, stopping at the first reward; a NaN truncates the count and unresolved lag 1 yields NaN | up to 10 known outcomes | count | model-free uncertainty block; excluded from the MU rate fit — it is another summary of the same lag vector the `reward_history_rcos_*` basis already carries |
| `wsls_deviation` | 0 on win-stay or lose-shift, 1 on win-shift or lose-stay. Reads `completed_trial_reward` (the trip's *realized* outcome, ungated) paired with `is_stay`. `NaN` where no trip has completed, the outcome was not observed, or the outbound choice is unresolved | 1 completed trial | binary | **descriptive only** — embeds the eventual choice, so future-referencing however the outcome is clocked |

(D4 (2026-06-29) kept both; **superseded 2026-08-21**:
`patch_switch_fraction_last5_completed_trips` (renamed 2026-08-21; see the
[naming scheme](design/2026-08-21-covariate-naming-scheme.md)) remains the
canonical graded covariate, but `is_after_switch` is retired as a fitted
predictor — see the row above.)

### 4. Spatial / role — already consistent, no change

**Two spatial constructs.** `occupancy_well_arm` is a *task state*: which well
the animal is visiting, derived
from trip intervals, and **NaN during a trip** — the destination is never named
before arrival, which is what makes it safe as a predictor.
`modal_observed_track_segment_id` is a *position*: the modal track segment the
tracking falls on, always defined,
trunks included. They disagree on 46 of 59,374 at-well events (0.077%), where
the animal is still inside its well-visit interval but a mean 48.4 cm from the
well. Neither is a fallback for the other; a cohort needing occupancy off-well
must state which construct it means.

**Per-builder event windows.** The event-window-derived task-state fields
(`occupancy_well_arm`, `current_patch`, `is_stay`, `prev_arm`/`next_arm`/
`last_rewarded_arm`, `is_rewarded_outcome`) are computed by each builder over
*its own* event window, not joined from a shared source. NonLocal and ContFrag
events are genuinely different windows inside the same immobility bout, so on a
handful of shared stops the two builders' values legitimately differ (~0.01-0.37%
of shared stops, e.g. 58/15,763 for `occupancy_well_arm`). This is expected, not a
bug — do not treat these as a single canonical stop-level value across builders.

**Scope and support traps, measured 2026-08-19.** Six covariates whose names promise
more than the data delivers:

- **`is_periswitch_post = 0` does not mean "no switch yet".** It means the
  *nearest* switch is ahead rather than behind (`periswitch_index_trials < 0`).
  Of the 35.1% of trials
  labelled 0, **94.75% already had a switch earlier in the session**. It is a
  nearest-neighbour tie-break, not a causal ordering flag.
- **`periswitch_index_trials` is signed and clipped.** 35.1% of trials are negative
  (counting down to a *future* switch) and **26.9% sit exactly on the ±20 clip**, so a
  linear slope is fit partly on a saturation plateau.
- **`candidate_arm_distance_rank` is ranked over all nine arms**, but the choice set drops the current
  arm and the trunks, leaving only ranks **{2, 6, 8}**. In that set the 2→6 step spans
  **108 cm** of real distance and the 6→8 step spans **1.3 cm** — entered linearly, a
  third of its slope is assigned to a 1 cm difference. Do not read its coefficient as a
  distance gradient. (Removed from `BASEc` 2026-08-18; still the only geometry control
  in `C_full`.)
- **A "stop" is an immobility bout, not a well visit.** Median **0.33 s**, 73.5% under
  1 s, and only 22.7% contain any event. Those sub-second bouts are 73.5% of the
  per-stop rate ladder's rows but hold 6.4% of the events.
- **`patch_count_hard` includes the patch's trunk**, and trunk arms carry **24.1%** of
  all decoded MAP bins — so every patch-level coefficient is partly about corridor
  representation.
- **Choice-model `replay_fraction` does not sum to 1** over the candidate set: mean
  **0.49**, because `arm_fraction` is normalised over nine arms and the choice set keeps
  five. The excluded current arm alone holds mean 0.24.

**Per-SD support: leaf-only is SHIPPED.** The candidate-arm belief mean, task
probability, reward fraction, and belief variance are standardized from leaf rows
only — structural-zero trunks cannot shrink the unit of a leaf-defined value or
uncertainty effect. This is materialized: as of the 2026-08-21 leaf-support
regeneration (re-promoted with the 2026-08-25 initial-port-experience corpus),
`standardization_reference.json` carries `"support": "leaf_arms"` on all four
columns and every persisted `_z` reproduces exactly under the **leaf** moments
(e.g. `reward_belief_variance_candidate_arm_preoutcome` std = 0.021175 = leaf SD,
n_reference_rows = 390120 = the leaf-row count). **Every shipped `_z`, and every
coefficient quoted per 1 SD from one, is already on the leaf-support unit — do not
apply any nine-arm→leaf conversion.** (Historical note: before 2026-08-21 the
persisted `_z` used all-nine-arm moments; coefficients quoted from that era read
~2.09× / ~1.62× / ~1.17× / ~1.02× larger for the four columns above,
respectively, and must be divided by those ratios to compare to a current
leaf-support coefficient. The current tables need no such conversion.) The raw
SWR × candidate-arm belief-mean interaction stays on probability units, so it
still must not be compared numerically to a standardized uncertainty coefficient
without an explicit SD conversion.

**`relative_value_switch_minus_stay_preoutcome` (ΔV) is Switch − Stay**, confirmed against the
manuscript (`references/Spatial_bandit_manuscript_v3.2_biorxiv_figs.pdf`, Methods): Stay is
the *upcoming* port within the patch, Switch is the greater of the two alternative patches'
mean port values, and the difference is taken *"without accounting for the cost of Switching
or other biases"*. The shipped column equals Switch − Stay exactly on all 102,975 rows
carrying a finite value (r = 1.000000); 102,975 counts *stop rows*, not distinct trials.
Use the `_mu` variant, not `_Qstem`: the latter is on the decision-value scale and is
negative on 99.90% of rows (99 positive, max +0.755) because it carries the stay bonus.
ΔV is ~ −(the current arm's stay value): r = −0.952 with
`stay_value_preoutcome`, −0.890 with
`reward_belief_mean_chosen_arm_preoutcome`, with near-identical SD
(0.158). A positive
`relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm` coefficient
therefore means content sharpens onto the
current arm when that arm's value is **low**. Earlier labels called it "Stay-vs-Switch",
which inverts the reading; corrected 2026-08-19, though one stale label remains in
`scripts/rate/fit_swr_contrast_rate.py:23`.

**`in_transit` does not mean "moving".** It labels the trip interval. The
modelled cohort is speed-gated immobility throughout, and off-well events have
a median speed of 1.35 cm/s — they are animals standing still partway through a
trip. The content models exclude them because the well-visit construct is
undefined there (so `is_current_arm` and everything derived from it is NaN), not
because of motion. The at-well cohort is stated explicitly in the runner rather
than emerging from that NaN pattern.


`is_current_arm`, `is_current_patch`, `is_prev_arm`, `is_prev_patch`,
`is_last_rewarded_noncurrent_arm` / `is_last_rewarded_noncurrent_patch`
(disjoint support), `is_candidate_arm_{0..7}` (content baseline), and
`candidate_arm_distance_rank` (choice baseline). Definitions are unchanged.

Both noncurrent last-rewarded indicators are **pooled-only as of 2026-08-18**: they are built
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

`bs(tc)` (fixed knots (1,2,3) s, [0,4]), the `reward_history_rcos_*` kernel, `onset_lag{j}` /
`hist_rcos` self-history. These are per-bin within-stop terms and live only in the
rate families; conditional models access within-stop time only through an
explicit current-arm interaction when a within-stop content effect is tested.

## C. Per-family covariate matrix (target)

| construct | rate B (per-bin) | rate A (per-stop) | coupling | content (arm/patch) | choice |
|---|---|---|---|---|---|
| `known_reward_outcome_lag_{j}` | — (stop metadata) | — (stop metadata) | raw lag 1, z-scored (SWR-power OLS) | current-arm interaction if tested | — |
| `reward_fraction_last10_known_trials` | — | — | — | — | — |
| `reward_history_rcos_{0,1,2}` + `reward_history_incomplete` | **main** (`M1_rew`/`M7r_time_histcos_rew`/`M9r_rew_x_time`/`M7ra_time_arrival_histcos_rew`, `BASE`) | **main** (`R1`) | — | — | — |
| `reward_fraction_candidate_arm_last10_visits` | current-arm interaction | — | current-arm interaction | **main** | **main** |
| current-arm task probability / `task_reward_probability_candidate_arm` | main | — | main | candidate-arm main | nominal-value sensitivity |
| chosen-arm / candidate-arm reward-belief mean | main | — | main | candidate-arm main | candidate-arm main |
| `rpe_postoutcome`, `task_reward_probability_patch_mean_gap_best_vs_second` | main after timing gate | — | main after timing gate | interaction if tested | — |
| `patch_switch_fraction_last5_completed_trips` | main | main | main | current-arm interaction | optional main |
| `did_last_completed_trip_switch_patch` | — | — | cohort selector only | — | — |
| spatial role / `is_candidate_arm_{0..7}` | — | — | — | main | choice set + `candidate_arm_distance_rank` |
| `bs(tc)`, self-history | main | — | — | `log1p_time_since_stop_start_s_z_x_is_current_arm` only | — |

## D. Decisions of record (2026-06-29)

- **D1** — recent-reward window standardized to **N = 10** in each construct's
  natural unit: `reward_fraction_last10_known_trials` (10 known outcomes) and
  `reward_fraction_candidate_arm_last10_visits`
  (10 visits, per-arm). Per-arm window changes 5→10. **Caveat:** "10" is not the same
  *real-time* span (a single arm is visited far less often than trials elapse) — it is
  10 in each construct's own unit, not 10 in time.
- **D2** — add candidate-arm nominal value
  `task_reward_probability_candidate_arm` to content and retain it
  for an explicit choice sensitivity — by
  **replacing the formerly-deferred `nominal_reward_prob` column** (renamed and now
  populated), not a new parallel column.
- **D3** — add the chosen/candidate-arm Beta reward-belief mean from qvals,
  not a local leaky-Q learner, as canonical,
  and **keep `leaky_q_value_diff`** (the completed-trial pre-update margin, a
  different local construct) as a *reported comparison*. (Revised from "drop":
  the two are different
  algorithms, and the margin is deliberately kept in the MU dissociation table.)
- **D4** — exploration canonical =
  `patch_switch_fraction_last5_completed_trips`.
  `did_last_completed_trip_switch_patch` is retired as a fitted predictor
  (rank deficiency) and kept only as metadata/cohort selector.
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
- Every regression model that consumes these covariates, in R formula notation, per
  family: [`models_table.md`](models_table.md). The record of which rungs changed or
  held parity across refits lives in [`design/model_history.md`](design/model_history.md).

## F. Temporal causality (pre- vs post-outcome) — leakage audit

Each stop-level covariate is broadcast to the non-local events within that stop.
A covariate is safe iff its value is determined by information available **before**
the event it predicts. Classification of the covariates fitted by the rate models:

| covariate | class | available before event? |
|---|---|---|
| `task_reward_probability_candidate_arm`, `task_reward_probability_patch_mean_gap_best_vs_second`, `trials_since_block_change` | nominal schedule | yes — fixed by the block, known at trial start |
| `task_reward_probability_chosen_arm`, `reward_belief_mean_chosen_arm_preoutcome` | chosen-arm-selected schedule/belief | **only after the destination choice is known**; current rate/coupling uses conservatively post-outcome-gated rows |
| candidate-arm reward-belief means, `relative_value_switch_minus_stay_preoutcome`, and reward-belief variance | model, **pre-outcome** belief (row *t* = belief entering trial *t*) | yes |
| destination/visit recency, `did_last_completed_trip_switch_patch`, `patch_switch_fraction_last5_completed_trips` | **arrival clock** | yes — destination and inbound patch change become known at goal arrival |
| reward lags/rates/counts, last-rewarded well | **outcome clock** | yes — advance at first pump onset (reward) or at the latest possible pump latency after arrival, capped at the collapsed task visit's final poke-out (omission) |
| `reward_belief_differential_entropy_leaf_mean_preoutcome` | mean per-leaf Beta differential entropy | yes (pre-outcome posterior) |
| `rpe_postoutcome`, `reward_belief_mean_update_l1_all_ports_postoutcome` | model, **post-outcome** (uses the current trial's realized reward) | **only if the event follows reward delivery** |
| `is_rewarded_outcome` (omission derived as `1 - reward`) | most recently arrived trip's outcome | finite **only after** that outcome is observable; otherwise NaN |
| `is_stay`, `wsls_deviation` | eventual outbound choice | no — future-referencing; descriptive use only |
| `completed_trial_reward` | completed physical trip's realized outcome, **ungated** | no — post-outcome; exists only so `wsls_deviation` can name the outcome the animal just experienced |

**Arrival is not the outcome boundary.** Canonical trials therefore carry two
clocks. `destination_first_poke_in_time` controls destination, visit, and
inbound-switch history (`goal_arrival_time` is a temporary compatibility alias).
Consecutive same-port re-pokes are one task experience, matching Comrie: arrival
is its first poke-in and departure is its final poke-out. This task-port context
is deliberately broader than literal photogate engagement during each bout.
`outcome_known_time` is the first pump onset for a rewarded trial. For
an omission it is destination arrival + `OMISSION_KNOWN_LATENCY_SEC` (0.30 s),
capped at the destination task visit's **final** poke-out. A later same-port
physical re-poke does not create a new task experience. A delayed pump anywhere
inside that whole visit is credited to the same experience, matching Comrie's
visit parser; therefore a transient beam release cannot establish an omission.
The 0.30-s latency exceeds every observed rewarded-visit pump latency and does
not otherwise condition the boundary on the animal's dwell duration. Between
arrival and that boundary, arrival-derived fields are populated while all
current reward-bearing fields remain NaN. Older/manual trial frames without a
physical outcome column use `end_time` for both outcomes: conservative for
rewards, but never anticipatory.

Experience 0 is the recorded origin poke preceding the first complete movement.
A recorded pump gives it an exact reward-on clock; a physical poke without a
pump supplies an observed omission at the same operational feedback boundary
used for destination visits. An epoch boundary supplies neither occupancy nor
outcome. Likewise, movement after the final physical poke is not a completed
trial. This prevents left- or right-censored epoch fragments from being
manufactured as visits or omissions.

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
| **experienced** | current/previous arm indicators, distance, stop time, known reward lags, omission streak, reward count/fraction, last-rewarded arm, completed-trip switch, visit recency, novelty | content covaries with something the animal directly underwent |
| **inferred** (model belief) | candidate/chosen-arm reward-belief means and variance, Beta differential entropy, relative value, local leaky-Q margin, RPE, and belief updates | content covaries with a *hypothesised* internal state, conditional on the belief model being right |
| **scheduled** (experimenter-only) | candidate/current-arm task reward probability, `task_reward_probability_patch_mean_gap_best_vs_second`, `trials_since_block_change` | content covaries with the **programmed contingency** |

The third row is the one that gets over-claimed. The animal never observes a
block boundary or a nominal p(reward); it only ever holds a noisy estimate. A
coefficient on `trials_since_block_change` is informative *insofar as the
animal has learned the schedule* — it is not evidence that the animal
represents trials-since-block-change. Report these as schedule effects, and if
the claim is about representation, use the inferred-belief covariates instead.

(Timing itself is fine in all three: `trials_since_block_change` is a
backward `cumcount` within a block, unlike `periswitch_index_trials`, which is
signed toward the nearest switch — including future ones — and is therefore
future-referencing.)

### F.1 Two clocks can coexist in one design row

The candidate-arm reward-belief mean/variance and task-probability block is
fixed on the qvals row entering the trial. The reward-history block
(`reward_count_candidate_arm_last10_known_trials`,
`reward_fraction_candidate_arm_last10_visits`, and
`known_reward_outcome_lag_{j}` — with the `reward_history_rcos_*` basis built
from that lag vector) advances on
the outcome clock. On **83.9 % of events** (54,537 / 65,020 matched) the event starts after its
trial's `outcome_known_time`, and 54,451 of those trials were rewarded — so
within a single design row the two blocks summarise history through
**different endpoints, differing by the just-received reward**.

> **That 83.9 % must be re-derived.** It was measured against the *previous*
> omission boundary (destination poke-out). The boundary is now poke-in +
> `OMISSION_KNOWN_LATENCY_SEC`, capped at the collapsed visit's final poke-out;
> it is earlier by a median 0.87 s, so more
> events clear it and the share will move — upward. The qualitative claim (the
> two blocks close at different endpoints, differing by the just-received
> reward) is unaffected; only the number is stale.

This is not leakage: the value block carries strictly *less* recent
information than the reward block, never more. It is verified pre-update —
the chosen entry in `reward_belief_mean_port_{0..5}` moves +0.078 after a reward
and −0.091 after an omission on the following trial, and
`rpe_postoutcome == trial_reward − reward_belief_mean_chosen_arm_preoutcome`
exactly.

It is an interpretation hazard. A coefficient on
`reward_belief_mean_candidate_arm_preoutcome` read against one on
`reward_fraction_candidate_arm_last10_visits` is not "value vs. reward history
at the same moment";
the value term is one outcome behind. Say so when the two appear in the same
rung (M v, C4/CS5, C13/CS6).

A related, smaller offset: the value block is frozen at `stop_start` (it is
joined by `stop_id` from the per-stop qvals artifact) while every other
event-arm history covariate is frozen at `event_start`. Median gap 2.39 s. The
practical impact is small — qvals' assigned arm equals the event's own
`occupancy_well_arm` on 100.00 % of at-well events whose stop is `mixed` — but the
two are not the same instant.

### F.2 Scope condition: the at-well cohort is not SWR-neutral

`is_current_arm` is undefined off-well, so every rung containing it fits the
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

### F.3 `time_since_stop_start_s` and `tc` are different quantities

`event_table.time_since_stop_start_s` is the **median over the event's bins**, i.e.
the event *midpoint* minus `stop_start`. The Family-B design's `tc` is the bin
centre, and `event_onset_table.t_since_stop_onset` is the event *start* minus
`stop_start`. They agree on **0 of 65,913 events**, differing by exactly
`event_duration_s / 2` (median 31 ms, max 1.71 s). Negligible for
time-by-current-arm interactions, but they are not one construct.

### F.4 Per-bin reward history is recomputed at the outcome boundary

A stop is an immobility **fragment**, not a well visit: 68,810 of the 69,224
at-well stops carry an arrival clock and fall in 35,043 visits (mean **1.96
fragments per visit**, median 1, 90th pct 3), only 50.9 % of
at-well stops are the first fragment of their visit, and **69.9 % of at-well
non-local events occur in a non-first fragment**. **80.1 %** of at-well stops
already begin after their trial's outcome is knowable.

The predicate `stop_start < outcome_known_time_s <= stop_end` — the stop *spans*
the boundary — holds for **13.5 %** of all stops (14,494 of 107,400: 12,393
at-well and 2,101 `mixed`; no in-transit bout ever spans it). The shift applies
to the **13.4 %** (14,343 stops) that also keep at least one bin at or past the
boundary. For those, holding the
reward-history features at their bout-start values would let bins recorded after
the animal learned the outcome carry a reward history that predates it. So from
the first bin whose left edge is at or after `outcome_known_time_s`, the per-bin
lag matrix is shifted (`lag_1 <- completed_trial_reward`, `lag_j <- lag_{j-1}`)
and `reward_history_rcos_0..2` plus `reward_history_incomplete` are recomputed
from the shifted row **by the same basis functions the stop level uses**. Bins
before the boundary keep the bout-start values.

The one exclusion is a bout that genuinely **spans a well arrival**: the next
stop of the same session (ordered by `stop_start`) arrived inside
`(stop_start, stop_end]`, so the well the animal occupies changes partway
through and `completed_trial_reward` is not the outcome that becomes known at
the boundary. Those stops are not shifted, and the ones with at least one bin
past the boundary are counted in `out.attrs["n_stops_arrival_spanning"]`.
`occupancy_state == "mixed"` is **not** that test and was a bad proxy for it: on
the canonical stop table 173 stops (0.16 %) contain a later arrival and **none**
of them also contains an outcome boundary, while the 2,099 `mixed` stops that do
contain a boundary begin a median 0.22 s *after* their own arrival and last
~0.7 s. A stop whose boundary lies outside it is untouched.

## G. Base covariate reference

Definitions for the base (non-interaction) covariates on the canonical tables.
Interactions are products of a canonical base covariate with a modifier
(`is_swr`, `is_mua`, the time course, a current-arm role, or a derived omission)
and are not catalogued separately. `_patch` variants aggregate the per-arm column
to the 3 patches. Value/RL covariates (task reward probability, reward-belief
mean/variance, RPE, relative value, entropy, and choice probability) are in
[`behavior_model_covariates.md`](behavior_model_covariates.md).

**Spatial / trajectory** (per event-arm):
- `occupancy_state` / `at_well` / `occupancy_well_id` — whole-window behavioral
  location. An event is `at_well` only if every bin is at the same well;
  `in_transit`, `mixed`, and `unknown` events remain in the canonical tables so
  analyses can filter them explicitly. `modal_observed_track_segment_id` is
  the modal observed track segment, including trunks.
  `occupancy_well_arm` is NaN off-well; there is no
  trunk-to-destination fallback.
  9.10 % of events are off-well on the regenerated tables — see
  [`known_issues/current_arm_trunk_fallback.md`](known_issues/current_arm_trunk_fallback.md).
- `is_current_arm`, `is_prev_arm`, `is_next_arm` — candidate arm equals the
  animal's at-well / previous-trip-origin / upcoming-trip-destination arm.
  `is_next_arm` is **future-referenced**
  (the destination the animal will run to); used only where a spec includes it, and
  interpreted as forward representation, not a causal predictor.
  `is_current_arm` is **NaN** (not 0) wherever `occupancy_well_arm` is, so any spec
  containing it is fit on the at-well cohort by complete-case selection. That
  is the only cohort in which "the current arm" is defined — it is not an
  optional sensitivity subset. `is_prev_arm` / `is_next_arm` /
  `is_last_rewarded_arm`
  instead encode an absent reference as 0, because "no previous trip" genuinely
  means no arm qualifies.
- `is_same_patch_as_current` — candidate arm shares a patch with
  `occupancy_well_arm`; NaN wherever current occupancy is undefined.
- `distance_to_candidate_arm_cm` — track-graph distance from the animal's
  current position to the candidate arm; `candidate_arm_distance_rank` is its
  within-event min-rank.

**Reward / value history** (per event-arm, from the port-experience sequence):
- The canonical stream starts with the port occupied before the first movement,
  followed by one destination experience per movement. Spatial visit history
  advances at each experience's arrival. Reward history advances only when its
  outcome is physically available. Movement trials remain a separate table and
  are not renumbered or padded.
- `reward_count_candidate_arm_last10_known_trials` — number of rewards
  received at that candidate arm among the prior
  (up to) 10 outcome-known port experiences; this is a finite-window count, not a
  session-to-date cumulative total.
- `visit_recency_trials` — **port experiences ago** since the animal last visited that
  candidate arm (a discrete
  trial count, not elapsed time), measured at the **event start** using only
  port arrivals completed by then. A reachable arm not yet visited receives the
  time-local right-censored value `n_arrived_experiences + 1`;
  `is_candidate_arm_never_visited` separately marks that state. Appending future trials
  therefore cannot change an earlier event's recency. **Trunk arms carry a
  structural `0.0`** once any history exists — they are not reward sites, so
  "trials since last visit" is undefined for them and zero is a placeholder,
  not "just visited". Any aggregation over arms must mask trunks first
  (`visit_recency_trials_patch` takes its minimum over leaf arms only for exactly
  this reason).
  `visit_recency_trials_event_leaf_centered` subtracts the mean across the **visited** leaf arms
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
  `visit_recency_trials_event_leaf_centered` with
  `is_candidate_arm_never_visited`, which absorbs that state —
  a rung using the former without the latter would estimate roughly 5 % of its
  slope off the censoring rule. Pair them.
- **Trunk-arm convention (applies to the value block too).** Trunk arms 6/7/8
  are not reward sites. Candidate-arm reward fraction and visit recency
  document their structural `0.0`; candidate-arm belief mean, task probability,
  and belief variance follow the
  identical convention (structural `0.0` for a mapped stop, `NaN` when the stop
  has no qvals row at all) even though it was previously stated only for the
  first two. Any aggregation over arms must mask trunks first — a structural
  zero entering a `min` or a `mean` is silently wrong, which is why
  `visit_recency_trials_patch` takes its minimum over leaf arms only.
- `is_candidate_arm_never_visited` — 1.0 where the animal has never visited that leaf arm by
  the event start, else 0.0; trunk arms 0.0. Carries the right-censored state
  that `visit_recency_trials`'s fill value can only approximate, and enters M4+,
  `BASEc`, and the choice ladder alongside `visit_recency_trials_event_leaf_centered`. Because a
  never-visited arm always holds the maximum pre-centering recency, the two
  move together by construction — check conditioning before reading their
  coefficients separately.
- `visit_recency_trials_patch` / `is_patch_never_visited` — per-patch aggregates,
  both taken as the **minimum over the patch's two leaf arms** (trunks
  excluded). A patch is novel only if both its leaves are novel.
- `reward_fraction_candidate_arm_last10_visits` — rewarded fraction of the
  last 10 outcome-known visits to that candidate arm, including the starting
  port when it is that arm,
  (trunk arms 0.0; a leaf arm not yet visited is also encoded 0.0, *not* NaN —
  only rows with no physically known port outcome stay NaN). See §2.

**Recent-trial context** (per stop, broadcast to the stop's events):
- `is_rewarded_outcome` — the most recently arrived trip's outcome, nullable
  until feedback is known; omission is derived as its complement with the same
  missingness.
- `did_last_completed_trip_switch_patch` — whether the completed inbound
  transition switched patch (available at destination arrival); stay is
  derived only where needed.
- `is_stay` — whether the later outbound trip stays within the patch. This is a
  future-referencing response, not the completed inbound-trip construct.

**Event / state indicators** (per event):
- `is_swr` — event overlaps a sharp-wave ripple (strict positive-duration overlap;
  see the classified-event-table contract in CLAUDE.md).
- `is_mua` — event overlaps a multi-unit-activity population event.
- `non_local_event` — the event is a non-local (replay) event.

**Timing / exposure** (per event unless noted):
- `immobility_duration_s` — stop (immobility bout) duration; the
  **`log(immobility_duration_s)`
  exposure offset** in the rate models (per stop).
- `time_since_stop_start_s` — time from the immobility-bout onset to the event.
- `time_since_last_well_arrival_s` — time from the most recent well arrival
  completed by the event start to the event start. An arrival during an event
  never enters this value. It is nonnegative when present; an in-progress trip's future arrival is
  never used.
- `time_since_last_reward_s` — time from the last reward available by event start to
  the event start.

**Stop-grain clocks** (columns on `stop_table`, absolute seconds on the session
clock; both are NaN for a bout with no completed trial, e.g. a pre-trial stop):
- `last_well_arrival_time_s` — arrival at the well occupied at the bout start
  (`_goal_arrival_time` of the completed trial). Clock GOAL_ARRIVAL, timing
  PRE_OUTCOME: it is known the moment the animal reaches the well.
- `outcome_known_time_s` — the moment the trial's outcome became observable: the
  pump onset on a rewarded trial, poke-in + `OMISSION_KNOWN_LATENCY_SEC` on an
  omission. Clock OUTCOME_AVAILABILITY, timing POST_OUTCOME. Finite at bout start
  for 80.1 % of at-well stops; the remaining at-well stops begin inside the
  arrival → outcome window, so their outcome is not yet knowable when they start.
- `event_duration_s`, `event_start_s`, `event_end_s`, `event_midpoint_s` — the event's
  boundary/mid timestamps and inclusive decoded duration:
  `event_end_s - event_start_s + 1 / DEFAULT_SAMPLING_FREQUENCY_HZ`. Thus a
  one-bin event has one-bin duration rather than zero.

**Decode content** (per event-arm; the content-model response family):
- `arm_fraction` — MAP-based: fraction of decoded timebins assigned to this arm;
  a MAP coordinate exactly at a junction is assigned to its nearest real arm by
  the hard decoder mapping (sums to 1 over the 9 arms when MAP is finite).
- `arm_fraction_posterior` — posterior-mass-based: this arm's summed posterior mass
  over the total real-arm mass (also sums to 1). **Builder-specific denominator:**
  the NonLocal builder normalizes by `nonlocal_probability_sum` (probability-weighted
  mass); ContFrag, which has no non-local-probability concept, normalizes by
  `n_timebins` (bin-count mass). Both sum to 1 over the nine arms, but the column
  means probability-weighted mass in NonLocal and bin-count mass in ContFrag — do
  not compare the two builders' values as the same quantity.
- `arm_posterior_mass` — summed non-local-probability-weighted posterior mass.
- `arm_count_soft` — non-local-probability-weighted count of timebins whose
  **MAP** arm is this arm; despite the historical “soft” name, it is not a full
  posterior-mass allocation.
- `arm_count_hard` — count of timebins whose MAP arm is this arm.
- `patch_fraction` — the patch analogue of `arm_fraction`: the event's share of
  decoded content allocated to this patch, over the same denominator (the
  in-event timebins whose MAP landed on a real arm). A patch fraction is exactly
  the sum of its three arms' fractions, and an event's three patch fractions sum
  to one; NaN when the event has no real-arm bin. `patch_count_hard` is the
  corresponding timebin count.
- `replay_fraction` — mean `arm_fraction` across the events in the stop (choice dataset);
  `replay_fraction_swr` / `replay_fraction_non_swr` restrict to the stop's SWR / non-SWR events.

`arm_fraction` / `patch_fraction` are the **shipped content-model response**: each
event contributes one observation of unit mass, so the likelihood, AIC and
standard errors are stated per event. `arm_count_hard` / `patch_count_hard` weight
an event by its number of decoded timebins and are retained only for the
count-response comparison (`--response hard_count`).

**Event-length control** (per event; built at fit time by the `is_swr`-contrast
content runner, not stored):
- `log_n_nonlocal_bins` — `log(n_nonlocal_bins)`; `log_n_nonlocal_bins_z` is its
  z-score over the fit cohort. SWR-coupled events run ~1.8× longer than non-SWR
  ones and a longer trajectory visits more arms, so without this an
  `swr_x_is_current_arm` coefficient partly reads "longer events leave the
  current arm".
- `log_n_nonlocal_bins_z_x_is_current_arm` / `log_n_nonlocal_bins_z_x_is_prev_arm`
  (`_x_is_current_patch` for the patch hierarchy) — the length products that put
  that channel in the design. Members of `BASEc`, so every rung built on the base
  reports the SWR effect NET of event length (a post-treatment adjustment — length
  may be a mediator); the `nt` twins (`BASEc_NT`: `CS1nt_current`,
  `CS4nt_armreward`, `CS5nt_armvalue`, `CS6nt_armuncert`, `CPS1nt_current`) omit
  them and report the total SWR effect.
- `log1p_time_since_stop_start_s_z_x_is_current_arm` — the within-stop time-course
  product, also a `BASEc` member and also absent from the `nt` twins.

**Per-bin clocks** (Family-B design; built by `aggregate_to_timebins` at fit
time, not stored). All three are bin-centre quantities, so they are the per-bin
analogues of the per-event columns above and must not be joined with them:
- `tc` — time (s) since **stop onset** (the speed-threshold crossing that opened
  the immobility bout), entered via a `bs(tc)` spline with fixed knots (1, 2, 3) s
  over `[0, TMAX = 4]`. The default time-course backbone of Family B. A stop is an
  immobility *fragment*, not a well visit (see §F.4), so `tc = 0` is the start of a
  fragment and need not be an arrival or a reward.
- `time_since_last_well_arrival_s` — bin centre minus the stop's
  `last_well_arrival_time_s`; NaN where that clock is NaN. The arrival-anchored
  alternative to `tc`, entered via a `bs(...)` spline with knots (1, 2, 4, 6) s
  over `[0, TMAX_ARRIVAL = 8]` by the arrival rungs (`M0a_const_arrival`,
  `M5a_time_arrival`, `M7ra_time_arrival_histcos_rew`). Same *name* as the
  per-event column in the Timing / exposure block above and a different *grain*.
- `time_since_outcome_known_s` — bin centre minus the stop's
  `outcome_known_time_s`; NaN where that clock is NaN, and **negative before the
  outcome is knowable**. Not a fitted predictor of the shipped ladder; it is the
  boundary the per-bin reward features are recomputed at (§F.4).

**Self-history** (per-bin rate design; built at fit time, not stored):
- `onset_lag{j}` — count of the event's own onsets `j` bins back (past-only, resets
  per stop); `hist_rcos_k` projects those causal lags onto raised-cosine basis `k`.
- `cross_hist_rcos_k` — the same raised-cosine history for the *other* event stream.

**Dirichlet-multinomial estimator parameters** (content fits under
`--estimator dirichlet_multinomial`; **model parameters, not table columns** —
nothing on the canonical tables holds them, and they exist only in a fitted DM
result):
- `log_phi_intercept` — the log within-event Dirichlet concentration of a
  **non-SWR** event: `log φ_e = log_phi_intercept + log_phi_is_swr · is_swr_e`,
  with `c_e ~ DirichletMultinomial(L_e, φ_e · p_e)`. It lives on the
  **per-category** scale that φ parameterizes (φ multiplies the softmax
  probability vector to give the Dirichlet parameters), so it is a redundancy
  scale, not an effect on any arm. **Larger φ is *less* overdispersion**: the
  event's probability vector sits close to the model mean and its timebins are
  near-independent draws (φ → ∞ is exactly the count multinomial). **Smaller φ
  means the event's mass clumps onto a few arms and its bins are near-copies of
  one another.** It is the DM analogue of `Intercept`: a **structural nuisance
  term**, excluded from the separation judgment (`_DM_NUISANCE_TERMS`) and
  **never aggregated or reported** as a scientific effect. Because it is a
  nuisance term it can **never appear in `separated_terms`** — that flag says
  nothing about it either way. Where the session's concentration was pinned
  rather than estimated, the coefficient **is** the pinned value (not NaN) and
  it is `bse_log_phi_intercept` that is `NaN`, with the parameter excluded from
  `n_params`. It is also genuinely unidentified when a session's events are
  near-independent (large φ). Judge it from `phi_source` and its SE — a
  non-`session` `phi_source` means it was imposed, and a large or non-finite SE
  means the data did not pin it — never from `separated_terms`.
- `log_phi_is_swr` — the additive shift of that log concentration on SWR
  events, on the same per-category scale. **This one is a scientific term**: it
  stays in the separation judgment and is aggregated in its own BH-FDR family
  (one row per rung × population in the audited DM table), separately from the
  β contrast, because it is the same quantity on every rung rather than any
  rung's key term. Positive means an SWR event's mass is spread over *more*
  arms and its bins are closer to independent draws — less within-event timebin
  redundancy — and `+1.8` means an SWR event's φ is ≈ e^1.8 ≈ 6× that of a
  non-SWR event. It is a statement about redundancy
  across an event's bins, not about which arms are represented. Absent from the
  within-SWR / within-non-SWR population splits, where the flag is constant.
  **TOTAL vs LENGTH-ADJUSTED — say which you mean.** The fitter assumes φ is
  **length-invariant**: an event's length `L_e` enters the likelihood only as
  the multinomial index, and `log φ_e` carries no length term. Event length is
  **post-treatment** with respect to `is_swr` — on the canonical 65,913-event
  cohort an SWR event runs **1.73×** longer (median 45 vs 26 bins) — so the
  shipped `_dm` coefficient is the **TOTAL** concentration contrast, length
  included. The length-adjusted twin comes from
  `fit_swr_contrast_content.py --concentration-length` (`_dm_len` artifacts),
  which puts `log_n_nonlocal_bins_z` in the concentration design as well: on
  `C2_base` it is **+1.502** (5/5 animals, q = 2.7e-37) against the total
  **+1.799**, next to a length coefficient
  `log_phi_log_n_nonlocal_bins_z` = **+0.522 per SD** (5/5, q = 2.7e-37).
  About a sixth of the total runs through event length and the effect survives
  the adjustment. Full per-rung table and the estimand the recommendation cites:
  [`design/content_estimator_comparison.md`](design/content_estimator_comparison.md)
  §7.1.
- `log_phi_log_n_nonlocal_bins_z` — the additive shift of the log concentration
  **per SD of `log n_nonlocal_bins`**, present only in `_dm_len` fits. Positive
  (+0.43 … +0.53 across rungs, 5/5 animals on every one) means a longer event's
  bins are *also* closer to independent draws, which is why omitting it inflates
  `log_phi_is_swr`. It is reported per SD, not per category: it is the one
  concentration term whose design column is continuous.
- `sd_log_phi_<col>` — the SD of that concentration column over the fitted
  events, persisted so the aggregation's identifiability backstop is read on the
  effect scale (`|coef| × sd`) like every other term. `sd_log_phi_intercept` is
  **NaN** by construction — a constant column has no spread — so the intercept
  is judged by the separation flag alone.
- `n_events_swr` / `n_events_nonswr` — modelled events on each side of the
  `is_swr` modifier, on every DM row; **NaN** where the concentration design has
  no `is_swr` column (the within-SWR splits). A modifier resting on one or two
  SWR events is finite and unflagged, and no identifiability rule can see it —
  these counts are the only way to find such a fit. On the `all` stratum the
  minimum is 25 SWR events per fit.
- `phi_source` — provenance of the concentration intercept used for that
  session's fit, one of `session` (the session estimated its own),
  `animal_pooled` (fewer modelled events than `--min-events-session-phi`,
  default 40, so the median intercept of that animal's longer sessions was
  pinned), or `pooled_all_animals` (the animal had no qualifying session
  either, so the all-animal median was pinned). A pinned fit's
  `bse_log_phi_intercept` is `NaN` and the parameter is excluded from
  `n_params`. Report the share of pinned rows alongside any concentration
  result — it is negligible in the comparison and the base ladder and large in
  the thin `omitted` stratum.
