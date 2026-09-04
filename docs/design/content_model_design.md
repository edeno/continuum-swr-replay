# Content models — design spec (parallel to the is_swr per-bin RATE ladder)

Design spec for the content ladder, now **implemented** in
`scripts/content/fit_swr_contrast_content.py` (the RATE ladder is
`scripts/rate/fit_swr_contrast_rate.py`; see `docs/models_table.md` "is_swr version").
The goal: ask the SAME task-dimension questions of replay *content* (what an event
represents) that the rate ladder asks of replay *rate* (how often events occur),
using the beta-bernoulli covariates (the HMM was retired 2026-07-13).

## Covariate dictionary (shared with the rate ladder)
Task dimensions, all from the beta-bernoulli model, per event/stop scalars:
- `known_reward_outcome_lag_1` — most recent outcome known at bout start.
- `relative_value_switch_minus_stay_preoutcome` (ΔV) — beta relative value,
  **Switch − Stay**
  (r = −0.952 with the stay value, so a positive
  `relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm` means the current
  arm's value is *low*; an earlier "Stay-vs-Switch" label here was backwards).
- `rpe_postoutcome` — beta reward prediction error (reward − pre-outcome
  chosen-arm belief mean).
- `reward_belief_differential_entropy_leaf_mean_preoutcome` — mean per-leaf
  Beta differential entropy.
- `reward_belief_mean_update_l1_all_ports_postoutcome` — beta global value
  update Σ|Δμ|.
- CONTEXT split: `PERISWITCH` = `periswitch_index_trials` +
  `is_periswitch_post` + their interaction; `BLOCK` =
  `trials_since_block_change`; `SWITCHRATE` =
  `patch_switch_fraction_last5_completed_trips`.
- `is_swr` — event-level SWR-coincidence flag.

Arm-/patch-level features (vary WITHIN an event → real main effects):
- `is_current_arm` / `is_current_patch` — the animal's current arm/patch.
- `is_prev_arm` / `is_prev_patch` — previous arm/patch.
- `is_last_rewarded_noncurrent_arm` / `is_last_rewarded_noncurrent_patch` —
  last rewarded arm/patch when distinct from current.
- `is_next_patch` — the upcoming patch (LOOK-AHEAD; see guardrails).
- `visit_recency_trials` / `visit_recency_trials_patch` — trial recency of
  visiting each arm/patch.
- `candidate_arm_distance_rank` — distance ordering from current position.
- `reward_fraction_candidate_arm_last10_visits` — per-arm reward memory.
- `reward_belief_mean_candidate_arm_preoutcome` — per-arm beta belief mean.
- `ARMFE` — arm fixed effects (8 dummies).

## Structural fact that shapes everything
Content is a **conditional Poisson over arms within event** (`strata(event_id)`,
≡ clogit). Any covariate CONSTANT within an event cancels in the conditional
likelihood → has NO main effect. All the task dimensions above are event-level
scalars → they enter ONLY as interactions with arm-/patch-features. This is why
SWR status enters as `swr_x_is_current_arm`, never as a bare term.

Consequence for the `is_swr` dissociation:
- feature dimensions (current, last rewarded, value, ...) enter as a canonical
  `swr_x_<feature>` term, which is a
  2-way term, identifiable, pooled-testable.
- event-scalar dimensions (RPE, value, entropy, update, context) → their content
  effect is already a two-way `X_x_is_current_arm` product; a pooled SWR contrast
  is three-way and
  separation-prone (cf. rate S10). Dissociation for these comes from fitting the
  ladder PER is_swr split {all, SWR, non-SWR} and comparing coefficients.

## RATE ↔ CONTENT correspondence
| | rate | content |
|---|---|---|
| unit | (stop × 15ms bin), Poisson count | (event × arm), conditional Poisson |
| "offset" | log(exposure) | strata(event_id) |
| task dim asks | change HOW OFTEN events occur? | change WHICH ARM is represented? |
| dim enters as | main effect | X : arm-feature interaction |
| dissociation | is_swr:X (2-way) | is_swr:X:feature (3-way) or via split |

## ARM ladder (conditional Poisson over 9 arms, strata(event_id); per is_swr split)
Baseline (spatial nuisance; `visit_recency_trials`/`is_prev_arm` are the autocorrelation
controls — the fix for the previously-invalidated content claims):
```
C0   arm_count ~ ARMFE
C1   + is_current_arm
C2   + is_prev_arm + visit_recency_trials + candidate_arm_distance_rank
```
Per-arm value/reward CONTENT (main effects; "represents which arms?"; no rate analog):
```
C3   + reward_fraction_candidate_arm_last10_visits
C4   + reward_belief_mean_candidate_arm_preoutcome
C5   + is_last_rewarded_noncurrent_arm
```
State modulation of current-arm bias (event scalar × `is_current_arm`; rate-ladder analogues):
```
C6   + rpe_postoutcome_z_x_is_current_arm
C7   + relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm
C8   + reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm
C9   + reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm
```
CONTEXT split (event scalar × `is_current_arm`; parallels rate G1/G2/G3):
```
C10  + periswitch_index_trials_z_x_is_current_arm
C11  + trials_since_block_change_z_x_is_current_arm
C12  + patch_switch_fraction_last5_completed_trips_z_x_is_current_arm
```
is_swr dissociation (pooled, 2-way, identifiable — the robust prior findings):
```
CS1  BASEc + swr_x_is_current_arm
CS2  BASEc + swr_x_is_prev_arm
CS3  BASEc + swr_x_is_last_rewarded_noncurrent_arm
CS4  C3    + swr_x_reward_fraction_candidate_arm_last10_visits
CS5  C4    + swr_x_reward_belief_mean_candidate_arm_preoutcome
```
(Event-scalar dissociations = compare C6–C12 across the is_swr split, not pooled.)

## PATCH ladder (conditional Poisson over 3 patches; the natural home for SWITCH content)
```
CP0  patch_count ~ is_current_patch + is_prev_patch + visit_recency_trials_patch
CP1  + is_last_rewarded_noncurrent_patch
CP3  + periswitch_index_trials_z_x_is_current_patch
CP4  + patch_switch_fraction_last5_completed_trips_z_x_is_current_patch
```
Look-ahead (`is_next_patch`) — **DROPPED (decision 2026-07-02).** `is_next_patch`
does not exist in the codebase, and `next_patch == current_patch` 90.5% of events,
so a look-ahead coefficient is identified only from the ~9.5% next≠current events
(a switch-only restriction would not raise that ceiling, only discard baseline
information — and there are too few switches). It also reopens the already-
invalidated "predicts next patch" claim. The switch-anticipation question is
carried instead by `CP2`/`CP3` (old-patch decay / new-patch growth over
`periswitch_index_trials`), which uses the continuous time-from-switch axis. Forward-
bias (ahead/behind) is omitted for the same reason.

is_swr dissociation (2-way, identifiable):
```
CPS1 + swr_x_is_current_patch
CPS2 + swr_x_is_last_rewarded_noncurrent_patch
```

## Guardrails / caveats (project history + independent review, 2026-07-02)
- **No rate-based pruning** (decision): the content ladder keeps ALL task
  dimensions (incl. `uncert` C8, `update` C9), regardless of whether they were
  significant on the RATE side. Content and rate are different questions — a
  dimension inert for event *rate* can still shape event *content*.
- INVALIDATED-by-autocorrelation set (flag as confirmatory-of-a-prior-null):
  "represents reward locations" (= C3/C4). `is_next`/"predicts next patch" and
  forward-bias are DROPPED (see patch ladder). C3/C4 earn a place ONLY to test
  whether they survive the autocorrelation controls (`is_current_arm` +
  `is_prev_arm` +
  `visit_recency_trials`) AND are SWR-specific in a way autocorrelation can't produce.
- **`is_current_arm` uses strict whole-window at-well context**:
  `occupancy_well_arm` is finite only when every event bin is at the same well;
  transit, mixed,
  and unknown events have no current arm. `physical_arm` separately preserves
  the modal observed segment. Report the explicit location-state composition and
  use an at-well sensitivity cohort for any `X_x_is_current_arm` interpretation.
- **Multi-df terms need JOINT tests, not per-coefficient reporting:** `PERISWITCH`
  (3 df; C10) and any spline interaction → report a joint LR test + fitted curves
  (predicted content vs `periswitch_index_trials`), never signed-rank on one loading.
- **`CS4`/`CS5`/`CS2` are expected-sparse / noisy** (inherit the M7 caveat: the
  SWR × per-arm-reward/value subset is ~2% of events; `is_prev_arm` is 83%
  collinear with `is_next` and was bidirectional/null before). Do NOT present at
  the same confidence as the robust `CS1`/`CS3`.
- **`C4`/`CS5` complete-case shrinkage:**
  `reward_belief_mean_candidate_arm_preoutcome` and
  `task_reward_probability_candidate_arm` are NaN on
  absent-contingency stops (the documented `Mv` problem) — co-fitting restricts
  the shared event set. State the retained-n up front.
- **`C8`/`C9` collinearity:** smoke-test VIF/rank for
  `reward_belief_differential_entropy_leaf_mean_preoutcome`↔
  `reward_belief_mean_update_l1_all_ports_postoutcome`
  before trusting C8/C9 as separate (the HMM-era pair was r=0.42).
- G-ladder cross-split comparisons are INFORMAL (two independent fits, different
  residual structure); only the pooled `is_swr:feature` interactions carry a valid
  SE on the *difference*. State which is authoritative when they disagree.
- Separation: keep pooled interactions ≤2-way; use the is_swr split for 3-way.
- Aggregation: per-session fit → across-session signed-rank / LR (no random effects),
  identical to the rate side.

(This doc holds the content model DESIGN only. Model specifications live in
`docs/models_table.md` (R notation) and the runner `scripts/rate/fit_swr_contrast_rate.py`;
fitted rate output is `Processed-Data/replay_content/models_per_bin_rate/
swr_contrast_rate.pkl`. Neither catalogs nor design docs hold analysis RESULTS.
The only rate dependency here is the *decision* above that content is NOT pruned
on rate significance.)
