# Content models — proposed design (parallel to the is_swr per-bin RATE ladder)

Working spec for review. The RATE ladder is built and run
(`scripts/rate/fit_swr_contrast_rate.py`, `docs/models_table.md` "is_swr version").
The CONTENT ladder below is proposed, not yet implemented. The goal: ask the
SAME task-dimension questions of replay *content* (what an event represents)
that the rate ladder asks of replay *rate* (how often events occur), harmonized
to the paper-faithful HMM covariates.

## Covariate dictionary (shared with the rate ladder)
Task dimensions, all from ONE model (`default_hmm`), per event/stop scalars:
- `reward_lag1` — current (consummatory) reward at this stop.
- `hmm_relative_value` (ΔV) — relative value (paper's single value+decision axis).
- `hmm_rpe` — reward prediction error (reward − Q_current_leaf).
- `reward_entropy` — HMM reward-state entropy (outcome uncertainty).
- `hmm_value_update` — global value update Σ|ΔQ|.
- CONTEXT split (near-orthogonal, VIF≈1): `PERISWITCH` = trials_since_switch +
  is_post_switch + their interaction; `BLOCK` = trials_since_block_change;
  `SWITCHRATE` = recent_switch_rate.
- `is_swr` — event-level SWR-coincidence flag.

Arm-/patch-level features (vary WITHIN an event → real main effects):
- `is_current` / `is_current_patch` — the animal's current arm/patch.
- `is_prev` / `is_prev_patch` — previous arm/patch.
- `is_last_rewarded_(elsewhere|patch)` — last rewarded arm/patch.
- `is_next_patch` — the upcoming patch (LOOK-AHEAD; see guardrails).
- `visit_recency` / `patch_visit_recency` — recency of visiting each arm/patch.
- `arm_dist_rank` — distance ordering from current position.
- `reward_rate_arm_k10` — per-arm reward memory (last 10 visits).
- `q_arm` — per-arm value (TO HARMONIZE: derive from HMM Q1..6, currently beta).
- `ARMFE` — arm fixed effects (8 dummies).

## Structural fact that shapes everything
Content is a **conditional Poisson over arms within event** (`strata(event_id)`,
≡ clogit). Any covariate CONSTANT within an event cancels in the conditional
likelihood → has NO main effect. All the task dimensions above are event-level
scalars → they enter ONLY as interactions with arm-/patch-features. This is why
`is_swr` enters as `is_swr:is_current`, never as a bare term.

Consequence for the `is_swr` dissociation:
- feature dims (is_current, is_last_rewarded, q_arm, ...) → `is_swr:feature` is a
  2-way term, identifiable, pooled-testable.
- event-scalar dims (rpe, value, entropy, update, context) → their content
  effect is already 2-way `X:is_current`; a pooled `is_swr` contrast is 3-way and
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
Baseline (spatial nuisance; `visit_recency`/`is_prev` are the autocorrelation
controls — the fix for the previously-invalidated content claims):
```
C0   arm_count ~ ARMFE
C1   + is_current                                  # current-arm bias (dominant signal)
C2   + is_prev + visit_recency + arm_dist_rank     # = BASEc (net of autocorrelation & distance)
```
Per-arm value/reward CONTENT (main effects; "represents which arms?"; no rate analog):
```
C3   + reward_rate_arm_k10        # represents recently-REWARDED arms?  [prior-null risk]
C4   + q_arm                      # represents high-VALUE arms?          [prior-null risk]
C5   + is_last_rewarded_elsewhere # represents the last-rewarded arm?
```
State modulation of the current-bias (event-scalar : is_current; rate-ladder analogs):
```
C6   + hmm_rpe:is_current             # does RPE pull content toward/off current?   (~G5_rpe)
C7   + hmm_relative_value:is_current  # does ΔV shift the bias?                       (~G4_value)
C8   + reward_entropy:is_current      # does uncertainty BROADEN off-current?         (~G6_uncert)
C9   + hmm_value_update:is_current    # does updating shift content?                  (~G7_update)
```
CONTEXT split (event-scalar : is_current; parallels rate G1/G2/G3):
```
C10  + PERISWITCH:is_current          # current-bias change around a switch?
C11  + BLOCK:is_current               # content drift across a contingency block?
C12  + SWITCHRATE:is_current          # volatility → broaden off-current?
```
is_swr dissociation (pooled, 2-way, identifiable — the robust prior findings):
```
CS1  BASEc + is_swr:is_current                 # "SWR broadens off-current"       (expect −)
CS2  BASEc + is_swr:is_prev                     # SWR off-prev
CS3  BASEc + is_swr:is_last_rewarded_elsewhere  # "SWR avoids last-rewarded"       (expect −)
CS4  C3    + is_swr:reward_rate_arm_k10          # reward-arm representation SWR-specific?
CS5  C4    + is_swr:q_arm                        # value-arm representation SWR-specific?
```
(Event-scalar dissociations = compare C6–C12 across the is_swr split, not pooled.)

## PATCH ladder (conditional Poisson over 3 patches; the natural home for SWITCH content)
```
CP0  patch_count ~ is_current_patch + is_prev_patch + patch_visit_recency   # = BASEc_patch
CP1  + is_last_rewarded_patch                          # represents the abandoned/rewarded patch?
CP2  + is_last_rewarded_patch:trials_since_switch      # does OLD-patch representation DECAY post-switch?
CP3  + is_current_patch:trials_since_switch            # does NEW-patch representation GROW into the switch?
CP4  + recent_switch_rate:is_current_patch             # volatility → leaves current patch?
```
Look-ahead (`is_next_patch`) — **DROPPED (decision 2026-07-02).** `is_next_patch`
does not exist in the codebase, and `next_patch == current_patch` 90.5% of events,
so a look-ahead coefficient is identified only from the ~9.5% next≠current events
(a switch-only restriction would not raise that ceiling, only discard baseline
information — and there are too few switches). It also reopens the already-
invalidated "predicts next patch" claim. The switch-anticipation question is
carried instead by `CP2`/`CP3` (old-patch decay / new-patch growth over
`trials_since_switch`), which use the continuous time-from-switch axis. Forward-
bias (ahead/behind) is omitted for the same reason.

is_swr dissociation (2-way, identifiable):
```
CPS1 + is_swr:is_current_patch
CPS2 + is_swr:is_last_rewarded_patch
CPS3 + is_swr:(is_last_rewarded_patch:trials_since_switch)   # is the old-patch decay SWR-specific?
```

## Guardrails / caveats (project history + independent review, 2026-07-02)
- **No rate-based pruning** (decision): the content ladder keeps ALL task
  dimensions (incl. `uncert` C8, `update` C9), regardless of whether they were
  significant on the RATE side. Content and rate are different questions — a
  dimension inert for event *rate* can still shape event *content*.
- INVALIDATED-by-autocorrelation set (flag as confirmatory-of-a-prior-null):
  "represents reward locations" (= C3/C4). `is_next`/"predicts next patch" and
  forward-bias are DROPPED (see patch ladder). C3/C4 earn a place ONLY to test
  whether they survive the autocorrelation controls (is_current + is_prev +
  visit_recency) AND are is_swr-specific in a way autocorrelation can't produce.
- **`is_current` is a modal-over-event summary** (`tables.py`: modal
  `track_segment_id`, trunk→`current_well` fallback), not a per-timebin state.
  `X:is_current` interactions on switch-straddling events (esp. `C10`) are
  boundary-noisy exactly where the effect concentrates — report the fallback-path
  fraction near `trials_since_switch≈0`, or window those events out.
- **Multi-df terms need JOINT tests, not per-coefficient reporting:** `PERISWITCH`
  (3 df; C10) and any spline interaction → report a joint LR test + fitted curves
  (predicted content vs `trials_since_switch`), never signed-rank on one loading.
- **`CS4`/`CS5`/`CS2` are expected-sparse / noisy** (inherit the M7 caveat: the
  is_swr × per-arm-reward/value subset is ~2% of events; `is_prev` is 83%
  collinear with `is_next` and was bidirectional/null before). Do NOT present at
  the same confidence as the robust `CS1`/`CS3`.
- **`C4`/`CS5` complete-case shrinkage:** `q_arm`/`p_reward_arm` are NaN on
  absent-contingency stops (the documented `Mv` problem) — co-fitting restricts
  the shared event set. State the retained-n up front.
- **`C8`/`C9` collinearity:** `reward_entropy`↔`hmm_value_update` r=0.42 (rpe↔value
  is fine at r=0.14) — VIF/rank smoke-test C8/C9 before trusting them as separate.
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
