# Content and rate uncertainty model ladders

**Status:** proposed Aim-1 design; not yet the implemented model catalog  
**Date:** 2026-07-13

This design reorganizes the replay analyses around two complementary estimands:

1. **content allocation:** conditional on observed replay content, where is it
   allocated?;
2. **event occurrence:** when does replay occur, and at what rate?

A small event-size and marginal-content bridge distinguishes more events from
longer events and from redistribution of a fixed amount of content:

\[
\frac{\text{well-specific content}}{\text{time}}
=
\frac{\text{events}}{\text{time}}
\times
\frac{\text{content}}{\text{event}}
\times
\frac{\text{well-specific content}}{\text{content}}.
\]

Aim 1 pools SWR-overlapping and non-SWR replay. SWR status is introduced only
after the Aim-1 ladders have been selected and validated.

## R-formula notation

For conditional content models, notation such as

```r
arm_count_hard ~ predictors | event_id
```

uses the `fixest` convention: `| event_id` is an absorbed event fixed effect.
It is equivalent to conditioning the Poisson likelihood on the total decoded
content in each event. It is not a random intercept. All continuous belief
predictors are standardized within session unless otherwise stated.

The formulas are scientific specifications rather than a commitment to a
particular R package. The production implementation may remain in Python as
long as it preserves the same response, candidate set, conditioning set,
covariate timing, and nested comparisons.

## Spatial decomposition of uncertainty

Let `U_leaf` be the pre-outcome Beta posterior variance for one of the six
reward wells. For trial `t`, patch `p`, and leaf `j`, define:

```r
U_global  <- rowMeans(cbind(U_leaf_1, ..., U_leaf_6))
U_patch   <- rowMeans(cbind(U_leaf_1, U_leaf_2))  # separately within each patch

U_local   <- U_leaf  - U_global             # relative well uncertainty
U_between <- U_patch - U_global             # relative patch uncertainty
U_within  <- U_leaf  - U_patch              # relative uncertainty within patch
```

Thus

```r
U_leaf == U_global + U_between + U_within
```

up to floating-point error. The components answer different questions:

| Component | Model family | Question |
|---|---|---|
| `U_global` | occurrence and event size | Does overall uncertainty engage more replay processing? |
| `U_between` | patch allocation | Is replay directed toward the more uncertain patch? |
| `U_within` | within-patch allocation | Within a represented patch, is replay directed toward its more uncertain well? |
| `U_local` | six-well allocation | Which relatively uncertain well receives content, combining between- and within-patch targeting? |

An event-level global scalar has no content-allocation main effect: because it
is constant across candidate locations in an event, it cancels from the
conditional likelihood. It can only modify a location-varying feature, such as
`U_global:is_current`; that is a different question from local uncertainty
targeting.

## A. Six-well content-allocation ladder

### Estimand and cohort

- Data grain: one row per `event_id x leaf`.
- Candidate set: reward-well arms 0--5 only.
- Response: `arm_count_hard`.
- Keep events with positive total leaf content.
- Primary estimand: decoded-bin-weighted allocation among reward wells,
  conditional on an observed replay event containing leaf content.
- Freeze one complete-case event cohort across A0--A6.

The primary response keeps continuity with the current conditional-Poisson
models. Repeat the final comparison with `arm_count_soft`, posterior mass, and
an event-balanced response such as dominant-well identity so longer events do
not automatically receive more weight.

```r
A0 <- arm_count_hard ~
    factor(arm) |
    event_id

A1 <- arm_count_hard ~
    factor(arm) +
    is_current +
    is_prev +
    is_current_patch +
    is_prev_patch +
    arm_dist_rank |
    event_id

A2 <- arm_count_hard ~
    factor(arm) +
    is_current + is_prev +
    is_current_patch + is_prev_patch +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere |
    event_id

A3 <- arm_count_hard ~
    factor(arm) +
    is_current + is_prev +
    is_current_patch + is_prev_patch +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere +
    p_reward_arm |
    event_id

A4 <- arm_count_hard ~
    factor(arm) +
    is_current + is_prev +
    is_current_patch + is_prev_patch +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere +
    p_reward_arm +
    q_arm + I(q_arm^2) |
    event_id

A5 <- arm_count_hard ~
    factor(arm) +
    is_current + is_prev +
    is_current_patch + is_prev_patch +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere +
    p_reward_arm +
    q_arm + I(q_arm^2) +
    U_local |
    event_id

A6 <- arm_count_hard ~
    factor(arm) +
    is_current + is_prev +
    is_current_patch + is_prev_patch +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere +
    p_reward_arm +
    q_arm + I(q_arm^2) +
    U_local +
    is_current:U_local |
    event_id
```

The primary uncertainty comparison is:

```r
A4 versus A5
```

`q_arm + I(q_arm^2)` protects against mistaking a nonlinear value effect for
posterior uncertainty. A spline can replace the quadratic if its support and
per-session identifiability are adequate.

### Current-location confirmation

Repeat A0--A5 on remote alternatives:

```r
data = subset(leaf_event, is_current == 0)
```

Remove `is_current` from these formulas but retain current-patch and distance
controls. If chosen and occupied wells are not identical in some rows, also run
the predeclared unchosen-well restriction. The primary uncertainty claim should
not depend on content assigned to the animal's occupied arm.

## P. Patch-content allocation

The patch response includes both reward wells and the trunk belonging to the
patch. `U_patch` is derived only from the two reward-well beliefs. A trunk is
therefore represented as patch content but is never assigned uncertainty zero.

```r
P0 <- patch_count_hard ~
    factor(patch) |
    event_id

P1 <- patch_count_hard ~
    factor(patch) +
    is_current_patch |
    event_id

P2 <- patch_count_hard ~
    factor(patch) +
    is_current_patch +
    reward_fraction_candidate_arm_last10_visits_patch |
    event_id

P3 <- patch_count_hard ~
    factor(patch) +
    is_current_patch +
    reward_fraction_candidate_arm_last10_visits_patch +
    swr_x_is_current_patch |
    event_id

Pv <- patch_count_hard ~
    factor(patch) +
    is_current_patch +
    reward_fraction_candidate_arm_last10_visits_patch +
    task_reward_probability_candidate_arm_patch +
    reward_belief_mean_candidate_arm_preoutcome_patch |
    event_id
```

The primary patch comparison is:

```r
P2 versus Pv
```

## W. Within-patch well allocation

Use the two leaf rows in each `event_id x patch` stratum and retain strata with
positive leaf content. This asks which well is selected after conditioning on
the patch being represented.

```r
W0 <- arm_count_hard ~
    factor(arm) +
    is_current +
    is_prev +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere +
    p_reward_arm +
    q_arm + I(q_arm^2) |
    interaction(event_id, patch)

W1 <- arm_count_hard ~
    factor(arm) +
    is_current +
    is_prev +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere +
    p_reward_arm +
    q_arm + I(q_arm^2) +
    U_within |
    interaction(event_id, patch)
```

The primary within-patch comparison is:

```r
W0 versus W1
```

## O. Replay-event occurrence ladder

### Estimand and cohort

- Data grain: one row per `stop x 15-ms bin`.
- Primary outcome: pooled non-local event-onset count.
- Offset: log of the bin's actual exposure.
- Primary window: first 4 s of each stop, matching Family B.
- Freeze one complete-case stop/trial cohort across O0--O5.
- Cross-validation splits whole behavioral trials, never bins or events sharing
  one belief state.

Define the reusable blocks:

```r
TIME <- splines::ns(
    time_since_stop,
    knots = c(1, 2, 3),
    Boundary.knots = c(0, 4)
)

HISTORY <- hist_rcos_0 + hist_rcos_1 +
           hist_rcos_2 + hist_rcos_3

REWARD <- reward_lag1 + reward_lag2 + reward_lag3 +
          reward_lag4 + reward_lag5 + reward_lag6 +
          reward_lag7 + reward_lag8 + reward_lag9 +
          reward_lag10

TASK <- factor(current_segment) +
        is_after_switch +
        trials_since_block_change +
        recent_switch_rate

VALUE <- q_current_pre + I(q_current_pre^2) +
         switch_advantage_mu_preoutcome_sess
```

`q_current_pre` denotes the pre-outcome Beta mean for the current/chosen option;
implementation should map it to the canonical current `qvals_table` column.

```r
O0 <- onset_count ~
    1 +
    offset(log(exposure))

O1 <- onset_count ~
    TIME +
    offset(log(exposure))

O2 <- onset_count ~
    TIME +
    HISTORY +
    offset(log(exposure))

O3 <- onset_count ~
    TIME +
    HISTORY +
    REWARD +
    TASK +
    offset(log(exposure))

O4 <- onset_count ~
    TIME +
    HISTORY +
    REWARD +
    TASK +
    VALUE +
    offset(log(exposure))

O5 <- onset_count ~
    TIME +
    HISTORY +
    REWARD +
    TASK +
    VALUE +
    U_global +
    offset(log(exposure))
```

The primary global-rate comparison is:

```r
O4 versus O5
```

Family A's full-stop count model with a log-immobility offset remains a
full-duration sensitivity. Same-trial RPE and value update do not enter
O0--O5 because they are post-outcome. If studied, they belong in a separate
post-outcome-restricted descriptive branch.

## D. Event-size bridge

Use one row per detected replay event. `n_nonlocal_bins` is positive, so use a
zero-truncated negative-binomial model with a log link and no exposure offset.

```r
D0 <- n_nonlocal_bins ~
    1

D1 <- n_nonlocal_bins ~
    splines::ns(time_since_stop, df = 4) +
    factor(current_segment) +
    REWARD +
    TASK

D2 <- n_nonlocal_bins ~
    splines::ns(time_since_stop, df = 4) +
    factor(current_segment) +
    REWARD +
    TASK +
    VALUE

D3 <- n_nonlocal_bins ~
    splines::ns(time_since_stop, df = 4) +
    factor(current_segment) +
    REWARD +
    TASK +
    VALUE +
    U_global
```

The primary event-size comparison is:

```r
D2 versus D3
```

## I. Marginal absolute well content

Use every `stop x leaf` row, including zero-replay stops. This analysis bridges
the occurrence and conditional-allocation estimands and protects against a
conclusion driven only by conditioning on detected events.

First model whether any content appears at a well:

```r
I_presence <- I(leaf_content_bins > 0) ~
    factor(arm) +
    is_current + is_prev +
    is_current_patch + is_prev_patch +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere +
    p_reward_arm +
    q_arm + I(q_arm^2) +
    U_global +
    U_between +
    U_within +
    log(immobility_time)
```

Fit `I_presence` with a binomial likelihood. Then model the positive amount:

```r
I_amount <- leaf_content_bins ~
    factor(arm) +
    is_current + is_prev +
    is_current_patch + is_prev_patch +
    arm_dist_rank +
    visit_recency_trials_event_leaf_centered +
    reward_rate_arm_k10 +
    is_last_rewarded_elsewhere +
    p_reward_arm +
    q_arm + I(q_arm^2) +
    U_global +
    U_between +
    U_within +
    offset(log(immobility_time))
```

Fit `I_amount` on positive rows with a zero-truncated negative-binomial
likelihood. Cluster or bootstrap inference at the stop, session, and animal
levels as appropriate; the headline biological sample size is five animals.

## Alternative uncertainty definitions

The primary construct is Beta posterior variance: uncertainty about reward
probability. Alternative constructs are sibling replacement models on the same
rows and folds, not additional terms in one model:

```r
A5_variance <- update(A4, . ~ . + U_local_variance)
A5_entropy  <- update(A4, . ~ . + U_local_beta_entropy)
A5_scarcity <- update(A4, . ~ . + U_local_inverse_concentration)

O5_variance <- update(O4, . ~ . + U_global_variance)
O5_entropy  <- update(O4, . ~ . + U_global_beta_entropy)
O5_scarcity <- update(O4, . ~ . + U_global_inverse_concentration)
```

Here evidence scarcity may be defined from Beta concentration, for example
`1 / (alpha + beta + 1)`. Do not co-fit the correlated uncertainty definitions
and interpret their individual coefficients. Moreover, `q * (1 - q)` is
algebraically contained in `q + I(q^2)`, so predictive ambiguity cannot be an
additional independent term after the quadratic value adjustment.

Report the construct-correlation matrix, common support, coefficient stability,
and held-out improvement for each sibling model.

## SWR-status extension for Aim 2

Aim 1 reports pooled-event A/P/W/O/D/I results. After selecting the Aim-1 model,
either fit the full model separately to SWR-overlapping and non-SWR events and
compare paired session/animal coefficients, or fully interact SWR status with
the complete location-varying baseline.

For content, the saturated interaction has the form:

```r
A5_SWR <- arm_count_hard ~
    BASE_A5 +
    is_swr:(
        factor(arm) +
        is_current + is_prev +
        is_current_patch + is_prev_patch +
        arm_dist_rank +
        visit_recency_trials_event_leaf_centered +
        reward_rate_arm_k10 +
        is_last_rewarded_elsewhere +
        p_reward_arm +
        q_arm + I(q_arm^2) +
        U_local
    ) |
    event_id
```

The bare `is_swr` main effect is event-constant and is absorbed by `event_id`.
Interacting only `is_swr:U_local` while forcing all other coefficients to be
shared is not the primary contrast.

For rate, first retain O5 as the pooled replay-occurrence model. Then model SWR
overlap as an event mark:

```r
SWR_mark <- is_swr ~
    splines::ns(time_since_stop, df = 4) +
    factor(current_segment) +
    REWARD +
    TASK +
    VALUE +
    U_global
```

This separates a change in total replay occurrence from a change in the
fraction of replay events overlapping an SWR.

## Primary comparisons and claim boundaries

```r
A4 versus A5   # relative well targeting
P3 versus P4   # between-patch targeting
W0 versus W1   # within-patch targeting
O4 versus O5   # global event occurrence
D2 versus D3   # content amount per event
```

| Result | Licensed interpretation |
|---|---|
| A5 positive, O5 null | uncertainty reallocates existing replay content |
| A5 null, O5 positive | uncertainty increases general replay engagement, not location-specific targeting |
| A5 and O5 positive | uncertainty both gates replay and targets its content |
| adjusted I positive | uncertain locations receive more absolute replay content, including zero-replay opportunities |
| D3 positive | uncertainty is associated with longer/richer detected events |

A5 alone supports: "among observed replay events, content is preferentially
allocated to locations with greater model-inferred reward uncertainty."
Only O5 licenses an event-rate claim, and only the adjusted marginal I models
license the broader statement that uncertain locations receive more absolute
replay. None of these models alone establishes that replay reduces uncertainty,
causes exploration, or improves learning.

## Relationship to current repository models

- Existing content M0a--M5 supply most of A0--A2.
- Mv supplies the objective/subjective value ingredients for A3--A4.
- Mu/Mvu and C13 supply the local-uncertainty machinery for A5, but the primary
  refit should use leaves only rather than nine arms with trunk uncertainty zero.
- C8 (`global uncertainty x current`) remains a separate current-anchoring
  question and is not the local-uncertainty rung.
- Family-B M0/M5/M3r/M7r supply the occurrence baseline mechanics for O0--O3.
- The current G models are `BASE + one block` sibling tests, not the nested
  O0--O5 Aim-1 ladder.
- The primary O5 construct should be mean six-leaf Beta variance;
  `beta_reward_entropy` is a sibling sensitivity.
- Existing SWR C/CS and G/S models move to the Aim-2 extension rather than being
  interleaved with Aim 1.

Once implemented, the exact production formulas belong in `docs/models_table.md`
and `src/continuum_swr_replay/analysis/behavior/model_specs.py`; this document
remains the scientific design and claim contract.
