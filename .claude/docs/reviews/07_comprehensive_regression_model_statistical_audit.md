# Comprehensive Regression-Model Statistical Audit

**Date:** 2026-07-10  
**Scope:** Replay-content, replay-rate, SWR-contrast, uncertainty/value, behavioral-choice,
Stage-6 stay/switch, SWR-power, cross-history, and PFC regression models.  
**Question:** Do the models follow good statistical practice, use accurate and appropriately
timed covariates, manage collinearity and dependence, and answer the scientific questions claimed
for them?  
**Method:** Read-only code/specification review, inspection of persisted production artifacts,
production-data design and alignment checks, and targeted test execution. No model artifacts were
regenerated and no scientific code was changed.

This is a cross-family synthesis of the earlier reviews in this directory. Where this review
conflicts with an earlier assessment, the production-data checks dated 2026-07-10 supersede the
earlier statement. In particular, the earlier claim that Family-B AIC/CV comparisons use a common
cohort is incorrect.

---

## Executive verdict

**No: not all current regressions ask or answer the claimed question.**

The arm/patch content and per-bin event-intensity frameworks have sensible core estimands. Several
other models either answer a substantially narrower question than their labels imply or are not
currently statistically interpretable. The main blockers are:

1. `recent_switch_rate` leaks the current trial's future decision.
2. Many likelihood, AIC, BIC, and CV comparisons use different observations.
3. The SWR-power model is algebraically rank deficient in 37.2% of sessions.
4. Choice and Stage-6 models pseudo-replicate decisions by using stops rather than distinct trials.
5. Population inference treats sessions or neurons as biological replicates despite only five
   animals.
6. Model-based uncertainty ignores important within-event, within-stop, temporal, and hierarchical
   dependence.
7. Persisted artifacts come from mixed code/specification versions and cannot safely be summarized
   together.

The current cached results should therefore not be treated as manuscript-ready. A clean, versioned
refit is required after the critical issues below are fixed.

---

## Critical findings

### 1. `recent_switch_rate` leaks the current trial's future choice

`add_uncertainty_covariates` computes the variable by shifting and rolling over **stop rows**
([`covariates.py:518`](../../../src/continuum_swr_replay/analysis/replay/covariates.py#L518)). The
source `is_stay` variable is post-hoc: it is defined from `current_patch == next_patch`
([`tables.py:996`](../../../src/continuum_swr_replay/analysis/replay/tables.py#L996)). Shifting by one
stop is not equivalent to shifting by one completed decision because a trial can contain multiple
immobility stops.

Production-data checks found:

- 60.9% of mapped stops immediately follow another stop from the same trial.
- This rises to 77.2% for event-bearing stops.
- `corr(recent_switch_rate, current is_stay) = -0.665` overall.
- The correlation is `-0.800` where the preceding stop belongs to the same trial.

This invalidates the current switch-rate terms in **MU, G3, S5, C12, CP4**, the SWR-power model,
and related exploratory content/rate interactions. The timing registry currently labels this
variable as backward-safe ([`covariate_timing.py:59`](../../../src/continuum_swr_replay/analysis/replay/covariate_timing.py#L59)); that label is false for the materialized data.

**Required correction:** compute switching once per distinct completed trial from the realized
patch change, lag by one completed trial, roll over five completed trials, and join the resulting
value to stops/events by timestamp. Add a test containing duplicate stops within a trial.

### 2. Model comparisons frequently use different observations

The per-bin and per-stop fitters independently perform complete-case filtering for each model
([`per_bin_rate_model.py:129`](../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L129),
[`per_stop_rate_model.py:86`](../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L86)).
Building one upstream design does not make the retained fit rows common.

Persisted Family-B fits show, for example:

- M0/M5 median retained bins: approximately 31,518.
- M2/M7/M7r median retained bins: approximately 29,049.
- M0 versus reward-kernel rungs differ in all 218 paired sessions, with a median loss of about
  2,349 bins.
- R1 differs from R0 in 188/218 sessions; Rlag differs from R0 in 217/217 sessions.

Because `-2 log L`, AIC, and BIC depend on the observed response rows, the claim that M7r has lower
AIC than M5 is not a valid head-to-head comparison. Cross-model CV is also cohort-confounded when
each model is evaluated on its own complete cases.

The generic comparison helper checks only counts, not exact IDs, row hashes, nesting, likelihood
family, convergence, or cohort identity
([`model_comparison.py:48`](../../../src/continuum_swr_replay/analysis/replay/model_comparison.py#L48)).
Equal counts can still refer to different observations. It also clips negative LR statistics and
non-positive degrees of freedom into apparently valid p-values rather than rejecting the comparison.

The base arm/patch content runner is an important exception: it freezes a union-complete event
cohort before fitting. Comparisons within that run are valid if the models are estimable, although
the addition of Mv/Mu/Mvu makes simpler-model estimates conditional on the richer models' complete
cases. The SWR-content contrast ladder does not freeze a common cohort.

**Required correction:** freeze and persist exact row IDs and fold assignments for every declared
comparison family. Store a cohort hash and verify it exactly before computing likelihood-based or
predictive differences.

### 3. SWR-power coefficients are frequently unidentified

The model jointly includes `reward_lag1`, `rpe`, and `q_current`
([`fit_swr_power_model.py:45`](../../../scripts/rate/fit_swr_power_model.py#L45)), although
`rpe = reward - q_chosen`. On post-arrival/event cohorts, `reward_lag1` often represents that same
realized reward.

Production checks on the response-complete design found:

- 218 fitted sessions total.
- Full rank: 137 sessions.
- Rank 7 rather than 8: 72 sessions.
- Rank 6 rather than 8: 9 sessions.
- Therefore **81/218 sessions (37.2%) are rank deficient**.
- `reward_lag1 == rpe + q_current` for a median 98.7% of events per session.
- Some maximum VIFs approach `1e15`.

The script uses unchecked `np.linalg.lstsq`
([`fit_swr_power_model.py:75`](../../../scripts/rate/fit_swr_power_model.py#L75)), which silently
returns an arbitrary minimum-norm solution. It saves no rank, condition number, SE, CI, residual,
influence, heteroskedasticity, or predictive diagnostics.

**Conclusion:** current reward/RPE/Q coupling coefficients are not interpretable.

**Required correction:** choose a scientifically identified parameterization—typically reward + Q
or RPE alone, not all three—then enforce SVD rank/condition gates. Include nonlinear event duration
in the model rather than checking duration only by a separate marginal correlation.

### 4. Choice and Stage-6 models pseudo-replicate decisions

The choice likelihood groups observations by `stop_id`
([`choice_prediction.py:423`](../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py#L423)).
However, multiple stops can precede the same eventual behavioral decision.

Production checks found:

- 22,232 eligible replay stops.
- Only 18,230 distinct mapped trial decisions.
- 27.1% of stop observations belong to repeated decisions.
- One decision is represented by as many as 236 stops.

Consequently, long or multi-stop trials receive more weight and the same response can enter the
likelihood repeatedly. Stage-6 stay/switch regressions have the same stop-weighted target problem.

The choice builder also removes the current arm from the candidate set
([`choice_prediction.py:398`](../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py#L398))
and then drops groups without a remaining chosen arm. In production, 1,378/23,610 replay-stop
choices (5.84%) have `next_arm == current_arm` and are silently excluded.

**Required correction:** aggregate all eligible pre-choice replay to one choice set per distinct
decision, or prespecify one stop such as the final pre-departure stop. Either include the current
arm or explicitly label the estimand as choice conditional on not repeating the current arm and
show a sensitivity analysis.

### 5. Persisted artifacts do not match the current specifications

Examples include:

- The saved per-bin results predate current convergence/separation diagnostics.
- The saved SWR-contrast rate artifact lacks diagnostics emitted by the current runner.
- The cached choice fits use old predictor names and omit the current `arm_dist_rank` baseline.
- Cached SWR-split choice fits contain extreme estimates: 14/183 `replay_frac_swr` and 7/183
  `replay_frac_non_swr` coefficients have `|beta| >= 10`, with ranges beyond +/-1,000.
- Stage-6 and PFC artifacts predate relevant code/schema changes.
- Individual model files and combined `all_models` files have mixed run dates.

No current claim should combine these files. Each run should use an atomic versioned output
directory containing git hash, data-build hash, model-spec hash, CLI arguments, exact formulas,
cohort hashes, fold hashes, and schema version. Consumers should fail closed on mismatch.

---

## High-priority statistical issues

### Hierarchical inference and pseudo-replication

Many paper-facing summaries apply Wilcoxon tests across up to 218 sessions even though there are
only five animals. Stage-6 GEE clusters on session rather than animal
([`learning.py:585`](../../../src/continuum_swr_replay/analysis/replay/learning.py#L585)); PFC
summaries similarly treat neurons as independent biological replicates.

Session- and neuron-level results can describe consistency, but they do not independently support
population generalization. The newer contrast aggregator correctly calls session signed-rank tests
descriptive ([`contrast_aggregation.py:8`](../../../src/continuum_swr_replay/analysis/replay/contrast_aggregation.py#L8)).
With five animals, even 5/5 same-direction effects have a minimum two-sided exact sign-test
`p = 0.0625`.

Primary summaries should show all five animal effects and an animal-balanced effect estimate.
Animal-block bootstrap, leave-one-animal-out sensitivity, or a hierarchical animal/session model
can be used, but uncertainty from only five top-level units must remain explicit.

### Dependence, overdispersion, and optimistic uncertainty

- Conditional-Poisson content fits treat decoded category counts as if their standard likelihood
  uncertainty were adequate. The repository's own simulation demonstrates that within-event
  trajectory autocorrelation makes nominal SEs too small and p-values anti-conservative
  ([`test_replay_content_simulation.py:417`](../../../tests/unit/test_replay_content_simulation.py#L417)).
- Content CV splits by event, not stop. Because 86.2% of events occur in multi-event stops, a median
  of roughly 80% of held-out events share their stop with training events. Folds should be grouped
  by stop or contiguous trial blocks.
- Per-stop Poisson uses model-based SEs despite serial stops and possible overdispersion.
- Per-bin Poisson partially represents dependence through self-history but still uses ordinary GLM
  SEs and lacks an explicit design-rank gate.
- SWR/non-SWR stacked rate rows share a stop-bin but are not treated as paired or clustered.
- PFC models treat 2-ms spike bins as independent while omitting spike history and baseline drift
  ([`glm.py:446`](../../../src/continuum_swr_replay/analysis/pfc/glm.py#L446)). Their LR p-values
  are consequently optimistic.

### Separation, nonconvergence, and high-dimensional fits

Separation is common and is not consistently excluded from inference:

- Base M5: 74/213 session fits flagged separated.
- Sparse last-rewarded contrast rungs are frequently or universally quasi-separated in some strata.
- Per-stop Rlag non-local-SWR fits include 84/217 separated fits; the headline lag-1 median rate
  ratio changes from approximately 12.6 to 6.7 when separated fits are excluded.
- M9, MU, and S-rung designs can have 30-40+ coefficients despite a generic gate of only 20 positive
  event bins.

The universal `|coefficient| >= 10` separation heuristic is itself not scale invariant and often
flags raised-cosine history terms. Diagnostics should use standardized designs, Hessian/singular
values, fitted probabilities/intensities, and profile behavior. Sparse terms should use penalized
or partially pooled fits, with explicit sensitivity excluding bad fits.

### Conditional-model collinearity is measured on the wrong design

Conditional Poisson/logit identify only **within-event** or **within-choice-set** contrasts. The
rank check correctly centers by group, but current VIF and condition-number calculations use raw
`X` ([`fitting.py:437`](../../../src/continuum_swr_replay/analysis/replay/fitting.py#L437)).

Correct centered and standardized diagnostics on production data found:

| Design | Median maximum VIF | Additional result |
|---|---:|---|
| Base C2 | 6.0 | 90th percentile about 12.2; maximum about 68 |
| C13 arm uncertainty | 19.5 | Most sessions exceed VIF 10 |
| CS6 uncertainty contrast | 20.4 | Strong instability |
| Mvu value + uncertainty | 23.2 | 90th percentile about 40; maximum about 599; 5 rank-deficient sessions |

The Beta-model mean and uncertainty covariates are mechanically coupled and differ structurally
between leaf and trunk arms. Mvu-style coefficients should not be interpreted as cleanly separating
exploitation from exploration. Prefer a leaf-only estimand, within-arm centering/residualization,
or a prespecified low-dimensional behavioral construct.

Choice is a useful contrast: its current centered VIFs are modest (about 1.26-1.85), but geometry
still dominates. The selected candidate is nearest in 92.37% of stops and in a median 96.1% of each
session. Replacing raw distance with distance rank improves numerical stability; it does not remove
geometric confounding.

### Post-outcome and future-referencing covariates

`hmm_rpe` and `hmm_value_update` contain the current outcome, while `trials_since_switch` and
`is_post_switch` reference the next switch. The code's descriptive allowlists prevent accidental
forward-prediction labels, but the runners generally do not gate rows to those occurring after the
outcome. Q/RPE/update values can therefore be broadcast over pre-outcome exposure bins or events.

Affected examples include G5/S7, G7/S9, C6/C9, MU, and Stage 6. These models can support only an
association with the eventual outcome unless every fitted bin/event is explicitly gated after
outcome delivery. Peri-switch variables support descriptive alignment, not prospective switching
prediction.

### Multiple testing and formal contrasts

- Separate SWR and non-SWR coefficients are not a test of their difference. Only pooled S/CS/CPS
  interactions answer that question formally.
- "Significant in one stratum but not the other" is not evidence of a stratum difference.
- "Shared" or "equivalent" effects require an interaction CI and a prespecified equivalence margin;
  failure to reject a difference is not equivalence.
- Multi-degree-of-freedom splines, reward histories, and peri-switch terms need joint Wald/LR tests,
  not a conclusion based on one selected loading.
- Multiplicity correction should cover one prespecified confirmatory family across all claims, not
  be restarted opportunistically for each plot or subgroup.

---

## Covariate-accuracy assessment

### What is working

- Event-arm and event-patch cardinalities and fraction normalization pass their production
  invariants.
- Current-arm builders agree on approximately 99.95% of rows.
- Q-value table coverage is approximately 98.1%, with about 99.92% current-arm alignment at wells
  and 99.86% reward alignment.
- `reward_lag1` is constructed using a strict past-only goal-arrival cutoff and is the safest
  primary immediate-reward covariate.
- Arm and patch fixed-effect construction and within-event rank checks are generally careful.

### Qualifications and errors

1. The missing 1.9% of Q-value rows are concentrated at epoch ends and are later/reward-enriched;
   complete-case loss is therefore not obviously ignorable.
2. `recent_switch_rate` is future-leaky, as described above.
3. `reward_rate_arm_k10 = 0` conflates a never-visited arm with a visited arm having known-zero
   reward history. Add an observation-count/history indicator or a shrinkage prior.
4. `cumulative_reward` is actually a count over the last ten trials and should be renamed.
5. Nominal reward probabilities show imperfect empirical calibration: the aggregate calibration
   slope is about 0.89 and the maximum absolute nominal-versus-empirical deviation is about 0.093.
   Use these as scheduled probabilities, not empirically exact reward probabilities.
6. An existing audit reports no events before reward pump, but that check filters on the known-broken
   `is_reward_stop` flag. Re-audit using actual pump and port-entry timestamps.
7. External Beta-Bernoulli and HMM behavior models are not fitted in this repository. Local code can
   verify join alignment and timing, but cannot establish upstream parameter recovery or model fit.

---

## Model-family question and validity map

| Family | Actual estimand/question | Audit verdict |
|---|---|---|
| Arm/patch content | Relative allocation of decoded time bins across arms/patches, conditional on an observed event | Good core compositional question. It is not event occurrence or the probability that an entire event "represents" an arm. Requires block-robust uncertainty and soft-posterior sensitivity. |
| Post-switch/ContFrag content | Same composition estimand on selected event definitions or post-switch subsets | Useful sensitivity/selected-cohort description, not independent confirmatory evidence. |
| Per-stop Poisson | Average event rate per second over an immobility stop | Reasonable secondary/literature model. Needs common comparison cohorts, overdispersion checks, and trial/animal-aware uncertainty. |
| Per-bin point-process | Event-onset intensity during the first four seconds as a function of within-stop time, reward history, and event history | Strongest rate framework. Current ranking is invalid because cohorts differ; late-time effects are also conditional on long stops. |
| G rate / split C content | Separate stratum-specific SWR and non-SWR associations | Descriptive only; cannot establish a difference by comparing significance labels. |
| S/CS/CPS pooled interactions | Formal SWR-versus-non-SWR contrasts | Conceptually the right difference question, subject to dependence, timing, leakage, and estimability issues. |
| MU/value/uncertainty | Conditional rate/content association with model value, uncertainty, RPE, and updates | High-dimensional and unstable; current coefficients do not cleanly isolate value, RPE, or uncertainty. |
| Choice conditional logit | Destination among non-current arms, conditional on a replay-containing stop and a retained non-repeat choice | Much narrower than "replay predicts upcoming choice." Uses the wrong decision grain, omits a direct Q/value baseline, is geometry-dominated, and has no held-out validation. |
| Stage-6 stay/switch | Association among eventual outcome, replay content, and eventual stay/switch on replay-containing stops | Does not establish replay-mediated learning or behavioral gating. Needs one row per decision, temporal gating, behavioral controls, and animal hierarchy. |
| SWR-power OLS | Intended conditional association between event power and task variables | Invalid in its current form because of algebraic collinearity and absent inferential diagnostics. |
| Cross-history rate | Whether past events in one stream improve conditional prediction of the other stream | A lagged predictive/Granger-like association, not evidence that one stream drives the other; common inputs remain possible. |
| PFC theta/event GLMs | Descriptive theta or peri-event firing modulation within a neuron/session | Useful exploratory association. Does not establish event-driven firing, a matched-control effect, population generality, or SWR/non-SWR equivalence. |

---

## Required model-comparison practice

### 1. Freeze the comparison data first

For each declared comparison, persist:

- exact response-row IDs;
- one cohort hash;
- exact train/test fold IDs;
- response and likelihood identity;
- parent/child formula hashes;
- exclusions and retained-N by hierarchy level.

Parent and child models must use identical rows. A count equality check is insufficient.

### 2. Match the comparison to the question

| Question | Appropriate comparison |
|---|---|
| Does a nested unregularized block improve in-sample fit? | LRT plus paired delta-AIC/delta-BIC, on identical rows and only estimable/converged fits |
| Which nonnested model predicts better? | Identical-fold held-out delta log score, deviance, or ELPD |
| Does SWR differ from non-SWR? | One pooled interaction/contrast with an uncertainty estimate |
| Are two effects practically equivalent? | Equivalence interval/test with a prespecified margin |
| Does the model generalize across time? | Contiguous trial/stop-block CV |
| Does the claim generalize across sessions or animals? | Leave-session/animal-out validation where estimable, plus animal-level effects |

Do not summarize absolute AIC across heterogeneous sessions. Summarize paired within-session
differences only when each pair used the same observations. A Wilcoxon p-value on delta-AIC is not
a substitute for a calibrated model-selection procedure.

### 3. Treat folds as part of the design

- Content: split by stop or contiguous decision blocks, not event.
- Per-bin rate: use identical stop/trial blocks across candidate models; prefer contiguous blocks
  over random interleaving.
- Choice/Stage 6: split distinct decisions, not stops.
- Population prediction: explicitly state whether the target is a new stop, new session, or new
  animal; these require different validation schemes.

---

## Required reporting practice

Each model-summary row should report:

1. response, link, exposure/offset, observation grain, and exact estimand;
2. covariate timing class: past-only, contemporaneous, post-outcome, or future-referencing;
3. animals, sessions, decisions, stops, events, and bins retained;
4. cohort-retention percentage and missingness reason;
5. convergence, separation, rank, and per-term estimability;
6. centered/scaled singular values, condition number, and maximum VIF;
7. overdispersion, residual, calibration, or time-rescaling diagnostics appropriate to the family;
8. transformed effect size—rate ratio or odds ratio—with a defensible CI;
9. all five animal-specific effects and an animal-balanced summary;
10. formula, code hash, data-build hash, cohort hash, and fold hash.

Do not lead with counts of significant sessions or neurons. Do not interpret separated estimates,
arbitrary pseudo-inverse solutions, or stale artifacts. Distinguish clearly between coefficient
inference and held-out prediction.

### Family-specific sensitivity analyses

- **Content:** hard MAP counts, soft counts/posterior mass, and an event-weighted dominant-category
  analysis; block bootstrap by stop/session.
- **Rate:** Poisson versus negative-binomial/quasi-Poisson or robust/bootstrapped uncertainty;
  common-cohort raised-cosine model; time-rescaling and held-out residual checks.
- **Choice:** include direct arm value/Q and geometry baselines, replay dose as well as composition,
  same-arm choices, and decision-blocked CV.
- **Stage 6:** behavioral baseline, one row per decision, post-outcome gating, nonlinear content,
  animal/session hierarchy, and multiplicity across all role/stratum tests.
- **PFC:** canonical event labels, spike history and slow drift, matched immobile controls,
  session/animal-cluster population summaries, and an explicit equivalence margin when claiming
  similar SWR/non-SWR responses.

---

## Recommended primary analysis set

1. **Content primary:** one prespecified arm conditional-composition model, with patch scale as a
   resolution sensitivity and pooled CS interactions for formal SWR differences.
2. **Rate primary:** the raised-cosine Family-B point-process model on a frozen common cohort, with
   safe past-only reward history. Keep per-stop Family A as a literature-comparable secondary model.
3. **Choice primary:** one decision-grain, value- and distance-controlled conditional logit with
   blocked held-out evaluation. Label it association unless temporal/causal identification changes.
4. **Stage 6:** descriptive association unless a stronger longitudinal/mediation design is added.
5. **SWR power:** rebuild from a non-collinear parameterization before reporting any coefficient.
6. **PFC:** retain per-neuron GLMs as descriptive first-stage models, followed by animal/session-aware
   population inference.

---

## Priority remediation order

### P0: blocks current interpretation

1. Rebuild `recent_switch_rate` at completed-trial grain and regenerate every dependent table/model.
2. Rebuild the choice and Stage-6 datasets at distinct-decision grain; resolve same-arm choices.
3. Freeze exact common cohorts and folds for all model comparisons.
4. Replace the rank-deficient SWR-power specification.
5. Add run manifests/schema checks and refit all canonical models into a clean output directory.

### P1: required for defensible inference

1. Add rank/condition/estimability gates to every high-dimensional rate and OLS fitter.
2. Replace model-based SEs with block-/cluster-aware uncertainty where dependence is material.
3. Move population inference to animal-balanced or hierarchical summaries.
4. Gate post-outcome covariates by actual outcome time or relabel them strictly as retrospective.
5. Use formal interaction and joint multi-df tests for difference claims.
6. Add soft-posterior content fits and stop-blocked CV.

### P2: reporting and maintenance

1. Correct stale model tables and covariate definitions.
2. Rename misleading variables such as `cumulative_reward`.
3. Add exact cohort/fold hashes and scientific-estimand labels to every artifact.
4. Update paper claims only from the clean refit and include animal-level effect displays.

---

## Verification performed

Production invariant/audit scripts passed their implemented checks, including event/arm/patch
cardinality, fraction normalization, Q-value alignment, reward-lag identities, and current-arm
agreement. Targeted tests also passed:

- PFC GLM tests: 35 passed.
- Content simulations: 22 non-slow plus 2 slow tests passed.
- Core fitting/rate tests: 79 passed.
- Choice/covariate tests: 62 passed.

These results validate implemented mechanics. They do **not** establish statistical adequacy: the
same content simulation suite explicitly demonstrates under-coverage from trajectory
autocorrelation, and ordinary unit tests do not detect wrong estimands, pseudo-replication,
informative complete-case selection, or invalid population generalization.

No repository files other than this review were changed as part of documenting the audit.
