# Overview - scope, questions, gates, risks

> **SUPERSEDED — DO NOT EXECUTE (2026-07-29).** This file is retained only as
> historical design context. Canonical corrective work is owned by
> [`regression-integrity-fixes`](../regression-integrity-fixes/PLAN.md); the only
> surviving downstream rate/content task is
> [`rate-content-followups` Phase 2](../rate-content-followups/phase-2-swr-continuum.md).

[<- back to PLAN.md](PLAN.md)

## Motivation

The rate and content models answer different scientific questions:

- **Rate models:** how often non-local replay onsets occur per unit immobility
  time, after accounting for stop time, self-history, reward history, and task
  covariates. These are Poisson intensity models, so key coefficients are rate
  ratios.
- **Content models:** conditional on an event occurring, how its decoded content
  is allocated over arms or patches. These are conditional Poisson allocation
  models, so key coefficients are allocation ratios within the event.
- **Choice models:** whether stop-level replay content predicts the animal's next
  destination beyond behavioral predictors. These are conditional logits, so key
  coefficients are choice odds ratios.

The coefficient inspection suggests a biologically useful split:

- Immediate reward history has a large SWR-overlap rate effect: in the per-bin
  rate ladder, `reward_lag1` was large for `non_local_swr` and small for
  `non_local_no_swr`, with the paired SWR/non-SWR contrast strongly positive.
- Longer reward history looked suppressive across populations, suggesting the
  immediate post-reward SWR gate sits on top of a broader post-reward rate
  reduction.
- Content models show strong current-arm/current-patch allocation. SWR-overlap
  content appears less local than non-SWR content in the existing coefficient
  readout, but the arm reward-rate SWR-contrast artifacts must be regenerated
  before using those numbers.
- HMM/task content terms suggest state-dependent allocation: RPE and switching
  variables modulate current-location bias, while per-arm value contrasts need a
  careful artifact-freshness check.

The implementation task is therefore not "fit one more model." It is to turn
that interpretation into an audited pipeline that can say which conclusions are
supported by current artifacts, which are stale, and which follow from joint
rate/content evidence.

## Questions this plan can answer

- **Rate vs content:** Does reward change the probability of replay events, the
  represented content of those events, or both?
- **SWR specificity:** Are the reward and task effects carried by
  SWR-overlapping non-local replay, non-SWR non-local replay, or both?
- **Binary vs continuous SWR coupling:** Is `is_swr` a useful binary process
  label, or do the effects scale continuously with ripple overlap/power?
- **Task-state content:** Does RPE, relative value, uncertainty, update size, or
  switching state shift content toward/away from the current arm or patch?
- **Behavioral relevance:** Does replay content improve prediction of next
  choice beyond recency, arm reward rate, cumulative reward, and distance rank?
- **Claim safety:** Which paper-level claims survive current covariate names,
  event-wise complete-case filtering, provenance checks, and held-out or
  cross-session validation?

## Current codebase integration points

- `src/continuum_swr_replay/analysis/replay/model_specs.py:1` defines the model
  families and the content identifiability rule: event-level constants cancel
  inside conditional content models.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:201` defines the
  arm content hierarchy with `reward_rate_arm_k10`, `q_arm`, and
  `is_last_rewarded_elsewhere`.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:715` defines the
  choice models and the behavioral baseline.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:791` defines the
  SWR-contrast rate ladders (`G0`-`G8`, `S0`-`S10`).
- `src/continuum_swr_replay/analysis/replay/model_specs.py:867` defines the
  SWR-contrast content ladders and key terms.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:55` extracts
  the columns used by a per-bin rate spec; `:79` fits one session and emits
  `coef_*`, `bse_*`, `pval_*`, and `rr_*`.
- `scripts/rate/fit_swr_contrast_rate.py:1` fits the stratified and stacked
  `is_swr` per-bin rate contrast models.
- `src/continuum_swr_replay/analysis/replay/fitting.py:61` fits conditional
  content models and enforces event-wise missing-data filtering.
- `scripts/content/fit_replay_content_models.py:493` applies common
  complete-case event filtering across the primary content hierarchy.
- `scripts/content/fit_swr_contrast_content.py:4` is the content analog of the
  SWR-contrast rate runner; `:159` fits one session after complete-event
  filtering.
- `src/continuum_swr_replay/analysis/replay/choice_prediction.py:91` computes
  model-predicted/residualized arm fractions; `:241` builds the stop-by-arm
  choice set; `:418` fits the conditional-logit choice model.

## Escalation logic

Each phase ships as a separable PR and ends with a decision gate.

1. **Phase 1:** audit and regenerate content artifacts. Gate: no downstream
   interpretation can use an artifact that fails current coefficient-name,
   complete-case, or provenance checks.
2. **Phase 2:** build shared summary tables. Gate: all downstream analyses must
   consume the summary contract rather than hand-reading pickle columns.
3. **Phase 3:** run rate-content dissociation tests. Gate: decide whether binary
   `is_swr` is sufficient or whether continuous ripple-overlap/power terms are
   required for the reportable analysis.
4. **Phase 4:** test choice prediction/mediation. Gate: if replay content does
   not add held-out predictive value beyond behavior, do not frame the
   rate/content effects as behavioral mediation.
5. **Phase 5:** generate figures/docs only from audited artifacts. Gate: no
   stale artifact flag, missing key coefficient, or ambiguous estimand enters a
   paper-facing figure.

## Non-goals

- No re-decoding and no new event detector.
- No replacement of the existing rate/content ladders; this plan builds
  audited summaries and follow-up tests around them.
- No full hierarchical Bayesian refit. Session-level paired tests and
  per-animal summaries are the default.
- No direct magnitude comparison between rate ratios, content allocation
  ratios, and choice odds ratios.
- No causal claim that replay causes choice. Phase 4 asks whether replay content
  predicts choice beyond measured behavioral covariates.

## Metrics

- Artifact audit passes current model specs: expected `coef_*` terms exist,
  stale aliases are absent, and retained content events preserve 9 arm rows or
  3 patch rows.
- Summary output contains one row per requested `(family, artifact, model,
  population, session, term)` with effect scale, provenance, n, and stale flag.
- Dissociation report includes signed session effects, per-animal sign
  consistency, paired SWR/non-SWR contrasts, and bootstrap or signed-rank
  uncertainty.
- Choice report includes held-out improvement over `C_behavior`, replay-only
  reference performance, SWR/non-SWR split coefficients with unfit counts, and
  common retained-stop cohorts for each nested comparison.
- Figure/doc generation fails if any source artifact is stale or lacks
  provenance.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Stale content artifacts encode old covariate names (`recent_reward_rate`) or old row-wise missing filtering | Phase 1 audit is a hard gate; regenerate before interpreting. |
| Conditional content models silently change the candidate set by dropping rows | Preserve complete events: all retained arm events have 9 rows and all retained patch events have 3 rows. |
| Rate/content effects are put on incompatible scales | Shared contract labels every effect as `rate_ratio`, `allocation_ratio`, or `choice_odds_ratio`. |
| Sparse terms such as last-rewarded-elsewhere interactions overfit or vanish in many sessions | Report unfit counts, n events, n animals, and keep these terms secondary. |
| Multiple testing across many task covariates | Define key terms up front and separate primary dissociation tests from exploratory tables. |
| Choice prediction launders distance/current-location bias | Keep the behavioral baseline visible, report residualized replay-content fits, and preserve the existing distance-rank baseline. |
| Nested choice comparisons use different retained stops after SWR/non-SWR split predictors introduce NaNs | Refit baseline and full models on an identical complete-case stop cohort for every comparison, and report cohort shrinkage. |

## Open questions

1. **Continuous SWR mark:** use ripple-overlap fraction, ripple-power z-score,
   or both. Default: test both if available, then choose the simpler term with
   held-out support.
2. **Behavioral outcome:** next arm or next patch. Default: next arm first,
   patch as a robustness/sensitivity analysis.
3. **Claim level:** whether the final paper claim is "reward gates SWR-overlap
   replay rate" only, or also "reward/task state changes represented content."
   Phase 1-3 decide this.

## Estimated effort

- Phase 1: ~150 LOC audit/refit helpers + tests; runtime dominated by full
  content refits.
- Phase 2: ~150 LOC shared summary builder + tests.
- Phase 3: ~200 LOC analysis runner, continuous-SWR sensitivity, report tables.
- Phase 4: ~150-250 LOC choice feature/runner extensions + tests.
- Phase 5: ~150 LOC figures/docs/claim ledger.
