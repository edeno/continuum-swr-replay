# Phase 4 - Choice prediction and mediation checks

[<- back to PLAN.md](PLAN.md) . [overview](overview.md) . [shared contracts](shared-contracts.md)

After rate/content dissociation is audited, test whether replay content predicts
future behavior. This phase asks a predictive question, not a causal one:
does replay content during a stop improve next-arm prediction beyond measured
behavioral covariates?

**Questions:** Does replay content predict the next choice above the behavioral
baseline? Is any predictive value SWR-specific? Does residualized content
predict choice after removing current-location/value/task content structure?

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/model_specs.py:715` - current
  choice model specs and behavioral baseline.
- `src/continuum_swr_replay/analysis/replay/choice_prediction.py:91` -
  residualized replay-content helper.
- `src/continuum_swr_replay/analysis/replay/choice_prediction.py:241` -
  stop-by-arm choice set construction.
- `src/continuum_swr_replay/analysis/replay/choice_prediction.py:418` -
  conditional-logit fitter and stop-wise missing-data policy.
- Phase 1 audited content artifacts and Phase 3 selected SWR representation.

## Tasks

- Extend the choice runner or add `scripts/analysis/fit_choice_rate_content.py`.
  - Build `replay_frac`, `replay_frac_swr`, and `replay_frac_non_swr` from
    canonical event-arm tables.
  - Build `replay_frac_residual` using an audited content fit and
    `predict_arm_fractions`.
  - If Phase 3 promotes a continuous SWR mark, add mark-weighted replay
    fractions as secondary predictors.
- Fit and summarize:
  - `C_replay_only`;
  - `C_behavior`;
  - `C_full`;
  - `C_full_swr_split`;
  - optional residualized-content model, e.g. behavioral baseline plus
    `replay_frac_residual`.
- For every nested comparison, build a common complete-case retained-stop cohort
  first, then refit both baseline and richer models on that same cohort.
  - `C_full` vs `C_behavior` gets its own common cohort.
  - `C_full_swr_split` vs `C_behavior` gets its own common cohort because split
    SWR/non-SWR predictors introduce additional NaNs.
  - Residualized-content models get their own common cohort because residuals
    propagate content-model missingness.
  - Report `n_stops_before`, `n_stops_common`, and the dropped-stop fraction for
    each comparison.
- Add held-out or cross-session predictive scoring if not already available for
  these choice fits.
- Report:
  - incremental log-likelihood over `C_behavior`;
  - coefficient signs and odds ratios for replay terms;
  - unfit counts and retained stop counts;
  - SWR/non-SWR split comparison;
  - whether residualized content still predicts choice.

## Gate

Proceed to Phase 5 with behavior claims only if:

- replay terms improve prediction over `C_behavior` on held-out or
  cross-validated scoring;
- baseline and replay models in each comparison were refit and scored on the
  same retained-stop cohort;
- the candidate set and stop-wise complete-case checks pass;
- residualized replay-content results are consistent enough to argue the effect
  is not only current-location/value leakage.

If replay does not improve prediction over behavior, final figures can still
show the rate/content dissociation, but should not claim behavioral mediation.

## Deliberately not in this phase

- No causal mediation model.
- No post-choice variables or next-arm-derived predictors.
- No distance-free headline result without also reporting the behavioral
  baseline and residualized sensitivity.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_choice_features_no_next_arm_leakage` | Predictors exclude `next_arm`, `is_next`, and post-choice reward. |
| `test_choice_complete_stop_drop` | A NaN in any candidate row drops the whole stop. |
| `test_residualized_choice_propagates_nan` | Residualized replay features do not convert missing residuals to zero. |
| `test_choice_nested_models_common_cohort` | Baseline and richer models in a comparison receive identical retained stop IDs. |
| real smoke | Runner fits at least `C_behavior` and `C_full` on one session/animal slice. |

## Fixtures

- Tiny stop/event-arm tables with a known candidate set and one chosen arm.
- Fixture content fit with simple coefficients for residualized replay tests.

## Review

Confirm that the output is framed as prediction. If the behavioral baseline is
strong and replay adds little, the implementation should say that directly
rather than rescue the claim with an underspecified model.
