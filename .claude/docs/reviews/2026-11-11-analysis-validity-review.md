# Analysis validity and consistency review

Reviewed at `ed6789f8` on 2026-11-11. The summaries are traceable and their
existing tests pass, but that does not establish the validity of every upstream
analysis. This review found reproducible defects in the content fitter, the
stay/leave GEE wrapper and the rate figure, plus unresolved limits on population
inference. No production code, fitted output or reported value was changed.

The focus was the path from the five summaries through their statistics and
selected upstream fitters, comparisons and plotting code. The recording data
and fitted bundles are not present in this checkout. Synthetic counterexamples
establish the code defects below; their prevalence and effect on the current
recording-data results still require a cluster audit. This is not a validation
of every decoder, data loader or recorded event.

The earlier [reported-numbers review](2026-11-11-reported-numbers-review.md)
mainly checked the extraction and reporting of statistics. Preserving its
numbers through a refactor does not validate the original estimator. The
existing [choices register](../../../docs/known_issues/reported_statistics_choices.md)
and [TODO](../../../TODO.md) already acknowledge some of the inference and
figure-cohort issues; those are identified as existing below.

**Reproduced defects**

1. **High: positive, uniform allocations are excluded from the fitted content
   objective but included in its null and reported sample size.**

   `analysis/replay/fitting.py:281–298, 798, 502–519` drops every event whose
   response is constant over candidates. An event assigning a third of its
   mass to each patch is valid data: it supports a uniform allocation. In the
   conditional objective implemented at `:223–278`, its contribution depends
   on the coefficients and has nonzero curvature. It is not an uninformative
   all-zero event. Copying statsmodels' old exclusion rule does not make this
   appropriate for the project's allocation response.

   The `uniform_allocations` probe fits synthetic patch fractions. Adding
   uniform events leaves the fitted target-patch probability near 0.8, although
   the full-data optimum is 0.5667. The fitted log objective is unchanged,
   while the reported event count and null log objective double. The sandwich
   covariance also receives the full rows (`:887–889`), unlike the fitted
   objective. Thus coefficients, uncertainty, pseudo-R² and comparisons with
   the null can disagree about which observations were used. The `n_events`
   comparison guard cannot detect this because the fit reports the prefilter
   count.

   The existing test
   `tests/unit/test_fitting.py:892` explicitly preserves the constant-response
   exclusion. Replace that expectation with a check of the stated conditional
   objective; reproducing the old backend is not an independent validation.

   This reaches the arm/patch content fits and their comparison outputs;
   relevant summary families include `behavior.json: leaf_*`,
   `est_*` and the content-contrast coefficients. Impact on their current
   values is not measured. Count uniform positive responses on each actual
   fitted candidate set, especially patch and leaf-only fits. Keep these
   events in the objective, use the same observations for the null and
   covariance, then refit affected bundles and regenerate their summaries.

2. **Medium: canceled predictors still incur model-complexity penalties.**

   `analysis/replay/fitting.py:798–815` removes columns constant within every
   event because they cancel from the conditional model. Nevertheless,
   `:903–910` sets `n_params = len(predictors)`, counting them in AIC and BIC.
   `model_comparison.py:143–145` also uses that count for its test degrees of
   freedom.

   In the `canceling_predictor` probe, adding an event-level column leaves the
   fitted model and log objective unchanged but increases AIC by 2. Its record
   explicitly says `z:no_within_event_variation`. Model ranking should not
   change when an algebraically irrelevant column is added.

   Use the identifiable conditional-model dimension for canceled columns and
   retain the requested-column count as separate metadata if needed. Handle
   coefficients at separation boundaries separately: simply subtracting every
   unidentified term does not restore ordinary likelihood-test assumptions.
   Audit the saved `unidentified_terms` before trusting the affected
   `behavior.json: leaf_step_*` and estimator-comparison criteria.

3. **Medium: a failed GEE fit is reported as successful and enters FDR.**

   `analysis/replay/learning.py:630–657` suppresses fitting warnings and sets
   `status = "ok"` whenever `GEE.fit()` returns, without checking
   `model.converged`. This differs from the per-session logistic wrapper
   immediately above it, which checks convergence. `pooled_gee_role_summary`
   then uses `status == "ok"` as the definition of a converged fit.

   The `gee_convergence` probe supplies a backend result explicitly marked
   `converged=False`; the wrapper returns `status="ok"` and retains its
   interaction p-value. This reaches `behavior.json: stay_min_q_gee` through
   `reported/behavior.py:2378–2399`. The probe verifies failure handling, not
   that a current lab fit failed. Check convergence and finite inferential
   quantities before marking a fit usable, persist those diagnostics, and
   inspect or rerun the saved GEE bundles.

4. **Medium: the rate figure can reverse the sign of its labeled median AIC
   difference.**

   `scripts/rate/plot_per_bin_rate_summary.py:345–353` plots
   `median(AIC_child) - median(AIC_null)` as “median ΔAIC”. The summary's
   `aic_difference_statistics` instead matches sessions and takes
   `median(AIC_child - AIC_null)`. These are different even with identical
   complete cohorts; missing fits can make the discrepancy worse.

   The `paired_aic_plot` probe executes the actual panel function on synthetic
   paired AICs: null `[0, 100, 101]`, child `[90, 91, 201]`. It plots −9
   (apparently better), while the median paired difference is +90 (worse).
   Use the shared comparison function for the plotted bars, with each rung
   paired against its appropriate null and comparable observations.

5. **Medium: the reward-kernel figure and summary use different sessions.**

   `scripts/rate/plot_per_bin_rate_summary.py:299–303` passes every session
   with the reward block to `reward_kernel_rate_ratios`.
   `analysis/replay/rate_fit_statistics.py:166–198, 248–251` first excludes a
   session if any reward-basis term fails the per-term gate. This is the
   existing choices-register entry **B21**, independently reproduced here.

   In the `kernel_cohort` probe a flagged session contributes to the plotted
   curve and is absent from the summary, producing different lag-one values.
   Both implementations reconstruct each session's curve before taking its
   median; agreeing on that arithmetic is insufficient without agreeing on the
   cohort. Use the same gated curves for both and report the kept/dropped
   counts. Check the related rate plotters when regenerating them.

**Scientific limits that need decisions or additional validation**

- **Behavioral p-values do not establish inference across animals.**
  `coefficient_statistics.py:168` and `learning.cross_session_role_summary`
  apply Wilcoxon to session coefficients. The pooled GEE groups by
  `(animal, session_date, epoch)` at `learning.py:599–601`, so epochs of one
  recording day and animal are separate clusters. This permits dependence
  within an epoch, not between epochs. In contrast, the primary state-evidence
  analysis explicitly takes the animal as its replicate
  (`hippocampus/state_evidence/summary.py`).

  The `repeated_sessions` probe illustrates the consequence: copying the same
  five animal values into additional epoch rows changes the session p from
  0.0625 to about 6.3e-10 without adding an independent animal. This is a
  synthetic illustration, not a recomputation of a reported p.
  [SciPy's Wilcoxon assumptions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
  require independent observations, and
  [statsmodels' GEE documentation](https://www.statsmodels.org/stable/gee.html)
  assumes independent clusters. The code's “descriptive consistency” caveat
  is appropriate; treating those p/q values as calibrated population evidence
  is not. This limitation is already in TODO, but the GEE docstring still
  promises valid population inference.

  Define the target population and replication level for each claim. Keep
  animal-balanced effects and animal signs visible; use a method respecting
  the animal/day hierarchy, with an explicit small-sample justification, for
  uncertainty. Clustering only by recording day still does not establish
  generalization across animals. Do not interpret a nonsignificant stay/leave
  test as evidence of equivalence or no meaningful effect.

- **Fractional-response model scores need appropriate interpretation.**
  The design document correctly calls the content scores quasi-likelihood
  quantities (`docs/design/content_model_design.md:87–96`). Nevertheless,
  `model_comparison.py:142–145` automatically supplies an ordinary chi-square
  likelihood-ratio p-value, and the leaf summary reports “AIC” and sessions
  with a difference greater than 2 without carrying the quasi-likelihood
  qualification (`reported/behavior.py:2227–2245`). Normalizing event mass
  and using a sandwich SE do not by themselves calibrate likelihood-ratio
  tests or the ordinary AIC penalty. The
  [statsmodels quasi-likelihood warning](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html)
  states the general limitation; its GLM example is not this custom fitter.

  Suppress unsupported LRT output for the fraction estimator. Label existing
  score differences descriptively and validate model improvements with held-out
  prediction on common event/stop blocks, or justify a criterion appropriate
  to this estimating objective. The absent content-ladder cross-validation
  is already recorded in TODO. No claim is made here that a particular
  current model preference reverses.

- **Convergence rules differ between rate inference and validation.**
  `reward_gated_fits` deliberately retains nonconverged fits, whereas
  `per_bin_rate_model.cross_validate_per_bin_rate_model` invalidates a score when a
  fold does not converge. This occurs in the committed results: see
  `paper/reported/behavior.json` keys
  `rate_kernel_nonconverged_non_local_no_swr`,
  `rate_kernel_nonconverged_non_local_swr` and
  `rate_kernel_nonconverged_swr` (all nonzero and not pending).
  A flag alone does not prove a coefficient wrong, but passing the separation
  gate does not prove optimization succeeded either. Refit or assess the
  final objective/score, then show whether the kernel conclusions depend on
  these sessions. Apply the same documented usability rule to summaries and
  figures, accounting for exclusions rather than silently changing cohorts.

- **The stay/leave FDR family is within stratum, not the combined search.**
  `learning.py:727–763` corrects roles separately in each stratum;
  `reported/behavior.py:2372–2399` then selects the smallest adjusted p over
  strata. That minimum is a valid descriptive minimum of within-stratum
  adjusted values, not a correction over every searched role–stratum pair.
  State that scope in the value descriptions, or define and correct the full
  family before selecting its minimum. The current summary reports no small
  minimum; this review is not alleging a current false discovery.

**Verification and next steps**

The code-only release check passed for all five summaries, including tags,
citations and generated macros. It did not check the lab inputs. The targeted
summary/statistics suite passed **499 tests**; an additional upstream suite
passed **143 tests**, with one existing statsmodels warning in a constant-column
test. Thus **642 existing tests passed**, while the independent probes above
still exposed the defects. These are complementary checks, not contradictory
results.

The [probe script](2026-11-11-analysis-validity-probes.py) uses synthetic data
only. Its output is saved in
[analysis-validity-probes.json](2026-11-11-analysis-validity-probes.json).
To reproduce from the repository root:

```bash
.venv/bin/python scripts/reported/check_release.py --code-only
.venv/bin/python .claude/docs/reviews/2026-11-11-analysis-validity-probes.py
```

The installed project environment was used because uv's cache lock was blocked
by the sandbox and the escalated launch was interrupted. Dependencies were
not updated.

First correct the fitter's observation and parameter-count rules and the GEE
status handling, with tests against independent mathematical expectations.
Then align the two rate panels with the summary's paired estimator and cohort.
On the cluster, measure the affected observations/fits, rebuild the affected
bundles from a committed fix, and compare summaries by key before regenerating
figures. Independently settle the replication level and permissible model
comparison evidence before drafting inferential Results. A successful
release check should remain a provenance check, not a claim of statistical
validation.

**Follow-up: do the regressions answer the stated questions?**

The model families are largely reasonable for their response variables. That
does not mean every reported coefficient or test answers its question. The
following assessment is of the design and implemented readout, not a new fit
to the recordings.

| Question | Current model | Assessment |
|---|---|---|
| 3.3: reward history and event occurrence | Per-stop Poisson with exposure; per-bin Poisson conditional intensity with time and history terms | Appropriate families for two distinct quantities. Per-stop rate is events per immobility second; per-bin rate is onset intensity while the detector is at risk. Preserve this distinction. Compare class-specific reconstructed kernels directly, on compatible cohorts, with animal-level uncertainty. |
| 3.1–3.2: represented arms and recent reward | Conditional softmax allocation model, fit by a conditional-Poisson objective on per-event fractions | Appropriate conditional-mean approach to compositional shares, including zeros. It describes allocation given an observed event, not event occurrence or ordered trajectories. Repair the fitter and use inference appropriate to a fractional response. |
| 3.6–3.7: class differences, value and uncertainty | Content interactions, leaf-only belief ladder, duration-adjusted variants | Reasonable for adjusted associations. Uncertainty is a fitted behavioral-model quantity related to value and visit history; its unique contribution requires those alternatives in the comparison. Leaf-only results are conditional on represented leaf content. Adjustment for duration does not identify a causal direct effect. |
| 3.4: content associated with upcoming choice beyond behavior | Stop-stratified conditional logit over eligible destination wells | Appropriate for the narrower association question. The cohort contains at-well stops with replay and a known next choice; the split model additionally needs both event classes. The behavioral baseline controls distance rank, not a flexible distance/patch-choice process. A predictive-gain claim needs blocked held-out comparison against the behavioral baseline. |
| 3.5: content predicts staying/leaving | Logistic/GEE with outcome, content and their interaction | Appropriate binary-response family, but the reported interaction test does not answer the broad question. See below. |
| 3.8: ripple power and behavior | Per-session OLS on continuous event-mean ripple z-score, restricted to post-outcome events | Reasonable as a linear association model, subject to residual/nonlinearity/influence checks. It is conditional on detected events and on the post-outcome cohort; it is not a model of SWR occurrence or a test of distinct biological mechanisms. |
| 4.1: mPFC response around events | Penalized Poisson temporal kernels, blocked held-out deviance and matched non-event controls; marked extensions | Well aligned with event-associated firing and incremental prediction by event properties. Validity depends on the control exchangeability and conditioning assumptions. It does not alone establish a causal hippocampus-to-mPFC effect, a difference between mechanisms, or equivalence of classes. |

The allocation approach should not be rejected merely because its response is
fractional: fractional multinomial quasi-likelihood is a recognized way to
model the conditional mean of shares with boundary values
([Mullahy](https://www.nber.org/papers/w16354)). Event-history and behavioral
covariates also have a natural place in a point-process model
([Truccolo et al.](https://pubmed.ncbi.nlm.nih.gov/15356183/)). These references
support the model families, not the particular implemented fits or their
calibration. Blocked prediction is a separate target from a regression
association ([Shmueli](https://arxiv.org/abs/1101.0891)).

The clearest mismatch is **question 3.5**. The implemented model is

```text
logit P(stay) = intercept + b_outcome * omission
                         + b_content * content
                         + b_interaction * omission * content
```

`learning.cross_session_role_summary` tests only `beta_inter`, and
`pooled_gee_role_summary` reads only `p_inter`; the builder's
`stay_min_q_signrank` and `stay_min_q_gee` report their minima. Therefore these
values ask whether the content–stay association differs with omission on the
log-odds scale. A content association of the same strength after reward and
omission can have `b_content != 0` and `b_interaction = 0`: this analysis would
report no interaction even though content is associated with staying.

For the broad association question, compare a prespecified behavioral baseline
with that baseline plus the content main effect and interaction, testing the
content block jointly and showing the fitted content slopes in both outcome
conditions. If the claim is prediction, evaluate the increment in held-out
log loss/calibration on blocks chosen for the intended generalization target.
The current formula adjusts for outcome alone; it is not the richer behavioral
baseline of the destination-choice model. Alternatively, narrow the question
to “Does reward omission modify the association between replay content and
staying?” Even then, fix GEE convergence handling and the replication level.

Across the families, retain “associated with,” “duration-adjusted association,”
and “held-out predictive improvement” where those are what is estimated.
“Confound-free” in the rate guide and “direct/total effect” in the content and
marked-model descriptions are stronger causal interpretations than these
observational fits establish. A significant coefficient in one event class and
a nonsignificant one in another is also not a class-difference test: use the
existing interaction or paired-kernel contrast, with valid uncertainty.
