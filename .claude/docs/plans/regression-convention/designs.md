# Designs

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Per-component designs. Names inside `analysis/regression/` are the builder's to choose (design section 10);
the ones for code not yet built are proposals, marked new. The fitter is built: its section below is an
interface summary, and the code is the truth for it.

- [The shared fitter (built)](#the-shared-fitter-built)
- [Smooth terms and bases](#smooth-terms-and-bases)
- [Support rule](#support-rule)
- [Collinearity measures](#collinearity-measures)
- [Trial folds](#trial-folds)
- [Simultaneous bands](#simultaneous-bands)
- [Session to day to animal](#session-to-day-to-animal)
- [Rate design on the visit](#rate-design)
- [Content design](#content-design)
- [JAX batch twin](#jax-batch-twin)

## The shared fitter (built)

Built in Phase 1 (merged to master in 6e54eda8, cleaned up in 98fe5c06). Every name, signature and rule is
in the code, `src/continuum_swr_replay/analysis/regression/` (each module's docstring states its rules),
and in `docs/design/regression_fitter.md` (the rules with their measurements, the agreement with mgcv,
neurospatial and the current fitters, and the cost). This section says only what a caller passes in and
gets back. It does not repeat the internals; read them there.

**Likelihoods** (`likelihoods.py`): `Poisson` (aux: the offset `log(exposure)` per row), `Bernoulli` (aux
None; 0/1 outcomes only), `Gaussian` (aux: the scale in `penalized_newton`; None in the REML functions,
which profile it), `GroupedSoftmax` (aux: the group id of each row, sorted so that a group's rows are
contiguous; the content response is the event's shares). Under the softmax a column constant within every
group (an intercept, an event-level covariate) is not identified and must not be in the design; the fitter
checks identification on `Likelihood.identifying_design`.

**Inputs.**

- `reml.SessionData(X, y, aux=None, name=None, active=None)`: one session's design `(n_rows, p)`, response
  and aux; `name` labels messages; `active` is a boolean mask over the columns that marks the columns the
  session estimates (unpenalized columns only; the others get NaN coefficients and covariance). It holds
  no start: the fitter chooses its starts.
- `reml.PenaltyBlock(columns, S)`: one quadratic penalty on its columns. The blocks of a model must be
  disjoint; a matrix that is not symmetric positive semidefinite is refused when the block is built.
- `pspline.PSplineBasis(lo, hi, n_basis, degree=3)` (or `PSplineBasis.from_spacing(lo, hi, spacing)`):
  `design(x)` (clamped to the range), `penalty(order=2)`, and `constrained(x, order=2)`, which returns a
  `ConstrainedSmooth(design, penalty, constraint)`: the constrained design `B Z`, its penalty `Z' P Z` and
  the constraint `Z`, computed on the rows given (the pooled rows). Another session's design is
  `basis.design(x_s) @ constraint`.

**Entry points** (`reml.py` unless named):

- `select_penalties(lik, sessions, blocks) -> PooledFit`: one log penalty per block, shared by the
  sessions, by pooled REML from a grid start. Before the search it fits every session at the grid center
  and sets aside the sessions with no finite maximum; after it, it refits them at the chosen penalties
  with their running-away unpenalized columns inactive (at most `REFIT_MAX_ROUNDS` rounds).
- `refine_penalties(lik, sessions, blocks, rho0) -> PooledFit`: the same search from `rho0`, without the
  grid.
- `fit_at_penalties(lik, sessions, blocks, rho) -> PooledFit`: the fits at given log penalties (no block
  and an empty `rho` for no penalty), without a search. It raises at the first session whose fit does not
  converge: it sets no session aside.
- `pooled_reml(lik, sessions, blocks, rho, *, derivatives=True) -> PooledScore` and
  `reml_and_derivatives(lik, X, y, aux, blocks, rho, beta0=None, *, derivatives=True) -> RemlScore`: the
  pooled score and one session's score, with the exact gradient and Hessian in the log penalties.
  `PooledScore` names the first failing session (`failed_session`, `failed`, `failed_status`) instead of
  raising.
- `newton.penalized_newton(lik, X, y, aux, S, beta0, *, free_basis=None, max_iter=NEWTON_MAX_STEPS,
  max_step=MAX_STEP_LENGTH) -> NewtonFit`: one fit at a summed penalty `S` (`reml.assemble_penalty`). It
  takes no tolerance and no halving count: those are rules in `constants.py`.

**What comes back.** `PooledFit(rho, score, gradient, sessions, search)`. `sessions` holds one
`RemlScore` per session given, in order (there is no separate per-session fit class): `fit` (a
`NewtonFit`: `beta`, `A`, `status`, `newton_decrement`, `n_iter`, `loglik`, and for a fit with no finite
maximum its `step` and `running_away` columns), `covariance` (`A^-1`, mgcv's `Vp`), `edf` per block,
`edf_total`, `scale` (the Gaussian's), `score`, and `running_away_directions` and
`unidentified_directions` (`FreeDirection`s: a design column, a block's null space or a block's penalized
direction). `in_null_space(session, blocks)` flags, per block, a term shrunk to its penalty's null space (a
straight line). `search` is a `PenaltySearch` (None from `fit_at_penalties`): `reference_rho`, `in_choice`
(the sessions whose scores chose the penalties; `PooledFit.score` is over these only), `set_aside`
(`SetAsideSession`s with `reference`, `refits`, `left_out`, `reason` and `fitted`), `grid_failures` per
status, `n_outer_steps`, `at_boundary` per term and `share_without_fit`. A set-aside session has its refit
in `sessions`, NaN at the columns left out, or NaN values when it has no fit.

**Statuses** (`newton.FitStatus`; `NewtonFit.converged` is true only for the first): "converged", "design
not identified", "information not positive definite", "step halving failed", "step limit reached", "no
finite maximum". `penalized_newton` returns every status without raising.

**Errors** (`ValueError`; in a pooled fit each names the session by index and `SessionData.name`):
inputs that are not finite, of the wrong shape or outside what the family takes; overlapping blocks; an
inactive column inside a penalty block; in a search, a session whose design is not identified (the message
names the coordinates that take part and says to mark the columns inactive) or whose fit fails for a
numerical reason at the reference point; no finite grid point; a search that does not converge (no
argument returns an unconverged result); more than `MAX_FAILING_SHARE` (20 %) of the sessions without a
fit; and `fit_at_penalties` at the first session whose fit does not converge.

**Constants** (`constants.py`): only the fitter's own rules (convergence, step control, identifiability,
penalty bounds and search, `REFIT_MAX_ROUNDS`, `MAX_FAILING_SHARE`). The convention's constants that the
fitter does not read are defined in the Phase 2 module that reads each
([C5](shared-contracts.md#c5-the-convention)).

**Known limits** (`TODO.md`, section "Regression fitter"): the step cap is in coefficient units, so a fit
with a column whose values are all very small ends at the step limit (Phase 2's design check refuses such
a column); only `select_penalties` handles a session without a finite maximum (Phase 2's per-session loop
must handle it at given penalties); each evaluation prepares every session again (to fix before the real
models' searches).

## Smooth terms and bases

> WAITS FOR THE LEAD (smooth-term rule): the basis and the penalty of a per-session smooth term. Option A:
> a natural cubic spline of 4 columns, knots from the pooled data, no penalty (the author's
> recommendation, checked at 3, 4 and 5 columns); option B: the P-spline below with one fixed weak penalty
> per term. Under either, no penalty is chosen from the data for a per-session smooth term with few
> events (simulation: a penalty chosen by REML makes each session's curve a line at about 140 events, and
> the belief-variance test is then wrong). Depends on it: which basis this section prescribes (option A
> needs a natural-spline basis the package does not have), the initial basis sizes, the basis check, and
> model D's penalty.

- Cubic B-spline on equally spaced knots over a fixed range `[lo, hi]` per term (the knots are the same
  for every session), with a second-order difference penalty `P = D2' D2`. Beyond `hi` the clock is clamped
  to `hi` (the curve is flat there), below `lo` to `lo`. As built: `pspline.PSplineBasis`.
- Identifiability: one sum-to-zero constraint per smooth, computed once on the pooled cohort rows of the
  model and kind, and the same `Z` used for every session, so that the coefficients mean the same thing in
  every session and the session curves can be summarized together. As built:
  `PSplineBasis.constrained(x)` returns the constrained design, its penalty `Z' P Z` and `Z`. The
  constrained penalty has rank `n_basis - 2` (take it from `PenaltyBlock.rank`, never from a count).
- Model D (reward history as a linear functional): `eta_i += sum_{l=1}^{10} f(log l) r_{i,l}`,
  `f(x) = sum_b B_b(x) gamma_b` over `x in [0, log 10]`, so column `b` of the design is
  `sum_l B_b(log l) r_{i,l}` with `r_{i,l}` the outcome (1 rewarded, 0 omitted) at lag `l`. No sum-to-zero
  constraint (the term is not confounded with the intercept unless every outcome is the same; the support
  rule checks that). Second-order difference penalty on `gamma`. mgcv's equivalent is
  `s(L, by = R)` with matrix arguments `L` (lags) and `R` (outcomes), the "linear functional" form of
  `?linear.functional.terms`. The log scale lets the curve change fast between lags 1 and 2 (design call).
- Initial basis sizes (calls, checked as below): time since arrival `[0, 20] s`, knots 0.5 s apart; time
  until leaving `[0, 3] s`, 0.1 s apart (the design asks for closer knots here, section 2.4: 0.33 s was too
  smooth for a 0.05 s change); time until next movement `[0, 1.5] s`, 0.1 s apart; log trials since last
  visit `[log 1, log 200]`, 10 basis functions; model D lags 8 basis functions over `[0, log 10]`.
- Basis check: after the fit, a term whose edf exceeds 0.8 times its basis dimension is refit with twice
  the basis functions; the rule and both sizes are recorded in `basis` of the fit record.

## Support rule

Decided from the data before the fit, per unpenalized declared term and session (design 4.3):

- A 0/1 term (rate: last outcome rewarded): count the units (visits) with the term 1 that hold at least one
  event of the kind, and those with the term 0 that do; the term passes when both counts are at least 5.
- A continuous slope restricted to an arm role (content): among the role's candidate rows of events of the
  kind, count events with the covariate above and below the session median of those rows (ties to neither);
  pass when both are at least 5 and the covariate is not constant within the role.
- A role category (content: previous arm, next arm): pass when at least 5 events of the kind have a
  candidate in the role.
- A term that fails in more than 20 % of sessions with a fit (the value of the fitter's
  `MAX_FAILING_SHARE`) leaves the per-session model. The design says it is then asked in a pooled model;
  that model is stage 3, outside this plan, so the phase stops and reports the term to the lead.
- A term that fails in a session is made inactive there (`SessionData.active`). When the term is a 0/1 role
  column (a role category, such as the previous arm), the rows with that role are dropped from that
  session's fit and counted, not merged into the reference role. Found by simulation (scratch
  `proto_uncertainty_measures/REPORT.md`): merging inflated the belief variance's false positives, to 20 %
  in one case. The same row rule holds for a role column made inactive because the session has no finite
  maximum ([Phase 2](phase-2-convention-tools.md)).
- As built (after the phase's independent review): a category level passes on its own count alone, and
  only a thin level's own rows are dropped. After any rows are dropped, the support is counted again on the
  rows kept, until nothing changes, and the record holds the counts the fit used. A term whose column is
  below the design check's threshold in one session fails the support rule there (on the pooled rows it is
  the caller's error). A session left with no rows or no events is recorded as not fitted, with the reason,
  and the loop goes on. A category's reference level must have rows in every session; otherwise the design
  is not identified and the fit raises, naming the session.
- The minimum count (5) is defined in `analysis/regression/support.py`, the module that reads it; the
  fitter's `constants.py` does not hold it.

Smooth terms carry a penalty and have no support rule; their basis check and edf are recorded.

> WAITS FOR THE LEAD (smooth-term rule): under option A (a natural spline with no penalty) a smooth term
> is an unpenalized set of columns, and whether it then needs a support rule (enough events across its
> range in each session) depends on the decision.

## Collinearity measures

Measured on the quantity the likelihood sees, with the information weights at the fit: for the diagonal
families `X~ = diag(sqrt(w)) X`; for the grouped softmax, per event,
`X~_g = diag(sqrt(p_g)) (X_g - 1 p_g' X_g)` (then `X~' X~ = X' (diag p - p p') X` summed over events).

For a term `i` with columns `Xi` and the rest of the design `X0` (intercept or arm effects included), QR of
`[X~0, X~i]` gives `R = [[R00, R0i], [0, Rii]]`. Then (mgcv's three concurvity measures; mgcv reports them by
`concurvity(fit, full = TRUE)`):

    observed = ||R0i b_i||^2 / (||R0i b_i||^2 + ||Rii b_i||^2)          (b_i the fitted coefficients)
    worst    = largest eigenvalue mu of  R0i' R0i v = mu (R0i' R0i + Rii' Rii) v
    estimate = ||R0i||_F^2 / (||R0i||_F^2 + ||Rii||_F^2)

For one column the three equal `R^2` of that column on the rest, and `VIF = 1 / (1 - R^2)`. The rule of
[C5](shared-contracts.md#c5-the-convention) reads `R^2` for one-column terms and `estimate` for smooths
(call), and stores all three. Phase 1's mgcv script (`scripts/checks/compare_regression_to_mgcv.R`) does not
print concurvity: Phase 2 adds `concurvity(full = TRUE)` to it and to the fixtures, and tests the three
numbers against it. The threshold (R² 0.8) is defined in `analysis/regression/collinearity.py`, the module
that reads it.

## Trial folds

One fold function for every family (design 4.4), in `analysis/regression/folds.py` (new). Same semantics and
signature as `trial_folds(time, segment_starts, segment_classes, n_folds=5, max_segment_s=30.0)` of branch
`swr-trial-folds` (`analysis/event_evidence/tetrode_holdout.py:297` there, with `trial_segments` `:129`,
`capped_groups` `:255`, `task_segment_starts` `:177`): scikit-learn `StratifiedGroupKFold(n_splits=5,
shuffle=False)` with one sample per row, class = the destination well of the row's trial segment, group =
the row's piece of that segment (a segment longer than 30 s cut into equal pieces). Deterministic. For a
regression the rows are bins (rate, mPFC) or events (content, choice), placed by their start time. The
number of folds (5) and the longest stay piece (30 s) are defined in `folds.py`, the module that reads them.

That branch is not on master (2026-11-21; Phase 4 was done without it). When it is, its `trial_folds` calls
the shared function (or the shared function is that one, moved), and a test asserts both give the same folds
on a synthetic epoch. Folds compare designs and check over-fitting; they do not choose penalties and are not
the test that an effect exists.

## Simultaneous bands

For a curve `f(x_1..x_G) = C beta` (C the basis rows for the term, a fixed grid):

1. Per session, the pointwise standard error `se_g = sqrt(diag(C V C'))` with `V = (H + S)^-1`.
2. A band valid for the whole curve: draw `M = 10000` vectors `z ~ N(0, V)` (seed fixed and stored), compute
   `max_g |C z|_g / se_g`, take its 0.95 quantile `c`, band `f +/- c se`.
3. For the animal-balanced curve across sessions: nested bootstrap draws of the curve (animals, then
   sessions; `shared_statistics.nested_bootstrap_draws`, `analysis/shared_statistics.py:377`), `sd_g` over
   the draws, `c` = 0.95 quantile of `max_g |draw_g - mean_g| / sd_g`, then the band is
   `estimate +/- c * sd_g * k` with `k = t(0.975, A - 1) sqrt(A / (A - 1)) / z(0.975)`, the same small-sample
   correction `bootstrap_interval` (`shared_statistics.py:456`) applies to a scalar (call: combining the two
   is this plan's choice; Phase 9 checks its coverage on simulated data).

## Session to day to animal

New functions in `analysis/shared_statistics.py` (task-agnostic, so they belong there; the module's rule,
`docs/README.md:100-103`, is kept):

- `animal_day_session_values(values, animal, day, session, statistic)`: session values -> recording-day
  values -> animal values (a mean or median at each level), the three-level analog of
  `animal_session_values` (`shared_statistics.py:191`).
- `day_level_test(values, animal, day)`: the signed-rank p over recording-day values
  (`signed_rank_p`, `shared_statistics.py:51`, which returns NaN below six non-zero differences).
- The estimate is the animal-balanced value; its interval is `bootstrap_interval` over
  `nested_bootstrap_draws` (animals, then sessions); "animals with the sign" counts animal values with the
  estimate's sign. No claim rests on a p-value over five animals.

The scratch script `verify_rate/s5_inference.py` (`infer`) shows the pattern on real values.

## Rate design

The visit at-risk bin design (Phase 3 builds it; Phase 5a fits it). Per session and kind:

- Visits from the port experiences (`arrival_time` = first poke-in, `departure_time` = last poke-out). The
  at-risk set is the union over the session's immobility stops of `[stop_start, stop_end]` intersected with
  each visit interval, less excluded time (`analysis/event_definition/analyzed_time.py`), less time inside an
  ongoing event of the outcome's kind (as `aggregate_to_timebins` does with `at_risk_events`,
  `analysis/replay_models/per_bin_design.py:394`).
- 15 ms bins (`RATE_BIN_S`, `analysis/behavior/design_constants.py:106`) laid from each in-visit immobile
  stretch's start; the last bin is partial (its `exposure` the remainder). A stop that crosses the final
  poke-out is cut there; a stop that starts before the first poke-in starts at it. Each bin carries
  `visit_context = "in_visit"`, `well_visit_id`, `stretch_index` within the visit, and the three clocks at
  the bin center: `s` = t - first poke-in, `u` = last poke-out - t, `m` = end of the current immobile stretch
  (clipped at the last poke-out) - t. In the visit's last stretch `m = u`. As built:
  `replay_models/visit_design.py:visit_timebins` (columns `time_since_visit_start_s`,
  `time_until_visit_end_s`, `time_until_stretch_end_s`).
- Counts: event starts of the kind whose event starts inside a visit (`select_visit_context` default,
  `STARTS_IN_WELL_VISIT` = `in_visit` and `spans_visit_end`, C3). An event that starts in a bin and crosses
  the final poke-out is counted (lead, L4); the bins end at the poke-out, so its time after it is not at
  risk. The counting rule that remains is the event definition's: an event must be immobile as a whole, so
  a candidate that starts in a stop and runs into movement is not an event. The rate simulation of Phase 5a
  plants that rule at every stop end (small movements and the last one), so its effect is measured, not
  assumed.
- One exception, as built: an event whose start has no at-risk time before it is not counted (it starts at
  the start of a piece of analyzed time, such as a stop's first sample, and fills that bin, or right after
  an event of the kind that fills the bin before it). `visit_timebins` records these in `attrs`
  (`n_starts_not_at_risk`, `n_starts_not_at_risk_by_reason`). The events that start on a stop's first
  sample are about 0.2 % of the No SWR events and none of the SWR events (28 events), so the rate and
  content models would differ by them. Whether that stands is a question for the lead at
  [Phase 5a](phase-5a-rate-models.md).
- Reward covariates per bin: the last known outcome and the last ten known outcomes at the bin's time, with
  the existing outcome-boundary shift (the outcome of the current trial becomes lag 1 at
  `outcome_known_time_s`; `per_bin_design.py:85-91`, `_OUTCOME_SHIFT_STOP_COLUMNS`).
- Model A: `count ~ offset(log exposure) + f_s(s) + f_u(u) [+ f_m(m)] + last_outcome_rewarded`.
  Model D: the same with the model D functional in place of the 0/1 term, on bins with ten known lags.
  `f_m` stays only if the collinearity rule separates it from `f_u` (design 4.1, "to try").

## Content design

Per session and kind, one row per (event, candidate arm), all nine arms (lead's decision 1):

- Response: the event's share on each arm (`arm_fraction` of the event-arm table), weight one per event.
- Events: those that start inside a visit (C3), `in_visit` and `spans_visit_end`. The current arm of a
  `spans_visit_end` event is the arm of the visit's well (Phase 3 sets it so in the event-arm table).
- Roles, exclusive, per (event, arm): `current` (the arm of the well being visited), `previous` (the arm of
  the previous trial's well, when different), `other_leaf` (any other leaf arm), `trunk` (arms 6 to 8).
  Each role is a category (reference: `other_leaf`). Arm fixed effects stay in every content model, per kind
  (`model_specs.py:2072`, `add_arm_fixed_effects`); a role contrast is then read against them.
- Leaf-arm covariates by role (current, previous, other_leaf): the reward fraction over the last ten visits
  enters as a **penalized smooth per role** (lead, 2026-11-20: a curved reward effect belongs to Q4, not to an
  uncertainty test), over `[0, 1]` with 6 basis functions (call; the fraction takes the values k/10, so a
  small basis); value (beta-belief mean) as one slope per role. Trunk rows carry none (set to 0, with the
  trunk role absorbing the level). Q4a, the "slope" of the current arm's share on its reward fraction, is
  then a linear functional of the smooth: `f(m + sd/2) - f(m - sd/2)`, `m` and `sd` the pooled mean and SD
  of the fraction on current-arm rows (call), so it stays a change per pooled SD with an exact variance.
  That reading of Q4a is a call to confirm with the lead at [Phase 6a](phase-6a-content-models.md).
- WAITS FOR THE LEAD (smooth-term rule): the reward-fraction smooth per role, the recency smooth and the
  current role's smooth of `u` are per-session smooth terms; their basis and penalty (above, "Smooth terms
  and bases") follow the decision.
- Recency: a smooth of `log(trials since last visit)` on leaf rows, not centered within the event (lead:
  absolute recency), with the never-visited indicator; recency is clamped at 200 trials. The shipped
  within-event centering (`behavior/covariates/history/visit_recency.py:23`) is not used.
- Q7b: a smooth of time until leaving (`u` at the event start) for the current role, per kind.
- Length-adjusted comparison of kinds: every role slope and role category also interacts with
  `log(event duration) - L0`, `L0` the pooled median log duration over both kinds (fixed, stored), so the
  per-kind estimate is at the same reference length; the unadjusted model omits the interactions. Both are
  stored and reported (see [C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct)).
- Uncertainty: **one model** (lead, 2026-11-20, after the simulation: "Yes"), the main design plus, by role
  on the leaf arms:
  1. the belief variance (`reward_belief_variance_candidate_arm_preoutcome`, built by
     `behavior/covariates/history/value_block.py:206`), pooled-SD slope per role;
  2. the sample count: `log(1 + n)`, `n` the visits to the well with a known outcome earlier in the epoch,
     before the event, pooled-SD slope per role;
  3. the trial number in the epoch (the visits the animal has arrived at by the event start, Phase 3's
     `n_visits_arrived`; this definition is a call to confirm with the lead at Phase 6a), a pooled-SD
     slope per role on the same leaf roles the measures enter on. It is an event-level quantity, so it is
     identified only as a role interaction; without it the sample count is a drift with the trial number;
  with the smooth reward fraction above. The reward variability (the variance of the arm's 0/1 outcomes
  over its last 10 visits) is dropped (lead, "Yes"): for 0/1 outcomes it is `p(1 - p)` of the reward
  fraction and could not be told from a curved reward-fraction effect in the simulation.
  - Recency in this model (call, stated before the fit): the simulation found the belief-variance test
    calibrated only when the recency curve is flexible and not shrunk (a straight line, a small ridge or the
    production centered spline moved recency into the belief variance). The repeat with the shared
    fitter's REML penalty (design, thirteenth round; simulation) found that a penalty chosen from the data
    makes each session's recency curve a straight line at about 140 events of a kind, and the
    belief-variance test is then wrong (61 % and 68 % false positives). The design stays one fit for each
    session (fourteenth round).
  - WAITS FOR THE LEAD (smooth-term rule): how the recency curve of this model is fitted (a natural spline
    of 4 columns with no penalty, or a P-spline with a fixed weak penalty). Either is stated before the
    first real fit.
  - The recency curve the paper shows is read from this model, the one that holds the belief variance, the
    sample count and the trial number: read from the model without them, the curve absorbs a true
    belief-variance effect or a drift (simulation, bias 0.3 to 0.9). Its supported range in the
    simulation was 5 to 80 trials since the last visit.
  - The belief variance is predicted by recency within a session (VIF far above the collinearity limit by
    construction), and its simulated power is low; it is reported under C5's rule for a term that fails the
    collinearity rule by design, and a null is reported as "no evidence beyond recency; the design cannot
    detect an effect of this size", with the simulated power beside it (call, stated before the fits).
  The identifiability study, and its repeat with the shared fitter's recency penalty, are the gate of
  Phase 6a.
- Q6 (Phase 7): one more role, `next`, for the arm of the next trial's well when it is neither current nor
  previous (call: when the next arm is the current or previous arm, the event adds nothing to this term).

## JAX batch twin

For the mPFC unit models only (Phase 8): the same objective, Newton rule and pooled REML (over a session's
units) as the NumPy reference, vectorized over units with `jax.vmap` and run in float64 on CPU for tests and
in float32 on the GPU (design 5, check 6). Read for method, not imported:
`/cumulus/edeno/neurospatial/src/neurospatial/encoding/_glm_jax.py:215-459` (`_newton_loop_jax`,
`_reml_score_jax`, `select_penalty_by_reml_jax`) and this repository's `analysis/pfc/batched_glm.py:460-640`
(`_sweep`, `_solve`, `_parameter_covariance`). Known pitfalls with the pinned library stack are in the project
memory: float32 drift in marginals, A100 matmul precision (set `jax.default_matmul_precision("highest")` for
the check).
