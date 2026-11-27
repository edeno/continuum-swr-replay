# One convention for every regression model, and the behavior figure (Figure 4)

Design, 2026-11-20. Status: **for the lead's approval; nothing is built.** No production model, table or
reported value changes before this document is approved.

Marks used below: **(lead)** is a decision the project lead made, with the date; **(call)** is a choice the
author of this document made and the lead has not confirmed; **(open)** needs the lead's decision.

## 1. Why

Figure 4 is the behavior figure (lead, 2026-11-20; the mPFC figure is Figure 5). Its question, in the lead's
words: "is there anything different about the role of non local events that overlap with SWR and those that
don't". It also has to show that events with no SWR vary with behavior, and at the least describe how both
kinds of event relate to behavior.

The lead set four conditions for the models behind it (2026-11-20):

- the behavioral models are computed correctly and in one consistent manner, to a standard a journal's
  statistics review accepts;
- the covariates are valid;
- each model asks and answers a stated question;
- "it is also important that we are consistent across all the regression models", and "mPFC models should be
  included".

An independent check of the current models (section 2) found that three of the six questions cannot be
answered by the models as they are. So this document covers two things: one convention that every regression
model in the paper follows, and the models and panels of Figure 4 built on it.

### Goals

- One convention (section 4) for the data, the design, the fit, the folds, the inference and the report of
  every regression model: rate, content, choice, ripple power and the mPFC unit models.
- One shared fitter inside this repository (lead: "We should keep our own copy of the method and make sure it
  is correct"), checked against outside references (section 6).
- Figure 4: the six questions of section 3, each answered for SWR events, for No SWR events and for their
  difference, plus the time course relative to when the animal leaves a well (lead).

### Not goals

- A claim that uncertainty, as distinct from recency, shapes content. The evidence does not support it
  (section 2.3). It can return only as a pre-declared test under the new convention.
- The question whether No SWR events are decoding spillover from the animal's own position. The lead placed
  it in a supplement or the text; the wider-Local-kernel re-decode (question 2c.6) stays separate work.
- A change to event definitions, cohorts, the decoder or the canonical tables' meaning.
- A single joint model of all sessions as the first step. The lead wants the 242 per-session fits first
  (section 7).

### What would make this a failure even if it runs

- A reported effect that a simulation with no such effect also produces (this is the present state for the
  uncertainty term and for part of the reward effect on SWR rate).
- Two families of model that state the same kind of result under different rules.
- A p-value that treats the 242 epochs as independent.

## 2. What the independent check found

The lead asked for a check that does not rely on earlier audits ("Verify yourself", 2026-11-20). Three
independent checks, three prototypes on simulated data, and scripts by the author re-ran the key points on
2026-11-20. Their scripts, small result tables and plots are kept as scratch (ignored by git) in the main
checkout under .claude/research/2026-11-20-regression-prototypes/. Every number in this section is a
point-in-time diagnostic from those scripts. None is a reported value. The implementation turns each finding
it depends on into a test or an analysis with a stored output (section 6).

What holds in the current code:

- The reward-history covariates never use an outcome before the animal knew it (checked against raw port
  data in five sessions, one per animal). Current arm, previous arm, next arm, reward fraction and value
  match independent recomputation.
- Every value of the behavior summary that was checked reproduces exactly from the stored fits, and the
  stored fits match the current tables.

### 2.1 Rate

- Event starts are fewer before a stop ends. In stops of 2.5 s or more, SWR event starts in the last 100 to
  200 ms run at about a quarter of their rate one second earlier and are reduced from about 0.8 s before the
  end, in 5 of 5 animals; counted No SWR starts are reduced only in the last 50 ms. Section 2.7 shows what
  makes this: for SWR events it is in the data, not the counting rule.
- Stops after an omitted reward are short, so most of their time is this end-of-stop time, and a model
  without a clock for it reads the difference as a reward effect. On simulated events laid over the real
  stops, with no true reward effect and only the counting rule, the production rung M7r returned a reward
  rate ratio of about 1.10 and called it significant in 18 of 20 simulated data sets; a model with a second
  clock (time until the stop ends) returned about 1.02.
- On the real data the SWR lag-1 reward kernel falls from the reported value to about 1.19 when each stop's
  last 0.5 s is removed. The simulated artifact explains part of that fall; the rest is not explained
  (**open**: real suppression before the animal leaves, or more censoring than the simulation planted).
- The difference between kinds in the reward effect survives every variant (about 1.4, 5 of 5 animals,
  recording-day and animal-level tests, leave-one-animal-out).
- The rise of the rate over the first seconds of a stop is robust in both kinds (model and model-free).
- The per-bin model covers only the first 4 s of a stop, which leaves out about a fifth of the cohort's
  events. Its reward basis is strongly collinear and its fixed-knot spline fails in sparse sessions.

### 2.2 Content

- The models fit one slope per covariate across all arms. Within session and arm, the slope of content
  share on reward fraction and on value is negative on the current arm for SWR events (0 of 5 animals
  positive) and positive on the other arms (value, both kinds, 5 of 5). One pooled slope mixes them. On
  simulated content with opposite true effects by role, the pooled rungs returned no effect; a model with
  slopes by role recovered both.
- The share of content on the current arm rises as the stop's end nears, in both kinds, 5 of 5 animals. It
  does not change with time since arrival. At a well the current arm is the only way out, so this may be the
  path ahead.
- Comparisons between kinds depend on whether event length is adjusted for. The reported per-kind
  coefficients are evaluated at the pooled mean length, which the descriptions do not say.
- The cross-validation of the content models uses a different optimizer from the fit it scores.

### 2.3 Uncertainty

- The belief variance is mostly a clock of how long an arm has gone unvisited: it climbs back to its
  ceiling when an arm is not visited, and among the arms the animal is not using it correlates about 0.77
  with log trials since the last visit within an event.
- On simulated content whose truth is recency only, the production rung C13 returned an uncertainty slope
  of about +0.22 and called it significant at the animal level in every simulated data set at 242 sessions.
  The reported real values are of that size. The cause was isolated: the recency spline is centered within
  the event, so it cannot move the other arms as a group against the current arm, and that signal goes to
  the pooled uncertainty slope.
- A model with slopes by role and an uncentered spline of recency keeps the false-positive rate near its
  nominal level. With a straight-line recency term it does not when the true curve is curved.
- On the real data under that model, uncertainty beyond recency is about zero in No SWR events and small in
  SWR events (5 of 5 animals, recording-day p about 0.02). At the real spread of the counts, separating the
  two needs about the whole cohort.

### 2.4 Penalized splines

- A minimal P-spline with the penalty strength set by restricted maximum likelihood (REML) reproduced mgcv
  on the same data and basis (effective degrees of freedom to 1e-10 at the same penalty; curves to about
  1e-5; the same coefficient and standard error for a covariate beside the smooth).
- The REML score can have two optima. mgcv's search stopped at the worse one in 12 of 3,600 simulated data
  sets; a local search of our own did the same in about 1 % until it started from a grid.
- With no true change, the penalized spline reported one in 5 to 7 % of data sets. A flat or straight-line
  truth shrinks to a straight line. A 4-fold rise is found 80 % of the time from about 100 events; a rate
  that halves in the last 0.3 s needs about 300.
- With knots 0.33 s apart, a change that takes 0.05 s is drawn too smooth and the 95 % interval covers the
  truth only 76 to 86 % of the time.
- The fixed-knot, unpenalized fit the project uses now failed outright on a few simulated data sets and gave
  wild curves at 50 events.

### 2.5 Folds and held-out scores

- Inside one session, a held-out score on trial folds cannot show that a small real effect exists: a true
  reward rate ratio of 1.3 lowered the held-out score in most sessions, and at the real spread of the content
  counts every added term lowered it, true or not. The real data show the same pattern.
- A penalty strength chosen per session is unstable; one strength shared by all sessions recovered the
  curves as well or better.

### 2.6 The models differ from each other

An inventory read from the code lists 15 families (the scratch folder holds a condensed copy). They use at
least nine convergence tests and six rules for a term that cannot be estimated; no penalty in the rate,
content and choice families, a cross-validated one in one history model and a fixed one in the mPFC models;
six fold designs; six ways of scaling a covariate; uncentered interactions everywhere; and seven kinds of
test with six different multiple-comparison families.

### 2.7 Stop ends and small movements at the well (exploration of all 242 epochs)

The lead asked whether the end-of-stop drop is a problem in the data, and worried about "cases where the
animal moves a little bit and the animal is still at the well". The decode frames of all 242 epochs were
read with the project's own event functions; the rebuild reproduces every row of the event table and of the
stop table. All numbers are diagnostics.

- **The counting rule is not what makes the SWR drop.** Without the rule that the animal is immobile for the
  whole event, 233 more SWR candidates start in a stop and cross its end (0.6 % of counted SWR events).
  Adding them leaves 94 to 99 % of the drop. SWR starts fall from about 0.8 s before the stop ends; head speed
  begins to rise about 0.4 s before. One limit: the SWR detector itself needs low speed at an SWR's first and
  last sample, so an SWR across a stop end cannot be in the inventory; treating crossing candidates with
  high ripple power as SWR fills only the last 100 ms.
- **For No SWR events the rule makes the last-50-ms drop, and hides a rise.** 6,541 No SWR candidates start
  in a stop and continue into movement (12 % of the counted No SWR events; median duration 0.42 s). With them
  the start rate rises two to three times in the last 100 ms. The paper's definition leaves these out.
- **Most stop ends are small movements, not the animal leaving.** 58 % of well-anchored stop ends are
  followed by another stop at the same well. The movement between is small: median 0.22 s, peak speed
  5.7 cm/s, net head displacement 1.6 cm, the head within about 4 cm of the well. Position and speed alone
  identify them in every epoch (98 to 99.8 % of stops that are not the last of a visit).
- **The stop clock mixes arrivals with re-stops.** Counted from the start of a stop, the SWR start rate at
  0.5 to 1 s is about 0.18 per second; counted from the animal's arrival at the well it is about 0.03.
  Counted from arrival, the current-arm share falls to about 0.08 at 0.5 to 1 s and rises to 0.3 to 0.4 by 5
  to 8 s; an earlier first look on the stop clock had shown no change.
- **Before the animal leaves** the current-arm share reaches about 0.5 in the last 0.25 s in both kinds;
  before a small movement it is about 0.31 to 0.35. The SWR drop appears before every movement, small or
  large.
- **Three ways to set the unit** (each stop alone; stops with their ends marked "stays" or "leaves"; the
  well visit, with stops joined across small movements) give nearly the same curves against time until the
  animal leaves. They differ in the clock since arrival and in the units at risk: 93,126 stops against
  45,005 visits, with nearly the same events.

## 3. The questions

The lead accepted this list (2026-11-20) and added the last row. Each is asked for SWR events, for No SWR
events, and for the difference between the kinds.

| # | Question | Response | Status after section 2 |
|---|---|---|---|
| Q1 | Does the rate of event starts change over the time at a well? | rate | strong; model needs the whole stop |
| Q2 | Does the outcome just received, and recent reward history, change the rate? | rate | difference between kinds strong; size for SWR needs the end-of-stop fix |
| Q3 | Is content biased to where the animal is and where it came from? | content | strong (current arm, current patch; previous arm is suppressed) |
| Q4 | Does content follow an arm's reward history or value? | content | not answerable with pooled slopes; ask by arm role |
| Q5 | Does content follow how long an arm has gone unvisited, and does uncertainty add to that? | content | recency strong; uncertainty beyond recency not supported |
| Q6 | Does content relate to where the animal goes next? | content, choice | null once behavior is in the model; next arm equals previous arm in over 80 % of events |
| Q7 | How do rate and content change as the animal gets ready to leave the well, and before a small movement at the well? | rate, content | strong first look; needs the model |

"SWR" and "No SWR" mean non-local events that overlap an SWR or overlap none, as in Figures 2 and 3. Each
comparison between the kinds states whether event length is held fixed ("direct") or not ("total"); both are
reported.

## 4. The convention

### 4.1 Data

- **The unit is the well visit** (lead, 2026-11-20, after the exploration of section 2.7): from the animal's
  arrival at a well to its last nose-poke out. The immobile stretches of a visit are joined across the small
  movements between them. Time after the last nose-poke out is run time and is not analyzed (lead: "if the
  animal has poked out then we count it as part of the run time and don't analyze them"); an event counts
  when it lies wholly inside the visit. The ripple-power model joins this cohort.
- At-risk time for event starts is the immobile time of the visit, less time inside an ongoing event, as
  now. Time in a small movement is not at risk.
- Each time bin carries three clocks (lead): time since arrival at the well; time until the animal leaves
  (the last nose-poke out); and time until the next movement of any size. The third is "to try": SWR starts
  fall before every movement, small or large (section 2.7), and it is kept only if the collinearity rule of
  section 4.7 lets it be told apart from the second.
- The whole visit is modeled, not the first 4 s of a stop.
- The stop table gains what the visit needs: a visit identifier, the visit's arrival and last nose-poke out,
  and each stretch's kind of end (a small movement or the animal leaving). This is a rebuild of the canonical
  tables as one set.
- A row with a missing value is never dropped inside a fitter. The cohort step drops it and counts it, per
  kind.

**Label in the tables, filter at analysis** (lead, 2026-11-20: "lets label these events in the data
structures and then when it comes to analyze them (summary stats, summary plots, before regression, etc)
then we can filter, we can always choose to add them later or analyze them as groups").

- No event leaves the canonical tables. The event table and the stop table gain a label that says where each
  event or stop is relative to the well visits (**call** for the names and thresholds, from the exploration
  of section 9): inside a visit; next to the well just left, after the last nose-poke out; crossing the last
  nose-poke out; stopped on the track (leaf arm, trunk arm, patch junction, center); next to the next well
  before its first nose-poke in; before the epoch's first visit; other. Beside the label: the visit
  identifier, time since the last nose-poke out and time until the next nose-poke in. Position is the
  animal's projected position at the event's midpoint; "next to" a well or a junction is within 10 cm (the
  counts do not hinge on it).
- The labels add columns and drop nothing, so the eligible events and their identifiers do not change.
- One function owns the filter, and every summary statistic, summary plot and model of Figures 2 to 5 calls
  it; none filters by itself. Its default is the common subset: events inside a well visit, in the 242
  analyzed epochs (Figure 5: in the epochs with mPFC units). Each summary records how many events each
  label holds and how many the filter kept.
- The events outside the visits can be added later or analyzed as groups by a change of that one call.

### 4.2 Design

- A variable has one form in every model:
  - a clock, trials since the last visit, or peri-event lag: a penalized smooth term (cubic B-spline basis
    with a second-order difference penalty);
  - an arm's role (current, previous, other): a category, with a separate slope of each arm covariate for
    each role;
  - a kind of event: a category; a smooth or slope that differs by kind is a separate term per kind, so the
    difference between kinds is a stated contrast.
- The number of basis functions is set generously and checked, not tuned by hand. The time-until-leave clock
  uses closer knots than the time-since-arrival clock, because its feature is sharp (section 2.4).
- Recency enters as a smooth of log trials since the last visit, not centered within the event. This form is
  provisional: stage 1 runs the comparison of open decision 6 first and the lead decides before the content
  design is frozen.
- The content response is each event's share over its candidate arms, one unit of weight per event, as the
  shipped fits use (lead, 2026-09-05: fractions stay shipped). The count response and the
  Dirichlet-multinomial are robustness estimators on the same design.
- The candidate set is all nine arms, for every content question (lead, 2026-11-20: "so that we can better
  describe the activity"). The trunk arms are their own role, with no reward, value or visit covariates, and
  every event stays in the cohort. Arm fixed effects stay in every content model, per kind. (The prototypes
  used the six leaf arms; the nine-arm design goes through the same simulations before it is used.)
- Uncertainty has three measures, declared before the fit and each tested once with recency in the model
  (lead, 2026-11-20): the belief variance as now; how variable an arm's rewards were; and how many times the
  animal has sampled the arm.
- One standardization: a covariate that is reported per standard deviation is scaled once by a pooled
  standard deviation computed on the cohort, with one stated definition (**call**: population form,
  ddof = 0), stored with the fit.
- A continuous factor that enters an interaction is centered at a fixed pooled reference. A 0/1 factor is
  not.
- Reward history, in place of three overlapping bumps (lead, 2026-11-20: "A and D separately"), is asked by
  two separate models:
  - A: the last outcome alone, a 0/1 term. Its rate ratio is the tested estimand of Q2.
  - D: one penalized smooth curve over the last ten outcomes (the effect of a reward at each lag; the form
    mgcv documents as a linear functional of a smooth). It describes how the effect fades and is shown with a
    band; it is not in the list of tests. The lags enter on a log scale, so the curve can change fast between
    the first lags (**call**), and a simulation with a sharp effect at the last outcome checks what it returns.
  The last outcome is only weakly related to older ones (a diagnostic on the stop table, 2026-11-20:
  correlation about 0.03 with the outcome before it and about 0.14 three and five back), so A's ratio is close
  to the effect of the last outcome alone.

### 4.3 Fit

- Every model is a penalized likelihood fit by the shared fitter (section 5).
- The penalty strength of each smooth term is set by REML, one strength per term shared by all sessions of a
  model and kind (section 2.5), starting from a grid (section 2.4). Per-session strengths are not used.
- Terms with no smooth (role slopes, kind contrasts, the reward effect) carry no penalty and must pass the
  support rule below.
- One convergence rule for every likelihood: the Newton decrement of the penalized objective below a stated
  tolerance, in double precision.
- One rule for a term that cannot be estimated: decided from the data before the fit (the term varies in too
  few units with events on each side), stated per term and session, and stored. A term that fails it in more
  than a stated share of sessions leaves the per-session model and is asked in a pooled model instead.
  (Not applied to the support rule: no fit read that share, and its function,
  `support.terms_failing_share`, was removed on 2026-11-24, the lead's choice. The constant
  `MAX_FAILING_SHARE` remains: the penalty search and the rate model's third-clock decision,
  `visit_rate.third_clock_decision`, read it.)

### 4.4 Folds

- One fold rule (lead): inside an epoch, whole trials are the folds, five of them, balanced by destination
  well, with long stays cut into pieces, as the function trial_folds of the Figure 2 branch
  `swr-trial-folds` does. That branch is not merged, and this work does not wait for it (**call**): the rule
  is written once in the shared regression package, and when that branch merges its code calls the shared
  function, with a test that the two give the same folds.
- Folds are used to compare designs and to check over-fitting. They are not used to set penalty strengths
  (REML does that) and not as the per-session test that a small effect exists (section 2.5).
- A held-out claim in the paper is made on the sum over all sessions, with recording days and animals as the
  units of its uncertainty.

### 4.5 Inference

- The 242 epochs are not independent. Each result is reduced session to recording day to animal.
- The reported estimate is the animal-balanced summary; its interval is the project's nested bootstrap
  (animals, then their sessions) with the small-sample t correction; beside it stand the number of animals
  with the estimate's sign and the recording-day-level test.
- With five animals a sign test cannot go below p = 0.0625, so no claim rests on a p-value over animals
  alone.
- One family for multiple comparisons, declared before the fits. Each question has scalar estimands, each
  tested for SWR, for No SWR and for their difference; the false discovery rate is controlled over the whole
  family. The proposed members (**call**; the lead can edit the list before stage 1):

  | Question | Scalar estimand |
  |---|---|
  | Q1 | rate 2 to 4 s after arrival at the well over rate in the first second |
  | Q2 | rate ratio for a rewarded against an omitted last outcome |
  | Q3 | share on the current arm over the share on another leaf arm; the same for the previous arm |
  | Q4 | slope of the current arm's share on its reward fraction; slope of another arm's share on its value |
  | Q5 | share of an arm 20 trials since its last visit over one 3 trials since; slope on uncertainty with recency in the model |
  | Q6 | share on the next arm over the share on another arm that is neither current nor previous |
  | Q7 | rate in the last 0.25 s before the last nose-poke out over rate 1 to 2 s before; the same for the current-arm share |

  That is ten estimands, each for two kinds and their difference: 30 tests.
- A curve is shown with a band that is valid for the whole curve, from draws of the fitted coefficients. The
  curve describes; the scalar estimand above is what is tested.
- Later, a joint model with session, recording day and animal as grouping terms may replace the two-stage
  summary (section 7, stage 3).

### 4.6 Report

- A rate effect is a rate ratio; a content effect is a ratio of content shares; each with its interval.
- Every stored fit carries the same record: penalty strengths, effective degrees of freedom per term, the
  convergence measure, the support of each term, and the basis.
- Each statistic lives in `analysis/` and the figure calls the function the summary reads, as now.

### 4.7 Collinearity

The lead (2026-11-20): "we should also be concerned about collinearity in this model and all the models".

- Every model's design is measured before its coefficients are read, and the measure is stored with the fit:
  for a term with one column, how well the other terms predict it (the variance inflation factor); for a
  smooth term, the same idea for its whole curve (concurvity, as mgcv reports it).
- The measure is taken on the quantity the likelihood sees: within the event for the content models, on
  at-risk time for the rate models.
- A threshold is set once for every family (**call**: a term is "separable" when the other terms explain
  less than 80 % of it, a variance inflation of 5). A term over it is not reported by itself: it is reported
  together with the term it cannot be told from (as recency and uncertainty are today), or one of the two
  leaves the model by a rule written before the fit.
- A penalty steadies a fit but adds no information. A pair of terms the design cannot separate stays
  inseparable under any penalty, so the rule above applies with or without one.
- Each model is also checked on simulated data laid over its real design: an effect is planted on one term,
  and the fit must not report it on another.
- Measured for the three clocks on the real at-risk time (a diagnostic; visits from position, 45,005 of
  them): time since arrival is nearly independent of the other two (they predict 6 % of it). Time until the
  animal leaves and time until the next movement predict about half of each other (43 % and 51 %), because
  61 % of at-risk time is in a visit's last immobile stretch, where the two are equal. They can still be
  told apart: of the at-risk time within 0.5 s of a movement, about half is before a small movement and
  half before the animal leaves.

## 5. The shared fitter

One module in this repository (**call**: a new package `analysis/regression/`), used by every family.

- Likelihoods: Poisson with a log-exposure offset (rate, mPFC units); softmax over candidates within an event
  (content, choice); Bernoulli (stay or leave); Gaussian (ripple power). The Dirichlet-multinomial stays a
  robustness estimator on the same design.
- Penalties: a list of quadratic penalties, one per smooth term, each with its own strength.
- Fit: penalized Newton with step halving. Penalty strengths by Laplace-approximate REML with exact
  derivatives (the prototype's derivative-free search for two strengths was slightly worse than mgcv).
- Batches: many fits of the same shape in one call (sessions, kinds, folds, units).
- Precision and device: a double-precision NumPy reference and a JAX version for batches. A GPU is used only
  after `nvidia-smi` shows the card has no other job (lead, 2026-11-20). The measured CPU cost of the rate and
  content fits is small (about 0.08 s for one two-clock rate fit, 0.03 s for one content fit), so those run
  on CPU; the mPFC unit models are the GPU case.
- References, read for method, not imported (lead): the lead's penalized Poisson model with REML in the
  neurospatial repository (`encoding/_glm_numpy.py`, `_glm_jax.py`), the Laplace rate model and the softmax
  Newton code in state_space_practice, the spline Poisson regression in non_local_detector, the two batched
  fitters already here (`analysis/replay_models/rate/ridge_history_cv.py`, `analysis/pfc/batched_glm.py`),
  and mgcv.

## 6. How the fitter is shown to be correct

Each is a test in the suite unless marked.

1. No penalty: the same coefficients as the current fitters on real sessions, to a stated tolerance.
2. One penalty: the same selected strength and curve as the neurospatial fitter on a one-term model.
3. Softmax: the same result as the current conditional Poisson fitter on the same data.
4. Hand cases: models of two or three parameters against a direct calculation.
5. Known truth: on simulated data the fitter recovers planted coefficients and curves, its 95 % intervals
   cover at the stated rate, and a term with no true effect is reported at the nominal rate. The three
   prototypes of section 2 become these simulations, including the two false effects found (reward from the
   end-of-stop rule; uncertainty from recency).
6. GPU against CPU: single precision on GPU agrees with double precision on CPU to a stated tolerance.
7. Outside reference (a documented script, not in the suite): agreement with mgcv on a few sessions. The
   environment is a separate conda environment, `csr-mgcv` (R 4.5.3, mgcv 1.9-4), created with
   `mamba create -y -n csr-mgcv --override-channels -c conda-forge r-base r-mgcv r-jsonlite` (lead approved,
   2026-11-20). It is not part of the project's uv environment.

## 7. Order of work

Stage 1 comes first (lead: "I want to do the 242 fits first").

- **Stage 0, the fitter.** The module of section 5 and checks 1 to 6. No model changes.
- **Stage 0b, the labels and the one subset.** The visit labels and the visit's clocks in the canonical
  tables (one regeneration of the table set; identifiers unchanged), the one filter function, and Figures 2
  and 3 with the events summary rebuilt on the common subset. Their numbers change: 75,675 events become
  69,011 and the No SWR share goes from 56.8 % to about 54.5 % (diagnostics of 2026-11-20). The two data
  points of section 9 (one epoch's head positions; a few label and position disagreements) are checked here.
- **Stage 1, per-session fits on the convention.** In this order, each with its refit, its summary values
  and its tests:
  1. rate (Q1, Q2, Q7): two clocks, whole stop, the reward term, by kind;
  2. content (Q3, Q4, Q5, Q7): slopes by role, smooth recency, the current-arm term against time until the
     stop ends, by kind, with the direct and total contrasts;
  3. choice and stay or leave (Q6), and ripple power, on the same cohort and rules;
  4. the mPFC unit models (lead: included): the same penalty rule (REML in place of the fixed prior), the
     trial-fold rule, the same convergence and support rules. Their last fit took about 14 GPU-hours, so this
     step is measured on a few sessions first.
- **Stage 2, inference.** The session to day to animal reduction, the bootstrap interval, the declared
  family, the simultaneous bands (section 4.5), applied to every family.
- **Stage 3, pooled model (only if stage 2 is not enough).** One model per question with session, recording
  day and animal as grouping terms. The per-session fits are its first step.
- **Stage 4, Figure 4.** Drawn to the standards of Figures 1 to 3 and 5 from the stage 1 and 2 outputs.

### What happens to each existing family

Every family whose values the paper reports either moves to the convention or leaves the paper with its
values. Nothing is removed without the lead's yes (open decision 4), and never before its replacement's
values are in the summary.

| Family (section 2.6) | Proposed (**call**) |
|---|---|
| Per-bin rate ladder | replaced by the visit model with its clocks (stage 1.1) |
| Per-stop rate ladder | kept, fit by the shared fitter, as the "total" reward effect per second of stop, if decision 3 asks for it |
| SWR-contrast rate ladder | its kind contrast becomes part of the two-clock model; its other covariates on rate (prediction error, value, entropy, switch rate) are in none of the seven questions: lead decides |
| Uncertainty rate model | in none of the seven questions: lead decides |
| Event-history models (duration-aware, ridge cross-validated) | kept; moved to the shared fitter after Figure 4, since they already use at-risk time and a penalty |
| Older all-bins cross-history model | deleted in the first commit of the build (lead: "ok delete"); nothing reports it |
| Content ladders (arm, patch, contrast, leaf) | replaced by the model with slopes by role (stage 1.2) |
| Dirichlet-multinomial content | kept as the robustness estimator on the new design |
| Within-arm comparison of uncertainty, recency, value | deleted when the content model's Q5 values replace its values (lead: "ok delete") |
| Choice | moved (stage 1.3) |
| Stay or leave | 7 % of its fits converge now; refit on the convention, then lead decides whether it stays in the paper |
| Ripple power per event | moved, onto the shared cohort (stage 1.3) |
| mPFC peri-event ladder, marked kernels, stop context | moved (stage 1.4): the penalty by REML shared over the units of a session, trial folds, the shared convergence and support rules, and population statements reduced session to day to animal. The peri-event design, the matched-control draws and the per-unit test keep their definitions (**call**) |

## 8. Figure 4, first plan

Every panel draws SWR and No SWR separately, named and colored as in Figures 2, 3 and 5. This plan is the
starting point for the figure's own design pass after stage 1.

| Panel | Shows | Question |
|---|---|---|
| a | An example stop: events of both kinds against the two clocks | orientation |
| b | Rate against time since arrival, each kind | Q1 |
| c | Rate against time until the animal leaves, and against time until a small movement, each kind | Q7 |
| d | Effect of the last outcome on the rate, each kind and the difference | Q2 |
| e | Content share on the current arm, the previous arm and other arms, each kind | Q3 |
| f | Current-arm share against time until the animal leaves, each kind | Q7 |
| g | Content share against trials since the last visit, other arms, each kind | Q5 |
| h | Slopes by arm role for reward history and value, each kind | Q4 |
| i | Content and the next choice | Q6 |

## 9. Decisions

Settled by the lead on 2026-11-20 (the rounds below record the words): the nine-arm candidate set; the well
visit as the unit, ending at the last nose-poke out, with three clocks; both forms of the reward effect; every
remaining family moved to the convention, and two deleted; the mPFC models in this work; absolute recency;
reward history asked by models A and D; two more measures of uncertainty; one data subset for Figures 2 to 5,
made by labels in the tables and one filter at analysis. Nothing in this list is open now.
The questions as first put:

A second reader checked this document as a builder would (2026-11-20). Its questions that only the lead can
answer are 1 to 5; 6 was open before.

1. The candidate set of the content models. The six leaf arms (content shares rescaled over them; events
   with no leaf content leave the cohort, about 12 % in the current tables; the prototypes used this), or
   all nine arms with the trunk arms as their own role (every event stays). The project's history shows the
   choice moves coefficients, so it is fixed once for every content question. Proposed: the six leaf arms,
   with the count of events left out per kind reported and the nine-arm fit as a robustness check.
2. What "the time the animal leaves" is: the end of the immobility bout (speed crosses 4 cm/s), as the
   prototypes used and as the stop table gives, or the nose-poke out of the well. With the nose-poke, the
   unit of the models becomes the well visit, which can hold several stops, and the cohort, the at-risk time
   and the folds change with it.
3. The estimand of the reward effect on rate. With the second clock, the reward effect is the effect at a
   given time before the animal leaves, and leaving is itself a consequence of reward. Report this "direct"
   effect, the "total" effect per second of stop (the per-stop family), or both?
4. The families in none of the seven questions (table in section 7): rate against prediction error, value,
   entropy and switch rate; the uncertainty rate model; the within-arm comparison; the older all-bins
   history model; stay or leave. Keep each in the paper and move it, or let it go?
5. The mPFC step changes Figure 5's numbers and needs GPU time (the last fit took about 14 GPU-hours). Do it
   within this work, or after Figure 4 is drawn? And the uncertainty covariate: report "no evidence beyond
   recency" with the beta variance as it is, or also test a second definition of uncertainty that is not a
   clock of time since the last visit?
6. Whether recency acts through the absolute time since the last visit (what the simulation assumed) or only
   through the order among the other arms. Stage 1 runs this comparison on the real data first; the lead
   decides how recency is centered from its result.

### Decisions recorded (lead, 2026-11-20, after reading this section)

- **1, candidate set:** all nine arms, "so that we can better describe the activity". The trunk arms are
  their own role and every event stays in the cohort.
- **3, reward effect:** "whatever is the most fair". Read here as: report both, each named (**call**): the
  total effect per second of stop, and the direct effect at the same time before the animal leaves.
- **4, other families:** "move it to the convention and fix if needed". Two are still with the lead: the
  within-arm comparison and the older all-bins history model, each proposed for removal because the content
  model's Q5 and the two newer history models answer their questions.
- **5, mPFC:** the mPFC models go onto the convention in this work. The second definition of uncertainty is
  still with the lead.
- **6, recency:** absolute time since the last visit. The stage 1 comparison stays as a check.
- **2, what "leaves" means:** still with the lead. Measured on the five sessions with cached photogate data
  (one per animal, 981 well visits; a diagnostic): the last stop of a visit ends within 0.25 s of the final
  nose-poke out in 90 % of visits and within 0.5 s in 96 %, but a visit holds a median of two stops and 63 %
  of stop ends are not the animal leaving (it moves at the well and stops again). Proposed: keep the
  immobility stop as the unit, mark each stop end as "leaves" or "stays", and take the final nose-poke out
  as the time the animal leaves. The poke times are cached for five sessions only; the other sessions need a
  read of the lab database and two added columns in the stop table.

### Second round (lead, 2026-11-20)

- **4:** "ok delete" for the within-arm comparison and the older all-bins history model. The older history
  model goes in the first commit of the build (nothing reports it). The within-arm comparison goes when the
  content model's Q5 values replace its values in the summary, by the rule of section 7.
- **5:** "uncertainty would be a new result if true". So the second definition of uncertainty is wanted;
  which measures, and how they are declared before the fit, is still being discussed.
- **2:** still deciding; the lead wants to talk it through and explore the implications.
- Calls confirmed by the lead ("ok"): one scale for covariates, one shared fitter, the fold rule written
  once in the shared package, the table of proposed fates. Calls still being explained: the end of a stop,
  the form of reward history, the declared list of tests, and what changes in the mPFC models.

### Third round (lead, 2026-11-20)

- **5:** "ok" to two more measures of uncertainty, declared before the fit: how variable an arm's rewards
  were, and how many times the animal has sampled the arm. Each is tested once, with recency in the model.
- **Calls confirmed:** the declared list of tests; what changes in the mPFC models; the second clock for the
  end of a stop ("A sounds good"), with the condition that the data are explored first: an exploration of
  the decode frames is running to show whether the drop in event starts before a stop ends is made by the
  counting rule or is in the data, and what the small movements at a well are (for decision 2).
- **Reward history:** models A and D, separately (section 4.2).

### Fifth round (lead, 2026-11-20): one data subset for Figures 2 to 5

- "all our analysis should be on the same real data subset (figure 2-5)". So the data rule of section 4.1
  is meant for Figures 2, 3 and 5 too, not only for the models of Figure 4. Today Figures 2 and 3 use every
  immobile non-local event (75,675); 69,011 of them are at a well by the event's own label, and 6,381 are in
  transit (81 % of those have no SWR, against 54.5 % at a well; 4 to 12 % of an animal's events).
- **Open, and not to be decided before the data are understood** (lead: "Before we exclude these events,
  lets make sure we understand what the events that don't correspond to our rule are, when they occur,
  etc."): whether the common subset is immobile time inside a well visit, and whether the events outside it
  are described anywhere. An exploration of those events is running (what they are, when and where they
  occur, how they compare with events at a well, what exclusion changes).

### What the events outside the well visits are (exploration of all 242 epochs, 2026-11-20)

Diagnostics; the scripts, tables and plots are scratch in the main checkout under
.claude/research/2026-11-20-regression-prototypes/explore_outside_visits/.

- The at-well label already is the lead's rule: a well visit runs from the first nose-poke in at a well to
  the last nose-poke out, and position is not used. The nose-poke times are cached for every epoch (an
  earlier note in this document said five; that was one of two caches). So "inside a well visit" is the
  69,011 events the event table labels at a well, and no database read is needed for the rule.
- 6,664 events are outside (5,383 No SWR, 1,281 SWR; 8.8 % of all events, 4 to 12 % per animal):

  | Where the animal is | No SWR | SWR | Share of outside |
  |---|---|---|---|
  | still next to the well it just left, after the last nose-poke out | 2,584 | 420 | 45 % |
  | the event spans the last nose-poke out | 144 | 45 | 3 % |
  | stopped on a leaf arm (mostly the arm of the well just left, about 30 cm from it) | 1,654 | 526 | 33 % |
  | at a patch junction | 597 | 105 | 10.5 % |
  | at the center | 149 | 96 | 4 % |
  | on a trunk arm | 118 | 48 | 2.5 % |
  | next to the next well, before the first nose-poke in | 54 | 13 | 1 % |
  | before the epoch's first nose-poke in | 70 | 21 | 1.4 % |
  | other (some are a disagreement of label and tracked position) | 13 | 7 | under 1 % |

- So about half are the animal still at the well after its last nose-poke out (a median 1.6 s after it for
  No SWR events), and about half are stops on the track during a run (a median 6 s after the nose-poke out,
  about 40 % of the way to the next well, more often on a run that changes patch).
- They are concentrated: ten epochs hold 42 % of them, eight of those from one animal's first days; 768 occur
  after an epoch's last visit, where a few recordings run on for up to two minutes.
- No SWR events at a stop on the track differ most from No SWR events at a well: lower ripple power and
  multiunit activity (0 of 5 animals higher), content that ends at a well 57 % of the time against 20 %
  (5 of 5), and a higher held-out spike gain (5 of 5). No SWR events next to the well just left are like
  events at a well in duration, extent and speed, with a lower held-out gain (0 of 5 higher).
- Excluding all of them takes the events from 75,675 to 69,011, the No SWR share from 56.8 % to 54.5 %,
  5.75 % of the SWRs, 8.3 % of the immobile time, and 1.2 % of the mPFC cohort's events.
- Two data points to check: one epoch (chimi20200212_02_r1) has 141 events with the head 50 to 90 cm off the
  track while its projection is at the well; and a few events are labeled at one well while the tracked
  position is far from it.

### Sixth round (lead, 2026-11-20)

- No field for "where the animal was" in the literature survey: "no field survey, lets just categorize the
  events and understand them".
- The lead expects events at stops in the middle of a run "might be different functionally", and the
  exploration agrees for No SWR events (above).
- Decision: label the events in the data structures and filter at analysis (section 4.1). The outside
  events are not described in a supplement for now ("We might decide to at some point").

### Seventh round (lead, 2026-11-20): how the labels are built

- The labels go into the main tables as new columns, not into a separate label table ("new table is just
  avoiding future work").
- New columns change the table files, so every stored output that recorded the old files (the per-event
  measurements of Figures 2 and 3, the behavior fits, the mPFC bundles) is marked stale and its values
  pending until it is run again. The lead accepts that ("pending is fine, this whole system gets in the way
  of iteration"); no tool is built to avoid it.
- The labels and the one filter function are built first, on their own branch (`event-visit-labels`), and
  merged early so that other work can use them. That step changes no number. Figures 2 and 3 move to the
  filter in a second step, timed with the other session's Figure 2 work.

### Eighth round (lead, 2026-11-20): the order

- "whatever it is most efficient, we want to ... make sure figure 2-3 use the same data and then we want to
  execute our regression plan for figure 4-5."
- Order chosen for that (**call**): the labels merge and the canonical tables change first, before any long
  run, so that each stored run records the new tables once; the other session (Figure 2's held-out analysis)
  then takes master and uses the selection function, and its branch goes to master with the lead's approval;
  Figures 2 and 3 and the events summary move onto the common subset; then the regression plan of section 7
  for Figures 4 and 5. The plan is written in `.claude/docs/plans/regression-convention/`.
- One known cost of the table change: the other session's step that sums its scores over Figure 2's event
  windows refuses to run until the held-out tetrode run and the participation run record the new event
  table.

### Ninth round (lead, 2026-11-20): legacy code and the ContFrag tables

- "Let's make sure to cleanup any legacy code." Each phase of the plan removes the code it replaces, with
  no compatibility code left for older tables or fits, and the last phase is a sweep that proves nothing
  is left (`.claude/docs/plans/regression-convention/`). A value the manuscript cites is still removed
  only with the lead's yes, and an analysis the plan does not replace is a question for the lead.
- The ContFrag tables (the SWR-event and MUA-event content tables) get the same well-visit label as the
  non-local tables ("Label them now"), so that every table with `occupancy_state` carries `visit_context`
  and the table contract requires it.

### Tenth round (lead, 2026-11-20): the plan's first decisions

- **The two new uncertainty measures** (plan decision L2): as recommended, and "maybe we should check on
  simulated". Reward variability is the variance of the arm's 0/1 outcomes over the last 10 visits to its
  well; sample count is the log of one plus the visits to the well earlier in the epoch. Before any real
  fit, a simulation on the real data structure checks that each of the three measures can be told apart
  from recency, the reward fraction and the time in the epoch (false positives and power for each).
- **A stop that crosses the last nose-poke out** (L4): keep its time before the poke-out (option A). A
  check on all 242 epochs: 15,803 of the 93,738 stops that start inside a visit cross the final poke-out;
  their part before the poke-out has a median of 1.75 s and their part after it 0.07 s; that part before
  the poke-out is 31 % of the immobile time inside visits and holds 20,497 of the 69,011 in-visit events.
- **"Rate and content models should analyze the same events."** One event set for both families. Open
  point put to the lead: 189 events start inside a visit and end after the final poke-out. If the set is
  "the whole event inside the visit", the rate model must not count them, and it then loses 8 to 11 % of
  the event starts in the last 0.25 s before the poke-out and most starts in the last 0.05 s (a counting
  artifact at the end of the visit). If the set is "the event starts inside the visit", there is no
  artifact and the content models gain those 189 events.
- **"Arrival"** (L3) needs no decision: the stop table's arrival clock equals the first nose-poke in for
  every stop that starts inside a visit where both exist (93,302 stops, no difference), and is missing for
  554. The visit starts at the first nose-poke in.

### Eleventh round (lead, 2026-11-20): the remaining decisions

- **The event set**: the events that start inside a well visit (`visit_context` of `in_visit` or
  `spans_visit_end`; 69,011 + 189 events), for every model family and every figure. It replaces "the whole
  event inside the visit" wherever this document says so. The selection function's default selects it.
- **The dense time-bin table** is removed ("Remove"); the bins of the visit replace it.
- **The ContFrag regressions** (content and rate models of SWR-sourced and MUA-sourced events) move onto
  the new design ("Move them"), as two more event sources of the same runners, for a supplementary
  comparison.
- **The list of tests** is thirteen quantities, 39 tests ("Ok"). Section 4.5's "ten" was a miscount.
- **Figure 2's panels that count SWRs or seconds** use only time inside a well visit ("Ok").
- **The total reward effect: per stop or per visit?** The lead: "We need to be consistent in the repo."
  Read as (**call**): the well visit is the unit everywhere, the total reward effect is per visit, no
  per-stop variant is kept beside it, and stay or leave is one decision per visit. The same rule holds
  for the event set, the selection function, the fitter and the clocks, in every family, summary and
  figure.

### Twelfth round (lead, 2026-11-20): the uncertainty measures after the simulation

A simulation on the real data structure (content made from a known model; 100 data sets for each of seven
truths; `.claude/research/2026-11-20-regression-prototypes/proto_uncertainty_measures/REPORT.md`) gave:

- The belief variance is nearly recency within a session. Its test is correct (about 5 % false positives
  when the truth is recency alone) only when the recency curve in the model is flexible and not shrunk; a
  straight line in log recency gives 86 %, and the present production model 66 % (No SWR) and 100 % (SWR).
  Fitted alone it reports a true sample-count effect. A true effect of the size of the recency effect's
  linear part is found in 30 to 50 % of the data sets.
- The reward variability cannot be told apart from a curved effect of the reward fraction.
- The sample count is a drift with the trial number unless the trial number is in the model.

Decisions:

- **Reward variability is dropped** as an uncertainty measure ("Yes"). The reward fraction gets a smooth
  term instead. The declared family is twelve quantities, 36 tests.
- **The belief variance and the sample count are tested together in one model, with the trial number**
  ("Yes").
- (**call**) Before any real fit, the simulation is repeated with the shared fitter's penalty selection
  on the recency curve; if the selected penalty moves recency into the belief variance, the recency curve
  of this model is fitted without a penalty. The low power of the belief-variance test is stated before
  the fits: a null result is "no evidence beyond recency, and the design cannot detect an effect of this
  size".

### Thirteenth round (lead, 2026-11-21, 03:07): "Okay to decisions"

Put to the lead as a table; all four accepted:

- **The visit unit goes to master and into the tables.** `visit-clocks` was merged (master d14ff888) and
  the 13 canonical tables were published with the visit columns. Existing columns changed only on the rows
  that cross the final poke-out: the 189 crossing events and the 15,803 crossing stops take the visit's
  well as their current arm (**call**, for stops: the same rule as for events).
- **The control windows of tetrode participation and of the held-out tetrode score are drawn inside well
  visits**, and both runs are done again (the held-out one with trial folds), then the other session's
  window sums.
- **The shared fitter's merge waits** for a check of its penalty search on the content design.
- **One curve shared across sessions, or one for each session:** waits for the same simulation round. Its
  second round showed that a curve for each session, with one penalty chosen from the data, becomes a
  straight line when a session has about 140 events, and that the belief-variance test is then wrong
  (61 % and 68 % false positives); a recency curve with almost no penalty restores the test.

### Fourteenth round (lead, 2026-11-21, 04:40): one curve for each session stays

A third simulation round showed that one recency curve shared by the sessions of an animal recovers the
planted curve and keeps the belief-variance test correct with high power. I recommended it for the smooth
terms. The lead: "I don't want one curve for each animal. This is a whole new design and implementation."

- **The design stays: one fit for each session**, summarized by session, recording day and animal.
- What is known inside this design (simulation): a penalty chosen from the data makes each session's
  curve a line (a session has about 140 events of a kind) and the belief-variance test wrong; a small
  fixed penalty, or a small basis with no penalty, keeps that test correct (false positives 0 to 7 %) with
  low power (23 to 50 % at an effect of the size of the recency effect's linear part).
- (**call**, being checked by simulation) For smooth terms the per-session curve is fitted with a fixed
  weak penalty, and the reported curve is the average of the per-session curves. The check: does that
  average recover the planted shape, and at which strength.
- The shared fitter is corrected for the failures that the content design showed (sessions with no finite
  maximum, fits at very strong penalties, a flat score); its merge waits for that.
- Changed on 2026-11-24 for the behavior models: see the twenty-fourth round ("the unit of the fit").

### Fifteenth round (2026-11-21): the shared fitter on master; two questions open with the lead

No decision of the lead is recorded in this round. It records what was built and the two questions the
lead has not yet answered.

- **The shared fitter is on master.** It was merged in 6e54eda8 and cleaned up in 98fe5c06
  (`docs/design/regression_fitter.md`). The cleanup fixed two defects: the penalty search could return a
  point of negative curvature (a maximum or a saddle of the score) as converged when its last allowed step
  came there (df1d0625); and when two sessions failed first at different grid points with the same status,
  the count of failing grid points kept one session's count (9 failing points recorded as 3; 52d5b633).
  A change that fit the Gaussian by one exact Newton step was tried (726adc45) and reverted (538f83e9): on
  a near-collinear design one step was less accurate than the loop (largest coefficient error 3.9e-8
  against 2.9e-11), and it returned "converged" for an objective that was not finite.
- **The step limit depends on the scaling of a column** (`TODO.md`, a prototype on simulated designs of
  600 rows). With one N(0,1) column multiplied by 1e-4 or less, the Poisson, Bernoulli and softmax fits
  end at the step limit; multiplied by 1e-2, or by any factor above 1, every fit converges to the same
  optimum. Two replacement rules failed on the saved failure cases, so the fitter keeps its rule, and the
  convention's design check will refuse a column whose values are all very small (its threshold is set and
  tested in the phase that builds the check).
- **Open: the rule for a smooth term in a per-session fit.** The design stays one fit for each session
  (fourteenth round). Option A: a natural cubic spline with 4 columns, knots from the pooled data, no
  penalty. Option B: a P-spline with one fixed weak penalty per term. (**call**, the author's
  recommendation: option A, with a check that the conclusions hold at 3, 4 and 5 columns.) Known from
  simulation: a penalty chosen from the data by REML makes each session's curve a line when a session has
  about 140 events, and the belief-variance test is then wrong; a fixed, declared smoothness keeps that
  test correct. Either way the REML choice is not used for per-session smooth terms with few events; it
  stays in the package. Section 4.3's REML penalty shared by the sessions waits for this answer.
- **Open: a session whose simple terms have no finite maximum.** Today only the penalty search handles one
  (`select_penalties` sets the session aside and fits it again with the running-away unpenalized columns
  made inactive); `fit_at_penalties` raises at the first such session. Under either option for smooth terms
  the models are fitted at given penalties or with none, so this handling must exist outside the search.
  (**call**, the author's recommendation: the caller owns it, since it owns the support rule, and such a
  session is reported with those terms missing, not left out; leaving it out depends on the outcome,
  because the sessions that fail are those with no events in one condition.) In simulated content data 4 of
  242 sessions were of this kind, with 5 or 6 arm and role columns left out in each (simulation, not a
  result).
- A rule found by simulation for the support rule: when a 0/1 role column is made inactive for a session,
  the rows with that role are dropped from the session's fit, not merged into the reference role (merging
  inflated the belief variance's false positives, to 20 % in one case).

### Sixteenth round (lead, 2026-11-21, 10:10): a session with no finite maximum keeps its other terms

The lead: "A session with no finite best value: do terms missing."

- A session whose simple terms have no finite maximum is fitted again with the running-away unpenalized
  columns left out, and it is **reported with those terms missing**. It is not left out of the summary.
- The handling is built outside the penalty search, with the support rule (the per-session loop), and the
  record of the session names the columns left out and the reason.
- Still open: the rule for a smooth term in a per-session fit (fifteenth round).

### Seventeenth round (lead, 2026-11-22): the rule for a smooth term in a per-session fit

The lead, 10:59: "For decision 2, I think I would prefer the penalized splines". Then, 12:19: "I want to
stick with one fit per session for now". (Decision 2 was the question of the fifteenth round: the rule
for a smooth term in a per-session fit.)

The author's reading of these two answers (**call**). The author told the lead this reading on
2026-11-22 and on 2026-11-23, and the lead did not object:

- The fit stays one fit for each session, for every model.
- A per-session smooth term is the cubic P-spline that the package has (`pspline.PSplineBasis`). It has
  **one fixed log penalty per term**, declared in the model and the same in every session.
- No natural-spline basis is added. Option A of the fifteenth round is dropped.
- No penalty is chosen from the data for a per-session smooth term. The REML selection stays in the
  package (`reml.select_penalties`), for a family that may still need it.
- The value of each declared penalty is set per model, on that model's own real rows, by a target median
  number of effective parameters over the sessions (`regression/calibration.py:log_penalty_for_edf`,
  built on branch `regression-declared-penalty`, b47faec9 and 39a51d3a). A simulation on those rows
  checks the value. The fit is reported at that value and also at a stiffer and a looser value.
- A target does not carry from one model to another. Simulation: the same target (6 effective
  parameters) gave different log penalties in two designs (about 3.4 for one arrival curve alone, about
  1.8 with two clocks and the reward term), and biases of the Q1 contrast of opposite sign.
- Changed on 2026-11-24 for the behavior models: "one fit for each session" gives way to one fit for each
  recording day and for each animal (the twenty-fourth round, "the unit of the fit").

### Eighteenth round (lead, 2026-11-23, 09:52): no bias-reduced fit

The lead: "I don't want to use firth".

- The fit is plain penalized maximum likelihood. No bias-reduced fit is used in any family. This covers
  the full form and the partial form that the author had proposed.

### Nineteenth round (lead, 2026-11-23, 11:31): the rate questions are answered by the regression model

The author proposed a stratified estimate for each animal (a Mantel-Haenszel rate ratio), and then
plain window counts for each animal. The lead: "this all seems very complicated", then "I want the
regression model".

- The rate questions (Q1, Q2, Q7a) are answered by the regression model of section 4: one fit for each
  session, the plain penalized fit, the declared penalties.
- The model-free curves by outcome stay a diagnostic. They are not an estimator.

### Twentieth round (lead, 2026-11-23, 12:08 to 12:11): a simulation package and one JAX fitter

The lead: "yeah, maybe we should have a simulation package so that we have these for the rate, content,
etc models, all with the same principle, we should always be using the real regression code". The
author proposed this shape, and the lead said "yes":

- Each likelihood gets a function that draws outcomes.
- A task-free engine, `analysis/regression/simulation.py`, takes a model, its real rows and a scenario.
  It draws the outcome, runs the real fit, the real declared tests and the real reduction, and returns
  one results table (bias, spread, coverage, rejections, sessions without a value).
- A named scenario list sits beside each model in `analysis/replay_models/<family>/`.
- One runner script runs the scenario lists.
- Tests run on tiny cohorts.
- A scenario is either inside the model (planted coefficients) or outside it (any expected outcome as a
  function of the rows).
- The rule: a simulation replaces only the outcome column. Nothing rebuilds a model by hand.

The lead: "I would prefer a jax fitter for all models".

- There is one fitter, in JAX, for the rate, content, stay or leave, ripple power and mPFC models.
- The separate JAX twin for the mPFC models only is dropped.
- By the lead's rule on legacy code (ninth round), the NumPy solver is removed when the JAX one replaces
  it. Its outputs are kept as test fixtures.
- The port design is
  `.claude/research/2026-11-20-regression-prototypes/jax_fitter/DESIGN.md` (main checkout; a copy is in
  the plan folder as `jax-fitter-design.md`).

### Twenty-first round (lead, 2026-11-23, 12:32): precision and the port's acceptance

- **Precision.** Asked "Double precision on the GPU too? I recommend yes. The single-precision path and
  its extra check are then dropped", the lead answered "sure". So the fitter runs in float64 on the CPU
  and on the GPU. Float64 is turned on for each public call of the fitter, not for the whole process
  (the decoder in the same process stays in float32). The single-precision GPU path of section 5 and
  check 6 of section 6 (single against double precision) are dropped. The GPU check is GPU float64
  against CPU float64.
- **The acceptance rule of the port.** Asked whether the JAX fitter must match the stored NumPy outputs
  within 1e-9 relative, with the statuses and counts exactly equal (bit-for-bit equality is not
  possible), the lead answered "ok".
- **A slower fit on the CPU.** The lead: "depends on how much time things take". This is not decided.
  Before the port starts, the time of whole workloads is measured with the present fitter and with a
  JAX prototype, and the table goes to the lead.
- (**call**) The simulation engine lands before the port, so that the same seeds give the same data
  before and after it and the engine checks the port. The author told the lead this order and its
  reason, and the lead did not object.

### Twenty-second round (lead, 2026-11-23, 13:00 and 13:02): a weak penalty on simple coefficients

At 13:00 the author listed five options for the reward coefficient of the rate model (item a of the open
list below): the event-based support rule of the plan; a lower minimum; counting visits with at-risk time;
a weak penalty on the coefficient; one coefficient for several sessions. The lead: "I think weak penalty".

At 13:02 the author asked: "For consistency, should the same weak penalty apply to every simple tested
coefficient in every model? I recommend yes, with a simulation check for each family before use." The
lead: "yes".

- The reward coefficient of the rate model has a weak penalty. Item a of the open list is decided.
- Every simple tested coefficient (a 0/1 step, a slope, a level of a tested category) in every model has
  the same weak penalty. Each family's simulation (Phase 2b's engine, the family's scenario list) checks
  the penalty before the family's fit is used.
- This replaces two sentences of section 4.3: "Terms with no smooth ... carry no penalty and must pass the
  support rule below", and the support rule "decided from the data before the fit (the term varies in too
  few units with events on each side)".

The author's calls that go with these decisions (**call**, each):

- Baselines are not penalized: the intercept, the arm effects and the other nuisance categories.
  Greenland and Mansournia (2015, Statistics in Medicine 34:3133, doi 10.1002/sim.6537) advise a weak
  penalty on coefficients as a default for sparse data, caution against penalizing intercepts, and advise
  a sensitivity analysis over the strength.
- A guard stays, and it reads the design only, never the outcome. A term is absent in a session when its
  column does not vary enough there: fewer than 5 units with a row on each side (a level: with a row of the
  level; a continuous slope: above and below the median of its rows; a constant column fails). No rule
  reads the event counts before a fit. What happens to a failing term is unchanged (inactive; a level's
  rows dropped), and the 20 % rule stays, on these counts. Under the grouped softmax every candidate row
  belongs to an event, so the counts there do not change.
- A penalized coefficient always has a finite maximum, so a session with no event on one side of a
  penalized term is fitted with the term in. An unpenalized term in that case still has no finite maximum
  and is left out by the sixteenth round's rule.
- Each fit records how much the penalty mattered: the effective parameters of each penalized simple
  coefficient (near 1: the data decide; near 0: the penalty decides).
- The width is one declared number for all models: a Normal prior SD on the coefficient in the
  convention's reported unit (a 0/1 step; one pooled SD of a continuous covariate). The log penalty is
  `log(1 / SD^2)`. Its value is not chosen yet (open item g below).

Built: the penalty on `Slope` and `Category` and the support rule on the design, branch `weak-penalty`,
commit aa8f8425 (`docs/design/regression_fitter.md`, "Terms and their names" and "The support rule").

### Twenty-third round (lead, 2026-11-23, 12:34 and 13:21): the JAX port goes ahead without blocking Figure 4

At 12:34 the lead said, on one JAX fitter: "I think I would in general prefer one fitter, but if there's a
good reason not to, then that's fine". The author then measured whole workloads with the present fitter
and a JAX prototype, under load (absolute times rough;
`.claude/docs/plans/regression-convention/jax-fitter-timing.md`):

| Workload | NumPy, 32 processes | JAX CPU, 32 processes | JAX, one A100 |
|---|---|---|---|
| One whole fit of a rate session | 0.349 s (one core) | 0.727 s (one core) | 0.102 s |
| The real rate fit | 32 s | 1.4 min | 5.2 min |
| One penalty calibration | 6.3 min | 13.5 min | 60 min |
| One simulation sweep | 42 min | 1.5 h | 6.6 h |
| The content fit | under 1 min | under 1 min | under 1 min |
| The mPFC fit at its present size (extrapolated) | 3.1 h | 2.0 h | 15 min |

Fits that share one design, batched, took 0.8 ms each on the GPU against 311 ms in a NumPy loop.

The lead asked "what do you recommend?". The author recommended that the port go ahead without blocking
Figure 4, in this order:

1. the weak-penalty change in the present fitter (twenty-second round);
2. the simulation engine;
3. in parallel, on separate branches, the rate model and the JAX port;
4. the port replaces the solver, and every fit is rerun once to confirm equality.

The lead: "ok".

- Item d of the open list (the port's speed) is decided. The entry condition of the JAX fitter phase is
  met.
- (**call**) The acceptance rule of the port (twenty-first round) gets an absolute floor: a value `b`
  matches when the difference is at most `1e-11 + 1e-9 |b|`, with the statuses and counts exactly equal.
  The prototype had one coefficient near zero (about 4e-4) at a relative difference of 5.4e-9. The author
  told the lead, and the lead did not object.

### Twenty-fifth round (lead, 2026-11-23, 19:19 to 19:34): the reward term of the rate model

(Master's twenty-fourth round is the unit of the fit, 2026-11-24, below; this round is placed by its
date, before it. The content model's recency round, numbered twenty-fourth on the branch
`role-content-models`, is renumbered when that branch merges.)

What was shown to the lead: the realistic simulation of the weak penalty at prior SD 4, 2 and 1
(`.claude/research/2026-11-20-regression-prototypes/proto_rate_realistic/REPORT.md`: every session gets
a value, but the result on the reward coefficient depends on the SD more than on the data, because a
session expects only 2 to 6 SWR events after an omitted reward); a toy calculation with two Poisson counts
that gives the same pattern (`proto_rate_realistic/weak_penalty_toy/`); and the literature's advice to fit
one model over sparse units (one reward coefficient per animal, shared by its sessions, unpenalized).
(The author's reading, not the lead's words: the lead heard that advice and set it aside for consistency
across the model families.)

The lead's words, in order:

- About one coefficient per animal (19:19): "I just feel like once we start this, then we have to do it
  for every type of model".
- About an unpenalized per-session fit (19:27): "it's not consistent to have no penalty".
- (19:34): "lets just pick a penalty that's reasonable".

So:

- Decided by the lead: the reward coefficient keeps the weak penalty in each session, as every simple
  tested coefficient does, at a reasonable declared value. There is no unpenalized fit.
- (The author's reading at the time, not the lead's decision; superseded by the twenty-fourth round)
  There is no per-animal coefficient as the method. The lead's words at 19:19 ("I just feel like once we
  start this, then we have to do it for every type of model") are a hesitation, not a refusal, and the
  twenty-fourth round did exactly that for every behavior model: one fit for each recording day and for
  each animal.
- (**call**, the author's, not values the lead chose) The values are declared per model, because the unit
  of the coefficient differs: prior SD 2 on the rate model's log rate ratio (a rate ratio between about
  1/50 and 50 is plausible, `exp(1.96 x 2)` is about 50), with SD 1 and 4 reported beside it
  (`visit_rate.WEAK_PRIOR_SD`, `WEAK_PRIOR_SD_SENSITIVITY`); prior SD 8 on the content model's
  coefficients, with 4 and 16 beside it (the author's pick from the content validation, told to the lead
  on 2026-11-23). No single SD was decided for every family; the other families' values are not picked
  yet.
- (The author's text, not the lead's) A family's simulation measures the error that the paper states
  beside the number (bias, spread, coverage and false-positive rate at the declared SD and at the pair
  beside it); it does not veto a term.
- The per-animal fit built for the comparison (`visit_rate.fit_animal_rewards`, `reward_level="animal"`)
  is a check, not the paper's method; whether it stays in the code is a question for the lead.

Later rounds change this one. The twenty-fourth round (2026-11-24) fits the behavior models for each
recording day and for each animal (design C), so the reward coefficient is no longer fitted in each
session alone; the weak penalty stays on it. The twenty-sixth round (2026-11-25) reads the reward effect
from an arrival curve for each outcome over two windows and keeps the one coefficient as a secondary value.
The per-animal check was removed on 2026-11-26, when the branch took master, by task 12 of
`.claude/docs/plans/regression-convention/phase-4b-unit-of-fit.md`: a fit at `fit_by="animal"` with the
shared curves replaces it. The lead was not asked the question of the bullet above again.

### Open after the twenty-third round

Each of these is open unless it says it is decided.

a. *(Decided in the twenty-second round: a weak penalty on the reward coefficient, and on every simple
   tested coefficient; the support rule reads the design.)* The evidence that led to it: the planned
   event-based rule (5 visits with an event on each side) failed in 86 % of the real SWR sessions and 42 %
   of the No SWR sessions (counts of the input tables, not results), over the plan's 20 % limit
   (`MAX_FAILING_SHARE`).
b. **The form of the rate model.** Diagnostics on the real at-risk bins (pooled; not results): omission
   visits have a median length of 0.9 s against 5.4 s after reward. SWR events almost never start in the
   first second after arrival after either outcome (none in 93 % of sessions), so the first window of Q1
   is empty for SWR events. The reward ratio at the same time since arrival is about 2 to 3 from 1 to
   6 s and about 1 beyond 10 s, so it is not constant. Visits longer than 10 s are 705 of 44,435; ten
   sessions hold 72 % of that time, and that time holds 56 % of the SWR events after an omission. The two
   clocks add from 2 s on, but not between 1 and 2 s. A simulation with planted curves shaped like the
   real ones is running (scratch `proto_rate_realistic/`). The reward term itself is decided (twenty-fifth
   round). Open: the reference window of Q1; the flexibility (penalty) of the leaving clock, and of the
   other clocks; the reward effect as one coefficient or as a contrast of two arrival curves (whether it
   may change with time since arrival); whether the model covers a declared time range.
c. *(Decided in the twenty-first round: float64 on CPU and GPU.)*
d. *(Decided in the twenty-third round: the port goes ahead in the order given there; the acceptance rule
   has an absolute floor.)*
e. **The penalties of the mPFC models and of the history kernels.** Whether a penalty chosen over the
   many units of one session, or the declared-penalty rule, applies there. Decided at those phases.
f. **A support rule per grid point** for curves averaged over sessions (an idea of the author; not built,
   not decided). Simulation: few sessions have bins near the 20 s end of the arrival clock, and the
   average curve is far off there.
g. *(Decided in the twenty-fifth round: the reward coefficient per session with the weak penalty. The
   values are the author's, per model: SD 2 on the rate model's log rate ratio, with 1 and 4 beside it;
   SD 8 on the content model's coefficients, with 4 and 16 beside it.)* Each family reports
   a sensitivity analysis over the strength (Greenland and Mansournia 2015).
   (Superseded in part: the twenty-fourth round fits per day and per animal, not per session, with the
   weak penalty kept; the twenty-sixth round reads the reward effect from an arrival curve for each
   outcome over two windows, the one coefficient kept as a secondary value.)

### Twenty-fourth round (lead, 2026-11-24, 02:09 to 09:55): the unit of the fit

This round's number is settled when the unmerged branches merge: they hold rounds of their own with the
same numbers. The record is `docs/design/2026-11-24-unit-of-fit.md` (sections 2 and 11).

The lead's words, 2026-11-24:

- 02:09 and 02:10: "I'd like to be consistent with what we do with the behavior models" / "By behavior
  models, I mean rate, content, etc"
- 02:14: "Does not include the mpfc models for now."
- 03:13: "I'm inclined to do C and then attempt D as we are doing C, but we will see if it is
  prohibitive"
- 03:33: "Maybe we make it easy in the code to do day or animal grouping"
- 09:47: "I suspect with will end up doing design C for all. And then running both the day and animal
  model and compare"
- 09:48: "I think there will be a decent amount of variation from day to day (or even session to
  session) from learning, motivation, etc."
- 09:49: "So maybe design C with day and animal model and then design D later to check."
- 09:55: "I did say maybe, lets just make this the decision now and we can change if the evidence is
  against it"

What the lead decided (only what the quoted words carry):

- **Design C, with the day and the animal model, both run and compared**: one fit for each recording day
  and one for each animal (`analysis/replay_models/fit_by.py`, `BEHAVIOR_FIT_BY = ("day", "animal")`).
  In design C each session in the unit keeps a baseline of its own (the intercept of a rate model, the
  arm effects of a content model) and the tested terms are shared. One fit for each session stays in the
  code for the validation and for comparison.
- Design D (random session effects, their strengths selected by REML) later, as a check on the same
  mechanism.
- The decision changes if the evidence is against it.
- Consistent across the behavior models ("rate, content, etc"); not the mPFC models for now.

What the author decided or read (labeled; not the lead's words):

- **Fixed, declared penalties: the author's decision**, on "Should we keep the fixed penalty?" / "Ok decide
  then" (03:18, 03:20), from the evidence in the design note, section 11.2. The weak penalty on every
  simple tested coefficient is the lead's decision of the twenty-second round.
- **The curves are among the terms shared inside a unit: the author's reading.** The author's message to
  the lead just before 09:55 said that design C shares the curves inside a unit, that the lead had
  declined "one curve for each animal" on 2026-11-21 for the per-session design, and that the author read
  the lead's messages of that morning as changing that for design C. The lead's next message was "I did
  say maybe, lets just make this the decision now and we can change if the evidence is against it". The
  rate model's clock curves are the point that the running scenario may change (the design note, 11.4).
- **Which models: the author's reading.** That "etc" means the choice, stay-or-leave and ripple-power
  models is the author's reading, stated to the lead several times and not contradicted.

This round changes two earlier rounds. The fourteenth round kept one fit for each session and declined
one curve for each animal (2026-11-21, for the per-session design); that the shared curves of design C
replace that decline is the author's reading above. The seventeenth round kept one fit
for each session for every model (2026-11-22).

The evidence (simulation numbers; the design note, section 11.2): with nothing planted, the content
model's per-session estimates are off by up to 0.47 (Q3a -0.22 / -0.15, Q7b -0.34 / -0.47, Q5b-var
+0.26 / +0.25, No SWR / SWR), and one fit per animal takes these to 0.03 or less; one fit per day takes
them to between 0.03 and 0.18. The rate model's reward coefficient goes from +0.11 / +0.09 per session to
+0.011 / +0.006 per animal. Per-session arm effects gave a smaller largest bias than shared ones (0.035
against 0.062), and penalties selected by REML over-smoothed the curves that the tests read at their ends.
The reports are `.claude/research/2026-11-20-regression-prototypes/design_d_trial/rate/REPORT.md`,
`.../design_d_trial/content/REPORT.md` and `.../content_validation/unit_of_fit/tables.md`.

Open (the design note, section 11.4): whether the rate model's clock curves are shared inside a unit or
are per session; the rate model at the day unit and the declared penalties of its clocks at a larger unit;
which target the paper states when sessions differ; which unit the headline values come from.

### Twenty-sixth round (lead, 2026-11-25, 14:54 and 15:07 EST): the form of the rate model's reward effect

(The twenty-fifth round, the reward coefficient's weak penalty, 2026-11-23, is above, before the
twenty-fourth by its date.)

What was shown to the lead: the rate follow-up's later-onset scenario
(`.claude/research/2026-11-20-regression-prototypes/design_d_trial/rate/followup/REPORT.md`, section 3;
simulation numbers, `mgcv::bam`, the animal unit). With a later onset after an omission and no reward
ratio planted, the one reward coefficient came out at 0.20 (SWR) and 0.17 (No SWR) and excluded zero in
every data set; an arrival curve for each outcome read over a window returned the planted shape (4 to
10 s: -0.057 against a planted -0.053 for SWR; 2 to 4 s: 0.27 / 0.22 against 0.23 / 0.155). Under a
constant ratio every window recovered it. The windows' costs, with nothing planted: intervals excluding
zero 2 to 4 s 0 % / 5 %, 4 to 10 s 15 % / 5 %, 1 to 2 s 0 % / 20 % (and biased +0.05 / +0.06); half-widths
against the one coefficient 15 to 45 % wider at 2 to 4 s, two to three times at 4 to 10 s. The window
contrast is unusable at the day unit (+0.10 to +0.5 with nothing planted).

The lead's words, 2026-11-25:

- 14:54: "okay so no interaction of outcome of leaving"
- 15:07: "agree", to the author's recommendation below.

What the lead decided:

- **The leaving curve is shared by both outcomes**: no interaction of the outcome with the time until
  leaving, in any form of the model.
- **The reward effect is read from an arrival curve for each outcome** (the author's recommendation,
  agreed): `outcome + session + s(arrival, by = outcome) + s(leaving) + offset(log exposure)`, Poisson,
  fitted per animal (and per day, under the twenty-fourth round) and per kind, the two arrival curves
  penalized each on its own (the form the follow-up simulated, its form O), not the shared curve plus a
  difference curve of `visit_rate.REWARD_FORMS`' `"time_varying"`.
- **Two declared tests**, each the outcome term plus the mean over the window of the rewarded arrival
  curve minus the omitted one (log rate ratio):
  - Q2, the reward effect on the rate once events are going: the window 4 to 10 s after arrival;
  - a new question, the reward effect on when events start: the window 2 to 4 s after arrival.
- **The difference curve** (rewarded minus omitted, over the time since arrival, with its band) is shown,
  not tested.
- **The one reward coefficient stays as a secondary value**, for comparison with other studies, described
  as mixing the timing with the rate.

These windows are declared before any real arrival curve by outcome is looked at; the real curves are a
check on this choice, not a way to choose it.

**What the window tests compare (added 2026-11-25, with the lead's "ok").** A visit after an omission
is short (median 0.9 s, against 5.4 s after a reward; item b above), so the omission side of each
window comes only from the omission visits that last into it. Each test is therefore worded as the rate
after a reward against after an omission, at the same time since arrival and before leaving, among
visits that last into the window; the visits that linger after an omission may differ from the rest,
and the paper says so. The support (counts of the input tables, not results; model A's at-risk bins of
the 242 sessions, by each bin's last outcome; no event counted;
`.claude/research/2026-11-20-regression-prototypes/design_d_trial/rate/omission_support/`): per animal,
146 to 429 omission visits have bins at 4 to 10 s (208 to 620 at-risk seconds, against 3,321 to 4,669
rewarded visits), and 375 to 955 at 2 to 4 s; per session a median of 4 at 4 to 10 s (10 % of sessions
2 or fewer), which is why the window tests are read per animal, not per day. An animal or kind whose
window lacks support under the support rule is reported as unsupported; the windows are not moved to
fit the data. What follows for the builder (in order): the form and the
window contrast in `analysis/replay_models/rate/visit_rate.py` at `fit_by="animal"`, with the declared
tests in `declared_tests.py`; the rate validation through the project's simulation engine, with more data
sets for the rates of excluding zero (the follow-up's 15 % for SWR at 4 to 10 s rests on few) and with a
declared penalty for each clock curve (about 15 effective parameters for each arrival curve, 10 or more
for the leaving curve; not yet run); then the real fits. Under this form Q1 is read from the arrival
curve after an omission.

This round changes the declared Q2, which was the one coefficient (`declared_tests.py`), and decides the
open "form of the reward effect" of `visit_rate.REWARD_FORMS` and of item b above (whether the reward
effect may change with time since arrival: it may, along the arrival clock only).

**Built (2026-11-26, branch `rate-reward-form`; the author's note, not the lead's words).** The form
is `visit_rate.REWARD_FORMS`' `"by_outcome"`, the default (`OMITTED_ARRIVAL_TERM`,
`REWARDED_ARRIVAL_TERM`: each a smooth of the arrival clock `by` the last outcome, with its own
constraint and penalty); `"time_varying"` is removed and `"coefficient"` stays for the secondary value
(`visit_rate.SECONDARY_ESTIMAND`, `"reward_coefficient"`). The two tests are `declared_tests` Q2
(`Q2_WINDOW_S`, 4 to 10 s) and `Q2-onset` (`ONSET_WINDOW_S`, 2 to 4 s; the id is the author's, after
the convention of `Q5b-var`), both `OutcomeWindowContrast`; Q1 reads the arrival curve after an
omission. The support rule of a window (the author's rule, `visit_rate.window_support`): for each
outcome and window a contrast reads, the unit's visits with a row there and their at-risk seconds;
supported when every read has at least `support.MIN_UNITS` (5) visits; the counts are recorded beside
each value, and a value without support is kept, marked and left out of the reduction. The difference
curve is `visit_rate_statistics.difference_curves`, with its bands.

## 10. Left to the builder

File and function names inside `analysis/regression/`; the tolerance values; the number of basis functions
per term, within the check of section 4.2; batch sizes; the storage layout of the new fits; the order of
tests. Each such choice is recorded where it is made.
