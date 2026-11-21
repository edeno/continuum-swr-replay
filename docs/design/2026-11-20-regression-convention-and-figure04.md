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

## 10. Left to the builder

File and function names inside `analysis/regression/`; the tolerance values; the number of basis functions
per term, within the check of section 4.2; batch sizes; the storage layout of the new fits; the order of
tests. Each such choice is recorded where it is made.
