# The unit of the fit: session, day or animal, one setting for the behavior models

2026-11-24. Status: **proposal**. Nothing here is built. The implementation plan is
`.claude/docs/plans/regression-convention/phase-4b-unit-of-fit.md`. The convention this changes is
[2026-11-20-regression-convention-and-figure04.md](2026-11-20-regression-convention-and-figure04.md)
(the "design" below), sections 4.3 and 4.5 and the fourteenth and seventeenth rounds of its section 9.

How to read the labels:

- **Decided (lead)**: the lead's words, quoted, with the time.
- **Call**: the author's reading or choice. The lead can change it.
- **OPEN**: not decided. Each says what evidence will decide it.

Every number in this note is a simulation number or a count of the design, and says which. No number of
a fit of the real data is quoted.

## 1. Why

Today every behavior model is fitted to one recording session at a time. The estimates are then
averaged, session to recording day to animal. Simulations on the real rows, with the real regression
code, show that one session holds too little information for some of the tested terms. Each session's
estimate is biased, and the average keeps the bias.

The evidence (all simulation):

- **Rate model, reward coefficient (Q2).** With no effect planted, the per-session fit at the decided
  prior SD of 2 gives +0.102 (SWR) and +0.105 (No SWR), over 40 data sets on 40 real sessions. A check
  with one reward coefficient per animal, each session keeping its own baseline and clock curves, gives
  +0.027 (SWR) and +0.030 (No SWR), over 20 data sets. Source:
  `.claude/research/2026-11-20-regression-prototypes/rate_validation/REPORT.md`, sections 1 and 3.
- **Content model, noise matched to the real clustering of the shares.** With nothing planted, most
  tests are off by 0.2 to 0.3 (all 242 sessions, prior SD 8, No SWR / SWR): Q3a -0.22 / -0.15, Q5a
  -0.20 / -0.15, Q5b-var +0.25 / +0.21, Q7b -0.29 / -0.46. A session's error goes toward zero only at
  several hundred events: keeping sessions of 300 events or more brings Q3a, Q3b, Q5a and Q5b-var within
  0.05, but leaves 16 No SWR and 9 SWR sessions, and Q7b stays at -0.20 or worse at every threshold that
  keeps more than 8 sessions. Three changes inside the per-session fit did not remove the bias (arm
  effects with the weak penalty, no arm effects, smooths 6 log units stiffer); Q5b-var's bias was
  untouched by every change. Source: `.../content_validation/realistic_noise/REPORT.md`.
- **For scale (counts of the design, content cohort).** The median session has about 115 No SWR and 110
  SWR events. The median recording day has about 650 events of a kind; a day holds a median of 5
  sessions and at most 8. An animal holds 3,284 to 12,682 events of a kind. There are 45 recording days
  and 5 animals. Source: `.../content_validation/unit_of_fit/design_units.json`.

The literature read for this (abstracts only) says the same in general terms:

- One model over sparse units, not an average of per-unit estimates: Stijnen et al. 2010
  (doi 10.1002/sim.4040), Sweeting et al. 2004 (10.1002/sim.1761), Debray et al. 2013
  (10.1371/journal.pone.0060650).
- Many small strata, each with its own terms, bias a logit model: Pike et al. 1980 (10.1093/ije/9.1.89).
  This is the case of per-session baselines in the content model's softmax (section 4.2).
- A Poisson model with a term for each stratum is safe: Armstrong et al. 2014 (10.1186/1471-2288-14-122).
  This is the case of per-session intercepts in the rate model (section 4.1).
- Few top-level clusters (here, five animals) limit what a model with random effects can estimate:
  McNeish and Stapleton 2016 (10.1080/00273171.2016.1167008).

## 2. What the lead said

All on 2026-11-24, in order.

- 01:35: "So it seems like the major problem right now is per session power for rate and content models?"
- 02:09 and 02:10: "I'd like to be consistent with what we do with the behavior models" / "By behavior
  models, I mean rate, content, etc"
- 02:14: "Does not include the mpfc models for now."
- 03:13: "I'm inclined to do C and then attempt D as we are doing C, but we will see if it is
  prohibitive". As the options were put to the lead: C is one fit per animal with a baseline for each
  session; D is one model with random session effects whose penalties are estimated by REML, as in
  Simon Wood's `mgcv`.
- 03:18 and 03:20, on whether the smooth penalties stay fixed and declared under a larger unit or are
  estimated by REML: "Should we keep the fixed penalty?" / "Ok decide then". The author decides after the
  running simulations report (section 5).
- 03:31: "And grouping by day wouldn't work?"
- 03:33: "Maybe we make it easy in the code to do day or animal grouping"

Earlier decisions that still hold (2026-11-23, recorded in the design's section 9): no bias-reduced
(Firth) fit (eighteenth round); a regression model, not a stratified estimate (nineteenth round); the
weak penalty stays on every simple tested coefficient (twenty-second round); simulations always use the
real regression code (twentieth round).

This note changes one earlier rule. The design's fourteenth and seventeenth rounds (2026-11-21 and
2026-11-22) record that the fit stays one fit for each session, and the fourteenth round records that
one curve per animal for the smooth terms was declined then. The 2026-11-24 words above reopen the unit
of the fit. **Call**: they do not by themselves settle whether curves are shared inside a larger unit;
that is point 4.1 (a) below.

## 3. What the code does today

The session is the unit of the fit in these places (master 2e0ea0d0; the plan's inventory has every
place, the unmerged branches included).

- The loop. `analysis/regression/sessions.py:526` (`fit_sessions`) takes a mapping from
  `(animal, session_date, epoch)` (`SessionKey`, `:92`) to that session's rows and fits each alone
  (`:581-586`, `_fit_session` at `:440`). The design is built once on the pooled rows of every session
  (`terms.build_design`, `terms.py:985`): the knots, the constraints and the standardizations are shared.
  Only the coefficients are per session.
- The support unit. Inside a fit, `unit` is the replicate the support rule counts (a well visit for the
  rate model, an event for content). It is factorized on the rows given (`sessions.py:142`). The rate
  model's unit, `well_visit_id`, is a port experience number that starts at 0 in every session
  (`data_loaders/trials.py:595`, `visit_design.py:540`). It is unique only within a session. A fit of
  several sessions must therefore count the support unit as (session, visit), or visit 3 of one session
  and visit 3 of another become one replicate. The content model's unit, `event_id`, is a hash that is
  unique across sessions.
- The reference level. A category's reference level must have rows in every session
  (`terms.py:15-18`; the error at `sessions.py:334-341`).
- The record. `analysis/regression/record.py:103` (`FitRecord`): one row per (model, kind, session), the
  session in `animal`, `session_date`, `epoch` (`:180-182`).
- The penalties. Each smooth has one fixed log penalty, declared, the same in every session.
  `analysis/regression/calibration.py:103` (`log_penalty_for_edf`) sets it so that the median edf over
  sessions is a target.
- The shares over sessions. A term that fails the support rule in more than 20 % of the sessions leaves
  the model (`support.py:305`, `terms_failing_share`; `constants.py:110`, `MAX_FAILING_SHARE`). The rate
  model's third-clock rule reads the same share (`visit_rate.py:1043`).
- The reduction. `analysis/shared_statistics.py:1013` (`estimand_summary`): session to day to animal, a
  mean of means; the nested bootstrap (animals, then their sessions) with the t correction over animals
  (`bootstrap_interval`, `:621`); the animals with the estimate's sign; the day-level signed-rank test
  (`day_level_test`, `:978`).
- The simulation engine. `analysis/regression/simulation.py:668` (`_run_task`) draws a data set, fits it
  with `fit_sessions` and reduces it with `estimand_summary` (`_data_set_row`, `:617`).

## 4. The setting

One setting, three values (**call**, on the lead's 03:33 words):

| value | one fit for | the unit's key |
|---|---|---|
| `"session"` | each session (today) | `(animal, session_date, epoch)` |
| `"day"` | each recording day of an animal | `(animal, session_date)` |
| `"animal"` | each animal | `(animal,)` |

- **Name** (**call**): `fit_by`. The word "unit" is taken: the fitter's `unit` argument is the support
  replicate (a visit, an event). `fit_sessions(..., unit="well_visit_id", fit_by="animal")` reads without
  a clash.
- **One declaration** (**call**). The type and the key of each value are task-free. They live beside
  the session key, in `tables/session_keys.py` (new there: `FitBy`, `FIT_BY_COLUMNS`, `unit_key`); the
  fitter reads them (the fitter's package may import that foundation module, as `record.py` already
  imports `tables.io`). The value that every behavior model uses is declared once, in
  `analysis/replay_models/fit_by.py` (new: `BEHAVIOR_FIT_BY`). The rate, content, choice, stay-or-leave
  and ripple-power models read it. The mPFC models do not (decided, 02:14).
- **One code path.** `fit_sessions` takes `fit_by` and groups the sessions it is given into units. Each
  unit is one fit through the same steps as today's session (support, levels, refits, record). There is
  no second loop. At `fit_by="session"` every unit holds one session, and the records equal today's.
- **Overrides.** A model's settings (`VisitRateSettings`, `RoleContentSettings`) carry `fit_by`, with
  `BEHAVIOR_FIT_BY` as the default, so that a simulation can fit every unit on the same data sets. The
  record stores `fit_by`. The reported-values builders refuse a set of behavior fits with more than one
  value (**call**: this is how "consistent" is enforced).

### 4.1 What is shared and what stays per session

Inside a unit of several sessions, the tested terms are shared, and each session keeps a baseline. This
is design C of the 03:13 answer. A baseline is never tested and is never read by a declared contrast.

| model | per session inside the unit | shared inside the unit |
|---|---|---|
| rate (`visit_rate_A`, `visit_rate_D`) | the intercept (the session's level of rate) | the reward coefficient (A); the lag curve (D; described, not tested: **call**); the clock curves: **OPEN (a)** |
| content (`role_content`, `role_content_uncertainty`) | the arm effects: **OPEN (b)** | the role levels, the role slopes, the smooths |
| choice, stay or leave, ripple power (Phase 7, not built) | the model's baseline, declared with the model | its tested terms |

**OPEN (a), rate: are the clock curves shared inside the unit?** The model has two clocks that Q1 and
Q7a test (arrival and leaving) and, by default, a third (time until the end of the immobile stretch;
`VisitRateSettings.include_stretch_clock`). Two forms:

- *Shared curves.* The clock tests (Q1, Q7a) read the curves, so they gain from the unit only if the
  curves are shared. The risk: if sessions differ in their curves, a shared curve can push part of the
  difference into the reward coefficient, because the share of rewarded visits differs between sessions.
- *Per-session curves, shared reward coefficient.* This is what the +0.03 check did. Q1 and Q7a then stay
  per session; in that check they moved by at most 0.01 (simulation, rate report section 4).

What decides it: a scenario that plants curves that differ between sessions (none does yet; the plan adds
one), fitted in both forms, reporting Q1, Q2 and Q7a against both targets of section 6. Cost also
matters. A unit with per-session curves is a block design that a dense fit cannot hold for an animal
(section 8), so that form needs the profile path of the rate branch (the plan, "Order of work").

**OPEN (b), content: are the arm effects shared or per session?** Per-session arm effects in a softmax
are the case of many small strata, each with its own terms (Pike et al. 1980): about 115 events per
session for 8 free arm effects. Shared arm effects cannot follow sessions whose arm preferences differ.
A third form, random arm effects for each session, is design D (section 9). What decides it: the running
simulation `.../content_validation/unit_of_fit/` fits both forms (its "S" and "B") at a day and an
animal, with arm effects that differ between sessions (a Normal deviation of SD 0.3 per session and arm,
`driver.py`). Its results are not in. No number from it is quoted here.

**Call: a session without support for its baseline.** A session whose baseline fails the support rule
(fewer than `MIN_UNITS` = 5 visits or events of a kind) has its rows dropped from the unit's fit and
counted, as a level that fails is dropped today (`support.py:42-49`: merging into another level inflated
false positives in simulation). The scratch driver instead shared one set of arm effects among such
sessions. The plan builds the drop and reports how many sessions it removes.

## 5. Penalties under a larger unit

- A smooth's fixed log penalty was set so that the median edf over sessions is a target (6 for the rate
  clocks in the validation; `rate_validation/REPORT.md`, section 6). Under a larger unit the same penalty
  meets many times more data, so it is too weak: the curve takes more edf. Penalties are therefore set for
  the unit, never carried from the session fit.
- **OPEN (lead's 03:20 "Ok decide then", so the author decides after the runs): fixed or REML.**
  - *Fixed, recalibrated*: `log_penalty_for_edf` with the median taken over the units (5 animals, or 45
    days, per kind). Which target edf a unit should have is part of this question.
  - *REML*: the penalties selected from the data for each unit (`reml.select_penalties`, `reml.py:1883`).
    The design's thirteenth and fourteenth rounds record that a penalty chosen from the data made a
    session's curve a line, at about 140 events of a kind (simulation). A unit holds far more events, so
    that result does not carry over; it has to be measured again.
  - Evidence awaited: the running unit simulation fits each unit at the session penalties ("pS") and at
    penalties recalibrated on the unit ("pU", `unit_calibration.json`); the design D trials
    (`.../design_d_trial/rate/`, `.../design_d_trial/content/`) fit penalties by REML with `mgcv`. No
    results yet.
- The weak penalty on every simple tested coefficient stays fixed, at each model's declared prior SD
  (decided 2026-11-23). Under a larger unit it weighs less against the data, which is what a fixed prior
  should do. **Note**: the rate branch's animal check had the reward coefficient unpenalized, so the
  general mechanism's estimator is not the one that gave +0.03, and the check must be rerun.

## 6. What the estimate means when sessions differ

The present design estimates each session's effect and averages: day means of session values, animal
means of day values, then the mean over animals (`estimand_summary`). Its target is the plain mean of
session effects, reduced the same way.

One fit over a unit estimates a single effect for the unit. When sessions differ, that effect is close to
an information-weighted average of the session effects: sessions with more events, or with a better
balance of the two sides of a term, weigh more. Strictly, it is the projection of the truth onto the
unit's model (the fit on unlimited data).

- The two targets differ when sessions differ and their information differs. They are the same when the
  effect is the same in every session.
- **Every simulation reports both targets** for every estimand and unit (**call**): (1) the present
  target, the session -> day -> animal mean of the sessions' planted values; (2) the unit's own target,
  the unit model fitted by the real fitter to the expected outcome (no noise), reduced over units as the
  estimates are. The fitter accepts a response that is not a whole number (`likelihoods.py:420` checks
  only that it is not negative), so the expected counts or shares can be fitted as they are.
- **Call**: in that target fit the weak penalty is off. The target is a property of the truth, not of
  the estimator. (The rate branch computes this target by hand, `shared_ratio_projection`; the engine
  replaces it with the real fit.)
- **OPEN**: which target the paper states. Evidence awaited: how far apart the two are under the
  scenarios that plant session differences (effects, arm effects, curves).

## 7. The reduction and the inference for each unit

| `fit_by` | the values | estimate | interval | beside it |
|---|---|---|---|---|
| `"session"` | one per session | session -> day -> animal, mean of means | nested bootstrap (animals, sessions), t over animals (`estimand_summary`) | animals with the sign; the day-level test |
| `"day"` | one per day | day -> animal, mean of means | nested bootstrap (animals, days), t over animals: `estimand_summary` with the day as its own session | animals with the sign; the day-level test |
| `"animal"` | one per animal | the mean over the animals | mean +/- t(0.975, 4) SD / sqrt(5) | animals with the sign; no day-level test |

- For the animal unit the function exists on the rate branch: `shared_statistics.animal_mean_interval`
  (branch `visit-rate-validation`). It is the interval `bootstrap_interval` gives a mean of animal values,
  without resampling. **Call**: it becomes the one animal-level reduction.
- With five animals the day-level test is not defined at the animal unit (`signed_rank_p` needs six
  non-zero differences). The design's rule that no claim rests on a p over five animals stays.
- The kind difference (SWR minus No SWR) is taken unit by unit, over the units with both kinds, then
  reduced as above (as today, by session).
- What is lost at each step up:
  - session to day: the view of sessions within a day (the first against the later sessions of a day),
    and the session-level values behind the figures that show them;
  - day to animal: the day-level test, and the view over learning (how an estimate changes over the days
    of an animal).

## 8. Checks that become standard with a larger unit

- **Leave one session out.** Each unit's value refitted with each of its sessions left out, in turn. The
  report gives, per unit, the largest change and the session that caused it. One session should not
  carry a unit.
- **What is behind each value.** Each unit's value carries its numbers of sessions, events and visits
  (or events, for content), and the sessions dropped for their baseline (section 4.1), by cause.
- **Cost and memory** (counts of the design and a timing; not results). The rate model has about 40,000
  bins per session (1.6 million SWR bins over 40 sessions, `rate_validation/REPORT.md`, section 6). An
  animal of 58 sessions is then about 2.3 million rows (the author's arithmetic). Its clocks have 43, 33
  and 18 basis columns (`CLOCK_BASES`, measured with `PSplineBasis.from_spacing`). With shared curves an
  animal's design has about 150 columns. With per-session curves it has about 5,300 (58 times an intercept and the curves' 91 constrained
  columns), and a dense design
  of that size does not fit in memory: on one A100 the present fitter fits 0.5 million rows by 1,100
  columns in 6.6 s and runs out of memory at 2 million by 1,100, because it holds about three copies of
  the design (`.../design_d_trial/gpu_size/run.log`: peak 13.4 GB for a 4.4 GB design). A content
  animal is at most about 114,000 rows (12,682 events of nine arms) by a few hundred columns; it fits
  either way.

## 9. Design D, later, on the same mechanism

Design D is one model per unit with random session effects, their strengths estimated by REML. It is a
later step (03:13: "attempt D as we are doing C"). The mechanism of section 4 is built so that D adds
to it and does not replace it:

- A per-session baseline (section 4.1) is a block of columns, one copy per session. In D the block
  carries a penalty, the identity (a Normal random intercept per session), whose strength is estimated
  by REML within the unit. Per-session curves become a factor smooth: one curve per session with one
  shared penalty, as `mgcv` fits them. The fitter already has penalty blocks and REML selection
  (`reml.PenaltyBlock`, `reml.select_penalties`).
- An animal's rate model with session curves does not fit as one dense design (section 8). D needs a
  Gram matrix accumulated over sessions (or chunks of rows) and the block structure of the per-session
  columns, so that one animal fits on one GPU. That is a change to the fitter's Newton step, outside this
  phase.
- With five animals, D's animal-level variance is poorly estimated (McNeish and Stapleton 2016). The unit
  of D is the animal or the day; animals stay the top level of the inference (section 7).
- Evidence awaited: the design D trials (`.../design_d_trial/rate/`, `.../design_d_trial/content/`). No
  results yet.

## 10. What is not decided

1. Rate: clock curves shared inside the unit, or per session (section 4.1 a).
2. Content: arm effects shared or per session (section 4.1 b).
3. Penalties under a larger unit: fixed and recalibrated, or REML; and the target edf of a unit
   (section 5).
4. The declared value of `fit_by`: day or animal (the lead's 03:31 question and 03:33 words leave both).
5. Which target the paper states when sessions differ (section 6).
6. Whether design D follows, and for which models (section 9).

The plan lists these as questions for the lead, each with the evidence that is still awaited.
