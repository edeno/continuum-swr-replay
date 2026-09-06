# Model design history

The design-decision narrative for the replay-content model ladders, moved here
from the `analysis/replay/model_specs.py` module docstring so that module reads
as the terse `ModelSpec` catalog it is. This is the scientific audit trail —
including reversed reasoning and retracted numbers — and is preserved verbatim.
The machine-readable catalog (which predictors each rung carries) lives in
`model_specs.py`; this file is why.

The notes below concern the conditional-Poisson **arm and patch** hierarchies.
Their defining constraint: only arm-varying predictors are identifiable —
event-level constants (time since stop start, `is_swr`, etc.) cancel in the
within-event normalization and appear only as interactions with arm-varying
predictors. That rule does NOT apply to `MODEL_PER_STOP_RATE` (ordinary Poisson,
carries an intercept) or `MODEL_CHOICE`; see their own sections in the catalog.

## Design decisions (exploratory analysis on 7 j16 sessions)

- **Arm fixed effects** (`is_candidate_arm_0` … `is_candidate_arm_7`, 8 dummies
  with arm 8 as reference) enter as a baseline model `M0a` and are carried by
  every richer model. They absorb each arm's static replay occupancy so
  `is_current_arm` / `is_prev_arm` / `is_last_rewarded_arm` measure bias *above*
  an arm's own baseline rather than being inflated by it — important in
  camp-heavy sessions where the animal sits at one well and `is_current_arm`
  then concentrates on a single arm. See `ARM_FE_PREDICTORS`.

- **Reward history uses `reward_fraction_candidate_arm_last10_visits`** (rewarded
  fraction of the last 10 visits to an arm) at M3/M4 rather than
  `cumulative_reward` (a raw reward *count*). The count conflates value with
  visit frequency — a frequently-visited arm scores high even at a mediocre
  reward fraction — so its M3 coefficient partly re-encodes spatial recency.
  Once any trial has completed, a never-visited leaf is encoded as 0.0 rather
  than NaN; M4 adds `is_candidate_arm_never_visited` so that right-censored
  novelty is distinguishable from an observed zero reward rate. Events before
  the first completed trial remain unknown and are removed by complete-case
  selection.

- **M4 uses an event-local recency spline** rather than a session-wide
  transformation. At each event, the mean recency over the *visited* reward-site
  arms is subtracted from every leaf-arm recency; never-visited leaves are
  excluded from that mean because their `n_completed + 1` value is a censoring
  rule, not a measured recency. `is_candidate_arm_never_visited` carries that
  state separately and trunk arms remain structural zero. Appending future
  events cannot change an earlier design row, and the coefficient represents
  relative recency within the event above the arm-FE baseline.

- **Reward memory has NO rung in the per-session arm ladder.** Neither form of
  the last-rewarded indicator is usable here: the raw `is_last_rewarded_arm`
  correlates 0.919 with `is_current_arm` (it marks the same arm on 93.2% of
  location-resolved events), and the `is_last_rewarded_noncurrent_arm` residual
  has support on only 6.1% of events and is perfectly separated in 99 of 208
  sessions. The pooled SWR-contrast rungs (C5_lastrew, CS3_lastrew,
  CS5b_valuevsprevgoal) are still defined in the catalog, but pooling did not
  rescue them either — see "Correction (2026-09-06)" below.

  History, because the reasoning reversed twice and the intermediate numbers are
  still quoted elsewhere. The "elsewhere" residual was introduced to break a
  ~98% overlap between `is_last_rewarded_arm` and `is_current_arm`. On 2026-08-17
  that overlap was measured at 33.7% and the residual at 58.2% of events, which
  read as the collinearity having dissolved. Both figures were artifacts: they
  came from tables where an unresolved location made `is_current_arm` 0 rather
  than NaN, so every in-transit event silently counted the last-rewarded arm as
  "elsewhere". Off-well events are 9.1% of the corpus and carry
  `is_last_rewarded_arm` on 93.4% of themselves. Once they are NaN, the overlap
  is 93.2% and the residual 6.1% — the original ~98% regime, essentially intact.
  **Do not cite the 33.7% or 58.2% figures.**

- **M3 is the only rung carrying `reward_fraction_candidate_arm_last10_visits`
  without `is_candidate_arm_never_visited`.** A never-visited leaf is encoded
  `0.0` there, identical to "visited and never rewarded", and M4 adds the
  disambiguator. So the M3→M4 step is not a pure novelty test: part of it is M4
  cleaning up an encoding M3 was already using ambiguously. The nesting is
  deliberate and the step is a valid likelihood comparison, but do not report it
  as "replay tracks novelty" on its own — cite `is_candidate_arm_never_visited`'s
  own coefficient for that.

- **`is_next_arm` excluded:** the alternation structure makes
  `prev_arm == next_arm` on **84.5%** of the events where both are defined
  (measured 2026-08-18; an earlier ">90%" here was never re-derived). They
  separate on 15.5% of events, but **98.4% of those are also patch switches**,
  so the separating population is the same event class `is_prev_arm` plus the
  completed-trip switch flag already indexes — which is why the exclusion still
  stands on the corrected number.

- **`current_x_last_rewarded` excluded:** VIF 97-100 with both `is_current_arm`
  and `is_last_rewarded_arm`. Measured under the ~98% overlap regime, which the
  93.2% figure above shows still holds, so the exclusion stands.

- **`distance_to_candidate_arm_cm` and `is_same_patch` excluded:** proven
  uninformative (null when estimated independently; VIF 28 and 12 with
  `is_current_arm`). The spatial effect is binary "here vs not here," not a
  distance gradient.

- **SWR modulation is split across two rungs:** M5 adds the spatial interactions
  (`swr_x_is_current_arm`, `swr_x_is_prev_arm`), and M6 adds the reward-rate and
  relative-recency interactions. Superseded note: an elsewhere-reward-memory
  interaction was once a third term here and was removed, because
  `is_last_rewarded_noncurrent_arm` itself has support on only ~2% of events, so
  convergence and uncertainty require real-data re-validation after canonical
  table regeneration.

- **MUA interactions are a separate analysis** (split, not interaction) to avoid
  the same collinearity.

### Correction (2026-09-06): the whole last-rewarded family is refused

Earlier wording — the reward-memory bullet in this section, and the moved
`model_specs.py` comment below — said reward memory "lives only in the pooled
contrast rungs C5_lastrew / CS3_lastrew, where pooling supplies the variation a
per-session fit lacks". That is superseded: pooling across sessions did not
supply the missing variation, and the **entire last-rewarded family is refused
as unidentified**. `CS5b_valuevsprevgoal` — the previous-goal discriminant
against Gillespie et al. 2021 — fits only **132 sessions** under the
Dirichlet-multinomial with **94 flagged separated** (−0.316, 1/5 animals,
q = 0.294) and **133 sessions** on the shipped fraction response (+0.1840, 3/5,
q = 0.396). The rungs remain in the catalog so the refusal is reproducible, but
no coefficient from `C5_lastrew`, `CS3_lastrew`, `CS5b_valuevsprevgoal`,
`CP1_lastrew` or `CPS2_lastrew` is citable, and the previously quoted
−2.45 / −2.43 / −1.03 value-avoidance figures and the −0.79 SWR-by-previous-goal
interaction are withdrawn. Current wording: `paper/claims_table.md` row 19f.

## Retirements in the per-bin rate ladder (Family B)

- **The five `onset_lag1..10` dummy rungs are retired (2026-09-06):**
  `M3_hist`, `M4_hist_rew`, `M6_time_hist`, `M7_time_hist_rew` and
  `M9_rew_x_time`. Ten collinear lag dummies over a 150 ms window separate on
  the refractory bins — the first lags have essentially no events, the
  coefficients run to −inf — and a fit carrying a separated non-nuisance term
  has its whole cross-validation voided. Measured on the 2026-09-06 refit of
  218 sessions, `cv_bits_per_event` was **NaN in 99.08 % of the non-local fits**
  of each of the five (99.08 % non-local, 99.54 % non-local-no-SWR, 99.54 %
  non-local-with-SWR, 100.00 % detected SWR; 99.54 % pooled over the four
  populations). A rung that can never be scored out of sample cannot take part
  in the ladder's held-out comparison, which is one of the two things the
  ladder is read on.

  The raised-cosine variants `M3r_histcos` and `M7r_time_histcos_rew` span the
  same 150 ms window with four smooth, non-separating parameters and **are**
  scored (61.9 % / 63.8 % NaN on the non-local population — high, but not
  categorical). They were built for exactly this reason and now carry the
  self-history block alone. `M7r_time_histcos_rew` becomes the ladder's full
  additive rung and is what the summarizer, the figures and the refit audit
  read.

  The retirement also removed `M7r`'s natural null: with `M6_time_hist` gone it
  parented straight onto `M5_time`, so "the reward step" was the reward block
  and the self-history block together. **`M6r_time_histcos`** (2026-09-06)
  restores the missing rung — `M5_time` + the raised-cosine self-history block —
  so the reward-given-time-and-self-history step `M7r − M6r` exists again.

  The reward × time interaction survives, re-parented onto the raised-cosine
  full model as **`M9r_rew_x_time`** (`M7r` + `reward_history_rcos_0:bs(tc)`).
  Its specification is otherwise unchanged.

  Consequence for the per-lag self-history kernel: it is no longer read off ten
  `coef_onset_lag{j}` values but **reconstructed** from the four
  `coef_hist_rcos_k` per session
  (`per_bin_design.reconstruct_self_history_kernels`), which projects through the
  **un-normalized** raised-cosine basis — `hist_rcos_k` is a plain projection of
  the lag counts, unlike the reward block's bump-weighted fractions. The
  reported kernel is therefore smooth: it cannot represent a lag-specific spike.
  Measured against the retired dummies on the same sessions, that costs less
  than it sounds: **both** parameterizations read RR ≈ 0 at lags 1–2 in all
  four populations, and the smooth kernel is shallower only from lag 3 on,
  in 4/4 populations (non-local 0.31 vs 0.22, non-local with SWR 0.15 vs 0.00,
  non-local without SWR 0.02 vs 0.00, detected SWR 0.01 vs 0.00).
  Refractory/bursty shape and its ordering across
  populations are what the reconstruction preserves.

## Covariate-harmonization refit classification (2026-08)

Kept from the model catalog, which now describes only the models as they are
fitted today. Under the covariate-harmonization pass each spec was classified as
either CHANGED (covariates moved → coefficients move → true refit and report) or
parity-only (covariate-unchanged → a refit should reproduce the design columns;
checked by `_verify_harmonization_parity`).

| family | CHANGED (report old→new) | parity-only |
|---|---|---|
| MU uncertainty | MU_uncertainty (**done**) | — |
| SWR power | `ripple_power_z` model (canonical `rpe_postoutcome` and `reward_belief_mean_chosen_arm_preoutcome`) | — |
| Content arm | M3, M4, M5, M6, Mv | M0, M0a, M1, M2 |
| Content patch | P2, P3, Pv (patch reward rate replaces count) | P0–P1 |
| ContFrag content | arm M3–M4 | arm M0–M2, patch P0–P2 |
| Choice | C_behavior, C_full, C_full_swr_split | C_replay_only |
| Per-bin rate (B) | — covariates unchanged; **verify** the table regen / TMAX fix did not move onset counts, else refit | all specs |
| Per-stop rate (A) | — | R0, R1 |

Changed drivers: candidate-arm reward fraction (5→10 visits), the candidate-arm
task-probability/reward-belief block, post-outcome RPE, and the choice geometry
control now named `candidate_arm_distance_rank`.

Two standing qualifications on that table:

- **Never a parity pass (2026-08-20).** The key-aligned gate reproduced every
  declared unchanged column on shared rows but failed coverage, because a
  transient rebuild error omitted three Replay sessions (218 → 215). No row
  there should be treated as verified without regenerating those sessions.
- **Superseded for the rate families (2026-09-04).** Both rate ladders were
  respecified onto the reward-history raised-cosine basis and refit, so the rate
  rows above describe the pre-basis ladder: `R1` is a different specification
  from the one classified parity-only, and the per-bin reward rungs changed with
  it. The old→new comparison is in
  [`../audits/2026-09-04-rate-refit-comparison.md`](../audits/2026-09-04-rate-refit-comparison.md).

## Primary arm-level hierarchy: arm fixed effects, the retired reward-memory rung, SWR splits (moved 2026-09-06)

Superseded by § "Design decisions (exploratory analysis on 7 j16 sessions)" above;
kept as the verbatim `model_specs.py` comment as of its removal on 2026-09-06.
(References such as "above", "below" and "the module docstring" inside the
moved blocks in this file are to `model_specs.py`, not to this document.)

Moved verbatim from `model_specs.py`:

```
── Primary arm-level hierarchy ──

M0a (arm fixed effects) enters as a baseline below M1, and every
model from M1 on carries the 8 arm-FE dummies so the spatial /
reward predictors are estimated above each arm's static occupancy.

CONVERGENCE / VIF NOTE: the pre-arm-FE hierarchy was tuned for
max VIF < 3 through M4 and converged 7/7 on the j16 pilot. Adding
arm fixed effects deliberately introduces competition between
``is_current`` and the dummy of whichever arm the animal occupies
— VIF will rise (especially in camp-heavy sessions) and a session
where the animal never leaves one well can become rank-deficient
(is_current ≡ that arm's dummy) and fail to converge. The
``n_events < 2 * n_predictors`` gate in ``fit_conditional_poisson``
also drops more small sessions now (M6 needs ≥26 events).

There is no reward-memory rung. There was one -- M5 carried
``is_last_rewarded_elsewhere`` and the SWR rungs sat above it at M6
and M7 -- and it was removed on 2026-08-18, with the SWR rungs
renumbered down into M5 and M6.

The predictor is not identifiable per-session, and it does not measure
what its name says. It is an **omission-only** indicator: it cannot fire
on a rewarded stop, because if the animal was just rewarded here then
the last-rewarded arm is here and the residual is zero. It fires on 0.0%
of rewarded at-well events and 67.4% of omission ones, and every event
where it fires is an omission. So its support is capped by the ~10%
omission rate, and any coefficient answers "within omission stops, is
the previously-rewarded arm over- or under-represented" rather than
anything about reward memory in general.

On top of that it is not estimable per-session. It fires on 6.1% of
events, and in 99 of 208 sessions no replay mass ever lands on the arm
it marks, which is perfect separation: the coefficient runs to -inf
and the standard error explodes (median -3.503, SE > 10 in 49% of
sessions). Which sessions separate tracks their omission count (median
14 omission events, against 22 for the identified ones). Its SWR interaction was worse -- constant within every
event in 58 sessions -- and because the ladder freezes one
complete-case cohort over the union of all rungs, the refusals
propagated to every rung including M0, fitting 145 of 215 sessions
where 215 of 218 had fitted before.

The raw ``is_last_rewarded`` cannot take its place: it correlates
0.919 with ``is_current``, which is the near-collinearity the
"elsewhere" residual existed to remove. Reward memory therefore lives
only in the pooled contrast rungs C5_lastrew / CS3_lastrew, where
pooling across sessions supplies the within-event variation that a
per-session conditional fit does not have.

SWR interactions enter at M5 / M6. M5 carries the *spatial* × SWR
interactions (``swr_x_is_current_arm``, ``swr_x_is_prev_arm``) — does the local/previous
bias differ for SWR-overlapping events? M6 adds the *value* × SWR
interactions (``swr_x_reward_fraction_candidate_arm_last10_visits`` and the
centered-recency spline interactions) — does the reward structure of replay
differ on SWR events? Splitting M5 from M6 lets the nested LR cleanly
separate "spatial is SWR-modulated" from "reward is SWR-modulated".

An SWR split (fit M0–M4 separately on is_swr=True and is_swr=False
event subsets) is the unrestricted complement to these interaction
terms — same pattern documented for MUA below.

MUA event-type effects are assessed via split analysis (fit M0–M4
separately on MUA-only and non-MUA events) rather than interaction
terms, because mua_x_current ≈ mua_x_last_rewarded (r = 0.98,
VIF > 70).
```

The last-rewarded paragraphs in the block above are superseded — see
§ "Correction (2026-09-06): the whole last-rewarded family is refused" in
*Design decisions*.

## Visit-recency spline: masking, knot placement, raw-vs-log1p (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
Natural-cubic-spline basis for visit recency, on the raw trial-count scale.

Recency is deterministic on the two arms that already have their own
indicators: exactly 1 on the current arm and exactly 2 on the previous arm, on
100.00% of location-resolved events. A basis evaluated there is a fixed
vector, hence exactly collinear with ``is_current`` + ``is_prev`` -- fitting it
unmasked is rank-deficient. The basis is therefore masked to the arms where
recency is not already determined (visited leaves that are neither current nor
previous, where recency is >= 3 by construction), and one basis column is
dropped as reference because a natural spline basis is a partition of unity
and would otherwise sum to the mask.

Knots are fixed globally rather than fitted per session: a per-session basis
would give each session its own units and make coefficients incomparable
across the medians the ladders aggregate over. 12 and 40 trials sit near the
tertiles of the identified support (13 and 36), and are the best-conditioned
of the placements tried.

Raw rather than log1p, by preference for knots that read directly in trials.
Measured trade-off: the two are indistinguishable on fit (median paired Δllf
5.6 on a ~22,700 session log-likelihood, log1p ahead in 13 of 30 sessions),
but raw is less well conditioned -- within-event demeaned condition number
4.12 against 2.74, and the third basis column has SD 0.032 against the first's
0.386 because recency is right-skewed (median 21 on the support, max 198). If
per-session instability in these columns becomes a problem, the log1p scale is
the first thing to try: only these two constants change.
```

## Patch ladder: the retired patch reward-memory rung (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
There is no patch reward-memory rung, for the reason the arm ladder
has none (see the M5 note above). ``is_last_rewarded_patch`` overlaps
``is_current_patch`` at r = 0.945, and the "elsewhere" residual that
was meant to break that overlap fires on 2.9% of events and is
perfectly separated in 77 of 184 sessions. At patch level it also
carries no signal to recover: mean ``patch_count_hard`` is 14.76 on
the marked patch against 14.81 elsewhere, where ``is_current_patch``
shows 17.16 against 13.63.

It cost more here than in the arm ladder. Carrying it in P2/P3 lost 31
sessions and its SWR interaction in P4 lost 120, leaving 95 of 215 --
the frozen-union cohort propagates each refusal to every rung. Removed
2026-08-18, P3 and P4 renumbered down into P2 and P3.
``MODEL_PATCH_SWITCH`` remains the dedicated test of the
previous-vs-current contrast.
```

## Event-source-restricted hierarchies: the deferred cross-modality rung (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
A cross-modality rung (mua_x_* inside SWR-events, swr_x_* inside
MUA-events) is potentially identifiable when the cross-modality
flag varies, but should only be added once the ContFrag event_arm
tables are built and the cross-modality variability is verified.
```

## Post-switch patch model: the 7-session pilot result (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
Results on 7 j16 sessions:
- is_current_patch is STRONGER after switches (median +1.66 larger
  coefficient, 7/7, p=0.016) — replay immediately represents the
  new patch
- is_prev_patch is reliably NEGATIVE (7/7, median −1.8) — the
  patch the animal just left is actively suppressed, below even
  the unrelated third patch
```

## Per-stop rate ladder: why it carries the reward basis and stops at R1 (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
R1 carries the whole reward history in three numbers: the known-outcome lag
vector projected onto the log-spaced raised-cosine basis
(``REWARD_HISTORY_COLUMNS``), plus the structural-missingness indicator
``REWARD_HISTORY_INCOMPLETE``. It is the stop-level analog of the content
hierarchy's reward-history axis (``reward_fraction_candidate_arm_last10_visits``
at M3): R1 vs R0 tests whether replay *occurrence* tracks reward history at
all, and is the confound check for whether an M3 content effect is partly a
rate effect. The basis replaces both the single newest-outcome indicator and
the overlapping window-10 fraction — one smooth kernel instead of two
encodings of the same history at two timescales. Candidate-arm visit recency
has no stop-level analog (every trial visits some arm), so the rate hierarchy
stops at R1.

The hierarchy intentionally stops at R1. An ``is_swr_stop`` term
(1 if any timebin in the stop had an SWR) was dropped: ``n_selected_events``
counts non-local events, which overlap SWRs heavily by construction,
so ``is_swr_stop`` is largely tautological with the response and says
little about whether SWRs *drive* replay occurrence. The SWR-vs-non-SWR
question is answered at the event level (the ``swr_x_*`` content
interactions / SWR-split fits), not via the per-stop rate. A future
R-level term (+ ``trials_since_block_change``, ``block_number``) is
deferred until the contingency-block loader is wired.
```

## Per-bin rate ladder: the retired per-lag dummy rungs (moved 2026-09-06)

Superseded by § "Retirements in the per-bin rate ladder (Family B)" above; kept as
the verbatim `model_specs.py` comment as of its removal on 2026-09-06.

Moved verbatim from `model_specs.py`:

```
The five ``onset_lag1..10`` dummy rungs (M3_hist, M4_hist_rew,
M6_time_hist, M7_time_hist_rew, M9_rew_x_time) are RETIRED. The dummy
history block separates on the refractory bins, which voids the whole
rung's held-out metric: measured on the 2026-09-06 refit of 218 sessions,
``cv_bits_per_event`` was NaN in 99.08 % of non-local fits (99.08 /
99.54 / 99.54 / 100.00 % across the four populations), so the rungs could
not be cross-validated at all. The raised-cosine variants below span the
same 150 ms window with four smooth parameters and ARE scored. The
reward x time interaction survives as ``M9r_rew_x_time`` on the
raised-cosine parent. See § "Retirements in the per-bin rate ladder (Family B)".
```

## Per-bin rate ladder: the retired M8 switch rung (moved 2026-09-06)

Companion to § "Retirements in the per-bin rate ladder (Family B)" above; kept as
the verbatim `model_specs.py` comment as of its removal on 2026-09-06.

Moved verbatim from `model_specs.py`:

```
M8_full_switch (the per-bin switch main-effect extension) was retired:
The completed-trip switch flag is often constant within a session's complete-case
cohort, making the design rank-deficient. See the naming spec's
"Immediate switch is not a nuisance adjustment" section. Nothing parents
off it, so removal leaves the ladder intact.
```

## Content is_swr-contrast base: why candidate_arm_distance_rank is absent (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
``candidate_arm_distance_rank`` is deliberately absent. It takes five tied
values over a fixed track topology, its rank-1 level is *identical* to
``is_current_arm``
(agreement 1.000000 across 59,374 location-resolved events), and it ranks last
in unique likelihood of any covariate here. Carrying it collapsed the
current-arm coefficient from +0.705 (5/5 animals) to +0.102 (3/5) and doubled
its standard error, making every C*/CS* ``is_current_arm`` non-comparable with
``M1``/``C1_current``. Its removal leaves the SWR interactions untouched
(the SWR × candidate-arm belief-mean coefficient changed -2.390 -> -2.407).
This also restores consistency with the
module docstring above, which excludes ``distance_to_candidate_arm_cm`` and
``is_same_patch``
on the same grounds -- candidate-arm distance rank re-imported most of their
content.
```

The `-2.390 -> -2.407` figure is an **invariance check only** — it says the
exclusion did not move the SWR × belief-mean term, not that the term is
estimable. That −2.4 family is withdrawn; see § "Correction (2026-09-06): the
whole last-rewarded family is refused".

## Content is_swr-contrast base: the event-length control (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
Event-length control. SWR-coupled events run ~1.8x longer than non-SWR ones, and a
longer trajectory visits more arms, so an uncontrolled ``swr_x_is_current_arm``
partly reads "longer events leave the current arm". z(log n_nonlocal_bins) crossed
with the current and previous arm indicators removes that channel from every rung
built on the base; the ``nt`` twins below keep BASEc_NT as the uncontrolled
sensitivity fit.
```

## Content is_swr-contrast: the direct/total estimand of every swr_x term (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
CS1 inherits the within-stop-time and event-length current-arm interactions
from BASEc UNCONDITIONALLY, so ``swr_x_is_current_arm`` is the SWR-specific
current-arm effect BEYOND the general time course and BEYOND event length
(SWR and non-SWR events differ in both).

ESTIMAND. ``n_nonlocal_bins`` is post-treatment with respect to ``is_swr``
(SWR events run ~1.7x longer), so conditioning on it is a post-treatment
adjustment and length may be a mediator (SWR -> longer replay -> leaves the
current arm). Every ``swr_x_`` coefficient on a BASEc rung is therefore the
LENGTH-ADJUSTED (DIRECT) SWR effect; the ``nt`` twin, fitted on BASEc_NT,
holds the TOTAL SWR effect. Both are shipped and are reported side by side;
neither supersedes the other.
```

## Content patch is_swr-contrast: the retired CP2/CPS3 rungs (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
``CP2_olddecay`` and ``CPS3_olddecay`` were retired 2026-08-18. Both crossed
``is_last_rewarded_noncurrent_patch`` with a peri-switch term, and
the indicator correlates -0.96/-0.97 with its own interaction, so the design
was rank-deficient in 83 and 87 of 215 sessions. What survived the |coef|<10
filter was selection on the estimate: CP2 reported +4.83 with a typical
session SE of 7,313, against +0.36 on the sessions that were actually
identified. A 3%-support indicator crossed with a time term is not estimable
per session in a 3-alternative model.
```

## Choice models: the two excluded behavioural predictors (moved 2026-09-06)

Moved verbatim from `model_specs.py`:

```
Two predictors are deliberately excluded from the behavioural block:
- ``is_same_patch``: a coarse binary of ``distance_to_candidate_arm_cm`` (same-patch
  arms are close); the two were near-collinear (VIF > 700).
- ``distance_to_candidate_arm_cm`` itself: the animal chose the closest arm in ~95-99%
  of stops (median 0.99 in sessions that previously failed to fit),
  producing quasi-separation that crashed ``ConditionalLogit.fit`` in
  ~25% of sessions. Dropping it restores identifiability across all
  sessions and de-saturates the baseline so ``C_full`` vs
  ``C_behavior`` has room to detect a replay effect. Trade-off: the
  replay coefficient is no longer cleanly controlled for distance,
  and ``replay_fraction`` is correlated with low candidate-arm distance (replay
  biases toward the current/closest arm), so a positive
  ``replay_fraction`` coefficient could partly launder distance. A
  switch-style restriction (fit only on stops where the animal did
  not pick the closest arm) is the principled alternative; defer.
```

## Where the rest of the narrative lives

The sections above marked *(moved 2026-09-06)* were lifted out of
`model_specs.py` verbatim; each left a one-line pointer back to this file at the
place it came from. What stays in the module is the short comment that says what
a constant or rung *is* — the reasoning about why it came to be, the retired
rungs, and the measured numbers behind an exclusion live here. This file is the
design journal; `model_specs.py` is the machine-readable catalog.
