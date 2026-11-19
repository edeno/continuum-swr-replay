# Regression pipeline review: models, covariates, timing, readability, tests

**Date:** 2026-09-03
**Branch / HEAD:** `collapse-covariate-boundary` @ `53fad0c`
**Live tables reviewed:** `Processed-Data/replay_content/*.parquet` regenerated 2026-09-03 18:05
(`covariate_contract_v5_dio_poke_trials`); fitted artifacts dated 2026-09-02.
**Method:** code reading plus executed checks (synthetic boundary scenarios against the real
functions; refits and statistics on the live tables and persisted fits). Every finding below is
labeled CONFIRMED (observed by execution) or HYPOTHESIS (reasoned, needs a check). No
repository file was modified. Scratch scripts live in the session scratchpad.

Prior reviews in this directory (`07_*`, `08_*`) were treated as leads, not as evidence; the
2026-07-29 code map references modules (`covariates.py`, `tables.py` builders) that no longer
exist at the cited paths.

---

## 0. Executive summary

The pipeline's *plumbing* is in good shape: the two-clock design (visit history at arrival,
reward history at outcome availability) is implemented and holds at exact boundaries; events
partition into stops with no leakage; the qvals join is exact on the stops it covers; and the
covariate timing registry blocks post-outcome and future-referencing predictors from predictive
rungs. The unit suite (1631 passed, 1 xfail, 2:43 wall) kills 7 of 8 hand mutations.

The *inference* layer is not in good shape. Five problems change what the fitted numbers mean:

| # | Finding | Effect on results | Status |
|---|---|---|---|
| A1 | Content models treat every 2 ms decoder bin as an independent multinomial draw; 66 % of events have one MAP arm throughout. | Per-session SEs ~6x too small; per-session ΔAIC/LRT inflated ~25-30x; events weighted by duration. Point estimates and across-session sign counts unaffected. | CONFIRMED |
| A2 | Per-bin (Family B) and per-stop (Family A) rungs are fit on different rows (spec-specific `dropna`). | Every "reward kernel improves fit" ΔAIC is dominated by dropped early-session bins (corr 0.98 with events dropped). Same-row ΔAIC is 10-100x smaller. | CONFIRMED |
| A3 | SWR-contrast rate runner uses the legacy `|coef|>=10` separation flag. | 70 % of fits flagged; the audited SWR-rate table has **0 rows**. The rate half of the SWR dissociation currently publishes nothing. | CONFIRMED |
| B1 | Content SWR contrasts do not control event length/dispersion; SWR events are ~1.8x longer and place 36 % vs 17 % of bins on trunks. | `swr_x_is_current_arm` ("SWR broadens off current") is partly "longer trajectories visit more arms". | CONFIRMED |
| C1 | A "stop" is an immobility fragment, ~2 per well visit; 69 % of at-well events are in a non-first fragment. | `tc`, the 4 s truncation, `time_since_stop_start_s` and the log(immobility) offset measure time since a speed-threshold crossing, not since arrival or reward. | CONFIRMED |

Everything fitted so far (all `models*/` dirs) predates the current tables and the paper claims
table still quotes removed predictors (`is_last_rewarded_elsewhere`, `is_after_omission`) and
old rung numbers. Nothing in `models*/` should be cited until A1-A3 are fixed and the ladders are
refit on the 2026-09-03 tables.

---

## A. Are the regression models right, and are they fit and summarized correctly?

### A1. CRITICAL: within-event pseudo-replication in the content models (CONFIRMED)

- `analysis/replay/fitting.py:182` fits statsmodels `ConditionalPoisson` on `arm_count_hard`, whose
  likelihood is `sum_e [sum_a c_ea eta_ea - L_e log sum_a exp(eta_ea)]`: L_e independent categorical
  draws per event. The decoder runs at 500 Hz (`data_aggregation.py:301`), so L_e is the number of
  2 ms bins in the event.
- Live data (65,913 events): bins/event median 32 (SWR 45, non-SWR 26); **66 % of events keep one
  MAP arm for the entire event**; P(arm_t == arm_{t-1}) mean 0.935; median runs/event = 1.
  Pearson dispersion of M4 at the (event, arm) grain ~28 (12 sessions, range 16-37).
- Refit of eight real sessions, M3 vs M2, hard counts vs one draw per event (dominant arm through the
  same fitter): SE on `reward_fraction_candidate_arm_last10_visits` 0.07-0.12 vs 0.40-0.73 (~6x);
  ΔAIC(M3 vs M2) 10-317 vs -2.0 to +6.7. Point estimates also move (e.g. -0.43 vs -0.03; -2.03 vs
  -0.81) because the count response weights events by duration and by within-event mixing.
- The persisted event-grain ladder (`models_event_grain/`) already tells the other story: M2->M3
  ΔAIC -1.2 (child better in 32/206 sessions, 0/5 animals); M3->M4 -0.3; M5->M6 -0.7. Only
  M0a, M1, M2 and M5 (SWR x spatial) improve fit at the event grain.
- What is affected: per-session `pval_*`, `pct_sig_session`, `typical_session_se`, `lrt_significant`,
  `child_better_aic`, `lr_pvalue` (`model_comparison.py:74-77,139`), and the BIC `n = n_events`.
  What is not: across-session medians and per-animal sign consistency.
- Recommendation: (i) stop reporting per-session p / LRT / ΔAIC>2 from the timebin grain; (ii) select
  ladders on a quasi-likelihood (QAIC with phi from the Pearson statistic) or on the event grain;
  (iii) if the within-event distribution is the scientific object, model it (Dirichlet-multinomial or
  a cluster-robust sandwich on events); (iv) state in `docs/models_table.md` that the response weights
  events by duration.

### A2. CRITICAL: rate ladders compare rungs fit on different rows (CONFIRMED)

- `per_bin_rate_model.py:135` `dropna(subset=needed)` with spec-specific `needed`; same in
  `per_stop_rate_model.py:103`. Rungs with `known_reward_outcome_lag_10` drop the first ~10 trials'
  bins; rungs without keep them. `compare_nested_models` guards `n_events`, not `n_bins`/`n_stops`.
- Persisted `models_per_bin_rate/all_models.pkl` (872 session x population cells): `n_bins` differs
  parent vs child in 748/872 (M0->M1), 871/871 (M6->M7), 871/871 (M5->M7r). Median 2,286 bins and
  8 event-bins dropped per cell; median ΔAIC(M6->M7) 97 with corr(ΔAIC, events dropped) = 0.98.
  Refitting M6 on exactly M7's rows in four sessions: ΔAIC 822/283/561/281 -> **24/42/29/2.6** (10
  added parameters). Per-stop R0->Rlag: median ΔAIC 48, corr 0.70. The dropped stops have a
  different onset rate (1.43/s vs 0.97/s), so this is not random thinning.
- The runner docstring (`scripts/rate/fit_per_bin_rate_models.py:213-218`) claims identical bins;
  `tests/unit/test_replay_tables_audit_regressions.py:299-330` already encodes the defect as a
  strict xfail.
- Recommendation: freeze one complete-case (stop, bin) cohort over the union of all rung columns
  (the content runner's `freeze_common_cohort` pattern); assert `n_bins`/`n_stops` equality in every
  ΔAIC/CV comparison; re-derive every Family-A/B ladder claim, including `docs/rate_models.md` and
  `per_bin_rate_results.md`.

### A3. CRITICAL: SWR-contrast rate table is empty (CONFIRMED)

- `scripts/rate/fit_swr_contrast_rate.py:182` calls `fit_status(res, "glm")` without `predictor_sd`;
  the effect-scale fix (`_fit_diagnostics.py:25-31`, applied in `per_bin_rate_model.py:165`) never
  reached this runner. `swr_contrast_rate.pkl` (8,115 fits): `separated=True` in 70 %, mostly on
  small-SD `hist_rcos_*` columns. `aggregate_contrast` drops flagged sessions and refuses terms with
  >20 % dropped, so `swr_contrast_rate_audited.pkl` has **0 rows**.
- Underneath: ~1,950 fits show genuine quasi-complete separation on `known_reward_outcome_lag_1`
  (no non-local onset after any omission in that session; ~65/218 sessions for `non_local_swr`).
  That term is unidentified per-session there.
- Recommendation: pass `predictor_sd=np.asarray(res.model.exog).std(axis=0)` and refit; for the
  lag-1 separation use a pooled or penalized (Firth/ridge) fit, or report per-animal sign only.

### A4. MAJOR: per-bin cross-validation is void for the dummy-history rungs (CONFIRMED)

`cv_bits_per_event` is NaN in 100 % of sessions for M3_hist, M4, M6, M7, M9 (98 % flagged
separated: `onset_lag1..3` are structurally zero because events outlast 45 ms) and present in only
22 % (M3r) / 16 % (M7r). Family B currently has neither a valid AIC (A2) nor a usable CV for most
rungs; only the time-rescaling KS statistic survives (0.49 -> 0.86-0.90 once the spline enters,
which is sound). Retire the `onset_lag` dummies from the shipped ladder; investigate why 78-84 % of
histcos folds still flag separated (HYPOTHESIS: lag-1 separation from A3 or the SE rule on tiny-SD
rcos columns).

### A5. MAJOR: cross-session aggregation

- `contrast_aggregation.py:117-121` drops a session when `SE > 10*|coef|`. This is selection on the
  estimate. Simulation (200 sessions, 2000 reps): true 0.10 with SE 0.5 -> reported median 0.15;
  true 0.10 with SE 1.0 -> 0.18; null unbiased. On the current content contrast tables it removes
  only 0-5 sessions per term, so present medians move little, but the rule is not an identifiability
  criterion. Replace with `bse * sd_x > c` (which `separated` already implements).
- Inference is per-session median + across-session Wilcoxon (`session_signrank_fdr_q`) with sessions
  nested in 5 animals; two animals supply >50 % of sessions. The code labels the Wilcoxon
  "descriptive" and puts per-animal medians first, which is honest, but the q-values are what gets
  quoted ("q=5e-7"). Report per-animal medians with an animal-level bootstrap/permutation, or fit the
  hierarchical model the provenance JSON already names.

### A6. MAJOR: the canonical content directory was clobbered (CONFIRMED)

`fit_replay_content_models.py:842,891` replaces the whole output dir. `models/all_models.pkl`
(2026-09-02 17:50) now holds only Mu/Mv/Mvu; M0-M6 exist only in `models_event_grain/` and backups;
`models/model_comparison.pkl` has one Mu->Mvu row. Key the output on the requested model set or
refuse `--models` subsets into the canonical dir.

### A7. MINOR

- Pooled CS rungs: the within-stop-time control `_TSC` is in CS1 and C6-C12 but absent from
  CS2/CS4/CS5/CS5b/CS6, and BASEc coefficients are forced equal across SWR/non-SWR, so
  `swr_x_<key>` can absorb SWR differences in correlated base terms.
- Event-grain conditional logit is a different estimand (dominant arm of L draws amplifies log-odds:
  1.2 -> 4.8 in simulation); do not compare magnitudes across grains.
- `dispersion = deviance/df_resid` (`per_bin_rate_model.py:187`) is uninformative at 0.015
  counts/bin; use Pearson chi2/df on aggregated bins or drop it.
- `aggregate_to_timebins` (`_helpers.py:874-889`): onsets at `t == immobility_duration_s` or in a
  bin removed by `exposure > 1e-4` are silently dropped by the left merge; float jitter at exact
  15 ms edges (`(t/bin_s).astype(int)`: 0.030/0.015 = 1.99999809). 204/137,745 live onsets sit on
  an edge, 0 at `t == duration`. Use `floor(t/bin_s + 1e-9)` or `searchsorted`, and assert
  `count.sum() == n_onsets`.

### Checked and found sound

`ConditionalPoisson` is a true conditional likelihood (llf identical with an event-constant column at
0 or 5); `within_group_full_rank` refuses within-event constants; known beta recovered unbiased under
sticky bins. Ladders are genuinely nested; leaf-only contrast rungs are a coherent 6-category
likelihood and correctly kept out of nested ΔAIC. Content runner freezes one complete-case cohort;
`n_events` equal across rungs. `aggregate_to_timebins` exposures sum to `min(immobility, TMAX)`;
`onset_lag` resets at stop boundaries. CV folds group by event (content) / stop (rate). Per-stop
GLM: intercept forced, rank check, correct offset. Time-rescaling GOF is the right check.

---

## B. Are the covariates right, valid, and complete?

### B1. MAJOR: event length / dispersion is unmodeled in the content SWR contrasts (CONFIRMED)

At-well events: `n_nonlocal_bins` median 45 (SWR) vs 25; `n_unique_arms` 3.31 vs 1.44; posterior
entropy 0.49 vs 0.29; **trunk share of MAP bins 36.2 % vs 16.8 %**. Current-arm fraction falls
monotonically with event length in both classes (non-SWR 0.33 -> 0.17, SWR 0.31 -> 0.12 across
quintiles); the raw gap 0.285 vs 0.207 shrinks to ~0.03-0.05 within a length quintile.
`time_since_stop_start_s` medians are equal (2.52 vs 2.59 s), so `_TSC` cannot absorb it. The
conditional Poisson conditions on the total, not on dispersion. Add `z(log n_nonlocal_bins) x role`
terms (or stratify on length) to BASEc and re-derive CS1/CS2/CS5; report the trunk asymmetry.

### B2. MAJOR: the Beta-Bernoulli value/uncertainty covariates carry ~1-3 visits of memory (CONFIRMED)

Reconstructed alpha+beta = mu(1-mu)/var - 1 from the shipped per-port moments: median 2.5, 90th pct
5.1, **57 % of leaf rows < 3** (less than one observation beyond Beta(1,1)); arms with recency > 20
sit at alpha+beta = 2.00 (fully decayed to prior). Belief variance R^2 = 0.48 from log-recency +
never-visited; belief mean R^2 = 0.36 from reward fraction + recency; within-event r(mean, reward
fraction) = 0.74, r(mean, variance) = 0.82. "SWR content avoids high beta value" is mechanically
close to "avoids arms rewarded on their last one or two visits". Report the alpha+beta distribution;
add a model-free comparator (last outcome on the arm x recency) in the same rungs; run a no-decay or
leaky-Q sensitivity; say so when reporting.

### B3. MAJOR: `relative_value_switch_minus_stay_preoutcome` is the completed trip's decision variable (CONFIRMED)

`_spatial_bandit_rl.py:421-466`: start leaf = previous chosen leaf; stay = sibling of start.
r(dV, stay value) = -0.953, r(dV, switch value) = +0.15, r(dV, q_chosen) = -0.89. On stay trips
(~92 %) the sibling of the origin is the current well, so dV ~ -belief(current well). C7/G4/S6 test
"content vs minus pre-outcome value of where I am", not deliberation about switching. The forward
DV (row t+1) is computable from the same table; add it (post-outcome gated) and rename the current
one in `definitions.py`.

### B4. MAJOR: in-transit stops carry future-destination qvals covariates into the rate ladder (CONFIRMED)

`fit_swr_contrast_rate.py:244-291` inner-joins qvals with no location filter. qvals-mapped stops are
65.5 % at_well, 20.1 % in_transit, 14.0 % mixed. For in-transit stops `chosen_arm == next_arm` on
99.99 % and `stop_start < outcome_known_time` on 99.97 %, so `rpe_postoutcome`,
`reward_belief_mean_update_*`, `*_chosen_arm_*` describe a well not yet reached. In-transit stops
hold 7.2 % of non-local events but **10.4 % of non-SWR vs 2.5 % of SWR**, so the leak is
SWR-asymmetric and `is_swr:rpe_postoutcome` partly reflects composition. G5/S7/G7/S9 are declared
descriptive, correctly. Gate to `stop_start >= outcome_known_time` (as `fit_uncertainty_rate.py:153`
does) or restrict to at-well stops.

### B5. MAJOR (completeness): the current outcome is absent from the content contrasts (CONFIRMED)

`is_rewarded_outcome` is known at event start for 98.5 % (non-SWR) / 98.9 % (SWR) of at-well
events, with rewarded share 89.1 % vs 93.1 %. The blanket POST_OUTCOME class excludes it from every
predictive content rung although it is genuinely pre-event once gated at
`event_start >= outcome_known_time`. Add `is_rewarded_outcome x is_current_arm` under the existing
`retrospective=True` gate, or stratify the CS rungs by outcome.

### B6. MAJOR: Mv/Mu/Mvu are still nine-arm (CONFIRMED)

Within-event demeaned VIF on pooled at-well events: belief mean 13.5, variance 11.5 nine-arm vs
1.7 / 3.0 leaf-only; r(arm FE 6/7, belief mean) = -0.47. The same mechanism produced the retracted
-2.39. Build a leaf-only parent chain so these can be nested, or stop reporting them.

### B7-B9. MINOR

- `known_reward_outcome_lag_1` is the *current* outcome for 98 % of at-well stops and the *previous*
  outcome for in-transit stops (7.2 % of events, non-SWR-heavy). Documented in comments; the label
  "most-recent-known" undersells the SWR-asymmetric mixture.
- Arrival -> outcome gap on a first-ever visit: recency 1, `never_visited` 0, reward fraction 0.0
  (an "observed zero" that isn't); <= 1.5 % of at-well events; undocumented.
- `is_prev_arm` is same-patch on 92 % of events; with distance dropped it is the only same-patch
  term in BASEc. Consider adding `is_same_patch_as_current` so `is_prev` reads as the switch-trip
  increment.
- `docs/task_description.md:49-59` says ~35 % of qvals stops are non-canonical; the shipped table is
  99.2 % in {0.2, 0.5, 0.8}.

### Missing covariates, ranked by relevance to SWR vs non-SWR

1. Event length / dispersion (B1) - the largest SWR/non-SWR difference in the data.
2. Current outcome, event-start gated (B5).
3. Graded ripple power / MUA rate (`ripple_power_z`, r = 0.69 with `is_swr`) as an alternative to the
   binary `is_swr`.
4. Session time / satiety (`trial_number_by_epoch` already in qvals).
5. Forward decision variable (B3).
6. Time since arrival / since pump as the consummatory clock in Family B (see C1).
Dropping raw distance is defensible (rank-1 == `is_current_arm` exactly); `is_same_patch_as_current`
should then be explicit. Number of events per stop (median 4 vs 4) and speed (immobility-gated) are
not concerns; `next_arm` is correctly response-only.

### Checked and found sound

Two-clock semantics at exact boundaries (synthetic); recency == 1 on current and == 2 on prev on
100 % of at-well events; `is_prev & is_current` never both; `chosen_arm == occupancy_well_arm` 100 %
at-well; `rpe == reward - belief` exactly; belief mean genuinely pre-outcome; centering excludes
censored arms; spline masked and centered on its support; standardization on leaf support with the
reference asserted at fit time; timing registry keeps POST_OUTCOME / FUTURE_REFERENCING atoms out of
predictive rungs; qvals missingness (1.9 %, session-end tail) is not a leak.

---

## C. Is the processing and timing correct?

### C1. MAJOR (interpretation): the modeled stop is an immobility fragment (CONFIRMED)

`_helpers.py:118-139` defines stops by `head_speed < 4 cm/s` crossings. Live: 69,224 at-well stops
for 35,043 visits (mean 1.96/visit, 90th pct 3); **50.6 % of at-well stops are the first fragment
of their visit; 69 % of at-well events occur in non-first fragments**. `stop_start - arrival` median
0.77 s, 18.6 % > 2 s, 9.2 % > 10 s; 81 % of at-well stops begin after the outcome is known.
`is_reward_stop` (pump inside bout) is True for 0.2 % of onsets. Neither `docs/rate_models.md` nor
the covariate dictionary mentions fragmentation. Consequence: `tc`, the 4 s truncation,
`time_since_stop_start_s`, and the per-stop offset describe time since a speed-threshold crossing.
Document it; add `t_since_arrival` / `t_since_pump` (already in the onset table) as the
consummatory clock or run a merged-fragment sensitivity fit; retire `is_reward_stop`.

### C2. MAJOR: reward covariates frozen at bout start; stale comments (CONFIRMED)

`tables_stop.py:138-147` evaluates `known_reward_outcome_lag_*` at `bout_starts` and
`aggregate_to_timebins` carries them to every bin. 13,225 at-well stops (19 %) start inside the
arrival -> outcome window, so ~295k of their 365k bins carry the *previous* outcome after the current
one is known. No leak (lag_1 matches the arriving trial's reward at chance, 52.7 %), but a
misattribution. Comments at `value_uncertainty.py:345-350, 446-454` and `qvals_table.py:1003-1006`
say `is_rewarded_outcome` is missing at essentially every at-well stop; it is finite at 80.1 %.
Evaluate lag_1 per bin (left edge >= `outcome_known_time`) or drop pre-outcome bins; fix the comments.

### C3. MAJOR (provenance): fits predate tables; no guard ties qvals time to the live stop table (CONFIRMED)

Tables 2026-09-03 18:05; `event_onset_table` 09-02 12:15; `qvals_table` 08-29; every `models*/`
artifact 09-02 (v3-cache era; pickles carry no stamp). v3 -> live drift: `is_rewarded_outcome`
differs on 1,361 stops, `known_reward_outcome_lag_1` on 636, `reward_fraction_last10_known_trials`
on 706, `event_table.is_rewarded_outcome` on 12 events. `prepare_data.py:1014-1021` checks only the
schema stamp, never that `qvals_context_time == stop_start`; a renumbered decode would misjoin
silently on `hash_stop_id`. Assert equality per `stop_id`; stamp model pickles with table hashes.

### C4-C6. MINOR / NOTE

- `trials.py:495-501` comment about wilbur pumps firing > 1 s after release contradicts the 0.3 s
  latency guard; corpus latency p99 91 ms, max 236 ms.
- `well_visits.is_reward` is set from the next physical bout (`trials.py:818-821`), so 424
  gap-recovered rewards are False there but True in `trials`; only the gallery reads it.
- qvals attaches the *upcoming* trial to in-transit stops (see B4); worth NaN-ing chosen-arm columns
  off-well or registering them in `covariate_timing`.
- `time_since_stop_start_s` is the in-event *median* bin time, differing from `event_start -
  stop_start` by up to 1.7 s (documented).
- HYPOTHESIS: an in-progress poke at epoch start gets `arrival = epoch start` via the rising edge at
  index 0 in `events.py:115-119`; check `port_experiences.arrival_time[0]` against the epoch's first
  timestamp.

### Verified sound (boundaries exercised with the real functions)

Trial clocks: `start_time` = origin final poke-out; `goal_arrival_time` = destination first poke-in;
`outcome_known_time` = pump onset (gap pump credited) or `min(arrival + 0.30 s, departure)`; same-port
re-pokes collapse; orphan pump warned and dropped. Context at `t == arrival` -> at_well, outcome NaN;
`t == outcome_known` -> reward visible (`side="right"` inclusive); `t == departure` -> in_transit.
Events subset of stops (0 violations); sum `n_selected_events` = 65,913; SWR/MUA split sums match;
`is_swr` strict positive-duration; junction bins excluded from `arm_count_hard`
(`n_undecoded_bins` all 0); `attach_is_swr_to_onsets` joins on onset time, not rank, and passes on
the live artifacts; session-first stop before any trial yields NaN context.

---

## E. Are the tests meaningful?

Observed: `uv run pytest tests/unit -q -x` -> **1631 passed, 1 xfailed** in 2:43 (3.4 GB peak);
`tests/analysis/hippocampus` baseline 63 passed. Integration/property/visualization not run.

Mutation spot-check (real import path, tree restored and verified after each):

| Mutation | Result |
|---|---|
| `context.py:564` completed-trial `side="right"` -> `"left"` | killed (1 test) |
| `event_detection.py:319` SWR overlap `>` -> `>=` | killed (1) |
| `reward_history.py:59` cutoff clock outcome -> arrival | killed (1) |
| `_helpers.py:856` drop TMAX `minimum(..., nmax)` | survived, **equivalent mutant** (exposure filter removes the bins anyway; `nmax` is redundant) |
| `reward_history.py:273` reward-lag index off by one | killed (3) |
| `per_stop_rate_model.py:112` offset `log(exposure)` -> `exposure` | killed (4) |
| `visit_recency.py:96` drop within-event centering | killed (4) |
| `context.py:581` `prev_well` from `to_wells` | killed (1) |

Findings:

1. MAJOR `test_per_bin_rate_model.py:235-257 test_cv_folds_split_on_stop` tests sklearn's KFold,
   never the function; a bin-level leakage bug in `cross_validate_per_bin_rate_model` passes.
2. MAJOR Builder fixtures (`tests/unit/_port_experience_fixtures.py:579-595`) put omission
   `outcome_known_time` at departure, not `poke_in + 0.30 s`; every table/covariate test runs on a
   later omission clock than production. Add one DIO -> tables end-to-end test asserting a lag_1
   value inside the arrival -> outcome gap.
3. MAJOR `test_fit_per_bin_rate_models.py` plants RR = 2 and asserts only schema; add a value check.
4. MAJOR The A2 defect is held open as a strict xfail (`test_rate_ladder_rungs_share_a_cohort`).
5. MINOR Near-vacuous assertions (`test_fitting.py:259-266`, `test_per_bin_rate_model.py:171-191`).
6. MINOR Real-data tests in `tests/unit` gated on cwd-relative paths and unmarked as slow.
7. MINOR `tests/baselines/` (35 PDFs) referenced by no test. Fixtures `_EnvStub` x6,
   `_bandit_graph` x7 duplicated across files.
8. MINOR Two sign-critical behaviors (arrival tie, prev/next arm) each hang on a single test;
   `TestMutationSurvivorGuards::test_is_prev_references_prev_arm_not_next_arm` did not fire on M8.
9. NOTE `docs/design/builder_test_floor.md` cites `history.py:702/748`, a module that no longer
   exists; re-run `scripts/qa/mutation_check.py`.

Top untested behaviors: stop-disjoint CV through the real function; planted-effect recovery in the
per-bin orchestrator; the inclusive `is_swr_stop` vs strict event `is_swr` dual convention at a
boundary-touching SWR; production omission clock reaching a table builder; `time_since_stop_start_s`
value; `_fit_alpha` objective; pooled `_z` moments through `prepare_data`; spline behavior at
`tc >= upper_bound`; `n_categories` vs actual arm count in pseudo-R2.

---

## D. Readability and architecture

Counts from grep/wc/AST scans over the regression pipeline (26,141 lines in scope). `ruff
--select F401,F841,F811` is clean; every argparse flag is read. The problems are structural, not
lint-level.

### D1. MAJOR: `model_specs.py` still carries the narrative its docstring says was moved out

Header L22-27 points to `docs/design/model_history.md` for "reversed reasoning and retracted
numbers", yet ~120 lines of dated, number-heavy narrative remain (L127-186, 200-212, 383-400,
469-480, 905-940, 1095-1110, 1264-1271: "removed on 2026-08-18", "99 of 208 sessions", "SE of
7,313", "Results on 7 j16 sessions ... p=0.016"). L157-159 says reward memory "lives only in
C5_lastrew / CS3_lastrew" while that family is recorded elsewhere as refused. A newcomer cannot tell
which numbers are current, and two copies will rot. Leave one-line pointers next to each rung.

### D2. MAJOR: inverted layering and a god-module

- `_helpers.py` (948 lines, "Shared helpers for event-table builders") also holds the Family-B design
  builder `aggregate_to_timebins` (153 lines), `raised_cosine_history_basis`, the `POPULATIONS`
  cohort lambdas, `attach_is_swr_to_onsets` (176 lines), arm fixed effects, ID hashing, posterior
  alignment and reducers; 12 modules import it.
- `model_specs` defines table column names (`ARM_FE_PREDICTORS`, `RECENCY_SPLINE_COLUMNS`) and knots
  that the *builders* import, so it is a schema module, not a spec catalog; changing a knot there
  silently changes the persisted table. `hippocampus.event_table -> replay._helpers`,
  `data_loaders.trial_clocks -> analysis.constants`, `behavior.qvals_table -> replay.{registry,
  standardization}` all point the wrong way.
- Split `_helpers.py` into bouts / ids / per-bin design; move the shared column constants into a
  small schema module both sides import; move `OMISSION_KNOWN_LATENCY_SEC` to `data_loaders`.

### D3. MAJOR: `covariate_registry.py` is 1535 lines, ~860 of them a finished rename ledger

`REGISTRY` L165-1027 holds 141 `_rename` and 9 `_alias` entries for a migration the header says is
complete. Production uses `require_covariate_schema` (x42), `stamp_covariate_schema` (x12),
`row_keys`, and the qvals ingestion quartet; `assert_canonical_vocabulary` is a deny-list that grows
with the ledger. Keep ~250 lines (row keys, stamp/require, the ~15 qvals ingestion entries, a fixed
frozenset of legacy names); move the ledger to the naming-scheme design doc.

### D4. MAJOR: standardization and interaction columns are built four ways

Persisted `_z` via `standardization.attach_standardized` over leaf support with a JSON reference
(`prepare_data.py:850`); `fit_swr_contrast_content.py:131` defines its own `_z` (nanstd over all
rows, no reference) and hand-builds 15 interaction columns (L184-227, again for patch L250-260);
`fit_uncertainty_rate.py:608` and `fit_swr_power_model.py:274` z-score inline. `candidate_arm.py`
builds `swr_x_*` for the base ladder. `_assemble_arm` also reads `.pkl` paths that only work because
`table_io.read_table` swaps suffixes (`table_io.py:177-207`). One `standardize()` and one interaction
builder driven by the `SWR_CONTRAST_CONTENT_*_LADDER` lists.

### D5. MAJOR: the same concept is defined in several places

- `is_stay`: five implementations (`tables_event.py:284`, `tables_stop.py:292`, `shuffles.py:325`,
  `contfrag/tables.py:422, 913`).
- Session key: `session_keys.py` plus `retrospective._session_keys`, `timebin_tables._session_keys`,
  `_helpers._SESSION_COLS`, `table_contracts._SESSION_COLUMNS`, `fit_replay_content_models.SESSION_COLS`,
  and an inline list in `prepare_data.py:456`; `day` vs `session_date` is live in 6 modules.
- Canonical column lists: `_tables_common` (8 tuples), `covariate_registry._ROW_KEYS`,
  `table_contracts._LEAF_SUPPORT_COLUMNS`, `definitions.COVARIATE_DEFINITIONS`,
  `fit_replay_content_models._CANONICAL_SCHEMA_COLUMNS`. Adding a column touches 4-5 lists.
- `versions.assert_versions_compatible` has zero production callers (tests only), so the "makes
  drift loud" promise is not wired.
- Event timing computed twice and cross-checked (`hippocampus/event_table.py:183` and
  `tables_event._aggregate_event_features`), then `_merge_classified_and_verify_timing` (100 lines)
  renames, asserts bit-equality and drops the copy. Compute once.
- `covariates/clocks.py` `trial_cutoff_times`/`trial_outcome_times` are 1-line wrappers over the
  private functions; make the private ones public and delete the wrappers.

### D6. MAJOR: docs are stale against code

- `.claude/docs/reviews/08_regression_pipeline_code_map.md`: all 13 spot-checked line anchors land on
  unrelated lines; links `covariates.py:47` (now a package).
- `docs/models_table.md` ("mirrors model_specs.py"): omits `Mu`, `Mvu`, and the entire SWR-contrast
  *content* ladders C0-C13 / CS1-CS6 / CP / CPS; 70 lines of process prose sit in the catalog.
- `docs/README.md:86` "content M0-M7 / P0-P4" (ladder is M0-M6 + Mv/Mu/Mvu, P0-P3 + Pv); L91 links a
  missing plan file. `docs/covariate_dictionary.md:13,90` cite `history.py` (now a package).
- `CLAUDE.md:216-217` says `label_event_*` is retained; `tables_event.py:558-565` drops it.
  `CLAUDE.md:162, 215` cite wrong module paths.
- `paper/claims_table.md` quotes `is_last_rewarded_elsewhere`, `is_after_omission`, `swr_x_q_arm`,
  `arm_uncertainty`, "M5 non-SWR-only" with the pre-2026-08-18 rung numbering; none of these names
  exist in `model_specs.py`.
- `docs/replay_rate_content_swr_synthesis.md:381` cites a retired term; bare "M7" is ambiguous
  between the deleted content M7 and per-bin `M7_time_hist_rew`. `docs/per_bin_rate_results.md`
  is undated and says 218 sessions. `scripts/README.md` omits four `qa/` scripts.

### D7-D9. MINOR

- Shims: `tables.py` (58 lines) is a pure re-export kept "for the parity/provenance tests"; 35
  `from ._helpers import (x as _x)` re-alias lines across the grain modules and package inits;
  12 private-name re-exports whose only external callers are two test files.
- Dead / test-only: 6 strictly dead symbols (`_helpers.immobility_bout_starts`,
  `retrospective._PATCH_TO_ARM_SET`, three `simulation._BASE_*`, `_spatial_bandit_rl.set_nan_at_max2`);
  54 public symbols in `src/` referenced only by tests (the `*_per_arm` scalar views,
  `history/diagnostics.py`, the registry query API, `versions.assert_versions_compatible`,
  `behavioral_location_at_times`, several `trajectory`/`retrospective` helpers); 43 of 78 names in
  `analysis/replay/__init__.__all__` are never imported via the package.
- CLIs inside `src/`: `prepare_data.py` (1102 lines) and `qvals_table.py` (1089) carry argparse
  `main`s; `run_prepare_data` mixes cache I/O, Spyglass fallback, assembly, standardization and
  manifest writing. 96 functions exceed 80 lines, 8 exceed 200 (`build_stop_table` 251,
  `attach_candidate_arm_covariates` 209, `build_event_table` 208, `add_trial_context` 203). The
  content runner's science (`_fit_content_ladder -> fit_model_per_session -> fit_conditional_poisson`)
  is buried under ~650 lines of validators that belong in `table_contracts.py`.
- Naming: `is_prev_arm` is the *origin* of the last completed trip (needs a gloss everywhere);
  `in_transit` needs a 9-line disclaimer that it does not mean moving (the mask is literally
  `is_between_visits`); `dot` (22 signatures) and `tc` are opaque; rung numbering has gaps after
  retirements (CP0, CP1, CP3, CP4; M9 with no M8) and "M7" collides across families; suffixes `nt`,
  `b`, `r` are undocumented; `_helpers.py`, `_tables_common.py`, `summaries.py`, `learning.py` say
  nothing; `uncertainty.py` / `uncertainty_content.py` / `history/value_uncertainty.py` collide.
  `definitions.py` `producer=` is free text in 8 of 33 entries.
- `event_immobility.py` L16-18 documents that its filtering is silent; return the dropped count.

### Traceability (what a newcomer must follow)

For `reward_fraction_candidate_arm_last10_visits` the chain is DIO -> `trials.make_well_visits_df ->
make_trials_df_from_well_visits -> make_port_experiences_df` (+ `trial_clocks`) -> `prepare_data`
cache -> `tables_arm_patch.build_event_arm_table -> candidate_arm.attach_candidate_arm_covariates ->
history.reward_history.recent_reward_rate_all_events(k=10)` -> literal column name at
`candidate_arm.py:237` (also spelled in five schema lists) -> `model_specs._M3` ->
`fit_replay_content_models` (validators, timing gate) -> `fitting.fit_conditional_poisson`. The `k=10`
is a literal baked into the name; the producer is reached through two `__init__` re-export layers.

For `known_reward_outcome_lag_1` the f-string name is re-derived at eleven sites and consumed by three
different fitters (numpy design matrix, spec-generated patsy formula, runner-assembled patsy
formula); the "lag 1 is not the previous trial" caveat is restated in four docstrings.

For content M4 a reader needs `model_specs` (names), `candidate_arm` (producers),
`visit_recency` + `context` (semantics), `covariate_timing` (leak class, via a hand-maintained
20-entry `DERIVED_COVARIATE_SOURCES` map), `_tables_common` (persisted or not), `definitions.py` and
`covariate_dictionary.md` (prose).

---

## F. Recommended order of work

1. Fix A3 (one-line `predictor_sd`), A2 (common cohort in both rate runners), A6 (output-dir
   keying). Refit Family A/B and the SWR-rate contrast on the 2026-09-03 tables.
2. Decide the content estimand: event grain (dominant arm / soft fraction) or duration-weighted
   bins with QAIC / cluster-robust SEs. Re-run the content and SWR-content ladders with event-length
   and current-outcome controls (B1, B5). Retire per-session p/ΔAIC>2 reporting from the timebin grain.
3. Replace the `SE > 10|coef|` drop (A5) and move headline inference to animal-level statistics.
4. Document fragmentation (C1) and add an arrival/pump clock to Family B; evaluate lag_1 per bin (C2).
5. Add the qvals `stop_start` equality guard and table hashes to model pickles (C3); gate or NaN
   chosen-arm columns off-well (B4).
6. Rewrite `paper/claims_table.md` numbers only from refits; drop the stale predictor names.
7. Tests: items E1-E4.
