# Cohort and covariate-timing rules

Three cross-cutting rules that every fit runner enforces, moved out of the
model catalog so that file stays a catalog:

1. **Shared stop cohort** — which stops (and therefore which events) the rate
   and content families are both about.
2. **Covariate causal hygiene** — which covariates may be a forward predictor,
   and which rungs are retained only under an explicitly *descriptive* claim.
3. **Cohort-audit tables** — the `<run>_cohort_audit.csv` schema, its
   mutually-exclusive drop buckets, and its reconciliation invariant.

The rungs themselves, in R notation, are in
[`../models_table.md`](../models_table.md); the covariate definitions are in
[`../covariate_dictionary.md`](../covariate_dictionary.md).

## Shared stop cohort

**The rule.** Every rate and content runner selects its population through one
function, `analysis.events.cohort_selection.select_stop_cohort` (stop grain) or
its sibling `select_event_cohort` (event grain), on the canonical
`occupancy_state` column:

| cohort | stop-grain predicate | event-grain predicate | artifacts |
| --- | --- | --- | --- |
| `at_well` (canonical, default) | `occupancy_state in {"at_well", "mixed"}` — *well-anchored* | `occupancy_state == "at_well"` | the unsuffixed names |
| `in_transit` | `occupancy_state == "in_transit"` | `occupancy_state == "in_transit"` | `_intransit` |
| `all_stops` | no occupancy restriction | no occupancy restriction | `_allstops` |

**The two grains need different predicates to mean the same thing.**
`occupancy_state` is not the same statement about a stop as about an event. A
stop is labelled `mixed` whenever its sampled states differ; an event is
`at_well` only when its whole window is at one well. Measured on 2026-09-12 on
the 218-session canonical tables of that date (no script re-derives the counts
in this section), 14,845 of the 107,400 canonical stops were `mixed`, and they
held **18,274 of the 59,699 content events (30.6%)**; today's stop table has
<!--v:behavior.stops_mixed-->16,061<!--/v--> `mixed` stops of
<!--v:behavior.stops_total-->120,320<!--/v-->. So a naive stop-grain `occupancy_state == "at_well"` filter
would make the rate and content cohorts **cross** rather than nest.

In that measurement, those `mixed` stops were at-well bouts. Of the 14,743 (99.3%) that
have a last-well arrival at all, **every one** begins *after* it — median
0.73 s after, p10 0.22 s (the other 102 begin before the session's first
arrival, so they have no clock, not a late one). Their nearest-well distance
over the bout has median 0 cm at both ends and mean 0.28 cm, and is exactly 0
throughout for 81.6% of them, against a 5.2 cm median for `in_transit` stops.
And **96.8%** of their events (18,274 of 18,883) are themselves event-grain
`at_well`. Their first settling samples simply read as in-transit. The
stop-grain predicate is therefore *well-anchored* — a bout that begins after
arriving at a well and sits at it.

**The counts** (measured 2026-09-12 on the canonical tables of that date, schema
`v5_dio_poke_trials`; today's stop table holds
<!--v:behavior.stops_total-->120,320<!--/v--> stops, of which
<!--v:behavior.stops_well_anchored-->93,560<!--/v--> are well-anchored):

- stop table: **107,400** stops → **84,069 well-anchored** (69,224 `at_well` +
  14,845 `mixed`), 22,850 `in_transit`, 481 `unknown`.
- arm event table: **65,913** events → **59,699 at-well** (90.6%).

What reached the rate fits of that date from those 84,069: Family A modelled
**84,064** stops on `R0` (today <!--v:behavior.per_stop_audit_stops_modeled-->93,555<!--/v-->) (5 dropped for a missing own predictor), Family B
**83,549** over **6,777,953** bins (the same 5, plus 515 with no arrival clock,
which the arrival rungs put into the frozen union). Counts from
`models_per_stop_rate/all_models_cohort_audit.csv` and
`models_per_bin_rate/per_bin_rate_cohort_audit.csv`; the three cohorts are placed
side by side in
[`2026-09-12-rate-cohort-atwell.md`](../../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md).

**Nesting, and its residual.** Every event in the content cohort lies in a stop
in the rate cohort. The converse does not hold exactly: in the 2026-09-12
measurement, 609 events sat in well-anchored stops without being event-grain `at_well` (446 `in_transit`, 144
`mixed`, 19 `unknown`) — **1.0%** of the 60,308 events those stops hold. So
`content ⊂ rate` holds up to that residual, which is in the rate cohort and not
the content one by design.

**How it is applied.** The four rate runners (`fit_per_bin_rate_models.py`,
`fit_per_stop_rate_models.py`, `fit_swr_contrast_rate.py`,
`fit_uncertainty_rate.py`) expose it as `--stop-cohort`, applied to the stop
table **before** the `--max-per-animal` smoke subset and **before** the
common-cohort freeze, logged with its counts, written onto every fitted row as
`stop_cohort`, and recorded in the provenance sidecar's `extra`. The two
content runners (`fit_replay_content_models.py`,
`fit_swr_contrast_content.py`) apply the event-grain `at_well` rule
unconditionally — the content ladders are only defined there, since every
location-derived predictor (`is_current_arm` and everything built from it) is
undefined off a well.

`model_comparison.METADATA_COLUMNS` carries `stop_cohort`, so an at-well rung
compared against an all-stops rung is NaN'd with a reason rather than
differenced. That guard is soft on legacy artifacts: fits written before the
flag existed carry no `stop_cohort` at all, and `compare_nested_models`
deliberately treats a missing value as *unknown*, not *different*, so the
`n_stops` / `n_bins` row-count guard is the backstop for those until the
refits land.

**The PFC analyses read the same rule, at three grains.** The mPFC unit GLM
(`scripts/pfc/run_pfc_analysis_bandit.py`) fits 2 ms bins, so it applies the
stop-grain cohort **to the bins**: the fit mask is `is_immobile &
stop_interval_mask(position_time, kept_stops)`, i.e. an immobile sample that
falls inside a `[stop_start, stop_end]` interval of a kept stop. The events its
design matrices are keyed on are the **canonical** ones, read rather than
re-detected (`analysis/pfc/canonical_events.py` is the only place the PFC path
loads either): the non-local events are the session's rows of the shipped event
table (`Processed-Data/replay_content/event_table.parquet` — `event_id`,
`event_start_s`/`event_end_s`, `is_swr`, `occupancy_state`), and the SWR
intervals are the `SWR_event` labels of the cached
`{epoch_identifier_HPC}_80_data_over_time.pkl` frame, speed-gated by the read
(`analysed_time.with_swr_source`) and reduced to their sampled bounds by
`intervals_from_labels`, as the hippocampal event table reads them. The SWR split is that table's own `is_swr` column — strict
positive-duration overlap with those same labels — so a Figure 4 event and a
content- or rate-family event with the same `event_id` are one event, classified
one way; `data["ripple_times"]` no longer enters any fit path. Events are
restricted **by onset**: an event enters when its onset lies inside a kept stop,
applied *before* the split is read so `non_local_SWR` and `non_local_no_SWR`
partition exactly the kept events. Any canonical event or SWR interval falling
outside the aligned position frame is counted (`n_events_outside_frame`,
`n_swr_intervals_outside_frame`) and dropped rather than silently clipped.
The regressor basis still spans ±250 ms around each kept event, so an
event dropped just outside a stop's leading edge could have covered bins that
*are* fit; those bins are fit with its contribution zero, and the count of such
events is written to every provenance sidecar
(`n_{nonlocal,ripple}_dropped_overlapping_fit`) rather than assumed away.
Measured on 2026-09-14 on the 71-session bundles of that date, before the
artifact rule existed (the PFC cohort is now <!--v:mpfc.sessions-->75<!--/v-->
sessions), that was **6 non-local events and 4 SWRs** corpus-wide out of the
15,143 and 11,997 that were kept; the bin mask kept **92.3%** of the immobile
bins (19,732,623 of 21,384,252), and the onset rule kept **91.9%** of the
non-local events (15,143 of 16,477) and 93.6% of the SWRs (11,997 of 12,821);
the per-session table is in
[`2026-09-14-pfc-atwell-cohort.md`](../../.claude/docs/reviews/2026-09-14-pfc-atwell-cohort.md);
those counts were measured on bundles cut with the earlier run-time event
construction, which on that population labelled the same events the same way —
identical event count and identical SWR-coupled count in **71 of 71** sessions,
median per-session SWR fraction **0.447** either way, and an identical SWR
population wherever the two constructions were run side by side (measured
2026-09-14 with `scripts/pfc/diagnose_pfc_event_definition.py`, before the
artifact rule existed).
Reading the canonical table therefore buys traceability, not a different split.
The script recomputes the comparison for any session; its old construction
applies no analysed-time rule, so in a session where that rule drops an event
the two now differ for that reason alone. Re-run on 2026-11-12 on
`wilbur20210329_02_r1`, it found the same 212 events under both, 84 of them
SWR-coupled, none reclassified.
The unit GLM also draws a **matched non-event control** from that same fit
mask, one window per kept non-local event: the control onset is drawn uniformly
from the legal sample times of the *same* contiguous run of fitted bins the
event's onset falls in. Its duration-matching window must lie wholly inside
that run and overlap no real event of any class (`SWR` included), and no two
controls use the same onset bin. Control duration windows may overlap one
another: duration constrains placement, but the fitted regressor contains only
the distinct onsets, not an event-in-progress box. It preserves its source
event's duration, so the
comparison `matched_control + non_local_no_SWR vs. matched_control` asks whether
non-SWR replay modulates firing beyond a duration-matched non-event baseline
sampled from the same bins — the question a replay regressor on its own cannot
separate from "the animal is in the stretch of immobility replay happens in".
An event whose bout holds no free window of its duration is dropped rather than
placed elsewhere, so the control count is `<=` the event count; the draw's seed
(a hash of the session key) and both counts are written to every provenance
sidecar (`matched_control_seed`, `n_matched_control`,
`n_matched_control_unplaced`).

Three properties of that control decide what the comparison can be read as.

- **It is matched one-per-non-local event, over both classes**, not against the
  non-SWR class alone, so the control regressor generally carries *more* windows
  than the class it is compared with (one smoked session: 275 controls against
  187 non-SWR events, from 277 kept events). The two regressors are therefore
  not equal-n.
- **The preserved duration governs only where a control may be placed** — it
  sets the window that must fit inside the bout and clear every real event.
  Legal onset bins are enumerated directly, rather than found by rejection;
  control onsets are unique even though their auxiliary duration windows may
  overlap. The regressor itself is built from onsets alone,
  for every event class: each class's indicator is a spike-count-per-bin of its
  `start_time`s convolved with the lagged spline basis, so a window's length
  never enters the design.
- **Control onsets are uniform over the fitted bout, while real replay clusters
  after arrival**, so the control matches bout membership and duration but not
  time-since-arrival; `matched_control vs. constant` is the diagnostic for that
  residual — it asks whether the baseline itself modulates firing. And because
  a draw is rejected where real events already sit, controls are pushed into the
  event-free stretches of a bout, which in a short event-dense bout is
  *anti-conservative* for the engagement test. The pair is also not
  SWR-controlled: the adjudicating rung
  `SWR + matched_control + non_local_no_SWR vs. SWR + matched_control` is not
  fit yet.

Stage 6
(`behavior/choice/run_stay_leave.py`) applies `select_stop_cohort` at the stop grain before
its per-stop summary, and Stage 5B (`behavior/choice/run_retrospective_alignment.py`) takes its
at-well membership from `select_event_cohort` at the event grain (keeping its
finite-`occupancy_well_arm` conjunct). All three carry `--stop-cohort`; only the
unit GLM and Stage 6 offer `all_stops`.

**The PFC path aligns the decode to the position frame first.** It is the only
consumer that reloads `load_data` alongside a saved decode, and decodes written
before `get_position_info` was trimmed carry more time bins than `load_data` now
returns for the same epoch. `analysis.hippocampus.data_aggregation.align_results_to_position`
intersects the two axes on exact timestamps and accepts the result only when the
mismatch is a head/tail trim within budget (600 s on the results axis, 0.1 s on
the position axis); interleaved mismatches and drift inside the shared run still
raise, and the restricted pair goes to the same strict 1-to-1 validator. On
matched axes it is the identity. What it trimmed is recorded per session under
`cohort_counts.alignment` in the sidecar.

**Why at-well is the shared cohort.** The rate ladders used to fit every
immobility stop while the content ladders fit only at-well events, so "non-SWR
replay is not reward-gated" (a rate claim) and every content claim about
non-SWR replay were statements about different populations. And the difference
is not neutral: measured on 2026-09-12 on the tables of that date (no script
re-derives it), in-transit stops held **7.0%** of the corpus's immobility time
(9,093 s of 129,997 s) but **11.9%** of the non-SWR non-local onsets (4,690 of
39,510) and only **3.3%** of the SWR-coupled ones (859 of 26,403). Per second
of immobility that is **1.79x** the well-anchored non-SWR onset rate (0.516 vs
0.288 Hz) and **0.45x** the well-anchored SWR-coupled rate (0.094 vs 0.212 Hz).
Pooling those stops into the rate family and not the content family therefore
moves exactly the SWR-vs-non-SWR contrast the two families are compared on.

The all-stops fits remain available as a sensitivity bundle under `_allstops`,
and in-transit stops may be fit as their own bundle under `_intransit`, but an
in-transit population is never pooled with a well-anchored one. The in-transit
population is thin: 50 of the 218 sessions of September 2026 had any (a dated
count, not computed by a summary).

The restriction is not balanced across the SWR contrast (in the same 2026-09-12
measurement, 3.3% of SWR events were off-well against 13.0% of non-SWR events) — expected, since SWR events
concentrate at wells — so the cohort belongs in Methods.

**The estimator: a penalised Poisson GLM.** The mPFC unit GLMs are fit by a
batched Poisson IRLS in JAX, one call per model for every unit of a session,
with a Gaussian prior on the event-basis coefficients and the intercept left
free. The prior is a stated scientific assumption, not a regularisation
convenience. It exists because 23% of units had at least one model whose
unpenalised fit *separates* (measured on 2026-09-15 on the maximum-likelihood
bundles of that date; no script re-derives it): the peri-event basis is a cubic B-spline with
50 ms knots, and a basis column whose support holds no spikes drives its
coefficient without bound. Such a fit has no maximum-likelihood estimate at
all, and the design stays full rank throughout, so no rank-deficiency gate can
see it.

The prior's job is to exclude implausible rates, not to shape plausible ones.
Its standard deviation is `DEFAULT_PENALTY_SIGMA` in `analysis/pfc/batched_glm.py`
— **σ = 3** on the log-rate scale, per coefficient. What that says about a
*curve* is not that number: the basis is a partition of unity, so the prior on
the kernel at one lag is wider than σ by `sqrt(sum_b B_b(lag)^2)` and the prior
on a modulation held across the reported window is narrower by `sqrt(n_basis)`.
On the shipped 11-function basis those are
**<!--v:mpfc.prior_sd_per_lag-->2.07<!--/v--> per lag** (95 % of the prior mass
between a rate multiplier of **×<!--v:mpfc.prior_multiplier_per_lag_low-->0.017<!--/v-->
and ×<!--v:mpfc.prior_multiplier_per_lag_high-->57<!--/v-->** at one lag) and
**<!--v:mpfc.prior_sd_sustained-->0.90<!--/v--> sustained**
(**×<!--v:mpfc.prior_multiplier_sustained_low-->0.17<!--/v--> to
×<!--v:mpfc.prior_multiplier_sustained_high-->5.9<!--/v-->** held across the
reported window). Both are
computed by `implied_prior_sd` and recorded in every bundle's sidecar
(`penalty_sigma_log_rate`, `implied_prior_sd`); the figures here are the mpfc
summary's, read from those sidecars, not restated by hand.

Measured against that choice on 2026-09-15 by `scripts/pfc/compare_pfc_cohorts.py`
on the 71-session batch of that date (10,907 units past the 1 Hz gate): the corpus's own spread — the unpenalised peaks' standard
deviation with their estimation error deconvolved, divided by what one unit of σ
buys at one lag — implied **σ = 1.05** (SWR), **1.15** (non-SWR), **1.28**
(SWR-coupled replay) and **1.12** (the replay-split non-SWR curve) per
coefficient, and **0.66** for the matched control. The shipped prior is about
three times wider, which is the intent: it bound on **1.7 / 2.1 / 2.8 / 1.8 /
1.2 %** of well-posed units for those five curves, and the median peak moved by
0.009–0.028 on the log-rate scale against a standard error near 0.20–0.31. The audit reports that table beside the chosen value for whatever
tree it is run on; on the current bundles the prior binds on
<!--v:mpfc.prior_binds_percent_min-->2.6<!--/v-->–<!--v:mpfc.prior_binds_percent_max-->4.9<!--/v-->%
of well-posed units, depending on the curve. Under the prior every unit x
model converges; a unit that still fails is non-estimable and reported as NaN
everywhere, counted. Sensitivity bundles at other priors are written under
`_sigma<value>` names.

Each bundle also states **where the prior did the work**. Every curve-bearing
model is fit unpenalised as well, and each of the five curves carries the peak
of the penalised kernel, the peak the same coefficient block reaches
unpenalised, that peak's own standard error (the standard error of the kernel
at its peak lag, off-diagonal terms included — a diagonal `bse` cannot answer
it), whether the unpenalised fit ran away, and whether the two peaks differ by
more than the data can resolve. A unit whose unpenalised fit ran away has no
peak to difference against and is reported separately rather than counted as
bound.

**The inference: blocked held-out deviance against a matched-control null.**
Two things make the usual asymptotics unusable here. The fits are penalised, so
a likelihood ratio has no chi-square reference distribution at any stated
degrees of freedom; and spike counts in neighbouring 2 ms bins are not
independent, so an in-sample deviance drop is worth much less than its nominal
degrees of freedom suggest. The models are therefore compared by how well they
predict spikes they were not fit on.

**K = 5** folds are cut over *support-separated blocks* of the fit mask. Whole
contiguous runs are never split, and runs close enough for the ±250 ms event
kernel's supports to overlap are joined before assignment. A bin-wise split
would leave the training set 2 ms either side of every held-out bin; assigning
nearby runs separately could also let one event contribute to both train and
test. Blocks are visited in a seeded random order and each goes to the lightest
fold, so the folds carry
comparable bin counts; the assignment is seeded from the session name and
recorded (`n_folds`, `fold_seed`). Each model is fit on `K-1` folds under the
same prior and scored by the Poisson deviance of the fold left out, summed over
folds. A nested pair reports `heldout_delta_dev = D_ho(reduced) - D_ho(full)`,
positive meaning the added regressors predict held-out spikes better, and the
per-fold values beside it. The best model per unit is the one with the lowest
held-out deviance. No likelihood-ratio statistic, p-value or AIC is produced.

A held-out improvement has no analytic null, so one is drawn. The matched
non-event control already defines "a duration-matched non-event regressor drawn
from the fitted bouts"; drawing it `R` times, **per class, matched to that
class's own onset count and durations**, gives the distribution of improvements
a regressor of that size and shape with no event content buys. When an event has
no legal non-event window in its own fitted bout, one deterministic,
jointly-matchable subset is selected from timing and duration alone. The
observed test statistic and every null draw use exactly that fixed subset; the
all-event `heldout_delta_dev` remains separately available and is not used for
that p-value. A class's empirical p-value counts draws whose improvement matched
or beat the subset-matched observed one, with the observed statistic included
through the plus-one correction; the engagement statistic gets its own null the
same way, with the tested regressor replaced and the baseline left as it is.

**R = 1,000**. The finite-sample Monte Carlo p-value is `(r + 1)/(R + 1)`, so its
minimum with 1,000 usable draws is **1/1,001 ≈ 0.001** and zero is impossible. A draw
that fails to place every class-matched window is a thinner regressor and is
excluded. Production requires every requested draw to be finite for a unit's
p-value, and records that effective count per unit; the lower 20-draw floor is
retained only for deliberately shortened diagnostics.
`n_control_draws`, `control_draw_seed`, exact-draw counts, tested source indices,
and per-class total/tested/excluded window counts are in the sidecar. The
`control_null_event_policy` provenance field prevents older all-event/thinner-null
bundles from being pooled with this fixed-subset estimand.

**What the draw costs, measured** (`scripts/pfc/measure_pfc_null_geometry.py`;
the population numbers below are the mpfc summary's, the per-session ones are from a
run of the script on 2026-11-12). *Placement.* Control windows are found by
enumerating legal same-bout, real-event-free onset bins and requiring the onsets
to be distinct; the auxiliary duration-matching windows of two controls may
overlap one another, because duration only constrains where an onset may be
drawn and never enters the onset-only design. The former disjoint-window policy
made an exact count-matched draw impossible in dense bouts — 0/40 usable draws
for several classes on the first corrected smoke — which is the worse trade. The
price is a drawn regressor whose onsets can bunch where a real one's never do:
on three rebuilt sessions (`j1620210710_02_r1`, `chimi20200223_06_r3`,
`wilbur20210329_02_r1`; 20 draws each, 2026-11-12) the drawn sets' median
spacing is within about a third of the real class's (non-local 1,348 / 1,684 / 3,259 ms drawn against
1,252 / 1,317 / 2,704 ms real; SWR 1,692 / 3,684 / 5,366 against 1,667 / 2,798 /
5,688) but **1.5–6.2 %** of consecutive drawn onsets sit closer than the earlier
window's own duration, against **0 %** of real ones, and the closest drawn pair
is 5–47 ms apart against 36–68 ms real. *Engagement-null overlap.* The
engagement null's draws are independent of the shipped control they are entered
beside and may land on it, which two real classes never do: over the
<!--v:mpfc.sessions-->75<!--/v--> sessions they overlap it on a **median
<!--v:mpfc.engagement_null_overlap_median-->11 %<!--/v--> of their windows**
(<!--v:mpfc.engagement_null_overlap_session_min-->5<!--/v-->–<!--v:mpfc.engagement_null_overlap_session_max-->36 %<!--/v-->
across sessions; in the 2026-11-12 run one draw reached 47.4 %, worst session
`j1620210709_10_r5`). That much of the added regressor is a copy of its own
baseline, so the engagement p-value is anti-conservative by about that much.
*Units called without a gain.* A one-sided empirical p-value can be small while
the unit's own held-out improvement is zero or negative — the draws simply did
worse:
<!--v:mpfc.class_called_flat_fraction_min@pctnum-->4<!--/v-->–<!--v:mpfc.class_called_flat_fraction_max-->8 %<!--/v-->
of the units at raw p < .05 are in that state per event class (engagement 7.3 %,
259 of 3,531, in the 2026-11-12 run). They are evidence against a content-free
regressor of the same size, not evidence of out-of-fold prediction. *Events the
test could not use.* Corpus-wide
**<!--v:mpfc.unmatchable_percent_class_min-->0.27<!--/v-->–<!--v:mpfc.unmatchable_percent_class_max-->0.60<!--/v-->%**
of a class's events have no legal within-bout control and enter neither the
observed nor the null side; the worst single session/class
(`j1620210709_10_r5` / `non_local_SWR` in the 2026-11-12 run) excludes
<!--v:mpfc.unmatchable_worst_excluded-->8<!--/v--> of
<!--v:mpfc.unmatchable_worst_total-->102<!--/v-->, so a p-value there describes the
rest of that class's events.

The fraction of units with raw empirical `p < .05` is an explicitly
descriptive implementation preview. Units within a session and animal are not
independent replicates, so no Benjamini-Hochberg correction is applied across
them; the population inference that aggregates neuron → session → animal is a
separate, still-pending analysis. `analysis/pfc/load_population.py` refuses a bundle carrying
likelihood-ratio columns or a sidecar that does not state the corrected plus-one
Monte Carlo method and usable-draw floor.

**The replay-split curves were swapped until 2026-09-15.** The replay-split
design is concatenated `[constant, non_local_no_SWR, non_local_SWR]` — the
reverse of the order the model's *name* lists the classes — and
`process_neuron` read it by the name. So from 2026-04-24 (when the design gained
its second block in that order) until the fix, every bundle's
`non_local_SWR_modulation` carried the non-SWR kernel and
`non_local_no_SWR_modulation` the SWR-coupled one, and with them
`non_local_SWR_max_mod` / `non_local_no_SWR_split_max_mod` and their `_time`
twins. **Any per-class modulation magnitude quoted from the split model before
that date is mislabelled** and must be re-read from a bundle written after it;
this is the pair Q4 and Q5 are asked on. Whole-model comparisons never read a
block and are unaffected — the likelihood-ratio tests then, the held-out
deviances now — and so are the SWR-controlled model's two curves and the
engagement model's. The order is pinned by a regression test that plants two
kernels differing in lag and in sign, one per class, and asserts each shipped
curve recovers its own.

**History.** Before this, the models were fit by maximum likelihood — a
per-unit statsmodels IRLS, later a batched JAX one that handed separating fits
back to statsmodels — and tested by likelihood ratio with AIC model selection.
Over the 71 bundles fit that way on 2026-09-15, 22,902 of 154,350 neuron x
model fits (14.8%, over 4,640 of 17,150 neurons) ran to `maxiter` without
converging, and their likelihood-ratio tests and AICs were shipped like any
other fit's. That is the estimator this one replaces; those bundles and the
audits of them are kept as the record.

## Covariate causal hygiene

Every ladder runner enforces covariate timing at model-build via
`analysis.behavior.covariate_timing.assert_predictor_timing`; a rung whose
predictor is post-outcome or future-referencing must be **declared
descriptive**, else the run fails.

- **Per-stop `R1`** uses the reward-history basis over the outcomes physically
  known at bout start, instead of a current-outcome flag. The newest known
  outcome is not a fixed current/previous-trial relabeling (see
  [`../rate_models.md`](../rate_models.md)).
- **Descriptive rungs** are retained under an explicit *association* /
  *peri-switch alignment* claim, **not** a forward prediction. The allowlists
  are `DESCRIPTIVE_CONTENT_RUNGS` and `DESCRIPTIVE_RATE_RUNGS` in
  `model_specs.py`, and the catalog enumerates them under "Descriptive rungs".
  A new rung that introduces a leaky covariate but is not on an allowlist fails
  the timing assertion rather than being silently promoted to a predictive
  claim. Two kinds of leak are covered:
  - **Future-referencing.** `periswitch_index_trials` and `is_periswitch_post`
    sign toward the *next* switch, so they can only ever be descriptive.
  - **Post-outcome.** `rpe_postoutcome` and
    `reward_belief_mean_update_l1_all_ports_postoutcome` embed the current
    trial's reward. They were relabeled descriptive here rather than gated.
- **Never-visited encoding.** An unvisited leaf's
  `reward_fraction_candidate_arm_last10_visits` is `0.0` (not NaN), while
  `is_candidate_arm_never_visited` explicitly identifies novelty. The two enter
  together wherever recency/reward history is modeled.
- The stay/leave model `is_stay ~ outcome * content`
  (`scripts/behavior/choice/run_stay_leave.py`) legitimately uses the post-outcome
  `is_omitted_outcome`: its modeled unit (the stay/leave decision) is
  post-outcome by construction, so the timing gate does not apply.

## Cohort-audit tables

The **primary** fit runners emit a `<run>_cohort_audit.csv` next to their
outputs: **one row per (model, population)** stating who entered the model and
who was dropped, so complete-case filtering is visible rather than implied.
Covered runners: `fit_replay_content_models.py` (content base ladder, common
cohort), `fit_swr_contrast_content.py` and `fit_swr_contrast_rate.py` (the
`is_swr`-contrast ladders), `fit_per_stop_rate_models.py` (Family A) and
`fit_per_bin_rate_models.py` (Family B). The emitter is
`analysis.behavior.cohort_audit`. Counts are in the row's `grain` (`event` for
content, `stop` for the rate families). Drops are **mutually exclusive**,
assigned by priority:

`n_drop_qvals_gap` (mechanism B — the stop has no `qvals_table` row, the
unmapped stop tail, and one of this model's own qvals-sourced covariates is NaN
there; decided per row, so a qvals column masked on a stop that does have a row
is not the gap) → `n_drop_own_predictor_missing` (this model's own predictor NaN
otherwise, e.g. an early-session reward-history gap) → `n_drop_nonfinite` (a required
value is present but invalid, such as ±Inf) → `n_drop_zero_response`
(Σcount == 0) → `n_drop_common_cohort_other` (union complete-case: dropped only
for *another* rung's predictor) → `n_excluded_session` (data-level
`min_*`/rank gates).

**The rate families populate `n_drop_common_cohort_other`.** Both rate runners
freeze one cohort per fitted unit before any rung is fit —
`freeze_per_bin_cohort` over the union of every `MODEL_PER_BIN_RATE` design
column (per session × population, `exposure > 0`) and `freeze_common_cohort`
over the union of every `MODEL_PER_STOP_RATE` predictor (per session, positive
`immobility_duration_s` offset) — and the fitters raise on a residual NaN
instead of dropping rows, so a union drop is a recorded number rather than an
invisible per-rung `dropna`. Read the column as "complete for this rung, dropped
for another rung's predictor". For the **shipped** Family-A and Family-B
ladders no rung needs a covariate another rung lacks, so the frozen cohort is
simply every stop with a positive immobility offset and the column is zero on
every row (`models_per_stop_rate/all_models_cohort_audit.csv`,
`models_per_bin_rate/per_bin_rate_cohort_audit.csv`). A non-zero value appears
where rungs genuinely differ in what they need — the `is_swr`-contrast rate
ladder (`swr_contrast_rate_cohort_audit.csv`), whose value-covariate rungs also
charge their missing tail to `n_drop_qvals_gap` before the union is taken, as
does `fit_uncertainty_rate.py`.

**Reconciliation invariant (asserted, fail-loud):**
`n_in − Σ n_drop_* == n_modeled`. Convergence/separation counts
(`n_sessions_flagged_*`) and the `|coef| >= 10` quasi-separation counts (the
separate `<run>_term_diagnostics.csv`) are **session/term-level** and excluded
from the reconciliation. `scripts/checks/summarize_cohort_audits.py` stacks every
audit CSV and flags low-retention (`n_modeled/n_in < 0.9`) or biased
(mechanism-A early / mechanism-B qvals-tail) drops.

Also covered: `compare_event_source_models.py` maps its native
`_freeze_common_cohort` provenance (the content / patch / rate / **choice**
blocks) onto the standardized schema via
`records_from_common_cohort_provenance`, writing
`*_comparison_fits_cohort_audit.csv`. **This audit is block-grained, not
per-rung:** every model in a block shares ONE frozen common cohort, so it emits
one row per block with `model` = the block/analysis name (`choice`,
`choice_swr_split`, `patch_hierarchy`, …) rather than duplicating identical
counts per rung. The block's up-front selection — e.g. choice's
valid-replay-stop filter — is *definitional*, so `n_in` is the post-selection
candidate count. The qvals-merging auxiliary runners `fit_uncertainty_rate.py`
(inner join → `n_drop_qvals_gap`) and `fit_swr_power_model.py` (left join +
per-session dropna on qvals-derived predictors → `n_drop_qvals_gap`) each emit
their own cohort audit. `fit_cross_history_rate.py` (a bivariate
self-/cross-history variant with no new qvals dependence) is **not** audited —
documented, not silently dropped.
