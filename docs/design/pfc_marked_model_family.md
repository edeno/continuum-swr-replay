# Descriptive mPFC marked-kernel model family

## Question and scope

The existing PFC analysis shows that mPFC units change firing around both
SWR-coupled and non-SWR replay onsets. The single-mark extension asks whether
that response changes with current-arm content. This model family broadens the
question: **which properties of an individual replay event change the size or
timing of an mPFC unit's peri-event response?**

This is not an mPFC population decoder. It uses event properties already in the
canonical hippocampal replay tables as continuous marks on the temporal GLM.
It also is not a neuron-selection pipeline: every recorded unit is fit, and no
p-value, firing-rate cutoff, model choice, or sign filters a later population.

## Shared temporal estimand

For each replay class (`non_local_no_SWR` and `non_local_SWR`) the full model
contains:

1. an ordinary peri-onset spline kernel;
2. one spline kernel per adjustment mark; and
3. one spline kernel per target mark.

Each raw mark is centered and divided by its population SD separately within
session and replay class. A target curve therefore says how the unit's
log-firing-rate response changes, lag by lag, for a one-SD change in that event
property. Both replay classes and every mark in a named family are fit jointly.
The ordinary kernel is the response to an event whose marks are at their
within-class means.

**Per-stop baseline (bundle contract 3).** Every design also carries
per-stop columns right after the intercept
(`marked_glm.build_baseline_columns`): one indicator per kept well other than
the reference well (well 0, or the session's lowest kept well when well 0 has
no fitted bin), `well_unknown` (the stop has no `occupancy_well_id`: a mixed
stop of the well-anchored cohort; added only when the session has such a
fitted bin), `rewarded_outcome` (the stop's known outcome was rewarded) and
`outcome_unknown`. A column constant over the fitted bins is dropped and named
in the sidecar (`baseline_column_names` lists the kept ones). The columns are
free unless `--baseline-penalty-sigma` is given; the shipped follow-up run
gives them a prior SD of 3 log-rate units. These columns take up stationary
rate differences between stops, which a design with one intercept per session
would otherwise read as event-locked effects of a context mark (the
contract-1 corpus showed reward, switch and previous-arm "effects" as large
200 ms before onset as after). Because mixed stops have their own level, the
intercept is the reference well's level and the typical-event kernel is the
response at the reference well after an unrewarded, known outcome. Contract 2
lacked `well_unknown`: mixed-stop bins (a median 43 % of fitted bins over the
75 sessions of `marked_family_baseline_prior/`, measured 2026-10-29; no script
re-derives it) were coded exactly like the
reference well, so contract-2 typical kernels mix the two levels. A model
whose target is a stop-level context mark other than the reward outcome
(`trips_to_patch_switch`, `next_trip_switches_patch`) adds that mark as a
baseline column too, z-scored over the fitted bins with an `_unknown`
indicator. `--no-baseline` fits the
contract-1 single-intercept designs.

For each class × target, the descriptive comparison removes only that target's
spline block and refits the model on the same blocked folds. The reported
held-out deviance change is predictive evidence that the target carries unique
information beyond the named adjustments. It is not converted to a p-value.

## Derived content marks

The primary content allocation is the shipped MAP-based `arm_fraction`; the
posterior-mass allocation remains a later sensitivity, not a second default
family.

- `current_arm_fraction`: content on the animal's occupied arm.
- `previous_arm_fraction`: content on the preceding trip's origin arm.
- `current_patch_fraction`: content on the three arms in the occupied patch.
- `leaf_content_fraction`: total content on the six reward-bearing leaf arms.
- `content_weighted_belief_uncertainty`: the replay-weighted mean of
  `reward_belief_variance_candidate_arm_preoutcome` over leaf arms.
- `content_weighted_belief_value`: the corresponding Beta belief mean.
- `content_weighted_recent_reward`: the replay-weighted recent rewarded
  fraction for represented leaf arms.
- `content_weighted_relative_recency`: the replay-weighted, within-event
  leaf-centered visit recency. Never-visited leaf arms are excluded from this
  mean (numerator and denominator): their recency column carries the
  right-censoring fill, not a measured recency, exactly as
  `center_visit_recency_within_event` excludes them upstream. An event whose
  represented leaf mass is entirely on never-visited arms has undefined
  recency; `content_weighted_novelty_fraction` carries that mass.
- `content_weighted_novelty_fraction`: represented leaf mass on never-visited
  arms.

The last five marks divide by represented **leaf** mass. Trunk arms are excluded
before the numerator and denominator are formed; their structural zeros never
become zero uncertainty, value, reward, or recency. An all-trunk event has an
undefined leaf-content mark and stays in the ordinary event population while
contributing zero to that centered marked regressor. All-trunk events are the
dominant missing source (7,315 of 59,843 well-anchored events, 12.2 %, measured
2026-09-18 on the tables of that date; no script re-derives these counts); a
leaf-weighted mark is also undefined, rarely, when the arm-level source is
itself undefined for the event on every leaf (32 well-anchored events with
leaf content for the Beta belief, 0.05 %, same measurement), so not every missing value is an
all-trunk event.
`content_weighted_recent_reward` uses the source column's convention for a
never-visited arm, which is `0.0` (zero rewards from zero visits); the novelty
mark carries the never-visited state separately. `leaf_content_fraction`
is included separately so the model can distinguish *which leaf content* was
represented from *how much leaf content* the event contained.

Event duration uses the canonical inclusive `event_duration_s` (end minus
start plus one 2 ms decode bin), not a recomputed `end - start` approximation.

## Recommended models

All marks named in one row are fit jointly. “Targets” receive leave-one-block-
out held-out comparisons; “adjustments” remain in every comparison. Mean
posterior entropy is a decode-quality adjustment, not a decoded mPFC response.

| model | targets | adjustments | question |
|---|---|---|---|
| `spatial_role` | current arm, previous arm, current patch | duration, leaf mass, posterior entropy | Is the response tied to where content sits relative to the animal? |
| `belief_content` | represented-arm uncertainty, represented-arm value | duration, leaf mass, current-arm content, recent reward, relative recency, novelty, posterior entropy | Does uncertainty matter beyond value, spatial role, reward/visit history, content form, and decode quality? |
| `reward_memory` | recent reward, relative recency | duration, leaf mass, current-arm content, uncertainty, value, novelty, posterior entropy | Are reward-memory effects separable from spatial role and the current belief state? |
| `novelty_content` | represented never-visited fraction | duration, leaf mass, current-arm content, uncertainty, value, recent reward, relative recency, posterior entropy | Does novelty matter beyond graded recency and the other content axes? |
| `trajectory_geometry` | log spatial extent, log reversals, pause fraction, behind-minus-ahead extent | duration, posterior entropy | Is the response sensitive to extended, fragmented, or retrospective trajectories? |
| `within_stop_timing` | log time since stop onset | duration, current-arm content, leaf mass, value, posterior entropy | Does the neural response itself evolve over the stop? |
| `switch_context` | last trip switched patch | duration, current-arm content, value, posterior entropy, rewarded context | Does exploration change the average replay response beyond reward? |
| `reward_context` | latest known outcome was rewarded | duration, current-arm content, value, posterior entropy, switch context | Does reward change the average response beyond exploration? |
| `graded_coupling` | ripple power, hippocampal MUA, theta power | duration, leaf mass, posterior entropy | Is the binary SWR split a thresholded view of continuous coupling? |

An adjustment that is constant within a session/class is represented by an
all-zero nuisance block: there is no within-class confounding variation to
adjust. Every mark in a model, adjustment or target, must have at least ten
finite events in both replay classes, and a **target** must additionally have
nonzero variation in both. An ineligible session × model gets a provenance
marker, not a partial fit.

In `spatial_role`, `current_patch_fraction` contains `current_arm_fraction` by
construction (median within-session × class r = 0.68 on the canonical table,
measured 2026-09-18; no script re-derives it),
so its two leave-one-block-out comparisons are not independent evidence and
must be read together.

Three optional interaction sensitivities are available with
`--include-interactions`:

- `exploration_content_interaction`: current-arm content × patch switch and
  uncertainty × patch switch;
- `reward_content_interaction`: the analogous known-reward interactions; and
- `coupling_content_interaction`: the analogous ripple-power interactions.

Each interaction includes both constituent main-effect kernels plus duration,
leaf mass, value, and posterior entropy. The interaction mark is the product of
the two raw marks, standardized afterwards; with a 0/1 modifier the content
main-effect kernel in an interaction model is therefore the response at
modifier = 0 (non-switch, unrewarded), not at the class mean. The
context/coupling modifier is required to vary; this prevents an interaction
that collapses onto its content main effect from being interpreted as
separately identified.

## Follow-up models

`FOLLOWUP_MODEL_SPECS` (`--model-set followup`) asks three questions of the
two replay classes and one of every class, SWRs without replay included when
the session is cut with `--event-set with_swr_without_replay`:

| model | targets | adjustments | classes |
|---|---|---|---|
| `replay_content` | `current_arm_fraction`, `content_weighted_recent_reward`, `content_weighted_belief_uncertainty` | `log_event_duration_s`, `leaf_content_fraction`, `posterior_entropy_mean` | replay |
| `event_strength` | `log_event_duration_s`, `ripple_power_z`, `mua_rate_z` | `posterior_entropy_mean` | replay |
| `stop_context` | `rewarded_context`, `trips_to_patch_switch` (also a baseline column) | `log_event_duration_s`, `current_arm_fraction`, `posterior_entropy_mean` | replay |
| `ripple_power` | `ripple_power_z` | `log_event_duration_s` | replay + SWRs without replay |
| `ripple_power_pooled` | `ripple_power_z`, one block shared by every class | `log_event_duration_s` | replay + SWRs without replay |

The "recently rewarded arm" is `content_weighted_recent_reward` (each arm's
reward fraction over its last ten visits) and the uncertain arm
`content_weighted_belief_uncertainty`. The SWRs without replay carry only
duration and ripple power; their `ripple_power_z` is the event table's rule,
the mean of the decode frame's `SWR_zscore` over the SWR's own bins
(`canonical_events.event_mean_ripple_power`, which reproduces the event
table's column exactly), and their duration is the sampled span plus one bin.
Ripple power is not matched between classes: the models measure how the
response scales with it in each class (`ripple_power`) and overall
(`ripple_power_pooled`: each event's mark is standardized within its class,
then every class's events feed one marked block, the sum of the per-class
blocks; each class keeps its own base kernel and duration block). A class a
session cannot carry for a model -- fewer than ten finite events of a mark
(for example fewer than ten SWRs without replay), a constant target, or a
two-level mark with fewer than ten events at a level -- is dropped from that
model in that session and named in the sidecar (`dropped_event_classes`); the
model is ineligible only when no class is left.

`trips_to_patch_switch` (`marked_model_family.stop_patch_switch_marks`) is a
forward mark of the stop. The session's completed trials are rebuilt from the
stop table (`completed_trial_arm` of trial k agrees with the `next_arm` of the
stops after trial k-1 in every session); the trip that starts at a stop whose
last completed trial is k is trial k+1. The mark is the number of trips, from
that one on, that stay in the patch before the first that leaves it -- 0 when
the next trip leaves -- capped at 3, and undefined for a stop with no
completed trial and after the session's last switch.
`next_trip_switches_patch` is `trips_to_patch_switch == 0`.

## Marginal tier

`marginal_specs()` (`--tier marginal`, or `all` for joint then marginal)
generates one model per follow-up target, plus `next_trip_switches_patch`,
named `marginal_<mark>`: the target alone, adjusted for
`log_event_duration_s` and `posterior_entropy_mean` (duration itself only for
entropy), content targets also for `leaf_content_fraction`, on the replay
classes. A joint model credits variance its targets share to none of them; the
marginal tier credits it to each. The marginal tier is the family's total
effect and the joint tier its direct effect; neither supersedes the other, and
the summary prints them side by side.

## Slope reference

With `--slope-permutations R` (30 by default for the follow-up set) each
slope kernel gets a shuffled-mark reference: R full-data refits (no held-out
scoring, the observed fit's prior) in which only that target block is rebuilt
from the target shuffled within its class, every other block -- the other
classes' included -- keeping its observed values. The reference is drawn only
after the observed fit passes the runner's convergence gate.

The shuffle is residualized (`slope_permutation_method =
residualized_within_class`): within the class the target is regressed, with an
intercept, on the model's other marks (adjustments and other targets), and only
the residuals are permuted; each draw is fitted plus permuted residual,
re-standardized within the class as the observed mark is. Permuting the raw
target would decorrelate it from those marks and make the reference too
narrow by about `sqrt(1 - R^2)` (simulated at r = 0.75: estimator SD 0.0100,
residualized reference 0.0103, raw 0.0071). The target's R² on the other marks
is recorded per class. An event-level mark's residuals are permuted among the
events where it is defined. A stop-level mark (`rewarded_context`,
`trips_to_patch_switch`, `next_trip_switches_patch`) is regressed and permuted
at the stop level (a stop's other marks averaged over its events): each
stop's value moves to another stop and all its events take it, so the
reference keeps the within-stop dependence (a stop whose events disagree on
the value -- the latest known outcome can change inside a stop -- is split
into one unit per value). A pooled block is shuffled within each class. Each
model's draws use their own seed (session, model name and optional base
seed), so different models' references are independent.

The draws of a block are fit together in device batches
(`batched_glm.fit_poisson_column_variants`: the design stays on the device and
only the replaced columns cross per batch), each warm-started from the
observed fit's coefficients. The penalised objective is strictly concave, so
the warm start moves the iterations, not the answer: against one draw at a
time, cold, the curves agree to 1.5e-8.

The summary aggregates the observed and every shuffled population curve the
same way (unit -> session -> animal, as every population quantity in the
summary is). The call to quote is the max statistic: the observed largest
`|curve|` over the ±200 ms window against each draw's largest `|curve|`,
outside when the plus-one p `(1 + #draws >= observed) / (R + 1)` is at most
0.05. (Comparing with the interpolated 95th percentile of 30 draws instead
would call about 8 % of exchangeable observations outside.)

Under the shipped mark prior the target's slope picks up part of a correlated
mark's effect (ridge leakage). Simulated with the target correlated r 0.75
with an adjustment that carries a planted kernel and no target effect (120
sessions, 30 draws): the target's window mean is +0.0088 ± 0.0010 and its peak
+0.023 ± 0.003 at σ_mark = 0.5, and −0.0012 ± 0.0016 at σ_mark = 3. The
residualized reference carries the same leakage (its window mean +0.0086 ±
0.0002), so the call stays calibrated: 5 of 120 sessions outside at p ≤ 0.05.
A raw-permutation reference does not (18 of 40 outside). A small slope on a
target strongly correlated with another mark should therefore be read with
the reference, not against zero.
The per-lag 95 % band at the observed peak lag (overall and per animal) is
descriptive, since the peak lag is selected as the most extreme. In
simulation the stop-shuffled reference of `rewarded_context` was well
calibrated with no stationary offset between stops (z RMS about 0.8 over 24
fits) and somewhat narrow when the baseline absorbed a ×1.5 offset (about 1.2
over 16 fits, one |z| of 3.0). The reference describes the population curve;
it is not a per-unit test.

## Missing values and denominators

Undefined marks never remove an onset from the ordinary kernel. Standardization
uses the finite events for that mark, records the finite/missing counts and raw
mean/SD, and assigns zero only to the centered marked regressor for an
undefined event. A model requires at least ten finite events per mark per class.
The provenance sidecar records every scale and availability count. Downstream
summaries must report finite denominators per comparison; they must not silently
substitute the total unit count.

## Performance contract

Six compact core models have six marked predictors per class and a 155-column
full design on the current 11-basis representation. The three content
horse-races and the three interactions each have nine marked predictors per
class and a 221-column full design. The full family therefore uses only two
compiled widths, while the uncertainty/value, reward/recency, and novelty
comparisons retain the spatial, novelty/recency, and decode-quality controls
needed for their scientific interpretation.

The held-out scorer bounds `design_batch_size × n_features` near the workload
of the measured four-design, 67-column single-mark profile (about 34 GB on the
largest profiled session). A 7.5% headroom permits two 144-column compact-core
reductions per batch; 210-column wide-tier reductions use one. Wider designs
therefore cannot inherit the narrow model's four-design batch and multiply the
tall QR working set. Session neural data and event marks are loaded once; the
fold labels and spike-count response are transferred once and kept resident on
the device across every requested family. There are no
permutation refits in this family.

The following profile is a one-off measurement from the development session;
no preflight script or log for it is in the repository, so it is not
reproducible from the tree and no production bundle exists yet. On the 344-unit
Wilbur 2021-03-29 `02_r1` real-session gate, a four-comparison
155-column compact-tier preflight used about 33.7 GB on an otherwise idle 80 GB
A100; held-out scoring took 206 s and the full-data fit 22 s (first call
includes JIT compilation). Batch size 1 and 2 produced bit-identical tables and
curves. The current 221-column `coupling_content_interaction` model used about 36.2 GB,
scored held-out data in 460 s and fit the full data in 34 s; its design was full
rank (221/221). This cost is why interactions are opt-in.

## What the first smoke showed

The three-session GPU smoke
([2026-09-18-pfc-marked-family-smoke.md](../../.claude/docs/reviews/2026-09-18-pfc-marked-family-smoke.md))
ran end to end, but it also showed that a family bundle's `typical` curve is
**not** the Figure 4 kernel: with the base kernels alone the family reproduces
the canonical curves exactly, and adding the mark blocks under the σ = 3 prior
inflates the base peaks by about 25 % (IQR 1.0–1.6) whether the marks are
real or permuted within class, and lets held-out predictions blow up on
dozens to hundreds of units per model on the larger sessions. The marked
blocks therefore carry their own prior, `DEFAULT_MARK_PENALTY_SIGMA = 0.5`
(`--mark-penalty-sigma`), while the base kernels keep σ = 3. Total held-out
deviance improves monotonically as that prior tightens and never beats the
base-only model on the smoke sessions, so 0.5 is a stated operating point
rather than an optimum: it restores the canonical base kernels within 2 %,
removes the blow-ups, and still admits a per-SD modulation of roughly ×0.7 to
×1.4. Read target slopes only with their held-out delta and never quote a
`typical` curve as a Figure 4 modulation; the two now agree closely but are
not the same fit.

## Artifacts

`scripts/pfc/run_pfc_marked_model_family.py` writes one bundle per session ×
model under `Processed-Data/pfc_marked_model_family/`:

- `*_marked_family.pkl`: per-unit full/reduced held-out deviances, their
  difference, per-fold values, convergence flags, spike count, and firing rate;
- `.lag_time.npy`: the reported ±200 ms lag axis;
- `.{class}.typical.npy`: the typical-event kernel for each replay class;
- `.{class}.{target}.slope.npy`: each target's one-SD modulation kernel (the
  class is `pooled` for a pooled model);
- `.{class}.{target}.slope_null.npy`: with a slope reference, the shuffled
  refits' kernels, float32, shape `(R, 201, n_units)`; and
- `.provenance.json`: exact inputs, hashes, model definition, mark scales,
  ranks, timing, device, eligibility, and NaN/convergence counts.

The provenance file is written last and is the bundle commit marker. A stale,
partial, or mismatched bundle is removed before refitting. **The 71-session
corpus was fit 2026-09-19** at git `4d69b83` (on `feature/pfc-marked-event-kernels`;
its code reached this branch in `cc04527d`) (816 bundles, 36 ineligible
markers): [2026-09-19-pfc-marked-family-corpus.md](../../.claude/docs/reviews/2026-09-19-pfc-marked-family-corpus.md).
That corpus predates the analysed-time rule: its sessions were assembled
without excluded time and without the artifact-dropped events, on the
218-session tables of that date. The runner now assembles each session with
the canonical PFC assembly (`analysis/pfc/session.py`), so a refit's fit mask
and event onsets are the Figure 4 ones. Contract-2 bundles (per-stop baseline,
`tier`, `event_set`, the slope-reference keys) never match a contract-1
sidecar, and contract-3 bundles (`well_unknown` added) never match a
contract-2 one; the follow-up corpus is written to its own directory
(`pfc_response_followups/marked_family_well_unknown/` for contract 3, an
explicit `--output-dir`), and
`summarize_pfc_marked_family.py` refuses a directory that mixes contracts or
event sets. It writes `summary.parquet` there, one row per model, tier,
class, target and statistic, and its figures under
`paper/figures/pfc_marked_family/` (`--figure-dir`).
