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
dominant missing source (7,315 of 59,843 well-anchored events, 12.2 %); a
leaf-weighted mark is also undefined, rarely, when the arm-level source is
itself undefined for the event on every leaf (32 well-anchored events with
leaf content for the Beta belief, 0.05 %), so not every missing value is an
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
construction (median within-session × class r = 0.68 on the canonical table),
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
([audits/2026-09-18-pfc-marked-family-smoke.md](../audits/2026-09-18-pfc-marked-family-smoke.md))
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
- `.{class}.{target}.slope.npy`: each target's one-SD modulation kernel; and
- `.provenance.json`: exact inputs, hashes, model definition, mark scales,
  ranks, timing, device, eligibility, and NaN/convergence counts.

The provenance file is written last and is the bundle commit marker. A stale,
partial, or mismatched bundle is removed before refitting. **No production
bundle has been fit yet.**
