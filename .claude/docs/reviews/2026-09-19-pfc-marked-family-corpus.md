# mPFC marked-model family: 71-session corpus batch (2026-09-18/19)

Branch `feature/pfc-marked-event-kernels` at `4d69b83`. Runner
`scripts/pfc/run_pfc_marked_model_family.py`, nine core models plus the three
interactions, base prior σ = 3, mark prior 0.5, K = 5 blocked folds, at-well
cohort. Three tmux workers on A100s 4, 7 and 9, sessions balanced by unit
count (24/23/24 sessions, 5,754–5,801 units each); launched 2026-09-18 19:34,
last worker finished 2026-09-19 about 10:10. Logs and per-worker session
lists under `logs/pfc_marked_family_20260918/`; output under
`Processed-Data/pfc_marked_model_family/`. Prior selection and the smoke that
motivated it: [2026-09-18-pfc-marked-family-smoke.md](2026-09-18-pfc-marked-family-smoke.md).

## Completeness

| quantity | value |
|---|---|
| sessions | 71 (all mPFC at-well sessions) |
| session × model pairs | 852 = 71 × 12 |
| complete bundles | 816 |
| ineligible markers | 36 |
| no-units markers, failed sessions | 0, 0 |
| distinct fit contracts across sidecars | 1 (σ 3.0 / 0.5, K = 5, window 0.25 s, git 4d69b83) |
| unit rows over all bundles | 195,700 |

Ineligible pairs by model: `novelty_content` 11, `reward_context` 8,
`reward_content_interaction` 8, `exploration_content_interaction` 7,
`switch_context` 2. By the mark that failed: `rewarded_context` constant in a
class 16, `content_weighted_novelty_fraction` constant 11,
`current_arm_fraction_x_switch` constant 6, `switch_patch_context` constant 3.
Every marker carries the model name and the reason, and the skip check
recognises them, so a rerun re-fits nothing.

## Fit health per model

Per bundle, a unit is a "blow-up" when any held-out delta exceeds 1,000 in
absolute value, and "NaN" when any comparison's delta is undefined (a fold or
the full-data fit did not converge). `frac_pos` is the mean over bundles of
the fraction of unit × comparison deltas that are positive; `med_delta` and
`p99_delta` are medians over bundles of the per-bundle median and 99th
percentile.

| model | bundles | units | blow-up units | NaN units | non-converged full fits | frac_pos | med_delta | p99_delta |
|---|---|---|---|---|---|---|---|---|
| spatial_role | 71 | 17,328 | 1 | 144 | 48 | 0.30 | −0.80 | 8.2 |
| belief_content | 71 | 17,328 | 0 | 144 | 48 | 0.31 | −0.67 | 10.1 |
| reward_memory | 71 | 17,328 | 0 | 144 | 48 | 0.29 | −0.66 | 8.9 |
| novelty_content | 60 | 14,881 | 2 | 130 | 44 | 0.40 | −0.03 | 13.0 |
| trajectory_geometry | 71 | 17,328 | 0 | 144 | 48 | 0.29 | −0.74 | 11.3 |
| within_stop_timing | 71 | 17,328 | 0 | 144 | 48 | 0.33 | −0.83 | 14.1 |
| switch_context | 69 | 16,651 | 0 | 145 | 47 | 0.37 | −0.43 | 13.6 |
| reward_context | 63 | 13,934 | 0 | 118 | 38 | 0.40 | −0.15 | 14.3 |
| graded_coupling | 71 | 17,328 | 1 | 144 | 48 | 0.29 | −0.92 | 9.9 |
| exploration_content_interaction | 64 | 15,004 | **37** | **452** | 40 | 0.32 | −0.23 | 6.2 |
| reward_content_interaction | 63 | 13,934 | 0 | 283 | 41 | 0.25 | −0.88 | 4.3 |
| coupling_content_interaction | 71 | 17,328 | 0 | 144 | 48 | 0.26 | −0.84 | 5.6 |

Reading this:

- The mark prior did what the smoke selection said it would. Across the nine
  core models, four units in 17,328 have an exploding delta, against 177 in
  one session alone at σ = 3. Non-converged full-data fits are 48 of 17,328
  (0.28 %), the same units in every model.
- Median deltas are slightly negative everywhere and a quarter to two fifths
  of unit × comparison deltas are positive, with 99th percentiles of 4 to 14
  deviance units. That is the shape expected when most units carry no effect
  of a given mark and a minority do; it is descriptive, and no p-value is
  attached.
- `exploration_content_interaction` is the exception: 37 blow-up units and
  452 NaN units. Its `current_arm_fraction_x_switch` mark is a fraction
  times a rare binary, so after within-class standardization a handful of
  switch events carry very large weights, and a fold that holds them out
  destabilises the held-out prediction. Any summary of this model must drop
  non-finite and exploding deltas with a stated rule, and the fix (a minimum
  count of the rare level for the modifier of an interaction) is a
  follow-up, not applied here.
- Downstream summaries must use each comparison's finite denominator, as
  the design doc requires; the NaN counts above are that denominator's
  complement.

## Reading the population summary

`scripts/pfc/summarize_pfc_marked_family.py` aggregates unit → session →
animal and writes
[2026-09-19-pfc-marked-family-summary.md](2026-09-19-pfc-marked-family-summary.md)
(tables, a CSV, and one figure per model under
`paper/figures/pfc_marked_family/`). Two numbers carry the reading. The
fraction of units whose held-out prediction improved with a target's kernel
is 0.24–0.41 for every target (animal-balanced median), with no reference
value: the marks add nothing in aggregate, and a quarter to two fifths of
units gain. And the population mean one-SD slope kernel is small everywhere,
|peak| 0.004–0.027 log rate per SD (0.4–2.7 % of rate), but for many targets
its sign is shared by 80–98 % of sessions and 5/5 animals.

The kernels split into two kinds, told apart by their value before onset
(lags ≤ −100 ms) against after it (0 to +150 ms):

- **Event-locked** (near zero before onset, non-zero after): more content on
  the current arm or patch lowers the response after onset by about 1.3–1.6 %
  per SD in both replay classes (5/5 animals, 85–92 % of sessions); the same
  current-arm effect is stronger at higher ripple power and after reward
  (interaction post-onset means −0.017); a trajectory that extends farther
  behind the animal than ahead raises the response by about 1 % per SD (5/5
  animals; the trajectory mark with the most gaining units, 0.34–0.36); and
  higher hippocampal multiunit activity around non-SWR replay lowers it
  (−0.014). Represented-arm uncertainty (+0.008 non-SWR) and time within the
  stop (−0.005) are event-locked but at the resolution floor.
- **Context offsets** (as large before onset as after): rewarded stops
  (+0.010 to +0.021 throughout), patch switches (−0.02), never-visited
  content (−0.02) and previous-arm content (−0.015 to −0.02). These are
  stationary rate differences between the contexts the events occur in, which
  the model cannot separate from an event-locked response because it has one
  intercept per session and no per-stop baseline terms. They are real for
  prediction (reward context and novelty have the most gaining units,
  0.36–0.41) but they are not evidence of replay-specific modulation until a
  per-stop baseline is in the model. The occupancy-well and reward-outcome
  baseline terms of the original design are exactly what would absorb them.

Represented-arm value, recency, recent reward, theta and ripple power as main
effects, spatial extent, and pauses show kernels under 1 % per SD or
inconsistent signs: no population-level effect at this resolution. Population
means average heterogeneous units, so single units can carry much larger
effects (95th-percentile held-out gains of 3–14 deviance units); no
permutation reference exists at the population level, and the tighter mark
prior shrinks every slope toward zero, so magnitudes are lower bounds.

## Follow-ups

1. Require a minimum count of each level of a binary modifier (say ten
   events per class) before an interaction with it is eligible; refit the
   interaction bundles that change.
2. A population-level permutation reference for the slope kernels (the
   single-mark runner's label shuffle exists only for current-arm content and
   was not run on the corpus), so that a 1 % consistent slope can be read
   against what shuffled marks give.
3. Per-stop baseline terms (occupancy well, reward outcome) in the base
   model, so that context offsets and event-locked responses are estimated
   separately; the context marks above are the test case.
