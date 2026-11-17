# Deferred validation: what would license a performance claim

*Design spec. Status: **parked**, not scheduled and not a prerequisite for the initial
descriptive report.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

## Purpose and trigger for reopening

[03](03-inventory-comparison.md) and [04](04-event-features.md) establish overlap,
differences in the events, and reasons for selection under the chosen definitions.
Those results can characterize practical strengths and limitations without identifying
a universally best detector.

Claims about greater sensitivity, detection precision, boundary accuracy, specificity,
or calibration require additional validation appropriate to the target. More events,
smoother trajectories, or larger posteriors do not establish those claims. Prior-excluded
likelihood on the same spikes is a diagnostic, not independent validation, and an
interval with no detected event is not a known negative.

Reopen this work only when a particular scientific or reviewer question requires a
performance claim or independent evidence. Define the target, reference, scoring rule,
and feasible scope at that time. The studies below are options with different limits;
neither automatically supplies every performance measure.

Threshold robustness is a separate, planned follow-up in 04. It describes dependence on
settings and does not require either validation study. It also does not substitute for
them.

## Simulation with planted activity

Plant traveling sequences, stationary remote representation, fragmented activity,
local representation, and no-sequence activity, varying duration and spike counts.
Include generators that do not reproduce our transition assumptions. Score non-local
representation and extended traveling sequences separately so methods are not penalized
for activity outside their stated target.

Known event timing allows boundary errors, merges, and splits to be measured; known
positive and negative examples permit detection-performance estimates. Those estimates
apply to the tested generators, not automatically to the recordings. Choose a small
study answering the disputed mechanism before considering a broader benchmark.

## Held-out neural evidence

Separate discovery tetrodes from evaluation tetrodes and rerun every spike-dependent
selection and trajectory-estimation step using discovery data alone. Fit or tune
transitions and thresholds without the evaluation spikes. Document the LFP channels
used by SWR triggers and distinguish LFP-triggered selection from spike-based selection.

Predefine a common evaluation score for all inventories. Compare discovery trajectories
with held-out neural evidence, accounting for coverage, duration, and spike counts.
Positive held-out evidence tests generalization across recorded neurons; its absence
does not establish a false event when held-out spatial coverage is poor. This study
alone does not supply real-data sensitivity without known missed events.

A first version of the rule has been tried on simulated data, where the truth is known
(02 §"Held-out evidence on simulated data"; design in
[memoryless_simulation_controls.md](memoryless_simulation_controls.md)): expected
Poisson log-likelihood of the evaluation spikes under a discovery content posterior,
spike and silence terms apart, against the animal's own position and permuted
evaluation fields, with a coverage floor. It is implemented as
`detector_comparison.heldout` and can be applied to recorded tetrodes unchanged. On the
simulation it supports <!--v:detection.heldout_ours_only_support_fold_c-->79<!--/v-->–<!--v:detection.heldout_ours_only_support_max-->90<!--/v-->%
of the informative true replays only our detector reported and <!--v:detection.heldout_placeless_supported_events-->0<!--/v--> of
<!--v:detection.heldout_placeless_informative_events-->44<!--/v--> informative place-less events
(<!--v:detection.heldout_placeless_loose_supported_events-->1<!--/v--> of <!--v:detection.heldout_placeless_loose_events-->106<!--/v-->
at a looser coverage floor); no event of pure local coding
reached it, and most short events are uninformative.

Before scheduling a study, resolve these issues on a pilot:

- Reduced discovery data change the detected population, so results do not transfer
  automatically to the full-data cohorts.
- Held-out tetrodes may provide little information about some locations.
- Silence has a spatially varying likelihood through ground-process intensity;
  normalizing a posterior does not remove that effect. A score must state whether and
  how silence contributes, and how extreme scores are handled.
- A shuffle or shift null needs an explicit operation and exchangeability rationale;
  no scoring formula or null is committed by this parked spec.
- Every method must use the same folds, eligible sessions, and evaluation rule. Report
  failures and missing coverage rather than selecting favorable folds.

## Reporting boundary

The initial report states that its comparison is descriptive and distinguishes
algorithmic selection properties from measured detection performance. A later validation
study has its own configuration, results, and limitations; it does not retroactively
turn the initial overlap comparison into a performance benchmark.
