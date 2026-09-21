# Replay detection method comparison — overview

*Design spec, cross-cutting. Status: approved, not yet implemented.*

This directory holds the design for comparing replay-detection methods on our data.
This file carries the purpose, the governing rule, the scope boundary, and the
cross-cutting decisions. **Method-specific decisions and deviations live in that
method's own spec** — deliberately not pooled into one table, because a combined
register invites the cross-method blending this design forbids.

| spec | scope | status |
|---|---|---|
| [01-yu-swr-detector.md](01-yu-swr-detector.md) | SWR detector with a noise-percentile threshold (Yu et al. 2017) | design complete |
| [02-memoryless-sequence.md](02-memoryless-sequence.md) | Memoryless sliding-window sequence detector (Widloski & Foster 2025) | design complete |
| [03-inventory-comparison.md](03-inventory-comparison.md) | Six inventories → common schema → N-way overlap accounting | design complete |
| [04-event-features.md](04-event-features.md) | Symmetric event characterization and selection diagnostics; threshold robustness as a later follow-up | initial scope agreed |
| [05-validation.md](05-validation.md) | Simulation and held-out-tetrode validation: what a performance claim would need | parked |
| [detector_comparison.md](detector_comparison.md) | One row per detection method considered: input, signal, normalization, threshold rule, durations, state handling, requirements, status | reference, kept current |

01 and 02 are independent and can proceed in parallel. 03 depends on both; the initial
diagnostics in 04 follow 03. Record the primary settings and analyses before running the
comparison. Threshold robustness follows the default-setting report, starting with a
small cost pilot; it does not block that report.

## Purpose

The field detects replay in ways that are not interchangeable, and the choice of
detector is upstream of every claim made about the events it returns. We run several
published detection methods on the same recordings and ask which events each finds,
where they agree, and what a reader should attribute a difference to.

The immediate motivation is our own non-local event cohort and its SWR-overlap split.
Agreement across methods is evidence of robustness to detection choices. It is not
independent validation of the recordings or of the encoding model, and disagreement
can reflect a different scientific target as much as different sensitivity.

The comparison characterizes the strengths and limitations of every method, **ours
included**, for the question being asked. Record a short set of expected tradeoffs (04),
use the same primary diagnostics for all inventories, and report disagreements in both
directions. Additional questions prompted by the results are allowed and identified as
exploratory. There is no ground truth in this design; accuracy, sensitivity, precision,
and calibration claims require validation such as the studies parked in 05.

The initial report contains overlap and time coverage, event differences, boundary and
segmentation summaries, selection diagnostics, and reproducibly sampled examples at the
retained default settings. It can be completed before threshold robustness. Leave-one-
tetrode-out studies, encoding refits, native-speed controls, and task-model refits are
follow-ups motivated by particular findings, not requirements for the initial report.
No method is designated the reference detector against which others are judged.

## The governing rule

**Each method runs as itself. Shared substrate is harmonized; method internals are not.**

| | harmonized | left to each method |
|---|---|---|
| **Substrate** — same recordings, epochs, speed-eligible population, and the fitted encoding model where a method needs one | ✅ | |
| **Method internals** — thresholds, noise models, summary statistics, event assembly, scoring rules | | ✅ |

The speed/immobility rule is shared eligibility: every method's candidates are qualified
under our canonical whole-event rule (< 4 cm/s), retaining the per-bin gates needed for
candidate assembly. Running each method under its native speed gate is a **deferred
control** (03) if needed to explain a particular difference.

Shared epochs alone do not guarantee shared observation time. Every inventory carries
the availability intervals defined in 03; comparison denominators use shared valid,
speed-eligible exposure. Missing coverage produces unknown labels, not non-detections.

Everything downstream of that is method-internal and must not be blended. Concretely,
and these are live hazards rather than hypotheticals:

- Yu's noise-threshold estimator does **not** adopt Widloski's five-bin robustified mode,
  even though both implement a mode-mirroring noise model. Yu's mode is Yu's mode; the
  robustified variant is a sensitivity check inside 01, not the default.
- No single SWR definition is imposed on the sequence cohorts. Each cohort is labelled
  under **each** SWR definition and the matrix is reported (03). This also makes the
  label-sensitivity analysis fall out of the design rather than needing a separate sweep.
- Deviations that are *forced* — clusterless rather than sorted units, a 1-D branched
  track rather than a 2-D arena, a 500 Hz grid — are unavoidable and documented per
  method. Deviations that merely *import another method's choice* are not permitted.

## The three levels, and which one this is

Detection pipelines in the literature vary on three axes:

1. **Trigger** — what becomes a candidate event.
2. **Scoring** — what counts as replay given a candidate.
3. **Label** — what counts as "has a ripple."

**This design covers level 1**, with level 3 riding along because each cohort is
labelled under every SWR definition. Level 2 is deferred to 04.

The structural fact that organizes level 1: **a triggered method cannot detect replay
outside its own trigger, by construction.** SWR-triggered methods cannot find replay
without a ripple; MUA-triggered methods cannot find replay without a population burst.
Those are two different exclusions. An MUA-triggered inventory *can* find ripple-less
replay — a burst with no ripple — so the ripple-less question is askable from
inventories 4, 5, and 6. The question only posterior-triggered methods can ask is replay
with **neither** a ripple **nor** a burst, the cell Widloski & Foster report as
"neither." Keep the two questions separate wherever a non-SWR fraction is reported: the
inventory comparison (03) yields both cells, and they must not be conflated under the
label "ripple-less." That neither-cell is why the posterior-triggered family exists and
why our own pipeline is in it. Worth noting that our lab lineage moved families without
measuring the cost: Denovellis et al. 2021 and Gillespie et al. 2021 are state-space but
SWR-*triggered*, where our current pipeline is posterior-triggered.

## The six inventories

Each produces an event table over the same epochs and the same speed-eligible population.

| # | Trigger | Family | Method | Status |
|---|---|---|---|---|
| 1 | SWR, consensus trace | LFP | `Kay_ripple_detector` | exists (`SWR_event`) |
| 2 | SWR, per-channel with merging | LFP | `Karlsson_ripple_detector` (Karlsson et al. 2009) | in package; needs a run |
| 3 | SWR, noise-percentile threshold | LFP | Yu et al. 2017 | **build in `ripple_detection` — [01](01-yu-swr-detector.md)** |
| 4 | MUA high-synchrony events | population rate | `multiunit_HSE_detector` | exists (`MUA_event`) |
| 5 | Posterior threshold | decoding | our state-space non-local detection | exists (`non_local_event`) |
| 6 | Posterior sequence | decoding | Widloski & Foster 2025 | **build — [02](02-memoryless-sequence.md)** |

1, 2 and 3 are three different SWR *algorithms*, not three thresholds on one algorithm:
Kay thresholds a root-sum-of-squares consensus trace, Karlsson detects per channel and
merges, and Yu uses a median consensus with a noise-percentile threshold. Their
aggregation and event-assembly rules can produce different populations and boundaries;
the comparison measures those differences rather than assuming an ordering of sensitivity.

The `ripple_detection` package supplying 1, 2 and 4 is
[Eden-Kramer-Lab/ripple_detection](https://github.com/Eden-Kramer-Lab/ripple_detection);
Spyglass reaches 1 and 2 through `RippleTimesV1`.

## Scope boundary: clusterless compatibility

HPC in this project is decoded clusterless. That excludes every method requiring
spike-sorted units, and the exclusion is principled rather than convenient — it should
be stated in the paper, because otherwise the absences read as oversights.

| method | needs sorted units? | in scope |
|---|---|---|
| SWR triggers — Kay, Karlsson, Yu 2017 | no (LFP) | ✅ |
| MUA trigger — `multiunit_HSE_detector` | no (spike times) | ✅ |
| Posterior triggers — ours, Widloski port | no (clusterless encoding model) | ✅ |
| HMM over units — Maboudi et al. 2018, eLife 34467 | **yes** | ❌ for HPC |
| Immobility- vs movement-associated place cells — Yu et al. 2017 | **yes** | ❌ for HPC |
| Theta-cycle coactivity — Yu & Frank 2021 | **yes**, and wrong epochs | ❌ |

Two of these deserve their reasons recorded rather than a bare exclusion:

- **Maboudi et al. 2018** ("Uncovering temporal structure in hippocampal output
  patterns") fits an HMM with 30–50 latent states to population burst events themselves,
  with no place fields and no behavioural training data, then scores replay as model
  congruence against a transition-matrix shuffle. It is a genuinely different family —
  unsupervised sequence structure with no spatial template — and reports 60.9% event-level
  agreement with Bayesian replay detection. **Deferred, not rejected**: it could run on
  mPFC/OFC sorted units as a separate comparison.
- **Yu & Frank 2021** detects nonlocal representations from theta-cycle coactivity
  *during movement* (> 2 cm/s) and explicitly excludes SWR periods. Despite the shared
  phrase "nonlocal hippocampal representations," it and our detector examine disjoint
  time periods for different phenomena. Including it would be a category error.

## Cross-cutting decisions

Method-specific decisions live in the component specs. These apply across the comparison.

| # | Decision | Reasoning |
|---|---|---|
| **X1** | Compare detection *triggers* (level 1); defer scoring (level 2) to 04 | The trigger is upstream of everything else and determines which of the ripple-less and neither-ripple-nor-burst questions each inventory can ask at all |
| **X2** | Six inventories, not two | A within-family comparison of two posterior-triggered methods cannot say what the standard SWR/MUA pipelines would have found |
| **X3** | Harmonize substrate, not method internals | Comparability without blending; see the governing rule above |
| **X4** | Shared whole-event < 4 cm/s eligibility for every method; native-speed controls are deferred | Apply the same eligibility criterion to each method's candidates. Different event boundaries can still affect qualification |
| **X5** | Hold the fitted encoding model fixed for both decoding-based methods; any later running-only control applies to **both** | Substrate under X3. An encoding refit is not an initial deliverable |
| **X6** | Label every cohort under every SWR definition; report the matrix | Imposing one definition would blend methods, and the matrix delivers label sensitivity for free |
| **X7** | Clusterless compatibility bounds the factorial | See the scope boundary above; stated so the exclusions are legible |
| **X8** | Each cohort gets its own key; cohorts relate only by temporal overlap, never by shared ids | `event_id` hashes are cohort-dependent, so a shared key would mis-join silently (CLAUDE.md on atomic regeneration) |
| **X9** | Record the primary configuration and analyses; report all primary contrasts symmetrically and label subsequent exploratory analyses | Objectivity depends on fair measurements and reporting, not on a frozen list of expected weaknesses. Keep original results and any changes traceable |
| **X10** | Retain threshold robustness as a planned follow-up after the default-setting report, beginning with a small cost pilot | Reuse valid cached work, measure the expensive re-decoding/shuffle costs, and choose a limited scope before expanding. Until run, robustness to thresholds is unassessed |
| **X11** | Record observation availability for every detector and use shared exposure for comparisons (03) | Missing recordings, window support, or outputs must not create apparent detector-only events or ripple-less labels |

## Reference material

Vendored source for the Widloski method is at
[reference/widloski_foster_2025/](../../../reference/widloski_foster_2025/), whose README
records provenance, licences, the pinned commit, and the functions absent from every
published source.
