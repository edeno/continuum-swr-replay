# Replay detection method comparison — overview

*Design spec, cross-cutting. Status (2026-10-18): approved; the Yu detector exists in the
`ripple_detection` package, and the memoryless sequence detector, its shuffle, its event
table and a per-epoch runner are implemented and tested in this project, and the detector
ran on all 242 analysed epochs (run `d0964cc76ed2-seed20260922`); its two remaining pilot
checks (null validation, two sensitivities) are still to do; the inventory comparison's
accounting is implemented and tested on synthetic inventories, and its per-detector
adapters, per-epoch runner, report and examples are built and tested on simulated epochs,
and the comparison ran on all 242 analysed epochs (run `afeb897d850f`); the event
features are built, their smoke ran and the pilot ran on 5 epochs (run `86ed2e7fa90a`: 4 `ok`, and peanut, which failed there on a bug at its last window, re-checked `ok` under `f8a6984ae781`); the full run is pending. The Kay, Karlsson, Yu and Zugaro SWR inventories are
computed directly from the package, on shared CA1 inputs, by
`scripts/detection/run_swr_detection.py`, which ran on every epoch in the full pipeline
refresh (R3, then R10, run `056eee0dcb5a`); Spyglass integration is the
last step. Inventory 8, Carey, is decided (2026-10-17); its runner is built and tested, and it ran
on all 242 analysed epochs (run `fa6537e48bcf`).
Each spec's own header carries its status; the per-spec sections below carry the open
decisions.*

This directory holds the design for comparing replay-detection methods on our data.
This file carries the purpose, the governing rule, the scope boundary, and the
cross-cutting decisions. **Method-specific decisions and deviations live in that
method's own spec** — deliberately not pooled into one table, because a combined
register invites the cross-method blending this design forbids.

| spec | scope | status |
|---|---|---|
| [01-yu-swr-detector.md](01-yu-swr-detector.md) | SWR detector with a noise-percentile threshold (Yu et al. 2017) | package side implemented; direct project inventory produced by `scripts/detection/run_swr_detection.py`, run on every epoch in R3 and R10 (run `056eee0dcb5a`); Spyglass integration last |
| [02-memoryless-sequence.md](02-memoryless-sequence.md) | Memoryless sliding-window sequence detector (Widloski & Foster 2025) | implemented and tested, shuffle and event table included; ran on all 242 analysed epochs (`d0964cc76ed2-seed20260922`); the pilot checks (null validation, two sensitivities) are still to do |
| [03-inventory-comparison.md](03-inventory-comparison.md) | Nine inventories (eight detectors) → common schema → N-way overlap accounting | accounting, adapters, per-epoch runner, report and examples implemented and tested on synthetic or simulated epochs; ran on all 242 analysed epochs (run `afeb897d850f`), with its report and examples |
| [04-event-features.md](04-event-features.md) | Symmetric event characterization and selection diagnostics; threshold robustness as a later follow-up | initial scope agreed; the initial report is implemented (features, evidence, selection, populations, runner, report, examples); the smoke ran (`68f51dabdb3c`) and the pilot ran on 5 epochs (run `86ed2e7fa90a`: 4 `ok`, and peanut, which failed there on a bug at its last window, re-checked `ok` under `f8a6984ae781`); the full run is pending |
| [05-validation.md](05-validation.md) | Simulation and held-out-tetrode validation: what a performance claim would need | parked |
| [memoryless_simulation_controls.md](memoryless_simulation_controls.md) | Simulated timing, false positives, multi-part events and a first held-out test, extending 02's comparison figure | implemented 2026-09-25 |
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
- No single SWR definition is imposed on the sequence cohorts. Each cohort is labeled
  under **each** SWR definition (Kay, Karlsson, Yu, Zugaro and Carey) and the matrix is reported (03). This also makes the
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
labeled under every SWR definition. Level 2 is deferred to 04.

The structural fact that organizes level 1: **a triggered method cannot detect replay
outside its own trigger, by construction.** SWR-triggered methods cannot find replay
without a ripple; MUA-triggered methods cannot find replay without a population burst.
Those are two different exclusions. An MUA-triggered inventory *can* find ripple-less
replay — a burst with no ripple — so the ripple-less question is askable from
inventories 4, 5, and 6. The question only posterior-triggered methods can ask is replay
with **neither** a ripple **nor** a burst, the cell Widloski & Foster report as
"neither." Keep the two questions separate wherever a non-SWR fraction is reported: the
inventory comparison (03) yields both cells, and they must not be conflated under the
label "ripple-less." Nor is either the paper's own classification: in this design a
ripple-less event is one with no detected SWR overlap under a stated SWR definition,
whereas Widloski & Foster also compared each event's ripple power against
duration-matched background snippets, which this design does not reproduce. Report "no
detected SWR under these definitions", with the definition attached, never "ripple-less"
as the paper uses it. That neither-cell is why the posterior-triggered family exists and
why our own pipeline is in it. Worth noting that our lab lineage moved families without
measuring the cost: Denovellis et al. 2021 and Gillespie et al. 2021 are state-space but
SWR-*triggered*, where our current pipeline is posterior-triggered.

## The nine inventories (eight detectors)

Each produces an event table over the same epochs and the same speed-eligible population.
The comparison runner reads the memoryless one under two cohorts, `published` and
`replay_criteria`, so it compares nine inventories (03): `kay`, `karlsson`, `yu`, `zugaro`, `carey`,
`mua`, `state_space`, `memoryless` and `memoryless_criteria`.

| # | Trigger | Family | Method | Status |
|---|---|---|---|---|
| 1 | SWR, consensus trace | LFP | `Kay_ripple_detector` | computed directly at the pinned package commit by `scripts/detection/run_swr_detection.py` (R3, then R10, run `056eee0dcb5a`); it is the paper's SWR source (D7, closed 2026-10-17). The canonical Spyglass `SWR_event` is its version check, not an inventory |
| 2 | SWR, per-channel with merging | LFP | `Karlsson_ripple_detector` (Karlsson & Frank 2009) | in package; computed directly by `scripts/detection/run_swr_detection.py` (R3, then R10, run `056eee0dcb5a`) |
| 3 | SWR, noise-percentile threshold | LFP | `Yu_ripple_detector` (Yu et al. 2017) | in package; computed directly by `scripts/detection/run_swr_detection.py` (R3, then R10, run `056eee0dcb5a`) — [01](01-yu-swr-detector.md) |
| 4 | MUA high-synchrony events | population rate | `multiunit_HSE_detector` | exists (`MUA_event`) |
| 5 | Posterior threshold | decoding | our state-space non-local detection | exists (`non_local_event`) |
| 6 | Posterior sequence | decoding | Widloski & Foster 2025 | implemented (`memoryless_sequence/`, `scripts/detection/run_memoryless_detection.py`); ran on all 242 analysed epochs (`d0964cc76ed2-seed20260922`); the pilot checks (null validation, two sensitivities) are still to do — [02](02-memoryless-sequence.md) |
| 7 | SWR, one channel, two thresholds | LFP | `Zugaro_ripple_detector` (the `FindRipples` lineage: Hirase and Zugaro, FMAToolbox; Csicsvari et al. 1999 as ancestor) | in package; computed directly by `scripts/detection/run_swr_detection.py` on one CA1 channel per epoch (R3, then R10, run `056eee0dcb5a`) — see below |
| 8 | Ripple power and multiunit activity jointly | joint LFP × rate | `Carey_candidate_detector` (Carey, Tanaka & van der Meer 2019; vandermeerlab `GenCandidateEvents`) | in package (PR #17; `normalization_mask` since `9766fd4`, pinned from `e1decf2`); decided 2026-10-17; runner `scripts/detection/run_carey_detection.py` built and tested, and run on all 242 epochs (run `fa6537e48bcf`) — see below |

1, 2, 3 and 7 are four different SWR *algorithms*, not four thresholds on one algorithm:
Kay thresholds a root-sum-of-squares consensus trace, Karlsson detects per channel and
merges, Yu uses a median consensus with a noise-percentile threshold, and Zugaro bounds
one channel's smoothed squared ripple band at 2 SD and keeps the events that peak above
5 SD. Their aggregation and event-assembly rules can produce different populations and
boundaries; the comparison measures those differences rather than assuming an ordering
of sensitivity.

The `ripple_detection` package supplying 1, 2, 3, 4, 7 and 8 is
[Eden-Kramer-Lab/ripple_detection](https://github.com/Eden-Kramer-Lab/ripple_detection);
Spyglass reaches 1 and 2 through `RippleTimesV1`. For the comparison, 1, 2, 3 and 7 are
computed directly by calling the package at its pinned commit, one epoch at a time, in
`scripts/detection/run_swr_detection.py` (the refresh design's Stage 3,
[2026-10-03-full-pipeline-refresh-design.md](../2026-10-03-full-pipeline-refresh-design.md)).
Kay, Karlsson and Yu read the same stored CA1 ripple band; Zugaro reads one referenced
raw channel from the same CA1 electrodes and references. All four share the epoch's
speed trace and time span. The runner checks that the raw LFP, re-filtered as Spyglass
filtered it, reproduces the stored band. It also checks Kay against the canonical
`SWR_event` twice: Kay at the pinned commit on Spyglass's own inputs (the package version
alone), and the direct Kay inventory (version and inputs together). Populating them
through Spyglass is deferred to the end and must reproduce the direct inventories.

**Decision D1, resolved (2026-10-17).** D1 in the comparison plan
(`.claude/docs/plans/replay_detection_comparison/PLAN.md`) asked whether Zugaro, Long and
Carey get project inventories or stay as package-only comparison rows. Zugaro gets one
(inventory 7, refresh design R3, 2026-10-03). Carey gets one (inventory 8, below). Long is
dropped: it needs a stratum radiatum channel, and every tetrode in this dataset targets CA1
stratum pyramidale ([task description](../../task_description.md), "Neural implant").

**Inventory 7, Zugaro.** Zugaro runs with the
Buzsáki-lab Databank files' settings: a 120–180 Hz Butterworth band-pass of order 3,
zero-phase; events bounded at 2 SD and kept if they peak above 5 SD; events closer than
50 ms merged with no length cap; no minimum duration; events longer than 150 ms then
dropped. It reads one channel per epoch, chosen by buzcode's `bz_GetBestRippleChan` rule
from the CA1 candidates. The refresh design lists the filter design and the departures
from the lineage (Stage 3, "Where this run departs from the lineage"). One more departure:
channel selection's moving RMS (13 samples at 1000 Hz) truncates its window at gaps and
stretch edges (NaN-aware), where the package's own Zugaro moving average zero-pads
(`np.convolve(..., mode="same")`); it affects the 6 samples at each end of a stretch.

**Inventory 8, Carey.** `Carey_candidate_detector` runs at the package default, which is
the code's Hilbert ripple score, on the same shared inputs and speed rule as the other SWR
detectors (the 4 cm/s rule, the artifact mask, rule R-T for the LFP), with the multiunit
trains built from the marks the decode used. It is a fifth SWR definition: it is a column of
the SWR label matrix (X6) and one of the triggered inventories in the neither-ripple-nor-burst
cell. Four adaptations are recorded:

1. Clusterless tetrodes stand in for units. Each tetrode's multiunit train is one "unit"
   for `minimum_active_units` and for the per-unit `spike_cap`. That is a coarser count
   than sorted cells.
2. The Hilbert configuration is used, not the published spectral one. The paper's
   published candidates used a spectral ripple score built from hand-picked ripples, which
   this dataset does not have. The threshold changes with it. The published candidates
   (`precand.m:19-24`) keep the joint score, rescaled to a mean of 0.5, above 4 (eight
   times its mean), with event bounds at that same level. We use the Hilbert wrapper's
   rule (`GenCandidateEvents.m:58-59, 164-168`): the joint score is z-scored, events are
   bounded where it exceeds 1 and kept if it peaks above 3. Its effect against the
   published rule is not measured (SWR detector audit, 2026-10-19).
3. The ripple signal is the stored 150–250 Hz band averaged over the tetrodes rule R-T
   kept, as the package averages channels for Kay, Karlsson and Yu. The original reads one
   channel in 140–250 Hz.
4. There is no theta restriction. The package default has none; the original leaves out
   high-theta periods. The 4 cm/s speed rule gates. **To change at the next full rebuild (lead, 2026-10-19;
   PLAN.md D10):** the theta gate is added, on the raw LFP of the corpus callosum tetrode
   the project uses for theta, at the package defaults (6–10 Hz, z < 2).

The time grid is the LFP's, with each tetrode's spikes binned onto it. Carey's artifact
handling is the one the other four SWR detectors have: the package's `normalization_mask`
(`ripple_detection` `9766fd4`) leaves the masked samples out of its normalization, and an
event that touches the mask is dropped.

## Scope boundary: clusterless compatibility

HPC in this project is decoded clusterless. That excludes every method requiring
spike-sorted units, and the exclusion is principled rather than convenient — it should
be stated in the paper, because otherwise the absences read as oversights.

| method | needs sorted units? | in scope |
|---|---|---|
| SWR triggers — Kay, Karlsson, Yu 2017, Zugaro | no (LFP) | ✅ |
| Joint ripple × multiunit trigger — Carey 2019 | no (LFP and multiunit; tetrodes stand in for units) | ✅ |
| MUA trigger — `multiunit_HSE_detector` | no (spike times) | ✅ |
| Posterior triggers — ours, Widloski port | no (clusterless encoding model) | ✅ |
| HMM over units — Maboudi et al. 2018, eLife 34467 | **yes** | ❌ for HPC |
| Immobility- vs movement-associated place cells — Yu et al. 2017 | **yes** | ❌ for HPC |
| Theta-cycle coactivity — Yu & Frank 2021 | **yes**, and wrong epochs | ❌ |

Two of these deserve their reasons recorded rather than a bare exclusion:

- **Maboudi et al. 2018** ("Uncovering temporal structure in hippocampal output
  patterns") fits an HMM with 30–50 latent states to population burst events themselves,
  with no place fields and no behavioral training data, then scores replay as model
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
| **X1** | Compare detection *triggers* (level 1); defer scoring (level 2) to 04 | The trigger is upstream of everything else and determines which of the ripple-less and neither-ripple-nor-burst questions each inventory can ask at all. What level 1 actually compares, though, is the event inventories of complete pipelines: acceptance criteria, significance filters and boundary rules differ between them too, so a disagreement between two inventories is not attributable to the trigger alone |
| **X2** | Nine inventories (eight detectors), not two | A within-family comparison of two posterior-triggered methods cannot say what the standard SWR/MUA pipelines would have found |
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
records provenance, licenses, the pinned commit, and the functions absent from every
published source.
